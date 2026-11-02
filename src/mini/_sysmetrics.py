"""
System metrics for one task attempt: GPU utilization and memory, plus the worker's CPU and RAM, sampled in the background while the task runs.

A task's own numbers say what it learned; these say what it cost the machine. "Is the GPU busy, or waiting on the host?" is the first question whenever a run is slower than expected (seed packs, a new data path, a fallback card), and without a record it can only be asked of a run that's still going. So every attempt samples as a matter of course: dependency-free (``/proc`` and ``nvidia-smi``, no torch/jax/pynvml import), cheap (one long-lived ``nvidia-smi`` process rather than a spawn per sample), and best-effort (a field the sandbox can't read is simply absent).

Two outputs, sized to where they live:

- **Rows**, one per ``interval_s``, appended to a JSONL file beside the attempt's result (``MemoStore.sysmetrics_path``). Appended as they're taken, so a worker that dies mid-run still leaves its trace up to the last row.
- **A summary** (means and peaks over the rows) on the task record as ``sys``, refreshed every few rows and at the end, so ``status`` can show it while the task runs.

GPU utilization is the driver's own figure: the share of the last ~1 s in which any kernel was running. ``nvidia-smi`` reports it every second, and each row carries the mean and peak of those reports since the previous row. Memory is what the driver sees allocated, so under JAX's default preallocation it reads ~75% of the card whatever the model needs; a task that wants its true peak asks the framework.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

# Seconds between rows. Ten is ~360 rows an hour: dense enough to see a stall or a slow epoch, small enough
# that a long run's file stays in the hundreds of KB. `MINI_SYSMETRICS_S` overrides it; 0 turns sampling off.
DEFAULT_INTERVAL_S = 10.0
INTERVAL_ENV = "MINI_SYSMETRICS_S"

# The summary rides the hot control-plane record, so it's refreshed every few rows rather than on each one.
_SUMMARY_EVERY = 3

_GPU_QUERY = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"
_GPU_FIELDS = ("util", "mem_util", "mem_used_mib", "mem_total_mib", "power_w", "temp_c")

_GIB = 1024**3


def interval_from_env() -> float | None:
    """The sampling interval for this worker, or ``None`` when sampling is off."""
    raw = os.environ.get(INTERVAL_ENV)
    try:
        s = float(raw) if raw else DEFAULT_INTERVAL_S
    except ValueError:
        s = DEFAULT_INTERVAL_S
    return s if s > 0 else None


def parse_gpu_line(line: str) -> tuple[int, dict[str, float]] | None:
    """One ``nvidia-smi --query-gpu=index,<_GPU_QUERY>`` CSV line as ``(index, fields)``; unreadable values (``[N/A]``) are dropped."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(_GPU_FIELDS) + 1:
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    fields = {}
    for name, raw in zip(_GPU_FIELDS, parts[1:], strict=True):
        try:
            fields[name] = float(raw)
        except ValueError:
            continue
    return index, fields


class _GpuStream:
    """A long-lived ``nvidia-smi`` reporting once a second, folded into per-row means and peaks.

    One process for the whole attempt instead of one per row: a spawn costs a noticeable slice of a core inside gVisor, and the task's own host work is often what we're trying to measure.
    """

    def __init__(self, period_ms: int = 1000):
        self._lock = threading.Lock()
        self._acc: dict[int, list[dict[str, float]]] = {}
        try:
            self._proc: subprocess.Popen[str] | None = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu=index,{_GPU_QUERY}",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={period_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self._proc = None  # no NVIDIA driver here: a CPU task, or a sandbox without the tool
            return
        threading.Thread(target=self._read, name="mini-sysmetrics-gpu", daemon=True).start()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if (parsed := parse_gpu_line(line)) is not None:
                index, fields = parsed
                with self._lock:
                    self._acc.setdefault(index, []).append(fields)

    def take(self) -> dict[str, float]:
        """GPU fields for one row, from the reports since the last call; ``{}`` without any.

        Several cards fold into one row: utilization is averaged over them and memory summed, which is what "how busy is this task's hardware" means for a task that owns them all.
        """
        with self._lock:
            acc, self._acc = self._acc, {}
        if not acc:
            return {}
        per_gpu = []
        for reports in acc.values():
            g: dict[str, float] = {}
            for name in _GPU_FIELDS:
                vals = [r[name] for r in reports if name in r]
                if vals:
                    g[name] = sum(vals) / len(vals)
                    if name == "util":
                        g["util_max"] = max(vals)
            per_gpu.append(g)
        out: dict[str, float] = {}

        def mean(name: str) -> float | None:
            vals = [g[name] for g in per_gpu if name in g]
            return sum(vals) / len(vals) if vals else None

        def total(name: str) -> float | None:
            vals = [g[name] for g in per_gpu if name in g]
            return sum(vals) if vals else None

        for key, val in (
            ("gpu_util", mean("util")),
            ("gpu_util_max", max((g["util_max"] for g in per_gpu if "util_max" in g), default=None)),
            ("gpu_mem_util", mean("mem_util")),
            ("gpu_mem_gb", total("mem_used_mib")),
            ("gpu_mem_total_gb", total("mem_total_mib")),
            ("gpu_power_w", total("power_w")),
            ("gpu_temp_c", max((g["temp_c"] for g in per_gpu if "temp_c" in g), default=None)),
        ):
            if val is not None:
                out[key] = val / 1024 if key.startswith("gpu_mem") and key.endswith("_gb") else val
        return out

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _host_cpu_ticks() -> tuple[int, int] | None:
    """``(busy, total)`` jiffies across all host CPUs, from ``/proc/stat``."""
    try:
        with open("/proc/stat") as f:
            fields = [int(x) for x in f.readline().split()[1:]]
    except OSError, ValueError:
        return None
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
    return sum(fields) - idle, sum(fields)


def _meminfo_gb() -> tuple[float, float] | None:
    """``(used, total)`` host RAM in GiB, from ``/proc/meminfo`` (used = total − available)."""
    try:
        info = {
            k: int(v.split()[0])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if (kv := line.split(":", 1)) and len(kv) == 2
            for k, v in [kv]
        }
        total, avail = info["MemTotal"], info["MemAvailable"]
    except OSError, ValueError, KeyError, IndexError:
        return None
    return (total - avail) / 1024**2, total / 1024**2  # kB -> GiB


def _rss_gb() -> float | None:
    """Resident set of this process, in GiB, from ``/proc/self/statm``."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
    except OSError, ValueError, IndexError:
        return None
    return pages * os.sysconf("SC_PAGE_SIZE") / _GIB


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means and peaks over *rows*: the ``sys`` field on a record.

    Means for the rates (utilization, CPU, power), peaks for the levels (memory); a field no row carries is left out.
    """

    def vals(name: str) -> list[float]:
        return [r[name] for r in rows if r.get(name) is not None]

    out: dict[str, Any] = {"rows": len(rows)}
    for name in ("gpu_util", "gpu_mem_util", "gpu_power_w", "proc_cpu", "host_cpu"):
        if v := vals(name):
            out[f"{name}_mean"] = round(sum(v) / len(v), 2)
    for name in ("gpu_util_max", "proc_cpu", "gpu_mem_gb", "gpu_temp_c", "rss_gb", "ram_used_gb"):
        if v := vals(name):
            out[f"{name.removesuffix('_max')}_max"] = round(max(v), 2)
    for name in ("gpu_mem_total_gb", "ram_total_gb"):
        if v := vals(name):
            out[name] = round(v[-1], 2)
    return out


class SystemSampler:
    """Background sampler for one attempt: rows to *path*, a summary through *record*.

    ``with SystemSampler(...):`` samples for the span of the block, taking one last row on exit so a task shorter than the interval still leaves one. Every failure inside is swallowed: metrics are an observation of the task and must never be the reason it fails.
    """

    def __init__(
        self,
        path: Path,
        record: Callable[..., bool],
        interval_s: float,
        step: Callable[[], int | None] = lambda: None,
    ):
        self._path, self._record, self._interval, self._step = path, record, interval_s, step
        self._rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu: _GpuStream | None = None
        self._file: TextIO | None = None
        self._t0 = self._last_t = time.monotonic()
        self._last_cpu = self._last_host = None

    def __enter__(self) -> SystemSampler:
        try:
            self._file = self._path.open("a", buffering=1)  # line-buffered: each row lands as it's taken
            self._gpu = _GpuStream()
            self._t0 = self._last_t = time.monotonic()
            self._last_cpu = _proc_cpu_s()
            self._last_host = _host_cpu_ticks()
        except Exception:
            return self
        self._thread = threading.Thread(target=self._run, name="mini-sysmetrics", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self._sample()
            self._publish()
        except Exception:
            pass
        finally:
            if self._gpu is not None:
                self._gpu.close()
            if self._file is not None:
                self._file.close()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._sample()
                if len(self._rows) % _SUMMARY_EVERY == 0:
                    self._publish()
            except Exception:
                pass

    def _sample(self) -> None:
        now = time.monotonic()
        dt = now - self._last_t
        if dt <= 0:
            return
        row: dict[str, Any] = {"t": round(now - self._t0, 1)}
        if (step := self._step()) is not None:
            row["step"] = step
        if self._gpu is not None:
            row |= self._gpu.take()
        cpu = _proc_cpu_s()
        if self._last_cpu is not None:
            row["proc_cpu"] = (cpu - self._last_cpu) / dt  # cores this process kept busy
        host = _host_cpu_ticks()
        if host is not None and self._last_host is not None and host[1] > self._last_host[1]:
            row["host_cpu"] = 100 * (host[0] - self._last_host[0]) / (host[1] - self._last_host[1])
        if (rss := _rss_gb()) is not None:
            row["rss_gb"] = rss
        if (mem := _meminfo_gb()) is not None:
            row["ram_used_gb"], row["ram_total_gb"] = mem
        self._last_t, self._last_cpu, self._last_host = now, cpu, host
        row = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
        self._rows.append(row)
        if self._file is not None:
            self._file.write(json.dumps(row) + "\n")

    def _publish(self) -> None:
        if self._rows:
            self._record(sys=summarize(self._rows) | {"interval_s": self._interval})


def _proc_cpu_s() -> float:
    """CPU seconds this process has used so far, over all its threads."""
    t = os.times()
    return t.user + t.system


def read_rows(text: str) -> list[dict[str, Any]]:
    """Parse a sysmetrics JSONL file, skipping a torn last line (a worker killed mid-write)."""
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows
