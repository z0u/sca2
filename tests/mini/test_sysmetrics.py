"""System metrics recorded with each attempt: GPU/CPU/RAM rows beside the result, a summary on the record."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from mini import _sysmetrics
from mini._sysmetrics import SystemSampler, _GpuStream, parse_gpu_line, read_rows, summarize
from mini.gc import apply_gc, plan_gc
from mini.local_apparatus import LocalApparatus


@pytest.fixture
def fake_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``nvidia-smi`` on PATH that streams two cards' reports ten times a second, whatever it's asked."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        "while true; do\n"
        "  echo '0, 80, 30, 2048, 24576, 50.5, 60'\n"
        "  echo '1, 40, [N/A], 1024, 24576, 30.0, 70'\n"
        "  sleep 0.1\n"
        "done\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_parse_gpu_line_drops_unreadable_values():
    assert parse_gpu_line("0, 87, 12, 1500, 23034, [N/A], 55") == (
        0,
        {"util": 87.0, "mem_util": 12.0, "mem_used_mib": 1500.0, "mem_total_mib": 23034.0, "temp_c": 55.0},
    )
    assert parse_gpu_line("garbage") is None
    assert parse_gpu_line("x, 1, 2, 3, 4, 5, 6") is None


@pytest.mark.usefixtures("fake_nvidia_smi")
def test_gpu_stream_folds_cards_into_one_row():
    stream = _GpuStream(period_ms=100)
    try:
        time.sleep(0.5)
        row = stream.take()
    finally:
        stream.close()
    assert row["gpu_util"] == pytest.approx(60)  # mean over the two cards
    assert row["gpu_util_max"] == 80
    assert row["gpu_mem_util"] == pytest.approx(30)  # card 1 reported N/A, so card 0 alone
    assert row["gpu_mem_gb"] == pytest.approx(3.0)  # summed: 2048 + 1024 MiB
    assert row["gpu_mem_total_gb"] == pytest.approx(48.0)
    assert row["gpu_temp_c"] == 70  # the hotter card


def test_gpu_stream_without_driver_is_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "")
    stream = _GpuStream()
    assert stream.take() == {}
    stream.close()


def test_summary_means_rates_and_peaks_levels():
    rows = [
        {"t": 1, "gpu_util": 50, "gpu_util_max": 90, "proc_cpu": 1.0, "rss_gb": 1.0, "gpu_mem_total_gb": 22.0},
        {"t": 2, "gpu_util": 70, "gpu_util_max": 75, "proc_cpu": 0.5, "rss_gb": 1.5, "gpu_mem_total_gb": 22.0},
    ]
    assert summarize(rows) == {
        "rows": 2,
        "gpu_util_mean": 60,
        "proc_cpu_mean": 0.75,
        "gpu_util_max": 90,
        "proc_cpu_max": 1.0,
        "rss_gb_max": 1.5,
        "gpu_mem_total_gb": 22.0,
    }


def test_read_rows_skips_a_torn_last_line():
    assert read_rows('{"t": 1}\n{"t": 2}\n{"t": ') == [{"t": 1}, {"t": 2}]


@pytest.mark.usefixtures("fake_nvidia_smi")
def test_sampler_writes_rows_and_summary(tmp_path: Path):
    published: list[dict] = []
    path = tmp_path / "sysmetrics.jsonl"
    with SystemSampler(path, lambda **f: published.append(f) or True, 0.1, step=lambda: 7):
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:  # keep a core busy, so proc_cpu has something to see
            pass
    rows = read_rows(path.read_text())
    assert len(rows) >= 3
    assert all(r["step"] == 7 for r in rows)
    assert max(r["proc_cpu"] for r in rows) > 0.5
    assert any("gpu_util" in r for r in rows)
    final = published[-1]["sys"]
    assert final["rows"] == len(rows) and final["interval_s"] == 0.1
    # Each row folds the reports since the last one, and a row can fall between the two cards' lines (80 and 40), so
    # a row reads 60 only on average. The summary is the mean of the rows.
    gpu_rows = [r["gpu_util"] for r in rows if "gpu_util" in r]
    assert final["gpu_util_mean"] == pytest.approx(sum(gpu_rows) / len(gpu_rows))
    assert 40 <= final["gpu_util_mean"] <= 80


def test_sampler_never_raises_into_the_task(tmp_path: Path):
    def broken_record(**_: object) -> bool:
        raise RuntimeError("control plane down")

    with SystemSampler(tmp_path / "missing-dir" / "rows.jsonl", broken_record, 0.05):
        time.sleep(0.15)


def test_each_attempt_records_its_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sweep, drive):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_sysmetrics.INTERVAL_ENV, "0.1")  # the local worker inherits it
    app = LocalApparatus("sys")

    def _work(x: int) -> int:  # local, so the worker gets it by value
        time.sleep(0.4)
        return x * 2

    drive(sweep("sys", _work, [1]), app)
    store = app.memo_store()
    [rec] = store.records()
    rows = store.sysmetrics(rec["key"])
    assert len(rows) >= 2
    assert {"t", "proc_cpu", "rss_gb"} <= rows[-1].keys()
    assert rec["sys"]["rows"] == len(rows)

    # A replaced attempt's rows are garbage, like its result; the current attempt's are not.
    stale = store.result_dir(rec["key"]) / "sysmetrics-deadbeef.jsonl"
    stale.write_text(json.dumps({"t": 0}) + "\n")
    [item] = plan_gc(store).by_kind("attempt-files")
    assert item.names == ["sysmetrics-deadbeef.jsonl"]
    apply_gc(store, plan_gc(store))
    assert store.sysmetrics(rec["key"]) == rows


def test_sampling_can_be_turned_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sweep, drive):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_sysmetrics.INTERVAL_ENV, "0")
    app = LocalApparatus("sys-off")

    def _work(x: int) -> int:
        return x * 2

    drive(sweep("sys-off", _work, [1]), app)
    store = app.memo_store()
    [rec] = store.records()
    assert "sys" not in rec and store.sysmetrics(rec["key"]) == []
