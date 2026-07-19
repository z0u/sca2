"""
Detached worker for a single memoized task: ``python -m mini._taskworker <data_dir> <key>``.

Loads the cloudpickled call, runs it with the data-dir + progress context (so
``get_data_dir``/``emit_progress``/``emit_metrics`` work), and records the
result or traceback under the content key. Spawned in its own session so it
outlives the orchestration tick that launched it.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import cloudpickle

from contextlib import nullcontext

from mini._queues import EndOfQueue
from mini._watchdog import Watchdog, WatchdogStall
from mini.memo import MemoStore
from mini.progress import progress_context
from mini.runs import SETTLED, RunState, compute_env
from mini.store import (
    Artifact,
    StaleWriteError,
    Store,
    artifact_shas,
    producer_context,
    resolved_refs_context,
    store_context,
    store_for,
    store_root_for,
)
from mini.volume import data_dir_context


# Throughput is re-computed over a trailing window at most this often; it also
# bounds how stale the reported rate can be (a monitor should judge liveness by
# ``progress_at`` age, not by the rate — a wedge stops updating both).
_RATE_WINDOW_S = 60.0


class _MemoSink:
    """Writes the latest progress/metrics for a task straight to its memo record.

    Writes are fenced on the attempt's *gen*: once a successor attempt (or a
    ``cancel``) takes the record, this worker's progress stops landing — and stops
    trying (``_fenced``), since a superseded attempt can never own it again.

    Beyond relaying emissions, the sink derives the record's liveness-vs-progress
    split: ``heartbeat_at`` (any emission — the worker breathes) vs ``progress_at``
    (the ``(step, total)`` pair advanced — the task *moves*), plus a
    ``steps_per_min`` throughput over a trailing window. A wedged worker can keep
    the former fresh while the latter freezes; monitors key on the difference.
    The same advance signal feeds the *watchdog*, which aborts the worker when
    progress stalls too long (see :mod:`mini._watchdog`).
    """

    def __init__(self, store: MemoStore, key: str, gen: str | None = None, watchdog: Watchdog | None = None):
        self._store = store
        self._key = key
        self._gen = gen
        self._fenced = False
        self._watchdog = watchdog
        self._last: tuple[int, int] | None = None
        self._progress_at: float | None = None
        self._rate_anchor: tuple[float, int] | None = None  # (time, step)

    def put(self, item: Any, /, block: bool = True, timeout: float | None = None) -> None:
        del block, timeout
        if isinstance(item, EndOfQueue) or self._fenced:
            return
        now = time.time()
        if (item.step, item.total) != self._last:
            self._last = (item.step, item.total)
            self._progress_at = now
            if self._watchdog is not None:
                self._watchdog.poke(item.step, item.total)
        fields = dict(
            step=item.step,
            total=item.total,
            message=item.message,
            metrics=item.metrics,
            heartbeat_at=now,
        )
        if self._progress_at is not None:
            fields["progress_at"] = self._progress_at
        if self._rate_anchor is None:
            self._rate_anchor = (now, item.step)
        elif (elapsed := now - self._rate_anchor[0]) >= _RATE_WINDOW_S:
            fields["steps_per_min"] = round((item.step - self._rate_anchor[1]) / elapsed * 60.0, 1)
            self._rate_anchor = (now, item.step)
        if self._gen is None:
            self._store.update(self._key, **fields)
        elif not self._store.update_if(self._key, self._gen, **fields):
            self._fenced = True

    def get(self, /, block: bool = True, timeout: float | None = None) -> Any:
        del block, timeout
        raise NotImplementedError

    def empty(self) -> bool:
        return True


class _FencedStore(Store):
    """The ambient store for one attempt, with mutable-name writes gen-fenced.

    Record writes and results are already fenced on the attempt generation, but
    ``set_ref`` / ``publish`` mutate *names* in the artifact store — unfenced,
    a stale worker's name write would silently last-writer-win its successor's
    (CAS blobs are immune, so everything else passes straight through). The name
    lives in a different backend than the record (files/HF vs the record store),
    so the fence is check → write → re-check rather than atomic: a supersession
    landing *during* the write can't be prevented, but the re-check turns it from
    silent corruption into a loud :class:`~mini.store.StaleWriteError` — and the
    successor's own completing write then heals the name.
    """

    def __init__(self, inner: Store, memo: MemoStore, key: str, gen: str):
        self._inner, self._memo, self._key, self._gen = inner, memo, key, gen

    def _fence(self, verb: str, after: bool = False) -> None:
        if self._memo.record(self._key).get("gen") != self._gen:
            raise StaleWriteError(
                f"{verb}: attempt {self._gen} of task {self._key} was superseded (relaunched or cancelled)"
                + (" during the write — the name may briefly hold this attempt's value" if after else "")
            )

    # Fenced mutable-name verbs — everything else passes through untouched.

    def set_ref(self, name: str, art: Artifact) -> None:
        self._fence(f"set_ref({name!r})")
        self._inner.set_ref(name, art)
        self._fence(f"set_ref({name!r})", after=True)

    def publish(self, art: Artifact, path: str) -> str:
        self._fence(f"publish({path!r})")
        url = self._inner.publish(art, path)
        self._fence(f"publish({path!r})", after=True)
        return url

    def _write_ref(self, name: str, payload: str) -> None:
        self._fence(f"set_ref({name!r})")
        self._inner._write_ref(name, payload)
        self._fence(f"set_ref({name!r})", after=True)

    # Pass-throughs: forward the public verbs (not the shared high-level logic)
    # so a backend's own overrides — HF batching, cache warming — stay in play.

    def put(self, data: bytes | Path, *, name: str) -> Artifact:
        return self._inner.put(data, name=name)

    def get(self, art: Artifact, dest: Path) -> Path:
        return self._inner.get(art, dest)

    def get_ref(self, name: str) -> Artifact | None:
        return self._inner.get_ref(name)

    def has(self, sha256: str) -> bool:
        return self._inner.has(sha256)

    def _write_blob(self, sha256: str, src: Path) -> None:
        self._inner._write_blob(sha256, src)

    def _read_blob(self, sha256: str, dest: Path) -> None:
        self._inner._read_blob(sha256, dest)

    def _read_ref(self, name: str) -> str | None:
        return self._inner._read_ref(name)


def _producer_stamp(experiment: str | None, store: MemoStore, key: str) -> dict[str, Any] | None:
    """The identity ``set_ref`` stamps into refs this task writes, or ``None``.

    A compact provenance record (the :func:`~mini.lineage.upstream_snapshot` shape
    plus the task key): enough for a consumer — a downstream run, a report — to
    attribute the bytes to this experiment and the code state that produced them,
    without a lookup into this run's control plane. The code state comes from the
    run's stored lineage (stamped by the driver at wake start); best-effort, since
    provenance must never take a task down.
    """
    if experiment is None:
        return None
    from mini.lineage import upstream_snapshot

    try:
        meta = store.meta()
    except Exception:
        meta = {}
    stamp = upstream_snapshot(experiment, meta)
    stamp.pop("modal_app_ids", None)  # app ids accumulate run-wide; they don't identify this write
    stamp["task"] = key
    return stamp


def _upstream_refs(resolved: dict[str, dict[str, Any] | None]) -> list[dict[str, str]]:
    """Compact ``{ref, experiment?}`` entries for the record, from the resolved-ref set."""
    return [
        {"ref": name, **({"experiment": p["experiment"]} if p and p.get("experiment") else {})}
        for name, p in sorted(resolved.items())
    ]


def _attempt_already_settled(store: MemoStore, key: str, gen: str | None) -> bool:
    """Is this worker a backend re-run of an attempt that already settled?

    Concretely: a watchdog abort exits the process, which Modal sees as a
    *container crash* and re-schedules the input regardless of ``retries=0`` —
    the re-run carries the same gen, so without this guard it would flip the
    settled FAILED back to RUNNING, wedge again, and crash-loop until the role
    timeout. The record already tells the story; the re-run should run nothing
    and return cleanly, so the input completes and the loop ends.
    """
    if gen is None:
        return False
    prior = store.record(key)
    return prior.get("gen") == gen and prior.get("state") in SETTLED


def _arm_watchdog(
    watchdog_s: float | None,
    store: MemoStore,
    key: str,
    gen: str | None,
    commit: Callable[[], None] | None,
    record: Callable[..., bool],
) -> tuple[Watchdog | None, dict[str, Any]]:
    """Build the progress watchdog (or not) plus the record fields that go with it.

    The ``watchdog_s`` stamp lands on the record so client-side staleness views
    can match the worker's own threshold.
    """
    if not watchdog_s:
        return None, {}
    return Watchdog(watchdog_s, _stall_handler(store, key, gen, commit, record)), {"watchdog_s": watchdog_s}


def _stall_handler(
    store: MemoStore, key: str, gen: str | None, commit: Callable[[], None] | None, record: Callable[..., bool]
) -> Callable[[str], None]:
    """The watchdog's ``on_stall``: persist the stall as a task failure.

    The stall twin of ``execute_task``'s except-path, run on the watchdog's
    thread while the main thread is presumed wedged: persist first (error file →
    commit → settle FAILED, same order), then the watchdog exits the process.
    Fenced like every other write — a superseded attempt settles nothing and
    just dies.
    """

    def abort_stalled(diagnosis: str) -> None:
        summary = diagnosis.strip().splitlines()[-1]
        now = time.time()
        try:
            store.error_path(key, gen).write_text(diagnosis)
            if commit is not None:
                commit()
        finally:
            record(
                state=RunState.FAILED,
                error=summary,
                exc_type=f"{WatchdogStall.__module__}.{WatchdogStall.__qualname__}",
                heartbeat_at=now,
                finished_at=now,
            )

    return abort_stalled


def execute_task(
    store: MemoStore,
    key: str,
    fn: Any,
    args: tuple,
    hooks: list,
    commit: Callable[[], None] | None = None,
    artifacts: Store | None = None,
    gen: str | None = None,
    experiment: str | None = None,
    watchdog_s: float | None = None,
) -> None:
    """Run one memoized call and persist its result/state — backend-agnostic.

    Shared by the local subprocess worker and the Modal remote worker: only how
    the call *arrives* (staged on disk vs passed to ``spawn``) and where state
    lands (``RecordStore``) differ; the run/persist core is identical.

    *gen* is the attempt generation this worker runs under. Every record write is
    fenced on it, and the result/error land in gen-qualified files — so a stale
    worker (superseded by a relaunch, or cancelled but surviving SIGTERM) can
    neither merge DONE over its successor's RUNNING nor overwrite its result. A
    worker that finds itself already superseded at startup exits without running.

    *commit* is called after the result/error is written to the I/O plane and
    *before* the record flips to DONE/FAILED — so a poller never sees a settled
    state whose artifact hasn't been committed yet (the Modal Volume needs this).

    *artifacts* binds the content-addressed :class:`~mini.store.Store` as ambient
    for ``mini.store.put`` / ``get`` inside the step. Because ``put`` uploads
    synchronously, by the time the result is written its handles already resolve —
    so the existing write → commit → DONE order extends from "the volume flushed"
    to "the referenced blobs are durable" for free. Its mutable-name verbs
    (``set_ref`` / ``publish``) are fenced on *gen* via :class:`_FencedStore`, so
    a stale worker fails loudly instead of clobbering a name its successor owns.

    *experiment* is the experiment this task belongs to. It powers ref provenance
    both ways: refs the step writes are stamped with it (:func:`_producer_stamp`),
    and refs the step *resolves* land on the settled record as ``upstream_refs`` —
    the evidence the driver aggregates into ``lineage.upstreams``, without the
    experiment declaring its deps by hand.

    *watchdog_s* arms the progress watchdog: if the task's ``(step, total)``
    hasn't advanced in that many seconds, the worker settles its own record
    FAILED (with an all-thread stack dump as the traceback) and hard-exits —
    a silent wedge becomes a fast, retryable failure instead of burning the
    whole role ``timeout`` (see :mod:`mini._watchdog`). It covers the task call
    only — result upload rides on the role timeout as before.
    """

    def record(**fields: Any) -> bool:
        if gen is None:
            store.update(key, **fields)
            return True
        return store.update_if(key, gen, **fields)

    if _attempt_already_settled(store, key, gen):
        return

    result_dir = store.result_dir(key)
    result_dir.mkdir(parents=True, exist_ok=True)
    watchdog, wd_fields = _arm_watchdog(watchdog_s, store, key, gen, commit, record)
    sink = _MemoSink(store, key, gen, watchdog=watchdog)
    if artifacts is not None and gen is not None:
        artifacts = _FencedStore(artifacts, store, key, gen)
    # Record what we actually ran on (host/GPU/…) and when the worker truly began,
    # captured here in the worker. ``started_at`` (worker's first write) pairs with
    # ``finished_at`` (below) for a real execution duration — distinct from the
    # client-side ``created_at`` stamped before the worker was even scheduled.
    started_at = time.time()
    if not record(
        state=RunState.RUNNING, heartbeat_at=started_at, started_at=started_at, **wd_fields, env=compute_env()
    ):
        return  # superseded before we even started — nothing here is wanted anymore
    producer = _producer_stamp(experiment, store, key)
    resolved: dict[str, dict[str, Any] | None] = {}
    try:
        with (
            data_dir_context(store.data_dir),
            store_context(artifacts) if artifacts is not None else nullcontext(),
            producer_context(producer) if producer is not None else nullcontext(),
            resolved_refs_context(resolved),
            progress_context(key, key, queue=sink, emission_interval=0.2),
            watchdog if watchdog is not None else nullcontext(),
        ):
            for hook in reversed(hooks):
                hook()
            result = fn(*args)
        # The artifact sidecar rides with the result: which blobs the result
        # references, written even when empty ("none" beats "unknown" — the GC
        # mark phase then never has to unpickle this result). Sidecar first, so
        # a readable result always has its reference set alongside.
        store.artifacts_path(key, gen).write_text(json.dumps(sorted(artifact_shas(result))))
        store.result_path(key, gen).write_bytes(cloudpickle.dumps(result))
        if commit is not None:
            commit()
        now = time.time()
        # Which shared refs this step resolved (and whose experiment wrote each) —
        # the per-task evidence the driver rolls up into ``lineage.upstreams``.
        extra = {"upstream_refs": _upstream_refs(resolved)} if resolved else {}
        record(state=RunState.DONE, heartbeat_at=now, finished_at=now, **extra)
    except Exception as exc:
        tb = traceback.format_exc()
        store.error_path(key, gen).write_text(tb)
        if commit is not None:
            commit()
        now = time.time()
        extra = {"upstream_refs": _upstream_refs(resolved)} if resolved else {}
        record(
            state=RunState.FAILED,
            error=tb.strip().splitlines()[-1],
            exc_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            heartbeat_at=now,
            finished_at=now,
            **extra,
        )


def run_task(data_dir: Path, key: str) -> None:
    """Local subprocess entry: read the staged call from disk and run it."""
    store = MemoStore(data_dir)
    fn, args, hooks, gen, watchdog_s = store.read_call(key)
    # Project-scoped artifact store sits beside the experiment's data dir (or the
    # shared HF bucket, if MINI_STORE_BUCKET is set), so a blob put here resolves
    # from any experiment in the project (and from reports).
    artifacts = store_for(store_root_for(data_dir))
    # The local data dir is <data_root>/<experiment>, so its leaf names the experiment.
    execute_task(
        store, key, fn, args, hooks, artifacts=artifacts, gen=gen, experiment=data_dir.name, watchdog_s=watchdog_s
    )


def main() -> None:
    run_task(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
