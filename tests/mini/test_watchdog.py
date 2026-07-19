"""The wedged-worker liveness guard: progress watchdog + staleness surfacing.

A wedged worker (hung device call, deadlocked thread) holds its resources while
making no step progress — and can keep emitting heartbeats, so heartbeat
staleness never trips. The watchdog aborts it from inside (FAILED + stack dump
+ hard exit); the record carries the progress/heartbeat split so monitors can
tell "dead" from "slow"; per-key cancel reaps one wedge without stopping its
healthy siblings.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mini._taskworker import _MemoSink
from mini._watchdog import Watchdog
from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.memo import MemoStore
from mini.orchestration import tick
from mini.progress import ProgressMessage
from mini.runs import RunState, progress_age, stale_progress


# ---------------------------------------------------------------------------
# Watchdog unit behavior
# ---------------------------------------------------------------------------


def test_watchdog_fires_only_when_progress_stalls():
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(0.4, stalls.append, _exit=exits.append)
    with wd:
        for step in range(8):  # steady progress: each poke resets the clock
            wd.poke(step, 8)
            time.sleep(0.1)
        assert stalls == []
    assert exits == []


def test_watchdog_aborts_on_frozen_step():
    stalls: list[str] = []
    exits: list[int] = []
    wd = Watchdog(0.3, stalls.append, _exit=exits.append)
    with wd:
        deadline = time.monotonic() + 5.0
        while not exits and time.monotonic() < deadline:
            wd.poke(7, 100)  # emissions keep coming, but the step never advances
            time.sleep(0.05)
    assert exits == [70]
    (diagnosis,) = stalls
    assert "at step 7/100" in diagnosis
    assert "WatchdogStall" in diagnosis.strip().splitlines()[-1]
    assert "--- thread" in diagnosis  # the stack dump is the wedge's "traceback"


# ---------------------------------------------------------------------------
# Record surfacing: progress_at / steps_per_min / staleness views
# ---------------------------------------------------------------------------


def _msg(step: int, total: int = 10) -> ProgressMessage:
    return ProgressMessage(run_id="r", job_id="j", step=step, total=total)


def test_sink_splits_heartbeat_from_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = MemoStore(tmp_path / "sink")
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])

    sink = _MemoSink(store, "k")
    sink.put(_msg(1))  # anchors the rate window; first emission is an advance
    now[0] = 1030.0
    sink.put(_msg(4))
    assert store.record("k").get("steps_per_min") is None  # window (60s) not yet elapsed
    now[0] = 1070.0
    sink.put(_msg(8))
    rec = store.record("k")
    assert rec["progress_at"] == 1070.0
    assert rec["steps_per_min"] == pytest.approx(6.0)  # (8-1) steps over 70s
    now[0] = 1080.0
    sink.put(_msg(8))  # same step: heartbeat advances, progress doesn't
    rec = store.record("k")
    assert (rec["heartbeat_at"], rec["progress_at"]) == (1080.0, 1070.0)


def test_stale_progress_is_the_wedge_signature():
    base = {"state": RunState.RUNNING, "env": {"host": "x"}, "started_at": 900.0}
    fresh_hb = {**base, "heartbeat_at": 999.0, "progress_at": 998.0}
    wedged = {**base, "heartbeat_at": 999.0, "progress_at": 500.0}  # beating, not moving
    assert stale_progress(fresh_hb, now=1000.0) is False
    assert stale_progress(wedged, now=1000.0) is True
    assert progress_age(wedged, now=1000.0) == pytest.approx(500.0)
    # The worker's own watchdog threshold wins over the generic one when stamped.
    assert stale_progress({**wedged, "watchdog_s": 600.0}, now=1000.0) is False
    # No emission yet: age anchors on started_at; queued (no env) has no referent.
    assert progress_age(base | {"heartbeat_at": 901.0}, now=1000.0) == pytest.approx(100.0)
    assert progress_age({"state": RunState.RUNNING, "heartbeat_at": 901.0}, now=1000.0) is None


# ---------------------------------------------------------------------------
# End to end: a wedged task aborts fast; healthy siblings are untouched
# ---------------------------------------------------------------------------


def test_wedged_worker_settles_failed_with_stack_dump(tmp_path: Path):
    # Local def so cloudpickle serializes it by value for the detached worker.
    def wedge_or_work(delay: float):
        from mini import emit_progress

        emit_progress(1, 100)
        time.sleep(delay)  # delay≥60 stands in for a hung device call
        emit_progress(100, 100)
        return delay

    def main(ctx):
        return ctx.map(wedge_or_work, [0.0, 60.0])

    exp = Experiment(name="wedge", main=main)
    app = LocalApparatus("wedge", max_workers=2, data_dir=tmp_path / "wedge").w(watchdog=1)
    store = app.memo_store()
    done, _ = tick(exp, app)
    assert not done  # both cells launched, in flight

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        by_state = {RunState(r["state"]): r for r in store.records() if r.get("state")}
        if RunState.FAILED in by_state and RunState.DONE in by_state:
            break
        time.sleep(0.2)
    else:
        raise AssertionError(f"expected one DONE + one FAILED, got: {store.records()}")

    wedged = by_state[RunState.FAILED]
    assert wedged["exc_type"] == "mini._watchdog.WatchdogStall"
    assert "no step progress" in wedged["error"]
    assert wedged["watchdog_s"] == 1
    diagnosis = store.error(wedged["key"])
    assert "wedge_or_work" in diagnosis  # the stack dump names the wedged frame
    # The healthy sibling finished normally under the same watchdog.
    assert by_state[RunState.DONE].get("exc_type") is None


def test_backend_rerun_of_settled_attempt_is_a_noop(tmp_path: Path):
    """After a watchdog abort, Modal sees a container crash and re-schedules the
    input (regardless of retries=0). The re-run carries the same gen, so it must
    not flip the settled FAILED back to RUNNING and wedge again — it runs
    nothing and returns, ending the reschedule loop."""
    from mini._taskworker import execute_task
    from mini.memo import task_key_parts

    def boom():  # what the re-run would execute if the guard failed
        (tmp_path / "ran").touch()

    store = MemoStore(tmp_path / "rerun")
    key, parts = task_key_parts(boom, ())
    gen = store.mark_running(boom, key, parts, expect_gen=None)
    store.update(key, state=RunState.FAILED, error="WatchdogStall: …")  # the abort settled it

    execute_task(store, key, boom, (), [], gen=gen)
    assert store.record(key)["state"] == RunState.FAILED  # not resurrected to RUNNING
    assert not (tmp_path / "ran").exists()


def test_cancel_by_key_leaves_siblings_running(tmp_path: Path):
    def linger(x):
        time.sleep(30.0)
        return x

    def main(ctx):
        return ctx.map(linger, [1, 2])

    exp = Experiment(name="onecancel", main=main)
    app = LocalApparatus("onecancel", max_workers=2, data_dir=tmp_path / "onecancel")
    store = app.memo_store()
    done, _ = tick(exp, app)
    assert not done  # both cells launched, in flight
    k1, k2 = sorted(r["key"] for r in store.records())

    assert app.cancel(store, keys=[k1]) == [k1]
    states = {r["key"]: RunState(r["state"]) for r in store.records()}
    assert states == {k1: RunState.CANCELLED, k2: RunState.RUNNING}
    assert app.cancel(store) == [k2]  # cleanup: the unbounded form still sweeps the rest
