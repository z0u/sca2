"""The memoized local backend caps its concurrent workers and queues the rest (``mini.local_apparatus.launch_queued``)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from rich.console import Console

from mini import local_apparatus
from mini.experiment import Experiment
from mini.local_apparatus import (
    LocalApparatus,
    _launch_env,
    _pid_alive,
    _proc_start,
    _stage_spec,
    launch_queued,
    spec_path,
)
from mini.monitor import drive_and_watch
from mini.orchestration import tick
from mini.runs import RunState


def _timed(x):
    import time

    start = time.time()
    time.sleep(0.4)
    return x, start, time.time()


def _sweep(name: str, n: int) -> Experiment:
    return Experiment(name=name, main=lambda ctx: ctx.map(_timed, list(range(n))))


def _max_overlap(spans: list[tuple[float, float]]) -> int:
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    live = peak = 0
    for _, step in events:
        live += step
        peak = max(peak, live)
    return peak


def test_workers_cap_concurrent_tasks(tmp_path: Path):
    app = LocalApparatus("capped", max_workers=2, data_dir=tmp_path / "capped")
    results = drive_and_watch(_sweep("capped", 5), app, poll=0.01, console=Console(quiet=True))
    assert [x for x, _, _ in results] == list(range(5))
    assert _max_overlap([(s, e) for _, s, e in results]) <= 2


def test_queue_drains_without_a_driver(tmp_path: Path):
    """One tick and no watch: each exiting worker hands its slot to the next queued task."""
    app = LocalApparatus("detached", max_workers=1, data_dir=tmp_path / "detached")
    done, _ = tick(_sweep("detached", 3), app)
    assert not done
    store = app.memo_store()
    queued = [r for r in store.records() if not r.get("pid")]
    assert len(queued) == 2  # one started, two waiting
    deadline = time.time() + 30
    while any(r.get("state") == RunState.RUNNING for r in store.records()):
        assert time.time() < deadline, "queue did not drain"
        time.sleep(0.05)
    assert {r["state"] for r in store.records()} == {RunState.DONE}
    assert not any(spec_path(store, r["key"]).exists() for r in store.records())  # removed at launch


def test_call_staged_by_an_earlier_attempt_is_not_launched(tmp_path: Path):
    """A key claimed under a new gen but not yet re-staged must wait, not start the old call."""
    app = LocalApparatus("stale", max_workers=1, data_dir=tmp_path / "stale")
    store = app.memo_store()
    store.records_backend.write("k", {"key": "k", "state": RunState.RUNNING, "gen": "new", "created_at": 0})
    store.write_call("k", _timed, (1,), gen="old")
    _stage_spec(store, "k", "old", {})
    assert launch_queued(store) == []
    assert "pid" not in store.record("k")


def test_an_unstaged_claim_does_not_hold_a_slot(tmp_path: Path, monkeypatch):
    """An older queued record with no spec for its attempt (a tick that died mid-batch) is passed over, not counted against the cap."""
    app = LocalApparatus("wedge", max_workers=1, data_dir=tmp_path / "wedge")
    store = app.memo_store()
    store.set_meta(local_workers=1)
    store.records_backend.write("old", {"key": "old", "state": RunState.RUNNING, "gen": "g", "created_at": 0})
    store.records_backend.write("new", {"key": "new", "state": RunState.RUNNING, "gen": "g", "created_at": 1})
    store.write_call("new", _timed, (1,), gen="g")
    _stage_spec(store, "new", "g", {})
    monkeypatch.setattr(local_apparatus, "spawn_taskworker", lambda *a, **k: os.getpid())
    assert launch_queued(store) == ["new"]


def test_cancelling_a_queued_task_drops_its_spec(tmp_path: Path):
    app = LocalApparatus("cq", max_workers=1, data_dir=tmp_path / "cq")
    store = app.memo_store()
    store.set_meta(local_workers=1)
    store.records_backend.write("k", {"key": "k", "state": RunState.RUNNING, "gen": "g", "created_at": 0})
    store.write_call("k", _timed, (1,), gen="g")
    _stage_spec(store, "k", "g", {"ROLEVAR": "x"})
    assert app.cancel(store) == ["k"]
    assert not spec_path(store, "k").exists()
    assert launch_queued(store) == []
    assert store.record("k")["state"] == RunState.CANCELLED


def test_launch_spec_lives_outside_the_project(tmp_path: Path):
    """The spec may hold an env overlay, so it sits in the runtime dir (memory), readable by its owner only."""
    store = LocalApparatus("spec", data_dir=tmp_path / "project" / ".mini" / "spec").memo_store()
    _stage_spec(store, "k", "g1", {"XLA_FLAGS": "--xla_cpu_enable_fast_math=false"})
    path = spec_path(store, "k")
    assert path.is_relative_to(os.environ["XDG_RUNTIME_DIR"])
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(path.read_text()) == {"gen": "g1", "env": {"XLA_FLAGS": "--xla_cpu_enable_fast_math=false"}}


def test_without_a_runtime_dir_specs_go_to_a_private_shm_dir(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    monkeypatch.setattr(local_apparatus, "_SHM", tmp_path / "shm")
    (tmp_path / "shm").mkdir()
    store = LocalApparatus("shm", data_dir=tmp_path / "shm-run").memo_store()
    assert spec_path(store, "k").is_relative_to(tmp_path / "shm" / f"mini-{os.getuid()}")
    assert (tmp_path / "shm" / f"mini-{os.getuid()}").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("plant", ["open", "symlink"])
def test_an_unsafe_shm_dir_is_passed_over(tmp_path: Path, monkeypatch, plant: str):
    """/dev/shm is shared: a ``mini-<uid>`` someone else could read, or a symlink planted there, sends specs to the state home instead."""
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    monkeypatch.setattr(local_apparatus, "_SHM", tmp_path / "shm")
    mine = tmp_path / "shm" / f"mini-{os.getuid()}"
    (tmp_path / "shm").mkdir()
    if plant == "open":
        mine.mkdir()
        mine.chmod(0o777)
    else:
        (tmp_path / "elsewhere").mkdir(mode=0o700)
        mine.symlink_to(tmp_path / "elsewhere")
    store = LocalApparatus("shm", data_dir=tmp_path / "shm-run").memo_store()
    assert spec_path(store, "k").is_relative_to(tmp_path / "xdg-state")


def test_a_wiped_spec_is_restaged_once_its_call_is(tmp_path: Path):
    """A reboot clears the runtime dir; the next wake writes a waiting task's spec again, but not before the claiming tick has staged its call."""
    app = LocalApparatus("wiped", max_workers=1, data_dir=tmp_path / "wiped").w(env={"ROLEVAR": "now"})
    store = app.memo_store()
    rec = {"key": "k", "state": RunState.RUNNING, "gen": "g2", "created_at": 0}
    store.records_backend.write("k", rec)
    store.write_call("k", _timed, (1,), gen="g1")  # an earlier attempt's call: g2's is still to come
    app.refresh_queued(store, rec)
    assert not spec_path(store, "k").exists()
    store.write_call("k", _timed, (1,), gen="g2")
    app.refresh_queued(store, rec)
    assert json.loads(spec_path(store, "k").read_text()) == {"gen": "g2", "env": {"ROLEVAR": "now"}}


def test_launch_env_undoes_the_launchers_own_overlay(monkeypatch):
    """A worker launching a sibling passes on the sibling's overlay, not its own."""
    monkeypatch.setenv("SHARED", "base")
    monkeypatch.delenv("ONLY_MINE", raising=False)
    monkeypatch.delenv("MINI_TASK_BASE_ENV", raising=False)
    mine = _launch_env({"SHARED": "mine", "ONLY_MINE": "1"})  # as the tick launches "mine"
    # Now act as that worker, launching a sibling with an overlay of its own.
    for k, v in mine.items():
        monkeypatch.setenv(k, v)
    sibling = _launch_env({"OTHER": "2"})
    assert sibling["SHARED"] == "base"
    assert "ONLY_MINE" not in sibling
    assert sibling["OTHER"] == "2"
    assert json.loads(sibling["MINI_TASK_BASE_ENV"]) == {"OTHER": None}


def test_a_worker_launches_siblings_from_the_env_it_started_with(monkeypatch):
    """What a task sets in ``os.environ`` stays with that task."""
    started = {"PATH": "/bin", "MINI_TASK_BASE_ENV": "{}"}
    monkeypatch.setenv("SET_BY_TASK", "1")
    assert "SET_BY_TASK" not in _launch_env({}, started)


def _role_env(x):
    import os
    import time

    time.sleep(1)
    return os.environ.get("ROLEVAR")


def test_queued_tasks_launch_with_the_latest_wakes_env(tmp_path: Path):
    """A wake re-stages waiting tasks, so a config edited between wakes (or a new shell after a restart) applies to them."""
    exp = Experiment(name="fresh", main=lambda ctx: ctx.map(_role_env, [0, 1, 2]))
    old = LocalApparatus("fresh", max_workers=1, data_dir=tmp_path / "fresh").w(env={"ROLEVAR": "old"})
    new = LocalApparatus("fresh", max_workers=1, data_dir=tmp_path / "fresh").w(env={"ROLEVAR": "new"})
    tick(exp, old)  # the first task starts under the old config; two wait
    tick(exp, new)
    assert drive_and_watch(exp, new, poll=0.01, console=Console(quiet=True)) == ["old", "new", "new"]


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="identity check needs /proc")
def test_a_reused_pid_does_not_pass_for_the_worker():
    """After a restart a worker's pid may name some other process; the recorded start identity tells them apart."""
    me = os.getpid()
    start = _proc_start(me)
    assert start is not None
    assert _pid_alive(me, start)
    assert _pid_alive(me)  # a record from before pid_start: existence only
    assert not _pid_alive(me, "another-boot:" + start.split(":")[1])


def test_a_claim_never_staged_is_reaped_after_the_grace(tmp_path: Path):
    """A tick that died between claiming a key and staging its call leaves RUNNING with no pid; past the grace it settles FAILED, and a queued task is left alone."""
    app = LocalApparatus("unstaged", data_dir=tmp_path / "unstaged")
    store = app.memo_store()
    old = time.time() - 2 * local_apparatus.CLAIM_GRACE_S
    for key, created_at in [("fresh", time.time()), ("orphan", old), ("queued", old)]:
        store.records_backend.write(key, {"key": key, "state": RunState.RUNNING, "gen": "g", "created_at": created_at})
    store.write_call("queued", _timed, (0,), gen="g")  # staged for its attempt, waiting for a slot
    store.write_call("orphan", _timed, (0,), gen="stale")  # a previous attempt's call

    assert app.reap_dead(store) == ["orphan"]
    assert RunState(store.record("orphan")["state"]) == RunState.FAILED
    assert RunState(store.record("fresh")["state"]) == RunState.RUNNING
    assert RunState(store.record("queued")["state"]) == RunState.RUNNING


def test_a_reused_pid_is_reaped_and_not_signalled(tmp_path: Path, monkeypatch):
    """A RUNNING record whose pid now belongs to a stranger reads dead, and cancel doesn't signal the stranger."""
    app = LocalApparatus("reuse", data_dir=tmp_path / "reuse")
    store = app.memo_store()
    store.records_backend.write(
        "k", {"key": "k", "state": RunState.RUNNING, "gen": "g", "pid": os.getpid(), "pid_start": "old-boot:1"}
    )
    killed: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append(pid))
    assert app.reap_dead(store) == ["k"]
    app._stop_task({"pid": os.getpid(), "pid_start": "old-boot:1"})
    assert killed == []
