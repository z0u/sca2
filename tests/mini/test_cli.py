"""CLI inspect/cancel commands over the memo store.

`ls`/`status` surface memo-orchestration experiments (state lives in the memo store, addressed by name); `cancel` stops in-flight tasks. Commands are driven against ``.mini`` under a tmp cwd, so DATA_ROOT (a relative path) resolves there.
"""

from __future__ import annotations

import argparse
import os
import textwrap
import time
from pathlib import Path

import pytest

from mini.experiment import Experiment
from mini.local_apparatus import LocalApparatus
from mini.orchestration import tick
from mini.runs import RunState


def _drive(exp: Experiment, app: LocalApparatus, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, _ = tick(exp, app)
        if done:
            return
        time.sleep(0.1)
    raise AssertionError("orchestration did not complete")


def test_data_root_anchors_at_project_root(tmp_path: Path, monkeypatch):
    """`.mini` follows the project root (a marker dir), not the cwd, so `mini` finds the same store from any subdirectory; with no marker it falls back to cwd."""
    from mini.runs import data_root

    proj = tmp_path / "proj"
    (proj / "sub" / "deep").mkdir(parents=True)
    (proj / "pyproject.toml").touch()  # the project marker
    monkeypatch.chdir(proj / "sub" / "deep")
    assert data_root() == proj.resolve() / ".mini"  # walked up past sub/deep to the marker

    bare = tmp_path / "bare"  # no marker anywhere above → fall back to cwd
    bare.mkdir()
    monkeypatch.chdir(bare)
    assert data_root() == bare.resolve() / ".mini"


def test_ls_and_status_surface_memo_experiments(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no project marker under /tmp → store resolves under cwd

    def train(x):
        return x * 2

    exp = Experiment(name="cli", main=lambda ctx: ctx.map(train, [1, 2]))
    _drive(exp, LocalApparatus("cli"))  # default data_dir → .mini/cli

    from mini.__main__ import cmd_ls, cmd_status

    cmd_ls(argparse.Namespace())
    ls_out = capsys.readouterr().out
    assert "cli" in ls_out and "tasks" in ls_out  # discovered via the memo store

    cmd_status(argparse.Namespace(name="cli", app="local"))
    status_out = capsys.readouterr().out
    assert "train-" in status_out and "2 tasks" in status_out  # per-task memo records


def test_status_and_ls_report_done_despite_superseded_failure(tmp_path: Path, monkeypatch, capsys):
    """The scenario a monitor agent hits: a task fails, the fn is *replaced* by one with a new name (a new identity — re-keying every cell), and the run completes under the new keys. ``status``/``ls`` must report the *run* as done — aggregating over the requested set — with the orphaned old records shown but marked, not poisoning the state a poller acts on."""
    monkeypatch.chdir(tmp_path)

    def bad(x):
        raise RuntimeError("bug")

    def good(x):
        return x

    def sweep(fn):
        return Experiment(name="super", main=lambda ctx: ctx.map(fn, [1]))

    app = LocalApparatus("super")  # default data_dir → .mini/super
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:  # drive the buggy version to its failure
        try:
            tick(sweep(bad), app)
        except ExceptionGroup:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("map never surfaced the failure")
    _drive(sweep(good), LocalApparatus("super"))  # the "hotfix": new source, new keys

    from mini.__main__ import cmd_ls, cmd_status

    cmd_status(argparse.Namespace(name="super", app="local"))
    status_out = capsys.readouterr().out
    assert "—  done  (1 tasks, +1 superseded)" in status_out  # aggregate ignores the orphan, but counts it
    assert "(superseded)" in status_out  # …but the orphan stays visible, marked

    cmd_ls(argparse.Namespace())
    ls_out = capsys.readouterr().out
    assert "done" in ls_out and "+1 superseded" in ls_out


def test_explain_walks_the_attempt_timeline_after_a_hotfix(tmp_path: Path, monkeypatch, capsys):
    """After a hotfix, ``explain <key>`` must answer "why did this re-run": the record healed *in place* (same identity), so the story is its attempt timeline — the failed attempt under the old code, then the current one, naming the dependency that moved."""
    monkeypatch.chdir(tmp_path)

    def make(fixed: bool):
        if fixed:

            def work(x):
                return x
        else:

            def work(x):
                raise RuntimeError("bug")

        return work

    def sweep(fn):
        return Experiment(name="explain", main=lambda ctx: ctx.map(fn, [1]))

    app = LocalApparatus("explain")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            tick(sweep(make(fixed=False)), app)
        except ExceptionGroup:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("map never surfaced the failure")
    _drive(sweep(make(fixed=True)), LocalApparatus("explain"))

    store = app.memo_store()
    (rec,) = store.records()  # same qualname + inputs — one identity across the fix
    assert RunState(rec["state"]) == RunState.DONE

    from mini.__main__ import cmd_explain

    cmd_explain(argparse.Namespace(name="explain", key=rec["key"], app="local"))
    out = capsys.readouterr().out
    assert "(superseded)" not in out  # healed in place — nothing orphaned
    assert "attempts (2):" in out
    assert "failed" in out and "!! RuntimeError: bug" in out  # the old attempt and its error
    assert "changed" in out  # …and the dependency that moved to heal it


def test_status_shows_queued_distinct_from_running(tmp_path: Path, monkeypatch, capsys):
    """A RUNNING record with no ``env`` is launched-but-unstarted (the worker writes ``env`` as its first action): ``status`` must read it as *queued*, not silently lump it in with tasks actually running on a worker."""
    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "queuedexp")
    now = time.time()
    pid = os.getpid()  # a live pid, so reap_dead doesn't settle the records
    common = {"state": "running", "fn": "train", "pid": pid, "heartbeat_at": now}
    store.records_backend.merge("train-queued", {"key": "train-queued", **common})
    store.records_backend.merge("train-live", {"key": "train-live", "env": {"host": "worker.test"}, **common})

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="queuedexp", app="local"))
    out = capsys.readouterr().out
    lines = {line.split()[2]: line for line in out.splitlines() if "train-" in line}
    assert "◌" in lines["train-queued"] and "queued" in lines["train-queued"]
    assert "♥" not in lines["train-queued"]  # its heartbeat is just the launch stamp, not liveness
    assert "▸" in lines["train-live"] and "running" in lines["train-live"] and "♥" in lines["train-live"]


def test_status_badges_stale_heartbeat(tmp_path: Path, monkeypatch, capsys):
    """A RUNNING task whose heartbeat has gone quiet for minutes gets an advisory badge — the honest signal when a backend liveness probe has a blind spot (#20). Queued tasks never badge: their heartbeat is just the launch stamp."""
    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import STALE_HEARTBEAT_S, data_root

    store = MemoStore(data_root() / "staleexp")
    now = time.time()
    pid = os.getpid()  # a live pid, so reap_dead doesn't settle the records
    env = {"env": {"host": "worker.test"}}
    common = {"state": "running", "fn": "train", "pid": pid}
    store.records_backend.merge("t-fresh", {"key": "t-fresh", "heartbeat_at": now, **env, **common})
    store.records_backend.merge(
        "t-stale", {"key": "t-stale", "heartbeat_at": now - STALE_HEARTBEAT_S - 60, **env, **common}
    )
    store.records_backend.merge("t-queued", {"key": "t-queued", "heartbeat_at": now - STALE_HEARTBEAT_S - 60, **common})

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="staleexp", app="local"))
    out = capsys.readouterr().out
    lines = {line.split()[2]: line for line in out.splitlines() if "t-" in line}
    assert "⚠ stale — worker may be dead" in lines["t-stale"]
    assert "stale" not in lines["t-fresh"]
    assert "stale" not in lines["t-queued"] and "⧖" in lines["t-queued"]


def test_status_names_a_declared_phase_instead_of_badging_it(tmp_path: Path, monkeypatch, capsys):
    """A task whose body is store transfers emits no progress, so its heartbeat goes quiet and its step freezes while it is healthy. The declared span suppresses both badges and gets named, so the line says what the pause is instead of leaving 3300/3300 sitting there — the shape a wedge has too."""
    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import STALE_HEARTBEAT_S, data_root

    store = MemoStore(data_root() / "phaseexp")
    now = time.time()
    quiet = now - STALE_HEARTBEAT_S - 60
    common = {"state": "running", "fn": "publish", "pid": os.getpid(), "env": {"host": "worker.test"}}
    store.records_backend.merge(
        "t-uploading",
        {"key": "t-uploading", "heartbeat_at": quiet, "progress_at": quiet, "step": 3300, "total": 3300}
        | {"phase": "put model", "phase_until": now + 600, "watchdog_s": 120.0}
        | common,
    )
    # Same record, but its own budget has run out: the badge comes back.
    store.records_backend.merge(
        "t-overrun",
        {"key": "t-overrun", "heartbeat_at": quiet, "progress_at": quiet}
        | {"phase": "put model", "phase_until": now - 30, "watchdog_s": 120.0}
        | common,
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="phaseexp", app="local"))
    out = capsys.readouterr().out
    lines = {line.split()[2]: line for line in out.splitlines() if "t-" in line}
    assert "⋯ put model" in lines["t-uploading"] and "⚠" not in lines["t-uploading"]
    assert "⚠ stale — worker may be dead" in lines["t-overrun"]


def test_status_json_is_machine_readable(tmp_path: Path, monkeypatch, capsys):
    """``status --json`` emits one stable JSON object — the agent-facing signal (#20): aggregate state, per-task state, and honest liveness hints (queued / heartbeat age / stale), instead of grepping the human lines."""
    import json
    from unittest.mock import ANY

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import STALE_HEARTBEAT_S, data_root

    store = MemoStore(data_root() / "jsonexp")
    now = time.time()
    store.records_backend.merge("t-done", {"key": "t-done", "state": "done", "fn": "train", "metrics": {"loss": 0.5}})
    store.records_backend.merge(
        "t-zombie",
        {
            "key": "t-zombie",
            "state": "running",
            "fn": "train",
            "pid": os.getpid(),  # a live pid, so reap_dead doesn't settle it
            "env": {"host": "worker.test", "gpu": "L4"},
            "heartbeat_at": now - STALE_HEARTBEAT_S - 60,
            "step": 3,
            "total": 10,
        },
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="jsonexp", app="local", json=True))
    payload = json.loads(capsys.readouterr().out)
    payload["tasks"].sort(key=lambda t: t["key"])
    assert payload == {
        "experiment": "jsonexp",
        "app": "local",
        "state": "running",
        "settled": False,
        "tasks": [
            {
                "key": "t-done",
                "fn": "train",
                "state": "done",
                "queued": False,
                "superseded": False,
                "metrics": {"loss": 0.5},
            },
            {
                "key": "t-zombie",
                "fn": "train",
                "state": "running",
                "queued": False,
                "superseded": False,
                "step": 3,
                "total": 10,
                "env": {"host": "worker.test", "gpu": "L4"},
                "heartbeat_age_s": ANY,
                "stale_heartbeat": True,
            },
        ],
    }
    assert payload["tasks"][1]["heartbeat_age_s"] > STALE_HEARTBEAT_S


def test_status_brief_is_counts_plus_attention_only(tmp_path: Path, monkeypatch, capsys):
    """``status --brief`` answers the monitor's question — "anything to act on?" — without one line per task: counts by state, plus only the tasks needing attention (here a failed cell and a long-queued one; the healthy lines are the payload bulk on a big sweep and carry no actionable signal)."""
    import json
    from unittest.mock import ANY

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import STALE_HEARTBEAT_S, data_root

    store = MemoStore(data_root() / "briefexp")
    now = time.time()
    pid = os.getpid()  # a live pid, so reap_dead doesn't settle the records
    env = {"env": {"host": "worker.test"}}
    store.records_backend.merge("t-done", {"key": "t-done", "state": "done", "fn": "train"})
    store.records_backend.merge(
        "t-live", {"key": "t-live", "state": "running", "fn": "train", "pid": pid, "heartbeat_at": now, **env}
    )
    store.records_backend.merge(
        "t-failed", {"key": "t-failed", "state": "failed", "fn": "train", "error": "RuntimeError: boom"}
    )
    store.records_backend.merge(  # queued (no env) far past the staleness window — capacity starvation
        "t-parked", {"key": "t-parked", "state": "running", "fn": "train", "pid": pid, "heartbeat_at": now - 400}
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="briefexp", app="local", json=True, brief=True))
    payload = json.loads(capsys.readouterr().out)
    payload["attention"].sort(key=lambda t: t["key"])
    assert payload == {
        "experiment": "briefexp",
        "app": "local",
        "state": "running",
        "settled": False,
        "counts": {"done": 1, "running": 1, "failed": 1, "queued": 1},
        "attention": [
            {
                "key": "t-failed",
                "fn": "train",
                "state": "failed",
                "error": "RuntimeError: boom",
                "cause": "RuntimeError: boom",
            },
            {"key": "t-parked", "fn": "train", "state": "running", "queued_s": ANY, "cause": "queued too long"},
        ],
        "attention_total": 2,
        "attention_summary": [  # two distinct causes, one task each (launch order)
            {"fn": "train", "state": "failed", "cause": "RuntimeError: boom", "count": 1},
            {"fn": "train", "state": "queued", "cause": "queued too long", "count": 1},
        ],
    }
    assert payload["attention"][1]["queued_s"] > STALE_HEARTBEAT_S

    cmd_status(argparse.Namespace(name="briefexp", app="local", json=False, brief=True))
    out = capsys.readouterr().out
    assert "1 done · 1 running · 1 queued · 1 failed" in out  # the counts line, in fixed order
    assert "t-failed" in out and "t-parked" in out  # the actionable tasks are listed
    assert "t-done" not in out and "t-live" not in out  # …healthy ones are not


def test_status_brief_collapses_a_mass_failure(tmp_path: Path, monkeypatch, capsys):
    """The 30-containers-all-die case: identical failures must collapse to one counted line (and a bounded JSON list + cause rollup), not N near-identical lines — the difference between noise and a diagnosis, and the whole point of --brief on a wide fan-out."""
    import json

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "massexp")
    for i in range(8):  # one fan-out, one shared cause
        key = f"train_one-{i:04d}"
        store.records_backend.merge(
            key, {"key": key, "state": "failed", "fn": "train_one", "error": "RuntimeError: CUDA OOM"}
        )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="massexp", app="local", json=False, brief=True))
    out = capsys.readouterr().out
    assert "8 failed" in out  # counts line
    assert "8× train_one" in out and "RuntimeError: CUDA OOM" in out  # one collapsed line…
    assert out.count("train_one-") == 1  # …not eight — only the sample key ("— e.g. …") survives

    cmd_status(argparse.Namespace(name="massexp", app="local", json=True, brief=True))
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["attention"]) == 6  # capped list of representative records
    assert payload["attention_total"] == 8
    assert payload["attention_summary"] == [
        {"fn": "train_one", "state": "failed", "cause": "RuntimeError: CUDA OOM", "count": 8}
    ]


def test_status_brief_caps_many_distinct_causes(tmp_path: Path, monkeypatch, capsys):
    """When the causes really are distinct, the per-task listing is bounded with a ``+N more`` rollup (mirroring the failed-task hint), so the payload can't blow up however wide the fan-out — while the summary still names every cause."""
    import json

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "distinctexp")
    for i in range(8):  # eight different failures → eight singleton groups
        key = f"train_one-{i:04d}"
        store.records_backend.merge(key, {"key": key, "state": "failed", "fn": "train_one", "error": f"Err{i}"})

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="distinctexp", app="local", json=False, brief=True))
    out = capsys.readouterr().out
    assert out.count("train_one-") == 6  # first six shown in full (with their keys)…
    assert "… +2 more cause(s), 2 task(s)" in out  # …the rest rolled up

    cmd_status(argparse.Namespace(name="distinctexp", app="local", json=True, brief=True))
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["attention"]) == 6 and payload["attention_total"] == 8
    assert len(payload["attention_summary"]) == 8  # every distinct cause still named


def test_watch_timeout_exits_124_with_brief_summary(tmp_path: Path, monkeypatch, capsys):
    """A bounded ``watch --timeout`` must come back by itself — exit 124 with a compact JSON summary — so a monitor's wait can't hang past its budget."""
    import json

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "watchto")
    store.records_backend.merge(
        "t-slow",
        {
            "key": "t-slow",
            "state": "running",
            "fn": "train",
            "pid": os.getpid(),  # a live pid, so reap_dead doesn't settle it
            "env": {"host": "worker.test"},
            "heartbeat_at": time.time(),
        },
    )

    from mini.__main__ import cmd_watch

    with pytest.raises(SystemExit) as ei:
        cmd_watch(argparse.Namespace(name="watchto", app="local", poll=0.01, timeout="0.1s", json=True))
    assert ei.value.code == 124
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "timeout" and "still in flight" in payload["reason"]
    assert payload["counts"] == {"running": 1} and payload["settled"] is False


def test_watch_exit_code_reflects_settle_outcome(tmp_path: Path, monkeypatch, capsys):
    """``watch`` is the wake trigger for scripts/agents (#20): it blocks until the run settles and its exit code carries the outcome — 0 iff DONE."""
    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    from mini.__main__ import cmd_watch

    store = MemoStore(data_root() / "watchfail")
    store.records_backend.merge("t1", {"key": "t1", "state": "failed", "error": "boom"})
    with pytest.raises(SystemExit) as ei:
        cmd_watch(argparse.Namespace(name="watchfail", app="local", poll=0.01))
    assert ei.value.code == 1

    store = MemoStore(data_root() / "watchdone")
    store.records_backend.merge("t1", {"key": "t1", "state": "done"})
    cmd_watch(argparse.Namespace(name="watchdone", app="local", poll=0.01))  # no raise — exit 0
    assert "done" in capsys.readouterr().out


def test_app_resolution_precedence(tmp_path: Path, monkeypatch):
    """--app flag > launch marker > $MINI_APP > [tool.mini] app > local (#47)."""
    monkeypatch.chdir(tmp_path)
    from mini.__main__ import _resolve_app

    def ns(app: str | None = None) -> argparse.Namespace:
        return argparse.Namespace(app=app)

    assert _resolve_app("exp", ns()) == "local"  # nothing configured
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\napp = "modal"\n')
    assert _resolve_app("exp", ns()) == "modal"  # project default travels with the repo
    monkeypatch.setenv("MINI_APP", "local")
    assert _resolve_app("exp", ns()) == "local"  # env overrides pyproject (one-off shell / CI)
    marker = tmp_path / ".mini" / "exp" / ".app"
    marker.parent.mkdir(parents=True)
    marker.write_text("modal\n")
    assert _resolve_app("exp", ns()) == "modal"  # the launch marker is per-experiment ground truth
    assert _resolve_app("exp", ns(app="local")) == "local"  # explicit flag beats everything


def test_run_stamps_backend_for_later_reads(tmp_path: Path, monkeypatch, capsys):
    """A launch remembers its backend (``.mini/<name>/.app``), so ``status`` with no ``--app`` reads the store the experiment actually lives on (#47)."""
    monkeypatch.chdir(tmp_path)
    exp_file = tmp_path / "stamp.py"
    exp_file.write_text(
        textwrap.dedent("""
        from mini import Experiment
        def work(x):
            return x
        experiment = Experiment(name='stampexp', main=lambda ctx: ctx.map(work, [7]))
        """)
    )
    from mini.__main__ import cmd_run, cmd_status

    cmd_run(argparse.Namespace(path=str(exp_file), watch=True, poll=0.05, app=None, workers=1))
    assert (tmp_path / ".mini" / "stampexp" / ".app").read_text().strip() == "local"
    capsys.readouterr()

    cmd_status(argparse.Namespace(name="stampexp", app=None))  # no flag — resolved via the marker
    assert "done" in capsys.readouterr().out


def test_run_captures_lineage_and_lineage_command_reports_it(tmp_path: Path, monkeypatch, capsys):
    """A run stamps provenance into meta; ``mini lineage`` reads it back.

    Under ``/tmp`` there's no git repo, so the git block is absent — but the identity/driver/timeline half is always captured, which is what this asserts.
    """
    monkeypatch.chdir(tmp_path)
    exp_file = tmp_path / "prov.py"
    exp_file.write_text(
        textwrap.dedent("""
        from mini import Experiment
        def work(x):
            return x
        experiment = Experiment(name='provexp', main=lambda ctx: ctx.map(work, [1, 2]))
        """)
    )
    from mini.__main__ import cmd_lineage, cmd_run

    cmd_run(argparse.Namespace(path=str(exp_file), watch=True, poll=0.05, app=None, workers=1))
    capsys.readouterr()

    cmd_lineage(argparse.Namespace(name="provexp", app="local", diff=False))
    out = capsys.readouterr().out
    assert "provexp — lineage" in out
    assert "when" in out and "driver" in out  # timeline + spawning environment always present


def test_lineage_snapshots_declared_upstreams(tmp_path: Path, monkeypatch):
    """An experiment that declares ``deps`` records each upstream's provenance, so a downstream run can trace which A produced its inputs."""
    monkeypatch.chdir(tmp_path)

    def work(x):
        return x

    from mini.__main__ import _stamp_lineage

    up = Experiment(name="prep", main=lambda ctx: ctx.map(work, [1]))
    _drive(up, LocalApparatus("prep"))
    up_args = argparse.Namespace(name="prep", app="local")
    _stamp_lineage(up, LocalApparatus("prep").memo_store(), up_args)

    down = Experiment(name="train", main=lambda ctx: ctx.map(work, [2]), deps=["prep"])
    _drive(down, LocalApparatus("train"))
    store = LocalApparatus("train").memo_store()
    _stamp_lineage(down, store, argparse.Namespace(name="train", app="local"))

    upstreams = store.meta()["lineage"]["upstreams"]
    assert [u["experiment"] for u in upstreams] == ["prep"]
    assert "run_at" in upstreams[0]  # carries when the upstream first ran


def test_lineage_detects_upstreams_from_resolved_refs(tmp_path: Path, monkeypatch, capsys):
    """A run that reads another experiment's ref records it as an upstream — no ``deps=`` declared. The producer is stamped onto the ref at ``set_ref`` time (in the upstream's worker), the consumer's worker records the resolution on its task record, and the driver rolls it up into ``lineage.upstreams``."""
    monkeypatch.chdir(tmp_path)
    # Workers must hit the tmp LocalStore: an ambient bucket would write the test ref
    # to the real shared store, and a publish-repo alone builds a CAS-less store.
    monkeypatch.delenv("MINI_STORE_BUCKET", raising=False)
    monkeypatch.delenv("MINI_PUBLISH_REPO", raising=False)
    (tmp_path / "prep.py").write_text(
        textwrap.dedent("""
        from mini import Experiment
        def produce(x):
            from mini.store import put, set_ref
            set_ref('shared/thing', put(b'payload', name='thing.bin'))
            return x
        experiment = Experiment(name='prep', main=lambda ctx: ctx.map(produce, [1]))
        """)
    )
    (tmp_path / "train.py").write_text(
        textwrap.dedent("""
        from mini import Experiment
        def consume(x):
            from mini.store import get_ref
            assert get_ref('shared/thing') is not None
            return x
        experiment = Experiment(name='train', main=lambda ctx: ctx.map(consume, [2]))
        """)
    )
    from mini.__main__ import cmd_lineage, cmd_run

    ns = lambda f: argparse.Namespace(path=str(tmp_path / f), watch=True, poll=0.05, app=None, workers=1)  # noqa: E731
    cmd_run(ns("prep.py"))
    cmd_run(ns("train.py"))
    capsys.readouterr()

    store = LocalApparatus("train").memo_store()
    (rec,) = store.records()
    assert rec["upstream_refs"] == [{"ref": "shared/thing", "experiment": "prep"}]  # worker-side evidence
    upstreams = store.meta()["lineage"]["upstreams"]  # driver rollup (stamped at end of the watch)
    assert [u["experiment"] for u in upstreams] == ["prep"]
    assert upstreams[0]["refs"] == ["shared/thing"]
    assert "run_at" in upstreams[0]  # snapshotted from prep's own stored lineage

    cmd_lineage(argparse.Namespace(name="train", app="local", diff=False))
    out = capsys.readouterr().out
    assert "⇐ prep" in out and "via shared/thing" in out


def test_empty_read_names_backend_and_hints_at_the_other(tmp_path: Path, monkeypatch):
    """A read that finds nothing must say which backend it looked on and — when the run lives on the other one — name the flag to get there (#47)."""
    monkeypatch.chdir(tmp_path)
    import mini.__main__ as cli

    monkeypatch.setattr(cli, "_peek", lambda name, backend: 3 if backend == "modal" else 0)
    with pytest.raises(SystemExit) as e:
        cli.cmd_status(argparse.Namespace(name="ghost", app=None))
    assert "no tasks found for experiment 'ghost' on local" in str(e.value)
    assert "found 3 task(s) on modal — try: --app modal" in str(e.value)


def test_cancel_stops_running_task(tmp_path: Path):
    def slow(x):
        import time

        time.sleep(30)  # long enough that only a cancel ends it within the test
        return x

    app = LocalApparatus("cancelexp", data_dir=tmp_path / "cancelexp")
    tick(Experiment(name="cancelexp", main=lambda ctx: ctx.map(slow, [1])), app)  # launch + suspend

    store = app.memo_store()
    (rec,) = store.records()
    pid = rec["pid"]  # recorded synchronously at spawn
    assert pid and RunState(rec["state"]) == RunState.RUNNING

    assert app.cancel(store) == [rec["key"]]
    assert all(RunState(r["state"]) == RunState.CANCELLED for r in store.records())

    # the worker really took the SIGTERM (reap it to confirm + avoid a zombie)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("worker did not exit after cancel")


def test_retry_cli_heals_failed_task(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # DATA_ROOT + the experiment file resolve under here
    exp_file = tmp_path / "retrycli.py"
    exp_file.write_text(
        textwrap.dedent("""
        from mini import Experiment, get_data_dir
        def flaky(x):
            f = get_data_dir() / 'attempts'
            n = int(f.read_text()) if f.exists() else 0
            f.write_text(str(n + 1))
            if n == 0:  # fail on the first attempt only
                raise RuntimeError('boom once')
            return x
        experiment = Experiment(name='retrycli', main=lambda ctx: ctx.map(flaky, [7]))
        """)
    )
    from mini.__main__ import cmd_retry, cmd_run

    def ns():  # run/retry share flags; --watch drives synchronously to settle
        return argparse.Namespace(path=str(exp_file), watch=True, poll=0.05, app="local", workers=1, key=None)

    with pytest.raises(SystemExit):  # FAILED is terminal — watch surfaces it and exits 1
        cmd_run(ns())
    capsys.readouterr()  # drop the failure output

    cmd_retry(ns())  # resets the failed task, then the rerun (attempt 2) succeeds
    out = capsys.readouterr().out
    assert "retrying 1 task" in out and "✓ complete" in out


def test_run_with_a_name_instead_of_a_file_hints_at_the_split(tmp_path: Path, monkeypatch):
    """``run``/``retry`` take a *file*; the read verbs take a *name*. Typing a known experiment name here must not die in a raw ``ImportError`` — it should name the mistake and point at the verbs that take names (#57)."""
    monkeypatch.chdir(tmp_path)  # no project marker under /tmp → store resolves under cwd
    _drive(Experiment(name="stale-probe", main=lambda ctx: ctx.map(lambda x: x, [1])), LocalApparatus("stale-probe"))

    from mini.__main__ import cmd_retry

    def ns(path: str) -> argparse.Namespace:
        return argparse.Namespace(path=path, watch=False, poll=0.05, app="local", workers=1, key=None)

    with pytest.raises(SystemExit) as e:  # the name of a real experiment, at a file-taking verb
        cmd_retry(ns("stale-probe"))
    assert "'stale-probe' is an experiment name" in str(e.value)
    assert "status/results/cancel take names" in str(e.value)

    with pytest.raises(SystemExit) as e:  # an unknown token → the plain missing-file error, no name hint
        cmd_retry(ns("nope"))
    assert "no experiment file at 'nope'" in str(e.value)
    assert "experiment name" not in str(e.value)


def _running(key: str, **fields) -> dict:
    """A healthy-looking RUNNING record: live worker, fresh heartbeat, real env."""
    now = time.time()
    return {
        "key": key,
        "state": "running",
        "fn": "train_one",
        "pid": os.getpid(),
        "heartbeat_at": now,
        "progress_at": now,
        "started_at": now - 600,
        "env": {"host": "worker.test", "gpu": "L4"},
        "step": 1000,
        "total": 8000,
    } | fields


def test_status_flags_deviation_from_expectation(tmp_path: Path, monkeypatch, capsys):
    """Healthy is not the same as "nothing failed". A cell crawling at a fraction of its siblings' throughput, one projected to hit its timeout before it finishes, and one whose loss has gone to NaN all pass every liveness check — a monitor that only looks for failures reports "progressing normally" while the run burns (ex-2.1.5). The comparison belongs in the tool, not the reader."""
    import json

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "devexp")
    for i in range(3):  # the fleet: a median to compare against
        store.records_backend.merge(f"t-fast{i}", _running(f"t-fast{i}", steps_per_min=3000))
    store.records_backend.merge("t-slow", _running("t-slow", steps_per_min=100))
    # 4000 steps left at 3000/min is ~80s, but it has already been running 600s of
    # its 660s timeout — it will be killed just short of the finish line.
    store.records_backend.merge("t-late", _running("t-late", steps_per_min=3000, step=4000, timeout_s=660))
    store.records_backend.merge(
        "t-nan", _running("t-nan", steps_per_min=3000, metrics={"loss": float("nan")}, metrics_nonfinite=["loss"])
    )
    store.records_backend.merge(
        "t-rising",
        _running(
            "t-rising",
            steps_per_min=3000,
            metrics={"loss": 2.5},
            metrics_wrong_way={"loss": 4},
            metric_goals={"loss": "down"},
        ),
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="devexp", app="local", json=True, brief=True))
    payload = json.loads(capsys.readouterr().out)
    flagged = {t["key"]: t["cause"] for t in payload["attention"]}
    assert set(flagged) == {"t-slow", "t-late", "t-nan", "t-rising"}, "the healthy siblings must stay quiet"
    assert "sibling median" in flagged["t-slow"]
    assert "timeout" in flagged["t-late"]
    assert "diverged" in flagged["t-nan"]
    assert "rising" in flagged["t-rising"]


def test_a_metric_meant_to_rise_is_flagged_when_it_falls(tmp_path: Path, monkeypatch, capsys):
    """Direction comes from the job, so the flag reads the same alarm either way: a sliding accuracy is as much of a problem as a climbing loss, and the CLI matches on no metric names at all. An undeclared metric is measured but never flagged — silence beats guessing which way a domain-specific score is supposed to go."""
    import json

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "goalexp")
    for i in range(3):
        store.records_backend.merge(f"t-ok{i}", _running(f"t-ok{i}", steps_per_min=3000))
    store.records_backend.merge(
        "t-sliding",
        _running(
            "t-sliding",
            steps_per_min=3000,
            metrics={"accuracy": 0.4},
            metrics_wrong_way={"accuracy": 5},
            metric_goals={"accuracy": "up"},
        ),
    )
    store.records_backend.merge(  # climbing, but climbing is what it's for
        "t-climbing",
        _running(
            "t-climbing",
            steps_per_min=3000,
            metrics={"accuracy": 0.9},
            metrics_delta={"accuracy": 0.02},
            metric_goals={"accuracy": "up"},
        ),
    )
    store.records_backend.merge(  # moving, but nobody said which way it should go
        "t-undeclared",
        _running("t-undeclared", steps_per_min=3000, metrics={"score": 7.0}, metrics_delta={"score": 1.5}),
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="goalexp", app="local", json=True, brief=True))
    payload = json.loads(capsys.readouterr().out)
    flagged = {t["key"]: t["cause"] for t in payload["attention"]}
    assert set(flagged) == {"t-sliding"}
    assert flagged["t-sliding"] == "accuracy falling for several windows"


def test_deviation_flags_let_go_of_a_finished_task(tmp_path: Path, monkeypatch, capsys):
    """The deviation flags describe a task *now*. A settled record keeps the numbers from its final window forever, so without a state guard a cell whose last minute was slow reads "under a third of the sibling median" after it finished, and a loss that ticked up at the end parks a completed run in the attention list for good — which would make "healthy means the scan came back clean" unreachable."""
    import json

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "settledexp")
    for i in range(3):
        store.records_backend.merge(f"t-fast{i}", _running(f"t-fast{i}", steps_per_min=3000))
    store.records_backend.merge(
        "t-was-slow", _running("t-was-slow", state="done", steps_per_min=100, metrics_wrong_way={"loss": 9})
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="settledexp", app="local", json=True, brief=True))
    assert "attention" not in json.loads(capsys.readouterr().out)


def test_throughput_outlier_needs_a_fleet_to_compare_against(tmp_path: Path, monkeypatch, capsys):
    """Two cells are not a distribution: with too few reporters there is no expectation to deviate from, and flagging the slower of a pair would cry wolf on every uneven sweep."""
    import json

    monkeypatch.chdir(tmp_path)
    from mini.memo import MemoStore
    from mini.runs import data_root

    store = MemoStore(data_root() / "lonelyexp")
    store.records_backend.merge("t-fast", _running("t-fast", steps_per_min=3000))
    store.records_backend.merge("t-slow", _running("t-slow", steps_per_min=100))

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="lonelyexp", app="local", json=True, brief=True))
    assert "attention" not in json.loads(capsys.readouterr().out)


def test_region_defaults_from_the_project_config(tmp_path: Path, monkeypatch):
    """Placement is usually a property of where the run's storage lives, not of one experiment, so it has a project-wide default — with the explicit flag winning, and a role's own ``region=`` winning over both (roles are applied on top)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\nregion = "us-east"\n')

    pytest.importorskip("modal")
    from mini.__main__ import _build_apparatus
    from mini.modal_apparatus import ModalApparatus

    def built(region: str | None = None) -> ModalApparatus:
        args = argparse.Namespace(app="modal", gpu=None, timeout=None, max_containers=None, region=region)
        app = _build_apparatus("regionexp", args)
        assert isinstance(app, ModalApparatus)
        return app

    assert built().modal_fn_kwargs["region"] == "us-east"
    assert built("eu-west").modal_fn_kwargs["region"] == "eu-west"
    assert built().w(region="asia-northeast3").modal_fn_kwargs["region"] == "asia-northeast3"


def test_worker_env_defaults_from_the_project_config(tmp_path: Path, monkeypatch):
    """What a task computes under (``XLA_FLAGS``) is a property of the project, not of one experiment, so it has a project-wide default on both backends — with a role's own ``env=`` merging over it key by key rather than replacing it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.mini]\nenv = { XLA_FLAGS = "-x", KEEP = "1" }\n')

    from mini.__main__ import _build_apparatus

    args = argparse.Namespace(app="local", workers=1, watchdog=None, watchdog_grace=None)
    local = _build_apparatus("envexp", args)
    assert isinstance(local, LocalApparatus)
    assert local.env == {"XLA_FLAGS": "-x", "KEEP": "1"}
    assert local.w(env={"XLA_FLAGS": "-y"}).env == {"XLA_FLAGS": "-y", "KEEP": "1"}

    pytest.importorskip("modal")
    from mini.modal_apparatus import ModalApparatus

    margs = argparse.Namespace(app="modal", gpu=None, timeout=None, max_containers=None, region=None)
    remote = _build_apparatus("envexp", margs)
    assert isinstance(remote, ModalApparatus)
    assert remote.modal_fn_kwargs["env"] == {"XLA_FLAGS": "-x", "KEEP": "1"}


# ---------------------------------------------------------------------------
# `results`: eliding the bulk
# ---------------------------------------------------------------------------


def test_a_long_sequence_is_elided_with_its_count():
    """The reason the default view exists: a per-step metric list is one float per step, and a sweep of them buries every scalar worth reading."""
    from mini.__main__ import _abbreviated

    assert _abbreviated({"val_loss": [0.5, 0.4, 0.3, 0.2, 0.1]}) == "{'val_loss': [0.5, 0.4, 0.3, … +2]}"
    assert _abbreviated([1, 2, 3]) == "[1, 2, 3]"  # nothing dropped, nothing said
    assert _abbreviated([]) == "[]"


def test_a_long_numeric_sequence_summarizes_rather_than_showing_a_head():
    """Three floats out of hundreds say nothing about a metric trace — not where it ended, not whether it spiked. The summary answers both in about the same width."""
    from mini.__main__ import _abbreviated

    assert _abbreviated({"val_loss": [5.0, 9.0, *(4.0 for _ in range(8))]}) == (
        "{'val_loss': [10 floats: 5 → 4, max 9, mean 4.6, std 1.58]}"  # 4 is the last, so only the spike is named
    )
    # Integral sequences keep integral bounds; only the mean and spread go fractional.
    assert _abbreviated([*range(19), -1]) == "[20 ints: 0 → -1, max 18, mean 8.5, std 5.92]"


def test_a_summary_names_an_extreme_only_where_it_is_an_interior_one():
    """A metric trace usually runs one way, which puts its extremes on the ends, where they already print. Naming them there would be the largest repetition in the line."""
    from mini.__main__ import _abbreviated

    descending = [float(10 - i) for i in range(10)]
    assert _abbreviated(descending) == "[10 floats: 10 → 1, mean 5.5, std 3.03]"
    assert _abbreviated([*descending, 20.0]) == "[11 floats: 10 → 20, min 1, mean 6.82, std 5.23]"


def test_a_sequence_short_enough_to_print_in_full_is_never_rounded():
    """The summary is the one rounded thing in this view, so the length floor confines it to sequences already losing most of themselves to the elision."""
    from mini.__main__ import _STATS_MIN, _abbreviated

    exact = [1 / 3] * _STATS_MIN
    assert _abbreviated(exact) == f"[{1 / 3!r}, {1 / 3!r}, {1 / 3!r}, … +{_STATS_MIN - 3}]"
    assert "floats:" in _abbreviated([*exact, 1 / 3])  # one past the floor, and it summarizes


def test_a_summary_counts_non_finite_values_and_leaves_them_out_of_the_stats():
    """A nan mid-curve is the thing you most want a results dump to tell you about, and it would take min/max/mean/std with it if it reached them."""
    from mini.__main__ import _abbreviated

    assert _abbreviated([1.0, float("nan"), *(3.0 for _ in range(8))]) == (
        "[10 floats: 1 → 3, 1 non-finite, mean 2.78, std 0.667]"
    )
    assert _abbreviated([float("nan")] * 10) == "[10 floats: nan → nan, 10 non-finite]"


def test_a_sequence_only_summarizes_when_the_summary_would_mean_something():
    """Flags are ints to Python but don't read as numbers; a mixed sequence has no stats; a set has no first or last to name."""
    from mini.__main__ import _abbreviated

    assert _abbreviated([True, False] * 5) == "[True, False, True, … +7]"
    assert _abbreviated([1, 2, 3, "four", *range(5, 11)]) == "[1, 2, 3, … +7]"
    assert _abbreviated(set(range(20))) == "{0, 1, 2, … +17}"


def test_a_number_that_resists_arithmetic_costs_the_summary_not_the_verb():
    """`math.isfinite` raises on an int too large to be a float, and the whole summary is a nicety — so it steps aside for the head view rather than taking `results` down with it."""
    from mini.__main__ import _abbreviated

    assert _abbreviated([10**400] * 10).endswith("… +7]")  # the head view, not an OverflowError


def test_scalars_and_keys_survive_verbatim():
    """What a reader takes from the abbreviated view has to be what the result says, so only lengths are lost: no rounding, no reformatting, and every key kept."""
    from mini.__main__ import _abbreviated

    stats = {"em": 0.8333333333333334, "nll": 1.204, "converged": True, "tau": None, "name": "lam0"}
    assert _abbreviated(stats) == (
        "{'em': 0.8333333333333334, 'nll': 1.204, 'converged': True, 'tau': None, 'name': 'lam0'}"
    )


def test_containers_keep_their_shape():
    from mini.__main__ import _abbreviated

    assert _abbreviated((1, 2)) == "(1, 2)"
    assert _abbreviated({"a": {"b": [1, 2, 3, 4]}}) == "{'a': {'b': [1, 2, 3, … +1]}}"
    assert _abbreviated(frozenset({1})) == "{1}"


def test_a_container_subclass_renders_rather_than_raising():
    """A namedtuple is a tuple, and a result carrying one shouldn't cost the verb — which is what dispatching on the exact type did."""
    from collections import OrderedDict, namedtuple

    from mini.__main__ import _abbreviated

    assert _abbreviated(namedtuple("P", "a b")(1, 2)) == "(1, 2)"
    assert _abbreviated(OrderedDict(a=1)) == "{'a': 1}"


def test_a_long_string_says_how_much_it_dropped():
    from mini.__main__ import _abbreviated
    from mini.__main__ import _STR_CAP as cap

    assert _abbreviated("x" * cap) == repr("x" * cap)
    assert _abbreviated("x" * (cap + 5)) == f"{'x' * cap!r}… +5 chars"


def test_an_array_shows_its_shape_rather_than_its_payload():
    import numpy as np

    from mini.__main__ import _abbreviated

    assert _abbreviated(np.zeros((3300, 4), dtype=np.float32)) == "float32[3300, 4]"


def test_an_artifact_shows_the_handle_a_listing_wants():
    """Returning bulk as an artifact is the house pattern, so this handle is the one a results listing meets most; its full repr is 64-character shas and, for a tree, every child blob."""
    from mini.__main__ import _abbreviated
    from mini.store import Artifact

    blob = Artifact(sha256="ab" * 32, size=9021, name="evals.json")
    assert _abbreviated(blob) == "Artifact('evals.json', 9021 bytes, sha256=abababababab…)"

    tree = Artifact(sha256="cd" * 32, size=48213, name="ckpt", kind="tree", children=(blob, blob, blob))
    assert _abbreviated(tree) == "Artifact('ckpt', 48213 bytes, 3 files, sha256=cdcdcdcdcdcd…)"


def test_nesting_is_walked_only_so_far():
    from mini.__main__ import _abbreviated

    deep = value = {}
    for _ in range(10):
        value["a"] = value = {}
    assert _abbreviated(deep).endswith("…" + "}" * 7)


def test_an_unrenderable_value_falls_back_to_a_capped_repr():
    from mini.__main__ import _abbreviated
    from mini.__main__ import _STR_CAP as cap

    class Sprawling:
        def __repr__(self):
            return "S(" + "y" * 300 + ")"  # 303 characters, and no quotes added on the way out

    assert _abbreviated(Sprawling()) == f"{'S(' + 'y' * (cap - 2)}… +{303 - cap} chars"


def test_results_elides_by_default_and_full_restores_the_repr(tmp_path: Path, monkeypatch, capsys):
    """End to end over a real store: the default line is short enough to read, and every number in it is one `--full` away from being checked."""
    monkeypatch.chdir(tmp_path)

    def train(x):
        return {"label": f"cell-{x}", "val_loss": [0.1 * i for i in range(500)]}

    exp = Experiment(name="res", main=lambda ctx: ctx.map(train, [1]))
    _drive(exp, LocalApparatus("res"))

    from mini.__main__ import cmd_results

    cmd_results(argparse.Namespace(name="res", key=None, app="local", full=False))
    brief = capsys.readouterr().out
    assert "'label': 'cell-1'" in brief and "500 floats: 0 → 49.9" in brief
    assert len(brief) < 200

    cmd_results(argparse.Namespace(name="res", key=None, app="local", full=True))
    full = capsys.readouterr().out
    assert "500 floats" not in full and len(full) > 3000


def test_status_flags_results_computed_under_older_numerics(tmp_path: Path, monkeypatch, capsys):
    """A finished result carries the library versions it was computed under, so a read-only view can say — without importing anything or re-running the DAG — that the current environment might not reproduce it."""
    monkeypatch.chdir(tmp_path)
    import json

    from mini.memo import MemoStore
    from mini.runs import data_root, installed_numerics

    now_jax = installed_numerics()["jax"]
    store = MemoStore(data_root() / "driftexp")
    common = {"state": "done", "fn": "train"}
    # Two aged baselines of the same package: a sweep computed half under one jax
    # and half under another must report both, not whichever record read last.
    store.records_backend.merge(
        "train-old", {"key": "train-old", "env": {"numerics_packages": {"jax": "0.0.1"}}, **common}
    )
    store.records_backend.merge(
        "train-older", {"key": "train-older", "env": {"numerics_packages": {"jax": "0.0.2"}}, **common}
    )
    store.records_backend.merge(
        "train-now", {"key": "train-now", "env": {"numerics_packages": {"jax": now_jax}}, **common}
    )

    from mini.__main__ import cmd_status

    cmd_status(argparse.Namespace(name="driftexp", app="local"))
    assert f"⚠ 2 result(s) computed under different numerics: jax 0.0.1/0.0.2 → {now_jax}" in capsys.readouterr().out

    # Same shape in both payloads: a monitor polling --brief is the reader most likely
    # to be the only one watching while a sweep straddles an upgrade.
    expected = {"tasks": 2, "packages": {"jax": {"recorded": ["0.0.1", "0.0.2"], "current": now_jax}}}
    for brief in (False, True):
        cmd_status(argparse.Namespace(name="driftexp", app="local", json=True, brief=brief))
        assert json.loads(capsys.readouterr().out)["numerics_drift"] == expected
