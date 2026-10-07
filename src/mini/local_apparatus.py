"""
Apparatus for running sweeps locally with thread-based concurrency.

Example::

    from mini.local_apparatus import LocalApparatus

    app = LocalApparatus("my-experiment", max_workers=4)
    results = list(app.map(train, configs))

On the memoized path (``bin/mini run``) each task is a detached subprocess, and ``max_workers`` caps how many run at once. The rest wait staged and RUNNING without a pid, which ``mini status`` shows as queued. :func:`launch_queued` starts them as slots free up. It runs when a batch is staged, when a worker exits, and on each tick or watch poll, so a detached run drains its queue with nobody watching.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import secrets
import signal
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Iterable, Iterator, TypeVar, cast, override

from mini._queues import QueueLike
from mini.apparatus import Apparatus
from mini.local_queue import LocalQueue
from mini.local_volume import LocalVolume
from mini.memo import MemoStore
from mini.progress import ProgressMessage, progress_context
from mini.progress_display import RichProgressDisplay
from mini.runs import RunState, data_root, spawn_taskworker
from mini.store import Store, project_store, store_context, store_for, store_root_for
from mini.volume import data_dir_context

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["LocalApparatus"]

CLAIM_GRACE_S = 60.0
"""How long a claimed, unlaunched task may go without a staged call before ``reap_dead`` settles it. A claiming tick stages within seconds."""


class LocalApparatus(Apparatus[LocalVolume]):
    """
    Run functions locally using a thread pool.

    Jobs can report progress via ``emit_progress()`` which is automatically displayed using Rich progress bars when running in a terminal.
    """

    def __init__(self, name: str, max_workers: int | None = None, data_dir: Path | str | None = None):
        self.name = name
        self.max_workers = max_workers
        self.watchdog_s: float | None = None
        self.watchdog_grace_s: float | None = None
        self.env: dict[str, str] = {}
        self._before_hooks: list[Callable[[], Any]] = []
        self._volume: LocalVolume | None = LocalVolume(Path(data_dir) if data_dir else data_root() / name)

    def __str__(self) -> str:
        return f'Local apparatus "{self.name}"'

    def clone(self) -> LocalApparatus:
        new_app = LocalApparatus(self.name, self.max_workers)
        new_app.watchdog_s = self.watchdog_s
        new_app.watchdog_grace_s = self.watchdog_grace_s
        new_app.env = dict(self.env)
        new_app._before_hooks = self._before_hooks[:]
        new_app._volume = self._volume
        return new_app

    @override
    def w(self, **kwargs: Any) -> LocalApparatus:
        """Honor the backend-agnostic ``watchdog=`` (seconds without step progress before the worker aborts itself), ``watchdog_grace=`` (the looser threshold until the first emission) and ``env=`` (environment for the task worker, merged key by key); every other option is a backend-native knob this apparatus has no use for, ignored as before — so a role table written for Modal still loads locally.

        ``env`` reaches the *memoized* path only, where each task is its own subprocess. The interactive ``map``/``arun`` path runs threads in the caller's process, which has one shared environment and, for anything that reads env at init, has usually already read it.
        """
        if not ({"watchdog", "watchdog_grace", "env"} & kwargs.keys()):
            return self
        new_app = self.clone()
        new_app.watchdog_s = kwargs.get("watchdog", self.watchdog_s)
        new_app.watchdog_grace_s = kwargs.get("watchdog_grace", self.watchdog_grace_s)
        new_app.env |= kwargs.get("env") or {}
        return new_app

    @override
    def before_each(self, hook: Callable[[], Any]) -> LocalApparatus:
        new_app = self.clone()
        new_app._before_hooks = self._before_hooks + [hook]
        return new_app

    @override
    def memo_store(self) -> MemoStore:
        from mini.memo import MemoStore

        return MemoStore(self.volume.path)

    @override
    def spawn_tasks(self, store: MemoStore, batch: list[tuple[str, str, Callable, tuple, list]]) -> None:
        """Stage every call, then start as many as the worker cap allows; the rest queue."""
        for key, gen, fn, args, hooks in batch:
            store.write_call(key, fn, args, hooks, gen, self.watchdog_s, self.watchdog_grace_s)  # stage for worker
            _stage_spec(store, key, gen, self.env)  # after the call: it marks the staging complete
        self.launch_queued(store)

    @override
    def launch_queued(self, store: MemoStore) -> list[str]:
        # The cap lives in the run's meta, so a worker launching a sibling on exit
        # applies it too; the latest wake's --workers wins.
        if store.meta().get("local_workers") != self.task_slots:
            store.set_meta(local_workers=self.task_slots)
        return launch_queued(store)

    @override
    def refresh_queued(self, store: MemoStore, rec: dict[str, Any]) -> None:
        key, gen = rec["key"], rec.get("gen")
        if rec.get("pid") or not gen:
            return
        spec = _read_spec(store, key)
        if spec is not None and (spec.get("gen") != gen or spec.get("env") == self.env):
            return  # current already, or the claiming tick is mid-batch
        # A missing spec was wiped with the runtime dir (a reboot or logout). Re-stage it
        # only once this attempt's call is staged; before that, the claiming tick is mid-batch.
        if spec is None and store.staged_gen(key) != gen:
            return
        with _launch_lock(store):  # a launch reads then deletes the spec under this lock
            cur = store.record(key)
            if cur.get("gen") == gen and not cur.get("pid"):
                _stage_spec(store, key, gen, self.env)

    @override
    def cancel(self, store: MemoStore, keys: list[str] | None = None) -> list[str]:
        # Under the launch lock, so a worker exiting mid-cancel can't start a queued
        # task after this snapshot and leave it running unstopped.
        with _launch_lock(store):
            cancelled = super().cancel(store, keys)
            for key in cancelled:
                spec_path(store, key).unlink(missing_ok=True)  # a task cancelled while queued
            return cancelled

    @property
    def task_slots(self) -> int:
        """How many detached task workers may run at once: ``max_workers``, else the CPU count."""
        return self.max_workers or os.cpu_count() or 1

    @override
    def _stop_task(self, rec: dict[str, Any]) -> None:
        """SIGTERM the worker's process group (it's a session leader: pgid == pid)."""
        if (pid := rec.get("pid")) and _pid_alive(pid, rec.get("pid_start")):  # not a stranger reusing the pid
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGTERM)

    @override
    def _is_task_alive(self, rec: dict[str, Any], store: MemoStore) -> bool:
        """Is the recorded worker pid still a live process? (for ``reap_dead``).

        With no pid, the task is queued or its claiming tick is mid-batch. A queued task has its call staged for its attempt; a claim that has gone :data:`CLAIM_GRACE_S` without one belongs to a tick that died before staging, and nothing will ever launch it.
        """
        if pid := rec.get("pid"):
            return _pid_alive(pid, rec.get("pid_start"))
        if store.staged_gen(rec["key"]) == rec.get("gen"):
            return True  # queued: a worker slot will launch it
        claimed_at = rec.get("created_at")
        return claimed_at is None or time.time() - claimed_at < CLAIM_GRACE_S

    @override
    async def amap(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[R, None]:
        # TODO: support lazy iterables
        iterables_lists: list[list] = [list(it) for it in iterables]
        sizes = [len(it) for it in iterables_lists]
        n = min(sizes) if sizes else None

        # Name the backend (symmetric with Modal's 'Running … on Modal'), so a
        # local run — e.g. a fallback when a notebook meant to use Modal — is
        # visible in the logs rather than only inferable from the *absence* of
        # Modal's image-build output. ('locally', not 'on CPU': a local box may
        # well have a GPU that JAX/torch will use.)
        workers = self.max_workers or 1
        log.info("Running %d jobs locally (%d workers)", n, workers)
        run_id = secrets.token_hex(4)

        if self._volume is not None:
            self._volume.path.mkdir(parents=True, exist_ok=True)

        progress_display = RichProgressDisplay(n or 0, queue=LocalQueue())
        # Target ~10 emissions/sec overall: interval = max_workers / target_rate_hz
        emission_interval = workers / 10.0
        # Project-scoped artifact store, so a mapped fn's put/get resolves the ambient
        # store on the interactive path too (not only the detached memo worker). Built
        # caller-side and closed over: local execution is in-process threads.
        store = store_for(store_root_for(self._volume.path)) if self._volume is not None else project_store()
        local_fn = _wrap_for_local(
            fn,
            self._before_hooks,
            run_id,
            progress_display.queue,
            kwargs=kwargs or {},
            emission_interval=emission_interval,
            data_dir=self._volume.path if self._volume is not None else None,
            store=store,
        )

        loop = asyncio.get_running_loop()

        with progress_display, ThreadPoolExecutor(max_workers=workers) as pool:
            # Submit all tasks
            tasks = [
                loop.run_in_executor(pool, local_fn, i, *args)
                for i, args in enumerate(zip(*iterables_lists, strict=False))
            ]

            # Yield results in input order to match map semantics
            for task in tasks:
                yield await task


# Shared memory, where a runtime dir is missing (containers often lack $XDG_RUNTIME_DIR).
_SHM = Path("/dev/shm")


def _state_dir(store: MemoStore) -> Path:
    """Where this run's queued launch specs wait, outside the project and preferably in memory.

    A spec holds a task's env overlay, which could carry a credential if a role passes one through. Out of the project tree, it stays away from tooling that reads the checkout (search, agents, the site build); in memory (tmpfs), it never reaches a disk or a backup. The first of these that is usable wins:

    - ``$XDG_RUNTIME_DIR/mini/launch``: the user's own tmpfs, cleared at logout or reboot.
    - ``/dev/shm/mini-<uid>/launch``: shared memory, used only if ``mini-<uid>`` is a directory this user owns and nobody else can read. ``/dev/shm`` is shared by all users, so someone else could have created that name first.
    - ``$XDG_STATE_HOME/mini/launch`` (``~/.local/state``): on disk, where there is no tmpfs (macOS).

    A spec wiped from memory is re-staged by the next wake (:meth:`LocalApparatus.refresh_queued`). The directory name hashes the run's data dir, so two checkouts never share one.
    """
    run = store.data_dir.resolve()
    digest = hashlib.sha256(str(run).encode()).hexdigest()[:12]
    return _launch_base() / f"{run.name}-{digest}"


def _launch_base() -> Path:
    if (runtime := os.environ.get("XDG_RUNTIME_DIR")) and Path(runtime).is_dir():
        return Path(runtime) / "mini" / "launch"
    if _SHM.is_dir() and (shm := _private_dir(_SHM / f"mini-{os.getuid()}")):
        return shm / "launch"
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "mini" / "launch"


def _private_dir(d: Path) -> Path | None:
    """*d*, created if need be, if it is a real directory owned by this user with no access for anyone else; else ``None``."""
    with suppress(FileExistsError):
        d.mkdir(mode=0o700)
    try:
        st = d.lstat()  # lstat: a symlink planted by someone else must not pass
    except OSError:
        return None
    ok = stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and st.st_mode & 0o077 == 0
    return d if ok else None


def spec_path(store: MemoStore, key: str) -> Path:
    """The launch spec for a queued *key* (see :func:`_stage_spec`)."""
    return _state_dir(store) / f"{key}.json"


def _stage_spec(store: MemoStore, key: str, gen: str, env: dict[str, str]) -> None:
    """Write *key*'s launch spec: its env overlay, stamped with the attempt's *gen*.

    Written after the call, so it also marks the call as staged for this attempt: a call file left by an earlier attempt doesn't count. Only the overlay is written (config the experiment or project declares), never the launching shell's environment, and only for as long as the task is queued: :func:`launch_queued` deletes the spec once the worker starts. Owner-only permissions, like an SSH key.
    """
    d = _state_dir(store)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(spec_path(store, key), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"gen": gen, "env": env}, f)


def _read_spec(store: MemoStore, key: str) -> dict[str, Any] | None:
    try:
        return json.loads(spec_path(store, key).read_text())
    except OSError, ValueError:
        return None


# The launching process's values for the keys a worker's overlay replaced (``null`` = unset).
# It travels in the worker's own environment, which already holds those values, so the
# shell's environment never reaches disk; a worker reads it to undo its own overlay
# before it launches a sibling.
_BASE_ENV_VAR = "MINI_TASK_BASE_ENV"


def _launch_env(overlay: dict[str, str], env: dict[str, str] | None = None) -> dict[str, str]:
    """The whole environment to launch a task with: *env* (this process's, by default), minus the overlay it was itself launched with, plus *overlay*.

    A worker passes the environment it started with, so whatever its task set in ``os.environ`` stays with that task.
    """
    env = dict(os.environ) if env is None else env
    restore: dict[str, str | None] = {}
    if own := env.get(_BASE_ENV_VAR):  # we are a worker: undo our own overlay first
        with suppress(ValueError):
            restore = cast("dict[str, str | None]", json.loads(own))
    base = {k: v for k, v in env.items() if k not in restore and k != _BASE_ENV_VAR}
    base |= {k: v for k, v in restore.items() if v is not None}
    shadowed = {k: base.get(k) for k in overlay}
    return base | overlay | {_BASE_ENV_VAR: json.dumps(shadowed)}


@contextmanager
def _launch_lock(store: MemoStore) -> Iterator[None]:
    store.root.mkdir(parents=True, exist_ok=True)
    with open(store.root / ".launch.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield  # released when the file closes


def launch_queued(store: MemoStore, env: dict[str, str] | None = None) -> list[str]:
    """Start queued tasks while live workers number fewer than the run's cap; return the keys started.

    A queued task is RUNNING with a current ``gen`` and no ``pid``: claimed and staged, waiting for a slot. Live workers are RUNNING records whose pid still runs, so a settled worker (even one still exiting) and a vanished one both free their slot. Launches serialize on a lock file, and each stamps its pid before the lock drops, so two exiting workers can't both start the same task. Nothing starts past the run's wall-clock budget.
    """
    if store.budget_expired():
        return []
    cap = store.meta().get("local_workers") or os.cpu_count() or 1
    started: list[str] = []
    with _launch_lock(store):
        running = [r for r in store.records() if r.get("state") == RunState.RUNNING and r.get("gen")]
        live = sum(1 for r in running if r.get("pid") and _pid_alive(r["pid"], r.get("pid_start")))
        # Staged for its current attempt; one that isn't (its claiming tick is mid-batch,
        # or it runs on another backend) must not take a slot from one that is.
        queued = [
            (r, spec)
            for r in sorted((r for r in running if not r.get("pid")), key=lambda r: r.get("created_at") or 0)
            if (spec := _read_spec(store, r["key"])) is not None and spec.get("gen") == r["gen"]
        ]
        for rec, spec in queued[: max(0, cap - live)]:
            key, gen = rec["key"], rec["gen"]
            launch_env = _launch_env(spec.get("env") or {}, env)
            pid = spawn_taskworker(store.data_dir, key, env=launch_env, inherit=False)  # pid == pgid, for cancel
            if store.update_if(key, gen, pid=pid, pid_start=_proc_start(pid)):
                started.append(key)
                spec_path(store, key).unlink(missing_ok=True)  # the worker holds its env now
            else:  # cancelled or re-claimed since the snapshot; its gen fences the worker's writes
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGTERM)
    return started


def _proc_start(pid: int) -> str | None:
    """An identity for the process now holding *pid*: the boot it belongs to and its start time, or ``None`` where there's no ``/proc`` (or no such process).

    A pid is reused once its process exits, and across a restart any pid may belong to anything. Recorded next to a worker's pid, this tells a later probe whether the pid still names that worker.
    """
    try:
        return f"{_boot_id()}:{_stat_fields(pid)[19]}"  # field 22 of stat: start time since boot
    except OSError, IndexError:
        return None


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _stat_fields(pid: int) -> list[str]:
    """``/proc/<pid>/stat`` from the state field on. It reads "pid (comm) state ...", and comm may hold spaces or parens, so split after the final ')'."""
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()


def _pid_alive(pid: int, start: str | None = None) -> bool:
    """Whether *pid* is a running process, and, given the *start* identity from :func:`_proc_start`, the same one that was recorded.

    A zombie (an exited child not yet reaped) counts as dead: ``os.kill(pid, 0)`` succeeds on one, which would keep a hard-killed worker looking alive when it's a direct child of the watcher. On Linux we read ``/proc/<pid>/stat``; elsewhere we fall back to a signal-0 probe, with no zombie or identity check. A record from before ``pid_start`` existed (``start`` is ``None``) skips the identity check.
    """
    if Path("/proc").is_dir():
        try:
            fields = _stat_fields(pid)
        except OSError:
            return False  # no such process, or it vanished mid-read
        if fields[0] == "Z":
            return False
        return start is None or _proc_start(pid) == start
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours (shouldn't happen for our own worker)


def _wrap_for_local(
    fn: Callable[..., R],
    hooks: list[Callable[[], None]],
    run_id: str,
    queue: QueueLike[ProgressMessage],
    kwargs: dict[str, Any],
    emission_interval: float,
    data_dir: Path | None,
    store: Store | None,
) -> Callable[..., R]:
    def run_one(index: int, *args) -> R:
        dir_ctx = data_dir_context(path=data_dir) if data_dir is not None else nullcontext()
        store_ctx = store_context(store) if store is not None else nullcontext()
        with (
            progress_context(run_id, str(index), queue=queue, emission_interval=emission_interval),
            dir_ctx,
            store_ctx,
        ):
            for hook in reversed(hooks):
                hook()
            result = fn(*args, **kwargs)
            return result

    return run_one
