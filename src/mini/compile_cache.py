"""
JAX's compilation cache, shared by every worker, with the first compile of each program settled among them.

Without XLA's determinism flags, a GPU run is reproducible only while it runs the same compiled program: the autotuner times its candidates on the card at hand and keeps the fastest, and a noisy timing picks differently from one container to the next. Loading the compiled program from a shared cache pins the picks, so flags-off runs agree bit for bit (``eng/determinism.md``). That makes the cache part of what a result means, and three things follow:

- **One compile per program and card model.** A fan-out launched cold would otherwise have every task compile and time its own picks. So each program is *claimed* before it compiles: the worker whose claim lands compiles and writes the entry, and the others wait for it and load it. A claim is a lease, renewed while its holder compiles, so a worker that dies mid-compile holds the rest up by a lease at most. JAX's own key would also split one card model by host pool (:meth:`SharedCache._name`), so the key here names the card instead.
- **Write once.** An entry is never replaced. A worker that compiled without the claim (its wait ran out) keeps its own program for the attempt, and the record says so.
- **Provenance.** Each attempt records the entries it loaded or wrote (:meth:`SharedCache.summary`), so two attempts can be checked for running the same programs, and a cache that was lost and refilled shows up as a new digest.

Two backends, with the same contract: :class:`LocalBackend` (a directory and ``flock``, for subprocess workers) and :class:`ModalBackend` (entries on a Volume, through its API rather than a mount, so a write is visible to every container at once; claims in a ``modal.Dict``, whose entries expire after a week idle, which suits a lease). The worker installs the cache when ``MINI_COMPILE_CACHE`` names a namespace (:func:`namespace_from_env`). Different namespaces never share entries, so a probe can start cold without touching the project cache.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

ENV = "MINI_COMPILE_CACHE"
"""Env var naming the cache namespace a worker uses; unset, empty, ``0`` or ``off`` leaves JAX's own (unshared) behavior."""

DETERMINISTIC_XLA_FLAGS = (
    "--xla_cpu_enable_concurrency_optimized_scheduler=false"
    " --xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true"
)
"""The ``XLA_FLAGS`` this project ran under until 2026-10-31: reproducible on any host of a card model with no cache, at about a sixth of a step. A role that must reproduce a run from before the cache sets ``env={"XLA_FLAGS": DETERMINISTIC_XLA_FLAGS}`` (``eng/determinism.md``)."""

VOLUME_NAME = "mini-compile-cache"
CLAIMS_DICT_NAME = "mini-compile-cache-claims"

# How long a waiting worker gives a claimed compile before it compiles on its own. Long enough for the slowest
# compile we have seen by a wide margin (a training step's is 10 to 30 s); the lease below catches a dead holder
# much sooner, so this only matters for a holder that is alive and stuck.
WAIT_S = 20 * 60.0
LEASE_S = 90.0
RENEW_S = 10.0
POLL_S = 0.5

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def namespace_from_env() -> str | None:
    """The namespace ``MINI_COMPILE_CACHE`` names, or ``None`` when the shared cache is off."""
    raw = os.environ.get(ENV, "").strip()
    return None if raw.lower() in ("", "0", "off", "false", "no") else _UNSAFE.sub("_", raw)


class Claim(Protocol):
    def release(self) -> None: ...


class Backend(Protocol):
    """Where entries and claims live. Names are already namespaced and filesystem-safe."""

    def read(self, name: str) -> bytes | None:
        """The entry's bytes, or ``None`` if there is none."""
        ...

    def write_once(self, name: str, data: bytes) -> bool:
        """Store *data* unless an entry exists; ``True`` if this call stored it."""
        ...

    def claim(self, name: str) -> Claim | None:
        """Take the right to compile *name*, without blocking; ``None`` while another live worker holds it."""
        ...


# ---------------------------------------------------------------------------
# The cache JAX talks to
# ---------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SharedCache:
    """A JAX ``CacheInterface`` over a :class:`Backend`, claiming each program before it compiles.

    JAX calls :meth:`get` before compiling and :meth:`put` after. A miss in :meth:`get` either returns ``None`` holding the claim (this worker compiles, and :meth:`put` writes the entry and lets go) or waits for the holder entry to appear and returns it.
    """

    _path = None  # read by JAX's logging; there is no single local path

    def __init__(
        self,
        backend: Backend,
        namespace: str,
        *,
        wait_s: float = WAIT_S,
        poll_s: float = POLL_S,
        clock: Any = time.monotonic,
        sleep: Any = time.sleep,
        device: str | Callable[[], str] | None = None,
    ):
        self.backend, self.namespace, self.device = backend, namespace, device
        self.wait_s, self.poll_s, self._clock, self._sleep = wait_s, poll_s, clock, sleep
        self._held: dict[str, Claim] = {}
        self._lock = threading.Lock()
        self.reset_attempt()

    def _name(self, key: str) -> str:
        """Where *key* lives: JAX's key, or, once its parts are known and the device is named, a key of our own.

        JAX folds in a fingerprint of the card as XLA describes it (``accelerator_config``), and that differs between host pools with the same card, driver and CUDA: two L40S pools on Modal give two values. Each pool would then compile and time its own picks, and a seed re-run in the other pool would land on other weights. So the key here swaps that part for the device as JAX names it (kind, compute capability, core count, device count), which holds one program per card model.
        """
        parts = self.parts.get(key)
        if callable(self.device):  # named on first use, by when the task has brought its backend up
            self.device = self.device()
        if self.device and parts and "accelerator_config" in parts:
            kept = [f"{p}={d}" for p, d in sorted(parts.items()) if p != "accelerator_config"]
            key = f"{key.rsplit('-', 1)[0]}-{_sha(chr(10).join([*kept, f'device={self.device}']).encode())}"
        return f"{self.namespace}/{_UNSAFE.sub('_', key)}"

    def get(self, key: str) -> bytes | None:
        with self._lock:
            self.parts[key], self._pending = self._pending, {}
        name = self._name(key)
        deadline = self._clock() + self.wait_s
        waited = False
        while True:
            if (data := self.backend.read(name)) is not None:
                self._note(key, "waited" if waited else "loaded", data)
                return data
            if (claim := self.backend.claim(name)) is not None:
                # The holder may have written and let go between our read and our claim.
                if (data := self.backend.read(name)) is not None:
                    claim.release()
                    self._note(key, "waited" if waited else "loaded", data)
                    return data
                with self._lock:
                    self._held[key] = claim
                return None
            if self._clock() > deadline:
                log.warning("compile cache: gave up waiting for %s after %.0f s; compiling here", key, self.wait_s)
                with self._lock:
                    self._unclaimed.add(key)
                return None
            waited = True
            self._sleep(self.poll_s)

    def put(self, key: str, value: bytes) -> None:
        name = self._name(key)
        try:
            if self.backend.write_once(name, value):
                self._note(key, "compiled", value)
            else:  # someone else's entry got there first: this attempt ran a program the cache doesn't hold
                self._note(key, "unshared", value)
                log.warning("compile cache: %s was already stored; this attempt compiled its own", key)
        finally:
            self._release(key)

    def clear(self) -> None:
        pass

    def _release(self, key: str) -> None:
        with self._lock:
            claim = self._held.pop(key, None)
            self._unclaimed.discard(key)
        if claim is not None:
            claim.release()

    def release_all(self) -> None:
        """Let go of every claim still held: a compile that failed never reaches :meth:`put`."""
        for key in list(self._held):
            self._release(key)

    # -- provenance --------------------------------------------------------

    def reset_attempt(self) -> None:
        with getattr(self, "_lock", contextlib.nullcontext()):
            self._seen: dict[str, tuple[str, str]] = {}
            self._unclaimed: set[str] = set()
            self.parts: dict[str, dict[str, str]] = {}
            self._pending: dict[str, str] = {}

    def note_part(self, part: str, digest: str) -> None:
        """One part of the key JAX is computing (its own debug log, :class:`_KeyParts`); the next :meth:`get` claims them."""
        with self._lock:
            self._pending.setdefault(part, digest)

    def _note(self, key: str, how: str, data: bytes) -> None:
        if how == "compiled" and key in self._unclaimed:
            how = "unclaimed"
        name = self._name(key)  # the entry, rather than JAX's key, which differs between host pools of one card
        with self._lock:
            self._seen[name] = (how, _sha(data))

    def summary(self) -> dict[str, Any] | None:
        """What this attempt took from the cache: a count per outcome and a digest over the entries.

        Outcomes: ``loaded`` (the entry was there), ``waited`` (another worker compiled it while this one waited), ``compiled`` (this worker compiled and stored it), ``unclaimed`` (compiled and stored after giving up on a stuck holder), and ``unshared`` (compiled, but an entry was already stored, so this attempt ran a program the cache does not hold). Attempts with the same ``digest`` ran the same compiled programs; the digest leaves out ``unshared`` programs, which no later attempt can load.
        """
        with self._lock:
            seen = dict(self._seen)
        if not seen:
            return None
        counts: dict[str, int] = {}
        for how, _ in seen.values():
            counts[how] = counts.get(how, 0) + 1
        shared = sorted(f"{k}={sha}" for k, (how, sha) in seen.items() if how != "unshared")
        out = {"namespace": self.namespace, **counts, "digest": _sha("\n".join(shared).encode())[:16]}
        if key_env := self.key_env():
            out["key_env"] = key_env
        return out

    def key_env(self) -> dict[str, list[str]]:
        """The parts of this attempt's keys that describe where it ran rather than what it compiled, each part's distinct digests.

        Two attempts that miss each other's entries differ in one of these (or in the program itself): ``backend version`` is the CUDA and driver versions, ``accelerator_config`` the card as XLA describes it, ``XLA flags`` the flags.
        """
        env: dict[str, set[str]] = {}
        with self._lock:
            for parts in self.parts.values():
                for part, digest in parts.items():
                    if part in _ENV_PARTS:
                        env.setdefault(part, set()).add(digest[:16])
        return {part: sorted(digests) for part, digests in sorted(env.items())}


_ENV_PARTS = ("jax_lib version", "backend version", "XLA flags", "accelerator_config")


class _KeyParts(logging.Handler):
    """Hands JAX's per-part key digests (logged at debug level by ``jax._src.cache_key``) to the installed cache."""

    _PREFIX = "get_cache_key hash of serialized "

    def __init__(self, cache: SharedCache):
        super().__init__(logging.DEBUG)
        self.cache = cache

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith(self._PREFIX):
            part, _, digest = msg.removeprefix(self._PREFIX).rpartition(": ")
            self.cache.note_part(part, digest)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _FileClaim:
    def __init__(self, fd: int):
        self._fd = fd

    def release(self) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)


class LocalBackend:
    """Entries as files under *root*, claims as ``flock`` locks beside them. The kernel drops a dead holder's lock, so no lease is needed."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, name: str) -> Path:
        return self.root / "entries" / name

    def read(self, name: str) -> bytes | None:
        try:
            return self._path(name).read_bytes()
        except FileNotFoundError:
            return None

    def write_once(self, name: str, data: bytes) -> bool:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(data)
        try:
            os.link(tmp, path)  # fails if the entry exists: write-once, atomically
        except FileExistsError:
            return False
        finally:
            tmp.unlink(missing_ok=True)
        return True

    def claim(self, name: str) -> _FileClaim | None:
        lock = self.root / "claims" / f"{name}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        return _FileClaim(fd)


class _LeaseClaim:
    """A claim in a ``modal.Dict``, renewed on a thread until released.

    The Dict has no compare-and-swap, so a lease that runs out is never taken over in place: the next worker claims the *next generation* of the key (``name#1``, ``name#2``, …) with ``put(skip_if_exists=True)``, which only one worker can win.
    """

    def __init__(self, claims: Any, key: str, owner: str, renew_s: float):
        self._claims, self._key, self._owner = claims, key, owner
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._renew, args=(renew_s,), daemon=True, name="compile-cache-lease")
        self._thread.start()

    def _renew(self, every: float) -> None:
        while not self._stop.wait(every):
            with contextlib.suppress(Exception):
                self._claims[self._key] = {"owner": self._owner, "at": time.time()}

    def release(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self._claims.pop(self._key)


class DictClaims:
    """Lease-based claims over any mapping with ``put(key, value, skip_if_exists=)`` (a ``modal.Dict``)."""

    def __init__(self, claims: Any, *, lease_s: float = LEASE_S, renew_s: float = RENEW_S, now: Any = time.time):
        self._claims, self.lease_s, self.renew_s, self._now = claims, lease_s, renew_s, now
        self.owner = f"{os.environ.get('MODAL_TASK_ID', os.uname().nodename)}:{os.getpid()}:{uuid.uuid4().hex[:6]}"

    def claim(self, name: str) -> _LeaseClaim | None:
        for gen in range(1000):
            key = f"{name}#{gen}"
            if self._claims.put(key, {"owner": self.owner, "at": self._now()}, skip_if_exists=True):
                return _LeaseClaim(self._claims, key, self.owner, self.renew_s)
            held = self._claims.get(key)
            if held is None:  # released (or expired) since our put: try this generation again next poll
                return None
            if self._now() - held.get("at", 0.0) <= self.lease_s:
                return None  # a live holder
            # a lapsed lease: its holder stopped renewing, so move on to the next generation
        return None


class ModalBackend:
    """Entries on a Modal Volume, read and written through its API; claims in a ``modal.Dict``.

    The Volume API sees committed data from any container without a reload, and an upload without ``force`` is refused for a path that exists, which is the write-once rule enforced by Modal itself.
    """

    def __init__(self, volume: Any, claims: Any, **lease: Any):
        self._volume = volume
        self._claims = DictClaims(claims, **lease)

    @classmethod
    def from_names(cls, environment: str | None) -> ModalBackend:
        import modal

        return cls(
            modal.Volume.from_name(VOLUME_NAME, environment_name=environment, create_if_missing=True),
            modal.Dict.from_name(CLAIMS_DICT_NAME, environment_name=environment, create_if_missing=True),
        )

    def read(self, name: str) -> bytes | None:
        try:
            return b"".join(self._volume.read_file(name))
        except FileNotFoundError:
            return None

    def write_once(self, name: str, data: bytes) -> bool:
        try:
            with self._volume.batch_upload() as batch:
                batch.put_file(io.BytesIO(data), name)
        except FileExistsError:
            return False
        return True

    def claim(self, name: str) -> _LeaseClaim | None:
        return self._claims.claim(name)


# ---------------------------------------------------------------------------
# Installing it in JAX
# ---------------------------------------------------------------------------

_installed: SharedCache | None = None


def install(backend: Backend, namespace: str, **kw: Any) -> SharedCache:
    """Make a :class:`SharedCache` the process-wide JAX compilation cache, once per process.

    Sets the thresholds so every program goes through the cache (JAX skips small or quick compiles by default, and those are autotuned too), and turns off XLA's own per-fusion autotune cache, which would write files of its own beside the entries: the entry already holds the picks.
    """
    global _installed
    if _installed is not None:
        return _installed
    import jax
    from jax._src import compilation_cache as cc

    jax.config.update("jax_enable_compilation_cache", True)
    # Only consulted as a flag that the cache is on, and as the parent for XLA's own caches (off below).
    jax.config.update("jax_compilation_cache_dir", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mini-jax-cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
    cache = SharedCache(backend, namespace, device=_device_name, **kw)
    # JAX logs each part of a key as it hashes it, at debug level only; kept out of the root handlers.
    key_log = logging.getLogger("jax._src.cache_key")
    key_log.setLevel(logging.DEBUG)
    key_log.propagate = False
    key_log.addHandler(_KeyParts(cache))
    cc.reset_cache()
    with cc._cache_initialized_mutex:
        cc._cache = cache  # ty: ignore[invalid-assignment]
        cc._cache_initialized = True
    _installed = cache
    return cache


def _device_name() -> str:
    """The card as JAX names it, with the device count: what one program per card model is keyed on."""
    import jax

    dev = jax.devices()[0]
    attrs = (str(getattr(dev, a, "")) for a in ("platform", "device_kind", "compute_capability", "core_count"))
    return ":".join([*attrs, str(len(jax.devices()))])


def installed() -> SharedCache | None:
    return _installed


@contextlib.contextmanager
def attempt(backend_factory: Any) -> Generator[SharedCache | None]:
    """Install the shared cache for one task attempt, if ``MINI_COMPILE_CACHE`` turns it on, and yield it (or ``None``).

    *backend_factory* builds the backend on first use (``None``: no shared cache on this worker), so a worker that never turns the cache on never touches Modal. The yielded cache collects this attempt's provenance (:meth:`SharedCache.summary`), and any claim a failed compile left behind is released on the way out.
    """
    if backend_factory is None or (namespace := namespace_from_env()) is None:
        yield None
        return
    cache = install(backend_factory(), namespace)
    cache.reset_attempt()
    try:
        yield cache
    finally:
        cache.release_all()
