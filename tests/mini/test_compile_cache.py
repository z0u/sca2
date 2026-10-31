"""The shared compilation cache: one compile per program across workers, write-once entries, and what each attempt loaded."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from unittest.mock import ANY

import numpy as np
import pytest

from mini import compile_cache
from mini.compile_cache import DictClaims, LocalBackend, SharedCache


def _worker(root: Path, **kw) -> SharedCache:
    """A cache as one worker sees it; two of these over one root are two workers sharing it."""
    return SharedCache(LocalBackend(root), "ns", poll_s=0.01, **kw)


def test_first_worker_compiles_and_the_next_loads(tmp_path):
    a, b = _worker(tmp_path), _worker(tmp_path)
    assert a.get("jit_step-1") is None  # a miss: `a` holds the claim and compiles
    a.put("jit_step-1", b"program")
    assert b.get("jit_step-1") == b"program"
    assert a.summary() == {"namespace": "ns", "compiled": 1, "digest": ANY}
    assert b.summary() == {"namespace": "ns", "loaded": 1, "digest": a.summary()["digest"]}


def test_a_cold_fan_out_waits_for_the_one_compile(tmp_path):
    """While one worker holds the claim, another waits for its entry instead of compiling its own."""
    a, b = _worker(tmp_path), _worker(tmp_path)
    assert a.get("k") is None
    got: list[bytes | None] = []
    waiter = threading.Thread(target=lambda: got.append(b.get("k")))
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive()  # still waiting: the program is claimed but not stored
    a.put("k", b"program")
    waiter.join(5)
    assert got == [b"program"]
    assert b.summary() == {"namespace": "ns", "waited": 1, "digest": a.summary()["digest"]}


def test_a_failed_compile_hands_the_claim_on(tmp_path):
    """A compile that raises never reaches `put`; letting go at the end of the attempt frees the program for the next worker."""
    a, b = _worker(tmp_path), _worker(tmp_path)
    assert a.get("k") is None
    a.release_all()
    assert b.get("k") is None  # `b` claims it and compiles
    b.put("k", b"program")
    assert b.summary()["compiled"] == 1


def test_entries_are_written_once(tmp_path):
    """A second store of the same program is refused, and the attempt that compiled it is marked as running a program the cache doesn't hold."""
    a, b = _worker(tmp_path), _worker(tmp_path)
    a.put("k", b"first")
    b.put("k", b"second")
    assert LocalBackend(tmp_path).read("ns/k") == b"first"
    assert b.summary() == {"namespace": "ns", "unshared": 1, "digest": ANY}
    assert b.summary()["digest"] != a.summary()["digest"]


def test_a_stuck_holder_is_given_up_on(tmp_path):
    """Past the wait, a worker compiles on its own, and the record says it compiled without the claim."""
    clock = iter(range(0, 10_000, 10))
    a = _worker(tmp_path)
    b = _worker(tmp_path, wait_s=30, clock=lambda: next(clock), sleep=lambda s: None)
    assert a.get("k") is None
    assert b.get("k") is None
    b.put("k", b"program")
    assert b.summary() == {"namespace": "ns", "unclaimed": 1, "digest": ANY}


def test_namespaces_do_not_share_entries(tmp_path):
    a = SharedCache(LocalBackend(tmp_path), "probe-1")
    a.put("k", b"program")
    assert SharedCache(LocalBackend(tmp_path), "probe-2").get("k") is None


@pytest.mark.parametrize(
    ("raw", "namespace"), [(None, None), ("", None), ("0", None), ("off", None), ("1", "1"), ("shared", "shared")]
)
def test_namespace_from_env(monkeypatch, raw, namespace):
    if raw is None:
        monkeypatch.delenv(compile_cache.ENV, raising=False)
    else:
        monkeypatch.setenv(compile_cache.ENV, raw)
    assert compile_cache.namespace_from_env() == namespace


class _FakeDict(dict):
    """The two calls `DictClaims` makes on a `modal.Dict`."""

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def test_lease_claims_move_to_the_next_generation_when_a_holder_stops_renewing():
    now = [1000.0]
    d = _FakeDict()
    a = DictClaims(d, lease_s=90, renew_s=3600, now=lambda: now[0])
    b = DictClaims(d, lease_s=90, renew_s=3600, now=lambda: now[0])
    held = a.claim("ns/k")
    assert held is not None and set(d) == {"ns/k#0"}
    now[0] += 60
    assert b.claim("ns/k") is None  # within the lease: a live holder
    now[0] += 60
    taken = b.claim("ns/k")  # lapsed: `b` takes generation 1, leaving generation 0 to expire
    assert taken is not None and set(d) == {"ns/k#0", "ns/k#1"}
    taken.release()
    held.release()
    assert d == {}


def test_jax_loads_a_program_another_worker_compiled(tmp_path, monkeypatch):
    """End to end on the CPU: once JAX's cache is ours, a second compile of the same function loads the stored entry."""
    jax = pytest.importorskip("jax")
    from jax._src import compilation_cache as cc

    saved = {k: getattr(jax.config, k) for k in (
        "jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes", "jax_persistent_cache_enable_xla_caches",
    )}  # fmt: skip
    monkeypatch.setattr(compile_cache, "_installed", None)
    try:
        cache = compile_cache.install(LocalBackend(tmp_path), "ns")

        def f(x):
            return (x * 3.0).sum()

        x = np.arange(4.0, dtype=np.float32)  # a host array, so `f` is the only program compiled
        first = float(jax.jit(f)(x))
        # One program, so each part of the key that describes the machine has one value.
        key_env = dict.fromkeys(["XLA flags", "accelerator_config", "backend version", "jax_lib version"], [ANY])
        assert cache.summary() == {"namespace": "ns", "compiled": 1, "digest": ANY, "key_env": key_env}
        cache.reset_attempt()
        jax.clear_caches()  # a new worker process starts with nothing in memory
        again = float(jax.jit(f)(x))
        assert cache.summary() == {"namespace": "ns", "loaded": 1, "digest": ANY, "key_env": key_env}
        assert again == first == 18.0
    finally:
        key_log = logging.getLogger("jax._src.cache_key")
        key_log.handlers.clear()
        key_log.setLevel(logging.NOTSET)
        key_log.propagate = True
        cc.reset_cache()
        for k, v in saved.items():
            jax.config.update(k, v)


def _task_using_the_cache(x):
    cache = compile_cache.installed()
    assert cache is not None
    if cache.get("jit_step-1") is None:
        cache.put("jit_step-1", b"program")
    return x


@pytest.mark.parametrize(("env", "recorded"), [("shared", True), ("off", False)])
def test_the_settled_record_says_what_the_attempt_took(tmp_path, monkeypatch, env, recorded):
    from mini._taskworker import execute_task
    from mini.memo import MemoStore, task_key_parts

    monkeypatch.setattr(compile_cache, "_installed", SharedCache(LocalBackend(tmp_path / "cc"), "shared"))
    monkeypatch.setenv(compile_cache.ENV, env)
    store = MemoStore(tmp_path / "exp")
    key, parts = task_key_parts(_task_using_the_cache, (1,))
    gen = store.mark_running(_task_using_the_cache, key, parts, expect_gen=None)
    execute_task(store, key, _task_using_the_cache, (1,), [], gen=gen, compile_backend=lambda: LocalBackend(tmp_path))
    rec = store.record(key)
    assert rec["state"] == "done"
    expected = {"namespace": "shared", "compiled": 1, "digest": ANY} if recorded else None
    assert rec.get("compile_cache") == expected  # off: the attempt neither resets nor reports the cache


def _keyed(cache: SharedCache, accelerator: str) -> None:
    """Feed the cache the parts of a key, as JAX logs them while it computes `jit_step-<hash>`."""
    for part, digest in [("computation", "c0ffee"), ("backend version", "cuda"), ("accelerator_config", accelerator)]:
        cache.note_part(part, digest)


@pytest.mark.parametrize(("device_b", "shared"), [("gpu:NVIDIA L40S:8.9:142:1", True), ("gpu:NVIDIA L4:8.9:58:1", False)])
def test_hosts_of_one_card_model_share_a_program(tmp_path, device_b, shared):
    """Two host pools with the same card describe it to XLA differently; the key keeps one program per card model."""
    a = SharedCache(LocalBackend(tmp_path), "ns", device="gpu:NVIDIA L40S:8.9:142:1")
    b = SharedCache(LocalBackend(tmp_path), "ns", device=device_b, poll_s=0.01)
    _keyed(a, "aws-pool")
    assert a.get("jit_step-aws") is None
    a.put("jit_step-aws", b"program")
    _keyed(b, "oci-pool")
    got = b.get("jit_step-oci")
    assert got == (b"program" if shared else None)
    if shared:  # one program, so one digest, whatever JAX called it on each host
        assert b.summary()["digest"] == a.summary()["digest"]
