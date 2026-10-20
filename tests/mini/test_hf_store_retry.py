"""Offline unit tests for the HFStore's retry ring — no network, fake ``api``.

Every bucket call is wrapped so that a server-side blip doesn't take a run down with it (see ``_retrying`` in ``mini.hf_store``). These check the three things that ring has to get right: a 5xx comes back, a 4xx doesn't, and the number of attempts is bounded. The waits are recorded rather than slept, so the suite stays fast.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterator

import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError

import mini.hf_store as hf_store
from mini.hf_store import _RETRY_WAITS, HFStore
from mini.store import LocalStore, _cas_key

SHA = "a" * 64
PAYLOAD = b"the blob bytes"


def http_error(status: int) -> HfHubHTTPError:
    """What ``huggingface_hub`` raises once ``hf_raise_for_status`` sees a bad response."""
    request = httpx2.Request("POST", "https://bucket.invalid/api/buckets/ns/bucket/batch")
    return HfHubHTTPError(f"HTTP {status}", response=httpx2.Response(status, request=request))


class FlakyApi:
    """Stands in for ``HfApi``, failing the first *fail_times* calls of each kind with *error*."""

    def __init__(self, *, fail_times: int = 0, error: HfHubHTTPError | Exception | None = None):
        self.error = error or http_error(500)
        self.remaining = {"batch": fail_times, "paths_info": fail_times, "download": fail_times}
        self.calls = {"batch": 0, "paths_info": 0, "download": 0}

    def _attempt(self, kind: str) -> None:
        self.calls[kind] += 1
        if self.remaining[kind] > 0:
            self.remaining[kind] -= 1
            raise self.error

    def batch_bucket_files(self, bucket: str, **ops: Any) -> None:
        self._attempt("batch")

    def get_bucket_paths_info(self, bucket: str, paths: list[str]) -> Iterator[Any]:
        # A generator, as the real one is: the request only goes out on iteration,
        # which is what the retry has to wrap.
        self._attempt("paths_info")
        for p in paths:
            if p == _cas_key(SHA):
                yield type("BucketFile", (), {"path": p, "xet_hash": "xet-1234"})()

    def download_bucket_files(self, bucket: str, files: list[tuple[Any, str]]) -> None:
        self._attempt("download")
        for _, dest in files:
            Path(dest).write_bytes(PAYLOAD)


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record what the ring would have slept, instead of sleeping it."""
    recorded: list[float] = []
    monkeypatch.setattr("mini.hf_store._sleep_before_retry", recorded.append)
    return recorded


def store(tmp_path: Path, api: FlakyApi) -> HFStore:
    s = HFStore("ns/bucket", cache=LocalStore(tmp_path / "cache"))
    s._api = api  # inject the stub, bypassing the real HfApi
    return s


def test_a_transient_server_error_on_a_write_is_retried(tmp_path: Path, waits: list[float]):
    api = FlakyApi(fail_times=2)
    art = store(tmp_path, api).put(PAYLOAD, name="blob")

    assert art.size == len(PAYLOAD)
    # One paths-info (the remote ``has`` probe) and one batch commit, each of which
    # failed twice before landing.
    assert api.calls == {"paths_info": 3, "batch": 3, "download": 0}
    assert waits == [10.0, 20.0, 10.0, 20.0]


def test_a_permission_error_is_not_retried(tmp_path: Path, waits: list[float]):
    # A 403 is final: the token can't write, and coming back in ten seconds only
    # delays an error the caller needs now.
    api = FlakyApi(fail_times=1, error=http_error(403))
    with pytest.raises(HfHubHTTPError):
        store(tmp_path, api).put(PAYLOAD, name="blob")

    assert api.calls["paths_info"] == 1
    assert waits == []


def test_a_persistent_server_error_gives_up_after_a_bounded_number_of_attempts(tmp_path: Path, waits: list[float]):
    api = FlakyApi(fail_times=99)
    with pytest.raises(HfHubHTTPError):
        store(tmp_path, api).put(PAYLOAD, name="blob")

    # Four attempts in all, and the error the caller sees is the last one's.
    assert api.calls["paths_info"] == len(_RETRY_WAITS) + 1 == 4
    assert waits == list(_RETRY_WAITS)


def test_the_dedup_race_is_retried_although_it_is_a_4xx(tmp_path: Path, waits: list[float]):
    # Two processes committing the same Xet hash at once race in the bucket's dedup
    # and one comes back 422 — transient, despite the 4xx. See _RETRY_STATUSES.
    api = FlakyApi(fail_times=1, error=http_error(422))
    art = store(tmp_path, api).put(PAYLOAD, name="blob")

    assert art.size == len(PAYLOAD)
    assert api.calls == {"paths_info": 2, "batch": 2, "download": 0}
    assert waits == [10.0, 10.0]


def test_a_transient_error_on_a_download_is_retried(tmp_path: Path, waits: list[float]):
    api = FlakyApi(fail_times=1)
    s = store(tmp_path, api)  # a cold cache, so the pull really goes to the bucket

    s._pull_blobs([SHA])

    assert s._cache._blob_path(SHA).read_bytes() == PAYLOAD
    assert api.calls == {"paths_info": 2, "batch": 0, "download": 2}
    assert waits == [10.0, 10.0]


def test_a_timeout_is_retried(tmp_path: Path, waits: list[float]):
    # Transport failures reach us as the HTTP library's own types rather than as an
    # HfHubHTTPError, so this is the case that breaks quietly if the library moves.
    request = httpx2.Request("POST", "https://bucket.invalid/api/buckets/ns/bucket/batch")
    api = FlakyApi(fail_times=1, error=httpx2.ReadTimeout("timed out", request=request))
    art = store(tmp_path, api).put(PAYLOAD, name="blob")

    assert art.size == len(PAYLOAD)
    assert api.calls == {"paths_info": 2, "batch": 2, "download": 0}
    assert waits == [10.0, 10.0]


def test_a_hung_bucket_write_is_abandoned_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
):
    release = threading.Event()

    class HangingOnceApi(FlakyApi):
        def batch_bucket_files(self, bucket: str, **ops: Any) -> None:
            self.calls["batch"] += 1
            if self.calls["batch"] == 1:
                release.wait(5)  # hangs past the deadline, as the Modal commits did

    monkeypatch.setattr(hf_store, "_COMMIT_TIMEOUT_S", 0.05)
    api = HangingOnceApi()
    store(tmp_path, api).set_ref("x", store(tmp_path, api).put(PAYLOAD, name="blob"))
    release.set()
    assert api.calls["batch"] == 3  # the put's commit twice (the first abandoned), then the ref's
    assert len(waits) == 1
