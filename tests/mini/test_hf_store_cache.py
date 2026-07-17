"""Offline unit tests for the HFStore warm cache — no network, fake ``api``.

These exercise the local cache path (``_local_blob``) with a stub bucket API, so
they run without a bucket or token (unlike the integration suite in
``test_hf_store.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini.hf_store import HFStore
from mini.store import LocalStore, _cas_key


class FakeApi:
    """Stands in for ``HfApi``: serves blob bytes from a dict, or fails a download."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs  # cas-key -> bytes
        self.fail_next = False
        self.downloads = 0

    def download_bucket_files(self, cas, files):  # noqa: ANN001 — mirrors HfApi's signature
        for key, dest in files:
            self.downloads += 1
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if self.fail_next:
                # Simulate a Ctrl-C mid-download: the tool has created the file but
                # written nothing before the interrupt lands.
                dest.write_bytes(b"")
                self.fail_next = False
                raise KeyboardInterrupt("interrupted download")
            dest.write_bytes(self.blobs[key])


def _store(tmp_path: Path, blobs: dict[str, bytes]) -> tuple[HFStore, FakeApi]:
    store = HFStore("ns/bucket", cache=LocalStore(tmp_path / "cache"))
    api = FakeApi(blobs)
    store._api = api  # inject the stub, bypassing the real HfApi
    return store, api


def test_interrupted_download_leaves_no_poisoned_cache_entry(tmp_path: Path):
    payload = b"the real checkpoint bytes"
    sha = "a" * 64
    blobs = {_cas_key(sha): payload}
    store, api = _store(tmp_path, blobs)

    # First pull is interrupted after creating an empty file.
    api.fail_next = True
    with pytest.raises(KeyboardInterrupt):
        store._local_blob(sha)

    # The regression guard: no 0-byte file is left at the final cache path, so the
    # cache doesn't "hit" on a corrupt entry — and no stray temp file lingers either.
    assert not store._cache._blob_path(sha).exists()
    assert list((tmp_path / "cache").rglob("*.tmp.*")) == []

    # A retry re-pulls and serves the real bytes.
    out = store._local_blob(sha)
    assert out.read_bytes() == payload


def test_second_read_hits_the_warm_cache(tmp_path: Path):
    payload = b"cached once, served twice"
    sha = "b" * 64
    store, api = _store(tmp_path, {_cas_key(sha): payload})

    assert store._local_blob(sha).read_bytes() == payload
    assert store._local_blob(sha).read_bytes() == payload
    assert api.downloads == 1  # the second read never touches the bucket
