---
status: done
tags: [storage, mini]
opened: 2026-10-12
closed: 2026-10-12
---
# Moving to huggingface_hub 2

`pyproject.toml` asks for `huggingface-hub>=1.19`, so a plain `uv lock --upgrade` would take us from 1.26.0 to 2.2.0. The weekly dependency review on 2026-10-12 held it back, because one change in 2.0 reaches `mini.hf_store` and would fail without any error.

In 2.0 the library moved its HTTP stack from `httpx` to `httpx2`, so its transport errors are now `httpx2` types. `_is_transient` in `src/mini/hf_store.py` imports `httpx` and checks `isinstance(exc, httpx.TimeoutException | httpx.NetworkError | httpx.RemoteProtocolError)`. `datasets` 5.0.1 still depends on `httpx`, so that import would keep working, but the check would stop matching. Timeouts and dropped connections would then skip the outer retry ring added in [hf-store-retry-transient-5xx](hf-store-retry-transient-5xx.md), and only the library's own shorter backoff would cover them. 5xx responses arrive as `HfHubHTTPError` and would still be retried. The offline tests in `tests/mini/test_hf_store_retry.py` only raise `HfHubHTTPError`, so they wouldn't notice.

The release notes suggest importing `httpx` from `huggingface_hub.utils` in code that has to work on both majors, but 1.26.0 doesn't have that name. So the fix is either a lower bound of 2 with a direct `httpx2` import, or a guarded import that checks against both.

Two smaller things to check on the way:

- `httpx2` uses the OS trust store by default. `SSL_CERT_FILE` and `SSL_CERT_DIR` are still honored, but sessions that reach the Hub through a proxy with its own CA should get a quick test.
- 2.0 also removed several deprecated APIs (`upload_large_folder`, `huggingface-cli`, and others). `mini` doesn't call any of those, but the bucket calls (`batch_bucket_files`, `download_bucket_files`, `get_bucket_paths_info`, `list_bucket_tree`, `sync_bucket`) aren't mentioned in the release notes either way, so the gated `pytest -m hf` suite against the dev pair is the real test.

Nothing in 2.x is something we need yet, so this can wait until a reason comes up, or until a `datasets` release requires it.

## Resolution

Taken the same day. `datasets` 5.1.0 had dropped its own `httpx` dependency and accepts hub 2, so raising both lower bounds (`huggingface-hub>=2`, `datasets>=5.1`) removed `httpx` and `httpcore` from the lock completely. `_is_transient` now imports `httpx2` with no fallback, and `test_a_timeout_is_retried` raises an `httpx2.ReadTimeout` so the transport case has a test of its own. The `pytest -m hf` suite passed against the dev pair through the session proxy, so the new trust-store default works there.
