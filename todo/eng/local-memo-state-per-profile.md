---
status: open
tags: [mini, storage]
opened: 2026-10-10
---
# Keep local memo state apart per storage profile

A local run keeps its memo records under `.mini/<name>/` (`mini.runs.data_root`), whatever `MINI_PROFILE` is. So an experiment run first under `MINI_PROFILE=dev` and then on production reads as already done on the second run: every task is skipped, nothing is written to the production bucket, and the report then fails to load its results ("not published"). This happened with the equals-evidence experiment, and moving `.mini/equals-evidence/` aside was the workaround. On Modal the profile gets its own `modal-environment`, which keeps the two apart; the local apparatus has no counterpart.

A fix could put a profile's local state under its own directory (say `.mini/<profile>/<name>/`), matching how `publish.<profile>.lock` is kept apart.

A related small one: `tests/mini/test_store.py::test_store_for_threads_publish_repo_into_the_hfstore` fails where `MINI_CACHE_DIR` is set in the environment (the project cloud containers set it to the shared folder), since the test doesn't clear it. CI doesn't set it, so it passes there.
