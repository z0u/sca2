---
status: done
tags: [mini, storage]
opened: 2026-10-10
closed: 2026-11-01
---
# Keep local memo state apart per storage profile

A local run keeps its memo records under `.mini/<name>/` (`mini.runs.data_root`), whatever `MINI_PROFILE` is. So an experiment run first under `MINI_PROFILE=dev` and then on production reads as already done on the second run: every task is skipped, nothing is written to the production bucket, and the report then fails to load its results ("not published"). This happened with the equals-evidence experiment, and moving `.mini/equals-evidence/` aside was the workaround. On Modal the profile gets its own `modal-environment`, which keeps the two apart; the local apparatus has no counterpart.

A fix could put a profile's local state under its own directory (say `.mini/<profile>/<name>/`), matching how `publish.<profile>.lock` is kept apart.

A related small one: `tests/mini/test_store.py::test_store_for_threads_publish_repo_into_the_hfstore` fails where `MINI_CACHE_DIR` is set in the environment (the project cloud containers set it to the shared folder), since the test doesn't clear it. CI doesn't set it, so it passes there.

## Notes

**2026-11-01, tech-debt routine** — Done. `mini.runs.experiments_root()` is `.mini/` under the base configuration and `.mini/profiles/<profile>/` under a profile, and every local reader and writer of experiment state goes through it: the CLI verbs and the `.app` stamp, `LocalApparatus`, `mini gc`, `project_store()`, and the Modal apparatus's client-side store. A subdirectory rather than `.mini/<profile>/<name>/` keeps a profile name from colliding with an experiment name. Dev state written before this change still sits at `.mini/<name>/` and now reads as production state, so move or delete it by hand where it exists. The `MINI_CACHE_DIR` test leak had already been fixed by the autouse fixture in `tests/conftest.py`. Covered by `tests/mini/test_cli.py::test_local_state_is_kept_apart_per_storage_profile`.
