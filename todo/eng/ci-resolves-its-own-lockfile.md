---
status: open
tags: [ci, tooling]
opened: 2026-08-24
---
# CI re-resolves the lockfile instead of failing on a stale one

`./go install` runs `uv sync --all-groups --no-group cuda`, with neither `--locked` nor `--frozen`, and CI runs `./go install`. Plain `uv sync` updates `uv.lock` when it disagrees with `pyproject.toml`, so a dependency edit pushed without a re-lock doesn't fail the build — CI quietly resolves its own lockfile, tests against that, and reports green. Whatever gets merged is a `pyproject.toml` the committed lock no longer describes, and the next person to run `uv sync` locally picks up a third resolution.

Confirmed rather than inferred, on 2026-08-24: bump one floor in `pyproject.toml` and leave the lock alone, and `uv lock --check` reports it stale and `uv sync --locked` errors with "The lockfile at `uv.lock` needs to be updated" — but plain `uv sync` installs the new version and rewrites `uv.lock` on the spot, no warning.

The relative cooldown is what makes this more than tidiness. `exclude-newer = "3 days"` resolves against wall-clock time at resolution, so a CI re-lock weeks after the pyproject edit sees a different eligible set than the developer who made it: newer releases, and a lockfile nobody reviewed. That is the same reproducibility claim the cooldown is there to make, arriving through the back door.

The fix is one flag, so the question is which. `--locked` fails the build and names the remedy, which reads right for CI and costs a red build on the first forgotten `uv lock` — the point. `--frozen` installs from the lock and ignores `pyproject.toml` entirely, which hides the drift rather than reporting it. But `install.sh` is shared with developers, where a hard failure on a half-finished dependency edit would be irritating, so it probably wants to be conditional on `$CI` (or a `--locked` flag on `./go install` that only the workflow passes) rather than unconditional.

The other two workflows have it too, and they don't go through `install.sh`: `pr-preview.yml:42` and `publish-docs.yml:41` both run a bare `uv sync --group pages`. So the fix is three edits, not one — and those two are CI-only, so they can take the flag unconditionally.
