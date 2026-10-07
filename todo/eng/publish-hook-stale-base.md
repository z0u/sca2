---
status: done
tags: [publishing, tooling]
opened: 2026-09-29
closed: 2026-10-04
---
# The pre-push publish check misfires on a stale `origin/main` after a merge that skipped publishing

`scripts/unpublished_reports.py` says a stale base is self-correcting: the extra reports it sees as changed had their pins moved in the same range. That holds only when every merge in the range published. A PR merged without publishing (under `skip-publish-check`, say) changes a report directory without moving its pin, so a session whose `origin/main` predates that merge sees the report as changed by its own branch.

On 2026-09-29 a fresh container had `origin/main` behind #229 (the ty upgrade, which edited `report.py` of ex-2.1.7 to ex-2.1.10 and left `docs/publish.lock` as it was). The pre-push hook then blocked a push that touched no report, naming those four. A `git fetch origin main` cleared it.

A fix could fetch the base in the hook before comparing (soft, like the rest of the hook: skip the fetch when offline), or have the session-start hook fetch `origin/main`.

## Notes

- 2026-10-04: the pre-push hook now fetches `origin main` (20-second timeout, skipped on failure) before resolving the base. `test_a_stale_base_misfires_past_a_merge_that_skipped_publishing` records the case, and the script docstring and `eng/publishing.md` no longer call a stale base self-correcting without qualification. The session-start fetch was the other option; the hook was the better place, since a push needs the network anyway and the fetch then happens right before the comparison.
