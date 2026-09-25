---
status: open
tags: [publishing, ci]
opened: 2026-09-25
---
# The publish check does not re-run when `skip-publish-check` is added

The lint workflow skips the publish check when the PR carries the `skip-publish-check` label, but it triggers on `pull_request` with the default types (opened, synchronize, reopened). A label added after a push does not start a run, and a re-run of the failed job reuses the original event payload, without the label. So a PR labelled after its last push stays red until something else is pushed. PR #218 hit this: the label went on a minute after the run started.

Two ways out. Add `types: [opened, synchronize, reopened, labeled, unlabeled]` under `pull_request` in `.github/workflows/lint-check.yml`, so labelling re-runs the checks (the concurrency group cancels the stale run). Or add the label before the push in the agent workflow. The first is one line and removes the ordering rule.
