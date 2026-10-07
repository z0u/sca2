---
name: steward
description: How to handle activity events on a PR you opened or drive in this repo — which wakes need a reply and which need nothing.
---

Most wakes on a PR here need no reply. Before acting on one, decide which kind it is.

- CI passed (a successful check-suite rollup): nothing to do. End the turn without a message unless you were waiting on this result to continue a task.
- CI failed: a real finding. The `Reports published` step fails when a report changed without `./go publish`, and its job summary names the command to run.

PR previews arrive as commit statuses (`preview`, `preview: <report>`) set by the `Site` workflow, and they don't wake you. A red `preview` status means the preview failed to build; its Details link goes to the run log. That is usually this PR's report failing to render, so look at it on your next push.

Dead-code detection is not part of CI. It runs monthly (`.claude/agents/deadcode-routine.md`), so a vulture finding is never a reason to touch a PR.
