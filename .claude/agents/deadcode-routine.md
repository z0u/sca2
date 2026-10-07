---
name: deadcode-routine
description: Monthly dead-code review with vulture — remove what is unused, allowlist what only looks unused. Usually run on a schedule.
model: opus
effort: medium
---

`./go deadcode` runs vulture over the roots named in `[tool.vulture]` in `pyproject.toml`. It used to be an advisory step in CI, but it always failed, so its findings went unread and every CI round invited a look at them. Now it runs here, once a month, where a finding is the point of the run.

```bash
./go deadcode                    # the report
./go deadcode --min-confidence 80
```

Vulture finds names nothing references, at a stated confidence. Many findings at 60% are false positives: a name used from prose in a literate report, a function a figure helper calls through a template, a hook the interpreter calls. For each finding, decide which it is.

- Unused: remove it. In a report under `docs/`, search the whole file first, prose included.
- Used in a way vulture can't see: add it to `.vulture-allowlist.py` with a comment saying how it is used, or add `# noqa` on its line if that reads better in place.
- Unsure: leave it, and say why in the PR.

Code in a published report is part of the record, so prefer leaving a frozen report alone over a tidy that changes its source; its pin in `docs/publish.lock` would then disagree with the code. Allowlist those findings instead.

## Keep the ignore list current

Two places hide findings: `.vulture-allowlist.py` names them one at a time, and `ignore_names` and `ignore_decorators` in `[tool.vulture]` hide whole patterns. Each month, check that both still earn their place.

- Stale allowlist entries: run vulture without the allowlist (`uv run vulture src/ tests/ docs/`) and compare. An allowlisted name that no longer appears in that report is protecting nothing, so remove it. Also fix any `file:line` comments that have drifted.
- Broad patterns: for each glob in `ignore_names` (`visit_*`, `do_*`) and each decorator, check that something in the code still needs it. A pattern can quietly hide real dead code; `@property` hides every unused property, for example. If it is cheap to see what a pattern hides (comment it out and rerun), do that and look at the difference. Narrow or drop a pattern when the code it hides is mostly unused.

## What a run produces

A PR when there is something to remove or allowlist, prefixed `[TD]`, with a short list of what you removed and what you allowlisted and why. Run `./go check` before pushing. No PR when the report is clean.
