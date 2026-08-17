---
status: partial
tags: [agents, testing, tooling]
opened: 2026-08-14
---
# Agent worktrees live inside the repo, and nothing clears them out

Agents get their own git worktree under `.claude/worktrees/`, which is inside the repo rather than beside it. That placement has two consequences: tools that walk the tree find a whole second checkout, and nothing removes the checkout once its branch is finished.

**Collection, now fixed.** `pytest` walked into `.claude/worktrees/` and collected a second copy of `tests/`, reporting four `ImportPathMismatchError`s on every `./go check`. The cause is that setting `norecursedirs` in `pyproject.toml` *replaces* pytest's default list rather than extending it, and `.*` is the default entry that keeps it out of dotted directories — so listing `node_modules` and `.venv` quietly opened `.claude` up. Fixed by putting `.*` back at the head of the list. The general lesson is worth carrying to any other tool configured with a directory-exclusion list: check whether the setting extends the defaults or replaces them. `ruff`, `ty`, `vulture`, and the `marimo check` hook each have their own, and none of them have been looked at for this.

**Lifecycle, still open.** Nothing removes a worktree when its branch is done. The one found on 2026-08-14 dated from 08-05 — ten days after its PR (#74) had merged — and it was clean and level with `main` the whole time. It just sat there, and the only reason it surfaced was the collection noise above, which is now gone. So the next one will be quieter, not rarer.

`git worktree remove` is the manual answer, but something has to notice first. Candidates, roughly cheapest first: a line in `AGENTS.md` telling an agent to remove its own worktree on the way out; a warning in `./go check` when a worktree's branch is merged; or a `./go` subcommand that prunes them, which the housekeeping routine could run on its pass. The `AGENTS.md` note only fires when a session ends tidily, and a stranded worktree is evidence that one didn't — so a prune command looks like the better shape, with the routine as the thing that remembers to call it.

Worth checking at the same time whether the worktrees need to be inside the repo at all. If they can sit in a sibling directory, both halves of this item stop being possible.
