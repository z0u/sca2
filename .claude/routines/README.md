# Routines

Scheduled jobs that run in the project's **Maintenance routines** thread. Each scheduled task's stored prompt is short: it names a file here and asks the thread to read this README and that file from `origin/main`. So a change to a routine is a PR to this folder, and it takes effect on the next run after merge. Schedules, and which thread each task wakes, live in the scheduled tasks themselves.

| File | Schedule (UTC) | Helper |
|---|---|---|
| `advisory.md` | Tue–Sun 16:07 | general-purpose, opus |
| `deps.md` | Mon 16:00 | the `deps-routine` agent spec |
| `tech-debt.md` | Sun & Tue 16:00 | general-purpose, opus |
| `housekeeping.md` | Sun & Tue 21:00 | general-purpose, opus |
| `dead-code.md` | 1st of month 15:51 | the `deadcode-routine` agent spec |

The science desk has its own thread and prompt, and its brief is `.claude/agents/science-desk.md`.

## How a run works (for the thread)

1. Read this README and the routine's file from `origin/main` (`git fetch origin main`, then `git show origin/main:.claude/routines/<file>`), whatever this thread's own checkout holds. If the fetch fails, say so in the thread and stop.
2. Hand the job to one fresh helper so each run starts clean: start one Agent in the background with `isolation: "worktree"`, so runs that overlap don't share a checkout. Pass on the routine's brief as written, plus the environment note and branch rule below, and end your turn with a status checklist. Tell the helper not to call any `mcp__hearthbot__` tools.
3. Where a routine's helper is an agent spec, start a general-purpose helper on opus and pass it the body of `.claude/agents/<name>.md` from `origin/main`: repo agent types don't load in project threads.
4. When it finishes: if it opened a PR, subscribe to the PR's activity, reply in the thread with the link and what it needs from the user, and drive the PR to green as the thread's own. If it changed nothing, refresh the status checklist and skip the reply (housekeeping always replies, with the priority list). If the report has a Concerns section, pass it on to the user word for word.
5. Keep the thread open: never call `set_thread_resolved` there, because scheduled tasks can't fire into a resolved thread.

## Branch rule (for the helper)

Run `git fetch origin main` first. Push to the routine's branch, named in its file: restart it from `origin/main` each run, unless an open PR already uses it, in which case build on that PR. If the remote branch is gone, push with `--force-with-lease=<branch>:`, else `--force-with-lease`.

## Practical notes (for the helper)

Pass these on with every brief:

- Use the GitHub MCP tools (`mcp__github__*`, loaded with ToolSearch) for PRs; there's no `gh` CLI.
- Before writing prose (PR body, todo notes, docstrings), read `.agents/skills/writing/SKILL.md` and `.agents/skills/style-terms/SKILL.md`, since the Skill tool may not know repo skills here. Don't name the user in commits, PRs, todos or code.
- Open a PR ready for review (not a draft).
- Run `./go check` before pushing.

## Environment note (for the helper)

This project's environment holds production credentials: `HF_TOKEN` can write the production storage pair, and Modal tokens are set. Run `export MINI_PROFILE=dev` before any `mini` or `./go` command, and never write to production storage. Reading production to check results is fine. Each routine's file states its Modal limit.
