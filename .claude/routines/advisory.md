# Daily advisory check

Security advisories only; the full dependency review (`deps.md`) runs on Mondays. Branch: `claude/routine-advisory`. Modal: none.

## The helper's brief

Hi Opus-san! This is the daily security check on our dependencies. It keeps a package with a known advisory from sitting in the lockfile until Monday's full review, which protects the experiment environments and everyone who builds on this repo. Most days the audit is clean, and a one-line "all clear" is a good result.

1. Run `export MINI_PROFILE=dev`, then `./go deps --audit` only.
2. If the audit is clean, report that and change nothing.
3. If it finds advisories: restart the branch from `origin/main`, upgrade the affected packages with `uv lock --upgrade-package <pkg>`, and run `./go check`. Respect the three-day `exclude-newer` window: report any fix it still holds back, with the time it clears, and don't override it. Open a [DEP] PR, following `.claude/agents/deps-routine.md` for the PR body and advisory table, with the project's attribution header. Report each advisory (package, versions, ID, whether it could reach us, what it did), the PR link, and what the user needs to do.

`./go check` passing is your check that an upgrade is safe. If the audit itself errors, or `./go check` fails in a way that looks unrelated to the upgrade, stop and say what you saw; that's a useful finding, and there's no need to work around it. If anything about the task or the environment seems off, add a Concerns section to your report, and it will reach the user word for word.

## For the thread

If a fix is held back by the `exclude-newer` window, schedule a one-off follow-up in the thread for after it clears.
