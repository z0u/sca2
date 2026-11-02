# Find tuning

Branch: `claude/routine-find-tuning`. Modal: none. The thread replies only when the helper opens a PR or raises a concern.

## The helper's brief

Hi Opus-san! This one looks after `./go find`, the ranked keyword search that agents use to find where something was found or decided (`scripts/find.py`; its docstring explains the ranking and the log). Every search made through `./go` in a project thread is logged to the project's shared folder, and agents record the answers it missed with `--miss`. Your job is to read that evidence and, where it points at something worth changing, change it.

Run `export FIND_NO_LOG=1` first, so your own searches stay out of the log.

1. Run `./go find --log-report --since <date of the last find-tuning PR, or two weeks ago>`. It lists the searches that came back empty, the ones reworded within a few minutes (the first of each pair may have missed), and each recorded miss with the rank its answer has today.
2. For each miss whose answer is still outside the top five, and each empty or reworded search where you can find the answer yourself, work out why: a word the report spelled differently, a term the glossary doesn't list, a passage too long or too short, a stemming slip, or a document kind the search doesn't cover.
3. Add the cases you settled to `tests/find_cases.json` (query plus the path that should rank in the top five). Add a case even when no change is needed, as long as it ranks there today: it guards the next change.
4. If a pattern of misses points at a fix, make the smallest one that covers it: a glossary wording (`docs/glossary.md`, following its comment), a stemming rule, the alias weight, or the BM25 constants. Run `uv run pytest tests/test_find.py tests/test_find_cases.py` and keep every case passing. One miss alone is a case, not a reason to retune.
5. Prefix the PR title with `[FIND]`, and say in the body which misses each change fixes and which ones are left.

If the log has nothing new, or nothing that needs changing, stop without a PR: that's a fine outcome. If anything about the task or the environment seems off, add a Concerns section to your report; it reaches the user word for word.
