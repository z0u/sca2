---
name: report-review
description: >-
  Iterative fresh-eyes review of a draft experiment report or design doc, via
  Use when the user wants a report reviewed, sanity-checked, or wants to know
  if it's ready to run/freeze.
context: fork
model: sonnet
---

Let's iteratively improve this report.

For up to 3 rounds:

1. **Spawn a fresh `report-reviewer` agent**. Give it:
    - The file path
    - Anything the user said should change about the review's focus this round
    - A one-line summary of what the *previous* round changed, so it doesn't
      redo that work blind — but not the previous round's reasoning or
      verdict, so its judgment stays independent.
2. **Read its report** (`Changes` / `Blockers` / `Recommendation`).
3. **Decide whether to continue:**
    - If it reports no blockers and only minor/no changes — stop, this is
      converged.
    - If it made real fixes but recommends running/freezing as-is — stop,
      this also counts as converged.
    - If it found something structural it couldn't fix itself, or flagged a
      genuine design question, stop the loop and escalate to the user.
    - Otherwise (it made substantive fixes and isn't yet at "no blockers"),
      go to another round.
4. After the loop ends (converged, escalated, or 3 rounds reached), give the
    user a summary of what changed across all rounds and the final
    recommendation. Diffs are left staged, not committed — offer to commit if
    the user wants.
