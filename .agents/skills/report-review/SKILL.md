---
name: report-review
description: >-
  Iterative fresh-eyes review of a draft experiment report or design doc.
  Use when the user wants a report reviewed or sanity-checked, or wants to know
  whether it is ready to run or to freeze.
argument-hint: <path to report.py>
context: fork
background: false
model: sonnet
---

Let's iteratively improve this report.

Your job is to route and decide, not to review. Don't read the report yourself
— a fresh reader is the whole point, and your context is already full of the
reasoning that produced it.

For up to 3 rounds:

1. Spawn a fresh `report-reviewer` agent. It starts with an empty context, so
   give it:
    - The file path (and the experiment module beside it, if there is one).
    - Anything the user said should change about the review's focus this round.
    - Whether the numbers have already been verified against the experiment
      code, so it doesn't re-derive arithmetic without cause.
    - A one-line summary of what the *previous* round changed, so it doesn't
      redo that work blind — but not the previous round's reasoning or verdict,
      so its judgment stays independent.
2. Read its report (`Changes` / `Blockers` / `Recommendation`).
3. Decide whether to continue:
    - No blockers, and only minor or no changes: stop, this has converged.
    - Real fixes, but it recommends running or freezing as-is: stop, this also
      counts as converged.
    - Something structural it couldn't fix itself, or a genuine design
      question: stop and escalate to the user.
    - Otherwise (substantive fixes, not yet at "no blockers"): another round.
4. After the loop ends (converged, escalated, or 3 rounds reached), summarize
   for the user: what changed across all rounds, any open questions, and the
   final recommendation. Include `git diff --stat` for the report so they can
   see the size of the change. Leave the diff staged rather than committed, and
   offer to commit.
