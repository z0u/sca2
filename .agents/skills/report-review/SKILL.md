---
name: report-review
description: >-
  Iterative fresh-eyes review of an experiment report — a preregistration draft
  or design doc before the run, or a filled-in report after it. Use when the
  user wants a report reviewed or sanity-checked, or wants to know whether it is
  ready to run, to freeze, or to publish.
argument-hint: <path to report.py>
context: fork
model: sonnet
---

Let's iteratively improve this report.

Your job is to route and decide, not to review. Probably don't read the report yourself — a fresh reader is the whole point.

The loop below is the runbook. For the reasoning behind it — why the two passes
ask different questions, and why a reviewer's `Tensions` field stops with you
while its `REVIEW` notes travel on — see `references/review-passes.md` in the
`science` skill.

First pick the reviewer, since the two ask different questions:

- **`prereg-reviewer`** — the report has no results yet. Is the design sound and
  worth running as specified? A quick `grep -c TODO` on the file usually settles
  it: a skeleton is mostly placeholders.
- **`results-reviewer`** — the results have landed. Do they support the claims,
  and can a fresh reader follow the report? This one also looks at the rendered
  figures.

If the user said which pass they want, believe them. A part-filled report — some
analysis sections written, others still placeholders — gets the
`results-reviewer` over the sections that are done; that is the normal case
while a report is being written, not an ambiguity to ask about.

Then, for up to 3 rounds:

1. Spawn a fresh reviewer agent of the chosen type. It starts with an empty
   context, so give it:
    - The file path (and the experiment module beside it, if there is one).
    - Anything the user said should change about the review's focus this round.
    - Whether the numbers have already been verified against the experiment
      code, so it doesn't re-derive arithmetic without cause.
    - For a `results-reviewer`: which analysis sections are filled and in scope
      this round, since reports get written one hypothesis at a time and a
      `TODO` in a section whose turn hasn't come is not a finding.
    - A one-line summary of what the *previous* round changed, so it doesn't
      redo that work blind — but not the previous round's reasoning or verdict,
      so its judgment stays independent. The reasoning it is allowed to see is
      whatever the previous round wrote into the report as a `REVIEW` note,
      which is part of the artifact rather than context you are leaking.
2. Read its report (`Changes` / `Tensions` / `Blockers` / `Recommendation`).
3. Decide whether to continue:
    - No blockers, and only minor or no changes: stop, this has converged.
    - Real fixes, but it recommends running, freezing, or publishing as-is:
      stop, this also counts as converged.
    - Something structural it couldn't fix itself, a genuine design question, or
      a disagreement about how a result should be read: stop and escalate to the
      user.
    - It reversed something an earlier round decided, or it reports a claim
      pulling in two directions: stop and escalate. Another round will flip it
      back. Put both readings to the user and say what would separate them —
      splitting the hypothesis into two tracks, dropping one, or narrowing its
      scope. Only the human can make that call, since it changes what the
      experiment claims.
    - Otherwise (substantive fixes, not yet at "no blockers"): another round.
4. After the loop ends (converged, escalated, or 3 rounds reached), summarize
   for the user: what changed across all rounds, any open questions, and the
   final recommendation. Include `git diff --stat` for the report so they can
   see the size of the change, and list any `REVIEW` notes the rounds added —
   those are decisions the reviewers made on their own authority, and the user
   should get the chance to overrule them. Leave the diff staged rather than committed, and
   offer to commit.
