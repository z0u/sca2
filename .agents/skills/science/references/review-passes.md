# Running review passes

Read this when you are the one *commissioning* a review — driving the report
yourself, or running the `report-review` skill. A reviewer agent doesn't need
any of it; what a reviewer needs is the `REVIEW` note convention in
[SKILL.md](../SKILL.md).

## Which pass

The `report-review` skill runs the report past fresh readers who haven't seen
the conversation that produced it. Use it twice, for two different questions:

- **Before freezing the hypotheses, and again before running.** The
  `prereg-reviewer` asks whether the design is sound and worth running as
  specified, and comes back with a run/freeze/discuss recommendation.
- **Once the results are in and the report is filled, before publishing.** The
  `results-reviewer` asks whether the results support the claims and whether a
  fresh reader can follow the report — every hypothesis scored against its
  frozen threshold, post-hoc readings kept out of the primary sections, figures
  and tables captioned and legible.

A part-filled report gets the `results-reviewer` over the sections that are
done. Say which sections are in scope, so a `TODO` in a section whose turn
hasn't come isn't read as an omission.

## Why rounds oscillate

Each reviewer sees the report and nothing else. That isolation is what makes
them useful, and it is also why two rounds can each be locally right about a
claim and correct it in opposite directions. Two mechanisms keep that in check,
and they are deliberately separate:

- **The `REVIEW` note**, in the report. It carries a decision and its warrant
  forward to the next round, which *is* allowed to read it — the note is part of
  the artifact, so passing it on leaks nothing. This is what stops round 3
  re-litigating a sentence round 1 settled.
- **The `Tensions` field** in the reviewer's structured report. It carries a
  round's own reading — its doubts, its confidence, what it nearly changed — to
  you and no further. Nothing downstream sees it, which is what lets the next
  round's judgment stay independent.

Keeping those channels distinct is the whole design. A note that starts
editorialising about the report's quality has drifted into the second channel's
job and has begun priming the next reader; see the note-content rule in
SKILL.md.

## When to stop

Escalate rather than run another round when a reviewer reverses a decision an
earlier round recorded, or reports a claim pulling two ways. Another round will
just flip it back. Put both readings to the human and say what would separate
them — splitting the hypothesis into two tracks, dropping one, or narrowing its
scope. That call changes what the experiment claims, so it is theirs.

Also stop and escalate for anything structural a reviewer couldn't fix itself,
and for a disagreement about how a result should be read.

## The prose pass

Prose gets a simplification pass regardless of whether a review round runs —
see "Collaborating on a report" in SKILL.md for when it applies. No reviewer
agent is involved. In one turn:

1. Write
2. Stage changes (unless you have another way to see what the agent changes)
3. Hand it to the `prose-simplifier` agent on the same turn
4. Review the agent's changes; check for correctness
