---
name: science
description: |
  How we do experimental science in this project: designing experiments, preregistering falsifiable hypotheses, and collaborating on reports.
---

## Preregistration

Design the experiment with the human, and draft the report skeleton before writing any experiment code. The skeleton doubles as the analysis plan: writing it before the data exists is what lets a later "we predicted X and found Y" carry weight, because the prediction is verifiably older than the result.

The skeleton is usually a text-only notebook. It runs: intro (the question, why it matters for anchoring, lineage from earlier experiments), a "How to read this draft" note, the method (data spec, measurements), the hypotheses with decision thresholds, analysis sections (consider having one section per hypothesis), an "Exploratory analyses" section, and a discussion.

Conventions:

- **Placeholders are admonitions marked `TODO`.** Each states what its figure or table will show (axes, panels), the hypothesis it scores, the expected pattern, and what a contrary result would look like. The marker is greppable, so no placeholder survives to publication; results replace placeholders in place, so review reads as a prediction → observation diff.
- **Hypotheses are falsifiable:** state the measurement, the threshold, and which outcomes count as partial.
- **Freeze the hypotheses once the skeleton is agreed** (immaterial edits aside), and say so in the report itself under "How to read this draft": results replace placeholders, and anything conceived after seeing the data goes under "Exploratory analyses", marked as post hoc.
- **Avoid over-claiming in the analysis and discussion.** An experiment may _inform_ the next, but committing to an interpretation now may blind us when we run the follow-up.

Example:

```md
## Hypotheses

- **H1.** Describe what we're testing (no title).

<!-- Then in the results/analysis section further down... -->

## Short name for H1 (H1)

/// admonition | TODO
Describe what is needed (figure, table, expectations).
///
```

## Best practices

- Choose a measurement site by a criterion independent of the statistic you're judging.
- (more in `/todo-science.md`)

## Collaborating on a report

The human wants to be involved in the writing, so the skeleton is a review artifact in its own right. Iterate on it together in a PR before any experiment code lands (although feel free to run small prototypes that don't get committed) — this is where the hypotheses and thresholds get agreed and frozen.

When results arrive, fill the report in order of stakes rather than all at once. The mechanical sections — where the number either clears its threshold or it doesn't — can be filled in one pass. Pause for a discussion round before writing the prose where interpretation lives, since that is the part the human most wants a hand in, and the part most likely to over-reach.

Whatever you have written, run a review round over it before handing back to the human — the sections that are done, not the whole report. Say in the request which sections are in scope, so a `TODO` in a section whose turn hasn't come isn't read as an omission.

The publishing mechanics — exporting the report as a bundle, wiring result refs, verifying the render — are a separate concern, covered by the `mi-ni` skill.

### Review passes

The `report-review` skill runs the report past fresh readers who haven't seen
the conversation that produced it. Use it twice, for two different questions:

- **Before freezing the hypotheses, and again before running.** The
  `prereg-reviewer` asks whether the design is sound and worth running as
  specified, and comes back with a run/freeze/discuss recommendation.
- **Once the results are in and the report is filled, before publishing.** The
  `results-reviewer` asks whether the results support the claims and whether a
  fresh reader can follow the report — every hypothesis scored against its frozen
  threshold, post-hoc readings kept out of the primary sections, figures and
  tables captioned and legible.

Reviewers see the report and nothing else, which is what makes them useful and
also what makes rounds oscillate: two readers can each be locally right about a
claim and correct it in opposite directions, round after round. The fix is to
record decisions in the artifact, where the next reader will find them.

**Leave a `REVIEW` note wherever a review changes a claim.** Not for typos or
prose polish — for a threshold, a verdict, a scope, or a wording that changes
what is being asserted. Say what the change was, why, and what a later reader
should check to disagree with it. Usually a Python comment in the cell:

```python
# REVIEW: narrowed "anchoring transfers" to "transfers at layer 4" — H2 only
# measures layer 4, so the broader claim outruns the data. Verify: if the sweep
# in the exploratory section covers other layers, this can widen again.
```

An HTML comment inside a Markdown string works when the note has to sit beside
one specific paragraph; it stays invisible in the render. Make it a visible
admonition only when a reader of the published report benefits from it. The
marker is greppable either way, so a review pass can find every prior decision
before touching the same text.

**A note you would reverse is a finding, not an edit.** Wanting to undo a
recorded decision usually means the claim is doing two jobs at once, and each
reviewer is right about a different one. Say so, name both readings, and stop —
the resolution is structural (split the hypothesis into two tracks, drop one,
or state the scope that separates them) and it needs the human, because it
changes what the experiment claims.

For prose alone, the pass is lighter. Text should be reviewed before handing
back to the human. In one turn:

1. Write
2. Stage changes (unless you have another way to see what the agent changes)
3. Hand it to the `prose-simplifier` agent on the same turn
4. Review the agent's changes; check for correctness
