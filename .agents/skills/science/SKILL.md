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
- **A constants-only `experiment.py` is marked `DESIGN_ONLY = True`.** It is often worth landing the design constants — grid sizes, thresholds, schedules — as a module during preregistration, so the report imports them instead of restating numbers the code will later own. But `tests/mini/test_experiments_e2e.py` globs every `docs/**/experiment.py` and asserts it loads into a named experiment with a callable `main(ctx)`, which a design module doesn't have yet. `DESIGN_ONLY = True` at module level skips that check. Delete the line in the same change that adds the DAG — an implemented experiment that still carries it silently loses its load coverage.
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

Any prose you write also gets a `prose-simplifier` pass on the same turn, whether or not a review round is warranted — stage your changes first so you can see what it did, then read its edits for correctness. It gets the path and line range and nothing else. This applies to a single filled-in section as much as to a whole draft, so it is a habit of writing rather than a step in the review; the sequence is in [references/review-passes.md](references/review-passes.md).

The publishing mechanics — exporting the report as a bundle, wiring result refs, verifying the render — are a separate concern, covered by the `mi-ni` skill.

### The tl;dr

A report opens with one, directly under the title and above the intro prose:

```md
# Ex 2.1.7: a repulsive term and a narrower pull

/// tip |
<!-- tl;dr -->
We tested two mechanisms to improve anchor selectivity:
**1.** Apply the anchor term only to operand 1 (no other tokens), and
**2.** Add a repulsive term to clear the target subspace.
Both work, but 1. worked better, and their effects stack.
///
```

Four lines or so (two when reflowed): what we tried, and which way it came out.
It orients someone deciding whether to read on, so keep numbers, hypothesis IDs,
thresholds and caveats out of it — the analysis sections and the discussion are
where a reader who stayed gets the full accounting. Left to itself this box
grows into a second conclusion; if a sentence in it would also belong in the
discussion, cut it.

The title is empty (`/// tip |`) so the box reads as a lede rather than a
labelled aside, and the `<!-- tl;dr -->` comment keeps the marker greppable. In
a preregistration draft, write the "what we tried" half and leave the outcome
line for later.

### Recording a review decision

Reports go through several fresh-eyes review rounds, each reader starting from
the report alone. `REVIEW` notes are how one round's decisions reach the next.

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
one specific paragraph; it stays invisible in the render. Make it visible only
when a reader of the published report benefits from it. The marker is greppable
either way, so a review pass can find every prior decision before touching the
same text.

**A note records the change and its warrant, and nothing else.** It says what
the report now claims and why that follows from the data — the same category of
thing as a code comment explaining a non-obvious invariant, which is why the
next round may read it. It never carries a judgment of the report's quality, a
round's confidence, or anything phrased as "I suspect" or "this felt weak":
that primes the next reader instead of informing them. Observations of that kind
go in the round's own report, under `Tensions`, where they reach the supervisor
and stop.

**A note you would reverse is a finding, not an edit.** Wanting to undo a
recorded decision usually means the claim is doing two jobs at once, and each
reviewer is right about a different one. Say so, name both readings, and stop —
the resolution is structural (split the hypothesis into two tracks, drop one,
or state the scope that separates them) and it needs the human, because it
changes what the experiment claims.

Commissioning the rounds — which pass to run when, how to brief a reviewer, when
to stop and escalate, and the lighter pass for prose alone — is in
[references/review-passes.md](references/review-passes.md). Always read this if
you are the lead.
