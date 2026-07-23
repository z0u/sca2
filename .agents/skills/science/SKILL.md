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

The publishing mechanics — exporting the report as a bundle, wiring result refs, verifying the render — are a separate concern, covered by the `mi-ni` skill.
