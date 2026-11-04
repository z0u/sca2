---
status: open
tags: [D2.2, post-hoc]
opened: 2026-11-04
---
# A per-position gate for the fitters without role labels

In the post-hoc-compare experiment, the like-for-like comparison restricts every fitter to fits pooled over every position, and edits every position. A real pipeline with no role labels could still choose where to edit from the fitted direction itself: edit a position when the score of the direction there passes a threshold fixed in advance. If that gate lets LEACE or difference in means recover most of the edge they had at the example answers, the pooled comparison is a lower bound on the fitters rather than their fair level, and the write-up should say so.

A scoped test: on the eight controls at seeds 712 to 719, take LEACE and difference in means at the slice each selected over the pooled fits, gate positions by the projection onto the fitted direction at a threshold set on the tuning set, and score removal and spill with the same baselines. No new training, and the scoring is a small pass over stored checkpoints.
