---
status: done
tags: [D2.2, anchoring, schedules, ex-2.2.23]
opened: 2026-10-08
closed: 2026-10-08
---
# Where the second rise falls on the anchor schedules

The anchor and anti-subspace schedules are ex-2.1.10's, chosen when *red* was anchored in a grammar of one equation per line, and have not been revisited since the in-context grammar. One reading of the spill in ex-2.2.23 is that the model learns the HSV ops while the anti-subspace term is easing off, and a 400-epoch run spends more steps learning while it is weak.

A small look at ex-2.2.23's stored trajectories, with no compute: the anchor and anti-subspace weights against the share of training, each run's rise epoch marked on them, and the lean (the mean alignment of every state) through training, at both lengths. If the spill turns out to follow the schedules, a longer anti-subspace hold is the change to try.

Split from the withdrawn ex-2.2.24 draft (its E5; see [PR #255](https://github.com/z0u/sca2/pull/255)).

## Notes

**2026-10-08, schedules-at-the-rise report** — The schedules part needed no report: the anti-subspace weight and the learning rate correlate at r ≈ 0.99 over a run in the recipe, as expected, so the stored runs cannot separate them. `AntiSpec` now takes `anneal_start` (default 0, the old schedule), which holds the weight at its peak until then, so a later run can move the anneal off the LR cosine. The lean part became [a report](/docs/m2/schedules-at-the-rise/report.py) that pools the alignment by its peak over roles on contexts of the other ops, at the end of training only, since the trajectories keep only the mean over every state.
