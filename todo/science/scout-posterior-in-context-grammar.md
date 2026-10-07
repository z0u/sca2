---
status: done
tags: [D2.2, task-grammar]
opened: 2026-09-24
---
# Scouting report: the posterior over ops in the in-context grammar

The [D2.2 pivot](/docs/m2/d2.2/pivot.md) proposes a grammar where the model infers the op from a few solved examples. Its design numbers come from scratch simulations: how the posterior on the true op spreads with the number of examples and the replacement rate, and the Bayes ceiling (the best accuracy any model could reach by answering with the answer distribution weighted by the posterior), about 0.74 with three clean examples and 0.58 at a replacement rate of 0.35.

Those would be easier to review as a `lit` report with figures: the posterior distribution across contexts for a grid of example counts and replacement rates, the ceiling on the same grid, and how much random-cube noise adds. All of it is computable from the op table (`answer_dist` in `sca.data.ops`) under stochastic rounding, with no training. It would also fix the replacement rate and example count for the in-context control.

The [quick route](/docs/m2/d2.2/design-2026-10.md#quick-route) in the design folds this report into the method section of [the pilot](/docs/m2/d2.2/design-2026-10.md#the-pilot), so it needs no round of its own.

## Notes

**2026-09-27, ex-2.2.16 prereg** — Computed in ex-2.2.16's method section (`docs/m2/ex-2.2.16/posterior.py` beside the report). The pivot's 0.74 and 0.58 are the hard-accuracy form (the predictor naming the mode); the calibrated predictor the reports' expected exact match scores gets 0.69 and 0.49, and 0.53 at the centre condition (3 examples, ρ = 0.3). The floor is 0.11 at every grid point. Proposed conditions: `k3-r0.2`, `k3-r0.3`, `k4-r0.3`.

**2026-09-27, Opus, PM** — In progress in [#221](https://github.com/z0u/sca2/pull/221) (open): `docs/m2/ex-2.2.16/posterior.py` computes the posterior and the Bayes ceiling over the example-count and replacement grid, with cube noise, and the pilot's report cites this item from its method section. Close this when #221 merges.

**2026-09-28, Opus** — Done in #221: the posterior and the Bayes ceiling over the grid are in the method section of the [ex-2.2.16 report](/docs/m2/ex-2.2.16/report.py), and [ex-2.2.17](/docs/m2/ex-2.2.17/report.py) scores trained models against the same posterior, context by context.
