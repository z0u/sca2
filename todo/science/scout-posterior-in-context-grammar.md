---
status: open
tags: [D2.2, task-grammar]
opened: 2026-09-24
---
# Scouting report: the posterior over ops in the in-context grammar

The [D2.2 pivot](/docs/m2/d2.2/pivot.md) proposes a grammar where the model infers the op from a few solved examples. Its design numbers come from scratch simulations: how the posterior on the true op spreads with the number of examples and the replacement rate, and the Bayes ceiling (the best accuracy any model could reach by answering with the answer distribution weighted by the posterior), about 0.74 with three clean examples and 0.58 at a replacement rate of 0.35.

Those would be easier to review as a `lit` report with figures: the posterior distribution across contexts for a grid of example counts and replacement rates, the ceiling on the same grid, and how much random-cube noise adds. All of it is computable from the op table (`answer_dist` in `sca.data.ops`) under stochastic rounding, with no training. It would also fix the replacement rate and example count for the in-context control.

The [quick route](/docs/m2/d2.2/design.md#quick-route) in the design folds this report into the method section of [the pilot](/docs/m2/d2.2/design.md#the-pilot), so it needs no round of its own.
