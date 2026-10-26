---
status: open
tags: [D2.2, edit-effects, measurement]
opened: 2026-10-26
---
# An absolute floor for the write tolerance

The write bound of ex-2.2.1 (carried into [edit-effects](/docs/m2/edit-effects/report.py)) allows the write a quarter over the bound, as a ratio. Where the bound is small, a ratio over the tolerance can be a shift of a degree or two. In [recipe-compare](/docs/m2/recipe-compare/report.py), `pool 1–3, soft τ` has the largest ratio of the three recipes (1.33, at slice 1 on the example operands), where the bound is about 5°, so the write is under 2° over it. Every single site past the tolerance under any recipe there has a bound of 6° or less.

The paper review asked whether a gate should ignore small absolute shifts. That seems right to us: a site would go over only when the write is past both the ratio and the bound plus some number of degrees. The floor should come from the measurement rather than be picked by eye, for example from the seed spread of the bound at the low-bound sites, or from the write that an edit along a reference direction gives at those sites. Stored arrays from edit-effects and recipe-compare should be enough to choose it without training.

If adopted, the D2.2 design would name the floor next to the tolerance, and the reports that quote the ratio would say how many sites are over both.
