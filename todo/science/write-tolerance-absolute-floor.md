---
status: partial
tags: [D2.2, edit-effects, measurement]
opened: 2026-10-26
---
# An absolute tolerance for the write

The write bound of ex-2.2.1 (carried into [edit-effects](/docs/m2/edit-effects/report.py)) allows the write a quarter over the bound, as a ratio. Where the bound is small, a ratio over the tolerance can be a shift of a degree or two. In [recipe-compare](/docs/m2/recipe-compare/report.py), `pool 1–3, soft τ` has the largest ratio of the three recipes (1.33, at slice 1 on the example operands), where the bound is about 5°, so the write is under 2° over it. Every single site past the tolerance under any recipe there has a bound of 6° or less.

The paper review asked whether a gate should ignore small absolute shifts. That seems right to us: a site would go over only when the write is past both the ratio and the bound plus some number of degrees. The floor should come from the measurement rather than be picked by eye, for example from the seed spread of the bound at the low-bound sites, or from the write that an edit along a reference direction gives at those sites. Stored arrays from edit-effects and recipe-compare should be enough to choose it without training.

If adopted, the D2.2 design would name the floor next to the tolerance, and the reports that quote the ratio would say how many sites are over both.

## Notes

**2026-10-30, science desk** — The [write-floor report](/docs/m2/write-floor/report.py) proposes a floor from stored arrays, on `τ_ℓ fixed`, `falloff`, `step, soft τ` and `pool 1–3, soft τ`. Drawing the held-out contexts again moves the excess by hundredths of a degree, so it cannot set the floor. The seed spread of the bound at sites with a bound under 10° gives one: 1.13° at the 95% quantile. With it, every site over the tolerance at a `?`, `,` or ⏎ drops out, and what stays is at operands at slice 1 (slice 2 too on the step): one run each on `τ_ℓ fixed` and `falloff`, three on the step, and all eight on the pool. A rule relative to each site (an excess over twice its seed spread) flags a few symbol sites instead, so adopting the floor is a choice between the two. Left for review: whether to adopt it, and naming it in the D2.2 design. The reference-direction source was not measured, since it needs a new pass.

**2026-10-30, science desk** — Renamed on review: the floor is now the *absolute tolerance*, beside the ratio as the *relative tolerance* (as in numpy's `atol` and `rtol`), since "floor" already names the chance-level baseline and the end of an anneal in other reports. The review was fine with adopting it. Proposed for a gate, not yet agreed: count runs with any site over both (at most one in eight), with no role labels.
