---
status: open
tags: [D2.2, anchoring, measurement]
opened: 2026-10-09
---
# A measure of off-concept alignment that carries to other concepts

The lean measurements of [embedding-lean](/docs/m2/embedding-lean/report.py) and [late pull](/docs/m2/late-pull/report.py) are the slope of the e₁ component of the color embeddings against lightness, and the correlation of α with lightness in the states. Both are about lightness, which is specific to `difference`: it is the stand-in the pull happened to find there, and we did not predict it. Anchoring another op, or another kind of concept, would need a different stand-in found after the fact, so these numbers cannot be compared across concepts.

The review of late pull suggested something like the mean α of the states in contexts that are not of the anchored op, graded by the posterior over ops, so contexts that look most like the anchored op count least. That asks the general question (how far does the anchor direction reach into states it should not), and it needs no knowledge of the stand-in. The trajectories of late pull already keep α at every slice and position for the other ops, ungraded; a graded version needs the posterior of each probe context, which the replacement-noise corpus provides. The lightness measures stay useful as a mechanism check on `difference`, since they show what the stand-in is.

Also from that review: "lean" now names both the slope in the table and the correlation in the states, and does not say that it is about lightness. Late pull now calls them the lightness slope and the lightness correlation, and keeps "lean" for the effect.

## Notes

**2026-10-24, backlog refresh** — [Spill-timing](/docs/m2/spill-timing/report.py) used another measurement that needs no knowledge of the stand-in: the spill along training, scored as the drop on each other op under the edit, net of the median drop under edits along other coordinate axes. It measures the effect on the other ops rather than how far the anchor direction reaches into their states, and its E4 found the two can differ: e₁ held almost no color at the other ops on the runs that spilled. So a graded α and the reference-netted spill would measure different things, and both may be worth keeping.
