---
status: done
tags: [D2.1, anchoring, ex-2.1.6]
closed: 2026-08-18
---
# Add the anti-subspace term and re-run ex-2.1.6

Add the anti-subspace term and re-run ex-2.1.6. The next step from that result.

one attractive term is satisfied largely by a color-independent shift of the whole cube onto the axis, so the margin stalls at 0.27 while the mean alignment climbs with λ. M1's loss carried anti-subspace (`mean(cos²)` over all points) at 3% of the anchor weight, and it constrains exactly that quantity. Same testbed, same schedule, one more term — the cheapest experiment that could turn the M2 transfer result around. Anti-anchor (the hemisphere gate) is the second candidate; `separate` is the third — ex-2.1.6's exploratory section found the cube swinging bodily onto the axis while keeping ~68% of its extent at the scoring rung, which is a common-component problem, though the extent does fall to ~50% at λ=0.3, so `separate` would start to matter at weights past the ones we swept.

## Notes

**2026-08-18, housekeeping** — done, and three experiments deep. [Ex-2.1.7](https://z0u.github.io/sca2/ex-2.1.7/) is the experiment this item asked for: ex-2.1.6's pull with M1's repulsive term added, same testbed, crossed against an op1-only pull. The term works, but its main effect came out smaller than the narrower pull's, so it didn't turn the result around on its own. [Ex-2.1.8](https://z0u.github.io/sca2/ex-2.1.8/) then tuned its schedule and found the trailing hold is what contains the cube-wide drift, and [ex-2.1.11](https://z0u.github.io/sca2/ex-2.1.11/) showed that schedule is load-bearing: a constant delivering the same total dose loses grading. The `separate` candidate is settled too, by [a heavier anchor buys alignment and loses selectivity](./heavier-anchor-buys-alignment-loses-selectivity.md) — the compression it would address only appears at weights past the selectivity optimum. Of the three candidates only anti-anchor (the hemisphere gate) was never tried; it survives as hypothesis 3 in the [D2.1 kickoff queue](./d21-kickoff-carry-over-lessons.md), so closing here doesn't drop it.
