---
status: open
tags: [D2.1, anchoring, ex-2.1.6]
---
# Is the alignment decay worth chasing on its own?

Read as a gate, the H4 slide in ex-2.1.6 is a failure; read as a phenomenon it may be the more informative half. The margin peaks near epoch 10 and loses about a quarter over the next forty epochs at constant λ, so the optimizer walks away from selectivity under steady pressure rather than losing it when protection is withdrawn. One reading is that the selective response arrives first and the rest of the cube follows it onto the axis, which the mean alignment climbing over the same window is consistent with; ex-2.1.6 cannot separate that from the alternatives. Note that stopping early is not a rescue — the peak (≈0.36) never cleared the 0.5 gate either, so there is no epoch at which selectivity was demonstrated, and picking a stopping point from these trajectories would be selecting on noise.

## Notes

**2026-08-16, housekeeping** — ex-2.1.11's `flat-anchor` ablation (holding λ_a constant, no schedule) is adjacent evidence but doesn't settle this: on the *current* recipe (op1-only pull + softmin pooling + anti-subspace repulsion), the margin "reaches its margin sooner and then holds it (retention 0.9975)" — no decay (`### Is the anchor schedule needed?`). That's a different setup from the whole-stream, unpooled, no-repulsion anchor that produced the original H4 decay in ex-2.1.6, so it's reassuring for the recipe we're actually using rather than a resolution of the mechanism question this item asks. Leaving open — still needs the closer look #101 flagged, ideally by someone re-reading both trajectories side by side rather than a grep pass.
