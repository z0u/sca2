---
status: open
tags: [ex-2.1.9, anchoring, representations]
opened: 2026-08-23
---
# Record per-role drift in the anchor trajectory

The ex-2.1.9 `+` embedding ends at 0.30 alignment even under the op1 oracle, which never pulls `+` and whose op1 states can't read it. Whatever pays for that pays *against* the repulsion, which runs from epoch 0 — so *when* it climbs is diagnostic: early would mean the repulsion never priced it out; late (as the anneal decays) would fit a task-loss equilibrium.

The current traj records only op1 drift and the softmin weights, and mid-run checkpoints are overwritten, so this needs a re-run with per-role ᾱ(ℓ, t) recorded. Neither ex-2.1.10 nor ex-2.1.11 records it either. Candidate mechanism worth testing then: `+` appears in every line, so a constant e₁ component there can carry the cube-mean redness to every answer, leaving op1 to carry deviations.

Split from the older ex-2.1.9 follow-ups item on 2026-08-23.
