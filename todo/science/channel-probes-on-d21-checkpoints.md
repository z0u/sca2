---
status: open
tags: [probes, anchoring, representations]
opened: 2026-08-18
---
# Fit per-channel probes on the d2.1 checkpoints, ex-2.1.5 style

Alex's idea, from comparing the d2.1 grading grids with ex-2.1.5's per-channel probe figure: fit ridge probes for the RGB channels of each operand at every (slice, position) site of the d2.1 conditions, and draw the same kind of figure. The machinery exists — checkpoints and probe sets are published, `sca.anchoring._stream_axis` captures the residual slices, and the ridge code is in ex-2.1.5. The models are 4-layer, 64-wide, over 5832 lines, so the cost is minutes.

Expectations worked out on 2026-08-18, which the run would test:

- Off-key sites are bounded by the probe set, and identically so for every model. At the op1 slot the state is a function of the op1 token ([[causal-model-s-first-position-context-free]]), so an op2-target probe there can only read E[op2 | op1], and the closure rule caps that at R² ≈ 0.086 per channel — the parity sawtooth E[op2 level | op1 level] = 2, 3, 2, 3, 2, 3, the same for R, G, and B. This holds for the un-anchored condition too: fitted probes read the model's own color representation, which is there whether or not it aligns with the anchor axis. So these figures would show the off-key tilt in every condition, and that model-independence is their value — a companion demonstration that the tilt in the α figures is partner composition. Evaluating the same probes on uniform (non-closed) pairs would flatten the off-key sites and show the same thing the other way.
- The on-key R channel would track α only loosely: α carries sim¹·⁵ to red, which correlates with redness at 0.90 but with the raw R channel at 0.61 over the palette. G and B stay fully decodable in the anchored models if the task is intact.
- The real open question the figure would answer: does anchoring cost any ordinary color decodability, per channel and per site? That is the bounded-side-effects claim read through a probe, and nothing published measures it yet.

This is a new measurement on published checkpoints, so it belongs in a small experiment page beside d2.1 rather than in `docs/m2/d2.1/report.py`, which declares itself not an experiment.
