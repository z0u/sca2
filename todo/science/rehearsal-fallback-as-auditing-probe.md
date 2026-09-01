---
status: open
tags: [D2.3, anchoring, auditing]
opened: 2026-09-01
---
# Rehearsed fallback as an auditing probe

The D2.2 [design](/docs/m2/d2.2/design.md) considered training the fallback by rehearsal: apply the axis-projection operator on a small fraction of training steps, and train the output toward the designed fallback. Rejected there as a fallback mechanism, because it trains the model under the operator the eval applies — the observed fallback then shows the training converged, and the cheapest policy keeps the concept readable off-axis and emits the fallback only where the projected state is detected.

That failure mode makes rehearsal a good probe. It is a controlled way to construct masked-rather-than-removed, so it can measure the sensitivity of the auditing rows (relearning rebound, ActPert, off-axis recoverability) on our models: train the rehearsed condition, run the rows, and see whether they flag it. If they don't, we learn the rows' detection floor before leaning on them anywhere else. D2.3-shaped, beside verification.

Related: [Redirect between two anchored ops](redirect-between-two-anchored-ops.md) resolves the same train/eval mismatch from the other side, by choosing an intervention target that training visits.
