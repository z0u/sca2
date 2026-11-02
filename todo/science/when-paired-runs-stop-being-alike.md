---
status: open
tags: [D2.2, seeds]
---
# When do runs at one seed stop being alike?

We now compare conditions as two samples and pair runs by seed only when they branch from one stored state and are measured soon after ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)). "Soon after" has no number yet. The evidence for the rule is two reports: removal under `τ_ℓ fixed` and `τ_ℓ anneal` correlated about 0.2 across eight seeds at epoch 400 (*fixed-confirm*), and in the sans-diff report two runs at one seed were about as far apart as runs at different seeds.

The stored runs could say more without a GPU: the correlation across seeds between two conditions, at each kept checkpoint, for pairs that share seeds (*fixed-confirm* with *anneal-confirm*, sans-diff with its controls, the arms of the late-spill scout after their shared epoch 64). That gives the number of epochs over which a branch stays paired, and whether the task scores (which net against a control) decorrelate as fast as removal does.

## Notes

**2026-11-02, low-runs report** — The [low-runs report](/docs/m2/low-runs/report.py) measured the distance between the weights of twin runs (same labels, seed and batches; L4 against L40S arithmetic) at every kept checkpoint. By epoch 16 the twins are at 0.81 to 1.00 of the √2 that two unrelated runs reach, and at 1.26 to 1.33 by epoch 400, so in weight space a pair stops being alike within the first sixteen epochs. The task loss and α of a pair agree to a hundredth for the whole run all the same. So "soon after" is short in the weights and can be long in the scores; the correlation across seeds proposed above would say which scores stay paired.
