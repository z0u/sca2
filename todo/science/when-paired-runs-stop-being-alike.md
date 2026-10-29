---
status: open
tags: [D2.2, seeds]
---
# When do runs at one seed stop being alike?

We now compare conditions as two samples and pair runs by seed only when they branch from one stored state and are measured soon after ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)). "Soon after" has no number yet. The evidence for the rule is two reports: removal under `τ_ℓ fixed` and `τ_ℓ anneal` correlated about 0.2 across eight seeds at epoch 400 (*fixed-confirm*), and in the sans-diff report two runs at one seed were about as far apart as runs at different seeds.

The stored runs could say more without a GPU: the correlation across seeds between two conditions, at each kept checkpoint, for pairs that share seeds (*fixed-confirm* with *anneal-confirm*, sans-diff with its controls, the arms of the late-spill scout after their shared epoch 64). That gives the number of epochs over which a branch stays paired, and whether the task scores (which net against a control) decorrelate as fast as removal does.
