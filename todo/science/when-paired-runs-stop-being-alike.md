---
status: open
tags: [D2.2, seeds]
---
# When do runs at one seed stop being alike?

We now compare conditions as two samples and pair runs by seed only when they branch from one stored state and are measured soon after ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)). "Soon after" has no number yet. The evidence for the rule is two reports: removal under `τ_ℓ fixed` and `τ_ℓ anneal` correlated about 0.2 across eight seeds at epoch 400 (*fixed-confirm*), and in the sans-diff report two runs at one seed were about as far apart as runs at different seeds.

The stored runs could say more without a GPU: the correlation across seeds between two conditions, at each kept checkpoint, for pairs that share seeds (*fixed-confirm* with *anneal-confirm*, sans-diff with its controls, the arms of the late-spill scout after their shared epoch 64). That gives the number of epochs over which a branch stays paired, and whether the task scores (which net against a control) decorrelate as fast as removal does.

## Notes

**2026-11-02, low-runs report** — The [low-runs report](/docs/m2/low-runs/report.py) measured the distance between the weights of twin runs (same labels, seed and batches; L4 against L40S arithmetic) at every kept checkpoint. By epoch 16 the twins are at 0.81 to 1.00 of the √2 that two unrelated runs reach, and at 1.26 to 1.33 by epoch 400, so in weight space a pair stops being alike within the first sixteen epochs. The task loss and α of a pair agree to a hundredth for the whole run all the same. So "soon after" is short in the weights and can be long in the scores; the correlation across seeds proposed above would say which scores stay paired.

**2026-11-06, science desk** — The [paired-runs report](/docs/m2/paired-runs/report.py) takes the rank correlation across seeds between two conditions at each record, with no new runs. From one initialization (four recipes at 32 seeds to epoch 64, and the fixed and anneal confirmations to epoch 400 at eight), the task score keeps about half of its pairing through epoch 64, removal about a third at forty seeds, and spill none we can detect; at eight seeds nothing after epoch 16 can be told from zero, the 0.2 quoted above included. The size of the change (a 0.6% temperature change against a different pull) made no clear difference. From one stored state (the late spill scout arms from epoch 64, and the same schedule on L4), removal, spill and task stay at about 0.8 to 0.95 for at least 150 epochs, and spill to the end. So the stored state seems to do the work in the rule, more than the time since. Left out: the sans-diff runs, which share a seed with their controls but train on another corpus and have end scores only. Open until the report is reviewed; the human may want to turn this into a `finding` and tighten the rule in `eng/determinism.md`.
