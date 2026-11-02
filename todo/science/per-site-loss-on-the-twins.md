---
status: open
tags: [D2.2, seeds, analysis]
opened: 2026-11-02
---
# Per-site loss on the twin runs

The [low-runs report](/docs/m2/low-runs/report.py) compared the validation loss of the twin pairs that split (seeds 715 and 719) as one number per epoch and found the two runs of a pair within 0.014 of each other, the same as pairs that did not split. A loss broken down by role (the example answers, the query answer, the operands, the syntax tokens) and by op might still show where a low run spends its loss differently, since the reader blocks of a low run take part of the evidence from off e₁ and that could change the prediction at some site by a little. The reviewer asked for this; the report's twins figure shows a small bump in the low run's loss near epochs 30 to 70 at both seeds, which the windowed comparison calls ordinary.

No training: score the stored kept checkpoints of the four twin pairs (`reports/m2/low-runs/cross-kept/{label}` and the fixed-labels kept states) on the validation holdout with the loss kept per position and role, as `ex2216` scores them, and plot the difference between the two runs of a pair by role along training beside the two pairs that did not split.
