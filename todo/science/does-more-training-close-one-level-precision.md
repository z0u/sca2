---
status: open
tags: [D2.1, ex-2.1.3, vocab]
---
# Does more training close the one-level precision gap at the full grid?

Ex-2.1.3's v4096 cells plateau at seen 0.85 / holdout 0.65 under the fixed 100-epoch schedule, with misses one grid level off in one channel — the geometry is right and the precision isn't. Candidates: a longer or reshaped schedule, weight decay (grokking-style late snap-in).
