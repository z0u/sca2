---
status: open
tags: [anchoring, ex-2.1.9, M3]
opened: 2026-08-23
---
# Position dropout in the pooled anchor

Mask a random subset of span positions out of the softmin each step before pooling. The latch ex-2.1.9 saw is self-reinforcing because the cheapest position absorbs the whole pull every step; dropout keeps the runners-up receiving gradient, so the race stays open longer. Same role as noisy gating / expert dropout in mixture-of-experts routing, where it counters the analogous router collapse. Unlike a τ schedule it needs no timing decision — the cost is a drop-rate hyperparameter and a noisier anchor loss.

Caveat at very low τ: on steps where the favorite is masked, the pull concentrates fully on the next-cheapest position rather than spreading, so it randomizes the winner per step rather than softening the pooling; likely most useful combined with a moderate τ.

No implementation exists on the anchoring path today — `dropout` appears nowhere in `src/sca` or the m2 experiment modules — so this begins in `src/sca/anchoring.py`, as an extra path inside `pooled_anchor_term` keyed on a `drop_rate` argument that `make_anchored_train_step` forwards.

Split from the older ex-2.1.9 follow-ups item on 2026-08-23. Raised in ex-2.1.9's 2026-08-07 review round, round 3.
