---
status: open
tags: [D2.2, anchoring, schedules, selectivity]
opened: 2026-10-24
---
# Does the spill land on ops the run is still learning?

[Spill-timing](/docs/m2/spill-timing/report.py) found two kinds of spill on `pool 1–3, soft τ`. The early kind appears during the rise, as the pool sharpens and the anchor weight reaches its plateau. The late kind appears after about epoch 100 on `value-hsv`, on two of the seeds that learn it last, while they learn it, and stays. Both land on an op that is being learned at the time (E3). When an op is learned changes with the condition, and `falloff` and `step, soft τ` learn `value-hsv` as late on some seeds without spilling on it, so learning late is not enough by itself.

One reading is that e₁ is open to an op while it is being learned, and that what keeps it closed in the quiet conditions is the anti term, which is partway through its slow anneal when the late kind appears. If so, an anti anneal that starts later, or runs slower, should shrink the late kind on `pool 1–3, soft τ` and leave the early kind as it is. The other reading, that whatever draws an op to e₁ also slows its learning, predicts no change.

An arm or two of `pool 1–3, soft τ` with a later or slower anti anneal, at seeds 704 to 711, recording the spill along training as spill-timing did. Both changes are schedules, so they need no role labels. The anti anneal nearly coincides with the learning-rate cosine (r ≈ 0.99, found in the closed schedules-at-the-rise PR #262), so moving the anti anneal alone would also tell the two apart. It could ride in the [soft-pool follow-ups](./soft-pool-spill-follow-ups.md), whose other arms aim at the early kind.
