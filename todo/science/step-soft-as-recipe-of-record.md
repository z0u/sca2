---
status: open
tags: [D2.2, anchoring, recipe]
opened: 2026-10-24
priority: high
---
# Should `step, soft τ` be the recipe of record?

`falloff` became the recipe of record in [split-anti-seeds](/docs/m2/split-anti-seeds/report.py), before the soft start existed. Since then, `step, soft τ` (the same step pull and falloff, with a pooling temperature that starts soft) has looked at least as good on the two things that set them apart:

- Lost anchors. [The loss-rate screen](/docs/m2/loss-rate/report.py) put `falloff` at one loss in 40 seeds and `step, soft τ` at none (95% bounds of about 11% and 7%). The difference is not significant at these counts.
- Spill. In [soft-pool](/docs/m2/soft-pool/report.py) two of its eight runs went over the spill criterion against e₁ in the control, which is why it stayed an arm. Against the median baseline of [reference-baseline](/docs/m2/reference-baseline/report.py) (E4), both fall under it, so it now has none over, like `falloff`.

Removal on the kept runs was close to that of `falloff` in soft-pool. So the case for keeping `falloff` may be only that it came first.

The question needs no training: a side-by-side of the two conditions on the stored runs at seeds 704 to 711 (and the finished loss-rate runs where they have end checkpoints), under the D2.2 baseline. Removal, spill at every dose, task score per op, and the edit-effects measurements (write bound, grading, landing), which are cheap to repeat. If `step, soft τ` matches or beats `falloff` on all of them, it would become the recipe of record, and the [recipe sweep](./full-recipe-sweep-after-the-lean-fix.md) would center on it. The soft start also adds a dimension to the sweep (where the temperature starts, and how fast it sharpens).

The two conditions also differ in when the spill appears: [spill-timing](/docs/m2/spill-timing/report.py) found the early kind of spill only on the pool over slices 1 to 3, and at epoch 64 `falloff` and `step, soft τ` look alike.
