---
status: open
tags: [D2.2, anchoring, pooling]
opened: 2026-10-09
---
# Pool the pull over slices as well as positions

The pull asks each labeled context to align somewhere in it at every slice. Pooled over slices too, it would ask for alignment somewhere at some slice, and the model would meet it at the slice where a contextual feature is cheapest to make, with no need to choose slices by hand. Mellowmax is separable at one τ: pooling over positions and then over slices is the same term as one mellowmax over all the position × slice pairs, so either form will do, and two τ's (a harder pool over depth) are possible with the sequential form.

On the current recipe this would make things worse on its own: the cheapest slice is the embedding, where a latch or the lean satisfies every context at once ([lean review](/docs/m2/embedding-lean/report.py)), so the later slices would never be pulled. It is worth running only with the embedding held off e₁, by the anti term on every slice or by `clean_embedding_rows` on every token, as an arm beside the fixed late-slice pull in [pull-slice-restrictions](./pull-slice-restrictions-at-more-seeds.md). The three separate "which slices" from "let the model choose". Proposed by the reviewer on the lean review.
