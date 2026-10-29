---
status: open
tags: [D2.2, anchoring, intervention]
opened: 2026-10-29
---
# Anchor and anti terms at the mid-block states

The [mid-block-edit report](/docs/m2/mid-block-edit/report.py) edited the state between the attention and the MLP of each block, beside the usual edit between blocks. On the recipe of record it gained no removal and spilled on half the runs. The attention of block 1 writes about as much e₁ on the other ops as on `difference`, and the MLP of block 1 seems to use it, so taking it away there changes what that MLP computes for every op.

The anchor and anti terms read only the slices between blocks (`NGPT.stream_and_logits`), so nothing in training keeps e₁ for the op alone at the mid-block states. With both terms acting there as well, the attention of block 1 might keep e₁ off the other ops, and an edit at any point along the stream might then be selective. That would add no removal on the four-block model, where the edit between blocks already takes out nearly all of the op, so its value is for deeper models, where the sites an edit can reach are not ours to choose.

A version: expose the mid-block states from the training forward pass, apply the anti term at them (and the pull, pooled with the slices it already pools), and train eight seeds of `τ_ℓ fixed` with it (about $7 on L4). Score with `mid=` in `sca.intervention.apply`, as the mid-block-edit experiment does.
