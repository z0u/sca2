---
status: partial
tags: [performance, modal, experiments]
opened: 2026-10-30
---
# A training step with no scatter-add

The seed-pack probe ([`seed-packs-on-larger-gpus`](./seed-packs-on-larger-gpus.md), `docs/m2/pack-timing`) found five scatter-adds in the anchored training step. Each is the gradient of an indexing operation: two picks of residual slices by an index array (`states[sel]` in `make_anchored_train_step`), the embedding lookup (`wte[idx]` in `NGPT`), and two in the cross-entropy (optax's `softmax_cross_entropy_with_integer_labels` picks the target logit). The GPU determinism flags ([eng/determinism.md](/eng/determinism.md)) make a scatter-add run serially.

On the L40S, with the flags on, removing all five made one seed's step about a third faster: 3.3 ms a step against 4.5, with batch drawing and the readback included. It would make every training run cheaper, so it is worth having whether or not seeds are ever packed.

## Done

The slice picks are static slices now (`_take_slices` in `sca.anchoring`). After one epoch on the CPU the weights are bit-identical to the indexed version, which fits the change: a static slice selects the same values, and its gradient is a pad where the gather's was a scatter-add onto zeros. I haven't checked it on a GPU.

It does change the code every anchored training task depends on, so re-ticking an existing experiment retrains its runs unless the run passes `--keep-stale-done`. Published reports read pinned artifacts and are unaffected.

## Left to do

The other three were only tried in the probe, as a patch made at run time (`_without_gathers` in `docs/m2/pack-timing/experiment.py`): the embedding as a product of a one-hot matrix with the table, and the cross-entropy as the sum of log-softmax times a one-hot target. Moving them into `sca.model.ngpt` and `sca.training.loop` is a numerics change. At full precision the forward pass selects the same values, but the embedding gradient sums the same terms in another order, so a run would not reproduce its earlier weights bit for bit. That makes it a change of the kind a jax upgrade is, best made between experiments rather than under one. `train_model` and the scoring passes share the model code and would move with it.

Before adopting it, time it on the L4, the card most runs train on. The probe queued an L4 run of the scatter-free step (`L4-onehot`) that had not been scheduled when this item was written.

## Cost per step by card

From the probe, with the flags on and one seed per card, in microdollars a step at Modal list prices (milliseconds a step in brackets):

| card | as trained now | scatter-free |
| --- | --- | --- |
| L4 | 1.9 (8.7, two containers) | |
| L40S | 2.4 (4.5) | 1.8 (3.3) |
| A10G | 4.2 (13.6) | |
| A100 | 4.0 (6.8) | |

The L4 figure is the step with the slice picks already static; one L4 container on the earlier code took 11.2 ms. So the L40S comes second to the L4 per dollar and runs twice as fast, and the A10G costs about twice the L4 per step. The fallback list for fresh-seed training is now `["L4", "L40S", "A10G"]` (the `mi-ni` authoring reference and [eng/determinism.md](/eng/determinism.md)), which was `["L4", "A10G", "L40S"]` when it was first written. If the scatter-free step holds on the L4 as it does on the L40S, the order may want another look.
