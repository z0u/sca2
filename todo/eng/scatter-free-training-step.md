---
status: done
tags: [performance, modal, experiments]
opened: 2026-10-30
closed: 2026-10-31
---
# A training step with no scatter-add

The seed-pack probe ([`seed-packs-on-larger-gpus`](./seed-packs-on-larger-gpus.md), `docs/m2/pack-timing`) found five scatter-adds in the anchored training step. Each is the gradient of an indexing operation: two picks of residual slices by an index array (`states[sel]` in `make_anchored_train_step`), the embedding lookup (`wte[idx]` in `NGPT`), and two in the cross-entropy (optax's `softmax_cross_entropy_with_integer_labels` picks the target logit). The GPU determinism flags ([eng/determinism.md](/eng/determinism.md)) make a scatter-add run serially.

With the flags on, removing all five made one seed's step about a third faster on the L40S (3.3 ms a step against 4.5, with batch drawing and the readback included) and about an eighth faster on the L4 (7.6 against 8.7, the L4 already on static slices). It would make every training run cheaper, so it is worth having whether or not seeds are ever packed.

## Done

The slice picks are static slices now (`_take_slices` in `sca.anchoring`). After one epoch on the CPU the weights are bit-identical to the indexed version, which fits the change: a static slice selects the same values, and its gradient is a pad where the gather's was a scatter-add onto zeros. I haven't checked it on a GPU.

It does change the code every anchored training task depends on, so re-ticking an existing experiment retrains its runs unless the run passes `--keep-stale-done`. Published reports read pinned artifacts and are unaffected.

The embedding and the cross-entropy followed on 2026-10-31. `NGPT.embed` is the lookup as a product of a one-hot matrix with the table, at full precision, and every pass of the model (`__call__`, `residual_stream`, `stream_and_logits`, and the edited pass in `sca.intervention`) goes through it. The block1-edit experiment keeps its own indexed copy of the edited pass, which selects the same values and takes no gradient. `sca.training.loop.cross_entropy` is optax's arithmetic (the log-normalizer less the target logit) with the target logit picked by a mask. On the CPU the forward pass, the logits and the loss are bit-identical to the indexed versions, and the compiled anchored step has no scatter or gather left in it. The weights after a step differ in the last bits, as expected, so this is a numerics change: a re-ticked training task retrains, and its runs will not match the weights they had.

`train_anchored` also draws each dispatch's batches on a thread of its own now (`_Prefetch`). The call that dispatches a scan returns only near the end of the card's work, so batches drawn between dispatches left the card idle for about a millisecond a step. The draws are unchanged: the same weights bit for bit on the CPU and on an L40S.

## Measured in the training loop

The step-speed probe (`docs/m2/step-speed`, dev profile) timed `train_anchored` itself on the recipe of record and its full corpus, records and validation included, with the determinism flags on. Milliseconds a step, one container each:

| card | indexed step, batches drawn in line | scatter-free, drawn in line | scatter-free, drawn on a thread |
| --- | --- | --- | --- |
| L40S | 3.45 | 3.25 | 2.42 |
| L4 | 7.47 | 7.95 | 6.74 |

On the L40S the scatter-free step saved less in the loop than in the seed-pack probe, where the batches were already on the card, and the thread saved more: drawn in line, the batches cost the card about as much as the scatters did. L4 timings vary by about a sixth between containers, so its middle column is within noise of the first.

The probe also tried the scan with a select in place of its conditional on the live flag, in case the conditional was what held the dispatch call: 2.26 ms on the L40S and 6.79 on the L4, too little to be worth a change. Turning the flags off is in [eng/determinism.md](/eng/determinism.md#with-no-scatter-left-2026-10-31): it saves about another sixth, but needs a shared compilation cache to stay reproducible.

## Cost per step by card

At Modal list prices ($0.80 an hour for the L4, $1.95 for the L40S), the training loop now costs 1.3 microdollars a step on the L40S and 1.5 on the L4, so the L40S is the cheaper card per step as well as the faster, by about 2.8×. The fallback list for fresh-seed training is now `["L40S", "L4", "A10G"]` (the `mi-ni` authoring reference and [eng/determinism.md](/eng/determinism.md)). The A10G was not timed in the loop; in the seed-pack probe it cost about twice the L4 per step, so it stays last.

From the seed-pack probe, before these changes, with the flags on and one seed per card, in microdollars a step (milliseconds a step in brackets, batch drawing and the readback included):

| card | as trained then | scatter-free |
| --- | --- | --- |
| L4 | 1.9 (8.7, two containers) | 1.7 (7.6, two containers) |
| L40S | 2.4 (4.5) | 1.8 (3.3) |
| A10G | 4.2 (13.6) | |
| A100 | 4.0 (6.8) | |

## Notes

**2026-10-31, follow-up thread** — The fifth round of the pack-timing probe timed the scatter-free step with the determinism flags off: one L40S seed 2.37 → 1.95 ms a seed-step on the card, one A100 seed 5.7 → 3.33. The flags make XLA's autotuner take its first candidate rather than its fastest (`RequireDeterminism` in `stream_executor_util.cc`; the XLA determinism guide says the same), so the saving is the autotuner. Scatter-adds were the one atomically-reduced op in the step, so once they are gone the program may be deterministic without the flags, given a persisted autotune cache; the determinism probe (`eng/determinism.md`) would check that for about a dollar. Worth folding into this item: the two together take an L40S step from 3.3 to 2.8 ms with the host draw, and a prefetch thread for the host draw would take it to about 1.95.

**2026-10-31, single-seed thread** — Folded in above. In the training loop the prefetch thread saved more than expected and the flags less: the flags-off program was not deterministic on its own (no persisted cache), and with a shared compilation cache it was, on the L40S. That part is now [`flags-off-with-a-compilation-cache`](./flags-off-with-a-compilation-cache.md).
