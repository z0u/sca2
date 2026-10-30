---
status: done
tags: [performance, modal, experiments]
opened: 2026-10-29
closed: 2026-10-30
---
# Seed packs on larger GPUs

Packing seeds was closed at d64 because, once `train_anchored` scans 16 steps per dispatch, an L4 is doing real work: a scanned step takes about 3.4 ms for one seed, and eight seeds vmapped together come to about 6.2 ms per seed-step ([`training-step-is-host-bound`](./training-step-is-host-bound.md), [`pack-runs-per-gpu`](./pack-runs-per-gpu.md)). That looks like a limit of the card (memory bandwidth, perhaps) rather than of the loop, so a larger card might change the answer.

Nobody has measured it. On paper an L40S has about 2.9× the memory bandwidth of an L4 at about 2.4× the price, so the best case is a saving of about a fifth per seed. An A100 or H100 has far more bandwidth, and a pack of 8 to 16 seeds there might halve the cost per seed.

The probe: time the scanned step for packs of 1, 8 and 16 seeds on an L4, an L40S and an A100, with the probe recipe of `pack-runs-per-gpu` (a throwaway mini experiment under `MINI_PROFILE=dev`, several single-use containers per cell), and compare the cost per seed-step at list prices.

Two things changed since packing was closed, both in its favor. Conditions are now compared as two samples ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)), so the new numerics of a pack and a move off the L4 matter less: a whole experiment on one card is fine, as long as no run in it pairs with or resumes from a run on the L4. The cost that stays is that the pack is the memo unit, so one failure retrains every seed in it. Training is a few dollars per experiment, so the main gain would be wall time.

L4 supply on Modal is a second reason. Since about 2026-10-27 a training task has waited a median of 12 to 28 minutes for an L4, with a tail past an hour, counting only the wait that our own container cap does not explain. A pack needs one card where eight runs need eight, so a confirmation run would wait for one machine rather than for the slowest of eight. If a pack runs well on an L40S or an A100, that card also gives a second capacity pool to draw on, with each experiment still on one card.

## What the probe found (2026-10-30)

Packing does not pay on any card we tried, as long as the GPU determinism flags are on. The probe is `docs/m2/pack-timing`, run under the dev profile: the step of the recipe of record (`τ_ℓ fixed`, as the fixed-confirm experiment trains it), at epoch 100 of its schedules, vmapped over packs of seeds and timed per seed-step, three containers a card unless noted. The "loop" figure adds what `train_anchored` does around each dispatch (drawing the batches on the host, about 0.8–1.5 ms a seed-step, and reading the losses back). Milliseconds per seed-step, loop, the median over containers (one seed is the plain, unvmapped dispatch):

| card | one seed | pack of 8 | pack of 32 | pack of 64 |
| --- | --- | --- | --- | --- |
| L4 (one container) | 11.2 | 28.6 | 35.7 | |
| A10G | 13.6 | 18.2 | 20.3 | |
| L40S | 4.5 | 8.4 | 12.7 | |
| A100 | 6.8 | 7.1 | 7.5 | 7.6 |
| A100, determinism flags off | 7.8 | 4.6 | 4.3 | 4.3 |
| L40S, no scatter-add in the step | 3.3 | 4.7 | 5.7 | |

On every card with the flags on, a pack costs at least as much per seed as one seed alone, and on the L4, A10G and L40S it costs two to three times as much. With the flags off, the A100 halves its cost per seed from a pack of 8 on, but that only brings it level with the L4 per dollar.

The step had five scatter-adds, which are the gradients of indexing: two residual-slice selections (`states[sel]` with an index array), the embedding lookup, and two in the cross-entropy. The determinism flags make a scatter-add run serially, and a vmapped scatter-add grows with the pack. Removing all five (static slices, and one-hot products for the other three) makes one seed about a third faster on the L40S and a pack much cheaper, but a pack still costs more per seed than one seed does. Something else in the batched step also grows with the pack. I did not chase it, since even the best pack on the best card would save little beside the single-seed gain.

Peak memory was about 125 MB per seed, so memory was never the limit: a pack of 64 used 8 GB of the A100.

Two results bear on other work, and they are taken up in [`scatter-free-training-step`](./scatter-free-training-step.md): the scatter-free step is faster for every run, packed or not, and the fallback cards differ a lot per dollar. The probe cost about $6.60 in all.
