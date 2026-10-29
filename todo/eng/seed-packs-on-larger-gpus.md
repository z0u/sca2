---
status: open
tags: [performance, modal, experiments]
opened: 2026-10-29
---
# Seed packs on larger GPUs

Packing seeds was closed at d64 because, once `train_anchored` scans 16 steps per dispatch, an L4 is doing real work: a scanned step takes about 3.4 ms for one seed, and eight seeds vmapped together come to about 6.2 ms per seed-step ([`training-step-is-host-bound`](./training-step-is-host-bound.md), [`pack-runs-per-gpu`](./pack-runs-per-gpu.md)). That looks like a limit of the card (memory bandwidth, perhaps) rather than of the loop, so a larger card might change the answer.

Nobody has measured it. On paper an L40S has about 2.9× the memory bandwidth of an L4 at about 2.4× the price, so the best case is a saving of about a fifth per seed. An A100 or H100 has far more bandwidth, and a pack of 8 to 16 seeds there might halve the cost per seed.

The probe: time the scanned step for packs of 1, 8 and 16 seeds on an L4, an L40S and an A100, with the probe recipe of `pack-runs-per-gpu` (a throwaway mini experiment under `MINI_PROFILE=dev`, several single-use containers per cell), and compare the cost per seed-step at list prices.

Two things changed since packing was closed, both in its favor. Conditions are now compared as two samples ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)), so the new numerics of a pack and a move off the L4 matter less: a whole experiment on one card is fine, as long as no run in it pairs with or resumes from a run on the L4. The cost that stays is that the pack is the memo unit, so one failure retrains every seed in it. Training is a few dollars per experiment, so the main gain would be wall time.
