---
status: open
tags: [experiments, storage, measurement]
opened: 2026-10-24
---
# Keep a few checkpoints along training

Training runs keep only their end checkpoint. When [spill-timing](/docs/m2/spill-timing/report.py) asked when the spill appears, it had to train `pool 1–3, soft τ` again at eight seeds to apply the edit along the way. The [loss-rate screen](/docs/m2/loss-rate/report.py) left epoch-64 checkpoints only because its runs stopped there, and those had to be pinned with refs to outlive the memo records.

A checkpoint of the d64, four-block model is under a megabyte, so keeping one at epochs 16, 32, 64, 128 and 256 beside the end checkpoint would add a few megabytes per run. Any later question about when something appears (spill, the lean, a lost anchor) could then be scored on stored runs, at a coarse spacing, without retraining. Proposed in the planning of spill-timing.

The change goes in the shared training loop (`train_one` in the [ex-2.2.21 module](/docs/m2/ex-2.2.21/experiment.py), which later experiments reuse), so it changes the memo evidence of every experiment that trains through it, unless it is an opt-in argument.
