---
status: open
tags: [modal, performance]
opened: 2026-11-01
---
# Reserve a core for the training role

Since the batch prefetch (#328), a training task uses about 0.8 of a core on an L40S: the prefetch thread draws batches while the main thread dispatches. The train roles reserve nothing (Modal's default, 0.125 cores), so everything above that is borrowed from idle cores on the host. Modal bills the larger of reserved and used, so the borrowing already costs what a reservation would; what it lacks is the guarantee. On a busy host a task could fall back toward its reservation and leave the card waiting on batches, as the host-bound L4 step did (`training-step-is-host-bound`).

Setting `cpu=1` on the train roles would cost about 0.2 of a core more per task than now and hold the core. The fixed-labels crossed runs (16 on L40S, 2026-11-01) showed the 0.8 cores at about 80% GPU utilization. Worth a check with `bin/mini sysmetrics` on a few runs before and after, since the earlier probe found a reservation made no difference to the L4 step.
