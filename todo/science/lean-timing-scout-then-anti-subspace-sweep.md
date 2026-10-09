---
status: done
tags: [D2.2, anchoring, schedules, in-context]
opened: 2026-10-08
closed: 2026-10-09
---
# When the lean appears, then the anti-subspace schedule

Two steps, in this order.

**A tiny scout of when the lean appears.** A few seeds at 400 epochs on the current schedule, recording α per role along the trajectory (and keeping a few mid-run checkpoints), to see when in training the alignment builds up at the positions that carry the spill. The stored trajectories of ex-2.2.23 and [tau-lambda-sweep](/docs/m2/tau-lambda-sweep/report.py) hold only pooled measurements (the op margin and the leans on the probe set), and only final checkpoints are kept, so the look at stored data in [the schedules item](/todo/science/schedules-at-the-rise.md) could compare roles only at the end of training. The change to `train_one` is small: score α by role in its `on_record` hook.

**Then a sweep of the anti-subspace schedule**, holding the anchor schedule fixed: its weight, its anneal, or both. The anti-subspace weight falls from 2.5 λ_a to 0.3 λ_a by 90% of training, almost in step with the learning-rate cosine, so no run so far can tell a weak anti-subspace term late in training from training length itself. Two things to carry in. A flat weight did not work well in D2.1: ex-2.1.11's brackets found that flat at the hold ratio latched onto `+`, and flat at the peak ratio cost grading. And in tau-lambda-sweep, raising λ_a sixteenfold (which scales the anti-subspace term with it) did not lower the spill, though the pull grew too, so that does not isolate the anti-subspace term. What the scout shows about timing should decide which part of the schedule to sweep.

## Notes

**2026-10-09, tau-lambda-sweep** — Superseded by the [embedding-lean report](/docs/m2/embedding-lean/report.py). It found the lean by 200 epochs, and the remaining timing question rides on the next run as [record α per slice and per role](/todo/eng/record-per-slice-per-role-alpha-along-training.md). The anti-subspace sweep gives way to [an anti weight of its own](/todo/eng/detach-anti-weight-from-anchor-weight.md) and the pull-side items ([pull-slice-restrictions](./pull-slice-restrictions-at-more-seeds.md), [mellowmax over slices](./mellowmax-over-slices.md)).
