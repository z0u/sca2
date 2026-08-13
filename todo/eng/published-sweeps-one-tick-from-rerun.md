---
status: open
tags: [memoization, storage]
opened: 2026-07-27
---
# Published sweeps are one tick away from a full re-run after an evidence-scheme change

Widening what the fingerprint tracks re-stamps every task's evidence, so the next `mini run` re-runs the whole DAG in place, even though no experiment code moved. Adding a small step to ex-2.1.5 tripped this: the deferred-import tracing from #58/#59 landed after the sweep, so the tick re-ran `prepare_corpus` and would have re-trained all 24 cells. Cost is the smaller half of the problem; the real one is that a re-trained sweep may not reproduce the numbers a published report already quotes (determinism landed after that run too), so the report and the store would silently disagree. Nothing to fix in the mechanism itself — over-invalidation is the right bias — but two things would help. A read-only `mini plan <exp>` that lists what a tick would launch and why, so the choice to re-run is made before the launch and not after; and something that records, per published ref, the evidence the run was produced under, so "this report's numbers predate the current scheme" is a fact the report can state rather than a thing you rediscover. The workaround for now is what `docs/m2/ex-2.1.5/cross_eval.py` does: read the published checkpoints from a standalone script and write results back under their own ref, leaving the DAG alone.
