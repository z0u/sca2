---
status: open
tags: [cli, local]
opened: 2026-09-18
---
# The memoized local backend spawns every staged task at once

`LocalApparatus.spawn_tasks` (the path `bin/mini run` takes for a memoized experiment) loops over the batch and calls `spawn_taskworker` for each record, so a `ctx.map` over N inputs starts N detached processes on the same tick. `max_workers` (`--workers`) only bounds the interactive `amap` thread pool; the memoized path never reads it. A map over 65 checkpoints on `geometry-rsa` (the whole-geometry read in `docs/m2/geometry-rsa/`) started 65 JAX processes on a small box, took the load average past 140, and had to be killed; the workaround was to chunk the inputs in the experiment (`SEEDS_PER_TASK`) and cap BLAS/XLA threads through the role's `env`, which is a workaround the experiment should not have to carry.

The fix is a concurrency bound on the memoized path: stage every record, spawn at most `max_workers` of them, and let the next tick (or a small local queue) start the rest as workers exit. `_is_task_alive` already gives the tick a way to count live workers. Modal's path is unaffected, since each task is its own container.
