---
status: done
tags: [cli, local]
opened: 2026-09-18
closed: 2026-09-27
---
# The memoized local backend spawns every staged task at once

`LocalApparatus.spawn_tasks` (the path `bin/mini run` takes for a memoized experiment) loops over the batch and calls `spawn_taskworker` for each record, so a `ctx.map` over N inputs starts N detached processes on the same tick. `max_workers` (`--workers`) only bounds the interactive `amap` thread pool; the memoized path never reads it. A map over 65 checkpoints on `geometry-rsa` (the whole-geometry read in `docs/m2/geometry-rsa/`) started 65 JAX processes on a small box, took the load average past 140, and had to be killed; the workaround was to chunk the inputs in the experiment (`SEEDS_PER_TASK`) and cap BLAS/XLA threads through the role's `env`, which is a workaround the experiment should not have to carry.

The fix is a concurrency bound on the memoized path: stage every record, spawn at most `max_workers` of them, and let the next tick (or a small local queue) start the rest as workers exit. `_is_task_alive` already gives the tick a way to count live workers. Modal's path is unaffected, since each task is its own container.

## Notes

- 2026-09-27: done. `LocalApparatus.spawn_tasks` stages every call and starts at most `--workers` (default: the CPU count, where it used to be 1 for the thread pool only). The overflow stays RUNNING with no pid, which `status` already shows as queued. `launch_queued` fills free slots when a batch is staged, when a worker exits, and on each tick and watch poll, so a detached run drains its queue with no driver. geometry-rsa's `SEEDS_PER_TASK` chunking and its thread caps are left alone: they still earn their place (fewer JAX start-ups, fewer threads per worker). Each queued task's env overlay waits in a launch spec (mode 0600, deleted when the worker starts) outside the checkout and in memory where possible: `$XDG_RUNTIME_DIR/mini/launch/`, else a private `/dev/shm/mini-<uid>/` (checked for owner and mode, since `/dev/shm` is shared), else `$XDG_STATE_HOME` on disk (macOS). A spec wiped by a reboot or logout is re-staged by the next wake, once a `<key>.gen` marker shows the claiming tick has staged that attempt's call; the launching shell's own values travel in the worker's environment, never on disk. Each wake re-stages the tasks still waiting with its own config, so a restart or an edited `env=` applies to them. A worker's pid is recorded with its boot and start time (`pid_start`), so after a restart a reused pid reads as dead to the reaper and the slot count, and `cancel` doesn't signal the process that now holds it.
