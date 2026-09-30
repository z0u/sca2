---
status: open
tags: [local]
opened: 2026-09-28
---
# A local claim whose tick died before staging stays RUNNING forever

A tick claims a key (`mark_running`: RUNNING under a new `gen`, no `pid`) before `LocalApparatus.spawn_tasks` stages its call and launch spec. If the tick dies in between, the record is RUNNING with no pid and nothing staged for its attempt. `launch_queued` passes it over (it no longer takes a worker slot), but `_is_task_alive` treats every no-pid record as alive ("can't probe; assume alive"), so `reap_dead` never settles it, and `--watch` waits on it until someone runs `mini cancel` or `retry`. This predates the worker cap.

A fix: let `reap_dead` settle a RUNNING, no-pid local record whose `MemoStore.staged_gen` doesn't match its `gen` once it is older than a grace period (a claiming tick stages within seconds, so a minute is plenty). The record's `created_at` or claim time gives the age. It should settle as FAILED like any other reaped task, so recovery stays a deliberate `retry`.
