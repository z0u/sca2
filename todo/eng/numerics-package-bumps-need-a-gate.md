---
status: open
tags: [determinism, memoization]
opened: 2026-08-05
---
# Numerics-relevant package bumps need a gate

Found while working on #73.

A jax 0.10.1 → 0.11.0 upgrade changed the final-weights digest of a fixed nGPT d64-L4 run on a fixed L4 (`e4b7c106…` → `884ee00c…`, loss drift ~1 part in 4×10⁶), while the memo key stayed identical — package versions feed neither `task_key_parts`' identity nor its validity evidence, since `_is_project_file` excludes site-packages. So a DONE record survives the upgrade serving its old number, and a re-run for any other reason quietly writes a new one under the same key. Written up in `eng/determinism.md`. Options, none implemented: fold a pinned set of numerics packages into the evidence (invalidates broadly, and every past record with it); surface a warning when a record's `compute_env` numerics differ from the current environment (cheap, detection-only, probably the right first move); or keep it manual and bump `version=` per affected task on each numerics upgrade.
