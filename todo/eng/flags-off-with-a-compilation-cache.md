---
status: done
tags: [performance, modal, determinism]
opened: 2026-10-31
closed: 2026-10-31
---
# Determinism flags off, with a shared compilation cache

With no scatter left in the training step ([`scatter-free-training-step`](./scatter-free-training-step.md)), the GPU determinism flags cost about a sixth of a step on an L40S (2.42 ms a step with them, 2.0 to 2.1 without). Turned off alone, they cost reproducibility: in the step-speed probe (`docs/m2/step-speed`), six L40S containers and four L4 containers each ended on weights of their own. With JAX's compilation cache in a directory on the Volume, filled by one task and loaded by three more, the four L40S runs agreed bit for bit, and so did three L4 runs, at 5.6 to 5.7 ms a step against 6.74 with the flags ([eng/determinism.md](/eng/determinism.md#with-no-scatter-left-2026-10-31)). So the autotuner's timed picks look like the remaining source of drift, and pinning the compiled program removes it.

Adopting it would mean a run reproduces only while its compiled program is in the cache. Things to settle first:

- The first compile per card and per program. A fan-out launched cold has every task compile and time its own picks, so the seeds of one launch would run on different programs, and a later re-run of a seed would load whichever entry was written last. The workers could settle it among themselves through the Modal Dict, which already carries the task records: `Dict.put(key, value, skip_if_exists=True)` returns whether the put happened, so it works as a claim. The key would name the card, the JAX and XLA versions, the flags and a hash of the lowered program (cheap to get before compiling, from `jit(...).lower(...)`). The worker whose claim lands compiles, writes the cache entry, commits the Volume and marks the key ready; the others wait for the mark, reload the Volume and load the entry. The claim needs a lease, so that a worker that dies while compiling doesn't hold the rest up, and the waiting is a compile's worth of idle card per task, ten to thirty seconds. Locally, a lock file does the same job.
- Losing the cache. Deleting it, or a `gc` that reaches it, would let one memo key map to two numerically different results with no warning, the same quiet failure a JAX upgrade has. Recording the cache entry each attempt loaded (or whether it compiled afresh) beside `XLA_FLAGS` on the record would make that visible, as `numerics_drift` does for package versions.
- Concurrent writers on the Modal Volume, and how a reader sees an entry written after its container started.
- XLA's own per-fusion autotune cache (`--xla_gpu_per_fusion_autotune_cache_dir`) is a narrower alternative: it pins the picks and leaves the rest of compilation alone. Untested here.

The flags stay on until this is settled. The numbers are from the fourth round of the probe.

## Notes

2026-10-31 — Done, and the flags are off. `mini.compile_cache` settles each point above: first compiles are claimed through the Modal Dict with a lease (locally a `flock`), entries are write-once, and each attempt records `compile_cache` (counts, a digest over the entries it used, and the machine parts of the key). The cold fan-out turned up one more thing: JAX's key splits a card model by host pool (AWS and OCI L40S hosts each got their own programs and their own weights), so the key names the card model instead. Ten flags-off L40S runs over both pools, six launched cold together, compiled once and agreed bit for bit at 1.94 to 2.03 ms a step; six flags-on runs gave 2.36 to 2.46. The per-fusion autotune cache was not needed. Details and the deterministic mode for exact reproductions of older runs are in [eng/determinism.md](/eng/determinism.md#a-shared-compilation-cache-2026-10-31).
