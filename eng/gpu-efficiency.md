# Spending GPU time on arithmetic

*Part of the [engineering notes](./README.md).*

Our models are small (d64, four layers), so on a GPU most tasks are bound by latency and fixed costs: kernel launches, compilation, container start, uploads. The arithmetic is the cheap part. That shapes which choices matter, and this note records them as conventions for new experiments, with the measurements behind each.

## Conventions

1. **Default to an L4.** A training step takes about 10.5 ms on an L4, 9.3 ms on an A10 and 7.7 ms on an L40S, while the hourly price goes $0.80, $1.10, $1.95. Per step, the L4 is cheapest; a T4 ($0.59) is half the speed. Bigger batches would use the larger cards better, but batch size is a science parameter here, so it isn't ours to tune for cost. Measurements in [determinism](./determinism.md#re-measured-at-full-loop-length-2026-09-27). Since the loop runs sixteen steps a dispatch, the cards differ more: on 2026-10-30 the recipe-of-record step took about 8.7 ms on an L4, 4.5 on an L40S and 13.6 on an A10G, so the L4 is still cheapest per step, the L40S comes second, and the A10G costs about twice the L4 ([`scatter-free-training-step`](/todo/eng/scatter-free-training-step.md)). Packing several seeds onto one card does not help ([`seed-packs-on-larger-gpus`](/todo/eng/seed-packs-on-larger-gpus.md)).
2. **Keep the deterministic XLA flags.** At full loop length they cost a few seconds of compilation per task and nothing per step, and without them replicas of one run drift apart (see [determinism](./determinism.md)).
3. **Jit once, at module level, and pass what varies as arguments.** JAX caches a compiled function per *function object*, keyed on the shapes and static values of its arguments. A closure or lambda built inside a loop (`eqx.filter_jit(lambda m, x: m(x, op))`) is a new function object each time, and whatever it closes over is baked into the program, so every call compiles from scratch. Define the jitted function once, and pass the model, the batch and any operator in as arguments. An operator with array parameters is an `eqx.Module`: arrays are leaves (traced, so a new subspace of the same shape reuses the program) and Python scalars that select code paths are `eqx.field(static=True)`. `_stream_axis` in `sca/anchoring.py` and `_forward_jit` in `sca/intervention.py` are the models to copy.
4. **Keep shapes fixed across calls.** Each new shape is a new compile. Batch with a fixed `batch_size` and let the last chunk be the only odd one. A static argument (`slices`, `axes`) compiles once per value, which is fine for a handful of values and costly for hundreds.
5. **When a task is slower than its arithmetic, count compiles first.** `jax.config.update("jax_log_compiles", True)` logs one line per compile with the function name, and `jax.monitoring.register_event_duration_secs_listener` receives `/jax/core/compile/backend_compile_duration` events to total the time. A name that repeats dozens of times is the lead.
6. **Leave the volume commit alone.** A task commits the experiment Volume after writing its result and before recording `DONE`, so a reader that sees `DONE` can always read the result. It costs about 3 s per task. Modal's background sync would get the bytes there eventually, but with no ordering against the record.

## Why no persistent compilation cache

JAX can write compiled programs to disk (`JAX_COMPILATION_CACHE_DIR`) and load them in a later process. We tried it on the shared cache Volume and it doesn't carry across Modal containers: the cache key includes a fingerprint of the GPU topology (`accelerator_config` in `jax/_src/cache_key.py`), and that fingerprint differs from one L4 container to the next. Every other component of the key matched. Each container misses, compiles, and writes its own entries, so the cache adds Volume writes and no hits. Within one process it does dedup identical programs, which rescued the old scoring code (below), but jitting once gets the same result without it. Worth re-checking after a JAX upgrade; the probe logs the key components with `logging.getLogger("jax._src.cache_key").setLevel(logging.DEBUG)`.

## What we measured: the scoring pass (2026-09-27)

In ex-2.2.12 and ex-2.2.13 the scoring role used more L4 time than training: 160 against 91 GPU-minutes, and 477 against 324. One ex-2.2.13 `score_one` compiled 112 programs, 74 of them the forward pass in `intervention.apply`, which built a new jitted lambda for every (operator, subspace) pair. On CPU:

| variant | wall | compiling | compiles |
| --- | --- | --- | --- |
| before | 225 s | 166 s | 112 |
| before, cold persistent cache | 68 s | 25 s | |
| before, warm persistent cache | 44 s | 2.5 s | |
| jit once | 53 s | 22 s | 51 (11 of them `_forward`) |
| jit once, warm persistent cache | 30 s | 0.5 s | |

The cold-cache row is faster than "before" because the cache dedups identical programs within one process: the 74 lambdas lowered to a few distinct programs. (The warm rows are one local process reading another's cache, which works on CPU; on Modal it doesn't, see above.) Every variant's output was bit-identical to the original.

On L4, four ex-2.2.13 runs each, one per single-use container:

| code | wall | compiling | compiles |
| --- | --- | --- | --- |
| before | 113–208 s (median 164) | 91–175 s | 112–134 |
| jit once | 35–40 s | 17–21 s | 51–54 |

That's about 4.5× less wall time per task. Scoring was 477 GPU-minutes of ex-2.2.13, so at this ratio it would have been about 100.

Old and new code agree on the GPU to float32 rounding (largest absolute difference 6e-6 over about 10,000 scalars per run), and so does the *same* code run in two containers: in two of the four runs, the old code alone gave results that differed at that level. So the scoring pass is reproducible across GPU containers to rounding and not to the bit, even with the deterministic flags, which means a re-run can change the content hash of a score artifact. In ex-2.2.x the step downstream of the scores is `publish_results`, a CPU task, so a re-run that moves a hash costs one cheap re-publish; a step that feeds scores into more GPU work would pay more.
