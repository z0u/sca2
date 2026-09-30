---
status: done
tags: [performance, modal, experiments]
opened: 2026-09-27
closed: 2026-09-30
---
# Pack several training runs onto one GPU

Training a d64-L4 model leaves the GPU mostly idle, so the cheapest way to cut the training bill is to give each GPU more to do per step without changing any run's recipe.

The evidence is from a timing probe of ex-2.2.14's `train_one` (2,970 steps, two containers per cell; the table is in [`eng/determinism.md`](/eng/determinism.md)). The steady-state step took about 10.5 ms on an L4, 9.3 ms on an A10, and 7.7 ms on an L40S, which has roughly four times the L4's compute. A step that barely speeds up on a much larger card is latency-bound: kernel launches, Python dispatch, and the `float(loss)` sync each step, rather than arithmetic. Host-side batch sampling is not the cause (about 2 s of an 80 s task).

Per step, the L4 is already the cheapest card on Modal's price list (about 2.3 µ$ against 3.1 for T4, 2.9 for A10, 4.2 for L40S), so a GPU swap has nothing to offer. Larger batches would fill the card, but batch size is part of the recipe and would change what the experiments measure.

What keeps the recipe is giving one GPU several cells at a time. There are three ways to do it, which differ in how much they disturb numerics and the memo.

### Option 1: concurrent tasks in one container (numerics and memo unchanged)

Let one container run K ordinary `train_one` calls at once on the same GPU, e.g. through Modal's input concurrency (`@modal.concurrent(max_inputs=K)`). Each cell is still its own task, record and memo key, and it runs the same kernels at the same shapes, so its bytes match an unpacked run under the determinism flags. Because the step is latency-bound, K streams can interleave on a card that one stream leaves idle.

What it needs: mini support for a concurrency option on a role; `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or a memory fraction) so K JAX clients fit; and per-task state that is already per-context (`get_data_dir` is a `ContextVar`; progress and the watchdog would need checking). The watchdog's `os._exit` would take down every task in the container, so a stall would cost K cells. The unknown is the GIL: if Python dispatch is most of the step, threads in one process won't overlap much. A probe of K = 1, 2, 4 on an L4 answers that before any mini work.

### Option 2: a packed task, by seed (simplest code, new numerics)

Write `train_pack(configs, ...)` that `jax.vmap`s the train step over a leading axis of models, optimizer states and batches, and returns one result per seed. Packing by seed is the easy case: every slot in a pack shares a condition, so the anchor and anti schedules stay scalars per step, and only the seed (init key and batch RNG) differs per slot.

The pack is the memo unit: one task and one record per pack, so a failure retrains all its seeds. Downstream can still be per cell if indexing a pack result (`packs[i][j]`) keys `eval_one` on the element's content. Worth checking in `mini.orchestration` before relying on it. Batched kernels are a different computation, so a packed run won't reproduce an unpacked one bit for bit. Fine for a new experiment; it can't be slipped under an existing one's memo.

### Option 3: a packing helper outside the memo path (per-cell memo, batched execution)

Keep one task per cell in the DAG, and let the apparatus gather pending cells of the same function into one container call that runs a vmapped implementation, then settle each cell's record separately. This is option 2's throughput with option 1's per-cell records. It is also the most mini work: the fn needs a declared batched twin, and the scheduler has to group, split results, and fail cells individually.

The numerics catch is sharper here: a cell's bytes would depend on which cells shared its pack and how many, so a cell re-run alone after a failure would not match its first attempt. A fixed pack width (padding empty slots) probably makes each slot's arithmetic independent of its neighbours. A probe would need to confirm that (slot 0 bit-identical across different pack-mates) before trusting the memo with it.

### Suggested order

Probe option 1 first, since a win there costs no science and no memo changes. If threads don't overlap, option 2 by seed is a contained change in `sca.compute.training` plus one experiment's DAG. Option 3 is worth it only if packing becomes the default for every experiment.

### Probe recipe

The 2026-09-27 probe was a throwaway mini experiment: it wrapped `train_one` and swapped `sca.compute.training.sample_anchored_batches` for a generator that timestamps each batch, one role per variant with `single_use_containers=True`, run under `MINI_PROFILE=dev`.

## Notes

**2026-09-27, ex-2.2.16 review** — Sandy asks whether the pilot could pack its seeds. It fits option 2 (every arm has three seeds at one condition, and the pilot has no memo to keep), but its training comes to about $2 over 42 runs, so packing would save pennies and some wall time there. The pilot does not wait on it; a larger sweep is the better first user.

**2026-09-28, ex-2.2.17 round 6** — The d128-L4 run trains at about 2,100–3,100 steps/min, the same as d64-L4, so doubling the width costs nothing per step: more evidence the step is latency-bound, and that a pack of four d64 seeds would likely run in the time of one. The scouting rounds are where this now pays: eight- and sixteen-times runs of about 105k–211k steps, three seeds per arm, and ex-2.2.17 has passed $13. Sandy asks about packing seeds again. A fourth, cheaper step to try alongside option 1: keep the per-step `float(loss)` and metric values on the device and sync every N steps (they only feed `emit_metrics`), which removes one host round trip per step for every run, packed or not.

**2026-09-30, housekeeping** — Closed, with Sandy. Probe 4 in [`training-step-is-host-bound`](./training-step-is-host-bound.md) answered it: a `lax.scan` over the steps cut a d64 step from about 10 ms to 3–4, and once the step is scanned, vmapping 8 seeds costs about 8× one, so packing adds little. The scanned loop landed in `train_anchored` in #228. The next ways to cut the training bill are fewer steps or epochs per run (see [`cheaper-center-control-recipe`](/todo/science/cheaper-center-control-recipe.md)) and changes to the architecture.
