"""Pack HLO: what the GPU compiler makes of a pack of seeds, so the extra bytes a pack moves per seed can be named.

The pack-timing probe found that a `jax.vmap`ped pack of seeds saturates an L40S's memory bus and moves two to five
times the bytes per seed that one seed moves (`/mnt/project-files/pack-timing/why-packing-doesnt-pay.md`). On the
CPU the pack scales as it should, so the extra bytes come from how the GPU backend compiles the vmapped program.
This scout looks at that program.

Each task compiles the scanned dispatch `train_anchored` runs, unvmapped (`plain`) and vmapped over a pack of
`PACK` seeds, on the recipe of record (`τ_ℓ fixed`, as `fixed-confirm` trains it) at the pack-timing probe's point
on the schedules, and keeps for each:

- the optimized HLO module (`as_text` of the compiled program): the fusions, the GEMM custom calls with their
  shapes and layouts, and the copies and transposes between them;
- XLA's cost analysis of the whole program (flops and bytes accessed);
- a profiler trace of a few dispatches (`jax.profiler`, written as a Perfetto JSON), for each kernel's time on the
  card, if the container lets the profiler see the card;
- the time per dispatch, as a check against the pack-timing numbers.

Everything is put to the store as one tree per task; the comparison happens on the driver's side, from the files.
Corpus preparation is the pack-timing probe's (a memo hit); the tasks run under the dev profile, and nothing here is
read by a report.

    MINI_PROFILE=dev bin/mini run docs/m2/pack-hlo/experiment.py --app modal --max-containers 4 --budget 1h
    MINI_PROFILE=dev bin/mini results --full pack-hlo --app modal
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir
from mini.orchestration import Pending


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


pt = _load_sibling("pack-timing", "pack_hlo_pt")
fc = pt.fc
ex2221 = pt.ex2221
ex2216 = pt.ex2216

PACK = 8
"""Seeds in the vmapped pack: the size the pack-timing probe's fifth round measured on the L40S (bus 32% → 99%)."""
SEEDS: tuple[int, ...] = pt.SEEDS[:PACK]
TIMED = 8
"""Dispatches timed per program, after `WARMUP`."""
PROFILED = 3
"""Dispatches inside the profiler trace."""
WARMUP = 3
VARIANTS: dict[str, tuple[str, dict[str, str]]] = {
    "L40S-off": ("L40S", {"XLA_FLAGS": pt.NONDET_XLA_FLAGS}),
    "L40S-on": ("L40S", {}),
}
"""Role name → (card, environment over the project baseline). `off` is the pack-timing fifth round's setting
(determinism flags off, where the byte inflation was measured); `on` is the project's baseline, which training runs
use today."""


def _trace_kernels(trace_dir) -> dict | None:
    """Kernel time on the card from the Perfetto trace under *trace_dir*: total and by kernel name, in µs.

    Returns `None` if there is no trace, or no events on a GPU process (the profiler could not reach the card).
    """
    import gzip
    import json
    from collections import defaultdict
    from pathlib import Path

    paths = sorted(Path(trace_dir).rglob("*.json.gz"))
    if not paths:
        return None
    with gzip.open(paths[-1], "rt") as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    names = {e["pid"]: e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu = {pid for pid, n in names.items() if "GPU" in n or "gpu" in n}
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.get("ph") == "X" and e.get("pid") in gpu:
            row = by_name[e["name"]]
            row[0] += float(e.get("dur", 0.0))
            row[1] += 1
    if not by_name:
        return {"processes": sorted(set(names.values())), "total_us": 0.0, "kernels": []}
    rows = sorted(((n, us, c) for n, (us, c) in by_name.items()), key=lambda r: -r[1])
    return {
        "processes": sorted(set(names.values())),
        "total_us": sum(r[1] for r in rows),
        "kernels": [{"name": n, "us": us, "count": c} for n, us, c in rows[:60]],
    }


def inspect_packs(corpus, labels, rows: list[dict], variant: str) -> dict:  # noqa: C901 — one setup, then one pass per program
    """Compile, dump, time and profile the plain dispatch and a pack of `PACK`, in this container."""
    import json
    import time

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec, sample_anchored_batches
    from sca.compute.data_pipelines import load_data
    from sca.compute.training import SCAN_STEPS, _anchored_step, _anti_at, _scanned
    from sca.data.batches import batches_per_epoch, split_data
    from sca.data.incontext import context_length
    from sca.data.named_colors import WordTokenizer
    from sca.model import build_model
    from sca.training.optimizer import configure_optimizer
    from sca.training.scheduler import configure_schedule
    from mini.progress import emit_progress
    from mini.store import get, put

    workdir = get_data_dir() / "pack-hlo" / variant
    out_dir = workdir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = get(corpus, workdir / "corpus")
    config = rows[0]["config"]
    anchor = AnchorSpec(**rows[0]["anchor"])
    anti = AntiSpec(**rows[0]["anti"]) if rows[0]["anti"] is not None else None
    tokenizer = WordTokenizer(config.tokenizer)
    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=fc.OP_NAMES.index(fc.ANCHORED_OP),
            label_rate=ex2216.LABEL_RATE,
            variant="whole",
            context_len=z["context_len"],
        )
        loss_mask = z["loss_mask"] if "loss_mask" in z.files else None
    data, _ = load_data(corpus_dir)
    train_data, _ = split_data(data, config.data.train_split)
    epoch_length = batches_per_epoch(len(train_data), config.data, config.model)
    min_line = context_length(fc.K)

    template = build_model(config.model, key=jr.key(0))
    schedule = configure_schedule(config.scheduler, config.optimizer.learning_rate, epoch_length)
    optimizer = configure_optimizer(template, config.optimizer, schedule)
    n_lines = config.model.block_size // min_line + 2
    run = _scanned(
        _anchored_step(optimizer, anchor, n_lines, None, 0.0, 0.0, rows[0]["anchor_slices"], None, fc.sps.EVERY),
        SCAN_STEPS,
        anchor.scheduled_tau,
    )
    ats = [pt.AT_EPOCH + i / epoch_length for i in range(SCAN_STEPS)]
    weights = np.asarray([float(anchor(a)) for a in ats], np.float32)
    anti_weights = np.asarray([_anti_at(anti, a) for a in ats], np.float32)
    taus = np.nan_to_num(np.asarray([anchor.taus(a) for a in ats], np.float32), nan=1.0, posinf=1.0)
    live = np.ones(SCAN_STEPS, bool)

    def draw(seed: int) -> tuple:
        rng = np.random.default_rng(seed)
        gen = sample_anchored_batches(
            train_data, config.data, config.model, epoch_length, rng, spec, anchor.span, lines=True,
            crop=ex2221.CROP_POLICY, newline_id=tokenizer.stoi["\n"], min_line_tokens=min_line, loss_mask=loss_mask,
        )  # fmt: skip
        chunk = [next(gen) for _ in range(SCAN_STEPS)]
        return tuple(np.stack([b[f] for b in chunk]) for f in range(len(chunk[0])))

    total = 2 * (WARMUP + TIMED + PROFILED)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        emit_progress(done, total)

    results = {}
    for pack in ("plain", PACK):
        name = str(pack)
        seeds = list(SEEDS[:1] if pack == "plain" else SEEDS)
        if pack == "plain":
            model = build_model(config.model, key=jr.key(seeds[0]))
            opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
            step = run
            batches = draw(seeds[0])
        else:
            model = eqx.filter_vmap(lambda key: build_model(config.model, key=key))(
                jnp.stack([jr.key(s) for s in seeds])
            )
            opt_state = eqx.filter_vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_inexact_array)))(model)
            step = eqx.filter_jit(eqx.filter_vmap(run, in_axes=(0, 0, 0, None, None, None, None)))
            per_seed = [draw(s) for s in seeds]
            batches = tuple(np.stack(f) for f in zip(*per_seed, strict=True))
        batches = jax.device_put(batches)
        args = (weights, anti_weights, taus, live)

        t0 = time.perf_counter()
        lowered = step.lower(model, opt_state, batches, *args)  # ty: ignore[unresolved-attribute] — both are `filter_jit` wrappers
        compiled = lowered.compile()
        compile_s = time.perf_counter() - t0
        hlo = compiled.compiled.as_text()
        (out_dir / f"{name}.hlo.txt").write_text(hlo)
        (out_dir / f"{name}.pre.txt").write_text(lowered.lowered.as_text(dialect="hlo"))
        try:
            cost = compiled.compiled.cost_analysis()
            cost = cost[0] if isinstance(cost, list) else cost
            cost = {k: float(v) for k, v in (cost or {}).items() if k in ("flops", "bytes accessed", "transcendentals")}
        except Exception as e:  # noqa: BLE001 — the analysis is optional on some backends
            cost = {"error": repr(e)}
        try:
            m = compiled.compiled.memory_analysis()
            memory = {
                k: int(getattr(m, k))
                for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
                if hasattr(m, k)
            }
        except Exception as e:  # noqa: BLE001
            memory = {"error": repr(e)}

        for _ in range(WARMUP):
            model, opt_state, outs = compiled(model, opt_state, batches, *args)
            jax.block_until_ready(outs)
            tick()
        times = []
        for _ in range(TIMED):
            t = time.perf_counter()
            model, opt_state, outs = compiled(model, opt_state, batches, *args)
            jax.block_until_ready(outs)
            times.append(time.perf_counter() - t)
            tick()
        device_ms = 1000 * float(np.median(times)) / (SCAN_STEPS * len(seeds))

        trace_dir = out_dir / f"{name}-trace"
        profile_error = None
        try:
            with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
                for _ in range(PROFILED):
                    model, opt_state, outs = compiled(model, opt_state, batches, *args)
                    jax.block_until_ready(outs)
                    tick()
        except Exception as e:  # noqa: BLE001 — a profiler failure leaves the HLO, which is the main output
            profile_error = repr(e)
        kernels = _trace_kernels(trace_dir)
        if kernels is not None:
            (out_dir / f"{name}-kernels.json").write_text(json.dumps(kernels, indent=1))

        results[name] = {
            "seeds": len(seeds),
            "compile_s": compile_s,
            "device_ms": device_ms,
            "cost": cost,
            "memory": memory,
            "hlo_lines": hlo.count("\n"),
            "profile_error": profile_error,
            "kernel_total_us": None if kernels is None else kernels["total_us"],
            "kernel_top": None if kernels is None else kernels["kernels"][:12],
        }
        print(f"{name}: device {device_ms:.3f} ms per seed-step, compile {compile_s:.0f} s, cost {cost}", flush=True)
        del model, opt_state, outs, compiled, lowered

    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    return {
        "variant": variant,
        "device": jax.devices()[0].device_kind,
        "programs": results,
        "files": put(out_dir, name=f"pack-hlo-{variant}"),
    }


def main(ctx: Ctx) -> dict:
    prep = ctx.run(
        ex2221.prepare_condition,
        fc.OP_NAMES,
        fc.K,
        ex2221.RHO,
        0.0,
        ex2221.CUBE_RATE,
        pt.N_LINES,
        ex2221.CORPUS_SEED,
        ex2216.HOLDOUT_CONTEXTS,
        ex2216.N_TRAJ_PROBE,
        f"{ex2221.MAIN_KEY}-n{pt.N_LINES}",
        role="prep",
    )
    rows = fc.rows_of(prep["meta"], SEEDS, None)
    done, waiting = {}, []
    for variant in VARIANTS:
        try:
            done[variant] = ctx.run(inspect_packs, prep["corpus"], prep["labels"], rows, variant, role=variant)
        except Pending as p:
            waiting.append(p)
    if waiting:
        raise waiting[0]
    return done


COMPUTE: dict[str, Any] = {"prep": pt.COMPUTE["prep"]} | {
    # Two compiles of 10 to 60 seconds, then a few dozen dispatches.
    variant: dict(gpu=card, env=env, timeout=1800, watchdog=600, watchdog_grace=900)
    for variant, (card, env) in VARIANTS.items()
}

experiment = Experiment(name="pack-hlo", main=main, roles=COMPUTE)
