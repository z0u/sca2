"""Pack timing: the cost of a training step when several seeds share one GPU, on each card we might train on.

`todo/eng/seed-packs-on-larger-gpus.md` asks whether a pack of seeds, `jax.vmap`ped over a leading axis of models,
optimizer states and batches, costs less per seed-step than one seed alone on a card larger than the L4. On an L4 the
answer was no (8 seeds cost about 8× one, `todo/eng/training-step-is-host-bound.md`), but the larger cards have far
more memory bandwidth and compute to fill.

Each task times every pack size in one container, one after another, on the recipe of record (`τ_ℓ fixed`, as
`fixed-confirm` trains it) at a mid-training point of its schedules, so the anchor and anti terms are both on. A pack
of *K* is the scanned dispatch `train_anchored` runs (`SCAN_STEPS` steps), vmapped over *K* seeds; `plain` is the
unvmapped dispatch itself. Two timings per pack:

- `device`: the dispatch alone, on batches already on the device, synced at its end. What the card costs.
- `loop`: what `train_anchored` does per dispatch, for every seed of the pack: draw the batches on the host, stack
  them, dispatch, and read the losses back. What a pack would cost if `train_anchored` were vmapped as it stands,
  with the host sampling every seed in turn.

Corpus preparation and the timing tasks run under the dev profile; nothing here is read by a report.

    MINI_PROFILE=dev bin/mini run docs/m2/pack-timing/experiment.py --app modal --max-containers 15 --budget 2h
    MINI_PROFILE=dev bin/mini status pack-timing
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


fc = _load_sibling("fixed-confirm", "pack_timing_fc")
ex2221 = fc.ex2222.ex2221
ex2216 = ex2221.ex2216

CARDS: tuple[str, ...] = ("L4", "A10G", "L40S", "A100")
"""Modal GPU names: the three of the training fallback list, and one large card. `A100` is the 40 GB card. The H100
is left out until the A100 shows that a pack can fill a large card."""
PACKS: dict[str, tuple[int | str, ...]] = dict.fromkeys(CARDS, ("plain", 1, 8, 16, 32)) | {
    "A100": ("plain", 1, 8, 16, 32, 64)
}
"""Pack sizes per card, timed in this order (rotated between replicas). `plain` is the unvmapped dispatch."""
SEEDS: tuple[int, ...] = tuple(range(900, 900 + max(p for ps in PACKS.values() for p in ps if isinstance(p, int))))
"""Seeds no experiment trains, one per slot of the largest pack."""
AT_EPOCH = 100.0
"""Where on the schedules the weights and temperatures are read: past the τ_p anneal, with both terms on."""
STEP_BUDGET = 4096
"""Seed-steps timed per pack and per timing, so the small packs time enough dispatches to be steady."""
WARMUP = 3
"""Dispatches before the timed ones, which hold the compile."""
N_LINES = 60_000
"""A fifth of the recipe corpus: enough for an epoch longer than the timed span, and a quicker build. The corpus
size sets the epoch length and nothing in the step."""
N_REPLICAS = 3
REPLICAS = {"L4": 1}
"""Replicas by variant where not `N_REPLICAS`: the L4 ran a pack of 32 so slowly (about an hour a task) that one is
enough."""
PRICE_PER_HOUR = {"L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100": 2.10}
"""Modal list prices in USD (October 2026), for the cost per seed-step."""
NONDET_CARDS: tuple[str, ...] = ("A100",)
"""Cards also timed without the GPU determinism flags (`eng/determinism.md`). The first round on the A100 found the
cost per seed-step flat in the pack size; a deterministic scatter, as in an embedding gradient, runs serially and
would cost the same per seed however many share the card."""
NONDET_XLA_FLAGS = "--xla_cpu_enable_concurrency_optimized_scheduler=false"
"""The project's `XLA_FLAGS` with the two GPU determinism flags left out."""
ONE_HOT_CARDS: tuple[str, ...] = ("L40S",)
"""Cards timed in a third round, with the flags on, on the step with no scatter-add left in it: the slice selection
already static in `sca.anchoring` (since the second round), and here the embedding lookup and the cross-entropy as
products with a one-hot matrix in place of indexing (`_without_gathers`). The second round found the L40S's cost per
seed-step rising most with the pack size."""
VARIANTS: dict[str, tuple[str, dict[str, str]]] = (
    {card: (card, {}) for card in CARDS}
    | {f"{card}-nondet": (card, {"XLA_FLAGS": NONDET_XLA_FLAGS}) for card in NONDET_CARDS}
    | {f"{card}-onehot": (card, {}) for card in ONE_HOT_CARDS}
)
"""Role name → (card, environment over the project baseline)."""


def _without_gathers() -> None:
    """Swap the training step's two remaining gathers for one-hot products, in this process.

    The embedding lookup and the cross-entropy's pick of the target logit are gathers, whose gradients are
    scatter-adds. A product with a one-hot matrix selects the same values (at full precision, so no TF32 rounding) and
    has a matrix product for its gradient. Only `time_packs` calls this, in the `-onehot` roles' own containers.
    """
    import jax
    import jax.numpy as jnp

    import sca.anchoring
    from sca.model import ngpt

    highest = jax.lax.Precision.HIGHEST

    def embed(model, idx):
        table = model.transformer.wte
        return ngpt.normalize(
            jnp.matmul(jax.nn.one_hot(idx, table.shape[0], dtype=table.dtype), table, precision=highest)
        )

    def stream_and_logits(self, idx):
        x = embed(self, idx)
        states = [x]
        enc = self.transformer.rotary_enc
        allowed = self.attention_mask(idx)
        run_block = ngpt.eqx.filter_checkpoint(lambda block, h: block(h, enc, allowed))
        for block in self.transformer.blocks:
            x = run_block(block, x)
            states.append(x)
        return jnp.stack(states), (x @ self.transformer.readout.T) * self.s_z()

    def cross_entropy(logits, targets):
        logp = jax.nn.log_softmax(logits)
        losses = -(logp * jax.nn.one_hot(targets, logits.shape[-1], dtype=logp.dtype)).sum(-1)
        mask = targets != 0
        return jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1)

    ngpt.NGPT.stream_and_logits = stream_and_logits
    sca.anchoring.cross_entropy = cross_entropy


def time_packs(corpus, labels, rows: list[dict], packs: tuple, variant: str, replica: int) -> dict:  # noqa: C901 — one setup, then two timings per pack
    """Time the scanned dispatch for each pack size in *packs*, in order, in this container. *variant* names the
    role, so that each card and flag set has its own memo key.
    """
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
    from mini.store import get

    if variant.endswith("-onehot"):
        _without_gathers()
    workdir = get_data_dir() / "pack-timing" / f"r{replica}"
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
    # The step `train_anchored` builds for `ex2221.train_one` as the soft pool scout calls it (anti on every slice).
    run = _scanned(
        _anchored_step(optimizer, anchor, n_lines, None, 0.0, 0.0, rows[0]["anchor_slices"], None, fc.sps.EVERY),
        SCAN_STEPS,
        anchor.scheduled_tau,
    )
    # The weights and temperatures of one dispatch at AT_EPOCH, shared by every seed of a pack.
    ats = [AT_EPOCH + i / epoch_length for i in range(SCAN_STEPS)]
    weights = np.asarray([float(anchor(a)) for a in ats], np.float32)
    anti_weights = np.asarray([_anti_at(anti, a) for a in ats], np.float32)
    taus = np.nan_to_num(np.asarray([anchor.taus(a) for a in ats], np.float32), nan=1.0, posinf=1.0)
    live = np.ones(SCAN_STEPS, bool)

    def batch_stream(seed: int):
        rng = np.random.default_rng(seed)
        while True:
            yield from sample_anchored_batches(
                train_data, config.data, config.model, epoch_length, rng, spec, anchor.span, lines=True,
                crop=ex2221.CROP_POLICY, newline_id=tokenizer.stoi["\n"], min_line_tokens=min_line,
                loss_mask=loss_mask,
            )  # fmt: skip

    def draw(streams, packed: bool) -> tuple:
        """One dispatch of batches per stream, stacked: fields of shape (steps, B, …), or (seeds, steps, B, …)."""
        per_seed = []
        for s in streams:
            chunk = [next(s) for _ in range(SCAN_STEPS)]
            per_seed.append(tuple(np.stack([b[f] for b in chunk]) for f in range(len(chunk[0]))))
        return tuple(np.stack(f) for f in zip(*per_seed, strict=True)) if packed else per_seed[0]

    def n_dispatch_of(k: int) -> int:
        return max(8, STEP_BUDGET // (SCAN_STEPS * k))

    ks = [1 if p == "plain" else int(p) for p in packs]
    total = sum(WARMUP + 2 * n_dispatch_of(k) for k in ks)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        emit_progress(done, total)

    results = []
    for pack in packs:
        k = 1 if pack == "plain" else int(pack)
        seeds = [r["config"].seed for r in rows[:k]]
        streams = [batch_stream(s) for s in seeds]
        if pack == "plain":
            model = build_model(config.model, key=jr.key(seeds[0]))
            opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
            step = run
        else:
            model = eqx.filter_vmap(lambda key: build_model(config.model, key=key))(
                jnp.stack([jr.key(s) for s in seeds])
            )
            opt_state = eqx.filter_vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_inexact_array)))(model)
            step = eqx.filter_jit(eqx.filter_vmap(run, in_axes=(0, 0, 0, None, None, None, None)))

        def dispatch(model, opt_state, batches, step=step):
            return step(model, opt_state, batches, weights, anti_weights, taus, live)

        n_dispatch = n_dispatch_of(k)
        t0 = time.perf_counter()
        for _ in range(WARMUP):
            model, opt_state, outs = dispatch(model, opt_state, draw(streams, pack != "plain"))
            np.asarray(outs)
            tick()
        compile_s = time.perf_counter() - t0

        # device: the dispatch alone, on one stacked draw put on the device once.
        on_device = jax.device_put(draw(streams, pack != "plain"))
        jax.block_until_ready(on_device)
        times = []
        for _ in range(n_dispatch):
            t = time.perf_counter()
            model, opt_state, outs = dispatch(model, opt_state, on_device)
            jax.block_until_ready(outs)
            times.append(time.perf_counter() - t)
            tick()
        device_ms = 1000 * float(np.median(times)) / (SCAN_STEPS * k)

        # loop: draw, stack, dispatch, read back, as train_anchored does per dispatch.
        sample_s = []
        t_loop = time.perf_counter()
        for _ in range(n_dispatch):
            t = time.perf_counter()
            batches = draw(streams, pack != "plain")
            sample_s.append(time.perf_counter() - t)
            model, opt_state, outs = dispatch(model, opt_state, batches)
            np.asarray(outs)
            tick()
        loop_ms = 1000 * (time.perf_counter() - t_loop) / (n_dispatch * SCAN_STEPS * k)
        sample_ms = 1000 * float(np.sum(sample_s)) / (n_dispatch * SCAN_STEPS * k)

        finite = bool(np.isfinite(np.asarray(outs)).all())
        stats = jax.devices()[0].memory_stats() or {}
        results.append(
            {
                "pack": pack,
                "seeds": k,
                "n_dispatch": n_dispatch,
                "compile_s": compile_s,
                "device_ms": device_ms,
                "loop_ms": loop_ms,
                "sample_ms": sample_ms,
                "finite": finite,
                "peak_bytes": stats.get("peak_bytes_in_use"),
            }
        )
        print(
            f"pack {pack}: device {device_ms:.3f}, loop {loop_ms:.3f}, sampling {sample_ms:.3f} ms per seed-step;"
            f" compile {compile_s:.0f} s",
            flush=True,
        )
        del model, opt_state, outs

    return {"variant": variant, "replica": replica, "device": jax.devices()[0].device_kind, "packs": results}


def main(ctx: Ctx) -> dict:
    prep = ctx.run(
        ex2221.prepare_condition,
        fc.OP_NAMES,
        fc.K,
        ex2221.RHO,
        0.0,
        ex2221.CUBE_RATE,
        N_LINES,
        ex2221.CORPUS_SEED,
        ex2216.HOLDOUT_CONTEXTS,
        ex2216.N_TRAJ_PROBE,
        f"{ex2221.MAIN_KEY}-n{N_LINES}",
        role="prep",
    )
    rows = fc.rows_of(prep["meta"], SEEDS, None)
    # One map per variant, each with its own role. A map in flight raises `Pending`, which would hold back the
    # variants after it, so every variant launches before the wake suspends.
    timed, waiting = {}, []
    for variant, (card, _) in VARIANTS.items():
        packs, n = PACKS[card], REPLICAS.get(variant, N_REPLICAS)
        orders = [packs[r % len(packs) :] + packs[: r % len(packs)] for r in range(n)]
        try:
            timed[variant] = ctx.map(
                time_packs,
                [prep["corpus"]] * n,
                [prep["labels"]] * n,
                [rows] * n,
                orders,
                [variant] * n,
                list(range(n)),
                role=variant,
                allow_partial=True,
            )
        except Pending as p:
            waiting.append(p)
    if waiting:
        raise waiting[0]
    return {"timed": timed}


COMPUTE: dict[str, Any] = {
    # One corpus build of 60,000 contexts with its held-out set and probes.
    "prep": dict(cpu=2, timeout=1800),
} | {
    # Five or six packs, each a compile of 10 to 50 seconds and two timings of 4,096 seed-steps. The first round's
    # L4 tasks ran past fifteen minutes, so a slow card gets room.
    variant: dict(gpu=card, env=env, timeout=3600, watchdog=600, watchdog_grace=900)
    for variant, (card, env) in VARIANTS.items()
}

experiment = Experiment(name="pack-timing", main=main, roles=COMPUTE)
