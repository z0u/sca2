"""Step speed: the scatter-free training step and the prefetched batch draw, timed in `train_anchored` itself, and
whether the step stays run-to-run deterministic with the GPU determinism flags off.

The pack-timing probe found three gains for a single run (`todo/eng/scatter-free-training-step.md`): a step with no
scatter-add (now in `sca.model.ngpt` and `sca.training.loop`), the host's batch draw overlapped with the card's work
(now in `train_anchored`), and the determinism flags off. The flags were there to make the scatter-adds
deterministic (`eng/determinism.md`); with no scatter left, the one other source of run-to-run drift the XLA guide
names is the autotuner's pick, which with the flags off is timed afresh in every process.

Each task trains the recipe of record (`τ_ℓ fixed`, as `fixed-confirm` trains it) through `train_anchored` for
`STOP` epochs of the unchanged 400-epoch schedule. The time between the first trajectory record and the last, over
the steps between them, is the steady cost of a step with the records included and the compile left out.
(`train_anchored` builds its step with `_scanned` from its module, which is what lets `_without_cond` swap it.) The
weights at the stop are hashed; replicas of one variant run in separate containers (each its own role), so equal
digests mean the same weights from separate processes, each with its own compile and autotune.

Corpus preparation and the training tasks run under the dev profile; nothing here is read by a report.

    MINI_PROFILE=dev bin/mini run docs/m2/step-speed/experiment.py --app modal --max-containers 12 --budget 1h
    MINI_PROFILE=dev bin/mini status step-speed
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


fc = _load_sibling("fixed-confirm", "step_speed_fc")
ex2221 = fc.ex2222.ex2221
ex2216 = ex2221.ex2216

SEED = 900
"""A seed no experiment trains (the pack-timing probe's first)."""
N_LINES = 60_000
"""The pack-timing probe's corpus, a fifth of the recipe's: the corpus size sets the epoch length and nothing in the
step. A second round trains on the recipe's own corpus (`full`), where the work done once an epoch is spread over
five times the steps."""
STOP = 40
"""Epochs trained in each task: ten records at the recipe's stride of four epochs."""
NONDET_XLA_FLAGS = "--xla_cpu_enable_concurrency_optimized_scheduler=false"
"""The project's `XLA_FLAGS` with the two GPU determinism flags left out."""
VARIANTS: dict[str, dict[str, Any]] = {
    "L40S-cache-seed": dict(card="L40S", env={"XLA_FLAGS": NONDET_XLA_FLAGS}, n=1, full=True, cache=True),
    "L40S-cache": dict(
        card="L40S", env={"XLA_FLAGS": NONDET_XLA_FLAGS}, n=3, full=True, cache=True, after="L40S-cache-seed"
    ),
    "L4-cache-seed": dict(card="L4", env={"XLA_FLAGS": NONDET_XLA_FLAGS}, n=1, full=True, cache=True),
    "L4-cache": dict(card="L4", env={"XLA_FLAGS": NONDET_XLA_FLAGS}, n=2, full=True, cache=True, after="L4-cache-seed"),
}
"""Variant → card, environment over the project baseline, replicas, and the options: `full` trains on the recipe's
corpus, `gather` puts back the indexed embedding and cross-entropy the step had before, `profile` runs the training
under `cProfile`, `select` swaps the scan's conditional for a select (`_without_cond`), and `cache` keeps JAX's
compilation cache in a directory on the Volume, one per card, so that a task after the first loads the compiled
programs, with the first one's autotuned picks, rather than compiling and timing them again. A variant with `after`
launches once that variant's tasks are done.

The earlier rounds' variants (results on the dev store, under their own roles), each one container a replica:

- round 1, the small corpus, the batch draw between dispatches on the training thread: `L40S-off` and `L4-off`
  (flags off, three replicas each), `L40S-on` (two), `L4-on` (one).
- round 2, the recipe's corpus, flags on: `L40S-full`, `L40S-full-gather`, `L40S-full-profile`, `L4-full`,
  `L4-full-gather`.
- round 3, the recipe's corpus, flags on, the batch draw on a thread of its own: `L40S-r3`, `L40S-r3-select`,
  `L4-r3`, `L4-r3-select`.
- round 4, the recipe's corpus, flags off, a compilation cache shared within each card: the variants above.
"""


def _without_cond() -> None:
    """Swap the scan's conditional on the live flag for a select, in this process: a padded step is computed and its
    result dropped, where the conditional skips it. Every live step computes what it did.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    import sca.compute.training as training

    def scanned(step, n_steps: int, scheduled_tau: bool = False):
        @eqx.filter_jit
        def run(model, opt_state, batches, weights, anti_weights, taus, live):
            carry, static = eqx.partition((model, opt_state), eqx.is_array)

            def body(carry, xs):
                batch, weight, anti_weight, tau, on = xs
                kw = {"taus": tau} if scheduled_tau else {}
                m, s = eqx.combine(carry, static)
                m, s, *outs = step(m, s, *batch[:4], weight, anti_weight, *batch[4:], **kw)
                new = eqx.filter((m, s), eqx.is_array)
                kept = jax.tree.map(lambda a, b: jnp.where(on, a, b), new, carry)
                return kept, jnp.where(on, jnp.stack([jnp.asarray(o, jnp.float32) for o in outs]), 0.0)

            carry, outs = jax.lax.scan(body, carry, (batches, weights, anti_weights, taus, live), length=n_steps)
            return (*eqx.combine(carry, static), outs)

        return run

    training._scanned = scanned  # ty: ignore[invalid-assignment] — a patch for the probe alone


def _with_gathers() -> None:
    """Put back the embedding lookup and the cross-entropy as they were before the scatter-free step, in this process."""
    import jax.numpy as jnp
    import optax

    import sca.anchoring
    from sca.model import ngpt

    def cross_entropy(logits, targets):
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        mask = targets != 0
        return jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1)

    ngpt.NGPT.embed = lambda self, idx: ngpt.normalize(self.transformer.wte[idx])
    sca.anchoring.cross_entropy = cross_entropy  # ty: ignore[invalid-assignment] — a patch for the probe alone


def train_timed(  # noqa: C901 — one setup, one run, and the probe's options
    corpus,
    labels,
    probes,
    row: dict,
    variant: str,
    replica: int,
    gather: bool = False,
    profile: bool = False,
    select: bool = False,
    cache: bool = False,
) -> dict:
    """Train *row* for `STOP` epochs in this container; return the steady ms per step, the digest of the weights at
    the stop, and the card. *variant* and *replica* give each task its own memo key; *gather*, *profile*, *select*
    and *cache* are the `VARIANTS` options.
    """
    import cProfile
    import hashlib
    import io
    import pstats
    import os
    import time

    import equinox as eqx
    import jax

    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from sca.data.incontext import context_length
    from sca.data.named_colors import WordTokenizer
    from mini.runs import gpu_name
    from mini.store import get

    if gather:
        _with_gathers()
    if select:
        _without_cond()
    if cache:
        card = (gpu_name() or "cpu").replace(" ", "-")
        jax.config.update("jax_compilation_cache_dir", str(get_data_dir() / "step-speed" / f"jax-cache-{card}"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    started = time.perf_counter()
    workdir = get_data_dir() / "step-speed" / f"{variant}-r{replica}"
    corpus_dir = get(corpus, workdir / "corpus")
    config = row["config"]
    tokenizer = WordTokenizer(config.tokenizer)
    ops = fc.OP_NAMES
    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=ops.index(fc.ANCHORED_OP),
            label_rate=ex2216.LABEL_RATE,
            variant="whole",
            context_len=z["context_len"],
        )
        loss_mask = z["loss_mask"] if "loss_mask" in z.files else None
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]
    per_op = len(probe_tokens) // len(ops)
    weights = np.concatenate([np.full(per_op, 1.0 if o == fc.ANCHORED_OP else 0.0) for o in ops])
    weights = weights / weights.sum()
    epoch_length = fc.ex2222.ex2217.epoch_length_of(row["total_tokens"], config)
    stride = max(1, round(fc.sps.TRAJ_STRIDE_EPOCHS * epoch_length))

    stamps: list[float] = []
    profiler = cProfile.Profile() if profile else None
    if profiler is not None:
        profiler.enable()
    model, metrics, _ = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**row["anchor"]),
        anti=AntiSpec(**row["anti"]),
        label_p=spec,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        probe_line_w=weights,
        crop=ex2221.CROP_POLICY,
        anchor_slices=row["anchor_slices"],
        anti_slices=fc.sps.EVERY,
        checkpoint_dir=workdir,
        traj_stride=stride,
        newline_id=tokenizer.stoi["\n"],
        min_line_tokens=context_length(fc.K),
        loss_mask=loss_mask,
        on_record=lambda _i, _m: stamps.append(time.perf_counter()),
        stop_epoch=STOP,
    )
    profile_text = None
    if profiler is not None:
        profiler.disable()
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(30)
        profile_text = out.getvalue()
    # The last record is the one at the stop, made after the loop; the ones before it are a stride apart.
    steady = stamps[:-1]
    ms_per_step = 1000 * (steady[-1] - steady[0]) / ((len(steady) - 1) * stride)
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    digest = hashlib.sha256(b"".join(np.asarray(leaf).tobytes() for leaf in leaves)).hexdigest()
    print(f"{variant} r{replica}: {ms_per_step:.3f} ms/step, digest {digest[:12]}", flush=True)
    return {
        "variant": variant,
        "replica": replica,
        "gpu": gpu_name(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "epoch_length": epoch_length,
        "stride": stride,
        "records": len(stamps),
        "to_first_record_s": stamps[0] - started,
        "ms_per_step": ms_per_step,
        "digest": digest,
        "val_loss": metrics[-1].val_loss,
        "profile": profile_text,
    }


def main(ctx: Ctx) -> dict:
    def prep(n_lines: int, key: str) -> dict:
        return ctx.run(
            ex2221.prepare_condition,
            fc.OP_NAMES,
            fc.K,
            ex2221.RHO,
            0.0,
            ex2221.CUBE_RATE,
            n_lines,
            ex2221.CORPUS_SEED,
            ex2216.HOLDOUT_CONTEXTS,
            ex2216.N_TRAJ_PROBE,
            key,
            role="prep",
        )

    preps = {False: prep(N_LINES, f"{ex2221.MAIN_KEY}-n{N_LINES}"), True: prep(ex2221.N_LINES, ex2221.MAIN_KEY)}
    rows = {}
    for full, p in preps.items():
        (row,) = fc.rows_of(p["meta"], (SEED,), None)
        rows[full] = row | {"total_tokens": p["meta"].total_tokens}
    # One task per role, so that every replica gets a container of its own. A task in flight raises `Pending`,
    # which would hold back the ones after it, so every task launches before the wake suspends.
    timed, waiting = {}, []
    for variant, v in VARIANTS.items():
        full = v.get("full", False)
        opts = (v.get("gather", False), v.get("profile", False), v.get("select", False), v.get("cache", False))
        p = preps[full]
        if "after" in v and sum(k.startswith(f"{v['after']}-r") for k in timed) < VARIANTS[v["after"]]["n"]:
            continue  # the variant it follows is still running
        for r in range(v["n"]):
            try:
                timed[f"{variant}-r{r}"] = ctx.run(
                    train_timed, p["corpus"], p["labels"], p["probes"], rows[full], variant, r, *opts,
                    role=f"{variant}-r{r}",
                )  # fmt: skip
            except Pending as e:
                waiting.append(e)
    if waiting:
        raise waiting[0]
    return {"timed": timed}


COMPUTE: dict[str, Any] = {
    # Two corpus builds with their held-out sets and probes: 60,000 contexts (the pack-timing probe's) and the recipe's.
    "prep": dict(cpu=2, timeout=1800),
} | {
    # Forty epochs: about 2,000 steps on the small corpus and 10,000 on the recipe's, a few minutes at most.
    f"{variant}-r{r}": dict(gpu=v["card"], env=v["env"], timeout=3600, watchdog=600, watchdog_grace=900)
    for variant, v in VARIANTS.items()
    for r in range(v["n"])
}

experiment = Experiment(name="step-speed", main=main, roles=COMPUTE)
