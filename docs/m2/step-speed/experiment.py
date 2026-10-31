"""Step speed: the scatter-free training step and the prefetched batch draw, timed in `train_anchored` itself, and
whether the step stays run-to-run deterministic with the GPU determinism flags off.

The pack-timing probe found three gains for a single run (`todo/eng/scatter-free-training-step.md`): a step with no
scatter-add (now in `sca.model.ngpt` and `sca.training.loop`), the host's batch draw overlapped with the card's work
(now in `train_anchored`), and the determinism flags off. The flags were there to make the scatter-adds
deterministic (`eng/determinism.md`); with no scatter left, the one other source of run-to-run drift the XLA guide
names is the autotuner's pick, which with the flags off is timed afresh in every process.

Each task trains the recipe of record (`τ_ℓ fixed`, as `fixed-confirm` trains it) through `train_anchored` for
`STOP` epochs of the unchanged 400-epoch schedule. The time between the first trajectory record and the last, over
the steps between them, is the steady cost of a step with the records included and the compile left out. The
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
step."""
STOP = 40
"""Epochs trained in each task: ten records at the recipe's stride of four epochs, about 8,500 steps."""
NONDET_XLA_FLAGS = "--xla_cpu_enable_concurrency_optimized_scheduler=false"
"""The project's `XLA_FLAGS` with the two GPU determinism flags left out."""
VARIANTS: dict[str, tuple[str, dict[str, str], int]] = {
    "L40S-off": ("L40S", {"XLA_FLAGS": NONDET_XLA_FLAGS}, 3),
    "L4-off": ("L4", {"XLA_FLAGS": NONDET_XLA_FLAGS}, 3),
    "L40S-on": ("L40S", {}, 2),
    "L4-on": ("L4", {}, 1),
}
"""Variant → (card, environment over the project baseline, replicas)."""


def train_timed(corpus, labels, probes, row: dict, variant: str, replica: int) -> dict:
    """Train *row* for `STOP` epochs in this container; return the steady ms per step, the digest of the weights at
    the stop, and the card. *variant* and *replica* give each task its own memo key.
    """
    import hashlib
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
        "ms_per_step": ms_per_step,
        "digest": digest,
        "val_loss": metrics[-1].val_loss,
    }


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
    (row,) = fc.rows_of(prep["meta"], (SEED,), None)
    row = row | {"total_tokens": prep["meta"].total_tokens}
    # One task per role, so that every replica gets a container of its own. A task in flight raises `Pending`,
    # which would hold back the ones after it, so every task launches before the wake suspends.
    timed, waiting = {}, []
    for variant, (_, _, n) in VARIANTS.items():
        for r in range(n):
            try:
                timed[f"{variant}-r{r}"] = ctx.run(
                    train_timed, prep["corpus"], prep["labels"], prep["probes"], row, variant, r,
                    role=f"{variant}-r{r}",
                )  # fmt: skip
            except Pending as p:
                waiting.append(p)
    if waiting:
        raise waiting[0]
    return {"timed": timed}


COMPUTE: dict[str, Any] = {
    # One corpus build of 60,000 contexts with its held-out set and probes (the pack-timing probe's).
    "prep": dict(cpu=2, timeout=1800),
} | {
    # Forty epochs, about 8,500 steps: under a minute of steps on an L40S, a few minutes on an L4.
    f"{variant}-r{r}": dict(gpu=card, env=env, timeout=3600, watchdog=600, watchdog_grace=900)
    for variant, (card, env, n) in VARIANTS.items()
    for r in range(n)
}

experiment = Experiment(name="step-speed", main=main, roles=COMPUTE)
