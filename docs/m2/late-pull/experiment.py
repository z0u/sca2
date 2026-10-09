"""The pull on the later blocks alone, with the anti-subspace term kept on every slice.

The embedding-lean report found that pulling the first two slices (the embedding and block 1) toward e₁ loads
lightness onto e₁ in the color embedding table, and that this lean carries the spill of the edit. Leaving those
slices out of the pull has been tried before only with the anti-subspace term left out of them too (ex-2.2.21's
`no-emb`, the τ × λ_a sweep). This experiment pulls blocks 2 to 4 only and keeps the anti term on every slice, so
the first two slices are asked to stay off e₁ rather than left alone. A second arm adds a hard constraint: after
every step the e₁ component of every embedding is zeroed (`clean_embedding_rows` on the whole table), so the table
cannot lean at all.

Each run is paired by model seed with ex-2.2.23's every-slice and control runs at 400 epochs, so the two new arms
see the same initialization, batches and label draws as those runs. A fresh every-slice arm at the same seeds
records the alignment at every slice and position along training, which ex-2.2.23 did not keep, and checks that
its runs reproduce. Every run is then measured as embedding-lean measured ex-2.2.23's runs, with the edit
restricted to each slice set.

    bin/mini run docs/m2/late-pull/experiment.py --app modal --max-containers 12 --budget 8
    bin/mini status late-pull
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

from mini import Ctx, Experiment


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


lean = _load_sibling("embedding-lean", "late_pull_lean")
ex2223 = lean.ex2223
ex2222 = ex2223.ex2222
ex2221 = ex2223.ex2221
ex2216 = ex2221.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = ex2223.OP_NAMES
ANCHORED_OP = ex2223.ANCHORED_OP
K = ex2223.K
EPOCHS = ex2223.LONG
"""400 epochs, the recipe length since ex-2.2.23."""
N_SLICES = lean.N_SLICES
EVERY = tuple(range(N_SLICES))
LATE = tuple(range(2, N_SLICES))
"""Blocks 2 to 4: the slices where editing ex-2.2.23's runs removed most of the anchored op with almost no spill
(embedding-lean, E4)."""
assert LATE == lean.SLICE_SETS["blocks 2 to 4"]

# --- The arms ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    anchor_slices: tuple[int, ...] | None
    """The slices the pull acts on; `None` is every slice."""
    anti_slices: tuple[int, ...] | None
    """The slices the anti-subspace term acts on; `None` follows `anchor_slices`."""
    clean: bool
    """Hold every embedding off e₁ after each step."""


ARMS: tuple[Arm, ...] = (
    Arm("late", LATE, EVERY, clean=False),
    Arm("late-clean", LATE, EVERY, clean=True),
    Arm("whole", None, None, clean=False),
)
"""`whole` is the recipe of record (ex-2.2.23's `anchor`), trained again for its trajectory."""

SEEDS: tuple[int, ...] = ex2223.SEEDS[:4]
"""Model seeds 700 to 703, the first four of ex-2.2.23, so every run has a twin there in each condition."""

TRAJ_STRIDE_EPOCHS = ex2223.TRAJ_STRIDE_EPOCHS

BUDGET_USD = 8
"""Ex-2.2.23 cost about \\$0.25 to \\$0.3 per 400-epoch run with its scoring; twelve runs and their measurements
come to about \\$4."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/late-pull"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESULTS_REF = PREFIX + "/results"


def rows_of(meta, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run: ex-2.2.21's recipe config and schedules at the seed and length, with the arm's slices."""
    rows = []
    for a in ARMS:
        base = ex2216.Condition229(
            a.name, 1, a.name, lam=ex2216.LAM, tau=ex2216.TAU, epochs=epochs, ops=OP_NAMES, n_lines=ex2221.N_LINES
        )
        anchor, anti = ex2216.schedules(base)
        for s in seeds:
            config, _ = ex2221.recipe_config(meta, s, epochs)
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "arm": asdict(a),
                    "model_seed": s,
                    "label": label_of(a.name, s),
                }
            )
    return rows


def design() -> dict[str, Any]:
    return {
        "experiment": "late-pull",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arms": [asdict(a) for a in ARMS],
        "seeds": list(SEEDS),
        "slice_sets": {k: list(v) for k, v in lean.SLICE_SETS.items()},
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
    }


def publish(rows: list[dict], trained: list[dict], scored: list[dict]) -> dict:
    """The trajectories, every checkpoint, and the measurements (one record per run with its arm and seed)."""
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="late-pull-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="late-pull-results.json"))
    return {"n_runs": len(scored)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    """Resolve ex-2.2.21's corpus condition, train every run, measure it, and publish. *epochs*, *seeds* and
    *publishes* are arguments so a smoke run takes the same path at a fraction of the size, and leaves the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    rows = rows_of(resolved["meta"], epochs, seeds)
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    vocab = rows[0]["config"].model.vocab_size
    trained = ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        ["whole"] * n,
        [r["arm"]["anchor_slices"] for r in rows],
        [""] * n,
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [ex2221.N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        [r["arm"]["anti_slices"] for r in rows],
        [tuple(range(vocab)) if r["arm"]["clean"] else None for r in rows],
        [True] * n,
        role="train",
    )
    scored = ctx.map(
        lean.measure_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [r["label"] for r in rows],
        [True] * n,
        role="measure",
    )
    if not publishes:
        return {"labels": [s["label"] for s in scored], "traj_keys": sorted(trained[0]["traj"])}
    return ctx.run(publish, rows, trained, scored, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of every arm for eight epochs (past the five-epoch warm-up), through the whole DAG."""
    return run(ctx, epochs=8, seeds=SEEDS[:1], publishes=False)


COMPUTE = {
    "prep": dict(cpu=2, timeout=1800),
    # About 105,600 steps at 400 epochs, as ex-2.2.23's long runs; sized for a slow container
    # (eng: training-step-is-host-bound).
    "train": dict(gpu="L4", timeout=4 * 3600, watchdog=900, watchdog_grace=900),
    # One alignment pass and up to eight scoring passes over 14,000 held-out contexts (embedding-lean).
    "measure": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="late-pull", main=main, roles=COMPUTE)
