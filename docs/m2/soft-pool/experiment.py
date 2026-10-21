"""A pooling temperature that starts soft, and a pull on slices 1 to 3 only, beside the `falloff` recipe of record.

The recipe of record (split-anti-seeds' `falloff`) pulls `difference` toward e₁ with one weight on slices 2 to 4, pooled
over positions at a fixed τ_p = 0.1. In anchor-ramp, the runs that lost the anchor split off from the kept ones during
the pull's warm-up (epochs 16 to 24 of 40), and losses ordered by how little pull slices 1 to 3 had. Two changes, crossed:

- pull shape: the step on slices 2 to 4, or a ramp over slices 1 to 3 (⅓, ⅔, 1, with nothing on the embedding or slice
  4), whose weights add up to the step's over slices 1 to 3;
- τ_p: fixed at 0.1, or soft-to-sharp: 1 at epoch 0, sharpening to 0.1 by the end of the warm-up (epoch 40), so the
  pool is soft while the pull settles.

The step at a fixed τ_p is `falloff` itself, so three arms are new. A fourth pools the pull over the slice × position
pairs of slices 1 to 3 (slice-terms' nested pool), with both temperatures soft-to-sharp: τ_p from 1 to 0.1 and τ_ℓ from
4 to 0.4, over the same window.

All keep `falloff`'s anti weight on every slice, and everything else is split-anti-seeds' training, measurement and
dose scoring, at its seeds (704 to 711). Its `falloff` runs and ex-2.2.23's controls are the twins, so nothing they
trained is trained again, and every comparison is paired by initialization, batches and label draws.

    bin/mini run docs/m2/soft-pool/experiment.py --app modal --max-containers 16 --budget 4h
    bin/mini status soft-pool
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


sas = _load_sibling("split-anti-seeds", "soft_pool_sas")
st = sas.st
lean = sas.lean
ex2223 = sas.ex2223
ex2222 = sas.ex2222
ex2221 = sas.ex2221

OP_NAMES = sas.OP_NAMES
ANCHORED_OP = sas.ANCHORED_OP
K = sas.K
EPOCHS = sas.EPOCHS
EVERY = sas.EVERY
LATE = sas.LATE
TRAJ_STRIDE_EPOCHS = sas.TRAJ_STRIDE_EPOCHS
SEEDS = sas.SEEDS
"""Model seeds 704 to 711, as split-anti-seeds, so every run has its `falloff` and control twins."""
FALLOFF_HOLDS = sas.FALLOFF_HOLDS
BASELINE = "falloff"
"""Split-anti-seeds' arm the new ones are compared with: the recipe of record."""

# --- The arms ----------------------------------------------------------------------------------------------

TAU_START = 1.0
"""τ_p at epoch 0 under the soft-to-sharp schedule: soft enough that the pool spreads over the span roles."""
SLICE_TAU = 0.4
"""τ_ℓ at the end of the schedule, for the pool over slices. Slice-terms' E4 trained τ_ℓ from 0.2 to 1.6 at one seed
(701) and saw no trend in removal or spill, and its joint pool (τ_ℓ = τ_p = 0.1) lost the anchor on one seed of four,
so 0.4 sits inside the range that worked, a little sharper than its middle."""
SLICE_TAU_START = 4.0
"""τ_ℓ at epoch 0: the same tenfold sharpening as τ_p, starting near a plain mean over slices."""


@dataclass(frozen=True)
class Arm:
    name: str
    pull: tuple[float, ...]
    """The pull's weight on each slice, embedding first, relative to the recipe's (1 on each of slices 2 to 4)."""
    soft: bool
    """Whether τ_p (and τ_ℓ, when pooled over slices) starts soft and sharpens over the warm-up."""
    shown_as: str
    pool_slices: bool = False

    def anchor_slices(self) -> tuple[int, ...]:
        return tuple(s for s in EVERY if self.pull[s] > 0)

    def slice_w(self) -> tuple[float, ...] | None:
        """`AnchorSpec.slice_w` for `pull`: the term means over the pulled slices, so a slice of relative weight w
        among n pulled slices takes w × n / 3 to match the recipe's mean over three. `None` for a flat pull, whose
        spec then matches the recipe's.
        """
        n = len(self.anchor_slices())
        w = tuple(self.pull[s] * n / len(LATE) for s in self.anchor_slices())
        return None if self.pool_slices or all(abs(x - 1) < 1e-12 for x in w) else w

    def anchor(self, base: dict) -> dict:
        """The arm's `AnchorSpec` keywords on the recipe's *base*; only the fields it sets enter the spec."""
        spec = dict(base)
        if (w := self.slice_w()) is not None:
            spec["slice_w"] = w
        if self.pool_slices:
            spec |= {"pool_slices": True, "slice_tau": SLICE_TAU}
        if self.soft:
            spec |= {"tau_start": TAU_START, "tau_anneal_end": base["warmup_epochs"]}
            if self.pool_slices:
                spec["slice_tau_start"] = SLICE_TAU_START
        return spec


STEP = tuple(1.0 if s in LATE else 0.0 for s in EVERY)
"""The recipe's pull by slice: the `falloff` arm."""
RAMP_123 = (0.0, 1 / 3, 2 / 3, 1.0, 0.0)
"""A ramp over slices 1 to 3 whose weights add up to the step's over the same slices (two)."""
POOL_123 = (0.0, 1.0, 1.0, 1.0, 0.0)
"""Slices 1 to 3, pooled over together: the weights only say which slices are in the pool."""

ARMS: tuple[Arm, ...] = (
    Arm("step-soft", STEP, True, "step, soft τ"),
    Arm("ramp-123", RAMP_123, False, "ramp 1–3"),
    Arm("ramp-123-soft", RAMP_123, True, "ramp 1–3, soft τ"),
    Arm("pool-123-soft", POOL_123, True, "pool 1–3, soft τ", pool_slices=True),
)

BUDGET_USD = 14
"""Split-anti-seeds cost about \\$0.35 per 400-epoch run with its measurements: 32 runs come to about \\$11, and their
dose scoring about \\$1."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/soft-pool"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"


def rows_of(meta, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run: the slice-terms schedules with `falloff`'s anti weight, and the arm's pull and temperatures."""
    anchor, anti = st.schedules()
    if epochs != EPOCHS:
        # A smoke run keeps the shape of the schedule, scaled to its length.
        f = epochs / EPOCHS
        anchor = anchor | {k: anchor[k] * f for k in ("warmup_epochs", "anneal_start", "anneal_end")}
        # (`Arm.anchor` takes the temperature window from the scaled warm-up.)
        anti = anti | {k: anti[k] * f for k in ("anneal_end", "anchor_anneal_start", "anchor_anneal_end")}
    rows = []
    for a in ARMS:
        for s in seeds:
            config, _ = ex2221.recipe_config(meta, s, epochs)
            rows.append(
                {
                    "config": config,
                    "anchor": a.anchor(anchor),
                    "anti": anti | {"hold": FALLOFF_HOLDS},
                    "anchor_slices": a.anchor_slices(),
                    "arm": asdict(a),
                    "model_seed": s,
                    "label": label_of(a.name, s),
                }
            )
    return rows


def design() -> dict[str, Any]:
    return {
        "experiment": "soft-pool",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arms": [
            asdict(a) | {"anchor_slices": list(a.anchor_slices()), "anchor": a.anchor(st.schedules()[0])} for a in ARMS
        ],
        "baseline": {"name": BASELINE, "pull": list(STEP), "anchor_slices": list(LATE)},
        "anti_holds": list(FALLOFF_HOLDS),
        "tau_start": TAU_START,
        "slice_tau": SLICE_TAU,
        "slice_tau_start": SLICE_TAU_START,
        "seeds": list(SEEDS),
        "slice_sets": {k: list(v) for k, v in lean.SLICE_SETS.items()},
        "dose_gammas": list(ex2222.DOSE_GAMMAS),
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
    }


def publish(rows: list[dict], trained: list[dict], scored: list[dict], dosed: list[dict]) -> dict:
    """The trajectories, every checkpoint, the measurements (one record per run with its arm and seed), and the dose
    scoring of every run, with its per-context arrays under their own refs.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="soft-pool-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="soft-pool-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    dose = {"design": design(), "runs": [{k: v for k, v in d.items() if k != "arrays"} for d in dosed]}
    set_ref(DOSE_REF, put(json.dumps(dose).encode(), name="soft-pool-dose.json"))
    return {"n_runs": len(scored), "n_dosed": len(dosed)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    """Resolve ex-2.2.21's corpus condition, train every run, measure it, score it at every dose, and publish.
    *epochs*, *seeds* and *publishes* are arguments so a smoke run takes the same path at a fraction of the size, and
    leaves the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    rows = rows_of(resolved["meta"], epochs, seeds)
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    trained = ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        ["whole"] * n,
        [r["anchor_slices"] for r in rows],
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
        [EVERY] * n,
        [None] * n,
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
    dosed = ctx.map(
        ex2222.suppress_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [OP_NAMES] * n,
        [K] * n,
        [True] * n,
        [r["label"] for r in rows],
        role="measure",
    )
    if not publishes:
        return {"labels": [s["label"] for s in scored], "dosed": [d["label"] for d in dosed]}
    return ctx.run(publish, rows, trained, scored, dosed, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of every arm for eight epochs (past the five-epoch warm-up), through the whole DAG."""
    return run(ctx, epochs=8, seeds=SEEDS[:1], publishes=False)


COMPUTE = sas.COMPUTE

experiment = Experiment(name="soft-pool", main=main, roles=COMPUTE)
