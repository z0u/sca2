"""Per-slice terms: the anti-subspace weight set by slice, and the pull pooled over slices.

The late-pull experiment's `deep` arm pulls blocks 2 to 4 and keeps the anti-subspace term on every slice. Holding
that term at 0.2 kept the embedding table and block 1 off e₁ on the other ops and brought the spill within the
criterion on every seed, but it removed about a tenth less of `difference`. The second D2.2 review traced part of
that loss to the hold pressing on the pull at the blocks it pulls. This experiment tries two arms built on that:

- `split-anti`: the `deep` pull, with the anti weight held at 0.2 on the embedding and block 1 and at the recipe hold
  (0.03) on blocks 2 to 4. Both anneal from the same peak (0.25).
- `pool-slices`: the same anti weights, with the pull pooled by one mellowmax over the slice × position pairs of a
  context, over blocks 1 to 4, so the pull chooses the slice as well as the position. The embedding stays out of
  the pool, where a latch could win it.

Each run is paired by model seed with late-pull's `deep` runs (at the recipe hold and at 0.2) and with ex-2.2.23's
400-epoch controls, so no comparison run is trained again. Every new run is measured as the late-pull experiment
measured its runs (embedding-lean's `measure_one`), and the new runs, their late-pull twins, and the controls are
scored at every dose as ex-2.2.22 scored its runs (`suppress_one`).

    bin/mini run docs/m2/slice-terms/experiment.py --app modal --max-containers 12 --budget 6
    bin/mini status slice-terms
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


pull = _load_sibling("late-pull", "slice_terms_pull")
lean = pull.lean
ex2223 = pull.ex2223
ex2222 = pull.ex2222
ex2221 = pull.ex2221
ex2216 = pull.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = pull.OP_NAMES
ANCHORED_OP = pull.ANCHORED_OP
K = pull.K
EPOCHS = pull.EPOCHS
EVERY = pull.EVERY
LATE = pull.LATE
BLOCKS = tuple(range(1, pull.N_SLICES))
"""Blocks 1 to 4: the slices the pooled pull may choose among."""
SEEDS = pull.SEEDS
"""Model seeds 700 to 703, as the late-pull experiment, so every run has its twins."""
TRAJ_STRIDE_EPOCHS = pull.TRAJ_STRIDE_EPOCHS

# --- The arms ----------------------------------------------------------------------------------------------

ANTI_PEAK = 0.25
"""The anti weight at epoch 0, as the recipe (λ × 2.5 at λ = 0.1), now written as an absolute number."""
EARLY_HOLD = 0.2
"""The hold on the embedding and block 1: late-pull's higher hold, where those slices stayed off e₁."""
LATE_HOLD = 0.03
"""The hold on the pulled blocks: the recipe's (λ × 0.3)."""
SLICE_HOLDS = tuple(EARLY_HOLD if s < 2 else LATE_HOLD for s in EVERY)
"""One hold per slice, embedding first."""


@dataclass(frozen=True)
class Arm:
    name: str
    anchor_slices: tuple[int, ...]
    """The slices the pull acts on."""
    pool_slices: bool
    """Pool the pull over slices as well as positions."""


ARMS: tuple[Arm, ...] = (
    Arm("split-anti", LATE, pool_slices=False),
    Arm("pool-slices", BLOCKS, pool_slices=True),
)

TWINS: tuple[str, ...] = ("late", "late-anti0.2")
"""Late-pull's `deep` arm at the recipe hold and at 0.2 (stored under its training-time names)."""

BUDGET_USD = 6
"""Late-pull cost about \\$0.3 per 400-epoch run with its measurements: eight runs come to about \\$2.5, and the dose
scoring of twenty runs (the new ones, their twins, and the controls) well under \\$1."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/slice-terms"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"


def schedules() -> tuple[dict, dict]:
    """The recipe's `AnchorSpec` and `AntiSpec` keyword dicts at 400 epochs, with the anti weight absolute and set
    by slice. The keyframes (the anneal to the hold by epoch 360, the shared end anneal) are the recipe's.
    """
    base = ex2216.Condition229(
        "x", 1, "x", lam=ex2216.LAM, tau=ex2216.TAU, epochs=EPOCHS, ops=OP_NAMES, n_lines=ex2221.N_LINES
    )
    anchor, anti = ex2216.schedules(base)
    assert anti is not None
    assert abs(anti["lam"] * anti["peak_ratio"] - ANTI_PEAK) < 1e-12
    assert abs(anti["lam"] * anti["hold_ratio"] - LATE_HOLD) < 1e-12
    return anchor, anti | {"peak": ANTI_PEAK, "hold": SLICE_HOLDS}


def rows_of(meta, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run: ex-2.2.21's recipe config at the seed and length, with the arm's pull and anti weights."""
    anchor, anti = schedules()
    if epochs != EPOCHS:
        # A smoke run keeps the shape of the schedule, scaled to its length.
        f = epochs / EPOCHS
        anchor = anchor | {k: anchor[k] * f for k in ("warmup_epochs", "anneal_start", "anneal_end")}
        anti = anti | {k: anti[k] * f for k in ("anneal_end", "anchor_anneal_start", "anchor_anneal_end")}
    rows = []
    for a in ARMS:
        for s in seeds:
            config, _ = ex2221.recipe_config(meta, s, epochs)
            rows.append(
                {
                    "config": config,
                    "anchor": anchor | {"pool_slices": a.pool_slices},
                    "anti": anti,
                    "arm": asdict(a),
                    "model_seed": s,
                    "label": label_of(a.name, s),
                }
            )
    return rows


def resolve_twins(seeds: tuple[int, ...]) -> dict:
    """The checkpoints of the late-pull twins and ex-2.2.23's 400-epoch controls at *seeds*, by ref."""
    from mini.store import get_ref

    refs = {
        pull.label_of(arm, s): pull.CHECKPOINT_REF.format(label=pull.label_of(arm, s)) for arm in TWINS for s in seeds
    }
    refs |= {f"control-s{s}": ex2223.CHECKPOINT_REF.format(label=ex2223.label_of("control", EPOCHS, s)) for s in seeds}
    found = {label: get_ref(ref) for label, ref in refs.items()}
    missing = [label for label, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    return found


def design() -> dict[str, Any]:
    return {
        "experiment": "slice-terms",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arms": [asdict(a) for a in ARMS],
        "anti_peak": ANTI_PEAK,
        "slice_holds": list(SLICE_HOLDS),
        "twins": list(TWINS),
        "seeds": list(SEEDS),
        "slice_sets": {k: list(v) for k, v in lean.SLICE_SETS.items()},
        "dose_gammas": list(ex2222.DOSE_GAMMAS),
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
    }


def publish(rows: list[dict], trained: list[dict], scored: list[dict], dosed: list[dict]) -> dict:
    """The trajectories, every checkpoint, the measurements (one record per run with its arm and seed), and the dose
    scoring of every run, twins and controls included, with its per-context arrays under their own refs.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="slice-terms-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="slice-terms-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    dose = {"design": design(), "runs": [{k: v for k, v in d.items() if k != "arrays"} for d in dosed]}
    set_ref(DOSE_REF, put(json.dumps(dose).encode(), name="slice-terms-dose.json"))
    return {"n_runs": len(scored), "n_dosed": len(dosed)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    """Resolve ex-2.2.21's corpus condition and the twins, train every run, measure it, score every run at every
    dose, and publish. *epochs*, *seeds* and *publishes* are arguments so a smoke run takes the same path at a
    fraction of the size, and leaves the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    twins = ctx.run(resolve_twins, seeds, role="prep")
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
    checkpoints = {t["label"]: t["checkpoint"] for t in trained} | (twins if publishes else {})
    labels = list(checkpoints)
    m = len(labels)
    dosed = ctx.map(
        ex2222.suppress_one,
        [checkpoints[label] for label in labels],
        [resolved["holdout"]] * m,
        [OP_NAMES] * m,
        [K] * m,
        [True] * m,
        labels,
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


COMPUTE = pull.COMPUTE

experiment = Experiment(name="slice-terms", main=main, roles=COMPUTE)
