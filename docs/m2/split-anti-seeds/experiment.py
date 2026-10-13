"""The split anti weight at fresh seeds: a preregistered confirmation of the slice-terms arms.

Four arms, each pulling `difference` at slices 2 to 4 with the anti-subspace term on every slice (late-pull's `deep`
pull), differ only in the hold of the anti weight by slice:

- `hold0.2` and `hold0.1`: one hold on every slice (shown as `deep, hold 0.2` and `deep, hold 0.1`).
- `split-anti`: 0.2 on the embedding and slice 1, 0.03 on slices 2 to 4, as in the slice-terms experiment.
- `falloff`: a straight line from 0.2 at the embedding to 0.03 at slice 4.

Every arm trains at model seeds 704 to 711, none of which any of them has seen. Ex-2.2.23's 400-epoch controls at
those seeds are the twins, so every comparison is paired by initialization, batches and label draws. Training,
measurement and dose scoring are the slice-terms experiment's, unchanged.

    bin/mini run docs/m2/split-anti-seeds/experiment.py --app modal --max-containers 16 --budget 6h
    bin/mini status split-anti-seeds
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


st = _load_sibling("slice-terms", "split_anti_seeds_st")
pull = st.pull
lean = st.lean
ex2223 = st.ex2223
ex2222 = st.ex2222
ex2221 = st.ex2221

OP_NAMES = st.OP_NAMES
ANCHORED_OP = st.ANCHORED_OP
K = st.K
EPOCHS = st.EPOCHS
EVERY = st.EVERY
LATE = st.LATE
TRAJ_STRIDE_EPOCHS = st.TRAJ_STRIDE_EPOCHS

SEEDS: tuple[int, ...] = tuple(s for s in ex2223.SEEDS if s >= 704)
"""Model seeds 704 to 711: ex-2.2.23 has 400-epoch controls there, and no anchored arm has trained there before."""
assert SEEDS == tuple(range(704, 712))

# --- The arms ----------------------------------------------------------------------------------------------

ANTI_PEAK = st.ANTI_PEAK
EARLY_HOLD = st.EARLY_HOLD
LATE_HOLD = st.LATE_HOLD
FALLOFF_HOLDS = tuple(EARLY_HOLD + (LATE_HOLD - EARLY_HOLD) * s / (len(EVERY) - 1) for s in EVERY)
"""A straight line from the split's embedding hold to its slice-4 hold, one hold per slice, embedding first."""


@dataclass(frozen=True)
class Arm:
    name: str
    hold: float | tuple[float, ...]
    """The anti weight's hold: one number for every slice, or one per slice, embedding first."""
    shown_as: str

    def holds(self) -> tuple[float, ...]:
        return self.hold if isinstance(self.hold, tuple) else (self.hold,) * len(EVERY)


ARMS: tuple[Arm, ...] = (
    Arm("hold0.2", 0.2, "deep, hold 0.2"),
    Arm("hold0.1", 0.1, "deep, hold 0.1"),
    Arm("split-anti", st.SLICE_HOLDS, "split-anti"),
    Arm("falloff", FALLOFF_HOLDS, "falloff"),
)

BUDGET_USD = 16
"""Slice-terms cost about \\$0.3 per 400-epoch run with its measurements: 32 runs come to about \\$10, and the dose
scoring of 40 checkpoints (the new runs and the controls) about \\$1."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/split-anti-seeds"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"


def rows_of(meta, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run: the slice-terms schedules with the arm's hold, the `deep` pull, at the seed and length."""
    anchor, anti = st.schedules()
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
                    "anchor": anchor | {"pool_slices": False},
                    "anti": anti | {"hold": a.hold},
                    "arm": asdict(a),
                    "model_seed": s,
                    "label": label_of(a.name, s),
                }
            )
    return rows


def resolve_controls(seeds: tuple[int, ...]) -> dict:
    """The checkpoints of ex-2.2.23's 400-epoch controls at *seeds*, by ref."""
    from mini.store import get_ref

    refs = {f"control-s{s}": ex2223.CHECKPOINT_REF.format(label=ex2223.label_of("control", EPOCHS, s)) for s in seeds}
    found = {label: get_ref(ref) for label, ref in refs.items()}
    missing = [label for label, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    return found


def design() -> dict[str, Any]:
    return {
        "experiment": "split-anti-seeds",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arms": [asdict(a) | {"holds": list(a.holds())} for a in ARMS],
        "anti_peak": ANTI_PEAK,
        "seeds": list(SEEDS),
        "slice_sets": {k: list(v) for k, v in lean.SLICE_SETS.items()},
        "dose_gammas": list(ex2222.DOSE_GAMMAS),
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
    }


def publish(rows: list[dict], trained: list[dict], scored: list[dict], dosed: list[dict]) -> dict:
    """The trajectories, every checkpoint, the measurements (one record per run with its arm and seed), and the dose
    scoring of every run and control, with its per-context arrays under their own refs.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="split-anti-seeds-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="split-anti-seeds-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    dose = {"design": design(), "runs": [{k: v for k, v in d.items() if k != "arrays"} for d in dosed]}
    set_ref(DOSE_REF, put(json.dumps(dose).encode(), name="split-anti-seeds-dose.json"))
    return {"n_runs": len(scored), "n_dosed": len(dosed)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    """Resolve ex-2.2.21's corpus condition and the controls, train every run, measure it, score every run and
    control at every dose, and publish. *epochs*, *seeds* and *publishes* are arguments so a smoke run takes the
    same path at a fraction of the size, and leaves the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    controls = ctx.run(resolve_controls, seeds, role="prep")
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
        [LATE] * n,
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
    checkpoints = {t["label"]: t["checkpoint"] for t in trained} | (controls if publishes else {})
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


COMPUTE = st.COMPUTE

experiment = Experiment(name="split-anti-seeds", main=main, roles=COMPUTE)
