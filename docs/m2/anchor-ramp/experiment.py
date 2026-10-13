"""An anchor weight that ramps up with depth, beside the `falloff` recipe of record.

The recipe of record (split-anti-seeds' `falloff`) pulls `difference` toward e₁ with one weight on slices 2 to 4 and
none on the embedding or slice 1: a step in depth. Its anti weight falls off in a straight line from the embedding to
slice 4. Two arms replace the step with a straight-line ramp of the pull, rising to the step's weight at slice 4:

- `ramp-emb`: zero at the embedding, so slice 1 is pulled at a quarter of the slice-4 weight.
- `ramp-1`: zero at slice 1, so the pull starts at slice 2 (a third of the slice-4 weight) as the step does.

Both keep `falloff`'s anti weight on every slice, and everything else is split-anti-seeds' training, measurement and
dose scoring, at its seeds (704 to 711). Its `falloff` runs and ex-2.2.23's controls are the twins, so nothing they
trained is trained again, and every comparison is paired by initialization, batches and label draws.

    bin/mini run docs/m2/anchor-ramp/experiment.py --app modal --max-containers 16 --budget 4h
    bin/mini status anchor-ramp
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


sas = _load_sibling("split-anti-seeds", "anchor_ramp_sas")
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
"""Split-anti-seeds' arm the ramps are compared with: the recipe of record."""

# --- The arms ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    pull: tuple[float, ...]
    """The pull's weight on each slice, embedding first, relative to the recipe's (1 on each of slices 2 to 4)."""
    shown_as: str

    def anchor_slices(self) -> tuple[int, ...]:
        return tuple(s for s in EVERY if self.pull[s] > 0)

    def slice_w(self) -> tuple[float, ...]:
        """`AnchorSpec.slice_w` for `pull`: the term means over the pulled slices, so a slice of relative weight w
        among n pulled slices takes w × n / 3 to match the recipe's mean over three.
        """
        n = len(self.anchor_slices())
        return tuple(self.pull[s] * n / len(LATE) for s in self.anchor_slices())


STEP = tuple(1.0 if s in LATE else 0.0 for s in EVERY)
"""The recipe's pull by slice, for the figures: the `falloff` arm."""
ARMS: tuple[Arm, ...] = (
    Arm("ramp-emb", tuple(s / 4 for s in EVERY), "ramp from emb"),
    Arm("ramp-1", tuple(max(0, s - 1) / 3 for s in EVERY), "ramp from 1"),
)
assert all(a.pull[-1] == 1.0 for a in ARMS)

BUDGET_USD = 8
"""Split-anti-seeds cost about \\$0.35 per 400-epoch run with its measurements: 16 runs come to about \\$6, and their
dose scoring under \\$1."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/anchor-ramp"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"


def rows_of(meta, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run: the slice-terms schedules with `falloff`'s anti weight and the arm's pull by slice."""
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
                    "anchor": anchor | {"pool_slices": False, "slice_w": a.slice_w()},
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
        "experiment": "anchor-ramp",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arms": [asdict(a) | {"anchor_slices": list(a.anchor_slices()), "slice_w": list(a.slice_w())} for a in ARMS],
        "baseline": {"name": BASELINE, "pull": list(STEP), "anchor_slices": list(LATE)},
        "anti_holds": list(FALLOFF_HOLDS),
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
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="anchor-ramp-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="anchor-ramp-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    dose = {"design": design(), "runs": [{k: v for k, v in d.items() if k != "arrays"} for d in dosed]}
    set_ref(DOSE_REF, put(json.dumps(dose).encode(), name="anchor-ramp-dose.json"))
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

experiment = Experiment(name="anchor-ramp", main=main, roles=COMPUTE)
