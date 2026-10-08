"""The mellowmax temperature and the anchor weight, swept together on the pull that leaves out the embedding slice.

Spill-by-position found that on most anchored runs of the recipe one syntax embedding comes to lie on e₁ (a latch
of the pooled pull), and that the latch carries much of the spill of the edit. The `no-emb` pull of ex-2.2.21, which
leaves the embedding slice out of the anchor and anti-subspace terms, never latched and scored best on the task, but
only at three seeds and 200 epochs. This sweep trains `no-emb` at the recipe length over a log-log plane of the
mellowmax temperature τ and the anchor weight λ_a, one seed per trial, so that smooth fits over the plane can stand in
for repeated seeds. A few every-slice trials along τ at the default weight are the reference.

The DAG resolves ex-2.2.21's corpus, held-out set, and probes by ref, trains each trial with ex-2.2.21's code, and
scores it with ex-2.2.21's eval and ex-2.2.22's suppression pass. The controls are ex-2.2.23's, read by the report.

    bin/mini run docs/m2/tau-lambda-sweep/experiment.py --app modal --max-containers 20 --budget 20
    bin/mini status tau-lambda-sweep
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

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


ex2223 = _load_sibling("ex-2.2.23", "tau_lambda_sweep_ex2223")
ex2222 = ex2223.ex2222
ex2221 = ex2223.ex2221
ex2216 = ex2221.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = ex2223.OP_NAMES
ANCHORED_OP = ex2223.ANCHORED_OP
HSV_OPS = ex2223.HSV_OPS
K = ex2223.K
EPOCHS = ex2223.LONG
"""The recipe after ex-2.2.23: every run trains for 400 epochs, where every run of that scout made the second rise."""
LAMBDA_DEFAULT = ex2216.LAM
TAU_DEFAULT = ex2216.TAU
"""The recipe point, λ_a = 0.1 and τ = 0.1 (ex-2.2.3). The anti-subspace weight is a multiple of λ_a in the recipe
(2.5× falling to 0.3×), so a trial at another λ_a moves both terms together."""
assert LAMBDA_DEFAULT == 0.1 and TAU_DEFAULT == 0.1

# --- The space ---------------------------------------------------------------------------------------------

TAU_RANGE: tuple[float, float] = (0.01, 1.0)
"""Mellowmax pools the alignment over the positions of a context. At τ = 0.01 the pool is close to the max over
positions, and at τ = 1 close to their mean (the alignment is a cosine, so its spread is about 1). Log scale."""
LAMBDA_RANGE: tuple[float, float] = (0.025, 0.4)
"""A quarter of the recipe weight to four times it. Log scale."""

SOBOL_M = 5
SOBOL_SEED = 2026_10_08
"""32 trials (2⁵, the size at which a Sobol sequence covers the square evenly), scrambled with this seed."""


@dataclass(frozen=True)
class Trial:
    name: str
    arm: str
    """`no-emb` (the sweep) or `whole` (the every-slice reference)."""
    tau: float
    lam: float
    model_seed: int


SEED_OFFSET = 800
"""Fresh model seeds 800 onward, one per trial; no earlier experiment has used them."""


def sweep_points() -> np.ndarray:
    """The (τ, λ_a) of every `no-emb` trial, in Sobol order."""
    from scipy.stats import qmc

    u = qmc.Sobol(d=2, scramble=True, seed=SOBOL_SEED).random_base2(SOBOL_M)
    lo = np.log10([TAU_RANGE[0], LAMBDA_RANGE[0]])
    hi = np.log10([TAU_RANGE[1], LAMBDA_RANGE[1]])
    return 10 ** (lo + u * (hi - lo))


N_REFERENCE = 8
REFERENCE_TAUS: tuple[float, ...] = tuple(float(t) for t in np.geomspace(*TAU_RANGE, N_REFERENCE))
"""The every-slice reference: eight trials at the recipe weight, τ evenly spaced on a log scale over the same range.
Ex-2.2.23's twelve every-slice runs at 400 epochs sit at the recipe point beside them."""


def trials() -> tuple[Trial, ...]:
    pts = sweep_points()
    sweep = tuple(
        Trial(f"no-emb-{i:02d}", "no-emb", float(t), float(lam), SEED_OFFSET + i) for i, (t, lam) in enumerate(pts)
    )
    n = len(sweep)
    reference = tuple(
        Trial(f"whole-{i:02d}", "whole", t, LAMBDA_DEFAULT, SEED_OFFSET + n + i) for i, t in enumerate(REFERENCE_TAUS)
    )
    return sweep + reference


TRIALS = trials()
assert len(TRIALS) == 2**SOBOL_M + N_REFERENCE
assert len({t.model_seed for t in TRIALS}) == len(TRIALS), "one seed per trial"

TRAJ_STRIDE_EPOCHS = ex2223.TRAJ_STRIDE_EPOCHS

BUDGET_USD = 20
"""Ex-2.2.23 spent about $9 on 24 runs at 400 epochs and 14 at 200, with their scoring passes: about $0.25 to $0.3
per 400-epoch run. Forty runs come to about $12."""

# =============================================================================================
# The DAG
# =============================================================================================

MAIN_KEY = ex2221.MAIN_KEY
N_TRAJ_EEM_PER_OP = ex2221.N_TRAJ_EEM_PER_OP

PREFIX = "reports/m2/tau-lambda-sweep"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
EVAL_REF = PREFIX + "/eval"
EVAL_ARRAYS_REF = PREFIX + "/eval-arrays/{label}"
SUPPRESSION_REF = PREFIX + "/suppression"
SUPPRESSION_ARRAYS_REF = PREFIX + "/suppression-arrays/{label}"


def rows_of(meta, chosen: tuple[Trial, ...]) -> list[dict]:
    """One row per trial: ex-2.2.21's recipe config at the trial seed and 400 epochs, with the trial τ and λ_a in
    both schedules and the arm's label variant and slices.
    """
    n_layer = ex2216.model_dims(ex2223.MODEL)[1]
    rows = []
    for t in chosen:
        config, _ = ex2221.recipe_config(meta, t.model_seed, EPOCHS)
        base = ex2216.Condition229(
            t.name, 1, t.name, lam=t.lam, tau=t.tau, epochs=EPOCHS, ops=OP_NAMES, n_lines=ex2221.N_LINES
        )
        anchor, anti = ex2216.schedules(base)
        assert anchor["tau"] == t.tau and anchor["peak"] == t.lam and anti is not None and anti["lam"] == t.lam
        variant, anchor_slices, pull = ex2221.label_variant(t.arm, n_layer)
        rows.append(
            {
                "config": config,
                "anchor": anchor,
                "anti": anti,
                "variant": variant,
                "anchor_slices": anchor_slices,
                "pull": pull,
                "label": t.name,
                "trial": asdict(t),
            }
        )
    return rows


def design() -> dict[str, Any]:
    return {
        "experiment": "tau-lambda-sweep",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "hsv_ops": list(HSV_OPS),
        "epochs": EPOCHS,
        "tau_range": list(TAU_RANGE),
        "lambda_range": list(LAMBDA_RANGE),
        "sobol": {"m": SOBOL_M, "seed": SOBOL_SEED},
        "trials": [asdict(t) for t in TRIALS],
        "dose_gammas": list(ex2223.DOSE_GAMMAS),
    }


def publish(rows: list[dict], trained: list[dict], evaled: list[dict], suppressed: list[dict]) -> dict:
    """Every ref the report reads: the trajectories, the eval and the suppression pass (one record per trial with
    its τ, λ_a, arm, and seed), their per-context arrays, and every checkpoint.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: r["trial"] for r in rows}

    def slim(r: dict) -> dict:
        return meta[r["label"]] | {k: v for k, v in r.items() if k != "arrays"}

    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for r in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    for r in suppressed:
        set_ref(SUPPRESSION_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="tau-lambda-sweep-trajectories.json"))
    for ref, records, name in ((EVAL_REF, evaled, "eval"), (SUPPRESSION_REF, suppressed, "suppression")):
        body = {"design": design(), "runs": [slim(r) for r in records]}
        set_ref(ref, put(json.dumps(body).encode(), name=f"tau-lambda-sweep-{name}.json"))
    return {
        "n_trained": len(trained),
        "eem": {r["label"]: r["task"]["eem"]["all"] for r in evaled},
        "landing": {r["label"]: r["landing"] for r in suppressed},
    }


def run(ctx: Ctx, chosen: tuple[Trial, ...] = TRIALS, epochs: int = EPOCHS) -> dict:
    """The whole DAG: resolve ex-2.2.21's corpus condition, train every trial, score it, and publish. *chosen* and
    *epochs* are arguments so a smoke run takes the same path at a fraction of the size.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    rows = rows_of(resolved["meta"], chosen)
    if epochs != EPOCHS:
        rows = [smoke_row(r, resolved["meta"], epochs) for r in rows]
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    trained = ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["variant"] for r in rows],
        [r["anchor_slices"] for r in rows],
        [r["pull"] for r in rows],
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        role="train",
    )
    ckpt = [t["checkpoint"] for t in trained]
    labels = [r["label"] for r in rows]
    evaled = ctx.map(
        ex2221.eval_one,
        ckpt,
        [resolved["holdout"]] * n,
        [None] * n,
        [OP_NAMES] * n,
        [K] * n,
        labels,
        role="eval",
    )
    suppressed = ctx.map(
        ex2222.suppress_one,
        ckpt,
        [resolved["holdout"]] * n,
        [OP_NAMES] * n,
        [K] * n,
        [False] * n,
        labels,
        role="suppress",
    )
    return ctx.run(publish, rows, trained, evaled, suppressed, role="prep")


def smoke_row(row: dict, meta, epochs: int) -> dict:
    """*row* rebuilt at *epochs*, with both schedules scaled to the shorter length: for a smoke run only."""
    t = Trial(**row["trial"])
    config, _ = ex2221.recipe_config(meta, t.model_seed, epochs)
    base = ex2216.Condition229(
        t.name, 1, t.name, lam=t.lam, tau=t.tau, epochs=epochs, ops=OP_NAMES, n_lines=ex2221.N_LINES
    )
    anchor, anti = ex2216.schedules(base)
    return row | {"config": config, "anchor": anchor, "anti": anti}


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One `no-emb` trial and one reference trial for eight epochs (past the five-epoch warm-up), through the whole
    DAG.
    """
    return run(ctx, (TRIALS[0], TRIALS[-1]), epochs=8)


COMPUTE = {
    "prep": dict(cpu=2, timeout=1800),
    # About 105,600 steps at 400 epochs, as ex-2.2.23's long runs; sized for a slow container
    # (eng: training-step-is-host-bound).
    "train": dict(gpu="L4", timeout=4 * 3600, watchdog=900, watchdog_grace=900),
    "eval": dict(gpu="L4", timeout=900),
    "suppress": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="tau-lambda-sweep", main=main, roles=COMPUTE)
