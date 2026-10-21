"""How often the anchor is lost: `falloff` and the soft start at 32 new seeds each, stopped early.

The recipe of record (split-anti-seeds' `falloff`) kept the anchor on all eight of its seeds, which fits a loss rate as
high as three in ten. Its close variants lost it on one to three seeds of eight, and the soft start of soft-pool
(`step, soft τ`) kept it on all eight of its own. In every run of those experiments whose outcome we know, the outcome
was set in the first sixty or so epochs: on a run that lost the anchor, the example answers were off e₁ through epochs
44 to 64, and on every run that kept it they were on. So this experiment trains each new seed on the full 400-epoch
schedule and stops it at epoch 64, and the report calls each run from α at the example answers over that window. A
run whose window falls between the stored lost and kept runs is trained again to the end and dose-scored, so that the
call never rests on the window alone where the stored runs could not settle it.

Two seeds of each arm are also re-trained, stopped early, at seeds the stored runs used, so the report can check that
stopping early leaves the trajectory up to the stop unchanged.

    bin/mini run docs/m2/loss-rate/experiment.py --app modal --max-containers 16 --budget 2h
    bin/mini status loss-rate
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
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


sp = _load_sibling("soft-pool", "loss_rate_sp")
sas = sp.sas
st = sp.st
lean = sp.lean
ex2222 = sp.ex2222
ex2221 = sp.ex2221

OP_NAMES = sp.OP_NAMES
ANCHORED_OP = sp.ANCHORED_OP
K = sp.K
EPOCHS = sp.EPOCHS
EVERY = sp.EVERY
TRAJ_STRIDE_EPOCHS = sp.TRAJ_STRIDE_EPOCHS
FALLOFF_HOLDS = sp.FALLOFF_HOLDS

# --- The arms and the seeds --------------------------------------------------------------------------------

ARMS: tuple = (
    sp.Arm("falloff", sp.STEP, False, "falloff"),
    sp.Arm("step-soft", sp.STEP, True, "step, soft τ"),
)
"""`falloff` (the step pull on slices 2 to 4 at a fixed τ_p, as split-anti-seeds trained it) and the soft start of
soft-pool (the same step, with τ_p sharpening from 1 to 0.1 over the warm-up)."""

SEEDS = tuple(range(712, 744))
"""32 model seeds no earlier run has used, the same for both arms so the two are paired by initialization, batches and
label draws."""
REPRO_SEEDS = (704, 705)
"""Seeds whose full-length runs are stored for both arms (split-anti-seeds' `falloff`, soft-pool's `step-soft`)."""

STOP_EPOCH = 64
"""Where each screen run stops: the end of the window over which the report calls it."""
WINDOW = (44, 64)
"""The epochs (inclusive) over which the call averages α at the example answers."""
TRACK_SLICE = 3
"""The slice the call reads: one both arms pull, and the deepest one every stored arm pulls."""
LOST_MAX = 0.15
"""The highest window mean on a stored run that lost the anchor (rounded up)."""
KEPT_MIN = 0.28
"""The lowest window mean on a stored run that kept it (rounded down). A screen run between the two is finished."""
CALL_AT = 0.22
"""Below this window mean a screen run is called lost: the middle of the gap."""

BUDGET_USD = 8
"""Soft-pool cost about \\$0.35 per 400-epoch run with its measurements. A run stopped at epoch 64 trains for a sixth
as long with no measurement after it: 68 come to about \\$4 to \\$5, with room for a few finished runs."""


def label_of(arm: str, model_seed: int, stop: int | None = STOP_EPOCH) -> str:
    return f"{arm}-s{model_seed}" + (f"-e{stop}" if stop is not None else "")


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/loss-rate"
TRAJ_REF = PREFIX + "/trajectories"
CALLS_REF = PREFIX + "/calls"
FINISHED_TRAJ_REF = PREFIX + "/finished-trajectories"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"


def rows_of(meta, seeds: tuple[int, ...], stop: int | None, epochs: int = EPOCHS) -> list[dict]:
    """One row per run of each arm at *seeds*: soft-pool's rows, stopped at *stop* (`None` for the full schedule)."""
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
                    "anchor": a.anchor(anchor),
                    "anti": anti | {"hold": FALLOFF_HOLDS},
                    "anchor_slices": a.anchor_slices(),
                    "arm": asdict(a),
                    "model_seed": s,
                    "stop": stop,
                    "label": label_of(a.name, s, stop),
                }
            )
    return rows


def window_mean(traj: dict, window: tuple[float, float] = WINDOW, sl: int = TRACK_SLICE) -> float:
    """α at the example answers on the `difference` probe contexts, at slice *sl*, averaged over the records whose
    epoch falls in *window* (each epoch once: a run stopped at the window's end records its last epoch twice).
    """
    import numpy as np

    ep = np.asarray(traj["epoch"])
    a = np.asarray(traj["alpha_anchored"])[:, sl][:, list(lean.ANSWERS)].mean(axis=1)
    keep = (ep >= window[0] - 1e-6) & (ep <= window[1] + 1e-6)
    _, first = np.unique(np.round(ep[keep], 3), return_index=True)
    return float(a[keep][first].mean())


def call_of(value: float) -> str:
    if value < LOST_MAX:
        return "lost"
    if value >= KEPT_MIN:
        return "kept"
    return "lost?" if value < CALL_AT else "kept?"


def train(ctx: Ctx, resolved: dict, rows: list[dict]) -> list[dict]:
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    return ctx.map(
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
        [r["stop"] for r in rows],
        role="train",
    )


def design() -> dict[str, Any]:
    return {
        "experiment": "loss-rate",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "stop_epoch": STOP_EPOCH,
        "window": list(WINDOW),
        "track_slice": TRACK_SLICE,
        "lost_max": LOST_MAX,
        "kept_min": KEPT_MIN,
        "call_at": CALL_AT,
        "arms": [asdict(a) | {"anchor": a.anchor(st.schedules()[0])} for a in ARMS],
        "anti_holds": list(FALLOFF_HOLDS),
        "seeds": list(SEEDS),
        "repro_seeds": list(REPRO_SEEDS),
        "dose_gammas": list(ex2222.DOSE_GAMMAS),
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
    }


def publish_screen(rows: list[dict], trained: list[dict]) -> list[dict]:
    """The screen trajectories under one ref, and each run's call. Returns the calls."""
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"], "stop": r["stop"]} for r in rows}
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="loss-rate-trajectories.json"))
    calls = []
    for t in trained:
        m = meta[t["label"]]
        if m["model_seed"] not in SEEDS:
            continue
        v = window_mean(t["traj"])
        calls.append(m | {"label": t["label"], "window_mean": v, "call": call_of(v)})
    set_ref(CALLS_REF, put(json.dumps({"design": design(), "runs": calls}).encode(), name="loss-rate-calls.json"))
    return calls


def publish_finished(rows: list[dict], trained: list[dict], scored: list[dict], dosed: list[dict]) -> dict:
    """The finished runs (those whose window was between the stored lost and kept runs): trajectories, checkpoints,
    measurements and dose scoring, under refs of their own.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(FINISHED_TRAJ_REF, put(json.dumps(traj).encode(), name="loss-rate-finished-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="loss-rate-results.json"))
    dose = {"design": design(), "runs": [{k: v for k, v in d.items() if k != "arrays"} for d in dosed]}
    set_ref(DOSE_REF, put(json.dumps(dose).encode(), name="loss-rate-dose.json"))
    return {"n_finished": len(trained)}


def run(ctx: Ctx, epochs: int = EPOCHS, stop: int = STOP_EPOCH, seeds: tuple[int, ...] = SEEDS,
        repro: tuple[int, ...] = REPRO_SEEDS, publishes: bool = True) -> dict:  # fmt: skip
    """Resolve ex-2.2.21's corpus condition, train every screen run to *stop*, call it, and finish the runs the call
    can't settle: train them to the end, measure them and score them at every dose. The arguments let a smoke run
    take the same path at a fraction of the size, and leave the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    rows = rows_of(resolved["meta"], seeds + repro, stop, epochs)
    trained = train(ctx, resolved, rows)
    if not publishes:
        return {"labels": [t["label"] for t in trained]}
    calls = ctx.run(publish_screen, rows, trained, role="prep")
    unsure = {(c["arm"], c["model_seed"]) for c in calls if c["call"].endswith("?")}
    full = [r for r in rows_of(resolved["meta"], seeds, None, epochs) if (r["arm"]["name"], r["model_seed"]) in unsure]
    out = {"n_screen": len(calls), "n_unsure": len(full)}
    if not full:
        return out
    n = len(full)
    finished = train(ctx, resolved, full)
    scored = ctx.map(
        lean.measure_one,
        [t["checkpoint"] for t in finished],
        [resolved["holdout"]] * n,
        [r["label"] for r in full],
        [True] * n,
        role="measure",
    )
    dosed = ctx.map(
        ex2222.suppress_one,
        [t["checkpoint"] for t in finished],
        [resolved["holdout"]] * n,
        [OP_NAMES] * n,
        [K] * n,
        [True] * n,
        [r["label"] for r in full],
        role="measure",
    )
    return out | ctx.run(publish_finished, full, finished, scored, dosed, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of each arm for eight epochs, stopped at six (past the five-epoch warm-up), through the training."""
    return run(ctx, epochs=8, stop=6, seeds=SEEDS[:1], repro=(), publishes=False)


COMPUTE = sas.COMPUTE

experiment = Experiment(name="loss-rate", main=main, roles=COMPUTE)
