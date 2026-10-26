"""Soft pool scout: sixteen schedules of `pool 1–3, soft τ` screened at epoch 64, then the best few finished.

`pool 1–3, soft τ` (soft-pool) removes about as much as the recipe of record and settles on slice 3 on every seed, but
two of its eight runs go over the spill criterion, and recipe-compare traced that spill to block 1. Spill-timing found
the runs over the criterion spill early, while the pool sharpens, and that the early kind already shows at epoch 64.
So stage 1 trains sixteen schedule conditions at the eight soft-pool seeds (704 to 711) on the full 400-epoch schedule,
stops each at epoch 64 with its resume state, and scores it there:

- the loss-rate screen's call, from α at the example answers over epochs 44 to 64;
- the full edit on the epoch-64 checkpoint against the median of eight reference axes, as spill-timing scored it;
- α by slice and role, from the trajectory's map (`alpha_map`).

The conditions change one or two schedules of the base (τ_ℓ 4 → 0.4 and τ_p 1 → 0.1 by epoch 40, λ_a warming up over
epochs 0 to 40). Twelve cross the slice temperature at the end (0.1, 0.2, 0.4) with when the temperatures are sharp (by
epoch 10, 20 or 40, or τ_ℓ at its end value from the start while τ_p sharpens by 40); the base is one of them, stored
by soft-pool and scored at every record by spill-timing, so eleven are new. Five more move the λ_a warm-up (a delay:
zero until epoch 20, then up by 60; a stretch: up by 80), each with the soft temperatures and with both fixed at their
end values, and the fixed temperatures with the base warm-up.

Stage 2 carries the leaders (`LEADERS`, chosen from the stage-1 read) on from their stops to epoch 400, keeping
checkpoints along the way.

    bin/mini run docs/m2/soft-pool-scout/experiment.py --app modal --max-containers 10 --budget 3h
    bin/mini status soft-pool-scout
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


spt = _load_sibling("spill-timing", "soft_pool_scout_spt")
lr = spt.lr
sp = spt.sp
st = sp.st
ex2222 = sp.ex2222
ex2221 = sp.ex2221

OP_NAMES = sp.OP_NAMES
ANCHORED_OP = sp.ANCHORED_OP
K = sp.K
EPOCHS = sp.EPOCHS
EVERY = sp.EVERY
TRAJ_STRIDE_EPOCHS = sp.TRAJ_STRIDE_EPOCHS
FALLOFF_HOLDS = sp.FALLOFF_HOLDS
SEEDS = sp.SEEDS
"""704 to 711: soft-pool's seeds, so every run has its stored `pool 1–3, soft τ`, `falloff` and control twins."""
STOP_EPOCH = lr.STOP_EPOCH
"""Where stage 1 stops each run: the end of the loss-rate screen's window."""
KEEP_EPOCHS = ex2221.KEEP_EPOCHS
BASE = "pool-123-soft"
"""Soft-pool's arm, the base every condition changes."""

# --- The conditions ----------------------------------------------------------------------------------------

TAU_START = sp.TAU_START
SLICE_TAU_START = sp.SLICE_TAU_START


@dataclass(frozen=True)
class Condition:
    name: str
    shown_as: str
    slice_tau: float = sp.SLICE_TAU
    """τ_ℓ at the end of its schedule (throughout, when it is fixed)."""
    sharp_by: float | None = 40.0
    """The epoch by which both temperatures reach their end values, from τ_p = 1 and τ_ℓ = 4; `None` holds both at
    their end values from epoch 0."""
    slice_soft: bool = True
    """`False` holds τ_ℓ at its end value from epoch 0 while τ_p still sharpens by *sharp_by*."""
    warmup: tuple[float, float] = (0.0, 40.0)
    """The λ_a warm-up: zero until the first epoch, the peak from the second."""

    def anchor(self, base: dict, f: float = 1.0) -> dict:
        """The condition's `AnchorSpec` keywords on the recipe's *base*, with its epochs scaled by *f* (a smoke run).
        Only the fields it sets enter the spec.
        """
        spec = dict(base) | {"pool_slices": True, "slice_tau": self.slice_tau, "warmup_epochs": self.warmup[1] * f}
        if self.warmup[0] > 0:
            spec["warmup_start"] = self.warmup[0] * f
        if self.sharp_by is not None:
            spec |= {"tau_start": TAU_START, "tau_anneal_end": self.sharp_by * f}
            if self.slice_soft:
                spec["slice_tau_start"] = SLICE_TAU_START
        return spec


def _crossing() -> tuple[Condition, ...]:
    out = []
    for lt in (0.1, 0.2, 0.4):
        for by in (10.0, 20.0, 40.0):
            out.append(Condition(f"lt{lt:g}-by{by:g}", f"τ_ℓ {lt:g}, by {by:g}", lt, by))
        out.append(Condition(f"lt{lt:g}-fixed-lt", f"τ_ℓ {lt:g}, τ_ℓ fixed", lt, 40.0, slice_soft=False))
    return tuple(out)


CROSSING = _crossing()
"""τ_ℓ at the end × when the temperatures are sharp: twelve conditions, the base (`lt0.4-by40`) among them."""
DELAY = (20.0, 60.0)
STRETCH = (0.0, 80.0)
WARMUPS: tuple[Condition, ...] = (
    Condition("delay", "λ_a delay", warmup=DELAY),
    Condition("stretch", "λ_a stretch", warmup=STRETCH),
    Condition("fixed", "fixed τ", sharp_by=None),
    Condition("fixed-delay", "fixed τ, λ_a delay", sharp_by=None, warmup=DELAY),
    Condition("fixed-stretch", "fixed τ, λ_a stretch", sharp_by=None, warmup=STRETCH),
)
"""The λ_a warm-up moved, at the base temperatures and at fixed ones (τ_p = 0.1, τ_ℓ = 0.4 throughout)."""
STORED = "lt0.4-by40"
"""The crossing's base: soft-pool's `pool 1–3, soft τ`, not trained again."""
CONDITIONS = CROSSING + WARMUPS
NEW = tuple(c for c in CONDITIONS if c.name != STORED)

LEADERS: tuple[str, ...] = ()
"""Stage 2: the conditions carried on to epoch 400, chosen after the stage-1 read."""

BUDGET_USD = 20
"""Soft-pool cost about \\$0.35 per 400-epoch run with its measurements. Stage 1 is 128 runs to epoch 64, about a sixth
of a run each (\\$7.50) plus their scoring; stage 2 is the rest of 16 to 24 runs (\\$5 to \\$7) and their measurements."""


def label_of(condition: str, model_seed: int, stop: int | None = STOP_EPOCH) -> str:
    return f"{condition}-s{model_seed}" + (f"-e{stop}" if stop is not None else "")


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/soft-pool-scout"
SCREEN_TRAJ_REF = PREFIX + "/screen-trajectories"
SCREEN_REF = PREFIX + "/screen"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESUME_REF = PREFIX + "/resume/{label}"
KEPT_REF = PREFIX + "/kept/{label}"


def rows_of(meta, conditions: tuple[Condition, ...], stop: int | None, epochs: int = EPOCHS,
            seeds: tuple[int, ...] = SEEDS) -> list[dict]:  # fmt: skip
    """One row per run of each condition at *seeds*: soft-pool's config and `falloff` anti weight, the condition's
    anchor schedule, stopped at *stop* (`None` for the full schedule).
    """
    anchor, anti = st.schedules()
    f = epochs / EPOCHS
    if epochs != EPOCHS:
        # A smoke run keeps the shape of the schedule, scaled to its length.
        anchor = anchor | {k: anchor[k] * f for k in ("warmup_epochs", "anneal_start", "anneal_end")}
        anti = anti | {k: anti[k] * f for k in ("anneal_end", "anchor_anneal_start", "anchor_anneal_end")}
    rows = []
    for c in conditions:
        for s in seeds:
            config, _ = ex2221.recipe_config(meta, s, epochs)
            rows.append(
                {
                    "config": config,
                    "anchor": c.anchor(anchor, f),
                    "anti": anti | {"hold": FALLOFF_HOLDS},
                    "anchor_slices": (1, 2, 3),
                    "condition": asdict(c),
                    "model_seed": s,
                    "stop": stop,
                    "label": label_of(c.name, s, stop),
                }
            )
    return rows


def train(ctx: Ctx, resolved: dict, rows: list[dict], resume: list[dict | None] | None = None) -> list[dict]:
    """`train_one` on every row with the alignment map and kept checkpoints, carrying on *resume* where given."""
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    epochs = rows[0]["config"].scheduler.epochs
    keep = tuple(e for e in KEEP_EPOCHS if e < epochs) if epochs == EPOCHS else (2, 4, 6)
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
        [None] * n,
        [keep] * n,
        resume if resume is not None else [None] * n,
        role="train",
    )


def design() -> dict[str, Any]:
    anchor = st.schedules()[0]
    return {
        "experiment": "soft-pool-scout",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "stop_epoch": STOP_EPOCH,
        "base": BASE,
        "stored": STORED,
        "conditions": [asdict(c) | {"anchor": c.anchor(anchor)} for c in CONDITIONS],
        "leaders": list(LEADERS),
        "anti_holds": list(FALLOFF_HOLDS),
        "seeds": list(SEEDS),
        "window": list(lr.WINDOW),
        "track_slice": lr.TRACK_SLICE,
        "keep_epochs": list(KEEP_EPOCHS),
        "full_dose": spt.FULL_DOSE,
        "n_sham": spt.N_SHAM,
        "sham_seed": spt.SHAM_SEED,
        "probe_slices": list(spt.PROBE_SLICES),
        "probe_positions": spt.position_names(K),
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
    }


def publish_screen(rows: list[dict], trained: list[dict], scored: list[dict]) -> dict:
    """The stage-1 trajectories under one ref; each run's call and edit scoring at the stop under another; and each
    run's checkpoint at the stop, resume state and kept checkpoints under refs of their own, so stage 2 can carry
    them on after the memo records expire.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"condition": r["condition"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        set_ref(RESUME_REF.format(label=t["label"]), t["resume_state"])
        if "kept_checkpoints" in t:
            set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(SCREEN_TRAJ_REF, put(json.dumps(traj).encode(), name="soft-pool-scout-screen-trajectories.json"))
    calls = {t["label"]: lr.window_mean(t["traj"]) for t in trained}
    runs = [meta[s["label"]] | s | {"window_mean": calls[s["label"]], "call": lr.call_of(calls[s["label"]])} for s in scored]  # fmt: skip
    set_ref(
        SCREEN_REF, put(json.dumps({"design": design(), "runs": runs}).encode(), name="soft-pool-scout-screen.json")
    )
    return {"n_screen": len(runs)}


def run(ctx: Ctx, epochs: int = EPOCHS, stop: int = STOP_EPOCH, seeds: tuple[int, ...] = SEEDS,
        conditions: tuple[Condition, ...] = NEW, publishes: bool = True) -> dict:  # fmt: skip
    """Resolve ex-2.2.21's corpus condition, train every stage-1 run to *stop*, score each at the stop, and publish.
    The arguments let a smoke run take the same path at a fraction of the size, and leave the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    rows = rows_of(resolved["meta"], conditions, stop, epochs, seeds)
    trained = train(ctx, resolved, rows)
    per = spt.STORED_PER_TASK
    chunks = [trained[i : i + per] for i in range(0, len(trained), per)]
    scored = ctx.map(
        spt.score_checkpoints,
        [{t["label"]: t["checkpoint"] for t in c} for c in chunks],
        [resolved["holdout"]] * len(chunks),
        [OP_NAMES] * len(chunks),
        [K] * len(chunks),
        role="measure",
    )
    screen = [r for chunk in scored for r in chunk]
    if not publishes:
        return {"labels": [t["label"] for t in trained], "resume": [("resume_state" in t) for t in trained]}
    return ctx.run(publish_screen, rows, trained, screen, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of two conditions (a delay, and fixed τ_ℓ) for eight epochs, stopped at six, through the screen."""
    picked = tuple(c for c in NEW if c.name in ("fixed-delay", "lt0.1-fixed-lt"))
    return run(ctx, epochs=8, stop=6, seeds=SEEDS[:1], conditions=picked, publishes=False)


COMPUTE = sp.COMPUTE

experiment = Experiment(name="soft-pool-scout", main=main, roles=COMPUTE)
