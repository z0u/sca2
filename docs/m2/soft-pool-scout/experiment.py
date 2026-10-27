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
from collections.abc import Sequence
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
rb = _load_sibling("reference-baseline", "soft_pool_scout_rb")
ee = _load_sibling("edit-effects", "soft_pool_scout_ee")
lr = spt.lr
sp = spt.sp
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

LEADERS: tuple[str, ...] = ("lt0.1-by40", "lt0.1-fixed-lt")
"""Stage 2: the conditions carried on to epoch 400, chosen after the stage-1 read: τ_ℓ ending at 0.1, sharpening
by epoch 40 (one change from the base) and fixed from the start (the simplest schedule, with only τ_p moving)."""

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
FINISHED_TRAJ_REF = PREFIX + "/finished-trajectories"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"
AXES_REF = PREFIX + "/axes"
WRITE_REF = PREFIX + "/write"
WRITE_ARRAYS_REF = PREFIX + "/write-arrays/{label}"
ALONG_REF = PREFIX + "/along"
RESTRICTED_REF = PREFIX + "/restricted-axes"

RESTRICTED_SETS = {k: v for k, v in lean.SLICE_SETS.items() if k != "every slice"}
"""The slice sets of the restricted edits; the edit at every slice has its axis scoring already (`AXES_REF`)."""
STORED_RECIPES = {"falloff": rb.sas, "step-soft": sp, "pool-123-soft": sp}
"""The three recipes recipe-compare scored, by label, beside the experiment whose checkpoints they are."""


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


def train(ctx: Ctx, resolved: dict, rows: list[dict], resume: Sequence[dict | None] | None = None) -> list[dict]:
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


def score_kept(kept_checkpoints, checkpoint, holdout, label: str, epochs: tuple[int, ...]) -> list[dict]:
    """Spill-timing's record scoring (the full edit, the reference edits and the probe) on each checkpoint a finished
    run kept, and on its end checkpoint, in one task.
    """
    from sca.compute.model import load_checkpoint
    from mini.store import get

    workdir = ex2221.get_data_dir() / "soft-pool-scout" / label
    out, score = [], None
    for e in epochs:
        model, config, _ = ex2221.kept_checkpoint(kept_checkpoints, e, workdir / "kept")
        if score is None:
            score = spt.record_scoring(config, OP_NAMES, K, holdout, workdir)
        out.append({"label": label, "epoch": e} | score(model))
    get(checkpoint, workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    if score is None:
        score = spt.record_scoring(config, OP_NAMES, K, holdout, workdir)
    out.append({"label": label, "epoch": config.scheduler.epochs} | score(model))
    return out


def publish_finished(rows: list[dict], trained: list[dict], measured: list[dict], dosed: list[dict],
                     axes: list[dict], written: list[dict], along: list[list[dict]]) -> dict:  # fmt: skip
    """The finished runs: trajectories, checkpoints, measurements, dose scoring, the axis scoring, the write bound,
    and the scoring of the kept checkpoints, under refs of their own.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"condition": r["condition"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        if "kept_checkpoints" in t:
            set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(FINISHED_TRAJ_REF, put(json.dumps(traj).encode(), name="soft-pool-scout-finished-trajectories.json"))

    def body(runs: list[dict]) -> bytes:
        return json.dumps({"design": design(), "runs": [meta[r["label"]] | r for r in runs]}).encode()

    set_ref(RESULTS_REF, put(body(measured), name="soft-pool-scout-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    set_ref(DOSE_REF, put(body([{k: v for k, v in d.items() if k != "arrays"} for d in dosed]), name="soft-pool-scout-dose.json"))  # fmt: skip
    set_ref(AXES_REF, put(body(axes), name="soft-pool-scout-axes.json"))
    for w in written:
        set_ref(WRITE_ARRAYS_REF.format(label=w["label"]), w["arrays"])
    set_ref(WRITE_REF, put(body([{k: v for k, v in w.items() if k != "arrays"} for w in written]), name="soft-pool-scout-write.json"))  # fmt: skip
    set_ref(ALONG_REF, put(body([r for a in along for r in a]), name="soft-pool-scout-along.json"))
    return {"n_finished": len(trained)}


def restricted_axes_one(checkpoint, holdout, label: str) -> dict:
    """The full edit restricted to each slice set (as embedding-lean restricts it), along each coordinate axis, on
    every held-out completion context: per op, expected exact match on the clean pass and the drop under each axis and
    slice set (`drop[set][axis][op]`, e₁ first). The median over the other axes is the baseline of the D2.2 design,
    here for the restricted edits, which earlier reports netted against the same edit on the control.
    """
    import numpy as np

    from mini import get_data_dir
    from mini.store import get
    from sca.intervention import Subspace, logits_at, projection

    ex2216, ex2218 = lean.ex2216, lean.ex2218
    workdir = get_data_dir() / "soft-pool-scout" / "restricted" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    ctx = ex2216._post_queries(P, ho, K, tok2color)
    eq_role = ex2216.query_role(K, ex2216.QUERY_EQ)
    width = model.transformer.wte.shape[1]
    mask = ex2216.site_positions("every position", K)
    full = rb.DOSE_GAMMAS[-1]

    def per_op(v: np.ndarray) -> list[float]:
        return [float(v[ho.op_ids == o].mean()) for o in range(len(OP_NAMES))]

    def score(op, slices) -> np.ndarray:
        lg = logits_at(model, ho.tokens, op, slices, eq_role, mask if slices else None)
        return ex2216._match(table, ctx, ex2216._color_probs(lg, color_ids))

    clean = score(projection(Subspace.axis(width), 0.0), ())
    drop = {
        name: [per_op(clean - score(projection(Subspace.axis(width, i), full), slices)) for i in range(width)]
        for name, slices in RESTRICTED_SETS.items()
    }
    return {"label": label, "clean": per_op(clean), "drop": drop}


def resolve_finished(seeds: tuple[int, ...], leaders: tuple[str, ...]) -> dict:
    """The final checkpoints of the three stored recipes and of the finished leaders at *seeds*, by label."""
    from mini.store import get_ref

    refs = {f"{a}-s{s}": m.CHECKPOINT_REF.format(label=m.label_of(a, s)) for a, m in STORED_RECIPES.items() for s in seeds}  # fmt: skip
    refs |= {label_of(c, s, None): CHECKPOINT_REF.format(label=label_of(c, s, None)) for c in leaders for s in seeds}
    found = {label: get_ref(ref) for label, ref in refs.items()}
    missing = [label for label, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    return found


def publish_restricted(scored: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    body = {"design": design(), "slice_sets": {k: list(v) for k, v in RESTRICTED_SETS.items()}, "runs": scored}
    set_ref(RESTRICTED_REF, put(json.dumps(body).encode(), name="soft-pool-scout-restricted-axes.json"))
    return {"n_restricted": len(scored)}


def restrict(ctx: Ctx, resolved: dict, seeds: tuple[int, ...], leaders: tuple[str, ...], publishes: bool) -> dict:
    """Paper round of stage 2: the restricted edits along every axis of each finished run and recipe."""
    checkpoints = ctx.run(resolve_finished, seeds, leaders, role="prep")
    labels = list(checkpoints)
    n = len(labels)
    scored = ctx.map(restricted_axes_one, [checkpoints[x] for x in labels], [resolved["holdout"]] * n, labels, role="measure")  # fmt: skip
    if not publishes:
        return {"restricted": [r["label"] for r in scored]}
    return ctx.run(publish_restricted, scored, role="prep")


def finish(ctx: Ctx, resolved: dict, stopped: list[dict], leaders: tuple[str, ...], epochs: int,
           seeds: tuple[int, ...], publishes: bool, full: bool) -> dict:  # fmt: skip
    """Stage 2: carry each leader on from its stop to the end, then measure and score it as recipe-compare did, and
    score the checkpoints it kept along the way.
    """
    rows = rows_of(resolved["meta"], tuple(c for c in CONDITIONS if c.name in leaders), None, epochs, seeds)
    by_label = {t["label"]: t for t in stopped}
    stop = next(int(lab.rsplit("-e", 1)[1]) for lab in by_label)
    resume = [by_label[label_of(r["condition"]["name"], r["model_seed"], stop)] for r in rows]
    trained = train(ctx, resolved, rows, resume)
    n = len(rows)
    labels = [r["label"] for r in rows]
    holdout = [resolved["holdout"]] * n
    checkpoints = [t["checkpoint"] for t in trained]
    measured = ctx.map(lean.measure_one, checkpoints, holdout, labels, [True] * n, role="measure")
    dosed = ctx.map(ex2222.suppress_one, checkpoints, holdout, [OP_NAMES] * n, [K] * n, [True] * n, labels, role="measure")  # fmt: skip
    along = ctx.map(
        score_kept,
        [t["kept_checkpoints"] for t in trained],
        checkpoints,
        holdout,
        labels,
        [tuple(t["kept_epochs"]) for t in trained],
        role="measure",
    )
    if not full:  # a smoke run leaves out the two slow passes, which earlier experiments ran as they are here
        return {"finished": labels, "along": [len(a) for a in along]}
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    written = ctx.map(ee.measure_one, checkpoints, holdout, labels, role="write")
    if not publishes:
        return {"finished": labels}
    return ctx.run(publish_finished, rows, trained, measured, dosed, axes, written, along, role="prep")


def run(ctx: Ctx, epochs: int = EPOCHS, stop: int = STOP_EPOCH, seeds: tuple[int, ...] = SEEDS,
        conditions: tuple[Condition, ...] = NEW, leaders: tuple[str, ...] = LEADERS, publishes: bool = True,
        full: bool = True) -> dict:  # fmt: skip
    """Resolve ex-2.2.21's corpus condition, train every stage-1 run to *stop*, score each at the stop, and publish;
    then carry the *leaders* on to the end (stage 2). The arguments let a smoke run take the same path at a fraction
    of the size, and leave the refs alone.
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
    out = {"labels": [t["label"] for t in trained]}
    if publishes:
        out = ctx.run(publish_screen, rows, trained, screen, role="prep")
    if not leaders:
        return out
    out |= finish(ctx, resolved, trained, leaders, epochs, seeds, publishes, full)
    if full:
        out |= restrict(ctx, resolved, seeds, leaders, publishes)
    return out


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of two conditions (a delay, and fixed τ_ℓ) for eight epochs, stopped at six, through the screen, and
    the second carried on to the end and measured.
    """
    picked = tuple(c for c in NEW if c.name in ("fixed-delay", "lt0.1-fixed-lt"))
    return run(ctx, epochs=8, stop=6, seeds=SEEDS[:1], conditions=picked, leaders=("lt0.1-fixed-lt",),
               publishes=False, full=False)  # fmt: skip


COMPUTE = sp.COMPUTE | {"write": ee.COMPUTE["measure"]}

experiment = Experiment(name="soft-pool-scout", main=main, roles=COMPUTE)
