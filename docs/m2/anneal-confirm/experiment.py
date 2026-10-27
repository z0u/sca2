"""Anneal confirmation: `τ_ℓ 0.1 → 0.025` at fresh seeds, and its loss rate at forty.

The soft pool scout's candidate recipe pools the pull over slices 1 to 3 with τ_ℓ held at 0.1 and τ_p sharpening from
1 to 0.1 by epoch 40. The late spill scout found that sharpening τ_ℓ again late, to 0.025 over epochs 64 to 360,
lowered the late spill at no measured cost. The recipe here folds that into one move: τ_ℓ goes from 0.1 at epoch 0 to
0.025 at epoch 400, minimum-jerk in log τ (`AnchorSpec.slice_tau_late`), so it is still near 0.1 while the anchor
settles (0.096 at epoch 64) and reaches 0.05 at epoch 200. Everything else is the soft pool scout's `τ_ℓ 0.1, τ_ℓ
fixed`.

Two sets of runs, sharing their first 64 epochs:

- the loss-rate count: the recipe at the forty seeds of the loss-rate screen (704 to 743), each stopped at epoch 64
  with its resume state and called from α at the example answers over epochs 44 to 64, as that screen called
  `falloff` and `step, soft τ` at the same seeds;
- the confirmation: the runs at seeds 712 to 719 carried on from their stops to epoch 400, keeping a checkpoint every
  32 epochs, beside a control (no anchor) at each of those seeds, trained in full. Any run the call is unsure of is
  carried on and measured too, so that each loss counted is a measured one.

    bin/mini run docs/m2/anneal-confirm/experiment.py --app modal --max-containers 10 --budget 3h
    bin/mini status anneal-confirm
"""

from __future__ import annotations

import importlib.util
import sys
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


sps = _load_sibling("soft-pool-scout", "anneal_confirm_sps")
ex2223 = _load_sibling("ex-2.2.23", "anneal_confirm_ex2223")
lr = sps.lr
rb = sps.rb
ee = sps.ee
lean = sps.lean
ex2222 = sps.ex2222
ex2221 = sps.ex2221

OP_NAMES = sps.OP_NAMES
ANCHORED_OP = sps.ANCHORED_OP
K = sps.K
EPOCHS = sps.EPOCHS
STOP_EPOCH = lr.STOP_EPOCH
"""Where the loss-rate runs stop: the end of the loss-rate screen's window."""
BASE = "lt0.1-fixed-lt"
"""The soft pool scout's condition the recipe changes: τ_ℓ 0.1 throughout, τ_p 1 → 0.1 by epoch 40."""
SLICE_TAU_END = 0.025
"""Where τ_ℓ ends: the late spill scout's `lt-late` value, two halvings past 0.1."""
ANNEAL = (SLICE_TAU_END, 0.0, float(EPOCHS))
"""`AnchorSpec.slice_tau_late`: τ_ℓ moves from 0.1 to 0.025 over the whole schedule."""
NAME = "anneal"

COUNT_SEEDS: tuple[int, ...] = tuple(range(704, 712)) + lr.SEEDS
"""The forty seeds of the loss-rate screen (its 32 new seeds and the 8 of split-anti-seeds), so each call pairs with
that screen's calls of `falloff` and `step, soft τ` at the seed."""
SEEDS: tuple[int, ...] = tuple(range(712, 720))
"""Fresh seeds for the confirmation: no finished run of any recipe has used them."""
assert set(SEEDS) <= set(COUNT_SEEDS)
STOP_KEEP = (16, 32, 64)
"""Checkpoints a loss-rate run keeps before its stop, as the soft pool scout's stage 1 did."""
FINISH_KEEP = tuple(range(96, EPOCHS, 32))
"""Checkpoints kept after the stop, every 32 epochs, as in the late spill scout."""

TWINS = False
"""Whether `falloff` trains in full at the confirmation seeds too, so its removal and task pair with the recipe's by
seed (about \\$3 more)."""

BUDGET_USD = 12
"""From the task times of the soft pool scout and the late spill scout on L4s (about \\$0.80 an hour): forty runs to
epoch 64 at about four minutes each (\\$2.50); eight carried on to 400 and eight controls at about twenty minutes each
(\\$4.50); the end measurements on sixteen runs (\\$1.50); the axis scoring of fourteen checkpoints on eight runs
(\\$1); and room for a few finished runs the call is unsure of."""


def anchor_of(base: dict, f: float = 1.0) -> dict:
    """The recipe's `AnchorSpec` keywords on the soft pool scout's *base* condition (epochs scaled by *f*)."""
    v, a, b = ANNEAL
    return base | {"slice_tau_late": (v, a * f, b * f)}


def label_of(model_seed: int, stop: int | None = STOP_EPOCH, name: str = NAME) -> str:
    return f"{name}-s{model_seed}" + (f"-e{stop}" if stop is not None else "")


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/anneal-confirm"
SCREEN_TRAJ_REF = PREFIX + "/screen-trajectories"
CALLS_REF = PREFIX + "/calls"
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
RESTRICTED_REF = PREFIX + "/restricted-axes"
ALONG_REF = PREFIX + "/along"


def anchored_rows(meta, seeds: tuple[int, ...], stop: int | None, epochs: int = EPOCHS) -> list[dict]:
    """One row per run of the recipe at *seeds*: the soft pool scout's row of its base condition with τ_ℓ annealing,
    stopped at *stop* (`None` for the full schedule).
    """
    base = next(c for c in sps.CONDITIONS if c.name == BASE)
    f = epochs / EPOCHS
    return [
        r | {"anchor": anchor_of(r["anchor"], f), "kind": NAME, "label": label_of(r["model_seed"], stop)}
        for r in sps.rows_of(meta, (base,), stop, epochs, seeds)
    ]


def control_rows(meta, seeds: tuple[int, ...], epochs: int = EPOCHS) -> list[dict]:
    """One row per control at *seeds*: ex-2.2.23's control (ex-2.2.21's `control` arm, with no pull and no anti term)
    at the length, so it shares its initialization, batches and label draws with the anchored run at the seed.
    """
    control = next(c for c in ex2223.CONDITIONS if c.name == "control")
    return [
        r | {"kind": "control", "stop": None, "model_seed": s, "label": label_of(s, None, "control")}
        for s, r in zip(seeds, ex2223.new_rows(meta, [(control, epochs, s) for s in seeds]), strict=True)
    ]


def falloff_rows(meta, seeds: tuple[int, ...], epochs: int = EPOCHS) -> list[dict]:
    """One row per `falloff` run at *seeds*, as the loss-rate screen built it, trained in full."""
    return [
        r | {"kind": "falloff", "label": label_of(r["model_seed"], None, "falloff")}
        for r in lr.rows_of(meta, seeds, None, epochs)
        if r["arm"]["name"] == "falloff"
    ]


def train(ctx: Ctx, resolved: dict, rows: list[dict], resume: list[dict | None], keep: list[tuple[int, ...]]) -> list[dict]:  # fmt: skip
    """`train_one` on every row with the alignment map, carrying on *resume* where given and keeping *keep*."""
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(sps.TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    return ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r.get("variant", "whole") for r in rows],
        [r["anchor_slices"] for r in rows],
        [r.get("pull", "") for r in rows],
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [ex2221.N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        [sps.EVERY] * n,
        [None] * n,
        [True] * n,
        [r["stop"] for r in rows],
        [None] * n,
        keep,
        resume,
        role="train",
    )


def kept_axes_one(kept_checkpoints, checkpoint, holdout, label: str, epochs: tuple[int, ...], end: int) -> list[dict]:
    """The full edit at every position and slice along each coordinate axis, on each checkpoint a run kept (after
    *epochs*) and on its end checkpoint (after *end*): per op, expected exact match on the clean pass and the drop
    under each axis (`drop[axis][op]`, e₁ first), so the spill along training is net of the median over the other 63
    axes of the same run, as at the end.
    """
    import numpy as np

    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get

    ex2216, ex2218 = lean.ex2216, lean.ex2218
    workdir = ex2221.get_data_dir() / "anneal-confirm" / "along" / label
    end_model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    ctx = ex2216._post_queries(P, ho, K, tok2color)
    eq_role = ex2216.query_role(K, ex2216.QUERY_EQ)
    mask = ex2216.site_positions("every position", K)
    width = end_model.transformer.wte.shape[1]
    every = tuple(range(len(end_model.transformer.blocks) + 1))
    full = rb.DOSE_GAMMAS[-1]

    def per_op(v: np.ndarray) -> list[float]:
        return [float(v[ho.op_ids == o].mean()) for o in range(len(OP_NAMES))]

    def scored(model, epoch: int) -> dict:
        def score(op, slices) -> np.ndarray:
            lg = logits_at(model, ho.tokens, op, slices, eq_role, mask if slices else None)
            return ex2216._match(table, ctx, ex2216._color_probs(lg, color_ids))

        clean = score(projection(Subspace.axis(width), 0.0), ())
        drop = [per_op(clean - score(projection(Subspace.axis(width, i), full), every)) for i in range(width)]
        return {"label": label, "epoch": epoch, "clean": per_op(clean), "drop": drop}

    out = [scored(ex2221.kept_checkpoint(kept_checkpoints, e, workdir / "kept")[0], e) for e in epochs]
    return out + [scored(end_model, end)]


def design() -> dict[str, Any]:
    base = next(c for c in sps.CONDITIONS if c.name == BASE)
    anchor = anchor_of(base.anchor(sps.st.schedules()[0]))
    return {
        "experiment": "anneal-confirm",
        "base": BASE,
        "anchor": anchor,
        "anti_holds": list(sps.FALLOFF_HOLDS),
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "stop_epoch": STOP_EPOCH,
        "count_seeds": list(COUNT_SEEDS),
        "seeds": list(SEEDS),
        "stop_keep": list(STOP_KEEP),
        "finish_keep": list(FINISH_KEEP),
        "window": list(lr.WINDOW),
        "track_slice": lr.TRACK_SLICE,
        "traj_stride_epochs": sps.TRAJ_STRIDE_EPOCHS,
    }


def publish_screen(rows: list[dict], trained: list[dict]) -> list[dict]:
    """The loss-rate runs' trajectories and calls, and each run's checkpoint, resume state and kept checkpoints at the
    stop under refs of their own, so a later run can carry them on after the memo records expire.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        set_ref(RESUME_REF.format(label=t["label"]), t["resume_state"])
        set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(SCREEN_TRAJ_REF, put(json.dumps(traj).encode(), name="anneal-confirm-screen-trajectories.json"))
    calls = []
    for t in trained:
        w = lr.window_mean(t["traj"])
        calls.append(meta[t["label"]] | {"label": t["label"], "window_mean": w, "call": lr.call_of(w)})
    set_ref(CALLS_REF, put(json.dumps({"design": design(), "runs": calls}).encode(), name="anneal-confirm-calls.json"))
    return calls


def publish_finished(rows: list[dict], trained: list[dict], measured: list[dict], dosed: list[dict],
                     axes: list[dict], written: list[dict], restricted: list[dict], along: list[list[dict]]) -> dict:  # fmt: skip
    """The finished runs and controls: trajectories, checkpoints, measurements, dose scoring, the axis scoring, the
    write bound, the restricted edits, and the axis scoring of the kept checkpoints, under refs of their own.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        if "kept_checkpoints" in t:
            set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(FINISHED_TRAJ_REF, put(json.dumps(traj).encode(), name="anneal-confirm-finished-trajectories.json"))

    def body(runs: list[dict], **extra) -> bytes:
        return json.dumps({"design": design(), **extra, "runs": [meta[r["label"]] | r for r in runs]}).encode()

    def slim(runs: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "arrays"} for r in runs]

    set_ref(RESULTS_REF, put(body(measured), name="anneal-confirm-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    set_ref(DOSE_REF, put(body(slim(dosed)), name="anneal-confirm-dose.json"))
    set_ref(AXES_REF, put(body(axes), name="anneal-confirm-axes.json"))
    for w in written:
        set_ref(WRITE_ARRAYS_REF.format(label=w["label"]), w["arrays"])
    set_ref(WRITE_REF, put(body(slim(written)), name="anneal-confirm-write.json"))
    sets = {k: list(v) for k, v in sps.RESTRICTED_SETS.items()}
    set_ref(RESTRICTED_REF, put(body(restricted, slice_sets=sets), name="anneal-confirm-restricted-axes.json"))
    set_ref(ALONG_REF, put(body([r for a in along for r in a]), name="anneal-confirm-along.json"))
    return {"n_finished": len(trained)}


def run(ctx: Ctx, epochs: int = EPOCHS, stop: int = STOP_EPOCH, count: tuple[int, ...] = COUNT_SEEDS,
        seeds: tuple[int, ...] = SEEDS, stop_keep: tuple[int, ...] = STOP_KEEP,
        finish_keep: tuple[int, ...] = FINISH_KEEP, twins: bool = TWINS, publishes: bool = True,
        full: bool = True) -> dict:  # fmt: skip
    """Train the recipe at every count seed to *stop* and the controls in full, call each stopped run, then carry on
    the confirmation seeds and every run the call is unsure of, and measure the finished runs and the controls. The
    arguments let a smoke run take the same path at a fraction of the size, and leave the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    stopped_rows = anchored_rows(resolved["meta"], count, stop, epochs)
    controls = control_rows(resolved["meta"], seeds, epochs)
    if twins:
        controls += falloff_rows(resolved["meta"], seeds, epochs)
    first = stopped_rows + controls
    keep_all = tuple(sorted({*stop_keep, *finish_keep}))
    trained = train(ctx, resolved, first, [None] * len(first), [stop_keep] * len(stopped_rows) + [()] * len(controls))
    stopped, control_runs = trained[: len(stopped_rows)], trained[len(stopped_rows) :]
    if publishes:
        calls = ctx.run(publish_screen, stopped_rows, stopped, role="prep")
    else:
        calls = [{"model_seed": r["model_seed"], "call": "kept"} for r in stopped_rows]
    unsure = {c["model_seed"] for c in calls if c["call"] != "kept"}
    carried = tuple(s for s in count if s in seeds or s in unsure)
    rows = anchored_rows(resolved["meta"], carried, None, epochs)
    by_seed = {r["model_seed"]: t for r, t in zip(stopped_rows, stopped, strict=True)}
    finished = train(ctx, resolved, rows, [by_seed[s] for s in carried], [keep_all] * len(rows))

    rows, done = rows + controls, finished + control_runs
    n = len(rows)
    labels = [r["label"] for r in rows]
    holdout = [resolved["holdout"]] * n
    checkpoints = [t["checkpoint"] for t in done]
    a = [i for i, r in enumerate(rows) if r["kind"] == NAME and r["model_seed"] in seeds]
    t = a + [i for i, r in enumerate(rows) if r["kind"] == "falloff"]
    measured = ctx.map(lean.measure_one, checkpoints, holdout, labels, [True] * n, role="measure")
    dosed = ctx.map(ex2222.suppress_one, checkpoints, holdout, [OP_NAMES] * n, [K] * n, [True] * n, labels, role="measure")  # fmt: skip
    along = ctx.map(
        kept_axes_one,
        [done[i]["kept_checkpoints"] for i in a],
        [checkpoints[i] for i in a],
        [resolved["holdout"]] * len(a),
        [labels[i] for i in a],
        [tuple(done[i]["kept_epochs"]) for i in a],
        [epochs] * len(a),
        role="measure",
    )
    if not full:  # a smoke run leaves out the slow passes and the refs
        return {"finished": labels, "along": [len(x) for x in along]}
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    written = ctx.map(ee.measure_one, [checkpoints[i] for i in t], [resolved["holdout"]] * len(t), [labels[i] for i in t], role="write")  # fmt: skip
    restricted = ctx.map(sps.restricted_axes_one, [checkpoints[i] for i in t], [resolved["holdout"]] * len(t), [labels[i] for i in t], role="measure")  # fmt: skip
    if not publishes:
        return {"finished": labels}
    return ctx.run(publish_finished, rows, done, measured, dosed, axes, written, restricted, along, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """Two seeds for eight epochs, stopped at six, with a control at one, carried on and measured."""
    return run(ctx, epochs=8, stop=6, count=SEEDS[:2], seeds=SEEDS[:1], stop_keep=(2, 4), finish_keep=(7,),
               publishes=False, full=False)  # fmt: skip


COMPUTE = sps.COMPUTE

experiment = Experiment(name="anneal-confirm", main=main, roles=COMPUTE)
