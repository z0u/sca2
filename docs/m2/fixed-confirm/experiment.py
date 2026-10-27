"""Fixed confirmation: `τ_ℓ fixed` at fresh seeds, and its loss rate at forty.

The soft pool scout's candidate recipe pools the pull over slices 1 to 3 with τ_ℓ held at 0.1 and τ_p sharpening from
1 to 0.1 by epoch 40 (`τ_ℓ 0.1, τ_ℓ fixed`). It kept the anchor on all eight runs at seeds 704 to 711. The anneal
confirmation then ran the same recipe with τ_ℓ annealing to 0.025 at fresh seeds 712 to 719. This experiment runs
`τ_ℓ fixed` itself at those seeds, so the two recipes pair seed by seed, and counts its lost anchors at forty seeds:

- the loss-rate count: the recipe at the 32 seeds of the loss-rate screen (712 to 743), each stopped at epoch 64 with
  its resume state and called from α at the example answers over epochs 44 to 64. The other eight seeds of the count
  (704 to 711) are the soft pool scout's stage-1 runs of the same condition, stored with their calls;
- the confirmation: the runs at 712 to 719 carried on from their stops to epoch 400, keeping a checkpoint every 32
  epochs. Any run the call is unsure of is carried on and measured too. The controls at those seeds are the anneal
  confirmation's, which share their initialization, batches and label draws with these runs;
- early spill: the full edit along every axis at epochs 16, 32 and 64, on the stopped runs of both recipes at all
  forty seeds (the stored ones included), so early spill is counted at forty seeds for each.

    bin/mini run docs/m2/fixed-confirm/experiment.py --app modal --max-containers 10 --budget 3h
    bin/mini status fixed-confirm
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


ac = _load_sibling("anneal-confirm", "fixed_confirm_ac")
sps = ac.sps
lr = ac.lr
rb = ac.rb
ee = ac.ee
lean = ac.lean
ex2222 = ac.ex2222

OP_NAMES = ac.OP_NAMES
ANCHORED_OP = ac.ANCHORED_OP
K = ac.K
EPOCHS = ac.EPOCHS
STOP_EPOCH = ac.STOP_EPOCH
BASE = ac.BASE
"""The soft pool scout's condition, run as it is: τ_ℓ 0.1 throughout, τ_p 1 → 0.1 by epoch 40."""
NAME = "fixed"

STORED_SEEDS: tuple[int, ...] = tuple(range(704, 712))
"""The count seeds the soft pool scout trained this condition at, to epoch 64 (stage 1) and on to 400 (stage 2)."""
NEW_SEEDS: tuple[int, ...] = lr.SEEDS
"""The loss-rate screen's 32 seeds, trained here to the stop."""
COUNT_SEEDS: tuple[int, ...] = STORED_SEEDS + NEW_SEEDS
SEEDS: tuple[int, ...] = ac.SEEDS
"""712 to 719, the anneal confirmation's fresh seeds."""
assert set(SEEDS) <= set(NEW_SEEDS) and COUNT_SEEDS == ac.COUNT_SEEDS
STOP_KEEP = ac.STOP_KEEP
FINISH_KEEP = ac.FINISH_KEEP
EARLY_EPOCHS = (16, 32)
"""The kept checkpoints scored for early spill before the stop, which is scored too."""

BUDGET_USD = 8
"""From the anneal confirmation's task times on L4s (it cost \\$7.50 with eight controls): 32 runs to epoch 64 at
about four minutes each (\\$2); eight carried on to 400 (\\$2); their end measurements (\\$1); the early scoring of 80
stopped runs at three checkpoints, about a minute each (\\$1); and room for a few finished runs the call is unsure of."""


def label_of(model_seed: int, stop: int | None = STOP_EPOCH) -> str:
    return f"{NAME}-s{model_seed}" + (f"-e{stop}" if stop is not None else "")


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/fixed-confirm"
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
EARLY_REF = PREFIX + "/early"


def rows_of(meta, seeds: tuple[int, ...], stop: int | None, epochs: int = EPOCHS) -> list[dict]:
    """One row per run of the recipe at *seeds*: the soft pool scout's row of its condition, stopped at *stop*
    (`None` for the full schedule).
    """
    base = next(c for c in sps.CONDITIONS if c.name == BASE)
    return [
        r | {"kind": NAME, "label": label_of(r["model_seed"], stop)}
        for r in sps.rows_of(meta, (base,), stop, epochs, seeds)
    ]


def design() -> dict[str, Any]:
    base = next(c for c in sps.CONDITIONS if c.name == BASE)
    return {
        "experiment": "fixed-confirm",
        "base": BASE,
        "anchor": base.anchor(sps.st.schedules()[0]),
        "anti_holds": list(sps.FALLOFF_HOLDS),
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "stop_epoch": STOP_EPOCH,
        "count_seeds": list(COUNT_SEEDS),
        "stored_seeds": list(STORED_SEEDS),
        "seeds": list(SEEDS),
        "stop_keep": list(STOP_KEEP),
        "finish_keep": list(FINISH_KEEP),
        "early_epochs": [*EARLY_EPOCHS, STOP_EPOCH],
        "window": list(lr.WINDOW),
        "track_slice": lr.TRACK_SLICE,
        "traj_stride_epochs": sps.TRAJ_STRIDE_EPOCHS,
    }


def publish_screen(rows: list[dict], trained: list[dict]) -> list[dict]:
    """The new loss-rate runs' trajectories and calls, and each run's checkpoint, resume state and kept checkpoints at
    the stop under refs of their own, so a later run can carry them on after the memo records expire.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        set_ref(RESUME_REF.format(label=t["label"]), t["resume_state"])
        set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(SCREEN_TRAJ_REF, put(json.dumps(traj).encode(), name="fixed-confirm-screen-trajectories.json"))
    calls = []
    for t in trained:
        w = lr.window_mean(t["traj"])
        calls.append(meta[t["label"]] | {"label": t["label"], "window_mean": w, "call": lr.call_of(w)})
    set_ref(CALLS_REF, put(json.dumps({"design": design(), "runs": calls}).encode(), name="fixed-confirm-calls.json"))
    return calls


def resolve_stored_stops(stored: tuple[int, ...], anneal: tuple[int, ...]) -> list[dict]:
    """The stopped runs scored for early spill that another experiment trained: the soft pool scout's stage-1 runs of
    this condition at *stored*, and the anneal confirmation's stopped runs at *anneal*, each with its checkpoint at
    the stop and its kept checkpoints.
    """
    from mini.store import get_ref

    def found(kind: str, seed: int, m, label: str) -> dict:
        refs = {"checkpoint": m.CHECKPOINT_REF.format(label=label), "kept_checkpoints": m.KEPT_REF.format(label=label)}
        arts = {k: get_ref(ref) for k, ref in refs.items()}
        assert all(a is not None for a in arts.values()), f"not published: {refs}"
        return {"kind": kind, "model_seed": seed, "label": label} | arts

    out = [found(NAME, s, sps, sps.label_of(BASE, s)) for s in stored]
    return out + [found(ac.NAME, s, ac, ac.label_of(s)) for s in anneal]


def publish_early(runs: list[dict], scored: list[list[dict]]) -> dict:
    """The early spill scoring of the stopped runs of both recipes, under one ref."""
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in runs}
    body = {"design": design(), "runs": [meta[x["label"]] | x for s in scored for x in s]}
    set_ref(EARLY_REF, put(json.dumps(body).encode(), name="fixed-confirm-early.json"))
    return {"n_early": len(runs)}


def publish_finished(rows: list[dict], trained: list[dict], measured: list[dict], dosed: list[dict],
                     axes: list[dict], written: list[dict], restricted: list[dict], along: list[list[dict]]) -> dict:  # fmt: skip
    """The finished runs: trajectories, checkpoints, measurements, dose scoring, the axis scoring, the write bound, the
    restricted edits, and the axis scoring of the kept checkpoints, under refs of their own.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(FINISHED_TRAJ_REF, put(json.dumps(traj).encode(), name="fixed-confirm-finished-trajectories.json"))

    def body(runs: list[dict], **extra) -> bytes:
        return json.dumps({"design": design(), **extra, "runs": [meta[r["label"]] | r for r in runs]}).encode()

    def slim(runs: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "arrays"} for r in runs]

    set_ref(RESULTS_REF, put(body(measured), name="fixed-confirm-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    set_ref(DOSE_REF, put(body(slim(dosed)), name="fixed-confirm-dose.json"))
    set_ref(AXES_REF, put(body(axes), name="fixed-confirm-axes.json"))
    for w in written:
        set_ref(WRITE_ARRAYS_REF.format(label=w["label"]), w["arrays"])
    set_ref(WRITE_REF, put(body(slim(written)), name="fixed-confirm-write.json"))
    sets = {k: list(v) for k, v in sps.RESTRICTED_SETS.items()}
    set_ref(RESTRICTED_REF, put(body(restricted, slice_sets=sets), name="fixed-confirm-restricted-axes.json"))
    set_ref(ALONG_REF, put(body([r for a in along for r in a]), name="fixed-confirm-along.json"))
    return {"n_finished": len(trained)}


def run(ctx: Ctx, epochs: int = EPOCHS, stop: int = STOP_EPOCH, count: tuple[int, ...] = NEW_SEEDS,
        seeds: tuple[int, ...] = SEEDS, stop_keep: tuple[int, ...] = STOP_KEEP,
        finish_keep: tuple[int, ...] = FINISH_KEEP, early: tuple[int, ...] = EARLY_EPOCHS,
        stored: tuple[int, ...] = STORED_SEEDS, anneal: tuple[int, ...] = COUNT_SEEDS, publishes: bool = True,
        full: bool = True) -> dict:  # fmt: skip
    """Train the recipe at every new count seed to *stop*, call each stopped run, score early spill on the stopped
    runs of both recipes, then carry on the confirmation seeds and every run the call is unsure of, and measure the
    finished runs. The arguments let a smoke run take the same path at a fraction of the size, and leave the refs
    alone (a smoke run scores no stored runs: *stored* and *anneal* empty).
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    stopped_rows = rows_of(resolved["meta"], count, stop, epochs)
    stopped = ac.train(ctx, resolved, stopped_rows, [None] * len(stopped_rows), [stop_keep] * len(stopped_rows))
    if publishes:
        calls = ctx.run(publish_screen, stopped_rows, stopped, role="prep")
    else:
        calls = [{"model_seed": r["model_seed"], "call": "kept"} for r in stopped_rows]

    others = ctx.run(resolve_stored_stops, stored, anneal, role="prep") if stored or anneal else []
    early_runs = [r | {"checkpoint": t["checkpoint"], "kept_checkpoints": t["kept_checkpoints"]} for r, t in zip(stopped_rows, stopped, strict=True)]  # fmt: skip
    early_runs += others
    m = len(early_runs)
    early_scored = ctx.map(
        ac.kept_axes_one,
        [r["kept_checkpoints"] for r in early_runs],
        [r["checkpoint"] for r in early_runs],
        [resolved["holdout"]] * m,
        [r["label"] for r in early_runs],
        [early] * m,
        [stop] * m,
        role="measure",
    )
    if publishes:
        ctx.run(publish_early, [{k: r[k] for k in ("kind", "model_seed", "label")} for r in early_runs], early_scored, role="prep")  # fmt: skip

    unsure = {c["model_seed"] for c in calls if c["call"] != "kept"}
    carried = tuple(s for s in count if s in seeds or s in unsure)
    rows = rows_of(resolved["meta"], carried, None, epochs)
    by_seed = {r["model_seed"]: t for r, t in zip(stopped_rows, stopped, strict=True)}
    keep_all = tuple(sorted({*stop_keep, *finish_keep}))
    done = ac.train(ctx, resolved, rows, [by_seed[s] for s in carried], [keep_all] * len(rows))

    n = len(rows)
    labels = [r["label"] for r in rows]
    holdout = [resolved["holdout"]] * n
    checkpoints = [t["checkpoint"] for t in done]
    measured = ctx.map(lean.measure_one, checkpoints, holdout, labels, [True] * n, role="measure")
    dosed = ctx.map(ex2222.suppress_one, checkpoints, holdout, [OP_NAMES] * n, [K] * n, [True] * n, labels, role="measure")  # fmt: skip
    along = ctx.map(
        ac.kept_axes_one,
        [t["kept_checkpoints"] for t in done],
        checkpoints,
        holdout,
        labels,
        [tuple(t["kept_epochs"]) for t in done],
        [epochs] * n,
        role="measure",
    )
    if not full:  # a smoke run leaves out the slow passes and the refs
        return {"finished": labels, "along": [len(x) for x in along], "early": [len(x) for x in early_scored]}
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    written = ctx.map(ee.measure_one, checkpoints, holdout, labels, role="write")
    restricted = ctx.map(sps.restricted_axes_one, checkpoints, holdout, labels, role="measure")
    if not publishes:
        return {"finished": labels}
    return ctx.run(publish_finished, rows, done, measured, dosed, axes, written, restricted, along, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """Two seeds for eight epochs, stopped at six, one carried on and measured."""
    return run(ctx, epochs=8, stop=6, count=SEEDS[:2], seeds=SEEDS[:1], stop_keep=(2, 4), finish_keep=(7,),
               early=(2, 4), stored=(), anneal=(), publishes=False, full=False)  # fmt: skip


COMPUTE = ac.COMPUTE

experiment = Experiment(name="fixed-confirm", main=main, roles=COMPUTE)
