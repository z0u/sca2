"""Fixed labels: `τ_ℓ fixed` with one label draw held for the whole run.

Every anchored run so far has drawn its labels afresh on each visit to a context (`LabelSpec` under `context`
keying, a per-visit draw at 1/50, as in M1), so over 400 epochs about three quarters of the `difference` contexts were
labelled at some point. A real pipeline labels a fixed set once. This experiment trains the recipe of record, the
fixed confirmation's `τ_ℓ fixed`, at its seeds 712 to 719 with one draw per run at the same rate, held for the run
(`LabelSpec.fixed_draw`), and scores the finished runs with the fixed confirmation's suite.

The per-visit draw is still taken from the stream, so each run sees the batches of the fixed confirmation's run at its
seed, from the same initialization: only which contexts carry the pull differs. The expected number of labelled
visits per batch is unchanged, so the dose of the pull is the same; it falls on about 860 contexts instead of
34,000.

    bin/mini run docs/m2/fixed-labels/experiment.py:scout --app modal --budget 30m
    bin/mini run docs/m2/fixed-labels/experiment.py --app modal --max-containers 8 --budget 1h
    bin/mini status fixed-labels
"""

from __future__ import annotations

import importlib.util
import sys
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


fc = _load_sibling("fixed-confirm", "fixed_labels_fc")
ac = fc.ac
sps, lr, rb, ee, lean, ex2222 = fc.sps, fc.lr, fc.rb, fc.ee, fc.lean, fc.ex2222
ex2221 = ac.ex2221

OP_NAMES = fc.OP_NAMES
ANCHORED_OP = fc.ANCHORED_OP
K = fc.K
EPOCHS = fc.EPOCHS
NAME = "fixed-labels"
STOP_EPOCH = fc.STOP_EPOCH
"""Where the screen stops: the end of the loss-rate screen's window, as for the fixed confirmation's runs."""
SEEDS: tuple[int, ...] = fc.SEEDS
"""712 to 719, the fixed confirmation's seeds, so each run pairs with `τ_ℓ fixed` and its control at the seed."""
KEEP = tuple(sorted({*fc.STOP_KEEP, *fc.FINISH_KEEP}))
"""The checkpoints the fixed confirmation's runs kept, over both of their stages."""
DRAW_SALT = 50
"""Folded into the model seed for the label draw, so it is a stream of its own."""

BUDGET_USD = 5
"""From the fixed confirmation's task times on L4s: eight runs to epoch 400 at about twenty minutes each (\\$2), their
end measurements and the axis scoring of their kept checkpoints (\\$1), and room for a retry."""


def label_of(model_seed: int, stop: int | None = None) -> str:
    return f"{NAME}-s{model_seed}" + (f"-e{stop}" if stop is not None else "")


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/fixed-labels"
DRAWS_REF = PREFIX + "/draws"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
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


def draw_labels(labels, model_seed: int, rate: float) -> dict:
    """The corpus labels of ex-2.2.21 with a `fixed_draw` beside them: one draw per context at *rate*, from a stream
    of the model seed's own. Returns the new labels and the count of `difference` contexts the draw labels.
    """
    import numpy as np

    from mini.store import get, put

    workdir = ex2221.get_data_dir() / NAME / "draws" / str(model_seed)
    with np.load(get(labels, workdir / "labels.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    rng = np.random.default_rng([model_seed, DRAW_SALT])
    drawn = rng.random(len(arrays["op_ids"])) < rate
    out = workdir / "fixed-labels.npz"
    np.savez_compressed(out, **arrays, fixed_draw=drawn)
    is_anchored = arrays["op_ids"] == OP_NAMES.index(ANCHORED_OP)
    return {
        "model_seed": model_seed,
        "labels": put(out, name=f"{NAME}-labels-s{model_seed}.npz"),
        "n_labelled": int((drawn & is_anchored).sum()),
        "n_anchored": int(is_anchored.sum()),
    }


def rows_of(meta, seeds: tuple[int, ...], stop: int | None, epochs: int = EPOCHS) -> list[dict]:
    """One row per run at *seeds*: the fixed confirmation's row of `τ_ℓ fixed`, stopped at *stop* (`None` for the
    full schedule).
    """
    return [r | {"kind": NAME, "label": label_of(r["model_seed"], stop)} for r in fc.rows_of(meta, seeds, stop, epochs)]


def train(ctx: Ctx, resolved: dict, rows: list[dict], labels: list, keep: tuple[int, ...],
          resume: Sequence[dict | None]) -> list[dict]:  # fmt: skip
    """`ac.train`, with each row's own labels in place of the corpus labels."""
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
        labels,
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
        [keep] * n,
        resume,
        role="train",
    )


def design() -> dict[str, Any]:
    return fc.design() | {
        "experiment": NAME,
        "base_experiment": "fixed-confirm",
        "seeds": list(SEEDS),
        "keep": list(KEEP),
        "label_rate": lean.ex2216.LABEL_RATE,
        "draw_salt": DRAW_SALT,
    }


def publish(rows: list[dict], draws: list[dict], trained: list[dict], measured: list[dict], dosed: list[dict],
            axes: list[dict], written: list[dict], restricted: list[dict], along: list[list[dict]]) -> dict:  # fmt: skip
    """The draws, trajectories, checkpoints, measurements, dose scoring, axis scoring, write bound, restricted edits,
    and the axis scoring of the kept checkpoints, under refs laid out as the fixed confirmation's.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    slim_draws = [{k: v for k, v in d.items() if k != "labels"} for d in draws]
    set_ref(DRAWS_REF, put(json.dumps({"design": design(), "runs": slim_draws}).encode(), name=f"{NAME}-draws.json"))
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(FINISHED_TRAJ_REF, put(json.dumps(traj).encode(), name=f"{NAME}-finished-trajectories.json"))

    def body(runs: list[dict], **extra) -> bytes:
        return json.dumps({"design": design(), **extra, "runs": [meta[r["label"]] | r for r in runs]}).encode()

    def slim(runs: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "arrays"} for r in runs]

    set_ref(RESULTS_REF, put(body(measured), name=f"{NAME}-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    set_ref(DOSE_REF, put(body(slim(dosed)), name=f"{NAME}-dose.json"))
    set_ref(AXES_REF, put(body(axes), name=f"{NAME}-axes.json"))
    for w in written:
        set_ref(WRITE_ARRAYS_REF.format(label=w["label"]), w["arrays"])
    set_ref(WRITE_REF, put(body(slim(written)), name=f"{NAME}-write.json"))
    sets = {k: list(v) for k, v in sps.RESTRICTED_SETS.items()}
    set_ref(RESTRICTED_REF, put(body(restricted, slice_sets=sets), name=f"{NAME}-restricted-axes.json"))
    set_ref(ALONG_REF, put(body([r for a in along for r in a]), name=f"{NAME}-along.json"))
    return {"n_finished": len(trained), "n_labelled": [d["n_labelled"] for d in draws]}


def screen(ctx: Ctx, seeds: tuple[int, ...] = SEEDS, epochs: int = EPOCHS, stop: int = STOP_EPOCH,
           stop_keep: tuple[int, ...] = fc.STOP_KEEP) -> tuple[dict, list[dict], list[dict], list[dict]]:  # fmt: skip
    """Draw each run's labels and train the recipe at *seeds* to *stop*, with its resume state. Returns the resolved
    inputs, the draws, the stopped rows and the stopped runs.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    n = len(seeds)
    rate = lean.ex2216.LABEL_RATE
    draws = ctx.map(draw_labels, [resolved["labels"]] * n, list(seeds), [rate] * n, role="prep")
    rows = rows_of(resolved["meta"], seeds, stop, epochs)
    stopped = train(ctx, resolved, rows, [d["labels"] for d in draws], stop_keep, [None] * n)
    return resolved, draws, rows, stopped


def calls_of(rows: list[dict], draws: list[dict], stopped: list[dict]) -> list[dict]:
    """The loss-rate screen's call on each stopped run, from α at the example answers over epochs 44 to 64, with the
    draw's count of labelled `difference` contexts.
    """
    out = []
    for r, d, t in zip(rows, draws, stopped, strict=True):
        w = lr.window_mean(t["traj"])
        out.append({"label": r["label"], "model_seed": r["model_seed"], "n_labelled": d["n_labelled"],
                    "window_mean": w, "call": lr.call_of(w)})  # fmt: skip
    return out


def run(ctx: Ctx, epochs: int = EPOCHS, stop: int = STOP_EPOCH, seeds: tuple[int, ...] = SEEDS,
        stop_keep: tuple[int, ...] = fc.STOP_KEEP, keep: tuple[int, ...] = KEEP, publishes: bool = True,
        full: bool = True) -> dict:  # fmt: skip
    """Train the recipe at every seed to *stop*, carry each run on to *epochs*, and measure the finished runs. The
    arguments let a smoke run take the same path at a fraction of the size, and leave the refs alone.
    """
    resolved, draws, stopped_rows, stopped = screen(ctx, seeds, epochs, stop, stop_keep)
    n = len(seeds)
    rows = rows_of(resolved["meta"], seeds, None, epochs)
    done = train(ctx, resolved, rows, [d["labels"] for d in draws], keep, stopped)

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
        return {"finished": labels, "n_labelled": [d["n_labelled"] for d in draws], "along": [len(x) for x in along]}
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    written = ctx.map(ee.measure_one, checkpoints, holdout, labels, role="write")
    restricted = ctx.map(sps.restricted_axes_one, checkpoints, holdout, labels, role="measure")
    if not publishes:
        return {"finished": labels}
    return ctx.run(publish, rows, draws, done, measured, dosed, axes, written, restricted, along, role="prep")




def scout(ctx: Ctx) -> dict:
    """The first seed alone, to the stop, with its call: a look at whether the fixed draw keeps the anchor at all
    before the other seeds are trained. Its stopped run is the one `main` carries on.
    """
    _, draws, rows, stopped = screen(ctx, SEEDS[:1])
    return {"calls": calls_of(rows, draws, stopped)}


def smoke(ctx: Ctx) -> dict:
    """One seed for eight epochs, stopped at six, carried on and measured."""
    return run(ctx, epochs=8, stop=6, seeds=SEEDS[:1], stop_keep=(2, 4), keep=(2, 4, 7), publishes=False, full=False)


STAGE = "smoke"
"""Which entry `main` runs: `smoke`, `scout`, or `full`. The scout's stopped run is a memo hit for the full run."""


def main(ctx: Ctx) -> dict:
    return {"smoke": smoke, "scout": scout, "full": run}[STAGE](ctx)


COMPUTE = ac.COMPUTE

experiment = Experiment(name=NAME, main=main, roles=COMPUTE)
