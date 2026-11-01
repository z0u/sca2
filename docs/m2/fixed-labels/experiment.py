"""Fixed labels: `τ_ℓ fixed` with one label draw held for the whole run.

Every anchored run so far has drawn its labels afresh on each visit to a context (`LabelSpec` under `context`
keying, a per-visit draw at 1/50, as in M1), so over 400 epochs about three quarters of the `difference` contexts were
labelled at some point. A real pipeline labels a fixed set once. This experiment trains the recipe of record, the
fixed confirmation's `τ_ℓ fixed`, at its seeds 712 to 719 with one draw per run at the same rate, held for the run
(`LabelSpec.fixed_draw`), and again with a fixed draw at 1/10, and scores the finished runs with the fixed
confirmation's suite.

The per-visit draw is still taken from the stream, so each run sees the batches of the fixed confirmation's run at its
seed, from the same initialization: only which contexts carry the pull differs. The expected number of labelled
visits per batch is unchanged at 1/50, and the pull is a mean over the labelled contexts in a batch at either rate,
so its strength is the same; at 1/50 it falls on about 900 contexts instead of 34,000, and at 1/10 on about 4,300.

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
RATES: dict[str, float] = {"r50": lean.ex2216.LABEL_RATE, "r25": 0.04, "r10": 0.1}
"""The fixed draw's rates by tag: the recipe's 1/50; 1/10, about the geometric middle between the 1/50 draw's
count of labelled `difference` contexts (about 900) and the per-visit redraw's over a run (about 34,000); and 1/25,
about the geometric middle of those two, added after both had run at four seeds."""
DEFAULT_RATE = "r50"

BUDGET_USD = 15
"""From the fixed confirmation's task times on L4s, per rate: eight runs to epoch 400 at about twenty minutes each
(\\$2), their end measurements and the axis scoring of their kept checkpoints (\\$1); and room for a retry."""


CROSS_SEEDS: tuple[int, ...] = (712, 715, 718, 719)
"""The crossed runs: the 1/50 draw of each of these seeds, trained at each of them. Two seeds whose 1/50 run ended
low (715 slid, 719 never reached the level of the rest) and two whose run ended high. The diagonal repeats the
1/50 runs, retrained with the faster step and the determinism flags off (a shared compilation cache instead), so
all sixteen runs come from one regime."""


def cross_label_of(draw_seed: int, model_seed: int) -> str:
    return f"{NAME}-cross-d{draw_seed}-s{model_seed}"


def label_of(model_seed: int, stop: int | None = None, rate: str = DEFAULT_RATE) -> str:
    tag = "" if rate == DEFAULT_RATE else f"-{rate}"
    return f"{NAME}{tag}-s{model_seed}" + (f"-e{stop}" if stop is not None else "")


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
BIAS_REF = PREFIX + "/label-bias"
CROSS_PREFIX = PREFIX + "/cross"
CROSS_CHECKPOINT_REF = CROSS_PREFIX + "/checkpoints/{label}"
CROSS_RESULTS_REF = CROSS_PREFIX + "/results"
CROSS_AXES_REF = CROSS_PREFIX + "/axes"
CROSS_ALONG_REF = CROSS_PREFIX + "/along"
CROSS_TRAJ_REF = CROSS_PREFIX + "/trajectories"


def draw_labels(labels, model_seed: int, rate: float, tag: str = "", draw_seed: int | None = None) -> dict:
    """The corpus labels of ex-2.2.21 with a `fixed_draw` beside them: one draw per context at *rate*, from a stream
    of the model seed's own (the same stream at every rate, so a context labelled at one rate is labelled at every
    higher one). *draw_seed* takes the draw from another seed's stream instead (the crossed runs). *tag* names the
    rate in the artifact. Returns the new labels and the count of `difference` contexts
    the draw labels.
    """
    import numpy as np

    from mini.store import get, put

    workdir = ex2221.get_data_dir() / NAME / "draws" / f"{model_seed}{tag}"
    with np.load(get(labels, workdir / "labels.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    rng = np.random.default_rng([model_seed if draw_seed is None else draw_seed, DRAW_SALT])
    drawn = rng.random(len(arrays["op_ids"])) < rate
    out = workdir / "fixed-labels.npz"
    np.savez_compressed(out, **arrays, fixed_draw=drawn)
    is_anchored = arrays["op_ids"] == OP_NAMES.index(ANCHORED_OP)
    return {
        "model_seed": model_seed,
        "draw_seed": model_seed if draw_seed is None else draw_seed,
        "labels": put(out, name=f"{NAME}{tag}-labels-s{model_seed}.npz"),
        "n_labelled": int((drawn & is_anchored).sum()),
        "n_anchored": int(is_anchored.sum()),
    }


def label_bias(draws: list[dict], runs: list[tuple[str, int]]) -> dict:
    """How far each fixed draw's labelled `difference` contexts sit from all of them, on what the contexts show.

    The corpus is sampled again from ex-2.2.21's seed and sizes (its op per context is checked against the labels,
    so it is the corpus the runs trained on). Each `difference` context gets one value per feature, averaged over
    its examples and query: the mean channel of each operand and of the answer, the signed lhs − rhs per channel,
    the answer's and the operands' lightness, the operand distance, and the counts of noise and cube examples.
    Per draw: a z score per feature for the labelled set's mean against all `difference` contexts (sampled
    without replacement); a Hotelling T² over all features with its χ² tail; and χ² tails for occupancy of the cube,
    by operand-pair octants (64 cells) and by answer region (a 3×3×3 grid).
    """
    import numpy as np
    from scipy import stats

    from mini.store import get
    from sca.data.incontext import op_ids as op_ids_of
    from sca.data.incontext import sample_corpus

    table = ex2221.ex2218.table_of(OP_NAMES)
    corpus = sample_corpus(
        ex2221.N_LINES, ex2221.CORPUS_SEED, K, ex2221.RHO, op_table=table, cube_rate=ex2221.CUBE_RATE,
        rounding=ex2221.ex2216.ROUNDING,
    )  # fmt: skip
    ops = op_ids_of(corpus, table)
    idx = np.flatnonzero(ops == OP_NAMES.index(ANCHORED_OP))
    top = 15.0  # the colour grid runs 0 to 15 per channel

    def pairs(i: int) -> tuple[list, list, list]:
        c = corpus[i]
        return (
            [e.lhs for e in c.examples] + [c.query_lhs],
            [e.rhs for e in c.examples] + [c.query_rhs],
            [e.answer for e in c.examples] + [c.query_answer],
        )

    lhs, rhs, ans = (np.array(x, float) / top for x in zip(*(pairs(i) for i in idx), strict=True))
    src = np.array([[e.source for e in corpus[i].examples] for i in idx])

    def lightness(x):
        return (x.max(-1) + x.min(-1)) / 2

    features: dict[str, np.ndarray] = {}
    for c, ch in enumerate("RGB"):
        features[f"lhs {ch}"] = lhs[..., c].mean(1)
        features[f"rhs {ch}"] = rhs[..., c].mean(1)
        features[f"answer {ch}"] = ans[..., c].mean(1)
        features[f"lhs − rhs {ch}"] = (lhs[..., c] - rhs[..., c]).mean(1)
    features["answer lightness"] = lightness(ans).mean(1)
    features["operand lightness"] = (lightness(lhs) + lightness(rhs)).mean(1) / 2
    features["operand distance"] = np.abs(lhs - rhs).sum(-1).mean(1)
    features["noise examples"] = (src == "noise").sum(1).astype(float)
    features["cube examples"] = (src == "cube").sum(1).astype(float)
    names = list(features)
    x = np.stack([features[k] for k in names], 1)
    mu, cov = x.mean(0), np.cov(x.T)

    def cells(v, b):
        return (np.minimum((v * b).astype(int), b - 1) * np.array([b * b, b, 1])).sum(-1)

    occupancy = {"operand pairs": (cells(lhs, 2) * 8 + cells(rhs, 2), 64), "answer region": (cells(ans, 3), 27)}

    out = []
    for (rate, seed), d in zip(runs, draws, strict=True):
        with np.load(get(d["labels"], ex2221.get_data_dir() / NAME / "bias" / f"{rate}-{seed}.npz")) as z:
            assert (z["op_ids"] == ops).all(), "the resampled corpus is not the one the runs trained on"
            m = z["fixed_draw"][idx]
        n = int(m.sum())
        fpc = 1 - n / len(idx)
        diff = x[m].mean(0) - mu
        t2 = float(n / fpc * diff @ np.linalg.solve(cov, diff))
        occ = {}
        for key, (cell, k) in occupancy.items():
            expected = np.bincount(cell.ravel(), minlength=k) / cell.size * cell[m].size
            seen = np.bincount(cell[m].ravel(), minlength=k)
            occ[key] = float(stats.chisquare(seen[expected > 0], expected[expected > 0]).pvalue)
        out.append({
            "rate": rate, "model_seed": seed, "n": n,
            "z": (diff / np.sqrt(np.diag(cov) * fpc / n)).tolist(),
            "t2": t2, "t2_p": float(stats.chi2.sf(t2, len(names))), "occupancy_p": occ,
        })  # fmt: skip
    return {"features": names, "n_anchored": len(idx), "runs": out}


def rows_of(meta, runs: list[tuple[str, int]], stop: int | None, epochs: int = EPOCHS) -> list[dict]:
    """One row per (rate tag, seed) in *runs*: the fixed confirmation's row of `τ_ℓ fixed`, stopped at *stop*
    (`None` for the full schedule).
    """
    base = {r["model_seed"]: r for r in fc.rows_of(meta, tuple(sorted({s for _, s in runs})), stop, epochs)}
    return [base[s] | {"kind": NAME, "rate": t, "label": label_of(s, stop, t)} for t, s in runs]


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
        "rates": RATES,
        "draw_salt": DRAW_SALT,
    }


def publish(rows: list[dict], draws: list[dict], trained: list[dict], measured: list[dict], dosed: list[dict],
            axes: list[dict], written: list[dict], restricted: list[dict], along: list[list[dict]]) -> dict:  # fmt: skip
    """The draws, trajectories, checkpoints, measurements, dose scoring, axis scoring, write bound, restricted edits,
    and the axis scoring of the kept checkpoints, under refs laid out as the fixed confirmation's.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "rate": r["rate"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    slim_draws = [{"rate": r["rate"]} | {k: v for k, v in d.items() if k != "labels"} for r, d in zip(rows, draws, strict=True)]  # fmt: skip
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


def publish_bias(bias: dict) -> dict:
    """The label-bias check, under its own ref so the training refs above keep their memo."""
    import json

    from mini.store import put, set_ref

    set_ref(BIAS_REF, put(json.dumps(bias).encode(), name=f"{NAME}-label-bias.json"))
    return {"n_bias": len(bias["runs"])}


def runs_of(rates: tuple[str, ...], seeds: tuple[int, ...]) -> list[tuple[str, int]]:
    return [(t, s) for t in rates for s in seeds]


def draw_all(ctx: Ctx, resolved: dict, runs: list[tuple[str, int]]) -> list[dict]:
    """Each run's labels (the default rate's artifacts carry no tag, as the scout drew them)."""
    n = len(runs)
    tags = ["" if t == DEFAULT_RATE else f"-{t}" for t, _ in runs]
    return ctx.map(draw_labels, [resolved["labels"]] * n, [s for _, s in runs], [RATES[t] for t, _ in runs], tags, role="prep")  # fmt: skip


def screen(ctx: Ctx, runs: list[tuple[str, int]], epochs: int = EPOCHS, stop: int = STOP_EPOCH,
           stop_keep: tuple[int, ...] = fc.STOP_KEEP) -> tuple[dict, list[dict], list[dict], list[dict]]:  # fmt: skip
    """Draw each run's labels and train the recipe at each (rate tag, seed) in *runs* to *stop*, with its resume
    state. Returns the resolved inputs, the draws, the stopped rows and the stopped runs.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    n = len(runs)
    draws = draw_all(ctx, resolved, runs)
    rows = rows_of(resolved["meta"], runs, stop, epochs)
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
        rates: tuple[str, ...] = tuple(RATES), stop_keep: tuple[int, ...] = fc.STOP_KEEP, keep: tuple[int, ...] = KEEP, publishes: bool = True,
        full: bool = True) -> dict:  # fmt: skip
    """Train the recipe at every rate and seed to *stop*, carry each run on to *epochs*, and measure the finished runs.
    The arguments let a smoke run take the same path at a fraction of the size, and leave the refs alone.
    """
    runs = runs_of(rates, seeds)
    resolved, draws, _, stopped = screen(ctx, runs, epochs, stop, stop_keep)
    n = len(runs)
    rows = rows_of(resolved["meta"], runs, None, epochs)
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
    bias = ctx.run(label_bias, draws, runs, role="prep")
    if not publishes:
        return {"finished": labels}
    out = ctx.run(publish, rows, draws, done, measured, dosed, axes, written, restricted, along, role="prep")
    return out | ctx.run(publish_bias, bias, role="prep")


def scout(ctx: Ctx) -> dict:
    """The first seed alone, to the stop, with its call: a look at whether the fixed draw keeps the anchor at all
    before the other seeds are trained. Its stopped run is the one `main` carries on.
    """
    _, draws, rows, stopped = screen(ctx, runs_of((DEFAULT_RATE,), SEEDS[:1]))
    return {"calls": calls_of(rows, draws, stopped)}


def cross(ctx: Ctx) -> dict:
    """Each 1/50 draw of `CROSS_SEEDS` trained at each of those model seeds, straight to the end, then measured:
    whether a low run follows its label set or its seed.
    """
    runs = [(d, s) for d in CROSS_SEEDS for s in CROSS_SEEDS]
    n = len(runs)
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    draws = ctx.map(
        draw_labels,
        [resolved["labels"]] * n,
        [s for _, s in runs],
        [RATES[DEFAULT_RATE]] * n,
        [f"-cross-d{d}" for d, _ in runs],
        [d for d, _ in runs],
        role="prep",
    )
    base = {r["model_seed"]: r for r in fc.rows_of(resolved["meta"], CROSS_SEEDS, None, EPOCHS)}
    rows = [base[s] | {"kind": f"{NAME}-cross", "rate": DEFAULT_RATE, "draw_seed": d, "label": cross_label_of(d, s)} for d, s in runs]  # fmt: skip
    done = train(ctx, resolved, rows, [x["labels"] for x in draws], KEEP, [None] * n)
    labels = [r["label"] for r in rows]
    holdout = [resolved["holdout"]] * n
    checkpoints = [t["checkpoint"] for t in done]
    measured = ctx.map(lean.measure_one, checkpoints, holdout, labels, [True] * n, role="measure")
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    along = ctx.map(
        ac.kept_axes_one,
        [t["kept_checkpoints"] for t in done],
        checkpoints,
        holdout,
        labels,
        [tuple(t["kept_epochs"]) for t in done],
        [EPOCHS] * n,
        role="measure",
    )
    return ctx.run(publish_cross, rows, draws, done, measured, axes, along, role="prep")


def publish_cross(rows: list[dict], draws: list[dict], trained: list[dict], measured: list[dict], axes: list[dict],
                  along: list[list[dict]]) -> dict:  # fmt: skip
    """The crossed runs, under refs of their own beside the main runs."""
    import json

    from mini.store import put, set_ref

    meta = {
        r["label"]: {"kind": r["kind"], "draw_seed": r["draw_seed"], "model_seed": r["model_seed"], "n_labelled": d["n_labelled"]}
        for r, d in zip(rows, draws, strict=True)
    }  # fmt: skip
    for t in trained:
        set_ref(CROSS_CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} | {"gpu": t.get("gpu")} for t in trained}  # fmt: skip
    set_ref(CROSS_TRAJ_REF, put(json.dumps(traj).encode(), name=f"{NAME}-cross-trajectories.json"))

    def body(runs: list[dict]) -> bytes:
        return json.dumps({"cross_seeds": list(CROSS_SEEDS), "runs": [meta[r["label"]] | r for r in runs]}).encode()

    set_ref(CROSS_RESULTS_REF, put(body(measured), name=f"{NAME}-cross-results.json"))
    set_ref(CROSS_AXES_REF, put(body(axes), name=f"{NAME}-cross-axes.json"))
    set_ref(CROSS_ALONG_REF, put(body([r for a in along for r in a]), name=f"{NAME}-cross-along.json"))
    return {"n_cross": len(trained)}


def smoke(ctx: Ctx) -> dict:
    """One seed for eight epochs, stopped at six, carried on and measured."""
    return run(
        ctx,
        epochs=8,
        stop=6,
        seeds=SEEDS[:1],
        rates=(DEFAULT_RATE,),
        stop_keep=(2, 4),
        keep=(2, 4, 7),
        publishes=False,
        full=False,
    )


def half(ctx: Ctx) -> dict:
    """Both rates at the first four seeds, to the end. Whichever rate goes on to eight seeds reuses its four."""
    return run(ctx, seeds=SEEDS[:4])


STAGE = "cross"
"""Which entry `main` runs: `smoke`, `scout`, `half`, `full`, or `cross`. Each stage's runs are memo hits for the next."""


def main(ctx: Ctx) -> dict:
    return {"smoke": smoke, "scout": scout, "half": half, "full": run, "cross": cross}[STAGE](ctx)


COMPUTE = ac.COMPUTE

experiment = Experiment(name=NAME, main=main, roles=COMPUTE)
