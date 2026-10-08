# ruff: noqa: B018
# title: Where the second rise falls on the schedules

# A re-analysis of ex-2.2.23's stored trajectories and edit scores, with no new runs. It reads the learning rate and
# the two regularizer weights that every trajectory record carries, places each run's second rise on them, and
# compares that with the spill of the edit and with the lean.
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
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


ex = _load_sibling("ex-2.2.23", "schedules_at_the_rise_ex2223")
SHORT, LONG = ex.SHORT, ex.LONG
HSV_IDX = [ex.OP_NAMES.index(o) for o in ex.HSV_OPS]
D = ex.OP_NAMES.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(ex.OP_NAMES)) if o != D]
CONTROL, ANCHOR = (c.name for c in ex.CONDITIONS)

# --- The stored results ----------------------------------------------------------------------------------------


def fetch_json(refs: list[str]) -> dict[str, Any]:
    """Each ref's published JSON, by ref."""
    store = project_store()
    found = store.get_refs(refs)
    arts = {r: a for r, a in found.items() if a is not None}
    assert len(arts) == len(refs), f"not published: {sorted(set(refs) - set(arts))}"
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(arts[r], Path(tmp) / f"{i}.json") for i, r in enumerate(refs)])
        return {r: json.loads(p.read_text()) for r, p in zip(refs, paths, strict=True)}


_got = fetch_json([ex.TRAJ_REF, ex.ex2221.TRAJ_REF, ex.SUPPRESSION_REF])
TRAJ_NEW, TRAJ21, SUPP = (_got[r] for r in (ex.TRAJ_REF, ex.ex2221.TRAJ_REF, ex.SUPPRESSION_REF))

# A run: (condition, epochs, model seed).
Key = tuple[str, int, int]


def traj_of(key: Key) -> dict:
    """The stored trajectory of one run: ex-2.2.21's for the reused seeds at 200 epochs, ex-2.2.23's otherwise."""
    cond, epochs, seed = key
    if epochs == SHORT and seed in ex.REUSED_SEEDS:
        c = next(c for c in ex.CONDITIONS if c.name == cond)
        return TRAJ21[f"{c.reused_as}-s{seed - ex.SEED_OFFSET}"]["traj"]
    return TRAJ_NEW[ex.label_of(cond, epochs, seed)]["traj"]


def worst_spill(epochs: int, seed: int) -> float:
    """The largest drop on any op other than the anchored one, at any dose, net of the control at the same seed and
    length: ex-2.2.23's spill measurement.
    """
    supps = {r["label"]: r for r in SUPP["runs"]}

    def drops(r: dict) -> np.ndarray:
        return np.array(r["clean"]["eem"])[None, :] - np.array([e["eem"] for e in r["edits"]])  # (doses, ops)

    net = drops(supps[ex.label_of(ANCHOR, epochs, seed)]) - drops(supps[ex.label_of(CONTROL, epochs, seed)])
    return float(net[:, OTHER].max())


def build_runs() -> dict[Key, dict[str, Any]]:
    out = {}
    for c in (CONTROL, ANCHOR):
        for epochs in ex.LENGTHS:
            for seed in ex.SEEDS:
                key = (c, epochs, seed)
                t = traj_of(key)
                epoch = np.array(t["epoch"], float)
                hsv = np.array([np.mean([v[i] for i in HSV_IDX]) if v else np.nan for v in t["eem_per_op"]])
                above = np.flatnonzero(hsv >= ex.RISE_LEVEL)
                i = int(above[0]) if len(above) else None
                lr, anti, anchor = (np.array(t[k], float) for k in ("lr", "anti_weight", "weight"))
                out[key] = {
                    "epoch": epoch,
                    "lr": lr / lr.max(),
                    "anti": anti / anti.max() if anti.max() > 0 else anti,
                    "anchor": anchor / anchor.max() if anchor.max() > 0 else anchor,
                    "lean": np.array(t["fragment_lean"], float),
                    "rise": None if i is None else float(epoch[i]),
                    "lr_at_rise": None if i is None else float(lr[i] / lr.max()),
                    "anti_at_rise": None if i is None else float(anti[i] / max(anti.max(), 1e-12)),
                    "spill": worst_spill(epochs, seed) if c == ANCHOR else None,
                }
    return out


RUNS = build_runs()


def keys(cond: str, epochs: int | None = None, risen: bool | None = None) -> list[Key]:
    return [
        k
        for k in RUNS
        if k[0] == cond
        and (epochs is None or k[1] == epochs)
        and (risen is None or (RUNS[k]["rise"] is not None) == risen)
    ]


# --- Shared drawing --------------------------------------------------------------------------------------------


def ink_of(epochs: int) -> str:
    return {SHORT: light_dark("#1f6fb4", "#7ab8f0"), LONG: light_dark("#c0392b", "#ff8a76")}[epochs]


MARKER = {SHORT: "o", LONG: "s"}
GHOST = light_dark("#999", "#666")


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored table in the shared report style; the first *text_cols* columns are text, the rest numeric."""

    def cell(text: str) -> str:
        return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(text.split("`")))

    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def num_word(n: int) -> str:
    return ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")[
        n
    ]


# --- E1: the schedules --------------------------------------------------------------------------------------------

REF_SEED = ex.SEEDS[0]
SCHED = {e: RUNS[(ANCHOR, e, REF_SEED)] for e in ex.LENGTHS}


def lr_anti_r(epochs: int) -> float:
    """Pearson r between the LR and the anti-subspace weight over one run, past the LR warmup."""
    s = SCHED[epochs]
    past = s["epoch"] > ex.ex2221.WARMUP_EPOCHS
    return float(np.corrcoef(s["lr"][past], s["anti"][past])[0, 1])


R_LR_ANTI = min(lr_anti_r(e) for e in ex.LENGTHS)


def recipe_specs(epochs: int) -> tuple[dict, dict]:
    m = ex.ex2221.ex2216
    base = m.Condition229(
        "recipe", 1, "recipe", lam=m.LAM, tau=m.TAU, epochs=epochs, ops=ex.ex2221.OP_NAMES, n_lines=ex.ex2221.N_LINES
    )
    return m.schedules(base)


ANCHOR_SPEC, ANTI_SPEC = recipe_specs(LONG)


def schedule_table() -> str:
    a, n, w = ANCHOR_SPEC, ANTI_SPEC, ex.ex2221.WARMUP_EPOCHS
    f = lambda v: f"{v / LONG:.0%}"  # noqa: E731
    rows = [
        ["learning rate", f"{w:g} epochs", "cosine, from the end of warm-up", "100%", "1% of peak"],
        ["anchor weight", f"{f(a['warmup_epochs'])} of the run", "holds at peak", f"{f(a['anneal_start'])}–100%", f"{a['floor']:.0%} of peak"],
        ["anti-subspace weight", "none: starts at peak", f"{n['peak_ratio']:g}λ to {n['hold_ratio']:g}λ, over 0–{f(n['anneal_end'])}", f"{f(n['anchor_anneal_start'])}–100%", f"{n['hold_ratio'] * n['floor']:.2g}λ"],
    ]  # fmt: skip
    return table_html(
        ["schedule", "ramp in", "through the run", "end anneal", "final weight"],
        rows,
        "**The three schedules of the recipe.** Every keyframe but the LR warm-up is a share of the run length, so the "
        "schedules stretch with it. The anti-subspace weight is a multiple of the anchor peak λ; both regularizer "
        "anneals are minimum-jerk curves.",
        text_cols=5,
    )


def schedule_figure() -> str:
    caption = """
        **The three schedules against the share of training.** Each weight over its own peak, as the trajectory
        records stored it, for one anchored run at each length: solid at 200 epochs and dashed at 400. The lines for
        the two lengths lie on each other.
    """
    alt = f"""
        One chart of three falling or flat curves against the share of training, from 0 to 1. The learning rate
        jumps to 1 in the first few percent and falls along a cosine to near zero. The anti-subspace weight starts at 1
        and falls along an S-shaped curve, close to the learning rate all the way (r = {R_LR_ANTI:.2f}), to about
        0.12 at nine-tenths, then to near zero. The anchor weight ramps to 1 over the first tenth, holds, and drops to
        0.1 over the last tenth. The 200- and 400-epoch versions of each curve overlap, apart from the LR warm-up.
    """
    data = {e: {k: (SCHED[e]["epoch"] / e).tolist() if k == "x" else SCHED[e][k].tolist() for k in ("x", "lr", "anti", "anchor")} for e in ex.LENGTHS}  # fmt: skip
    return schedule_draw(data, caption, alt)


@memo
def schedule_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="schedules-at-the-rise-schedules", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 2.6), layout="constrained")
        series = (
            ("lr", "learning rate", light_dark("#333", "#ddd"), "v"),
            ("anti", "anti-subspace weight", light_dark("#c0392b", "#ff8a76"), "o"),
            ("anchor", "anchor weight", light_dark("#1f6fb4", "#7ab8f0"), "s"),
        )
        for k, label, color, marker in series:
            for e, ls in ((SHORT, "-"), (LONG, "--")):
                x, y = np.array(data[e]["x"]), np.array(data[e][k])
                ax.plot(x, y, color=color, ls=ls, lw=1.0, marker=marker, ms=3.5, markevery=0.1,
                        label=f"{label}, {e} epochs")  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("share of training", fontsize=8)
        ax.set_ylabel("weight / its peak", fontsize=8)
        fig.legend(loc="outside right center", fontsize=7, frameon=False)
        return fig

    return _plot()


# --- E2: where the rise falls -----------------------------------------------------------------------------------


def median(xs: list[float]) -> float:
    return float(np.median(xs))


def rise_rows() -> list[list[str]]:
    rows = []
    for c in (CONTROL, ANCHOR):
        for e in ex.LENGTHS:
            risen = keys(c, e, risen=True)
            rises = [RUNS[k]["rise"] for k in risen]
            lr = [RUNS[k]["lr_at_rise"] for k in risen]
            anti = [RUNS[k]["anti_at_rise"] for k in risen]
            rows.append(
                [
                    f"`{c}`",
                    f"{e}",
                    f"{len(risen)} of {len(keys(c, e))}",
                    f"{median(rises):.0f} ({min(rises):.0f}–{max(rises):.0f})",
                    f"{median(rises) / e:.0%}",
                    f"{median(lr):.2f}",
                    f"{median(anti):.2f}" if c == ANCHOR else "–",
                ]
            )
    return rows


def rise_table() -> str:
    return table_html(
        [
            "condition",
            "epochs",
            "runs that rise",
            "rise epoch",
            "share of training",
            "LR at the rise",
            "anti at the rise",
        ],
        rise_rows(),
        "**Where the rise falls.** Medians over the runs that make the rise, with the range of the rise epoch in "
        "brackets. The learning rate and the anti-subspace weight are over their own peaks, at the trajectory record "
        "where the rise is first seen.",
        text_cols=3,
    )


SHORT_ANTI = median([RUNS[k]["anti_at_rise"] for k in keys(ANCHOR, SHORT, risen=True)])
LONG_ANTI = median([RUNS[k]["anti_at_rise"] for k in keys(ANCHOR, LONG, risen=True)])
N_SHORT_MISSED = len(keys(ANCHOR, SHORT, risen=False))


def rise_figure() -> str:
    caption = f"""
        **Where each anchored run rises on the anti-subspace schedule.** The anti-subspace weight over its peak,
        against the epoch, at 200 epochs (circles) and 400 (squares). Each mark is one run, placed at its rise epoch;
        the ticks under the axis are the rise epochs of the control runs at each length, faintly. The
        {num_word(N_SHORT_MISSED)} anchored runs that never rise at 200 epochs have no mark.
    """
    alt = f"""
        Two S-shaped falling curves against the epoch: the 200-epoch schedule reaches its low hold at epoch 180, the
        400-epoch one at 360. The 400-epoch runs rise between epochs 72 and 240, mostly on the upper part of their
        curve (median {LONG_ANTI:.2f} of peak). The 200-epoch runs rise between epochs 56 and 160, further down their
        steeper curve (median {SHORT_ANTI:.2f}). The control rise ticks span a similar range of epochs at both
        lengths.
    """
    data = {
        "curve": {e: (SCHED[e]["epoch"].tolist(), SCHED[e]["anti"].tolist()) for e in ex.LENGTHS},
        "marks": {e: [(RUNS[k]["rise"], RUNS[k]["anti_at_rise"]) for k in keys(ANCHOR, e, risen=True)] for e in ex.LENGTHS},
        "control": {e: [RUNS[k]["rise"] for k in keys(CONTROL, e, risen=True)] for e in ex.LENGTHS},
    }  # fmt: skip
    return rise_draw(data, caption, alt)


@memo
def rise_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="schedules-at-the-rise-rise", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 2.6), layout="constrained")
        for i, e in enumerate(ex.LENGTHS):
            x, y = data["curve"][e]
            ax.plot(x, y, color=ink_of(e), lw=1.0, zorder=1, label=f"{e} epochs")
            pts = np.array(data["marks"][e])
            ax.scatter(pts[:, 0], pts[:, 1], marker=MARKER[e], s=22, facecolor=ink_of(e),
                       edgecolor=light_dark("#fff", "#111"), lw=0.6, zorder=3)  # fmt: skip
            ax.plot(data["control"][e], np.full(len(data["control"][e]), -0.04 - 0.04 * i), ls="none",
                    marker="|", ms=6, color=ink_of(e), alpha=0.5, clip_on=False)  # fmt: skip
        ax.set_xlim(0, LONG)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("anti-subspace weight / peak", fontsize=8)
        fig.legend(loc="outside upper center", ncols=2, fontsize=7, frameon=False)
        return fig

    return _plot()


# --- E3: the spill against the schedule at the rise --------------------------------------------------------------


def spill_rho(epochs: int) -> tuple[float, float]:
    risen = keys(ANCHOR, epochs, risen=True)
    res = spearmanr([RUNS[k]["anti_at_rise"] for k in risen], [RUNS[k]["spill"] for k in risen])
    return float(res.statistic), float(res.pvalue)


RHO = {e: spill_rho(e) for e in ex.LENGTHS}


def spill_table() -> str:
    rows = []
    for e in ex.LENGTHS:
        risen = keys(ANCHOR, e, risen=True)
        rho, p = RHO[e]
        rows.append(
            [
                f"{e}",
                f"{len(risen)}",
                f"{median([RUNS[k]['spill'] for k in risen]):.2f}",
                f"{rho:+.2f}",
                f"{p:.2f}",
            ]
        )
    return table_html(
        ["epochs", "anchored runs that rise", "median spill", "Spearman ρ with anti at the rise", "p"],
        rows,
        "**The spill against the anti-subspace weight at the rise.** Spill is the largest drop the edit causes on "
        "any other op, net of the control. ρ is a rank correlation over the runs at one length; the same ρ holds for "
        "the learning rate at the rise, which ranks the runs identically.",
    )


def spill_figure() -> str:
    caption = f"""
        **The spill of the edit against the anti-subspace weight at the rise.** One mark per anchored run that
        makes the rise: circles at 200 epochs, squares at 400. The dashed line is ex-2.2.21's selectivity criterion
        ({ex.SELECTIVITY_GATE:g}). The x axis would be the same, rank for rank, for the learning rate at the rise.
    """
    alt = f"""
        A scatter of spill against the anti-subspace weight at the rise, from 0.1 to 1. At 200 epochs the circles
        fall from left to right: the runs that rise late, where the weight is near 0.15, spill about 0.2, and those
        that rise early, near 0.6 to 0.85, stay near or under 0.05 (ρ = {RHO[SHORT][0]:+.2f}). At 400 epochs the
        squares sit at weights from 0.3 to 0.95 and mostly higher, up to 0.47, with a weaker downward trend
        (ρ = {RHO[LONG][0]:+.2f}). One 400-epoch run sits under the criterion.
    """
    data = {e: [(RUNS[k]["anti_at_rise"], RUNS[k]["spill"]) for k in keys(ANCHOR, e, risen=True)] for e in ex.LENGTHS}
    return spill_draw(data, caption, alt)


@memo
def spill_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="schedules-at-the-rise-spill", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.6, 2.8), layout="constrained")
        for e in ex.LENGTHS:
            pts = np.array(data[e])
            ax.scatter(pts[:, 0], pts[:, 1], marker=MARKER[e], s=26, facecolor=ink_of(e),
                       edgecolor=light_dark("#fff", "#111"), lw=0.6, label=f"{e} epochs", zorder=3)  # fmt: skip
        ax.axhline(ex.SELECTIVITY_GATE, color=light_dark("#000", "#fff"), lw=0.8, ls="--", alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 0.5)
        ax.set_xlabel("anti-subspace weight at the rise / peak", fontsize=8)
        ax.set_ylabel("spill (largest drop on another op)", fontsize=8)
        fig.legend(loc="outside upper center", ncols=2, fontsize=7, frameon=False)
        return fig

    return _plot()


# --- E4: the lean through training -------------------------------------------------------------------------------


def lean_rho(epochs: int) -> float:
    risen = keys(ANCHOR, epochs, risen=True)
    res = spearmanr([RUNS[k]["lean"][-1] for k in risen], [RUNS[k]["spill"] for k in risen])
    return float(res.statistic)


LEAN_RHO = {e: lean_rho(e) for e in ex.LENGTHS}


def lean_figure() -> str:
    caption = """
        **The lean through training.** The mean alignment of every state with e₁ at the last slice, one line per
        run, against the share of training: 200 epochs on the left, 400 on the right. Anchored runs in color, with
        their seed mean drawn heavier and marked; control runs as hairlines behind.
    """
    alt = """
        Two panels of the lean against the share of training. The control runs wander between about −0.05 and 0.08
        and end anywhere in that range. The anchored runs climb to about 0.02 within the first tenth. At 200 epochs
        their mean then creeps up to about 0.03 over the second half of training; at 400 epochs it holds near 0.02
        for the first third and creeps up to about 0.035 over the rest. The anchored runs end in a narrow band at
        both lengths.
    """
    data = {
        e: {
            c: [((RUNS[k]["epoch"] / e).tolist(), RUNS[k]["lean"].tolist()) for k in keys(c, e)]
            for c in (CONTROL, ANCHOR)
        }
        for e in ex.LENGTHS
    }
    return lean_draw(data, caption, alt)


@memo
def lean_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="schedules-at-the-rise-lean", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained", sharey=True)
        for ax, e in zip(axes, ex.LENGTHS, strict=True):
            for x, y in data[e][CONTROL]:
                ax.plot(x, y, color=GHOST, lw=0.4, alpha=0.7, zorder=0)
            ys = []
            for x, y in data[e][ANCHOR]:
                ax.plot(x, y, color=ink_of(e), lw=0.5, alpha=0.4, zorder=1)
                ys.append(y)
            x = data[e][ANCHOR][0][0]
            ax.plot(x, np.mean(ys, axis=0), color=ink_of(e), lw=1.6, marker=MARKER[e], ms=3.5, markevery=0.1,
                    zorder=2)  # fmt: skip
            ax.axhline(0, color=light_dark("#000", "#fff"), lw=0.5, alpha=0.4)
            ax.set_title(f"{e} epochs", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_xlabel("share of training", fontsize=8)
        axes[0].set_ylabel("lean (mean alignment)", fontsize=8)
        return fig

    return _plot()


# --- The report ----------------------------------------------------------------------------------------------------

rf"""
# Where the second rise falls on the schedules

/// tip |
<!-- lede -->
In the recipe, the anti-subspace weight falls through training in step with the learning rate, so the stored runs cannot tell the two apart. Within each length, the runs that learn the HSV ops later, when both have fallen further, spill more; across lengths the 400-epoch runs rise earlier on the schedule and spill more all the same.
///

Ex-2.2.23 found that the edit spills onto other ops on nearly every run that has made the second rise, and more at 400 epochs than at 200. Its discussion guessed that the anti-subspace term, which keeps other information off e₁, is easing off while the model learns the HSV ops, so a longer run spends more of its learning with the term weak. The [backlog item](/todo/science/schedules-at-the-rise.md) asked for a look at where the rise falls on the schedules, with no new runs. This report reads the trajectory records of ex-2.2.23's {len(RUNS)} runs, which store the learning rate and both regularizer weights beside the task score and the lean, and the edit scores of its anchored runs.
"""

# %%

rf"""
## Observations

- [The schedules move together (E1)](#the-schedules-move-together-e1): the anti-subspace weight and the learning rate fall along nearly the same curve, at both lengths. Neither is computed from the other; the recipe stretches both over the run.
- [Where the rise falls (E2)](#where-the-rise-falls-e2): runs rise at similar epochs at both lengths, so on the 400-epoch schedule they rise earlier in the run, while the anti-subspace weight is still high.
- [The spill and the schedule at the rise (E3)](#the-spill-and-the-schedule-at-the-rise-e3): within a length, the runs that rise later on the schedule tend to spill more. Across lengths it goes the other way: the 400-epoch runs rise higher on the schedule and spill more.
- [The lean through training (E4)](#the-lean-through-training-e4): the anchored runs reach most of their lean within the first tenth of training, and it nearly doubles over the rest of training as the anti-subspace weight falls, ending at about the same level at both lengths, still small against the spread of the control runs.

## Scope

This is an exploratory re-analysis of stored results, with no preregistration and no gate. It covers the {len(keys(ANCHOR))} anchored and {len(keys(CONTROL))} control runs of ex-2.2.23: twelve model seeds of each condition at each of 200 and 400 epochs, on the recipe of record. The trajectory records come every four epochs, so a rise epoch is known to within four epochs. With twelve or fewer runs per length, a rank correlation has to be large to mean much, and the p-values in E3 are there as a yardstick for that, uncorrected.

The {num_word(N_SHORT_MISSED)} anchored runs that never make the rise at 200 epochs have no rise epoch, and E2 and E3 leave them out; ex-2.2.23 found that none of them spills.

## The measurements

Rise epoch
:   The first trajectory record at which the HSV skill (expected exact match averaged over the three HSV ops) reaches {ex.RISE_LEVEL:g}, as in ex-2.2.23.

Weight at the rise
:   The learning rate, or the anti-subspace weight, at the rise epoch, over its own peak. 1 is full strength.

Spill
:   The largest drop the edit causes on any op other than the anchored one, at any dose, net of the control at the same seed and length: ex-2.2.23's measurement. Lower is better; ex-2.2.21's criterion is {ex.SELECTIVITY_GATE:g}.

Lean
:   The mean alignment with e₁ of every state at the last slice, over the probe contexts, as the trajectory records store it. The anti-subspace term acts on the mean of the squared alignment of the same states, so this is close to what the term sees.
"""

# %%

rf"""
## The schedules move together (E1)

Is the anti-subspace anneal tied to the learning rate? In the code it is not: the optimizer reads its learning rate from an optax warm-up-and-cosine schedule, and the training step takes the anti-subspace weight from its own minimum-jerk curve. But the recipe sets every keyframe of the regularizer schedules as a share of the run, and the cosine also runs over the whole run.

{schedule_table()}

{schedule_figure()}

The result is two curves that fall together. Past the LR warm-up, the learning rate and the anti-subspace weight correlate at r = {R_LR_ANTI:.2f} over a run, at either length. They differ in the last tenth, where the anti-subspace weight drops to its floor along with the anchor and the learning rate finishes its cosine, and in the first few epochs, where the learning rate is still warming up.

So, for these runs, anything that follows the anti-subspace weight through training also follows the learning rate, and the stored trajectories cannot say which of the two it follows. Separating them takes a run whose schedules differ: the anti-subspace anneal now has a start epoch (`AntiSpec.anneal_start`), which holds the weight at its peak until then. Its default of zero is the schedule every run so far used.
"""

# %%

rf"""
## Where the rise falls (E2)

At what point on the schedules does each run learn the HSV ops? If a 400-epoch run spends more of its learning with the anti-subspace term weak, its rise should fall later on the schedule than a 200-epoch run does.

{rise_table()}

{rise_figure()}

It is the other way round. The runs rise at similar epochs at both lengths (the anchored runs a little later at 400), so on the 400-epoch schedule, which is stretched to twice the length, the rise comes earlier in the run. The anti-subspace weight at the rise is higher at 400 epochs: about four-fifths of its peak, against a bit under half at 200 epochs. The learning rate tells the same story, as E1 implies. The control runs, which have no anti-subspace term, rise over a similar span of epochs, so the timing of the rise looks set by the task and the learning rate rather than by the anti-subspace term.

What the longer run does have is more training after the rise: about three times as many epochs after it, most of them with the weight below half its peak. So the guess in ex-2.2.23 survives in a different form. A 400-epoch run learns the HSV ops while the term is strong, then trains for a long time with it weak.
"""

# %%

rf"""
## The spill and the schedule at the rise (E3)

Does the spill follow the schedule at the rise? A run that rises when the term is weaker might store more lightness along e₁ as it learns the HSV ops.

{spill_table()}

{spill_figure()}

Within each length, the runs that rise later on the schedule tend to spill more. At 200 epochs the trend is the stronger of the two: the four runs that rise last spill several times as much as the five that rise first. At 400 epochs it is weaker, and with twelve runs it could be chance. Across lengths it goes the other way: the 400-epoch runs rise higher on the schedule and spill more.

So the weight at the rise does not account for the spill on its own, and neither does the length of training after the rise: within a length, the runs that spill most are the late ones, with the least training after their rise. The two trends fit a reading with two parts, one for each: a run spills more when it learns the HSV ops with the term weak (the late runs at 200 epochs), and more again when it trains for a long time with the term weak after learning them (every run at 400). That reading fits without being tested. A late rise also marks a seed that learns slowly, and the learning rate falls with the anti-subspace weight, so either trend may have nothing to do with the anti-subspace term.
"""

# %%

# REVIEW: "most of their lean ... creeps up by about half again" became "about half ... by most of that again": the
# seed mean is ~0.017-0.020 at a tenth of training and ~0.031-0.035 at the end at both lengths. Observations still says
# "creeps up a little", which holds against the control spread; check it against the figure if that reading changes.
rf"""
## The lean through training (E4)

If the anti-subspace term is what keeps states off e₁, the lean of the anchored runs might grow as its weight falls.

{lean_figure()}

Somewhat, though it stays small against the spread of the control runs. The anchored runs reach about half their final lean within the first tenth of training, while the anti-subspace weight is near its peak. After that the seed mean creeps up by most of that again, over the second half of the run at 200 epochs and the last two-thirds at 400, as the weight falls, and it ends at about the same level at both lengths. The control runs wander far more widely, so the anchored runs end in a narrow band at both lengths. At 400 epochs the runs that end with more lean tend to spill more (ρ = {LEAN_RHO[LONG]:+.2f}); at 200 epochs there is no such trend (ρ = {LEAN_RHO[SHORT]:+.2f}).

The lean is a mean over every state, and most states carry no lightness that an edit could remove. So a lean that moves this little says the cloud as a whole stays close to where the anti-subspace term leaves it. It does not rule out a few states that carry lightness drifting onto e₁ late in training, which is what the spill would need.
"""

# %%

r"""
## Discussion

The schedules of the recipe were never meant to track the learning rate, but they do, because the recipe stretches all of them over the run. That makes the stored runs a poor place to look for the effect of the anti-subspace term: any trend over training, or between runs that rise at different times, is a trend in the learning rate too.

What the runs do show is that learning the HSV ops under a weak anti-subspace term does not seem to be the whole of the spill. At 400 epochs the rise comes while the term is strong, and the spill is larger. The difference between the lengths lies after the rise, in the long stretch of training with both the term and the learning rate low. If the term matters, a hold that keeps it up through that stretch would lower the spill; if the spill comes from training at a low learning rate, such a hold would change little. A run with the anti-subspace anneal moved late, at an unchanged LR schedule, would separate the two.

The lean says little either way. It is a mean over every state, and it moves little on the anchored runs and ends at the same level at both lengths while the spill differs, so if lightness moves onto e₁ late in training it moves on a few states. A measurement on the states that carry lightness, which the backlog item on what e₁ holds besides the op asks for, would be the place to look.
"""
