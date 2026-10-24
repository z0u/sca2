# ruff: noqa: B018
# title: A baseline from the quiet axes

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


AXES_BODY = fetch_json(ex.AXES_REF)
AX = {r["label"]: r for r in AXES_BODY["runs"]}
DESIGN = AXES_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
GAMMAS: list[float] = DESIGN["dose_gammas"]
FULL_DOSE = len(GAMMAS) - 1
CRITERION = 0.02  # The selectivity criterion of ex-2.2.21: the largest net drop on another op the edit may cause.
E1 = 0  # The anchored axis, in code.
N_SYNTAX = 2  # The axes left out of the reference set: the two that hold the most energy (the syntax axes).
N_DRAW = 8  # Reference axes per draw, as in spill-timing.
N_DRAWS = 2000
# The dose scoring split-anti-seeds published, which the e₁ drops here should reproduce.
STORED_DOSE = {r["label"]: r for r in fetch_json(ex.sas.DOSE_REF)["runs"]}
ROT_DOSE = {r["label"]: r for r in fetch_json(ex.rb.DOSE_REF)["runs"]}

FALLOFF, CONTROL = "falloff", "control"
COORD, ROT = "coordinate", "rotated"  # Which stored pair: the coordinate basis of split-anti-seeds, or rotated-basis.


def label(arm: str, seed: int, which: str = COORD) -> str:
    base = ex.label_of(arm, seed)
    return ex.ROTATED + base if which == ROT else base


def drops(arm: str, seed: int, which: str = COORD) -> np.ndarray:
    """The drop under the edit along each direction, at each dose, per op: (directions, doses, ops). Along the
    coordinate axes for the stored pair, and along the rotated basis Adam acted in for the rotated-basis pair.
    """
    r = AX[label(arm, seed, which)]
    return np.asarray(r["drop_rotated"] if which == ROT else r["drop"])


def energy(arm: str, seed: int, which: str = COORD) -> np.ndarray:
    """The share of the energy of the stream each direction holds, averaged over slices."""
    r = AX[label(arm, seed, which)]
    return np.asarray(r["energy_rotated"] if which == ROT else r["energy"]).mean(axis=0)


def span(seed: int, which: str = COORD) -> float:
    """The clean-to-null span on the anchored op of the `falloff` run, which removal is a share of."""
    r = AX[label(FALLOFF, seed, which)]
    return r["clean"][D] - r["null"][D]


def syntax_dirs(arm: str, seed: int, which: str = COORD) -> list[int]:
    """The two directions other than e₁ that hold the most energy of the stream (code index)."""
    e = energy(arm, seed, which).copy()
    if which == COORD:
        e[E1] = -1
    return [int(i) for i in np.argsort(-e)[:N_SYNTAX]]


def reference_dirs(arm: str, seed: int, which: str = COORD) -> list[int]:
    """The reference set: every direction but e₁ and the two syntax directions. Under the rotated basis e₁ is not one
    of the directions, so only the syntax directions are left out.
    """
    n = drops(arm, seed, which).shape[0]
    out = set(syntax_dirs(arm, seed, which)) | ({E1} if which == COORD else set())
    return [i for i in range(n) if i not in out]


def median_drop(arm: str, seed: int, which: str = COORD, dirs: list[int] | None = None) -> np.ndarray:
    """The median drop over the reference directions (or *dirs*), per dose and op: (doses, ops)."""
    dirs = reference_dirs(arm, seed, which) if dirs is None else dirs
    return np.median(drops(arm, seed, which)[dirs], axis=0)


def e1_drop(arm: str, seed: int, which: str = COORD) -> np.ndarray:
    """The drop under the e₁ edit, per dose and op. In both pairs e₁ is the first coordinate axis."""
    return np.asarray(AX[label(arm, seed, which)]["drop"])[E1]


# The three baselines a drop on the anchored run is netted against, per dose and op.
PAIRED, CTRL_MED, SELF_MED = "paired", "control median", "own median"
BASELINES = (PAIRED, CTRL_MED, SELF_MED)
SHOWN = {PAIRED: "e₁ in the control", CTRL_MED: "reference axes, control", SELF_MED: "reference axes, own run"}


def baseline(kind: str, seed: int, which: str = COORD) -> np.ndarray:
    if kind == PAIRED:
        return e1_drop(CONTROL, seed, which)
    if kind == CTRL_MED:
        return median_drop(CONTROL, seed, which)
    return median_drop(FALLOFF, seed, which)


def scores(base: np.ndarray, seed: int, which: str = COORD) -> dict[str, Any]:
    """Removal at the full dose and spill at every dose, for the e₁ edit on `falloff` net of *base*."""
    net = e1_drop(FALLOFF, seed, which) - base
    return {
        "removal": float(net[FULL_DOSE, D] / span(seed, which)),
        "spill": [float(net[g, OTHER].max()) for g in range(len(GAMMAS))],
        "worst_spill": float(net[:, OTHER].max()),
    }


def scored(kind: str, seed: int, which: str = COORD) -> dict[str, Any]:
    return scores(baseline(kind, seed, which), seed, which)


def raw(seed: int, which: str = COORD) -> dict[str, Any]:
    """The same with nothing subtracted."""
    return scores(np.zeros_like(e1_drop(FALLOFF, seed, which)), seed, which)


def draws(arm: str, seed: int, which: str = COORD) -> np.ndarray:
    """Removal and worst spill under the median of *N_DRAW* reference directions drawn at random, *N_DRAWS* times:
    (draws, 2).
    """
    ref = reference_dirs(arm, seed, which)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_DRAWS):
        pick = list(rng.choice(ref, N_DRAW, replace=False))
        s = scores(median_drop(arm, seed, which, pick), seed, which)
        out.append((s["removal"], s["worst_spill"]))
    return np.array(out)


def mean_range(values) -> tuple[float, float, float]:
    v = np.asarray(list(values), float)
    return float(v.mean()), float(v.min()), float(v.max())


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    half = 0.5 * 10 ** -int(spec[1])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(v) < half else v for v in mean_range(values))
    return f"{num(m, spec)} ({num(lo, spec)} to {num(hi, spec)})"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    from mini.vis import figure_html

    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- Checks against the published dose scoring ---------------------------------------------------------------


def stored_e1(arm: str, seed: int, which: str = COORD) -> np.ndarray:
    lab = f"control-s{seed}" if (arm == CONTROL and which == COORD) else ex.label_of(arm, seed)
    d = (STORED_DOSE if which == COORD else ROT_DOSE)[lab]
    return np.array([e["drop"] for e in d["edits"]])


REPRO = max(
    float(abs(e1_drop(a, s, w) - stored_e1(a, s, w)).max())
    for a in (FALLOFF, CONTROL)
    for s in SEEDS
    for w in (COORD, ROT)
)
assert REPRO < 1e-4, f"the e₁ drops differ from the published dose scoring by {REPRO}"

# --- Summaries the prose quotes ---------------------------------------------------------------------------------

SC = {w: {k: {s: scored(k, s, w) for s in SEEDS} for k in BASELINES} for w in (COORD, ROT)}
RAW = {w: {s: raw(s, w) for s in SEEDS} for w in (COORD, ROT)}
REM = {w: {k: {s: SC[w][k][s]["removal"] for s in SEEDS} for k in BASELINES} for w in (COORD, ROT)}
SPILL = {w: {k: {s: SC[w][k][s]["worst_spill"] for s in SEEDS} for k in BASELINES} for w in (COORD, ROT)}
N_WITHIN = {w: {k: sum(v <= CRITERION for v in SPILL[w][k].values()) for k in BASELINES} for w in (COORD, ROT)}
MEAN_REM = {w: {k: float(np.mean(list(REM[w][k].values()))) for k in BASELINES} for w in (COORD, ROT)}
# Removal moved by the new baselines, per seed, against the paired one.
SHIFT = {
    w: {k: {s: REM[w][k][s] - REM[w][PAIRED][s] for s in SEEDS} for k in (CTRL_MED, SELF_MED)} for w in (COORD, ROT)
}
OWN_VS_CTRL = {w: max(abs(REM[w][SELF_MED][s] - REM[w][CTRL_MED][s]) for s in SEEDS) for w in (COORD, ROT)}


def cost(arm: str, seed: int, which: str = COORD) -> np.ndarray:
    """The cost of the full edit along each direction: the drop at the full dose, averaged over the ops."""
    return drops(arm, seed, which)[:, FULL_DOSE].mean(axis=1)


def e1_cost(arm: str, seed: int) -> float:
    return float(np.asarray(AX[label(arm, seed)]["drop"])[E1, FULL_DOSE].mean())


def e1_rank(seed: int) -> int:
    """Where e₁ falls among the 64 coordinate axes of the control, by cost: 1 is the costliest."""
    c = cost(CONTROL, seed)
    return int((c > c[E1]).sum()) + 1


RANK = {s: e1_rank(s) for s in SEEDS}
REF_COST = {s: float(np.median(cost(CONTROL, s)[reference_dirs(CONTROL, s)])) for s in SEEDS}
SYN_COST = {s: [float(cost(CONTROL, s)[i]) for i in syntax_dirs(CONTROL, s)] for s in SEEDS}
TOP_SHARE = {s: float(energy(CONTROL, s)[syntax_dirs(CONTROL, s)[0]]) for s in SEEDS}
MILD_SECOND = [s for s in SEEDS if min(SYN_COST[s]) < 0.25]
SPIKE_MOVED = [s for s in SEEDS if syntax_dirs(CONTROL, s)[0] != syntax_dirs(FALLOFF, s)[0]]
BUSIEST, QUIETEST = min(SEEDS, key=lambda s: RANK[s]), max(SEEDS, key=lambda s: RANK[s])
# With the syntax axes left in, the median over the other 63 axes barely moves.
NO_EXCL = max(
    abs(
        scores(median_drop(CONTROL, s, COORD, [i for i in range(64) if i != E1]), s)["removal"]
        - REM[COORD][CTRL_MED][s]
    )
    for s in SEEDS
)
DRAWS = {w: {(a, s): draws(a, s, w) for a in (CONTROL, FALLOFF) for s in SEEDS} for w in (COORD, ROT)}
DRAW_WIDTH = {
    w: max(float(np.subtract(*np.percentile(d[:, 0], [95, 5]))) for d in DRAWS[w].values()) for w in (COORD, ROT)
}
DRAW_SPILL_HI = {w: max(float(np.percentile(d[:, 1], 95)) for d in DRAWS[w].values()) for w in (COORD, ROT)}
LOST_PAIRED_ROT = [s for s in SEEDS if REM[ROT][PAIRED][s] < 0.5]

INK = {
    PAIRED: light_dark("#888", "#999"),
    CTRL_MED: light_dark("#0b3c73", "#cfe6ff"),
    SELF_MED: light_dark("#2a9d8f", "#7fd8c8"),
    "raw": light_dark("#c0392b", "#ff8a76"),
}
MARK = {PAIRED: "o", CTRL_MED: "s", SELF_MED: "D", "raw": "^"}
GHOST = light_dark("#bbb", "#555")


# %%

rf"""
# A baseline from the quiet axes

/// tip |
<!-- lede -->
Removal and spill have been measured net of what the same edit does to the control at the same seed, so they depend on where e₁ happens to sit in that control. We scored the stored runs of the recipe of record again against a baseline that does not: the median drop over edits along the other quiet coordinate axes. The numbers quoted so far barely move, and on the rotated-basis runs, where the old baseline fell short, the new one puts the short seeds back beside the rest.
///

The question comes from [rotated-basis](../rotated-basis/report.py). There, with Adam acting in a rotated basis, e₁ overlapped the syntax direction of the control on two seeds, the edit took half of the op from the control, and removal net of the control fell short although the anchored runs removed the op as before. That report and the [D2.2 design](../d2.2/design.md) proposed the median over reference edits along other coordinate axes, off the syntax axes, as in the spill-timing experiment.

This report trains nothing. It scores the edit along each of the 64 coordinate axes, at every dose, on the eight `falloff` runs (the recipe of record) of [split-anti-seeds](../split-anti-seeds/report.py) and the eight controls of ex-2.2.23 at the same seeds, and on the sixteen runs of rotated-basis.

## Observations

- [Where e₁ sits in the control (E1)](#where-e-sits-in-the-control-e1): it varies by seed. In the control e₁ is among the quietest axes on some seeds and among the five costliest on one, while the median over the reference axes is steady across seeds.
- [Net removal and spill against the new baseline (E2)](#net-removal-and-spill-against-the-new-baseline-e2): little changes. The mean removal rises by a hundredth or two, most of it on the one seed where e₁ was a busy axis of the control, and every run stays within the spill criterion. The median taken in the anchored run itself, with no control, agrees with the one taken in the control.
- [The rotated-basis runs (E3)](#the-rotated-basis-runs-e3): the new baseline, taken along the rotated axes, brings the removal on the two short seeds up to the level of the rest.

## Scope and method

This is an exploratory re-scoring of stored runs, with no preregistration and no gate. It covers `falloff` and the 400-epoch control at model seeds 704 to 711, as published by split-anti-seeds and ex-2.2.23, and the `falloff` and control runs of rotated-basis at the same seeds. Each anchored run pairs with the control at its seed, which shares its initialization, batches and label draws.

The edit is the dose edit of ex-2.2.22, unchanged: a direction projected out of the state at every position and slice, at a quarter, a half, three quarters, or all of its component. It is scored on the whole held-out set of every op, as before. For e₁ these drops reproduce the published dose scoring to rounding. What is new is the same edit along every other coordinate axis, and, for the rotated-basis runs, along every axis of the rotated basis Adam acted in.

A reference axis is any coordinate axis other than e₁ and the two that hold the most energy[^energy] of the stream, averaged over slices. Those two are where the model puts its syntax tokens: one coordinate holds about {num(np.mean(list(TOP_SHARE.values())), ".1f")} of the energy in every control, and projecting it out takes most of the score of every op, on some seeds all of it. For the rotated-basis runs
<!-- REVIEW: "costs every op about half its score" read the cost of the top axis (about 0.5, the drop averaged over ops) as a share of the score; the mean clean score over ops is about 0.58, so that drop is most to all of it. Verify: `drop[top][FULL_DOSE]` over `clean` per op in the control axes data is 0.7 to 1.0 on every seed, and 1.0 on 707 and 708. -->
<!-- REVIEW: "Those two are where the model puts its syntax tokens" is the rule's intent; E1 shows the second axis by energy is a mild one on 707 and 708, where no other axis costs more than about 0.2. -->
 the reference axes are the rotated axes, with the two highest in energy left out the same way.

[^energy]: Energy: the sum of the squared projections of the states onto a direction. The share of a direction is its energy over the total.

## The measurements

- Removal: the net drop in score on `difference` under the full edit along e₁, as a share of the way from the clean score to the target null. Higher is better.
- Spill: the largest net drop on another op under the e₁ edit, over the four doses. A run is within the criterion when it is at most {CRITERION:g}. Lower is better.
- The cost of an axis: the drop under the full edit along it, averaged over the ops.

Each of these is net of a baseline, and the report compares three, each taken per op and per dose:

- e₁ in the control: the drop the same e₁ edit causes in the control at the same seed. This is the measurement quoted so far.
- Reference axes, control: the median drop over the reference axes of the control.
- Reference axes, own run: the median drop over the reference axes of the anchored run itself. It needs no control.

E2 also shows removal with nothing subtracted (raw), as rotated-basis recommended.
"""

# %%


def cost_figure() -> str:
    data = {
        "seeds": SEEDS,
        "costs": [cost(CONTROL, s).tolist() for s in SEEDS],
        "syntax": [syntax_dirs(CONTROL, s) for s in SEEDS],
        "median": [REF_COST[s] for s in SEEDS],
    }
    alt = f"""
        The cost of projecting out each coordinate axis of the control, one column per seed. Most axes sit in a
        band from about 0.01 to 0.06, the median of the reference axes is near 0.03 on every seed, and the two
        syntax axes sit apart, near 0.5 on most seeds. e₁, drawn as a square, falls low in the band on most seeds
        and near its top at seed {BUSIEST}.
    """
    return cost_draw(data, alt)


@memo
def cost_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="reference-baseline-cost",
        alt_text=alt_text,
        caption="""
            **The cost of each coordinate axis in the control**, the full edit along it, averaged over the ops. One
            column per seed: the reference axes as faded dots, their median as a bar, the two syntax axes as open
            circles, and e₁ as a square. The axis is logarithmic.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
        rng = np.random.default_rng(0)
        for i, (c, syn, med) in enumerate(zip(data["costs"], data["syntax"], data["median"], strict=True)):
            c = np.asarray(c)
            ref = [j for j in range(len(c)) if j != E1 and j not in syn]
            ax.scatter(
                i + rng.uniform(-0.18, 0.18, len(ref)), np.maximum(c[ref], 1e-2), s=6, color=GHOST, lw=0, zorder=1
            )
            ax.plot([i - 0.28, i + 0.28], [med, med], color=INK[CTRL_MED], lw=1.8, zorder=2)
            ax.scatter([i] * len(syn), c[syn], s=22, facecolor="none", edgecolor=INK["raw"], lw=0.9, zorder=3)
            ax.scatter([i], [c[E1]], s=26, marker="s", color=INK[PAIRED], edgecolor=light_dark("#000", "#fff"),
                       lw=0.6, zorder=4)  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1)
        ax.set_xticks(range(len(data["seeds"])), [str(s) for s in data["seeds"]], fontsize=7)
        ax.set_xlabel("seed", fontsize=8)
        ax.set_ylabel("cost (mean drop over ops)", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig

    return _plot()


def rank_table() -> str:
    rows = [
        [
            str(s),
            f"{RANK[s]} of 64",
            num(e1_cost(CONTROL, s), ".3f"),
            num(REF_COST[s], ".3f"),
            ", ".join(num(v) for v in SYN_COST[s]),
            num(TOP_SHARE[s]),
            "moved" if s in SPIKE_MOVED else "same",
        ]
        for s in SEEDS
    ]
    return table_html(
        [
            "seed",
            "e₁ by cost, control",
            "e₁ cost, control",
            "reference median, control",
            "syntax axes cost, control",
            "top energy share, control",
            "top axis, `falloff`",
        ],
        rows,
        """
        **Where e₁ sits in the control**, per seed. The rank counts from the costliest of the 64 coordinate axes.
        The cost is the drop under the full edit, averaged over the ops. The top energy share is the share of the
        energy of the stream held by the axis with the most, averaged over slices, and the last column says whether
        the `falloff` run at that seed has its top axis on the same coordinate.
        """,
    )


rf"""
## Where e₁ sits in the control (E1)

How typical an axis is e₁ in a control? If it is always among the quiet ones, the paired baseline and the median of the quiet axes should agree, and nothing quoted so far would move. If its place varies by seed, so does the paired baseline.

**What we saw.** Its place varies. On two seeds e₁ is the second quietest axis of the control, and at seed {BUSIEST} it is the {RANK[BUSIEST]}th costliest of 64, at about twice the median of the reference axes. The median itself is steady across seeds.

{cost_figure()}

{rank_table()}

The axis with the most energy takes most of the score of every op on every seed, and all of it on 707 and 708. The second costs as much on most seeds, but on {" and ".join(str(s) for s in MILD_SECOND)} it is a milder axis, and on 706 two reference axes cost more than it does. So ranking by energy only roughly finds the costly axes. Leaving them out of the reference set hardly matters either way: the median over all 63 axes other than e₁ gives a removal within {num(NO_EXCL, ".3f")} of the median without them, on every seed. A median of a few dozen axes is not swayed by two.

One detail differs from what rotated-basis expected of the stored runs. On {len(SPIKE_MOVED)} of 8 seeds ({", ".join(str(s) for s in SPIKE_MOVED)}) the `falloff` run puts its most energetic axis on a different coordinate from the control at the same seed, at every slice, and that coordinate holds almost nothing in the control. Rotated-basis took the spike to sit on the same coordinate in both, and read a move under the rotated basis as fitting the anti term steering it. These moves under plain Adam suggest a move happens without the rotation too, so a change of loss alone may be enough.
<!-- REVIEW: rotated-basis E3 says "In the stored runs we found the spike on the same coordinate in the control and the anchored run at every seed we checked." Here the top-energy coordinate differs on the seeds listed. Verify: `energy` of `falloff-s<seed>` and `control-s<seed>` in the axes data, averaged over slices; the result holds at the last slice too. Checked against the axes data: on 706, 707 and 708 the top-energy coordinate of `falloff` differs from that of the control at each of the five slices, and the control holds under 0.02 of its energy on that coordinate; on the other five seeds it is the same coordinate at every slice. -->

So the paired baseline is one draw from the axes of the control, and a draw that varies from the quietest to among the busiest few.
"""

# %%


def scoring_figure(which: str) -> str:
    data = {
        "which": which,
        "seeds": SEEDS,
        "removal": {k: [REM[which][k][s] for s in SEEDS] for k in BASELINES}
        | {"raw": [RAW[which][s]["removal"] for s in SEEDS]},
        "spill": {k: [SPILL[which][k][s] for s in SEEDS] for k in BASELINES}
        | {"raw": [RAW[which][s]["worst_spill"] for s in SEEDS]},
    }
    if which == COORD:
        alt = f"""
            Removal (top) and spill (bottom) per seed, under each baseline and with nothing subtracted. Removal sits
            between about 0.77 and 0.91 under all three baselines, the three markers close together on every seed
            but {BUSIEST}, where the paired baseline sits about a tenth below the two medians. Raw removal sits
            above, near 0.97. Spill sits near zero under all three baselines, below the criterion; the raw spill is
            higher, up to about 0.04.
        """
    else:
        alt = f"""
            The same for the rotated-basis runs. Under the paired baseline removal falls to about 0.5 and 0.3 on
            seeds {" and ".join(str(s) for s in LOST_PAIRED_ROT)} and is lower on 707; under both medians it sits
            near 0.85 to 0.9 on every seed. Raw removal is near 0.97 throughout. Spill is near zero under every
            baseline.
        """
    return scoring_draw(data, alt)


@memo
def scoring_draw(data: dict, alt_text: str) -> str:
    which = data["which"]

    @themed(
        name=f"reference-baseline-{which}",
        alt_text=alt_text,
        caption=f"""
            **Removal and spill under each baseline**, {"the stored runs" if which == COORD else "the rotated-basis runs"},
            per seed: net of e₁ in the control (the measurement quoted so far), net of the median over the reference
            axes of the control, net of the median over the reference axes of the anchored run, and with nothing
            subtracted. Removal is at the full dose; spill is the largest over the four doses. The dashed rule is the
            spill criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.8), layout="constrained", sharex=True)
        x = np.arange(len(data["seeds"]))
        kinds = (*BASELINES, "raw")
        offset = {k: (i - 1.5) * 0.14 for i, k in enumerate(kinds)}
        names = {**SHOWN, "raw": "nothing subtracted"}
        for ax, key in zip(axes, ("removal", "spill"), strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for k in kinds:
                ax.scatter(x + offset[k], data[key][k], marker=MARK[k], s=18, color=INK[k], zorder=2, label=names[k])
            if key == "spill":
                ax.axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
                ax.set_ylim(-0.02, 0.05)
                ax.set_ylabel("spill", fontsize=8)
            else:
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("removal", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[1].set_xticks(x, [str(s) for s in data["seeds"]])
        axes[1].set_xlabel("seed", fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False, fontsize=7)
        return fig

    return _plot()


def summary_table(which: str) -> str:
    rows = [
        [
            SHOWN[k],
            fmt(REM[which][k].values()),
            f"{N_WITHIN[which][k]} of {len(SEEDS)}",
            fmt(SPILL[which][k].values(), ".3f"),
        ]
        for k in BASELINES
    ] + [
        [
            "nothing subtracted",
            fmt((RAW[which][s]["removal"] for s in SEEDS)),
            "",
            fmt((RAW[which][s]["worst_spill"] for s in SEEDS), ".3f"),
        ]
    ]
    return table_html(
        ["baseline", "removal ↑", "within criterion ↑", "spill, any dose ↓"],
        rows,
        f"""
        **Removal and spill under each baseline**, {"the stored `falloff` runs" if which == COORD else "the rotated-basis `falloff` runs"},
        seed mean and range over {len(SEEDS)} runs. Removal is at the full dose; spill is the largest net drop on
        another op over the four doses, and "within criterion" counts the runs where it is at most {CRITERION:g}.
        Arrows mark which direction is better.
        """,
    )


def seed_table(which: str) -> str:
    rows = [[str(s)] + [num(REM[which][k][s]) for k in BASELINES] + [num(RAW[which][s]["removal"])] for s in SEEDS]
    return table_html(
        ["seed", *[f"removal, {SHOWN[k]} ↑" for k in BASELINES], "removal, nothing subtracted ↑"],
        rows,
        f"""
        **Removal at the full dose by seed**, under each baseline, {"the stored runs" if which == COORD else "the rotated-basis runs"}.
        """,
    )


rf"""
## Net removal and spill against the new baseline (E2)

Netted against the median of the reference axes, do removal and spill of `falloff` change from the numbers split-anti-seeds quoted? The main number is removal at the full dose, per seed, beside spill over the four doses.

**What we saw.** Hardly at all. The mean removal is {num(MEAN_REM[COORD][PAIRED])} against e₁ in the control, {num(MEAN_REM[COORD][CTRL_MED])} against the median of the control, and {num(MEAN_REM[COORD][SELF_MED])} against the median of the anchored run. Every run stays within the spill criterion under every baseline.

{scoring_figure(COORD)}

{summary_table(COORD)}

The one seed that moves by more than a few hundredths is {BUSIEST}, where e₁ was among the busiest axes of the control. There the paired baseline took a tenth off the removal that the medians leave on. Elsewhere the three baselines agree within the spread of a few hundredths that a seed range shows anyway.

{seed_table(COORD)}

The two medians agree with each other on every seed, within {num(OWN_VS_CTRL[COORD])} of removal. So in these runs the quiet axes of an anchored run cost about what the quiet axes of its control cost, and the anchor hardly touches them. A baseline
<!-- REVIEW: "leaves them alone" softened to "hardly touches them": the median cost of the reference axes is a few thousandths higher in `falloff` than in the control on 7 of 8 seeds (0.031 to 0.040 against 0.029 to 0.036). Verify: `np.median(cost(FALLOFF, s)[reference_dirs(FALLOFF, s)])` against `REF_COST[s]`. -->
 taken in the anchored run itself would then do without a control.

Spill falls slightly under both medians, since on the other ops e₁ in the control costs a little less than the median does on most seeds, so the paired baseline subtracts less. The raw spill
<!-- REVIEW: the reason had its direction reversed ("costs a little more than the median"). Net spill is the drop on the anchored run less the baseline, so a higher spill under the paired baseline means that baseline is the smaller one. Verify: `e1_drop(CONTROL, s)[FULL_DOSE, OTHER] - baseline(CTRL_MED, s)[FULL_DOSE, OTHER]` is negative on most ops of most seeds; 705 is the exception. -->
, with nothing subtracted, is higher (up to {num(max(RAW[COORD][s]["worst_spill"] for s in SEEDS), ".3f")}), so some baseline is still needed to separate the anchored op from the general cost of an edit.

The median of eight reference axes drawn at random, as spill-timing takes it, varies with the draw. Over many draws at each seed the middle nine tenths of the removal spans at most {num(DRAW_WIDTH[COORD])}, and the spill stays at or below {num(DRAW_SPILL_HI[COORD], ".3f")} on nineteen draws in twenty. That is about half the seed range of removal, so eight axes are enough to tell a removable run from a weak one. The median of all the reference axes takes no draw at all, for a few minutes more of scoring.
"""

# %%

rf"""
## The rotated-basis runs (E3)

The failure rotated-basis found was a control in which e₁ overlaps the syntax direction. Taken along the rotated axes, where Adam put the cheap directions in those runs, does the median baseline put the short seeds back beside the rest?

**What we saw.** Yes. Against e₁ in the control, removal falls below half on seeds {" and ".join(str(s) for s in LOST_PAIRED_ROT)}, and to about six tenths on 707. Against either median it sits near nine tenths on every seed, with every run within the spill criterion.

{scoring_figure(ROT)}

{summary_table(ROT)}

{seed_table(ROT)}

Again the two medians agree, within {num(OWN_VS_CTRL[ROT])} of removal. The median of eight drawn axes spans at most {num(DRAW_WIDTH[ROT])} in removal over the middle nine tenths of draws, a little wider than under plain Adam.

The reference axes have to be the cheap ones for the basis the model trained in. Under the rotated basis a coordinate axis is an ordinary direction and costs more, as rotated-basis found, so a median over coordinate axes would overstate the baseline there. With an optimizer that has no privileged basis at all, every direction touches the spike, and no set of cheap reference directions exists in the same sense.
"""

# %%

r"""
## Discussion

The paired baseline was sound for the stored runs, in the sense that it rarely moved the result: e₁ happened to be a quiet axis of the control on most seeds, and where it was not, the error was a tenth of removal on one seed. The numbers split-anti-seeds quoted stand.

The median over the quiet axes removes that dependence on the draw, and it costs a few minutes more of scoring per run. Where the draw goes badly, as in the rotated-basis runs, it is the difference between a run read as having lost the anchor and one read as removing nine tenths of the op.

The agreement of the two medians suggests a baseline needs no control at all: the quiet axes of the anchored run serve. That would matter for a recipe meant for real models, where a matched control is costly or impossible, and it is what spill-timing already does along training. The agreement rests on these sixteen pairs, and a recipe that disturbed the quiet axes of the anchored run would break it, so a control is still the check that a run-internal baseline is sound.
"""
