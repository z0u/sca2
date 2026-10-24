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
SOFT_DOSE = {r["label"]: r for r in fetch_json(ex.sp.DOSE_REF)["runs"]}

FALLOFF, CONTROL = "falloff", "control"
COORD, ROT = "coordinate", "rotated"  # Which stored pair: the coordinate basis of split-anti-seeds, or rotated-basis.
# Soft-pool's arms near the spill criterion, each paired with the stored control; as a *which*, the arm stands in
# for `falloff`.
SOFT = tuple(DESIGN["soft_pool_arms"])
SOFT_SHOWN = {"step-soft": "step, soft τ", "pool-123-soft": "pool 1–3, soft τ"}


def label(arm: str, seed: int, which: str = COORD) -> str:
    if which in SOFT:
        return ex.label_of(CONTROL, seed) if arm == CONTROL else ex.SOFT_POOL + ex.sp.label_of(which, seed)
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
    if which != ROT:
        e[E1] = -1
    return [int(i) for i in np.argsort(-e)[:N_SYNTAX]]


def reference_dirs(arm: str, seed: int, which: str = COORD) -> list[int]:
    """The reference set: every direction but e₁ and the two syntax directions. Under the rotated basis e₁ is not one
    of the directions, so only the syntax directions are left out.
    """
    n = drops(arm, seed, which).shape[0]
    out = set(syntax_dirs(arm, seed, which)) | ({E1} if which != ROT else set())
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


def SPILL_OF(kind: str, seed: int, which: str) -> float:
    return scored(kind, seed, which)["worst_spill"]


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


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


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
    if which in SOFT:
        d = SOFT_DOSE[ex.sp.label_of(which, seed)]
        return np.array([e["drop"] for e in d["edits"]])
    lab = f"control-s{seed}" if (arm == CONTROL and which == COORD) else ex.label_of(arm, seed)
    d = (STORED_DOSE if which == COORD else ROT_DOSE)[lab]
    return np.array([e["drop"] for e in d["edits"]])


REPRO = max(
    float(abs(e1_drop(a, s, w) - stored_e1(a, s, w)).max())
    for a in (FALLOFF, CONTROL)
    for s in SEEDS
    for w in (COORD, ROT, *SOFT)
    if not (w in SOFT and a == CONTROL)
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
SOFT_SPILL = {w: {k: {s: SPILL_OF(k, s, w) for s in SEEDS} for k in BASELINES} for w in SOFT}
SOFT_RAW = {w: {s: raw(s, w)["worst_spill"] for s in SEEDS} for w in SOFT}
SOFT_OVER = {w: {k: [s for s in SEEDS if SOFT_SPILL[w][k][s] > CRITERION] for k in BASELINES} for w in SOFT}
SOFT_REM = {w: {k: float(np.mean([scored(k, s, w)["removal"] for s in SEEDS])) for k in BASELINES} for w in SOFT}
STEP, POOL = SOFT

# Light and dark inks, picked with `light_dark` inside the plot functions, where the theme is set.
INK = {
    PAIRED: ("#888", "#999"),
    CTRL_MED: ("#0b3c73", "#cfe6ff"),
    SELF_MED: ("#2a9d8f", "#7fd8c8"),
    "raw": ("#c0392b", "#ff8a76"),
}
MARK = {PAIRED: "o", CTRL_MED: "s", SELF_MED: "D", "raw": "^"}
GHOST = ("#bbb", "#555")


# %%

rf"""
# A baseline from the quiet axes

/// tip |
<!-- lede -->
Removal and spill have been measured net of what the same edit does to the control at the same seed, so they depend on where the anchored axis, e₁, happens to sit in that control. We re-scored the stored runs of the recipe of record against a baseline that does not depend on it: the median drop over edits along the other quiet coordinate axes. The numbers quoted so far barely move, and on the rotated-basis runs the new baseline brings the short seeds back up beside the rest.
///

The question comes from [rotated-basis](../rotated-basis/report.py), where Adam acted in a rotated basis.[^adam] On two seeds e₁ overlapped the syntax direction of the control, so the edit took half of the op from the control as well. Removal net of the control then fell short, although the anchored runs removed the op as before.

That report and the [D2.2 design](../d2.2/design.md) proposed a different baseline: the median over reference edits along other coordinate axes, leaving out the syntax axes, as in the spill-timing experiment.

[^adam]: Adam and the basis: Adam scales the update of each coordinate separately, so the coordinate axes are special to it. Rotated-basis ran it in a rotated basis instead.

This report trains nothing. It scores stored runs: the eight `falloff` runs (the recipe of record) of [split-anti-seeds](../split-anti-seeds/report.py), the eight controls of ex-2.2.23 at the same seeds, the sixteen runs of rotated-basis, and, for E4, the sixteen runs of two soft-pool arms. On each run it scores the edit along each of the 64 coordinate axes, at every dose.

## Observations

- [Where e₁ sits in the control (E1)](#where-e-sits-in-the-control-e1): it varies by seed. On some seeds e₁ is among the quietest axes of the control, and on one it is among the five costliest. The median over the reference axes is steady across seeds. Separately, on three seeds the anchored run keeps its spike of energy on a different coordinate from the control.
- [Net removal and spill against the new baseline (E2)](#net-removal-and-spill-against-the-new-baseline-e2): little changes. The mean removal rises by a hundredth or two, mostly on the one seed where e₁ was a busy axis of the control. Every run stays within the spill criterion. The median taken in the anchored run itself, with no control, agrees with the one taken in the control.
- [The rotated-basis runs (E3)](#the-rotated-basis-runs-e3): the new baseline, taken along the rotated axes, brings the removal on the two seeds below half, and a third at six tenths, up to the level of the rest.
- [Runs near the spill criterion (E4)](#runs-near-the-spill-criterion-e4): on two arms of soft-pool that sit near the criterion, the count of runs over it changes at the edge. `step, soft τ` goes from {len(SOFT_OVER[STEP][PAIRED])} runs over to none, and `pool 1–3, soft τ` keeps about as many but not the same ones.

## Scope and method

This is an exploratory re-scoring, with no preregistration and no gate. The model seeds are 704 to 711, and the controls are the 400-epoch ones. Each anchored run pairs with the control at its seed, which shares its initialization, batches, and label draws.

The edit is the dose edit of ex-2.2.22, unchanged. It projects a fixed direction out of the state at every position and slice, removing a quarter, a half, three quarters, or all of the component along it. It is scored on the whole held-out set of every op, as before.

Along e₁ the drops this edit causes reproduce the published dose scoring to rounding. New here is the same edit along every other coordinate axis, and, for the rotated-basis runs, along every axis of the rotated basis Adam acted in.

A reference axis is any coordinate axis other than e₁ and the two that hold the most energy[^energy] of the stream, averaged over slices. Those two are where the model puts its syntax tokens. For the rotated-basis runs the reference axes are the rotated axes, with the two highest in energy left out the same way.
<!-- REVIEW: "Those two are where the model puts its syntax tokens" is the rule's intent; E1 shows the second axis by energy is a mild one on 707 and 708, where no other axis costs more than about 0.2. -->

One coordinate holds about {num(np.mean(list(TOP_SHARE.values())), ".1f")} of the energy in every control, and E1 shows what projecting it out costs.

[^energy]: Energy: the sum of the squared projections of the states onto a direction. The share of a direction is its energy over the total.

## Measurements

- Removal: the net drop in score on `difference` under the full edit along e₁, as a share of the way from the clean score to the target null. Higher is better.
- Spill: the largest net drop on another op under the e₁ edit, over the four doses. A run is within the criterion when it is at most {CRITERION:g}. Lower is better.
- The cost of an axis: the drop under the full edit along it, averaged over the ops. A quiet axis is one that costs little.

Each of these is net of a baseline. We compare three, each taken per op and per dose:

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
        On every seed most axes cost between about 0.01 and 0.06, around a steady median near 0.03, and the two
        syntax axes stand far above them. e₁ falls low in that band on most seeds, and near its top at seed
        {BUSIEST}.
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
                i + rng.uniform(-0.18, 0.18, len(ref)),
                np.maximum(c[ref], 1e-2),
                s=6,
                color=light_dark(*GHOST),
                lw=0,
                zorder=1,
            )
            ax.plot([i - 0.28, i + 0.28], [med, med], color=light_dark(*INK[CTRL_MED]), lw=1.8, zorder=2)
            ax.scatter(
                [i] * len(syn), c[syn], s=22, facecolor="none", edgecolor=light_dark(*INK["raw"]), lw=0.9, zorder=3
            )
            ax.scatter([i], [c[E1]], s=26, marker="s", color=light_dark(*INK[PAIRED]), edgecolor=light_dark("#000", "#fff"),
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
            "e₁ rank by cost, control",
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

**What we saw.** Its place varies. On two seeds e₁ is the second quietest axis of the control. At seed {BUSIEST} it is the {ordinal(RANK[BUSIEST])} costliest of 64 (the high square in the figure below), with about twice the cost of the median reference axis. The median itself is steady across seeds.

{cost_figure()}

{rank_table()}

So the paired baseline is one draw from the axes of the control, and that draw ranges from one of the quietest axes to one of the busiest few.

The axis with the most energy takes most of the score of every op on every seed, and all of it on 707 and 708.
<!-- REVIEW: "costs every op about half its score" read the cost of the top axis (about 0.5, the drop averaged over ops) as a share of the score; the mean clean score over ops is about 0.58, so that drop is most to all of it. Verify: `drop[top][FULL_DOSE]` over `clean` per op in the control axes data is 0.7 to 1.0 on every seed, and 1.0 on 707 and 708. -->
 The second (the lower open circle) costs as much on most seeds, but on {" and ".join(str(s) for s in MILD_SECOND)} it is a milder axis, and on 706 two reference axes cost more than it does. So ranking by energy only roughly finds the costly axes.

Leaving the syntax axes out of the reference set hardly matters: the median over all 63 axes other than e₁ gives a removal within {num(NO_EXCL, ".3f")} of the median without them, on every seed. A median of a few dozen axes is not swayed by two.

On {len(SPIKE_MOVED)} of 8 seeds ({", ".join(str(s) for s in SPIKE_MOVED)}) the `falloff` run puts its most energetic axis (the spike) on a different coordinate from the control at the same seed, at every slice, and that coordinate holds almost nothing in the control. Rotated-basis expected otherwise: it found the spike on the same coordinate in both runs on the seeds it checked, and took a move under the rotated basis as a sign that the anti term steers it.
<!-- REVIEW: rotated-basis E3 says "In the stored runs we found the spike on the same coordinate in the control and the anchored run at every seed we checked." Here the top-energy coordinate differs on the seeds listed. Verify: `energy` of `falloff-s<seed>` and `control-s<seed>` in the axes data, averaged over slices; the result holds at the last slice too. Checked against the axes data: on 706, 707 and 708 the top-energy coordinate of `falloff` differs from that of the control at each of the five slices, and the control holds under 0.02 of its energy on that coordinate; on the other five seeds it is the same coordinate at every slice. -->

The moves seen here under plain Adam suggest the spike can move without the rotation, so a change of loss alone may be enough to move it.
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
            The three baselines give nearly the same removal, about 0.8 to 0.9, on every seed but {BUSIEST}, where
            the paired one is about a tenth lower. Spill stays below the criterion under all three, and only the raw
            spill crosses it.
        """
    else:
        alt = f"""
            Under the paired baseline removal drops below half on seeds {" and ".join(str(s) for s in LOST_PAIRED_ROT)}
            and to about 0.6 on 707, while under both medians it sits near 0.9 on every seed. Spill stays near zero
            under every baseline.
        """
    return scoring_draw(data, alt)


@memo
def scoring_draw(data: dict, alt_text: str) -> str:
    which = data["which"]

    @themed(
        name=f"reference-baseline-{which}",
        alt_text=alt_text,
        caption=f"""
            **Removal and spill under each baseline**, {"the stored `falloff` runs" if which == COORD else "the rotated-basis `falloff` runs"},
            per seed. The gray circles (e₁ in the control) are the measurement quoted so far. Removal is at the full
            dose; spill is the largest over the four doses. The dashed rule is the spill criterion.
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
                ax.scatter(
                    x + offset[k],
                    data[key][k],
                    marker=MARK[k],
                    s=18,
                    color=light_dark(*INK[k]),
                    zorder=2,
                    label=names[k],
                )
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
        **Removal at the full dose by seed**, under each baseline, {"the stored `falloff` runs" if which == COORD else "the rotated-basis `falloff` runs"}.
        These are the values in the top panel of the figure above.
        """,
    )


rf"""
## Net removal and spill against the new baseline (E2)

When netted against the median of the reference axes, do the removal and spill of `falloff` change from the numbers that split-anti-seeds quoted? The main measurement is removal at the full dose per seed, shown beside spill over the four doses.

**What we saw.** Hardly at all. The mean removal goes from {num(MEAN_REM[COORD][PAIRED])} against e₁ in the control to {num(MEAN_REM[COORD][CTRL_MED])} against the median of the control, and {num(MEAN_REM[COORD][SELF_MED])} against the median of the anchored run. Every run stays within the spill criterion under every baseline.

{scoring_figure(COORD)}

{summary_table(COORD)}

The one seed that moves by more than a few hundredths is {BUSIEST}, where e₁ was among the busiest axes of the control. There the paired baseline (the gray circle in the figure above) gives a removal about a tenth lower than the medians do. Elsewhere the three baselines agree within a few hundredths, small next to the spread of removal across seeds.

{seed_table(COORD)}

The two medians agree on every seed, within {num(OWN_VS_CTRL[COORD])} of removal. So in these runs the quiet axes of an anchored run cost about what the quiet axes of its control cost, and the anchor hardly touches them.
<!-- REVIEW: "leaves them alone" softened to "hardly touches them": the median cost of the reference axes is a few thousandths higher in `falloff` than in the control on 7 of 8 seeds (0.031 to 0.040 against 0.029 to 0.036). Verify: `np.median(cost(FALLOFF, s)[reference_dirs(FALLOFF, s)])` against `REF_COST[s]`. -->

Spill falls slightly under both medians. On most seeds, e₁ in the control costs a little less on the other ops than the median does, so the paired baseline subtracts less.
<!-- REVIEW: the reason had its direction reversed ("costs a little more than the median"). Net spill is the drop on the anchored run less the baseline, so a higher spill under the paired baseline means that baseline is the smaller one. Verify: `e1_drop(CONTROL, s)[FULL_DOSE, OTHER] - baseline(CTRL_MED, s)[FULL_DOSE, OTHER]` is negative on most ops of most seeds; 705 is the exception. -->

The raw spill, with nothing subtracted, is higher (up to {num(max(RAW[COORD][s]["worst_spill"] for s in SEEDS), ".3f")}), so some baseline is still needed to separate the anchored op from the general cost of an edit.

The median of eight reference axes drawn at random, as spill-timing takes it, varies with the draw. Over many draws at each seed, the middle nine tenths of the removal spans at most {num(DRAW_WIDTH[COORD])}, about half the seed range of removal, so eight axes are enough to tell a removable run from a weak one.

Over the same draws, spill stays at or below {num(DRAW_SPILL_HI[COORD], ".3f")} on nineteen draws in twenty. The median over all the reference axes needs no draw.
"""

# %%

rf"""
## The rotated-basis runs (E3)

In the rotated-basis runs, Adam put the cheap directions along the rotated axes. With the median taken along those axes, does the baseline put the short seeds back beside the rest?

**What we saw.** Yes. Against e₁ in the control, removal falls below half on seeds {" and ".join(str(s) for s in LOST_PAIRED_ROT)}, and to about six tenths on 707. Against either median it sits near nine tenths on every seed, with every run within the spill criterion.

{scoring_figure(ROT)}

{summary_table(ROT)}

{seed_table(ROT)}

Again the two medians agree, within {num(OWN_VS_CTRL[ROT])} of removal. The median of eight drawn axes spans at most {num(DRAW_WIDTH[ROT])} in removal over the middle nine tenths of draws, a little wider than under plain Adam.

The reference axes have to be the cheap ones for the basis the model trained in. Under the rotated basis a coordinate axis is an ordinary direction and costs more, as rotated-basis found, so a median over coordinate axes would overstate the baseline there.

With an optimizer that has no privileged basis at all, every direction touches the spike, and no set of cheap reference directions exists in the same sense.
"""

# %%


def soft_figure() -> str:
    data = {
        "arms": [
            {
                "name": SOFT_SHOWN[w],
                "spill": {k: [SOFT_SPILL[w][k][s] for s in SEEDS] for k in BASELINES}
                | {"raw": [SOFT_RAW[w][s] for s in SEEDS]},
            }
            for w in SOFT
        ],
        "seeds": SEEDS,
    }
    alt = f"""
        Spill per seed for the two soft-pool arms, under each baseline and with nothing subtracted. For step, soft
        τ, the two runs over the criterion against e₁ in the control (seeds {" and ".join(str(s) for s in SOFT_OVER[STEP][PAIRED])})
        fall under it against either median. For pool 1–3, soft τ, the runs over the criterion change: seeds
        {" and ".join(str(s) for s in SOFT_OVER[POOL][PAIRED] if s not in SOFT_OVER[POOL][CTRL_MED])} fall under it,
        seed 705 rises just over it, and seeds 707 and 711 stay over under every baseline. The raw spill is above
        the criterion on most runs.
    """
    return soft_draw(data, alt)


@memo
def soft_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="reference-baseline-soft-pool",
        alt_text=alt_text,
        caption="""
            **Spill under each baseline, soft-pool runs**, per seed, for the two arms of soft-pool that sit near the
            criterion. Spill is the largest net drop on another op over the four doses. The dashed rule is the
            spill criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), layout="constrained", sharey=True)
        x = np.arange(len(data["seeds"]))
        kinds = (*BASELINES, "raw")
        offset = {k: (i - 1.5) * 0.14 for i, k in enumerate(kinds)}
        names = {**SHOWN, "raw": "nothing subtracted"}
        for ax, arm in zip(axes, data["arms"], strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            ax.axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
            for k in kinds:
                ax.scatter(
                    x + offset[k], arm["spill"][k], marker=MARK[k], s=16, color=light_dark(*INK[k]), zorder=2,
                    label=names[k],
                )  # fmt: skip
            ax.set_title(arm["name"], fontsize=8)
            ax.set_xticks(x, [str(s) for s in data["seeds"]], fontsize=7)
            ax.set_xlabel("seed", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("spill", fontsize=8)
        axes[0].set_ylim(-0.02, 0.1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False, fontsize=7)
        return fig

    return _plot()


def soft_table() -> str:
    def cell(w: str, k: str, s: int) -> str:
        v = num(SOFT_SPILL[w][k][s], ".3f")
        return f"<strong>{v}</strong>" if SOFT_SPILL[w][k][s] > CRITERION else v

    rows = [
        [f"`{SOFT_SHOWN[w]}`", SHOWN[k], *[cell(w, k, s) for s in SEEDS], f"{len(SOFT_OVER[w][k])} of {len(SEEDS)}"]
        for w in SOFT
        for k in BASELINES
    ]
    return table_html(
        ["arm", "baseline", *[str(s) for s in SEEDS], "over criterion ↓"],
        rows,
        f"""
        **Spill by seed under each baseline**, the two soft-pool arms: the largest net drop on another op over the
        four doses. Values over the criterion of {CRITERION:g} are in bold. These are the values in the figure above.
        """,
    )


rf"""
## Runs near the spill criterion (E4)

The stored `falloff` runs are all well within the spill criterion, so a change of baseline could not move a run across it. Two arms of [soft-pool](../soft-pool/report.py), `step, soft τ` and `pool 1–3, soft τ`, sit near it at the same seeds: against e₁ in the control, {len(SOFT_OVER[STEP][PAIRED])} and {len(SOFT_OVER[POOL][PAIRED])} of their 8 runs spill over it. Do those counts hold against the median? The main number is the spill over the four doses, run by run.

**What we saw.** For `step, soft τ` they do not: both runs over the criterion fall under it against either median. For `pool 1–3, soft τ` the count barely changes ({len(SOFT_OVER[POOL][PAIRED])} against e₁ in the control, {len(SOFT_OVER[POOL][CTRL_MED])} against the median of the control, {len(SOFT_OVER[POOL][SELF_MED])} against the median of the run), but the runs in it change. Seeds {" and ".join(str(s) for s in SOFT_OVER[POOL][PAIRED] if s not in SOFT_OVER[POOL][CTRL_MED])} fall under the criterion, and seed 705 rises to it: just over against the median of the control, and over by less than a thousandth against the median of the run (the bold 0.020 in the table).
<!-- REVIEW: "seed 705 rises just over it" now says it rises to the criterion. Against the own-run median it is 0.0205, and against the median over all 63 other axes (the D2.2 rule) in the run it is 0.0198, so which side it lands on is a matter of rounding. Verify: `SOFT_SPILL[POOL][SELF_MED][705]`, and `scores(median_drop(FALLOFF, 705, POOL, [i for i in range(64) if i != E1]), 705, POOL)["worst_spill"]`. -->

{soft_figure()}

{soft_table()}

Which way a run moves depends on the op that spills. On every run over the criterion, the e₁ edit costs the spilling op less in the control than the median axis does, so the paired baseline subtracts less, and most runs that sat just over the criterion fall under it. On seed 705 of `pool 1–3, soft τ` it goes the other way: e₁ in that control (among its busiest axes, E1) costs `value-hsv` more than the median does, so the paired baseline hides the spill on that op. That fits the spill-timing experiment, which tracked this run against a reference edit and saw its spill grow late on `value-hsv`, ending near the criterion. The run that spills well over it (seed 711 of `pool 1–3, soft τ`) stays over under every baseline. So does seed 707, which sat as close to the criterion as 708 and 710: against the median its largest spill is on another op, `mix`, which e₁ in the control costs more than the median axis does.
<!-- REVIEW: "The runs that spill well over it (seeds 707 and 711)" split into two. Seed 707 is at 0.028 against e₁ in the control, beside 708 (0.025) and 710 (0.027), and it stays over because the op that spills changes: `darken` against e₁ in the control (baseline 0.024 against a median of 0.037), `mix` against either median (0.018 against 0.014). "most runs over the criterion" became "every run": the paired baseline is below the median on the spilling op for all six. Verify: for each seed over the criterion, the argmax over doses and other ops of `e1_drop(FALLOFF, s, w) - baseline(k, s, w)`, and `baseline(PAIRED, s, w)` against `baseline(CTRL_MED, s, w)` at that op and dose. -->
<!-- REVIEW: the spill-timing sentence said netting against the control "hid a spill on `value-hsv` that the run has on its own". That report (PR #285, not on this branch) nets against eight reference axes, finds the late spill on `value-hsv` at 705, and says it ends just under the criterion; it does not say the control hid it. Reworded to what it saw. Verify: spill-timing E2, "The late kind grows on seeds 705 and 710 ... It ends just under the criterion on 705". -->

Removal moves as little as for `falloff`, within a hundredth or two on average.

So the spill counts of soft-pool depend on the baseline at the edge: a run within about a hundredth of the criterion can land on either side. The counts against the median of the control are the ones to compare with later experiments, since the D2.2 design now scores that way. The median there keeps the syntax axes in, which leaves these counts as they are.
<!-- REVIEW: "The counts against the median" narrowed to the median of the control, and a clause added on the design's rule, which takes the median over all 63 other axes rather than the 61 reference axes here. Under that rule the control-median counts and seeds are the same (none, and 705, 707, 711); the own-run count for `pool 1–3, soft τ` drops to 2 because 705 lands at 0.0198. Verify: `scores(median_drop(arm, s, w, [i for i in range(64) if i != E1]), s, w)["worst_spill"]` for arm in (CONTROL, FALLOFF). -->
"""

# %%

r"""
## Discussion

For the stored runs of `falloff` the paired baseline was sound, in that it rarely moved the result. Only at the edge of the spill criterion, as in the soft-pool runs of E4, did it move a count. On most seeds e₁ happened to be a quiet axis of the control. Where it was not, the error was a tenth of removal, on one seed. The numbers that split-anti-seeds quoted stand.

The median over the quiet axes removes that dependence on the draw, and it takes a few minutes more of scoring per run. Where the draw goes badly, as in the rotated-basis runs, it is the difference between a run that seems to have lost the anchor and one that removes nine tenths of the op.

The agreement of the two medians suggests a baseline needs no control at all: the quiet axes of the anchored run serve. That would matter for a recipe meant for real models, where a matched control is costly or impossible, and it is what spill-timing already does along training.

But the agreement rests on these sixteen pairs, and a recipe that disturbed the quiet axes of the anchored run would break it. So a control is still the way to check that a run-internal baseline is sound.

For later experiments this suggests a simpler scoring rule. Leaving the syntax axes in barely moves the median, so the baseline could be the median over every coordinate axis but e₁, with no ranking by energy. If it is taken in the anchored run, an experiment would need controls only now and then, to confirm that the two medians still agree, rather than one per seed. All of this assumes an optimizer like Adam, which keeps most axes quiet. Under one with no privileged basis, any reference direction touches the spike, so the baseline would be larger and an edit of any one direction dearer.
"""
