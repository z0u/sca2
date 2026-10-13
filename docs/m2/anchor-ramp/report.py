# ruff: noqa: B018
# title: An anchor weight that ramps with depth

# An exploratory report: arms that ramp the pull up with depth, beside the `falloff` recipe of record, at the
# split-anti-seeds seeds. `experiment.py` beside this script trains, measures and dose-scores the new runs; the
# `falloff` runs and the controls come from split-anti-seeds and ex-2.2.23.
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
from mini.vis import figure_html, light_dark, themed
from sca.data.ops import colors

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


SAS = ex.sas
RESULTS = fetch_json(ex.RESULTS_REF)
TRAJ = fetch_json(ex.TRAJ_REF) | fetch_json(SAS.TRAJ_REF)
DOSE = {r["label"]: r for r in fetch_json(ex.DOSE_REF)["runs"] + fetch_json(SAS.DOSE_REF)["runs"]}
# Every run measured by embedding-lean's `measure_one`: the new runs, split-anti-seeds' and ex-2.2.23's controls.
RUNS = (
    {r["label"]: r for r in RESULTS["runs"]}
    | {r["label"]: r for r in fetch_json(SAS.RESULTS_REF)["runs"]}
    | {r["label"]: r for r in fetch_json(ex.lean.RESULTS_REF)["runs"]}
)
DESIGN = RESULTS["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
GAMMAS: list[float] = DESIGN["dose_gammas"]
CRITERION = 0.02  # The selectivity criterion of ex-2.2.21: the largest net drop on another op the edit may cause.
ANSWERS = list(ex.lean.ANSWERS)  # The positions of the example answers in a context.
QUERY_ANSWER = ex.lean.QUERY_EQ + 1  # The position of the query answer, just after the query `=`.
EVERY, LATE_SET, SLICE1 = "every slice", "blocks 2 to 4", "block 1"  # Stored slice-set keys.
LIGHT = (np.array(colors()) / 15.0).mean(1)

BASE = ex.BASELINE
ARMS = (BASE, *(a.name for a in ex.ARMS))
NAME = {BASE: "falloff"} | {a.name: a.shown_as for a in ex.ARMS}
PULL = {BASE: list(ex.STEP)} | {a.name: list(a.pull) for a in ex.ARMS}
ANTI = list(ex.FALLOFF_HOLDS)
EMB_RAMP, ONE_RAMP, EMB_TOTAL, ONE_TOTAL = (a.name for a in ex.ARMS)


def label_of(arm: str, seed: int) -> str:
    return ex.label_of(arm, seed)  # split-anti-seeds labels its runs the same way


def run_of(arm: str, seed: int) -> dict:
    return RUNS[label_of(arm, seed)]


def control_of(seed: int) -> dict:
    return RUNS[ex.ex2223.label_of("control", ex.EPOCHS, seed)]


def slice_removal(arm: str, seed: int, slice_set: str) -> float:
    """Removal under the full edit on *slice_set*, net of the control at the same seed."""
    r, c = run_of(arm, seed), control_of(seed)
    net = np.array(r["edits"][slice_set]) - np.array(c["edits"][slice_set])
    return float(net[D] / (r["clean"][D] - r["null"][D]))


def dose_curves(arm: str, seed: int) -> dict[str, list[float]]:
    """Removal and spill at each dose (ex-2.2.22's scoring, at every position and slice), net of the control."""
    d, c = DOSE[label_of(arm, seed)], DOSE[f"control-s{seed}"]
    span = d["clean"]["eem"][D] - d["null"]["eem"][D]
    rem, sp = [], []
    for e, ec in zip(d["edits"], c["edits"], strict=True):
        net = np.array(e["drop"]) - np.array(ec["drop"])
        rem.append(float(net[D] / span))
        sp.append(float(net[OTHER].max()))
    return {"removal": rem, "spill": sp}


def op_change(arm: str, seed: int) -> np.ndarray:
    """The task score of each op, less that of the control at the same seed."""
    return np.array(run_of(arm, seed)["clean"]) - np.array(control_of(seed)["clean"])


def alpha_by_slice(arm: str, seed: int, key: str = "alpha_anchored", pos: list[int] | int = ANSWERS) -> np.ndarray:
    """α at the end of training on the probe contexts, per slice: averaged over the example answers by default."""
    a = np.asarray(TRAJ[label_of(arm, seed)]["traj"][key])[-1]
    return a[:, pos].mean(axis=1) if isinstance(pos, list) else a[:, pos]


def slope(axis) -> float:
    """The lean: the least-squares slope of the e₁ component of the color embeddings against lightness (0 to 1), so
    how much further along e₁ black sits than white.
    """
    return float(np.polyfit(LIGHT, np.asarray(axis, float), 1)[0])


def mean_range(values) -> tuple[float, float, float]:
    v = np.asarray(list(values), float)
    return float(v.mean()), float(v.min()), float(v.max())


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    v = list(values)
    if not v:
        return "–"
    half = 0.5 * 10 ** -int(spec[1])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(x) < half else x for x in mean_range(v))
    return f"{num(m, spec)} ({num(lo, spec)} to {num(hi, spec)})"


def frac(w: float) -> str:
    """A pull weight as the fraction it was set from."""
    for d in (1, 2, 3, 4):
        if abs(w * d - round(w * d)) < 1e-9:
            n = round(w * d)
            return str(n) if d == 1 else f"{n}/{d}"
    return f"{w:.2g}"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


INK = {
    BASE: light_dark("#2a9d8f", "#7fd8c8"),
    "ramp-emb": light_dark("#c0392b", "#ff8a76"),
    "ramp-1": light_dark("#0b3c73", "#9fcbff"),
    "ramp-emb-total": light_dark("#e08e0b", "#f5b950"),
    "ramp-1-total": light_dark("#6c4fa3", "#b9a2e8"),
}
MARK = {BASE: "D", "ramp-emb": "^", "ramp-1": "s", "ramp-emb-total": "v", "ramp-1-total": "o"}
GHOST = light_dark("#bbb", "#555")
RULE = light_dark("#333", "#ddd")
SLICE_TICK = ["emb", "1", "2", "3", "4"]
SLICE_INK = [
    light_dark(c, d)
    for c, d in (
        ("#bbb", "#666"),
        ("#9ecae1", "#36628f"),
        ("#6baed6", "#4f86b8"),
        ("#2171b5", "#8cbdeb"),
        ("#08306b", "#d6e8fa"),
    )
]

# --- Summaries the prose quotes -------------------------------------------------------------------------------

CURVES = {a: {s: dose_curves(a, s) for s in SEEDS} for a in ARMS}
FULL = {a: {s: CURVES[a][s]["removal"][-1] for s in SEEDS} for a in ARMS}
WORST_SPILL = {a: {s: max(CURVES[a][s]["spill"]) for s in SEEDS} for a in ARMS}
N_WITHIN = {a: sum(v <= CRITERION for v in WORST_SPILL[a].values()) for a in ARMS}
TASK = {a: [float(op_change(a, s).mean()) for s in SEEDS] for a in ARMS}
LOST_BELOW = 0.5  # A run whose full edit removes less than half the op lost the anchor (as in split-anti-seeds).
LOST = {a: [s for s in SEEDS if FULL[a][s] < LOST_BELOW] for a in ARMS}
KEPT = {a: [s for s in SEEDS if s not in LOST[a]] for a in ARMS}
SLOPE = {a: [slope(run_of(a, s)["emb_axis_colors"]) for s in SEEDS] for a in ARMS}
FIRST_OPERANDS = [*ex.lean.OPERANDS_A, QUERY_ANSWER - 4]  # The first operand of every example and of the query.


def weights_table() -> str:
    rows = [[f"`{NAME[a]}`", "pull"] + [frac(w) for w in PULL[a]] for a in ARMS]
    rows.append(["every arm", "anti hold"] + [f"{w:.2f}" for w in ANTI])
    return table_html(
        ["arm", "term", *SLICE_TICK],
        rows,
        r"""
        **The term weights by slice.** The pull is relative to the anchor weight $\lambda_a$ on each slice of the
        step, so `falloff` pulls slices 2 to 4 at 1. The anti hold is the weight the anti term anneals to, the same
        `falloff` line in every arm, rounded here to two places.
        """,
    )


def dose_figure() -> str:
    data = {
        "doses": GAMMAS,
        "arms": [
            {"arm": a, "name": NAME[a], "runs": [CURVES[a][s] for s in SEEDS], "lost": [s in LOST[a] for s in SEEDS]}
            for a in ARMS
        ],
    }
    alt = """
        Removal (top row) and spill (bottom row) against the dose, one column per arm. In every arm most runs climb
        to between about 0.75 and 0.9 removal at the full dose. Falloff and the ramp from the embedding at the step
        total have no runs near zero; the other three ramps each have two or three dashed runs that stay near zero.
        Spill stays near zero on most runs; a few ramp runs go over the dashed criterion at the higher doses, the
        highest a lost run of the ramp from the embedding at about 0.13.
    """
    return dose_draw(data, alt)


@memo
def dose_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="anchor-ramp-dose",
        alt_text=alt_text,
        caption="""
            **Removal and spill by dose**, the edit at every position and slice, one column per arm. One hairline per
            run, dashed where the run lost the anchor; the seed mean is bold, and the other arms are ghosts behind.
            The dashed rule is the selectivity criterion.
        """,
    )
    def _plot() -> plt.Figure:
        arms = data["arms"]
        fig, axes = plt.subplots(
            2, len(arms), figsize=(7.4, 3.8), layout="constrained", sharex=True, sharey="row", squeeze=False
        )
        for row, key in enumerate(("removal", "spill")):
            for col, arm in enumerate(arms):
                ax = axes[row, col]
                ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
                for other in arms:
                    if other is not arm:
                        for r in other["runs"]:
                            ax.plot(data["doses"], r[key], color=GHOST, lw=0.4, alpha=0.5, zorder=1)
                a = arm["arm"]
                for r, lost in zip(arm["runs"], arm["lost"], strict=True):
                    ax.plot(data["doses"], r[key], color=INK[a], lw=0.7, alpha=0.8, ls="--" if lost else "-", zorder=2)
                mean = np.mean([r[key] for r in arm["runs"]], axis=0)
                ax.plot(data["doses"], mean, color=INK[a], lw=1.7, marker=MARK[a], ms=4.5, zorder=3)
                if row == 0:
                    ax.set_title(arm["name"], fontsize=7)
                else:
                    ax.axhline(CRITERION, color=RULE, lw=0.9, ls="--", zorder=1)
                    ax.set_xlabel("dose", fontsize=8)
                ax.set_xticks(data["doses"], [f"{g:g}" for g in data["doses"]], fontsize=6)
                ax.tick_params(labelsize=7)
            axes[row, 0].set_ylabel(key, fontsize=9)
        return fig

    return _plot()


def summary_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            frac(sum(PULL[a][1:4])),
            f"{len(KEPT[a])} of {len(SEEDS)}",
            fmt(FULL[a][s] for s in KEPT[a]),
            f"{N_WITHIN[a]} of {len(SEEDS)}",
            fmt(WORST_SPILL[a].values(), ".3f"),
            fmt(TASK[a], ".3f"),
            worst_op(a),
        ]
        for a in ARMS
    ]
    return table_html(
        [
            "arm",
            "pull on 1 to 3",
            "kept the anchor ↑",
            "removal, kept ↑",
            "within criterion ↑",
            "spill, any dose ↓",
            "task change ↑",
            "worst op change ↑",
        ],
        rows,
        f"""
        **Removal, spill and task**, seed mean and range. The pull on slices 1 to 3 is the sum of the relative pull
        weights there (the step has 2). A run kept the anchor when the full edit removes at least {LOST_BELOW:g} of
        the op, and removal is over those runs, at the full dose. "Spill, any dose" is the largest net spill over
        the four doses, and "within criterion" counts the runs where it is at most {CRITERION:g}. The task change is
        the mean score over ops less that of the control, and the worst op change is the largest loss on any one op
        at any seed. Arrows mark which direction is better.
        """,
    )


def worst_op(a: str) -> str:
    v, s, op = min((float(op_change(a, s).min()), s, OPS[int(op_change(a, s).argmin())]) for s in SEEDS)
    return f"{num(v, '.3f')} (`{op}`, {s})"


def lost_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            str(s),
            num(FULL[a][s]),
            num(WORST_SPILL[a][s], ".3f"),
            num(alpha_by_slice(a, s)[-1]),
            num(alpha_by_slice(a, s, pos=FIRST_OPERANDS)[-1]),
        ]
        for a in ARMS
        for s in LOST[a]
    ]
    kept = [alpha_by_slice(a, s) for a in ARMS for s in KEPT[a]]
    return table_html(
        ["arm", "seed", "removal", "spill, any dose", "α, example answers", "α, first operands"],
        rows,
        f"""
        **The runs that lost the anchor.** Removal at the full dose, the largest net spill at any dose, and α at
        slice 4 at the end of training on `{ex.ANCHORED_OP}` probe contexts, at the example answers and at the first
        operand of each equation. On the runs that kept the anchor, α at the example answers is
        {fmt((k[-1] for k in kept))} at slice 4.
        """,
    )


def schedule_of(arm: str) -> dict:
    """The pull weight on each slice along training (the scheduled λ_a times the relative pull), which every seed of
    an arm shares.
    """
    t = TRAJ[label_of(arm, SEEDS[0])]["traj"]
    w = np.asarray(t["weight"])
    return {"epoch": t["epoch"], "pull": [(w * p).tolist() for p in PULL[arm]]}


def draw_pull(ax: plt.Axes, sched: dict, label: bool = False) -> None:
    """A strip of the pull weight on each slice under an epoch chart, one line per pulled slice, shaded by depth."""
    for sl, w in enumerate(sched["pull"]):
        if max(w) > 0:
            ax.plot(sched["epoch"], w, color=SLICE_INK[sl], lw=1.1, label=f"slice {sl}" if label else None)
    ax.set_ylim(0, 0.16)
    ax.set_yticks([0, 0.1, 0.15])
    ax.tick_params(labelsize=7)


def track_figure() -> str:
    data = {
        "arms": [
            {
                "arm": a,
                "name": NAME[a],
                "schedule": schedule_of(a),
                "runs": [
                    {
                        "epoch": TRAJ[label_of(a, s)]["traj"]["epoch"],
                        "alpha": np.asarray(TRAJ[label_of(a, s)]["traj"]["alpha_anchored"])[:, 4][:, ANSWERS]
                        .mean(axis=1)
                        .tolist(),
                        "lost": s in LOST[a],
                    }
                    for s in SEEDS
                ],
            }
            for a in ARMS
        ]
    }
    alt = """
        Alignment at the example answers at slice 4 against epoch, one column per arm. Most runs rise to about 0.55
        within the first 20 to 40 epochs and settle near 0.64. The dashed lost runs rise to at most about 0.2 early
        and fall back to near zero for the rest of training. Two runs of the ramp from the embedding climb late,
        reaching the others only around epoch 230. Underneath, the pull weight on each slice: one level for falloff,
        and evenly spaced levels rising with depth for the ramps.
    """
    return track_draw(data, alt)


@memo
def track_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="anchor-ramp-track",
        alt_text=alt_text,
        caption=f"""
            **Alignment at the example answers along training**, at slice 4, on the probe contexts of
            `{ex.ANCHORED_OP}`, one column per arm. One line per run, dashed where the run lost the anchor; the other
            arms are ghosts behind. The strips underneath are the pull weight on each slice, darker with depth.
        """,
    )
    def _plot() -> plt.Figure:
        arms = data["arms"]
        fig, axes = plt.subplots(
            2,
            len(arms),
            figsize=(7.4, 3.4),
            layout="constrained",
            sharex=True,
            sharey="row",
            height_ratios=[3, 1],
            squeeze=False,
        )
        for col, arm in enumerate(arms):
            ax, wax = axes[0, col], axes[1, col]
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for other in arms:
                if other is not arm:
                    for r in other["runs"]:
                        ax.plot(r["epoch"], r["alpha"], color=GHOST, lw=0.4, alpha=0.4, zorder=1)
            for r in arm["runs"]:
                ax.plot(r["epoch"], r["alpha"], color=INK[arm["arm"]], lw=0.8, ls="--" if r["lost"] else "-", zorder=2)
            ax.set_title(arm["name"], fontsize=7)
            ax.set_ylim(-0.05, 0.8)
            ax.tick_params(labelsize=7)
            draw_pull(wax, arm["schedule"], label=True)
            wax.set_xlabel("epoch", fontsize=8)
        axes[0, 0].set_ylabel("α, example answers", fontsize=8)
        axes[1, 0].set_ylabel("pull", fontsize=8)
        seen: dict[str, Any] = {}
        for wax in axes[1]:
            for h, lab in zip(*wax.get_legend_handles_labels(), strict=True):
                seen.setdefault(lab, h)
        labels = sorted(seen)
        fig.legend(
            [seen[k] for k in labels], labels, loc="outside lower center", ncols=len(labels), frameon=False, fontsize=7
        )
        return fig

    return _plot()


def depth_figure() -> str:
    data = {
        "anti": ANTI,
        "arms": [
            {
                "arm": a,
                "name": NAME[a],
                "pull": PULL[a],
                "anchored": [alpha_by_slice(a, s).tolist() for s in KEPT[a]],
                "other": [alpha_by_slice(a, s, "alpha_other").tolist() for s in KEPT[a]],
            }
            for a in ARMS
        ],
    }
    alt = """
        Alignment at the example answers by slice for the runs that kept the anchor. On difference, every arm rises
        from near zero at the embedding to about 0.64 at slice 4, nearly on top of one another, with the ramps from
        slice 1 a little lower at slices 1 and 2. On the other ops every arm peaks at slice 1, at about 0.15 to 0.19,
        and stays below 0.1 after. Underneath, the pull weight by slice (a step for falloff, straight lines for the
        ramps), and the anti hold falling from 0.2 at the embedding to 0.03 at slice 4.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="anchor-ramp-depth",
        alt_text=alt_text,
        caption=f"""
            **Alignment at the example answers by slice**, at the end of training, on the probe contexts of
            `{ex.ANCHORED_OP}` (left) and of the other ops (right), over the runs that kept the anchor: the seed mean
            in the arm's marker. Underneath are the pull weight at each slice, relative to the step, and the anti
            hold, which every arm shares.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            3, 2, figsize=(6.4, 4.2), layout="constrained", sharex=True, height_ratios=[3, 1.2, 0.8]
        )
        x = np.arange(len(SLICE_TICK))
        for col, key in enumerate(("anchored", "other")):
            ax, pax, aax = axes[:, col]
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for arm in data["arms"]:
                a = arm["arm"]
                ax.plot(
                    x,
                    np.mean(arm[key], axis=0),
                    color=INK[a],
                    lw=1.4,
                    marker=MARK[a],
                    ms=4,
                    zorder=3,
                    label=arm["name"],
                )
                pax.plot(x, arm["pull"], color=INK[a], lw=1.1, marker=MARK[a], ms=3)
            aax.plot(x, data["anti"], color=RULE, lw=1.1, marker=".", ms=4)
            ax.set_ylim(-0.03, 0.75 if key == "anchored" else 0.25)
            pax.set_ylim(-0.05, 1.6)
            pax.set_yticks([0, 1])
            aax.set_ylim(0, 0.24)
            aax.set_yticks([0, 0.2])
            aax.set_xticks(x, SLICE_TICK)
            aax.set_xlabel("slice", fontsize=8)
            for a_ in (ax, pax, aax):
                a_.tick_params(labelsize=7)
        axes[0, 0].set_ylabel(f"α, {ex.ANCHORED_OP}", fontsize=8)
        axes[0, 1].set_ylabel("α, other ops", fontsize=8)
        axes[1, 0].set_ylabel("pull", fontsize=8)
        axes[2, 0].set_ylabel("anti", fontsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


def depth_table() -> str:
    rows = [
        [f"`{NAME[a]}`"]
        + [num(float(np.mean([alpha_by_slice(a, s)[sl] for s in KEPT[a]]))) for sl in range(1, 5)]
        + [num(float(np.mean([alpha_by_slice(a, s, "alpha_other")[sl] for s in KEPT[a]]))) for sl in (1, 2)]
        + [fmt(SLOPE[a])]
        for a in ARMS
    ]
    rows.append(["controls", "–", "–", "–", "–", "–", "–", fmt(slope(control_of(s)["emb_axis_colors"]) for s in SEEDS)])
    return table_html(
        ["arm", "α at 1", "α at 2", "α at 3", "α at 4", "α other ops, 1", "α other ops, 2", "lightness slope"],
        rows,
        f"""
        **Alignment by slice and the lean of the color table.** α at the example answers on `{ex.ANCHORED_OP}` and on
        the other ops, seed mean over the runs that kept the anchor. The lightness slope is how much further along e₁
        black sits than white in the color embedding table, seed mean and range over every run.
        """,
    )


# %%

rf"""
# An anchor weight that ramps with depth

/// tip |
<!-- lede -->
We tried replacing the step in the pull weight (nothing on the embedding or slice 1, full weight on slices 2 to 4) with a weight that rises in a straight line with depth. With the step weight kept at slice 4, the ramps lost the anchor on more seeds than the step did. Scaled up to the step total, the ramp from the embedding kept the anchor on every seed, but it spilled a little more, and the ramp from slice 1 still lost it.
///

The recipe of record ([`falloff`](/docs/m2/split-anti-seeds/report.py)) sets the anti weight $\lambda_{{\bar s}}$ by slice, falling in a straight line from the embedding to slice 4, so no slice boundary has to be chosen for it. The pull still has a boundary: it acts on slices 2 to 4 at the full anchor weight $\lambda_a$, and not at all on the embedding or slice 1.

The {len(ex.ARMS)} new arms, of {len(SEEDS)} runs each, try the mirror image of the falloff for the pull. They use the seeds of the split-anti-seeds experiment, so its `falloff` runs serve as the baseline with no new training.
"""

# %%

rf"""
## Observations

- [Removal and spill (E1)](#removal-and-spill-e1): every ramp but one lost the anchor on two or three seeds, where `falloff` lost it on none. `{NAME[EMB_TOTAL]}` kept it on every seed, with two runs over the spill criterion. The runs that kept the anchor removed about as much as `falloff`, and losses are fewer where the pull on slices 1 to 3 is higher.
- [When the ramped runs lose the anchor (E2)](#when-the-ramped-runs-lose-the-anchor-e2): the example answers on a lost run never come onto e₁, from the first tens of epochs, while the pull is still warming up.
<!-- REVIEW: the "losses are fewer where the pull on slices 1 to 3 is higher" line moved from E2 to E1, which is the
section that shows it; E2 shows the timing only. -->
- [Where the anchor sits by depth (E3)](#where-the-anchor-sits-by-depth-e3): on the runs that kept it, the alignment by slice is nearly the same in every arm, and the pull on slice 1 brings back neither the drift on the other ops nor the lean in the color table.

## Scope

This is an exploratory study, with no preregistration and no gate. The new arms differ from `falloff` only in the pull weight by slice. The first two reach the weight of the step at slice 4, and differ in where they start from zero:

- `{NAME[EMB_RAMP]}`: zero at the embedding, so slice 1 is pulled at a quarter of the slice-4 weight, slice 2 at a half and slice 3 at three quarters.
- `{NAME[ONE_RAMP]}`: zero at slice 1, so the pull starts on slice 2 as the step does, at a third of the slice-4 weight, and slice 3 at two thirds.

Matching the step at slice 4 means each of these pulls less in total than the step: five sixths as much for `{NAME[EMB_RAMP]}`, and two thirds for `{NAME[ONE_RAMP]}`. When the first results showed both losing the anchor more often than `falloff`, we added two arms with the same shapes scaled to the step total (`{NAME[EMB_TOTAL]}` and `{NAME[ONE_TOTAL]}`), which pull slice 4 harder than the step does. These were not in the plan.
<!-- REVIEW: the total-matched arms were added after the first results, to separate the shape of the ramp from the lower total pull; the plan named that as the first alternative reading of a removal drop. -->

{weights_table()}

The anti weight is the falloff in every arm, so the ramps from the embedding pull slice 1 against a hold of about 0.16.

[The late-pull experiment](/docs/m2/late-pull/report.py) moved the pull off the embedding and slice 1 to keep the lightness lean off the embedding table, and found that slice 1, unpulled, came to hold part of the concept. A pull there might bring some of the lean back; E3 looks for it. No pair of arms differs at slice 1 alone, so an effect there is read beside slices 2 and 3.

Every run trains at model seeds {SEEDS[0]} to {SEEDS[-1]} for 400 epochs, paired by seed with its `falloff` twin and with ex-2.2.23's control (same initialization, batches and label draws). With eight seeds per arm, one or two lost runs could come from the seeds alone, so we weigh the counts across arms together.

The edits on slice 1 alone and on slices 2 to 4 alone, which the plan gave a section to, are left out. On the runs that lost the anchor every edit removes almost nothing, except on seed 710 of `{NAME[ONE_RAMP]}`, where the slice-1 edit removes about half, as the full edit nearly does. On the others they look like those of `falloff`, with the slices-2-to-4 edit again falling short on seed 710.
<!-- REVIEW: "every edit removes almost nothing" excepted seed 710 of ramp from 1: slice_removal(ONE_RAMP, 710, SLICE1)
is 0.52 and the full dose removes 0.44, just under the 0.5 line. -->

## The measurements

The measurements and dose scoring are those of the split-anti-seeds experiment.

- Removal: the net drop in the score on `difference` under the edit, as a share of the way from the clean score to the target null. The edit projects e₁ out of the state at every position and slice, at a dose of a quarter, a half, three quarters, or all of the e₁ component. Higher is better.
- Spill: the largest net drop on another op under the edit. A run is within the criterion when its spill is at most {CRITERION:g} at every dose.
- Both are net of the control at the same seed.
- A run lost the anchor when the full edit removes less than half of the op, as in the split-anti-seeds experiment.
- Task change: the mean score over ops, less that of the control. Zero means no damage.
- The alignment α with e₁ at the example answers, at every slice, and the lightness slope of the color embedding table (how much further along e₁ black sits than white).
"""

# %%

rf"""
## Removal and spill (E1)

Does a pull that ramps with depth remove as much of the op as the step, and keep the edit as selective? The main measure is the removal at the full dose, with the spill at any dose beside it.

{dose_figure()}

Most ramped runs look like `falloff`: removal climbs with the dose to between three quarters and nine tenths of the op, and the spill stays near zero. But three of the five arms have runs that remove little at any dose (the dashed lines), all but one of them almost nothing. Those runs lost the anchor.

{summary_table()}

`{NAME[EMB_TOTAL]}` is the only ramp that kept the anchor on every seed, and its removal matches that of `falloff`. Two of its runs went over the spill criterion, one by a little and one by about twice the criterion, where `falloff` stayed within it on every seed.

The two ramps from slice 1 lost the anchor on {len(LOST[ONE_RAMP])} and {len(LOST[ONE_TOTAL])} seeds, so scaling that shape up to the step total did not fix it. Scaling it up also pulls slice 4 half again as hard as the step, and one of its runs (seed 705) dropped by more than 0.2 on `value-hsv` against its control, and spilled over the criterion. That is one run, so we take it as a sign to watch rather than a cost of the shape.

Across the five arms, the count of lost runs follows the pull on slices 1 to 3 (the second column): none at 1.8 and 2, two at 1.5, three at 1. Eight seeds per arm is too few to call this a trend, and we picked that sum after seeing the counts.

The pull on slices 2 and 3 alone is the same in the two step-total arms, which lost none and two, so the ordering rests on the quarter-weight pull on slice 1. Still, it fits a simpler reading than the shape of the ramp: the pull on the slices before the last one is what keeps the anchor.
<!-- REVIEW: added that the sum over slices 1 to 3 was picked after the counts, and what in the data the ordering
rests on, so the Discussion weighs the second reading as post hoc. Verify: the pull on slices 2 and 3 alone is 1.5 in
both step-total arms, which lost 0 and 2. -->

{lost_table()}

One of these runs is borderline: seed 710 of `{NAME[ONE_RAMP]}` removes a little under half the op, though its example answers are off e₁ at every slice, so it is a weak anchor rather than none. We count it as lost by the rule of the split-anti-seeds experiment.

The lost runs are not the same seeds in every arm: seed 706 and seed 708 each lost the anchor in two arms, and the rest in one. On most of them the example answers stay off e₁, and the query answer aligns as usual, as it does on every run that kept the anchor. This is the same kind of loss seen with the uniform holds of the split-anti-seeds experiment.

One run of `{NAME[EMB_RAMP]}` (seed 709) settled on the first operands instead, and its query answer is off e₁ too. On that run the edit removes nothing from `difference`. It spills onto the other ops instead, since every op reads its operands.
<!-- REVIEW: "the query answer, which aligns on every run" was wrong for seed 709 of ramp from emb (α −0.01 at the query
answer at slice 4); on every other run it is 0.96 to 1.0. Verify: alpha_by_slice(a, s, pos=QUERY_ANSWER)[-1]. -->
"""

# %%

rf"""
## When the ramped runs lose the anchor (E2)

When does a ramped run lose the anchor: early, as the pull first arrives, or later, as the anchor drifts off? The main measure is α at the example answers, at slice 4, along training.

{track_figure()}

The lost runs part ways with the others in the first tens of epochs. On a kept run, α at the example answers climbs past 0.4 by about epoch 20 to 40, while the pull is still warming up. On a lost run it rises a little, falls back to near zero, and stays there at full pull weight. So the outcome is set while the pull arrives, and the long hold after it does not recover it.

`{NAME[EMB_RAMP]}` also has two kept runs that climb late: they first crossed 0.4 only at about epoch 60 to 80, and stayed below the others until about epoch 230. In `falloff` one run dips briefly, and none climbs late.

No other arm has a run like these, including the two ramps from slice 1, which pull slices 2 and 3 less. So slow climbs do not follow the pull sum the way the losses do, and two runs in one arm may be the seeds.
<!-- REVIEW: "the ramps also have more runs that climb late" narrowed to ramp from emb, and the "fits the reading of E1"
sentence replaced: the first crossing of 0.4 on kept runs is at epoch 20 to 48 in every other arm (falloff reaches 40
at seed 711), and only ramp from emb seeds 710 and 711 cross later (76 and 60). Verify: the first epoch at which the
slice-4 α at the example answers exceeds 0.4, per run. -->
"""

# %%

rf"""
## Where the anchor sits by depth (E3)

On the runs that kept the anchor, does the alignment by slice follow the pull weight, and does a pull on slice 1 bring back the drift onto e₁ on the other ops, or the lean in the color table?

{depth_figure()}

The alignment by slice is nearly the same in every arm, though the pull weights differ a lot. On `difference`, α at the example answers rises from near zero at the embedding to about 0.64 at slice 4 in every arm. The two ramps from slice 1 sit a little lower at slices 1 and 2. So where the anchor sits follows the step at slice 4, and the lighter pull earlier does not move it.

On the other ops the alignment peaks at slice 1 in every arm. The ramps from the embedding sit about where `falloff` does there, though `falloff` does not pull slice 1, and the ramps from slice 1 sit a little lower. So the pull on slice 1 adds little to the drift on the other ops.

{depth_table()}

The lean in the color table stays small in every arm, close to that of `falloff` and well inside the spread of the controls. Within that, it grows with the pull on the first slices: the ramps from slice 1 lean least, then `falloff`, then the ramps from the embedding, and every arm leans the same way on every seed, where the controls lean either way.

But these are differences of a few hundredths. Before [the late-pull experiment](/docs/m2/late-pull/report.py) moved the pull off the embedding, the recipe had a slope of about −1. So a light pull on slice 1 does not bring that lean back.
<!-- REVIEW: added the ordering of the small lean by the early pull (claims review). Verify: the slope column of the table and the per-seed slopes. -->
"""

# %%

rf"""
## Discussion

A ramp with the step weight at slice 4 changes how often a run anchors at all, rather than how well: it loses the anchor on more seeds than the step, and the runs that keep it end up much like `falloff`.

Two readings fit. In one, the shape of the ramp matters: pulling slice 4 harder than the others makes it easier for the pull to settle on the query answer. (The query answer aligns at slice 4 on every run but the one that settled on the operands.) In the other, what matters is how much pull the earlier slices get, and the shape matters only through that.

The losses follow the pull on slices 1 to 3 across all five arms, which favors the second reading, but the counts are small, that sum was chosen after the counts were in, and the arms differ in more than it.
<!-- REVIEW: "aligns on every run" narrowed: on seed 709 of ramp from emb the query answer is off e₁ (α −0.01 at
slice 4) as well as the example answers. Verify: alpha_by_slice(EMB_RAMP, 709, pos=QUERY_ANSWER). -->

If the second reading holds, a ramp could keep the anchor as long as the earlier slices get as much pull as the step gives them, and `{NAME[EMB_TOTAL]}` is close to that. It then has no slice boundary for the pull, and its cost is the spill on two of eight seeds. The step in `falloff` gives the earlier slices more, at no cost we could see here.
"""

# %%

r"""
## Method

The per-slice pull weight is `AnchorSpec.slice_w`, added for this experiment. It multiplies each slice's pooled term inside the mean over the pulled slices, as the per-slice `AntiSpec.hold` does for the anti term, and the arm module converts the weights relative to the step into that form. Training follows the split-anti-seeds experiment unchanged.

The total-matched arms ran after the first two, with the same seeds and everything else unchanged. The 32 new runs cost about \$11.
"""
