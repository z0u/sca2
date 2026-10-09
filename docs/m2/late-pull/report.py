# ruff: noqa: B018
# title: Pulling the later blocks only

# Twelve new runs: two arms that pull blocks 2 to 4 only, with the anti-subspace term on every slice, and the recipe
# trained again at the same four seeds for its trajectory. `experiment.py` beside this script trains and measures every
# run; the twins and controls at the same seeds come from ex-2.2.23, as embedding-lean measured them.
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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


RESULTS = fetch_json(ex.RESULTS_REF)
TRAJ = fetch_json(ex.TRAJ_REF)
LEAN = {r["label"]: r for r in fetch_json(ex.lean.RESULTS_REF)["runs"]}
"""Embedding-lean's measurements of ex-2.2.23, for the twins and controls at the same seeds."""
DESIGN = RESULTS["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
ARMS = ("whole", "late", "late-clean")
RUNS = {r["label"]: r for r in RESULTS["runs"]}
CRITERION = 0.02  # The selectivity criterion of ex-2.2.21: the largest net drop on another op the edit may cause.
RGB = np.array(colors()) / 15.0
LIGHT = RGB.mean(1)
EVERY, LATE_SET, BLOCK1, EMB = "every slice", "blocks 2 to 4", "block 1", "embedding"


def runs_of(arm: str) -> list[dict]:
    return [RUNS[ex.label_of(arm, s)] for s in SEEDS]


def twin_of(r: dict) -> dict:
    return LEAN[ex.ex2223.label_of("anchor", ex.EPOCHS, r["model_seed"])]


def control_of(r: dict) -> dict:
    return LEAN[ex.ex2223.label_of("control", ex.EPOCHS, r["model_seed"])]


CONTROLS = [control_of(r) for r in runs_of("whole")]


def net_drop(r: dict, slice_set: str) -> np.ndarray:
    """The drop in task score per op under the edit on *slice_set*, net of the control at the same seed."""
    return np.array(r["edits"][slice_set]) - np.array(control_of(r)["edits"][slice_set])


def removal(r: dict, slice_set: str = EVERY) -> float:
    """The share of the way from the clean score to the target null that the net drop on the anchored op gets."""
    return float(net_drop(r, slice_set)[D] / (r["clean"][D] - r["null"][D]))


def spill(r: dict, slice_set: str = EVERY) -> float:
    """The largest net drop on another op."""
    return float(net_drop(r, slice_set)[OTHER].max())


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
    half = 0.5 * 10 ** -int(spec[1])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(v) < half else v for v in mean_range(values))
    return f"{num(m, spec)} ({num(lo, spec)} to {num(hi, spec)})"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def ink(arm: str) -> str:
    return {
        "whole": light_dark("#c0392b", "#ff8a76"),
        "late": light_dark("#1f6fb2", "#7ab8f5"),
        "late-clean": light_dark("#2a9d8f", "#7fd8c8"),
    }[arm]


MARK = {"whole": "o", "late": "^", "late-clean": "s"}
SLICE_INK = [
    light_dark(c, d)
    for c, d in (
        ("#999", "#777"),
        ("#e08e0b", "#f5b950"),
        ("#1f6fb2", "#7ab8f5"),
        ("#6c4fa3", "#b9a2e8"),
        ("#2a9d8f", "#7fd8c8"),
    )
]
SLICE_MARK = ["o", "D", "^", "s", "v"]
SLICE_NAME = ["emb", "block 1", "block 2", "block 3", "block 4"]

# A few summaries the prose quotes.
TWIN_GAP = max(max(abs(a - b) for a, b in zip(r["clean"], twin_of(r)["clean"], strict=True)) for r in runs_of("whole"))
"""The `whole` arm against its twins in ex-2.2.23: same code, seed, batches and label draws."""
SLOPE = {a: mean_range(slope(r["emb_axis_colors"]) for r in runs_of(a)) for a in ARMS}
SLOPE_UNLATCHED = mean_range(slope(r["emb_axis_colors"]) for r in runs_of("whole") if r["latched"] is None)
SLOPE_CONTROL = mean_range(slope(c["emb_axis_colors"]) for c in CONTROLS)
LATCHED = [r["model_seed"] for r in runs_of("whole") if r["latched"] is not None]
R_S1 = {a: mean_range(r["r_light_states_other"][1] for r in runs_of(a)) for a in ARMS}
N_WITHIN = {a: sum(spill(r) <= CRITERION for r in runs_of(a)) for a in ARMS}
WORST_CLEAN = max(spill(r) for r in runs_of("late-clean"))
TASK = {a: mean_range(np.mean(r["clean"]) for r in runs_of(a)) for a in ARMS}
TASK_CONTROL = mean_range(np.mean(c["clean"]) for c in CONTROLS)
LATE_LOW = {a: [r["model_seed"] for r in runs_of(a) if removal(r, LATE_SET) < 0.5] for a in ARMS}
EMB_SPILLERS = [r["model_seed"] for r in runs_of("late") if spill(r, EMB) > CRITERION]


def seeds_text(seeds: list[int]) -> str:
    """A list of seeds in words: 'seed 700', 'seeds 700 and 702', 'seeds 700, 701 and 703'."""
    listed = " and ".join(", ".join(str(s) for s in seeds).rsplit(", ", 1))
    return f"seed {listed}" if len(seeds) == 1 else f"seeds {listed}"


rf"""
# Pulling the later blocks only

/// tip |
<!-- lede -->
Pulling only blocks 2 to 4 cut the lean of the color table to about a quarter. Holding the table off e₁ by a hard constraint went further: the spill of the full edit fell to about the selectivity criterion, with removal and the task as before. In both new arms the concept settled partly at block 1, which nothing pulls, and most of the remaining spill comes from there.
///

[Embedding-lean](/docs/m2/embedding-lean/report.py) found that every anchored run of ex-2.2.23 leans lightness onto e₁ in its color embedding table. At the embedding there is no context yet for the pull to use, so the pull settles for a token-level stand-in: darker colors are the nearest one for `{ex.ANCHORED_OP}` answers. Editing only blocks 2 to 4 of those runs removed most of `{ex.ANCHORED_OP}` with almost no spill. Editing the embedding alone spilled as much as editing every slice.

Pulling every slice but the embedding had been tried before (ex-2.2.21 and the τ × λ_a sweep). But those runs also left the anti-subspace term off the embedding, and the table leaned all the same.

So this experiment trains twelve runs at {ex.EPOCHS} epochs and {len(SEEDS)} seeds:

- `late`: the pull on blocks 2 to 4 only, and the anti-subspace term on every slice, so the first two slices are asked to stay off e₁ rather than left alone.
- `late-clean`: the same, and after every step the e₁ component of every embedding is set to zero, so the table cannot lean at all. The readout table is untied on this recipe (a separate matrix from the embedding table) and stays free.
- `whole`: the recipe of record, every slice pulled, trained again to record the alignment at every slice and position along training.

Each run shares its seed, and so its initialization, batches and label draws, with one anchored run and one control of ex-2.2.23. The anchor weight, the pool temperature τ and both schedules are the recipe values.
"""

# %%

rf"""
## Observations

- [The color table (E1)](#the-color-table-e1): in `late` the table leans the same way as on the recipe on every seed, where the controls lean either way, but only about a quarter as far. The lightness in the states at block 1 fades to about half.
- [Removal and spill (E2)](#removal-and-spill-e2): the full edit removes as much in both new arms as on the recipe. Its spill falls in `late`, and falls to about the criterion in `late-clean`. In both arms, editing block 1 alone removes nearly everything.
- [Where the alignment settles (E3)](#where-the-alignment-settles-e3): in the new arms the alignment of `{ex.ANCHORED_OP}` contexts grows at blocks 2 to 4 early in training, as the pull asks. Block 1 follows more slowly, on other ops as well as on `{ex.ANCHORED_OP}`.

## Scope

This is an exploratory study, with no preregistration and no gate. Four seeds per arm can show a large change in the lean or the spill, against the seed range, but not a small one. The `whole` arm reproduces its twins in ex-2.2.23 (the task scores of each pair agree to within {TWIN_GAP:.1g}), so a difference between arms at one seed comes from the change of pull, not from nondeterminism in training.

The anchor weight is the recipe value, and the anti-subspace weight keeps its recipe schedule. That schedule is a multiple of the anchor weight: about two and a half times it early in training, annealing to a third of it by about the midpoint.
<!-- REVIEW: said that the anti schedule anneals, which the earlier text ("a multiple of the anchor weight") left
out. The stored trajectory has anti_weight 0.25 at epoch 20, 0.12 at 200, 0.04 at 300 and 0.03 after, against an
anchor weight of 0.1; E1 and E3 now name this as a reading of the second-half growth. Verify: traj["anti_weight"]. -->
<!-- REVIEW: "noise in training" narrowed to "nondeterminism in training", since the twin check shows runs reproduce
bit for bit at a seed, not that the seed range is small; the sentence before covers that. -->

The anchor term is a mean over the pulled slices, so at the same weight each of blocks 2 to 4 is pulled about five thirds as hard in the new arms as in `whole`. The anti term is a mean over the same five slices in every arm, so it is unchanged.

## The measurements

The measurements are those of embedding-lean, taken on every run:

- The lean of the color table: how much further along e₁ the color embeddings sit for black than for white, as the slope of the e₁ component against lightness over the {len(LIGHT)} colors. It is zero in `late-clean`, where the table is held off e₁.
- The lean in the states: the correlation of α with lightness at each slice, over the color positions of contexts of other ops.
- Removal and spill of the edit, which projects e₁ out of the state at every position on a set of slices. Both are net of the control at the same seed, and the selectivity criterion is a spill of {CRITERION:g}.
- The task score of each op, as expected exact match on held-out contexts.
- Along training, every {ex.TRAJ_STRIDE_EPOCHS} epochs: the mean alignment α with e₁ at every slice and position of the probe contexts, for `{ex.ANCHORED_OP}` and for the other ops, and the e₁ component of every color embedding.
"""

# %%


def table_figure() -> str:
    seed = SEEDS[0]
    data = {
        "panels": [
            {"title": f"{a}, seed {seed}", "axis": RUNS[ex.label_of(a, seed)]["emb_axis_colors"]}
            for a in ("whole", "late")
        ],
        "traj": {
            a: [
                {
                    "epoch": TRAJ[ex.label_of(a, s)]["traj"]["epoch"],
                    "slope": [slope(x) for x in TRAJ[ex.label_of(a, s)]["traj"]["emb_axis_colors"]],
                }
                for s in SEEDS
            ]
            for a in ("whole", "late")
        },
        "controls": [slope(c["emb_axis_colors"]) for c in CONTROLS],
    }
    alt = f"""
        Dark colors sit far along e₁ on the recipe (up to about 0.9) but only about a third as far with the late
        pull at seed {seed}; along training the recipe slope falls early to about −1 on most seeds, while the late
        slope stays flat for a hundred epochs, then falls to about −0.25, inside the spread of the controls.
    """
    return table_draw(data, alt)


@memo
def table_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="late-pull-table",
        alt_text=alt_text,
        caption=f"""
            **The lean of the color embedding table.** Left and middle: the e₁ component of each of the {len(LIGHT)}
            color embeddings against its lightness at the end of training, one run of each arm, each dot drawn in its
            own color. Right: the slope of that component against lightness along training, one hairline per run
            and the seed mean bold; the ticks at the right edge are the controls at the same seeds. `late-clean` is
            zero throughout and not drawn.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), layout="constrained")
        for ax, panel in zip(axes[:2], data["panels"], strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            ax.scatter(LIGHT, panel["axis"], s=8, c=RGB, edgecolor=light_dark("#333", "#ddd"), linewidth=0.2, zorder=3)
            ax.set_title(panel["title"], fontsize=8)
            ax.set_xlabel("lightness of the color", fontsize=9)
            ax.set_ylim(-0.15, 1.1)
        axes[1].sharey(axes[0])
        axes[1].tick_params(labelleft=False)
        axes[0].set_ylabel("e₁ component of the embedding", fontsize=9)
        ax = axes[2]
        ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
        for arm, runs in data["traj"].items():
            for r in runs:
                ax.plot(r["epoch"], r["slope"], color=ink(arm), lw=0.4, alpha=0.6, zorder=2)
            ep = np.asarray(runs[0]["epoch"])
            mean = np.mean([r["slope"] for r in runs], axis=0)
            ax.plot(ep, mean, color=ink(arm), lw=1.6, marker=MARK[arm], markevery=10, ms=4, label=arm, zorder=3)
        end = data["traj"]["whole"][0]["epoch"][-1]
        for v in data["controls"]:
            ax.plot([end * 1.02, end * 1.07], [v, v], color=light_dark("#333", "#ddd"), lw=1.2, zorder=3)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel("slope against lightness", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        return fig

    return _plot()


def lean_table() -> str:
    rows = [
        [f"`{a}`", fmt(slope(r["emb_axis_colors"]) for r in runs_of(a))]
        + [fmt(r["r_light_states_other"][s] for r in runs_of(a)) for s in (1, 2)]
        for a in ARMS
    ]
    rows.append(
        ["control", fmt(slope(c["emb_axis_colors"]) for c in CONTROLS)]
        + [fmt(c["r_light_states_other"][s] for c in CONTROLS) for s in (1, 2)]
    )
    return table_html(
        ["arm", "slope in the table", "r at block 1", "r at block 2"],
        rows,
        f"""
        **The lean, in the table and in the states**, seed mean and range over {len(SEEDS)} runs. The slope is that
        of the e₁ component of the color embeddings against lightness; r is the correlation of α with lightness at
        the color positions of contexts of other ops. The controls are ex-2.2.23's at the same seeds.
        """,
    )


rf"""
## The color table (E1)

If the lean is the pull meeting the embedding with a stand-in, it should fade in `late`, where the pull leaves the first two slices and the anti-subspace term stays on them. A slope of −1 means black sits a whole unit further along e₁ than white.

{table_figure()}

{lean_table()}

It fades to about a quarter, though not to nothing. The slope in `late` is between {num(SLOPE["late"][2])} and {num(SLOPE["late"][1])} by seed, against about {num(SLOPE_UNLATCHED[0], ".1f")} on the recipe runs with no latch ({seeds_text(LATCHED)} of the recipe is latched on `,` and has a shallow slope too). That is about as steep as the steepest control. But the controls lean either way by seed, and every `late` run leans the same way as the recipe, so what is left is still the pull at work, only weaker.

The lightness in the states at block 1 fades too, to about half the recipe value, and in `late-clean` it is smaller again. By block 2 it is small in every arm.
<!-- REVIEW: "fades by about as much as the slope" changed to "to about half": r at block 1 goes from −0.49 (−0.6 on
the unlatched seeds) to −0.23, where the slope goes to a quarter. Verify: the lean table. -->

On the recipe the slope grows from early in training. On `late` it stays near zero for about the first hundred epochs, then grows through the second half, while the whole table drifts a little onto e₁ (E3). The pull on block 2 reaches the table only through block 1, and that is enough to load some lightness onto e₁.

The growth also tracks the anti-subspace schedule, which anneals from its peak to its hold over the same epochs (Scope), so the table may be held off e₁ while the anti term is strong and lean once it is not; this run cannot separate the two. Part of the lean may also come from the per-slice pull being five thirds of the recipe value.
<!-- REVIEW: added the anti-schedule reading of the late onset. The anti weight falls from 0.22 at epoch 100 to 0.04
at epoch 300, the window in which the late slope grows; the earlier text gave only the route through block 1.
Verify: traj["anti_weight"] against the slope panel. -->

"""

# %%


def edit_figure() -> str:
    data = {
        "panels": [
            {
                "title": name,
                "runs": [
                    {"arm": a, "removal": removal(r, name), "spill": spill(r, name)} for a in ARMS for r in runs_of(a)
                ],
            }
            for name in (EVERY, BLOCK1, LATE_SET)
        ],
    }
    alt = """
        Under the full edit every run removes about 0.85 to 0.9, and spill drops from the recipe (up to 0.38) to
        late (up to 0.17) to late-clean (around the criterion); editing block 1 alone looks much the same, while
        editing blocks 2 to 4 spills almost nothing but removes nearly everything on seven runs and almost nothing on five.
    """
    return edit_draw(data, alt)


@memo
def edit_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="late-pull-edit",
        alt_text=alt_text,
        caption=f"""
            **Spill against removal under three slice-restricted edits**, one mark per run. Removal is the net drop on
            `{ex.ANCHORED_OP}` as a share of the way to the target null; spill is the largest net drop on another op.
            The dashed rule is the selectivity criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), layout="constrained", sharey=True, sharex=True)
        for ax, panel in zip(axes, data["panels"], strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            ax.axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
            for r in panel["runs"]:
                ax.plot(
                    r["removal"],
                    r["spill"],
                    MARK[r["arm"]],
                    ms=5.5,
                    color=ink(r["arm"]),
                    mec=light_dark("white", "#111"),
                    mew=0.5,
                    zorder=3,
                )
            ax.set_title(f"edit on {panel['title']}", fontsize=8)
            ax.set_xlabel("removal", fontsize=9)
        axes[0].set_ylabel("spill", fontsize=9)
        h = [Line2D([], [], ls="none", marker=MARK[a], color=ink(a), label=a) for a in ARMS]
        fig.legend(handles=h, loc="outside upper center", ncols=len(h), frameon=False, fontsize=7)
        return fig

    return _plot()


def edit_table() -> str:
    sets = (EVERY, EMB, BLOCK1, LATE_SET)
    rows = [[f"{name}: removal"] + [fmt(removal(r, name) for r in runs_of(a)) for a in ARMS] for name in sets]
    rows += [[f"{name}: spill"] + [fmt((spill(r, name) for r in runs_of(a)), ".3f") for a in ARMS] for name in sets]
    return table_html(
        ["slices edited"] + [f"`{a}`" for a in ARMS],
        rows,
        f"""
        **Removal and spill under each slice-restricted edit**, seed mean and range over {len(SEEDS)} runs. The
        `whole` runs are the same as their twins in ex-2.2.23.
        """,
    )


rf"""
## Removal and spill (E2)

The edit on every slice is the goal, since one that has to know which slices to skip is a weaker guarantee. The main measure is its spill, with removal beside it.

{edit_figure()}

{edit_table()}

The spill falls, and removal holds. Under the edit on every slice all twelve runs remove about as much as each other. The recipe runs spill well above the criterion on every seed, and `late` spills less on every seed but stays above it. `late-clean` is within it on {N_WITHIN["late-clean"]} of {len(SEEDS)} runs, and the fourth spills about twice the criterion ({WORST_CLEAN:.3f}).

The task score is the same in all three arms and on the controls (about {TASK_CONTROL[0]:.2f} expected exact match averaged over ops, and no run more than {max(TASK_CONTROL[0] - TASK[a][1] for a in ARMS):.2f} below the controls), so the new pull costs the task nothing we can see.

In both new arms the concept sits at block 1 as well as later. Editing block 1 alone removes at least as much as editing every slice, though block 1 is never pulled, and it gives most of the spill that is left. In `late` the embedding alone still spills on {seeds_text(EMB_SPILLERS)}, where the weaker lean of E1 is still used downstream.

Editing only blocks 2 to 4 spills almost nothing in any arm, but its removal splits by seed. On seven of the twelve runs it removes nearly everything, and on the other five almost nothing ({seeds_text(LATE_LOW["late"])} in `late`, {seeds_text(LATE_LOW["late-clean"])} in `late-clean`, and {seeds_text(LATE_LOW["whole"])} in `whole`).

On those five, the answer depends on the e₁ component at block 1, and not on the e₁ component the pull put at blocks 2 to 4. Perhaps the later blocks read block 1 along e₁ and write the answer elsewhere. Or the two edits may differ in some other way that this measurement cannot tell apart. Embedding-lean saw the same split on the recipe and took it as a sign that the concept was partly built from the lean. Here the split persists with the table held off e₁, so it does not need the lean.
<!-- REVIEW: "the later blocks rebuild the concept from what block 1 passes them, so taking it out after block 1 is
too late" restated as what the two edits show (block-1 e₁ needed, blocks-2-to-4 e₁ not), with the mechanism as one
reading. "Most runs" became "seven of twelve", since in the new arms it is half. Verify: the removal rows for
block 1 and blocks 2 to 4, per seed. -->

"""

# %%


def alpha_figure() -> str:
    def per_slice(lab: str, key: str) -> list[list[float]]:
        return np.asarray(TRAJ[lab]["traj"][key]).mean(axis=2).T.tolist()  # (slice, record), mean over positions

    data = {
        "panels": [
            {
                "arm": a,
                "runs": [
                    {
                        "epoch": TRAJ[ex.label_of(a, s)]["traj"]["epoch"],
                        "anchored": per_slice(ex.label_of(a, s), "alpha_anchored"),
                        "other": per_slice(ex.label_of(a, s), "alpha_other"),
                    }
                    for s in SEEDS
                ],
            }
            for a in ARMS
        ],
    }
    alt = f"""
        On the recipe the embedding and block 1 align with e₁ on every op, not just {ex.ANCHORED_OP}; in the late arms
        blocks 2 to 4 align quickly on {ex.ANCHORED_OP} only, while block 1 (and, in late, the embedding) creeps up
        slowly on all ops through the second half of training.
    """
    return alpha_draw(data, alt)


@memo
def alpha_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="late-pull-alpha",
        alt_text=alt_text,
        caption=f"""
            **Mean alignment with e₁ along training, by slice.** The mean of α over the positions of the probe
            contexts of `{ex.ANCHORED_OP}` (top) and of the other ops (bottom), one line per slice; bold is the
            seed mean and the hairlines are the {len(SEEDS)} runs.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), layout="constrained", sharex=True, sharey=True)
        for col, panel in enumerate(data["panels"]):
            ep = np.asarray(panel["runs"][0]["epoch"])
            for row, key in enumerate(("anchored", "other")):
                ax = axes[row, col]
                ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
                for s, name in enumerate(SLICE_NAME):
                    for r in panel["runs"]:
                        ax.plot(r["epoch"], r[key][s], color=SLICE_INK[s], lw=0.3, alpha=0.5, zorder=2)
                    mean = np.mean([r[key][s] for r in panel["runs"]], axis=0)
                    ax.plot(
                        ep, mean, color=SLICE_INK[s], lw=1.4, marker=SLICE_MARK[s], markevery=12, ms=3.5, label=name
                    )
                if row == 0:
                    ax.set_title(panel["arm"], fontsize=8)
                else:
                    ax.set_xlabel("epoch", fontsize=9)
        axes[0, 0].set_ylabel(f"α, {ex.ANCHORED_OP}", fontsize=9)
        axes[1, 0].set_ylabel("α, other ops", fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


rf"""
## Where the alignment settles (E3)

A concept that settles where the pull is should show up at blocks 2 to 4 on `{ex.ANCHORED_OP}` contexts and nowhere on the others. The measure is the mean of α over the positions of a probe context, which mixes the positions where the concept sits with those where it does not, so only its shape over training and slices matters.

{alpha_figure()}

On the recipe every slice rises together on `{ex.ANCHORED_OP}` contexts, and the embedding and block 1 rise nearly as much on the other ops. That is the lean seen from the states: at those slices the alignment follows the tokens, whatever the op.

On both new arms, blocks 2 to 4 rise on `{ex.ANCHORED_OP}` contexts within the first few dozen epochs and then level off, and stay low on the other ops. Block 1 rises later and more slowly, on the other ops as well as on `{ex.ANCHORED_OP}`, though less on the others than on the recipe. So block 1 comes to meet part of the pull on block 2, and the anti term does not stop it.

Much of the rise at block 1, and nearly all of it on the other ops, comes after epoch 150, as the anti weight anneals toward its hold (Scope), so the anti term may hold block 1 off e₁ only while it is strong. Its rise on other ops is the spill that E2 traced to the edit on block 1.
<!-- REVIEW: added the schedule reading beside "the anti term does not stop it", for the same reason as in E1.
Verify: block 1 on the other ops rises from about epoch 150 in both new arms, where anti_weight is 0.18 and falling. -->


On `late` the embedding also drifts onto e₁ in the second half of training, by about as much on the other ops as on `{ex.ANCHORED_OP}`. That is the whole color table shifting a little toward e₁, with the lightness slope of E1 on top. It does not happen in `late-clean`, where the table is held at zero.
"""

# %%

rf"""
## Discussion

Moving the pull off the first two slices did most of what embedding-lean expected of it. The color table leans much less, the lightness in the states fades, and the full edit spills less on every seed. Holding the table off e₁ outright took the spill most of the rest of the way, with nothing lost in removal or in the task. That fits the account that the spill comes from stand-ins at the slices where nothing contextual exists.

It did not keep the concept out of block 1. The pull on block 2 is cheapest to meet by having block 1 already lean toward e₁, and block 1 has some context to work with, so part of what it puts there is about `{ex.ANCHORED_OP}` and part is shared with other ops.

Keeping out the shared part is the job of the anti-subspace term. On its recipe schedule it does not manage it: block 1 and the table move onto e₁ mostly after the anti weight has annealed toward its hold. Embedding-lean estimated the anti term to be an order of magnitude or more too weak per embedding. This may be the same shortfall at block 1, or the anneal may let go too early; this run does not separate the two.

If, on some seeds, the answer to `{ex.ANCHORED_OP}` depends on the e₁ component at block 1 and not on the one at the later blocks (E2), the edit on every slice is the one to keep, and the remaining spill is a question of what block 1 carries.
<!-- REVIEW: the two paragraphs above follow the E1, E2 and E3 changes: the anneal named as an alternative to
"too weak", and the split restated as what the edits show rather than as the later blocks rebuilding the concept. -->


Four seeds show that `late-clean` spills much less than the recipe, but not whether its one run above the criterion is typical.

## Glossary

Alignment
α
:   How far a state points along e₁: the cosine between the state and e₁, which is 1 on the anchor direction and 0 off it.

Anti-subspace term
:   The companion to the pull: the mean of α² over every position and slice it acts on, which asks the states as a whole to stay off e₁.

Lean
:   Lightness on e₁ in the color embedding table: darker colors sit further along the anchor direction, a stand-in for `{ex.ANCHORED_OP}` at the slices where nothing contextual exists.

Latch
:   A syntax token embedding on e₁, which satisfies the pull at the embedding for every context at once. Found by spill-by-position.

Removal
:   How much of `{ex.ANCHORED_OP}` the edit takes out: the net drop in its score under the edit, as a share of the way from the clean score to the target null.

Spill
:   The largest net drop in score on another op under the edit. The selectivity criterion is {CRITERION:g}.

Target null
:   The score an ideal predictor would get on `{ex.ANCHORED_OP}` contexts if it had lost that op and answered from the remaining ones.
"""
