# ruff: noqa: B018
# title: Where the lean comes from

# A re-analysis of ex-2.2.23's checkpoints and the no-emb trials of the τ × λ_a sweep, with no new training.
# `experiment.py` beside this script measures every run and publishes one JSON of summaries; this script nets each
# anchored run of ex-2.2.23 against the control at the same seed and length, as ex-2.2.23 did, and draws the figures.
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
DESIGN = RESULTS["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
K: int = DESIGN["k"]
SETS: list[str] = list(DESIGN["slice_sets"])
RUNS = {r["label"]: r for r in RESULTS["runs"]}
ANCHORED = [r for r in RESULTS["runs"] if r.get("source") == "ex-2.2.23" and r["condition"] == "anchor"]
CONTROLS = [r for r in RESULTS["runs"] if r.get("source") == "ex-2.2.23" and r["condition"] == "control"]
SWEEP_NO_EMB = [r for r in RESULTS["runs"] if r.get("source") == "tau-lambda-sweep" and r["condition"] == "no-emb"]
LENGTHS = sorted({r["epochs"] for r in ANCHORED}, reverse=True)
LONG, SHORT = LENGTHS
CRITERION = 0.02  # The selectivity criterion of ex-2.2.21: the largest net drop on another op the edit may cause.
RGB = np.array(colors()) / 15.0
LIGHT = RGB.mean(1)
LATCH_NAME = {None: "no latch", ",": "latched `,`", "\n": "latched ⏎", "?": "latched `?`", "=": "latched `=`"}


def control_of(r: dict) -> dict:
    return RUNS[ex.ex2223.label_of("control", r["epochs"], r["model_seed"])]


def net_drop(r: dict, slice_set: str) -> np.ndarray:
    """The drop in task score per op under the edit on *slice_set*, net of the control at the same seed and length."""
    return np.array(r["edits"][slice_set]) - np.array(control_of(r)["edits"][slice_set])


def removal(r: dict, slice_set: str = "every slice") -> float:
    """The share of the way from the clean score to the target null that the net drop on the anchored op gets."""
    return float(net_drop(r, slice_set)[D] / (r["clean"][D] - r["null"][D]))


def spill(r: dict, slice_set: str = "every slice") -> tuple[float, str]:
    """The largest net drop on another op, and the op it lands on."""
    net = net_drop(r, slice_set)
    worst = OTHER[int(np.argmax(net[OTHER]))]
    return float(net[worst]), OPS[worst]


def by_length(epochs: int) -> list[dict]:
    return [r for r in ANCHORED if r["epochs"] == epochs]


def group(epochs: int, latch: str | None) -> list[dict]:
    return [r for r in by_length(epochs) if r["latched"] == latch]


def mean_range(values) -> tuple[float, float, float]:
    v = np.asarray(list(values), float)
    return float(v.mean()), float(v.min()), float(v.max())


def fmt(values, spec: str = ".2f") -> str:
    half = 0.5 * 10 ** -int(spec[1])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(v) < half else v for v in mean_range(values))
    return f"{m:{spec}} ({lo:{spec}} to {hi:{spec}})"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def ink(epochs: int | str) -> str:
    return {
        LONG: light_dark("#c0392b", "#ff8a76"),
        SHORT: light_dark("#1f6fb2", "#7ab8f5"),
        "control": light_dark("#777", "#999"),
        "sweep": light_dark("#2a9d8f", "#7fd8c8"),
    }[epochs]


MARK = {None: "o", ",": "^", "\n": "v", "?": "D", "=": "s"}

# A few summaries the prose quotes.
LATCH_COUNTS = {e: {t: len(group(e, t)) for t in (None, ",", "\n", "?", "=") if group(e, t)} for e in LENGTHS}
N_SPILL_LONG = sum(spill(r)[0] > CRITERION for r in by_length(LONG))
N_LATE_WITHIN = sum(spill(r, "blocks 2 to 4")[0] <= CRITERION for r in by_length(LONG))
LOW_LATE = sorted(
    (r["model_seed"], removal(r, "blocks 2 to 4")) for r in group(LONG, None) if removal(r, "blocks 2 to 4") < 0.65
)
R_EMB_ANCHORED = mean_range(r["r_light_emb"] for r in ANCHORED)
R_EMB_CONTROL = mean_range(r["r_light_emb"] for r in CONTROLS)
LOW_LATE_BLOCK1 = [removal(RUNS[ex.ex2223.label_of("anchor", LONG, s)], "block 1") for s, _ in LOW_LATE]
SD_NO_LATCH = mean_range(r["emb_sd"] for r in ANCHORED if r["latched"] is None)
SD_LATCHED = mean_range(r["emb_sd"] for r in ANCHORED if r["latched"] is not None)
SD_CONTROL = mean_range(r["emb_sd"] for r in CONTROLS)
ALPHA_FIT = mean_range(r["alpha_fit_other"] for r in by_length(LONG))
ALPHA_NOFIT = mean_range(r["alpha_nofit_other"] for r in by_length(LONG))
R_DROP_NFIT = mean_range(r["r_drop_nfit_other"] for r in by_length(LONG))
R_READOUT = mean_range(abs(r["r_light_readout"]) for r in ANCHORED)
R_SWEEP_EMB = mean_range(r["r_light_emb"] for r in SWEEP_NO_EMB)

rf"""
# Where the lean comes from

/// tip |
<!-- lede -->
On the runs of ex-2.2.23, the edit that removes `{ex.ANCHORED_OP}` spills onto other ops, and the spill comes from the first two slices. At those slices the model has not yet read the context, so it meets the pull with a stand-in: either lightness loaded onto e₁ in the color embedding table, or a syntax token latched to e₁. The edit at those slices then takes the lightness away from every op. Editing only the later blocks removes `{ex.ANCHORED_OP}` with almost no spill.
///

[Spill-by-position](/docs/m2/spill-by-position/report.py) found that the removal and the spill share the example answers, and that a syntax token latched onto e₁ shapes the spill. The [τ × λ_a sweep](https://github.com/z0u/sca2/pull/260) found that the spill persists with no latch at all. That left two questions: what causes the spill when there is no latch, and why the anti-subspace term does nothing about it.

This report re-analyzes the stored checkpoints of [ex-2.2.23](/docs/m2/ex-2.2.23/report.py) ({len(ANCHORED)} anchored and {len(CONTROLS)} control runs, at {SHORT} and {LONG} epochs) and the no-emb trials of the sweep near the recipe τ ({len(SWEEP_NO_EMB)} trials), with no new training.
"""

# %%

rf"""
## Observations

- [What e₁ holds on the other ops (E1)](#what-e₁-holds-on-the-other-ops-e1): at the example answers of other ops, e₁ holds a per-example judgement of whether that one example fits `{ex.ANCHORED_OP}`. The spill does not follow it, so the spill is not the concept firing on the wrong contexts.
- [Lightness on e₁ in the embedding table (E2)](#lightness-on-e₁-in-the-embedding-table-e2): every anchored run loads lightness onto e₁ in the color embedding table, darker colors further along. The lean is about three times larger on runs with no latch, and the latch decides how the run spills.
- [The lean fades with depth (E3)](#the-lean-fades-with-depth-e3): the lean is strongest at the embedding and fades through the blocks. It is there at {SHORT} and {LONG} epochs and on the no-emb trials alike, and the readout table has none of it.
- [The edit by slice (E4)](#the-edit-by-slice-e4): editing only blocks 2 to {ex.N_SLICES - 1} removes most of `{ex.ANCHORED_OP}` with almost no spill. Editing only the embedding gives half the removal and all the spill.

## Scope

This is a re-analysis with no preregistration and no gate. It covers every run of ex-2.2.23: {len(by_length(LONG))} anchored runs at {LONG} epochs and {len(by_length(SHORT))} at {SHORT}, each with its control at the same model seed, scored on ex-2.2.21's held-out contexts. The prose quotes the {LONG}-epoch runs, since they have learned every op, and the {SHORT}-epoch runs are shown beside them.

The sweep trials are the no-emb trials with τ between {DESIGN["sweep_tau"][0]:g} and {DESIGN["sweep_tau"][1]:g}, measured on the tables and states only.

The measurements follow ex-2.2.23 and spill-by-position. The task score of an op is the expected exact match of the answer on held-out contexts of that op. Removal is the drop in the `{ex.ANCHORED_OP}` score under the edit, net of the same edit on the paired control run, as a share of the way from the clean score to the target null. Spill is the largest net drop on any other op; the selectivity criterion is {CRITERION:g}.

The full edit removes the whole e₁ component at every position and slice; E4 restricts it to a set of slices. The alignment α is the cosine of the state with e₁,[^cosine] averaged over slices 2 to {ex.N_SLICES - 1} where a section says so.

[^cosine]: Cosine similarity: how closely two vectors point the same way, ignoring length. 1 means the state lies along e₁, 0 means it is perpendicular to it.

The checkpoints are the ones the recipe was tuned on, so the seed-to-seed spread here is the spread the recipe has, with nothing held out. The slice-restricted edits of E4 are on models trained with every slice pulled, so they say where the removable part of the concept sits in those models. Only a new run can say whether a pull kept off the early slices would put the concept in the later blocks.
"""

# %%


def nfit_table() -> str:
    """Per example count on the other ops: the share of contexts, α at the query `=`, and the drop under the full edit."""
    rows = []
    long = by_length(LONG)
    n = np.array([r["n_by_nfit_other"] for r in long]).mean(0)
    for k in range(K + 1):
        drop = [r["drop_by_nfit_other"][k] for r in long if r["drop_by_nfit_other"][k] is not None]
        ctrl = [
            control_of(r)["drop_by_nfit_other"][k] for r in long if control_of(r)["drop_by_nfit_other"][k] is not None
        ]
        qeq = [r["alpha_qeq_by_nfit_other"][k] for r in long if r["alpha_qeq_by_nfit_other"][k] is not None]
        rows.append([str(k), f"{n[k] / n.sum():.0%}", fmt(qeq), fmt(drop, ".3f"), fmt(ctrl, ".3f")])
    return table_html(
        ["examples that fit", "share of contexts", "α at the query <code>=</code>", "drop, anchored", "drop, control"],
        rows,
        f"""
        **The other ops' contexts by how many of their examples fit `{ex.ANCHORED_OP}` on their own**, on the
        {LONG}-epoch runs: the share of contexts, α at the query `=` (slices 2 to {ex.N_SLICES - 1}), and the drop
        in task score under the full edit, on the anchored run and on its control. Seed mean and range.
        """,
    )


rf"""
## What e₁ holds on the other ops (E1)

Is the spill the concept itself firing on contexts that are not `{ex.ANCHORED_OP}`? At an example answer the anchor follows the posterior given that one example ([example-evidence](/docs/m2/example-evidence/report.py)), so a `mix` context whose first example happens to fit `{ex.ANCHORED_OP}` should have some alignment there.

It does, strongly. On the {LONG}-epoch runs the alignment at an example answer of another op is near {ALPHA_FIT[0]:.1f} when that example fits `{ex.ANCHORED_OP}` on its own and near {ALPHA_NOFIT[0]:.1f} when it does not, on every seed, latched or not. So e₁ at the example answers holds a clean per-example judgement, and other ops' contexts get it wherever an example fits.

{nfit_table()}

But that is not what spills. The contexts with no fitting example lose the most under the edit, and the few contexts with two or three fitting examples gain a little, so the correlation between the drop and the number of fitting examples is negative on every run (about {R_DROP_NFIT[0]:.2f}). The control loses a little under the same edit at every count, with no pattern.

So the spill comes from something the edit removes at every context of the affected op, whatever its examples say. The one place the pull reaches but no context can is the embedding table (E2).

At slices 2 to {ex.N_SLICES - 1} the alignment at the query `=` is small on the other ops' contexts, and it rises with the number of fitting examples. That fits the query position tallying the per-example judgements, though this analysis can't tell whether it is a count or a posterior.
"""

# %%


def emb_figure() -> str:
    picks = [
        (ex.ex2223.label_of("control", LONG, ANCHORED[0]["model_seed"]), "control"),
        (next(r["label"] for r in group(LONG, ",")), "anchored, latched ,"),
        (next(r["label"] for r in group(LONG, None)), "anchored, no latch"),
    ]
    data = {
        "panels": [
            {
                "title": f"{title} (seed {RUNS[lbl]['model_seed']})",
                "axis": RUNS[lbl]["emb_axis_colors"],
                "r": RUNS[lbl]["r_light_emb"],
                "sd": RUNS[lbl]["emb_sd"],
            }
            for lbl, title in picks
        ],
    }
    alt = """
        Three scatter panels of the e₁ component of each color embedding against the lightness of the color, marks
        colored by the color itself. The control is a cloud with no trend, spread between about −0.3 and 0.4. The
        anchored run latched on `,` is a narrow band that falls from about 0.2 at the dark end to 0 at the light
        end. The anchored run with no latch is a wide wedge that falls from about 0.7 at the dark end to 0 at the
        light end.
    """
    return emb_draw(data, alt)


@memo
def emb_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="embedding-lightness",
        alt_text=alt_text,
        caption=f"""
            **Lightness on e₁ in the color embedding table, three {LONG}-epoch runs.** Each mark is one of the
            {len(RGB)} color embeddings, colored by the color it names; x is the lightness of the color (its mean
            channel), y its component along e₁ after normalization. Left: a control. Middle: an anchored run latched
            on `,`. Right: an anchored run with no latch. Each panel's title gives the correlation and the standard
            deviation of the component over the table.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), layout="constrained", sharey=True)
        for ax, panel in zip(axes, data["panels"], strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            ax.scatter(LIGHT, panel["axis"], s=14, c=RGB, edgecolor=light_dark("#333", "#ddd"), linewidth=0.3, zorder=3)
            ax.set_title(f"{panel['title']}\nr = {panel['r']:+.2f}, SD = {panel['sd']:.2f}", fontsize=8)
            ax.set_xlabel("lightness of the color", fontsize=9)
            ax.set_ylim(-0.75, 0.75)
        axes[0].set_ylabel("e₁ component of the embedding", fontsize=9)
        return fig

    return _plot()


def spill_figure() -> str:
    data = {
        "runs": [
            {
                "epochs": r["epochs"],
                "latched": r["latched"],
                "sd": r["emb_sd"],
                "r_states": r["r_light_states_other"][2],
                "spill": spill(r)[0],
            }
            for r in ANCHORED
        ],
    }
    alt = """
        Two scatter panels of spill against a measure of the lean, marks shaped by the latch and colored by
        training length. Left, against the spread of the e₁ component over the color table: the comma-latched runs
        spill most at a small spread, the no-latch runs spill moderately at a large spread, and the newline-latched
        runs sit at the criterion with a small spread. Right, against the lightness correlation in the states of
        other ops at slice 2: spill rises as that correlation moves away from zero, with the newline-latched runs
        closest to zero and lowest.
    """
    return spill_draw(data, alt)


@memo
def spill_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="spill-by-lean",
        alt_text=alt_text,
        caption=f"""
            **Spill against the lean, all {len(ANCHORED)} anchored runs of ex-2.2.23.** Left: against the standard
            deviation of the e₁ component over the color embeddings. Right: against the correlation between
            lightness and α at the color positions of other ops, at slice 2. Color is the training length; the
            marker is the latch: circles have none, up-triangles are latched on `,`, down-triangles on ⏎, and the
            diamond on `?`. The dashed rule is the selectivity criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained", sharey=True)
        for ax, key in zip(axes, ("sd", "r_states"), strict=True):
            for r in data["runs"]:
                ax.plot(
                    r[key],
                    r["spill"],
                    MARK[r["latched"]],
                    ms=5.5,
                    color=ink(r["epochs"]),
                    mec=light_dark("white", "#111"),
                    mew=0.5,
                    zorder=3,
                )
            ax.axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
        axes[0].set_xlabel("SD of the e₁ component over the color embeddings", fontsize=9)
        axes[1].set_xlabel("r(lightness, α) at color positions of other ops, slice 2", fontsize=9)
        axes[0].set_ylabel("spill", fontsize=9)
        h = [Line2D([], [], ls="none", marker="s", color=ink(e), label=f"{e} epochs") for e in LENGTHS]
        h += [
            Line2D(
                [],
                [],
                ls="none",
                marker=MARK[t],
                color=light_dark("#666", "#aaa"),
                label=LATCH_NAME[t].replace("`", ""),
            )
            for t in (None, ",", "\n", "?")
        ]
        fig.legend(handles=h, loc="outside upper center", ncols=len(h), frameon=False, fontsize=7)
        return fig

    return _plot()


rf"""
## Lightness on e₁ in the embedding table (E2)

At the embedding slice the state at a position is the token embedding alone, with nothing yet read from the context. The pull asks each labeled `{ex.ANCHORED_OP}` context to have some position with α near 1 at that slice too, so it can only be met at the token level: by a latch, or by a statistical stand-in, where tokens more common in `{ex.ANCHORED_OP}` contexts sit further along e₁.

Lightness is such a stand-in. `{ex.ANCHORED_OP}` answers are the channel-wise absolute difference of the operands, so they are darker than the colors around them.

{emb_figure()}

Every anchored run has this lean: the correlation between lightness and the e₁ component is about {R_EMB_ANCHORED[0]:.1f} on all {len(ANCHORED)}, with the same sign on every one. On the controls it averages zero but runs from {R_EMB_CONTROL[1]:.1f} to {R_EMB_CONTROL[2]:.1f}, because lightness is a major direction of the color table and a fixed axis picks up some of it by chance, with a sign that depends on the seed. The spread of the component on the controls ({SD_CONTROL[0]:.2f}) is what projecting unit vectors onto a random direction in {ANCHORED[0]["n_embd"]} dimensions gives.

The size of the lean depends on the latch. Where a syntax token sits on e₁, the pull is met at that token and the color table barely moves, so the spread of the e₁ component over the colors is small ({SD_LATCHED[0]:.2f}). Where nothing latches, the colors take the whole pull, and the spread is about three times larger ({SD_NO_LATCH[0]:.2f}).

{spill_figure()}

The runs latched on ⏎ barely spill, and their color table hardly leans. That token ends the context and attention stops at it, so nothing downstream reads it.

The runs latched on `,` have a small lean and the largest spill. The `,` ends every example, and on these runs nearly the whole `,` embedding lies along e₁. So the projection that removes e₁ removes the token itself, and every context of every op loses its example boundaries, as spill-by-position saw.

The runs with no latch have the largest lean and spill moderately, onto `darken` on all but one of them. The right panel puts the three groups on one line: the further the lightness correlation in the states has moved from zero, the more the run spills.
"""

# %%


def depth_figure() -> str:
    groups = [
        (f"anchored {LONG}, no latch", group(LONG, None), ink(LONG), "o", "-"),
        (f"anchored {LONG}, latched ,", group(LONG, ","), ink(LONG), "^", "--"),
        (f"anchored {SHORT}, latched ⏎", group(SHORT, "\n"), ink(SHORT), "v", "--"),
        (f"control {LONG}", [control_of(r) for r in by_length(LONG)], ink("control"), "s", "-"),
        ("no-emb sweep trials", SWEEP_NO_EMB, ink("sweep"), "D", ":"),
    ]
    data = {
        "groups": [
            {
                "name": f"{name} ({len(rs)})",
                "rows": [r["r_light_states_other"] for r in rs],
                "ink": c,
                "marker": m,
                "ls": ls,
            }
            for name, rs, c, m, ls in groups
        ],
    }
    alt = f"""
        A line chart of the lightness correlation at color positions of other ops against slice, from the embedding
        to slice {ex.N_SLICES - 1}. All anchored groups start near −0.7 at the embedding. The no-latch group fades to
        −0.6, −0.4, −0.25, and −0.2; the two latched groups jump to about −0.2 at slice 1 and fade toward −0.07; the
        no-emb sweep trials follow the no-latch group. The control stays at zero. Faint lines show the individual
        runs.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="lightness-by-slice",
        alt_text=alt_text,
        caption=f"""
            **The lightness correlation by slice.** Each line is the seed mean of the correlation between lightness
            and α at the color positions of other ops, with the individual runs as faint lines behind. Circles:
            {LONG}-epoch runs with no latch. Up-triangles: latched on `,`. Down-triangles: {SHORT}-epoch runs
            latched on ⏎. Squares: the controls. Diamonds: the no-emb trials of the τ × λ_a sweep, τ from
            {DESIGN["sweep_tau"][0]:g} to {DESIGN["sweep_tau"][1]:g}.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
        x = np.arange(ex.N_SLICES)
        for g in data["groups"]:
            rows = np.array(g["rows"])
            for row in rows:
                ax.plot(x, row, "-", color=g["ink"], lw=0.5, alpha=0.25)
            ax.plot(
                x,
                rows.mean(0),
                g["ls"],
                marker=g["marker"],
                color=g["ink"],
                lw=1.2,
                ms=5,
                mec=light_dark("white", "#111"),
                mew=0.5,
                label=g["name"],
            )
        ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
        ax.set_xticks(x, ["emb", *(str(i) for i in x[1:])])
        ax.set_xlabel("slice", fontsize=9)
        ax.set_ylabel("r(lightness, α), other ops", fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


rf"""
## The lean fades with depth (E3)

If the lean is a property of the embedding table, it should be strongest at the embedding and fade as the blocks add contextual signal.

{depth_figure()}

The lean is there at {SHORT} epochs as at {LONG}, so it comes with the pull; training longer only grows it on the runs that lose their latch. On the latched runs the lean in the states falls away at slice 1, which fits the latch taking the pull from the color table.

The no-emb trials lean just as far at the embedding ({R_SWEEP_EMB[0]:.1f}), although that slice is not pulled. Leaving slice 0 out of `anchor_slices` takes the anti term off it as well, so nothing holds the table in place while the pull at block 1 moves the embedding beneath it.

The controls spread to either side of zero at every slice, a different sign on each seed, as their embedding tables do (E2). The anchored runs all lean the same way.

The readout table has no lightness on e₁ on any run (the correlation is below {R_READOUT[2]:.2f} in size), so the lean is on the input side only.
"""

# %%

SHOWN_SETS = ["embedding", "block 1", f"blocks 2 to {ex.N_SLICES - 1}", f"blocks 1 to {ex.N_SLICES - 1}", "every slice"]


def edit_figure() -> str:
    data = {
        "panels": [
            {
                "epochs": e,
                "sets": {s: [net_drop(r, s).tolist() for r in by_length(e)] for s in SHOWN_SETS},
            }
            for e in LENGTHS
        ],
        "ops": [OPS[o] for o in [D, *OTHER]],
        "order": [D, *OTHER],
    }
    alt = f"""
        Two dot panels, one per training length, of the net drop in score for each of {len(OPS)} ops under five
        slice-restricted edits, seeds jittered behind the seed mean. At {LONG} epochs every edit that includes block
        1 takes {ex.ANCHORED_OP} down by about 0.6; the embedding-only edit takes it down by 0.4. On the other ops
        the embedding-only and every-slice edits sit at 0.1 to 0.25, the block-1-only and blocks-1-to-4 edits at
        about 0.05, and the blocks-2-to-4 edit at zero. At {SHORT} epochs the drops on other ops are small for every
        edit.
    """
    return edit_draw(data, alt)


@memo
def edit_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="edit-by-slice",
        alt_text=alt_text,
        caption="""
            **The edit restricted to a set of slices, by op.** Each column is an op and each shade a slice set, from
            the embedding (darkest) to every slice (lightest); the large mark is the seed mean and the small marks
            behind it are the runs, with a thin bar spanning their range. The y axis is the drop in task score under
            the edit, net of the paired control. The dashed rule is the selectivity criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained", sharey=True)
        rng = np.random.default_rng(0)
        cmap = plt.get_cmap(light_dark("viridis", "viridis"))
        stops = np.linspace(*light_dark((0.0, 0.9), (0.35, 1.0)), len(SHOWN_SETS))
        for ax, panel in zip(axes, data["panels"], strict=True):
            for si, name in enumerate(SHOWN_SETS):
                net = np.array(panel["sets"][name])  # runs × ops
                color = cmap(stops[si])
                for oi, o in enumerate(data["order"]):
                    x = oi + (si - 2) * 0.15
                    col = net[:, o]
                    ax.plot([x, x], [col.min(), col.max()], "-", color=color, lw=0.8, alpha=0.5, zorder=2)
                    ax.plot(
                        x + rng.uniform(-0.02, 0.02, len(col)),
                        col,
                        "o",
                        ms=1.8,
                        color=color,
                        alpha=0.4,
                        mew=0,
                        zorder=3,
                    )
                    ax.plot(
                        x,
                        col.mean(),
                        "o",
                        ms=4.5,
                        color=color,
                        mec=light_dark("white", "#111"),
                        mew=0.5,
                        zorder=4,
                        label=name if oi == 0 else None,
                    )
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            ax.axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
            ax.set_xticks(range(len(data["ops"])), data["ops"], rotation=30, ha="right")
            ax.set_title(f"{panel['epochs']} epochs", fontsize=8)
        axes[0].set_ylabel("net drop in task score", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def edit_table() -> str:
    rows = []
    for name in SETS:
        row = [name]
        for e in LENGTHS:
            runs = by_length(e)
            row += [fmt(removal(r, name) for r in runs), fmt(spill(r, name)[0] for r in runs)]
        rows.append(row)
    head = ["slices edited"] + [f"{m}, {e}" for e in LENGTHS for m in ("removal", "spill")]
    return table_html(
        head,
        rows,
        f"""
        **Removal and spill under each slice-restricted edit**, seed mean and range over the {len(by_length(LONG))}
        runs at each training length. Removal is the net drop on `{ex.ANCHORED_OP}` as a share of the way to the
        target null; spill is the largest net drop on another op.
        """,
    )


LOW_LATE_TEXT = ", ".join(f"{s}" for s, _ in LOW_LATE)
LOW_LATE_VALUES = ", ".join(f"{v:.2f}" for _, v in LOW_LATE)
LOW_LATE_BLOCK1_TEXT = ", ".join(f"{v:.2f}" for v in LOW_LATE_BLOCK1)

rf"""
## The edit by slice (E4)

Is the spill caused by the edit at the early slices? Here the full edit is applied to one set of slices at a time, on every anchored run.

{edit_figure()}

{edit_table()}

At {LONG} epochs the embedding slice alone gives half the removal and all the spill: the spill under the embedding-only edit is the same size as under the full edit, run by run. Blocks 2 to {ex.N_SLICES - 1} alone give most of the removal, with spill within the criterion on {N_LATE_WITHIN} of the {len(by_length(LONG))} runs.

Block {ex.N_SLICES - 1} alone does nothing. That matches the readout having no lightness on e₁, and suggests the concept is read out of the stream before the last block.

The {SHORT}-epoch runs spill little under any edit because most of them are latched on ⏎. Their removal under the late edit is lower, because the unlatched ones lean at the embedding like the {LONG}-epoch runs.

But the late edit varies a lot between runs. On {len(LOW_LATE)} of the {len(group(LONG, None))} unlatched {LONG}-epoch runs (seeds {LOW_LATE_TEXT}) the edit on blocks 2 to {ex.N_SLICES - 1} removes much less ({LOW_LATE_VALUES}) than block 1 alone does on the same runs ({LOW_LATE_BLOCK1_TEXT}), while on the other unlatched runs it removes as much as the full edit. On those runs the model has come to compute `{ex.ANCHORED_OP}` partly from the lightness that the lean put on e₁. So taking the lean away is part of how the full edit works, which makes a late-only edit on the current recipe a weak fix on its own.
"""

# %%

rf"""
## Discussion

The spill has one source on these runs: the pull at the slices where no contextual concept can exist. At the embedding, and largely still at block 1, the state is the token, so the pull is met by whatever token-level feature separates `{ex.ANCHORED_OP}` contexts from the rest. The cheapest is a syntax token shared by every context (the latch); failing that, the darkness of the answers (the lean).

The edit then removes the stand-in from every op, and the ops that depend on lightness lose accuracy. This fits the four observations and the two earlier reports, and the edit-by-slice result is the same mechanism seen from the other side.

Two things stop the anti-subspace term from keeping the color table off e₁. The first is normalization. The anti term is the mean of cos² over every live position and every slice, so each color embedding is one of a few thousand terms in a mean, and moving one of them along e₁ costs almost nothing.

The pull, by contrast, divides by the number of labeled contexts. Mellowmax at the recipe τ[^mellowmax] also concentrates the gradient of each context on its best position, so the embedding of the darkest token in a `{ex.ANCHORED_OP}` context gets most of the pull of that context.

[^mellowmax]: Mellowmax: a smooth version of the maximum over positions, here of the alignment within a context. A small τ makes it favor the best position more sharply; a large τ makes it closer to the mean.

A rough count puts the pull on a color embedding an order of magnitude or more above what the anti term costs it at the hold ratio, and further above once the anti term anneals.[^weights] So a lean of this size is the cheapest way the model has to satisfy the slice-0 pull, and the anti term only slows it a little.

The second is that the anti weight is set relative to the anchor weight, so the sweep that varied λ_a scaled both terms together and left the ratio between them unchanged. That is why λ_a barely mattered.

[^weights]: A rough count: at the hold the anti weight is about 0.03, spread over roughly 300 live positions per window and five slices, so one embedding at cos² = 0.3 costs on the order of 0.03 × 0.3 / 1500 per window. The pull on the same embedding, where it is the best position in a labeled context, is on the order of 0.1 × (1 − α) / (labeled contexts per window, about 3) per window at each slice it is pulled on. The ratio is in the tens, and it depends on how many contexts the token appears in, so only the order of magnitude is meaningful.

The no-emb trials fit the same account: nothing holds their table in place (E3), and the edit still touches slice 0, so the lean is still removed from every op. That is also why [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) found that every slice restriction spilled more, with the states of other ops further along e₁ at the first two slices. The restriction took the anti term off the slices where the stand-in lives while the edit went on removing it.

If this account is right, a pull kept off the first two slices, with those slices held off e₁ by the anti term or by a hard constraint on the embedding table, would leave the model nothing to meet the pull with but a contextual feature, and the edit would then remove `{ex.ANCHORED_OP}` without the lightness.

The unlatched runs where the late edit removed little show the risk: on the current recipe the concept is partly built out of the lean, and a run with no lean has to build it another way. Whether it can is a question for a new run, since the stored checkpoints only say where the concept sits once the lean is there.

## Glossary

Latch
:   A syntax token embedding on e₁: a token every context contains, so putting it on the anchor direction satisfies the pull at the embedding for every context at once. Found by spill-by-position.

Lean
:   Lightness on e₁ in the color embedding table: darker colors sit further along the anchor direction, a stand-in for `{ex.ANCHORED_OP}` at the slices where nothing contextual exists.

Stand-in
:   Something token-level that satisfies the pull where no contextual concept can: a latch or a lean.

Removal
:   How much of `{ex.ANCHORED_OP}` the edit takes out: the net drop in its score under the edit, as a share of the way from the clean score to the target null.

Spill
:   The largest net drop in score on another op under the edit. The selectivity criterion is {CRITERION:g}.

Target null
:   The score an ideal predictor would get on `{ex.ANCHORED_OP}` contexts if it had lost that op and answered from the remaining ones.
"""
