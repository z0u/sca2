# ruff: noqa: B018
# title: The split anti weight at fresh seeds

# A preregistered confirmation: four arms at eight fresh seeds, differing only in the hold of the anti-subspace weight
# by slice. `experiment.py` beside this script trains, measures and dose-scores every run; the controls at the same
# seeds come from ex-2.2.23. The predictions were frozen at ff44a17; the results replaced the placeholders in place.
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

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


RESULTS = fetch_json(ex.RESULTS_REF)
TRAJ = fetch_json(ex.TRAJ_REF)
DOSE = {r["label"]: r for r in fetch_json(ex.DOSE_REF)["runs"]}
# Every run measured by embedding-lean's `measure_one`: the new runs and ex-2.2.23's controls.
RUNS = {r["label"]: r for r in RESULTS["runs"]} | {r["label"]: r for r in fetch_json(ex.lean.RESULTS_REF)["runs"]}
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

ARMS = tuple(a.name for a in ex.ARMS)
NAME = {a.name: a.shown_as for a in ex.ARMS}
HOLDS = {a.name: list(a.holds()) for a in ex.ARMS}
SPLIT, FALLOFF, HIGH = "split-anti", "falloff", "hold0.2"

# Task gate (frozen with the predictions): mean task change below this, or any op at any seed below the other.
GATE_MEAN, GATE_OP = -0.02, -0.05
# H1 and H2 rules (frozen with the predictions).
H1_SEEDS_AHEAD, H1_MEAN_GAIN, H1_WITHIN = 6, 0.05, 7
H2_BAND, H2_WITHIN = 0.03, 7


def label_of(arm: str, seed: int) -> str:
    return ex.label_of(arm, seed)


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
    rem, sp, raw = [], [], []
    for e, ec in zip(d["edits"], c["edits"], strict=True):
        net = np.array(e["drop"]) - np.array(ec["drop"])
        rem.append(float(net[D] / span))
        sp.append(float(net[OTHER].max()))
        raw.append(float(np.array(e["drop"])[OTHER].max()))
    return {"removal": rem, "spill": sp, "raw_spill": raw}


def op_change(arm: str, seed: int) -> np.ndarray:
    """The task score of each op, less that of the control at the same seed."""
    return np.array(run_of(arm, seed)["clean"]) - np.array(control_of(seed)["clean"])


def alpha_by_slice(arm: str, seed: int, key: str = "alpha_anchored", pos: list[int] | int = ANSWERS) -> np.ndarray:
    """α at the end of training on the probe contexts, per slice: averaged over the example answers by default."""
    a = np.asarray(TRAJ[label_of(arm, seed)]["traj"][key])[-1]
    return a[:, pos].mean(axis=1) if isinstance(pos, list) else a[:, pos]


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
    "hold0.2": light_dark("#0b3c73", "#cfe6ff"),
    "hold0.1": light_dark("#6fa3d6", "#5f8fbf"),
    "split-anti": light_dark("#c0392b", "#ff8a76"),
    "falloff": light_dark("#2a9d8f", "#7fd8c8"),
}
MARK = {"hold0.2": "s", "hold0.1": "o", "split-anti": "^", "falloff": "D"}
GHOST = light_dark("#bbb", "#555")
SLICE_TICK = ["emb", "1", "2", "3", "4"]

# --- Summaries the prose and the verdicts quote -----------------------------------------------------------------

CURVES = {a: {s: dose_curves(a, s) for s in SEEDS} for a in ARMS}
FULL = {a: {s: CURVES[a][s]["removal"][-1] for s in SEEDS} for a in ARMS}
WORST_SPILL = {a: {s: max(CURVES[a][s]["spill"]) for s in SEEDS} for a in ARMS}
N_WITHIN = {a: sum(v <= CRITERION for v in WORST_SPILL[a].values()) for a in ARMS}
TASK = {a: [float(op_change(a, s).mean()) for s in SEEDS] for a in ARMS}
WORST_OP = {a: min((float(op_change(a, s).min()), s, OPS[int(op_change(a, s).argmin())]) for s in SEEDS) for a in ARMS}
GATE_FAILS = {
    a: [(s, OPS[o], float(op_change(a, s)[o])) for s in SEEDS for o in np.flatnonzero(op_change(a, s) < GATE_OP)]
    for a in ARMS
}
PASSES_GATE = {a: np.mean(TASK[a]) >= GATE_MEAN and not GATE_FAILS[a] for a in ARMS}

GAIN = {s: FULL[SPLIT][s] - FULL[HIGH][s] for s in SEEDS}
H1_AHEAD = sum(g > 0 for g in GAIN.values())
H1_MEAN = float(np.mean(list(GAIN.values())))
H1_PASS = H1_AHEAD >= H1_SEEDS_AHEAD and H1_MEAN >= H1_MEAN_GAIN and N_WITHIN[SPLIT] >= H1_WITHIN
DIFF = {s: FULL[FALLOFF][s] - FULL[SPLIT][s] for s in SEEDS}
H2_MEAN = float(np.mean(list(DIFF.values())))
H2_PASS = abs(H2_MEAN) < H2_BAND and N_WITHIN[FALLOFF] >= H2_WITHIN

# Runs where the edit removes less than half the op (post hoc: the anchor settled elsewhere).
LOST = [(a, s) for a in ARMS for s in SEEDS if FULL[a][s] < 0.5]
# Seeds where the edit on slices 2 to 4 alone removes less than a quarter of the op, in every arm.
LATE_SHORT = [s for s in SEEDS if all(slice_removal(a, s, LATE_SET) < 0.25 for a in ARMS)]


# %%

r"""
# The split anti weight at fresh seeds

/// tip |
<!-- lede -->
We trained the candidate recipes from [the slice-terms experiment](/docs/m2/slice-terms/report.py) at eight fresh seeds. Holding the anti-subspace weight high on the first slices only kept the op removable and the edit selective, and a weight that falls off linearly with depth did as well. Holding it at one level on every slice, high or gentler, lost the anchor altogether on one seed in each arm.
///

We chose `falloff` as the recipe of record. By the frozen gate on damage to the task only `split-anti` qualified, and `falloff` missed that gate on one op at one seed.
<!-- REVIEW: the decision was open in the previous draft; the human chose `falloff` after reading the results. This departs from the frozen task gate, and the report says so here and in the decision. -->
"""

# %%

rf"""
## Findings

- [The split keeps both removal and selectivity (H1)](#the-split-keeps-both-removal-and-selectivity-h1) — **{"pass" if H1_PASS else "TODO"}**. `split-anti` removed more than `deep, hold 0.2` on {H1_AHEAD} of {len(SEEDS)} seeds and stayed within the criterion on {N_WITHIN[SPLIT]}, with the whole gain on the four seeds where the uniform hold anchored less well or not at all.
- [A linear falloff does as well as the split (H2)](#a-linear-falloff-does-as-well-as-the-split-h2) — **{"pass" if H2_PASS else "TODO"}**. `falloff` removed about as much as `split-anti` and stayed within the criterion on every seed.
- [Which slices the edit needs (E1)](#which-slices-the-edit-needs-e1) — the edit on slices 2 to 4 alone falls short on the same seed in every arm, which fits the seed rather than the terms deciding where it misses. The edit on slice 1 alone does depend on the arm: it removes less under the high uniform hold.
- [Decision](#decision) — `falloff` is the recipe of record. Only `split-anti` cleared the frozen task gate; `falloff` missed it on one op at one seed, and we kept it for its spill and its shape.

/// admonition | How to read this report
This report was preregistered: the predictions and the criteria for the decision were frozen at commit `ff44a17`, before any run of this experiment. Each section opens with what we expect, and the results replaced the placeholders in place. The subsection on runs that lost the anchor, in H1, was not planned, and the decision departs from the frozen task gate.
///
"""

# %%

r"""
## Why this experiment

The pull indirectly asks the states at the example answers of `difference` contexts to align with e₁, and the anti-subspace term asks every other state to stay off it. When the anti weight anneals low, the color embedding table and slice 1 drift onto e₁ on the other ops, and the edit that projects e₁ out takes some of those ops with it: the spill. [The late-pull experiment](/docs/m2/late-pull/report.py) found that a higher hold everywhere keeps the spill within the criterion, at some cost in removal. [The slice-terms experiment](/docs/m2/slice-terms/report.py) traced that cost to the hold countering the pull at the pulled slices, and held the weight high on the embedding and slice 1 only. On its four seeds this gave the removal back and kept the spill within the criterion on three.

Those four seeds are the ones the arms were chosen on, so the result may be partly fitted to them. Before the split becomes the recipe a deliverable rests on, it should hold at fresh seeds, beside the arms it was chosen over.

The split also sets a boundary by hand: high on the first two slices, low after. If a weight that falls off smoothly with depth does as well, the recipe has one less choice in it.
"""

# %%

r"""
## Parameters

Every arm pulls `difference` toward e₁ at slices 2 to 4, pooled over the positions of each labeled context, so that most of the pull lands on the example answers. The anti-subspace term acts on every slice (the `deep` pull of the late-pull experiment). The anchor weight $\lambda_a$ (0.1), the pooling temperature $\tau_p$ (0.1), the label, the corpus and the schedules are the recipe values, at 400 epochs. The anti weight $\lambda_{\bar s}$ starts at 0.25 on every slice and anneals to its *hold* by epoch 360, as in the slice-terms experiment. The arms differ only in that hold:

<figure class="report-figure"><table class="report-table dense"><thead><tr><th>arm</th><th class=num>emb</th><th class=num>1</th><th class=num>2</th><th class=num>3</th><th class=num>4</th></tr></thead><tbody>
<tr><td><code>deep, hold 0.2</code></td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.2</td></tr>
<tr><td><code>deep, hold 0.1</code></td><td class=num>0.1</td><td class=num>0.1</td><td class=num>0.1</td><td class=num>0.1</td><td class=num>0.1</td></tr>
<tr><td><code>split-anti</code></td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.03</td><td class=num>0.03</td><td class=num>0.03</td></tr>
<tr><td><code>falloff</code></td><td class=num>0.2</td><td class=num>0.16</td><td class=num>0.12</td><td class=num>0.07</td><td class=num>0.03</td></tr>
</tbody></table><figcaption><p><strong>The hold of the anti weight by slice</strong>, for each arm. Slice 0 is the embedding. The <code>falloff</code> holds are a straight line from 0.2 to 0.03, rounded here to two places.</p></figcaption></figure>

`falloff` runs in a straight line between the two holds of `split-anti`, so it matches the split at the embedding and slice 4 and differs between them. `deep, hold 0.1` is the gentler uniform hold, a candidate for the recipe in its own right.

Each arm trains at model seeds 704 to 711, 32 runs in all. Ex-2.2.23 trained 400-epoch controls (no pull, no anti term) at these seeds, so every run has a control with the same initialization, batches and label draws. No arm has been trained at these seeds before.

`deep` at the recipe hold (0.03 everywhere) is left out, since it spilled over the criterion on every seed so far.
"""

# %%

r"""
## Measurements

The measurements and dose scoring are those of the slice-terms experiment.

- Removal: the net drop in the score on `difference` under the edit, as a share of the way from the clean score to the target null. The edit projects e₁ out of the state at every position, on a set of slices, at a dose of a quarter, a half, three quarters, or all of the e₁ component. Higher is better.
- Spill: the largest net drop on another op under the edit. A run is within the criterion when its spill is at most 0.02 at every dose.
- Both are net of the control at the same seed.
- Landing: the share of the distance to the target null that the full edit covers, as total variation on `difference` contexts. Higher is better.
- Task change: the mean score over ops, less that of the control. Zero means no damage; below zero is a loss.
- The alignment α with e₁ at the example answers, at every slice, at the end of training, and the slope of the color embedding table (the e₁ component of the color embeddings against lightness).
"""

# %%

r"""
## The split keeps both removal and selectivity (H1)

**What we expect.** On the slice-terms seeds, `split-anti` removed more than `deep, hold 0.2` at every seed and stayed within the criterion on three of four. We expect the same at the fresh seeds. The main number is the removal at the full dose, paired by seed against `deep, hold 0.2`.

A pass needs both removal and spill to hold:

- A pass would be `split-anti` removing more than `deep, hold 0.2` on at least 6 of 8 seeds, by at least 0.05 on average, with at least 7 of 8 runs within the criterion.
- A partial pass would be the removal gain holding with 5 or 6 runs within the criterion.
- A miss would be no removal gain (fewer than 5 seeds ahead, or an average gain under 0.02), or 4 or fewer runs within the criterion.

If the removal gain lands between 0.02 and 0.05 on average, or the seeds split evenly on its direction, the result would be outside the plan, and the verdict would be Unresolved.
"""

# %%


def dose_figure() -> str:
    data = {
        "doses": GAMMAS,
        "arms": [{"arm": a, "name": NAME[a], "runs": [CURVES[a][s] for s in SEEDS]} for a in ARMS],
    }
    alt = """
        Removal (top row) and spill (bottom row) against the dose, one column per arm. Removal grows with the dose on nearly every run and reaches about 0.8 to 0.9 at the full dose;
        the uniform holds each have one run that stays near zero, and the high uniform hold has three more that end
        between 0.55 and 0.75. Spill stays near zero on almost every run, with two hold-0.1 runs and one split-anti
        run going over the criterion at the higher doses.
    """
    return dose_draw(data, alt)


@memo
def dose_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="split-anti-seeds-dose",
        alt_text=alt_text,
        caption="""
            **Removal and spill by dose**, the edit at every position and slice, one column per arm. One hairline per
            run and the seed mean bold; the other arms are ghosts behind. The dashed rule is the selectivity
            criterion.
        """,
    )
    def _plot() -> plt.Figure:
        arms = data["arms"]
        fig, axes = plt.subplots(
            2, len(arms), figsize=(7.2, 3.8), layout="constrained", sharex=True, sharey="row", squeeze=False
        )
        for row, key in enumerate(("removal", "spill")):
            for col, arm in enumerate(arms):
                ax = axes[row, col]
                ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
                for other in arms:
                    if other is arm:
                        continue
                    for r in other["runs"]:
                        ax.plot(data["doses"], r[key], color=GHOST, lw=0.4, alpha=0.5, zorder=1)
                a = arm["arm"]
                for r in arm["runs"]:
                    ax.plot(data["doses"], r[key], color=INK[a], lw=0.6, alpha=0.8, zorder=2)
                mean = np.mean([r[key] for r in arm["runs"]], axis=0)
                ax.plot(data["doses"], mean, color=INK[a], lw=1.7, marker=MARK[a], ms=4.5, zorder=3)
                if row == 0:
                    ax.set_title(arm["name"], fontsize=8)
                else:
                    ax.axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
                    ax.set_xlabel("dose", fontsize=8)
                ax.set_xticks(data["doses"], [f"{g:g}" for g in data["doses"]], fontsize=7)
                ax.tick_params(labelsize=7)
            axes[row, 0].set_ylabel(key, fontsize=9)
        return fig

    return _plot()


def dose_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            fmt(FULL[a].values()),
            f"{N_WITHIN[a]} of {len(SEEDS)}",
            fmt(WORST_SPILL[a].values(), ".3f"),
            fmt((CURVES[a][s]["raw_spill"][-1] for s in SEEDS), ".3f"),
            fmt(DOSE[label_of(a, s)]["landing"] for s in SEEDS),
            fmt(TASK[a], ".3f"),
        ]
        for a in ARMS
    ]
    return table_html(
        ["arm", "removal ↑", "within criterion ↑", "spill, any dose ↓", "raw spill ↓", "landing ↑", "task change ↑"],
        rows,
        f"""
        **Removal, spill, landing and task**, seed mean and range over {len(SEEDS)} runs. Removal is at the full
        dose; "spill, any dose" is the largest net spill over the four doses, and "within criterion" counts the runs
        where it is at most {CRITERION:g}. The raw spill is the largest drop on another op at the full dose without
        subtracting the control. The landing is the share of the distance to the target null that the full edit
        covers, and the task change is the mean score over ops less that of the control. Arrows mark which direction
        is better.
        """,
    )


def paired_table() -> str:
    rows = [[str(s), num(FULL[SPLIT][s]), num(FULL[HIGH][s]), num(GAIN[s], "+.2f")] for s in SEEDS] + [
        [
            "mean",
            num(np.mean(list(FULL[SPLIT].values()))),
            num(np.mean(list(FULL[HIGH].values()))),
            num(H1_MEAN, "+.2f"),
        ]
    ]
    return table_html(
        ["seed", "`split-anti`", "`deep, hold 0.2`", "gain"],
        rows,
        "**Removal at the full dose, paired by seed**, and the gain of `split-anti` over `deep, hold 0.2`.",
    )


def lost_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            str(s),
            num(FULL[a][s]),
            num(alpha_by_slice(a, s)[-1]),
            num(alpha_by_slice(a, s, pos=QUERY_ANSWER)[-1]),
        ]
        for a, s in LOST
    ]
    return table_html(
        ["arm", "seed", "removal", "α, example answers", "α, query answer"],
        rows,
        f"""
        **The runs that lost the anchor.** Removal at the full dose, and α at slice 4 at the end of training on
        `{ex.ANCHORED_OP}` probe contexts, at the example answers and at the query answer.
        """,
    )


rf"""
**What we saw.** `split-anti` removed more than `deep, hold 0.2` on {H1_AHEAD} of {len(SEEDS)} seeds, by {num(H1_MEAN)} on average, and stayed within the criterion on {N_WITHIN[SPLIT]}. By the frozen rule that is a pass.

{dose_figure()}

The gain is uneven across seeds. On four of them the two arms remove about the same share of the op, with the split a little behind on two. On the other four the high hold everywhere removed much less, and on one of those it removed almost nothing.

{paired_table()}

So the high hold at the pulled slices does cost removal, as the slice-terms experiment found, but here only on some seeds. With the weight split by slice, no seed shows that cost.
<!-- REVIEW: "took that cost away on every seed" implied a cost on every seed; on four seeds the two arms are level or the split is a little behind. The claim is that no split run is weak (0.79 or more on every seed). Verify: the paired table. -->

The gain seems to come from where the anti weight sits, and not only from how much there is. `deep, hold 0.1` sums to about the same weight over the five slices as `split-anti` (0.50 against 0.49), and it removed less, lost the anchor on one seed, and spilled over the criterion on two.
<!-- REVIEW: added the total-weight comparison, since `deep, hold 0.1` is the arm that separates amount from placement. Verify: the holds in the parameters table sum to 1.0, 0.5, 0.49, and 0.575. -->

The one `split-anti` run over the spill criterion (seed {max(SEEDS, key=lambda s: WORST_SPILL[SPLIT][s])}) goes over at the two highest doses only. `falloff` and `deep, hold 0.2` were within the criterion on every seed, and `deep, hold 0.1` on {N_WITHIN["hold0.1"]}.

{dose_table()}

The edit lands a little closer to the target null in the two arms with a split or falloff hold than in the uniform ones. The task change is small in every arm, and smallest in `split-anti`.

/// admonition | Pass
`split-anti` removed more than `deep, hold 0.2` on {H1_AHEAD} of {len(SEEDS)} seeds, by {num(H1_MEAN)} on average, with {N_WITHIN[SPLIT]} of {len(SEEDS)} runs within the criterion.
///

### Runs that lost the anchor (post hoc)

Each uniform hold lost the anchor on one seed. On those runs the example answers stay off e₁ at every slice, and the query answer aligns instead, as in seed 700 of `pool-slices` in the slice-terms experiment. The prediction of the query answer is made at the query `=`, before that position, so an edit there cannot reach it.

{lost_table()}

Neither the split nor the falloff lost the anchor on any seed. One reading is that a high hold at the pulled slices makes the example answers costly to align, so the pull settles on the position that aligns most readily. Eight seeds per arm give two such runs in all, so this is a pattern to watch rather than an established cause. A different pooling temperature τ_p might also keep the pull on the example answers; we have not tried one here.
"""


# %%

r"""
## A linear falloff does as well as the split (H2)

**What we expect.** `falloff` holds slice 1 a little lower than `split-anti` and slices 2 and 3 higher. The lower hold at slice 1 may let a little more spill through, and the higher holds at slices 2 and 3 may counter the pull there and cost a little removal. We expect both effects to be small. The main number is the removal at the full dose, paired by seed against `split-anti`.

As in H1, a pass needs both removal and spill to hold:

- A pass would be `falloff` removing within 0.03 of `split-anti` on average, with at least 7 of 8 runs within the criterion.
- A miss would be `falloff` removing 0.03 or more less than `split-anti` on average, or 6 or fewer runs within the criterion.
- If `falloff` removes 0.03 or more than `split-anti` on average, that would also be outside what we expect, and we would report it as a pass with the gain named.

If the seeds split evenly on the direction of the removal difference and the average is near the 0.03 edge, the verdict would be Unresolved.
"""

# %%


def pair_figure() -> str:
    data = {
        "seeds": SEEDS,
        "removal": {a: [FULL[a][s] for s in SEEDS] for a in (SPLIT, FALLOFF)},
        "spill": {a: [WORST_SPILL[a][s] for s in SEEDS] for a in (SPLIT, FALLOFF)},
    }
    alt = """
        Two scatter plots of falloff against split-anti, one point per seed. In removal, seven points sit on the
        line of equality between 0.79 and 0.91, and one sits below it at about 0.79 against 0.91. In spill, every
        falloff value is under the criterion; one split-anti value is above it, at about 0.04.
    """
    return pair_draw(data, alt)


@memo
def pair_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="split-anti-seeds-pair",
        alt_text=alt_text,
        caption=f"""
            **`falloff` against `split-anti`, paired by seed.** Left: removal at the full dose, with the line of
            equality and a band of ±{H2_BAND:g}. Right: the largest spill at any dose, with the line of equality and the
            criterion on each axis (dashed). Seeds off the diagonal are labeled.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7), layout="constrained")
        ink = INK[FALLOFF]
        for ax, key in zip(axes, ("removal", "spill"), strict=True):
            x, y = np.array(data[key][SPLIT]), np.array(data[key][FALLOFF])
            lo, hi = (0.7, 0.95) if key == "removal" else (-0.015, 0.05)
            ax.plot([lo, hi], [lo, hi], color=light_dark("#999", "#666"), lw=0.8, zorder=1)
            if key == "removal":
                ax.fill_between(
                    [lo, hi],
                    [lo - H2_BAND, hi - H2_BAND],
                    [lo + H2_BAND, hi + H2_BAND],
                    color=light_dark("#ddd", "#333"),
                    lw=0,
                    zorder=0,
                )
            else:
                for f in (ax.axhline, ax.axvline):
                    f(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
            ax.scatter(x, y, color=ink, marker=MARK[FALLOFF], s=22, zorder=3)
            for s, xi, yi in zip(data["seeds"], x, y, strict=True):
                if abs(yi - xi) < (0.02 if key == "removal" else 0.008):
                    continue  # label only the seeds off the diagonal; the rest overlap
                ax.annotate(str(s), (xi, yi), xytext=(3, 2), textcoords="offset points", fontsize=6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{key}, split-anti", fontsize=8)
            ax.set_ylabel(f"{key}, falloff", fontsize=8)
            ax.tick_params(labelsize=7)
        return fig

    return _plot()


def depth_figure() -> str:
    data = {
        "arms": [
            {
                "arm": a,
                "name": NAME[a],
                "holds": HOLDS[a],
                "anchored": [alpha_by_slice(a, s).tolist() for s in SEEDS],
                "other": [alpha_by_slice(a, s, "alpha_other").tolist() for s in SEEDS],
            }
            for a in (SPLIT, FALLOFF)
        ]
    }
    alt = """
        Alignment at the example answers by slice, for split-anti and falloff. In difference contexts both arms
        rise from near zero at the embedding to about 0.6 at slices 3 and 4, nearly on top of each other. In the contexts of the other ops both stay low, at about 0.1 to 0.2
        from slice 1 on. Underneath, the hold of the anti weight by slice: split-anti steps from 0.2 to 0.03
        after slice 1, and falloff descends in a straight line between the same two values.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="split-anti-seeds-depth",
        alt_text=alt_text,
        caption=f"""
            **Alignment at the example answers by slice**, at the end of training, on the probe contexts of
            `{ex.ANCHORED_OP}` (left) and of the other ops (right). One hairline per run and the seed mean bold.
            Underneath is the hold of the anti weight at each slice.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            2, 2, figsize=(6.4, 3.4), layout="constrained", sharex=True, sharey="row", height_ratios=[3, 1]
        )
        x = np.arange(len(SLICE_TICK))
        for col, key in enumerate(("anchored", "other")):
            ax, wax = axes[0, col], axes[1, col]
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for arm in data["arms"]:
                a = arm["arm"]
                for r in arm[key]:
                    ax.plot(x, r, color=INK[a], lw=0.5, alpha=0.6, zorder=2)
                ax.plot(
                    x,
                    np.mean(arm[key], axis=0),
                    color=INK[a],
                    lw=1.6,
                    marker=MARK[a],
                    ms=4.5,
                    zorder=3,
                    label=arm["name"],
                )
                wax.plot(x, arm["holds"], color=INK[a], lw=1.2, marker=MARK[a], ms=3.5)
            ax.set_ylim(-0.05, 0.8)
            wax.set_ylim(0, 0.24)
            wax.set_yticks([0, 0.1, 0.2])
            wax.set_xticks(x, SLICE_TICK)
            wax.set_xlabel("slice", fontsize=8)
            for a_ in (ax, wax):
                a_.tick_params(labelsize=7)
        axes[0, 0].set_ylabel(f"α, {ex.ANCHORED_OP}", fontsize=8)
        axes[0, 1].set_ylabel("α, other ops", fontsize=8)
        axes[1, 0].set_ylabel("hold", fontsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


assert H2_PASS
FALLOFF_LOW = min(SEEDS, key=lambda s: DIFF[s])

rf"""
**What we saw.** `falloff` removed about the same share of the op as `split-anti`, {num(abs(H2_MEAN), ".3f")} {"less" if H2_MEAN < 0 else "more"} on average, and stayed within the criterion on {N_WITHIN[FALLOFF]} of {len(SEEDS)} seeds. By the frozen rule that is a pass.

{pair_figure()}

On seven seeds the two arms remove nearly the same share of the op. On seed {FALLOFF_LOW} `falloff` removes about a tenth less, and the spill points sit low for both arms apart from the one `split-anti` run over the criterion.

The alignment by slice agrees. In `{ex.ANCHORED_OP}` contexts the two arms are nearly the same at every slice. In the contexts of the other ops, `falloff` sits a little higher at slice 1, where its hold is lower, and a little lower at slices 2 and 3, where its hold is higher. Both differences are small next to the alignment the pull gives.

{depth_figure()}

So the boundary that the split sets by hand does not seem to matter much: a straight line between the same two levels gives nearly the same anchor.

/// admonition | Pass
`falloff` removed within {num(abs(H2_MEAN), ".3f")} of `split-anti` on average, with {N_WITHIN[FALLOFF]} of {len(SEEDS)} runs within the criterion.
///
"""


# %%

r"""
## Which slices the edit needs (E1)

On the slice-terms seeds and the late-pull seeds, editing slice 1 alone removed most of the op on every seed, while editing slices 2 to 4 alone removed nearly everything on some seeds and little on others. One reading is that the evidence for the op is carried from the example answers to the `=` sites by attention, in some other direction than e₁, and that the block whose heads make that move reads the example answers as they are at its input. Slice 1 is the input to every block after it, so clearing e₁ there removes the evidence before it moves. Clearing slices 2 to 4 leaves the copy at slice 1 for block 2 to read. This reading is not tested here.

We repeat the edit on slice 1 alone and on slices 2 to 4 alone for every run, to see whether the split by seed holds at the fresh seeds and in each arm.
"""

# %%


def slices_figure() -> str:
    data = {
        "seeds": SEEDS,
        "late_short": LATE_SHORT,
        "arms": [
            {
                "arm": a,
                "name": NAME[a],
                "slice1": [slice_removal(a, s, SLICE1) for s in SEEDS],
                "late": [slice_removal(a, s, LATE_SET) for s in SEEDS],
            }
            for a in ARMS
        ],
    }
    alt = f"""
        Removal by the edit on slice 1 alone (left) and on slices 2 to 4 alone (right), one column per arm, seeds as
        dots. On slice 1 alone most runs remove 0.6 to 0.95, with a scatter of lower runs. On slices 2 to 4 alone
        most runs remove about 0.9, and seed {", ".join(map(str, LATE_SHORT))}, drawn as an open marker, removes
        almost nothing in every arm.
    """
    return slices_draw(data, alt)


@memo
def slices_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="split-anti-seeds-slices",
        alt_text=alt_text,
        caption="""
            **Removal by which slices the edit covers**, at the full dose. One column per arm, a thin bar spanning
            the seeds, the seeds as faded dots, and the seed mean in the arm's marker. Open markers are the seed where the edit on slices 2
            to 4 alone falls short in every arm.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), layout="constrained", sharey=True)
        rng = np.random.default_rng(0)
        for ax, key, title in zip(axes, ("slice1", "late"), ("slice 1 alone", "slices 2 to 4 alone"), strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for i, arm in enumerate(data["arms"]):
                a, v = arm["arm"], np.array(arm[key])
                ax.plot([i, i], [v.min(), v.max()], color=INK[a], lw=4, alpha=0.2, solid_capstyle="round", zorder=1)
                jit = i + rng.uniform(-0.15, 0.15, len(v))
                for s, xj, yj in zip(data["seeds"], jit, v, strict=True):
                    short = s in data["late_short"]
                    ax.scatter(
                        xj,
                        yj,
                        s=12,
                        marker="o",
                        zorder=2,
                        facecolor="none" if short else INK[a],
                        edgecolor=INK[a],
                        alpha=1 if short else 0.55,
                    )
                ax.scatter(i, v.mean(), s=36, marker=MARK[a], color=INK[a], zorder=3)
            ax.set_xticks(range(len(data["arms"])), [arm["name"] for arm in data["arms"]], fontsize=7, rotation=20)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("removal", fontsize=8)
        return fig

    return _plot()


def slices_table() -> str:
    edits = (("slice 1", SLICE1), ("2 to 4", LATE_SET))
    rows = [
        [f"`{NAME[a]}`", label] + [num(slice_removal(a, s, sl)) for s in SEEDS] for label, sl in edits for a in ARMS
    ]
    head = ["arm", "slices"] + [str(s) for s in SEEDS]
    return table_html(head, rows, "**Removal by which slices the edit covers**, at the full dose, by arm and seed.")


# The seed where split-anti alone falls to about half on slices 2 to 4 (one run; reported without a reading).
SPLIT_HALF = min(s for s in SEEDS if s not in LATE_SHORT and slice_removal(SPLIT, s, LATE_SET) < 0.6)

rf"""
**What we saw.** The edit on slices 2 to 4 alone removes nearly all of the op on almost every run, and falls short on seed {", ".join(map(str, LATE_SHORT))} in every arm. On seed {SPLIT_HALF} it removes about half under `split-anti`, and nearly all under the other arms that anchored.

The edit on slice 1 alone removes most of the op on most runs of the split, the falloff, and the low uniform hold. It removes less under the high uniform hold, even on the seeds where that hold anchored well.

{slices_figure()}

The edit on slices 2 to 4 alone falls short on the same seed in every arm. Every arm shares the initialization, batches, and label draws of its seed, so this fits the seed deciding where the evidence for the op is read. The reading in the introduction to this section would trace that to which block moves the evidence to the `=` sites.

The edit on slice 1 alone removes different amounts in different arms. So the hold at the pulled slices shifts some of what the edit reaches, but it does not move the seed on which the later slices miss.
<!-- REVIEW: narrowed the E1 claim to the slices-2-to-4 edit. The slice-1 edit removes less under `deep, hold 0.2` on the well-anchored seeds too (0.39 to 0.64 against 0.74 to 0.87 for the split), so "which slices the edit needs is set by the seed rather than by the terms" outran the table. Verify: the slice-1 columns of the table. -->
<!-- The seed-708 shortfall under `split-anti` on slices 2 to 4 is one run and is reported without a reading. -->

{slices_table()}

The runs that lost the anchor (in H1) remove almost nothing under any edit, and appear here as the low points of the uniform holds.
"""


# %%

r"""
## Decision

We chose the recipe of record for D2.2 among the four arms by the criteria frozen with the predictions: runs within the spill criterion, removal at the full dose, landing, and a hard gate on the task change. These were fresh seeds for every arm, so the numbers for the chosen arm can be quoted as a result without a further confirmation.
"""

# %%


def decision_table() -> str:
    def worst(a: str) -> str:
        v, s, op = WORST_OP[a]
        return f"{num(v, '.3f')} (`{op}`, seed {s})"

    rows = [
        [
            f"`{NAME[a]}`",
            f"{N_WITHIN[a]} of {len(SEEDS)}",
            fmt(FULL[a].values()),
            fmt(DOSE[label_of(a, s)]["landing"] for s in SEEDS),
            num(float(np.mean(TASK[a])), ".3f"),
            worst(a),
            "yes" if PASSES_GATE[a] else "no",
        ]
        for a in ARMS
    ]
    return table_html(
        [
            "arm",
            "within criterion ↑",
            "removal ↑",
            "landing ↑",
            "task change ↑",
            "worst op change ↑",
            "clears the task gate",
        ],
        rows,
        f"""
        **Every criterion for every arm.** Removal and landing as seed mean and range. The task change is the mean
        over seeds of the mean score over ops, less that of the control. The worst op change is the largest loss
        on any one op at any seed against its control. The gate excludes an arm whose task change is below
        {num(GATE_MEAN)}, or whose worst op change is below {num(GATE_OP)}. Arrows mark which direction is better.
        """,
    )


CONTROL_SPREAD = float(np.ptp([control_of(s)["clean"] for s in SEEDS], axis=0).max())
GATE_OPS = sorted({op for a in ARMS for _, op, _ in GATE_FAILS[a]})

rf"""
{decision_table()}

Only `split-anti` clears the task gate. The three other arms each lose more than {num(abs(GATE_OP))} on some op at some seed, and every such loss is on one of the HSV ops ({", ".join(f"`{o}`" for o in GATE_OPS)}). These losses are larger than the spread between the controls at these seeds, at most {num(CONTROL_SPREAD, ".3f")} on any op.

We chose `falloff` anyway, which departs from the frozen gate. Its one miss is on `{WORST_OP[FALLOFF][2]}` at seed {WORST_OP[FALLOFF][1]}, a little past the limit. That seed is also where `falloff` removed a tenth less than the split, and where its task change over all ops was lowest, so the miss looks like one weaker run rather than an op that `falloff` harms. The HSV ops are the last to be learned, so a gate on any op at any seed may be stricter for them than for the others.

In its favor, `falloff` kept every run within the spill criterion, and its spill varied less across seeds than that of the split. That advantage rests on one run, the `split-anti` run at seed 704. The falloff also has no slice boundary to choose, and its levels can still be tuned, for example with a lower anti weight on the earliest slices.
"""


# %%

r"""
## Discussion

Setting the anti weight by slice holds up at fresh seeds. Both the split and the straight-line falloff anchored on every seed, kept the op removable, and kept the edit within the criterion on nearly every seed. The high uniform hold gave a weaker anchor on several seeds, and each uniform hold lost the anchor altogether on one.

The gain over the uniform high hold depends on the seed, so the high hold at the pulled slices seems to cost removal through the seeds where the pull struggles to settle on the example answers, rather than by trimming every run a little.

The falloff matching the split suggests the hold matters at its two ends: high on the embedding and slice 1, where the slice-terms experiment found the drift onto e₁ that causes the spill seemed to sit, and low on the slices the pull acts on. Where in between the weight steps down seems not to matter much at this resolution.
"""

# %%

rf"""
## Method

Training, measurement and dose scoring follow the slice-terms experiment, and its experiment module is reused with new arms and seeds. The anti weight per slice is the `AntiSpec.hold` tuple added there; the `falloff` holds are the unrounded straight line. The controls are ex-2.2.23's 400-epoch runs at seeds {SEEDS[0]} to {SEEDS[-1]}.

Two of the 32 trainings were stopped by the training watchdog while their checkpoints were uploading, after the last step, and were trained again from the start with the same seed and configuration.

The run cost about \$12.
"""
