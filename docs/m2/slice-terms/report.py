# ruff: noqa: B018
# title: Setting the terms by slice

# Eight new runs: the `deep` pull with the anti-subspace weight set by slice, and the same with the pull pooled over
# slices. `experiment.py` beside this script trains, measures and dose-scores every run; the twins at the same seeds
# come from the late-pull experiment, and the controls from ex-2.2.23.
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
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
TRAJ = fetch_json(ex.TRAJ_REF) | fetch_json(ex.pull.TRAJ_REF)
DOSE = {r["label"]: r for r in fetch_json(ex.DOSE_REF)["runs"]}
# Every run measured by embedding-lean's `measure_one`: the new runs, their late-pull twins, and the controls.
RUNS = (
    {r["label"]: r for r in RESULTS["runs"]}
    | {r["label"]: r for r in fetch_json(ex.pull.RESULTS_REF)["runs"]}
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
EVERY, LATE_SET, BLOCK1, EMB = "every slice", "blocks 2 to 4", "block 1", "embedding"

# The arms, by stored key, in the order the report shows them; the late-pull `late` arms are shown as `deep`.
ARMS = ("late", "late-anti0.2", "split-anti", "pool-slices")
NAME = {"late": "deep", "late-anti0.2": "deep, hold 0.2", "split-anti": "split-anti", "pool-slices": "pool-slices"}
TAU_ARMS = tuple(a.name for a in ex.ARMS if a.slice_tau is not None)
NEW = ("split-anti", "pool-slices", *TAU_ARMS)
TAU_SEED = ex.SLICE_TAU_SEED
# The slice-temperature ladder at its one seed: `pool-slices` is the rung at the recipe τ.
LADDER = (("pool-slices", ex.ex2216.TAU), *((a.name, a.slice_tau) for a in ex.ARMS if a.slice_tau is not None))


def label_of(arm: str, seed: int) -> str:
    return (ex.label_of if arm in NEW else ex.pull.label_of)(arm, seed)


def runs_of(arm: str) -> list[dict]:
    return [RUNS[label_of(arm, s)] for s in SEEDS]


def control_of(r: dict) -> dict:
    return RUNS[ex.ex2223.label_of("control", ex.EPOCHS, r["model_seed"])]


def dose_of(arm: str, seed: int) -> dict:
    return DOSE[label_of(arm, seed)]


def dose_control(seed: int) -> dict:
    return DOSE[f"control-s{seed}"]


def net_drop(r: dict, slice_set: str = EVERY) -> np.ndarray:
    """The drop in task score per op under the full edit on *slice_set*, net of the control at the same seed."""
    return np.array(r["edits"][slice_set]) - np.array(control_of(r)["edits"][slice_set])


def removal(r: dict, slice_set: str = EVERY) -> float:
    """The share of the way from the clean score to the target null that the net drop on the anchored op gets."""
    return float(net_drop(r, slice_set)[D] / (r["clean"][D] - r["null"][D]))


def spill(r: dict, slice_set: str = EVERY) -> float:
    """The largest net drop on another op."""
    return float(net_drop(r, slice_set)[OTHER].max())


def dose_curves(arm: str, seed: int) -> dict[str, list[float]]:
    """Removal and spill at each dose (ex-2.2.22's scoring, at every position and slice), net of the control."""
    d, c = dose_of(arm, seed), dose_control(seed)
    span = d["clean"]["eem"][D] - d["null"]["eem"][D]
    rem, sp, raw = [], [], []
    for e, ec in zip(d["edits"], c["edits"], strict=True):
        net = np.array(e["drop"]) - np.array(ec["drop"])
        rem.append(float(net[D] / span))
        sp.append(float(net[OTHER].max()))
        raw.append(float(np.array(e["drop"])[OTHER].max()))
    return {"removal": rem, "spill": sp, "raw_spill": raw}


def task_change(r: dict) -> float:
    """The mean task score over ops, less that of the control at the same seed."""
    return float(np.mean(r["clean"]) - np.mean(control_of(r)["clean"]))


def slope(axis) -> float:
    """The lean: the least-squares slope of the e₁ component of the color embeddings against lightness."""
    from sca.data.ops import colors

    light = (np.array(colors()) / 15.0).mean(1)
    return float(np.polyfit(light, np.asarray(axis, float), 1)[0])


def answers_by_slice(lab: str, key: str = "alpha_anchored", at: int = -1) -> np.ndarray:
    """The mean α at the example answers of the probe contexts, per slice, at one trajectory record."""
    return np.asarray(TRAJ[lab]["traj"][key])[at][:, ANSWERS].mean(axis=1)


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


def span_text(values, spec: str = ".2f") -> str:
    _, lo, hi = mean_range(values)
    return f"{num(lo, spec)} to {num(hi, spec)}"


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
    "late": light_dark("#8fbbe0", "#4f7fa8"),
    "late-anti0.2": light_dark("#0b3c73", "#cfe6ff"),
    "split-anti": light_dark("#c0392b", "#ff8a76"),
    "pool-slices": light_dark("#2a9d8f", "#7fd8c8"),
}
MARK = {"late": "o", "late-anti0.2": "s", "split-anti": "^", "pool-slices": "D"}
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
SCHED_INK = light_dark("#b4531f", "#dd8f5f")

# --- Summaries the prose quotes -------------------------------------------------------------------------------

CURVES = {a: [dose_curves(a, s) for s in SEEDS] for a in ARMS}
FULL_REMOVAL = {a: mean_range(c["removal"][-1] for c in CURVES[a]) for a in ARMS}
WORST_SPILL = {a: [max(c["spill"]) for c in CURVES[a]] for a in ARMS}
N_WITHIN = {a: sum(v <= CRITERION for v in WORST_SPILL[a]) for a in ARMS}
LANDING = {a: mean_range(dose_of(a, s)["landing"] for s in SEEDS) for a in ARMS}
TASK = {a: mean_range(task_change(r) for r in runs_of(a)) for a in ARMS}
MONOTONE = {a: all(np.all(np.diff(c["removal"]) > 0) for c in CURVES[a]) for a in ARMS}
# Where the pooled pull settles: the slice with the largest α at the example answers, by seed.
SETTLED = {s: int(np.argmax(answers_by_slice(label_of("pool-slices", s))[1:]) + 1) for s in SEEDS}
POOL_FAILED = [s for s in SEEDS if removal(RUNS[label_of("pool-slices", s)]) < 0.5]
POOL_OK = [s for s in SEEDS if s not in POOL_FAILED]
SPLIT_SPILLER = max(SEEDS, key=lambda s: max(dose_curves("split-anti", s)["spill"]))
SPLIT_SPILL_OP = OPS[OTHER[int(np.argmax(net_drop(RUNS[label_of("split-anti", SPLIT_SPILLER)])[OTHER]))]]
QUERY_ANSWER = ex.lean.QUERY_B + 2  # The position of the query answer: after the second operand and the `=`.
assert QUERY_ANSWER == ex.lean.QUERY_EQ + 1


def seeds_text(seeds: list[int]) -> str:
    """A list of seeds in words: 'seed 700', 'seeds 700 and 702', 'seeds 700, 701 and 703'."""
    listed = " and ".join(", ".join(str(s) for s in seeds).rsplit(", ", 1))
    return f"seed {listed}" if len(seeds) == 1 else f"seeds {listed}"


rf"""
# Setting the terms by slice

/// tip |
<!-- lede -->
We held the anti-subspace weight high on the embedding and block 1 only, and at the recipe level on the pulled blocks. This kept most of the selectivity of a high hold everywhere, and it gave back the removal that the high hold cost. We also pooled the pull over slices as well as positions. On most seeds the pull then settled on the last block. But on one seed it settled on a position the prediction does not use, and there the op could not be removed. On one of the block-4 seeds, a softer pool over slices let the earlier blocks align too, and removal and spill stayed about where they were.
///

[The late-pull experiment](/docs/m2/late-pull/report.py) pulled `{ex.ANCHORED_OP}` toward e₁ at blocks 2 to 4 only, with the anti-subspace term on every slice (the `deep` arm). Holding the anti weight at 0.2 for all of training, in place of the recipe hold of 0.03, kept the color embedding table and block 1 off e₁ on the other ops and brought the spill within the criterion on every seed. But it removed about a tenth less of the op.

A second review of those runs found that the hold also presses on the pull at the pulled blocks, which may account for the lost removal. So this report sets the two terms by slice, in two arms of {len(SEEDS)} runs each at {ex.EPOCHS} epochs:

- `split-anti`: the `deep` pull, with the anti weight held at {ex.EARLY_HOLD:g} on the embedding and block 1 and at the recipe hold of {ex.LATE_HOLD:g} on blocks 2 to 4.
- `pool-slices`: the same anti weights, with the pull on blocks 1 to 4 pooled over slices as well as positions, so that it may choose where along the stream to align.

A second round asked whether a softer pool over slices would settle less of the pull on block 4. It adds {len(TAU_ARMS)} runs of `pool-slices` at seed {TAU_SEED}, with the pool over slices at a temperature of its own.
"""

# %%

rf"""
## Observations

- [Removal and spill (E1)](#removal-and-spill-e1): with the anti weight split by slice, removal is back to that of `deep` at the recipe hold, and the spill stays within the criterion at every dose on {N_WITHIN["split-anti"]} of {len(SEEDS)} seeds. The fourth spills a little past it at the full dose only.
- [What the anti weight presses on (E2)](#what-the-anti-weight-presses-on-e2): the split keeps the alignment at the fitting example answers where the recipe hold has it, and keeps block 1 and the color embedding table nearly as far off e₁ on other ops as the high hold does.
- [Where the pooled pull settles (E3)](#where-the-pooled-pull-settles-e3): pooled over slices, the pull settles on block 4 on three seeds, and there the edit removes the op about as selectively as the split does. On {seeds_text(POOL_FAILED)} the pull settled on the query answer alone, a position the prediction does not read, and the op is not removable there.
- [A softer pool over depth (E4)](#a-softer-pool-over-depth-e4): on seed {TAU_SEED}, softening the pool over slices leaves block 4 where it was and lets blocks 1 and 2 align about as far as in `split-anti`. Removal and spill move around from one temperature to the next with no trend, and every run stays within the criterion.

## Scope

This is an exploratory study, with no preregistration and no gate. Four seeds per arm resolve the large differences between the late-pull holds, but not whether a single run past the criterion is typical. E4 has one run per temperature, so it can show a change in where the pull settles, but not a difference in removal of the size that seeds differ by.

Each run shares its seed, and so its initialization, batches and label draws, with a late-pull `deep` run at each hold and an ex-2.2.23 control. So a difference between arms at one seed comes from the change of terms rather than the initialization or the batches.

The anchor weight ({ex.ex2216.LAM:g}), τ ({ex.ex2216.TAU:g}), the label, the corpus and the schedules are the recipe values. Both arms keep the recipe anti schedule: {ex.ANTI_PEAK:g} at the start on every slice, annealing to its hold by epoch {ex.schedules()[1]["anneal_end"]:.0f}, then sharing the anchor anneal at the end. The anti weight is now written as an absolute number rather than a ratio to the anchor weight, which gives the same schedule here.

## The measurements

The measurements are those of the late-pull experiment, with the dose scoring of ex-2.2.22 added for every run and its twins:

- Removal: the net drop in the score on `{ex.ANCHORED_OP}` under the edit, as a share of the way from the clean score to the target null. The edit projects e₁ out of the state at every position, on a set of slices, and at a dose of a quarter, a half, three quarters, or all of the e₁ component.
- Spill: the largest net drop on another op. Both are net of the control at the same seed, and the selectivity criterion is a spill of at most {CRITERION:g} at every dose.
- The alignment α with e₁ at the example answers, the positions where the anchor sits in this grammar, along training at every slice.
- The task score of each op, as expected exact match on held-out contexts, against the control at the same seed.
"""

# %%


def dose_figure() -> str:
    data = {
        "doses": GAMMAS,
        "arms": [{"arm": a, "name": NAME[a], "runs": CURVES[a]} for a in ("late", "late-anti0.2", "split-anti")],
    }
    alt = """
        Removal grows with the dose in every arm; deep at the recipe hold and split-anti reach about 0.85 to 0.9 at
        the full dose, and deep at a hold of 0.2 reaches 0.64 to 0.86. Spill stays near zero at every dose for the
        two arms with a high hold on the first slices, with one split-anti run reaching 0.025 at the full dose,
        while deep at the recipe hold rises past the criterion from the three-quarter dose on.
    """
    return dose_draw(data, alt)


@memo
def dose_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="slice-terms-dose",
        alt_text=alt_text,
        caption="""
            **Removal and spill by dose**, the edit at every position and slice. One hairline per run and the seed
            mean bold. The dashed rule is the selectivity criterion. `deep` and `deep, hold 0.2` are the late-pull
            runs at the same seeds.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained", sharex=True)
        for ax, key, ylabel in zip(axes, ("removal", "spill"), ("removal", "spill"), strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for arm in data["arms"]:
                a = arm["arm"]
                for r in arm["runs"]:
                    ax.plot(data["doses"], r[key], color=INK[a], lw=0.5, alpha=0.6, zorder=2)
                mean = np.mean([r[key] for r in arm["runs"]], axis=0)
                ax.plot(data["doses"], mean, color=INK[a], lw=1.6, marker=MARK[a], ms=4.5, label=arm["name"], zorder=3)
            ax.set_xlabel("dose", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(data["doses"])
        axes[1].axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def dose_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            fmt(c["removal"][-1] for c in CURVES[a]),
            f"{N_WITHIN[a]} of {len(SEEDS)}",
            fmt(WORST_SPILL[a], ".3f"),
            fmt((c["raw_spill"][-1] for c in CURVES[a]), ".3f"),
            fmt(dose_of(a, s)["landing"] for s in SEEDS),
            fmt((task_change(r) for r in runs_of(a)), ".3f"),
        ]
        for a in ARMS
    ]
    return table_html(
        ["arm", "removal", "within criterion", "spill, any dose", "raw spill", "landing", "task change"],
        rows,
        f"""
        **Removal, spill, landing and task**, seed mean and range over {len(SEEDS)} runs. Removal is at the full
        dose; "spill, any dose" is the largest net spill over the four doses, and "within criterion" counts the runs
        where it is at most {CRITERION:g}. The raw spill is the largest drop on another op at the full dose without
        subtracting the control. The landing is the share of the distance to the target null that the full edit
        covers, as total variation on `{ex.ANCHORED_OP}` contexts. The task change is the mean score over ops, less
        that of the control. `pool-slices` is the subject of E3.
        """,
    )


rf"""
## Removal and spill (E1)

If the high hold costs removal by pressing on the pull at blocks 2 to 4, holding it high only on the embedding and block 1 should give the removal back. If the stand-ins live at those first two slices, the spill should also stay low. The main measure is removal at the full dose, with the spill at every dose beside it.

{dose_figure()}

Both hold. On every `split-anti` run, removal grows with the dose to about what `deep` removes at the recipe hold (the table below has the ranges). The spill stays within the criterion at every dose on {N_WITHIN["split-anti"]} of {len(SEEDS)} seeds. On seed {SPLIT_SPILLER} it passes the criterion at the full dose only ({max(dose_curves("split-anti", SPLIT_SPILLER)["spill"]):.3f}, on `{SPLIT_SPILL_OP}`), the same seed and op where `deep` at the recipe hold spilled least. The high hold everywhere is the only arm within the criterion on every seed.

{dose_table()}

The edit lands about as close to the target null in `split-anti` as in both `deep` arms. So the extra removal seems to be the op itself going away, rather than answer mass scattered onto other ops.

The task change is small, as in `deep`. One `split-anti` run (seed 702) has the lowest task score against its control of any run so far. The loss is spread across all ops rather than concentrated on `{ex.ANCHORED_OP}`, which suggests seed variation more than a cost of the terms. Four seeds cannot separate the two.

So on four seeds the split gets nearly all the selectivity of the high hold with none of its cost in removal. Which slices the edit has to cover is unchanged from late pull: as in `deep`, editing blocks 2 to 4 alone removes nearly everything on two seeds and less than half on the other two, and editing block 1 alone removes most of the op on every seed.
"""

# %%


def alpha_figure() -> str:
    def series(arm: str, s: int) -> dict:
        t = TRAJ[label_of(arm, s)]["traj"]
        a = np.asarray(t["alpha_anchored"])[:, :, ANSWERS].mean(axis=2)  # (record, slice)
        o = np.asarray(t["alpha_other"])[:, :, ANSWERS].mean(axis=2)
        return {"epoch": t["epoch"], "late": a[:, 2:].mean(axis=1).tolist(), "b1_other": o[:, 1].tolist()}

    def anti(arm: str) -> dict:
        t = TRAJ[label_of(arm, SEEDS[0])]["traj"]
        return {"epoch": t["epoch"], "w": np.asarray(t["anti_weight"]).reshape(len(t["epoch"]), -1)[:, 0].tolist()}

    data = {
        "arms": [
            {"arm": a, "name": NAME[a], "runs": [series(a, s) for s in SEEDS]}
            for a in ("late", "late-anti0.2", "split-anti")
        ],
        "anti": {"recipe hold": anti("late"), "hold 0.2": anti("late-anti0.2")},
    }
    alt = """
        At the example answers of difference contexts, blocks 2 to 4 reach about 0.5 early in every arm, then climb
        to about 0.65 in deep and split-anti and stay near 0.55 at the high hold. At block 1 on the other ops, deep
        drifts up through the second half to about 0.3, while the high hold and split-anti stay near 0.1.
    """
    return alpha_draw(data, alt)


@memo
def alpha_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="slice-terms-alpha",
        alt_text=alt_text,
        caption=f"""
            **Alignment at the example answers along training.** Left: α at the example answers of
            `{ex.ANCHORED_OP}` contexts, averaged over blocks 2 to 4. Right: α at block 1 at the example answers of
            the other ops. One hairline per run and the seed mean bold. The strip below is the anti weight: `deep`
            at the recipe hold (solid) and at 0.2 (dashed). `split-anti` follows the dashed line on the embedding
            and block 1 and the solid line on blocks 2 to 4.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.4), layout="constrained", sharex=True, height_ratios=[3, 1])
        for col, key in enumerate(("late", "b1_other")):
            ax = axes[0, col]
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for arm in data["arms"]:
                a = arm["arm"]
                for r in arm["runs"]:
                    ax.plot(r["epoch"], r[key], color=INK[a], lw=0.4, alpha=0.5, zorder=2)
                ep = arm["runs"][0]["epoch"]
                mean = np.mean([r[key] for r in arm["runs"]], axis=0)
                ax.plot(ep, mean, color=INK[a], lw=1.5, marker=MARK[a], markevery=12, ms=4, label=arm["name"], zorder=3)
            ax.set_ylim(-0.05, 0.75)
            sax = axes[1, col]
            for (name, sched), ls in zip(data["anti"].items(), ("-", "--"), strict=True):
                sax.plot(sched["epoch"], sched["w"], color=SCHED_INK, lw=1.1, ls=ls, label=f"anti weight, {name}")
            sax.set_ylim(0, 0.27)
            sax.set_yticks([0, 0.1, 0.2])
            sax.tick_params(labelsize=7)
            sax.set_xlabel("epoch", fontsize=9)
        axes[0, 0].set_ylabel(f"α, blocks 2 to 4,\n{ex.ANCHORED_OP}", fontsize=8)
        axes[0, 1].set_ylabel("α, block 1,\nother ops", fontsize=8)
        axes[1, 0].set_ylabel("weight", fontsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        sh, sl = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles + sh, labels + sl, loc="outside upper center", ncols=5, frameon=False, fontsize=7)
        return fig

    return _plot()


def alpha_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            fmt(r["alpha_fit_anchored"] for r in runs_of(a)),
            fmt(r["alpha_nofit_anchored"] for r in runs_of(a)),
            fmt(r["alpha_fit_other"] for r in runs_of(a)),
            fmt(answers_by_slice(label_of(a, s), "alpha_other")[1] for s in SEEDS),
            fmt(slope(r["emb_axis_colors"]) for r in runs_of(a)),
        ]
        for a in ("late", "late-anti0.2", "split-anti")
    ]
    return table_html(
        [
            "arm",
            f"fitting, `{ex.ANCHORED_OP}`",
            f"non-fitting, `{ex.ANCHORED_OP}`",
            "fitting, other ops",
            "block 1, other ops",
            "slope in the table",
        ],
        rows,
        f"""
        **Where the anchor sits at the end of training**, seed mean and range over {len(SEEDS)} runs. The first
        three columns are α at the example answers averaged over blocks 2 to 4, on held-out contexts, split by
        whether the example fits `{ex.ANCHORED_OP}` (whether that op could give the answer it shows) and by the op
        of the context. "Block 1, other ops" is α at block 1 at the example answers of the other ops, on the probe
        contexts. The slope is the lean of the color embedding table: the e₁ component of the color embeddings
        against lightness. The controls are at about 0.04 in the first three columns and lean either way, up to
        about ±0.25.
        """,
    )


rf"""
## What the anti weight presses on (E2)

Did the removal come back for the reason we expected, the alignment at the pulled blocks restored with block 1 and the table still off e₁ on the other ops? The main measure is α at the example answers: of `{ex.ANCHORED_OP}` contexts at blocks 2 to 4, and of the other ops at block 1.

{alpha_figure()}

It did. At blocks 2 to 4, `split-anti` follows `deep` at the recipe hold: the alignment reaches about a half early in training and climbs through the second half, where the high hold everywhere holds it flat. At block 1 on the other ops it follows the high hold, and stays low where `deep` drifts up through the second half.

{alpha_table()}

Split by whether an example fits `{ex.ANCHORED_OP}`, the alignment at the fitting answers is back to nearly the recipe-hold level, so the pressure that the high hold put on the pulled blocks has been lifted.

The non-fitting answers are back near the recipe-hold level too. That alignment was never about the op: the high hold everywhere had cleared it and the split leaves it in place, but it does not show up in the spill.

The color embedding table leans about a third as much as in `deep`, a little more than at the high hold, and on every seed it leans the way the anchored runs lean. Block 1 on the other ops is a little above the high hold too. Both fit the small spill that remains on one seed.
"""

# %%


def pool_figure() -> str:
    data = {
        "seeds": [
            {
                "seed": s,
                "answers": answers_by_slice(label_of("pool-slices", s)).tolist(),
                "query": np.asarray(TRAJ[label_of("pool-slices", s)]["traj"]["alpha_anchored"])[-1][
                    :, QUERY_ANSWER
                ].tolist(),
                "split": np.mean([answers_by_slice(label_of("split-anti", s))], axis=0).tolist(),
            }
            for s in SEEDS
        ],
    }
    alt = f"""
        On seeds 701 to 703 the pooled pull leaves the example answers at about 0.2 at block 1, rising to 0.65 to
        0.75 at block 4, close to split-anti at the same seed; on seed {POOL_FAILED[0] if POOL_FAILED else ""} the
        example answers stay near zero at every slice while the query answer reaches 1 at blocks 3 and 4.
    """
    return pool_draw(data, alt)


@memo
def pool_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="slice-terms-pool",
        alt_text=alt_text,
        caption=f"""
            **Where the pooled pull settles, by seed.** α at the end of training on `{ex.ANCHORED_OP}` probe
            contexts, at each slice: at the example answers (filled markers) and at the query answer (open
            markers), for `pool-slices`. The hairline is `split-anti` at the example answers, at the same seed.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(data["seeds"]), figsize=(7.2, 2.4), layout="constrained", sharey=True)
        x = np.arange(len(SLICE_NAME))
        ink = INK["pool-slices"]
        for ax, d in zip(axes, data["seeds"], strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            ax.plot(x, d["split"], color=INK["split-anti"], lw=0.6, alpha=0.8, zorder=1, label="split-anti, answers")
            ax.plot(x, d["answers"], color=ink, lw=1.4, marker="D", ms=4.5, zorder=3, label="example answers")
            ax.plot(
                x, d["query"], color=ink, lw=0.9, ls="--", marker="o", mfc="none", ms=4.5, zorder=3,
                label="query answer",
            )  # fmt: skip
            ax.set_title(f"seed {d['seed']}", fontsize=8)
            ax.set_xticks(x, SLICE_NAME, rotation=60, fontsize=7)
            ax.set_ylim(-0.1, 1.08)
        axes[0].set_ylabel("α at the end of training", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


def pool_table() -> str:
    rows = []
    for s in SEEDS:
        r = RUNS[label_of("pool-slices", s)]
        c = dose_curves("pool-slices", s)
        rows.append(
            [
                str(s),
                SLICE_NAME[SETTLED[s]],
                num(removal(r)),
                num(removal(r, BLOCK1)),
                num(removal(r, LATE_SET)),
                num(max(c["spill"]), ".3f"),
            ]
        )
    return table_html(
        ["seed", "largest α at answers", "removal", "block 1 only", "blocks 2 to 4 only", "spill, any dose"],
        rows,
        f"""
        **The pooled pull, by seed.** The slice with the largest α at the example answers of `{ex.ANCHORED_OP}`
        probe contexts, among blocks 1 to 4. Removal at the full dose of the edit at every slice, then of the edit
        on block 1 alone and on blocks 2 to 4 alone. The spill is the largest net spill over the four doses of the
        edit at every slice.
        """,
    )


rf"""
## Where the pooled pull settles (E3)

Pooled over slices as well as positions, the pull only needs each labeled context to align at one place along the stream. Each place gets a share of the pull that grows steeply with how well aligned it already is, so the pull should settle on whichever slice aligned first.

Which slice is that, and does the edit at every slice still remove the op? The main measure is α at the example answers at each slice.

{pool_figure()}

On three seeds the alignment is largest at block 4, and the profile over slices looks much like that of `split-anti`: low at block 1 and rising to block 4, with blocks 2 and 3 nearly as high. Block 1 aligns less than in `split-anti`. On those seeds the spill is within the criterion at every dose, and the edit removes the op as well as in `split-anti` on two of them and about a tenth less on the third (seed 702).

{pool_table()}

On {seeds_text(POOL_FAILED)} the example answers stayed near zero at every slice, and the edit removes nothing. What aligned instead is the query answer (the state at the answer token of the query, which has seen the answer and so can show the op), reaching α of 1 at blocks 3 and 4. That position aligns in every arm, since it is part of every labeled context, but in `pool-slices` on this seed it took the whole of the pull.

Along training the example answers rose to about a fifth early on, then fell back as the query answer took over. The prediction of the query answer is made at the query `=`, before that position, so an edit there cannot reach it.
<!-- REVIEW: the query answer is position {QUERY_ANSWER} of the probe contexts. Verify: traj["alpha_anchored"][-1][:, {QUERY_ANSWER}] for pool-slices-s{POOL_FAILED[0] if POOL_FAILED else ""}, and the same at the example answers. -->

This is the concentration the pooling was expected to bring, landing on a position no one meant it to use. The arm changes two things against `split-anti`, since block 1 joins the pulled slices as well as the pull being pooled; but block 1 took almost none of the pull on this seed, so the pooling is the likelier cause.
<!-- REVIEW: added the second difference of `pool-slices` (block 1 in the pulled set), which the first draft left out. Verify: answers_by_slice for pool-slices-s700 at block 1 (about 0.06) and the query answer at block 1 (0.42), against blocks 3 and 4. --> The per-slice pool includes the query answer too, and in every arm it aligns there about as far at blocks 2 to 4 as here. Why no per-slice run lost its example answers the same way, we can't say from these runs, and with one run in four it may be chance which position gets ahead.
"""

# %%


def ladder_profile(arm: str) -> dict:
    lab = label_of(arm, TAU_SEED)
    return {
        "answers": answers_by_slice(lab).tolist(),
        "query": np.asarray(TRAJ[lab]["traj"]["alpha_anchored"])[-1][:, QUERY_ANSWER].tolist(),
    }


def depth_figure() -> str:
    data = {
        "rungs": [{"arm": a, "slice_tau": t} | ladder_profile(a) for a, t in LADDER],
        "split": ladder_profile("split-anti")["answers"],
    }
    alt = """
        At every temperature the pull reaches the same level at block 4; softening the pool raises blocks 1 and 2 at the
        example answers to about where split-anti has them, and the query answer aligns at blocks 3 and 4 throughout.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="slice-terms-depth",
        alt_text=alt_text,
        caption=f"""
            **Where the pull settles as the pool over slices softens**, seed {TAU_SEED}. α at the end of training on
            `{ex.ANCHORED_OP}` probe contexts at each slice: at the example answers (left) and at the query answer
            (right), one line per temperature of the pool over slices, from the recipe τ (`pool-slices`, lightest)
            to the softest. The dotted line on the left is `split-anti` at the same seed.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained", sharey=True)
        x = np.arange(len(SLICE_NAME))
        n = len(data["rungs"])
        shades = [
            light_dark(
                to_hex(plt.cm.viridis(0.85 - 0.75 * i / (n - 1))), to_hex(plt.cm.viridis(0.25 + 0.7 * i / (n - 1)))
            )
            for i in range(n)
        ]
        marks = ["o", "D", "^", "s", "v"]
        axes[0].plot(x, data["split"], color=INK["split-anti"], lw=1.0, ls=":", zorder=4, label="split-anti")
        for ax, key in zip(axes, ("answers", "query"), strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for r, ink, m in zip(data["rungs"], shades, marks, strict=True):
                ax.plot(x, r[key], color=ink, lw=1.3, marker=m, ms=4, zorder=3, label=f"slice τ {r['slice_tau']:g}")
            ax.set_xticks(x, SLICE_NAME, rotation=60, fontsize=7)
            ax.set_ylim(-0.1, 1.08)
        axes[0].set_title("example answers", fontsize=8)
        axes[1].set_title("query answer", fontsize=8)
        axes[0].set_ylabel("α at the end of training", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def depth_table() -> str:
    rows = []
    for arm, t in (*LADDER, ("split-anti", None)):
        r = RUNS[label_of(arm, TAU_SEED)]
        c = dose_curves(arm, TAU_SEED)
        a = answers_by_slice(label_of(arm, TAU_SEED))
        rows.append(
            [
                f"{t:g}" if t is not None else "`split-anti`",
                num(a[1]),
                num(a[4]),
                num(removal(r)),
                num(removal(r, BLOCK1)),
                num(removal(r, LATE_SET)),
                num(max(c["spill"]), ".3f"),
                num(task_change(r), ".3f"),
            ]
        )
    return table_html(
        [
            "slice τ",
            "α, block 1",
            "α, block 4",
            "removal",
            "block 1 only",
            "blocks 2 to 4 only",
            "spill, any dose",
            "task change",
        ],
        rows,
        f"""
        **The pooled pull by temperature of the pool over slices**, seed {TAU_SEED}, with `split-anti` at the same
        seed below. α at the example answers of `{ex.ANCHORED_OP}` probe contexts at block 1 and block 4. Removal at
        the full dose of the edit at every slice, then on block 1 alone and on blocks 2 to 4 alone. The spill and
        task change are as in E1. The first line is `pool-slices`, whose pool over slices is at the recipe τ.
        """,
    )


rf"""
## A softer pool over depth (E4)

With the slices pooled at the recipe τ, the pull may put nearly all its weight on whichever slice is ahead, and on three seeds that was block 4. A softer pool over slices spreads the pull back over depth: at a high enough temperature it is the mean of one pool per slice. So we trained the `pool-slices` arm again at seed {TAU_SEED}, a seed where it settled on block 4, with the pool over slices at {span_text([t for _, t in LADDER[1:]], ".1f")}, each double the last, while the pool over positions stays at the recipe τ. At the recipe τ the two pools are the one joint pool of E3, so the `pool-slices` run is the first rung. The main measure is again α at the example answers at each slice.

{depth_figure()}

Softening the pool did not take the alignment away from block 4: it stays at about {num(np.mean([answers_by_slice(label_of(a, TAU_SEED))[4] for a, _ in LADDER]))} at every temperature, the level `split-anti` has it at too. What changes is the earlier blocks. At the recipe τ, block 1 is at {num(answers_by_slice(label_of("pool-slices", TAU_SEED))[1])} and block 2 below the blocks after it. From the first softer step on, both rise, and the profile over slices looks like that of `split-anti` at this seed. So on this seed the pool over slices at the recipe τ had not latched onto block 4 so much as held back the blocks before it, and a softer pool lets them align too.

The query answer shifts the same way, with blocks 1 and 2 higher at the softer temperatures. It aligns at blocks 2 to 4 at every temperature, as in the other arms, and the example answers keep their alignment, so this seed shows nothing of the seed {POOL_FAILED[0] if POOL_FAILED else ""} outcome. One seed cannot say whether a softer pool would have kept that seed on the example answers.

{depth_table()}

Every run on this ladder is within the criterion at every dose, and the task change is small. The largest spill rises a little at the two softest temperatures, toward that of `split-anti` at this seed. Removal under the full edit is {span_text(removal(RUNS[label_of(a, TAU_SEED)]) for a, _ in LADDER)} with no trend across the temperatures, and the edit on block 1 alone or on blocks 2 to 4 alone moves around from one run to the next, also with no trend. With one run per temperature, differences of this size are within the spread between seeds that `pool-slices` and `deep` at a hold of 0.2 showed in E1, so they do not show an effect of the temperature.
"""

# %%

r"""
## Discussion

The high hold everywhere had two effects, at different slices. On the embedding and block 1 it keeps the stand-ins off e₁, which fits where the spill came from; on the pulled blocks it presses on the alignment at the fitting example answers, which fits where the removal went. Splitting the weight keeps the first and lifts the second, as the second review expected.

The one `split-anti` seed past the criterion spills on the same op where `deep` spilled least, and a little more of block 1 and the table sit on e₁ than at the high hold everywhere. So the split may trade a small amount of selectivity for the removal it gives back. If that trade is typical, a hold a little above 0.2 on the first two slices, or the hold of 0.2 with a higher one on block 1, would be the place to look.

Where the example answers aligned early, the pooled pull settled on block 4 and the result was much like the split. Where the query answer got ahead, the pull settled there alone, a position that is in every labeled context but comes after the prediction it would matter for. As a recipe the pooled pull would need the query answer left out of its pool, and even then it would make the anchor depend on which position aligns first.

On the one seed tried, a softer pool over slices made the pooled pull look like the split: blocks 1 to 4 all aligned, with block 4 still the furthest along. So the softer pool takes away the concentration over slices, which was the point of pooling over them, and on this seed what remained was close to the per-slice pull of `split-anti`.
"""

# %%

rf"""
## Glossary

Alignment
α
:   How far a state points along e₁: the cosine between the state and e₁, which is 1 on the anchor direction and 0 off it.

Anti-subspace term
:   The companion to the pull: the mean of α² over every position and slice it acts on, which asks the states as a whole to stay off e₁. Its weight starts at {ex.ANTI_PEAK:g} and anneals to a level it then holds, the hold.

Fitting example
:   An example whose answer is one that `{ex.ANCHORED_OP}` could give on its operands, whatever op the context is under.

Stand-in
:   Something token-level that satisfies the pull where no contextual concept can: lightness on e₁ in the color embedding table (the lean), or a syntax token embedding on e₁ (a latch).

Removal
:   How much of `{ex.ANCHORED_OP}` the edit takes out: the net drop in its score under the edit, as a share of the way from the clean score to the target null.

Spill
:   The largest net drop in score on another op under the edit. The selectivity criterion is a spill of at most {CRITERION:g}.

Target null
:   The score an ideal predictor would get on `{ex.ANCHORED_OP}` contexts if it had lost that op and answered from the remaining ones.
"""
