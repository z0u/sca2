# ruff: noqa: B018
# title: A soft pool while the anchor settles

# An exploratory report: a pooling temperature that starts soft, and a pull on slices 1 to 3 only, crossed, beside
# the `falloff` recipe of record at the split-anti-seeds seeds, and a pool over slices 1 to 3 with both temperatures
# soft. `experiment.py` beside this script trains, measures and dose-scores the new runs; the `falloff` runs and the
# controls come from split-anti-seeds and ex-2.2.23.
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
TRACK_SLICE = 3  # The deepest slice every arm pulls: the step pulls 2 to 4, the ramp and the pool 1 to 3.

BASE = ex.BASELINE
ARMS = (BASE, *(a.name for a in ex.ARMS))
STEP_SOFT, RAMP, RAMP_SOFT, POOL_SOFT = (a.name for a in ex.ARMS)
NAME = {BASE: "falloff"} | {a.name: a.shown_as for a in ex.ARMS}
PULL = {BASE: list(ex.STEP)} | {a.name: list(a.pull) for a in ex.ARMS}
SOFT = {BASE: False} | {a.name: a.soft for a in ex.ARMS}
POOLED = {BASE: False} | {a.name: a.pool_slices for a in ex.ARMS}
ANTI = list(ex.FALLOFF_HOLDS)


def label_of(arm: str, seed: int) -> str:
    return ex.label_of(arm, seed)  # split-anti-seeds labels its runs the same way


def run_of(arm: str, seed: int) -> dict:
    return RUNS[label_of(arm, seed)]


def control_of(seed: int) -> dict:
    return RUNS[ex.ex2223.label_of("control", ex.EPOCHS, seed)]


def traj_of(arm: str, seed: int) -> dict:
    return TRAJ[label_of(arm, seed)]["traj"]


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
    a = np.asarray(traj_of(arm, seed)[key])[-1]
    return a[:, pos].mean(axis=1) if isinstance(pos, list) else a[:, pos]


def alpha_track(arm: str, seed: int, pos: list[int] | int, sl: int = TRACK_SLICE) -> list[float]:
    """α on the `difference` probe contexts at slice *sl* along training, at *pos* (averaged over a list)."""
    a = np.asarray(traj_of(arm, seed)["alpha_anchored"])[:, sl]
    return (a[:, pos].mean(axis=1) if isinstance(pos, list) else a[:, pos]).tolist()


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
    STEP_SOFT: light_dark("#e08e0b", "#f5b950"),
    RAMP: light_dark("#0b3c73", "#9fcbff"),
    RAMP_SOFT: light_dark("#c0392b", "#ff8a76"),
    POOL_SOFT: light_dark("#6c4fa3", "#b9a2e8"),
}
MARK = {BASE: "D", STEP_SOFT: "v", RAMP: "s", RAMP_SOFT: "^", POOL_SOFT: "o"}
GHOST = light_dark("#bbb", "#555")
RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
SLICE_TICK = ["emb", "1", "2", "3", "4"]

# --- Summaries the prose quotes -------------------------------------------------------------------------------

CURVES = {a: {s: dose_curves(a, s) for s in SEEDS} for a in ARMS}
FULL = {a: {s: CURVES[a][s]["removal"][-1] for s in SEEDS} for a in ARMS}
WORST_SPILL = {a: {s: max(CURVES[a][s]["spill"]) for s in SEEDS} for a in ARMS}
N_WITHIN = {a: sum(v <= CRITERION for v in WORST_SPILL[a].values()) for a in ARMS}
TASK = {a: [float(op_change(a, s).mean()) for s in SEEDS] for a in ARMS}
LOST_BELOW = 0.5  # A run whose full edit removes less than half the op lost the anchor (as in split-anti-seeds).
LOST = {a: [s for s in SEEDS if FULL[a][s] < LOST_BELOW] for a in ARMS}
KEPT = {a: [s for s in SEEDS if s not in LOST[a]] for a in ARMS}
SYNTAX_E1 = {a: {s: max(abs(v) for v in run_of(a, s)["syntax"].values()) for s in SEEDS} for a in ARMS}


def tau_of(arm: str) -> tuple[str, str]:
    """The two pooling temperatures of *arm*, as the table writes them."""
    tau = f"{ex.TAU_START:g} → 0.1" if SOFT[arm] else "0.1"
    slice_tau = f"{ex.SLICE_TAU_START:g} → {ex.SLICE_TAU:g}" if POOLED[arm] else "–"
    return tau, slice_tau


def arms_table() -> str:
    rows = [[f"`{NAME[a]}`", *(frac(w) for w in PULL[a]), *tau_of(a)] for a in ARMS]
    rows.append(["every arm, anti hold", *(f"{w:.2f}" for w in ANTI), "", ""])
    return table_html(
        ["arm", *SLICE_TICK, "τ_p", "τ_ℓ"],
        rows,
        r"""
        **The arms.** The pull weight on each slice, relative to the anchor weight $\lambda_a$ on each slice of the
        step, and the two pooling temperatures. An arrow is a temperature that sharpens from its first value at epoch 0
        to its second by the end of the pull warm-up (epoch 40). In `pool 1–3, soft τ` the weights only say which
        slices are in the pool. The anti hold is the weight the anti term anneals to, the same `falloff` line in every
        arm, rounded here to two places.
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
        to between about 0.7 and 0.9 removal at the full dose. One dashed run of the fixed ramp stays near zero at
        every dose. Spill stays near zero on most runs; each soft arm has a couple of runs just over the dashed
        criterion at the full dose, and one run of the soft pool reaches about 0.07.
    """
    return dose_draw(data, alt)


@memo
def dose_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-dose",
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
                ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
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


def worst_op(a: str) -> str:
    v, s, op = min((float(op_change(a, s).min()), s, OPS[int(op_change(a, s).argmin())]) for s in SEEDS)
    return f"{num(v, '.3f')} (`{op}`, {s})"


def summary_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
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
            "kept the anchor ↑",
            "removal, kept ↑",
            "within criterion ↑",
            "spill, any dose ↓",
            "task change ↑",
            "worst op change ↑",
        ],
        rows,
        f"""
        **Removal, spill and task**, seed mean and range. A run kept the anchor when the full edit removes at least
        {LOST_BELOW:g} of the op, and removal is over those runs, at the full dose. "Spill, any dose" is the largest
        net spill over the four doses, and "within criterion" counts the runs where it is at most {CRITERION:g}. The
        task change is the mean score over ops less that of the control, and the worst op change is the largest loss
        on any one op at any seed. Arrows mark which direction is better.
        """,
    )


def lost_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            str(s),
            num(FULL[a][s]),
            num(WORST_SPILL[a][s], ".3f"),
            num(alpha_by_slice(a, s)[TRACK_SLICE]),
            num(alpha_by_slice(a, s, pos=QUERY_ANSWER)[TRACK_SLICE]),
        ]
        for a in ARMS
        for s in LOST[a]
    ]
    kept = [alpha_by_slice(a, s)[TRACK_SLICE] for a in ARMS for s in KEPT[a]]
    return table_html(
        ["arm", "seed", "removal", "spill, any dose", "α, example answers", "α, query answer"],
        rows,
        f"""
        **The runs that lost the anchor.** Removal at the full dose, the largest net spill at any dose, and α at
        slice {TRACK_SLICE} at the end of training on `{ex.ANCHORED_OP}` probe contexts, at the example answers and
        at the query answer. On the runs that kept the anchor, α at the example answers is {fmt(kept)} at slice
        {TRACK_SLICE}.
        """,
    )


def schedule_of(arm: str) -> dict:
    """The anchor weight λ_a and the position pool temperature τ_p along training, which every seed of an arm
    shares. A fixed τ_p is not recorded in the trajectory, so it is drawn flat at the recipe value.
    """
    t = traj_of(arm, SEEDS[0])
    tau = np.asarray(t["tau"])[:, 0] if "tau" in t else np.full(len(t["epoch"]), ex.st.ex2216.TAU)
    return {"epoch": t["epoch"], "weight": t["weight"], "tau": tau.tolist()}


def track_figure() -> str:
    data = {
        "slice": TRACK_SLICE,
        "arms": [
            {
                "arm": a,
                "name": NAME[a],
                "schedule": schedule_of(a),
                "runs": [
                    {
                        "epoch": traj_of(a, s)["epoch"],
                        "answers": alpha_track(a, s, ANSWERS),
                        "query": alpha_track(a, s, QUERY_ANSWER),
                        "lost": s in LOST[a],
                    }
                    for s in SEEDS
                ],
            }
            for a in ARMS
        ],
    }
    alt = """
        Alignment at slice 3 against epoch, first hundred epochs, one column per arm. Top row, example answers:
        under falloff most runs rise past 0.4 by epoch 20; under the soft arms every run rises a few epochs later,
        close together; in the fixed ramp one dashed run rises to 0.3, drops back and wobbles below 0.4 until about
        epoch 100. Second row, the query answer rises to about 0.8 in every run, a little later under the soft arms.
        Underneath, the anchor weight ramps from 0 to 0.1 by epoch 40 in every arm, and the temperature falls from 1
        to 0.1 by epoch 40 in the soft arms and stays at 0.1 in the others.
    """
    return track_draw(data, alt)


@memo
def track_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-track",
        alt_text=alt_text,
        caption=f"""
            **Alignment along the first hundred epochs**, at slice {data["slice"]}, on the probe contexts of
            `{ex.ANCHORED_OP}`, one column per arm: at the example answers (top) and at the query answer (second
            row). One line per run, dashed where the run lost the anchor; the other arms are ghosts behind the top
            row. The strips underneath are the anchor weight $\\lambda_a$ and the position pool temperature
            $\\tau_p$ (log scale).
        """,
    )
    def _plot() -> plt.Figure:
        arms = data["arms"]
        fig, axes = plt.subplots(
            4,
            len(arms),
            figsize=(7.4, 4.6),
            layout="constrained",
            sharex=True,
            sharey="row",
            height_ratios=[3, 2, 0.8, 0.8],
            squeeze=False,
        )
        for col, arm in enumerate(arms):
            ax, qax, wax, tax = axes[:, col]
            ink = INK[arm["arm"]]
            for a_ in (ax, qax):
                a_.axhline(0, color=ZERO, lw=0.5, zorder=0)
            for other in arms:
                if other is not arm:
                    for r in other["runs"]:
                        ax.plot(r["epoch"], r["answers"], color=GHOST, lw=0.4, alpha=0.4, zorder=1)
            for r in arm["runs"]:
                ls = "--" if r["lost"] else "-"
                ax.plot(r["epoch"], r["answers"], color=ink, lw=0.8, ls=ls, zorder=2)
                qax.plot(r["epoch"], r["query"], color=ink, lw=0.8, ls=ls, zorder=2)
            ax.set_title(arm["name"], fontsize=7)
            ax.set_ylim(-0.05, 0.8)
            qax.set_ylim(-0.05, 1.05)
            sch = arm["schedule"]
            wax.plot(sch["epoch"], sch["weight"], color=RULE, lw=1.0)
            wax.set_ylim(0, 0.12)
            wax.set_yticks([0, 0.1])
            tax.plot(sch["epoch"], sch["tau"], color=ink, lw=1.1)
            tax.set_yscale("log")
            tax.set_ylim(0.07, 1.4)
            tax.set_yticks([0.1, 1], ["0.1", "1"])
            tax.minorticks_off()
            tax.set_xlim(0, 100)
            tax.set_xlabel("epoch", fontsize=8)
            for a_ in (ax, qax, wax, tax):
                a_.tick_params(labelsize=7)
        axes[0, 0].set_ylabel("α, example\nanswers", fontsize=8)
        axes[1, 0].set_ylabel("α, query\nanswer", fontsize=8)
        axes[2, 0].set_ylabel("$\\lambda_a$", fontsize=8)
        axes[3, 0].set_ylabel("$\\tau_p$", fontsize=8)
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
            if KEPT[a]
        ],
    }
    alt = """
        Alignment at the example answers by slice for the runs that kept the anchor. On difference every arm rises
        from near zero at the embedding to about 0.6 at slice 3; at slice 4 the step arms stay near 0.64 while the
        ramp and pool arms fall to about 0.46. On the other ops every arm peaks at slice 1, at about 0.15 to 0.21.
        Underneath, the pull weight by slice and the anti hold falling from 0.2 at the embedding to 0.03 at slice 4.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-depth",
        alt_text=alt_text,
        caption=f"""
            **Alignment at the example answers by slice**, at the end of training, on the probe contexts of
            `{ex.ANCHORED_OP}` (left) and of the other ops (right), over the runs that kept the anchor: the seed mean
            in the arm's marker; the two top panels have their own scales. Underneath are the pull weight at each
            slice, relative to the step (for the pooled arm, the slices in its pool), and the anti hold, which every
            arm shares.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            3, 2, figsize=(6.4, 4.2), layout="constrained", sharex=True, height_ratios=[3, 1.2, 0.8]
        )
        x = np.arange(len(SLICE_TICK))
        for col, key in enumerate(("anchored", "other")):
            ax, pax, aax = axes[:, col]
            ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
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
            pax.set_ylim(-0.05, 1.2)
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
        + [num(float(np.mean([alpha_by_slice(a, s)[sl] for s in KEPT[a]]))) if KEPT[a] else "–" for sl in range(1, 5)]
        + [
            num(float(np.mean([alpha_by_slice(a, s, pos=QUERY_ANSWER)[sl] for s in KEPT[a]]))) if KEPT[a] else "–"
            for sl in (3, 4)
        ]
        + [
            num(float(np.mean([alpha_by_slice(a, s, "alpha_other")[sl] for s in KEPT[a]]))) if KEPT[a] else "–"
            for sl in (1, 2)
        ]
        + [fmt(SYNTAX_E1[a].values(), ".3f")]
        for a in ARMS
    ]
    return table_html(
        [
            "arm",
            "α at 1",
            "α at 2",
            "α at 3",
            "α at 4",
            "query, 3",
            "query, 4",
            "α other ops, 1",
            "α other ops, 2",
            "syntax e₁",
        ],
        rows,
        f"""
        **Alignment by slice, and the syntax embeddings.** α at the example answers on `{ex.ANCHORED_OP}` and on the
        other ops, seed mean over the runs that kept the anchor. "Query" is α at the query answer on `{ex.ANCHORED_OP}`.
        "Syntax e₁" is the largest e₁ component (in size)
        among the embeddings of `?`, `=`, `,` and the newline, seed mean and range over every run; a latch would show
        as a value near one.
        """,
    )


RISE = 0.4  # The level at which E2 calls the example answers on e₁, as in anchor-ramp.
EARLY = 16  # The epoch E2 compares the arms at, early in the warm-up where anchor-ramp's lost runs fell behind.


def first_rise(arm: str, seed: int) -> float:
    """The first recorded epoch at which α at the example answers, at the tracked slice, goes over `RISE`."""
    t = np.asarray(alpha_track(arm, seed, ANSWERS))
    above = np.flatnonzero(t > RISE)
    return float(np.asarray(traj_of(arm, seed)["epoch"])[above[0]]) if len(above) else float("nan")


def at_epoch(arm: str, seed: int, pos: list[int] | int, epoch: float) -> float:
    ep = np.asarray(traj_of(arm, seed)["epoch"])
    return float(np.asarray(alpha_track(arm, seed, pos))[int(np.argmin(abs(ep - epoch)))])


RISES = {a: [first_rise(a, s) for s in SEEDS] for a in ARMS}


def settle_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            fmt((at_epoch(a, s, QUERY_ANSWER, EARLY) for s in SEEDS)),
            fmt((at_epoch(a, s, ANSWERS, EARLY) for s in SEEDS)),
            fmt(RISES[a], ".0f"),
            fmt((at_epoch(a, s, ANSWERS, 48) for s in SEEDS)),
        ]
        for a in ARMS
    ]
    return table_html(
        [
            "arm",
            f"α query answer, epoch {EARLY}",
            f"α example answers, epoch {EARLY}",
            f"first epoch over {RISE:g}",
            "α example answers, epoch 48",
        ],
        rows,
        f"""
        **How the runs settle.** α at slice {TRACK_SLICE} on the `{ex.ANCHORED_OP}` probe contexts, seed mean and
        range over every run: at the query answer and the example answers at epoch {EARLY}, the first recorded epoch
        at which the example answers go over {RISE:g} (records are every four epochs), and the example answers at
        epoch 48, just after the warm-up.
        """,
    )


# %%

rf"""
# A soft pool while the anchor settles

/// tip |
<!-- lede -->
We tried the two changes the anchor-ramp losses suggested, crossed: a pooling temperature that starts soft and sharpens over the pull warm-up, and a pull on slices 1 to 3 only, with nothing on slice 4. Every arm with the soft start kept the anchor on every seed, but the edit spilled more under each of them than at the fixed temperature. The ramp over slices 1 to 3 at the fixed temperature lost the anchor on one seed, in a new way.
///

In [the anchor-ramp experiment](/docs/m2/anchor-ramp/report.py), a pull that ramps with depth lost the anchor on two or three of eight seeds, where the step in [`falloff`](/docs/m2/split-anti-seeds/report.py) lost it on none. The lost runs split off from the kept ones during the pull warm-up, around epochs 16 to 24: on a kept run the example answers rise onto e₁ along with the query answer, and on a lost run they fall back. The losses also followed how little pull slices 1 to 3 had, though that ordering was picked after the counts were in.

The pull is pooled over the positions of each context at a sharp temperature, $\tau_p$ = 0.1,[^pool] so while the query answer is the position closest to e₁, it gets almost all of the pull. A soft pool early on would share the pull with the example answers while they are still deciding.

A pull on slices 1 to 3 alone, with nothing on slice 4, would put the pull where it mostly moves the example answers.

[^pool]: Pooling temperature: the pull on a context is a softmin-weighted mix over its positions. A low temperature gives nearly all the weight to the closest position; a high one spreads it evenly.

The {len(ex.ARMS)} new arms, of {len(ex.SEEDS)} runs each, cross the two changes and add a pool over slices 1 to 3 with both temperatures soft. They use the seeds of the split-anti-seeds experiment, so its `falloff` runs serve as the baseline with no new training.
"""

# %%

rf"""
## Observations

- [Removal and spill (E1)](#removal-and-spill-e1): the soft start kept the anchor on every seed in all three arms that had it, and the ramp at the fixed temperature lost it on one. Removal on the kept runs was close to that of `falloff` everywhere. But each soft arm had runs over the spill criterion, and the pool over slices 1 to 3 had the most.
- [When the runs settle (E2)](#when-the-runs-settle-e2): a soft pool holds every position back through the first part of the warm-up, then the example answers come onto e₁ with the query answer on every seed, a record or two later than under `falloff`. The one lost run fell back after it had risen partway, climbed back only slowly from about epoch 100, and ended below every kept run, with the edit removing almost nothing.
- [Where the anchor sits by depth (E3)](#where-the-anchor-sits-by-depth-e3): without a pull on slice 4, the alignment there falls back from its slice-3 level, and the pooled arm settles where the ramp does. The syntax embeddings stay off e₁ in every arm, so the extra spill under a soft pool does not come from a latch.

## Scope

This is an exploratory study, with no preregistration and no gate. Two factors are crossed, with `falloff` as the condition where both are at their recipe level:

- The pull shape: the step on slices 2 to 4 (the recipe), or a ramp over slices 1 to 3 at a third, two thirds and the full weight, with nothing on the embedding or slice 4. The ramp adds up to the step over slices 1 to 3 (two each), so the earlier slices keep the pull the step gives them, and the pull on slice 4 goes.
- The position pool temperature $\tau_p$: fixed at 0.1 (the recipe), or soft-to-sharp, starting at {ex.TAU_START:g} and sharpening to 0.1 by epoch 40, the end of the pull warm-up. The schedule is smooth in log $\tau_p$, so the temperature halves at a steady pace and comes to rest at 0.1. At epoch 20, when the anchor-ramp runs split, it is about halfway there in log terms: a query answer well ahead of an example answer takes a few times as much pull as the example answer, against about fifty times at 0.1.

{arms_table()}

The fourth arm pools the pull over the slice × position pairs of slices 1 to 3 (the nested pool of [the slice-terms experiment](/docs/m2/slice-terms/report.py)), so the pull can choose the slice as well as the position. Its slice pool temperature $\tau_\ell$ starts at {ex.SLICE_TAU_START:g}, near a plain mean over slices, and sharpens to {ex.SLICE_TAU:g} over the same window as $\tau_p$.

The end value rests on one seed. In E4 of slice-terms, at seed 701, every $\tau_\ell$ from 0.2 to 1.6 removed about three quarters to nine tenths of the op, with no trend and every run within the criterion, while the joint pool at $\tau_\ell$ = 0.1 lost the anchor on one seed of four. So {ex.SLICE_TAU:g} is inside the range that worked there.

A soft pool and a ramp each change more than one thing:

- A softer $\tau_p$ spreads the pull of each context over its positions, but the total pull per context stays the same at every temperature (the softmin weights sum to one). In [the τ×λ_a sweep](/docs/m2/tau-lambda-sweep/report.py), a softer fixed $\tau_p$ meant more spill, so spill is the thing to watch; here it is soft only during the warm-up.
- The ramp pulls two thirds as much in total as the step, all of it moved off slice 4. A difference between the ramp and the step can come from the shape or from the lower total.
- The pool over slices 1 to 3 pulls as much in total as the step, and lets the pull choose where it goes.

Every run trains at model seeds {ex.SEEDS[0]} to {ex.SEEDS[-1]} for 400 epochs, paired by seed with its `falloff` twin and with ex-2.2.23's control (same initialization, batches and label draws).

With eight seeds per arm, one or two lost runs could come from the seeds alone, so we weigh the counts across arms together. `falloff` lost no runs at these seeds, so the step pair can only show a soft start doing harm, and only the ramp pair can show it helping.

## The measurements

The measurements and dose scoring are those of the split-anti-seeds experiment.

- Removal: the net drop in the score on `difference` under the edit, as a share of the way from the clean score to the target null. The edit projects e₁ out of the state at every position and slice, at a dose of a quarter, a half, three quarters, or all of the e₁ component. Higher is better.
- Spill: the largest net drop on another op under the edit. A run is within the criterion when its spill is at most {CRITERION:g} at every dose.
- Both are net of the control at the same seed.
- A run lost the anchor when the full edit removes less than half of the op, as in the split-anti-seeds experiment.
- Task change: the mean score over ops, less that of the control. Zero means no damage.
- The alignment α with e₁ at the example answers and at the query answer, at every slice, along training and at the end.
"""

# %%

rf"""
## Removal and spill (E1)

Does a soft pool early, or a pull on slices 1 to 3 alone, keep the anchor on every seed, and keep the edit as selective as `falloff`? The main measure is the removal at the full dose, with the spill at any dose beside it. The soft step against `falloff` isolates the temperature, and the fixed ramp against `falloff` isolates the shape.

{dose_figure()}

Every run but one climbs with the dose to about seven tenths to nine tenths of the op, as `falloff` does. The exception is one run of `{NAME[RAMP]}` (the dashed line), which removes almost nothing at any dose.

With the soft start the same ramp kept the anchor on that seed, so in the one pair where the soft start could help, it did. But at a loss rate of one in eight, the soft twin would most often have kept that seed anyway, so this fits the soft start helping without being evidence that it does.
<!-- REVIEW: lowered the one-seed claim by its base rate. With one loss in eight under the fixed ramp, the soft
twin keeping that seed is the likely outcome even if the temperature does nothing, so it supports the reading only
as a consistency check. Verify: the counts in the summary table (7 of 8 against 8 of 8). -->

{summary_table()}

The soft start makes the edit less selective. `falloff` and `{NAME[RAMP]}` stayed within the criterion on every seed, and their soft twins went over it on two seeds each, by a little. The pool over slices 1 to 3 went over on half its seeds, one of them (seed 711) by more than three times the criterion. The τ×λ_a sweep found the same for a softer fixed temperature, so this seems to be the price of a soft pool, even one that is soft for the first tenth of training only.

The ramp also removes a little less. Over the kept runs, both ramps remove a few hundredths less than the step, and `{NAME[RAMP_SOFT]}` has three runs near seven tenths. The mean task score is unchanged in every arm, and no new arm loses more on any one op than `falloff` does.
"""

# %%

rf"""
## When the runs settle (E2)

Does the soft pool carry the example answers onto e₁ with the query answer through the warm-up, where the lost runs of the anchor-ramp experiment fell back? The main measure is α at the example answers along training, at slice {TRACK_SLICE}, the deepest slice every arm pulls. The query answer is in the row underneath, and the anchor weight and the temperature in strips below that.

{track_figure()}

A soft pool slows the start. In the step pair at epoch {EARLY}, the query answer under `falloff` is already partway onto e₁ with the example answers following, while under the soft start both are still near zero. A record or two later the example answers come up on every seed of `{NAME[STEP_SOFT]}`, all crossing {RISE:g} at the same record.

The ramp starts slowly at slice {TRACK_SLICE} at either temperature, since it pulls that slice at two thirds of the step weight. With the soft start its runs cross {RISE:g} within a few records of one another; at the fixed temperature one run is left behind.

{settle_table()}

So the soft pool does not lift the example answers ahead of the query answer, as we had pictured. A likely reading is that it spreads the pull so thinly that nothing moves much until the temperature is low enough to choose, and by then the example answers rise in step with the query answer.

On that reading the soft start works mostly as a delay. A warm-up that simply started later might settle the runs the same way, and no arm here tells the two apart. The step pair does show the soft start tightening the spread: under `falloff` one seed (711) crossed {RISE:g} late, at epoch 40, and under the soft start every seed crossed at the same record. Eight seeds per arm fit this reading without proving it.
<!-- REVIEW: named the delay reading as an alternative the arms don't separate, since the soft pool's effect at epoch
16 is indistinguishable from a later warm-up. Added the seed-711 tightening from the settle table as the one
observation on the step pair that bears on the benefit. Verify: first-rise column of the settle table. -->

The lost run of `{NAME[RAMP]}` (seed 708) is a different kind of loss from those in the anchor-ramp experiment. Its example answers rose partway with the others, fell back at the next record, and stayed below {RISE:g} until about epoch 100, while its query answer stayed on e₁. After that they climbed only to about four fifths of the kept level.

{lost_table()}

By the end its query answer is as aligned as on the kept runs, and its example answers end below every kept run but well above zero, yet the edit removes almost nothing on it, at any slice.

A likely reading is that the model learned to solve `difference` while the anchor was away, along a path that does not go through e₁, and does not use the alignment that came back later. One run can't settle that. But alignment alone would understate this kind of loss: the anchor-ramp losses ended near zero, and this one ends at four fifths of the kept level.
<!-- REVIEW: "nearly as aligned as the kept runs" was wrong for the example answers (0.49 against a kept range of 0.59
to 0.64, in the table above), so the claim is now that alignment understates the loss, not that it hides it. The
query answer (0.93 against 0.94 to 0.98) does support "as aligned". Verify: the lost-run table and its caption. -->
"""

# %%

rf"""
## Where the anchor sits by depth (E3)

On the runs that kept the anchor, where does the alignment sit by slice? With nothing on slice 4, does slice 4 still hold the concept as the residual stream (the running sum each block adds its output to) carries it up? Where does the pool over slices 1 to 3 settle, given that the pool over every block settled on block 4 in the slice-terms experiment? We also check the syntax embeddings, since in the τ×λ_a sweep a soft pool spilled through the `=` embedding coming onto e₁.

{depth_figure()}

Up to slice 3 the arms are nearly alike, and the soft start changes nothing there. At slice 4, where the ramp and the pool put no pull, the example answers fall back to about three quarters of their slice-3 level, and the query answer falls from near one to about four fifths. Under the step they hold their slice-3 level. So the concept rides up past the last pulled slice, though only partly, and the edit still finds it, since it acts on every slice.

The pool over slices 1 to 3 settles much as the ramp does, a little higher at slices 1 and 2. It peaks on slice 3, the last slice in the pool, as the slice-terms pool peaked on its last block. The other ops peak at slice 1 in every arm, as in `falloff`.

{depth_table()}

The syntax embeddings stay off e₁ in every arm, within a few hundredths, so nothing latches, and the spill of E1 does not come from the route the τ×λ_a sweep found. The other ops end about as aligned in the step pair, and a little more aligned under the soft ramp than under the fixed one, though not by enough to account for the spill. So where the extra spill comes from is still open.
"""

# %%

r"""
## Discussion

The soft start got the example answers onto e₁ on every seed it was tried on, apparently by holding the pull back until it can choose. That fits the anchor-ramp reading, that runs are lost in the first tens of epochs when the sharp pool gives the query answer nearly all of the pull. But the evidence for a benefit is one seed, which a one-in-eight loss rate would often keep anyway (E1).

The cost is clear: both soft twins spilled more than their fixed arms, and the pool over slices 1 to 3, which has no fixed twin, spilled most.

The ramp over slices 1 to 3 lost the anchor on one seed of eight at the fixed temperature, where the anchor-ramp ramps from slice 1 lost two or three. That fits the second anchor-ramp reading, that the earlier slices need about as much pull as the step gives them: this ramp gives slices 1 to 3 that much, and drops slice 4 instead.

Its one loss came later and in another way, with the alignment partly back by the end but no longer doing the work. That is a reason to keep scoring removal, and not alignment alone, when judging a run.

The soft ramp kept the anchor on every seed here without the step between slices 1 and 2, so it would be a candidate if its spill can be kept down. As it stands, `falloff` still keeps the anchor on every seed with the edit inside the criterion, which no new arm matched on both counts.
"""

# %%

r"""
## Method

The pooling temperatures are step inputs here, scheduled like the weights (`AnchorSpec.tau_start`, `slice_tau_start` and `tau_anneal_end`); before this experiment they were fixed when the training step was built. The schedule is minimum-jerk in log τ from the start value at epoch 0 to the end value at epoch 40, and holds after. Training otherwise follows the split-anti-seeds experiment unchanged. The trajectory of each run also keeps α by slice and position for each op separately, and the e₁ component of every syntax embedding, for later use. The 32 runs cost about \$11.
"""
