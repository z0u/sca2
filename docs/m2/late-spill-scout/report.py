# ruff: noqa: B018
# title: Late spill scout

# An exploratory report: three late-schedule changes to the soft pool scout's `τ_ℓ 0.1, τ_ℓ fixed`, each carried on
# from that condition's stored epoch-64 states at seeds 704 to 711. `experiment.py` beside this script trains and
# scores the runs.

import json
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


AXES_BODY = fetch_json(ex.AXES_REF)
DESIGN = AXES_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
FIXED, BY40 = ex.CONDITION, ex.SECOND
RISERS = [(FIXED, 707), (BY40, 705), (BY40, 707)]
# The runs that rise late in the soft pool scout: 707 under `τ_ℓ fixed`, and 705 and 707 under `by 40`. (A comment,
# not an attribute docstring: the weave renders a bare string at module level as prose at the top of the report.)
ARMS = [ex.Arm(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in a.items()}) for a in DESIGN["arms"]]
NAMES = [a.name for a in ARMS]
SHOWN = {a.name: a.shown_as for a in ARMS} | {"base": "base"}
BASE = "base"
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
TASK_MEAN_GATE = -0.02  # the task gate of the D2.2 design, on the mean over ops
LOST_BELOW = 0.5  # a finished run has lost the anchor when the full edit removes less than this
UNIT = 6  # tokens per equation: a ? b = y and a separator
ANSWERS = [UNIT * e + 4 for e in range(ex.K)]
FULL = len(ex.rb.DOSE_GAMMAS) - 1
E1 = 0  # the anchored axis, in code
STOP = ex.STOP_EPOCH


def num(v: float, spec: str = ".3f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} ({num(float(v.min()), spec)} to {num(float(v.max()), spec)})"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    import re

    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- At the end: the e₁ edit net of the median over the other 63 axes of the run (the D2.2 rule) ---------------


def axis_scores(r: dict) -> dict[str, Any]:
    """Removal at the full dose and the worst spill over doses for the e₁ edit on one run, net of the median drop
    over every other coordinate axis of the same run, and the clean score per op.
    """
    drop = np.asarray(r["drop"])  # (axes, doses, ops)
    net = drop[E1] - np.median(drop[1:], axis=0)
    span = r["clean"][D] - r["null"][D]
    spill = net[:, OTHER].max(axis=1)
    worst = int(spill.argmax())
    return {
        "removal": float(net[FULL, D] / span),
        "spill": float(spill.max()),
        "op": OPS[OTHER[int(net[worst, OTHER].argmax())]],
        "clean": np.asarray(r["clean"]),
    }


END = {(r["condition"], r["arm"], r["model_seed"]): axis_scores(r) for r in AXES_BODY["runs"]}
_CONTROL = {r["label"]: r for r in fetch_json(ex.rb.AXES_REF)["runs"]}
CONTROL_CLEAN = {s: np.asarray(_CONTROL[f"control-s{s}"]["clean"]) for s in SEEDS}
# The soft pool scout's finished runs of the same conditions, trained on L4 from the same epoch-64 states.
STAGE2 = {(r["condition"], r["model_seed"]): axis_scores(r) for r in fetch_json(ex.sps.AXES_REF)["runs"]}


def end_seed(arm: str, key: str, cond: str = FIXED) -> np.ndarray:
    return np.array([END[(cond, arm, s)][key] for s in SEEDS])


def task(arm: str) -> np.ndarray:
    """The clean score per op less that of the control at the seed, (seeds, ops)."""
    return np.array([END[(FIXED, arm, s)]["clean"] - CONTROL_CLEAN[s] for s in SEEDS])


REMOVAL = {a: end_seed(a, "removal") for a in NAMES}
SPILL = {a: end_seed(a, "spill") for a in NAMES}
TASK = {a: task(a) for a in NAMES}
OVER = {a: int((SPILL[a] > CRITERION).sum()) for a in NAMES}
LOST = {a: int((REMOVAL[a] < LOST_BELOW).sum()) for a in NAMES}
GPU_SHIFT = np.array([END[(FIXED, BASE, s)]["spill"] - STAGE2[(FIXED, s)]["spill"] for s in SEEDS])
# `base` less the soft pool scout's run of the same seed: the same schedule, on another GPU after epoch 64.


# --- Along training: the full edit on each kept checkpoint against the median of eight reference edits ---------


def reference_spill(edited, ref) -> float:
    """The largest drop on another op under the full edit, less the median drop under the reference edits
    (spill-timing): the median reference score less the edit score, at its largest over the other ops.
    """
    net = np.median(np.asarray(ref), axis=0) - np.asarray(edited)
    return float(net[OTHER].max())


_ALONG = fetch_json(ex.ALONG_REF)["runs"]
ALONG_EPOCHS = sorted({r["epoch"] for r in _ALONG})
_ALONG_BY = {(r["condition"], r["arm"], r["model_seed"], r["epoch"]): reference_spill(r["es_edit"], r["es_sham"]) for r in _ALONG}  # fmt: skip


def along(arm: str, cond: str, seed: int) -> np.ndarray:
    return np.array([_ALONG_BY[(cond, arm, seed, e)] for e in ALONG_EPOCHS])


ALONG = {a: np.array([along(a, FIXED, s) for s in SEEDS]) for a in NAMES}  # (seeds, epochs)
SPREAD_96 = np.ptp([_ALONG_BY[(FIXED, a, 707, 96)] for a in NAMES])
# How far the four runs of 707 under `τ_ℓ fixed` have already parted at the first kept checkpoint, where the arms'
# schedules differ by very little: the noise floor of the along-training score on one run.


# --- The schedules and α along training, from the trajectories --------------------------------------------------

TRAJ = {(v["condition"], v["arm"], v["model_seed"]): v["traj"] for v in fetch_json(ex.TRAJ_REF).values()}


def schedule(arm: str, key: str, cond: str = FIXED) -> tuple[np.ndarray, np.ndarray]:
    """A recorded schedule of one run of *arm* (they are the same at every seed): epochs, values."""
    t = TRAJ[(cond, arm, SEEDS[0] if cond == FIXED else 707)]
    return np.asarray(t["epoch"]), np.asarray(t[key])


def alpha_answers(arm: str, cond: str, seed: int, slice_: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """α at the example answers of `difference` contexts at *slice_*, at every record."""
    t = TRAJ[(cond, arm, seed)]
    a = np.asarray(t["alpha_anchored"])  # (records, slices, positions)
    return np.asarray(t["epoch"]), a[:, slice_, ANSWERS].mean(axis=1)


# --- Figures ---------------------------------------------------------------------------------------------------

RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
GHOST = light_dark("#bbb", "#555")
ARM_INK = {
    "base": light_dark("#08306b", "#c6dbef"),
    "anti-late": light_dark("#d95f02", "#fdae6b"),
    "anti-held": light_dark("#a6611a", "#dfc27d"),
    "lt-late": light_dark("#7570b3", "#bcbddc"),
}
ARM_MARKER = {"base": "s", "anti-late": "^", "anti-held": "v", "lt-late": "D"}


def tex(shown: str) -> str:
    """An arm name for a figure label, with its symbols as mathtext."""
    return shown.replace("τ_ℓ", "$\\tau_\\ell$")


def hatch(ax, lo: float, hi: float, pattern: str) -> None:
    """Hatch the band from *lo* to *hi* as the failing side of a line (style-fig)."""
    ax.axhspan(lo, hi, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch=pattern, lw=0, alpha=0.12, zorder=0)  # fmt: skip


def _dots(ax, x: float, v, color, m: str, rng) -> None:
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def along_figure() -> str:
    """Spill along training on the rising runs and on the mean over the eight seeds, with λ_s̄ and τ_ℓ underneath."""
    data: dict[str, Any] = {
        "epochs": list(ALONG_EPOCHS),
        "names": NAMES,
        "titles": [f"{'τ_ℓ fixed' if c == FIXED else 'by 40'}, seed {s}" for c, s in RISERS] + ["τ_ℓ fixed, mean of 8"],
        "spill": [{a: along(a, c, s).tolist() for a in NAMES} for c, s in RISERS]
        + [{a: ALONG[a].mean(axis=0).tolist() for a in NAMES}],
        "criterion": CRITERION,
        "stop": STOP,
        "anti": {
            a: [schedule(a, "anti_weight")[0].tolist(), schedule(a, "anti_weight")[1][:, 3].tolist()] for a in NAMES
        },
        "tau": {a: [schedule(a, "tau")[0].tolist(), schedule(a, "tau")[1][:, 1].tolist()] for a in NAMES},
    }
    alt = """
        Four columns of three panels, epoch 64 to 400 on the x axis. Top row, spill per arm. τ_ℓ fixed, seed 707:
        base rises to about the criterion by epoch 224 and stays there; anti late stays near zero to 192, then climbs
        to just under the criterion by the end; anti held stays below 0.008 throughout; τ_ℓ late wanders between 0
        and 0.019 and ends at 0.014. By 40, seed 705: every arm falls below zero after epoch 128. By 40, seed 707: base
        climbs to 0.014; the three arms stay near zero. Mean of 8: all arms sit below zero, base highest. Middle row,
        the anti weight at slice 3: base and τ_ℓ late anneal from 0.24 to 0.07 by 360; anti late holds until 200 and
        then anneals to the same hold; anti held stays at 0.24 until 360. Bottom row, τ_ℓ: 0.1 throughout except τ_ℓ
        late, which falls to 0.025 by 360.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="late-spill-scout-along",
        alt_text=alt_text,
        caption=f"""
            **Spill along training, with the schedules the arms change.** Top: the spill of each arm at the
            checkpoints kept every 32 epochs from epoch {data["stop"]}, where the arms part, and at the end, against the median
            of eight reference edits; the first three panels are the runs that rise late in the soft pool scout, the
            last the mean over the eight seeds of `τ_ℓ 0.1, τ_ℓ fixed`. The dashed line is the criterion. Middle: the
            anti weight λ_s̄ at slice 3. Bottom: the slice temperature τ_ℓ, on a log scale.
        """,
    )
    def _plot() -> plt.Figure:
        names, ep = data["names"], np.asarray(data["epochs"])
        n = len(data["titles"])
        fig, axes = plt.subplots(3, n, figsize=(6.6, 4.2), layout="constrained", sharex=True,
                                 height_ratios=(3, 1, 1), sharey="row")  # fmt: skip
        for j in range(n):
            ax = axes[0, j]
            for a in names:
                ax.plot(ep, data["spill"][j][a], color=ARM_INK[a], marker=ARM_MARKER[a], ms=2.5, lw=1.0, label=tex(SHOWN[a]))  # fmt: skip
            ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
            ax.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=1)
            ax.set_title(tex(data["titles"][j]), fontsize=7)
            for a in names:
                axes[1, j].plot(*data["anti"][a], color=ARM_INK[a], lw=1.0)
                axes[2, j].plot(*data["tau"][a], color=ARM_INK[a], lw=1.0)
            axes[2, j].set_yscale("log")
            axes[2, j].set_yticks([0.025, 0.05, 0.1], ["0.025", "0.05", "0.1"])
            axes[2, j].minorticks_off()
            axes[2, j].set_xlabel("epoch", fontsize=7)
            for ax in axes[:, j]:
                ax.axvline(data["stop"], color=RULE, lw=0.6, ls=":", zorder=0)
                ax.tick_params(labelsize=6)
        axes[0, 0].set_ylabel("spill ↓", fontsize=7)
        axes[1, 0].set_ylabel("$\\lambda_{\\bar s}$", fontsize=7)
        axes[2, 0].set_ylabel("$\\tau_\\ell$", fontsize=7)
        axes[2, 0].set_xlim(data["stop"] - 8, ep[-1] + 8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def end_figure() -> str:
    data = {
        "names": NAMES,
        "removal": {a: REMOVAL[a].tolist() for a in NAMES},
        "spill": {a: SPILL[a].tolist() for a in NAMES},
        "task": {a: TASK[a].mean(axis=1).tolist() for a in NAMES},
        "by40": {key: {a: [END[(BY40, a, s)][key] for s in (705, 707)] for a in NAMES} for key in ("removal", "spill")},
        "criterion": CRITERION,
        "task_gate": TASK_MEAN_GATE,
    }
    alt = """
        Three panels, one column per arm with seeds as dots. Removal sits near 0.89 for base and τ_ℓ late, a little
        lower for anti late, and lower again for anti held, whose seed 710 is at 0.63. Spill: base has one seed at
        0.022, just over the criterion; anti late tops out at 0.018, anti held at 0.003 and τ_ℓ late at 0.013. Task
        change: every arm between about −0.012 and 0.010, far above the gate at −0.02.
    """
    return end_draw(data, alt)


@memo
def end_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="late-spill-scout-end",
        alt_text=alt_text,
        caption=f"""
            **Removal, spill, and task at epoch {ex.EPOCHS}, by arm.** One column per arm on `τ_ℓ 0.1, τ_ℓ fixed`,
            the {len(SEEDS)} seeds as small dots over a thin bar for their range, and the seed mean as the large mark;
            the hollow marks beside are the same arm on `τ_ℓ 0.1, by 40` at seeds 705 and 707. **Left:** removal at
            the full dose. **Middle:** spill, the largest net drop on another op at any dose; the dashed line is the
            criterion, with the side over it hatched. Both are net of the median over the other 63 axes of the run.
            **Right:** the task change, the mean clean score over ops less that of the control at the seed.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), layout="constrained")
        rng = np.random.default_rng(0)
        for ax, key in zip(axes, ("removal", "spill", "task"), strict=True):
            for i, a in enumerate(names):
                _dots(ax, i, data[key][a], ARM_INK[a], ARM_MARKER[a], rng)
                if key in data["by40"]:
                    v = data["by40"][key][a]
                    ax.plot([i + 0.3] * len(v), v, ARM_MARKER[a], ms=3.5, mfc="none", mec=ARM_INK[a], mew=0.8, zorder=4)  # fmt: skip
            ax.set_xticks(range(len(names)), [tex(SHOWN[a]) for a in names], rotation=40, ha="right", fontsize=6)
            ax.set_xlim(-0.6, len(names) - 0.3)
            ax.tick_params(labelsize=7)
        axes[0].set_title("removal ↑", fontsize=8)
        axes[1].set_title("spill ↓", fontsize=8)
        axes[2].set_title("task change ↑", fontsize=8)
        for ax, y, below in ((axes[1], data["criterion"], False), (axes[2], data["task_gate"], True)):
            ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
            ax.axhline(y, color=RULE, lw=0.7, ls="--", zorder=1)
            lo, hi = ax.get_ylim()
            if below:
                lo = min(lo, y - 0.005)
                hatch(ax, lo, y, "//")
            else:
                hi = max(hi, y + 0.005)
                hatch(ax, y, hi, "//")
            ax.set_ylim(lo, hi)
        return fig

    return _plot()


def alpha_figure() -> str:
    """α at the example answers at slice 3 along training, seed mean per arm, with the rising runs."""
    ep = alpha_answers(BASE, FIXED, SEEDS[0])[0]
    data = {
        "names": NAMES,
        "epochs": ep.tolist(),
        "mean": {a: np.mean([alpha_answers(a, FIXED, s)[1] for s in SEEDS], axis=0).tolist() for a in NAMES},
        "seeds": {a: [alpha_answers(a, FIXED, s)[1].tolist() for s in SEEDS] for a in NAMES},
        "stop": STOP,
    }
    alt = """
        α at the example answers at slice 3 from epoch 64 to 400. Base and τ_ℓ late rise from about 0.57 to 0.63;
        anti late lags until about epoch 250, then catches up to about 0.62; anti held levels off near 0.59, with one
        seed lower still.
    """
    return alpha_draw(data, alt)


@memo
def alpha_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="late-spill-scout-alpha",
        alt_text=alt_text,
        caption="""
            **The alignment at the example answers along training.** The mean alignment α with e₁ at the example
            answers of `difference` contexts, at slice 3, recorded every four epochs from epoch 64, where the arms part: the seed mean of each arm on `τ_ℓ 0.1, τ_ℓ fixed`
            as a line, the eight seeds as hairlines (a few dip below the lower edge).
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.2, 2.4), layout="constrained")
        ep = np.asarray(data["epochs"])
        for a in data["names"]:
            for v in data["seeds"][a]:
                ax.plot(ep, v, color=ARM_INK[a], lw=0.3, alpha=0.35, zorder=1)
            ax.plot(ep, data["mean"][a], color=ARM_INK[a], lw=1.3, label=tex(SHOWN[a]), zorder=2)
        ax.axvline(data["stop"], color=RULE, lw=0.6, ls=":", zorder=0)
        ax.set_xlim(data["stop"] - 8, ep[-1] + 8)
        ax.set_ylim(0.45, 0.7)
        ax.set_xlabel("epoch", fontsize=7)
        ax.set_ylabel("α at the example answers", fontsize=7)
        ax.tick_params(labelsize=7)
        fig.legend(loc="outside upper center", ncols=len(data["names"]), frameon=False, fontsize=7)
        return fig

    return _plot()


def end_table() -> str:
    rows = []
    for a in NAMES:
        t = TASK[a].mean(axis=1)
        if a == BASE:
            pr = ps = "—"
        else:
            g, h = REMOVAL[a] - REMOVAL[BASE], SPILL[a] - SPILL[BASE]
            pr, ps = fmt(g), fmt(h, ".3f")
        rows.append([f"`{SHOWN[a]}`", fmt(REMOVAL[a]), pr, fmt(SPILL[a], ".3f"), ps, f"{OVER[a]} of {len(SEEDS)}", fmt(t, ".3f")])  # fmt: skip
    a = table_html(
        ["arm", "removal ↑", "less base", "spill ↓", "less base", "over criterion ↓", "task change ↑"],
        rows,
        f"""
        **Removal, spill and task at epoch {ex.EPOCHS} on `τ_ℓ 0.1, τ_ℓ fixed`**, seed mean and range over
        {len(SEEDS)} runs, net of the median drop over the other 63 coordinate axes of the run. "Less base" is the arm
        less `base` at the same seed. The task change is the mean over ops of the clean score less that of the control
        at the seed.
        """,
    )
    risers = []
    for c, s_ in RISERS:
        name = f"`{'τ_ℓ fixed' if c == FIXED else 'by 40'}`, {s_}"
        stage2 = STAGE2[(c, s_)]["spill"]
        risers.append([name, num(stage2)] + [num(END[(c, x, s_)]["spill"]) for x in NAMES])
    b = table_html(
        ["run", "stage 2 (L4)", *[f"`{SHOWN[x]}`" for x in NAMES]],
        risers,
        """
        **Spill at the end on the runs that rose late in the soft pool scout.** The first column is the finished
        run of the soft pool scout, trained on L4 from the same epoch-64 state; the others ran on A10G.
        """,
    )
    return a + "\n\n" + b


R_FIX = {a: END[(FIXED, a, 707)]["spill"] for a in NAMES}
R_705 = {a: END[(BY40, a, 705)]["spill"] for a in NAMES}
R_707 = {a: END[(BY40, a, 707)]["spill"] for a in NAMES}
HELD_710 = END[(FIXED, "anti-held", 710)]["removal"]
BASE_710 = END[(FIXED, BASE, 710)]["removal"]
HELD_REST = np.delete(REMOVAL["anti-held"] - REMOVAL[BASE], SEEDS.index(710))
_ANTI = schedule("anti-held", "anti_weight")
HELD_W3 = float(np.asarray(_ANTI[1])[np.searchsorted(_ANTI[0], 300), 3])
BASE_W3 = float(np.asarray(schedule(BASE, "anti_weight")[1])[np.searchsorted(_ANTI[0], 300), 3])
ALPHA_END = {a: np.array([alpha_answers(a, FIXED, s_)[1][-1] for s_ in SEEDS]) for a in NAMES}

r"""
# Late spill scout

/// tip |
<!-- lede -->
The late spill on the candidate recipe of the soft pool scout follows the anti anneal. Holding the anti weight high stops it, and pausing the anneal delays it until the anneal resumes. Sharpening the slice temperature during the anneal also shrinks it, at no cost we could see. Holding the anti weight lowers removal.
///

The [soft pool scout](/docs/m2/soft-pool-scout/report.py) left a candidate recipe with no early spill: the pool over slices 1 to 3, with the slice temperature τ_ℓ held at 0.1.[^tau] But on one of its eight seeds (707) the spill rises after epoch 128 and ends at the criterion, and with more seeds some runs would likely go over it.

The rise comes while the anti weight λ_s̄ anneals from its peak down to the per-slice holds.[^anti] By then this is the only loss term still moving (the cosine decay of the learning rate runs alongside it, which is a reason to move the anti anneal on its own).

Spill-timing found that the late kind of spill falls on ops the run is still learning. So one reading is that the anti term keeps e₁ closed while an op is learned, and the anneal opens it.

This report asks whether a late change to the schedules keeps the late spill down, and what it costs.

Each arm changes one schedule, and only after epoch 64, so each run carries on from the stored epoch-64 state of the same seed and trains for the last 336 of 400 epochs. That makes four sets of eight runs, one of them the condition unchanged. Only one seed rises under this condition, so the same four also carry on the other finished condition of the soft pool scout, `τ_ℓ 0.1, by 40`, at the two seeds that rose under it.

[^tau]: Slice temperature: how sharply the anchor pull picks among the pooled slices. A lower τ_ℓ puts more of the pull on the one slice where the evidence peaks.

[^anti]: Anti weight: the strength of the anti term, which pushes states off e₁. It anneals (ramps down on a schedule) from a peak to a hold set for each slice.
"""

# %%

rf"""
## Observations

- [Spill along training (E1)](#spill-along-training-e1): on the runs that rise, the rise follows the anti anneal. Held high, the anti weight keeps the spill near zero; paused until epoch 200, the anneal delays the rise until it resumes. Sharpening τ_ℓ lowers it partway.
- [At the end of training (E2)](#at-the-end-of-training-e2): every arm ends with no run over the criterion, against one for the condition unchanged. Sharpening τ_ℓ keeps removal and the task where they were.
- [Alignment and removal (E3)](#alignment-and-removal-e3): holding the anti weight keeps the alignment at the example answers lower. It also lowers removal, a little on most seeds and by about a quarter on one.

## Scope and method

This is an exploratory study, with no preregistration and no gate. It covers one condition of the soft pool scout, `τ_ℓ 0.1, τ_ℓ fixed`, at its eight seeds (704 to 711), and four ways of carrying it on from epoch 64. The same four carry on `τ_ℓ 0.1, by 40` at 705 and 707, the seeds that rose late under it:

`base`
:   The condition unchanged, with a checkpoint every 32 epochs as the arms have.

`anti late`
:   The anti anneal pauses at epoch 64, holding the weight it has reached, and finishes from epoch 200 to 360. That weight is close to its peak (by epoch 64 the anneal has gone about 4% of the way to the holds). So the anti weight stays high while the late ops are learned, and reaches the same per-slice holds by the same epoch. Under `base` the anneal is a quarter of the way to the holds at epoch 128, and two-fifths of the way at 160, where spill-timing saw the late kind begin. So the two arms differ modestly there, and widely from epoch 200 on.

`anti held`
:   The pause runs to epoch 360. So the anti weight never anneals to the holds, and only comes down with the anchor in the last 40 epochs. At slice 3 it stays near {HELD_W3:.2f} through epoch 300, against {BASE_W3:.2f} under `base`. This is the largest change to the anti term that keeps its first 64 epochs, so it shows the most the anti term can do about the late spill, and what that costs. We expected a cost. An early experiment, [ex-2.1.11](/docs/m2/ex-2.1.11/report.py), found that the shape of the anti schedule mattered (a constant weight, high or low, cost grading), and [late-pull](/docs/m2/late-pull/report.py) found that a higher hold cost removal.

`τ_ℓ late`
:   The slice temperature sharpens again, from 0.1 to 0.025 over epochs 64 to 360, alongside the anti anneal. Stage 1 of the soft pool scout found that a sharper τ_ℓ lowers the alignment at slice 1 and the early spill. This arm asks whether sharpening further does the same for the late kind. At 0.025 the pool over three slices is close to a max, so the pull falls almost wholly on the slice where the evidence peaks (slice 3 on every seed).

Every arm uses only schedules and temperatures, nothing keyed to a token role, so each would transfer to real text as the recipe does.

Every run here trained on one GPU type, and every run of a seed shares its first 64 epochs. So a difference between an arm and `base` at the same seed comes from the change after epoch 64 alone. Two arms with the same schedule up to an epoch score the same to the last digit up to it (`anti late` and `anti held` to epoch 192).

But the late rise is small. On the runs it reaches, it ends near the criterion, while quiet runs scatter by about 0.006 around a value a little below zero (one standard deviation, over the stored runs of quiet conditions in spill-timing).

The score along training nets against eight reference axes rather than 63, and is at least as noisy. At epoch 96, where the schedules of the arms have barely parted, the four runs of 707 under `τ_ℓ fixed` already spread over {SPREAD_96:.3f}.

So the comparison that counts is the paired one on the runs that rise, judged over the whole of training rather than at one checkpoint: do the arms move them the same way?
<!-- REVIEW: added the noise floor of the along-training score (the spread of the four arms at epoch 96 on 707, where the anti weight differs by 0.014 and τ_ℓ by 0.002 between arms), since E1 rests on that score. Verify: `along(a, FIXED, 707)[0]` per arm. -->

Modal had no L4 GPUs free when this ran, so every run carried on from epoch 64 on an A10G, where the soft pool scout ran on L4. `base` therefore differs from the finished runs of the soft pool scout only in the numerics of the GPU, and that difference is informative in itself.

On the eight seeds of `τ_ℓ 0.1, τ_ℓ fixed`, `base` ends within {np.abs(GPU_SHIFT).max():.3f} of the L4 run, and the two runs at 707 rise late on both GPUs. But `τ_ℓ 0.1, by 40` at 705, which ended at the criterion on L4, stays below zero on the A10G (second table of E2). So whether a seed rises late is partly a matter of numerics, and whether a single run rises says little by itself.
<!-- REVIEW: added how large the GPU numerics are (GPU_SHIFT: `base` ends within 0.003 of the L4 run on every seed of `τ_ℓ fixed`) and that the two anti arms score the same to the last digit until their schedules part, so the paragraph gives the size of the numerics as well as their existence. The L4 statement follows the soft pool scout's roles through soft-pool, split-anti-seeds, slice-terms and late-pull to `gpu="L4"`; the A10G is `GPU` in experiment.py here, for training and measurement alike, so the stage-2 column of the second table of E2 is the one scored on another GPU. Verify: GPU_SHIFT, and along("anti-late", FIXED, s) against along("anti-held", FIXED, s) at epochs 96 to 192. -->
<!-- REVIEW: was "the seeds that rise (705 and 707)". Under `τ_ℓ 0.1, τ_ℓ fixed` only 707 rises (0.020, on `mix`); 705 rises under `τ_ℓ 0.1, by 40`, the other finished condition, and sits at −0.005 here. Verify: the by-seed table of E4 and the along-training figure of E5 in the soft pool scout. -->
<!-- REVIEW: the noise statement quotes spill-timing (Scope and method: 0.006 over 27 quiet runs) in place of "about a third of the criterion either side of zero". -->

The measurements are those of stage 2 of the soft pool scout, netted by the D2.2 rule:

Spill
:   The largest drop on another op under the full edit at epoch 400, at any dose, net of the median of the same edit along every other coordinate axis of the run. The criterion is 0.02.

Spill along training
:   The full edit on each checkpoint kept from epoch 96 on (every 32 epochs, and the end), net of the median of eight reference axes, as spill-timing scored it. The draw of axes is the same for every run.

Removal
:   How much of the anchored op the full edit removes at epoch 400, net as for spill. A run that removes less than 0.5 has lost the anchor.

Task change
:   The clean score of each op less that of the control at the same seed, averaged over ops.

Alignment at the answers
:   The mean alignment α with e₁ at the example answers of `difference` contexts, at slice 3, along training.
"""

# %%

rf"""
## Spill along training (E1)

Does any arm keep the late rise down on the runs where it appears, and does the rise follow the anti weight in time? The figure has the three runs that rose late in the soft pool scout, and the mean over the eight seeds of `τ_ℓ 0.1, τ_ℓ fixed`. The anti weight at slice 3 and τ_ℓ are plotted underneath.

{along_figure()}

On both runs that rise here, the late spill follows the anti weight. Where the weight is held high (`anti held`), the spill stays near zero to the end. The third run, 705 under `by 40`, does not rise on this GPU in any arm, `base` included, so it says nothing about the arms.
<!-- REVIEW: was "on all three rising runs ... on all three". Only two of the three rise on this GPU (the paragraph says so of the third two sentences on), so the count is two. Verify: the by-40 705 panel, where `base` is below zero from epoch 160. -->

Where the anneal pauses until epoch 200 (`anti late`), the spill of 707 under `τ_ℓ fixed` stays flat until the anneal resumes. Then it climbs as the weight comes down, and reaches the level of `base` by the end. On 707 under `by 40` the same arm keeps it flat throughout.

Sharpening τ_ℓ during the anneal lowers the spill on both runs that rise: partway on 707 under `τ_ℓ fixed`, and to zero on 707 under `by 40`. Its spill along training is bumpier than that of the other arms, which may be the pull moving between slices as the pool sharpens.

The mean over the eight seeds moves little, since seven of them are quiet under every arm. All three arms end a little below `base` on it.

So the timing fits the reading from the introduction, that the anneal opens e₁ to the other ops. The `anti late` run of 707 is the clearest evidence: the same model, from the same state, rises only once the weight comes down. One run is not enough to be sure, but nothing here points the other way.
"""

# %%

rf"""
## At the end of training (E2)

How does each arm compare with `base` at epoch 400, on spill, on removal and on the task?

{end_figure()}

{end_table()}

No arm has a run over the criterion, against one for `base` (707, at {num(R_FIX[BASE])}). On that run the spill falls to {num(R_FIX["anti-held"])} with the anti weight held, and to {num(R_FIX["lt-late"])} with τ_ℓ sharpened. `anti late` ends at {num(R_FIX["anti-late"])}, just under the line, as E1 showed. On 707 under `by 40`, all three arms end below zero, against {num(R_707[BASE])} for `base`.

`τ_ℓ late` keeps removal where `base` has it, on the mean and seed by seed. The two anti arms lower removal: a little on most seeds, and on one seed by much more under `anti held` (E3). None of the arms changes the task by more than a few thousandths.
"""

# %%

rf"""
## Alignment and removal (E3)

Holding the anti weight keeps the late spill down, but the anti term pushes on e₁ everywhere, the anchored op included. How much does holding it cost?

{alpha_figure()}

Under `base` the alignment at the example answers keeps growing after epoch 64, as the anti weight comes down. Under both anti arms it grows more slowly while the weight is held. `anti held` levels off below `base` (a seed mean of {ALPHA_END["anti-held"].mean():.2f} at epoch 400, against {ALPHA_END[BASE].mean():.2f}), and `anti late` catches up most of the way once its anneal resumes. `τ_ℓ late` stays with `base` throughout.

Removal follows. Under `anti held` it is lower than under `base` on every seed: by a few hundredths on seven of them, a change of {fmt(HELD_REST)} as mean and range, and by about a quarter on 710 ({BASE_710:.2f} to {HELD_710:.2f}), the seed with the lowest alignment.

The run at 710 keeps the anchor by our rule, but at that size it is closer to a lost run than any of the others. `anti late` lowers removal less, and on one seed (710 again) by a few hundredths.

So holding the anti weight high shows what the anti term can do about the late spill, but it is not a recipe. It trades the late spill for removal, and on one seed in eight that cost is large, as we expected.
"""

# %%

r"""
## Discussion

The late spill on the candidate recipe follows the anti anneal, which fits the reading in the [late-spill backlog item](/todo/science/late-spill-on-ops-still-being-learned.md): the anti term keeps e₁ closed to the other ops, and the anneal opens it.

The other reading is that whatever draws an op to e₁ also slows its learning. It predicted no change, and here the schedule alone moves the rise. But the two readings need not exclude each other, since we did not measure when the spilled op (`mix`) is learned on these runs. The evidence rests on two rising runs, so it is a strong hint, short of a settled result.
<!-- REVIEW: softened from "It does not fit the other reading": the readings are not exclusive, and this report does not measure when `mix` is learned (claims review, round 1). -->

Of the three changes, sharpening τ_ℓ during the anneal is the one that costs nothing we measured. It lowers the late spill on both rising runs, and keeps removal and the task where they were. It adds no term or weight to the recipe, only a second step in a temperature the recipe already has.

Why it helps is less clear than for the anti arms. A sharper slice pool puts the pull almost wholly on slice 3, which may leave slices 1 and 2 freer for the anti term to clear, but nothing here measures that.

The anti arms show the trade the anti term has always had. Held high, it keeps the late spill off at a cost to removal that is small on most seeds and large on one. Paused, it delays the spill without preventing it under `τ_ℓ fixed`, so the pause alone does not settle anything at the end of training.

The GPU difference is a reminder of how small this effect is. A run that reaches the criterion late on one GPU can stay below zero on another, from the same state with the same schedule. So a count of runs over the criterion at a handful of seeds cannot tell these arms apart. If one of them carries on to more seeds, the paired spill along training is the measurement to watch.
"""
