# ruff: noqa: B018
# title: Anneal confirmation

# An exploratory report: the soft pool scout's candidate recipe with τ_ℓ annealing from 0.1 to 0.025 over the whole
# schedule, at fresh seeds 712 to 719 and in a loss-rate count at forty seeds. `experiment.py` beside this script
# trains and scores the runs.

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


AXES_BODY = fetch_json(ex.AXES_REF)
DESIGN = AXES_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
COUNT_SEEDS: list[int] = DESIGN["count_seeds"]
OLD_SEEDS = list(range(704, 712))
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
TASK_MEAN_GATE, TASK_OP_GATE = -0.02, -0.05  # the task gate of the D2.2 design
LOST_BELOW = 0.5  # a finished run has lost the anchor when the full edit removes less than this
FULL = len(ex.rb.DOSE_GAMMAS) - 1
E1 = 0  # the anchored axis, in code
STOP = ex.STOP_EPOCH
NEW, FIXED, FALLOFF, STEP = "anneal", "fixed", "falloff", "step-soft"
RECIPES = [FALLOFF, STEP, FIXED, NEW]
SHOWN = {NEW: "τ_ℓ anneal", FIXED: "τ_ℓ fixed", FALLOFF: "falloff", STEP: "step, soft τ"}


def num(v: float, spec: str = ".3f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} ({num(float(v.min()), spec)} to {num(float(v.max()), spec)})"


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


# --- At the end: the e₁ edit net of the median over the other 63 axes of the run (the D2.2 rule) ---------------


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


def axis_scores(r: dict, control: dict) -> dict[str, Any]:
    """Removal at the full dose, the worst spill over doses and the op it falls on, and the task change per op
    against *control*, for the e₁ edit on one run.
    """
    net = net_of(r["drop"])  # (doses, ops)
    spill = net[:, OTHER].max(axis=1)
    worst = int(spill.argmax())
    return {
        "removal": float(net[FULL, D] / (r["clean"][D] - r["null"][D])),
        "spill": float(spill.max()),
        "op": OPS[OTHER[int(net[worst, OTHER].argmax())]],
        "task": np.asarray(r["clean"]) - np.asarray(control["clean"]),
    }


_NEW_AX = {r["label"]: r for r in AXES_BODY["runs"]}
_RB = {r["label"]: r for r in fetch_json(ex.rb.AXES_REF)["runs"]}
_SPS = {(r["condition"], r["model_seed"]): r for r in fetch_json(ex.sps.AXES_REF)["runs"]}
STORED_AX = {
    FALLOFF: lambda s: _RB[f"falloff-s{s}"],
    STEP: lambda s: _RB[ex.rb.SOFT_POOL + f"step-soft-s{s}"],
    FIXED: lambda s: _SPS[(ex.BASE, s)],
}
END = {(NEW, s): axis_scores(_NEW_AX[ex.label_of(s, None)], _NEW_AX[ex.label_of(s, None, "control")]) for s in SEEDS}
END |= {(n, s): axis_scores(get(s), _RB[f"control-s{s}"]) for n, get in STORED_AX.items() for s in OLD_SEEDS}
SEEDS_OF = dict.fromkeys((FALLOFF, STEP, FIXED), OLD_SEEDS) | {NEW: SEEDS}


def by_seed(n: str, key: str) -> np.ndarray:
    return np.array([END[(n, s)][key] for s in SEEDS_OF[n]])


REMOVAL = {n: by_seed(n, "removal") for n in RECIPES}
SPILL = {n: by_seed(n, "spill") for n in RECIPES}
TASK = {n: by_seed(n, "task") for n in RECIPES}  # (seeds, ops)
OVER = {n: int((SPILL[n] > CRITERION).sum()) for n in RECIPES}
LOST = {n: int((REMOVAL[n] < LOST_BELOW).sum()) for n in RECIPES}
TASK_MISS = {n: int(((TASK[n].mean(axis=1) < TASK_MEAN_GATE) | (TASK[n].min(axis=1) < TASK_OP_GATE)).sum()) for n in RECIPES}  # fmt: skip

# The restricted edit at block 1 alone, net of the median along the other 63 axes of the run.
_RESTRICTED = {r["label"]: r for r in fetch_json(ex.RESTRICTED_REF)["runs"]}
_RESTRICTED_OLD = {r["label"]: r for r in fetch_json(ex.sps.RESTRICTED_REF)["runs"]}
_OLD_LABEL = {FALLOFF: "falloff-s{s}", STEP: "step-soft-s{s}", FIXED: ex.BASE + "-s{s}"}


def block1_spill(n: str, s: int) -> float:
    r = _RESTRICTED[ex.label_of(s, None)] if n == NEW else _RESTRICTED_OLD[_OLD_LABEL[n].format(s=s)]
    return float(net_of(r["drop"]["block 1"])[OTHER].max())


BLOCK1 = {n: np.array([block1_spill(n, s) for s in SEEDS_OF[n]]) for n in RECIPES}
BLOCK1_OVER = {n: int((BLOCK1[n] > CRITERION).sum()) for n in RECIPES}

# --- Along training: the full edit on each kept checkpoint, net of the median over the other 63 axes ----------

_ALONG = fetch_json(ex.ALONG_REF)["runs"]
ALONG_EPOCHS = sorted({r["epoch"] for r in _ALONG})
_ALONG_BY = {(r["label"], r["epoch"]): r for r in _ALONG}


def along(s: int) -> np.ndarray:
    """Spill at each kept checkpoint and the end, for the run at seed *s*."""
    return np.array([net_of(_ALONG_BY[(ex.label_of(s, None), e)]["drop"])[OTHER].max() for e in ALONG_EPOCHS])


def along_clean(s: int, op: str) -> np.ndarray:
    return np.array([_ALONG_BY[(ex.label_of(s, None), e)]["clean"][OPS.index(op)] for e in ALONG_EPOCHS])


ALONG = {s: along(s) for s in SEEDS}
RISER = max(SEEDS, key=lambda s: ALONG[s][-1])
RISER_OP = END[(NEW, RISER)]["op"]
EARLY = max(SEEDS, key=lambda s: ALONG[s][:3].max())
EARLY_PEAK = float(ALONG[EARLY][:3].max())
EARLY_EPOCH = ALONG_EPOCHS[int(ALONG[EARLY][:3].argmax())]
BUMP_713 = float(ALONG[RISER][ALONG_EPOCHS.index(96)])
EARLY_NEXT = float(ALONG[EARLY][ALONG_EPOCHS.index(EARLY_EPOCH) + 1])  # the early bump at the checkpoint after its peak
GPU_SHIFT = 0.003  # how far a change of GPU moved the end spill on `τ_ℓ fixed` (the late spill scout, Scope and method)

# The schedules, from the trajectory of one finished run (they are the same at every seed).
_TRAJ = {k: v["traj"] for k, v in fetch_json(ex.FINISHED_TRAJ_REF).items()}
_T0 = _TRAJ[ex.label_of(SEEDS[0], None)]
SCHED_EPOCH = np.asarray(_T0["epoch"])
SCHED_TAU = np.asarray(_T0["tau"])[:, 1]
SCHED_ANTI = np.asarray(_T0["anti_weight"])[:, 3]

# --- The loss-rate count: α at the example answers over epochs 44 to 64 ---------------------------------------

CALLS = {r["model_seed"]: r for r in fetch_json(ex.CALLS_REF)["runs"]}
_LR_CALLS = fetch_json(ex.lr.CALLS_REF)["runs"]
_STORED_TRAJ = {FALLOFF: fetch_json(ex.lr.sas.TRAJ_REF), STEP: fetch_json(ex.lr.sp.TRAJ_REF)}


def window_of(n: str, s: int) -> float:
    if n == NEW:
        return CALLS[s]["window_mean"]
    if s in OLD_SEEDS:
        return ex.lr.window_mean(_STORED_TRAJ[n][f"{n}-s{s}"]["traj"])
    return next(r["window_mean"] for r in _LR_CALLS if r["arm"] == n and r["model_seed"] == s)


WINDOW = {n: np.array([window_of(n, s) for s in COUNT_SEEDS]) for n in (FALLOFF, STEP, NEW)}
NOT_KEPT = sum(c["call"] != "kept" for c in CALLS.values())
LOWEST = min(CALLS, key=lambda s: CALLS[s]["window_mean"])
ABOVE_FALLOFF = int((WINDOW[NEW] > WINDOW[FALLOFF]).sum())
ABOVE_STEP = int((WINDOW[NEW] > WINDOW[STEP]).sum())
LOSS_RATE_BOUND = 1 - 0.05 ** (1 / len(COUNT_SEEDS))  # one-sided 95% upper bound on the rate, at 0 of n


# --- Figures ---------------------------------------------------------------------------------------------------

RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
GHOST = light_dark("#bbb", "#555")
INK = {
    FALLOFF: light_dark("#a6611a", "#dfc27d"),
    STEP: light_dark("#1b7837", "#a6dba0"),
    FIXED: light_dark("#08306b", "#c6dbef"),
    NEW: light_dark("#7570b3", "#bcbddc"),
}
MARKER = {FALLOFF: "v", STEP: "^", FIXED: "s", NEW: "D"}


def tex(shown: str) -> str:
    """A recipe name for a figure label, with its symbols as mathtext."""
    return shown.replace("τ_ℓ", "$\\tau_\\ell$").replace("τ", "$\\tau$")


def hatch(ax, lo: float, hi: float) -> None:
    """Hatch the band from *lo* to *hi* as the failing side of a line (style-fig)."""
    ax.axhspan(lo, hi, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, alpha=0.12, zorder=0)  # fmt: skip


def _dots(ax, x: float, v, color, m: str, rng) -> None:
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def end_figure() -> str:
    data = {
        "names": RECIPES,
        "shown": SHOWN,
        "removal": {n: REMOVAL[n].tolist() for n in RECIPES},
        "spill": {n: SPILL[n].tolist() for n in RECIPES},
        "task": {n: TASK[n].mean(axis=1).tolist() for n in RECIPES},
        "criterion": CRITERION,
        "task_gate": TASK_MEAN_GATE,
    }
    r, s, t = REMOVAL[NEW], SPILL[NEW], TASK[NEW].mean(axis=1)
    alt = f"""
        Three panels, one column per recipe (falloff, step soft τ and τ_ℓ fixed at seeds 704 to 711, τ_ℓ anneal at
        712 to 719), seeds as dots. Removal: τ_ℓ fixed sits tightly near 0.90; falloff and step soft τ spread from
        about 0.76 to 0.91; τ_ℓ anneal spreads from {r.min():.2f} to {r.max():.2f}, mean {r.mean():.2f}. Spill: every
        recipe has most seeds below zero; τ_ℓ anneal tops out at {s.max():.3f}, at the criterion of 0.02, and
        τ_ℓ fixed has one seed at 0.020 too. Task change: τ_ℓ anneal between {t.min():.3f} and {t.max():.3f}, above the
        gate at −0.02; falloff has one seed at −0.032.
    """
    return end_draw(data, alt)


@memo
def end_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="anneal-confirm-end",
        alt_text=alt_text,
        caption="""
            **Removal, spill, and task at epoch 400, by recipe.** One column per recipe: the stored runs of `falloff`,
            `step, soft τ` and `τ_ℓ fixed` at seeds 704 to 711, and `τ_ℓ anneal` at the fresh seeds 712 to 719; the
            seeds as small dots over a thin bar for their range, the seed mean as the large mark. **Left:** removal at
            the full dose. **Middle:** spill, the largest net drop on another op at any dose; the dashed line is the
            criterion, with the side over it hatched. Both are net of the median over the other 63 axes of the run.
            **Right:** the task change, the mean clean score over ops less that of the control at the seed, with the
            gate of the D2.2 design dashed.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), layout="constrained")
        rng = np.random.default_rng(0)
        for ax, key in zip(axes, ("removal", "spill", "task"), strict=True):
            for i, n in enumerate(names):
                _dots(ax, i, data[key][n], INK[n], MARKER[n], rng)
            ax.axvline(len(names) - 1.5, color=GHOST, lw=0.5, ls=":", zorder=0)
            ax.set_xticks(range(len(names)), [tex(data["shown"][n]) for n in names], rotation=40, ha="right", fontsize=6)  # fmt: skip
            ax.set_xlim(-0.6, len(names) - 0.4)
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
                hatch(ax, lo, y)
            else:
                hi = max(hi, y + 0.005)
                hatch(ax, y, hi)
            ax.set_ylim(lo, hi)
        return fig

    return _plot()


def end_table() -> str:
    rows = []
    for n in RECIPES:
        seeds = SEEDS_OF[n]
        t = TASK[n]
        rows.append([
            f"`{SHOWN[n]}`",
            f"{seeds[0]}–{seeds[-1]}",
            fmt(REMOVAL[n]),
            fmt(SPILL[n], ".3f"),
            f"{OVER[n]} of {len(seeds)}",
            f"{BLOCK1_OVER[n]} of {len(seeds)}",
            fmt(t.mean(axis=1), ".3f"),
            num(float(t.min()), ".3f"),
        ])  # fmt: skip
    return table_html(
        [
            "recipe",
            "seeds",
            "removal ↑",
            "spill ↓",
            "over criterion ↓",
            "over, block 1 ↓",
            "task change ↑",
            "worst op ↑",
        ],
        rows,
        """
        **Removal, spill and task at epoch 400**, seed mean and range over eight runs, net of the median drop over the
        other 63 coordinate axes of the run. "Over, block 1" counts the runs over the criterion under the same edit
        restricted to block 1. The task change is the mean over ops of the clean score less that of the control at
        the seed; "worst op" is the lowest such change on any op of any run. The three stored recipes are at seeds
        704 to 711 and `τ_ℓ anneal` at 712 to 719, so the rows compare different seeds.
        """,
    )


def along_figure() -> str:
    data = {
        "epochs": list(ALONG_EPOCHS),
        "seeds": SEEDS,
        "spill": {s: ALONG[s].tolist() for s in SEEDS},
        "riser": RISER,
        "early": EARLY,
        "op": RISER_OP,
        "clean": {s: along_clean(s, RISER_OP).tolist() for s in SEEDS},
        "criterion": CRITERION,
        "sched": [SCHED_EPOCH.tolist(), SCHED_TAU.tolist(), SCHED_ANTI.tolist()],
    }
    end = ALONG[RISER][-1]
    alt = f"""
        Four stacked panels sharing an epoch axis from 16 to 400. Top: spill at each kept checkpoint for the eight
        seeds, as faint lines, with seed {RISER} and seed {EARLY} drawn darker. Most seeds sit between −0.03 and 0.01
        throughout. Seed {EARLY} jumps to {EARLY_PEAK:.2f} at epoch {EARLY_EPOCH}, is at the criterion at the next
        checkpoint, and is back near zero by the one after. Seed {RISER} has a bump to {BUMP_713:.3f} at epoch 96, then
        wanders near zero until about epoch 300, then rises to {end:.3f} at the end, at the criterion. Second: the
        clean score on {RISER_OP} for every seed, rising slowly from about 0.1 to 0.29, with seed {RISER} no different
        from the others. Third: τ_ℓ on a log scale, from 0.1 to 0.025. Bottom: the anti weight at slice 3, from 0.25
        to its hold of 0.07 by epoch 360, then down to near zero with the anchor over the last 40 epochs.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="anneal-confirm-along",
        alt_text=alt_text,
        caption=f"""
            **Spill along training, and the op it falls on.** Top: the spill at the checkpoints kept at epochs 16, 32
            and 64 and then every 32 epochs, and at the end, net of the median over the other 63 axes of the run, for
            each of the eight seeds; seed {data["riser"]} (the highest at the end) and seed {data["early"]} (the
            highest early) are drawn darker, and the dashed line is the criterion. Second: the clean score on
            `{data["op"]}`, the op the late spill falls on, for each seed. Below them, the two schedules that move
            late: the slice temperature τ_ℓ (log scale) and the anti weight λ_s̄ at slice 3.
        """,
    )
    def _plot() -> plt.Figure:
        ep = np.asarray(data["epochs"])
        fig, axes = plt.subplots(4, 1, figsize=(4.6, 5.0), layout="constrained", sharex=True, height_ratios=(2.2, 1.4, 0.8, 0.8))  # fmt: skip
        dark = {data["riser"]: "D", data["early"]: "o"}
        for s in data["seeds"]:
            on = s in dark
            kw = dict(marker=dark[s], ms=2.8, lw=1.1, alpha=1.0, zorder=3, label=f"seed {s}") if on else dict(lw=0.5, alpha=0.6, zorder=2)  # fmt: skip
            color = INK[NEW] if on else GHOST
            axes[0].plot(ep, data["spill"][s], color=color, **kw)
            axes[1].plot(ep, data["clean"][s], color=color, **{k: v for k, v in kw.items() if k != "label"})
        axes[0].axhline(0, color=ZERO, lw=0.5, zorder=0)
        axes[0].axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=1)
        axes[0].set_ylabel("spill ↓", fontsize=7)
        axes[0].legend(fontsize=6, frameon=False, loc="upper right")
        axes[1].set_ylabel(f"clean, {data['op']}", fontsize=7)
        e, tau, anti = (np.asarray(v) for v in data["sched"])
        axes[2].plot(e, tau, color=INK[NEW], lw=1.0)
        axes[2].set_yscale("log")
        axes[2].set_yticks([0.025, 0.05, 0.1], ["0.025", "0.05", "0.1"])
        axes[2].minorticks_off()
        axes[2].set_ylabel("$\\tau_\\ell$", fontsize=7)
        axes[3].plot(e, anti, color=RULE, lw=1.0)
        axes[3].set_ylabel("$\\lambda_{\\bar s}$", fontsize=7)
        axes[3].set_xlabel("epoch", fontsize=7)
        for ax in axes:
            ax.tick_params(labelsize=6)
        return fig

    return _plot()


def count_figure() -> str:
    names = [FALLOFF, STEP, NEW]
    data = {
        "names": names,
        "shown": SHOWN,
        "window": {n: WINDOW[n].tolist() for n in names},
        "lost_max": ex.lr.LOST_MAX,
        "kept_min": ex.lr.KEPT_MIN,
    }
    w = WINDOW[NEW]
    alt = f"""
        Window means of α at the example answers for three recipes at the same forty seeds, as dots in three columns
        with grey lines joining each seed across them. Dashed lines at 0.15 (lost below) and 0.28 (kept above).
        Falloff spreads from about 0 to 0.6, with one seed below 0.15 and a few near 0.2; step soft τ from about 0.33
        to 0.6; τ_ℓ anneal from {w.min():.2f} to {w.max():.2f}, every seed above 0.28.
    """
    return count_draw(data, alt)


@memo
def count_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="anneal-confirm-count",
        alt_text=alt_text,
        caption=f"""
            **The early call at forty seeds.** Mean α at the example answers at slice 3 over epochs 44 to 64, for
            each run, with a grey line joining the runs of a seed. `falloff` and `step, soft τ` are the calls of the
            loss-rate screen (and at seeds 704 to 711, the same window mean on their stored full runs). Below {data["lost_max"]} a run is called
            lost and from {data["kept_min"]} kept, with the band between finished and measured.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, ax = plt.subplots(figsize=(3.6, 2.6), layout="constrained")
        rng = np.random.default_rng(1)
        jit = {n: rng.uniform(-0.1, 0.1, len(data["window"][n])) for n in names}
        v = np.array([data["window"][n] for n in names])
        for j in range(v.shape[1]):
            ax.plot([i + jit[n][j] for i, n in enumerate(names)], v[:, j], color=GHOST, lw=0.3, zorder=1)
        for i, n in enumerate(names):
            ax.plot(i + jit[n], data["window"][n], MARKER[n], ms=3, color=INK[n], alpha=0.8, mew=0, zorder=3)
        for y in (data["lost_max"], data["kept_min"]):
            ax.axhline(y, color=RULE, lw=0.7, ls="--", zorder=0)
        hatch(ax, 0.0, data["lost_max"])
        ax.set_ylim(0.0, max(v.max() + 0.03, 0.65))
        ax.set_xticks(range(len(names)), [tex(data["shown"][n]) for n in names], fontsize=7)
        ax.set_ylabel("α at the example answers,\nepochs 44 to 64", fontsize=7)
        ax.tick_params(labelsize=7)
        return fig

    return _plot()


R_MIN = float(REMOVAL[NEW].min())
R_MIN_SEED = SEEDS[int(REMOVAL[NEW].argmin())]
LOW2 = sorted(SEEDS, key=lambda s: END[(NEW, s)]["removal"])[:2]
REST = np.array([END[(NEW, s)]["removal"] for s in SEEDS if s not in LOW2])
LOW_SEEDS = [s for s in SEEDS if END[(NEW, s)]["removal"] < float(REMOVAL[FALLOFF].min())]
WORST_TASK = float(TASK[NEW].min())
WORST_TASK_SEED = SEEDS[int(TASK[NEW].min(axis=1).argmin())]
WORST_TASK_OP = OPS[int(TASK[NEW][SEEDS.index(WORST_TASK_SEED)].argmin())]
RISER_END = float(ALONG[RISER][-1])
RISER_B1 = float(BLOCK1[NEW][SEEDS.index(RISER)])

r"""
# Anneal confirmation

/// tip |
<!-- lede -->
The pool over slices 1 to 3, with its slice temperature annealing slowly from 0.1 to 0.025, keeps the anchor on every run at eight fresh seeds. An early call at epoch 64 finds the anchor kept on every run of a forty-seed count too. Spill and the task stay within the criteria of the D2.2 design (one run ends at the spill line), and removal is about that of `falloff`.
///

The [soft pool scout](/docs/m2/soft-pool-scout/report.py) found a candidate for the recipe of record: the pull pooled over slices 1 to 3, with the slice temperature $\tau_\ell$ held at 0.1.[^tau] It keeps the anchor and stays within the spill criterion at seeds 704 to 711. But on one or two seeds, spill rises late and ends near the criterion. The [late spill scout](/docs/m2/late-spill-scout/report.py) then found that sharpening $\tau_\ell$ further late in training, to 0.025, lowered that late rise at no measured cost.

Here the two moves become one. $\tau_\ell$ falls from 0.1 at the start to 0.025 at epoch 400, smoothly on a log scale. So it is still close to 0.1 while the anchor settles, and reaches the halfway point (0.05) at epoch 200. We call this recipe `τ_ℓ anneal`.

[^tau]: Slice temperature: how sharply the pooled pull picks among slices. The pull is a softmin-weighted mix over slice and position pairs, and $\tau_\ell$ sets how peaked the weights over slices are, as the pooling temperature $\tau_p$ does over positions.

This is the confirmation run for adopting `τ_ℓ anneal` as the recipe of record. It trains the recipe to the end at eight fresh seeds (712 to 719), with a control at each seed. It also trains it to epoch 64 at forty seeds (704 to 743), to count lost anchors.
"""

# %%

r"""
## Observations

- [The anchor at fresh seeds (E1)](#the-anchor-at-fresh-seeds-e1): every run keeps the anchor and passes the task gate. Removal varies more across seeds than under `τ_ℓ fixed`, and about as much as under `falloff`. One seed is a little below the lowest `falloff` run.
- [Spill at the end and along training (E2)](#spill-at-the-end-and-along-training-e2): no run ends over the criterion. One seed rises late to the line, and the same seed goes over under the edit at block 1 alone. Two seeds have brief bumps over the line in the first hundred epochs, which clear within 64 epochs. The soft pool scout saw none under `τ_ℓ fixed` at the same checkpoints.
- [Lost anchors at forty seeds (E3)](#lost-anchors-at-forty-seeds-e3): the call finds all forty runs kept, each well clear of the band where the call is unsure.

## Scope and method

This is an exploratory study, with no preregistration and no gate. We set four checks before the runs. They guide the choice of recipe without deciding it, so a recipe that misses one narrowly can still be the better choice:

1. Every run at seeds 712 to 719 keeps the anchor (removal at least 0.5).
2. No run at those seeds ends over the spill criterion of 0.02.
3. Removal is no lower than that of `falloff` at seeds 704 to 711. The task passes the gate of the D2.2 design against the control at the same seed: a mean change over ops of at least −0.02, and at least −0.05 on any one op.
4. At most one run of the forty loses the anchor. That is about as few as `step, soft τ` (none of forty) and `falloff` (one) lost in the loss-rate screen, and forty seeds can't tell those two counts apart.

`τ_ℓ anneal` is `τ_ℓ fixed` from the soft pool scout (in full, `τ_ℓ 0.1, τ_ℓ fixed`) with one change: $\tau_\ell$ moves from 0.1 to 0.025 over epochs 0 to 400, on a minimum-jerk curve in log $\tau_\ell$.[^jerk] The position temperature $\tau_p$ still sharpens from 1 to 0.1 by epoch 40, the anchor weight still warms up over the first 40 epochs, and the anti weight still anneals to the per-slice holds of `falloff` by epoch 360.

[^jerk]: Minimum-jerk: a smooth S-shaped path that starts and ends at rest, so the change is slow at both ends and fastest in the middle.

The forty runs of the count use the same seeds as the `falloff` and `step, soft τ` runs of the loss-rate screen. So each run pairs with a run of each of those recipes by initialization, batches, and label draws. Each run stops at epoch 64 for its call, made as in that screen.

The runs at 712 to 719 continue from those stops to epoch 400, keeping a checkpoint every 32 epochs. Each control trains for the full schedule at the same seed, with no pull and no anti term. All runs trained and were measured on L4 GPUs.

The stored recipes ran at seeds 704 to 711, and `τ_ℓ anneal` at 712 to 719. So a difference between them includes the variation between two draws of eight seeds. Under `falloff` and `step, soft τ`, removal varies by about 0.15 across seeds, with one or two low seeds in eight. A difference in the seed mean smaller than a few hundredths is within that variation.

## The measurements

Every edit below is the e₁ edit. Each score is the drop under that edit, less the median drop under the same edit along each of the other 63 coordinate axes of the run (the rule of the D2.2 design).

Removal
:   The drop on `difference` under the full edit at every position and slice, as a share of the way from the clean score to the target null. Higher is better, and a run below 0.5 has lost the anchor.

Spill
:   The largest drop on any other op, at the dose where it is largest. Lower is better, and a run above 0.02 is over the criterion.

Spill along training
:   Spill at the full dose, measured on each checkpoint a run kept (epochs 16, 32, and 64, then every 32 epochs) and at the end.

Task change
:   The clean score per op, less that of the control at the seed.

Call
:   The early verdict on whether a run kept the anchor, from mean α at the example answers at slice 3 over epochs 44 to 64. A run below 0.15 is called lost, and one from 0.28 kept. A run in between is finished and measured.
"""

# %%

rf"""
## The anchor at fresh seeds (E1)

Does every run at the fresh seeds keep the anchor, and are removal and the task as good as under `falloff`?

{end_figure()}

{end_table()}

Every run keeps the anchor, and every run passes the task gate. The task stays within about a hundredth of the control on the mean over ops. The worst single op is `{WORST_TASK_OP}` at seed {WORST_TASK_SEED} ({num(WORST_TASK)}), against a gate of −0.05. `falloff` has one run that misses the gate on one op.

Removal is close to that of `falloff` on the seed mean, and below that of `τ_ℓ fixed`. Most of the difference comes from two seeds, {LOW2[0]} and {LOW2[1]}, which remove about as little as the lowest runs of the step recipes (`falloff` and `step, soft τ`). The lower of the two, seed {R_MIN_SEED} at {R_MIN:.2f}, is the only run below the lowest `falloff` run. It is still well clear of the line where a run counts as lost. The other six spread from {REST.min():.2f} to {REST.max():.2f}.

So removal under `τ_ℓ anneal` varies across seeds the way it does under the step recipes, while under `τ_ℓ fixed` it was unusually even at its eight seeds. We can't tell from this whether the anneal or the seeds make the difference, since the two recipes never ran at the same seeds.

The late spill scout found that sharpening $\tau_\ell$ late kept removal where it was, seed by seed, which points to the seeds. But there the sharpening ran over epochs 64 to 360, carried on from stored epoch-64 states, where here it runs over the whole schedule.
"""

# %%

rf"""
## Spill at the end and along training (E2)

Does any run end over the spill criterion, and does the late rise of the soft pool scout show up along training? The figure shows the spill of each seed along training. Beneath it are the clean score of the op the spill falls on, and the two schedules that move late.

{along_figure()}

No run ends over the criterion, and one ends at it. Seed {RISER}, after an early bump, stays near zero until about epoch 300, then rises to {num(RISER_END)} at the end. That is {num(CRITERION - RISER_END, ".4f")} under the line, less than a change of GPU moved this score in the late spill scout (up to {GPU_SHIFT}). So we count it as at the line rather than under it. The other seven end below or near zero.

This is the late kind of spill that the soft pool scout saw on one or two seeds in eight, at about the same size and on the same op (`{RISER_OP}`, as on seed 707 under `τ_ℓ fixed`). So the slow anneal does not remove it.

Every seed is still learning `{RISER_OP}` over the whole run, which fits the earlier finding, quoted in the late spill scout, that the late kind falls on ops a run is still learning. But seed {RISER} learns it no later than the others, so when `{RISER_OP}` is learned does not say why this seed rises and the others don't.
<!-- REVIEW: was "just under the line" and "the late rise is not tied to an op still being learned". The end value is 0.0198 against a criterion of 0.02, within the GPU shift the late spill scout measured, so the pass is at the line; and `mix` is still being learned on every seed through epoch 400 (second panel), so the result fits the ops-still-learned reading and only says that the timing of learning `mix` does not single out 713. Verify: RISER_END, the clean-mix panel. -->

Early on, two seeds have brief bumps over the line. Seed {EARLY} reaches {EARLY_PEAK:.2f} at epoch {EARLY_EPOCH}, while the position pool sharpens. It is still at the line ({num(EARLY_NEXT)}) at the next checkpoint, and near zero by the one after. Seed {RISER} reaches {BUMP_713:.3f} at epoch 96, and is near zero again by 128.

These bumps are the early kind of spill. The soft pool scout scored `τ_ℓ fixed` at seeds 704 to 711 on the same early checkpoints (16, 32, 64, and 128) and saw no spill above 0.003 there. By epoch 32 the anneal has moved $\tau_\ell$ by about one percent, so the bumps do not come from the anneal.

Two runs of eight against none of eight can't tell whether the two draws of seeds differ here or the bumps are simply uncommon. The stored runs of the step recipes were not scored this densely, so we don't know how common such bumps are under them.
<!-- REVIEW: was "which clear by the next checkpoint" (714 is at 0.0203 at epoch 64, over the line, and near zero at 96) and said nothing of the soft pool scout's early checkpoints on `τ_ℓ fixed`, which it scored at 16, 32, 64 and 128 (its E5: largest spill 0.003 up to epoch 128). Verify: ALONG[714], and E5 of the soft pool scout. -->

Under the edit at block 1 alone, one run goes over the criterion ({RISER}, at {num(RISER_B1)}). The run that rises late is also the one that spills at block 1. That fits what recipe-compare found under the pool: the spill enters with the edit at block 1. But one run of every recipe goes over at block 1 (table in E1), the step recipes included, so the block-1 spill on its own does not single out the pool.
<!-- REVIEW: "So the spill of seed 713 enters through block 1" lowered to "fits": one restricted edit on one run, and the E1 table has one run over at block 1 under every recipe. Verify: BLOCK1_OVER. -->
"""

# %%

rf"""
## Lost anchors at forty seeds (E3)

How many of the forty runs lose the anchor by the call at epoch 64, and how does that compare with `falloff` and `step, soft τ` at the same seeds?

{count_figure()}

The call finds all forty runs kept. The lowest window mean (mean α over epochs 44 to 64) is seed {LOWEST} at {CALLS[LOWEST]["window_mean"]:.2f}, above the line where a run counts as kept, so no run needed finishing to settle its call.

With none lost of forty, the rate of lost anchors is below {LOSS_RATE_BOUND:.0%} (a one-sided 95% bound), the same bound as for `step, soft τ` in the loss-rate screen. The call is made at epoch 64. The eight runs that went on to epoch 400 kept the anchor there too.

Seed by seed, the window means are above those of `falloff` on {ABOVE_FALLOFF} of forty seeds and above those of `step, soft τ` on {ABOVE_STEP}. A soft position temperature at the start, which `τ_ℓ anneal` shares with `step, soft τ`, may be what brings the example answers onto e₁ early on every seed. But `τ_ℓ anneal` is also above `step, soft τ` on most seeds, so the soft start is not the whole of it. The pool over three slices may add to it, but nothing here separates the two.
<!-- REVIEW: "which brings the example answers onto e₁ early" hedged: the recipe also sits above `step, soft τ`, which shares the soft start, on 35 of forty seeds, so the soft start can't be the whole explanation and no arm here isolates it. Verify: ABOVE_STEP. -->
"""

# %%

r"""
## Discussion

`τ_ℓ anneal` meets three of the four checks outright. Every run at the fresh seeds keeps the anchor (check 1), none ends over the spill criterion though one ends at it (check 2), and the count of lost anchors is the lowest a count of forty can give (check 4).

Check 3 has two halves, and the task half passes on every run. On the removal half, `τ_ℓ anneal` is a little below `falloff` on the seed mean, by less than two draws of eight seeds can differ (Scope and method). So that check is met within the seed spread rather than outright.
<!-- REVIEW: was "meets the four checks". Check 3 asks for removal no lower than `falloff`, and the seed mean is 0.858 against 0.872, so it is met only within the seed spread the Scope allows for. Verify: REMOVAL[NEW].mean() against REMOVAL[FALLOFF].mean(). -->

The late rise is still there on one seed in eight, at about the size the soft pool scout saw. So the slow anneal of $\tau_\ell$ does not remove the late kind of spill; at most it keeps it at the line here.

The late spill scout found the effect of the late sharpening to be about the size of the difference between two GPUs, so a confirmation at eight seeds can't tell whether it helps. What the run shows is no cost we can separate from the seed draw: removal is lower than under `τ_ℓ fixed` on the mean, and E1 can't say whether the anneal or the seeds did that.
<!-- REVIEW: "does not cost anything we can see" became "no cost we can separate from the seed draw", since removal is 0.04 below `τ_ℓ fixed` on the mean and E1 leaves open whether the anneal or the seeds account for it. Verify: REMOVAL[NEW].mean() against REMOVAL[FIXED].mean(). -->

Two things favor the pool recipe over `falloff`. It chooses the slice within a range set by the architecture (slices 1 to $L-1$), and its early call is never unsure, where `falloff` stayed in the unsure band on three seeds of forty. Its count of lost anchors, none of forty against one, is a tie at this many seeds.

Against the pool recipe, one seed in eight ends at the spill line, while the eight stored runs of `falloff` end at most at 0.014. That is one run, set against the seed draw and the GPU shift, so it is a small mark rather than a difference we can rely on. Under the block-1 edit it goes over on one seed, as one run of every recipe does, and its removal varies across seeds as under the step recipes.

The early bumps on two seeds are the other mark, and the larger one goes well over the line. Under `τ_ℓ fixed` the soft pool scout saw none at the same checkpoints, but at epoch 32 the anneal has moved $\tau_\ell$ by about one percent, so the two recipes are nearly the same there and the difference is most likely between the two draws of seeds. Both bumps cleared within 64 epochs, and the step recipes were never scored this early, so we can't say whether they bump too.
<!-- REVIEW: added the early bumps to the case against, as the style review's retelling weighed them and the Discussion had left them out. Hedged as a seed-draw difference since τ_ℓ is 0.0994 at epoch 32 against 0.1 under `τ_ℓ fixed`. Verify: ALONG[714], ALONG[713], and E5 of the soft pool scout. -->
<!-- REVIEW: the "for" and "against" lists were reconciled with the data: "loses no anchor in forty" moved from a point for to a tie, as check 4 in the Scope says forty seeds can't tell none from one; the block-1 spill moved from a point against to a tie, since the E1 table has one run over at block 1 under every recipe; and the one run at the spill line was added as the one visible mark against, hedged. Verify: BLOCK1_OVER, SPILL[FALLOFF].max(), and the falloff window means in the band (3 of forty). -->
"""

# %%

r"""
## References

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The pool over slices 1 to 3 with $\tau_\ell$ held at 0.1 keeps the anchor on all eight seeds with no early spill, and one seed rises late to near the criterion.

[late spill scout](/docs/m2/late-spill-scout/report.py)
:   The late spill follows the anti anneal, and sharpening $\tau_\ell$ to 0.025 over epochs 64 to 360 lowers it with no measured cost to removal or the task.

[loss-rate screen](/docs/m2/loss-rate/report.py)
:   A call from α at the example answers over epochs 44 to 64 sorts lost anchors from kept ones; at forty seeds `falloff` lost one and `step, soft τ` none.

[recipe-compare](/docs/m2/recipe-compare/report.py)
:   The spill of the pool with a soft slice temperature enters at block 1, the one slice the pool pulls and the step recipes leave alone.
"""
