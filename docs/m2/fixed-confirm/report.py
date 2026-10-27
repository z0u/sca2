# ruff: noqa: B018
# title: Fixed confirmation

# An exploratory report: the soft pool scout's candidate recipe, `τ_ℓ fixed`, at the anneal confirmation's fresh
# seeds 712 to 719 and in a loss-rate count at forty seeds, paired seed by seed with `τ_ℓ anneal`. `experiment.py`
# beside this script trains and scores the runs.

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


ac, sps, lr, rb = ex.ac, ex.sps, ex.lr, ex.rb
AXES_BODY = fetch_json(ex.AXES_REF)
DESIGN = AXES_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
COUNT_SEEDS: list[int] = DESIGN["count_seeds"]
OLD_SEEDS: list[int] = DESIGN["stored_seeds"]
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
TASK_MEAN_GATE, TASK_OP_GATE = -0.02, -0.05  # the task gate of the D2.2 design
LOST_BELOW = 0.5  # a finished run has lost the anchor when the full edit removes less than this
E1 = 0  # the anchored axis, in code
STOP = ex.STOP_EPOCH
GPU_SHIFT = 0.003  # how far a change of GPU moved the end spill on `τ_ℓ fixed` (the late spill scout, Scope and method)

# Recipes, as (recipe, seed set): `τ_ℓ fixed` appears twice, at the soft pool scout's seeds and at the fresh ones.
FALLOFF, STEP, FIXED, ANNEAL = "falloff", "step-soft", "fixed", "anneal"
FIXED_OLD = "fixed-old"
COLUMNS = [FALLOFF, STEP, FIXED_OLD, FIXED, ANNEAL]
SHOWN = {
    FALLOFF: "falloff",
    STEP: "step, soft τ",
    FIXED_OLD: "τ_ℓ fixed",
    FIXED: "τ_ℓ fixed",
    ANNEAL: "τ_ℓ anneal",
}
SEEDS_OF = dict.fromkeys((FALLOFF, STEP, FIXED_OLD), OLD_SEEDS) | dict.fromkeys((FIXED, ANNEAL), SEEDS)


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


FULL = len(rb.DOSE_GAMMAS) - 1


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


_FIX_AX = {r["label"]: r for r in AXES_BODY["runs"]}
_AC_AX = {r["label"]: r for r in fetch_json(ac.AXES_REF)["runs"]}
_RB = {r["label"]: r for r in fetch_json(rb.AXES_REF)["runs"]}
_SPS = {(r["condition"], r["model_seed"]): r for r in fetch_json(sps.AXES_REF)["runs"]}


def _control(s: int) -> dict:
    return _AC_AX[ac.label_of(s, None, "control")] if s in SEEDS else _RB[f"control-s{s}"]


_RUN = {
    FALLOFF: lambda s: _RB[f"falloff-s{s}"],
    STEP: lambda s: _RB[rb.SOFT_POOL + f"step-soft-s{s}"],
    FIXED_OLD: lambda s: _SPS[(ex.BASE, s)],
    FIXED: lambda s: _FIX_AX[ex.label_of(s, None)],
    ANNEAL: lambda s: _AC_AX[ac.label_of(s, None)],
}
END = {(n, s): axis_scores(get(s), _control(s)) for n, get in _RUN.items() for s in SEEDS_OF[n]}


def by_seed(n: str, key: str) -> np.ndarray:
    return np.array([END[(n, s)][key] for s in SEEDS_OF[n]])


REMOVAL = {n: by_seed(n, "removal") for n in COLUMNS}
SPILL = {n: by_seed(n, "spill") for n in COLUMNS}
TASK = {n: by_seed(n, "task") for n in COLUMNS}  # (seeds, ops)
OVER = {n: int((SPILL[n] > CRITERION).sum()) for n in COLUMNS}
LOST = {n: int((REMOVAL[n] < LOST_BELOW).sum()) for n in COLUMNS}
TASK_MISS = {n: int(((TASK[n].mean(axis=1) < TASK_MEAN_GATE) | (TASK[n].min(axis=1) < TASK_OP_GATE)).sum()) for n in COLUMNS}  # fmt: skip

# The restricted edit at block 1 alone, net of the median along the other 63 axes of the run.
_RESTRICTED = {r["label"]: r for r in fetch_json(ex.RESTRICTED_REF)["runs"]}
_RESTRICTED_AC = {r["label"]: r for r in fetch_json(ac.RESTRICTED_REF)["runs"]}
_RESTRICTED_OLD = {r["label"]: r for r in fetch_json(sps.RESTRICTED_REF)["runs"]}
_RESTRICTED_OF = {
    FALLOFF: lambda s: _RESTRICTED_OLD[f"falloff-s{s}"],
    STEP: lambda s: _RESTRICTED_OLD[f"step-soft-s{s}"],
    FIXED_OLD: lambda s: _RESTRICTED_OLD[ex.BASE + f"-s{s}"],
    FIXED: lambda s: _RESTRICTED[ex.label_of(s, None)],
    ANNEAL: lambda s: _RESTRICTED_AC[ac.label_of(s, None)],
}


def block1_spill(n: str, s: int) -> float:
    return float(net_of(_RESTRICTED_OF[n](s)["drop"]["block 1"])[OTHER].max())


BLOCK1 = {n: np.array([block1_spill(n, s) for s in SEEDS_OF[n]]) for n in COLUMNS}
BLOCK1_OVER = {n: int((BLOCK1[n] > CRITERION).sum()) for n in COLUMNS}

# --- Along training: the full edit on each kept checkpoint, net of the median over the other 63 axes ----------

_ALONG = {FIXED: fetch_json(ex.ALONG_REF)["runs"], ANNEAL: fetch_json(ac.ALONG_REF)["runs"]}
_LABEL = {FIXED: lambda s: ex.label_of(s, None), ANNEAL: lambda s: ac.label_of(s, None)}
ALONG_EPOCHS = sorted({r["epoch"] for r in _ALONG[FIXED]})
assert ALONG_EPOCHS == sorted({r["epoch"] for r in _ALONG[ANNEAL]})
_ALONG_BY = {(n, r["label"], r["epoch"]): r for n, runs in _ALONG.items() for r in runs}


def along(n: str, s: int) -> np.ndarray:
    """Spill at each kept checkpoint and the end, for the run of *n* at seed *s*."""
    return np.array([net_of(_ALONG_BY[(n, _LABEL[n](s), e)]["drop"])[OTHER].max() for e in ALONG_EPOCHS])


ALONG = {(n, s): along(n, s) for n in (FIXED, ANNEAL) for s in SEEDS}

# --- The loss-rate count: α at the example answers over epochs 44 to 64 ---------------------------------------

_CALLS = {r["model_seed"]: r for r in fetch_json(ex.CALLS_REF)["runs"]}
_SPS_SCREEN = {r["model_seed"]: r for r in fetch_json(sps.SCREEN_REF)["runs"] if r["condition"] == ex.BASE}
_AC_CALLS = {r["model_seed"]: r for r in fetch_json(ac.CALLS_REF)["runs"]}
_LR_CALLS = fetch_json(lr.CALLS_REF)["runs"]
_STORED_TRAJ = {FALLOFF: fetch_json(lr.sas.TRAJ_REF), STEP: fetch_json(lr.sp.TRAJ_REF)}


def window_of(n: str, s: int) -> float:
    if n == FIXED:
        return (_SPS_SCREEN if s in OLD_SEEDS else _CALLS)[s]["window_mean"]
    if n == ANNEAL:
        return _AC_CALLS[s]["window_mean"]
    if s in OLD_SEEDS:
        return lr.window_mean(_STORED_TRAJ[n][f"{n}-s{s}"]["traj"])
    return next(r["window_mean"] for r in _LR_CALLS if r["arm"] == n and r["model_seed"] == s)


COUNTED = [FALLOFF, STEP, ANNEAL, FIXED]
WINDOW = {n: np.array([window_of(n, s) for s in COUNT_SEEDS]) for n in COUNTED}
CALL = {n: [lr.call_of(float(w)) for w in WINDOW[n]] for n in COUNTED}
NOT_KEPT = {n: sum(c != "kept" for c in CALL[n]) for n in COUNTED}
LOWEST = COUNT_SEEDS[int(WINDOW[FIXED].argmin())]
ABOVE = {n: int((WINDOW[FIXED] > WINDOW[n]).sum()) for n in (FALLOFF, STEP, ANNEAL)}
LOSS_RATE_BOUND = 1 - 0.05 ** (1 / len(COUNT_SEEDS))  # one-sided 95% upper bound on the rate, at 0 of n

# --- Early spill: the full edit at epochs 16, 32 and 64 on every stopped run of both recipes ------------------

_EARLY = fetch_json(ex.EARLY_REF)["runs"]
EARLY_EPOCHS: list[int] = DESIGN["early_epochs"]
_EARLY_BY = {(r["kind"], r["model_seed"], r["epoch"]): r for r in _EARLY}
EARLY = {
    n: np.array([[net_of(_EARLY_BY[(n, s, e)]["drop"])[OTHER].max() for e in EARLY_EPOCHS] for s in COUNT_SEEDS])
    for n in (FIXED, ANNEAL)
}  # (seeds, epochs)
EARLY_OVER = {n: (EARLY[n] > CRITERION).sum(axis=0) for n in EARLY}  # runs over the criterion, per epoch
EARLY_ANY = {n: int((EARLY[n] > CRITERION).any(axis=1).sum()) for n in EARLY}  # runs over at any early epoch


# --- Figures ---------------------------------------------------------------------------------------------------

RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
GHOST = light_dark("#bbb", "#555")
INK = {
    FALLOFF: light_dark("#a6611a", "#dfc27d"),
    STEP: light_dark("#1b7837", "#a6dba0"),
    FIXED_OLD: light_dark("#08306b", "#c6dbef"),
    FIXED: light_dark("#08306b", "#c6dbef"),
    ANNEAL: light_dark("#7570b3", "#bcbddc"),
}
MARKER = {FALLOFF: "v", STEP: "^", FIXED_OLD: "s", FIXED: "s", ANNEAL: "D"}


def tex(shown: str) -> str:
    """A recipe name for a figure label, with its symbols as mathtext."""
    return shown.replace("τ_ℓ", "$\\tau_\\ell$").replace("τ", "$\\tau$")


def tick_of(n: str) -> str:
    seeds = SEEDS_OF[n]
    return f"{tex(SHOWN[n])}\n{seeds[0]}–{seeds[-1]}"


def hatch(ax, lo: float, hi: float) -> None:
    """Hatch the band from *lo* to *hi* as the failing side of a line (style-fig)."""
    ax.axhspan(lo, hi, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, alpha=0.12, zorder=0)  # fmt: skip


def _dots(ax, x: float, v, color, m: str, jit) -> None:
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def end_figure() -> str:
    data = {
        "names": COLUMNS,
        "ticks": [tick_of(n) for n in COLUMNS],
        "removal": {n: REMOVAL[n].tolist() for n in COLUMNS},
        "spill": {n: SPILL[n].tolist() for n in COLUMNS},
        "task": {n: TASK[n].mean(axis=1).tolist() for n in COLUMNS},
        "paired": [COLUMNS.index(FIXED), COLUMNS.index(ANNEAL)],
        "criterion": CRITERION,
        "task_gate": TASK_MEAN_GATE,
    }
    r, ra = REMOVAL[FIXED], REMOVAL[ANNEAL]
    alt = f"""
        Three panels (removal, spill, task change), each with five columns: falloff, step soft τ and τ_ℓ fixed at
        seeds 704 to 711, then τ_ℓ fixed and τ_ℓ anneal at the fresh seeds 712 to 719, with grey lines joining the
        two runs of each fresh seed. Removal under τ_ℓ fixed at the fresh seeds spreads from {r.min():.2f} to
        {r.max():.2f}, mean {r.mean():.2f}; under τ_ℓ anneal from {ra.min():.2f} to {ra.max():.2f}, mean
        {ra.mean():.2f}. TODO: describe the pairing once seen. Spill: τ_ℓ fixed at the fresh seeds tops out at
        {SPILL[FIXED].max():.3f}, against a criterion of 0.02. Task change: every fresh run of τ_ℓ fixed between
        {TASK[FIXED].mean(axis=1).min():.3f} and {TASK[FIXED].mean(axis=1).max():.3f}.
    """
    return end_draw(data, alt)


@memo
def end_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-confirm-end",
        alt_text=alt_text,
        caption="""
            **Removal, spill, and task at epoch 400, by recipe.** One column per recipe and seed set: the stored runs
            of `falloff`, `step, soft τ` and `τ_ℓ fixed` at seeds 704 to 711, and `τ_ℓ fixed` and `τ_ℓ anneal` at the
            fresh seeds 712 to 719, with a grey line joining the two runs of each fresh seed. The seeds are small dots
            over a thin bar for their range, and the seed mean is the large mark. **Left:** removal at the full dose.
            **Middle:** spill, the largest net drop on another op at any dose; the dashed line is the criterion, with
            the side over it hatched. Both are net of the median over the other 63 axes of the run. **Right:** the
            task change, the mean clean score over ops less that of the control at the seed, with the gate of the
            D2.2 design dashed.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), layout="constrained")
        rng = np.random.default_rng(0)
        jit = {n: rng.uniform(-0.12, 0.12, len(data["removal"][n])) for n in names}
        a, b = data["paired"]
        for ax, key in zip(axes, ("removal", "spill", "task"), strict=True):
            va, vb = data[key][names[a]], data[key][names[b]]
            for j in range(len(va)):
                ax.plot([a + jit[names[a]][j], b + jit[names[b]][j]], [va[j], vb[j]], color=GHOST, lw=0.4, zorder=1)
            for i, n in enumerate(names):
                _dots(ax, i, data[key][n], INK[n], MARKER[n], jit[n])
            ax.axvline(a - 0.5, color=GHOST, lw=0.5, ls=":", zorder=0)
            ax.set_xticks(range(len(names)), data["ticks"], rotation=50, ha="right", fontsize=5.5)
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
    for n in COLUMNS:
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
        ["recipe", "seeds", "removal ↑", "spill ↓", "over criterion ↓", "over, block 1 ↓", "task change ↑", "worst op ↑"],  # fmt: skip
        rows,
        """
        **Removal, spill and task at epoch 400**, seed mean and range over eight runs, net of the median drop over the
        other 63 coordinate axes of the run. "Over, block 1" counts the runs over the criterion under the same edit
        restricted to block 1. The task change is the mean over ops of the clean score less that of the control at
        the seed; "worst op" is the lowest such change on any op of any run. The last two rows share their seeds, and
        the controls at those seeds.
        """,
    )


def pair_table() -> str:
    rows = []
    for s in SEEDS:
        f, a = END[(FIXED, s)], END[(ANNEAL, s)]
        rows.append([
            str(s),
            f"{f['removal']:.2f}",
            f"{a['removal']:.2f}",
            num(f["spill"]),
            num(a["spill"]),
            num(float(f["task"].mean())),
            num(float(a["task"].mean())),
        ])  # fmt: skip
    return table_html(
        ["seed", "removal, fixed ↑", "removal, anneal ↑", "spill, fixed ↓", "spill, anneal ↓", "task, fixed ↑", "task, anneal ↑"],  # fmt: skip
        rows,
        """
        **The two recipes seed by seed at epoch 400**, at the fresh seeds. Removal and spill as in the table above;
        the task change is the mean over ops against the shared control at the seed.
        """,
    )


def along_figure() -> str:
    data = {
        "epochs": list(ALONG_EPOCHS),
        "seeds": SEEDS,
        "spill": {n: {s: ALONG[(n, s)].tolist() for s in SEEDS} for n in (FIXED, ANNEAL)},
        "shown": {n: SHOWN[n] for n in (FIXED, ANNEAL)},
        "criterion": CRITERION,
    }
    alt = """
        Two stacked panels sharing an epoch axis from 16 to 400 and a spill axis: τ_ℓ fixed above, τ_ℓ anneal below,
        each with one line per seed at 712 to 719 and the other recipe drawn as faint ghosts. TODO: describe.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-confirm-along",
        alt_text=alt_text,
        caption="""
            **Spill along training, paired by seed.** The spill at the checkpoints kept at epochs 16, 32 and 64 and
            then every 32 epochs, and at the end, net of the median over the other 63 axes of the run, for each of the
            fresh seeds. **Top:** `τ_ℓ fixed`; **bottom:** `τ_ℓ anneal`, with the other recipe as faint lines behind.
            The dashed line is the criterion, with the side over it hatched.
        """,
    )
    def _plot() -> plt.Figure:
        ep = np.asarray(data["epochs"])
        names = list(data["spill"])
        fig, axes = plt.subplots(2, 1, figsize=(4.6, 3.8), layout="constrained", sharex=True, sharey=True)
        for ax, n in zip(axes, names, strict=True):
            other = next(m for m in names if m != n)
            for s in data["seeds"]:
                ax.plot(ep, data["spill"][other][s], color=GHOST, lw=0.4, alpha=0.6, zorder=1)
            for s in data["seeds"]:
                ax.plot(ep, data["spill"][n][s], color=INK[n], marker=MARKER[n], ms=1.8, lw=0.8, alpha=0.8, zorder=3)
            ax.text(0.01, 0.95, tex(data["shown"][n]), transform=ax.transAxes, fontsize=7, va="top")
            ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
            ax.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=1)
            ax.set_ylabel("spill ↓", fontsize=7)
            ax.tick_params(labelsize=6)
        lo, hi = axes[0].get_ylim()
        for ax in axes:
            hatch(ax, data["criterion"], hi)
        axes[0].set_ylim(lo, hi)
        axes[1].set_xlabel("epoch", fontsize=7)
        return fig

    return _plot()


def count_figure() -> str:
    data = {
        "names": COUNTED,
        "ticks": [tex(SHOWN[n]) for n in COUNTED],
        "window": {n: WINDOW[n].tolist() for n in COUNTED},
        "lost_max": lr.LOST_MAX,
        "kept_min": lr.KEPT_MIN,
    }
    w = WINDOW[FIXED]
    alt = f"""
        Window means of α at the example answers for four recipes at the same forty seeds, as dots in four columns
        (falloff, step soft τ, τ_ℓ anneal, τ_ℓ fixed) with grey lines joining each seed across them. Dashed lines at
        0.15 (lost below) and 0.28 (kept above). τ_ℓ fixed spreads from {w.min():.2f} to {w.max():.2f}. TODO: describe.
    """
    return count_draw(data, alt)


@memo
def count_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-confirm-count",
        alt_text=alt_text,
        caption=f"""
            **The early call at forty seeds.** Mean α at the example answers at slice 3 over epochs 44 to 64, for
            each run, with a grey line joining the runs of a seed. `falloff` and `step, soft τ` are the calls of the
            loss-rate screen (and at seeds 704 to 711, the same window mean on their stored full runs); `τ_ℓ anneal`
            is the anneal confirmation's count. Below {data["lost_max"]} a run is called lost and from
            {data["kept_min"]} kept, with the band between finished and measured.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, ax = plt.subplots(figsize=(4.2, 2.6), layout="constrained")
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
        ax.set_xticks(range(len(names)), data["ticks"], fontsize=7)
        ax.set_ylabel("α at the example answers,\nepochs 44 to 64", fontsize=7)
        ax.tick_params(labelsize=7)
        return fig

    return _plot()


def early_figure() -> str:
    data = {
        "epochs": EARLY_EPOCHS,
        "names": [FIXED, ANNEAL],
        "shown": {n: SHOWN[n] for n in (FIXED, ANNEAL)},
        "spill": {n: EARLY[n].tolist() for n in (FIXED, ANNEAL)},
        "criterion": CRITERION,
    }
    alt = f"""
        Spill at epochs 16, 32 and 64 for the forty stopped runs of τ_ℓ fixed and of τ_ℓ anneal, as dots in pairs of
        columns per epoch, with grey lines joining the two runs of each seed. Runs over the criterion of 0.02: τ_ℓ
        fixed {", ".join(str(int(c)) for c in EARLY_OVER[FIXED])}; τ_ℓ anneal
        {", ".join(str(int(c)) for c in EARLY_OVER[ANNEAL])} at the three epochs. TODO: describe.
    """
    return early_draw(data, alt)


@memo
def early_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-confirm-early",
        alt_text=alt_text,
        caption="""
            **Early spill at forty seeds.** The spill on the checkpoints kept at epochs 16 and 32 and at the stop at
            epoch 64, for each of the forty stopped runs of the two recipes, net of the median over the other 63 axes
            of the run; a grey line joins the two runs of each seed. The dashed line is the criterion, with the side
            over it hatched.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
        rng = np.random.default_rng(2)
        ticks, labels = [], []
        for k, e in enumerate(data["epochs"]):
            xs = {n: 3 * k + i for i, n in enumerate(names)}
            v = {n: np.asarray(data["spill"][n])[:, k] for n in names}
            jit = {n: rng.uniform(-0.15, 0.15, len(v[n])) for n in names}
            for j in range(len(v[names[0]])):
                ax.plot([xs[n] + jit[n][j] for n in names], [v[n][j] for n in names], color=GHOST, lw=0.3, zorder=1)
            for n in names:
                ax.plot(xs[n] + jit[n], v[n], MARKER[n], ms=2.6, color=INK[n], alpha=0.75, mew=0, zorder=3)
            ticks += [xs[n] for n in names]
            labels += [f"{tex(data['shown'][n])}\nepoch {e}" for n in names]
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        ax.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=1)
        lo, hi = ax.get_ylim()
        hi = max(hi, data["criterion"] + 0.005)
        hatch(ax, data["criterion"], hi)
        ax.set_ylim(lo, hi)
        ax.set_xticks(ticks, labels, fontsize=5.5)
        ax.set_ylabel("spill ↓", fontsize=7)
        ax.tick_params(labelsize=6)
        return fig

    return _plot()


def early_table() -> str:
    rows = []
    for n in (FIXED, ANNEAL):
        rows.append(
            [f"`{SHOWN[n]}`"]
            + [f"{int(c)} of {len(COUNT_SEEDS)}" for c in EARLY_OVER[n]]
            + [f"{EARLY_ANY[n]} of {len(COUNT_SEEDS)}", num(float(EARLY[n].max()))]
        )
    return table_html(
        ["recipe", *[f"over, epoch {e} ↓" for e in EARLY_EPOCHS], "over at any ↓", "largest ↓"],
        rows,
        """
        **Early spill at forty seeds**: the runs over the criterion at each early checkpoint, the runs over it at any
        of the three, and the largest spill of any run at any of them.
        """,
    )


r"""
# Fixed confirmation

/// tip |
<!-- lede -->
TODO
///

The [anneal confirmation](/docs/m2/anneal-confirm/report.py) ran the pool over slices 1 to 3 with its slice temperature $\tau_\ell$ annealing from 0.1 to 0.025, at eight fresh seeds (712 to 719).[^tau] Every run kept the anchor. But removal varied across seeds more than it had under the simpler recipe it grew from, `τ_ℓ fixed` from the [soft pool scout](/docs/m2/soft-pool-scout/report.py), which holds $\tau_\ell$ at 0.1. Two seeds removed about as little as the lowest runs of the step recipes. The two recipes had run at different seeds, so the anneal and the seed draw could each account for that.

[^tau]: Slice temperature: how sharply the pooled pull picks among slices. The pull is a softmin-weighted mix over slice and position pairs, and $\tau_\ell$ sets how peaked the weights over slices are, as the pooling temperature $\tau_p$ does over positions.

This report runs `τ_ℓ fixed` at the same fresh seeds, so the two recipes pair seed by seed. It is also the confirmation run for adopting `τ_ℓ fixed`, the simpler of the two, as the recipe of record: it trains the recipe to the end at 712 to 719, and to epoch 64 at forty seeds (704 to 743) to count lost anchors. On the stopped runs of both recipes it also scores spill at epochs 16, 32 and 64, which the anneal confirmation left unscored on most of its forty.
"""

# %%

r"""
## Observations

- TODO

## Scope and method

This is an exploratory study, with no preregistration and no gate. The checks are the four the anneal confirmation set for its recipe. They guide the choice of recipe without deciding it, so a recipe that misses one narrowly can still be the better choice:

1. Every run at seeds 712 to 719 keeps the anchor (removal at least 0.5).
2. No run at those seeds ends over the spill criterion of 0.02.
3. Removal is no lower than that of `falloff` at seeds 704 to 711. The task passes the gate of the D2.2 design against the control at the same seed: a mean change over ops of at least −0.02, and at least −0.05 on any one op.
4. At most one run of the forty loses the anchor.

`τ_ℓ fixed` (in full, `τ_ℓ 0.1, τ_ℓ fixed`) pulls e₁ at the pool over slices 1 to 3, with $\tau_\ell$ at 0.1 throughout. The position temperature $\tau_p$ sharpens from 1 to 0.1 by epoch 40, the anchor weight warms up over the first 40 epochs, and the anti weight anneals to the per-slice holds of `falloff` by epoch 360. `τ_ℓ anneal` differs in one schedule: $\tau_\ell$ moves from 0.1 to 0.025 over the whole run. The two are nearly the same over the first 64 epochs, where the anneal has moved $\tau_\ell$ by about four percent.

The forty runs of the count use the seeds of the loss-rate screen, so each pairs with a run of `falloff`, `step, soft τ`, and `τ_ℓ anneal` by initialization, batches, and label draws. Each stops at epoch 64 for its call. At seeds 704 to 711 the stopped runs are the soft pool scout's runs of this recipe, and the other 32 are new.

The runs at 712 to 719 continue from their stops to epoch 400, keeping a checkpoint every 32 epochs. They share their controls with the anneal confirmation (no pull and no anti term, at the same seed). All runs trained and were measured on L4 GPUs, as the anneal confirmation was.

Within the fresh seeds, the two recipes pair seed by seed, so a difference between them is not a difference of seed draws. It can still be as small as the noise a run carries from its numerics: in the late spill scout a change of GPU moved the end spill of one run by up to 0.003. Against the stored recipes at 704 to 711, a difference includes the variation between two draws of eight seeds, and under `falloff` and `step, soft τ` removal varies by about 0.15 across seeds.

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
## The two recipes at the same seeds (E1)

Does removal vary across the fresh seeds under `τ_ℓ fixed` as it did under `τ_ℓ anneal`, and does every run keep the anchor and pass the task gate?

{end_figure()}

{end_table()}

{pair_table()}

TODO
"""

# %%

rf"""
## Spill at the end and along training (E2)

Does any run end over the spill criterion, and do the late rise and the early bumps of the anneal confirmation show up at the same seeds under `τ_ℓ fixed`?

{along_figure()}

TODO
"""

# %%

rf"""
## Lost anchors and early spill at forty seeds (E3)

How many of the forty runs lose the anchor by the call at epoch 64, and how often does either recipe spill early?

{count_figure()}

{early_figure()}

{early_table()}

TODO
"""

# %%

r"""
## Discussion

TODO
"""

# %%

r"""
## References

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   With $\tau_\ell$ annealing from 0.1 to 0.025, the pool keeps the anchor on all eight runs at seeds 712 to 719 and on all forty in the count, with removal varying across seeds about as under `falloff`.

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The pool over slices 1 to 3 with $\tau_\ell$ held at 0.1 keeps the anchor on all eight seeds with no early spill, and one seed rises late to near the criterion.

[late spill scout](/docs/m2/late-spill-scout/report.py)
:   The late spill follows the anti anneal, and sharpening $\tau_\ell$ to 0.025 over epochs 64 to 360 lowers it with no measured cost to removal or the task.

[loss-rate screen](/docs/m2/loss-rate/report.py)
:   A call from α at the example answers over epochs 44 to 64 sorts lost anchors from kept ones; at forty seeds `falloff` lost one and `step, soft τ` none.
"""
