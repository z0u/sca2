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


def removal_along(n: str, s: int) -> np.ndarray:
    """Removal under the full edit at each kept checkpoint and the end, for the run of *n* at seed *s*: the net drop on
    the anchored op as a share of the way from the clean score to the target null of the end record of the run.
    """
    null = _RUN[n](s)["null"][D]
    rs = [_ALONG_BY[(n, _LABEL[n](s), e)] for e in ALONG_EPOCHS]
    return np.array([net_of(r["drop"])[D] / (r["clean"][D] - null) for r in rs])


REMOVAL_ALONG = {(n, s): removal_along(n, s) for n in (FIXED, ANNEAL) for s in SEEDS}

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
        {ra.mean():.2f}. The lines joining the paired runs cross: the two lowest anneal seeds sit high under τ_ℓ fixed, and the lowest τ_ℓ fixed seed is high under the anneal. Spill: τ_ℓ fixed at the fresh seeds tops out at
        {SPILL[FIXED].max():.3f}, against a criterion of 0.02. Task change: every fresh run of τ_ℓ fixed between
        {TASK[FIXED].mean(axis=1).min():.3f} and {TASK[FIXED].mean(axis=1).max():.3f}, the lowest one under the gate at −0.02.
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
        [
            "recipe",
            "seeds",
            "removal ↑",
            "spill ↓",
            "over criterion ↓",
            "over, block 1 ↓",
            "task change ↑",
            "worst op ↑",
        ],  # fmt: skip
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
        [
            "seed",
            "removal, fixed ↑",
            "removal, anneal ↑",
            "spill, fixed ↓",
            "spill, anneal ↓",
            "task, fixed ↑",
            "task, anneal ↑",
        ],  # fmt: skip
        rows,
        """
        **The two recipes seed by seed at epoch 400**, at the fresh seeds. Removal and spill as in the table above;
        the task change is the mean over ops against the shared control at the seed.
        """,
    )


def removal_along_figure() -> str:
    data = {
        "epochs": list(ALONG_EPOCHS),
        "seeds": SEEDS,
        "removal": {n: {s: REMOVAL_ALONG[(n, s)].tolist() for s in SEEDS} for n in (FIXED, ANNEAL)},
        "shown": {n: SHOWN[n] for n in (FIXED, ANNEAL)},
    }
    alt = f"""
        Eight small panels, one per fresh seed from 712 to 719, each with removal along training from epoch 16 to 400
        under τ_ℓ fixed and under τ_ℓ anneal. Removal starts near zero at epoch 16 under both. At epoch 32 the two runs
        of a seed are often far apart, by up to {GAP_32:.2f}, and most pairs come together by about epoch 160 near 0.9. Seeds
        714, 715, 718 and 719 keep a gap to the end: under the anneal 718 and 719 sit around 0.7 to 0.8 from about epoch
        64 to 96 on, and under τ_ℓ fixed 715 sits a little lower than its anneal twin from epoch 64 on.
    """
    return removal_along_draw(data, alt)


@memo
def removal_along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-confirm-removal-along",
        alt_text=alt_text,
        caption="""
            **Removal along training, paired by seed.** One panel per fresh seed: removal under the full edit at the
            checkpoints kept at epochs 16, 32 and 64 and then every 32 epochs, and at the end, for `τ_ℓ fixed` and
            `τ_ℓ anneal`. Net of the median over the other 63 axes of the run, as a share of the way to the target null
            of the run at the end.
        """,
    )
    def _plot() -> plt.Figure:
        ep = np.asarray(data["epochs"])
        fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.2), layout="constrained", sharex=True, sharey=True)
        for ax, s in zip(axes.flat, data["seeds"], strict=True):
            for n in data["removal"]:
                ax.plot(ep, data["removal"][n][s], color=INK[n], marker=MARKER[n], ms=2.0, lw=0.9, label=tex(data["shown"][n]))  # fmt: skip
            ax.text(0.95, 0.06, f"seed {s}", transform=ax.transAxes, fontsize=6.5, ha="right")
            ax.set_ylim(-0.1, 1.0)
            ax.tick_params(labelsize=6)
        for ax in axes[:, 0]:
            ax.set_ylabel("removal ↑", fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("epoch", fontsize=7)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


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
        each with one line per seed at 712 to 719 and the other recipe drawn as faint ghosts. Under τ_ℓ fixed every seed stays between about −0.02 and 0.005 after epoch 96; one seed spikes to 0.047 at epoch 64 and is back near zero by 96. Under τ_ℓ anneal one seed spikes to 0.08 at epoch 32, another bumps to 0.03 at epoch 96, and that second seed rises from about epoch 300 to 0.02 at the end.
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
        0.15 (lost below) and 0.28 (kept above). τ_ℓ fixed spreads from {w.min():.2f} to {w.max():.2f} and τ_ℓ anneal over about the same range, both above every step soft τ run but a few; falloff has four runs below 0.28, one of them near 0.
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
        {", ".join(str(int(c)) for c in EARLY_OVER[ANNEAL])} at the three epochs. Almost every run sits between −0.02 and 0.02; the runs over the line are at different seeds under the two recipes, except one seed that is at 0.05 under both at epoch 16.
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


# --- Numbers the prose quotes ----------------------------------------------------------------------------------

_AC_TRAJ = fetch_json(ac.FINISHED_TRAJ_REF)[ac.label_of(SEEDS[0], None)]["traj"]
TAU_64 = float(np.interp(STOP, _AC_TRAJ["epoch"], np.asarray(_AC_TRAJ["tau"])[:, 1]))  # τ_ℓ of the anneal at the stop
DIFF = REMOVAL[FIXED] - REMOVAL[ANNEAL]  # removal, fixed less anneal, by fresh seed
DIFF_MAX = float(np.abs(DIFF).max())
_GAP = {s: REMOVAL_ALONG[(FIXED, s)] - REMOVAL_ALONG[(ANNEAL, s)] for s in SEEDS}
GAP_32 = float(max(abs(_GAP[s][ALONG_EPOCHS.index(32)]) for s in SEEDS))  # the largest paired gap at epoch 32
TAU_32 = float(np.interp(32, _AC_TRAJ["epoch"], np.asarray(_AC_TRAJ["tau"])[:, 1]))
TAU_96 = float(np.interp(96, _AC_TRAJ["epoch"], np.asarray(_AC_TRAJ["tau"])[:, 1]))
WORD = "no one two three four five six seven eight nine ten".split()
N_FIXED_HIGHER = int((DIFF > 0).sum())
LOW_ANNEAL = sorted(SEEDS, key=lambda s: END[(ANNEAL, s)]["removal"])[:2]
LOW_FIXED = sorted(SEEDS, key=lambda s: END[(FIXED, s)]["removal"])[:2]
MISS = next(s for s in SEEDS if TASK[FIXED][SEEDS.index(s)].mean() < TASK_MEAN_GATE or TASK[FIXED][SEEDS.index(s)].min() < TASK_OP_GATE)  # fmt: skip
MISS_TASK = TASK[FIXED][SEEDS.index(MISS)]
MISS_OPS = [OPS[o] for o in np.argsort(MISS_TASK)[:2]]
LATE_EPOCH = 160  # where the late learner of the two HSV ops is still well behind the other seeds


def clean_at(n: str, s: int, op: str, e: int) -> float:
    return float(_ALONG_BY[(n, _LABEL[n](s), e)]["clean"][OPS.index(op)])


MISS_AT = clean_at(FIXED, MISS, MISS_OPS[0], LATE_EPOCH)
OTHERS_AT = np.array([clean_at(FIXED, s, MISS_OPS[0], LATE_EPOCH) for s in SEEDS if s != MISS])
LATE_ANNEAL = 713  # the seed whose spill rose late under the anneal
END_MAX = {n: float(max(ALONG[(n, s)][-1] for s in SEEDS)) for n in (FIXED, ANNEAL)}
EARLY_SEEDS = {n: sorted({COUNT_SEEDS[i] for i, _ in np.argwhere(EARLY[n] > CRITERION)}) for n in EARLY}
EARLY_SHARED = sorted(set(EARLY_SEEDS[FIXED]) & set(EARLY_SEEDS[ANNEAL]))
assert len(EARLY_SHARED) == 1, EARLY_SHARED  # the prose names one
BUMP = next(s for s in EARLY_SEEDS[FIXED] if s in SEEDS)  # the early bump on a finished run of `τ_ℓ fixed`
BUMP_PEAK = float(ALONG[(FIXED, BUMP)][ALONG_EPOCHS.index(STOP)])
BUMP_NEXT = float(ALONG[(FIXED, BUMP)][ALONG_EPOCHS.index(STOP) + 1])
OLD_EARLY_MAX = float(EARLY[FIXED][[COUNT_SEEDS.index(s) for s in OLD_SEEDS]].max())


r"""
# Fixed confirmation

/// tip |
<!-- lede -->
The pool over slices 1 to 3, with its slice temperature held fixed, keeps the anchor on all eight runs at fresh seeds, and on all forty runs of a count that calls each run at epoch 64. Paired seed by seed with the annealed version of the recipe, it removes about as much, with no late rise in spill.

But some seeds that did well under one recipe did poorly under the other, and the two runs of a seed parted ways early, while the two recipes were still nearly the same. So much of the variation between runs seems to come from how sensitive a run is to small changes, rather than from either recipe.
///

The [anneal confirmation](/docs/m2/anneal-confirm/report.py) ran the pool over slices 1 to 3 with its slice temperature $\tau_\ell$ annealing from 0.1 to 0.025, at eight fresh seeds (712 to 719).[^tau] Every run kept the anchor, but removal varied across seeds more than under the recipe it grew from: `τ_ℓ fixed` from the [soft pool scout](/docs/m2/soft-pool-scout/report.py), which holds $\tau_\ell$ at 0.1.

Two seeds removed about as little as the lowest runs of the step recipes. The two recipes had run at different seeds, so either the anneal or the seed draw could account for that.

[^tau]: Slice temperature: how sharply the pooled pull picks among slices. The pull is a softmin-weighted mix over slice and position pairs, and $\tau_\ell$ sets how peaked the weights over slices are, as the pooling temperature $\tau_p$ does over positions.

This report runs `τ_ℓ fixed` at the same fresh seeds, so the two recipes pair seed by seed. It is also the confirmation run for adopting `τ_ℓ fixed`, the simpler of the two, as the recipe of record: it trains to the end at seeds 712 to 719, and to epoch 64 at forty seeds (704 to 743) to count lost anchors. On the stopped runs of both recipes it also scores spill at epochs 16, 32, and 64, which the anneal confirmation left unscored on most of its forty runs.
"""

# %%

rf"""
## Observations

- [The two recipes at the same seeds (E1)](#the-two-recipes-at-the-same-seeds-e1): every run keeps the anchor. On the seed mean, removal is a little higher than under `τ_ℓ anneal` and spreads less, but seed by seed the difference runs both ways, by up to about a tenth. Of the two seeds that removed least under the anneal, one removes about as much as the seed mean here and the other is still among the lowest; a third seed removes less. One run misses the task gate, on a seed that learns two ops late.
<!-- REVIEW: was "The two seeds that removed least under the anneal remove well here". Seed 719 is 0.85 under `τ_ℓ fixed`, the second-lowest of the eight, and E1 itself calls it among the lowest under both; only 718 (0.88) sits at the seed mean. Verify: LOW_ANNEAL, REMOVAL[FIXED]. -->

- [Spill at the end and along training (E2)](#spill-at-the-end-and-along-training-e2): no run ends near the spill line, and none rises late, including the seed that rose to the line under the anneal. One run has a brief early bump over the line, which clears by the next checkpoint.
- [Lost anchors and early spill at forty seeds (E3)](#lost-anchors-and-early-spill-at-forty-seeds-e3): the call finds all forty runs kept. At epochs 16 to 64, {WORD[EARLY_ANY[FIXED]]} runs of forty spill over the line at some checkpoint under `τ_ℓ fixed` and {WORD[EARLY_ANY[ANNEAL]]} under `τ_ℓ anneal`, mostly at different seeds, though the two recipes barely differ before epoch 64.

## Scope and method

This is an exploratory study, with no preregistration and no gate. It uses the four checks that the anneal confirmation set for its recipe, which guide the choice of recipe without deciding it, so a recipe that misses one narrowly can still be the better choice:

1. Every run at seeds 712 to 719 keeps the anchor (removal at least 0.5).
2. No run at those seeds ends over the spill criterion of 0.02.
3. Removal is no lower than that of `falloff` at seeds 704 to 711. The task passes the gate of the D2.2 design against the control at the same seed: a mean change over ops of at least −0.02, and at least −0.05 on any one op.
4. At most one run of the forty loses the anchor.

`τ_ℓ fixed` (in full, `τ_ℓ 0.1, τ_ℓ fixed`) pulls e₁ at the pool over slices 1 to 3, with $\tau_\ell$ at 0.1 throughout. The position temperature $\tau_p$ sharpens from 1 to 0.1 by epoch 40, the anchor weight warms up over the first 40 epochs, and the anti weight anneals to the per-slice holds of `falloff` by epoch 360.

`τ_ℓ anneal` differs in one schedule: $\tau_\ell$ moves from 0.1 to 0.025 over the whole run, slowly at first, so it is still {TAU_64:.3f} at epoch 64. Over the first 64 epochs the two recipes are nearly the same.

The forty runs of the count use the seeds of the loss-rate screen, so each shares its initialization, batches, and label draws with a run of `falloff`, of `step, soft τ`, and of `τ_ℓ anneal`. Each stops at epoch 64 for its call. At seeds 704 to 711 the stopped runs are those of this recipe from the soft pool scout; the other 32 are new.

The runs at 712 to 719 continue from their stops to epoch 400, keeping a checkpoint every 32 epochs. They share their controls with the anneal confirmation (no pull and no anti term, at the same seed). All runs trained and were measured on L4 GPUs, as the anneal confirmation was.

Within the fresh seeds, the two recipes pair seed by seed, so a difference between them is not a difference of seed draws. Against the stored recipes at seeds 704 to 711, a difference also includes the variation between two draws of eight seeds. Under `falloff` and `step, soft τ`, removal varies by about 0.15 across seeds.

## The measurements

Every edit below is the e₁ edit. Each score is the drop under that edit, less a baseline: the median drop when the same edit is made along each of the other 63 coordinate axes of the run (the rule of the D2.2 design).

Removal
:   The drop on `difference` under the full edit at every position and slice, as a share of the way from the clean score to the target null. Higher is better, and a run below 0.5 has lost the anchor.

Spill
:   The largest drop on any other op, at the dose where it is largest. Lower is better, and a run above 0.02 is over the criterion.

Spill along training
:   Spill at the full dose, measured on each checkpoint a run kept (epochs 16, 32, and 64, then every 32 epochs) and at the end. On the 32 runs stopped at epoch 64 only the first three exist.

Task change
:   The clean score per op, less that of the control at the seed.

Call
:   The early verdict on whether a run kept the anchor, from mean α at the example answers at slice 3 over epochs 44 to 64. A run below 0.15 is called lost, and one at 0.28 or above is called kept. A run in between is finished and measured.
"""

# %%

rf"""
## The two recipes at the same seeds (E1)

Does removal vary across the fresh seeds under `τ_ℓ fixed` as it did under `τ_ℓ anneal`, and does every run keep the anchor and pass the task gate? If the anneal made removal uneven, the two low seeds of the anneal confirmation should remove well here. If those seeds are hard for the pool, they should be low under both.

{end_figure()}

{end_table()}

Every run keeps the anchor. Removal under `τ_ℓ fixed` is a little above that of `τ_ℓ anneal` on the seed mean, and above `falloff` too. It spreads more than it did under the same recipe at seeds 704 to 711, though less than under the anneal.

{pair_table()}

Seed by seed, the differences run both ways, by up to {DIFF_MAX:.2f}: `τ_ℓ fixed` removes more on {WORD[N_FIXED_HIGHER]} of the eight seeds. The two low seeds of the anneal confirmation ({LOW_ANNEAL[0]} and {LOW_ANNEAL[1]}) remove more here, but seed {LOW_FIXED[0]}, which removed well under the anneal, is the lowest here, and seed 719 is among the lowest under both.

So neither account from the anneal confirmation fits on its own. The low seeds there were not low because of the seed alone, since seed {LOW_ANNEAL[1]} removes about as much as the seed mean here. Nor does the anneal lower removal across the board, since it removes more on some seeds.

By epoch 400 the anneal has taken $\tau_\ell$ down fourfold, so the end gaps could still be a late effect of the anneal that differs by seed. Removal along training shows when the gaps open.

{removal_along_figure()}

The runs of a seed part ways early. At epoch 32, where the anneal has moved $\tau_\ell$ by under one percent ({TAU_32:.4f} against 0.1), the two runs of some seeds are up to {GAP_32:.2f} apart in removal, and most pairs come back together by about epoch 160.

The two largest end gaps (seeds 718 and 719, a tenth each) are set by epoch 64 to 96, when $\tau_\ell$ under the anneal is still within about a tenth of 0.1 ({TAU_96:.3f} at epoch 96). They change little afterward, while the anneal does most of its work. The smaller lasting gaps, on seeds 714 and 715, wander by a few hundredths along the way.

So the end gaps were set while the two recipes were nearly the same. That fits a picture in which the final removal of a run is sensitive to small changes early on, while the anchor settles. If so, eight seeds of either recipe give a noisy estimate of its mean.

This does not rule out that a slightly sharper $\tau_\ell$ at epochs 64 to 96 matters, but a difference that small would be hard to tell from the same sensitivity. Seed 719, low under both, may also be a hard seed for the pool.
<!-- REVIEW: the warrant for the sensitivity reading moved from the end gaps to removal along training (claims review round 1): the end gaps alone could be a seed-dependent effect of the fourfold late sharpening, but the paired runs diverge at epoch 32 (τ_ℓ 0.6% apart) and the two largest end gaps (718, 719) are set by epoch 64 to 96. Verify: REMOVAL_ALONG, GAP_32, TAU_32, TAU_96. -->

Seven runs pass the task gate, with the mean over ops within about a hundredth of the control. Seed {MISS} misses it: its mean over ops is {num(float(MISS_TASK.mean()))}, and two ops, `{MISS_OPS[0]}` and `{MISS_OPS[1]}`, end {num(float(MISS_TASK.min()), ".2f")} and {num(float(np.sort(MISS_TASK)[1]), ".2f")} below the control.

This run learns those two ops late: at epoch {LATE_EPOCH} its clean score on `{MISS_OPS[0]}` is {MISS_AT:.2f}, where the other seven runs are at {OTHERS_AT.min():.2f} or more, and it is still rising at the end. Under the anneal, the same seed learned them on time, and another seed (717) was the late one but caught up. So this too looks like a difference between runs more than a property of the recipe.
"""

# %%

rf"""
## Spill at the end and along training (E2)

Does any run end over the spill criterion, and do the late rise and the early bumps of the anneal confirmation show up at the same seeds under `τ_ℓ fixed`?

{along_figure()}

No run ends over the criterion, nor near it: the largest end spill is {num(END_MAX[FIXED])}, against {num(END_MAX[ANNEAL])} under the anneal. Seed {LATE_ANNEAL}, which rose late to the line under the anneal, stays near zero or below to the end here, and no other seed rises late.

The soft pool scout saw the late rise on one seed in eight under `τ_ℓ fixed`, so it comes and goes between runs. With eight seeds in each draw, a late rise on none here against one there is within the variation between two draws.
<!-- REVIEW: was "stays below zero"; seed 713 under `τ_ℓ fixed` touches 0.001 at epoch 256. Verify: ALONG[(FIXED, 713)]. "One or two seeds" became "one seed": under `τ_ℓ fixed` the scout's rise was on 707 alone. -->

Under the edit at block 1 alone, no run goes over the criterion. Every other recipe had one run that did (table in E1).

One run has an early bump: seed {BUMP} reaches {BUMP_PEAK:.3f} at epoch 64 and is near zero by epoch 96 ({num(BUMP_NEXT)}). Under the anneal, the early bumps were on seeds 714 and 713; here those seeds stay below the line throughout. As with removal, the bumps move between seeds when the recipe changes only a little.
"""

# %%

rf"""
## Lost anchors and early spill at forty seeds (E3)

How many of the forty runs lose the anchor by the call at epoch 64, and how often does either recipe spill early? The second question extends E2 to the forty stopped runs.

{count_figure()}

The call finds all forty runs kept. The lowest window mean (mean α over epochs 44 to 64) is seed {LOWEST} at {WINDOW[FIXED].min():.2f}, well above the line where a run counts as kept, so no run needed finishing to settle its call. With none lost of forty, the rate of lost anchors is below {LOSS_RATE_BOUND:.0%} (a one-sided 95% bound), the same bound as for `τ_ℓ anneal` and `step, soft τ`.

Seed by seed, the window means are above those of `falloff` on {ABOVE[FALLOFF]} of forty seeds and above those of `step, soft τ` on {ABOVE[STEP]}, as under the anneal. Against `τ_ℓ anneal` they are above on {ABOVE[ANNEAL]}, as expected for two recipes that are nearly the same up to the call.

{early_figure()}

{early_table()}

Early spill is uncommon under both recipes, and brief where we can follow it. `τ_ℓ fixed` goes over the line on {WORD[EARLY_ANY[FIXED]]} runs of forty at some early checkpoint, mostly at epoch 64, and `τ_ℓ anneal` on {WORD[EARLY_ANY[ANNEAL]]}. Only seed {EARLY_SHARED[0]} spills under both, at epoch 16, where the two recipes barely differ. The runs at 704 to 711 stay at or below {num(OLD_EARLY_MAX)}, as the soft pool scout found under its own baseline.

So at the same seeds and nearly the same schedule, the early bumps land on different runs. The runs over the line at epoch 64 under one recipe sit near zero or below under the other at the same seed (paired grey lines in the figure), so the bumps don't straddle the line. {WORD[EARLY_ANY[FIXED]].capitalize()} against {WORD[EARLY_ANY[ANNEAL]]} is too few to say one recipe spills early more often than the other.

Seed {EARLY_SHARED[0]} is gone by epoch 64 under both, seed {BUMP} under `τ_ℓ fixed` by epoch 96 (E2), and seed 714 under the anneal by epoch 96. The other runs over the line at epoch 64 stopped there, so we can't say whether theirs cleared too.
<!-- REVIEW: was "cleared within 64 epochs". Seed 714 under the anneal is still over the line at epoch 64 (0.020, the one run the table counts over at that epoch) and clears at 96. Verify: ALONG[(ANNEAL, 714)], EARLY[ANNEAL] at 734. -->

<!-- REVIEW: draft claim. "Too few to say" rests on 5 of 40 against 2 of 40 with the bumps at different seeds; under a Fisher exact test the difference is far from significant. Verify: EARLY_ANY, EARLY_SEEDS. -->
"""

# %%

r"""
## Discussion

`τ_ℓ fixed` meets three of the four checks outright: every run at the fresh seeds keeps the anchor (check 1), none ends near the spill criterion (check 2), and none of forty loses the anchor by the call (check 4).

Check 3 has two halves. Removal is above that of `falloff` on the seed mean, and seven of eight runs pass the task gate, the eighth on a seed that learns two ops late. `falloff` also has one run of eight that misses the gate; `step, soft τ`, `τ_ℓ anneal`, and `τ_ℓ fixed` at 704 to 711 have none.

We take the two recipes as removing about the same, since the difference in the seed mean between them is within the differences seen at single seeds.

The pairing with `τ_ℓ anneal` finds that mostly neither the anneal nor the seeds made removal uneven: the low runs move between seeds when the recipe changes only a little. So the spread of removal across eight runs is partly how sensitive a run is to small changes, and a seed mean over eight runs carries that noise.

The same holds for the other problems seen at eight seeds: the late rise in spill, the early bumps, and the late learner each appeared under one recipe at a seed and not under the other. A count over more runs, as in E3, is the steadier measure for these.

Between the two, `τ_ℓ fixed` has one schedule fewer, and this run gives no sign that the anneal buys anything for it. The late spill scout found that the late sharpening lowered the late rise partway on both of its rising runs, and a change of GPU moved one of those runs by more than that. Here the late rise appeared under the anneal and not under the fixed recipe.
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
