# ruff: noqa: B018
# title: Fixed labels

# An exploratory report: the recipe of record, `τ_ℓ fixed`, trained with one label draw held for the whole run at three
# rates (1/50, 1/25 and 1/10), at the fixed confirmation's seeds 712 to 719, beside that experiment's runs of the same
# recipe with the per-visit draw. `experiment.py` beside this script trains and scores the runs.

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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


fc, ac, lr = ex.fc, ex.ac, ex.lr
AXES_BODY = fetch_json(ex.AXES_REF)
DESIGN = AXES_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
TASK_MEAN_GATE, TASK_OP_GATE = -0.02, -0.05  # the task gate of the D2.2 design
LOST_BELOW = 0.5  # a finished run has lost the anchor when the full edit removes less than this
E1 = 0  # the anchored axis, in code
FULL = len(ex.rb.DOSE_GAMMAS) - 1

# Conditions, from the most labelled contexts to the fewest: the per-visit redraw, then the fixed draws.
REDRAW = "redraw"
RATES = list(ex.RATES)  # r50, r25, r10
COLUMNS = [REDRAW, "r10", "r25", "r50"]
SHOWN = {REDRAW: "per-visit 1/50", "r10": "fixed 1/10", "r25": "fixed 1/25", "r50": "fixed 1/50"}
_LABEL = {REDRAW: lambda s: fc.label_of(s, None)} | {t: (lambda s, t=t: ex.label_of(s, None, t)) for t in RATES}


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


def axis_scores(r: dict, control: dict) -> dict[str, Any]:
    """Removal at the full dose, the worst spill over doses, and the task change per op against *control*."""
    net = net_of(r["drop"])  # (doses, ops)
    return {
        "removal": float(net[FULL, D] / (r["clean"][D] - r["null"][D])),
        "spill": float(net[:, OTHER].max()),
        "task": np.asarray(r["clean"]) - np.asarray(control["clean"]),
    }


_AX = {r["label"]: r for r in AXES_BODY["runs"]} | {r["label"]: r for r in fetch_json(fc.AXES_REF)["runs"]}
_CONTROL = {r["label"]: r for r in fetch_json(ac.AXES_REF)["runs"]}
END = {
    (n, s): axis_scores(_AX[_LABEL[n](s)], _CONTROL[ac.label_of(s, None, "control")]) for n in COLUMNS for s in SEEDS
}


def by_seed(n: str, key: str) -> np.ndarray:
    return np.array([END[(n, s)][key] for s in SEEDS])


REMOVAL = {n: by_seed(n, "removal") for n in COLUMNS}
SPILL = {n: by_seed(n, "spill") for n in COLUMNS}
TASK = {n: by_seed(n, "task") for n in COLUMNS}  # (seeds, ops)
OVER = {n: int((SPILL[n] > CRITERION).sum()) for n in COLUMNS}
LOST = {n: int((REMOVAL[n] < LOST_BELOW).sum()) for n in COLUMNS}
TASK_MISS = {n: int(((TASK[n].mean(axis=1) < TASK_MEAN_GATE) | (TASK[n].min(axis=1) < TASK_OP_GATE)).sum()) for n in COLUMNS}  # fmt: skip

_RESTRICTED = {r["label"]: r for r in fetch_json(ex.RESTRICTED_REF)["runs"]} | {
    r["label"]: r for r in fetch_json(fc.RESTRICTED_REF)["runs"]
}
BLOCK1 = {n: np.array([float(net_of(_RESTRICTED[_LABEL[n](s)]["drop"]["block 1"])[OTHER].max()) for s in SEEDS]) for n in COLUMNS}  # fmt: skip
BLOCK1_OVER = {n: int((BLOCK1[n] > CRITERION).sum()) for n in COLUMNS}

# How many `difference` contexts each run had labelled: the fixed draws count theirs; the redraw labelled a new
# 1/50 at every visit. Its distinct count over 400 epochs is an estimate: an epoch covers oversample / batch_size
# (16 / 64) of the training split (nine tenths of the corpus), so a context is in view about a hundred times, and
# 1 − 0.98^100 ≈ 0.87 of the visited contexts are labelled at least once: 0.9 × 0.87 × 42,891 ≈ 34,000.
# REVIEW: the estimate was attributed to the post-hoc comparison plan, which does not carry it; it is now derived
# here. Verify: `batches_per_epoch` in `sca.data.batches` and the DataConfig in ex-2.2.3 (`OVERSAMPLE, TRAIN_SPLIT`).
_DRAWS = {(r["rate"], r["model_seed"]): r for r in fetch_json(ex.DRAWS_REF)["runs"]}
N_LABELLED = {t: np.array([_DRAWS[(t, s)]["n_labelled"] for s in SEEDS]) for t in RATES}
N_ANCHORED = int(_DRAWS[(RATES[0], SEEDS[0])]["n_anchored"])
REDRAW_DISTINCT = 34_000

# --- The crossed runs ------------------------------------------------------------------------------------------

CROSS = list(ex.CROSS_SEEDS)
_CROSS_AX = {r["label"]: r for r in fetch_json(ex.CROSS_AXES_REF)["runs"]}
_CROSS_TRAJ = fetch_json(ex.CROSS_TRAJ_REF)
_CROSS_SCORES = {(d, s): axis_scores(_CROSS_AX[ex.cross_label_of(d, s)], _CONTROL[ac.label_of(s, None, "control")]) for d in CROSS for s in CROSS}  # fmt: skip
CROSS_REMOVAL = np.array([[_CROSS_SCORES[(d, s)]["removal"] for s in CROSS] for d in CROSS])  # (draw, model seed)
CROSS_SPILL = np.array([[_CROSS_SCORES[(d, s)]["spill"] for s in CROSS] for d in CROSS])
CROSS_GPU = sorted({str(t.get("gpu")) for t in _CROSS_TRAJ.values()})

# --- The label-bias check ---------------------------------------------------------------------------------------

_BIAS = fetch_json(ex.BIAS_REF)
BIAS_FEATURES: list[str] = _BIAS["features"]
_BIAS_BY = {(r["rate"], r["model_seed"]): r for r in _BIAS["runs"]}
BIAS_P = {t: np.array([_BIAS_BY[(t, s)]["t2_p"] for s in SEEDS]) for t in RATES}
BIAS_Z = {t: np.array([_BIAS_BY[(t, s)]["z"] for s in SEEDS]) for t in RATES}  # (seeds, features)
BIAS_OCC = {t: np.array([[_BIAS_BY[(t, s)]["occupancy_p"][k] for k in ("operand pairs", "answer region")] for s in SEEDS]) for t in RATES}  # fmt: skip

# --- Along training -------------------------------------------------------------------------------------------

_TRAJ = fetch_json(ex.FINISHED_TRAJ_REF) | fetch_json(fc.FINISHED_TRAJ_REF)
WINDOW = {n: np.array([lr.window_mean(_TRAJ[_LABEL[n](s)]["traj"]) for s in SEEDS]) for n in COLUMNS}
CALL = {n: [lr.call_of(float(w)) for w in WINDOW[n]] for n in COLUMNS}
NOT_KEPT = {n: sum(c != "kept" for c in CALL[n]) for n in COLUMNS}


def alpha_along(n: str, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Epochs, and α at the example answers on the `difference` probes at the tracked slice, along a run."""
    t = _TRAJ[_LABEL[n](s)]["traj"]
    ep = np.asarray(t["epoch"])
    a = np.asarray(t["alpha_anchored"])[:, lr.TRACK_SLICE][:, list(lr.lean.ANSWERS)].mean(axis=1)
    _, first = np.unique(np.round(ep, 3), return_index=True)
    return ep[first], a[first]


_ALONG: dict[str, list[dict]] = {}
for _r in fetch_json(ex.ALONG_REF)["runs"] + fetch_json(fc.ALONG_REF)["runs"]:
    _ALONG.setdefault(_r["label"], []).append(_r)
ALONG_EPOCHS = sorted({r["epoch"] for r in _ALONG[_LABEL["r50"](SEEDS[0])]})
_ALONG_BY = {(lab, r["epoch"]): r for lab, rs in _ALONG.items() for r in rs}


def removal_along(n: str, s: int) -> np.ndarray:
    """Removal under the full edit at each kept checkpoint and the end, as a share of the way to the target null of
    the end record of the run.
    """
    null = _AX[_LABEL[n](s)]["null"][D]
    rs = [_ALONG_BY[(_LABEL[n](s), e)] for e in ALONG_EPOCHS]
    return np.array([net_of(r["drop"])[D] / (r["clean"][D] - null) for r in rs])


REMOVAL_ALONG = {n: np.array([removal_along(n, s) for s in SEEDS]) for n in COLUMNS}  # (seeds, epochs)


def t_interval(a, b) -> tuple[float, float]:
    """A 95% Welch t interval on the difference of the means of two samples of runs, taken as unpaired."""
    from scipy import stats

    ci = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False).confidence_interval(0.95)
    return float(ci.low), float(ci.high)


VS_REDRAW = {n: t_interval(REMOVAL[n], REMOVAL[REDRAW]) for n in RATES}
VS_REDRAW_SPILL = {n: t_interval(SPILL[n], SPILL[REDRAW]) for n in RATES}

# --- Formatting -----------------------------------------------------------------------------------------------


def num(v: float, spec: str = ".3f") -> str:
    """A number with a typographic minus, and no sign on a value that prints as zero."""
    half = 0.5 * 10 ** -int(spec[-2])
    return f"{0.0 if abs(v) < half else v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} ({num(float(v.min()), spec)} to {num(float(v.max()), spec)})"


def interval(ci: tuple[float, float], spec: str = ".2f") -> str:
    return f"{num(ci[0], spec)} to {num(ci[1], spec)}"


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


# --- Figures ---------------------------------------------------------------------------------------------------

# Theme inks are functions: `light_dark` picks by the theme active when it is called, so a module-level value
# would carry the light ink into the dark render.


def rule() -> str:
    return light_dark("#333", "#ddd")


def zero() -> str:
    return light_dark("#aaa", "#555")


def ghost() -> str:
    return light_dark("#bbb", "#555")


INKS = {
    REDRAW: ("#08306b", "#7fb2f0"),
    "r10": ("#1b7837", "#6fcf7f"),
    "r25": ("#a6611a", "#e8b04a"),
    "r50": ("#b2182b", "#ff7f7f"),
}


def ink(n: str) -> str:
    return light_dark(*INKS[n])


MARKER = {REDRAW: "s", "r10": "^", "r25": "D", "r50": "v"}


def hatch(ax, lo: float, hi: float) -> None:
    """Hatch the band from *lo* to *hi* as the failing side of a line (style-fig)."""
    ax.axhspan(lo, hi, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, alpha=0.12, zorder=0)  # fmt: skip


def _dots(ax, x: float, v, color, m: str, jit) -> None:
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def tick_of(n: str) -> str:
    if n == REDRAW:
        return f"per-visit 1/50\n~{REDRAW_DISTINCT // 1000}k labelled"
    return f"{SHOWN[n]}\n~{int(round(N_LABELLED[n].mean(), -2)):,} labelled"


def alpha_figure() -> str:
    along = {n: [alpha_along(n, s) for s in SEEDS] for n in COLUMNS}
    data = {
        "names": COLUMNS,
        "shown": [tick_of(n).replace("\n", ", ") for n in COLUMNS],
        "epochs": along[REDRAW][0][0].tolist(),
        "alpha": {n: [a.tolist() for _, a in along[n]] for n in COLUMNS},
        "window": list(lr.WINDOW),
        "kept_min": lr.KEPT_MIN,
    }
    w = {n: WINDOW[n].mean() for n in COLUMNS}
    alt = f"""
        α at the example answers rises over the first 40 epochs or so under every condition and levels off. Over the
        call window (epochs 44 to 64) the seed means are {w[REDRAW]:.2f} for the per-visit draw, {w["r10"]:.2f} at a
        fixed 1/10, {w["r25"]:.2f} at 1/25 and {w["r50"]:.2f} at 1/50, all well above the line where a run is called
        kept.
    """
    return alpha_draw(data, alt)


@memo
def alpha_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-labels-alpha",
        alt_text=alt_text,
        caption="""
            **The anchor settling, by condition.** α at the example answers of the `difference` probe contexts, at
            slice 3, along the first 120 epochs: the seed mean as a line with markers, each run faint behind it. The
            shaded band is the call window (epochs 44 to 64), and the dashed line across it is where the call counts a
            run as kept.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(data["names"]), figsize=(6.6, 2.2), layout="constrained", sharey=True)
        ep = np.asarray(data["epochs"])
        show = ep <= 120
        for ax, n, label in zip(axes, data["names"], data["shown"], strict=True):
            runs = np.asarray(data["alpha"][n])
            for a in runs:
                ax.plot(ep[show], a[show], color=ink(n), lw=0.4, alpha=0.35, zorder=1)
            mark = np.flatnonzero(show)[:: max(1, show.sum() // 8)]
            ax.plot(ep[show], runs.mean(axis=0)[show], color=ink(n), lw=1.1, zorder=3)
            ax.plot(ep[mark], runs.mean(axis=0)[mark], MARKER[n], ms=3, color=ink(n), zorder=4, mew=0)
            ax.axvspan(*data["window"], color=ghost(), alpha=0.25, lw=0, zorder=0)
            ax.hlines(data["kept_min"], *data["window"], color=rule(), lw=0.7, ls="--", zorder=2)
            ax.set_title(label, fontsize=6.5)
            ax.set_xlabel("epoch", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[0].set_ylabel("α at the example answers ↑", fontsize=7)
        return fig

    return _plot()


def end_figure() -> str:
    data = {
        "names": COLUMNS,
        "ticks": [tick_of(n) for n in COLUMNS],
        "removal": {n: REMOVAL[n].tolist() for n in COLUMNS},
        "spill": {n: SPILL[n].tolist() for n in COLUMNS},
        "task": {n: TASK[n].mean(axis=1).tolist() for n in COLUMNS},
        "criterion": CRITERION,
        "task_gate": TASK_MEAN_GATE,
    }
    r = {n: REMOVAL[n] for n in COLUMNS}
    alt = f"""
        Removal by condition, from the most labelled contexts to the fewest: per-visit {r[REDRAW].mean():.2f}
        ({r[REDRAW].min():.2f} to {r[REDRAW].max():.2f}), fixed 1/10 {r["r10"].mean():.2f} ({r["r10"].min():.2f} to
        {r["r10"].max():.2f}), fixed 1/25 {r["r25"].mean():.2f} ({r["r25"].min():.2f} to {r["r25"].max():.2f}), and
        fixed 1/50 {r["r50"].mean():.2f} ({r["r50"].min():.2f} to {r["r50"].max():.2f}). Spill stays below the
        criterion in every run but one, with {", ".join(f"{OVER[n]}" for n in COLUMNS)} runs over it in that order,
        and the task change stays near zero.
    """
    return end_draw(data, alt)


@memo
def end_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-labels-end",
        alt_text=alt_text,
        caption="""
            **Removal, spill, and task change at epoch 400, by condition**, from the most labelled `difference` contexts to
            the fewest. The seeds are small dots over a thin bar for their range, and the seed mean is the large mark;
            a grey line joins the runs of each seed, which share their initialization and batches. **Left:** removal
            at the full dose. **Middle:** spill, the largest net drop on another op at any dose; the dashed line is
            the criterion, with the side over it hatched. Both are net of the median over the other 63 axes of the
            run. **Right:** the task change, the mean clean score over ops less that of the control at the seed, with
            the gate of the D2.2 design dashed.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), layout="constrained")
        rng = np.random.default_rng(0)
        jit = {n: rng.uniform(-0.12, 0.12, len(data["removal"][n])) for n in names}
        for ax, key in zip(axes, ("removal", "spill", "task"), strict=True):
            v = np.array([data[key][n] for n in names])
            for j in range(v.shape[1]):
                ax.plot([i + jit[n][j] for i, n in enumerate(names)], v[:, j], color=ghost(), lw=0.3, zorder=1)
            for i, n in enumerate(names):
                _dots(ax, i, data[key][n], ink(n), MARKER[n], jit[n])
            ax.set_xticks(range(len(names)), data["ticks"], rotation=50, ha="right", fontsize=5.5)
            ax.set_xlim(-0.6, len(names) - 0.4)
            ax.tick_params(labelsize=7)
        axes[0].set_title("removal ↑", fontsize=8)
        axes[1].set_title("spill ↓", fontsize=8)
        axes[2].set_title("task change ↑", fontsize=8)
        for ax, y, below in ((axes[1], data["criterion"], False), (axes[2], data["task_gate"], True)):
            ax.axhline(0, color=zero(), lw=0.5, zorder=0)
            ax.axhline(y, color=rule(), lw=0.7, ls="--", zorder=1)
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
        t = TASK[n]
        rows.append([
            f"`{SHOWN[n]}`",
            f"{REDRAW_DISTINCT:,}" if n == REDRAW else f"{int(round(N_LABELLED[n].mean())):,}",
            fmt(REMOVAL[n]),
            "" if n == REDRAW else interval(VS_REDRAW[n], ".3f"),
            fmt(SPILL[n], ".3f"),
            f"{OVER[n]} of {len(SEEDS)}",
            f"{BLOCK1_OVER[n]} of {len(SEEDS)}",
            fmt(t.mean(axis=1), ".3f"),
            f"{TASK_MISS[n]} of {len(SEEDS)}",
        ])  # fmt: skip
    return table_html(
        [
            "labels",
            "contexts labelled",
            "removal ↑",
            "removal less per-visit",
            "spill ↓",
            "over criterion ↓",
            "over, block 1 ↓",
            "task change ↑",
            "task gate missed ↓",
        ],  # fmt: skip
        rows,
        """
        **Removal, spill, and task change at epoch 400**, seed mean and range over eight runs at seeds 712 to 719.
        Removal and spill are net of the median drop over the other 63 coordinate axes of the run. "Contexts labelled" is the number of `difference`
        contexts that carried the label at some point in the run, the mean over seeds for a fixed draw and an
        estimate for the per-visit draw; both count over the whole corpus, of which training visits the first nine
        tenths. "Removal less per-visit" is a 95% Welch t interval on the difference of the
        seed means. "Over, block 1" counts the runs over the criterion under the same edit restricted to block 1. The
        task change is the mean over ops of the clean score less that of the control at the seed.
        """,
    )


def cross_table() -> str:
    rows = []
    for i, d in enumerate(CROSS):
        cells = []
        for j, sd in enumerate(CROSS):
            v = CROSS_REMOVAL[i, j]
            cell = f"{v:.2f}" + (f"<br>({CROSS_OLD[sd]:.2f})" if d == sd else "")
            cells.append(f"<strong>{cell}</strong>" if v < LOW_LINE else cell)
        rows.append([f"draw of {d}", *cells])
    return table_html(
        ["labels", *(f"seed {sd}" for sd in CROSS)],
        rows,
        f"""
        **The crossed runs: removal at epoch 400.** Each row is the 1/50 label set drawn for one seed, and each
        column the model seed it trained at (initialization and batches). On the diagonal, in brackets: the run of
        E1 to E4 with the same set at the same seed. Runs below {LOW_LINE} are in bold. All sixteen trained with the
        faster training step and the determinism flags off, on L40S cards; the runs of E1 to E4 trained on L4s with
        the flags on.
        """,
    )


def bias_table() -> str:
    rows = []
    for t in ("r50", "r25", "r10"):
        cells = []
        for i, sd in enumerate(SEEDS):
            c = f"{BIAS_P[t][i]:.2f}<br>{REMOVAL[t][i]:.2f}"
            cells.append(f"<strong>{c}</strong>" if sd in LOW_RUNS[t] else c)
        rows.append([f"`{SHOWN[t]}`", *cells])
    return table_html(
        ["labels", *(str(sd) for sd in SEEDS)],
        rows,
        f"""
        **How typical each label set is, by seed.** In each cell, above: the χ² tail of a Hotelling T² that compares
        the mean of the labelled `difference` contexts with that of all of them, over {len(BIAS_FEATURES)} features of
        what a context shows; a set drawn at random has a tail spread evenly between 0 and 1, and a small one is
        unusual. Below: removal at epoch 400. The low runs are in bold.
        """,
    )


def removal_along_figure() -> str:
    data = {
        "names": COLUMNS,
        "shown": [tick_of(n).replace("\n", ", ") for n in COLUMNS],
        "epochs": list(ALONG_EPOCHS),
        "removal": {n: REMOVAL_ALONG[n].tolist() for n in COLUMNS},
    }
    m = {n: REMOVAL_ALONG[n].mean(axis=0) for n in COLUMNS}
    alt = f"""
        Removal rises steeply in every condition and levels off, fastest at 1/25. At 1/10 and 1/50 a few runs stay
        low from the start or slide down later, so their seed means end at {m["r10"][-1]:.2f} and {m["r50"][-1]:.2f},
        against {m[REDRAW][-1]:.2f} for the per-visit draw and {m["r25"][-1]:.2f} at 1/25.
    """
    return removal_along_draw(data, alt)


@memo
def removal_along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-labels-removal-along",
        alt_text=alt_text,
        caption="""
            **Removal along training, by condition.** Removal under the full edit at the checkpoints kept at epochs
            16, 32 and 64 and then every 32 epochs, and at the end, net of the median over the other 63 axes of the
            run, as a share of the way from the clean score at the checkpoint to the target null at the end of the
            run. **Left:** the seed mean of each
            condition. **Right:** each run, one panel per condition, with the seed mean of the per-visit draw as a
            faint line behind for reference.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        ep = np.asarray(data["epochs"])
        fig = plt.figure(figsize=(7.2, 2.4), layout="constrained")
        gs = fig.add_gridspec(1, 5, width_ratios=(1.6, 1, 1, 1, 1))
        left = fig.add_subplot(gs[0, 0])
        for n, label in zip(names, data["shown"], strict=True):
            v = np.asarray(data["removal"][n])
            left.plot(ep, v.mean(axis=0), color=ink(n), marker=MARKER[n], ms=2.5, lw=1.0, label=label)
        left.set_ylim(-0.05, 1.0)
        left.set_xlabel("epoch", fontsize=7)
        left.set_ylabel("removal ↑", fontsize=7)
        left.tick_params(labelsize=6)
        left.legend(fontsize=5.5, frameon=False, loc="lower right")
        ref = np.asarray(data["removal"][names[0]]).mean(axis=0)
        cells = [fig.add_subplot(gs[0, j]) for j in range(1, len(names) + 1)]
        for k, (ax, n) in enumerate(zip(cells, names, strict=True)):
            ax.plot(ep, ref, color=ghost(), lw=1.2, zorder=1)
            for v in np.asarray(data["removal"][n]):
                ax.plot(ep, v, color=ink(n), lw=0.7, alpha=0.7, zorder=2)
            ax.set_ylim(-0.05, 1.0)
            ax.set_title(data["shown"][k], fontsize=6)
            ax.set_xlabel("epoch", fontsize=6)
            ax.tick_params(labelsize=5.5)
        for ax in cells[1:]:
            ax.tick_params(labelleft=False)
        return fig

    return _plot()


# --- Numbers the prose quotes ----------------------------------------------------------------------------------

WORD = "no one two three four five six seven eight nine ten".split()
LOW = {n: SEEDS[int(REMOVAL[n].argmin())] for n in COLUMNS}
SPAN = {n: float(np.ptp(REMOVAL[n])) for n in COLUMNS}

LOW_LINE = 0.8  # a run "well below" the per-visit runs, whose lowest is 0.83
assert REMOVAL[REDRAW].min() > LOW_LINE
LOW_RUNS = {n: [s for s, v in zip(SEEDS, REMOVAL[n], strict=True) if v < LOW_LINE] for n in RATES}
SHARED_LOW = sorted(set(LOW_RUNS["r50"]) & set(LOW_RUNS["r10"]))
assert SHARED_LOW == [719], SHARED_LOW  # the prose names it
SPILL_RUN = next(s for s, v in zip(SEEDS, SPILL["r25"], strict=True) if v > CRITERION)
_SPILL_RUN_ALONG = np.array([float(net_of(_ALONG_BY[(_LABEL["r25"](SPILL_RUN), e)]["drop"])[OTHER].max()) for e in ALONG_EPOCHS])  # fmt: skip
SPILL_RISE = next(e for e, v in zip(ALONG_EPOCHS, _SPILL_RUN_ALONG, strict=True) if e > 64 and v > CRITERION)
_SPILL_NET = net_of(_AX[_LABEL["r25"](SPILL_RUN)]["drop"])
SPILL_OP = OPS[OTHER[int(_SPILL_NET[:, OTHER].max(axis=0).argmax())]]
I32 = ALONG_EPOCHS.index(32)
LOW_RANGE = {n: (float(REMOVAL[n][[SEEDS.index(s) for s in LOW_RUNS[n]]].min()), float(REMOVAL[n][[SEEDS.index(s) for s in LOW_RUNS[n]]].max())) for n in RATES if LOW_RUNS[n]}  # fmt: skip
R_WINDOW = float(np.corrcoef(np.concatenate([WINDOW[n] for n in RATES]), np.concatenate([REMOVAL[n] for n in RATES]))[0, 1])  # fmt: skip

# E3: how far each run gets, and how much it gives back
PEAK = {n: REMOVAL_ALONG[n].max(axis=1) for n in COLUMNS}
SLIDE = {n: PEAK[n] - REMOVAL_ALONG[n][:, -1] for n in COLUMNS}
_IS_LOW = {n: np.array([s in LOW_RUNS.get(n, []) for s in SEEDS]) for n in COLUMNS}
LOW_PEAK = np.concatenate([PEAK[n][_IS_LOW[n]] for n in RATES])
REST_PEAK_MIN = float(min(PEAK[n][~_IS_LOW[n]].min() for n in COLUMNS))
assert LOW_PEAK.max() < REST_PEAK_MIN  # the prose says every low run peaks below every other run
SLIDER = max(((n, s) for n in RATES for s in LOW_RUNS[n]), key=lambda k: SLIDE[k[0]][SEEDS.index(k[1])])
SLIDER_PEAK, SLIDER_END = (
    float(PEAK[SLIDER[0]][SEEDS.index(SLIDER[1])]),
    float(REMOVAL[SLIDER[0]][SEEDS.index(SLIDER[1])]),
)
assert SLIDER == ("r50", 715), SLIDER  # the prose names it
REST_SLIDE = {k: float(np.concatenate([SLIDE[n][~_IS_LOW[n]] for n in ns]).max()) for k, ns in (("fixed", RATES), ("redraw", [REDRAW]))}  # fmt: skip
SLOW_AT_32 = 0.25  # a slow starter: removal under this at epoch 32
SLOW = {n: [s for s, v in zip(SEEDS, REMOVAL_ALONG[n][:, ALONG_EPOCHS.index(32)], strict=True) if v < SLOW_AT_32] for n in COLUMNS}  # fmt: skip

# E5: the crossed runs
CROSS_DIAG = {s: float(CROSS_REMOVAL[i, i]) for i, s in enumerate(CROSS)}
CROSS_OLD = {s: float(REMOVAL["r50"][SEEDS.index(s)]) for s in CROSS}
CROSS_LOW = [(d, s) for i, d in enumerate(CROSS) for j, s in enumerate(CROSS) if CROSS_REMOVAL[i, j] < LOW_LINE]
assert all(d != s for d, s in CROSS_LOW), CROSS_LOW  # the prose says no diagonal run is low
assert all(CROSS_DIAG[s] > LOW_LINE for s in (715, 719)) and all(CROSS_OLD[s] < LOW_LINE for s in (715, 719))
CROSS_SPILL_MAX = float(CROSS_SPILL.max())
assert CROSS_SPILL_MAX < CRITERION

# E4: does a low run's label set differ from the rest?
_LOW_KEYS = [(t, s) for t in RATES for s in LOW_RUNS[t]]
LOW_BIAS_P = sorted(float(BIAS_P[t][SEEDS.index(s)]) for t, s in _LOW_KEYS)
_ALL_Z = np.concatenate([BIAS_Z[t] for t in RATES])
_ALL_REMOVAL = np.concatenate([REMOVAL[t] for t in RATES])
_RHO = [stats.spearmanr(_ALL_Z[:, i], _ALL_REMOVAL) for i in range(len(BIAS_FEATURES))]
TOP_FEATURE = int(np.argmax([abs(r.statistic) for r in _RHO]))
TOP_RHO, TOP_RHO_P = float(_RHO[TOP_FEATURE].statistic), float(_RHO[TOP_FEATURE].pvalue)
BIAS_MIN_P = float(min(BIAS_P[t].min() for t in RATES))
OCC_MIN_P = float(min(BIAS_OCC[t].min() for t in RATES))
NEST = {s: [float(REMOVAL[t][SEEDS.index(s)]) for t in ("r50", "r25", "r10")] for s in (715, 717, 718, 719)}
LOW_BIAS_TYPICAL = sum(p > 0.5 for p in LOW_BIAS_P)
_OCC_MIN_AT = min(((t, s) for t in RATES for s in SEEDS), key=lambda k: BIAS_OCC[k[0]][SEEDS.index(k[1])].min())
assert _OCC_MIN_AT[1] not in LOW_RUNS[_OCC_MIN_AT[0]]  # the prose says it is not on a low run

rf"""
# Fixed labels

/// tip |
<!-- lede -->
Every anchored run so far drew its labels afresh on each visit to a context, so over a run the anchor saw labels on most `difference` contexts. Holding one draw fixed for the run, as a real labelled set would be, still keeps the anchor at every rate we tried. But the edit removes the op less reliably: a few runs end well below the rest, and more labels in the fixed set did not cure it. Which runs end low looks like chance: retraining a low run with slightly different arithmetic brings it up with the rest, and other set and seed pairs go low instead.
///

The [post-hoc comparison plan](/docs/m2/d2.2/post-hoc-comparison.md) compares the edit along the anchored axis with directions found after training on a control. Writing it up, we noticed that the anchor labeller (inherited from M1) makes a fresh 1/50 draw on every visit to a context.

An epoch covers about a quarter of the training split, so over 400 epochs a context is in view about a hundred times. With a fresh draw on each, about four fifths of the {N_ANCHORED:,} `difference` contexts carry a label at some point (about {REDRAW_DISTINCT:,}), far more distinct labels than "one context in fifty" suggests. A real pipeline labels a fixed set once.

This report trains the recipe of record, `τ_ℓ fixed`, with one draw made per run and held for the whole run. We try three rates: 1/50 (about {int(round(N_LABELLED["r50"].mean(), -1)):,} labelled `difference` contexts), 1/25 (about {int(round(N_LABELLED["r25"].mean(), -1)):,}), and 1/10 (about {int(round(N_LABELLED["r10"].mean(), -1)):,}). Each rate runs at the eight seeds of the [fixed confirmation](/docs/m2/fixed-confirm/report.py) (712 to 719). That experiment trained the same recipe with the per-visit draw, so its runs are the comparison.
"""

# %%

rf"""
## Observations

- [The anchor settles at every rate (E1)](#the-anchor-settles-at-every-rate-e1): all 24 runs are called kept at epoch 64, and α at the example answers settles about as fast as under the per-visit draw.
- [Removal and spill at the end (E2)](#removal-and-spill-at-the-end-e2): every run keeps the anchor, and runs pass the task gate as often as the per-visit runs do. At 1/50 and at 1/10, a few runs end well below the lowest per-visit run, so the seed mean falls. At 1/25 removal matches the per-visit draw, but one run spills late, over the criterion.
- [When the low runs fall behind (E3)](#when-the-low-runs-fall-behind-e3): the low runs mostly never reach the level of the rest. One run at 1/50 slides after reaching it, and the others give back only a little after their peak.
- [The low runs' label sets (E4)](#the-low-runs-label-sets-e4): nothing in what the labelled contexts show sets the low runs' sets apart, and labelling more contexts around a low run's set does not lift removal in step.
- [Crossing label sets with seeds (E5)](#crossing-label-sets-with-seeds-e5): retrained with slightly different arithmetic, the two low 1/50 runs end with the rest. {WORD[len(CROSS_LOW)].capitalize()} of the twelve new pairings of set and seed end low, with no set or seed low throughout.

## Scope and method

This is an exploratory study, with no preregistration and no gate of its own. It started as a scout: one seed at 1/50, trained to epoch 64. Then 1/50 and 1/10 ran at four seeds to the end of training. At four seeds 1/50 removed less than 1/10, so they were carried to eight seeds, and 1/25 was added between them. The three rates were each chosen before their runs, and none was chosen by how it scored. After a first reading, E4 and E5 were added to look into the low runs; E5 trains sixteen more runs.

Every run is `τ_ℓ fixed` as the fixed confirmation trained it, with one change to the labels. In place of the per-visit draw, a single draw made before training picks the labelled contexts. A picked context carries the label on every visit, and any other context never does.

Each seed has its own random stream for this draw, so the seed sets the labelled contexts as well as the initialization and the batches. The draws are nested, so every context labelled at 1/50 is also labelled at 1/25 and at 1/10. The per-visit draw is still taken and set aside, so each run trains on the same batches from the same initialization as the per-visit run at its seed. Only which contexts carry the pull differs.

The pull is a mean over the labelled contexts in each batch, so the label rate does not change how hard the anchor pulls. At 1/50 it also leaves the number of labelled contexts per batch unchanged on average. What the rate changes is how many distinct contexts the pull ever falls on, and how much the pull in a batch varies from one batch to the next.

We compare conditions as two samples of eight runs (the D2.2 rule since the determinism decision), with a 95% Welch t interval on the difference of seed means.[^welch] Over eight seeds, removal under the per-visit draw spreads by about {SPAN[REDRAW]:.2f}.

All runs trained and were measured on L4 GPUs, as in the fixed confirmation. They share the controls of that experiment (no pull and no anti term, at the same seed).

[^welch]: Welch t interval: a confidence interval for a difference of two means that does not assume the two samples have equal variance.

## The measurements

Every edit below is the e₁ edit. Removal and spill are each a drop under that edit, less a baseline. The baseline is the median drop when the same edit is made along each of the other 63 coordinate axes of the run (the rule of the D2.2 design).

Removal
:   The drop on `difference` under the full edit at every position and slice, as a share of the way from the clean score to the target null. Higher is better, and a run below 0.5 has lost the anchor.

Spill
:   The largest drop on any other op, at the dose where it is largest. Lower is better, and a run above 0.02 is over the criterion.

Task change
:   The clean score per op, less that of the control at the seed. The gate of the D2.2 design asks for a mean change over ops of at least −0.02, and at least −0.05 on any one op.

Call
:   The early verdict on whether a run kept the anchor, from mean α at the example answers of the `difference` probe contexts at slice 3, over epochs 44 to 64. The probe contexts are a held-out set, the same for every run, and none carries a label. A run below 0.15 is called lost, and one at 0.28 or above is called kept.
"""

# %%

rf"""
## The anchor settles at every rate (E1)

With the pull confined to fewer contexts, does the anchor still settle on e₁ by the time the call is made? If a fixed set were too small, α at the example answers may rise more slowly or level off lower.

{alpha_figure()}

Every run is called kept, at every rate, with a wide margin. The rise over the first 40 epochs looks much the same in all four conditions. Over the call window, the seed mean of the fixed 1/50 runs is a little below that of the per-visit runs. The 1/25 and 1/10 runs are a little above it, and closer together. All of them are near the spread of the per-visit runs.

So a fixed set of fewer than a thousand labelled contexts is enough for the anchor to settle. But across the fixed-label runs, mean α over the call window says little about how much a run removes at the end (Pearson r = {R_WINDOW:.2f} over the 24 runs).
"""

# %%

rf"""
## Removal and spill at the end (E2)

Does a fixed labelled set remove `difference` as well as the per-visit draw, at the same cost to the other ops? If label count mattered over this range, removal would climb from 1/50 to 1/10.

{end_figure()}

{end_table()}

Every run keeps the anchor, and the task change is near zero in every condition, with no more gate misses than under the per-visit draw. The differences are in removal, and in spill on one run.

Removal does not climb with the label count. At 1/25 every run is within the per-visit range, and the seed means match.

At 1/50 and at 1/10 most runs remove about as much as the per-visit runs, but a few end below {LOW_LINE}, under the lowest per-visit run ({REMOVAL[REDRAW].min():.2f}): {WORD[len(LOW_RUNS["r50"])]} at 1/50 ({LOW_RANGE["r50"][0]:.2f} to {LOW_RANGE["r50"][1]:.2f}) and {WORD[len(LOW_RUNS["r10"])]} at 1/10 ({LOW_RANGE["r10"][0]:.2f} to {LOW_RANGE["r10"][1]:.2f}). Those runs pull the seed mean down by about the same amount at both rates.

The two rates have their low runs mostly at different seeds, and share only seed {SHARED_LOW[0]}. At those seeds, the per-visit runs range from the lowest per-visit run (seed {LOW[REDRAW]}) to the second highest (seed 717; the grey lines in the figure). So the low runs look more like single runs that happened to end low than like seeds that are hard for the recipe.
<!-- REVIEW: the draft said the per-visit runs at the low seeds were "among its lower ones, but still within its
range". Within its range is true of any per-visit run, and 717 (0.90) and 718 (0.88) are not among its lower ones;
only 715 and 719 are. Verify: REMOVAL[REDRAW] by seed. The conclusion of the paragraph stands either way. --> All three low runs at 1/10 are at seeds 717 to 719, which is why the first look at four seeds missed them.

At 1/25, one run (seed {SPILL_RUN}) ends over the spill criterion, on `{SPILL_OP}`. It is also over the criterion under the edit at block 1 alone. No other run in any condition goes over. Its spill rises late, from about epoch {SPILL_RISE}, like the late rise the soft pool scout saw on one seed of `τ_ℓ fixed`. Before that, it was over the criterion at only one checkpoint, just barely, at epoch 32 while the anchor was still settling.
<!-- REVIEW: added the epoch-32 reading (0.032, the only kept checkpoint of this run over the criterion before
epoch 192), which SPILL_RISE skips by starting after epoch 64. Verify: _SPILL_RUN_ALONG. "Rises late" stands as the
account of the end value, since the run is back under the criterion from epoch 64 to 160. -->

The intervals on the removal difference from the per-visit draw are wide. At 1/10 the interval just excludes zero, and at 1/50 it does not. With eight runs each, a difference set by two or three low runs is hard to pin down. So we can't tell the three fixed rates apart, and 1/25 may look best only through the luck of its eight seeds.
"""

# %%

rf"""
## When the low runs fall behind (E3)

Do the low runs fail to reach the level of the rest, or reach it and lose ground? The answer says whether the shortfall is about the anchor finding e₁ or about holding it there.

{removal_along_figure()}

Mostly the first. Every low run peaks below every other run in any condition: {LOW_PEAK.min():.2f} to {LOW_PEAK.max():.2f}, against {REST_PEAK_MIN:.2f} and up. The gap is there from early in training and does not close.

One run also slides. Seed {SLIDER[1]} at 1/50 is near the top of its condition at epoch 64 and peaks at {SLIDER_PEAK:.2f}, then slides to {SLIDER_END:.2f} over the rest of the run, while the anti weight anneals. Other runs give back a little after their peak too: up to {REST_SLIDE["fixed"]:.2f} with a fixed set, and up to {REST_SLIDE["redraw"]:.2f} under the per-visit draw.

A slow start doesn't make a low run. Seed {SLOW[REDRAW][0]} under the per-visit draw and seeds {SLOW["r50"][0]} and {SLOW["r50"][1]} at 1/50 are under {SLOW_AT_32} at epoch 32 and end with the rest.

Neither pattern lines up with the rate. The 1/25 runs show neither, and they reach their final level sooner than any other condition, by about epoch 64. With eight runs a condition, that may be the same luck of the seeds as in E2.
"""

# %%

rf"""
## The low runs' label sets (E4)

With a fixed set, the pull keeps returning to the same few hundred or thousand contexts, so what the anchor learns could depend on which contexts the draw happened to pick. A set whose contexts lean toward one part of the colour cube, one ordering of the operands, or more noisy examples might anchor the op only partly. Is there anything unusual about the label sets of the low runs?

To check, each `difference` context gets one value per feature, averaged over its examples and its query: the mean of each channel for each operand and the answer, the signed difference of the operands per channel, the lightness of the operands and the answer, the distance between the operands, and the counts of noise and cube examples. Each label set is then compared with all `difference` contexts on all {len(BIAS_FEATURES)} features at once (a Hotelling T² test[^hotelling]), and on how it fills the cube (by pairs of operand octants, and by answer region, χ² tests).

[^hotelling]: Hotelling T² test: the many-feature version of a t test, asking whether a sample mean differs from a reference mean, with the features' correlations taken into account.

{bias_table()}

The low runs' sets look like any other. Their tails run from {LOW_BIAS_P[0]:.2f} to {LOW_BIAS_P[-1]:.2f}, and {WORD[LOW_BIAS_TYPICAL]} of the {WORD[len(LOW_BIAS_P)]} are above 0.5. The smallest tail of any of the 24 sets is {BIAS_MIN_P:.2f}, about what 24 random draws give. On the cube, the smallest of the 48 occupancy tails is {OCC_MIN_P:.2f}, about what 48 tests give, and it is not on a low run. No single feature tracks removal over the 24 runs either: the strongest is `{BIAS_FEATURES[TOP_FEATURE]}`, at Spearman ρ = {f"{TOP_RHO:.2f}".replace("-", "−")} (p = {TOP_RHO_P:.2f}).

The nesting gives a second view. Every context labelled at 1/50 is also labelled at 1/25 and 1/10, so the three sets of a seed grow around one another. If a run ended low because of some contexts in its set, those contexts stay in every larger set, diluted by the new ones, so removal should climb as the set grows and not fall again. At seed 715 the low run at 1/50 recovers once more contexts join its set ({NEST[715][0]:.2f}, then {NEST[715][1]:.2f} and {NEST[715][2]:.2f}), which fits. But at seed 719 removal goes {NEST[719][0]:.2f}, {NEST[719][1]:.2f}, {NEST[719][2]:.2f} as the set grows. At seeds 717 and 718 the 1/25 runs end at {NEST[717][1]:.2f} and {NEST[718][1]:.2f}, while the 1/10 runs, whose sets contain theirs, end at {NEST[717][2]:.2f} and {NEST[718][2]:.2f}. Low runs come and go as the set grows.

So the label sets don't account for the low runs, as far as these features can see. That leaves run-to-run sensitivity, where a small change in which contexts carry the pull sends a run to a different end. These tests can't rule out a feature they don't measure, so E5 trains sets and seeds in new pairings.
"""

# %%

rf"""
## Crossing label sets with seeds (E5)

Does a low run follow its label set or its seed? Here the 1/50 sets of four seeds are each trained at all four of those seeds: {CROSS[1]} and {CROSS[3]}, whose 1/50 runs ended low, and {CROSS[0]} and {CROSS[2]}, whose runs ended high. A set that holds a run back would give a low row, and a seed that does would give a low column.

These sixteen runs came after a faster training step was merged, which also turns the GPU determinism flags off (a shared compilation cache keeps runs reproducible instead). They trained on L40S cards rather than L4s. A run sees the same batches from the same initialization as before, so on the diagonal only the arithmetic differs from the 1/50 runs of E1 to E4, in its last bits.

{cross_table()}

Neither the sets nor the seeds account for the low runs. On the diagonal, both low runs now end with the rest: seed 715 at {CROSS_DIAG[715]:.2f} (it was {CROSS_OLD[715]:.2f}) and seed 719 at {CROSS_DIAG[719]:.2f} (it was {CROSS_OLD[719]:.2f}). Meanwhile {WORD[len(CROSS_LOW)]} of the twelve new pairings end low ({", ".join(f"the set of {d} at seed {s_}" for d, s_ in CROSS_LOW)}), against five of sixteen at 1/50 and 1/10 in E2. No row and no column is low throughout. Every crossed run stays under the spill criterion (the largest is {CROSS_SPILL_MAX:.3f}).

So whether a fixed-label run ends low is decided by small differences somewhere along the run, which the same set, seed and batches don't fix. The per-visit runs of the fixed confirmation spread less, though the reason for that is still open. The task change of these runs isn't compared here: their controls trained on L4s with the flags on.
"""

# %%

r"""
## Discussion

A fixed labelled set at one context in fifty is enough for the anchor to settle on e₁ and keep it (E1, E2). That was the main worry, since the per-visit draw had shown the anchor most of the `difference` contexts over a run. On that count, the scarce-label claim of the D2.2 design holds with labels as a real pipeline would have them.

But removal is less reliable with a fixed set: a few runs in eight end well short of the rest (E2), mostly because they never reach the level of the others (E3). Since a fixed set five times larger did no better, more labels of the same kind do not seem to be the fix, at least between about 900 and 4,300 contexts. Nor do the low runs' sets differ from the others in what their contexts show (E4), and in new pairings of set and seed the low runs move about: the same set at the same seed can end low or high depending on the last bits of the arithmetic (E5). The per-visit draw spreads less, and that may come from averaging over almost every context, which could steady a run against such small differences.

The [low-runs](/docs/m2/low-runs/report.py) report follows the low tail into the model. Every run writes the evidence that an example fits `difference` both on e₁ and along a second direction, and the low runs are the ones whose reader blocks take more of that evidence from off the axis, a difference that the loss and the anchor term do not see. The same off-axis part is in the per-visit runs at a few hundredths, so the fixed set makes a general mechanism more visible rather than a new one.

If this holds, a comparison against post-hoc methods with fixed labels should expect a wider spread on the anchored side than the fixed confirmation showed. With that spread, eight runs give a noisy seed mean.
"""

# %%

r"""
## References

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed`, with the per-visit label draw, keeps the anchor on all eight runs at seeds 712 to 719 with removal near 0.9, and is the recipe of record for D2.2.

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The pool over slices 1 to 3 with $\tau_\ell$ held at 0.1 keeps the anchor on all eight seeds with no early spill, and one seed rises late to near the criterion.

[low-runs](/docs/m2/low-runs/report.py)
:   Re-analysis of the 48 stored runs of this experiment and the fixed confirmation: the low runs read part of the evidence for `difference` from off e₁, through the reader blocks, and twins trained from the same labels, seed and batches match in loss and α while parting in removal.

[post-hoc comparison plan](/docs/m2/d2.2/post-hoc-comparison.md)
:   Plans a comparison of the edit along e₁ with directions fitted after training on the controls at seeds 712 to 719, with the anchor's tolerance of scarce labels among the claims it tests.
"""
