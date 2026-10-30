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

RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
GHOST = light_dark("#bbb", "#555")
INK = {
    REDRAW: light_dark("#08306b", "#c6dbef"),
    "r10": light_dark("#1b7837", "#a6dba0"),
    "r25": light_dark("#a6611a", "#dfc27d"),
    "r50": light_dark("#b2182b", "#f4a582"),
}
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
            shaded band is the call window (epochs 44 to 64), and the dashed line is where the call counts a run as
            kept.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(data["names"]), figsize=(6.6, 2.2), layout="constrained", sharey=True)
        ep = np.asarray(data["epochs"])
        show = ep <= 120
        for ax, n, label in zip(axes, data["names"], data["shown"], strict=True):
            runs = np.asarray(data["alpha"][n])
            for a in runs:
                ax.plot(ep[show], a[show], color=INK[n], lw=0.4, alpha=0.35, zorder=1)
            mark = np.flatnonzero(show)[:: max(1, show.sum() // 8)]
            ax.plot(ep[show], runs.mean(axis=0)[show], color=INK[n], lw=1.1, zorder=3)
            ax.plot(ep[mark], runs.mean(axis=0)[mark], MARKER[n], ms=3, color=INK[n], zorder=4, mew=0)
            ax.axvspan(*data["window"], color=GHOST, alpha=0.25, lw=0, zorder=0)
            ax.axhline(data["kept_min"], color=RULE, lw=0.7, ls="--", zorder=0)
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
        and the task change sits near zero.
    """
    return end_draw(data, alt)


@memo
def end_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-labels-end",
        alt_text=alt_text,
        caption="""
            **Removal, spill, and task at epoch 400, by condition**, from the most labelled `difference` contexts to
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
                ax.plot([i + jit[n][j] for i, n in enumerate(names)], v[:, j], color=GHOST, lw=0.3, zorder=1)
            for i, n in enumerate(names):
                _dots(ax, i, data[key][n], INK[n], MARKER[n], jit[n])
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
        t = TASK[n]
        rows.append([
            f"`{SHOWN[n]}`",
            f"{REDRAW_DISTINCT:,}" if n == REDRAW else f"{int(round(N_LABELLED[n].mean())):,}",
            fmt(REMOVAL[n]),
            "" if n == REDRAW else interval(VS_REDRAW[n]),
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
        **Removal, spill and task at epoch 400**, seed mean and range over eight runs at seeds 712 to 719, net of the
        median drop over the other 63 coordinate axes of the run. "Contexts labelled" is the number of `difference`
        contexts that carried the label at some point in the run, the mean over seeds for a fixed draw and an
        estimate for the per-visit draw; both count over the whole corpus, of which training visits the first nine
        tenths. "Removal less per-visit" is a 95% Welch t interval on the difference of the
        seed means. "Over, block 1" counts the runs over the criterion under the same edit restricted to block 1. The
        task change is the mean over ops of the clean score less that of the control at the seed.
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
    i64 = ALONG_EPOCHS.index(64)
    alt = f"""
        Seed-mean removal along training rises from near zero at epoch 16 under every condition. At epoch 64 it is
        {m[REDRAW][i64]:.2f} for the per-visit draw, {m["r10"][i64]:.2f} at a fixed 1/10, {m["r25"][i64]:.2f} at
        1/25 and {m["r50"][i64]:.2f} at 1/50; at the end, {m[REDRAW][-1]:.2f}, {m["r10"][-1]:.2f},
        {m["r25"][-1]:.2f} and {m["r50"][-1]:.2f}.
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
            run, as a share of the way to the target null of the run at the end. **Left:** the seed mean of each
            condition. **Right:** each run, one panel per condition, with the seed mean of the per-visit draw as a
            faint line behind for reference.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        ep = np.asarray(data["epochs"])
        fig = plt.figure(figsize=(6.8, 2.4), layout="constrained")
        gs = fig.add_gridspec(1, 4, width_ratios=(1.6, 1, 1, 1))
        left = fig.add_subplot(gs[0, 0])
        for n, label in zip(names, data["shown"], strict=True):
            v = np.asarray(data["removal"][n])
            left.plot(ep, v.mean(axis=0), color=INK[n], marker=MARKER[n], ms=2.5, lw=1.0, label=label)
        left.set_ylim(-0.05, 1.0)
        left.set_xlabel("epoch", fontsize=7)
        left.set_ylabel("removal ↑", fontsize=7)
        left.tick_params(labelsize=6)
        left.legend(fontsize=5.5, frameon=False, loc="lower right")
        ref = np.asarray(data["removal"][names[0]]).mean(axis=0)
        cells = [fig.add_subplot(gs[0, j]) for j in range(1, len(names))]
        for k, (ax, n) in enumerate(zip(cells, names[1:], strict=True)):
            ax.plot(ep, ref, color=GHOST, lw=1.2, zorder=1)
            for v in np.asarray(data["removal"][n]):
                ax.plot(ep, v, color=INK[n], lw=0.7, alpha=0.7, zorder=2)
            ax.set_ylim(-0.05, 1.0)
            ax.set_title(data["shown"][k + 1], fontsize=6)
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

rf"""
# Fixed labels

/// tip |
<!-- lede -->
Every anchored run so far drew its labels afresh on each visit to a context, so over a run the anchor saw labels on most `difference` contexts. Holding one draw fixed for the run, as a real labelled set would be, still keeps the anchor at every rate we tried. But the edit removes the op less reliably: a few runs end well below the rest, and more labels in the fixed set did not cure it.
///

The [post-hoc comparison plan](/docs/m2/d2.2/post-hoc-comparison.md) compares the edit along the anchored axis with directions found after training on a control. Writing it up, we noticed that the anchor labeller (inherited from M1) redraws its 1/50 draw on every visit to a context. An epoch covers about a quarter of the training split, so over 400 epochs a context is in view about a hundred times, and a fresh draw on each visit labels about three quarters of the {N_ANCHORED:,} `difference` contexts at some point, about {REDRAW_DISTINCT:,}. So the anchor had far more distinct labels than "one context in fifty" suggests. A real pipeline labels a fixed set once.

This report trains the recipe of record, `τ_ℓ fixed`, with one draw per run held for the whole run, at three rates: 1/50 (about {int(round(N_LABELLED["r50"].mean(), -1)):,} labelled `difference` contexts), 1/25 (about {int(round(N_LABELLED["r25"].mean(), -1)):,}), and 1/10 (about {int(round(N_LABELLED["r10"].mean(), -1)):,}). Each rate runs at the eight seeds of the [fixed confirmation](/docs/m2/fixed-confirm/report.py) (712 to 719), whose runs of the same recipe with the per-visit draw are the comparison.
"""

# %%

rf"""
## Observations

- [The anchor settles at every rate (E1)](#the-anchor-settles-at-every-rate-e1): all 24 runs are called kept at epoch 64, and α at the example answers settles about as fast as under the per-visit draw.
- [Removal and spill at the end (E2)](#removal-and-spill-at-the-end-e2): every run keeps the anchor and passes the task gate as often as the per-visit runs do. At 1/50 and at 1/10, a few runs end well below the lowest per-visit run, so the seed mean falls. At 1/25 removal matches the per-visit draw, but one run spills late, over the criterion.
- [When the low runs part (E3)](#when-the-low-runs-part-e3): some low runs sit low from early in training, and others slide down after the anchor has settled. Neither kind tracks the label count.

## Scope and method

This is an exploratory study, with no preregistration and no gate. It started as a scout of one seed at 1/50 to epoch 64, then 1/50 and 1/10 at four seeds to the end. Those two differed at four seeds, so they were carried to eight seeds, and 1/25 was added between them. The three rates were each chosen before their runs, and none was chosen by how it scored.

Every run is `τ_ℓ fixed` as the fixed confirmation trained it, with one change: in place of the per-visit draw, a context carries the label on every visit if a draw made once before training picked it, and on no visit otherwise. The draw comes from a stream of its own per seed, and the draws are nested, so every context labelled at 1/50 is also labelled at 1/25 and at 1/10. The per-visit draw is still taken and set aside, so each run trains on the same batches from the same initialization as the per-visit run at its seed. Only which contexts carry the pull differs.

The pull is a mean over the labelled contexts in each batch, so the label rate does not change how hard the anchor pulls. At 1/50 it also leaves the number of labelled contexts per batch unchanged on average. What the rate changes is how many distinct contexts the pull ever falls on, and how much the pull in a batch varies from one batch to the next.

We compare conditions as two samples of eight runs (the D2.2 rule since the determinism decision), with a 95% Welch t interval on the difference of seed means. Over eight seeds, removal under the per-visit draw spreads by about {SPAN[REDRAW]:.2f}. All runs trained and were measured on L4 GPUs, as the fixed confirmation's were, and share that experiment's controls (no pull and no anti term, at the same seed).

## The measurements

Every edit below is the e₁ edit. Each score is the drop under that edit, less a baseline: the median drop when the same edit is made along each of the other 63 coordinate axes of the run (the rule of the D2.2 design).

Removal
:   The drop on `difference` under the full edit at every position and slice, as a share of the way from the clean score to the target null. Higher is better, and a run below 0.5 has lost the anchor.

Spill
:   The largest drop on any other op, at the dose where it is largest. Lower is better, and a run above 0.02 is over the criterion.

Task change
:   The clean score per op, less that of the control at the seed. The gate of the D2.2 design asks for a mean change over ops of at least −0.02, and at least −0.05 on any one op.

Call
:   The early verdict on whether a run kept the anchor, from mean α at the example answers at slice 3 over epochs 44 to 64. A run below 0.15 is called lost, and one at 0.28 or above is called kept.
"""

# %%

rf"""
## The anchor settles at every rate (E1)

With the pull confined to a few hundred contexts, does the anchor still settle on e₁ by the time the call is made? If a fixed set were too small, α at the example answers would rise more slowly or level off lower.

{alpha_figure()}

Every run is called kept, at every rate, with a wide margin. The rise over the first 40 epochs looks much the same in all four conditions. Over the call window, the fixed 1/50 runs sit a little lower than the per-visit runs on the seed mean, and the 1/25 and 1/10 runs a little higher and closer together, all near the spread of the per-visit runs.

So a fixed set of under a thousand labelled contexts is enough for the anchor to settle. Across the fixed-label runs, the window mean says little about how much a run removes at the end (r = {R_WINDOW:.2f} over the 24 runs), which is why the end measurements below are needed.
"""

# %%

rf"""
## Removal and spill at the end (E2)

Does a fixed labelled set remove `difference` as well as the per-visit draw, at the same cost to the other ops? If label count mattered over this range, removal would climb from 1/50 to 1/10.

{end_figure()}

{end_table()}

Every run keeps the anchor, and the task change sits near zero in every condition, with no more gate misses than under the per-visit draw. The differences are in removal and in one run of spill.

Removal does not climb with the label count. At 1/50 and at 1/10 most runs remove about as much as the per-visit runs, but a few end below {LOW_LINE}, where the lowest per-visit run is {REMOVAL[REDRAW].min():.2f}: {WORD[len(LOW_RUNS["r50"])]} at 1/50 ({LOW_RANGE["r50"][0]:.2f} to {LOW_RANGE["r50"][1]:.2f}) and {WORD[len(LOW_RUNS["r10"])]} at 1/10 ({LOW_RANGE["r10"][0]:.2f} to {LOW_RANGE["r10"][1]:.2f}). Those runs pull the seed mean down by about the same amount at both rates. At 1/25 every run sits within the per-visit range, and the seed means match.

The low runs fall mostly at different seeds for the two rates, sharing only seed {SHARED_LOW[0]}. The per-visit runs at those seeds range from its lowest (seed {LOW[REDRAW]}) to its second highest (seed 717; the grey lines in the figure). So they look more like runs that went one way than seeds that are hard for the recipe.
<!-- REVIEW: the draft said the per-visit runs at the low seeds were "among its lower ones, but still within its
range". Within its range is true of any per-visit run, and 717 (0.90) and 718 (0.88) are not among its lower ones;
only 715 and 719 are. Verify: REMOVAL[REDRAW] by seed. The conclusion of the paragraph stands either way. --> All three low runs at 1/10 are at seeds 717 to 719, which is why the first look at four seeds missed them.

At 1/25, one run (seed {SPILL_RUN}) ends over the spill criterion, on `{SPILL_OP}`, and over it again under the edit at block 1 alone. No other run of any condition does. Its spill rises late, from about epoch {SPILL_RISE} (after one reading just over the criterion at epoch 32, while the anchor was still settling), as the late rise the soft pool scout saw on one seed of `τ_ℓ fixed`.
<!-- REVIEW: added the epoch-32 reading (0.032, the only kept checkpoint of this run over the criterion before
epoch 192), which SPILL_RISE skips by starting after epoch 64. Verify: _SPILL_RUN_ALONG. "Rises late" stands as the
account of the end value, since the run is back under the criterion from epoch 64 to 160. -->

How sure can we be? The intervals on removal against the per-visit draw are wide: at 1/10 it just excludes zero, and at 1/50 it does not. With eight runs each, a difference set by two or three low runs is hard to pin down. So the three fixed rates are not separable, and 1/25 looking best may well be the luck of its eight seeds.
"""

# %%

rf"""
## When the low runs part (E3)

Do the low runs fail to settle, or settle and then lose ground? The answer says whether the shortfall is about the anchor finding e₁ or about holding it there.

{removal_along_figure()}

Both happen. Some low runs are already behind at epoch 32 and never catch up, sitting at about 0.6 for the rest of training. Others come close to the per-visit level by epoch 100 or so and then slide down over the rest of the run, while the anti weight anneals. Under the per-visit draw, slow starters (such as seed 717, still near zero at epoch 32) catch up by the end.

Neither kind lines up with the rate. Both appear at 1/50 and at 1/10, and the 1/25 runs have neither; they also reach their final level sooner than any other condition, by about epoch 64. With eight runs a condition, that may be the same luck of the seeds as in E2.

One account that fits, without our having tested it: with a fixed set, the pull keeps returning to the same few hundred or thousand contexts, so what the anchor learns depends on which contexts the draw happened to pick. A draw whose contexts sit unevenly over the examples or the replacement noise could anchor the op only partly, and keep it that way. The per-visit draw averages over almost every context, which would explain why its runs spread less. Telling this apart from plain run-to-run sensitivity would take several draws at the same seed.
"""

# %%

r"""
## Discussion

A fixed labelled set at one context in fifty is enough for the anchor to settle on e₁ and keep it (E1, E2). That was the main worry, since the per-visit draw had shown the anchor most of the `difference` contexts over a run. On that count, the scarce-label claim of the D2.2 design holds with labels as a real pipeline would have them.

But removal is less reliable with a fixed set: a few runs in eight end well short of the rest (E2), either from early in training or after a late slide (E3). Since a fixed set five times larger did no better, more labels of the same kind do not seem to be the fix, at least between about 900 and 4,300 contexts. The per-visit draw spreads less, and that may come from the variety of contexts it sees rather than their number.

If this holds, a comparison against post-hoc methods with fixed labels should expect a wider spread on the anchored side than the fixed confirmation showed, and eight runs give a noisy seed mean. The intermediate rate looking best is within what eight seeds can show by chance, so it would need fresh seeds before it counts for more.
"""

# %%

r"""
## References

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed`, with the per-visit label draw, keeps the anchor on all eight runs at seeds 712 to 719 with removal near 0.9, and is the recipe of record for D2.2.

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The pool over slices 1 to 3 with $\tau_\ell$ held at 0.1 keeps the anchor on all eight seeds with no early spill, and one seed rises late to near the criterion.

[post-hoc comparison plan](/docs/m2/d2.2/post-hoc-comparison.md)
:   Plans a comparison of the edit along e₁ with directions fitted after training on the controls at seeds 712 to 719, with the anchor's tolerance of scarce labels among the claims it tests.
"""
