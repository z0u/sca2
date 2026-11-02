# ruff: noqa: B018
# title: What the edit misses in the low runs

# An exploratory re-analysis: the stored checkpoints of every run of the fixed-labels experiment, its crossed runs,
# and the per-visit runs of the fixed confirmation, scored with edits restricted by slice and position, with
# every direction but e₁ removed at the example answers, and with linear probes for the per-example judgement off
# e₁. `experiment.py` beside this script takes the measurements; no new training.

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


fl, fc = ex.fl, ex.fc
END_BODY = fetch_json(ex.END_REF)
DESIGN = END_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
E1 = 0
FULL = len(fl.rb.DOSE_GAMMAS) - 1
END = {r["label"]: r for r in END_BODY["runs"]}
LABELS = [r["label"] for r in ex.runs()]
KIND = {r["label"]: r["kind"] for r in ex.runs()}
SEED = {r["label"]: r["model_seed"] for r in ex.runs()}
ALONG: dict[str, dict[int, dict]] = {}
for _r in fetch_json(ex.ALONG_REF)["runs"]:
    ALONG.setdefault(_r["label"], {})[_r["epoch"]] = _r
ALONG_EPOCHS = sorted(next(iter(ALONG.values())))
ROUTES = {r["label"]: r for r in fetch_json(ex.ROUTES_REF)["runs"]}
EVIDENCE = {r["label"]: r for r in fetch_json(ex.EVIDENCE_REF)["runs"]}
TWINS = fetch_json(ex.TWINS_REF)


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


# Removal as the fixed-labels experiment scores it: net of the median over the other 63 axes, at the full dose.
_AX = {r["label"]: r for ref in (fl.AXES_REF, fl.CROSS_AXES_REF, fc.AXES_REF) for r in fetch_json(ref)["runs"]}
REMOVAL = {
    lab: float(net_of(_AX[lab]["drop"])[FULL, D] / (_AX[lab]["clean"][D] - _AX[lab]["null"][D])) for lab in LABELS
}
_ALONG_AX: dict[tuple[str, int], dict] = {}
for _ref in (fl.ALONG_REF, fl.CROSS_ALONG_REF, fc.ALONG_REF):
    for _r in fetch_json(_ref)["runs"]:
        _ALONG_AX[(_r["label"], _r["epoch"])] = _r
KEPT_EPOCHS = sorted({e for _, e in _ALONG_AX})


def removal_along(lab: str) -> np.ndarray:
    null = _AX[lab]["null"][D]
    return np.array([net_of(_ALONG_AX[(lab, e)]["drop"])[D] / (_ALONG_AX[(lab, e)]["clean"][D] - null) for e in KEPT_EPOCHS])  # fmt: skip


REMOVAL_ALONG = {lab: removal_along(lab) for lab in LABELS}

LOW_LINE = 0.8  # the fixed-labels report's line: under the lowest per-visit run (0.83)
LOW = [lab for lab in LABELS if REMOVAL[lab] < LOW_LINE]
NORMAL = [lab for lab in LABELS if REMOVAL[lab] >= LOW_LINE]
assert all(KIND[lab] != "per-visit" for lab in LOW)
N_LOW, N_NORMAL = len(LOW), len(NORMAL)

# The twins: the same label set at the same seed, trained twice with the last bits of the arithmetic different.
TWINS_ALL = [(fl.label_of(s, None), fl.cross_label_of(s, s)) for s in fl.CROSS_SEEDS]
SPLIT = [(a, b) for a, b in TWINS_ALL if (REMOVAL[a] < LOW_LINE) != (REMOVAL[b] < LOW_LINE)]
assert [SEED[a] for a, _ in SPLIT] == [715, 719], SPLIT  # the prose names them


def share(lab: str, edit: str) -> float:
    """The drop on `difference` under *edit*, as a share of the way from the clean score to the target null."""
    edits = END[lab]["edits"]
    return float((edits if edit in edits else ROUTES[lab]["edits"])[edit][D])


def group(values: dict[str, float], labs: list[str]) -> np.ndarray:
    return np.array([values[lab] for lab in labs])


# --- Formatting -----------------------------------------------------------------------------------------------


def num(v: float, spec: str = ".2f") -> str:
    half = 0.5 * 10 ** -int(spec[-2])
    return f"{0.0 if abs(v) < half else v:{spec}}".replace("-", "−")


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


def shown(lab: str) -> str:
    """A run's name in a table: its seed, with the label set's seed where that differs."""
    r = next(r for r in ex.runs() if r["label"] == lab)
    if r["kind"] == "cross" and r["draw_seed"] != r["model_seed"]:
        return f"set of {r['draw_seed']} at {r['model_seed']}"
    tag = {
        "fixed r50": "fixed 1/50",
        "fixed r25": "fixed 1/25",
        "fixed r10": "fixed 1/10",
        "cross": "fixed 1/50, retrained",
        "per-visit": "per-visit",
    }[r["kind"]]
    return f"{tag} at {r['model_seed']}"


# --- Inks -----------------------------------------------------------------------------------------------------


def rule() -> str:
    return light_dark("#333", "#ddd")


def ghost() -> str:
    return light_dark("#bbb", "#555")


def ink_low() -> str:
    return light_dark("#b2182b", "#ff7f7f")


def ink_normal() -> str:
    return light_dark("#2166ac", "#7fb2f0")


def ink_of(lab: str) -> str:
    return ink_low() if lab in LOW else ink_normal()


def fig_legend(fig: plt.Figure, entries: list[tuple[str, str, str]], lines: bool = True) -> None:
    """A key above the panels: one (ink, marker, label) per entry, drawn as a short line with its marker."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=ink, marker=m, ms=3.5, lw=0.9 if lines else 0, mew=0, label=lab) for ink, m, lab in entries
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(entries), frameon=False, fontsize=6.5)


ROLE_LABELS = [*(t for i in "₁₂₃" for t in (f"a{i}", "?", f"b{i}", "=", f"y{i}", ",")), "a", "?", "b", "=", "y", "⏎"]
assert len(ROLE_LABELS) == ex.CONTEXT_LEN


def smooth_step(y: list[float], flat: float = 0.55, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A per-token profile drawn as plateaus joined by smoothstep ramps, so it reads as a sequence of tokens."""
    xs, ys = [], []
    for i, v in enumerate(y):
        xs += [i - flat / 2, i + flat / 2]
        ys += [v, v]
        if i + 1 < len(y):
            t = np.linspace(0, 1, n)[1:-1]
            xs += list(i + flat / 2 + t * (1 - flat))
            ys += list(v + (y[i + 1] - v) * (3 * t**2 - 2 * t**3))
    return np.array(xs), np.array(ys)


# --- Figures --------------------------------------------------------------------------------------------------

_TRAJ = fetch_json(fl.FINISHED_TRAJ_REF) | fetch_json(fl.CROSS_TRAJ_REF)
TRACK_SLICE = 3


def traj_of(lab: str) -> dict:
    t = _TRAJ[lab]["traj"]
    ep = np.asarray(t["epoch"])
    a = np.asarray(t["alpha_anchored"])[:, TRACK_SLICE][:, list(ex.ANSWERS)].mean(axis=1)
    _, first = np.unique(np.round(ep, 3), return_index=True)
    return {
        "epoch": ep[first].tolist(),
        "val_loss": np.asarray(t["val_loss"])[first].tolist(),
        "alpha": a[first].tolist(),
        "weight": np.asarray(t["weight"])[first].tolist(),
        "anti_weight": np.asarray(t["anti_weight"])[first, TRACK_SLICE].tolist(),
    }


def twins_figure() -> str:
    data = {
        "pairs": [[shown(a), shown(b)] for a, b in SPLIT],
        "seeds": [SEED[a] for a, _ in SPLIT],
        "traj": [[traj_of(a), traj_of(b)] for a, b in SPLIT],
        "kept": list(KEPT_EPOCHS),
        "removal": [[REMOVAL_ALONG[a].tolist(), REMOVAL_ALONG[b].tolist()] for a, b in SPLIT],
        "low_first": [REMOVAL[a] < REMOVAL[b] for a, b in SPLIT],
    }
    ends = "; ".join(f"seed {SEED[a]}: {num(REMOVAL[a])} against {num(REMOVAL[b])}" for a, b in SPLIT)
    alt = f"""
        In each of the two split twin pairs the validation loss and α at the example answers of the two runs stay
        within a hundredth of each other for the whole run, while removal under the full edit parts ways: {ends}. At seed 719 the
        two removal curves separate by epoch 32 and never meet; at seed 715 they run together to epoch 128 and the
        low run then slides.
    """
    return twins_draw(data, alt)


@memo
def twins_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-twins",
        alt_text=alt_text,
        caption="""
            **The two twin pairs that split, along training.** Each row is one label set at one seed, trained twice:
            the run of the fixed-labels experiment and its crossed retraining, which differ only in the last bits of
            the arithmetic. The low run of the pair is in red, the other in blue. **Left:** validation loss. **Middle:**
            α at the example answers of the `difference` probe contexts, at slice 3. **Right:** removal under the full
            edit at the kept checkpoints, net of the median over the other 63 axes.
        """,
    )
    def _plot() -> plt.Figure:
        n = len(data["pairs"])
        fig, axes = plt.subplots(n, 3, figsize=(6.8, 1.9 * n), layout="constrained", sharex=True)
        axes = np.asarray(axes).reshape(n, 3)
        for i in range(n):
            tr = data["traj"][i]
            inks = [ink_low(), ink_normal()] if data["low_first"][i] else [ink_normal(), ink_low()]
            zs = [3, 2] if data["low_first"][i] else [2, 3]
            for j in range(2):
                ep = np.asarray(tr[j]["epoch"])
                axes[i, 0].plot(ep, tr[j]["val_loss"], color=inks[j], lw=0.9, zorder=zs[j])
                axes[i, 1].plot(ep, tr[j]["alpha"], color=inks[j], lw=0.9, zorder=zs[j])
                marker = "o" if inks[j] == ink_low() else "s"
                axes[i, 2].plot(
                    data["kept"], data["removal"][i][j], color=inks[j], lw=0.9, marker=marker, ms=2.4, zorder=zs[j]
                )
            axes[i, 0].set_ylim(2.15, 2.5)
            axes[i, 1].set_ylim(-0.05, 0.75)
            axes[i, 2].set_ylim(-0.05, 1.0)
            axes[i, 0].set_ylabel(f"seed {data['seeds'][i]}", fontsize=7)
            for ax in axes[i]:
                ax.tick_params(labelsize=6)
        axes[0, 0].set_title("validation loss ↓", fontsize=7.5)
        axes[0, 1].set_title("α at the example answers ↑", fontsize=7.5)
        axes[0, 2].set_title("removal ↑", fontsize=7.5)
        for ax in axes[-1]:
            ax.set_xlabel("epoch", fontsize=7)
        fig_legend(fig, [(ink_low(), "o", "the low run of the pair"), (ink_normal(), "s", "its twin")])
        return fig

    return _plot()


def survivors_figure() -> str:
    """Per context: what the edited model still gets right, by how many of the context's examples fit `difference`."""
    n_fit = N_FIT
    by = {}
    for lab in LABELS:
        e = np.asarray(END[lab]["per_context"]["edited"])
        by[lab] = [float(e[n_fit == k].mean()) for k in range(ex.K + 1)]
    data = {
        "labels": LABELS,
        "low": LOW,
        "by_fit": by,
        "counts": [int((n_fit == k).sum()) for k in range(ex.K + 1)],
        "probe": {lab: END[lab]["probe"] for lab in LABELS},
        "removal": REMOVAL,
    }
    lo = np.array([by[lab][ex.K] for lab in LOW]).mean()
    hi = np.array([by[lab][ex.K] for lab in NORMAL]).mean()
    alt = f"""
        Left: the expected exact match on `difference` after the full edit rises with the number of fitting
        examples in every run, and far more steeply in the low runs: on contexts where all three examples fit, the
        edited low runs still score {lo:.2f} on average against {hi:.2f} for the rest. Right: a linear probe
        decodes whether an example fits `difference` from the state at its answer with e₁ zeroed at AUC above
        {PROBE_FLOOR:.2f} in every run, low or not, and from the e₁ coordinate alone about as well.
    """
    return survivors_draw(data, alt)


@memo
def survivors_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-survivors",
        alt_text=alt_text,
        caption="""
            **What survives the edit, and what the probes find.** **Left:** expected exact match on the held-out
            `difference` contexts under the full edit, by how many of the context's three examples fit `difference`
            (the rest are replacement-op noise), one line per run: the low runs in red, the others in blue. **Right:**
            AUC of a linear probe for whether an example fits `difference`, from the state at its answer at slice 3
            with e₁ zeroed (x), against the AUC of the e₁ coordinate alone (y), one mark per run.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.4), layout="constrained")
        ks = list(range(ex.K + 1))
        for lab in data["labels"]:
            low = lab in data["low"]
            left.plot(
                ks,
                data["by_fit"][lab],
                color=ink_low() if low else ink_normal(),
                lw=0.8 if low else 0.5,
                alpha=0.9 if low else 0.5,
                zorder=3 if low else 2,
            )
        left.set_xticks(ks, [f"{k}" for k in ks])
        left.set_xlabel("examples that fit `difference`", fontsize=7)
        left.set_ylabel("exact match after the edit ↓", fontsize=7)
        left.set_ylim(0, 0.6)
        for lab in data["labels"]:
            p = data["probe"][lab]["fits, slice 3"]
            low = lab in data["low"]
            right.plot(
                p["off-axis"],
                p["e1"],
                "o" if low else "s",
                ms=3.5 if low else 2.5,
                color=ink_low() if low else ink_normal(),
                alpha=0.9 if low else 0.6,
                mew=0,
                zorder=3 if low else 2,
            )
        right.plot([0.9, 1], [0.9, 1], color=ghost(), lw=0.5, zorder=1)
        right.set_xlim(0.94, 1.0)
        right.set_ylim(0.94, 1.0)
        right.set_xlabel("probe AUC, e₁ zeroed", fontsize=7)
        right.set_ylabel("AUC of the e₁ coordinate", fontsize=7)
        for ax in (left, right):
            ax.tick_params(labelsize=6)
        fig_legend(fig, [(ink_low(), "o", "low runs"), (ink_normal(), "s", "the rest")])
        return fig

    return _plot()


ROUTE_EDITS = [
    ("slice 1", "e₁\nslice 1"),
    ("slice 2", "e₁\nslice 2"),
    ("slice 3", "e₁\nslice 3"),
    ("every slice", "e₁\nevery slice"),
    ("only e1, slice 2, answers", "all but e₁\nanswers, slice 2"),
    ("full and answers cleared, slice 2", "full, then\nanswers cleared\nslice 2"),
    ("full and other example positions cleared, slice 2", "full, then other\nexample positions\ncleared, slice 2"),
]


def routes_figure() -> str:
    data = {
        "edits": [e for e, _ in ROUTE_EDITS],
        "ticks": [t for _, t in ROUTE_EDITS],
        "labels": LABELS,
        "low": LOW,
        "share": {lab: [share(lab, e) for e, _ in ROUTE_EDITS] for lab in LABELS},
        "seed": SEED,
    }
    every_lo = group({lab: share(lab, "every slice") for lab in LABELS}, LOW)
    every_hi = group({lab: share(lab, "every slice") for lab in LABELS}, NORMAL)
    alt = f"""
        The e₁ edit at every slice removes {fmt(every_lo)} of the way in the low runs against {fmt(every_hi)} in the
        rest, before netting. No single slice accounts for the gap: the slice that removes most differs by run, and
        in several low runs two slices each remove most of it. Removing all but e₁ at the answers after block 2
        costs half the low runs and almost no normal run. Clearing the answers, or the other example positions,
        on top of the full edit lifts most low runs to the range of the normal runs under the same edit, and
        which of the two does so differs by run.
    """
    return routes_draw(data, alt)


@memo
def routes_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-routes",
        alt_text=alt_text,
        caption="""
            **Where the edit reaches the judgement, by run.** The drop on `difference` under each edit, as a share of
            the way from the clean score to the target null, before netting against the other axes: one mark per run,
            the low runs in red and the rest in blue, with a thin grey line joining the edits of each low run. The
            first four edits remove e₁ at the slice named, at every position. The fifth removes every direction but
            e₁ at the example answers after block 2. The last two are the full edit with, on top of it, the whole
            state removed (every direction, no renormalization) at the example answers, or at the other example
            positions, after block 2.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.6, 2.6), layout="constrained")
        rng = np.random.default_rng(0)
        x = np.arange(len(data["edits"]))
        for lab in data["labels"]:
            low = lab in data["low"]
            jit = rng.uniform(-0.18, 0.18)
            v = data["share"][lab]
            if low:
                ax.plot(x + jit, v, color=ghost(), lw=0.4, zorder=1)
            ax.plot(
                x + jit,
                v,
                "o" if low else "s",
                ms=3.2 if low else 2.2,
                color=ink_low() if low else ink_normal(),
                alpha=0.9 if low else 0.45,
                mew=0,
                zorder=3 if low else 2,
            )
        ax.set_xticks(x, data["ticks"], fontsize=6)
        ax.set_ylabel("drop on `difference`, share of the way ↑", fontsize=7)
        ax.axhline(0, color=ghost(), lw=0.5, zorder=0)
        ax.set_ylim(-0.35, 1.05)
        ax.tick_params(labelsize=6)
        fig_legend(fig, [(ink_low(), "o", "low runs"), (ink_normal(), "s", "the rest")], lines=False)
        return fig

    return _plot()


def along_figure() -> str:
    data = {
        "kept": list(KEPT_EPOCHS),
        "epochs": list(ALONG_EPOCHS),
        "labels": LABELS,
        "low": LOW,
        "removal": {lab: REMOVAL_ALONG[lab].tolist() for lab in LABELS},
        "fits": {lab: [ALONG[lab][e]["probe"]["fits, slice 3"]["off-axis"] for e in ALONG_EPOCHS] for lab in LABELS},
        "schedule": traj_of(LABELS[0]),
    }
    fits = np.array([data["fits"][lab] for lab in LABELS])
    data["fits_mean"] = {
        "low": fits[[lab in LOW for lab in LABELS]].mean(0).tolist(),
        "rest": fits[[lab in NORMAL for lab in LABELS]].mean(0).tolist(),
    }
    a32 = np.array([ALONG[lab][32]["probe"]["fits, slice 3"]["off-axis"] for lab in LABELS])
    alt = f"""
        Top left: removal under the full edit along training for all 48 runs; the low runs, in red, mostly sit under
        the rest from the first checkpoints after the anchor lands, and the two groups separate fully only after
        epoch 200. Top right: the probe for whether an example fits `difference`, with e₁ zeroed, is above
        {a32.min():.1f} by epoch 32 in every run and climbs toward 1; the mean over the low runs and the mean over
        the rest are drawn as thick lines and lie close together throughout. Below: the anchor weight and the anti
        weight at slice 3 along training.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-along",
        alt_text=alt_text,
        caption="""
            **Along training, every run.** **Top left:** removal under the full edit at the kept checkpoints, net of
            the median over the other 63 axes, one line per run: the low runs in red, the rest in blue. **Top
            right:** AUC of the probe for whether an example fits `difference`, from the state at its answer at slice 3
            with e₁ zeroed, one thin line per run and the mean of each group as a thick line. **Bottom:** the anchor weight $\\lambda_a$ and the anti weight $\\lambda_{\\bar s}$ at
            slice 3, on the same epoch axis.
        """,
    )
    def _plot() -> plt.Figure:
        fig = plt.figure(figsize=(6.8, 3.0), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=(3, 1))
        top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        strips = [fig.add_subplot(gs[1, 0], sharex=top[0]), fig.add_subplot(gs[1, 1], sharex=top[1])]
        for lab in data["labels"]:
            low = lab in data["low"]
            kw: dict[str, Any] = dict(
                color=ink_low() if low else ink_normal(),
                lw=0.8 if low else 0.45,
                alpha=0.9 if low else 0.4,
                zorder=3 if low else 2,
            )
            top[0].plot(data["kept"], data["removal"][lab], **kw)
            top[1].plot(data["epochs"], data["fits"][lab], **kw)
        top[1].plot(data["epochs"], data["fits_mean"]["low"], color=ink_low(), lw=1.8, zorder=5)
        top[1].plot(data["epochs"], data["fits_mean"]["rest"], color=ink_normal(), lw=1.8, zorder=5)
        top[0].set_ylim(-0.05, 1.0)
        top[0].set_ylabel("removal ↑", fontsize=7)
        top[1].set_ylim(0.7, 1.0)
        top[1].set_ylabel("probe AUC, e₁ zeroed", fontsize=7)
        sc = data["schedule"]
        for s in strips:
            s.plot(sc["epoch"], sc["weight"], color=rule(), lw=0.7)
            s.plot(sc["epoch"], sc["anti_weight"], color=rule(), lw=0.7, ls="--")
            s.set_ylim(0, 0.27)
            s.set_xlabel("epoch", fontsize=7)
        strips[0].set_ylabel("weights", fontsize=6.5)
        strips[0].text(380, 0.11, "$\\lambda_a$", fontsize=6, ha="right", va="bottom")
        strips[0].text(60, 0.23, "$\\lambda_{\\bar s}$", fontsize=6, ha="left", va="bottom")
        for ax in top + strips:
            ax.tick_params(labelsize=6)
        for ax in top:
            ax.tick_params(labelbottom=False)
        fig_legend(fig, [(ink_low(), "", "low runs"), (ink_normal(), "", "the rest")])
        return fig

    return _plot()


def peak_figure() -> str:
    data = {"labels": LABELS, "low": LOW, "peak": PEAK, "end": REMOVAL, "line": SLIDE_LINE}
    alt = f"""
        Removal at the end of training against its peak along training, one mark per run. Every run lies on or
        under the diagonal. The low runs, in red, peak at {fmt(group(PEAK, LOW))} and the rest at
        {fmt(group(PEAK, NORMAL))}; {len(LOW_SLIDERS)} low runs lie more than {SLIDE_LINE} under the diagonal and no
        normal run does.
    """
    return peak_draw(data, alt)


@memo
def peak_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-peak",
        alt_text=alt_text,
        caption="""
            **Peak against end.** Removal under the full edit at epoch 400 against the highest value at any kept
            checkpoint, one mark per run: the low runs in red, the rest in blue. The solid line is where a run ends at
            its peak and the dashed line a tenth under it.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(3.0, 2.8), layout="constrained")
        ax.plot([0.55, 1], [0.55, 1], color=ghost(), lw=0.6, zorder=1)
        ax.plot([0.55, 1], [0.55 - data["line"], 1 - data["line"]], color=ghost(), lw=0.6, ls="--", zorder=1)
        for lab in data["labels"]:
            low = lab in data["low"]
            ax.plot(data["peak"][lab], data["end"][lab], "o" if low else "s", ms=4 if low else 3, color=ink_low() if low else ink_normal(), alpha=0.9 if low else 0.6, mew=0, zorder=3 if low else 2)  # fmt: skip
        ax.set_xlim(0.6, 0.96)
        ax.set_ylim(0.55, 0.96)
        ax.set_xlabel("peak removal along training", fontsize=7)
        ax.set_ylabel("removal at epoch 400 ↑", fontsize=7)
        ax.tick_params(labelsize=6)
        fig_legend(fig, [(ink_low(), "o", "low runs"), (ink_normal(), "s", "the rest")], lines=False)
        return fig

    return _plot()


SITES_RUN = fl.cross_label_of(715, 715)
SITE_SLICES = (1, 2, 3, 4)


def sites_figure() -> str:
    a = EVIDENCE[SITES_RUN]["alpha_by_position"]
    data = {
        "difference": a["difference"],
        "other": a["other"],
        "slices": list(SITE_SLICES),
        "answers": list(ex.ANSWERS),
        "query": ex.QUERY_EQ,
    }
    alt = """
        A stack of four smooth-step profiles, one per slice after a block, of the alignment with e₁ by position in a
        context, on `difference` contexts and on the other ops. On `difference` contexts the alignment is high at the
        three example answers from slice 1 on and near zero everywhere else, the query `=` included; on the other ops
        it is near zero at every position. The example answers are marked as the sites the fits probe reads, and the
        query `=` as the site the query probe reads.
    """
    return sites_draw(data, alt)


@memo
def sites_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-sites",
        alt_text=alt_text,
        caption=f"""
            **Where the probes read.** Alignment with e₁ (α) by position in a held-out context, one panel per slice
            after a block, on one run ({shown(SITES_RUN)}): the `difference` contexts solid and the mean over the other
            ops dashed. The three example answers $y_1$ to $y_3$, where the anchor pulls and the fits probe reads the
            state (with e₁ zeroed, or e₁ alone), are marked with circles, and the query `=`, where the query probe
            reads, with a square. Between one panel and the next sits one block: attention, then its MLP.
        """,
    )
    def _plot() -> plt.Figure:
        n = len(data["slices"])
        fig, axes = plt.subplots(n, 1, figsize=(6.4, 0.75 * n + 0.5), layout="constrained", sharex=True)
        axes = list(np.atleast_1d(axes))[::-1]
        for ax, sl in zip(axes, data["slices"], strict=True):
            ax.axhline(0, color=ghost(), lw=0.4)
            ax.plot(*smooth_step(data["other"][sl], flat=0.8), "--", color=rule(), lw=0.7)
            ax.plot(*smooth_step(data["difference"][sl], flat=0.8), color=ink_normal(), lw=1.2)
            ys = data["difference"][sl]
            ax.plot(data["answers"], [ys[p] for p in data["answers"]], "o", ms=4, color=ink_normal(), mew=0)
            ax.plot([data["query"]], [ys[data["query"]]], "s", ms=4, color=ink_normal(), mew=0)
            ax.set_ylim(-0.1, 1.0)
            ax.set_yticks([0, 0.5])
            ax.set_ylabel(f"slice {sl}", fontsize=6.5)
            ax.text(-0.4, 0.72, f"after block {sl}", fontsize=6, color=rule(), va="top")
            ax.tick_params(labelsize=6)
        axes[0].set_xticks(np.arange(len(ROLE_LABELS)), ROLE_LABELS, fontsize=6)
        axes[0].set_xlim(-0.6, len(ROLE_LABELS) - 0.4)
        fig.supylabel("α", fontsize=7)
        return fig

    return _plot()


def evidence_figure() -> str:
    ks = EVIDENCE[LABELS[0]]["pc_counts"]
    data = {
        "ks": ks,
        "labels": LABELS,
        "low": LOW,
        "auc": {lab: {s: EVIDENCE[lab]["slices"][s]["auc_by_pcs"] for s in ("2", "3")} for lab in LABELS},
    }
    one = {
        s: np.array([EVIDENCE[lab]["slices"][s]["mean_difference_direction_auc"] for lab in LABELS]) for s in ("2", "3")
    }
    share = {
        s: np.array([EVIDENCE[lab]["slices"][s]["e1_share_of_mean_difference"] for lab in LABELS]) for s in ("2", "3")
    }
    alt = f"""
        The fits probe's AUC against how many principal components of the off-axis answer state it is given, one
        line per run, at slice 2 (left) and slice 3 (right). At slice 2 one component gives about chance in most runs
        and two give most of the way to the ceiling; at slice 3 one component is already near the ceiling. The low
        runs are among the rest. A single off-axis direction, the one between the class means, scores
        {fmt(one["2"])} at slice 2 and {fmt(one["3"])} at slice 3, and e₁ holds {fmt(share["2"])} of the squared
        difference between the class means at slice 2 and {fmt(share["3"])} at slice 3.
    """
    return evidence_draw(data, alt)


@memo
def evidence_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-evidence",
        alt_text=alt_text,
        caption="""
            **How many directions the off-axis evidence takes.** AUC of the fits probe given only the top principal
            components of the state at the example answers with e₁ zeroed, by how many components, one line per run:
            the low runs in red, the rest in blue. **Left:** slice 2. **Right:** slice 3.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), layout="constrained", sharey=True)
        for ax, s in zip(axes, ("2", "3"), strict=True):
            for lab in data["labels"]:
                low = lab in data["low"]
                ax.plot(data["ks"], data["auc"][lab][s], color=ink_low() if low else ink_normal(), lw=0.8 if low else 0.45, alpha=0.9 if low else 0.4, zorder=3 if low else 2)  # fmt: skip
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 4, 8, 16, 32, 63], ["1", "2", "4", "8", "16", "32", "63"])
            ax.set_xlabel(f"principal components given to the probe, slice {s}", fontsize=7)
            ax.set_ylim(0.45, 1.0)
            ax.tick_params(labelsize=6)
        axes[0].set_ylabel("fits probe AUC, e₁ zeroed ↑", fontsize=7)
        fig_legend(fig, [(ink_low(), "", "low runs"), (ink_normal(), "", "the rest")])
        return fig

    return _plot()


def distance_figure() -> str:
    labs = TWINS["labels"]
    dist = np.asarray(TWINS["distance"])
    idx = {lab: i for i, lab in enumerate(labs)}
    pairs = [(fl.label_of(s, None), fl.cross_label_of(s, s)) for s in fl.CROSS_SEEDS]
    fixed = [fl.label_of(s, None) for s in fl.CROSS_SEEDS]
    seed_pairs = [(a, b) for i, a in enumerate(fixed) for b in fixed[i + 1 :]]
    data = {
        "epochs": TWINS["epochs"],
        "twins": {str(SEED[a]): dist[:, idx[a], idx[b]].tolist() for a, b in pairs},
        "split": [str(SEED[a]) for a, _ in SPLIT],
        "seeds": [dist[:, idx[a], idx[b]].tolist() for a, b in seed_pairs],
    }
    first = {k: v[0] for k, v in data["twins"].items()}
    last = {k: v[-1] for k, v in data["twins"].items()}
    seed_last = np.array([v[-1] for v in data["seeds"]])
    alt = f"""
        The relative distance between the weights of the two runs of each twin pair along training,
        with the distances between the fixed-labels runs at different seeds as grey lines for scale. At epoch 16 the
        twins are already {fmt(first.values(), ".2f")} apart, and the distance grows through the run to
        {fmt(last.values(), ".2f")} at epoch 400, close to the {fmt(seed_last, ".2f")} between different seeds. The
        two pairs that split are no further apart than the two that did not.
    """
    return distance_draw(data, alt)


@memo
def distance_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="low-runs-distance",
        alt_text=alt_text,
        caption="""
            **How far apart the twins drift.** The distance between the weights of the two runs of a twin pair, as a
            share of their norm, at each kept checkpoint and at the end, one line per pair: the pairs that split in
            red (circles, seeds 715 and 719) and the pairs that did not in blue (squares, seeds 712 and 718). The
            grey lines are the same distance between the fixed-labels runs at two different seeds.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.2, 2.6), layout="constrained")
        for v in data["seeds"]:
            ax.plot(data["epochs"], v, color=ghost(), lw=0.6, zorder=1)
        for seed, v in data["twins"].items():
            split = seed in data["split"]
            ax.plot(
                data["epochs"],
                v,
                color=ink_low() if split else ink_normal(),
                marker="o" if split else "s",
                ms=2.6,
                lw=0.9,
                mew=0,
                zorder=3 if split else 2,
            )
        ax.text(data["epochs"][0], np.sqrt(2) + 0.02, "unrelated runs: √2", fontsize=6, color=ghost(), va="bottom")
        ax.set_ylim(0, 1.6)
        ax.set_xlabel("epoch", fontsize=7)
        ax.set_ylabel("distance between weights, relative", fontsize=7)
        ax.tick_params(labelsize=6)
        fig_legend(
            fig,
            [
                (ink_low(), "o", "twins that split"),
                (ink_normal(), "s", "twins that did not"),
                (ghost(), "", "two different seeds"),
            ],
        )
        return fig

    return _plot()


# --- Numbers the prose quotes ----------------------------------------------------------------------------------


@memo
def n_fit_of() -> list[int]:
    """How many examples of each held-out `difference` context fit `difference`, from the corpus itself: the
    posterior table says whether an example's answer is one `difference` gives for its operands.
    """
    from sca.compute.data_pipelines import load_data
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from mini.store import get, get_ref, store_context

    with tempfile.TemporaryDirectory() as tmp, store_context(project_store()):
        holdout = get_ref(ex.ex2221.HOLDOUT_REF.format(key=ex.ex2222.MAIN_KEY))
        corpus = get_ref(ex.ex2221.CORPUS_REF.format(key=ex.ex2222.MAIN_KEY))
        assert holdout is not None and corpus is not None
        ho = ex.ex2216.load_holdout(get(holdout, Path(tmp) / "holdout.npz"), ex.K)
        _, meta = load_data(get(corpus, Path(tmp) / "corpus"))
    tok = WordTokenizer(meta.tokenizer_config)
    color_ids = np.array([tok.stoi[n] for n in PALETTE])
    tok2color = np.full(len(tok.itos), -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    P = ex.ex2216._get_posterior()
    table = P.build_table(ex.ex2218.table_of(OPS))
    y = tok2color[ho.tokens[:, list(ex.ANSWERS)]]
    a = tok2color[ho.tokens[:, [ex.UNIT * i for i in range(ex.K)]]]
    b = tok2color[ho.tokens[:, [ex.UNIT * i + 2 for i in range(ex.K)]]]
    fits = np.asarray(table.lookup(a * len(PALETTE) + b, y))[D] > 0
    return fits.sum(axis=1)[ho.op_ids == D].tolist()


N_FIT = np.asarray(n_fit_of())
PROBE_FLOOR = min(END[lab]["probe"]["fits, slice 3"]["off-axis"] for lab in LABELS)

EVERY = {lab: share(lab, "every slice") for lab in LABELS}
ONLY_E1 = {lab: share(lab, "only e1, slice 2, answers") for lab in LABELS}
ZERO_ANSWERS = {lab: share(lab, "zero, slice 2, answers") for lab in LABELS}
ZERO_OTHERS = {lab: share(lab, "zero, slice 2, other example positions") for lab in LABELS}
EVERY_ANSWERS = {lab: share(lab, "every slice, answers") for lab in LABELS}
ANSWERS_GAP = max(abs(EVERY[lab] - EVERY_ANSWERS[lab]) for lab in LABELS)
THEN_ANSWERS = {lab: share(lab, "full and answers cleared, slice 2") for lab in LABELS}
THEN_OTHERS = {lab: share(lab, "full and other example positions cleared, slice 2") for lab in LABELS}
assert all(
    abs(share(lab, "full") - EVERY[lab]) < 0.005 for lab in LABELS
)  # the composite pass reproduces the full edit
LIFT_LINE = 0.1  # a rise over the full edit no normal run shows under either clearing
LIFT_ANSWERS = [lab for lab in LOW if THEN_ANSWERS[lab] - EVERY[lab] >= LIFT_LINE]
LIFT_OTHERS = [lab for lab in LOW if THEN_OTHERS[lab] - EVERY[lab] >= LIFT_LINE]
LIFTED = [lab for lab in LOW if lab in LIFT_ANSWERS or lab in LIFT_OTHERS]
UNLIFTED = [lab for lab in LOW if lab not in LIFTED]
NORMAL_LIFT = max(max(THEN_ANSWERS[lab], THEN_OTHERS[lab]) - EVERY[lab] for lab in NORMAL)
assert NORMAL_LIFT < LIFT_LINE
assert (
    sorted(UNLIFTED, key=lambda lab: EVERY[lab]) == sorted(LOW, key=lambda lab: EVERY[lab])[-len(UNLIFTED) :]
)  # the smallest shortfalls
SLICE = {s: {lab: share(lab, f"slice {s}") for lab in LABELS} for s in (1, 2, 3)}
FITS_OFF = {lab: END[lab]["probe"]["fits, slice 3"]["off-axis"] for lab in LABELS}
FITS_E1 = {lab: END[lab]["probe"]["fits, slice 3"]["e1"] for lab in LABELS}
QUERY_OFF = {lab: END[lab]["probe"]["op at query, slice 3"]["off-axis"] for lab in LABELS}
ALPHA_END = {lab: END[lab]["alpha"]["held-out difference"]["answers"][TRACK_SLICE] for lab in LABELS}
MATCH_CLEAN = {lab: float(np.asarray(END[lab]["per_context"]["edited"])[N_FIT == ex.K].mean()) for lab in LABELS}
MATCH_NOISY = {lab: float(np.asarray(END[lab]["per_context"]["edited"])[N_FIT < ex.K].mean()) for lab in LABELS}
PEAK = {lab: float(REMOVAL_ALONG[lab].max()) for lab in LABELS}
PEAK_EPOCH = {lab: KEPT_EPOCHS[int(REMOVAL_ALONG[lab].argmax())] for lab in LABELS}
SLIDE = {lab: PEAK[lab] - REMOVAL[lab] for lab in LABELS}
# A run whose later blocks use the off-axis content of the answer states: keeping only e₁ there costs it.
ANSWER_LINE = 0.3
ANSWER_READERS = [lab for lab in LABELS if ONLY_E1[lab] > ANSWER_LINE]
LOW_ANSWER = [lab for lab in LOW if lab in ANSWER_READERS]
LOW_PLAIN = [lab for lab in LOW if lab not in ANSWER_READERS]
NORMAL_ANSWER = [lab for lab in NORMAL if lab in ANSWER_READERS]
SLIDE_LINE = 0.1
LOW_SLIDERS = [lab for lab in LOW if SLIDE[lab] >= SLIDE_LINE]
NORMAL_SLIDERS = [lab for lab in NORMAL if SLIDE[lab] >= SLIDE_LINE]
LOSS_END = {lab: traj_of(lab)["val_loss"][-1] for lab in LABELS if lab in _TRAJ}


def group_table() -> str:
    rows = [
        ("removal", REMOVAL),
        ("full edit, share of the way", EVERY),
        ("e₁ edit at slice 2 alone", SLICE[2]),
        ("e₁ edit at slice 3 alone", SLICE[3]),
        ("all but e₁ removed at the answers, slice 2", ONLY_E1),
        ("whole state removed at the answers, slice 2", ZERO_ANSWERS),
        ("whole state removed at the other example positions, slice 2", ZERO_OTHERS),
        ("full edit, then the answers cleared, slice 2", THEN_ANSWERS),
        ("full edit, then the other example positions cleared, slice 2", THEN_OTHERS),
        ("fits probe, e₁ zeroed, slice 3", FITS_OFF),
        ("fits probe, e₁ alone, slice 3", FITS_E1),
        ("query probe, e₁ zeroed, slice 3", QUERY_OFF),
        ("α at the example answers, slice 3", ALPHA_END),
        ("edited match on contexts whose three examples fit", MATCH_CLEAN),
        ("edited match on the other contexts", MATCH_NOISY),
        ("peak removal along training", PEAK),
        ("slide from the peak", SLIDE),
    ]
    body = [[name, fmt(group(v, LOW)), fmt(group(v, NORMAL))] for name, v in rows]
    return table_html(
        ["measurement", f"low ({N_LOW} runs)", f"the rest ({N_NORMAL} runs)"],
        body,
        f"""
        **End-of-training measurements, low runs against the rest.** Mean and range over the runs of each group.
        Removal is net of the median over the other axes; every other edit is the share of the way from the clean
        score to the target null with no baseline. The probes are AUCs on held-out contexts of every op. The match
        is the expected exact match on the query answer after the full edit. The low runs are the {N_LOW} of the 48 that
        end below {LOW_LINE} in removal.
        """,
    )


def twins_table() -> str:
    rows = []
    for a, b in SPLIT:
        for lab in (a, b):
            rows.append(
                [
                    shown(lab) + (" (low)" if lab in LOW else ""),
                    num(LOSS_END[lab], ".3f"),
                    num(ALPHA_END[lab]),
                    num(REMOVAL[lab]),
                    num(EVERY[lab]),
                    num(SLICE[2][lab]),
                    num(SLICE[3][lab]),
                    num(ONLY_E1[lab]),
                    num(THEN_ANSWERS[lab]),
                    num(THEN_OTHERS[lab]),
                ]
            )
    return table_html(
        [
            "run",
            "val loss ↓",
            "α, answers",
            "removal ↑",
            "full edit",
            "e₁, slice 2",
            "e₁, slice 3",
            "all but e₁, answers",
            "full, then answers cleared",
            "full, then others cleared",
        ],
        rows,
        """
        **The two split pairs at epoch 400.** Validation loss and α at the example answers (slice 3) are what training
        sees; the rest are edits. Removal is net of the median over the other axes; the other columns are shares of the
        way with no baseline, under the e₁ edit at every slice, at slice 2 or 3 alone, with every direction but e₁
        removed at the answers after block 2, and under the full edit with the whole state then removed at the answers,
        or at the other example positions, after block 2.
        """,
    )


def low_table() -> str:
    rows = []
    for lab in sorted(LOW, key=lambda lab: REMOVAL[lab]):
        rows.append(
            [
                shown(lab),
                num(REMOVAL[lab]),
                num(EVERY[lab]),
                num(SLICE[3][lab]),
                num(ONLY_E1[lab]),
                num(THEN_ANSWERS[lab]),
                num(THEN_OTHERS[lab]),
            ]
        )
    rows.append(
        [
            "the normal runs",
            fmt(group(REMOVAL, NORMAL)),
            fmt(group(EVERY, NORMAL)),
            fmt(group(SLICE[3], NORMAL)),
            fmt(group(ONLY_E1, NORMAL)),
            fmt(group(THEN_ANSWERS, NORMAL)),
            fmt(group(THEN_OTHERS, NORMAL)),
        ]
    )
    return table_html(
        [
            "run",
            "removal ↑",
            "full edit",
            "e₁, slice 3",
            "all but e₁, answers",
            "full, then answers cleared",
            "full, then others cleared",
        ],
        rows,
        f"""
        **The low runs under the edits of this section, at epoch 400**, lowest removal first, with the mean and range of
        the {N_NORMAL} normal runs in the last row. The columns are as in the table of E1. A run is counted as lifted
        by a clearing when its share rises by {LIFT_LINE} or more over the full edit; no normal run rises by more than
        {num(NORMAL_LIFT)} under either.
        """,
    )


# --- More numbers the prose quotes -----------------------------------------------------------------------------


def _windowed_gap(a: str, b: str, width: int = 40) -> float:
    """The largest mean gap in validation loss between two runs over any window of *width* epochs."""
    ta, tb = traj_of(a), traj_of(b)
    n = min(len(ta["epoch"]), len(tb["epoch"]))
    ep = np.asarray(ta["epoch"][:n])
    gap = np.asarray(ta["val_loss"][:n]) - np.asarray(tb["val_loss"][:n])
    return max(abs(float(gap[(ep >= lo) & (ep < lo + width)].mean())) for lo in range(0, ex.EPOCHS - width + 1, 8))


TWIN_GAP = {SEED[a]: _windowed_gap(a, b) for a, b in TWINS_ALL}
SPLIT_SEEDS = [SEED[a] for a, _ in SPLIT]
GAP_SPLIT = max(TWIN_GAP[s] for s in SPLIT_SEEDS)
GAP_SAME = max(v for s, v in TWIN_GAP.items() if s not in SPLIT_SEEDS)

# The first kept epoch from which every low run is under every normal run.
SCREEN_EPOCH = next(
    e for i, e in enumerate(KEPT_EPOCHS) if all(max(REMOVAL_ALONG[lab][j] for lab in LOW) < min(REMOVAL_ALONG[lab][j] for lab in NORMAL) for j in range(i, len(KEPT_EPOCHS)))
)  # fmt: skip


def worst_other_drop(lab: str, edit: str) -> float:
    """The largest raw drop in expected match on another op under a clearing edit."""
    clean, null = np.array(END[lab]["clean"]), np.array(END[lab]["null"])
    drop = np.array(ROUTES[lab]["edits"][edit]) * (clean - null)
    return float(drop[OTHER].max())


CLEAR_SPILL = {
    edit: np.array([worst_other_drop(lab, edit) for lab in LABELS])
    for edit in ("full", "full and answers cleared, slice 2", "full and other example positions cleared, slice 2")
}
FITS_64 = {
    "low": float(np.mean([ALONG[lab][64]["probe"]["fits, slice 3"]["off-axis"] for lab in LOW])),
    "rest": float(np.mean([ALONG[lab][64]["probe"]["fits, slice 3"]["off-axis"] for lab in NORMAL])),
}
ONE_DIR = {
    s: np.array([EVIDENCE[lab]["slices"][s]["mean_difference_direction_auc"] for lab in LABELS]) for s in ("2", "3")
}
E1_SHARE = {
    s: np.array([EVIDENCE[lab]["slices"][s]["e1_share_of_mean_difference"] for lab in LABELS]) for s in ("2", "3")
}
PROBE_LINE = 0.95


def _pcs_needed(lab: str, s: str) -> float:
    """How many principal components of the off-axis state the probe needs to reach PROBE_LINE (inf if it never does)."""
    counts, aucs = EVIDENCE[lab]["pc_counts"], EVIDENCE[lab]["slices"][s]["auc_by_pcs"]
    return next((float(k) for k, a in zip(counts, aucs, strict=True) if a >= PROBE_LINE), float("inf"))


PCS_NEEDED = {s: np.array([_pcs_needed(lab, s) for lab in LABELS]) for s in ("2", "3")}
_TD = np.asarray(TWINS["distance"])
_TI = {lab: i for i, lab in enumerate(TWINS["labels"])}
TWIN_DIST_FIRST = [float(_TD[0, _TI[a], _TI[b]]) for a, b in TWINS_ALL]
TWIN_DIST_LAST = [float(_TD[-1, _TI[a], _TI[b]]) for a, b in TWINS_ALL]
_FIXED = [fl.label_of(s, None) for s in fl.CROSS_SEEDS]
SEED_DIST_LAST = [float(_TD[-1, _TI[a], _TI[b]]) for i, a in enumerate(_FIXED) for b in _FIXED[i + 1 :]]

# %%

r"""
# What the edit misses in the low runs

/// tip |
<!-- lede -->
A few fixed-label runs end with the edit removing much less of `difference` than the rest, and nothing that training sees sets them apart: a run trained again from the same labels, seed, and batches matches a low run in loss and in alignment, and still removes the op well. What differs is what the later blocks read. In every run the evidence that an example fits `difference` sits partly on the anchored axis and partly off it, and in a low run the blocks that read that evidence take more of it from the part off the axis, which the edit never touches.
///

The [fixed-labels](/docs/m2/fixed-labels/report.py) experiment found that holding one label draw for a run leaves a few runs in eight well below the rest in removal (E2), mostly because they never reach the level of the others (E3), with label sets that look like any other (E4), and with retraining moving which runs end low (E5). This page reads the stored checkpoints of all 48 runs of that experiment and of the [fixed-confirm](/docs/m2/fixed-confirm/report.py) experiment, with no new training, to ask what a low run does differently with the op.
"""

# %%

rf"""
## Observations

- [Twins that split (E1)](#twins-that-split-e1): two pairs of runs were trained twice from the same labels, seed, and batches and ended one low and one not. Their validation loss and α at the example answers stay within a hundredth of each other for the whole run, while removal parts ways early in one pair and late in the other. By the first checkpoint the weights of a twin pair are already most of the way to unrelated, so the last bits of the arithmetic make two different runs that happen to share a function.
- [What survives the edit (E2)](#what-survives-the-edit-e2): what the edited model still gets right are the contexts whose three examples all fit `difference`. Unexpectedly, the low runs keep several times as many of them. In every run the evidence that an example fits is written both on e₁ and along one direction off it, and a linear probe reads either about as well.
- [Where the edit misses (E3)](#where-the-edit-misses-e3): before any baseline, the e₁ edit leaves a fifth of the way in the low runs against a twentieth in the rest, and removing e₁ at the three example answers alone does the whole of that. In half the low runs block 4 reads the answers too, through their off-axis content; in the other half the usual reader, block 3, takes part of the evidence from off the axis. Clearing the answers or the other example positions on top of the full edit lifts most low runs toward the rest, and which of the two does so differs by run.
- [Peak and slide (E4)](#peak-and-slide-e4): the low runs peak lower than the rest and give back a little more after the peak, while the anti weight anneals. The off-axis evidence is in place early in every run, low or not, and the two groups do not separate in removal until after epoch 200.

## Scope and method

This is an exploratory re-analysis of 48 stored runs, with no preregistration and no gate: the 24 fixed-label runs of the fixed-labels experiment (three rates at seeds 712 to 719), its 16 crossed runs (the 1/50 sets of four seeds, each trained at all four), and the eight per-visit runs of the fixed confirmation. All are the recipe of record, `τ_ℓ fixed`, trained for 400 epochs.

Four of the crossed runs pair a label set with its own seed, so they repeat four fixed-labels runs from the same initialization on the same batches with the same labels. The one difference is the card: the fixed-labels runs trained on L4s and the crossed runs on L40S cards. XLA compiles a different program for each card model, and a sum taken in a different order differs in its last bits, so the two runs of a pair see the same gradients to within rounding and part by a little on the first step. We call them twins. Two of the four pairs ended with one run low and one not.

A run is low when its removal ends below {LOW_LINE}, under the lowest per-visit run, as the fixed-labels report drew the line. That makes {N_LOW} low runs: the five of the fixed-labels report, and five of its crossed runs. No per-visit run is low. Removal spreads by about 0.07 over the eight per-visit seeds, so a gap of 0.2 between a low run and its twin is well outside the seed range.

Every pass uses the 14,000 held-out contexts of ex-2.2.21, 2,000 per op, and the probes score on half of them. Restricted edits (one slice, one set of positions) are scored without the median-over-axes baseline, which would need 63 more passes per edit; the baseline takes about a tenth off the full edit in every run, so a restricted edit is compared with the full edit on the same footing, and with removal only loosely. The slide and peak of E4 use the kept checkpoints of the fixed-labels experiment and the fixed confirmation, scored there.

## The measurements

{sites_figure()}

Removal
:   The drop on `difference` under the full e₁ edit, at every slice and position, less the median drop under the same edit along each of the other 63 axes, as a share of the way from the clean score to the target null, as the fixed-labels experiment scores it. Higher is better.

Edit share
:   The drop on `difference` under an edit as a share of the way from the clean score to the target null, with no baseline. The full edit, one slice of it, or an edit that removes every direction but e₁ (or the whole state, with no renormalization) at a set of positions.

Fits probe
:   Whether an example fits `difference`: AUC of a ridge probe on the state at its answer with the e₁ coordinate zeroed, fitted on half the held-out contexts of every op and scored on the rest, beside the AUC of the e₁ coordinate alone. An example fits when its answer is one `difference` gives for its operands; the others carry replacement op noise. The figure above shows where the probe reads.

Query probe
:   Whether the context's op is `difference`: the same probe on the state at the query `=`.

Edited match
:   The expected exact match on the query answer of a held-out `difference` context after the full edit, by how many of the context's examples fit `difference`. Lower means more of the op removed.

α
:   Alignment with e₁: the e₁ component of the unit-norm state at the example answers of the held-out `difference` contexts, at slice 3.
"""

# %%

rf"""
## Twins that split (E1)

Does a low run look different to training? The twins give the cleanest test: two of the four pairs ended one low and one not, at seeds 715 and 719. If a low run went a different way on the task or on the anchor, its loss or its α would say so somewhere along the run.

{twins_figure()}

They do not. In both pairs the validation loss and α at the example answers stay close for the whole run, while removal under the full edit separates by about 0.2. The loss curves do part by a little in places: averaged over any 40 epochs, the two runs of a split pair differ by at most {num(GAP_SPLIT, ".3f")}, and in the low run's favor or against it by turns. But the two pairs that did not split differ by as much ({num(GAP_SAME, ".3f")}), so these gaps are the ordinary difference between any two runs. Over the last hundred epochs α in the low run ends a few hundredths under its twin, less than the spread across seeds. At seed 719 the two removal curves part by epoch 32 and never meet again. At seed 715 they run together until about epoch 100 and the low run then slides, while its twin keeps climbing.

{twins_table()}

So the low state costs nothing on the task and nothing on the anchor term, which fits what the fixed-labels experiment found about label sets and seeds: whichever way a run goes on this, training is indifferent to it. What differs between twins has to be something the edit sees and the loss does not. The rest of the table narrows it a little: in both pairs the e₁ edit at slice 2 alone already misses more in the low run, so the reader at block 3 is where the twins part (block1-edit found the reader at various later blocks on other seeds; E3 covers those). And what it still finds there differs by pair: at seed 719, clearing the other example positions on top of the full edit brings the low run most of the way to its twin and clearing the answers does nothing, while at seed 715 it is the other way round. E3 follows this up across all the low runs.

How can the last bits of the arithmetic decide this? Because twins are not the same run with a small perturbation at the end; they are two runs that part early and keep parting. The distance figure measures how far apart the weights of a pair are, as a share of their size, so that two unrelated runs sit at √2 ≈ 1.41 (the fixed-labels runs at different seeds sit at {fmt(SEED_DIST_LAST, ".2f")} at every checkpoint). By epoch 16 the twins are already {fmt(TWIN_DIST_FIRST, ".2f")} apart, most of the way to unrelated, and the gap keeps widening to {fmt(TWIN_DIST_LAST, ".2f")} at epoch 400.

{distance_figure()}

So by the end a twin pair is nearly as far apart in weight space as two seeds, while its loss and α agree to a hundredth, and the pairs that split are no further apart than the pairs that did not. A split is not a rare watershed that a last bit happened to cross: twins are different runs that share a function, and which side of the line each lands on is drawn afresh, with odds that these seeds put near even. This is the known sensitivity of training to any perturbation at all; what is new here is which quantity the perturbation changes.
"""

# %%

rf"""
## What survives the edit (E2)

Which contexts does the edited model still get right, and what does the model know about `difference` that is not on e₁? The block1-edit account says block 1 writes the evidence that an example fits `difference` onto e₁ at the answer, and a later block reads it. The edit removes the op by clearing that coordinate. If the same evidence also sat off the axis, the reader could take it from there, and the edit would miss by however much it does.

{survivors_figure()}

After the full edit the expected match rises with how many of the three examples fit `difference` in every run, and far more steeply in the low runs: on contexts where all three fit, the edited low runs still score {fmt(group(MATCH_CLEAN, LOW))} against {fmt(group(MATCH_CLEAN, NORMAL))} for the rest, while on the noisier contexts both groups are near the floor. So what the edit misses is read from the examples that fit.

And the evidence is written off e₁ in every run. With the e₁ coordinate zeroed, a linear probe on the state at an example answer at slice 3 tells a fitting example from one that does not at AUC {num(min(FITS_OFF.values()))} or above in all 48 runs, about as well as the e₁ coordinate alone does ({fmt(FITS_E1.values())}). The low runs sit among the rest on both. The op at the query `=` is off-axis too, at AUC {fmt(QUERY_OFF.values())}, as block1-edit found: the anchor never held it there.

How many directions does the off-axis evidence take? Few. Given only the top principal components of the off-axis state, the probe reaches AUC {PROBE_LINE} with a median of {num(np.median(PCS_NEEDED["2"]), ".0f")} components at slice 2 (a handful of runs need 8 or more, and one never gets there) and with one component at slice 3 in {int(np.sum(PCS_NEEDED["3"] == 1))} of 48 runs. A single fixed direction, the one between the mean state of fitting and non-fitting examples, scores {fmt(ONE_DIR["2"])} at slice 2 and {fmt(ONE_DIR["3"])} at slice 3. That mean difference lies {fmt(E1_SHARE["2"])} along e₁ at slice 2 and {fmt(E1_SHARE["3"])} at slice 3, by squared length, with the rest along one off-axis direction.

{evidence_figure()}

So the evidence that an example fits is, to a first approximation, one vector with two parts: a component along e₁ that the anchor placed, and a component along one other direction that the loss made on its own. Both are there in every run, low or not, and nothing in the end state of the writer sets the low runs apart. What has to differ is how much the readers take from each part.
"""

# %%

rf"""
## Where the edit misses (E3)

Where does the reader take the rest of the evidence from in a low run? The edits here narrow the full edit to one slice, or go further than e₁ at one set of positions after block 2: every direction but e₁ removed at the example answers, so the next block sees the answers through e₁ alone, or the full edit with the whole state then removed at the answers or at the other example positions, so whatever the full edit left there is gone too. Where that last kind lifts a low run toward the normal runs under the same edit, the leftover was reaching the reader from those positions. Clearing a whole state is a diagnostic and not an intervention: it costs the other ops up to {num(max(CLEAR_SPILL["full and answers cleared, slice 2"].max(), CLEAR_SPILL["full and other example positions cleared, slice 2"].max()))} in expected match, against {num(CLEAR_SPILL["full"].max())} at most under the full edit.

{routes_figure()}

The full edit misses more in the low runs before any baseline: it takes {fmt(group(EVERY, LOW))} of the way against {fmt(group(EVERY, NORMAL))} in the rest, so the baseline is not what sets them apart. Removing e₁ at the three example answers alone does the whole of this: it removes as much as the full edit in every run, to within {num(ANSWERS_GAP)}, so the e₁ the full edit takes off the other positions makes no difference. No single slice accounts for the gap either. In most runs, low or not, the e₁ edit at slice 2 alone removes nearly as much as the full edit, which is the reader at block 3 of block1-edit; a few runs, all at seed 712, read at block 2 instead and are removed at slice 1.

The low runs come in two shapes. In {len(LOW_ANSWER)} of the {N_LOW}, the e₁ edit at slice 3 alone also removes much of the op, so block 4 reads it from the answers as well, and keeping only e₁ at the answers after block 2 costs them {fmt(group(ONLY_E1, LOW_ANSWER))} of the way: that reader uses the off-axis content of the answer states. The other {len(LOW_PLAIN)} look like normal runs in where the op is read (slice 2, block 3) and in the all-but-e₁ edit, which costs them nothing: block 3 takes e₁ when it is there. The clearing edits below say where both shapes get their leftover.

{low_table()}

In the twins, clearing the answers on top of the full edit brings the low run at seed 715 level with its twin and clearing the other example positions does not; at seed 719 it is the other way round. Across the ten low runs, one clearing or the other lifts the full edit by a tenth or more in {len(LIFTED)} ({len(LIFT_ANSWERS)} by the answers, {len(LIFT_OTHERS)} by the other positions, and {len([lab for lab in LIFT_ANSWERS if lab in LIFT_OTHERS])} by both), while in the normal runs neither clearing lifts it by more than a few hundredths. The {len(UNLIFTED)} it does not lift are the {len(UNLIFTED)} low runs with the smallest shortfall. So the leftover sits in the off-axis content of the example positions, answers or others by run, and the e₁ edit leaves it because it was never on e₁.

A reader after block 3 is not by itself what makes a run low: {len(NORMAL_ANSWER)} normal runs have one too, and remove the op well. What the low runs share is that their readers take enough of the evidence from off the axis for the e₁ edit to leave a fifth or so of it in place, by one route or another.

{group_table()}
"""

# %%

rf"""
## Peak and slide (E4)

When does a run become low? The fixed-labels report saw most low runs stay under the rest from early on and one slide after its peak. The kept checkpoints let removal and the off-axis evidence be followed together, with the term weights beside them.

{along_figure()}

Mostly the peak. The low runs peak lower ({fmt(group(PEAK, LOW))} against {fmt(group(PEAK, NORMAL))} for the rest, with the ranges barely touching). But the lines are hard to tell apart early on, and the two groups separate for good only from epoch {SCREEN_EPOCH}: before that, some normal runs are still climbing from a late start, below where some low runs will end. One or two low runs are still rising at the end.

{peak_figure()}

The slide is small and the peak-against-end figure shows it better than the lines do. The low runs give back about a tenth from their peak, against a few hundredths for the rest; {len(LOW_SLIDERS)} of the {N_LOW} give back more than a tenth, and no normal run does. The give-back happens while the anti weight anneals down and the anchor weight decays, and the one run the fixed-labels report saw slide, seed 715 at 1/50, peaks with the rest and is the clearest case.

The off-axis evidence is not what changes. The fits probe with e₁ zeroed is above {num(min(ALONG[lab][32]["probe"]["fits, slice 3"]["off-axis"] for lab in LABELS), ".1f")} by epoch 32 in every run but a few that learn the task late, and above {num(min(ALONG[lab][160]["probe"]["fits, slice 3"]["off-axis"] for lab in LABELS))} in every run from epoch 160, so the loss writes the evidence off e₁ about as early as the anchor settles, in low and normal runs alike. The mean over the low runs and the mean over the rest lie close together throughout (at epoch 64, {num(FITS_64["low"])} against {num(FITS_64["rest"])}), so the low runs do not hold more off-axis evidence; they read more of it. What moves during the anneal is how much the readers take from e₁, and nothing in the loss or the anchor term holds that where it was once the anti weight no longer presses on the other positions.
"""

# %%

rf"""
## Discussion

As far as the loss and the anchor term can see, a low run and its twin are the same solution: same loss, same alignment, and a different outcome under the edit. What differs is inside the reader blocks. The evidence that an example fits `difference` is one vector with a part along e₁, which the anchor put there, and a part along one other direction, which the loss made on its own; a reader can take the evidence from either part. The low runs' readers take more of it from the part off the axis. Whether that is a different weighting of the same two inputs or a reader built a little differently, these edits cannot tell; either way it is something we are not optimizing for, and which the model has to settle one way or the other. The anchor term asks for the e₁ part and the anti term keeps e₁ off the other positions, but no term asks the readers to use e₁, so training is flat along that choice. A run settles somewhere on it early, as the twins show, and then stays or drifts during the anneal. That fits the fixed-labels finding that neither the label set nor the seed decides which runs end low. It also says what a twitchy landscape means here: not a rugged loss, but a direction the loss does not see.

None of this is special to the fixed-label runs. The off-axis evidence is there in the per-visit runs too, and their readers also take some of it: the full edit leaves a few hundredths of the way in every run. The fixed-label runs lean on it more often, and more. The per-visit draw may spread less because the pull reaches nearly every `difference` context over a run, so the e₁ part is written on the whole corpus and the readers have less reason to prefer the other; a fixed set leaves most contexts with only the off-axis part during training. That is a guess that this re-analysis does not test.

If it holds, the lever is on the readers rather than on the writer. The most direct term would make the prediction depend on e₁: on the labelled contexts, a second forward pass with e₁ removed, trained toward the answer the op's absence would give, so that a run which reads from off the axis pays for it. That costs a forward pass per labelled context and needs nothing but the labels, so it would carry to real text; it would need its own experiment. A second anchored direction for the off-axis part is another, since the evidence has about two directions, though an earlier multi-axis anchor did not pay off and the reader could still prefer a third.

For scoring, the low state is invisible to the signals the loss-rate screen uses, so an early call cannot catch it. The e₁ edit itself cannot either until about half-way through training (epoch {SCREEN_EPOCH} on these runs), since some normal runs start late and sit under the low ones before then.
"""

# %%

r"""
## References

[fixed-labels](/docs/m2/fixed-labels/report.py)
:   With one label draw held for the run, the anchor settles at every rate, but a few runs in eight end well below the rest in removal, and which runs do so moves when a run is trained again with slightly different arithmetic.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed`, with the per-visit label draw, keeps the anchor on all eight runs at seeds 712 to 719 with removal near 0.9, and is the recipe of record for D2.2.

[block1-edit](/docs/m2/block1-edit/report.py)
:   Block 1 writes the evidence for the op onto e₁ at the example answers, one later block reads it from the query `=` through attention, and an edit removes the op when it sits between the two.

[ex-2.2.21](/docs/m2/ex-2.2.21/report.py)
:   The in-context grammar with three examples and its held-out set of 14,000 contexts, 2,000 per op.
"""
