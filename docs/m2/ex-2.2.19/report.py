# title: Ex 2.2.19: training length and seeds for the seven-op set

# The design constants come from `experiment.py` beside this script. The runs this draft builds on are published
# already: the 400-epoch `no-four` and `full` runs of ex-2.2.18, and the three ex-2.2.17 seeds of the same recipe on
# the full op set, which are the yardstick for a difference between runs.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter, NullLocator

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

FOUR = ex.OP_SET.name
YARDSTICK = ex.YARDSTICK
X = ex.ex2216
ALL_OPS: tuple[str, ...] = tuple(X.OP_NAMES)
OPS = ex.OP_SET.ops
HSV_CHANNEL = ("hue-hsv", "sat-hsv", "value-hsv")
REF_E = ex.REFERENCE_EPOCHS
# A context is confident when the posterior on its true op is above this, as in ex-2.2.17 and ex-2.2.18.
CONFIDENT = 0.99


# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored result table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- Loading ------------------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "not published yet"
    return json.loads(path.read_text())


# Arrays exist only for the runs the confirmation trained, so a candidate that was not trained comes back as None.
CANDIDATES = [ex.label_of(e, lr, s) for e in (*ex.SCOUT_EPOCHS, REF_E) for lr in ex.SCOUT_LRS for s in ex.CONFIRM_SEEDS]
with tempfile.TemporaryDirectory() as _tmp:
    _refs = [
        ex.EVAL_REF,
        ex.TRAJ_REF,
        ex.ex2218.EVAL_REF,
        ex.ex2218.TRAJ_REF,
        ex.ex2217.EVAL_REF,
        *(ex.EVAL_ARRAYS_REF.format(label=lbl) for lbl in CANDIDATES),
    ]
    _files = fetch(_refs, Path(_tmp))
    EVAL = read_json(_files[ex.EVAL_REF])
    TRAJ = read_json(_files[ex.TRAJ_REF])
    _eval18 = read_json(_files[ex.ex2218.EVAL_REF])
    SCOUT_18 = {r["label"]: r for r in _eval18["runs"]}
    TRAJ_18 = read_json(_files[ex.ex2218.TRAJ_REF])
    PRIOR = {r["label"]: r for r in read_json(_files[ex.ex2217.EVAL_REF])["runs"] if r["label"] in YARDSTICK}
    ARRAYS: dict[str, dict[str, np.ndarray]] = {}
    for _lbl in CANDIDATES:
        _f = _files[ex.EVAL_ARRAYS_REF.format(label=_lbl)]
        if _f is not None:
            with np.load(_f) as _z:
                ARRAYS[_lbl] = {k: _z[k] for k in _z.files}

SEL = EVAL["selection"]
STATS = EVAL["op_set"]
CEILING_18 = _eval18["op_sets"][FOUR]["ceiling"]
# The ex-2.2.18 `no-four` run is the 400-epoch run at the scout seed, so it joins the runs of this experiment.
REF_LABEL = ex.label_of(REF_E, ex.PEAK_LR, ex.SCOUT_SEED)
RUNS: dict[str, dict] = {r["label"]: r for r in EVAL["runs"]} | {REF_LABEL: SCOUT_18[FOUR]}
TRAJ = TRAJ | {REF_LABEL: TRAJ_18[FOUR]}
PICK: int = SEL["pick"]
SEEDS = ex.CONFIRM_SEEDS


def lab(epochs: int, seed: int = ex.SCOUT_SEED) -> str:
    """The label of the run at *epochs* and *seed*, at the peak rate the scout took at that length."""
    return ex.label_of(epochs, ex.PEAK_LR if epochs == REF_E else SEL["rate"][str(epochs)], seed)


def score(run: dict, key: str = "eem", op: str | None = None) -> float:
    r = run["task"][key]
    return r["all"] if op is None else r["per_op"][list(run.get("ops", ex.ex2216.OP_NAMES)).index(op)]


def gap(run: dict, op: str | None = None) -> float:
    return score(run, "ceiling", op) - score(run, "eem", op)


YARD_GAP = [gap(PRIOR[s]) for s in YARDSTICK]
YARD_RANGE = max(YARD_GAP) - min(YARD_GAP)
OP_TOL = ex.op_tolerances([PRIOR[s] for s in YARDSTICK], ex.OP_SET.ops)
COST_1 = len(ex.SCOUT_LRS) * sum(ex.cost_per_run(e) for e in ex.SCOUT_EPOCHS)
COST_2_MAX = len(ex.CONFIRM_SEEDS) * sum(
    ex.cost_per_run(e) for e in (ex.REFERENCE_EPOCHS, *ex.chosen_lengths(min(ex.SCOUT_EPOCHS)))
)


# --- Derived measurements -----------------------------------------------------------------------------------


def eem(label: str, op: str | None = None) -> float:
    return score(RUNS[label], "eem", op)


def shortfall(seed: int, op: str | None = None, epochs: int = PICK) -> float:
    """How far the run at *epochs* falls below the run at the reference length, at one model seed."""
    return eem(lab(REF_E, seed), op) - eem(lab(epochs, seed), op)


def listed(items) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f", and {items[-1]}"


LENGTHS = {x["epochs"]: x for x in SEL["lengths"]}
REF_EEM = SEL["reference_eem"]
CEILING = STATS["ceiling"]

SHORT = np.array([shortfall(s) for s in SEEDS])
SHORT_OP = np.array([[shortfall(s, op) for op in OPS] for s in SEEDS])
MEAN_SHORT = float(SHORT.mean())
MEAN_OP = dict(zip(OPS, SHORT_OP.mean(0), strict=True))
OPS_OVER = [op for op in OPS if MEAN_OP[op] > OP_TOL[op]]
H1_VERDICT = (
    "Miss" if OPS_OVER or MEAN_SHORT > ex.PARTIAL_TOL else "Pass" if MEAN_SHORT <= ex.SHORTFALL_TOL else "Partial"
)
assert H1_VERDICT == "Partial", "the H1 prose and verdict are written for a partial pass"
CLOSEST_OP = max(OPS, key=lambda op: MEAN_OP[op] / OP_TOL[op])

# Seed 0 has ex-2.2.18's `full` run as the partner; seeds 1 and 2 have ex-2.2.17's runs on the same initializations.
FULL_PARTNER = {0: SCOUT_18["full"], 1: PRIOR[YARDSTICK[1]], 2: PRIOR[YARDSTICK[2]]}
GAP_FOUR = {s: gap(RUNS[lab(REF_E, s)]) for s in (0, *SEEDS)}
GAP_FULL = {s: gap(r) for s, r in FULL_PARTNER.items()}
assert all(GAP_FOUR[s] < GAP_FULL[s] for s in GAP_FULL), "the H2 prose and verdict are written for a pass"

EEM_FOUR_200 = [eem(lab(PICK, s)) for s in (0, *SEEDS)]
EEM_FOUR_400 = [eem(lab(REF_E, s)) for s in (0, *SEEDS)]
EEM_FULL_400 = [score(PRIOR[y]) for y in YARDSTICK]


def spread(v) -> float:
    return float(max(v) - min(v))


def hsv_mean(label: str) -> float:
    """The mean held-out EEM over the three HSV-channel ops."""
    return float(np.mean([eem(label, op) for op in HSV_CHANNEL]))


def hsv_short(length: int) -> float:
    """The mean shortfall over the three HSV-channel ops of the scout run at *length*."""
    return float(np.mean([LENGTHS[length]["shortfall_per_op"][op] for op in HSV_CHANNEL]))


def other_short(length: int) -> float:
    """The mean shortfall over the other four ops of the scout run at *length*."""
    return float(np.mean([v for op, v in LENGTHS[length]["shortfall_per_op"].items() if op not in HSV_CHANNEL]))


def rate_effect(length: int) -> float:
    """EEM at the higher peak rate minus EEM at the recipe rate, at the scout seed."""
    lo, hi = (eem(ex.label_of(length, lr, ex.SCOUT_SEED)) for lr in ex.SCOUT_LRS)
    return hi - lo


assert rate_effect(50) > 0 and rate_effect(100) > 0 > rate_effect(PICK), "the S1 prose is written for this pattern"


assert LENGTHS[PICK]["shortfall"] < SHORT.min(), "the H1 prose says the scout seed was flattered"


def hsv_rate_effect(length: int) -> float:
    lo, hi = (hsv_mean(ex.label_of(length, lr, ex.SCOUT_SEED)) for lr in ex.SCOUT_LRS)
    return hi - lo


def kl(label: str) -> float:
    return RUNS[label]["task"]["kl"]["all"]


# --- Skill curves -------------------------------------------------------------------------------------------


def traj_skill(label: str) -> np.ndarray:
    """Skill on the probe set along a run: the row for all ops, then one per op; shape (op, point)."""
    t = TRAJ[label]
    e = np.array(t["eem_per_op"]).T
    f, c = np.array(STATS["floor_per_op"])[:, None], np.array(STATS["ceiling_per_op"])[:, None]
    total = (np.array(t["eem"]) - STATS["floor"]) / (CEILING - STATS["floor"])
    return np.vstack([total, (e - f) / (c - f)])


def hsv_curve(label: str) -> np.ndarray:
    """Mean skill over the three HSV-channel ops along a run."""
    return traj_skill(label)[[1 + OPS.index(op) for op in HSV_CHANNEL]].mean(0)


def frac_done(label: str) -> np.ndarray:
    ep = np.array(TRAJ[label]["epoch"])
    return ep / ep[-1]


HSV_HALF = 0.5


def hsv_rise(label: str) -> tuple[float, float]:
    """The first logged epoch at which the HSV-channel skill passes *HSV_HALF*, and the learning rate there."""
    reached = hsv_curve(label) >= HSV_HALF
    if not reached.any():
        return float("nan"), float("nan")
    i = int(np.argmax(reached))
    return float(TRAJ[label]["epoch"][i]), float(TRAJ[label]["lr"][i])


def end_skill(label: str, ops: Sequence[str]) -> float:
    """Skill at the end of the run, averaged over *ops*, on the probe set."""
    return float(traj_skill(label)[[1 + OPS.index(op) for op in ops], -1].mean())


OTHER_OPS = tuple(op for op in OPS if op not in HSV_CHANNEL)


def span(values, fmt: str = ".2f") -> str:
    """The range of *values*, as "low to high"."""
    values = list(values)
    return f"{min(values):{fmt}} to {max(values):{fmt}}"


def hsv_at_fraction(label: str, q: float) -> float:
    return float(hsv_curve(label)[int(np.argmin(abs(frac_done(label) - q)))])


# --- Op confusion ---------------------------------------------------------------------------------------------


@memo
def answer_table():
    """The answers of all eleven ops on every pair, so the answers of a dropped op stay defined."""
    return X._get_posterior().build_table(X.TABLE)


@memo
def op_confusion(p16: np.ndarray, op_ids: np.ndarray, post: np.ndarray, pair: np.ndarray, ops: tuple[str, ...]):
    """Where a run puts its mass on confident contexts, by op, as in ex-2.2.17. Rows are the true op and columns
    all eleven ops, in the order of ALL_OPS. The diagonal is the mass on the colors the true op can give for the
    query operands; off the diagonal, the mass on the colors the column op can give and the true op cannot.
    """
    table = answer_table()
    rows = np.arange(len(p16))
    conf = post[rows, op_ids] > CONFIDENT
    p, pair = p16[conf].astype(float), pair[conf]
    true_op = np.array([ALL_OPS.index(ops[i]) for i in op_ids[conf]])
    rows = np.arange(len(p))
    gives = np.zeros((len(ALL_OPS), *p.shape), dtype=bool)
    for o in range(len(ALL_OPS)):
        idx, ok = table.idx[o, pair], table.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            gives[o, rows[ok[:, j]], idx[ok[:, j], j]] = True
    true = gives[true_op, rows]
    per_col = np.stack(
        [(p * np.where((true_op == o)[:, None], true, gives[o] & ~true)).sum(1) for o in range(len(ALL_OPS))], 1
    )
    m = np.full((len(ALL_OPS), len(ALL_OPS)), np.nan)
    for o in range(len(ALL_OPS)):
        if (true_op == o).any():
            m[o] = per_col[true_op == o].mean(0)
    return m


_seven = np.ix_([ALL_OPS.index(o) for o in OPS], [ALL_OPS.index(o) for o in OPS])
# The confusion at each length: the mean over the fresh seeds, on the seven ops of the set.
CONFUSION = {
    e: np.mean(
        [
            op_confusion(a["p"], a["op_ids"], a["posterior"], a["query_pair"], tuple(OPS))[_seven]
            for a in (ARRAYS[lab(e, s)] for s in SEEDS)
        ],
        axis=0,
    )
    for e in (PICK, REF_E)
}
OFF_DIAGONAL = ~np.eye(len(OPS), dtype=bool)
assert abs(CEILING - CEILING_18) < 1e-9, "the rebuilt corpus differs from ex-2.2.18's"


def off_mass(e: int) -> float:
    """The mean over the seven rows of the mass off the diagonal at length *e*."""
    return float(CONFUSION[e][OFF_DIAGONAL].sum() / len(OPS))


def row_off(e: int, ops: Sequence[str]) -> float:
    """The mean over the rows of *ops* of the mass off the diagonal at length *e*."""
    return float(np.mean([CONFUSION[e][OPS.index(op)][OFF_DIAGONAL[OPS.index(op)]].sum() for op in ops]))


assert all(
    CONFUSION[PICK][i][OFF_DIAGONAL[i]].sum() > CONFUSION[REF_E][i][OFF_DIAGONAL[i]].sum() for i in range(len(OPS))
), "the E3 prose says the shorter run puts more mass off the diagonal in every row"
assert row_off(PICK, HSV_CHANNEL) - row_off(REF_E, HSV_CHANNEL) > row_off(PICK, OTHER_OPS) - row_off(REF_E, OTHER_OPS)


# Confusion values at or above this are printed; every off-diagonal value is below 0.03.
PRINT_FLOOR = 0.01


def worst_square(e: int) -> tuple[str, str, float]:
    m = np.where(OFF_DIAGONAL, CONFUSION[e], -1)
    i, j = np.unravel_index(np.argmax(m), m.shape)
    return OPS[i], OPS[j], float(m[i, j])


# REVIEW: E4 and the adoption paragraph said the spread over seeds is "similar" at 200 and 400 epochs; the 200-epoch
# range is 1.6x the 400-epoch one, and E4 itself says four seeds can hardly resolve it. Now "four seeds do not resolve a
# difference in spread". Verify against spread(EEM_FOUR_200) / spread(EEM_FOUR_400).
# --- Figures ------------------------------------------------------------------------------------------------


def rule_color() -> str:
    return light_dark("#333", "#ddd")


ALL_LENGTHS = (*ex.SCOUT_EPOCHS, REF_E)


def shade(i: int, n: int = 4) -> tuple:
    """The color of the *i*-th of *n* ordered lengths: one map, darker (or brighter, in the dark theme) for longer."""
    lo, hi = light_dark((0.3, 1.0), (0.35, 0.95))
    return plt.get_cmap(light_dark("Blues", "magma"))(lo + (hi - lo) * i / (n - 1))


def seq_cmap():
    cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
    cmap.set_bad(light_dark("#fff", "#111"))
    return cmap


def cell_text_color(v: float, vmax: float) -> str:
    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
    return "#fff" if dark_cell else "#000"


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line, a dotted partial level under it when there is one, and the failing side hatched."""
    ax.axhline(y, color=rule_color(), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=rule_color(), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        span = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
        ax.set_ylim(lo, hi)


def dots(ax: Axes, x: float, v, color, *, rng, marker: str = "o", ms: float = 5.0) -> None:
    """One column of per-seed dots with the seed mean on top; a thin bar behind spans the seed range."""
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.06, 0.06, len(v)), v, "o", ms=2.6, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), marker, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


@memo
def scout_draw(alt_text: str, caption: str) -> str:
    @themed(name="scout", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
        floor = STATS["floor"]
        ax.set_xscale("log")
        ax.set_ylim(0.3, 0.65)
        for i, lr in enumerate(ex.SCOUT_LRS):
            # Only the recipe rate was trained to the reference length.
            reaches_ref = lr == ex.PEAK_LR
            xs = [*ex.SCOUT_EPOCHS, *([REF_E] if reaches_ref else [])]
            ys = [eem(ex.label_of(e, lr, ex.SCOUT_SEED)) for e in ex.SCOUT_EPOCHS]
            ys += [eem(REF_LABEL)] if reaches_ref else []
            ax.plot(xs, ys, "-", color=f"C{i}", lw=1.2, label=f"peak rate {lr:g}")
            for e, y in zip(ex.SCOUT_EPOCHS, ys, strict=False):
                taken = SEL["rate"][str(e)] == lr
                ax.plot(e, y, "o", ms=5.5, color=f"C{i}", mfc=f"C{i}" if taken else "none", mew=1.2, zorder=3)
        ax.plot(REF_E, eem(REF_LABEL), "o", ms=5.5, color=rule_color(), zorder=4)
        ax.plot(PICK, eem(lab(PICK)), "o", ms=12, mfc="none", mec=rule_color(), mew=1.0, zorder=4)
        ax.axhline(CEILING, ls="--", color=rule_color(), lw=0.9)
        ax.annotate("Bayes ceiling", (52, CEILING), xytext=(0, 3), textcoords="offset points", fontsize=8)
        ax.set_xticks(ALL_LENGTHS, [str(e) for e in ALL_LENGTHS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("epochs")
        ax.set_ylabel("held-out EEM")
        ax.secondary_yaxis(
            "right", functions=(lambda y: (y - floor) / (CEILING - floor), lambda s: floor + s * (CEILING - floor))
        ).set_ylabel("skill")
        gate_line(ax, REF_EEM - ex.SHORTFALL_TOL, fail="below")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


@memo
def shortfall_draw(alt_text: str, caption: str) -> str:
    @themed(name="shortfall", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.4), layout="constrained", sharey=True, width_ratios=[1, 4.5])
        rng = np.random.default_rng(0)
        ink = shade(2)
        dots(a, 0, SHORT, ink, rng=rng, ms=7)
        a.set_xticks([0], ["all ops"])
        a.set_xlim(-0.8, 0.8)
        a.set_ylabel(f"EEM at {REF_E} epochs minus EEM at {PICK}")
        a.set_ylim(-0.005, 0.09)
        for j, op in enumerate(OPS):
            dots(b, j, SHORT_OP[:, j], ink, rng=rng, ms=6)
            b.plot([j - 0.3, j + 0.3], [OP_TOL[op]] * 2, "-", color=rule_color(), lw=1.6, zorder=5)
        b.set_xticks(range(len(OPS)), OPS, fontsize=8)
        b.set_xlim(-0.6, len(OPS) - 0.4)
        for ax in (a, b):
            ax.axhline(0, color="0.6", lw=0.6)
        gate_line(a, ex.SHORTFALL_TOL, partial=ex.PARTIAL_TOL, fail="above")
        return fig

    return _plot()


@memo
def pairs_draw(alt_text: str, caption: str) -> str:
    @themed(name="pairs", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
        for s in FULL_PARTNER:
            ax.plot([s, s], [GAP_FOUR[s], GAP_FULL[s]], "-", color="0.6", lw=1.0, zorder=1)
        ax.plot(list(GAP_FULL), list(GAP_FULL.values()), "o", mfc="none", mec="C1", mew=1.5, ms=7, label="full")
        ax.plot(list(GAP_FOUR), list(GAP_FOUR.values()), "o", color="C0", ms=7, label=FOUR)
        ax.set_xticks(list(GAP_FOUR), [str(ex.SEED_OFFSET + s) for s in GAP_FOUR])
        ax.set_xlim(-0.5, max(GAP_FOUR) + 0.5)
        ax.set_ylim(0, None)
        ax.set_xlabel("model seed")
        ax.set_ylabel(f"gap to own ceiling at {REF_E} epochs")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


@memo
def curves_draw(alt_text: str, caption: str) -> str:
    @themed(name="curves", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2), layout="constrained", sharex=True, sharey=True)
        for k, (ax, name) in enumerate(zip(axes.flat, ["all ops", *OPS], strict=True)):
            ax = cast(Axes, ax)
            for i, e in enumerate(ALL_LENGTHS):
                for s in SEEDS if e in (PICK, REF_E) else ():
                    ax.plot(frac_done(lab(e, s)), traj_skill(lab(e, s))[k], color=shade(i), lw=0.6, alpha=0.5)
                ax.plot(frac_done(lab(e)), traj_skill(lab(e))[k], color=shade(i), lw=1.5, label=f"{e} epochs")
            ax.axhline(1, ls="--", color=rule_color(), lw=0.8)
            ax.set_title(name, fontsize=9)
            ax.set_ylim(-0.1, 1.05)
        fig.supxlabel("fraction of training completed", fontsize=9)
        fig.supylabel("skill on the probe set", fontsize=9)
        fig.legend(
            *axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncols=4, frameon=False, fontsize=8
        )
        return fig

    return _plot()


@memo
def calibration_draw(alt_text: str, caption: str) -> str:
    @themed(name="calibration", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
        for label, r in RUNS.items():
            e = r["epochs"]
            taken = r["peak_lr"] == (ex.PEAK_LR if e == REF_E else SEL["rate"][str(e)])
            c = shade(ALL_LENGTHS.index(e))
            ax.plot(eem(label), r["task"]["kl"]["all"], "o", ms=6, color=c, mfc=c if taken else "none", mew=1.3)
        for i, e in enumerate(ALL_LENGTHS):
            ax.plot([], [], "o", color=shade(i), label=f"{e} epochs")
        ax.set_xlabel("held-out EEM")
        ax.set_ylabel("calibration KL")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=4, frameon=False, fontsize=8)
        return fig

    return _plot()


@memo
def confusion_draw(alt_text: str, caption: str) -> str:
    @themed(name="confusion", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), layout="constrained", sharex=True, sharey=True)
        n = len(OPS)
        eye = np.eye(n, dtype=bool)
        vmax = max(float(np.nanmax(np.where(eye, np.nan, CONFUSION[e]))) for e in CONFUSION)
        im = None
        for ax, e in zip(axes, (PICK, REF_E), strict=True):
            m = CONFUSION[e]
            im = ax.imshow(np.where(eye, np.nan, m), cmap=seq_cmap(), vmin=0, vmax=vmax)
            for i, j in np.ndindex(n, n):
                if i == j:
                    ax.plot(j, i, "s", ms=13, mfc="none", mec="0.6", mew=0.6)
                elif m[i, j] >= PRINT_FLOOR:
                    color = cell_text_color(m[i, j], vmax)
                    ax.text(j, i, f"{m[i, j]:.2f}"[1:], ha="center", va="center", fontsize=7, color=color)
            ax.set_title(f"{e} epochs", fontsize=9)
            ax.set_xticks(range(n), OPS, rotation=90, fontsize=8)
            ax.set_yticks(range(n), OPS, fontsize=8)
        fig.supxlabel("answers of this op, beyond those of the true op", fontsize=9)
        fig.supylabel("true op", fontsize=9)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.6, label="mass")
        return fig

    return _plot()


@memo
def spread_draw(alt_text: str, caption: str) -> str:
    @themed(name="spread", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
        rng = np.random.default_rng(0)
        cols = [
            (f"{FOUR}, {PICK} epochs", EEM_FOUR_200, "C0", "o"),
            (f"{FOUR}, {REF_E} epochs", EEM_FOUR_400, "C2", "s"),
            (f"full, {REF_E} epochs", EEM_FULL_400, "C1", "D"),
        ]
        for x, (_, v, c, m) in enumerate(cols):
            dots(ax, x, v, c, rng=rng, marker=m, ms=7)
        ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=8)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylabel("held-out EEM")
        return fig

    return _plot()


rf"""

# Ex 2.2.19: training length and seeds for the seven-op set

/// tip |
<!-- tl;dr -->
We looked for the shortest training run on the seven-op set that keeps most of the skill of the 400-epoch recipe. A scout at one seed picked a length, and three fresh seeds checked it.
Half the length keeps most of the skill, though it falls a little short of the gate we set; we adopt it. Runs of 50 and 100 epochs do not keep enough of the skill. The seven-op set also stays closer to its ceiling than the full set at every paired seed, so its narrower gap was not seed variation.
///

## Findings

- [The scout (S1)](#the-scout-s1) — the rule picked {PICK} epochs: at the scout seed that run fell short of the {REF_E}-epoch run by {LENGTHS[PICK]["shortfall"]:.3f}, against {ex.SHORTFALL_TOL}, with no op over its tolerance. The 50- and 100-epoch runs fell short by {LENGTHS[50]["shortfall"]:.3f} and {LENGTHS[100]["shortfall"]:.3f}.
- [A shorter run keeps most of the skill (H1)](#a-shorter-run-keeps-most-of-the-skill-h1) — partial pass. On the fresh seeds the {PICK}-epoch run falls short by {MEAN_SHORT:.4f} on average, just over the gate of {ex.SHORTFALL_TOL} and inside the partial band to {ex.PARTIAL_TOL}. No op is over its tolerance, though `{CLOSEST_OP}` is within {OP_TOL[CLOSEST_OP] - MEAN_OP[CLOSEST_OP]:.4f} of it. We adopt {PICK} epochs.
- [The seven-op set stays closer to its ceiling (H2)](#the-seven-op-set-stays-closer-to-its-ceiling-h2) — pass. `no-four` has the smaller gap at all three paired seeds, {span([GAP_FOUR[s] for s in FULL_PARTNER], ".3f")} against {span(GAP_FULL.values(), ".3f")} for `full`.
- [Skill curves (E1)](#skill-curves-e1) — the HSV-channel ops take off at about the same epoch in the {PICK}- and {REF_E}-epoch runs ({span([hsv_rise(lab(e, s))[0] for e in (PICK, REF_E) for s in (0, *SEEDS)], ".0f")}). So they come late in the shorter run, and not at all in runs that end before that epoch.
- [Calibration (E2)](#calibration-e2) — calibration KL at {PICK} epochs ({span([kl(lab(PICK, s)) for s in (0, *SEEDS)], ".3f")}) matches {REF_E} epochs ({span([kl(lab(REF_E, s)) for s in (0, *SEEDS)], ".3f")}).
- [Op confusion (E3)](#op-confusion-e3) — the {PICK}-epoch run puts more mass off the diagonal in every row, mostly in the HSV-channel rows, with no single similar op taking it.
- [Spread over seeds (E4)](#spread-over-seeds-e4) — the EEM range over four seeds is {spread(EEM_FOUR_200):.3f} at {PICK} epochs and {spread(EEM_FOUR_400):.3f} at {REF_E}, against {spread(EEM_FULL_400):.3f} over the three `full` seeds.

/// admonition | How to read this report
Preregistered: the selection rule, the hypotheses, and their gates were frozen at commit `0e0a17c0`, before any run of this experiment. Results replaced the placeholders in place, and analyses conceived after seeing the data are marked post hoc.
///

## Why

Ex-2.2.18 trained the center control on several smaller op sets, one run each. Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` gave a seven-op set, `no-four`, with a higher Bayes ceiling (defined [below](#the-runs)). The model came within {gap(SCOUT_18[FOUR]):.3f} of that ceiling, against {gap(SCOUT_18["full"]):.3f} for the full set. But that was one seed. The three ex-2.2.17 seeds of the same recipe on the full set had gaps spread over {YARD_RANGE:.3f}, so the difference could be seed variation.

We will probably adopt `no-four` for the anchoring experiments that follow, and each of those costs more the longer its runs: about \${ex.cost_per_run(ex.REFERENCE_EPOCHS):.2f} for 400 epochs.

The skill curves of ex-2.2.18 leveled off well before the end: the last fifth of training added at most about {ex.SHORTFALL_TOL} of held-out expected exact match, which suggests a shorter run could do nearly as well. But the learning-rate schedule is a cosine, so a shorter run anneals sooner,[^anneal] and its curve is not just a truncated longer one.

[^anneal]: The learning rate decays along a cosine curve that reaches its low point at the end of the run, so a shorter run lowers its rate earlier, rather than stopping partway down the curve of a longer run.

A scout trains `no-four` at shorter lengths from the initialization of the ex-2.2.18 run, and a rule fixed in advance picks a length. Three fresh seeds then train at that length and at 400 epochs, and the preregistered comparison uses those seeds alone: the scout seed helped pick the length, so it would flatter the pick.

The fresh seeds also give the first replicates of `no-four` at 400 epochs, so we can check the narrower gap of ex-2.2.18 as well.

## The runs

Every run uses the recipe of ex-2.2.17 and ex-2.2.18, on the `no-four` corpus that ex-2.2.18 built, with a different number of epochs. At each shorter length the scout also tries a higher peak learning rate, since a shorter run may want one. The warmup (the opening stretch in which the learning rate ramps up to its peak) stays at {ex.WARMUP_EPOCHS:g} epochs at every length.

"""

table_html(
    ["stage", "epochs", "peak learning rate", "model seeds", "runs"],
    [
        ["ex-2.2.18 (reused)", f"{ex.REFERENCE_EPOCHS}", f"{ex.PEAK_LR:g}", f"{ex.SEED_OFFSET + ex.SCOUT_SEED}", "1"],
        [
            "1: scout",
            ", ".join(str(e) for e in ex.SCOUT_EPOCHS),
            ", ".join(f"{lr:g}" for lr in ex.SCOUT_LRS),
            f"{ex.SEED_OFFSET + ex.SCOUT_SEED}",
            str(len(ex.SCOUT_EPOCHS) * len(ex.SCOUT_LRS)),
        ],
        [
            "2: confirmation",
            f"the pick T, 2T if shorter than {ex.REFERENCE_EPOCHS}, and {ex.REFERENCE_EPOCHS}",
            f"the better scout rate at T and 2T; {ex.PEAK_LR:g} at {ex.REFERENCE_EPOCHS}",
            f"{ex.SEED_OFFSET + min(ex.CONFIRM_SEEDS)}–{ex.SEED_OFFSET + max(ex.CONFIRM_SEEDS)}",
            f"up to {len(ex.CONFIRM_SEEDS) * (1 + ex.MAX_CONFIRM_LENGTHS)}",
        ],
    ],
    f"""
        **The runs.** All train `no-four`. The scout costs about \\${COST_1:.2f} and the confirmation at most about
        \\${COST_2_MAX:.2f}, scaled from the cost of an ex-2.2.18 run.
    """,
    text_cols=4,
)

rf"""

Model seeds {ex.SEED_OFFSET + 1} and {ex.SEED_OFFSET + 2} are the initializations of ex-2.2.17's `{YARDSTICK[1]}` and `{YARDSTICK[2]}`, and seed {ex.SEED_OFFSET} of `{YARDSTICK[0]}` and of the ex-2.2.18 `full` run. So each 400-epoch `no-four` run at those seeds has a `full` run that started from the same weights. Seed {ex.SEED_OFFSET + 3} has no such partner.

Held-out *expected exact match* (EEM) is the probability that an answer drawn from the model at the query `=` is a correct answer of the true op. The *Bayes ceiling* is the same score for an ideal predictor, one that weighs each op by how well it explains the examples. The *gap* is the ceiling minus EEM. All three are computed on the op set of the run, as in ex-2.2.18.

Given a fixed corpus, the ceiling is the same at every length, so a difference in gap between two lengths is a difference in EEM. A run *falls short* of another by how much lower its EEM is. We compare runs that share a model seed, and average those paired differences over seeds.

## The scout (S1)

The scout is a procedure, with no hypothesis; the rule below picks the length that stage 2 confirms.

**The rule.** At each scout length, take the run at the peak rate with the higher EEM. Of those runs, pick the shortest, T, that falls short of the ex-2.2.18 run by at most {ex.SHORTFALL_TOL}, with no op falling short by more than its own tolerance (below). If no scout length passes, stage 2 trains only the 400-epoch runs, and H1 is unresolved.

Stage 2 confirms T, and also 2T when 2T is shorter than {ex.REFERENCE_EPOCHS} epochs, so that a lucky pass at T still leaves a length to adopt. Each trains at the rate taken at its length.

The overall tolerance is the largest last-fifth gain of any ex-2.2.18 run. The tolerance for each op is its seed range in the three ex-2.2.17 runs, or {ex.PARTIAL_TOL}, whichever is larger, since some ops varied little across three seeds and a single scout run is noisier than that.

**What we expect.** At 50 epochs the run falls short by more than the tolerance: ex-2.2.17 needed several times that length for the HSV-channel ops. We don't have a strong expectation between 100 and 200 epochs. We expect the higher rate to help more the shorter the run, and at 200 epochs to make little difference. The HSV-channel ops may be the exception: in ex-2.2.17 they were learned while the rate passed down through about {ex.ex2217.HOLD_LR:g}, during the cosine, so a higher peak only delays that stretch, and a short run has the least time to spare.

**What we saw.** The rule picks {PICK} epochs. At 50 epochs the run falls short of the {REF_E}-epoch run by {LENGTHS[50]["shortfall"]:.3f}, with all {len(LENGTHS[50]["ops_over"])} ops over their tolerance, and at 100 epochs by {LENGTHS[100]["shortfall"]:.3f}, with {listed(f"`{op}`" for op in LENGTHS[100]["ops_over"])} over. At {PICK} epochs the shortfall is {LENGTHS[PICK]["shortfall"]:.3f} and no op is over. Stage 2 trains {PICK} epochs and the reference; 2T is {2 * PICK}, the reference length, so nothing between is trained.

"""

scout_draw(
    f"""
        A line chart of held-out expected exact match against epochs on a log axis, for two peak learning rates. Both
        lines rise, from {eem(ex.label_of(50, ex.SCOUT_LRS[0], 0)):.2f} and {eem(ex.label_of(50, ex.SCOUT_LRS[1], 0)):.2f} at
        50 epochs, and the line for the
        recipe rate goes on to {eem(REF_LABEL):.2f} at {REF_E}. The higher rate is above the lower at 50 and 100 epochs and below it at {PICK}. A
        dashed line marks the Bayes ceiling at {CEILING:.2f}, and a hatched region below {REF_EEM - ex.SHORTFALL_TOL:.3f}
        marks where a run falls short of the tolerance. Apart from the reference, only the {PICK}-epoch run, ringed, is above that level.
    """,
    f"""
        **Held-out EEM by length at the scout seed.** One line per peak rate. Only the recipe rate was trained for
        {REF_E} epochs, in ex-2.2.18. Filled dots mark the rate the rule takes at each length. The dashed line is the Bayes ceiling
        (skill on the right axis), and the hatched region is more than {ex.SHORTFALL_TOL} below the {REF_E}-epoch run.
        The pick is ringed.
    """,
)

# %%
table_html(
    ["epochs", "rate", "all ops", *(f"`{op}`" for op in OPS)],
    [
        [
            f"{e}{' (pick)' if e == PICK else ''}",
            f"{SEL['rate'][str(e)]:g}",
            f"<b>{x['shortfall']:.3f}</b>" if x["shortfall"] <= ex.SHORTFALL_TOL else f"{x['shortfall']:.3f}",
            *(
                f"<b>{x['shortfall_per_op'][op]:.3f}</b>"
                if op not in x["ops_over"]
                else f"{x['shortfall_per_op'][op]:.3f}"
                for op in OPS
            ),
        ]
        for e, x in LENGTHS.items()
    ]
    + [["tolerance", "", f"{ex.SHORTFALL_TOL}", *(f"{OP_TOL[op]:.3f}" for op in OPS)]],
    """
        **Shortfall of each scout run from the ex-2.2.18 run.** EEM at 400 epochs minus EEM at the length, for all ops
        and for each op; bold marks a value within the tolerance in the last row.
    """,
    text_cols=2,
)

rf"""

The jump between 100 and 200 epochs is almost all in the HSV-channel ops. Their mean shortfall falls from {hsv_short(100):.3f} to {hsv_short(PICK):.3f}, against {other_short(100):.3f} to {other_short(PICK):.3f} for the other four. This is the caveat we wrote in advance, that a short run has the least time to spare for these ops.

The higher rate helped as expected at the shortest lengths: EEM was {rate_effect(50):.3f} higher at 50 epochs and {rate_effect(100):.3f} higher at 100. At {PICK} epochs it was {abs(rate_effect(PICK)):.3f} lower, more than the little difference we expected. The HSV-channel ops were not an exception to the pattern: the higher rate lifted them by {hsv_rate_effect(100):.3f} at 100 epochs, more than the whole set, and cost them {abs(hsv_rate_effect(PICK)):.3f} at {PICK}. The rate taken is therefore {SEL["rate"]["50"]:g} at 50 and 100 epochs and {SEL["rate"][str(PICK)]:g} at {PICK}.

/// admonition | Partial
The rule picked {PICK} epochs, and the runs of 50 and 100 epochs fell short as expected. At {PICK} epochs the higher rate lowered EEM by more than the little difference we expected.
///


## A shorter run keeps most of the skill (H1)

**What we expect.** On the fresh seeds, the run at T falls short of the run at {ex.REFERENCE_EPOCHS} epochs by at most {ex.SHORTFALL_TOL} on average, and no op falls short by more than its tolerance on average. That is a pass.

A shortfall between {ex.SHORTFALL_TOL} and {ex.PARTIAL_TOL} (the seed range of ex-2.2.17) is a partial pass. A shortfall beyond {ex.PARTIAL_TOL}, or an op beyond its tolerance, is a miss for H1, and would mean the scout result was flattered by its seed. 2T, when trained, gets a verdict by the same rule.

**Which length we adopt.** The verdicts inform this choice without settling it, since a rule written now can't anticipate everything that might matter: a short run that matches on EEM could be poorly calibrated ([E2](#calibration-e2)), or spread more widely over seeds ([E4](#spread-over-seeds-e4)). We expect to adopt the shortest confirmed length that passes, and will say why if we choose otherwise.

**What we saw.** On the fresh seeds the {PICK}-epoch run falls short of the {REF_E}-epoch run by {SHORT[0]:.4f}, {SHORT[1]:.4f}, and {SHORT[2]:.4f} at model seeds {ex.SEED_OFFSET + SEEDS[0]}, {ex.SEED_OFFSET + SEEDS[1]}, and {ex.SEED_OFFSET + SEEDS[2]}. The mean is {MEAN_SHORT:.4f}, {MEAN_SHORT - ex.SHORTFALL_TOL:.4f} over the gate and well inside the partial band. The shortfall at the scout seed, {LENGTHS[PICK]["shortfall"]:.4f}, was smaller than any of the three, so the scout was flattered by its seed, as the design anticipated. 2T is the reference length, so there is no second length to score.

No op falls short by more than its tolerance on average. `{CLOSEST_OP}` is closest, at {MEAN_OP[CLOSEST_OP]:.4f} against {OP_TOL[CLOSEST_OP]:.3f}; the three HSV-channel ops have the largest shortfalls, averaging {np.mean([MEAN_OP[op] for op in HSV_CHANNEL]):.3f} against {np.mean([MEAN_OP[op] for op in OTHER_OPS]):.3f} for the other four.

"""

shortfall_draw(
    f"""
        Two dot plots of the shortfall in held-out EEM between the {PICK}-epoch and {REF_E}-epoch runs, one dot per
        model seed. On the left, all ops together: three dots between {SHORT.min():.3f} and {SHORT.max():.3f}, with
        the mean just above a dashed gate at {ex.SHORTFALL_TOL}; a hatched region above the gate runs up past a
        dotted line at {ex.PARTIAL_TOL}. On the right, one column per op, each with a horizontal bar at its
        tolerance. The mean of every column is below its bar; {CLOSEST_OP} comes closest.
    """,
    f"""
        **Shortfall of the {PICK}-epoch run from the {REF_E}-epoch run, by model seed.** The larger dot is
        the seed mean and the thin bar spans the seeds. Left: all ops, against the gate (dashed) and the end of the
        partial band (dotted); the hatched side misses. Right: each op, against its tolerance (horizontal bar).
    """,
)

# %%
table_html(
    ["", "all ops", *(f"`{op}`" for op in OPS)],
    [
        ["mean shortfall", f"{MEAN_SHORT:.4f}", *(f"{MEAN_OP[op]:.4f}" for op in OPS)],
        ["tolerance", f"{ex.SHORTFALL_TOL}", *(f"{OP_TOL[op]:.3f}" for op in OPS)],
    ],
    """
        **Mean shortfall by op against the tolerance.** The gate arithmetic behind the figure.
    """,
)

rf"""

**The length we adopt.** We adopt {PICK} epochs, at the peak rate {ex.PEAK_LR:g}. The preregistered plan was to adopt the shortest confirmed length that passes, and to say why if we chose otherwise. H1 is a partial pass and 2T is the reference length, so no shorter length passes outright; we made this choice after seeing the results. The cost is about {MEAN_SHORT:.3f} of EEM, larger on the HSV-channel ops. Against that, the calibration KL at {PICK} epochs matches {REF_E} epochs ([E2](#calibration-e2)); four seeds do not resolve a difference in spread ([E4](#spread-over-seeds-e4)); and every later run costs about \${ex.cost_per_run(PICK):.2f} instead of \${ex.cost_per_run(REF_E):.2f}.

/// admonition | Partial
The {PICK}-epoch run falls short of the {REF_E}-epoch run by {MEAN_SHORT:.4f} on average against the gate of {ex.SHORTFALL_TOL}, inside the partial band to {ex.PARTIAL_TOL}, and no op is over its tolerance.
///


## The seven-op set stays closer to its ceiling (H2)

**What we expect.** At {ex.REFERENCE_EPOCHS} epochs, `no-four` has a smaller gap than `full` at each of the three model seeds where both exist ({ex.SEED_OFFSET}, {ex.SEED_OFFSET + 1}, and {ex.SEED_OFFSET + 2}). This is a check on ex-2.2.18, with no gate, since we are likely to adopt `no-four` for its higher ceiling either way. If the gaps overlap, the narrower gap of ex-2.2.18 was seed variation.

The `full` runs are reused: those at seeds {ex.SEED_OFFSET + 1} and {ex.SEED_OFFSET + 2} are from ex-2.2.17 and trained on the ex-2.2.16 corpus, and the one at seed {ex.SEED_OFFSET} is from ex-2.2.18, on a corpus rebuilt the same way, so the pairs share a starting point and differ in corpus draw as well as op set.

**What we saw.** `no-four` has the smaller gap than `full` at all three paired seeds: {GAP_FOUR[0]:.3f} against {GAP_FULL[0]:.3f} at seed {ex.SEED_OFFSET}, {GAP_FOUR[1]:.3f} against {GAP_FULL[1]:.3f} at seed {ex.SEED_OFFSET + 1}, and {GAP_FOUR[2]:.3f} against {GAP_FULL[2]:.3f} at seed {ex.SEED_OFFSET + 2}. At seed {ex.SEED_OFFSET + 3}, which has no `full` partner, the `no-four` gap is {GAP_FOUR[3]:.3f}, within the range of the other three. The largest `no-four` gap, {max(GAP_FOUR.values()):.3f}, is below the smallest `full` gap, {min(GAP_FULL.values()):.3f}.

"""

pairs_draw(
    f"""
        A dot plot of the gap to the Bayes ceiling at {REF_E} epochs, by model seed, for no-four (filled dots) and
        full (hollow dots), joined by a line at each seed where both exist. At every seed with a pair, the filled
        dot is below the hollow one: the no-four gaps are {span(GAP_FOUR.values(), ".3f")} and the full gaps
        {span(GAP_FULL.values(), ".3f")}. Seed {ex.SEED_OFFSET + 3} has only a filled dot.
    """,
    """
        **Gap to the ceiling of each op set, by model seed.** Filled: `no-four`. Hollow: `full`. A line joins the
        two runs that started from the same weights. Each gap is to the ceiling of its own op set.
    """,
)

rf"""

The narrower gap of ex-2.2.18 was not seed variation. The gap is smaller at every seed, by {span([GAP_FULL[s] - GAP_FOUR[s] for s in GAP_FULL], ".3f")}, while the four `no-four` seeds vary by {spread(GAP_FOUR.values()):.3f} among themselves. The pairs differ in corpus draw as well as op set, so this shows that `no-four` gets closer to its ceiling, and not which of the two changes is responsible.

/// admonition | Pass
`no-four` has the smaller gap than `full` at all three paired seeds, {span([GAP_FOUR[s] for s in FULL_PARTNER], ".3f")} against {span(GAP_FULL.values(), ".3f")}.
///


## Exploratory analyses

These are planned but have no prediction.

### Skill curves (E1)

The skill curves of every run, to see whether the HSV-channel ops are learned later, or not at all, in the shorter runs.

**What we saw.** The 50- and 100-epoch runs end before the HSV-channel ops have been learned: their mean skill on the probe set is {end_skill(lab(50), HSV_CHANNEL):.2f} and {end_skill(lab(100), HSV_CHANNEL):.2f} at the end, against {end_skill(lab(50), OTHER_OPS):.2f} and {end_skill(lab(100), OTHER_OPS):.2f} for the other four ops. In the {PICK}-epoch runs the HSV-channel ops are learned, but later in the schedule than at {REF_E} epochs. A quarter of the way through, their mean skill is {span([hsv_at_fraction(lab(PICK, s), 0.25) for s in (0, *SEEDS)])} at {PICK} epochs and {span([hsv_at_fraction(lab(REF_E, s), 0.25) for s in (0, *SEEDS)])} at {REF_E}, over the four seeds. They finish at {span([end_skill(lab(PICK, s), HSV_CHANNEL) for s in (0, *SEEDS)])} and {span([end_skill(lab(REF_E, s), HSV_CHANNEL) for s in (0, *SEEDS)])}.

"""

curves_draw(
    f"""
        Eight small line charts of skill on the probe set against the fraction of training completed, one for all ops
        and one for each of the seven, with a line for each of four lengths. In the panels for hue-hsv, sat-hsv,
        and value-hsv, the {REF_E}-epoch lines rise first, the {PICK}-epoch lines rise later and reach nearly the
        same level, and the 50- and 100-epoch lines stay low. In the panels for the other four ops all four lengths
        rise together, with the shorter runs ending lower.
    """,
    """
        **Skill against the fraction of training completed.** One panel for all ops and one for each op; one color per
        length, darker for longer. Thick lines are the scout seed and thin lines the three fresh seeds, at the lengths
        that have them. The dashed line is the ceiling.
    """,
)

rf"""

The rest of this analysis is post hoc. In the {PICK}-epoch and {REF_E}-epoch runs alike, the HSV-channel skill first passes {HSV_HALF:g} between epoch {span([hsv_rise(lab(e, s))[0] for e in (PICK, REF_E) for s in (0, *SEEDS)], ".0f")}. So these ops come late as a share of a shorter run, yet at about the same number of epochs. The learning rate at that epoch differs: {span([hsv_rise(lab(PICK, s))[1] for s in (0, *SEEDS)], ".4f")} in the {PICK}-epoch runs and {span([hsv_rise(lab(REF_E, s))[1] for s in (0, *SEEDS)], ".4f")} in the {REF_E}-epoch runs. That points to the number of epochs, rather than the rate, as what matters. But the two lengths differ in their whole schedule, so eight runs cannot separate the two. It fits the 100-epoch run falling short on these ops, since it ends near that point.


### Calibration (E2)

The calibration KL of each run (the KL divergence from the Bayes answer distribution to that of the model) beside its EEM, since ex-2.2.17 found a model can score well and be poorly calibrated.

**What we saw.** Calibration KL falls as EEM rises across the lengths, from {kl(ex.label_of(50, ex.SCOUT_LRS[0], 0)):.2f} at 50 epochs at the recipe rate to {kl(REF_LABEL):.2f} at {REF_E}. The higher rate brings it down sooner: {kl(ex.label_of(50, ex.SCOUT_LRS[1], 0)):.2f} at 50 epochs and {kl(ex.label_of(100, ex.SCOUT_LRS[1], 0)):.2f} at 100. At {PICK} and {REF_E} epochs the points overlap: {span([kl(lab(PICK, s)) for s in (0, *SEEDS)], ".3f")} over the four seeds at {PICK} epochs and {span([kl(lab(REF_E, s)) for s in (0, *SEEDS)], ".3f")} at {REF_E}. Within those two groups, a higher EEM does not go with a lower KL. The {PICK}-epoch runs are no worse calibrated than the longer ones. So the case that ex-2.2.17 found, a model that scores well but is poorly calibrated, does not arise at {PICK} epochs.

"""

calibration_draw(
    f"""
        A scatter plot of calibration KL against held-out EEM, one dot per run, colored by training length. The
        50-epoch runs are at the upper left, at KL {kl(ex.label_of(50, ex.SCOUT_LRS[0], 0)):.2f} and
        {kl(ex.label_of(50, ex.SCOUT_LRS[1], 0)):.2f}, the 100-epoch runs lower, and the {PICK}-epoch and {REF_E}-epoch runs form one cluster at the lower right, all between
        {min(kl(lbl) for lbl in RUNS if RUNS[lbl]["epochs"] >= PICK):.2f} and {max(kl(lbl) for lbl in RUNS if RUNS[lbl]["epochs"] >= PICK):.2f}, with no clear order by length within it.
    """,
    """
        **Calibration KL against held-out EEM.** One dot per run, one color per length, darker for longer. A filled dot is
        a run at the rate taken at its length; a hollow dot is a scout run at the other rate.
    """,
)

rf"""


### Op confusion (E3)

The op confusion matrix of each run at the chosen length and at {ex.REFERENCE_EPOCHS} epochs: the mass the model puts on the answers of each op, on confident contexts[^confident] of each true op, as in ex-2.2.17. A shorter run might keep more of its mass on a similar op.

[^confident]: A context is confident when the Bayes posterior on its true op is above {CONFIDENT}, as in ex-2.2.17 and ex-2.2.18.

**What we saw.** The shorter run puts more mass off the diagonal. Summed over a row, the mass on the answers of other ops averages {off_mass(PICK):.3f} at {PICK} epochs and {off_mass(REF_E):.3f} at {REF_E}, and it is higher in every row. The rise is largest in the three HSV-channel rows, {row_off(PICK, HSV_CHANNEL):.3f} against {row_off(REF_E, HSV_CHANNEL):.3f}, and in the other four rows it goes from {row_off(REF_E, OTHER_OPS):.3f} to {row_off(PICK, OTHER_OPS):.3f}. The largest square is the same at both lengths, `{worst_square(PICK)[0]}` onto `{worst_square(PICK)[1]}`, at {worst_square(PICK)[2]:.3f} and {worst_square(REF_E)[2]:.3f}. So the extra mass sits mostly in the rows of the HSV-channel ops, and the shorter run leans hardest on the same op as the longer one. Every square stays small.

"""

confusion_draw(
    f"""
        Two small heatmaps of the seven ops by the seven ops, one for {PICK} epochs and one for {REF_E}. The diagonal is
        blank and outlined. Most squares are pale in both. The darkest squares in the {PICK}-epoch map are in the rows
        of the three HSV-channel ops, and the largest, {worst_square(PICK)[0]} onto {worst_square(PICK)[1]}, is
        {worst_square(PICK)[2]:.3f}; in the {REF_E}-epoch map the same square is {worst_square(REF_E)[2]:.3f}.
    """,
    f"""
        **Where the mass goes, by op, at {PICK} and {REF_E} epochs.** Mean over the three fresh seeds, on confident
        held-out contexts. Rows are the true op. Each off-diagonal square is the mass the model puts on colors the
        column op can give and the true op cannot, with values of {PRINT_FLOOR} and above printed. The diagonal is outlined and left
        blank. Columns are the seven ops of the set.
    """,
)

rf"""


### Spread over seeds (E4)

The spread of EEM over the four `no-four` seeds at {ex.REFERENCE_EPOCHS} epochs, against the spread of the three `full` seeds, and the spread at each confirmed length.

**What we saw.** Over four model seeds the EEM range is {spread(EEM_FOUR_200):.3f} at {PICK} epochs and {spread(EEM_FOUR_400):.3f} at {REF_E}, and over the three `full` seeds of ex-2.2.17 it is {spread(EEM_FULL_400):.3f}. The {PICK}-epoch runs vary about {spread(EEM_FOUR_200) / spread(EEM_FOUR_400):.1f} times as much as the {REF_E}-epoch ones, which four seeds can hardly resolve, and both are below the `full` range. The lowest {PICK}-epoch run, at {min(EEM_FOUR_200):.3f}, is below the lowest {REF_E}-epoch run, at {min(EEM_FOUR_400):.3f}, by {min(EEM_FOUR_400) - min(EEM_FOUR_200):.3f}.

"""

spread_draw(
    f"""
        A dot plot of held-out EEM with one column per condition: no-four at {PICK} epochs, no-four at {REF_E},
        and full at {REF_E}. Each column has one dot per seed, a larger mark for the mean, and a thin bar for the
        range. The two no-four columns are near {np.mean(EEM_FOUR_200):.2f} and {np.mean(EEM_FOUR_400):.2f} with short
        bars; the full column is near {np.mean(EEM_FULL_400):.2f} with a longer one.
    """,
    f"""
        **Held-out EEM by seed.** Four seeds for each `no-four` column (model seeds {ex.SEED_OFFSET} to
        {ex.SEED_OFFSET + 3}) and the three seeds of ex-2.2.17 for `full`. The larger mark is the seed mean and the thin
        bar spans the seeds.
    """,
)

rf"""


## What it means for what follows

The anchoring experiments that follow can train `no-four` for {PICK} epochs at the peak rate {ex.PEAK_LR:g}, at half the cost. The shortfall is small, but most of it is in the three HSV-channel ops, so for an experiment that leans on them, an unanchored control of the same length is a fairer comparison than a {REF_E}-epoch number.

The HSV-channel ops set how short a run can be. In the post hoc part of E1 they took off at about the same epoch at both long lengths, which is consistent with needing a count of epochs rather than a share of the schedule. That rests on eight runs whose schedules differ throughout; lengths between 100 and 200 epochs, or a schedule that decays later, could separate the two.

Calibration does not separate {PICK} from {REF_E} epochs, and four seeds cannot resolve a difference in spread, so the shortfall is the only cost we found. The scout used one seed, the confirmation three, and all runs one op set and one corpus, so we cannot say how the shortfall moves with either.

The narrower gap of `no-four` held at every paired seed, which supports it as the base for the anchoring runs.

## Method

**Recipe.** The unanchored d64-L4 model with an untied readout and the newline mask, trained with a cosine schedule after a linear warmup, as in ex-2.2.18. The corpus condition is `k3-r0.3` (three examples per context, with a replacement rate of 0.3) on the `no-four` op set (the eleven ops of ex-2.2.16 less `screen`, `multiply`, `hsvmix`, and `exclusion`), with the corpus, held-out set, and probe set that ex-2.2.18 built. So the runs differ from the ex-2.2.18 run only in length and model seed. This experiment rebuilds the corpus from the seed of ex-2.2.18, and its Bayes ceiling, {CEILING:.4f}, equals the {CEILING_18:.4f} that ex-2.2.18 published, so the runs share that corpus.

**Warmup and learning rate.** The warmup is {ex.WARMUP_EPOCHS:g} epochs rather than a fixed share of the run, as in ex-2.2.17 from its second round, and the peak learning rate is {ex.PEAK_LR:g}. A shorter run spends less of its schedule near the peak, so it may do better at a higher peak rate. The scout tries {ex.SCOUT_LRS[1]:g} beside it at each shorter length: the next rate up on the ex-2.2.17 grid, which at 400 epochs scored level with {ex.PEAK_LR:g} at one seed, where 0.01 scored lower. The 400-epoch runs stay at {ex.PEAK_LR:g}, so H1 compares a shorter recipe, rate included, with the recipe we have.

**Measurements.** EEM, ceiling, floor, and calibration KL are measured on the {ex.ex2216.HOLDOUT_CONTEXTS:,} held-out contexts per op, with the evaluation of ex-2.2.18. Skill curves are logged at about {ex.ex2217.N_TRAJ_POINTS} points per run on {ex.ex2217.N_TRAJ_EEM_PER_OP} held-out contexts per op.

**Per-op tolerances.**

"""

table_html(
    ["op", "tolerance"],
    [[f"`{op}`", f"{OP_TOL[op]:.3f}"] for op in ex.OP_SET.ops],
    f"""
        **Per-op tolerances.** The seed range of the gap of each op over ex-2.2.17's three runs, or {ex.PARTIAL_TOL},
        whichever is larger.
    """,
)
