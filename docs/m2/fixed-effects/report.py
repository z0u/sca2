# ruff: noqa: B018
# title: What the edit does on τ_ℓ fixed

# An exploratory re-analysis of the fixed confirmation's runs of `τ_ℓ fixed` at seeds 712 to 719, with no new
# training and no new measurement pass: the dose scoring, the axis scoring and the alignment arrays that experiment
# stored, with the anneal confirmation's controls at the same seeds. The edit-effects runs of `falloff` at seeds 704
# to 711 are scored the same way beside them.
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesGrid, AxesRow, figure_html, light_dark, themed


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


fc = _load_sibling("fixed-confirm", "fixed_effects_fc")
ac, rb, ee = fc.ac, fc.rb, fc.ee
sas = ee.sas

# --- The stored arrays ----------------------------------------------------------------------------------------

FIXED, FALLOFF = "fixed", "falloff"
RECIPES = (FIXED, FALLOFF)
SHOWN = {FIXED: "τ_ℓ fixed", FALLOFF: "falloff"}
SEEDS_OF = {FIXED: list(fc.SEEDS), FALLOFF: list(ee.SEEDS)}
# Where each recipe keeps its arrays: the alignment, the dose scoring of the run and of its control, and the axes.
ALPHA_REF = {FIXED: lambda s: fc.WRITE_ARRAYS_REF.format(label=fc.label_of(s, None)),
             FALLOFF: lambda s: ee.ARRAYS_REF.format(label=f"falloff-s{s}")}  # fmt: skip
DOSE_REF = {FIXED: lambda s: fc.DOSE_ARRAYS_REF.format(label=fc.label_of(s, None)),
            FALLOFF: lambda s: sas.DOSE_ARRAYS_REF.format(label=f"falloff-s{s}")}  # fmt: skip
CONTROL_DOSE_REF = {FIXED: lambda s: ac.DOSE_ARRAYS_REF.format(label=ac.label_of(s, None, "control")),
                    FALLOFF: lambda s: sas.DOSE_ARRAYS_REF.format(label=f"control-s{s}")}  # fmt: skip
AXES_LABEL = {FIXED: lambda s: fc.label_of(s, None), FALLOFF: lambda s: f"falloff-s{s}"}
DOSE_KEEP = ("op_ids", "posterior", "query_pair", "clean_eem", "null_eem", "eem", "p_clean", "p_full")


def fetch_all() -> tuple[dict, np.ndarray, dict, dict, dict]:
    """The axis scoring of both recipes, the fit of every example answer under every op, and per run the alignment
    arrays, the dose arrays, and the dose arrays of its control, keyed by (recipe, seed).
    """
    store = project_store()
    arrays = (
        {("alpha", n, s): ALPHA_REF[n](s) for n in RECIPES for s in SEEDS_OF[n]}
        | {("dose", n, s): DOSE_REF[n](s) for n in RECIPES for s in SEEDS_OF[n]}
        | {("control", n, s): CONTROL_DOSE_REF[n](s) for n in RECIPES for s in SEEDS_OF[n]}
    )
    refs = [fc.AXES_REF, rb.AXES_REF, ee.FITS_REF, *arrays.values()]
    got = store.get_refs(refs)
    have = {r: a for r, a in got.items() if a is not None}
    assert len(have) == len(refs), f"not published: {sorted(set(refs) - set(have))}"
    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        paths = dict(zip(refs, store.get_many([(have[r], into / f"{i}") for i, r in enumerate(refs)]), strict=True))
        axes = {FIXED: json.loads(paths[fc.AXES_REF].read_text()), FALLOFF: json.loads(paths[rb.AXES_REF].read_text())}
        with np.load(paths[ee.FITS_REF]) as z:
            fits = z["fits"]
        alpha, dose, control = {}, {}, {}
        for (kind, n, s), ref in arrays.items():
            with np.load(paths[ref]) as z:
                if kind == "alpha":
                    alpha[(n, s)] = {k: z[k] for k in ("clean", "arriving", "op_ids")}
                else:
                    (dose if kind == "dose" else control)[(n, s)] = {k: z[k] for k in DOSE_KEEP}
    return axes, fits, alpha, dose, control


AXES_BODY, FITS, ALPHA, DOSE, CONTROL = fetch_all()
OPS: tuple[str, ...] = tuple(AXES_BODY[FIXED]["design"]["ops"])
D = OPS.index(fc.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
DIFF = fc.ANCHORED_OP
P = ee.ex2216._get_posterior()
TABLE = P.build_table(ee.ex2222.ex2218.table_of(OPS))
d0 = DOSE[(FIXED, SEEDS_OF[FIXED][0])]
OP_IDS = d0["op_ids"].astype(int)
IS_D = OP_IDS == D
# Every run of both recipes was scored on the same held-out contexts.
for _a in (*ALPHA.values(), *DOSE.values(), *CONTROL.values()):
    assert (_a["op_ids"] == OP_IDS).all()
for _a in (*DOSE.values(), *CONTROL.values()):
    assert np.array_equal(_a["query_pair"], d0["query_pair"]) and np.allclose(_a["posterior"], d0["posterior"])

CRITERION = 0.02  # the selectivity criterion of the D2.2 design
GAMMAS: list[float] = list(rb.DOSE_GAMMAS)
UNIT = 6  # tokens per equation: a ? b = y and a separator
N_POS = UNIT * (fc.K + 1)
ANSWERS = [UNIT * e + 4 for e in range(fc.K)]  # the example answers
QUERY_ANSWER = UNIT * fc.K + 4
ROLE = ("a", "?", "b", "=", "y", ",")
POS_LABEL = [ROLE[p % UNIT] if p != N_POS - 1 else "⏎" for p in range(N_POS)]
SLICE_TICK = ["emb", "1", "2", "3", "4"]
Q = 0.99  # the quantile the bound and the write are taken at, as in H4 of ex-2.2.1
TOLERANCE = 1.25  # the tolerance of H4 in ex-2.2.1: the write at most a quarter over the bound
FITS_D = FITS[:, :, D]  # (contexts, examples + query): the answer is one `difference` can give
assert FITS_D.shape[1] == fc.K + 1


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus, and no sign on a value that prints as zero."""
    half = 0.5 * 10 ** -int(spec[-2])
    return f"{0.0 if abs(v) < half else v:{spec}}".replace("-", "−")


def span(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(v.mean(), spec)} ({num(v.min(), spec)} to {num(v.max(), spec)})"


def t_interval(a, b) -> tuple[float, float]:
    """A 95% Welch t interval on the difference of the means of two samples of runs, which are not paired."""
    from scipy import stats

    ci = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False).confidence_interval(0.95)
    return float(ci.low), float(ci.high)


# REVIEW: every comparison of the two recipes now takes a Welch t interval in place of the percentile bootstrap
# (paper round 1, claims follow-up). At eight runs per recipe the bootstrap covered about 90% of the time rather than
# 95% in a simulation at the observed spreads, and the t interval about 95%. Verify: `t_interval`, and each `gap` in
# the prose against the wording around it.
def gap(a, b, spec: str = ".2f", unit: str = "") -> str:
    """The difference of means of two samples of runs, with its 95% Welch t interval, for the prose."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo, hi = t_interval(a, b)
    return f"{num(a.mean() - b.mean(), spec)}{unit} (95% interval {num(lo, spec)}{unit} to {num(hi, spec)}{unit})"


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


# Theme inks are functions: `light_dark` picks by the theme active when it is called.


def ink(n: str = FIXED) -> str:
    return light_dark("#2a9d8f", "#7fd8c8") if n == FIXED else light_dark("#888", "#999")


def write_ink() -> str:
    return light_dark("#c0392b", "#ff8a76")


def rule() -> str:
    return light_dark("#aaa", "#555")


MARK = {FIXED: "D", FALLOFF: "o"}


# --- E1: removal and spill by dose, net of the median over the other 63 axes ----------------------------------


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[0] - np.median(drop[1:], axis=0)


_AX = {n: {r["label"]: r for r in AXES_BODY[n]["runs"]} for n in RECIPES}


def dose_scores(n: str, s: int) -> dict[str, np.ndarray]:
    r = _AX[n][AXES_LABEL[n](s)]
    net = net_of(r["drop"])  # (doses, ops)
    return {
        "removal": net[:, D] / (r["clean"][D] - r["null"][D]),
        "spill": net[:, OTHER].max(axis=1),
        "net": net,
    }


BY_DOSE = {(n, s): dose_scores(n, s) for n in RECIPES for s in SEEDS_OF[n]}


def dose_stack(n: str, key: str) -> np.ndarray:
    return np.array([BY_DOSE[(n, s)][key] for s in SEEDS_OF[n]])  # (seeds, doses)


REMOVAL = {n: dose_stack(n, "removal") for n in RECIPES}
SPILL = {n: dose_stack(n, "spill") for n in RECIPES}
SPILL_WORST = {n: SPILL[n].max(axis=1) for n in RECIPES}  # per run, over doses
# The op and dose of the largest spill of any run of `τ_ℓ fixed`.
_w = np.unravel_index(SPILL[FIXED].argmax(), SPILL[FIXED].shape)
SPILL_TOP = {
    "seed": SEEDS_OF[FIXED][_w[0]],
    "dose": GAMMAS[_w[1]],
    "op": OPS[OTHER[int(BY_DOSE[(FIXED, SEEDS_OF[FIXED][_w[0]])]["net"][_w[1], OTHER].argmax())]],
    "value": float(SPILL[FIXED].max()),
}
MONOTONE = {n: int((np.diff(REMOVAL[n], axis=1) > 0).all(axis=1).sum()) for n in RECIPES}
# Removal at each dose as a share of removal at the full dose, seed mean: how close to proportional the dose is.
DOSE_SHARE_RUNS = {n: REMOVAL[n] / REMOVAL[n][:, -1:] for n in RECIPES}  # (seeds, doses)
DOSE_SHARE = {n: DOSE_SHARE_RUNS[n].mean(axis=0) for n in RECIPES}


def turn_at_dose(alpha, gamma: float) -> np.ndarray:
    """The angle a dose turns a unit state with alignment *alpha*: it takes the share *gamma* of the e₁ component
    out and puts the state back on the sphere, so tan of its angle to the plane off e₁ shrinks by (1 − γ).
    """
    theta = np.arcsin(np.clip(np.abs(alpha), 0, 1))
    return theta - np.arctan((1 - gamma) * np.tan(theta))


# REVIEW: added in answer to the paper review (would arcsin(dose) straighten the removal curve?). The dose is a
# share of the e₁ component, not an alignment, so arcsin(dose) has no geometric meaning here; the angle a dose turns
# a state through is `turn_at_dose`. At slice 1 that is close to proportional to the dose, so the slow start is not
# in the turn there. Verify: DOSE_TURN against GAMMAS.
_states_d = np.concatenate([ALPHA[(FIXED, s)]["clean"][1][IS_D][:, ANSWERS].ravel() for s in SEEDS_OF[FIXED]])
_states_d = _states_d.astype(np.float64)
# Per dose, the mean turn of the states holding `difference` at the example answers at slice 1, as a share of the
# mean turn under the full edit.
DOSE_TURN = np.array([turn_at_dose(_states_d, g).mean() / turn_at_dose(_states_d, 1.0).mean() for g in GAMMAS])


def aligned_after(alpha, gamma: float) -> np.ndarray:
    """|α| of a unit state after a dose: the share *gamma* of its e₁ component taken out, then back on the sphere."""
    a = np.abs(np.asarray(alpha, np.float64))
    return (1 - gamma) * a / np.sqrt((1 - gamma) ** 2 * a**2 + 1 - a**2)


# REVIEW: added the geometry at the later slices (paper round 1, claims follow-up). There the states holding
# `difference` sit nearly on e₁, where a partial dose turns a state much less than its share and putting it back on
# the sphere gives most of the alignment back, so the turn at a single slice lags the measured removal at every dose;
# the shape has to come from across the slices. Verify: DOSE_MEDIAN_ALPHA, DOSE_LEFT_3, and `turn_at_dose` by slice.
_states_by_slice = {
    sl: np.concatenate([ALPHA[(FIXED, s)]["clean"][sl][IS_D][:, ANSWERS].ravel() for s in SEEDS_OF[FIXED]]).astype(
        np.float64
    )
    for sl in range(1, 5)
}
# The median |α| of the states holding `difference` at the example answers, at slices 2 to 4.
DOSE_MEDIAN_ALPHA = [float(np.median(np.abs(_states_by_slice[sl]))) for sl in (2, 3, 4)]
# At slice 3, the mean |α| left after three quarters of the edit, as a share of the clean mean |α|.
DOSE_LEFT_3 = float(aligned_after(_states_by_slice[3], 0.75).mean() / np.abs(_states_by_slice[3]).mean())
# At every slice past the first, a dose turns these states by a smaller share of the full turn than the share of the
# full removal it gives, at every dose short of the full edit: the per-slice geometry lags the measured curve.
_turn_share = {
    sl: np.array([turn_at_dose(st, g).mean() / turn_at_dose(st, 1.0).mean() for g in GAMMAS[:-1]])
    for sl, st in _states_by_slice.items()
}
assert all((_turn_share[sl] < DOSE_SHARE[FIXED][:-1]).all() for sl in (2, 3, 4))
assert 0.75 in GAMMAS


@memo
def dose_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-effects-dose",
        alt_text=alt_text,
        caption=f"""
            **Removal and spill by dose**, the edit at every position and slice, net of the median drop along the
            other 63 coordinate axes of the run. One hairline per run and the seed mean bold, for `τ_ℓ fixed` at
            seeds 712 to 719 and `falloff` at 704 to 711; dose 0 is the clean pass. **Left:** removal, the share of the way from the clean score
            on `{DIFF}` to the target null. **Right:** spill, the largest net drop on any other op; the dashed line
            is the criterion, with the side over it hatched.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), layout="constrained")
        axes = cast(AxesRow, axes)
        x = np.array(data["gammas"])
        for ax, key in zip(axes, ("removal", "spill"), strict=True):
            for n in reversed(RECIPES):
                for row in data[key][n]:
                    ax.plot(x, row, color=ink(n), lw=0.5, alpha=0.6)
                ax.plot(x, np.mean(data[key][n], axis=0), color=ink(n), lw=1.6, marker=MARK[n], ms=4, label=SHOWN[n])
            ax.set_xticks(x, [f"{g:g}" for g in x])
            ax.set_xlabel("dose (share of e₁ removed)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("removal", fontsize=8)
        ax = axes[1]
        ax.set_ylabel("spill", fontsize=8)
        ax.axhline(CRITERION, color=rule(), lw=0.8, ls="--")
        lo, hi = -0.03, 0.035
        ax.axhspan(CRITERION, hi, facecolor="none", edgecolor=rule(), hatch="//", lw=0, alpha=0.4, zorder=0)
        ax.set_ylim(lo, hi)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


def dose_figure() -> str:
    # REVIEW: added the point at dose 0 (paper round 1). A dose of 0 is the clean pass, so the drop is 0 along e₁
    # and along every other axis, and removal and spill are 0 by definition, not by measurement.
    def from_zero(m: np.ndarray) -> list[list[float]]:
        return np.concatenate([np.zeros((len(m), 1)), m], axis=1).tolist()

    data = {
        "gammas": [0.0, *GAMMAS],
        "removal": {n: from_zero(REMOVAL[n]) for n in RECIPES},
        "spill": {n: from_zero(SPILL[n]) for n in RECIPES},
    }
    alt = (
        "Left: from zero at dose 0, removal rises slowly over the first quarter of the edit, steeply over the second, "
        "and levels off from three quarters, on every run of both recipes. Right: spill stays below the criterion "
        f"of {CRITERION} at every dose on every run of both recipes."
    )
    return dose_draw(data, alt)


def dose_table() -> str:
    rows = []
    for n in RECIPES:
        rows.append([f"`{SHOWN[n]}`, removal ↑", *[span(REMOVAL[n][:, i]) for i in range(len(GAMMAS))]])
    for n in RECIPES:
        rows.append([f"`{SHOWN[n]}`, spill ↓", *[span(SPILL[n][:, i], ".3f") for i in range(len(GAMMAS))]])
    return table_html(
        ["recipe, measure", *[f"dose {g:g}" for g in GAMMAS]],
        rows,
        f"""**Removal and spill by dose**, seed mean and range over eight runs, net of the median over the other
        63 axes of the run: `τ_ℓ fixed` at seeds 712 to 719, `falloff` at 704 to 711. Spill is the largest net
        drop on any other op at that dose; the criterion is {CRITERION}.""",
    )


# --- E2: the grading ------------------------------------------------------------------------------------------

POST_BINS = (0.0, 0.2, 0.5, 0.8, 0.95, 1.0)
BIN_LABEL = ["< 0.2", "0.2–0.5", "0.5–0.8", "0.8–0.95", "≥ 0.95"]
OTHER_BINS = (0.0, 0.05, 0.2, 0.5, 1.0)
OTHER_LABEL = ["< 0.05", "0.05–0.2", "0.2–0.5", "≥ 0.5"]


def contexts_of(op_ids: np.ndarray, query_pair: np.ndarray) -> Any:
    n = len(op_ids)
    empty = np.zeros((n, 0), np.int64)
    return P.Contexts(op_ids.astype(np.int64), empty, empty, query_pair.astype(np.int64))


@memo
def bayes(posterior: np.ndarray, op_ids: np.ndarray, query_pair: np.ndarray) -> dict[str, np.ndarray]:
    """The Bayes predictor and the target null on every held-out context: expected exact match of each, and the
    answer distributions over the palette.
    """
    ctx = contexts_of(op_ids, query_pair)
    post = posterior.astype(np.float64)
    null = ee.ex2216._target_null(P, TABLE, ctx, post, D)
    return {
        "ideal_eem": P.expected_match(TABLE, ctx, post),
        "null_eem": P.expected_match(TABLE, ctx, null),
        "q_ideal": P.predictive(TABLE, ctx, post),
        "q_null": P.predictive(TABLE, ctx, null),
        "null_post": null,
    }


BAYES = bayes(d0["posterior"], d0["op_ids"], d0["query_pair"])
for _a in DOSE.values():
    assert np.allclose(BAYES["null_eem"], _a["null_eem"], atol=2e-4)
POST_D = d0["posterior"][:, D].astype(np.float64)
BIN = np.clip(np.digitize(POST_D, POST_BINS[1:-1]), 0, len(POST_BINS) - 2)
OBIN = np.clip(np.digitize(POST_D, OTHER_BINS[1:-1]), 0, len(OTHER_BINS) - 2)
DEP_BAYES = BAYES["ideal_eem"] - BAYES["null_eem"]  # how much each context needed `difference`, model-free
N_BIN_D = [int(((BIN == b) & IS_D).sum()) for b in range(len(BIN_LABEL))]
N_OBIN = [int(((OBIN == b) & ~IS_D).sum()) for b in range(len(OTHER_LABEL))]


# REVIEW: the scores by context were net of the control at the same seed; they are now net of the mean drop over
# the eight controls of each recipe, context by context (paper round 1). Two runs at one seed are about as far apart
# as runs at different seeds by the end of training (eng/determinism.md, "Seeds are draws"), so a single control
# added its own seed scatter. Verify: CONTROL_DROP, and the share removed and gain tables against the earlier render.
CONTROL_DROP = {
    n: np.mean([CONTROL[(n, s)]["clean_eem"] - CONTROL[(n, s)]["eem"][-1] for s in SEEDS_OF[n]], axis=0)
    for n in RECIPES
}  # (contexts,): the drop under the full edit along e₁, mean over the controls of each recipe


def grading(n: str, s: int) -> dict[str, Any]:
    """The share removed by bin of the posterior on `difference` contexts, net of the mean of the controls and not,
    and the gain on the contexts of the other ops by bin.
    """
    r = DOSE[(n, s)]
    raw = r["clean_eem"] - r["eem"][-1]
    net = raw - CONTROL_DROP[n]
    dep = r["clean_eem"] - r["null_eem"]

    def share(drop: np.ndarray, sel: np.ndarray) -> float:
        return float(drop[sel].mean() / dep[sel].mean())

    return {
        "share": np.array([share(net, (BIN == b) & IS_D) for b in range(len(BIN_LABEL))]),
        "share_all": share(net, IS_D),
        "raw_share": np.array([share(raw, (BIN == b) & IS_D) for b in range(len(BIN_LABEL))]),
        "raw_share_all": share(raw, IS_D),
        "r_bayes": float(np.corrcoef(net[IS_D], DEP_BAYES[IS_D])[0, 1]),
        "gain": np.array([-net[(OBIN == b) & ~IS_D].mean() for b in range(len(OTHER_LABEL))]),
    }


GRADE = {(n, s): grading(n, s) for n in RECIPES for s in SEEDS_OF[n]}
GAIN_NULL = np.array([-DEP_BAYES[(OBIN == b) & ~IS_D].mean() for b in range(len(OTHER_LABEL))])


def grade(n: str, key: str) -> np.ndarray:
    return np.array([GRADE[(n, s)][key] for s in SEEDS_OF[n]])


SHARE_MEAN = {n: grade(n, "share").mean(axis=0) for n in RECIPES}
GAIN_MEAN = {n: grade(n, "gain").mean(axis=0) for n in RECIPES}
assert (grade(FIXED, "share") < 1).all(), "the prose of E2 says no run goes past the target null in any bin"


@memo
def grading_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-effects-grading",
        alt_text=alt_text,
        caption=f"""
            **The drop under the full edit, by how much each context needed `{DIFF}`.** **Left:** on `{DIFF}`
            contexts, the share of the way to the target null that the edit takes in each bin of the posterior on
            `{DIFF}`, as a ratio of bin means; one hairline per run and the seed mean bold, with the target null as
            the dashed rule at 1. **Right:** on the contexts of the other ops, the gain in expected exact match under
            the edit in each bin of the posterior on `{DIFF}` (these bins differ from the left), seed mean and range,
            with the gain the target null predicts as open squares. Both net of the mean of the controls.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), layout="constrained")
        axes = cast(AxesRow, axes)
        ax = axes[0]
        x = np.arange(len(BIN_LABEL))
        for n in reversed(RECIPES):
            for row in data["share"][n]:
                ax.plot(x, row, color=ink(n), lw=0.5, alpha=0.6)
            ax.plot(x, np.mean(data["share"][n], axis=0), color=ink(n), lw=1.6, marker=MARK[n], ms=4, label=SHOWN[n])
        ax.axhline(1, color=rule(), lw=0.7, ls="--", zorder=0)
        ax.set_ylim(0, 1.2)
        ax.set_xticks(x, BIN_LABEL, fontsize=7)
        ax.set_xlabel(f"posterior on {DIFF}", fontsize=8)
        ax.set_ylabel("share of the way to the null", fontsize=8)
        ax.tick_params(labelsize=7)
        ax = axes[1]
        x = np.arange(len(OTHER_LABEL))
        for k, n in enumerate(reversed(RECIPES)):
            g = np.array(data["gain"][n])
            xx = x + (k - 0.5) * 0.18
            ax.vlines(xx, g.min(axis=0), g.max(axis=0), color=ink(n), lw=2.5, alpha=0.35)
            ax.plot(xx, g.mean(axis=0), ls="none", color=ink(n), marker=MARK[n], ms=4)
        ax.plot(x, data["null"], ls="none", marker="s", ms=6, mfc="none", color=rule(), label="target null")
        ax.axhline(0, color=rule(), lw=0.5, zorder=0)
        ax.set_xticks(x, OTHER_LABEL, fontsize=7)
        ax.set_xlabel(f"posterior on {DIFF}", fontsize=8)
        ax.set_ylabel("gain in expected exact match", fontsize=8)
        ax.tick_params(labelsize=7)
        h0, l0 = axes[0].get_legend_handles_labels()
        h1, l1 = axes[1].get_legend_handles_labels()
        fig.legend(h0[::-1] + h1, l0[::-1] + l1, loc="outside upper center", ncols=3, frameon=False, fontsize=8)
        return fig

    return _plot()


def grading_figure() -> str:
    data = {
        "share": {n: grade(n, "share").tolist() for n in RECIPES},
        "gain": {n: grade(n, "gain").tolist() for n in RECIPES},
        "null": GAIN_NULL.tolist(),
    }
    alt = (
        f"Left: the share removed is roughly level across the posterior on {DIFF} for both recipes, a little under "
        f"the target null. Right: on the other ops the gain under the edit rises with the posterior on {DIFF}, as "
        "the target null predicts, and the two recipes match bin by bin."
    )
    return grading_draw(data, alt)


def grading_table() -> str:
    rows = [["`difference` contexts", *[f"{v:,}" for v in N_BIN_D], f"{int(IS_D.sum()):,}"]]
    rows.append(
        [
            "dependence (Bayes)",
            *[num(DEP_BAYES[(BIN == b) & IS_D].mean()) for b in range(5)],
            num(DEP_BAYES[IS_D].mean()),
        ]
    )
    for n in RECIPES:
        rows.append([f"`{SHOWN[n]}`", *[span(grade(n, "share")[:, b]) for b in range(5)], span(grade(n, "share_all"))])
    rows.append(
        [
            f"`{SHOWN[FIXED]}`, not net of the controls",
            *[span(grade(FIXED, "raw_share")[:, b]) for b in range(5)],
            span(grade(FIXED, "raw_share_all")),
        ]
    )
    return table_html(
        [f"share removed, by posterior on `{DIFF}`", *BIN_LABEL, "all"],
        rows,
        f"""**The share of the dependence removed, by bin of the posterior**, on the held-out `{DIFF}` contexts: the
        mean drop under the full edit over the mean distance from the clean answer to the target null, in expected
        exact match, net of the mean drop of the eight controls unless marked; seed mean and range over eight runs. The
        dependence row is the drop the target null predicts for the Bayes predictor, which needs no model.""",
    )


def gain_table() -> str:
    rows = [
        ["contexts of other ops", *[f"{v:,}" for v in N_OBIN]],
        ["target null", *[num(v) for v in GAIN_NULL]],
    ]
    for n in RECIPES:
        rows.append([f"`{SHOWN[n]}`", *[span(grade(n, "gain")[:, b]) for b in range(len(OTHER_LABEL))]])
    return table_html(
        [f"gain, by posterior on `{DIFF}`", *OTHER_LABEL],
        rows,
        """**The gain on the contexts of the other ops**, in expected exact match under the full edit, net of the
        mean of the controls, by bin of the posterior on `difference`; seed mean and range over eight runs. The
        target null row is the gain it predicts for the Bayes predictor.""",
    )


# --- E3: the landing and where the answers go -----------------------------------------------------------------


def tv(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Total variation between a palette distribution *q* and a model distribution *p*, with the mass *p* puts off
    the palette counted as distance.
    """
    return 0.5 * (np.abs(q - p).sum(-1) + np.clip(1.0 - p.sum(-1), 0, None))


QN_D = BAYES["q_null"][IS_D]
QI_D = BAYES["q_ideal"][IS_D]


def landing(n: str, s: int) -> dict[str, Any]:
    r = DOSE[(n, s)]
    pc, pf = r["p_clean"][IS_D].astype(np.float64), r["p_full"][IS_D].astype(np.float64)
    return {
        "landing": float(1 - tv(QN_D, pf).mean() / tv(QN_D, pc).mean()),
        "left": float(tv(QN_D, pf).mean()),
        "clean_to_bayes": float(tv(QI_D, pc).mean()),
    }


# REVIEW: the split of the distance through the nearest mixture of the ops, and the table of where the answers put
# their mass, were dropped (paper round 1): the sans-diff report gives both for these runs (E3 there), and
# edit-effects for `falloff`, so E3 now points to them. Verify: the mixture and mass rows of the E3 tables of
# sans-diff, and that this report no longer quotes them.
LAND = {(n, s): landing(n, s) for n in RECIPES for s in SEEDS_OF[n]}


def land(n: str, key: str) -> np.ndarray:
    return np.array([LAND[(n, s)][key] for s in SEEDS_OF[n]])


LEFT_OVER_REF = {n: float(land(n, "left").mean() / land(n, "clean_to_bayes").mean()) for n in RECIPES}


def landing_table() -> str:
    keys = (("landing ↑", "landing"), ("distance left ↓", "left"), ("clean to Bayes", "clean_to_bayes"))
    rows = [[label, *[span(land(n, k)) for n in RECIPES]] for label, k in keys]
    return table_html(
        ["on `difference` contexts", *[f"`{SHOWN[n]}`" for n in RECIPES]],
        rows,
        f"""**How far the edited answers are from the target null**, in total variation on the held-out `{DIFF}`
        contexts, seed mean and range over eight runs. The landing is the share of the clean distance to the target
        null that the full edit covers. The distance from the clean model to the Bayes predictor it learned to match
        is a yardstick.""",
    )


# The confusion matrices of H2 in ex-2.2.22, on `τ_ℓ fixed`.
MID_LO, MID_HI = ee.ex2216.MIDDLE_BAND
CONF_BINS = (0.0, MID_LO, MID_HI, 1.0)
MIN_BIN = 30  # a row of a matrix with fewer contexts than this in a run is left blank, as in ex-2.2.22


@memo
def gives_table(pair: np.ndarray) -> np.ndarray:
    """For each op and context, which colors that op can give on the query pair: `(ops, n, colors)`."""
    n = len(pair)
    out = np.zeros((TABLE.n_ops, n, P.N_COLORS), dtype=bool)
    rows = np.arange(n)
    for o in range(TABLE.n_ops):
        idx, ok = TABLE.idx[o, pair], TABLE.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            out[o, rows[ok[:, j]], idx[ok[:, j], j]] = True
    return out


def confusion(p: np.ndarray, op_ids: np.ndarray, post: np.ndarray, pair: np.ndarray, lo: float, hi: float) -> np.ndarray:  # fmt: skip
    """Where a distribution puts its mass, by op, as in ex-2.2.20, over the contexts whose posterior on the true op
    lies in [lo, hi). Rows are the true op. The diagonal is the mass on the colors the true op can give for the
    query operands; off it, the mass on the colors the column op can give and the true op cannot.
    """
    rows = np.arange(len(p))
    on_true = post[rows, op_ids]
    sel = (on_true >= lo) & ((on_true < hi) if hi < 1 else (on_true <= hi))
    gives = gives_table(pair)[:, sel]
    p, true_op = p[sel].astype(float), op_ids[sel]
    rows = np.arange(len(p))
    true = gives[true_op, rows]
    per_col = np.stack(
        [(p * np.where((true_op == o)[:, None], true, gives[o] & ~true)).sum(1) for o in range(len(OPS))], 1
    )
    m = np.full((len(OPS), len(OPS)), np.nan)
    for o in range(len(OPS)):
        if (true_op == o).sum() >= MIN_BIN:
            m[o] = per_col[true_op == o].mean(0)
    return m


@memo
def confusion_set(p_clean: list[np.ndarray], p_full: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
    """Per bin of the posterior on the true op, the seed-mean confusion matrix of the clean model, the edited model,
    and the target null.
    """
    op_ids, post, pair = OP_IDS.astype(np.int64), d0["posterior"], d0["query_pair"]
    out: dict[str, list[np.ndarray]] = {"clean": [], "edited": [], "target null": []}
    for lo, hi in zip(CONF_BINS, CONF_BINS[1:], strict=False):
        args = (op_ids, post, pair, lo, hi)
        out["clean"].append(np.nanmean([confusion(p, *args) for p in p_clean], axis=0))
        out["edited"].append(np.nanmean([confusion(p, *args) for p in p_full], axis=0))
        out["target null"].append(confusion(BAYES["q_null"], *args))
    return out


CONFUSION = confusion_set(
    [DOSE[(FIXED, s)]["p_clean"] for s in SEEDS_OF[FIXED]], [DOSE[(FIXED, s)]["p_full"] for s in SEEDS_OF[FIXED]]
)
PRINT_FLOOR = 0.01


def d_row_off(kind: str) -> list[float]:
    """The mean entry off the diagonal in the `difference` row, by bin."""
    return [float(np.nanmean(np.delete(m[D], D))) for m in CONFUSION[kind]]


def other_rows_off(kind: str, among_others: bool = False) -> list[float]:
    """The mass off the diagonal in the rows of the other ops, summed over the row and averaged over rows, by bin.
    With *among_others*, the `difference` column is left out, so it is the mass the other ops put on each other.
    """
    out = []
    for m in CONFUSION[kind]:
        off = np.where(np.eye(len(OPS), dtype=bool), 0.0, m)
        if among_others:
            off = np.delete(off, D, axis=1)
        out.append(float(np.nanmean(off[OTHER].sum(1))))
    return out


@memo
def confusion_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-effects-confusion",
        alt_text=alt_text,
        caption=f"""
            **Where the answers go, by how sure the context is**, on `τ_ℓ fixed` (seed mean over eight runs). Op
            confusion, as in ex-2.2.22, over the held-out contexts of every op. Rows: the true op. Columns: the mass
            on the colors the column op can give for the query operands and the true op cannot. The diagonal (the
            mass on the answers of the true op) is outlined, printed in grey italics, and left out of the color
            scale. One row of panels per bin of the posterior on the true op; a row of a matrix with fewer than
            {MIN_BIN} contexts is blank. Values of at least {PRINT_FLOOR:g} are printed.
        """,
    )
    def _plot() -> plt.Figure:
        nb = len(CONF_BINS) - 1
        fig, axes = plt.subplots(nb, 3, figsize=(8.4, 2.7 * nb + 0.6), layout="constrained", sharex=True, sharey=True)
        axes = cast(AxesGrid, axes)
        n = len(OPS)
        eye = np.eye(n, dtype=bool)
        mats = [np.array(m) for v in data.values() for m in v]
        vmax = max(float(np.nanmax(np.where(eye, np.nan, m))) for m in mats)
        cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
        cmap.set_bad(light_dark("#fff", "#111"))
        im = None
        for c, (name, ms) in enumerate(data.items()):
            for b, m in enumerate(ms):
                m = np.array(m)
                ax = axes[b][c]
                im = ax.imshow(np.where(eye, np.nan, m), cmap=cmap, vmin=0, vmax=vmax)
                for i, j in np.ndindex(n, n):
                    if i == j:
                        ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec="0.6", lw=0.6))
                        if np.isfinite(m[i, j]):
                            ax.text(j, i, f"{m[i, j]:.2f}".removeprefix("0"), ha="center", va="center", fontsize=6,
                                    color=light_dark("#777", "#999"), fontstyle="italic")  # fmt: skip
                    elif np.isfinite(m[i, j]) and m[i, j] >= PRINT_FLOOR:
                        dark_cell = (m[i, j] / vmax > 0.55) == (light_dark(0, 1) == 0)
                        ax.text(j, i, f"{m[i, j]:.2f}"[1:], ha="center", va="center", fontsize=6,
                                color="#fff" if dark_cell else "#000")  # fmt: skip
                if b == 0:
                    ax.set_title(name, fontsize=9)
                ax.set_xticks(range(n), OPS, rotation=90, fontsize=7)
                ax.set_yticks(range(n), OPS, fontsize=7)
        for b in range(nb):
            axes[b][0].set_ylabel(f"posterior {CONF_BINS[b]:g} to {CONF_BINS[b + 1]:g}", fontsize=8)
        assert im is not None
        fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.4, label="mass")
        return fig

    return _plot()


def confusion_figure() -> str:
    data = {k: [m.tolist() for m in v] for k, v in CONFUSION.items()}
    alt = f"""
        Under the edit, the {DIFF} row spreads its mass over the other ops about as the target null does, in each
        of the {len(CONF_BINS) - 1} bins of the posterior, with less on lighten. The rows of the other ops change
        little, apart from their {DIFF} column, which empties.
    """
    return confusion_draw(data, alt)


# --- E4: the write bound --------------------------------------------------------------------------------------


@memo
def turn_maps(clean: np.ndarray, arriving: np.ndarray, op_ids: np.ndarray, fits_d: np.ndarray) -> dict:
    """The bound and the write of one run as alignments, `(slices, positions)`: the 99th percentile of |α| over the
    contexts of the other ops, clean and as it arrives under the edit. At the answers (of the examples and the
    query) only the answers that do not fit `difference` count; the example answers that do are summarized apart
    (median and 99th percentile, per slice).
    """
    other = op_ids != D
    c, a = np.abs(clean.astype(np.float32)), np.abs(arriving.astype(np.float32))
    bound, write = np.quantile(c[:, other], Q, axis=1), np.quantile(a[:, other], Q, axis=1)
    fit = {}
    for name, keep in (("fit", fits_d), ("nofit", ~fits_d)):
        cc = np.concatenate([c[:, other & keep[:, e], p] for e, p in enumerate(ANSWERS)], axis=1)
        aa = np.concatenate([a[:, other & keep[:, e], p] for e, p in enumerate(ANSWERS)], axis=1)
        fit[name] = {
            "n": cc.shape[1],
            "q_clean": np.quantile(cc, Q, axis=1),
            "q_arriving": np.quantile(aa, Q, axis=1),
            "med_clean": np.median(cc, axis=1),
            "med_arriving": np.median(aa, axis=1),
        }
    for e, p in enumerate(ANSWERS + [QUERY_ANSWER]):
        keep = other & ~fits_d[:, e]
        bound[:, p] = np.quantile(c[:, keep, p], Q, axis=1)
        write[:, p] = np.quantile(a[:, keep, p], Q, axis=1)
    return {"bound": bound, "write": write, **fit}


TURN = {(n, s): turn_maps(ALPHA[(n, s)]["clean"], ALPHA[(n, s)]["arriving"], OP_IDS, FITS_D) for n in RECIPES for s in SEEDS_OF[n]}  # fmt: skip


def seed_mean(n: str, key: str, sub: str | None = None) -> np.ndarray:
    return np.mean([TURN[(n, s)][key] if sub is None else TURN[(n, s)][key][sub] for s in SEEDS_OF[n]], axis=0)


def angle(alpha) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(alpha, 0, 1)))


BOUND = {n: seed_mean(n, "bound") for n in RECIPES}
WRITE = {n: seed_mean(n, "write") for n in RECIPES}
RATIO = {n: np.arcsin(WRITE[n]) / np.arcsin(BOUND[n]) for n in RECIPES}  # (slices, positions), seed-mean quantiles
for _n in RECIPES:
    assert np.allclose(RATIO[_n][0], 1.0, atol=1e-3), "the embedding sees the clean stream"
RATIO_MAX = {n: float(RATIO[n][1:].max()) for n in RECIPES}
RATIO_WORST = {n: np.unravel_index(RATIO[n][1:].argmax(), RATIO[n][1:].shape) for n in RECIPES}
# Per run: the sites past the embedding where the write goes over the tolerance, and the bound there in degrees.
OVER = {
    n: [
        (s, sl, p, float(r), float(angle(TURN[(n, s)]["bound"][sl, p])))
        for s in SEEDS_OF[n]
        for sl in range(1, 5)
        for p in range(N_POS)
        if (r := np.arcsin(TURN[(n, s)]["write"][sl, p]) / np.arcsin(TURN[(n, s)]["bound"][sl, p])) > TOLERANCE
    ]
    for n in RECIPES
}
OVER_SLICES = {n: sorted({o[1] for o in OVER[n]}) for n in RECIPES}
OVER_BOUND_MAX = {n: max((o[4] for o in OVER[n]), default=0.0) for n in RECIPES}
OVER_EXCESS_MAX = {n: max(((o[3] - 1) * o[4] for o in OVER[n]), default=0.0) for n in RECIPES}
ANS_RATIO = {n: np.arcsin(seed_mean(n, "nofit", "q_arriving")) / np.arcsin(seed_mean(n, "nofit", "q_clean")) for n in RECIPES}  # fmt: skip
N_FIT = TURN[(FIXED, SEEDS_OF[FIXED][0])]["fit"]["n"]
N_NOFIT = TURN[(FIXED, SEEDS_OF[FIXED][0])]["nofit"]["n"]


@memo
def turn_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-effects-turn",
        alt_text=alt_text,
        caption=f"""
            **The bound and the write on the contexts of the other ops, by slice**, at three roles: the example
            answers that do not fit `{DIFF}`, the query answer, and the largest over every other position. The
            {Q:.0%} quantile of |α| as an angle, clean (the bound, solid) and as it arrives under the full edit (the
            write, dashed), seed mean over eight runs of `τ_ℓ fixed` (seeds 712 to 719) and of `falloff` (704 to 711).
            The edit acts at every slice, so the write at a slice is the α a state arrives with there, before the edit
            at that slice takes e₁ out.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.5), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        x = np.arange(len(SLICE_TICK))
        for ax, (title, key) in zip(axes, (("example answers", "answers"), ("query answer", "query"), ("other positions", "rest")), strict=True):  # fmt: skip
            for n in reversed(RECIPES):
                ax.plot(
                    x, data[n][key]["bound"], color=ink(n), lw=1.4, marker=MARK[n], ms=3.5, label=f"{SHOWN[n]}, bound"
                )
                ax.plot(x, data[n][key]["write"], color=ink(n), lw=1.2, ls="--", marker=MARK[n], ms=3.5, mfc="none",
                        label=f"{SHOWN[n]}, write")  # fmt: skip
            ax.set_title(title, fontsize=8)
            ax.set_xticks(x, SLICE_TICK, fontsize=7)
            ax.set_xlabel("slice", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel(f"angle at the {Q:.0%} quantile (°)", fontsize=8)
        axes[0].set_ylim(0, None)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


def rest_of(m: np.ndarray) -> np.ndarray:
    """The largest value over the positions that are not answers, by slice."""
    keep = [p for p in range(N_POS) if p not in (*ANSWERS, QUERY_ANSWER)]
    return m[:, keep].max(axis=1)


TURN_DATA = {
    n: {
        "answers": {
            "bound": angle(seed_mean(n, "nofit", "q_clean")).tolist(),
            "write": angle(seed_mean(n, "nofit", "q_arriving")).tolist(),
        },
        "query": {
            "bound": angle(BOUND[n][:, QUERY_ANSWER]).tolist(),
            "write": angle(WRITE[n][:, QUERY_ANSWER]).tolist(),
        },
        "rest": {"bound": angle(rest_of(BOUND[n])).tolist(), "write": angle(rest_of(WRITE[n])).tolist()},
    }
    for n in RECIPES
}


def turn_figure() -> str:
    t = TURN_DATA[FIXED]
    alt = (
        "On both recipes the write falls far below the bound from slice 2 on at the example and query answers, and "
        "stays near it at the other positions. The τ_ℓ fixed bound peaks at slice 3, at "
        f"{t['answers']['bound'][3]:.0f}° for the example answers and {t['query']['bound'][3]:.0f}° for the query "
        "answer, where the falloff bound does not."
    )
    return turn_draw(TURN_DATA, alt)


def turn_table() -> str:
    rows = []
    for n in RECIPES:
        for name, sub in (("do not fit", "nofit"), ("fit", "fit")):
            c, a = seed_mean(n, sub, "q_clean"), seed_mean(n, sub, "q_arriving")
            rows.append([f"`{SHOWN[n]}`, {name}"] + [f"{angle(x):.0f}° → {angle(y):.0f}°" for x, y in zip(c, a, strict=True)])  # fmt: skip
    for n in RECIPES:
        rows.append(
            [f"`{SHOWN[n]}`, query answer"]
            + [
                f"{angle(x):.0f}° → {angle(y):.0f}°"
                for x, y in zip(BOUND[n][:, QUERY_ANSWER], WRITE[n][:, QUERY_ANSWER], strict=True)
            ]  # fmt: skip
        )
    return table_html(
        ["answers of the other ops", *SLICE_TICK],
        rows,
        f"""**The bound and the write at the answers of the other ops**, by slice: arcsin |α| at the
        {Q:.0%} quantile over those answers, clean → as it arrives under the full edit. Seed mean over eight runs,
        each quantile taken per run and then averaged. An example answer *fits* when `{DIFF}` could give it on its
        operands ({N_FIT:,} answers per run, against {N_NOFIT:,} that do not); at the query answer only those that
        do not fit count.""",
    )


def over_table() -> str:
    rows = []
    for n in RECIPES:
        sl, p = RATIO_WORST[n]
        rows.append(
            [
                f"`{SHOWN[n]}`",
                f"{RATIO_MAX[n]:.2f}",
                f"slice {sl + 1}, `{POS_LABEL[p]}` ({angle(BOUND[n][1:][sl, p]):.1f}°)",
                f"{len(OVER[n])} of {len(SEEDS_OF[n]) * 4 * N_POS}",
                f"{OVER_BOUND_MAX[n]:.1f}°",
                f"{OVER_EXCESS_MAX[n]:.1f}°",
            ]
        )
    return table_html(
        ["recipe", "largest write / bound ↓", "where (bound)", "sites over ↓", "largest bound there", "largest excess"],
        rows,
        f"""**The write against the bound past the embedding.** The largest ratio of write to bound in angle on the
        seed mean, over every slice and position past the embedding, and where it falls. The sites over count slice
        and position pairs over all eight runs, each run taken on its own, where the write is more than
        {TOLERANCE}× the bound (the tolerance of ex-2.2.1); the last two columns give the widest bound among them
        and the largest excess of the write over the bound there.""",
    )


def tail_share(n: str, s: int, sl: int = 3) -> float:
    """Of the example answers of other ops that do not fit `difference`, the share with |α| above 0.5 at slice *sl*
    on the clean pass: the tail that sets the bound there.
    """
    c = np.abs(ALPHA[(n, s)]["clean"][sl].astype(np.float32))
    keep = ~FITS_D[:, : fc.K] & ~IS_D[:, None]
    near = np.stack([c[:, p] for p in ANSWERS], 1) > 0.5
    return float((near & keep).sum() / keep.sum())


TAIL = {n: np.array([tail_share(n, s) for s in SEEDS_OF[n]]) for n in RECIPES}
NOFIT_BOUND_3 = {n: np.array([angle(TURN[(n, s)]["nofit"]["q_clean"][3]) for s in SEEDS_OF[n]]) for n in RECIPES}
# REVIEW: the t interval on the slice-3 bound only just clears zero (claims review 2), so E4 calls the difference
# suggestive. Now that `gap` takes the t interval, E4 quotes it once (paper round 1, claims follow-up). The assert keeps
# that wording tied to the data. Verify: NOFIT_BOUND_3_T.
NOFIT_BOUND_3_T = t_interval(NOFIT_BOUND_3[FIXED], NOFIT_BOUND_3[FALLOFF])
assert 0 < NOFIT_BOUND_3_T[0] < 0.05 * NOFIT_BOUND_3_T[1], "E4 calls the wider bound at slice 3 suggestive"
NOFIT_MEDIAN_3 = {n: float(angle(seed_mean(n, "nofit", "med_clean")[3])) for n in RECIPES}
# The share of the example answers that fit `difference`: about one in this many, on the contexts of other ops.
FIT_ONE_IN = (N_FIT + N_NOFIT) / N_FIT
# In the rows of the other ops, the mass on the answers only `difference` gives, by bin (the `difference` column).
D_COL = {k: [float(np.nanmean(m[OTHER, D])) for m in v] for k, v in CONFUSION.items()}
D_ROW_DIAG = {k: [float(m[D, D]) for m in v] for k, v in CONFUSION.items()}
# REVIEW: the prose on the rows of the other ops quotes the mass they put on each other, leaving the `difference`
# column out, since that column empties under the edit and hid part of the move when it was counted in the total
# (claims review). Verify: the confusion figure, the rows other than `difference`, columns other than `difference`.
OTHER_OFF = {k: other_rows_off(k, among_others=True) for k in CONFUSION}
D_OFF = {k: d_row_off(k) for k in CONFUSION}
# REVIEW: "about as much mass on the answers of each other op as the target null" narrowed to "in total", with
# `lighten` named as the op that gets about half its share in the two surer bins (claims review). Verify: the
# `difference` row of the edited and target null matrices in the confusion figure, `lighten` column.
# REVIEW: the Write entry of the measurements said "within the bound when it is no larger", while E4 scores the
# write against the 1.25× tolerance of ex-2.2.1 and the largest seed-mean ratio is above 1; the entry now gives the
# tolerance (claims review). Verify: the over table and RATIO_MAX.
# The one other op that gets much less of the mass of `difference` than the target null gives it, in the surer bins.
# REVIEW: "about half its share" is now "between a half and three quarters" (claims review 2): the ratios are about
# 0.7 and 0.5 in the two surer bins. Every run of `τ_ℓ fixed` is short there, so the pooled shortfall is not one run.
# Verify: the assert below, and `confusion` per run at [D, lighten].
D_ROW_SHORT = {k: [float(m[D, OPS.index("lighten")]) for m in CONFUSION[k]] for k in ("edited", "target null")}
assert all(e < 0.75 * t for e, t in zip(D_ROW_SHORT["edited"][1:], D_ROW_SHORT["target null"][1:], strict=True))
NOFIT_BOUND_3_FALLOFF = NOFIT_BOUND_3[FALLOFF]
N_SPILL_POSITIVE = {n: int((SPILL[n].max(axis=1) > 0).sum()) for n in RECIPES}
FALL_AFTER = {n: float((REMOVAL[n][:, -2] - REMOVAL[n][:, -1]).max()) for n in RECIPES}
N_FALL_AFTER = {n: int((REMOVAL[n][:, -1] < REMOVAL[n][:, -2]).sum()) for n in RECIPES}
MOVE_AFTER = {n: float(np.abs(REMOVAL[n][:, -1] - REMOVAL[n][:, -2]).max()) for n in RECIPES}


def tail_posterior(n: str, s: int, sl: int = 3) -> tuple[float, float]:
    """The mean posterior on `difference` over the contexts of other ops with an example answer in the tail at slice
    *sl* (one that does not fit, with |α| above 0.5), and over the rest of the contexts of other ops.
    """
    c = np.abs(ALPHA[(n, s)]["clean"][sl].astype(np.float32))
    near = (np.stack([c[:, p] for p in ANSWERS], 1) > 0.5) & ~FITS_D[:, : fc.K] & ~IS_D[:, None]
    tail = near.any(1)
    return float(POST_D[tail].mean()), float(POST_D[~IS_D & ~tail].mean())


TAIL_POST = {s: tail_posterior(FIXED, s) for s in SEEDS_OF[FIXED]}
# The runs whose tail contexts have at least twice the posterior on `difference` of the rest.
TAIL_POST_HIGH = [s for s, (t, r) in TAIL_POST.items() if t >= 2 * r]
# The runs whose bound at the example answers that do not fit is wider at slice 3 than at slice 1.
RISES_TO_3 = {
    n: sum(TURN[(n, s)]["nofit"]["q_clean"][3] > TURN[(n, s)]["nofit"]["q_clean"][1] for s in SEEDS_OF[n])
    for n in RECIPES
}


@memo
def example_colors(holdout_ref: str, checkpoint_ref: str) -> np.ndarray:
    """The palette index of the two operands and the answer of each example of the held-out contexts:
    `(contexts, examples, 3)`. The held-out set stores tokens, and the token to color map comes from a checkpoint;
    every run shares the tokenizer, as `fits` in the edit-effects experiment assumes.
    """
    from mini.store import get, get_ref, store_context

    ex2216 = ee.ex2216
    with store_context(project_store()), tempfile.TemporaryDirectory() as tmp:
        holdout, checkpoint = get_ref(holdout_ref), get_ref(checkpoint_ref)
        assert holdout is not None and checkpoint is not None
        _, _, _, tok2color = ex2216._load_run(checkpoint, Path(tmp))
        ho = ex2216.load_holdout(get(holdout, Path(tmp) / "holdout.npz"), fc.K)
    assert (ho.op_ids == OP_IDS).all()
    return np.stack([np.stack([tok2color[ho.tokens[:, UNIT * e + j]] for j in (0, 2, 4)], 1) for e in range(fc.K)], 1)


def grid_steps(colors: np.ndarray) -> np.ndarray:
    """`(contexts, examples)`: how many steps of the color grid (one level, in one channel) an example answer is from
    the nearest answer `difference` can give on its operands. Fitting answers are at zero steps.
    """
    from sca.data.ops import PALETTE

    a, b, y = colors[..., 0], colors[..., 1], colors[..., 2]
    assert (y >= 0).all()
    pair = a * len(PALETTE) + b
    idx, prob = TABLE.idx[D, pair], TABLE.prob[D, pair]  # (contexts, examples, answers)
    steps = np.abs(GRID[np.maximum(idx, 0)] - GRID[y][..., None, :]).sum(-1)
    return np.where(prob > 0, steps, 99).min(-1)


def palette_grid() -> np.ndarray:
    """`(colors, 3)`: the level of each palette color in each channel."""
    from sca.data.ops import PALETTE

    rgb = np.array(list(PALETTE.values()), float)
    levels = np.unique(rgb)
    assert np.allclose(np.diff(levels), levels[1] - levels[0]), "the palette is an even grid"
    return np.rint(rgb / (levels[1] - levels[0])).astype(int)


GRID = palette_grid()
EXAMPLE_COLORS = example_colors(
    ee.ex2221.HOLDOUT_REF.format(key=ee.ex2222.MAIN_KEY),
    ee.ex2223.CHECKPOINT_REF.format(label=ee.ex2223.label_of("control", ee.EPOCHS, SEEDS_OF[FALLOFF][0])),
)
STEPS = grid_steps(EXAMPLE_COLORS)
NEAR = STEPS <= 1  # a near miss, or an answer that fits
# The fitting answers are at zero steps by definition; check that the colors line up with the stored fits.
assert ((STEPS == 0) == FITS_D[:, : fc.K]).all()
# The example answers of other ops that do not fit `difference`, and the share of them that are near misses.
NOFIT_OTHER = ~FITS_D[:, : fc.K] & ~IS_D[:, None]
NEAR_BASE = float(NEAR[NOFIT_OTHER].mean())


def tail_near(n: str, s: int, sl: int = 3) -> tuple[float, float]:
    """Among the example answers of other ops that do not fit `difference`: the share within one grid step of a
    `difference` answer in the tail at slice *sl* (|α| above 0.5 on the clean pass), and among the rest.
    """
    c = np.abs(ALPHA[(n, s)]["clean"][sl].astype(np.float32))
    near = np.stack([c[:, p] for p in ANSWERS], 1) > 0.5
    return float(NEAR[near & NOFIT_OTHER].mean()), float(NEAR[~near & NOFIT_OTHER].mean())


TAIL_NEAR = {n: np.array([tail_near(n, s) for s in SEEDS_OF[n]]) for n in RECIPES}  # (seeds, [tail, rest])


def in_tail(n: str, sl: int = 3) -> np.ndarray:
    """`(seeds, contexts, examples)`: whether each example answer has |α| above 0.5 at slice *sl* on the clean pass."""
    return np.stack(
        [np.stack([np.abs(ALPHA[(n, s)]["clean"][sl][:, p].astype(np.float32)) for p in ANSWERS], 1) > 0.5
         for s in SEEDS_OF[n]]
    )  # fmt: skip


# REVIEW: added the rate in the tail by distance from the nearest `difference` answer, and the check by darkness
# (paper round 1, claims follow-up). The rate falls off steeply with each step, and near misses are in the tail far
# more often than the rest in every stratum of darkness, so the account does not rest on how dark the answers are.
# With that, E4 and the discussion say "much of the tail is" again. Verify: TAIL_BY_STEP and TAIL_BY_DARK.
STEP_SHOWN = (1, 2, 3, 4, 5)
TAIL_BY_STEP = {
    n: [float(in_tail(n)[:, NOFIT_OTHER & (STEPS == k)].mean()) for k in STEP_SHOWN] for n in RECIPES
}  # pooled over runs
N_AT_STEP = [int((NOFIT_OTHER & (STEPS == k)).sum()) for k in STEP_SHOWN]  # answers per run
assert all(np.all(np.diff(TAIL_BY_STEP[n]) < 0) for n in RECIPES)
# Darkness: the sum of the three channel levels of the answer, in quartiles over the answers that do not fit.
_dark = GRID[EXAMPLE_COLORS[..., 2]].sum(-1)
_dark_q = np.digitize(_dark, np.quantile(_dark[NOFIT_OTHER], [0.25, 0.5, 0.75]))
_tail_fixed = in_tail(FIXED)
TAIL_BY_DARK = np.array(
    [
        [float(_tail_fixed[:, NOFIT_OTHER & (_dark_q == q) & m].mean()) for m in (STEPS == 1, STEPS > 1)]
        for q in range(4)
    ]
)  # (quartiles, [near misses, the rest]) on `τ_ℓ fixed`, pooled over runs
assert TAIL_BY_DARK[:, 0].min() > 5 * TAIL_BY_DARK[:, 1].max()


def steps_table() -> str:
    rows = [["answers per run", *[f"{v:,}" for v in N_AT_STEP]]]
    rows += [[f"`{SHOWN[n]}`", *[f"{v:.1%}" for v in TAIL_BY_STEP[n]]] for n in RECIPES]
    return table_html(
        ["grid steps from a `difference` answer", *[str(k) for k in STEP_SHOWN]],
        rows,
        f"""**How often an answer is in the tail, by how near it comes to fitting**: of the example answers of the
        other ops that do not fit `{DIFF}`, the share with |α| above 0.5 at slice 3 on the clean pass, by the number
        of steps of the color grid to the nearest answer `{DIFF}` could give on its operands, pooled over the eight
        runs of each recipe. One step is a near miss.""",
    )


# %%

r"""
# What the edit does on τ_ℓ fixed

/// tip |
<!-- lede -->
We scored the edit on the recipe of record, at its fresh seeds, the way the edit-effects report scored the recipe before it. We also added the confusion matrices of ex-2.2.22. The two recipes come out about the same on every measure, apart from a bound at slice 3 that is wider on the seed mean, set by a small tail of states, many of them near misses for `difference`. <!-- REVIEW: restored "on the seed mean" (paper round 1, claims follow-up): the t interval on the difference at slice 3 only just clears zero, so the report calls it suggestive. Verify: NOFIT_BOUND_3_T. --> The edit takes out more of `difference` as the dose grows, and the drop follows what each context needed. The edited answers get about half of the way to the target null. And the blocks after the edit put back no more e₁ on the other ops than the clean model has there.
///

The [fixed confirmation](/docs/m2/fixed-confirm/report.py) adopted `τ_ℓ fixed` as the recipe of record for D2.2, and scored it on removal, spill, the task, and lost anchors. The [D2.2 design doc](/docs/m2/d2.2/design.md) asks for the rest of what the edit does on the same runs before the blog post: removal by dose, the grading, the landing and the write bound of the [edit-effects report](/docs/m2/edit-effects/report.py), and the confusion matrices of [ex-2.2.22](/docs/m2/ex-2.2.22/report.py).

This report needs no new runs. The fixed confirmation stored what this report reads: the dose scoring, the scoring along the other 63 axes, and the alignment with e₁ at every slice and position. E4 also reads the colors of the held-out examples. The `falloff` runs from the edit-effects report are scored the same way alongside.
"""

# %%

rf"""
## Observations

- [Removal and spill by dose (E1)](#removal-and-spill-by-dose-e1): removal rises with the dose, slowly over the first quarter of the edit and steeply over the second, and it is complete by three quarters. On {len(SEEDS_OF[FIXED]) - N_SPILL_POSITIVE[FIXED]} runs of {len(SEEDS_OF[FIXED])}, the other ops lose less under the edit along e₁ than under the same edit along a typical axis, at every dose. No run comes near the spill criterion.
- [Whether the drop follows what each context needed (E2)](#whether-the-drop-follows-what-each-context-needed-e2): the edit takes about the same share of the way to the target null whatever the posterior on `{DIFF}`, and, before the controls are subtracted, the two recipes lose the same amount bin by bin. On the contexts of other ops whose examples point at `{DIFF}`, it gives back part of what those examples cost, as on `falloff`.
- [Where the edited answers go (E3)](#where-the-edited-answers-go-e3): the edited answers get about half of the way to the target null, a little less far than a model trained without `{DIFF}`. Pooled over contexts and runs, the `{DIFF}` row of the confusion matrix matches the target null in total at every level of the posterior, though `lighten` gets well under its share. The other ops lose a little more mass to each other than on the clean pass.
- [How far the edit turns the other ops (E4)](#how-far-the-edit-turns-the-other-ops-e4): the write stays within the clean bound past the embedding, as on `falloff`. The bound at the answers of the other ops is wider at slice 3 than on `falloff` on the seed mean. It is set by about one answer in a hundred, and about half of those are near misses: one step of the color grid from an answer `{DIFF}` could give. The write there is small.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers the eight runs of `τ_ℓ fixed` that the fixed confirmation trained to the end, at seeds 712 to 719, and the eight controls at the same seeds from the [anneal confirmation](/docs/m2/anneal-confirm/report.py), trained without the anchor.

<!-- REVIEW: the recipes were compared by whether a difference was wider than the spread of the runs of each; they are now compared as two samples (paper round 1, per the design checks of the sci-report skill). Claims review 2 found that a percentile bootstrap at eight runs each covers about 90% of the time rather than 95%, and a Welch t interval about 95%, so every comparison now takes the t interval and the coverage caveat is gone (paper round 1, claims follow-up). Verify: `t_interval` and `gap`. -->
The comparison is `falloff` at seeds 704 to 711, the runs of the edit-effects report, with the eight controls of ex-2.2.23 at those seeds. The two recipes ran at different seeds, so we compare them as two samples of eight runs: the difference of the seed means, with a 95% Welch t interval, which does not assume the two recipes vary alike. The intervals are wide, since the end of a run is sensitive to small changes early in training (the fixed confirmation). With eight runs a side, a difference whose interval only just clears zero is suggestive rather than settled.

Every measurement is on the {len(OP_IDS):,} held-out contexts of ex-2.2.21, {int(IS_D.sum()):,} per op, each with three examples and a query. The edit removes a share of the e₁ component from the state at every position and every slice. That share is the dose, and E2 to E4 use the full dose.

The scores take different baselines, as the stored data allows. Removal and spill (E1) are net of the median drop along the other 63 coordinate axes of the run, the rule of the D2.2 design. That scoring keeps only a mean per op, so the scores by context (E2) are net of the mean drop of the eight controls of the recipe under the same edit, context by context. By the end of training a control is no closer to the anchored run at its own seed than to one at another seed ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)), so the mean of the eight is the steadier baseline. The landing (E3) has no baseline, and the write (E4) is compared with the clean pass of the same run.

## The measurements

Removal
:   The drop in expected exact match on `{DIFF}` under the edit, as a share of the way from the clean score to the target null. Higher is better.

Spill {{.manual}}
:   The largest drop on any other op. Lower is better, and the selectivity criterion is {CRITERION}.

Share removed
:   On a group of `{DIFF}` contexts, the mean drop divided by the mean distance from the clean score to the target null. The target null gives a share of 1 in every group. So a share that is level across the posterior means the drop is graded the way the target null predicts.

Gain
:   On the contexts of other ops, the rise in expected exact match under the edit. The target null predicts a gain where the examples pointed at `{DIFF}`, and none elsewhere.

Landing
:   How far the edit moves the answers on `{DIFF}` contexts toward the target null: the share of the clean distance that it covers, in total variation.

Distance left
:   The total variation between the edited answers and the target null, on `{DIFF}` contexts.

Bound
:   From the clean pass: arcsin of the {Q:.0%} quantile of |α| over the states of the other ops at a slice and position, as an angle. It is how far the edit turns those states there. At the answers it counts only those that do not fit `{DIFF}`.

Write
:   The same quantile, over the states as they arrive under the full edit: that is, after the preceding block has read the edited states of the slices below. The write counts as within the bound when it is at most a quarter over it (the tolerance of H4 in ex-2.2.1). We quote the ratio.

Fit {{.manual}}
:   An example or query answer fits when `{DIFF}` can give it on its operands. In a context of another op, about one example answer in {FIT_ONE_IN:.0f} fits. It is evidence for `{DIFF}`, so the edit is meant to turn it.

Near miss
:   An example answer that does not fit, but is one step of the color grid from an answer `{DIFF}` can give on its operands: one level away in one of the three channels of the 6 × 6 × 6 palette. About {NEAR_BASE:.0%} of the example answers of other ops that do not fit are near misses. E4 reads the colors of the examples from the held-out set to find them.
"""

# %%

rf"""
## Removal and spill by dose (E1)

A smaller dose takes a share of e₁ out rather than all of it. We expect removal to grow with the dose, and spill to stay low at every dose, as it did on `falloff`.

{dose_figure()}

Removal grows with the dose on every run, from zero at dose 0, which is the clean pass. It starts slowly: the first quarter of the edit removes {DOSE_SHARE[FIXED][0]:.0%} of what the full edit does ({DOSE_SHARE_RUNS[FIXED][:, 0].min():.0%} to {DOSE_SHARE_RUNS[FIXED][:, 0].max():.0%} by run), and the second quarter much more. By three quarters of the full edit, it has done all that the full edit does: on the seed mean, removal at that dose already matches removal at the full dose. Past three quarters it moves by no more than {MOVE_AFTER[FIXED]:.2f} either way on any run.

`falloff` has the same shape. On the seed mean it is a little lower at the smaller doses, by less than two sets of eight runs can tell apart: at half the edit, `τ_ℓ fixed` removes {gap(REMOVAL[FIXED][:, 1], REMOVAL[FALLOFF][:, 1])} more.

<!-- REVIEW: the paragraph below answers the paper review (what would the curve look like against arcsin(dose)?). arcsin(dose) is not an angle of the edit, since the dose is a share of the e₁ component; the angle is `turn_at_dose`, near proportional to the dose at slice 1. Verify: DOSE_TURN. Claims review 2: narrowed to slice 1, since at the later slices the states holding `difference` sit nearly on e₁ (median |α| of 0.98 at slice 3), where a partial dose turns a state by much less than its share and renormalization gives most of the alignment back (a dose of three quarters leaves about four fifths of it at slice 3), so the geometry there is back-loaded rather than proportional. Verify: `turn_at_dose` on the slice-3 states, as DOSE_TURN does on slice 1. -->
The slow start is not in how far the edit turns the states at slice 1. The dose is the share of the e₁ component taken out of each state before it is put back on the sphere, so how far a dose turns a state depends on how close the state is to e₁. At slice 1, where block 1 writes about half of the evidence for the op (the block1-edit experiment), the states holding `{DIFF}` are far enough from e₁ that a dose turns them by about that share of the full turn: {DOSE_TURN[0]:.0%} of it at a quarter of the edit, and {DOSE_TURN[1]:.0%} at half.
<!-- REVIEW: added the sentence on slices 2 to 4 (paper round 1, claims follow-up). Verify: DOSE_MEDIAN_ALPHA, DOSE_LEFT_3, and the assert on `_turn_share`. -->
At slices 2 to 4 the states holding `{DIFF}` sit nearly on e₁ (median |α| of {", ".join(f"{v:.2f}" for v in DOSE_MEDIAN_ALPHA[:-1])} and {DOSE_MEDIAN_ALPHA[-1]:.2f}), where a partial dose turns a state far less than its share, and putting the state back on the sphere gives most of the alignment back: at slice 3, three quarters of the edit leaves {DOSE_LEFT_3:.2f} of the clean alignment. So at a single slice the turn is back-loaded, with most of it in the last quarter of the edit, where the measured removal has already finished. The bend has to come from how the slices compound: the edit at slice 1 starves what arrives at the later slices, which the edit there cuts by the same share again.

Spill is below zero at every dose on {len(SEEDS_OF[FIXED]) - N_SPILL_POSITIVE[FIXED]} runs of {len(SEEDS_OF[FIXED])}: the other ops lose less under the edit along e₁ than under the same edit along a typical other axis. The largest spill of any run at any dose is {SPILL_TOP["value"]:.3f}, on seed {SPILL_TOP["seed"]} (`{SPILL_TOP["op"]}`, at the full dose), far below the criterion.

{dose_table()}

So the edit is selective at every dose, and the full edit is past the point where more dose removes more.
"""

# %%

rf"""
## Whether the drop follows what each context needed (E2)

The target null predicts the drop context by context. A context whose examples all fit `{DIFF}` should lose most, and one whose examples point elsewhere should lose little. If the anchor holds the evidence for the op, the edit should take about the same share of the way to the target null in every bin of the posterior on `{DIFF}`.

If it held something coarser, such as a color feature of `{DIFF}` answers, it would take off about the same amount everywhere, which is a larger share on the contexts that needed the op least.

The target null also predicts a gain on the contexts of other ops: where an example showed the answer of `{DIFF}`, removing the op should move the answer back toward the true op.

{grading_figure()}

<!-- REVIEW: was "3 runs [in the lowest bin] go past the target null", and "a little higher than on `falloff` in every bin, by less than the spread". Netted against the mean of the controls, no run goes past the target null, and the seed range of every bin narrows; the gap between the recipes is in the controls, since the drop without them matches bin by bin. Verify: the not-net row of the table against `raw_share` of `falloff`, and CONTROL_DROP of each recipe on `difference` contexts. -->
The share removed is about level across the posterior on both recipes, between {SHARE_MEAN[FIXED].min():.2f} and {SHARE_MEAN[FIXED].max():.2f} at the seed mean on `τ_ℓ fixed`. No run goes past the target null in any bin, so no bin shows the overshoot that a coarse feature would give. Without the controls subtracted (the last row of the table below), the lowest bin removes the smallest share of any.

The share is higher on `τ_ℓ fixed` than on `falloff` in the lower bins, but that comes from the two sets of controls rather than from the anchored runs. Without the controls subtracted, the two recipes lose the same bin by bin, and over all bins `τ_ℓ fixed` removes {gap(grade(FIXED, "raw_share_all"), grade(FALLOFF, "raw_share_all"))} more. The controls of `falloff` lose a little more under the edit than those of `τ_ℓ fixed`, so the net share comes out lower there.

Over all bins, without the controls subtracted, the edit takes `{DIFF}` contexts {span(grade(FIXED, "raw_share_all"))} of the way to the target null on its own. Under the same edit, the controls lose {CONTROL_DROP[FIXED][IS_D].mean():.2f} there on average; that is the part a model with no anchor loses too.

{grading_table()}

On the contexts of other ops, the gain grows with the posterior on `{DIFF}`, as the target null predicts, and matches `falloff` bin by bin. Where the examples point most at `{DIFF}` (a posterior of at least 0.5), the edit gains {GAIN_MEAN[FIXED][-1]:.2f} against {GAIN_NULL[-1]:.2f} for the target null. In the two lowest bins it gains a little more than the target null. The edit-effects report traced most of that excess to the controls, which lose a little there under the edit. A model trained without `{DIFF}` gains about what the target null predicts on these contexts (the sans-diff experiment, E4 there).

{gain_table()}

So the drop is graded on the recipe of record as it was on `falloff`: in proportion to what each context needed, with the contexts of other ops recovering where their examples pointed at `{DIFF}`.
"""

# %%

rf"""
## Where the edited answers go (E3)

On `{DIFF}` contexts the full edit covers {span(land(FIXED, "landing"))} of the distance from the clean answers to the target null, about the same as on `falloff`, which lands {gap(land(FALLOFF, "landing"), land(FIXED, "landing"))} further. The landing is a stricter measure than the share removed in E2, which scores only the probability on the right answer. The landing scores the whole answer distribution, including how the mass that leaves `{DIFF}` is shared among the other ops.

{landing_table()}

The distance left is about {LEFT_OVER_REF[FIXED]:.1f}× the distance between the clean model and the Bayes predictor it was trained to match.

The sans-diff experiment looks at this distance on the same runs, beside a model trained without `{DIFF}` (E2 and E3 there). Nearly all the mass the edit moves goes to answers another op could give, and the miss is in how the edited model shares it among those ops on each context, as edit-effects found on `falloff`. The model trained without `{DIFF}` lands only a little closer to the target null, and misses in the same kind of way, so most of this distance is there without the anchor and the edit.

The confusion matrices pool over contexts, and here over the eight runs too, so they show the overall proportions. They put the clean model, the edited model, and the target null side by side, at three levels of how sure the context is about its true op.

{confusion_figure()}

In the `{DIFF}` row, the edited model puts about as much mass on the answers of the other ops as the target null does in total, at every level of the posterior. The mean entry off the diagonal is {", ".join(f"{v:.2f}" for v in D_OFF["edited"])} by bin, against {", ".join(f"{v:.2f}" for v in D_OFF["target null"])} for the target null.

Most ops get about their share. `lighten` gets between a half and three quarters of its share in the two surer bins ({", ".join(f"{v:.2f}" for v in D_ROW_SHORT["edited"][1:])} against {", ".join(f"{v:.2f}" for v in D_ROW_SHORT["target null"][1:])}), as it also did on `falloff`, and `sat-hsv` gets a little more than its share. The model trained without `{DIFF}` gives `lighten` about its share (the sans-diff experiment), so the shortfall there comes with the edit.

The mass left on the answers of `{DIFF}`, on the diagonal, falls to a few hundredths.

<!-- REVIEW: added the sans-diff finding on scatter from run to run (paper round 1). Verify: the table of distances on answers averaged over seeds in E3 of sans-diff. -->
So, pooled over contexts and runs, the edit puts the mass about where the target null does, and the matrices make the edited model look closer to the target null than the landing says. Mismatches in opposite directions cancel in a pooled matrix and add up in the landing. The sans-diff experiment found that much of the mismatch on a context changes from run to run: averaging the answers of the eight runs, context by context, brings the edited model about a third closer to the target null, and averaging does about as much for a model on contexts where its ideal is not in doubt. So the matrices average away two kinds of scatter, over contexts and over runs, and what is left in them, such as the shortfall on `lighten`, is common to the runs.

In the rows of the other ops, the `{DIFF}` column empties under the edit. Where the context is least sure of its true op, the clean model puts {D_COL["clean"][0]:.2f} on the answers only `{DIFF}` gives, and the edited model {D_COL["edited"][0]:.2f}, near the zero of the target null. This is the gain of E2 seen from the other side.

The rest of those rows change little. Even on the clean pass, their diagonal is lower than that of the target null, since the clean model is some way from the Bayes predictor. Where the context is least sure, the mass that leaves the `{DIFF}` column goes to the true op and to the other ops alike, and both moves are toward the target null.

In the two surer bins, the edit moves a little more mass off the diagonal onto the other ops, which is away from the target null. The mass the other ops put on each other is {", ".join(f"{v:.2f}" for v in OTHER_OFF["clean"][1:])} clean against {", ".join(f"{v:.2f}" for v in OTHER_OFF["edited"][1:])} edited, and {", ".join(f"{v:.2f}" for v in OTHER_OFF["target null"][1:])} for the target null.
"""

# %%

rf"""
## How far the edit turns the other ops (E4)

The edit turns each state by an angle: the arcsin of the alignment (α) the state arrives with. So the clean pass sets how far the edit can turn the states of the other ops, and that is the bound. If the blocks after an edit put e₁ back on the other ops, beyond what the clean model has there, the write would rise above the bound.

{turn_figure()}

The write stays within the bound past the embedding. At the seed mean the largest ratio of write to bound is {RATIO_MAX[FIXED]:.2f}, at slice {RATIO_WORST[FIXED][0] + 1}, position `{POS_LABEL[RATIO_WORST[FIXED][1]]}`, where the bound is under a degree. Run by run, {len(OVER[FIXED])} sites of {len(SEEDS_OF[FIXED]) * 4 * N_POS} go over the tolerance of ex-2.2.1, all where the bound is at most {OVER_BOUND_MAX[FIXED]:.1f}°, and none by more than {OVER_EXCESS_MAX[FIXED]:.1f}°. That is about the same as on `falloff`.

{over_table()}

From slice 2 on, the other ops arrive with much less e₁ than the clean model gives them, as on `falloff`: at the example answers that do not fit, the write is {ANS_RATIO[FIXED][2]:.0%} of the bound at slice 2 and {ANS_RATIO[FIXED][4]:.0%} at slice 4. The edit-effects report traced this to the edit at the slices before, which leaves each block nothing on e₁ to carry through. The block1-edit experiment found why that is enough: the later blocks mostly add to the e₁ that arrives, rather than computing it again.

<!-- REVIEW: the comparison of the bound at slice 3 is now two-sample (paper round 1); the interval of the difference excludes zero, so the text no longer leans on the overlap of the runs to play it down. Verify: NOFIT_BOUND_3 under `gap`. Claims review 2: the percentile interval undercovers at eight runs, and a Welch t interval on the same runs reaches zero (a Mann-Whitney test gives p of about 0.07), so the text now gives both and calls the difference suggestive. Verify: NOFIT_BOUND_3_T. Claims follow-up: `gap` now takes the t interval, so the text quotes it once, to a tenth of a degree so that a lower end near zero does not print as zero. -->
The bound itself has a different shape. On `falloff` it was widest at slice 1 at the example answers and narrowed with depth. On `τ_ℓ fixed` it widens to slice 3, to {angle(seed_mean(FIXED, "nofit", "q_clean")[3]):.0f}° against {angle(seed_mean(FALLOFF, "nofit", "q_clean")[3]):.0f}° on `falloff`, and the query answer does the same ({angle(BOUND[FIXED][3, QUERY_ANSWER]):.0f}° against {angle(BOUND[FALLOFF][3, QUERY_ANSWER]):.0f}°). Taken run by run, the bound at the example answers at slice 3 is wider on `τ_ℓ fixed` by {gap(NOFIT_BOUND_3[FIXED], NOFIT_BOUND_3[FALLOFF], ".1f", "°")}. The interval only just clears zero, so the difference is suggestive rather than settled; the bound is wider at slice 3 than at slice 1 on {RISES_TO_3[FIXED]} runs of `τ_ℓ fixed` and also on {RISES_TO_3[FALLOFF]} of `falloff`.

A small tail sets the bound at slice 3. The median example answer that does not fit would turn by {NOFIT_MEDIAN_3[FIXED]:.0f}° there, while the share close to e₁ (|α| above 0.5) is {TAIL[FIXED].min():.1%} to {TAIL[FIXED].max():.1%} by run ({TAIL[FALLOFF].min():.1%} to {TAIL[FALLOFF].max():.1%} on `falloff`).

The bound is the 99th percentile, so a tail near one in a hundred moves it a long way: it ranges from {NOFIT_BOUND_3[FIXED].min():.0f}° to {NOFIT_BOUND_3[FIXED].max():.0f}° across the runs of `τ_ℓ fixed`, and from {NOFIT_BOUND_3_FALLOFF.min():.0f}° to {NOFIT_BOUND_3_FALLOFF.max():.0f}° across those of `falloff`.

<!-- REVIEW: was "so we have no explanation for the tail". The block1-edit account (block 1 judges each example answer against its operands, and blocks 2 and 3 build on it) suggested a graded judgement; the colors of the examples show the tail is rich in near misses. Verify: TAIL_NEAR, NEAR, and `near_fit`. -->
On {len(SEEDS_OF[FIXED]) - len(TAIL_POST_HIGH)} runs of {len(SEEDS_OF[FIXED])}, the contexts in the tail have about as little posterior on `{DIFF}` as the rest, as the edit-effects report found for a tail at the query answer on `falloff`. But the posterior counts only answers that fit exactly. The block1-edit experiment found that block 1 writes the evidence for the op as a judgement on each example, whether its answer fits `{DIFF}` on its operands, and that blocks 2 and 3 build on what block 1 writes. A judgement of that kind made from colors could be graded, giving an answer that nearly fits some of the weight of one that fits.

<!-- REVIEW: the base rate of near misses is now the pooled share over all the answers that do not fit, as in the measurement entry (paper round 1, claims follow-up); it was the share among the answers outside the tail. The gradient by grid step and the check by darkness are new, and with them the wording is back to "much of the tail is". Verify: NEAR_BASE, TAIL_BY_STEP, TAIL_BY_DARK. -->
The tail fits that account. Of the answers in the tail, {TAIL_NEAR[FIXED][:, 0].min():.0%} to {TAIL_NEAR[FIXED][:, 0].max():.0%} by run are near misses, against about {NEAR_BASE:.0%} of all the answers of other ops that do not fit ({TAIL_NEAR[FALLOFF][:, 0].min():.0%} to {TAIL_NEAR[FALLOFF][:, 0].max():.0%} on `falloff`). The nearer an answer comes to one that fits, the more often it is in the tail, and the rate falls off steeply past one step of the grid (the table below). That does not come from how dark the answers are: in each quarter of them by darkness, near misses are in the tail {TAIL_BY_DARK[:, 0].min():.0%} to {TAIL_BY_DARK[:, 0].max():.0%} of the time, against about {TAIL_BY_DARK[:, 1].mean():.0%} for the rest.

{steps_table()}

Much of the tail is the model taking a near miss as weak evidence for `{DIFF}`, and turning those states is partly removal at work, as with the answers that fit. That leaves the rest of the tail unexplained, and does not say why the tail is larger on some runs than on others.

{turn_table()}

So the bound holds on the recipe of record in the sense of ex-2.2.1: the blocks after the edit put back less than the clean model has. Where the tail is large, the bound is loose: the turn we can state in advance at slice 3 is large for about one state of another op in a hundred, many of them near misses, though most turn by a couple of degrees. The spill of E1 stays low, so whatever the later blocks make of those turns costs the other ops little.
"""

# %%

r"""
## Discussion

On the recipe of record, the edit does what it did on `falloff` on every measure here, apart from the wider bound at slice 3. It removes more of `difference` as the dose grows, and leaves the other ops alone at every dose. The drop on each context follows how much that context needed the op. And the blocks after the edit put back less e₁ than the clean model has. These are claims (b) and (d) of the design doc, now shown on the recipe of record at fresh seeds.

The landing is where the edit falls short, as before. Pooled over contexts the edited model sends the mass of `difference` to the other ops in about the proportions of the target null, but context by context it weighs them differently. <!-- REVIEW: replaced "A model trained without `difference` would say how close any model of this size comes to the target null" with what the sans-diff experiment found (paper round 1). Verify: the lede and E2 of sans-diff. -->
With both recipes landing in the same place, the shortfall looks like a property of the model or of the edit rather than of the recipe. The [sans-diff experiment](/docs/m2/filtered-corpus/report.py) points to the model: trained without `difference`, it lands only a little closer to the target null than the edited model, and misses in the same kind of way, sharing the mass among the other ops differently on each context and at each seed. The anchor and the edit add a smaller part on top, and the shortfall on `lighten` in the pooled matrices is part of what they add.

Two results add to what the edit-effects report showed. Removal is nearly complete by three quarters of the full edit, so the full edit has some margin. And the bound at slice 3 is wider than on `falloff` on the seed mean, set by a tail of about one state of another op in a hundred that varies a lot between runs. Much of the tail is near misses for `difference`, which the model seems to take as weak evidence for the op, so some of that turn is removal rather than a side effect. That does not change what the edit does to the answers here, but it makes the turn we can state in advance larger. If a later recipe spills, the tail at slice 3 would be a place to look.
"""

# %%

r"""
## References

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` keeps the anchor on all eight runs at seeds 712 to 719 and all forty in the count, with removal of 0.88 on the seed mean and no run near the spill criterion; it is the recipe of record.

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   With the slice temperature annealing, the pool keeps the anchor on all eight runs at seeds 712 to 719; its controls at those seeds are the ones used here.

[edit-effects report](/docs/m2/edit-effects/report.py)
:   On `falloff`, the write stays within the clean bound, the drop is graded by the posterior, and the edited answers get about half of the way to the target null, with the shortfall in how the edited model weighs the other ops context by context.

[sans-diff experiment](/docs/m2/filtered-corpus/report.py)
:   A model trained with no loss on `difference` contexts lands a little closer to the target null than the edited `τ_ℓ fixed` runs, 0.58 of the way from the control against 0.50, and misses in the same kind of way, with much of the miss changing from run to run.

[block1-edit experiment](/docs/m2/block1-edit/report.py)
:   Block 1 writes about half of the evidence for the op onto e₁ at the example answers, as a judgement on each example, and the later blocks mostly add to the e₁ that arrives rather than computing it again.

[ex-2.2.22](/docs/m2/ex-2.2.22/report.py)
:   On an earlier recipe the edit got a little under half of the way to the target null, and the confusion matrices pooled over contexts put the mass about where the target null does.
"""
