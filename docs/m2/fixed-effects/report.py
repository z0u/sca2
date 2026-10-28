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
from scipy.optimize import nnls

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
DOSE_SHARE = {n: (REMOVAL[n] / REMOVAL[n][:, -1:]).mean(axis=0) for n in RECIPES}


@memo
def dose_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="fixed-effects-dose",
        alt_text=alt_text,
        caption=f"""
            **Removal and spill by dose**, the edit at every position and slice, net of the median drop along the
            other 63 coordinate axes of the run. One hairline per run and the seed mean bold, for `τ_ℓ fixed` at
            seeds 712 to 719 and `falloff` at 704 to 711. **Left:** removal, the share of the way from the clean score
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
    data = {
        "gammas": GAMMAS,
        "removal": {n: REMOVAL[n].tolist() for n in RECIPES},
        "spill": {n: SPILL[n].tolist() for n in RECIPES},
    }
    alt = (
        "Left: removal rises with the dose on every run of both recipes and levels off from three quarters of the "
        f"full edit. Right: spill stays below the criterion of {CRITERION} at every dose on every run of both recipes."
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


def grading(n: str, s: int) -> dict[str, Any]:
    """The share removed by bin of the posterior on `difference` contexts, net of the control at the seed and not,
    and the gain on the contexts of the other ops by bin.
    """
    r, c = DOSE[(n, s)], CONTROL[(n, s)]
    raw = r["clean_eem"] - r["eem"][-1]
    net = raw - (c["clean_eem"] - c["eem"][-1])
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
        "control_drop_d": float((c["clean_eem"] - c["eem"][-1])[IS_D].mean()),
    }


GRADE = {(n, s): grading(n, s) for n in RECIPES for s in SEEDS_OF[n]}
GAIN_NULL = np.array([-DEP_BAYES[(OBIN == b) & ~IS_D].mean() for b in range(len(OTHER_LABEL))])


def grade(n: str, key: str) -> np.ndarray:
    return np.array([GRADE[(n, s)][key] for s in SEEDS_OF[n]])


SHARE_MEAN = {n: grade(n, "share").mean(axis=0) for n in RECIPES}
GAIN_MEAN = {n: grade(n, "gain").mean(axis=0) for n in RECIPES}


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
            with the gain the target null predicts as open squares. Both net of the control at the seed.
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
            f"`{SHOWN[FIXED]}`, not net of the control",
            *[span(grade(FIXED, "raw_share")[:, b]) for b in range(5)],
            span(grade(FIXED, "raw_share_all")),
        ]
    )
    return table_html(
        [f"share removed, by posterior on `{DIFF}`", *BIN_LABEL, "all"],
        rows,
        f"""**The share of the dependence removed, by bin of the posterior**, on the held-out `{DIFF}` contexts: the
        mean drop under the full edit over the mean distance from the clean answer to the target null, in expected
        exact match, net of the control at the seed unless marked; seed mean and range over eight runs. The
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
        control at the seed, by bin of the posterior on `difference`; seed mean and range over eight runs. The
        target null row is the gain it predicts for the Bayes predictor.""",
    )


# --- E3: the landing and where the answers go -----------------------------------------------------------------


def tv(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Total variation between a palette distribution *q* and a model distribution *p*, with the mass *p* puts off
    the palette counted as distance.
    """
    return 0.5 * (np.abs(q - p).sum(-1) + np.clip(1.0 - p.sum(-1), 0, None))


def support(op: int, pairs: np.ndarray) -> np.ndarray:
    """`(n, 216)`: the colors *op* can give on each query pair (any nonzero probability)."""
    n = len(pairs)
    dense = np.zeros((n, 217), bool)
    idx = TABLE.idx[op, pairs]
    dense[np.arange(n)[:, None], np.where(idx < 0, 216, idx)] = TABLE.prob[op, pairs] > 0
    return dense[:, :216]


def answer_dists(pairs: np.ndarray) -> np.ndarray:
    """`(n, 216, ops)`: the answer distribution of every op on each query pair."""
    n = len(pairs)
    out = np.zeros((n, 217, len(OPS)))
    for o in range(len(OPS)):
        idx = TABLE.idx[o, pairs]
        np.put_along_axis(out[:, :, o], np.where(idx < 0, 216, idx), TABLE.prob[o, pairs], axis=1)
    return out[:, :216]


@memo
def nearest_mixture(p: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Per context, the mixture of the answer distributions of the ops closest to *p* (non-negative weights, least
    squares over the palette): `(n, 216)`.
    """
    A = answer_dists(pairs)
    return np.stack([A[i] @ nnls(A[i], p[i])[0] for i in range(len(p))])


PAIRS_D = d0["query_pair"][IS_D].astype(np.int64)
SUP = np.stack([support(o, PAIRS_D) for o in range(len(OPS))])  # (ops, n, 216)
OTHER_SUP = SUP[OTHER].any(0)
CATS = {
    f"only `{DIFF}`": SUP[D] & ~OTHER_SUP,
    "another op": OTHER_SUP,
    "no op": ~SUP.any(0),
}
QN_D = BAYES["q_null"][IS_D]
QI_D = BAYES["q_ideal"][IS_D]


def mass(p: np.ndarray) -> list[float]:
    return [float((p * m).sum(1).mean()) for m in CATS.values()] + [float(np.clip(1 - p.sum(1), 0, None).mean())]


def landing(n: str, s: int) -> dict[str, Any]:
    r = DOSE[(n, s)]
    pc, pf = r["p_clean"][IS_D].astype(np.float64), r["p_full"][IS_D].astype(np.float64)
    mix_f = nearest_mixture(pf, PAIRS_D)
    return {
        "landing": float(1 - tv(QN_D, pf).mean() / tv(QN_D, pc).mean()),
        "left": float(tv(QN_D, pf).mean()),
        "clean_to_bayes": float(tv(QI_D, pc).mean()),
        "off_mixture": float((0.5 * np.abs(pf - mix_f).sum(1)).mean()),
        "mixture_to_null": float(tv(QN_D, mix_f).mean()),
        "mass": {"null": mass(QN_D), "clean": mass(pc), "edit": mass(pf)},
        "op_mass_edit": [float((pf * SUP[o]).sum(1).mean()) for o in range(len(OPS))],
    }


LAND = {(n, s): landing(n, s) for n in RECIPES for s in SEEDS_OF[n]}
OP_MASS_NULL = [float((QN_D * SUP[o]).sum(1).mean()) for o in range(len(OPS))]


def land(n: str, key: str) -> np.ndarray:
    return np.array([LAND[(n, s)][key] for s in SEEDS_OF[n]])


LEFT_OVER_REF = {n: float(land(n, "left").mean() / land(n, "clean_to_bayes").mean()) for n in RECIPES}


def landing_table() -> str:
    keys = (
        ("landing ↑", "landing"),
        ("distance left ↓", "left"),
        ("clean to Bayes", "clean_to_bayes"),
        ("edited to nearest mixture ↓", "off_mixture"),
        ("nearest mixture to null ↓", "mixture_to_null"),
    )
    rows = [[label, *[span(land(n, k)) for n in RECIPES]] for label, k in keys]
    return table_html(
        ["on `difference` contexts", *[f"`{SHOWN[n]}`" for n in RECIPES]],
        rows,
        f"""**How far the edited answers are from the target null**, in total variation on the held-out `{DIFF}`
        contexts, seed mean and range over eight runs. The landing is the share of the clean distance to the target
        null that the full edit covers. The distance from the clean model to the Bayes predictor it learned to match is a
        yardstick. The last two rows are the legs of the path through the nearest mixture of the answers of the
        ops: the edited answers against that mixture, and the mixture against the target null.""",
    )


def mass_table() -> str:
    heads = [*CATS, "off the palette"]
    rows = []
    for name, key in (("target null", "null"), ("clean pass", "clean"), ("full edit", "edit")):
        rows.append([name, *[num(float(np.mean([LAND[(FIXED, s)]["mass"][key][i] for s in SEEDS_OF[FIXED]]))) for i in range(len(heads))]])  # fmt: skip
    rows.append(
        [
            "full edit, `falloff`",
            *[
                num(float(np.mean([LAND[(FALLOFF, s)]["mass"]["edit"][i] for s in SEEDS_OF[FALLOFF]])))
                for i in range(len(heads))
            ],
        ]  # fmt: skip
    )
    return table_html(
        ["answer mass on colors that…", *heads],
        rows,
        f"""**Where the answers put their mass on `{DIFF}` contexts**, by the kind of color: one only `{DIFF}` gives
        on the query pair, one another op gives (some of these `{DIFF}` gives too), one no op gives, or a token that
        is not a color. `τ_ℓ fixed` unless marked, seed mean over eight runs.""",
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
D_ROW_SHORT = {k: [float(m[D, OPS.index("lighten")]) for m in CONFUSION[k]] for k in ("edited", "target null")}
assert all(e < 0.75 * t for e, t in zip(D_ROW_SHORT["edited"][1:], D_ROW_SHORT["target null"][1:], strict=True))
NOFIT_BOUND_3_FALLOFF = NOFIT_BOUND_3[FALLOFF]
N_SPILL_POSITIVE = {n: int((SPILL[n].max(axis=1) > 0).sum()) for n in RECIPES}
FALL_AFTER = {n: float((REMOVAL[n][:, -2] - REMOVAL[n][:, -1]).max()) for n in RECIPES}
N_FALL_AFTER = {n: int((REMOVAL[n][:, -1] < REMOVAL[n][:, -2]).sum()) for n in RECIPES}
LOW_BIN_OVER = int((grade(FIXED, "share")[:, 0] > 1).sum())
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


# %%

r"""
# What the edit does on τ_ℓ fixed

/// tip |
<!-- lede -->
We scored the edit on the recipe of record, at its fresh seeds, the way the edit-effects report scored the recipe before it. We also added the confusion matrices of ex-2.2.22. The two recipes come out about the same on every measure, apart from a wider bound at slice 3 on the seed mean, which a small tail sets. The edit takes out more of `difference` as the dose grows, and the drop follows what each context needed. The edited answers get about half of the way to the target null. And the blocks after the edit put back no more e₁ on the other ops than the clean model has there.
///

The [fixed confirmation](/docs/m2/fixed-confirm/report.py) adopted `τ_ℓ fixed` as the recipe of record for D2.2, and scored it on removal, spill, the task, and lost anchors. The [D2.2 design doc](/docs/m2/d2.2/design.md) asks for the rest of what the edit does on the same runs before the blog post: removal by dose, the grading, the landing and the write bound of the [edit-effects report](/docs/m2/edit-effects/report.py), and the confusion matrices of [ex-2.2.22](/docs/m2/ex-2.2.22/report.py).

This report needs no new runs. The fixed confirmation stored everything this report reads: the dose scoring, the scoring along the other 63 axes, and the alignment with e₁ at every slice and position. The `falloff` runs from the edit-effects report are scored the same way alongside.
"""

# %%

rf"""
## Observations

- [Removal and spill by dose (E1)](#removal-and-spill-by-dose-e1): removal rises with the dose and is nearly complete by three quarters of the full edit. On {len(SEEDS_OF[FIXED]) - N_SPILL_POSITIVE[FIXED]} runs of {len(SEEDS_OF[FIXED])}, the other ops lose less under the edit along e₁ than under the same edit along a typical axis, at every dose. No run comes near the spill criterion.
- [Whether the drop follows what each context needed (E2)](#whether-the-drop-follows-what-each-context-needed-e2): the edit takes about the same share of the way to the target null whatever the posterior on `{DIFF}`, as on `falloff`. On the contexts of other ops whose examples point at `{DIFF}`, it gives back part of what those examples cost, as on `falloff`.
- [Where the edited answers go (E3)](#where-the-edited-answers-go-e3): the edited answers get about half of the way to the target null. Pooled over contexts, the `{DIFF}` row of the confusion matrix matches the target null in total at every level of the posterior, though `lighten` gets about half its share. The other ops lose a little more mass to each other than on the clean pass.
- [How far the edit turns the other ops (E4)](#how-far-the-edit-turns-the-other-ops-e4): the write stays within the clean bound past the embedding, as on `falloff`. On the seed mean the bound at the answers of the other ops is wider at slice 3 than on `falloff`, though the runs of the two recipes overlap. It is set by about one answer in a hundred, and the write there is small.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers the eight runs of `τ_ℓ fixed` that the fixed confirmation trained to the end, at seeds 712 to 719, and the control at each seed from the [anneal confirmation](/docs/m2/anneal-confirm/report.py), trained without the anchor but with the same initialization, batches, and label draws.

The comparison is `falloff` at seeds 704 to 711, the runs of the edit-effects report, with their controls from ex-2.2.23. Its numbers here are the ones that report gives. The two recipes ran at different seeds, so a difference between them also includes the variation between two draws of eight seeds. The fixed confirmation found that variation large: two runs of one seed under nearly the same recipe could end a tenth apart in removal.

So we count a difference between the recipes only where it is wider than the spread between the runs of each.

Every measurement is on the {len(OP_IDS):,} held-out contexts of ex-2.2.21, {int(IS_D.sum()):,} per op, each with three examples and a query. The edit removes a share of the e₁ component from the state at every position and every slice. That share is the dose, and E2 to E4 use the full dose.

The scores take different baselines, as the stored data allows. Removal and spill (E1) are net of the median drop along the other 63 coordinate axes of the run, the rule of the D2.2 design. That scoring keeps only a mean per op, so the scores by context (E2) are net of the control at the seed, as in the edit-effects report. The landing (E3) has no baseline, and the write (E4) is compared with the clean pass of the same run.

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
"""

# %%

rf"""
## Removal and spill by dose (E1)

A smaller dose takes a share of e₁ out rather than all of it. We expect removal to grow with the dose, and spill to stay low at every dose, as it did on `falloff`.

{dose_figure()}

Removal grows with the dose on every run. By three quarters of the full edit, it has done all that the full edit does: on the seed mean, removal at that dose already matches removal at the full dose. Past three quarters it moves by no more than {MOVE_AFTER[FIXED]:.2f} either way on any run.

The removal is not proportional to the dose: the first quarter of the edit removes {DOSE_SHARE[FIXED][0]:.0%} of what the full edit does, and the second quarter much more. `falloff` has the same shape, a little lower at the smaller doses.

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

The share removed is about level across the posterior on both recipes, between {SHARE_MEAN[FIXED].min():.2f} and {SHARE_MEAN[FIXED].max():.2f} at the seed mean on `τ_ℓ fixed`. It is a little higher than on `falloff` in every bin, though by less than the spread between the runs of `τ_ℓ fixed`, so we don't read it as a difference between the recipes.

The lowest bin is the noisiest, since its contexts needed little to begin with, and {LOW_BIN_OVER} runs there go past the target null. No other bin shows the overshoot that a coarse feature would give. Without the control subtracted (the last row of the table below), the lowest bin removes the smallest share of any.

Over all bins, without the control subtracted, the edit takes `{DIFF}` contexts {span(grade(FIXED, "raw_share_all"))} of the way to the target null on its own. Under the same edit, the control loses {np.mean(grade(FIXED, "control_drop_d")):.2f} there; that is the part a model with no anchor loses too.

{grading_table()}

On the contexts of other ops, the gain grows with the posterior on `{DIFF}`, as the target null predicts, and matches `falloff` bin by bin. Where the examples point most at `{DIFF}` (a posterior of at least 0.5), the edit gains {GAIN_MEAN[FIXED][-1]:.2f} against {GAIN_NULL[-1]:.2f} for the target null. In the two lowest bins it gains a little more than the target null. The edit-effects report traced most of that excess to the control, which loses a little there under the edit.

{gain_table()}

So the drop is graded on the recipe of record as it was on `falloff`: in proportion to what each context needed, with the contexts of other ops recovering where their examples pointed at `{DIFF}`.
"""

# %%

rf"""
## Where the edited answers go (E3)

On `{DIFF}` contexts the full edit covers {span(land(FIXED, "landing"))} of the distance from the clean answers to the target null, about the same as on `falloff`. The landing is a stricter measure than the share removed in E2, which scores only the probability on the right answer. The landing scores the whole answer distribution, including how the mass that leaves `{DIFF}` is shared among the other ops.

The edit-effects report found that, on `falloff`, the mass leaves the answers that only `{DIFF}` gives and lands on the answers of the other ops, in about the right overall proportions. The rest of the distance is in how the edited model weighs the other ops, context by context. The table below splits the distance the same way on `τ_ℓ fixed`.

{landing_table()}

Each piece is within a few hundredths of `falloff`. The edited answers sit close to a mixture of the answers of the ops, and that mixture is far from the target null, so the shortfall is again in the weights the edited model gives the other ops in each context.

The distance left is about {LEFT_OVER_REF[FIXED]:.1f}× the distance between the clean model and the Bayes predictor it was trained to match.

{mass_table()}

The confusion matrices pool over contexts, so they show the overall proportions. They put the clean model, the edited model, and the target null side by side, at three levels of how sure the context is about its true op.

{confusion_figure()}

In the `{DIFF}` row, the edited model puts about as much mass on the answers of the other ops as the target null does in total, at every level of the posterior. The mean entry off the diagonal is {", ".join(f"{v:.2f}" for v in D_OFF["edited"])} by bin, against {", ".join(f"{v:.2f}" for v in D_OFF["target null"])} for the target null.

Most ops get about their share. `lighten` gets about half of its share in the two surer bins ({", ".join(f"{v:.2f}" for v in D_ROW_SHORT["edited"][1:])} against {", ".join(f"{v:.2f}" for v in D_ROW_SHORT["target null"][1:])}), as it also did on `falloff`, and `sat-hsv` gets a little more than its share.

The mass left on the answers of `{DIFF}`, on the diagonal, falls to a few hundredths.

So, pooled over contexts, the edit puts the mass about where the target null does, and the matrices make the edited model look closer to the target null than the landing says. The two measures differ because mismatches in opposite directions cancel in a pooled matrix and add up in the landing.

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

From slice 2 on, the other ops arrive with much less e₁ than the clean model gives them, as on `falloff`: at the example answers that do not fit, the write is {ANS_RATIO[FIXED][2]:.0%} of the bound at slice 2 and {ANS_RATIO[FIXED][4]:.0%} at slice 4. The edit-effects report traced this to the edit at the slices before, which leaves each block nothing on e₁ to carry through.

On the seed mean, the bound itself has a different shape. On `falloff` it was widest at slice 1 at the example answers and narrowed with depth. On `τ_ℓ fixed` it widens to slice 3, to {angle(seed_mean(FIXED, "nofit", "q_clean")[3]):.0f}° against {angle(seed_mean(FALLOFF, "nofit", "q_clean")[3]):.0f}° on `falloff`, and the query answer does the same ({angle(BOUND[FIXED][3, QUERY_ANSWER]):.0f}° against {angle(BOUND[FALLOFF][3, QUERY_ANSWER]):.0f}°).

Run by run the two recipes overlap, though: at the example answers the bound is wider at slice 3 than at slice 1 on {RISES_TO_3[FIXED]} runs of `τ_ℓ fixed` and {RISES_TO_3[FALLOFF]} of `falloff`.

A small tail sets the bound at slice 3. The median example answer that does not fit would turn by {NOFIT_MEDIAN_3[FIXED]:.0f}° there, while the share close to e₁ (|α| above 0.5) is {TAIL[FIXED].min():.1%} to {TAIL[FIXED].max():.1%} by run ({TAIL[FALLOFF].min():.1%} to {TAIL[FALLOFF].max():.1%} on `falloff`).

The bound is the 99th percentile, so a tail near one in a hundred moves it a long way: it ranges from {NOFIT_BOUND_3[FIXED].min():.0f}° to {NOFIT_BOUND_3[FIXED].max():.0f}° across the runs of `τ_ℓ fixed`, and from {NOFIT_BOUND_3_FALLOFF.min():.0f}° to {NOFIT_BOUND_3_FALLOFF.max():.0f}° across those of `falloff`.

The edit-effects report found a tail like this at the query answer on `falloff`, with nothing in the examples to set it apart. Here too, on {len(SEEDS_OF[FIXED]) - len(TAIL_POST_HIGH)} runs of {len(SEEDS_OF[FIXED])}, the contexts in the tail have about as little posterior on `{DIFF}` as the rest, so we have no explanation for the tail.

{turn_table()}

So the bound holds on the recipe of record in the sense of ex-2.2.1: the blocks after the edit put back less than the clean model has. Where the tail is large, the bound is loose: the turn we can state in advance at slice 3 is large for about one state of another op in a hundred, though most turn by a couple of degrees. The spill of E1 stays low, so whatever the later blocks make of those turns costs the other ops little.
"""

# %%

r"""
## Discussion

On the recipe of record, the edit does what it did on `falloff` on every measure here, apart from the wider bound at slice 3. It removes more of `difference` as the dose grows, and leaves the other ops alone at every dose. The drop on each context follows how much that context needed the op. And the blocks after the edit put back less e₁ than the clean model has. These are claims (b) and (d) of the design doc, now shown on the recipe of record at fresh seeds.

The landing is where the edit falls short, as before. Pooled over contexts the edited model sends the mass of `difference` to the other ops in about the proportions of the target null, but context by context it weighs them differently. With both recipes landing in the same place, the shortfall looks like a property of the model or of the edit rather than of the recipe. A model trained without `difference` would say how close any model of this size comes to the target null.

Two results add to what the edit-effects report showed. Removal is nearly complete by three quarters of the full edit, so the full edit has some margin. And the bound at slice 3 is wider than on `falloff` on the seed mean, set by a tail of about one state of another op in a hundred that varies a lot between runs. That does not change what the edit does to the answers here, but it makes the turn we can state in advance larger. If a later recipe spills, the tail at slice 3 would be a place to look.
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

[ex-2.2.22](/docs/m2/ex-2.2.22/report.py)
:   On an earlier recipe the edit got a little under half of the way to the target null, and the confusion matrices pooled over contexts put the mass about where the target null does.
"""
