# title: What the anchor follows at the example answers

# A re-analysis of ex-2.2.22's by-count pass, with no new runs. It reads the stored alignment at every answer of the
# three-example held-out set, rebuilds the evidence of each example from the held-out tokens with the posterior module
# of ex-2.2.16, and compares the two.
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed


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


ex = _load_sibling("ex-2.2.22", "example_evidence_ex2222")
P = ex.ex2216._POSTERIOR_MODULE
K = ex.K
CONDITIONS = (ex.BASE, "anchor-whole", "k-mixed", "control")
LATE = slice(2, None)
# The slices α is averaged over: from the output of the second block on, where ex-2.2.22 (E3) saw the rise.


# --- The stored arrays ----------------------------------------------------------------------------------------


def fetch_all() -> tuple[dict, dict[str, dict[str, np.ndarray]], np.ndarray]:
    """The by-count summary, the arrays of every by-count run at three examples, and the held-out tokens."""
    store = project_store()

    def published(refs: list[str]) -> dict[str, Any]:
        got = store.get_refs(refs)
        missing = [r for r, a in got.items() if a is None]
        assert not missing, f"not published: {missing}"
        return got

    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        holdout_ref = ex.ex2221.HOLDOUT_REF.format(key=ex.MAIN_KEY)
        have = published([ex.BY_COUNT_REF, holdout_ref])
        summary_path, holdout_path = store.get_many(
            [(have[ex.BY_COUNT_REF], into / "by-count.json"), (have[holdout_ref], into / "holdout.npz")]
        )
        summary = json.loads(summary_path.read_text())
        labels = [r["label"] for r in summary["runs"]]
        refs = [ex.BY_COUNT_ARRAYS_REF.format(label=lbl) for lbl in labels]
        arts = published(refs)
        paths = store.get_many([(arts[r], into / f"{lbl}.npz") for r, lbl in zip(refs, labels, strict=True)])
        arrays = {}
        for lbl, p in zip(labels, paths, strict=True):
            with np.load(p) as z:
                arrays[lbl] = {k: z[k] for k in z.files if not k.endswith("alpha_at_answer") or k.startswith(f"k{K}_")}
        tokens = ex.ex2216.load_holdout(holdout_path, K).tokens
    return summary, arrays, tokens


SUMMARY, ARRAYS, TOKENS = fetch_all()
OPS = tuple(SUMMARY["runs"][0]["ops"])
D = OPS.index(ex.ANCHORED_OP)

# --- The evidence at each example -----------------------------------------------------------------------------


def evidence(tokens: np.ndarray, op_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per held-out `difference` context and example, `(n, K)`: the posterior on `difference` given the examples up
    to and including that one, the same given the earlier examples only (1/7 before the first), the posterior given
    that example alone, whether its answer is one `difference` can give on its pair, and the answer color.
    """
    from sca.config import TokenizerConfig
    from sca.data.incontext import vocabulary
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE

    tok = WordTokenizer(TokenizerConfig(vocabulary=sorted(vocabulary())))
    tok2color = np.full(tok.vocab_size, -1)
    tok2color[[tok.stoi[n] for n in PALETTE]] = np.arange(len(PALETTE))
    table = P.build_table(ex.ex2218.table_of(OPS))
    t = tokens[op_ids == D]
    roles = ex.answer_roles(K)[:K]
    y, a, b = tok2color[t[:, roles]], tok2color[t[:, roles - 4]], tok2color[t[:, roles - 2]]
    p = table.lookup(a * len(PALETTE) + b, y)  # (ops, n, K)
    others = (p.sum(0, keepdims=True) - p) / (len(OPS) - 1)
    rate = ex.ex2221.CUBE_RATE
    log_like = np.log((1 - ex.RHO - rate) * p + ex.RHO * others + rate / len(PALETTE)).transpose(1, 2, 0)

    def normed(log_post: np.ndarray) -> np.ndarray:
        q = np.exp(log_post - log_post.max(-1, keepdims=True))
        return (q / q.sum(-1, keepdims=True))[..., D]

    so_far = normed(log_like.cumsum(1))
    before = np.concatenate([np.full((len(t), 1), 1 / len(OPS)), so_far[:, :-1]], axis=1)
    return {"so_far": so_far, "before": before, "alone": normed(log_like), "fits": p[D] > 0, "answer": y}


EV = evidence(TOKENS, ARRAYS[SUMMARY["runs"][0]["label"]][f"k{K}_op_ids"])


def alpha(label: str) -> np.ndarray:
    """α at the example answers of the held-out `difference` contexts, averaged over `LATE` slices: `(n, K)`."""
    a = ARRAYS[label]
    keep = a[f"k{K}_op_ids"] == D
    stored = a[f"k{K}_posterior_at_answer"][keep][:, :K]
    assert np.allclose(stored, EV["so_far"], atol=1e-4), "the rebuilt posterior matches the stored one"
    return a[f"k{K}_alpha_at_answer"][keep][:, LATE, :K].astype(np.float64).mean(axis=1)


def labels_of(cond: str) -> list[str]:
    got = [r for r in SUMMARY["runs"] if r["condition"] == cond]
    return [r["label"] for r in sorted(got, key=lambda r: r["model_seed"])]


ALPHA = {lbl: alpha(lbl) for c in CONDITIONS for lbl in labels_of(c)}
SEED_OF = {r["label"]: r["model_seed"] for r in SUMMARY["runs"]}
NO_ANCHOR = [lbl for lbl in labels_of("k-mixed") if ALPHA[lbl][EV["fits"]].mean() < 0.2]
# `k-mixed` runs whose α at the answers that fit `difference` stays near the control: no anchor to explain.


def scored(cond: str) -> list[str]:
    return [lbl for lbl in labels_of(cond) if lbl not in NO_ANCHOR]


# --- Fits ------------------------------------------------------------------------------------------------------


def r2(x: np.ndarray, y: np.ndarray) -> float:
    """The squared Pearson correlation, r²."""
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1] ** 2)


def weights(y: np.ndarray) -> np.ndarray:
    """The least-squares weights of α on the one-example posterior and on the posterior before that example."""
    X = np.column_stack([np.ones(y.size), EV["alone"].ravel(), EV["before"].ravel()])
    return np.linalg.lstsq(X, y.ravel(), rcond=None)[0][1:]


def answer_color_r2(y: np.ndarray, seed: int = 0) -> float:
    """Held-out R² of α predicted by the mean α of its answer color: means from a random half of the contexts,
    scored on the other half.
    """
    half = np.random.default_rng(seed).random(len(y)) < 0.5
    g, n = EV["answer"], P.N_COLORS
    mean = np.bincount(g[half].ravel(), y[half].ravel(), n) / np.maximum(np.bincount(g[half].ravel(), minlength=n), 1)
    test, pred = y[~half].ravel(), mean[g[~half].ravel()]
    return float(1 - ((test - pred) ** 2).sum() / ((test - test.mean()) ** 2).sum())


def per_run(lbl: str) -> dict[str, float]:
    y = ALPHA[lbl]
    w = weights(y)
    misfit = ~EV["fits"]
    return {
        "so_far": r2(EV["so_far"], y),
        "share": r2(EV["fits"].cumsum(1) / np.arange(1, K + 1), y),
        "alone": r2(EV["alone"], y),
        "w_alone": float(w[0]),
        "w_before": float(w[1]),
        "color": answer_color_r2(y),
        "misfit_low": float(y[misfit & (EV["before"] < 0.3)].mean()),
        "misfit_high": float(y[misfit & (EV["before"] > 0.8)].mean()),
        "fit": float(y[EV["fits"]].mean()),
    }


STATS = {lbl: per_run(lbl) for c in CONDITIONS for lbl in labels_of(c)}


def cond_stat(cond: str, key: str) -> tuple[float, float, float]:
    v = [STATS[lbl][key] for lbl in scored(cond)]
    return float(np.mean(v)), float(np.min(v)), float(np.max(v))


def mean_of(cond: str, key: str) -> float:
    return cond_stat(cond, key)[0]


HINGE, WHOLE = ex.BASE, "anchor-whole"
ANCHORED = (HINGE, WHOLE, "k-mixed")
LO_ALONE = min(mean_of(c, "alone") for c in ANCHORED)
HI_SO_FAR = max(mean_of(c, "so_far") for c in ANCHORED)
HI_W_BEFORE = max(mean_of(c, "w_before") for c in ANCHORED)
LO_W_ALONE = min(mean_of(c, "w_alone") for c in ANCHORED)
HI_COLOR = max(mean_of(c, "color") for c in ANCHORED)


BINS = np.linspace(0, 1, 6)
MIN_BIN = 20
# Bins of 0.2: the evidence sits on a few values (most one-example posteriors are near 0.05 or 0.69), so each bin
# holds one cluster of them, and a point is drawn at the mean evidence of its bin, not the bin center. A bin with
# fewer points than `MIN_BIN` in a run is left out of that run.


def bin_of(x: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(x, BINS[1:-1]), 0, len(BINS) - 2)


COUNTS = tuple(range(1, 6))
NEXT_CONDITIONS = ("control", HINGE, "k-mixed")


def next_answer(cond: str, k: int) -> dict[str, list]:
    """On the held-out `difference` contexts with *k* examples, by bin of the posterior given those examples: where
    the points sit, the seed mean of expected exact match at the query, its seed range, and the ceiling (the ideal
    predictor). Every run of the condition, the one without an anchor included, since this is about the task.
    """
    runs = [ARRAYS[lbl] for lbl in labels_of(cond)]
    keep = runs[0][f"k{k}_op_ids"] == D
    post = runs[0][f"k{k}_posterior_at_answer"][keep][:, k - 1]
    b = bin_of(post)
    full = [i for i in range(len(BINS) - 1) if (b == i).sum() >= MIN_BIN]
    eem = np.array([[r[f"k{k}_eem"][keep][b == i].mean() for i in full] for r in runs])
    ceiling = runs[0][f"k{k}_ceiling"][keep]
    return {
        "x": [float(post[b == i].mean()) for i in full],
        "mean": eem.mean(0).tolist(),
        "lo": eem.min(0).tolist(),
        "hi": eem.max(0).tolist(),
        "ceiling": [float(ceiling[b == i].mean()) for i in full],
    }


NEXT = {c: {k: next_answer(c, k) for k in COUNTS} for c in NEXT_CONDITIONS}


def top(cond: str, k: int, key: str = "mean") -> float:
    """The value of *key* in the top bin of the posterior, where nearly every context with *k* examples sits."""
    return NEXT[cond][k][key][-1]


r"""
# What the anchor follows at the example answers

/// tip |
<!-- lede -->
In ex-2.2.22 the alignment with the anchored direction at the example answers rose with the posterior on `difference`, but at the same posterior it sat higher at earlier answers. That difference goes away once each answer is compared with the evidence of its own example. The alignment at an example answer follows how well that one example fits `difference`, and hardly depends on the examples before it.
///

The [backlog item](/todo/science/anchor-grades-with-the-posterior.md) asks whether the anchor holds the op the model has inferred, which would make it grade with the posterior, rather than the binary label it was trained on. [Ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (E3) took a first look and found a rise with the posterior at the example answers, plus an unexplained dependence on the answer index. This page re-analyzes the same stored measurements, with no new runs.
"""

# %%

rf"""
## Observations

- [Each example on its own (E1)](#each-example-on-its-own-e1): plotted against the posterior given that example alone, the lines for the three answer indices lie on top of each other. On every anchored condition, that posterior accounts for at least {LO_ALONE:.0%} of the variance in α (r²), against at most {HI_SO_FAR:.0%} for the posterior given the examples so far.
- [What the earlier examples add (E2)](#what-the-earlier-examples-add-e2): fitted together with the evidence before that example, the earlier evidence gets a weight of at most {HI_W_BEFORE:.2f}, against at least {LO_W_ALONE:.1f} for the example itself. It shows most on answers that do not fit `difference`.
- [The answer color alone (E3)](#the-answer-color-alone-e3): the answer color, without its operands, predicts at most {HI_COLOR:.0%} of the variance, so α depends on how the answer relates to its operands.
- [Predicting the next answer (E4)](#predicting-the-next-answer-e4): at the query the model does combine the examples. With three examples, the control and `hinge` score {top(HINGE, 3):.2f} where the ideal predictor scores {top(HINGE, 3, "ceiling"):.2f}, more than one example alone could support. With one or two examples it falls well short of the ideal predictor, even on `k-mixed`, which was trained on those counts.

## Scope

This is a re-analysis of the stored by-count pass of ex-2.2.22 on its three-example held-out set, with no preregistration and no gate. It covers the `hinge` condition (six runs: three paired with ex-2.2.21 and three replicates), the whole-line condition (three runs), `k-mixed` (three runs, one left out as below), and the control (five runs). Only the {len(EV["so_far"]):,} held-out `{ex.ANCHORED_OP}` contexts are used, and only their example answers: as ex-2.2.22 found, nearly every query answer has a posterior near 1, so there is too little spread to compare there. E4 also uses the held-out sets with 1 to 5 examples, for the task score at the query.

The `k-mixed` run at model seed {", ".join(str(SEED_OF[lbl]) for lbl in NO_ANCHOR)} has no anchor at the example answers: α stays near the control even where the example fits `{ex.ANCHORED_OP}`. It also scores lowest on the task of all the runs here, as the runs that took the slow path through training did in ex-2.2.21. It is left out of the numbers below.

## The measurements

*α* is the alignment with e₁ (the cosine between a state and the anchored direction), averaged over slices 2 to {ex.N_LAYER}, at the answer of each example. Slice 1 shows the same pattern, more weakly, and the embedding shows none.

Each example answer is compared with three summaries of the evidence, all on the noise model of the corpus:

- the posterior on `{ex.ANCHORED_OP}` given the examples up to and including that one (*the posterior so far*, the one ex-2.2.22 used);
- the posterior given that example alone, from a uniform prior over the seven ops (*the one-example posterior*);
- the posterior given only the examples before it (*the earlier posterior*), which is 1/7 at the first example.

About two thirds of the examples in a `{ex.ANCHORED_OP}` context *fit* it, meaning the answer shown is one `{ex.ANCHORED_OP}` can give on that pair. The rest come from replacement op noise or a random color. A single example can lift the posterior on `{ex.ANCHORED_OP}` from 1/7 to about 0.7 at most, since other ops can give the same answer on some pairs. An example that does not fit has a one-example posterior near zero.
"""

# %%


def binned_by_index(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean of *y* in each bin of *x*, per answer index: `(K, bins)`, NaN where a bin is thin."""
    out = np.full((K, len(BINS) - 1), np.nan)
    for j in range(K):
        b = bin_of(x[:, j])
        for i in range(len(BINS) - 1):
            if (b == i).sum() >= MIN_BIN:
                out[j, i] = y[b == i, j].mean()
    return out


def by_evidence(cond: str, key: str) -> dict[str, list]:
    """Per answer index and bin of the evidence *key*: where the points sit, the seed mean of α, the seed range
    (the run means, pooled over contexts), and the interquartile range over contexts of α averaged over runs (pooled
    over seeds first).
    """
    runs = np.stack([binned_by_index(EV[key], ALPHA[lbl]) for lbl in scored(cond)])
    pooled = np.mean([ALPHA[lbl] for lbl in scored(cond)], axis=0)
    x = binned_by_index(EV[key], EV[key])
    q25, q75 = np.full_like(x, np.nan), np.full_like(x, np.nan)
    for j in range(K):
        b = bin_of(EV[key][:, j])
        for i in np.flatnonzero(~np.isnan(x[j])):
            q25[j, i], q75[j, i] = np.percentile(pooled[b == i, j], [25, 75])
    return {
        "x": x.tolist(),
        "mean": np.nanmean(runs, axis=0).tolist(),
        "lo": np.nanmin(runs, axis=0).tolist(),
        "hi": np.nanmax(runs, axis=0).tolist(),
        "q25": q25.tolist(),
        "q75": q75.tolist(),
    }


def ink(j: int) -> str:
    return (light_dark("#1f6fb2", "#7ab8f5"), light_dark("#c0392b", "#ff8a76"), light_dark("#2e8b57", "#7fd8a4"))[j]


def e1_figure() -> str:
    data = {
        "panels": {key: {c: by_evidence(c, key) for c in (HINGE, "control")} for key in ("so_far", "alone")},
        "inks": [ink(j) for j in range(K)],
    }
    alt = f"""
        Two panels of seed-mean α at the example answers of `{ex.ANCHORED_OP}` contexts, on the `hinge` condition,
        one line per answer index (the first, second, and third example), each with a faint band for the seed range
        and whiskers for the middle half of the contexts, and the control drawn as thin grey lines near zero. Left: α
        against the posterior so far; the three lines are offset, with the first answer highest at the same
        posterior. Right: α against the one-example posterior, which reaches about 0.7 at most; the three lines lie
        on top of each other, low near zero and near 0.9 at the top. The seed bands are narrow, and the context
        whiskers are wide, spanning a few tenths at the middle of each line.
    """
    return e1_draw(data, alt)


@memo
def e1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="alpha-by-evidence",
        alt_text=alt_text,
        caption=f"""
            **α against two summaries of the evidence, on `hinge`.** At each example answer of the held-out
            `{ex.ANCHORED_OP}` contexts, α averaged over slices 2 to {ex.N_LAYER}, against **left:** the posterior so
            far, and **right:** the one-example posterior, in bins of 0.2, each point at the mean evidence of its bin.
            One line per answer index, the seed mean of six runs; a bin with fewer than {MIN_BIN} points in a run is
            left out of that run. The band is the seed range of the run means (pooled over contexts first); the
            whiskers span the middle half of the contexts, on α averaged over the runs (pooled over seeds first).
            Thin grey lines: the control.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained", sharey=True)
        xlabels = {"so_far": "posterior so far", "alone": "one-example posterior"}
        nudge = (-0.012, 0.0, 0.012)
        # The whiskers of the three indices are nudged apart so they don't overlap.
        for ax, (key, curves) in zip(axes, data["panels"].items(), strict=True):
            ctrl = curves["control"]
            for j in range(K):
                ax.plot(ctrl["x"][j], ctrl["mean"][j], "-", lw=0.6, color=light_dark("#999", "#777"))
            h = {k: np.array(v) for k, v in curves[HINGE].items()}
            for j in range(K):
                ax.fill_between(h["x"][j], h["lo"][j], h["hi"][j], color=data["inks"][j], alpha=0.15, lw=0, zorder=1)
            for j in range(K):
                x = h["x"][j] + nudge[j]
                ax.vlines(x, h["q25"][j], h["q75"][j], color=data["inks"][j], lw=0.8, alpha=0.6, zorder=2)
                ax.plot(x, h["mean"][j], "-o", ms=3, lw=1.1, color=data["inks"][j], label=f"example {j + 1}", zorder=3)
            ax.axhline(0, color=light_dark("#333", "#ddd"), lw=0.5)
            ax.set_xlabel(xlabels[key], fontsize=9)
            ax.set_xlim(0, 1 if key == "so_far" else 0.75)
        axes[0].set_ylabel("α", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=8)
        return fig

    return _plot()


rf"""
## Each example on its own (E1)

If α followed the posterior so far, answers with the same posterior would have the same α whatever their position. The left panel below repeats the comparison of ex-2.2.22 on `hinge`, and the right panel plots the same answers against the one-example posterior.

{e1_figure()}

On the left the three lines are offset, as in ex-2.2.22: at a posterior between 0.6 and 0.8, α at the first answer is about as high as it gets, and at the third it is about a third lower. On the right they lie on top of each other. The seed bands are narrow on both sides. The whiskers are not: at the same posterior so far, α spreads over most of its range from one context to the next, and against the one-example posterior that spread shrinks to a few tenths or less. An answer whose example fits `{ex.ANCHORED_OP}` has α near {mean_of(HINGE, "fit"):.2f} at every index, and one that does not fit stays low.

That explains the offset. At the first answer, a middling posterior means one example that fits `{ex.ANCHORED_OP}` and also fits another op. At the third, it usually means a mix of examples that fit and examples that do not, and α then depends on which kind the latest one is.

The table gives the same comparison as variance explained, for each anchored condition. The share of the examples so far that fit, a summary that weights every example equally, does a little better than the posterior so far and much worse than the one-example posterior.
"""

# %%


def fmt(cond: str, key: str, spec: str) -> str:
    m, lo, hi = cond_stat(cond, key)
    return f"{m:{spec}} ({lo:{spec}}–{hi:{spec}})"


def short(cond: str) -> str:
    return {HINGE: "hinge", WHOLE: "whole-line"}.get(cond, cond)


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def r2_table() -> str:
    rows = [
        [f"<code>{short(c)}</code>", str(len(scored(c))), *(fmt(c, k, ".2f") for k in ("so_far", "share", "alone"))]
        for c in ANCHORED
    ]
    return table_html(
        ["condition", "runs", "posterior so far", "share that fit", "one-example posterior"],
        rows,
        caption="""
            **How much of the variance in α each summary accounts for.** The squared correlation r² between α at
            the example answers (averaged over slices 2 to 4) and each summary of the evidence, over every example
            answer of the held-out `difference` contexts. Seed mean, with the seed range in brackets.
        """,
    )


rf"""
{r2_table()}

## What the earlier examples add (E2)

The one-example posterior leaves some variance unexplained, so the earlier examples might still add something. A least-squares fit of α on the one-example posterior and the earlier posterior together gives each a weight; both are on the same scale, from 0 to 1.
"""

# %%


def weight_table() -> str:
    rows = [
        [
            f"<code>{short(c)}</code>",
            fmt(c, "w_alone", ".2f"),
            fmt(c, "w_before", ".2f"),
            fmt(c, "misfit_low", ".2f"),
            fmt(c, "misfit_high", ".2f"),
        ]
        for c in ANCHORED
    ]
    return table_html(
        [
            "condition",
            "weight, this example",
            "weight, earlier examples",
            "α, misfit after doubt",
            "α, misfit after support",
        ],
        rows,
        caption="""
            **The weight of each example and of the ones before it.** The first two columns: the least-squares
            weights of α on the one-example posterior and on the earlier posterior. The last two: mean α at answers
            whose example does not fit `difference`, when the earlier posterior is below 0.3 and when it is above
            0.8. Seed mean, with the seed range in brackets.
        """,
    )


rf"""
{weight_table()}

The earlier examples get a weight of a few hundredths, against more than one for the example itself. They show most on the answers that do not fit: there α is a little higher when the earlier examples favor `{ex.ANCHORED_OP}` than when they do not ({mean_of(HINGE, "misfit_high"):.2f} against {mean_of(HINGE, "misfit_low"):.2f} on `hinge`). On answers that fit, the earlier examples make almost no difference.

## The answer color alone (E3)

The one-example posterior depends on the operands as well as the answer. If α could be predicted from the answer color alone, the anchor would be closer to a property of the answer token. Averaged by answer color on half of the contexts and tested on the other half, the answer color accounts for {fmt(HINGE, "color", ".2f")} of the variance on `hinge` (held-out R²), and similar on the other conditions. So the model places the state at an example answer on e₁ according to how that answer relates to its operands.

"""

# %%


def e4_figure() -> str:
    data = {"next": NEXT, "names": [short(c) for c in NEXT_CONDITIONS]}
    alt = f"""
        Three panels side by side, for the control, `hinge`, and `k-mixed`. Each plots expected exact match at the
        query against the posterior on `{ex.ANCHORED_OP}` given the examples, one line per number of examples from 1
        to 5 in shades running light to dark, with a faint band for the seed range and a dashed diagonal for the
        ideal predictor. On the control and `hinge`, the line for three examples nearly reaches the diagonal at the
        top ({top(HINGE, 3):.2f} against {top(HINGE, 3, "ceiling"):.2f} on `hinge`), and the other counts sit lower.
        On `k-mixed` every line sits well below the diagonal ({top("k-mixed", 3):.2f} with three examples). With
        one example every condition reaches about {top(HINGE, 1):.2f} at the top bin, against
        {top(HINGE, 1, "ceiling"):.2f} for the ideal predictor. On `hinge` the band for four and five examples is
        wide, from one run that drops.
    """
    return e4_draw(data, alt)


@memo
def e4_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="next-answer",
        alt_text=alt_text,
        caption=f"""
            **How well the model predicts the next answer.** On the held-out `{ex.ANCHORED_OP}` contexts with 1 to 5
            examples, expected exact match at the query against the posterior on `{ex.ANCHORED_OP}` given the
            examples, in bins of 0.2, each point at the mean posterior of its bin. One line per number of examples;
            the seed mean, with the seed range as a band. Dashed: the ideal predictor, which gets the posterior
            right. The control and `hinge` were trained on three examples only.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), layout="constrained", sharex=True, sharey=True)
        cmap = plt.get_cmap(light_dark("viridis_r", "viridis"))
        shades = {k: cmap(0.15 + 0.7 * i / (len(COUNTS) - 1)) for i, k in enumerate(COUNTS)}
        for ax, name, by_k in zip(axes, data["names"], data["next"].values(), strict=True):
            ax.plot([0, 1], [0, 1], "--", lw=0.7, color=light_dark("#777", "#999"), zorder=0)
            for k, d in by_k.items():
                d = {key: np.array(v) for key, v in d.items()}
                ax.fill_between(d["x"], d["lo"], d["hi"], color=shades[int(k)], alpha=0.18, lw=0, zorder=1)
                ax.plot(d["x"], d["mean"], "-o", ms=2.5, lw=1.1, color=shades[int(k)], label=f"{k}", zorder=2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("posterior given the examples", fontsize=9)
            ax.set_title(name, fontsize=9)
        axes[0].set_ylabel("expected exact match", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="examples",
            loc="outside upper center",
            ncols=len(labels),
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )
        return fig

    return _plot()


rf"""
## Predicting the next answer (E4)

The anchor follows each example on its own, but the next-token loss asks the model to predict each example answer at its `=`, which means combining the examples before it. How well does it do that? The by-count pass scored the query on held-out sets with 1 to 5 examples. The query after one example asks for the answer a second example would show (without the noise an example can carry), and the query after two asks for the third. So these scores stand in for how well the model predicts the answers of the later examples. The control and `hinge` were trained on three examples only, so for them the other counts are outside training; `k-mixed` was trained on 1 to 5.

{e4_figure()}

The ideal predictor gets an expected exact match about equal to its posterior, so it follows the diagonal. With three examples, the count they were trained on, the control and `hinge` come close to it: {top(HINGE, 3):.2f} against {top(HINGE, 3, "ceiling"):.2f} in the top bin on `hinge`. With one example every condition falls well short, at about {top(HINGE, 1):.2f} against {top(HINGE, 1, "ceiling"):.2f}, and `k-mixed`, which was trained on that count, does only a little better ({top("k-mixed", 1):.2f}). With two examples the gap is wider still: {top(HINGE, 2):.2f} against {top(HINGE, 2, "ceiling"):.2f} on `hinge`, and {top("k-mixed", 2):.2f} on `k-mixed`. `k-mixed` stays below the diagonal at every count, reaching {top("k-mixed", 3):.2f} with three examples. A single example lifts the posterior to about 0.7 at most, and with three examples the control and `hinge` score well above that, so at the query the model does combine the examples. The anchor at the example answers does not follow that combining.

## Discussion

At the example answers the anchor marks examples that look like `{ex.ANCHORED_OP}` more than contexts that do. The rise with the posterior that ex-2.2.22 saw comes mostly from this: the more examples fit, the higher the posterior, and each fitting example has a high α of its own. So this measurement does not yet show the anchor holding an inferred op. The sites where that could show are the `=` tokens, where the model predicts an answer it has not yet seen. The next-token loss covers every one of them, so the model is asked to combine the examples so far at each example `=` as well as at the query `=`. The answer, one token later, is where the example just shown is in view, which is what α follows. Ex-2.2.21 found the anchor weak at the query `=`, and α at the example `=` tokens was not stored.

This fits how the label works. The pull is pooled over the positions of a whole `{ex.ANCHORED_OP}` context, so the model can meet it at positions of its choosing, and the answers of the examples that fit are where a single line shows the op most plainly. The answers that do not fit stay low, which a pooled pull allows.

It may also bear on the spill that ex-2.2.23 found. A per-example judgement of whether an answer fits `{ex.ANCHORED_OP}` given its operands would fire on examples of other ops whose answers happen to fit, and removing e₁ there would touch those ops. This re-analysis covers only `{ex.ANCHORED_OP}` contexts, so it does not test that.
"""
