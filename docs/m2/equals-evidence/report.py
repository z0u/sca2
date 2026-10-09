# title: What the anchor follows at the = tokens

# A re-analysis of the ex-2.2.23 checkpoints and the trials of the τ × λ_a sweep, with no new training. It reads the
# alignment with e₁ at every `=` of the held-out contexts, rebuilds the evidence before each `=` from the tokens with
# the posterior module of ex-2.2.16, and compares the two.
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex = _load_sibling("experiment.py", "equals_evidence_experiment")
P = ex.ex2216._POSTERIOR_MODULE
K = ex.K
LATE = slice(2, None)
# The slices α is averaged over: from the output of the second block on, as in the example-evidence report.
RECIPE_TAU = 0.1
SOFT_TAU = 0.3
# A sweep trial with τ at or above this has a soft pool; the sweep found the anchor on the `=` tokens from there.
RISE_LEVEL = 0.1
# A trial whose α at the query `=` of `difference` contexts is at least this much above its α at the first `=`
# has an anchor that follows the context, so it is one where grading can be tested.


# --- The stored arrays ----------------------------------------------------------------------------------------


def fetch_all() -> tuple[dict, dict[str, np.ndarray], Any]:
    """The results summary, α at the `=` positions of every run (`(slices, contexts, K + 1)`), and the held-out
    set of ex-2.2.21.
    """
    store = project_store()

    def published(refs: list[str]) -> dict[str, Any]:
        got = store.get_refs(refs)
        missing = [r for r, a in got.items() if a is None]
        assert not missing, f"not published: {missing}"
        return got

    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        holdout_ref = ex.ex2221.HOLDOUT_REF.format(key=ex.ex2223.MAIN_KEY)
        have = published([ex.RESULTS_REF, holdout_ref])
        summary_path, holdout_path = store.get_many(
            [(have[ex.RESULTS_REF], into / "results.json"), (have[holdout_ref], into / "holdout.npz")]
        )
        summary = json.loads(summary_path.read_text())
        labels = [r["label"] for r in summary["runs"]]
        refs = [ex.ARRAYS_REF.format(label=lbl) for lbl in labels]
        arts = published(refs)
        paths = store.get_many([(arts[r], into / f"{lbl}.npz") for r, lbl in zip(refs, labels, strict=True)])
        arrays = {}
        for lbl, p in zip(labels, paths, strict=True):
            with np.load(p) as z:
                arrays[lbl] = z["alpha_eq"]  # the full `alpha` array stays compressed
        holdout = ex.ex2216.load_holdout(holdout_path, K)
    return summary, arrays, holdout


SUMMARY, ARRAYS, HOLDOUT = fetch_all()
RUNS = {r["label"]: r for r in SUMMARY["runs"]}
OPS = tuple(SUMMARY["design"]["ops"])
D = OPS.index(ex.ANCHORED_OP)
IS_D = HOLDOUT.op_ids == D

# --- The evidence before each `=` -----------------------------------------------------------------------------


def evidence(tokens: np.ndarray) -> dict[str, np.ndarray]:
    """Per held-out context and `=` (the K example `=` then the query `=`), `(n, K + 1)`: the posterior on
    `difference` given the examples before that `=` (1/7 at the first), and the number of those examples whose
    answer is one `difference` can give on its pair. Per example, `(n, K)`: the posterior given that example alone,
    and whether it fits.
    """
    from sca.config import TokenizerConfig
    from sca.data.incontext import vocabulary
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE

    tok = WordTokenizer(TokenizerConfig(vocabulary=sorted(vocabulary())))
    tok2color = np.full(tok.vocab_size, -1)
    tok2color[[tok.stoi[n] for n in PALETTE]] = np.arange(len(PALETTE))
    table = P.build_table(ex.ex2218.table_of(OPS))
    answers = np.array([ex.UNIT * i + 4 for i in range(K)])
    y, a, b = tok2color[tokens[:, answers]], tok2color[tokens[:, answers - 4]], tok2color[tokens[:, answers - 2]]
    p = table.lookup(a * len(PALETTE) + b, y)  # (ops, n, K)
    others = (p.sum(0, keepdims=True) - p) / (len(OPS) - 1)
    log_like = np.log((1 - ex.RHO - ex.CUBE_RATE) * p + ex.RHO * others + ex.CUBE_RATE / len(PALETTE)).transpose(
        1, 2, 0
    )

    def normed(log_post: np.ndarray) -> np.ndarray:
        q = np.exp(log_post - log_post.max(-1, keepdims=True))
        return (q / q.sum(-1, keepdims=True))[..., D]

    n = len(tokens)
    fits = p[D] > 0
    return {
        "before": np.concatenate([np.full((n, 1), 1 / len(OPS)), normed(log_like.cumsum(1))], axis=1),
        "count": np.concatenate([np.zeros((n, 1), int), fits.cumsum(1)], axis=1),
        "alone": normed(log_like),
        "fits": fits,
    }


EV = evidence(HOLDOUT.tokens)


def alpha(lbl: str) -> np.ndarray:
    """α at each `=` of every held-out context, averaged over `LATE` slices: `(n, K + 1)`."""
    return ARRAYS[lbl][LATE].astype(np.float64).mean(axis=0)


def r2(x: np.ndarray, y: np.ndarray) -> float:
    """The squared Pearson correlation, r²."""
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1] ** 2)


BINS = np.array([0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.75])
# Bins of the one-example posterior: the first holds the examples that do not fit (near zero), and the rest split
# the fitting examples by how many other ops fit the same pair, from about 0.15 to about 0.7.
MIN_BIN = 20
ALONE_BIN = np.clip(np.digitize(EV["alone"][IS_D, 0], BINS[1:-1]), 0, len(BINS) - 2)
# The bin of the first example of each `difference` context.
FULL_BINS = [i for i in range(len(BINS) - 1) if (ALONE_BIN == i).sum() >= MIN_BIN]


def tally(x: np.ndarray) -> np.ndarray:
    """The sum of the one-example posteriors of the examples before each `=`: `(n, K + 1)`."""
    return np.concatenate([np.zeros((len(x), 1)), x.cumsum(1)], axis=1)


def nonfit_change(y: np.ndarray, x: np.ndarray) -> float:
    """The mean change in *x* when one more example that does not fit is added at a fixed count of fitting
    examples: from the `=` at index j with c fitting examples to the `=` at j + 1 with the same c, over every such
    pair with at least `MIN_BIN` contexts on each side.
    """
    count = EV["count"][IS_D]
    changes = []
    for j in range(1, K):
        for c in range(j + 1):
            m1, m2 = count[:, j] == c, count[:, j + 1] == c
            if m1.sum() >= MIN_BIN and m2.sum() >= MIN_BIN:
                changes.append(x[m2, j + 1].mean() - x[m1, j].mean())
    return float(np.mean(changes))


def per_run(lbl: str) -> dict[str, float]:
    y = alpha(lbl)
    yd = y[IS_D]
    j = slice(1, None)
    by_bin = [yd[ALONE_BIN == i, 1].mean() for i in FULL_BINS]  # α at the second `=`, by bin of the first example
    return {
        "first_d": float(yd[:, 0].mean()),
        "first_other": float(y[~IS_D, 0].mean()),
        "query_d": float(yd[:, K].mean()),
        "query_other": float(y[~IS_D, K].mean()),
        "rise": float(yd[:, K].mean() - yd[:, 0].mean()),
        "r2_post": r2(EV["before"][IS_D][:, j], yd[:, j]),
        "r2_count": r2(EV["count"][IS_D][:, j], yd[:, j]),
        "r2_tally": r2(tally(EV["alone"][IS_D])[:, j], yd[:, j]),
        "nonfit": nonfit_change(yd, yd),
        "step": float(by_bin[1] - by_bin[0]),
        "grade": float(by_bin[-1] - by_bin[1]),
    }


STATS = {lbl: per_run(lbl) for lbl in RUNS}


def tau_of(lbl: str) -> float:
    return RUNS[lbl].get("tau", RECIPE_TAU)


RECIPE = [lbl for lbl, r in RUNS.items() if r["source"] == "ex-2.2.23" and r["condition"] == "anchor"]
RECIPE_LONG = [lbl for lbl in RECIPE if RUNS[lbl]["epochs"] == ex.ex2223.LONG]
CONTROL = [lbl for lbl, r in RUNS.items() if r["condition"] == "control"]
SWEEP = [lbl for lbl, r in RUNS.items() if r["source"] == "tau-lambda-sweep"]
SOFT = [lbl for lbl in SWEEP if tau_of(lbl) >= SOFT_TAU]
RISING = [lbl for lbl in SOFT if STATS[lbl]["rise"] >= RISE_LEVEL]
N_D = int(IS_D.sum())


def mean_range(values) -> tuple[float, float, float]:
    v = np.asarray(list(values), float)
    return float(v.mean()), float(v.min()), float(v.max())


def fmt(values, spec: str = ".2f") -> str:
    half = 0.5 * 10 ** -int(spec[1])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(v) < half else v for v in mean_range(values))
    return f"{m:{spec}} ({lo:{spec}} to {hi:{spec}})"


def stat(labels: list[str], key: str) -> list[float]:
    return [STATS[lbl][key] for lbl in labels]


HI_QUERY_RECIPE = max(stat(RECIPE_LONG, "query_d"))
HI_QUERY_CONTROL = max(stat(CONTROL, "query_d"))
HI_QUERY_SOFT = max(stat(SOFT, "query_d"))
EQ_LATCH = [lbl for lbl in SWEEP if RUNS[lbl]["syntax"]["="] >= 0.7]
# The sweep trials whose `=` embedding came onto e₁ (its E2): the sharpest pools.
HI_R2_POST = max(stat(RISING, "r2_post"))
POST_NONFIT = nonfit_change(alpha(RISING[0])[IS_D], EV["before"][IS_D])
# How much the posterior falls per non-fitting example at a fixed count: a property of the stimulus, the same on
# every run.
N_MID = int(sum((ALONE_BIN == i).sum() for i in FULL_BINS[1:-1]))
N_FIT = int(EV["fits"][IS_D, 0].sum())


def words(n: int) -> str:
    """A small count as a word, for prose."""
    return ("zero one two three four five six seven eight nine ten eleven twelve".split())[n] if n <= 12 else str(n)


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


rf"""
# What the anchor follows at the `=` tokens

/// tip |
<!-- lede -->
Every `=` is a place where the model has to infer the op from the examples before it, so it is where an anchor that held the inferred op would follow the posterior. On the recipe, the anchor is nearly absent at every `=`. With a softer pool it reaches them, and there it grades with the examples that fit `difference`, each by how well it fits. Whether it also discounts the examples that do not fit, as the posterior does, is more than one seed per trial can say. Part of what the softer pool puts on the `=` is the same on every op.
///

The [backlog item](/todo/science/anchor-grades-with-the-posterior.md) asks whether the anchor holds the op the model has inferred, which would make the alignment grade with the posterior on `{ex.ANCHORED_OP}`. [Example-evidence](/docs/m2/example-evidence/report.py) found that at an example answer the anchor follows that one example, where the line itself shows the op, so the test has to move to the `=` tokens, where the answer is still to be predicted. This page reads the alignment at every `=` on the stored checkpoints of [ex-2.2.23](/docs/m2/ex-2.2.23/report.py) and of the [τ × λ_a sweep](/docs/m2/tau-lambda-sweep/report.py), with no new training.
"""

# %%

rf"""
## Observations

- [How much anchor the `=` tokens hold (E1)](#how-much-anchor-the-tokens-hold-e1): on the recipe, α at the query `=` of `{ex.ANCHORED_OP}` contexts is at most {HI_QUERY_RECIPE:.2f} (the control reaches {HI_QUERY_CONTROL:.2f}), against about 0.9 at a fitting example answer. On the sweep it rises with τ, to as much as {HI_QUERY_SOFT:.2f} on the soft pools, and part of what it rises to is the same on every op.
- [What the rise follows (E2)](#what-the-rise-follows-e2): on the trials where α rises along the context, the posterior before the `=` accounts for up to {HI_R2_POST:.0%} of its variance, and a sum of the one-example posteriors does about as well. The two come apart on the examples that do not fit: one lowers the posterior by {-POST_NONFIT:.2f} on average at a fixed count of fitting examples, and changes α by {fmt(stat(RISING, "nonfit"))}, which is within what a posterior-following α would show at this slope. Among the examples that fit, α grades with how specific the example is.
- [Where it sits by depth (E3)](#where-it-sits-by-depth-e3): the part shared by every op starts at the `=` embedding, is largest in the first slices, and fades with depth; the part that follows the examples appears after block 2 and stays through block 4.

## Scope

This is a re-analysis of stored checkpoints, with no preregistration and no gate. It covers every run of ex-2.2.23 (the recipe at τ = {RECIPE_TAU}: {len(RECIPE)} anchored runs and {len(CONTROL)} controls, at 200 and 400 epochs) and every trial of the τ × λ_a sweep ({len(SWEEP)} trials at 400 epochs, one seed each, τ from 0.01 to 1 on the `no-emb` and whole-line arms). The sweep trials are the only stored runs with an anchor at the `=` tokens, so E2 and E3 rest on the {words(len(SOFT))} trials with τ at or above {SOFT_TAU}, and within those on the {words(len(RISING))} whose α rises along the context by at least {RISE_LEVEL}. With one seed per trial, a difference between trials cannot be told from the seed.

The held-out set has {len(HOLDOUT.tokens):,} three-example contexts, {N_D:,} of them under `{ex.ANCHORED_OP}`. The fits of E2 use the `{ex.ANCHORED_OP}` contexts; the other ops' contexts give the level of α where the examples say nothing about `{ex.ANCHORED_OP}`.

## The measurements

*α* is the alignment with e₁ (the cosine between a state and the anchored direction), averaged over slices 2 to {ex.N_SLICES - 1}, at the `=` of each example and at the query `=`. E3 shows it by slice.

Each `=` is compared with the evidence before it, on the noise model of the corpus: the posterior on `{ex.ANCHORED_OP}` given the examples before that `=` (1/7 at the first `=`, where nothing has been shown), and the number of those examples that *fit*, meaning the answer shown is one `{ex.ANCHORED_OP}` can give on that pair. The two agree closely on this stimulus, since the posterior climbs with each fitting example. They come apart in two places. An example that does not fit pulls the posterior down and leaves the count alone. And a count ignores how specific a fitting example is: one that also fits several other ops lifts the posterior from 1/7 to about 0.15, and one that fits `{ex.ANCHORED_OP}` alone lifts it to about 0.7 (the *one-example posterior*, as in example-evidence). A third summary, the sum of the one-example posteriors of the examples before the `=`, is graded like the posterior and ignores non-fitting examples like the count.
"""

# %%


def e1_figure() -> str:
    def marks(labels: list[str], jitter: float) -> dict[str, list]:
        rng = np.random.default_rng(0)
        return {
            "tau": [tau_of(lbl) * float(np.exp(rng.uniform(-jitter, jitter))) for lbl in labels],
            "first": stat(labels, "first_d"),
            "query": stat(labels, "query_d"),
        }

    data = {
        "recipe": marks(RECIPE_LONG, 0.08),
        "control": marks(CONTROL, 0.08),
        "no-emb": marks([lbl for lbl in SWEEP if RUNS[lbl]["condition"] == "no-emb"], 0.0),
        "whole": marks([lbl for lbl in SWEEP if RUNS[lbl]["condition"] == "whole"], 0.0),
    }
    alt = f"""
        α at the `=` tokens of `{ex.ANCHORED_OP}` contexts against the pool temperature τ on a log axis. Below τ of
        about 0.2 nearly every mark sits near zero, the recipe runs and the controls among them, except four trials
        at the sharpest pools whose marks reach 0.6 to 0.95. Above it the marks climb to between 0.1 and 0.7, and on
        each trial the filled mark for the query `=` sits above the open mark for the first `=`, by up to a third.
    """
    return e1_draw(data, alt)


@memo
def e1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="alpha-at-equals-by-tau",
        alt_text=alt_text,
        caption=f"""
            **How much anchor the `=` tokens hold, by pool temperature.** On the held-out `{ex.ANCHORED_OP}`
            contexts, α at the first `=` (open marks, where no example has been shown) and at the query `=`
            (filled marks, after three), joined per run. Circles: the `no-emb` arm of the sweep; squares: its
            whole-line arm; diamonds: the 400-epoch anchored runs of ex-2.2.23, at the recipe τ, spread sideways
            so they can be told apart; grey: the controls of ex-2.2.23, drawn at the same τ.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        style = {
            "control": ("D", light_dark("#999", "#777"), "control"),
            "recipe": ("D", light_dark("#c0392b", "#ff8a76"), "recipe (ex-2.2.23)"),
            "no-emb": ("o", light_dark("#1f6fb2", "#7ab8f5"), "sweep, no-emb"),
            "whole": ("s", light_dark("#2e8b57", "#7fd8a4"), "sweep, whole-line"),
        }
        for key, (marker, color, label) in style.items():
            d = data[key]
            for t, lo, hi in zip(d["tau"], d["first"], d["query"], strict=True):
                ax.plot([t, t], [lo, hi], "-", lw=0.7, color=color, alpha=0.6, zorder=1)
            ax.plot(d["tau"], d["first"], marker, ms=4, mfc="none", mec=color, lw=0, zorder=2)
            ax.plot(d["tau"], d["query"], marker, ms=4, color=color, lw=0, label=label, zorder=3)
        ax.axhline(0, color=light_dark("#333", "#ddd"), lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("τ", fontsize=9)
        ax.set_ylabel("α", fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=8)
        return fig

    return _plot()


rf"""
## How much anchor the `=` tokens hold (E1)

Before asking what the alignment at the `=` follows, the question is whether there is any. The figure shows α at the first `=` and at the query `=` of `{ex.ANCHORED_OP}` contexts on every run, by the temperature of the pool.

{e1_figure()}

On the recipe there is almost none. The 400-epoch anchored runs of ex-2.2.23 reach {fmt(stat(RECIPE_LONG, "query_d"))} at the query `=`, and the controls {fmt(stat(CONTROL, "query_d"))}, where a fitting example answer on the same runs has α near 0.9 ([embedding-lean](/docs/m2/embedding-lean/report.py), E1). The sweep trials below τ ≈ 0.2 sit with them, apart from the {words(len(EQ_LATCH))} at the sharpest pools whose `=` embedding came onto e₁ (the sweep, E2); on those the `=` state carries the latched embedding rather than anything the context put there, and they are left out below. So on every run trained with the recipe pool, the `=` tokens hold too little anchor to follow anything.

From τ ≈ 0.3 the marks climb, as the sweep found, and they do so in two parts. The open marks rise too: at the first `=`, where no example has been shown, α reaches {fmt(stat(SOFT, "first_d"))} on the soft trials, and on the other ops' contexts it is the same, {fmt(stat(SOFT, "first_other"))}. That part is a property of the `=` state that every context shares, which a soft pool rewards since the pull is met at a position every `{ex.ANCHORED_OP}` context has. The second part is the rise from the first `=` to the query `=`, which happens on `{ex.ANCHORED_OP}` contexts ({fmt(stat(SOFT, "rise"))}) and not on the others ({fmt([STATS[lbl]["query_other"] - STATS[lbl]["first_other"] for lbl in SOFT])}). That part follows the examples, and it is what E2 looks at, on the {words(len(RISING))} trials where it is at least {RISE_LEVEL}.
"""

# %%


def by_count() -> dict[str, list]:
    """α at each `=` with evidence before it, against the number of fitting examples before it, on the
    `difference` contexts of the rising trials: the trial mean and range, per `=` index.
    """
    count = EV["count"][IS_D]
    out: dict[str, list] = {"x": [], "mean": [], "lo": [], "hi": []}
    for j in range(1, K + 1):
        xs = [c for c in range(j + 1) if (count[:, j] == c).sum() >= MIN_BIN]
        per = np.array([[alpha(lbl)[IS_D][count[:, j] == c, j].mean() for c in xs] for lbl in RISING])
        out["x"].append(xs)
        out["mean"].append(per.mean(0).tolist())
        out["lo"].append(per.min(0).tolist())
        out["hi"].append(per.max(0).tolist())
    return out


def by_alone() -> dict[str, list]:
    """α at the second `=` against the one-example posterior of the first example, binned, per rising trial."""
    x, b = EV["alone"][IS_D, 0], ALONE_BIN
    return {
        "x": [float(x[b == i].mean()) for i in FULL_BINS],
        "trials": [[float(alpha(lbl)[IS_D][b == i, 1].mean()) for i in FULL_BINS] for lbl in RISING],
    }


def e2_figure() -> str:
    data = {"count": by_count(), "alone": by_alone()}
    alt = """
        Left: α against the number of fitting examples before the `=`, one line per `=` index; the lines climb
        together and lie close to one another at the same count, so a later `=` with the same count sits about where
        an earlier one does.
        Right: α at the second `=` against the posterior given the first example alone, one thin line per trial;
        the mean rises steadily from the non-fitting bin near zero to the most specific fitting bin at 0.7.
    """
    return e2_draw(data, alt)


@memo
def e2_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="alpha-by-evidence-at-equals",
        alt_text=alt_text,
        caption=f"""
            **What α at the `=` follows, on the {words(len(RISING))} rising trials.** **Left:** α against the number of
            examples before the `=` that fit `{ex.ANCHORED_OP}`, one line per `=` index (the second and third
            example and the query), the mean over trials with the range as a band. **Right:** α at the second `=`
            against the posterior given the first example alone, in bins; the first bin holds the examples that do
            not fit, and the rest split the fitting examples by how specific they are. One thin line per trial,
            the mean in bold. A bin with fewer than {MIN_BIN} contexts is left out.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        inks = (light_dark("#1f6fb2", "#7ab8f5"), light_dark("#c0392b", "#ff8a76"), light_dark("#2e8b57", "#7fd8a4"))
        names = ("second =", "third =", "query =")
        markers = ("o", "s", "D")
        c = data["count"]
        for j in range(K):
            x = np.array(c["x"][j]) + (j - 1) * 0.04
            axes[0].fill_between(x, c["lo"][j], c["hi"][j], color=inks[j], alpha=0.15, lw=0, zorder=1)
            axes[0].plot(x, c["mean"][j], "-" + markers[j], ms=3.5, lw=1.1, color=inks[j], label=names[j], zorder=3)
        axes[0].set_xlabel("fitting examples before the =", fontsize=9)
        axes[0].set_xticks(range(K + 1))
        axes[0].set_ylabel("α", fontsize=9)
        a = data["alone"]
        for t in a["trials"]:
            axes[1].plot(a["x"], t, "-", lw=0.6, color=light_dark("#1f6fb2", "#7ab8f5"), alpha=0.5, zorder=2)
        axes[1].plot(a["x"], np.mean(a["trials"], axis=0), "-o", ms=3.5, lw=1.4, color=inks[0], zorder=3)
        axes[1].set_xlabel("posterior given the first example", fontsize=9)
        axes[1].set_xlim(0, 0.75)
        for ax in axes:
            ax.axhline(0, color=light_dark("#333", "#ddd"), lw=0.5)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=8)
        return fig

    return _plot()


rf"""
## What the rise follows (E2)

On the rising trials, does α at a `=` follow the posterior on `{ex.ANCHORED_OP}` given the examples before it? The three summaries agree on most contexts, so the comparison needs the places where they differ. The left panel plots α against the count of fitting examples, one line per `=` index, so a later `=` with the same count is one that has seen more non-fitting examples, which lower the posterior by {-POST_NONFIT:.2f} on average. The right panel takes the second `=`, after a single example, against the one-example posterior of that example, from near zero (it does not fit) to 0.7 (it fits `{ex.ANCHORED_OP}` alone).

{e2_figure()}

Against the count, the three `=` indices lie close together: at the same number of fitting examples, the query `=` has about the same α as the second `=`, whatever came between. Adding a non-fitting example at a fixed count changes α by {fmt(stat(RISING, "nonfit"))}, where the posterior falls by {-POST_NONFIT:.2f}. That looks like α ignoring the examples that do not fit, but the two numbers are on different scales. At the slope of α on the one-example posterior in the right panel, an α that followed the posterior would drop by only a few hundredths per non-fitting example, and on the two trials with the most anchor the observed drops are about that size. So on this stimulus the sum and the posterior are too close for one seed per trial to tell apart. Against the one-example posterior, α rises steadily across the fitting bins: an example that fits `{ex.ANCHORED_OP}` alone lifts α by {fmt(stat(RISING, "grade"))} more than one that fits several ops, which is more than the step from a non-fitting example to the loosest fit ({fmt(stat(RISING, "step"))}). So among the examples that fit, α does grade with how well. The middle bins are thin, with {N_MID} of the {N_FIT:,} fitting first examples between the loosest and the most specific, so that slope rests on few contexts, though every trial shows it.

The table gives the comparison over every `=` with evidence before it, for the soft trials. The posterior, the count, and the sum of one-example posteriors account for about the same share of the variance on each trial, as they must when they agree on most contexts. The sum leads on four trials by a few hundredths to a tenth. On the two trials with the most anchor at the `=`, the posterior leads by about a tenth.
"""

# %%


def trial_table() -> str:
    rows = []
    for lbl in sorted(SOFT, key=tau_of):
        s = STATS[lbl]
        name = f"<code>{RUNS[lbl]['condition']}</code>"
        rows.append(
            [
                name,
                f"{tau_of(lbl):.2f}",
                f"{s['first_d']:.2f}",
                f"{s['rise']:.2f}",
                f"{s['r2_post']:.2f}",
                f"{s['r2_count']:.2f}",
                f"{s['r2_tally']:.2f}",
                f"{s['nonfit']:+.2f}",
                f"{s['step']:.2f}",
                f"{s['grade']:.2f}",
            ]
        )
    return table_html(
        [
            "arm",
            "τ",
            "α at the first <code>=</code>",
            "rise to the query <code>=</code>",
            "r², posterior",
            "r², count",
            "r², sum",
            "Δα per non-fitting example",
            "step, non-fit to loosest fit",
            "grade, loosest to tightest fit",
        ],
        rows,
        caption=f"""
            **The sweep trials with a soft pool (τ ≥ {SOFT_TAU}), by τ.** α at the first `=` of
            `{ex.ANCHORED_OP}` contexts and the rise from there to the query `=`; the share of the variance in α over
            the `=` tokens with evidence before them (r²) accounted for by the posterior, by the count of fitting
            examples, and by the sum of the one-example posteriors; the mean change in α when one more non-fitting
            example is added at a fixed count (the posterior falls by {-POST_NONFIT:.2f}); then at the second `=`,
            the step in α from a first example that does not fit to the loosest fit, and the further rise from
            there to the most specific fit. The rising trials (rise ≥ {RISE_LEVEL}) are the ones E2 plots.
        """,
    )


rf"""
{trial_table()}
"""

# %%


def by_slice() -> dict[str, list]:
    """α by slice at the first `=` and the query `=`, on `difference` and on the other ops' contexts, the mean
    over the rising trials with the range.
    """
    out: dict[str, list] = {}
    for site, j in (("first", 0), ("query", K)):
        for group, m in (("difference", IS_D), ("other", ~IS_D)):
            per = np.array([ARRAYS[lbl][:, m, j].astype(np.float64).mean(1) for lbl in RISING])  # (trials, slices)
            out[f"{site}-{group}"] = [per.mean(0).tolist(), per.min(0).tolist(), per.max(0).tolist()]
    return out


def e3_figure() -> str:
    alt = f"""
        α by slice, from the embedding to block 4. The three lines for the first `=` and for the query `=` of other
        ops lie on top of each other: about 0.2 at the embedding, 0.35 after block 1, then falling to 0.15 at block
        4. The line for the query `=` of `{ex.ANCHORED_OP}` contexts leaves them at block 2 and reaches about
        0.45 at blocks 3 and 4.
    """
    return e3_draw(by_slice(), alt)


@memo
def e3_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="alpha-at-equals-by-slice",
        alt_text=alt_text,
        caption=f"""
            **Where the anchor at the `=` sits by depth, on the {words(len(RISING))} rising trials.** α at the first `=`
            and at the query `=`, on `{ex.ANCHORED_OP}` contexts and on the other ops' contexts, at each slice from
            the embedding (emb) to the output of block {ex.N_SLICES - 1}. The mean over trials, with the range as a
            band.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.6, 3.2), layout="constrained")
        x = np.arange(ex.N_SLICES)
        style = {
            "first-other": ("--", "o", light_dark("#999", "#777"), "first =, other ops"),
            "first-difference": ("-", "o", light_dark("#999", "#777"), f"first =, {ex.ANCHORED_OP}"),
            "query-other": ("--", "D", light_dark("#c0392b", "#ff8a76"), "query =, other ops"),
            "query-difference": ("-", "D", light_dark("#c0392b", "#ff8a76"), f"query =, {ex.ANCHORED_OP}"),
        }
        for key, (ls, marker, color, label) in style.items():
            mean, lo, hi = data[key]
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0, zorder=1)
            ax.plot(x, mean, ls, marker=marker, ms=3.5, lw=1.1, color=color, label=label, zorder=3)
        ax.axhline(0, color=light_dark("#333", "#ddd"), lw=0.5)
        ax.set_xticks(x, ["emb", *(f"block {i}" for i in range(1, ex.N_SLICES))])
        ax.set_ylabel("α", fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


rf"""
## Where it sits by depth (E3)

E1 split the anchor at the `=` into a part every context shares and a part that follows the examples. If the two sat at different depths, a pull or an edit could address one without the other.

{e3_figure()}

They do, in part. The shared part is there from the embedding slice, where α at a `=` is the e₁ component of the `=` embedding itself (the sweep found the syntax embeddings partway onto e₁ at a soft pool), and on the mean it is largest after block 1 and fades from there, to about half of its peak by block 4. The part that follows the examples is absent until block 2, grows through block 3, and holds at block 4. So the last two blocks carry the contextual part with the least of the shared one, and an edit at the early blocks would act on the shared part alone. The trials differ in the size of the shared part (the bands) and in some of the detail: on two of the six the shared part peaks at the embedding rather than after block 1, and the contextual part first shows at block 2 on two trials, at block 3 on three, and only at block 4 on one. What they share is the order: the shared part first and fading, the contextual part later and holding.
<!-- REVIEW: "a third of its peak" corrected to "about half": per trial the block-4 value is 0.08 to 0.66 of the peak (mean curve about 0.45), and the former "not in this shape" was not true of the individual trials (peak slice and onset slice both vary). Verify: per-trial `ARRAYS[lbl][:, ~IS_D, 0].mean(1)` and the query-`difference` minus query-other difference by slice. -->
## Discussion

The item asked whether the anchor grades with the posterior on the op. At the `=` tokens, the sites where the model has to infer it, the answer on the recipe is that there is nothing to grade: the pull is met at the example answers, and the `=` tokens are left about where the control has them. That agrees with ex-2.2.21, which found the anchor weak at the query `=`, and extends it to the `=` of every example.

A softer pool puts anchor on the `=` tokens, and some of it follows the examples. It grades with the per-example judgements that example-evidence found at the answers: each fitting example adds to α by how well it fits. Whether an example that does not fit also takes something away, as it does from the posterior, this stimulus can't resolve at one seed per trial, since the posterior moves little there. If the anchor turns out to sum rather than weigh, that would fit how the label trains: it is binary on the true op and treats a context with a misleading example the same as one without. The two trials with the most anchor at the `=` sit a little closer to the posterior than to the sum. A stimulus with more examples that do not fit, or more seeds, would separate the two.

The part every context shares is the larger of the two on most soft trials, and it is the price of the softer pool: a state the model can place on e₁ once, at a token every `{ex.ANCHORED_OP}` context has, meets the pooled pull more cheaply than a contextual feature does. The sweep found the spill of the edit growing with τ, and a component on the `=` state of every op is one place that spill would come from. It sits earlier in the stack than the contextual part, which is some help: an edit on the last two blocks would remove most of the contextual part and about half of the shared one. So on the stored runs, the anchor can be made to reach the sites where the op is inferred, but what it holds there is a sum of the examples that fit rather than the inferred op, and it brings a shared component along.
"""
