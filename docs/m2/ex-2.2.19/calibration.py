# title: Ex 2.2.19, follow-up: where the calibration KL sits

# A post hoc split of the calibration KL of ex-2.2.19 over every held-out context, by true op, by how sure the Bayes
# posterior is of that op, and by the op whose answers hold the mass the model has in excess of the Bayes answer
# distribution. It reads the per-context arrays ex-2.2.19 (and ex-2.2.18, for the 400-epoch run at the scout seed)
# published, and trains nothing.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import light_dark, themed

X = ex.ex2216
ALL_OPS: tuple[str, ...] = tuple(X.OP_NAMES)
OPS = ex.OP_SET.ops
N_OPS = len(OPS)
SET_IDS = [ALL_OPS.index(o) for o in OPS]
DROPPED_IDS = [i for i in range(len(ALL_OPS)) if i not in SET_IDS]
HSV_CHANNEL = ("hue-hsv", "sat-hsv", "value-hsv")
SEEDS = (ex.SCOUT_SEED, *ex.CONFIRM_SEEDS)
REF_E = ex.REFERENCE_EPOCHS
# The confident threshold of the ex-2.2.17 to ex-2.2.19 op confusion matrices, and the bins of posterior on the true
# op that this note splits the held-out set into; the last bin is the confident set.
CONFIDENT = 0.99
BINS = (0.0, 0.5, 0.9, CONFIDENT, 1.0 + 1e-9)
BIN_NAMES = ("below 0.5", "0.5 to 0.9", "0.9 to 0.99", "above 0.99")
# Columns of the excess split: the seven ops of the set, then colors only a dropped op gives, colors no op gives, and
# mass on tokens that are not colors.
COLUMNS = (*OPS, "dropped", "none", "non-color")


# --- Loading ------------------------------------------------------------------------------------------------


def read(ref: str, into: Path) -> Path:
    store = project_store()
    art = store.get_ref(ref)
    assert art is not None, f"{ref} is not published"
    return store.get(art, into / ref.replace("/", "-"))


with tempfile.TemporaryDirectory() as _tmp:
    EVAL = json.loads(read(ex.EVAL_REF, Path(_tmp)).read_text())
PICK: int = EVAL["selection"]["pick"]
LENGTHS = (PICK, REF_E)


def array_ref(epochs: int, seed: int) -> str:
    """The per-context arrays of the run at *epochs* and *seed*. The 400-epoch run at the scout seed is the ex-2.2.18
    `no-four` run, as in the ex-2.2.19 report.
    """
    if epochs == REF_E and seed == ex.SCOUT_SEED:
        return ex.ex2218.EVAL_ARRAYS_REF.format(label=ex.OP_SET.name)
    rate = ex.PEAK_LR if epochs == REF_E else EVAL["selection"]["rate"][str(epochs)]
    return ex.EVAL_ARRAYS_REF.format(label=ex.label_of(epochs, rate, seed))


def answer_table():
    """The answers of all eleven ops on every pair, so that the answers of a dropped op stay defined. Not memoized:
    the table class comes from a module loaded by path, which the disk cache cannot pickle, and it is quick to build.
    """
    return X._get_posterior().build_table(X.TABLE)


@memo
def split_run(ref: str) -> dict[str, np.ndarray]:
    """Per held-out context of one run: the true op, the posterior on it, the calibration KL, and that KL split over
    `COLUMNS` in proportion to where the model has mass in excess of the Bayes answer distribution.

    The excess on a color is the model mass beyond the Bayes mass, where it is positive. A color that ops of the set
    give is shared among them by the posterior over ops, renormalized over those that give it. A color that only a
    dropped op gives goes to `dropped`, and one that no op gives to `none`. Mass on tokens that are not colors goes to
    `non-color`. The shares sum to one in every context, so the columns of a context sum to its KL.
    """
    table = answer_table()
    with tempfile.TemporaryDirectory() as tmp, np.load(read(ref, Path(tmp))) as z:
        a = {k: z[k] for k in z.files}
    p, post, pair, true = a["p"].astype(np.float64), a["posterior"].astype(np.float64), a["query_pair"], a["op_ids"]
    n, n_colors = p.shape
    # The answer distribution of each of the eleven ops on each query pair, dense over the grid.
    dense = np.zeros((len(ALL_OPS), n, n_colors + 1))
    for o in range(len(ALL_OPS)):
        idx = table.idx[o, pair]
        np.put_along_axis(dense[o], np.where(idx < 0, n_colors, idx), table.prob[o, pair], axis=1)
    dense = dense[..., :n_colors]
    q = np.einsum("no,ony->ny", post, dense[SET_IDS])
    kl = (q * (np.log(np.where(q > 0, q, 1.0)) - np.log(np.maximum(p, 1e-300)))).sum(1)

    excess = np.maximum(p - q, 0.0)
    gives = dense[SET_IDS] > 0
    weight = post.T[:, :, None] * gives
    total = weight.sum(0)
    share = np.where(total > 0, weight / np.where(total > 0, total, 1.0), 0.0)
    by_op = (share * excess[None]).sum(2).T
    unclaimed = excess * (total == 0)
    by_dropped = (dense[DROPPED_IDS] > 0).any(0)
    cols = np.column_stack([by_op, (unclaimed * by_dropped).sum(1), (unclaimed * ~by_dropped).sum(1), 1.0 - p.sum(1)])
    frac = cols / np.maximum(cols.sum(1, keepdims=True), 1e-12)

    return {
        "true": true,
        "p_true": post[np.arange(n), true],
        "kl": kl,
        "kl_stored": a["kl"].astype(np.float64),
        "split": kl[:, None] * frac,
        "model_top": p.max(1),
        "bayes_top": q.max(1),
        "same_top": p.argmax(1) == q.argmax(1),
    }


RUNS = {(e, s): split_run(array_ref(e, s)) for e in LENGTHS for s in SEEDS}
N = len(RUNS[(PICK, ex.SCOUT_SEED)]["kl"])
assert all(len(r["kl"]) == N for r in RUNS.values())
# Answer distributions are stored in float16, so the recomputed KL differs a little from the stored one.
KL_DRIFT = max(abs(r["kl"].mean() - r["kl_stored"].mean()) for r in RUNS.values())
assert KL_DRIFT < 2e-3


# --- Derived measurements -----------------------------------------------------------------------------------


def bin_of(r: dict) -> np.ndarray:
    return np.digitize(r["p_true"], BINS[1:-1])


def kl_all(e: int) -> np.ndarray:
    """(seed,): the calibration KL of each run at length *e*."""
    return np.array([RUNS[(e, s)]["kl"].mean() for s in SEEDS])


def kl_share(e: int, b: int) -> np.ndarray:
    """(seed,): the part of the calibration KL of each run that sits in bin *b*: its sum over the bin, over all contexts."""
    return np.array([RUNS[(e, s)]["kl"][bin_of(RUNS[(e, s)]) == b].sum() / N for s in SEEDS])


def kl_frac(e: int, b: int) -> np.ndarray:
    """(seed,): the share of the calibration KL of each run that sits in bin *b*."""
    return kl_share(e, b) / kl_all(e)


def kl_mean(e: int, b: int) -> np.ndarray:
    """(seed,): the mean KL per context in bin *b*."""
    return np.array([RUNS[(e, s)]["kl"][bin_of(RUNS[(e, s)]) == b].mean() for s in SEEDS])


def top_mean(e: int, b: int) -> np.ndarray:
    """(seed,): the mean, over the contexts of bin *b*, of the mass the model puts on its top answer."""
    return np.array([RUNS[(e, s)]["model_top"][bin_of(RUNS[(e, s)]) == b].mean() for s in SEEDS])


def gap_mean(e: int, b: int) -> np.ndarray:
    """(seed,): the mean, over the contexts of bin *b*, of the mass the model puts on its top answer less the mass
    the Bayes answer distribution puts on its own.
    """
    return top_mean(e, b) - bayes_top(b)


# The posterior is a property of the contexts, which are the same in every run, so one run stands for all.
CONTEXTS = RUNS[(PICK, ex.SCOUT_SEED)]
BIN_IDX = bin_of(CONTEXTS)


def bin_frac(b: int) -> float:
    return float((BIN_IDX == b).mean())


def bayes_top(b: int) -> float:
    """The mean, over the contexts of bin *b*, of the mass the Bayes answer distribution puts on its top answer."""
    return float(CONTEXTS["bayes_top"][BIN_IDX == b].mean())


def same_top(e: int, b: int) -> float:
    """The seed mean of the share of contexts in bin *b* whose model top answer is the Bayes top answer."""
    return float(np.mean([RUNS[(e, s)]["same_top"][bin_of(RUNS[(e, s)]) == b].mean() for s in SEEDS]))


def matrix(e: int) -> np.ndarray:
    """(op, column): the seed mean of each part of the calibration KL at length *e*, over all held-out contexts.
    Rows are the true op. The whole matrix sums to the calibration KL.
    """
    return np.mean(
        [
            np.stack([r["split"][r["true"] == o].sum(0) / N for o in range(N_OPS)])
            for r in (RUNS[(e, s)] for s in SEEDS)
        ],
        axis=0,
    )


def row_kl(e: int, ops: Sequence[str]) -> float:
    return float(sum(matrix(e)[OPS.index(o)].sum() for o in ops))


def op_mean_kl(e: int, op: str) -> float:
    """The seed mean of the mean KL per context of true op *op*."""
    return float(np.mean([r["kl"][r["true"] == OPS.index(op)].mean() for r in (RUNS[(e, s)] for s in SEEDS)]))


def col_share(e: int, cols: Sequence[str], b: int | None = None) -> float:
    """The share of the KL (of bin *b*, or of every context) that the columns *cols* take."""
    rs = [RUNS[(e, s)] for s in SEEDS]
    masks = [np.ones(N, bool) if b is None else bin_of(r) == b for r in rs]
    num = np.mean([r["split"][m][:, [COLUMNS.index(c) for c in cols]].sum() for r, m in zip(rs, masks, strict=True)])
    den = np.mean([r["split"][m].sum() for r, m in zip(rs, masks, strict=True)])
    return float(num / den)


def diag_share(e: int, b: int | None = None) -> float:
    """The share of the KL that goes to the column of the true op."""
    rs = [RUNS[(e, s)] for s in SEEDS]
    masks = [np.ones(N, bool) if b is None else bin_of(r) == b for r in rs]
    num = np.mean([r["split"][m][np.arange(m.sum()), r["true"][m]].sum() for r, m in zip(rs, masks, strict=True)])
    den = np.mean([r["split"][m].sum() for r, m in zip(rs, masks, strict=True)])
    return float(num / den)


LAST = len(BIN_NAMES) - 1
CONF_SHARE = {e: kl_frac(e, LAST).mean() for e in LENGTHS}
LOW_SHARE = {e: kl_frac(e, 0).mean() for e in LENGTHS}
D_LOW = kl_mean(REF_E, 0) - kl_mean(PICK, 0)
D_CONF = kl_mean(REF_E, LAST) - kl_mean(PICK, LAST)
CONF_TOP_WITHIN = max(abs(gap_mean(e, LAST)).max() for e in LENGTHS)
# The largest distance of the model top mass from the Bayes top mass, over the three groups above 0.5, all seeds.
UPPER_TOP_WITHIN = max(abs(gap_mean(e, b)).max() for e in LENGTHS for b in range(1, LAST + 1))
DROPPED = tuple(o for o in ALL_OPS if o not in OPS)
assert (D_LOW > 0).all() and (D_CONF < 0).all(), "the prose says the two ends move apart at every seed"
assert all((gap_mean(e, 0) > 0.05).all() for e in LENGTHS), "the prose says the model commits more than Bayes there"
assert (gap_mean(REF_E, 0) > gap_mean(PICK, 0)).all(), "the prose says the longer run commits more, at every seed"
assert CONF_TOP_WITHIN < 0.03, "the prose says the top mass is near Bayes there"
assert UPPER_TOP_WITHIN < 0.05, "the prose says the model is near Bayes in the three upper groups"
assert all(top_mean(e, b).mean() > top_mean(e, b - 1).mean() for e in LENGTHS for b in range(1, LAST + 1)), (
    "the prose says the model is surer where the posterior is sharper"
)
assert max(OPS, key=lambda o: op_mean_kl(PICK, o)) == "mix"
_diag = {e: np.diag(matrix(e)[:, :N_OPS]) for e in LENGTHS}
assert (_diag[REF_E] > _diag[PICK]).all(), "the alt text says the diagonal is darker at the longer length in every row"
assert [o for i, o in enumerate(OPS) if matrix(PICK)[i].argmax() != i] == ["difference"], (
    "the alt text names the exception"
)


def fmt_range(v: np.ndarray, f: str = ".3f") -> str:
    return f"{v.min():{f}} to {v.max():{f}}"


# --- Figures ------------------------------------------------------------------------------------------------


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


# Each group is drawn at the log-odds of its median posterior, so the groups sit where their contexts do: the two
# middle groups are close, and the confident group is far to the right.
BIN_X = np.array([logit(float(np.median(CONTEXTS["p_true"][BIN_IDX == b]))) for b in range(len(BIN_NAMES))])
EDGES = BINS[1:-1]


def shade(i: int) -> tuple:
    lo, hi = light_dark((0.45, 0.9), (0.4, 0.85))
    return plt.get_cmap(light_dark("Blues", "magma"))(lo + (hi - lo) * i)


def seq_cmap():
    cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
    cmap.set_bad(light_dark("#fff", "#111"))
    return cmap


def cell_text_color(v: float, vmax: float) -> str:
    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
    return "#fff" if dark_cell else "#000"


def confidence_axis(ax: plt.Axes, ylabel: str) -> None:
    """The shared horizontal axis of the by-confidence figures: the posterior on a log-odds scale, the group edges as
    ticks and faint rules, and the group names above the panel.
    """
    muted = light_dark("#666", "#aaa")
    for e in EDGES:
        ax.axvline(logit(e), color=light_dark("#ccc", "#444"), lw=0.6, ls=":", zorder=0)
    for b, name in enumerate(BIN_NAMES):
        ax.text(
            BIN_X[b], 1.02, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=muted
        )
    ax.set_xticks([logit(e) for e in EDGES], [f"{e:g}" for e in EDGES], fontsize=8)
    ax.set_xlim(BIN_X[0] - 1.1, BIN_X[-1] + 0.8)
    ax.set_xlabel("Bayes posterior on the true op (log-odds scale)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def series(ax: plt.Axes, fn, rng: np.random.Generator) -> None:
    """One line per training length through the seed means of *fn(e, b)*, with the runs as small dots."""
    for i, e in enumerate(LENGTHS):
        vals = np.stack([fn(e, b) for b in range(len(BIN_NAMES))])  # (bin, seed)
        dx = (i - 0.5) * 0.14
        for b in range(len(BIN_NAMES)):
            x = BIN_X[b] + dx + rng.uniform(-0.03, 0.03, vals.shape[1])
            ax.plot(x, vals[b], "o", ms=2.4, color=shade(i), alpha=0.55, mew=0, zorder=2)
        ax.plot(
            BIN_X + dx,
            vals.mean(1),
            "o-",
            ms=5,
            lw=1.2,
            color=shade(i),
            mec=light_dark("white", "#111"),
            mew=0.6,
            label=f"{e} epochs",
            zorder=3,
        )


def legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=8)


@memo
def share_draw(alt_text: str, caption: str) -> str:
    @themed(name="kl-share-by-confidence", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
        ax.bar(
            BIN_X,
            [bin_frac(b) for b in range(len(BIN_NAMES))],
            width=0.6,
            color=light_dark("#ddd", "#333"),
            label="share of contexts",
            zorder=1,
        )
        series(ax, kl_frac, np.random.default_rng(0))
        confidence_axis(ax, "share")
        ax.set_ylim(0, None)
        legend(fig, ax)
        return fig

    return _plot()


@memo
def per_context_draw(alt_text: str, caption: str) -> str:
    @themed(name="kl-per-context-by-confidence", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
        series(ax, kl_mean, np.random.default_rng(1))
        confidence_axis(ax, "mean KL per context (nats)")
        ax.set_ylim(0, None)
        legend(fig, ax)
        return fig

    return _plot()


@memo
def top_mass_draw(alt_text: str, caption: str) -> str:
    @themed(name="top-mass-by-confidence", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
        ax.plot(
            BIN_X,
            [bayes_top(b) for b in range(len(BIN_NAMES))],
            "s--",
            ms=4.5,
            lw=1.0,
            color=light_dark("#777", "#999"),
            label="Bayes",
            zorder=2,
        )
        series(ax, top_mean, np.random.default_rng(2))
        confidence_axis(ax, "mass on the top answer")
        legend(fig, ax)
        return fig

    return _plot()


@memo
def matrix_draw(alt_text: str, caption: str) -> str:
    @themed(name="kl-by-op", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), layout="constrained", sharey=True)
        vmax = max(float(matrix(e).max()) for e in LENGTHS)
        im = None
        for ax, e in zip(axes, LENGTHS, strict=True):
            m = matrix(e)
            im = ax.imshow(m, cmap=seq_cmap(), vmin=0, vmax=vmax, aspect="equal")
            for i, j in np.ndindex(*m.shape):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.3f}"[1:],
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=cell_text_color(m[i, j], vmax),
                )
            for i in range(N_OPS):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="0.6", lw=0.8))
            ax.set_title(f"{e} epochs (KL {kl_all(e).mean():.3f})", fontsize=9)
            ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=90, fontsize=8)
            ax.set_yticks(range(N_OPS), [f"{o}  {m[i].sum():.3f}" for i, o in enumerate(OPS)], fontsize=8)
        fig.supxlabel("where the model has mass in excess of the Bayes answer distribution", fontsize=9)
        fig.supylabel("true op, and its part of the KL", fontsize=9)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.7, label="part of the calibration KL (nats)")
        return fig

    return _plot()


# --- Report -------------------------------------------------------------------------------------------------

rf"""
# Ex 2.2.19, follow-up: where the calibration KL sits

/// tip |
<!-- tl;dr -->
The largest share of the calibration KL of the seven-op model is in the quarter of held-out contexts whose examples fit several ops, where the model is surer of its answer than the examples warrant. Longer training leaves the total where it was and moves more of it onto those contexts.
///

A well-calibrated model is as sure of its answer as the examples warrant: sure when they point to one op, spread out when they fit several. The calibration KL of ex-2.2.19 is how far the seven-op model is from that, over all held-out contexts, and this note asks where that distance comes from. The contexts whose examples fit several ops are a quarter of the held-out set and hold {LOW_SHARE[PICK]:.0%} of the KL, and there the model is too sure of itself: it puts {top_mean(PICK, 0).mean():.2f} on its favorite answer where the examples warrant {bayes_top(0):.2f}. Where the examples point to one op, which is where the op confusion matrices look, it is about as sure as it should be, and those contexts hold {CONF_SHARE[PICK]:.0%} of the KL. Training for {REF_E} epochs instead of {PICK} leaves the total where it was but moves it: at every seed the longer run is a little better calibrated where the op is clear and worse where it is not. By op, the KL is spread over all seven, highest on `mix`.

## Scope

[Ex-2.2.19](./report.py) measured two things about how a model spreads its answers. The calibration KL[^kl] is taken over every held-out context. The op confusion matrix (E3) shows which op answers the model leans toward, but only on confident contexts, where the Bayes posterior on the true op is above {CONFIDENT}. We ask whether a matrix like the confusion one can be made over the whole held-out set, showing how much each op contributes to the KL ([backlog item](/todo/science/per-op-calibration-kl-over-all-contexts.md)).

[^kl]: The KL divergence from the Bayes answer distribution to that of the model, in nats, averaged over contexts. It is the extra cross-entropy loss of the model over a perfectly calibrated one, so zero is the best a model can do.

This note looks at the KL two ways, by how clearly the examples pick out the op and by op, on the {PICK}- and {REF_E}-epoch runs of the seven-op set at four seeds each ({len(SEEDS)} runs per length). It is post hoc, with no hypotheses or gates, and trains nothing. Each length has {N:,} held-out contexts, the same contexts in every run.

## By confidence

The first way groups contexts by the Bayes posterior on the true op: the probability a perfect reader of the examples would give the op that generated them. Where it is near one the examples fit one op alone; where it is below a half they fit several about as well. The four groups run from a posterior below 0.5 to above {CONFIDENT}, and the last is the confident set of the confusion matrices. The first figure shows how the contexts and the KL spread over them.

"""

share_draw(
    f"""
        Four groups of contexts on a log-odds axis of the Bayes posterior, from below 0.5 to above {CONFIDENT}; the two
        middle groups sit close together and the confident group far to the right. Grey bars give the share of
        contexts in each group, about {bin_frac(0):.0%}, {bin_frac(1):.0%}, {bin_frac(2):.0%}, and
        {bin_frac(LAST):.0%}. Two lines, for {PICK} and {REF_E} epochs, give the share of the KL: about
        {LOW_SHARE[PICK]:.0%} in the first group, well above its bar, then about {kl_frac(PICK, 1).mean():.0%} and
        {kl_frac(PICK, 2).mean():.0%} in the middle groups, near their bars, and about {CONF_SHARE[PICK]:.0%} in the
        last, well below its bar. The two lines nearly coincide.
    """,
    f"""
        **Where the contexts are, and where the KL is, by how sure the Bayes posterior is of the true op.** The
        horizontal axis is the posterior on a log-odds scale, with the group edges as ticks; each group is drawn at
        the log-odds of its median posterior. Bars: the share of the {N:,} held-out contexts in the group. Lines: the
        share of the calibration KL that the contexts of the group hold, one line per training length, with the
        {len(SEEDS)} runs as small dots and the seed mean as large dots.
    """,
)

rf"""
The largest share of the KL is in the least confident contexts. The group below 0.5 is {bin_frac(0):.0%} of the contexts and holds {LOW_SHARE[PICK]:.0%} of the KL at {PICK} epochs and {LOW_SHARE[REF_E]:.0%} at {REF_E}. The confident set is {bin_frac(LAST):.0%} of the contexts and holds {CONF_SHARE[PICK]:.0%} and {CONF_SHARE[REF_E]:.0%}. In the figure, the line sits above the bar on the left and below it on the right.

Per context, the KL falls steadily as the posterior sharpens (next figure), from about {kl_mean(PICK, 0).mean():.2f} nats below 0.5 to {kl_mean(PICK, LAST).mean():.2f} above {CONFIDENT}. The two lengths cross: from {PICK} to {REF_E} epochs, the KL per context below 0.5 rises by {fmt_range(D_LOW)} over the four seeds, and in the confident set it falls by {fmt_range(-D_CONF)}. Every seed moves the same way at both ends. The total barely changes ({kl_all(PICK).mean():.3f} against {kl_all(REF_E).mean():.3f}), which is what ex-2.2.19 reported as calibration not separating the two lengths.

"""

per_context_draw(
    f"""
        The mean KL per context in each of the four groups, on the same log-odds axis, with one line per training
        length. Both fall from about {kl_mean(PICK, 0).mean():.2f} in the first group to about
        {kl_mean(PICK, LAST).mean():.2f} in the last. The {REF_E}-epoch line is above the {PICK}-epoch line in the
        first group and below it in the last, and the four runs of each length sit close to their mean.
    """,
    f"""
        **The calibration KL per context, by group.** The mean KL over the contexts of the group, in nats, one line
        per training length; small dots are the {len(SEEDS)} runs and large dots the seed mean. Lower is better.
    """,
)

r"""
The last figure shows how sure the model is against how sure it should be: the mass each puts on its single most likely answer.

"""

top_mass_draw(
    f"""
        The mass on the top answer in each of the four groups, on the same log-odds axis. A grey dashed line for the
        Bayes answer distribution rises from about {bayes_top(0):.2f} in the first group to {bayes_top(LAST):.2f} in
        the last. The model lines rise in step with it and lie on it in the three upper groups, but in the first
        group they sit above it, at about {top_mean(PICK, 0).mean():.2f} for {PICK} epochs and
        {top_mean(REF_E, 0).mean():.2f} for {REF_E}.
    """,
    f"""
        **How sure the model is, against how sure the examples warrant.** The mean, over the contexts of the group,
        of the mass on the most likely answer: grey for the Bayes answer distribution, and one line per training
        length for the model, with the {len(SEEDS)} runs as small dots. Above the grey line the model is surer than the
        examples warrant.
    """,
)

rf"""
Below a posterior of 0.5, Bayes puts {bayes_top(0):.2f} on its top answer and the model puts {top_mean(PICK, 0).mean():.2f} at {PICK} epochs and {top_mean(REF_E, 0).mean():.2f} at {REF_E}, more at every seed. Its top answer is also the Bayes top answer in only {same_top(PICK, 0):.0%} of those contexts, so it leans the wrong way about as often as the right one. In the other three groups the model is within {UPPER_TOP_WITHIN:.2f} of Bayes. So on confident contexts the longer run gains elsewhere than the top answer, in the small amounts of mass on the other colors: at {REF_E} epochs, less of the KL of the confident set goes to colors that no op gives ({col_share(REF_E, ["none"], LAST):.0%}, against {col_share(PICK, ["none"], LAST):.0%} at {PICK}).

So the model does know when the examples are clear. Its confidence tracks the posterior from one end of the axis to the other, and where the examples fit one op it is as sure as it should be, and the little KL there is small amounts of mass spread over wrong colors. Where the examples fit several ops it does spread its answer, but it leans toward one op more than the examples justify, and often toward the wrong one. Longer training makes that lean stronger while tidying the clear contexts, so the total stands still. The largest part of the KL that remains is this one habit.

## By op

The second way makes the matrix. Rows are the true op, and a row adds up to the part of the KL on the contexts of that op. Columns say where the model has mass in excess of the Bayes answer distribution, and each context divides its KL among the columns in proportion to that excess. Excess on a color that several ops of the set give is shared among them by the Bayes posterior over ops, since the posterior the model itself holds over ops is not something we can measure: the arrays hold its answer distribution and nothing of how it got there.

Three columns catch excess on colors outside the set. The seven-op set is the `no-four` set of ex-2.2.18: the eleven ops of ex-2.2.16 less {", ".join(f"`{o}`" for o in DROPPED[:-1])}, and `{DROPPED[-1]}`. Those four dropped ops are left out of the corpus the model trains on, and nothing is edited: the model is the plain trained model. Their answers are still defined on the color table, so a color that only a dropped op gives has the column `dropped`; a color that no op gives on the query pair has `none`; and mass on tokens that are not colors has `non-color`. (`difference` is in the set, so it has a row like any other.)

The column split is somewhat arbitrary. The KL counts the mass that the model is short of on the Bayes answers, and the excess says where that mass went; giving the KL to the excess in proportion is one way to join them. The rows add up either way.

In both maps a higher number is more KL, so worse. A high row says the model is poorly calibrated on the contexts of that op, with too much or too little mass on any answer; it does not say which op the model answered with. A high column says the model puts too much mass on the answers of that op; a column never counts too little.

"""

matrix_draw(
    f"""
        Two heatmaps of the seven true ops by ten columns: the seven ops, then dropped, none, and non-color, with square
        cells. The left is {PICK} epochs and the right {REF_E}. The diagonal is outlined and is the darkest square in
        every row but difference at {PICK} epochs, where the none column is larger; it is largest for mix, at about
        {matrix(PICK)[0, 0]:.3f}, and darker at {REF_E} epochs in every row. The none column is the next darkest; the
        rest of each row is pale and even, and the non-color column is near zero. Row totals beside the op names run
        from about {min(matrix(PICK).sum(1)):.3f} to {max(matrix(PICK).sum(1)):.3f}.
    """,
    f"""
        **The calibration KL by true op and by where the model has excess mass, at {PICK} and {REF_E} epochs.** Seed
        mean over {len(SEEDS)} runs, over all held-out contexts. Each square is a part of the KL in nats, and the
        whole map sums to the calibration KL in the title; higher is worse. Beside each op name is the sum of its
        row. The diagonal, outlined, is the excess on the answers of the true op.
    """,
)

rf"""
By row, the KL is spread over all seven ops. The three HSV-channel ops hold {row_kl(PICK, HSV_CHANNEL) / kl_all(PICK).mean():.0%} of it at {PICK} epochs, for three ops of seven. The highest KL per context is on `mix` ({op_mean_kl(PICK, "mix"):.2f}), and the lowest on `lighten` and `darken` ({op_mean_kl(PICK, "lighten"):.2f} and {op_mean_kl(PICK, "darken"):.2f}). On confident contexts, the confusion matrix of ex-2.2.19 put most of the extra mass of the shorter run in the HSV-channel rows. Over all contexts, the KL does not gather there to the same degree.

By column, the true op takes {diag_share(PICK):.0%} of the KL, the other six ops of the set {col_share(PICK, OPS) - diag_share(PICK):.0%}, colors no op gives {col_share(PICK, ["none"]):.0%}, and the dropped ops {col_share(PICK, ["dropped"]):.0%}. Mass on tokens that are not colors is negligible. Below a posterior of 0.5 the other ops take {col_share(PICK, OPS, 0) - diag_share(PICK, 0):.0%}, consistent with a model that leans further than Bayes toward one op when the examples allow several. No single other op stands out in any row. From {PICK} to {REF_E} epochs the share of the true op rises to {diag_share(REF_E):.0%} and that of colors no op gives falls to {col_share(REF_E, ["none"]):.0%}: the longer run moves mass off the colors no op gives and onto the answers of the true op.

## What it means for what follows

The KL that remains at {PICK} and {REF_E} epochs is mostly in contexts whose op is unclear, which the confusion matrices leave out. So a lower KL would come mainly from better hedging between ops on those contexts, and less from the HSV-channel ops, whose shortfall shows in exact match on confident contexts.

Is Bayes a fair standard there? The Bayes answer distribution is the distribution of the answer given the examples under the process that made the corpus: an op drawn uniformly, examples with replacement op noise and cube noise at the rates the corpus used, and a clean query. It uses nothing about a context beyond its examples, which the model also sees, and a held-out context is a fresh draw from the same process. A model that has learned the process is calibrated on it, and cross-entropy training pulls toward that distribution, so zero KL is reachable in principle and the standard is no oracle. The geometry of the cube is part of the standard too: the answer table is each op applied to every ordered pair of grid colors, so an op that clips at a face of the cube, or rounds to several neighboring colors, has those answers at those probabilities in the posterior and in the answer distribution alike. What the model has to learn, and has not fully learned at either length, is how much the examples leave open.

The longer run trades calibration on unclear contexts for calibration on clear ones. That fits a model that keeps sharpening its answers with training, and it could be overfitting to the training corpus; the held-out arrays alone cannot tell the two apart, since there is no training-set KL to set them beside. One can be had without retraining: the end checkpoints of every run are stored, so an evaluation pass over a sample of training contexts, a few minutes of L4 time per run, would give it ([backlog item](/todo/science/train-vs-heldout-kl-on-unclear-contexts.md)). Either way, a level total KL between two lengths does not mean the two are calibrated alike.

## Method

The runs are the {PICK}- and {REF_E}-epoch runs of the seven-op set in ex-2.2.19, at seeds {", ".join(str(ex.SEED_OFFSET + s) for s in SEEDS)}; the {REF_E}-epoch run at seed {ex.SEED_OFFSET + ex.SCOUT_SEED} is the ex-2.2.18 `no-four` run. Each run stored, per held-out context, the true op, the posterior over the seven ops, the query pair, and the model answer distribution over the grid (in float16).

The model answer distribution is the softmax over the whole vocabulary at the query `=`, the position that predicts the answer, taken at the color tokens. Mass on tokens that are not colors is kept as loss rather than renormalized away, and the greedy answer plays no part. The Bayes answer distribution is rebuilt from the posterior and the answer table, and the KL from it agrees with the stored KL to within {KL_DRIFT:.4f} on average per run, the difference coming from the float16 storage.
"""
