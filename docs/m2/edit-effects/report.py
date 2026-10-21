# ruff: noqa: B018
# title: What the edit does on the recipe of record

# An exploratory re-analysis of split-anti-seeds' `falloff` runs at seeds 704 to 711, with no new training.
# `experiment.py` beside this script stores the alignment with e₁ at every slice and position, clean and under the
# full edit; the grading and the landing come from the dose arrays split-anti-seeds published.
import json
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, smooth_step, themed

# --- The stored arrays ----------------------------------------------------------------------------------------


def fetch_all() -> tuple[dict, np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """The results summary, the fit of every example answer under every op, the alignment arrays of each run (clean
    and arriving under the full edit), and the dose arrays split-anti-seeds published for the same runs.
    """
    store = project_store()

    def published(refs: list[str]) -> dict[str, Any]:
        got = store.get_refs(refs)
        missing = [r for r, a in got.items() if a is None]
        assert not missing, f"not published: {missing}"
        return got

    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        head = published([ex.RESULTS_REF, ex.FITS_REF])
        summary_path, fits_path = store.get_many(
            [(head[ex.RESULTS_REF], into / "results.json"), (head[ex.FITS_REF], into / "fits.npz")]
        )
        summary = json.loads(summary_path.read_text())
        labels = [r["label"] for r in summary["runs"]]
        alpha_refs = [ex.ARRAYS_REF.format(label=lbl) for lbl in labels]
        dose_refs = [ex.sas.DOSE_ARRAYS_REF.format(label=lbl) for lbl in labels]
        arts = published(alpha_refs + dose_refs)
        paths = store.get_many(
            [(arts[r], into / f"alpha-{lbl}.npz") for r, lbl in zip(alpha_refs, labels, strict=True)]
            + [(arts[r], into / f"dose-{lbl}.npz") for r, lbl in zip(dose_refs, labels, strict=True)]
        )
        with np.load(fits_path) as z:
            fits = z["fits"]
        alpha, dose = {}, {}
        for lbl, p in zip(labels + labels, paths, strict=True):
            with np.load(p) as z:
                if p.name.startswith("alpha-"):
                    alpha[lbl] = {k: z[k] for k in ("clean", "arriving", "op_ids")}
                else:
                    keep = ("op_ids", "posterior", "query_pair", "clean_eem", "null_eem", "eem", "p_clean", "p_full")
                    dose[lbl] = {k: z[k] for k in keep}
    return summary, fits, alpha, dose


SUMMARY, FITS, ALPHA, DOSE = fetch_all()
OPS: tuple[str, ...] = tuple(SUMMARY["design"]["ops"])
D = OPS.index(ex.ANCHORED_OP)
SEEDS: list[int] = SUMMARY["design"]["seeds"]
P = ex.ex2216._get_posterior()
TABLE = P.build_table(ex.ex2222.ex2218.table_of(OPS))
OP_IDS = ALPHA[f"falloff-s{SEEDS[0]}"]["op_ids"].astype(int)
IS_D = OP_IDS == D
for lbl in ALPHA:
    assert (ALPHA[lbl]["op_ids"] == OP_IDS).all() and (DOSE[lbl]["op_ids"] == OP_IDS).all()

UNIT = 6  # tokens per equation: a ? b = y and a separator
N_POS = UNIT * (ex.K + 1)
ANSWERS = [UNIT * e + 4 for e in range(ex.K)]  # the example answers
QUERY_ANSWER = UNIT * ex.K + 4
ROLE = ("a", "?", "b", "=", "y", ",")
POS_LABEL = [ROLE[p % UNIT] if p != N_POS - 1 else "⏎" for p in range(N_POS)]
SLICE_TICK = ["emb", "1", "2", "3", "4"]
Q = 0.99  # the quantile the bound and the write are taken at, as in H4 of ex-2.2.1
TOLERANCE = 1.25  # the tolerance of H4 in ex-2.2.1: the write at most a quarter over the bound
FITS_D = FITS[:, :, D]  # (contexts, examples + query): the answer is one `difference` can give
assert FITS_D.shape[1] == ex.K + 1
FALLOFF, CONTROL = "falloff", "control"


def label_of(arm: str, seed: int) -> str:
    return f"{arm}-s{seed}"


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


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


INK = {FALLOFF: light_dark("#2a9d8f", "#7fd8c8"), CONTROL: light_dark("#888", "#999")}
WRITE_INK = light_dark("#c0392b", "#ff8a76")
GHOST = light_dark("#bbb", "#555")
RULE = light_dark("#aaa", "#555")


# --- E1: the write and the bound -----------------------------------------------------------------------------


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


TURN = {
    label_of(arm, s): turn_maps(ALPHA[label_of(arm, s)]["clean"], ALPHA[label_of(arm, s)]["arriving"], OP_IDS, FITS_D)
    for arm in (FALLOFF, CONTROL)
    for s in SEEDS
}


def seed_mean(arm: str, key: str, sub: str | None = None) -> np.ndarray:
    return np.mean(
        [TURN[label_of(arm, s)][key] if sub is None else TURN[label_of(arm, s)][key][sub] for s in SEEDS], axis=0
    )


def angle(alpha) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(alpha, 0, 1)))


BOUND = seed_mean(FALLOFF, "bound")
WRITE = seed_mean(FALLOFF, "write")
RATIO = np.arcsin(WRITE) / np.arcsin(BOUND)  # (slices, positions), seed-mean quantiles as in ex-2.2.1
assert np.allclose(RATIO[0], 1.0, atol=1e-3), "the embedding sees the clean stream"
RATIO_MAX = RATIO[1:].max()
RATIO_WORST = np.unravel_index(RATIO[1:].argmax(), RATIO[1:].shape)
# Per seed: the sites past the embedding where the write goes over the tolerance, and the bound there.
OVER = [
    (s, sl, p, float(r), float(angle(TURN[label_of(FALLOFF, s)]["bound"][sl, p])))
    for s in SEEDS
    for sl in range(1, 5)
    for p in range(N_POS)
    if (
        r := np.arcsin(TURN[label_of(FALLOFF, s)]["write"][sl, p])
        / np.arcsin(TURN[label_of(FALLOFF, s)]["bound"][sl, p])
    )
    > TOLERANCE
]
OVER_BOUND_MAX = max((o[4] for o in OVER), default=0.0)
# The write at the example answers that do not fit, as a share of the bound there, by slice.
ANS_RATIO = np.arcsin(seed_mean(FALLOFF, "nofit", "q_arriving")) / np.arcsin(seed_mean(FALLOFF, "nofit", "q_clean"))
N_FIT = TURN[label_of(FALLOFF, SEEDS[0])]["fit"]["n"]
N_NOFIT = TURN[label_of(FALLOFF, SEEDS[0])]["nofit"]["n"]


def turn_table() -> str:
    rows = []
    for name, arm, sub in (
        ("do not fit", FALLOFF, "nofit"),
        ("fit", FALLOFF, "fit"),
        ("control, do not fit", CONTROL, "nofit"),
    ):
        for kind, which in (("q", "99th pct."), ("med", "median")):
            c, a = seed_mean(arm, sub, f"{kind}_clean"), seed_mean(arm, sub, f"{kind}_arriving")
            rows.append(
                [f"{name}, {which}"] + [f"{angle(x):.0f}° → {angle(y):.0f}°" for x, y in zip(c, a, strict=True)]
            )
    return table_html(
        ["example answers, other ops", *SLICE_TICK],
        rows,
        f"""**How far the edit turns the example answers of the other ops**, by slice, as the angle of the clean
        state (the bound, at the 99th percentile) and the angle it arrives with under the edit (the write); seed
        mean over {len(SEEDS)} runs. An answer *fits* when `{ex.ANCHORED_OP}` could give it on its operands
        ({N_FIT:,} answers per run, against {N_NOFIT:,} that do not). The control has no anchor; the seed
        means of a 99th percentile and a median are each taken per run, then averaged.""",
    )


# %%


@memo
def turn_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="edit-effects-turn",
        alt_text=alt_text,
        caption=f"""
            **The bound and the write on the contexts of the other ops**, by position along a context, one panel
            per slice (the embedding at the bottom). The {Q:.0%} quantile of |α| over the {int((~IS_D).sum()):,}
            contexts of the six other ops, seed mean: clean (the bound, solid) and as it arrives under the full edit
            (the write, dashed). At the answers (*y*) only those that do not fit `{ex.ANCHORED_OP}` count. The control, with no anchor, is the ghost.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(5, 1, figsize=(6.4, 5.2), layout="constrained", sharex=True, sharey=True)
        axes = cast(AxesRow, axes)
        x = np.arange(N_POS)
        for row, ax in enumerate(axes):
            sl = 4 - row
            smooth_step(ax, x, data["control"][sl], color=GHOST, lw=0.8, ramp=0.6, zorder=1)
            smooth_step(ax, x, data["bound"][sl], color=INK[FALLOFF], lw=1.4, ramp=0.6, zorder=3)
            ax.plot(x, data["bound"][sl], ls="none", marker="o", ms=3, color=INK[FALLOFF], zorder=4)
            smooth_step(ax, x, data["write"][sl], color=WRITE_INK, lw=1.2, ls="--", ramp=0.6, zorder=3)
            ax.plot(x, data["write"][sl], ls="none", marker="v", ms=3, color=WRITE_INK, zorder=4)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1])
            ax.text(1.01, 0.5, SLICE_TICK[sl], transform=ax.transAxes, va="center", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[-1].set_xticks(x, POS_LABEL, fontsize=7)
        axes[2].set_ylabel(f"|α|, {Q:.0%} quantile", fontsize=8)
        handles = [
            plt.Line2D([], [], color=INK[FALLOFF], marker="o", ms=3, lw=1.4, label="clean (bound)"),
            plt.Line2D([], [], color=WRITE_INK, marker="v", ms=3, lw=1.2, ls="--", label="under the edit (write)"),
            plt.Line2D([], [], color=GHOST, lw=0.8, label="control, clean"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False, fontsize=8)
        return fig

    return _plot()


def turn_figure() -> str:
    data = {"bound": BOUND, "write": WRITE, "control": seed_mean(CONTROL, "bound")}
    alt = (
        f"Five stacked panels, one per slice, of the {Q:.0%} quantile of |alpha| along the 24 positions of a context, "
        "on contexts of the other ops. At the embedding the clean and edited curves coincide near 0.08 at the colors. "
        f"At slice 1 both sit near 0.55 at the example answers and below 0.1 elsewhere. At slices 2 to 4 the clean "
        "curve stays low except at the example answers, and the curve under the edit sits at or below the clean one "
        "everywhere. The control ghost sits near 0.3 at the colors at every slice."
    )
    return turn_draw(data, alt)


# --- E2: the grading -----------------------------------------------------------------------------------------

POST_BINS = (0.0, 0.2, 0.5, 0.8, 0.95, 1.0)
BIN_LABEL = ["< 0.2", "0.2–0.5", "0.5–0.8", "0.8–0.95", "≥ 0.95"]


def contexts_of(d: dict) -> Any:
    n = len(d["op_ids"])
    empty = np.zeros((n, 0), np.int64)
    return P.Contexts(d["op_ids"].astype(np.int64), empty, empty, d["query_pair"].astype(np.int64))


@memo
def bayes(posterior: np.ndarray, op_ids: np.ndarray, query_pair: np.ndarray) -> dict[str, np.ndarray]:
    """The Bayes predictor and the target null on every held-out context: expected exact match of each, and the
    answer distributions over the palette.
    """
    ctx = contexts_of({"op_ids": op_ids, "query_pair": query_pair})
    post = posterior.astype(np.float64)
    null = ex.ex2216._target_null(P, TABLE, ctx, post, D)
    return {
        "ideal_eem": P.expected_match(TABLE, ctx, post),
        "null_eem": P.expected_match(TABLE, ctx, null),
        "q_ideal": P.predictive(TABLE, ctx, post),
        "q_null": P.predictive(TABLE, ctx, null),
        "null_post": null,
    }


d0 = DOSE[label_of(FALLOFF, SEEDS[0])]
BAYES = bayes(d0["posterior"], d0["op_ids"], d0["query_pair"])
for lbl_ in DOSE:
    assert np.allclose(BAYES["null_eem"], DOSE[lbl_]["null_eem"], atol=2e-4)
POST_D = d0["posterior"][:, D].astype(np.float64)
BIN = np.clip(np.digitize(POST_D, POST_BINS[1:-1]), 0, len(POST_BINS) - 2)
DEP_BAYES = BAYES["ideal_eem"] - BAYES["null_eem"]  # how much each context needed `difference`, model-free


def net_drop(seed: int) -> np.ndarray:
    """Per context: the drop in expected exact match under the full edit, less the drop on the control there."""
    r, c = DOSE[label_of(FALLOFF, seed)], DOSE[label_of(CONTROL, seed)]
    return (r["clean_eem"] - r["eem"][-1]) - (c["clean_eem"] - c["eem"][-1])


def dep_model(seed: int) -> np.ndarray:
    """Per context: how far the clean answer is above the target null, the most the edit could take off."""
    r = DOSE[label_of(FALLOFF, seed)]
    return r["clean_eem"] - r["null_eem"]


NET = {s: net_drop(s) for s in SEEDS}
DEPM = {s: dep_model(s) for s in SEEDS}
N_BIN_D = [int(((BIN == b) & IS_D).sum()) for b in range(len(BIN_LABEL))]
# The share of the dependence removed per bin of the posterior on `difference`, as a ratio of bin means.
SHARE = {
    s: np.array([NET[s][(BIN == b) & IS_D].mean() / DEPM[s][(BIN == b) & IS_D].mean() for b in range(5)]) for s in SEEDS
}
# The same, without subtracting the control: the share of the way to the target null the edit takes on its own.
RAW_SHARE = {
    s: np.array(
        [
            (DOSE[label_of(FALLOFF, s)]["clean_eem"] - DOSE[label_of(FALLOFF, s)]["eem"][-1])[(BIN == b) & IS_D].mean()
            / DEPM[s][(BIN == b) & IS_D].mean()
            for b in range(5)
        ]
    )
    for s in SEEDS
}
RAW_SHARE_ALL = {
    s: (DOSE[label_of(FALLOFF, s)]["clean_eem"] - DOSE[label_of(FALLOFF, s)]["eem"][-1])[IS_D].mean()
    / DEPM[s][IS_D].mean()
    for s in SEEDS
}
RAW_SHARE_MEAN = np.mean([RAW_SHARE[s] for s in SEEDS], axis=0)
CONTROL_DROP_D = np.mean(
    [(DOSE[label_of(CONTROL, s)]["clean_eem"] - DOSE[label_of(CONTROL, s)]["eem"][-1])[IS_D].mean() for s in SEEDS]
)
SHARE_ALL = {s: NET[s][IS_D].mean() / DEPM[s][IS_D].mean() for s in SEEDS}
R_BAYES = {s: float(np.corrcoef(NET[s][IS_D], DEP_BAYES[IS_D])[0, 1]) for s in SEEDS}
SHARE_MEAN = np.mean([SHARE[s] for s in SEEDS], axis=0)
SHARE_SPREAD = float(SHARE_MEAN.max() - SHARE_MEAN.min())
SHARE_LOW_BIN = int(SHARE_MEAN.argmin())

# The other ops: the contexts where the examples point partly at `difference`.
OTHER_BINS = (0.0, 0.05, 0.2, 0.5, 1.0)
OTHER_LABEL = ["< 0.05", "0.05–0.2", "0.2–0.5", "≥ 0.5"]
OBIN = np.clip(np.digitize(POST_D, OTHER_BINS[1:-1]), 0, len(OTHER_BINS) - 2)
N_OBIN = [int(((OBIN == b) & ~IS_D).sum()) for b in range(len(OTHER_LABEL))]
GAIN_NULL = np.array([-DEP_BAYES[(OBIN == b) & ~IS_D].mean() for b in range(len(OTHER_LABEL))])
GAIN_EDIT = {s: np.array([-NET[s][(OBIN == b) & ~IS_D].mean() for b in range(len(OTHER_LABEL))]) for s in SEEDS}
GAIN_EDIT_MEAN = np.mean([GAIN_EDIT[s] for s in SEEDS], axis=0)


def raw_gain(seed: int) -> np.ndarray:
    r = DOSE[label_of(FALLOFF, seed)]
    g = r["eem"][-1] - r["clean_eem"]
    return np.array([g[(OBIN == b) & ~IS_D].mean() for b in range(len(OTHER_LABEL))])


RAW_GAIN_MEAN = np.mean([raw_gain(s) for s in SEEDS], axis=0)


@memo
def grading_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="edit-effects-grading",
        alt_text=alt_text,
        caption=f"""
            **The drop under the full edit, by how much each context needed `{ex.ANCHORED_OP}`.** **Left:** on
            `{ex.ANCHORED_OP}` contexts, the share of the way to the target null that the edit takes each bin of
            the posterior on `{ex.ANCHORED_OP}`, as a ratio of bin means; one hairline per run and the seed mean
            bold. **Right:** on the contexts of the other ops, the change in expected exact match under the edit
            (up is a gain), against the change the target null predicts, one point per bin of the posterior on
            `{ex.ANCHORED_OP}` and run. The diagonal is where the edit would match the target null. Both net of
            the control.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), layout="constrained")
        axes = cast(AxesRow, axes)
        ax = axes[0]
        x = np.arange(len(data["bins"]))
        for s in data["share"]:
            ax.plot(x, s, color=INK[FALLOFF], lw=0.5, alpha=0.6)
        ax.plot(x, np.mean(data["share"], axis=0), color=INK[FALLOFF], lw=1.6, marker="D", ms=4)
        ax.axhline(1, color=RULE, lw=0.7, ls="--", zorder=0)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x, data["bins"], fontsize=7)
        ax.set_xlabel(f"posterior on {ex.ANCHORED_OP}", fontsize=8)
        ax.set_ylabel("share of the way to the null", fontsize=8)
        ax.tick_params(labelsize=7)
        ax = axes[1]
        lim = max(float(np.max(data["null"])), float(np.max(data["edit"]))) * 1.1
        ax.plot([0, lim], [0, lim], color=RULE, lw=0.7, ls="--", zorder=0)
        marks = ["o", "s", "^", "D"]
        for b, m in enumerate(marks):
            ys = [g[b] for g in data["edit"]]
            ax.plot([data["null"][b]] * len(ys), ys, ls="none", marker=m, ms=3.5, color=INK[FALLOFF], alpha=0.6)
            ax.annotate(
                data["obins"][b], (data["null"][b], max(ys)), xytext=(3, 3), textcoords="offset points", fontsize=6.5
            )
        ax.set_xlim(-0.02, lim)
        ax.set_ylim(-0.02, lim)
        ax.set_xlabel("gain under the target null", fontsize=8)
        ax.set_ylabel("gain under the edit", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig

    return _plot()


def grading_figure() -> str:
    data = {
        "bins": BIN_LABEL,
        "share": [SHARE[s].tolist() for s in SEEDS],
        "obins": OTHER_LABEL,
        "null": GAIN_NULL.tolist(),
        "edit": [GAIN_EDIT[s].tolist() for s in SEEDS],
    }
    alt = (
        f"Left: share removed against five bins of the posterior on {ex.ANCHORED_OP}, eight thin lines and a bold "
        f"mean, all between about {SHARE_MEAN.min():.1f} and {SHARE_MEAN.max():.1f}, roughly level. Right: gain "
        "under the edit against gain under the target null for the other ops, four clusters of points that rise "
        "with the diagonal: the two lowest sit on or a little above it, and the highest, at about 0.33 on the "
        "x axis, sits about a third under it."
    )
    return grading_draw(data, alt)


def grading_table() -> str:
    rows = [["contexts", *[f"{n:,}" for n in N_BIN_D], f"{int(IS_D.sum()):,}"]]
    rows.append(
        [
            "dependence (Bayes)",
            *[num(DEP_BAYES[(BIN == b) & IS_D].mean()) for b in range(5)],
            num(DEP_BAYES[IS_D].mean()),
        ]
    )
    rows.append(["share removed", *[span([SHARE[s][b] for s in SEEDS]) for b in range(5)], span(SHARE_ALL.values())])
    rows.append(
        [
            "not net of the control",
            *[span([RAW_SHARE[s][b] for s in SEEDS]) for b in range(5)],
            span(RAW_SHARE_ALL.values()),
        ]
    )
    return table_html(
        [f"posterior on `{ex.ANCHORED_OP}`", *BIN_LABEL, "all"],
        rows,
        f"""**The share of the dependence removed, by bin of the posterior**, on the held-out `{ex.ANCHORED_OP}`
        contexts: the mean net drop under the full edit over the mean distance from the clean answer to the target
        null, in expected exact match; seed mean and range over {len(SEEDS)} runs. The dependence row is the drop
        the target null predicts for the Bayes predictor, which needs no model.""",
    )


# --- E3: the landing -----------------------------------------------------------------------------------------


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


PAIRS_D = d0["query_pair"][IS_D].astype(np.int64)
SUP = np.stack([support(o, PAIRS_D) for o in range(len(OPS))])  # (ops, n, 216)
OTHER_SUP = SUP[[o for o in range(len(OPS)) if o != D]].any(0)
CATS = {
    "only `difference`": SUP[D] & ~OTHER_SUP,
    "another op": OTHER_SUP,
    "no op": ~SUP.any(0),
}
QN_D = BAYES["q_null"][IS_D]
QI_D = BAYES["q_ideal"][IS_D]


def mass(p: np.ndarray) -> list[float]:
    return [float((p * m).sum(1).mean()) for m in CATS.values()] + [float(np.clip(1 - p.sum(1), 0, None).mean())]


@memo
def nearest_mixture(p: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Per context, the mixture of the ops' answer distributions closest to *p* (non-negative weights, least
    squares over the palette): `(n, 216)`.
    """
    A = answer_dists(pairs)
    return np.stack([A[i] @ nnls(A[i], p[i])[0] for i in range(len(p))])


# A predictor that has lost `difference` and ignores the examples: the other ops, equally weighted.
_uniform = np.where(np.arange(len(OPS)) == D, 0.0, 1.0 / (len(OPS) - 1))
Q_UNIFORM = P.predictive(
    TABLE, contexts_of({"op_ids": OP_IDS[IS_D], "query_pair": PAIRS_D}), np.tile(_uniform, (int(IS_D.sum()), 1))
)
LAND: dict[int, dict[str, Any]] = {}
for s in SEEDS:
    r, c = DOSE[label_of(FALLOFF, s)], DOSE[label_of(CONTROL, s)]
    pc, pf = r["p_clean"][IS_D].astype(np.float64), r["p_full"][IS_D].astype(np.float64)
    mix_f, mix_c = nearest_mixture(pf, PAIRS_D), nearest_mixture(pc, PAIRS_D)
    LAND[s] = {
        "landing": 1 - tv(QN_D, pf).mean() / tv(QN_D, pc).mean(),
        "left": tv(QN_D, pf).mean(),
        "clean_to_null": tv(QN_D, pc).mean(),
        "clean_to_bayes": tv(QI_D, pc).mean(),
        "control_to_bayes": tv(QI_D, c["p_clean"][IS_D].astype(np.float64)).mean(),
        "other_clean_to_bayes": tv(BAYES["q_ideal"][~IS_D], r["p_clean"][~IS_D].astype(np.float64)).mean(),
        "off_mixture": (0.5 * np.abs(pf - mix_f).sum(1)).mean(),
        "clean_off_mixture": (0.5 * np.abs(pc - mix_c).sum(1)).mean(),
        "uniform_left": tv(Q_UNIFORM, pf).mean(),
        "mixture_to_null": tv(QN_D, mix_f).mean(),
        "mass": {"null": mass(QN_D), "clean": mass(pc), "edit": mass(pf)},
        "left_by_cat": [float((0.5 * np.abs(QN_D - pf) * m).sum(1).mean()) for m in CATS.values()],
        "op_mass_edit": [float((pf * SUP[o]).sum(1).mean()) for o in range(len(OPS))],
    }

OP_MASS_NULL = [float((QN_D * SUP[o]).sum(1).mean()) for o in range(len(OPS))]


def land(key: str) -> list[float]:
    return [float(LAND[s][key]) for s in SEEDS]


def mass_table() -> str:
    heads = [*CATS, "off the palette"]
    rows = []
    for name, key in (("target null", "null"), ("clean pass", "clean"), ("full edit", "edit")):
        rows.append([name, *[num(np.mean([LAND[s]["mass"][key][i] for s in SEEDS])) for i in range(len(heads))]])
    rows.append(["distance left", *[num(np.mean([LAND[s]["left_by_cat"][i] for s in SEEDS])) for i in range(3)], "—"])
    return table_html(
        ["answer mass on colors that…", *heads],
        rows,
        f"""**Where the answers put their mass on `{ex.ANCHORED_OP}` contexts**, by the kind of color: one only
        `{ex.ANCHORED_OP}` gives on the query pair, one another op gives (some of these `{ex.ANCHORED_OP}` gives
        too), one no op gives, or a token that is not a color. A color counts as given when its probability under
        the op is above zero. The last row splits the distance left between the full edit and the target null
        (total variation) the same way. Seed mean over {len(SEEDS)} runs.""",
    )


@memo
def landing_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="edit-effects-landing",
        alt_text=alt_text,
        caption=f"""
            **How far the answers sit from where they should be**, as total variation on the held-out
            `{ex.ANCHORED_OP}` contexts. One column per comparison, the {len(SEEDS)} seeds as faded dots and the seed
            mean on top. The first two columns are the yardstick: the clean model against the Bayes predictor it
            learned to match, and the control against the same. The third is the edited model against the target
            null, and the last two are the legs of the path through the nearest mixture: the edited answers against
            the nearest mixture of the ops' answers, and that mixture against the target null.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
        rng = np.random.default_rng(0)
        for i, (_, vals) in enumerate(data["cols"]):
            v = np.asarray(vals)
            ax.plot([i, i], [v.min(), v.max()], color=GHOST, lw=3, solid_capstyle="round", zorder=1)
            ax.plot(
                i + rng.uniform(-0.12, 0.12, len(v)),
                v,
                ls="none",
                marker="o",
                ms=3,
                color=INK[FALLOFF],
                alpha=0.4,
                zorder=2,
            )
            ax.plot(i, v.mean(), ls="none", marker="D", ms=6, color=INK[FALLOFF], zorder=3)
        ax.set_xticks(range(len(data["cols"])), [n for n, _ in data["cols"]], fontsize=7)
        ax.set_ylim(0, 0.5)
        ax.set_ylabel("total variation", fontsize=8)
        ax.axvline(2.5 - 1, color=RULE, lw=0.5, ls=":")
        ax.tick_params(labelsize=7)
        return fig

    return _plot()


def landing_figure() -> str:
    cols = [
        ("clean\nvs Bayes", land("clean_to_bayes")),
        ("control\nvs Bayes", land("control_to_bayes")),
        ("edited\nvs null", land("left")),
        ("edited vs\nnearest mixture", land("off_mixture")),
        ("mixture\nvs null", land("mixture_to_null")),
    ]
    alt = (
        "Five columns of seed dots. The clean model and the control sit about "
        f"{np.mean(land('clean_to_bayes')):.2f} from the Bayes predictor. The edited model sits about "
        f"{np.mean(land('left')):.2f} from the target null. The edited answers sit about {np.mean(land('off_mixture')):.2f} "
        f"from the nearest mixture of the ops, and that mixture about {np.mean(land('mixture_to_null')):.2f} from the target null."
    )
    return landing_draw({"cols": cols}, alt)


def op_mass_table() -> str:
    others = [o for o in range(len(OPS)) if o != D]
    rows = [
        ["target null", *[num(OP_MASS_NULL[o]) for o in others]],
        ["full edit", *[span([LAND[s]["op_mass_edit"][o] for s in SEEDS]) for o in others]],
    ]
    return table_html(
        ["mass on the answers of…", *[f"`{OPS[o]}`" for o in others]],
        rows,
        f"""**The answer mass on each other op**, on `{ex.ANCHORED_OP}` contexts: the mass on the colors that op
        gives on the query pair, averaged over contexts; the full edit as seed mean and range. The colors of two
        ops can coincide, so a row sums to more than one.""",
    )


# %%

r"""
# What the edit does on the recipe of record

/// tip |
<!-- lede -->
We looked at what the edit does on the recipe of record, at three scales: how far it turns the states of the other ops, how much each `difference` context loses, and where the edited answers go. The turn stays within the bound the clean geometry sets, and the loss on each context follows how much it needed `difference`. The edited answers go to the other ops in about the right overall proportions, but context by context they weigh those ops differently from the target null, and that is most of the distance left.
///

The [D2.2 design doc](/docs/m2/d2.2/design.md) lists three claims that the recipe of record has no measurement for yet: the write bound of claim (d), the grading context by context of claim (b), and where the edited model lands, claim (c). None needs new training. This page measures all three on the eight `falloff` runs of [split-anti-seeds](/docs/m2/split-anti-seeds/report.py), from the dose scoring that experiment stored and one new pass over the held-out contexts that records the alignment with e₁ at every slice and position, clean and under the edit.
"""

# %%

rf"""
## Observations

- [How far the edit turns the other ops (E1)](#how-far-the-edit-turns-the-other-ops-e1): the write stays within the bound at every slice past the embedding, at most {RATIO_MAX:.2f} times it (seed mean), and the blocks after an edit do not put e₁ back on the other ops. The bound is widest at the answers, and at the example answers most of all at block 1. Example answers of other ops that fit `difference` sit on e₁ and are turned most, as the evidence they are.
- [Whether the drop follows what each context needed (E2)](#whether-the-drop-follows-what-each-context-needed-e2): the edit takes about the same share of the way to the target null on contexts that barely needed `difference` as on those that needed it most. On the contexts of other ops whose examples point at `difference`, the edit gives back some of what those examples cost, as the target null says it should.
- [Where the edited answers fall short (E3)](#where-the-edited-answers-fall-short-e3): almost all the mass leaves the answers only `difference` gives and lands on the answers of the other ops, in about the right overall proportions. What is left is mostly how the edited model weighs those ops context by context. The distance left is about two and a half times the distance between the clean model and the Bayes predictor it was trained to match.

## Scope

This is an exploratory re-analysis, with no preregistration and no gate. It covers the `falloff` arm of split-anti-seeds (the recipe of record) at model seeds 704 to 711, and the control at each of those seeds from ex-2.2.23, trained without the anchor at the same initialization, batches and label draws. Every measurement is on the {len(OP_IDS):,} held-out contexts of ex-2.2.21, {int(IS_D.sum()):,} per op, each three examples and a query.

The edit is the full edit of the recipe: the whole e₁ component taken out of the state at every position and every slice. E2 and E3 use the full dose only; split-anti-seeds already showed that the removal grows with the dose. With eight runs, a pattern seen on every seed is one we can lean on; a difference between seeds is noted and not explained. E3 confirms on the recipe of record what ex-2.2.22 found informally on an earlier recipe, and takes it one step further.

## The measurements

- The turn, or *write*: the angle by which the edit turns a state. The states are unit vectors and the edit renormalizes, so the turn is arcsin of the alignment α the state arrives with. At the embedding that is the clean α; at a later slice it is the α left after the blocks have read the edited states of the slices before.
- Whether an example *fits* `difference`: its answer is one that `difference` gives on its operands. In a context of another op, a fitting example is usually one where the replacement op noise showed the answer of `difference`.
- The *bound* at a slice and position: arcsin of the 99th percentile of the clean |α| over the contexts of the six other ops, as in H4 of [ex-2.2.1](/docs/m2/ex-2.2.1/report.py), set before any edit from the clean geometry. At the example answers it counts only the answers that do not fit `difference`: the ones that fit are evidence for the op, and the anchor is meant to put them on e₁ (the [example-evidence report](/docs/m2/example-evidence/report.py) found it does). The write stays within the bound when its 99th percentile over the same states is no larger. Ex-2.2.1 allowed a quarter over (a ratio of {TOLERANCE}) as a tolerance; we quote the ratio.
- *Dependence*: how much a context needed `difference`, as the expected exact match of the clean model less that of the target null on the same context. It is the most the edit could take off without going past the target null.
- The *drop* on a context: its expected exact match on the clean pass less that under the edit, net of the drop the same edit causes on the control at the same context. The share removed in a group of contexts is the mean drop over the mean dependence. The control loses more on `difference` contexts under the same edit than on any other op ({CONTROL_DROP_D:.2f} in expected exact match), so the net share is the part of the removal beyond what taking out an arbitrary axis costs, and sits below the share the edit takes on its own.
- The *landing* and the distance left: the total variation distance between the edited answers and the target null, on `difference` contexts, and the share of the clean distance that the edit covers, as a ratio of the means over those contexts, the number split-anti-seeds quotes.
"""

# %%

rf"""
## How far the edit turns the other ops (E1)

The anti-subspace term keeps the states of other ops off e₁, so we expect the edit to turn them only a little: most at block 1, where split-anti-seeds found the other ops sit closest to e₁ at the end of training, and less at the pulled blocks, where they sit lower. The question is whether the turn under the edit stays within the bound the clean geometry sets, at every slice and role, or whether a later block puts e₁ back onto the other ops after the edit has taken it out upstream.
<!-- REVIEW: the first draft of the plan gave "where the anti weight is low but the pull has room" as the reason for the smallest turn at the pulled blocks; a low hold leaves more room on e₁, not less, so the reason now cites where the other ops were measured to sit. Verify: the alignment-by-slice figure of split-anti-seeds (H2), right panel. -->
<!-- REVIEW: the bound at the example answers counts only the answers that do not fit `difference` (assumptions review). About one example answer in seventeen in another op's context fits, enough to set the 99th percentile there, and those states are evidence the anchor is meant to place. Verify: the fitting rows of the table below. -->

The figure shows the bound and the write at every position of a context, slice by slice, with the control behind.

{turn_figure()}

The write stays within the bound past the embedding, give or take the tolerance of ex-2.2.1. At the seed mean the largest ratio of write to bound is {RATIO_MAX:.2f}, at block {RATIO_WORST[0] + 1}, position `{POS_LABEL[RATIO_WORST[1]]}` of the {["first", "second", "third", "query"][RATIO_WORST[1] // UNIT]} equation. Seed by seed, a few sites go past the tolerance of ex-2.2.1 ({len(OVER)} of the {len(SEEDS) * 4 * N_POS} sites past the embedding), all where the bound itself is tiny, at most {OVER_BOUND_MAX:.0f}°. At block 1 the write is about the bound, since only the embedding has been edited before it, and the embedding holds little. From block 2 on it falls well below: at the example answers that do not fit, block 2 turns them by {ANS_RATIO[2]:.0%} of their bound and block 4 by {ANS_RATIO[4]:.0%}. So the blocks after an edit do not put e₁ back on the other ops; they leave less there than the clean model had.

The bound is widest at the answers. At the example answers it is widest at block 1, as we expected: there the answers of the other ops come partway onto e₁, about twice as far as in the control. At the query answer it stays wide with depth, from {angle(BOUND[1, QUERY_ANSWER]):.0f}° at block 1 to {angle(BOUND[4, QUERY_ANSWER]):.0f}° at block 4, while at the example answers it narrows. That comes from a small tail: a few contexts of another op in a hundred have their query answer close to e₁ at blocks 2 to 4, while the median stays a few degrees. We found nothing in the examples that sets those contexts apart (their posterior on `difference` is about as low as in the rest), so this is unexplained.
<!-- REVIEW: "about one in a hundred" and "no higher" were sized from the 99th percentile alone; counting |α| > 0.5 at the query answer of the other ops gives 4% at block 2 and 2% at blocks 3 and 4, and the posterior on `difference` in that tail is 0.05 against 0.04 in the rest (claims review). -->
 The edit takes them off before they get there. At the other roles the bound is a few degrees at every slice, smaller than in the control at the colors and about the same at the syntax.

{turn_table()}

The table puts the example answers in degrees. On the clean pass the answers that fit `difference` climb onto e₁ with depth, to within a few degrees of it by block 3. Under the edit they never get there: block 1 turns them by about a quarter of a right angle, and each later block finds them far less aligned than the clean model had them, so the writes after block 1 are small. Turning these states is the edit doing its job: they are the evidence for `difference` in a context that otherwise points elsewhere, and E2 finds that removing them helps those contexts.

So on the recipe of record the bound holds in the sense of ex-2.2.1. The anchor gives a turn we can state in advance from the clean map, and nothing downstream adds to it. What the bound does not cover is what the later blocks make of a state that was turned at block 1, which is what the spill measures; at block 1 the other ops sit closer to e₁ than in a model with no anchor, so that is where any spill would start.
"""

# %%

rf"""
## Whether the drop follows what each context needed (E2)

The target null predicts the drop context by context: a context whose examples all fit `difference` should lose most, and one whose examples point elsewhere should lose little. If the anchor holds the inferred op, the edit should take about the same share of the way to the target null in every group of contexts, whatever their posterior on `difference`. If it holds something coarser, such as the label or a color feature of `difference` answers, it would take off about the same amount everywhere, which would be a larger share on the contexts that barely needed the op. The main measure is the share removed by bin of the posterior (the correlation of drop with the dependence of the Bayes predictor is a second, weaker sign, since the clean model already tracks the posterior).

The target null also predicts a change on the contexts of other ops: where an example showed the answer of `difference`, removing the op should move the answer back toward the true op, a gain. We look at both.

{grading_figure()}

The share removed is about level across the posterior: between {SHARE_MEAN.min():.2f} and {SHARE_MEAN.max():.2f} at the seed mean, lowest in the {BIN_LABEL[SHARE_LOW_BIN]} bin. The lowest bin is the noisiest, since its contexts needed little to begin with, and one seed there goes a little past the target null; no other bin shows the overshoot a coarse feature would give. (The drop correlates with the Bayes dependence at r = {span(R_BAYES.values())}.)

{grading_table()}

On the contexts of other ops the edit helps where their examples point at `difference`, and by more the more they point at it: in the direction the target null predicts, and most of the way. Where the posterior on `difference` is {OTHER_LABEL[3]} the edit gains {GAIN_EDIT_MEAN[3]:.2f} against {GAIN_NULL[3]:.2f} for the target null; in the two low bins it gains a little more than the target null, {GAIN_EDIT_MEAN[1]:.2f} against {GAIN_NULL[1]:.2f} at {OTHER_LABEL[1]}. Part of that excess comes from subtracting the control, which loses a little on every op under the edit, though on the contexts whose examples point at `difference` it gains a little instead, on every seed: without subtracting it, the gains are {num(RAW_GAIN_MEAN[1])} and {num(RAW_GAIN_MEAN[3])}.
<!-- REVIEW: the control's drop by bin on the other ops is +0.03 below a posterior of 0.2 and −0.03 at ≥ 0.5 (a gain), so the earlier wording, which had it losing everywhere, did not fit the raw gain of 0.26 being above the net 0.23 (claims review). -->
 Most contexts of other ops have almost no posterior on `difference` ({N_OBIN[0]:,} of {int((~IS_D).sum()):,} below 0.05), and there the target null predicts almost no change.

Without subtracting the control the share is level too, and higher: between {RAW_SHARE_MEAN.min():.2f} and {RAW_SHARE_MEAN.max():.2f} by bin. The shape of the net share across bins comes from the control, whose own drop under the edit varies by bin, so it says where an arbitrary edit costs the most rather than anything about the anchor.
<!-- REVIEW: cut "the edit is a little shy where the evidence is split" (claims review). Without the control subtraction the share by bin is level near 0.9 to 0.97; the dip in the net share is the control's own drop by bin. Verify: the "not net of the control" row of the table. -->

So the drop is graded the way claim (b) asks: in proportion to what each context needed. The gain on the other ops fits the anchor holding what the examples show about `difference`, wherever it appears, rather than what the label marks.
"""

# %%

rf"""
## Where the edited answers fall short (E3)

The full edit covers about half of the distance to the target null on `difference` contexts ({span(land("landing"))} here). The rest could be on the answers of `difference` the edit leaves behind, on colors no op gives, or on the answers of the other ops, weighed differently from the target null. Each would point somewhere different: a weak removal, damage to the answer, or a model that has lost `difference` and weighs what is left in its own way.

{mass_table()}

The edit moves nearly all the mass off the answers only `difference` gives, and onto the answers of the other ops. A little more lands on colors no op gives than on the clean pass. Nearly all of the distance left is on the answers of the other ops.

Overall, the edited model spreads that mass over the other ops about as the target null does: each op gets roughly its share, with `lighten` and `hue-hsv` a little short.

{op_mass_table()}

So the shortfall is context by context. To separate how the edited model weighs the ops from how it shapes the answer of each op, we fit each edited answer distribution with the nearest mixture of the ops' answer distributions (non-negative weights), and measure the two legs of the path through it: from the edited answers to that mixture, and from the mixture to the target null. Total variation is a distance, so the two legs can add to more than the whole.

For a yardstick, the clean model sits some distance from the Bayes predictor it learned to match, on `difference` contexts ({span(land("clean_to_bayes"))}, about as far as the control) and a little further on the other ops. That is a comparison point rather than a floor, since the two are distances between different pairs, but it says how close a model of this size gets to an ideal predictor when nothing has been edited.

{landing_figure()}

The edited answers are close to a mixture of the ops' answers ({span(land("off_mixture"))}, against {np.mean(land("clean_off_mixture")):.2f} on the clean pass), so they are mostly well formed: answers the other ops give, with about the right spread within each op. The mixture, though, sits {span(land("mixture_to_null"))} from the target null, so the edited model weighs the other ops differently from the posterior, context by context. The distance left in all is about two and a half times the yardstick.

The edited model does use the examples to weigh the other ops, partly. A predictor that has lost `difference` and weighs the other ops equally, whatever the examples show, sits further from the edited answers ({np.mean(land("uniform_left")):.2f}) than the target null does.

This fits the reading in claim (c) that the anchor gives the edit a direction and not the whole distance. The edit takes `difference` away cleanly, and the model answers with the other ops, weighed by the examples less sharply or less accurately than the target null weighs them.
"""

# %%

r"""
## Discussion

On the recipe of record, the edit behaves at all three scales the way the placed axis says it should, to a first approximation. The turn on the states of other ops is bounded by the clean map, slice by slice, and the blocks downstream of an edit leave less on e₁ than they found. The loss on each `difference` context follows how much that context needed the op, with no sign of a coarser feature, and the contexts of other ops whose examples point at `difference` recover as the target null predicts. That fits the anchor holding the evidence for the op, wherever it appears, and the edit removing that evidence.

The gap is in where the answers land. The edit sends the mass to the right ops overall, but within a context it shares it among them only partly by how well each fits the examples. One reading is that the evidence the model uses to tell the other ops apart travels partly along e₁ too, so that taking out the axis blurs it; another is that the model never needed fine weights among the other ops on `difference` contexts, since the answer there was `difference`. This page does not separate them. A model trained without `difference` contexts would show how close any model of this size comes to the target null, and so how much of the gap is the edit.

Block 1 is where the bound is widest and where the other ops sit closer to e₁ than in a model with no anchor. If the spill grows on a later recipe, the bound map at block 1 is the first place to look.
"""
