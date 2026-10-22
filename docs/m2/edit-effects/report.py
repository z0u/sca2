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


# Theme inks are functions: `light_dark` picks by the theme active when it is called, so a module-level value
# would freeze the light ink into the dark variant too.


def ink() -> str:
    return light_dark("#2a9d8f", "#7fd8c8")


def write_ink() -> str:
    return light_dark("#c0392b", "#ff8a76")


def ghost() -> str:
    return light_dark("#bbb", "#555")


def rule() -> str:
    return light_dark("#aaa", "#555")


def diff_ink() -> str:
    return light_dark("#8e6bb0", "#c4a6e3")


# --- Method: the geometry of the edit ------------------------------------------------------------------------


@memo
def geometry_draw(alt_text: str) -> str:
    @themed(
        name="edit-effects-geometry",
        alt_text=alt_text,
        caption=f"""
            **The geometry of the edit, drawn schematically from made-up points.** **Left:** a unit-length state in
            the plane of e₁ and the rest of its direction. Its alignment α is its component along e₁. The edit takes
            that component out and renormalizes, which turns the state by arcsin |α|. **Right:** the states of the
            other ops at one slice and position, clean on the outer ring and as they arrive under the edit on the
            inner ring. The clean bound is the angle that holds 99% of the clean states (solid wedge), and the write
            is the angle that holds 99% of the arriving ones (dashed wedge). The ideal is a narrow bound with the
            write inside it. Example answers that fit `{ex.ANCHORED_OP}` (hollow) sit near e₁ as evidence for it, and
            the bound at the example answers leaves them out.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="constrained")
        axes = cast(AxesRow, axes)
        arrow = dict(arrowstyle="->", color=rule(), lw=0.7)

        def frame(ax, lo: float, hi: float) -> None:
            t = np.radians(np.linspace(lo, hi, 200))
            ax.plot(np.cos(t), np.sin(t), color=rule(), lw=0.6)
            ax.annotate("", xy=(0, 1.18), xytext=(0, min(0.0, np.sin(np.radians(lo)))), arrowprops=arrow)
            ax.annotate("", xy=(1.22, 0), xytext=(-0.05, 0), arrowprops=arrow)
            ax.text(0.05, 1.15, "e₁", fontsize=8)
            ax.set_xlim(-0.1, 1.3)
            ax.set_ylim(min(-0.2, np.sin(np.radians(lo)) - 0.05), 1.25)
            ax.set_aspect("equal")
            ax.axis("off")

        ax = axes[0]
        frame(ax, -5, 95)
        ax.text(1.22, -0.12, "rest of the state", fontsize=7, ha="right", color=rule())
        th = np.radians(35)
        sx, sy = np.cos(th), np.sin(th)
        ax.annotate("", xy=(sx, sy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ink(), lw=1.4))
        ax.text(sx + 0.04, sy + 0.04, "state", fontsize=7, color=ink())
        ax.plot([sx, sx], [0, sy], color=ink(), lw=0.8, ls=":")
        ax.text(sx + 0.04, sy / 2, "α", fontsize=8, color=ink())
        ax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=write_ink(), lw=1.4))
        ax.text(0.9, 0.04, "edited", fontsize=7, color=write_ink(), ha="center")
        a = np.linspace(0, th, 50)
        ax.plot(0.32 * np.cos(a), 0.32 * np.sin(a), color=write_ink(), lw=1.0)
        ax.text(0.36, 0.08, "turn = arcsin |α|", fontsize=7, color=write_ink())

        ax = axes[1]
        frame(ax, -40, 95)
        rng = np.random.default_rng(3)
        clean = rng.normal(0, 7, 60)
        arriving = clean * 0.4
        fitting = rng.uniform(70, 88, 6)
        for deg, r, marker, color, fill in (
            (clean, 1.0, "o", ink(), True),
            (arriving, 0.72, "v", write_ink(), True),
            (fitting, 1.0, "o", ghost(), False),
        ):
            t = np.radians(deg)
            ax.plot(
                r * np.cos(t),
                r * np.sin(t),
                ls="none",
                marker=marker,
                ms=3,
                color=color,
                mfc=color if fill else "none",
                alpha=0.8,
            )
        for deg, r, color, ls, label in (
            (np.quantile(np.abs(clean), Q), 1.12, ink(), "-", "clean bound"),
            (np.quantile(np.abs(arriving), Q), 0.84, write_ink(), "--", "write"),
        ):
            for sign in (1, -1):
                t = np.radians(sign * deg)
                ax.plot([0, r * np.cos(t)], [0, r * np.sin(t)], color=color, lw=0.9, ls=ls)
            ax.text(r * np.cos(np.radians(deg)) + 0.02, r * np.sin(np.radians(deg)), label, fontsize=7, color=color)
        ax.text(0.3, 0.72, f"fit {ex.ANCHORED_OP}:\nevidence", fontsize=7, color=ghost())
        return fig

    return _plot()


def geometry_figure() -> str:
    alt = (
        "Left: a unit vector at 35 degrees above the horizontal, its vertical component marked alpha, and the edited "
        "vector lying on the horizontal, with an arc between them marked turn equals arcsine of alpha. Right: clean "
        "states of other ops scattered within a few degrees of the horizontal on the outer ring, inside a solid "
        "wedge marked clean bound; the same states under the edit on an inner ring, inside a narrower dashed wedge "
        "marked write; and a few hollow points near the vertical marked as evidence for difference."
    )
    return geometry_draw(alt)


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
# The largest excess over the bound at the sites past the tolerance, in degrees.
OVER_EXCESS_MAX = max(((o[3] - 1) * o[4] for o in OVER), default=0.0)
# Of the example answers of other ops that fit `difference`, the share that the true op of the context gives too.
_fit_other = ~IS_D[:, None] & FITS_D[:, : ex.K]
FIT_COINCIDE = float((_fit_other & FITS[np.arange(len(OP_IDS)), : ex.K, OP_IDS]).sum() / _fit_other.sum())


@memo
def all_answers_bound(clean: np.ndarray, op_ids: np.ndarray) -> np.ndarray:
    """The bound at the example answers of one run with every answer counted, the fitting ones too, by slice."""
    c = np.abs(clean.astype(np.float32))[:, op_ids != D]
    return np.quantile(np.concatenate([c[:, :, p] for p in ANSWERS], axis=1), Q, axis=1)


@memo
def difference_quantile(clean: np.ndarray, op_ids: np.ndarray) -> np.ndarray:
    """The 99th percentile of the clean |α| over the `difference` contexts of one run, `(slices, positions)`."""
    return np.quantile(np.abs(clean.astype(np.float32))[:, op_ids == D], Q, axis=1)


BOUND_ALL = np.mean([all_answers_bound(ALPHA[label_of(FALLOFF, s)]["clean"], OP_IDS) for s in SEEDS], axis=0)
DIFF_Q = np.mean([difference_quantile(ALPHA[label_of(FALLOFF, s)]["clean"], OP_IDS) for s in SEEDS], axis=0)


def tail_of(seed: int) -> np.ndarray:
    """The contexts of other ops whose query answer comes close to e₁ (|α| > 0.5) at a slice from 2 to 4, clean."""
    c = np.abs(ALPHA[label_of(FALLOFF, seed)]["clean"][2:, :, QUERY_ANSWER].astype(np.float32))
    return ~IS_D & ~FITS_D[:, -1] & (c.max(0) > 0.5)


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
        f"""**How far the edit turns the example answers of the other ops**, by slice. Each entry is the angle
        arcsin |α| at the 99th percentile or the median over those answers, clean → as it arrives under the full
        edit; at the 99th percentile these are the bound and the write. Seed mean over {len(SEEDS)} runs, each
        quantile taken per run and then averaged. An answer *fits* when `{ex.ANCHORED_OP}` could give it on its
        operands ({N_FIT:,} answers per run, against {N_NOFIT:,} that do not). The control has no anchor.""",
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
            contexts of the six other ops, seed mean: clean (the bound) and under the full edit (the write). The
            edit acts at every slice, so the write at a slice is the α a state arrives with there, after the block
            before it has read the states the earlier slices edited, and before the edit at this slice takes e₁ out.
            At the answers (*y*) only those that do not fit `{ex.ANCHORED_OP}` count. The control, with no anchor,
            is the gray ghost, and the dotted line is the same quantile on the `{ex.ANCHORED_OP}` contexts, for
            reference.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(5, 1, figsize=(6.4, 5.2), layout="constrained", sharex=True, sharey=True)
        axes = cast(AxesRow, axes)
        x = np.arange(N_POS)
        for row, ax in enumerate(axes):
            sl = 4 - row
            smooth_step(ax, x, data["control"][sl], color=ghost(), lw=0.8, ramp=0.6, zorder=1)
            smooth_step(ax, x, data["difference"][sl], color=diff_ink(), lw=0.6, ls=":", ramp=0.6, zorder=2)
            smooth_step(ax, x, data["bound"][sl], color=ink(), lw=1.4, ramp=0.6, zorder=3)
            ax.plot(x, data["bound"][sl], ls="none", marker="o", ms=3, color=ink(), zorder=4)
            smooth_step(ax, x, data["write"][sl], color=write_ink(), lw=1.2, ls="--", ramp=0.6, zorder=3)
            ax.plot(x, data["write"][sl], ls="none", marker="v", ms=3, color=write_ink(), zorder=4)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1])
            ax.text(1.01, 0.5, SLICE_TICK[sl], transform=ax.transAxes, va="center", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[-1].set_xticks(x, POS_LABEL, fontsize=7)
        axes[2].set_ylabel(f"|α|, {Q:.0%} quantile", fontsize=8)
        handles = [
            plt.Line2D([], [], color=ink(), marker="o", ms=3, lw=1.4, label="clean (bound)"),
            plt.Line2D([], [], color=write_ink(), marker="v", ms=3, lw=1.2, ls="--", label="under the edit (write)"),
            plt.Line2D([], [], color=ghost(), lw=0.8, label="control, clean"),
            plt.Line2D([], [], color=diff_ink(), lw=0.6, ls=":", label=f"{ex.ANCHORED_OP} contexts, clean"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


def turn_figure() -> str:
    data = {"bound": BOUND, "write": WRITE, "control": seed_mean(CONTROL, "bound"), "difference": DIFF_Q}
    alt = (
        "Five stacked panels, one per slice. The curve under the edit matches the clean one up to slice 1 and falls "
        "well below it from slice 2 on, most at the answers, where the clean curve peaks at 0.3 to 0.65. The control "
        "is higher than both at the colors. The difference contexts reach about 0.6 at the answers at slice 1 and "
        "nearly 1 from slice 2 on."
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
            `{ex.ANCHORED_OP}` contexts, the share of the way to the target null that the edit takes in each bin of
            the posterior on `{ex.ANCHORED_OP}`, as a ratio of bin means; one hairline per run and the seed mean
            bold, with the target null as the dashed rule at 1. **Right:** on the contexts of the other ops, the change in expected exact match under the edit
            (up is a gain), against the change the target null predicts, one point per run and bin of the posterior on
            `{ex.ANCHORED_OP}` (labeled; these bins differ from the left). The diagonal is where the edit would match the target null. Both net of
            the control.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), layout="constrained")
        axes = cast(AxesRow, axes)
        ax = axes[0]
        x = np.arange(len(data["bins"]))
        for s in data["share"]:
            ax.plot(x, s, color=ink(), lw=0.5, alpha=0.6)
        ax.plot(x, np.mean(data["share"], axis=0), color=ink(), lw=1.6, marker="D", ms=4)
        ax.axhline(1, color=rule(), lw=0.7, ls="--", zorder=0)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x, data["bins"], fontsize=7)
        ax.set_xlabel(f"posterior on {ex.ANCHORED_OP}", fontsize=8)
        ax.set_ylabel("share of the way to the null", fontsize=8)
        ax.tick_params(labelsize=7)
        ax = axes[1]
        lim = max(float(np.max(data["null"])), float(np.max(data["edit"]))) * 1.1
        ax.plot([0, lim], [0, lim], color=rule(), lw=0.7, ls="--", zorder=0)
        marks = ["o", "s", "^", "D"]
        for b, m in enumerate(marks):
            ys = [g[b] for g in data["edit"]]
            ax.plot([data["null"][b]] * len(ys), ys, ls="none", marker=m, ms=3.5, color=ink(), alpha=0.6)
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
        f"Left: the share removed is roughly level across the posterior on {ex.ANCHORED_OP}, with the seed mean "
        f"between {SHARE_MEAN.min():.2f} and {SHARE_MEAN.max():.2f} and the seeds spread widest in the lowest bin. "
        "Right: on the other ops the gain under the edit rises with the gain the target null predicts, on or a "
        "little above the diagonal in the two lowest bins and about a third under it in the highest."
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
        "pf": pf,
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


@memo
def reweighted(null_post: np.ndarray, op_ids: np.ndarray, pairs: np.ndarray, power: float) -> np.ndarray:
    """The target null with its weights on the other ops raised to *power* and renormalized: flatter below 1, sharper
    above.
    """
    w = null_post**power
    return P.predictive(TABLE, contexts_of({"op_ids": op_ids, "query_pair": pairs}), w / w.sum(1, keepdims=True))


POWERS = (0.5, 1.0, 2.0)
SHARPNESS = {
    k: [float(tv(reweighted(BAYES["null_post"][IS_D], OP_IDS[IS_D], PAIRS_D, k), LAND[s]["pf"]).mean()) for s in SEEDS]
    for k in POWERS
}

# The tail at the query answer (E1), against the distances of E3 on the same contexts.
TAIL = {s: tail_of(s) for s in SEEDS}
TAIL_N = [int(TAIL[s].sum()) for s in SEEDS]


def tail_stats(seed: int) -> dict[str, float]:
    r = DOSE[label_of(FALLOFF, seed)]
    tail, rest = TAIL[seed], ~IS_D & ~TAIL[seed]
    added = tv(BAYES["q_null"], r["p_full"].astype(np.float64)) - tv(BAYES["q_ideal"], r["p_clean"].astype(np.float64))
    return {
        "eem_tail": float(r["clean_eem"][tail].mean()),
        "eem_rest": float(r["clean_eem"][rest].mean()),
        "added_tail": float(added[tail].mean()),
        "added_rest": float(added[rest].mean()),
    }


TAIL_STATS = {k: float(np.mean([tail_stats(s)[k] for s in SEEDS])) for k in tail_stats(SEEDS[0])}


def land(key: str) -> list[float]:
    return [float(LAND[s][key]) for s in SEEDS]


# No figure beside this table or the next: they hold a few seed-mean proportions, and the landing figure below
# draws the distances they lead up to.
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
            **How far the answers are from where they should be**, as total variation on the held-out
            `{ex.ANCHORED_OP}` contexts. One column per comparison, the {len(SEEDS)} seeds as faded dots over a gray
            bar for their range, and the seed mean on top. The first two columns, left of the dotted rule, are the
            yardstick: the clean model against the Bayes predictor it
            learned to match, and the control against the same. The third is the edited model against the target
            null, and the last two are the legs of the path through the nearest mixture: the edited answers against
            the nearest mixture of the answers of the ops, and that mixture against the target null.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
        rng = np.random.default_rng(0)
        for i, (_, vals) in enumerate(data["cols"]):
            v = np.asarray(vals)
            ax.plot([i, i], [v.min(), v.max()], color=ghost(), lw=3, solid_capstyle="round", zorder=1)
            ax.plot(
                i + rng.uniform(-0.12, 0.12, len(v)),
                v,
                ls="none",
                marker="o",
                ms=3,
                color=ink(),
                alpha=0.4,
                zorder=2,
            )
            ax.plot(i, v.mean(), ls="none", marker="D", ms=6, color=ink(), zorder=3)
        ax.set_xticks(range(len(data["cols"])), [n for n, _ in data["cols"]], fontsize=7)
        ax.set_ylim(0, 0.5)
        ax.set_ylabel("total variation", fontsize=8)
        ax.axvline(2.5 - 1, color=rule(), lw=0.5, ls=":")
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
        f"The edited model is about {np.mean(land('left')):.2f} from the target null, more than twice as far as the "
        f"clean model and the control are from the Bayes predictor (about {np.mean(land('clean_to_bayes')):.2f}). "
        f"The nearest mixture of the ops is close to the edited answers ({np.mean(land('off_mixture')):.2f}) but "
        f"{np.mean(land('mixture_to_null')):.2f} from the target null."
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
We looked at what the edit does on the recipe of record: how far it turns the states of the other ops, how much each `difference` context loses, and where the edited answers go. The turn stays within the bound the clean geometry sets. The loss on each context follows how much it needed `difference`. The edited answers go to the other ops in about the right overall proportions, but context by context they weigh those ops differently from the target null, and that is most of the distance left.
///

The [D2.2 design doc](/docs/m2/d2.2/design.md) lists three claims that the recipe of record has no measurement for yet. None needs new training.

- (d) The side effects are bounded by construction: because we placed the axis, we can say in advance how far an edit turns each state. This is the write bound.
- (b) Removing the op is graded: the drop on each context follows how much its answer depended on `difference`, as the target null predicts context by context.
- (c) The edited model answers in a predictable way: once `difference` is removed, it answers as the target null does, weighing the other ops by how well they fit the examples.

This study measures all three on the eight `falloff` runs of [split-anti-seeds](/docs/m2/split-anti-seeds/report.py), the experiment that set the current recipe of record. It uses the dose scoring that experiment stored, plus one new pass over the held-out contexts. That pass records the alignment with e₁ at every slice and position, clean and under the edit.
"""

# %%

rf"""
## Observations

- [How far the edit turns the other ops (E1)](#how-far-the-edit-turns-the-other-ops-e1): the write stays within the clean bound at every slice past the embedding, at most {RATIO_MAX:.2f} times it (seed mean). From slice 2 on, the other ops arrive with less e₁ than the clean model gives them, and less than the control has. The bound is widest at the answers, and at the example answers it is widest at slice 1. Example answers of other ops that fit `difference` are evidence for it; they sit on e₁, and the edit turns them most.
- [Whether the drop follows what each context needed (E2)](#whether-the-drop-follows-what-each-context-needed-e2): the edit takes about the same share of the way to the target null on contexts that barely needed `difference` as on those that needed it most. Some contexts of other ops have examples that point at `difference`; there the edit gives back some of what those examples cost, as the target null says it should.
- [Where the edited answers fall short (E3)](#where-the-edited-answers-fall-short-e3): almost all the mass leaves the answers only `difference` gives. It lands on the answers of the other ops, in about the right overall proportions. What is left is mostly how the edited model weighs those ops context by context, and those weights are no flatter or sharper than the target null on the whole. The distance left is about {np.mean(land("left")) / np.mean(land("clean_to_bayes")):.1f}× the distance between the clean model and the Bayes predictor it was trained to match.

## Scope

This is an exploratory re-analysis, with no preregistration and no gate. It covers the `falloff` arm of split-anti-seeds (the recipe of record) at model seeds 704 to 711. It also covers the control at each of those seeds from ex-2.2.23, which was trained without the anchor at the same initialization, batches, and label draws.

Every measurement is on the {len(OP_IDS):,} held-out contexts of ex-2.2.21, {int(IS_D.sum()):,} per op, each with three examples and a query. The edit is the full edit of the recipe: the whole e₁ component taken out of the state at every position and every slice. E2 and E3 use the full dose only, since split-anti-seeds already showed that the removal grows with the dose.

With eight runs, a pattern seen on every seed is one we can lean on; a difference between seeds is noted and not explained. E3 confirms on the recipe of record what ex-2.2.22 found informally on an earlier recipe, and takes it one step further.

## Method

The states are unit vectors, and the edit takes the e₁ component out of each one and renormalizes. That turns each state by an angle set by its alignment, arcsin |α|, so how far the edit moves a state can be read off the clean geometry before any edit is made. The figure draws this, with the bound and the write of E1. The terms are defined in the glossary at the end, and each shows in the margin at its first use in a section.

{geometry_figure()}

In this grammar the op is not held at one token. The model infers it from the examples, and the anchor puts that evidence on e₁ at the example answers: an answer that `difference` could give comes onto e₁ in a context of any op (the [example-evidence report](/docs/m2/example-evidence/report.py) found this). In a context of another op, about one example answer in {1 / (N_FIT / (N_FIT + N_NOFIT)):.0f} [fits](term:fit) `difference`. Most of these come from the replacement op noise, which drew `difference` for that example; in {FIT_COINCIDE:.0%} the true op gives the same answer. Either way the model sees evidence for `difference` there, so turning those states is the removal at work rather than a side effect.

So the bound at the example answers counts only the answers that do not fit. Counting every answer would set it from the evidence instead: at slice 3 it would be {angle(BOUND_ALL[3]):.0f}°, against {angle(seed_mean(FALLOFF, "nofit", "q_clean")[3]):.0f}° without the fitting answers. The fitting answers get rows of their own in the E1 table.
"""

# %%

rf"""
## How far the edit turns the other ops (E1)

We measure the turn on the contexts of the other ops, since that is where a turn is a side effect. On `difference` contexts the edit is meant to turn the states, and E2 and E3 score what that does.

The anti-subspace term keeps the states of other ops off e₁, so we expect the edit to turn them only a little. We expect the turn to be largest at slice 1, where [split-anti-seeds](/docs/m2/split-anti-seeds/report.py) found the other ops closest to e₁ at the end of training. It should be smaller at the pulled slices, where they are lower.

Does the turn under the edit stay within the clean bound, at every slice and role? Or does a later block put e₁ back onto the other ops after the edit has taken it out at the slices before?
<!-- REVIEW: the first draft of the plan gave "where the anti weight is low but the pull has room" as the reason for the smallest turn at the pulled blocks; a low hold leaves more room on e₁, not less, so the reason now cites where the other ops were measured to sit. Verify: the alignment-by-slice figure of split-anti-seeds (H2), right panel. -->
<!-- REVIEW: the bound at the example answers counts only the answers that do not fit `difference` (assumptions review). The paper review found the reason too brief where the bound was defined, so the Method section now gives it in full, with the bound that counting every answer would give. Verify: the fitting rows of the table below. -->

The figure shows the bound and the write at every position of a context, slice by slice, with the control and the `difference` contexts behind for reference. The two curves part most at the answers (*y*) and a little at `=`, and the write falls further below the bound with each slice.

{turn_figure()}

The write stays within the bound past the embedding. At the seed mean, the largest ratio of write to bound is {RATIO_MAX:.2f}, at slice {RATIO_WORST[0] + 1}, position `{POS_LABEL[RATIO_WORST[1]]}` of the {["first", "second", "third", "query"][RATIO_WORST[1] // UNIT]} equation, where the bound is {angle(BOUND[1:][RATIO_WORST]):.1f}°. Run by run, {len(OVER)} of the {len(SEEDS) * 4 * N_POS} sites past the embedding go over the tolerance of ex-2.2.1, all where the bound is at most {OVER_BOUND_MAX:.0f}°, and none by more than {OVER_EXCESS_MAX:.1f}°.

At slice 1 the write is about the bound. Only the embedding has been edited before it, and the embedding holds little. From slice 2 on, the other ops arrive with much less e₁ than the clean model gives them, because each block reads states that the edit at the slices before has already turned. At the example answers that do not fit, the write is {ANS_RATIO[2]:.0%} of the bound at slice 2 and {ANS_RATIO[4]:.0%} at slice 4.

That is below the control too, which has no anchor. From slice 2 on, the write at those answers is {angle(seed_mean(FALLOFF, "nofit", "q_arriving")[2]):.0f}° to {angle(seed_mean(FALLOFF, "nofit", "q_arriving")[4]):.0f}°, against {angle(seed_mean(CONTROL, "nofit", "q_clean")[2]):.0f}° to {angle(seed_mean(CONTROL, "nofit", "q_clean")[4]):.0f}° on the clean pass of the control. So the blocks after an edit do not put e₁ back on the other ops; they leave less there than either model has without the edit.

The bound is widest at the answers of the other ops. At the example answers it is widest at slice 1, as we expected: there the answers of the other ops come partway onto e₁, about twice as far as in the control. At the query answer it stays wide with depth, from {angle(BOUND[1, QUERY_ANSWER]):.0f}° at slice 1 to {angle(BOUND[4, QUERY_ANSWER]):.0f}° at slice 4, while at the example answers it narrows.

The wide bound at the query answer comes from a small tail. At slices 2 to 4, a few in a hundred contexts of other ops have their query answer close to e₁, while the median stays at a few degrees. How many varies a lot between seeds, from {min(TAIL_N)} contexts to {max(TAIL_N):,}. We found nothing in the examples that sets those contexts apart (their posterior on `difference` is about as low as in the rest), so this is unexplained.

The tail contexts are ones the clean model already answers less well: expected exact match {TAIL_STATS["eem_tail"]:.2f} against {TAIL_STATS["eem_rest"]:.2f} in the rest, seed mean. Under the edit they end up further from the target null, but by about as much as they were already further from the Bayes predictor on the clean pass, so the edit adds about the same distance there as elsewhere ({TAIL_STATS["added_tail"]:.3f} against {TAIL_STATS["added_rest"]:.3f} in total variation). The edit takes these query answers off e₁ at the earlier slices, before they get there. At the other roles the bound is a few degrees at every slice. It is smaller than in the control at the colors, and about the same at the syntax.
<!-- REVIEW: "about one in a hundred" and "no higher" were sized from the 99th percentile alone; counting |α| > 0.5 at the query answer of the other ops gives 4% at block 2 and 2% at blocks 3 and 4, and the posterior on `difference` in that tail is 0.05 against 0.04 in the rest (claims review). -->

{turn_table()}

The table puts the example answers in degrees. On the clean pass, the answers that fit `difference` climb onto e₁ with depth, to within a few degrees of it by slice 3. Under the edit they never get there. The edit at slice 1 turns them by about a quarter of a right angle, and at each later slice they arrive far less aligned than the clean model had them, so the writes after slice 1 are small.

Turning these states is the edit doing its job (see the Method): they are the evidence for `difference` in a context that otherwise points elsewhere, and E2 finds that removing them helps the contexts of other ops whose examples point at `difference`.

So on the recipe of record the bound holds in the sense of ex-2.2.1. The anchor gives a turn we can state in advance from the clean map, and nothing downstream adds to it.

The bound does not cover what the later blocks make of a state that was turned at slice 1; that is what the spill measures. At slice 1 the other ops are closer to e₁ than in a model with no anchor, so that is where any spill would start.
"""

# %%

rf"""
## Whether the drop follows what each context needed (E2)

The target null predicts the [drop](term:drop) context by context. A context whose examples all fit `difference` should lose most, and one whose examples point elsewhere should lose little. If the anchor holds the inferred op, the edit should take about the same share of the way to the target null in every group of contexts, whatever their posterior on `difference`. The posterior is what the examples say about the op, so an anchor that holds the posterior itself predicts the same; this test treats the two as one reading.

If the anchor holds something coarser, such as the label or a color feature of `difference` answers, the edit would take off about the same amount everywhere. That would be a larger share on the contexts that barely needed the op.

The main measure is the share removed by bin of the posterior. The correlation of drop with the dependence of the Bayes predictor is a second, weaker sign, since the clean model already tracks the posterior.

The target null also predicts a change on the contexts of other ops. Where an example showed the answer of `difference`, removing the op should move the answer back toward the true op, which is a gain.

The figure shows both: the share removed on `difference` contexts (left) and the gain on the other ops (right).

{grading_figure()}

The share removed is about level across the posterior: between {SHARE_MEAN.min():.2f} and {SHARE_MEAN.max():.2f} at the seed mean, lowest in the {BIN_LABEL[SHARE_LOW_BIN]} bin. The lowest bin is the noisiest, since its contexts needed little to begin with, and two seeds there go a little past the target null. No other bin shows the overshoot a coarse feature would give. The drop correlates with the Bayes dependence at r = {span(R_BAYES.values())}.

{grading_table()}

Without subtracting the control the share is level too, and higher: between {RAW_SHARE_MEAN.min():.2f} and {RAW_SHARE_MEAN.max():.2f} by bin. So the edit on its own takes `difference` contexts nearly all the way to the target null in expected exact match. Part of that, {CONTROL_DROP_D:.2f} on average, is what the same edit costs a model with no anchor. The shape of the net share across bins comes from the control, whose own drop under the edit varies by bin, so that shape says where an arbitrary edit costs the most, rather than anything about the anchor.
<!-- REVIEW: cut "the edit is a little shy where the evidence is split" (claims review). Without the control subtraction the share by bin is level near 0.9 to 0.97; the dip in the net share is the control's own drop by bin. Verify: the "not net of the control" row of the table. -->

On the contexts of other ops, the edit helps where their examples point at `difference`, and the more they point at it, the more it helps. That is the direction the target null predicts, and the edit goes most of the way. It falls furthest short where the examples are most in doubt between `difference` and the true op.

Where the posterior on `difference` is {OTHER_LABEL[3]}, the edit gains {GAIN_EDIT_MEAN[3]:.2f} against {GAIN_NULL[3]:.2f} for the target null. In the two low bins it gains a little more than the target null, {GAIN_EDIT_MEAN[1]:.2f} against {GAIN_NULL[1]:.2f} at {OTHER_LABEL[1]}.

Part of the excess in the low bins comes from subtracting the control, which loses a little there under the edit. Where the examples point most at `difference` the control gains a little instead, on every seed, so there the net gain is the smaller one. Without subtracting the control, the gains are {num(RAW_GAIN_MEAN[1])} at {OTHER_LABEL[1]} and {num(RAW_GAIN_MEAN[3])} at {OTHER_LABEL[3]}. Most contexts of other ops have almost no posterior on `difference` ({N_OBIN[0]:,} of {int((~IS_D).sum()):,} below 0.05), and there the target null predicts almost no change.
<!-- REVIEW: the control's drop by bin on the other ops is +0.03 below a posterior of 0.2 and −0.03 at ≥ 0.5 (a gain), so the earlier wording, which had it losing everywhere, did not fit the raw gain of 0.26 being above the net 0.23 (claims review). -->

So the drop is graded the way claim (b) asks: in proportion to what each context needed. The gain on the other ops fits the anchor holding what the examples show about `difference`, wherever it appears, rather than what the label marks.
"""

# %%

rf"""
## Where the edited answers fall short (E3)

On `difference` contexts the full edit covers about half of the distance from the clean answer distributions at the query to the target null: a landing of {span(land("landing"))}. That is a different measure from the share removed in E2. E2 scores expected exact match, the probability on the right answer, which the edit takes most of the way to the null. The landing compares the whole answer distribution, including how the mass that leaves `difference` is shared among the other ops (the target null shares it by how well each op fits the examples).

Where is the rest? It could be on the answers of `difference` that the edit leaves behind, on colors no op gives, or on the answers of the other ops, weighed differently from the target null. These would point to different causes: a weak removal, damage to the answer, or a model that has lost `difference` and weighs what is left in its own way.

{mass_table()}

The edit moves nearly all the mass off the answers only `difference` gives, and onto the answers of the other ops. A little more lands on colors no op gives than on the clean pass. Nearly all of the distance left is on the answers of the other ops.

Overall, the edited model spreads that mass over the other ops about as the target null does. Each op gets roughly its share, with `lighten` and `hue-hsv` a little short.

{op_mass_table()}

So the shortfall is context by context. It could be in how the edited model weighs the ops, or in how it shapes the answer of each op. To separate the two, we fit each edited answer distribution with the nearest mixture of the answer distributions of the ops, with non-negative weights.

We measure the two legs of the path through that mixture: from the edited answers to the mixture, and from the mixture to the target null. Total variation is a distance, so the two legs can add to more than the whole.

For reference, the clean model is some distance from the Bayes predictor it learned to match. On `difference` contexts that distance is {span(land("clean_to_bayes"))}, about as far as the control, and it is a little further on the other ops. This is a comparison point rather than a floor, since the two are distances between different pairs. But it says how close a model of this size gets to an ideal predictor when nothing has been edited. The figure puts that reference beside the distance left and the two legs.

{landing_figure()}

The edited answers are close to a mixture of the answers of the ops ({span(land("off_mixture"))}, against {np.mean(land("clean_off_mixture")):.2f} on the clean pass). So they are mostly well formed: answers the other ops give, with about the right spread within each op.

The mixture, though, is {span(land("mixture_to_null"))} from the target null, so the edited model weighs the other ops differently from the posterior, context by context. In all, the distance left is about {np.mean(land("left")) / np.mean(land("clean_to_bayes")):.1f}× the reference.

The difference in weights is not a systematic one. Weighing the other ops more evenly than the target null, or more sharply, moves the prediction further from the edited answers: with the weights raised to a power of {POWERS[0]} or {POWERS[2]:.0f} and renormalized, the distance is {np.mean(SHARPNESS[POWERS[0]]):.2f} or {np.mean(SHARPNESS[POWERS[2]]):.2f}, against {np.mean(SHARPNESS[1.0]):.2f} for the target null itself. With the overall proportions about right as well, what is left looks like scatter from context to context.

The edited model does use the examples to weigh the other ops, partly. Take a predictor that has lost `difference` and weighs the other ops equally, whatever the examples show. It is further from the edited answers ({np.mean(land("uniform_left")):.2f}) than the target null is ({np.mean(land("left")):.2f}).

Claim (c) asks that the edited model answer as the target null does. This fits the reading of it in the design doc, that the anchor gives the edit a direction and not the whole distance. The edit takes `difference` away cleanly, and the model answers with the other ops. It weighs them by the examples, but less accurately than the target null weighs them.
"""

# %%

rf"""
## Discussion

On the recipe of record, the edit behaves the way the anchor says it should on all three claims, to a first approximation. The turn on the states of other ops is bounded by the clean map, slice by slice, and the blocks downstream of an edit leave less on e₁ than they found. The loss on each `difference` context follows how much that context needed the op, with no sign of an unintended, coarser feature.

The contexts of other ops whose examples point at `difference` recover as the target null predicts. That fits the anchor holding the evidence for the op, wherever it appears, and the edit removing that evidence. In this grammar the op seems to be carried that way, spread over the example answers rather than held at one token. Whether part of it also lives in the attention, in what the heads match on (their query and key spaces) or what they carry (their value space), is open; we have put it in the [backlog](/todo/science/is-the-inferred-op-held-in-attention.md).

The shortfall is in where the answers land. The edit sends the mass to the right ops overall. But within a context, it shares the mass among them only partly by how well each fits the examples.

One reading is that the evidence the model uses to tell the other ops apart also travels partly along e₁, so taking out the axis blurs it. That would fit the anchor holding the evidence rather than the op itself. Another is that the model never needed fine weights among the other ops on `difference` contexts, since the answer there was `difference`, and so it never learned to weigh them there. This study does not separate them. A model trained without `difference` contexts would show how close any model of this size comes to the target null, and so how much of the shortfall is the edit; it is in the [backlog](/todo/science/reference-model-without-the-anchored-op.md).

If the spill grows on a later recipe, the bound map at slice 1 is the first place to look, since that is where the bound is widest.

## Glossary

Write
Turn
:   How far the edit turns a state: arcsin |α| of the alignment the state arrives with, taken at the 99th percentile over a group of states. At the embedding that is the clean α. At a later slice it is the α left after the block before has read the states the edit turned at the slices below. It is taken before the edit at that slice, which leaves no e₁ behind.

Clean bound
Bound
:   From the clean pass of the anchored model, before any edit: arcsin of the 99th percentile of |α| over the states of the six other ops at a slice and position. At the answers it counts only those that do not fit `{ex.ANCHORED_OP}`. It follows H4 of [ex-2.2.1](/docs/m2/ex-2.2.1/report.py).

Within the bound
:   The write is no larger than the bound over the same states. Ex-2.2.1 allowed a quarter over (a ratio of {TOLERANCE}) as a tolerance; we quote the ratio.

Fit {{.manual}}
:   Of an example or query answer: one that `{ex.ANCHORED_OP}` can give on its operands.

Dependence
:   How much a context needed `{ex.ANCHORED_OP}`: the expected exact match of the clean model less that of the target null on the same context, so the most the edit could take off without going past the target null.

Drop {{.manual}}
:   The loss on one context under the edit: its expected exact match on the clean pass less that under the edit, net of the drop the same edit causes on the control at that context.

Share removed
Net share
:   Of a group of contexts: the mean drop over the mean dependence, so the part of the removal beyond what taking out an arbitrary axis costs. It is below the share without the control subtracted, since under the same edit the control loses more on `{ex.ANCHORED_OP}` contexts than on any other op ({CONTROL_DROP_D:.2f} in expected exact match).

Landing
:   How far the edit moves the answers toward the target null on `{ex.ANCHORED_OP}` contexts: the share of the clean distance to the target null that it covers, in total variation, as a ratio of the means over those contexts (the number split-anti-seeds quotes).

Distance left
:   The total variation distance between the edited answers and the target null, on `{ex.ANCHORED_OP}` contexts.
"""
