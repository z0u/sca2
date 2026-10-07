# title: Ex 2.2.21: the in-context grammar pilot, on the reworked recipe

# The design constants come from `experiment.py` beside this script (the directory of the script is on sys.path
# while it runs). The preview section scores the eval ex-2.2.16 published for its own anchored arms, the result
# sections read this experiment's published eval, suppression pass, and trajectories, and the method section computes
# the posterior from the seven-op table alone.
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from matplotlib.axes import Axes

from mini.vis import AxesRow, figure_html, light_dark, themed

P = ex.ex2216._POSTERIOR_MODULE
D16 = ex.ex2216.OP_NAMES.index(ex.ANCHORED_OP)

# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(
    head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset(), text_cols: int = 1
) -> str:
    """An authored result table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def rule_color() -> str:
    return light_dark("#333", "#ddd")


def fetch_jsons(*names: str) -> list[dict]:
    """Several published JSON refs, resolved in one round trip and downloaded in one batch."""
    store = project_store()
    got = store.get_refs(list(names))
    refs = [got[n] for n in names]
    assert all(r is not None for r in refs), (
        f"not published: {[n for n, r in zip(names, refs, strict=True) if r is None]}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(r, Path(tmp) / f"{i}.json") for i, r in enumerate(refs) if r is not None])
        return [json.loads(p.read_text()) for p in paths]


def stored_evals() -> tuple[dict, dict, dict, dict, dict, dict]:
    """Ex-2.2.16's eval and suppression pass and ex-2.2.19's eval, then this experiment's eval, suppression pass, and
    training trajectories, as each was published.
    """
    e16, s16, e19, e21, s21, t21 = fetch_jsons(
        ex.ex2216.EVAL_REF,
        ex.ex2216.SUPPRESSION_REF,
        ex.ex2219.EVAL_REF,
        ex.EVAL_REF,
        ex.SUPPRESSION_REF,
        ex.TRAJ_REF,
    )
    return e16, s16, e19, e21, s21, t21


EVAL16, SUPP16, EVAL19, EVAL, SUPP21, TRAJ = stored_evals()

# --- The recipe the control has to reproduce ------------------------------------------------------------------

REF19 = [r for r in EVAL19["runs"] if r["label"].startswith(ex.REGRESSION_REF + "-")]
REF19_EEM = [r["task"]["eem"]["all"] for r in REF19]
REF19_CEIL = REF19[0]["task"]["ceiling"]["all"]
REF19_FLOOR = REF19[0]["task"]["floor"]["all"]
REF19_KL = float(np.mean([r["task"]["kl"]["all"] for r in REF19]))
REF19_MEAN = float(np.mean(REF19_EEM))
REF19_SD = float(np.std(REF19_EEM, ddof=1))
REF19_SKILL = (REF19_MEAN - REF19_FLOOR) / (REF19_CEIL - REF19_FLOOR)


def seed_band(n1: int, n2: int, sd: float = REF19_SD) -> float:
    return ex.SEED_BAND_SD * sd * float(np.sqrt(1 / n1 + 1 / n2))


BAND_55 = seed_band(5, 5)
BAND_53 = seed_band(5, 3)
BAND_33 = seed_band(3, 3)

# --- The preview: ex-2.2.16's anchored arms ------------------------------------------------------------------

PREVIEW_ARMS = ("control-k3-r0.3", "anchor-whole", "anchor-hinge")
ROLE_LABELS = [
    *(t for i in "₁₂₃" for t in (f"a{i}", "?", f"b{i}", "=", f"y{i}", ",")),
    "a",
    "?",
    "b",
    "=",
    "y",
    "⏎",
]
Q_EQ = EVAL16["runs"][0]["roles"]["query ="]
Q_Q = EVAL16["runs"][0]["roles"]["query ?"]
Q_ANS = Q_EQ + 1
EX_ANS = [i for i, t in enumerate(ROLE_LABELS[:18]) if t.startswith("y")]


def runs16(arm: str) -> list[dict]:
    return [r for r in EVAL16["runs"] if r["arm"] == arm]


def alignment16(arm: str) -> np.ndarray:
    """Seed-mean alignment, ops × slices × positions, for the k = 3 contexts of one arm."""
    return np.mean([r["alignment"] for r in runs16(arm)], axis=0)


ALIGN16 = {a: alignment16(a) for a in PREVIEW_ARMS}


def last_block(arm: str, anchored: bool) -> np.ndarray:
    a = ALIGN16[arm][:, -1, :]
    return a[D16] if anchored else np.delete(a, D16, axis=0).mean(axis=0)


MARGIN16 = {
    a: float(np.mean([r["margin"]["value"] for r in runs16(a)]))
    for a in ("control-k3-r0.3", "anchor-whole", "anchor-hinge")
}
EEM16 = {
    a: float(np.mean([r["task"]["eem"]["all"] for r in runs16(a)]))
    for a in ("control-k3-r0.3", "anchor-whole", "control-mask", "anchor-mask")
}
CTRL16_EEM = float(np.mean([r["task"]["eem"]["all"] for r in runs16("control-k3-r0.3")]))
CTRL16_SKILL = float(
    np.mean(
        [
            (r["task"]["eem"]["all"] - r["task"]["floor"]["all"])
            / (r["task"]["ceiling"]["all"] - r["task"]["floor"]["all"])
            for r in runs16("control-k3-r0.3")
        ]
    )
)


def supp_mean(supp: dict, arm: str) -> dict:
    """Seed means of a suppression pass for one arm: clean and target-null scores per op, each edit, and the mean
    posterior on the anchored op in the contexts of each op (NaN where the pass did not record it).
    """
    rs = [r for r in supp["runs"] if r["arm"] == arm]
    keys = [(e["operator"], e["dose"], e["site"]) for e in rs[0]["edits"]]
    return {
        "clean": np.mean([r["clean"]["eem"] for r in rs], axis=0),
        "null": np.mean([r["null"]["eem"] for r in rs], axis=0),
        "edits": {k: np.mean([r["edits"][i]["eem"] for r in rs], axis=0) for i, k in enumerate(keys)},
        "posterior": np.mean([r.get("posterior_anchored", np.nan) for r in rs], axis=0),
    }


SUPP = {a: supp_mean(SUPP16, a) for a in PREVIEW_ARMS}


def net_drop(arm: str, key: tuple) -> np.ndarray:
    """Per op, the drop in expected exact match under an edit, less the drop the control shows under the same edit."""
    s, c = SUPP[arm], SUPP["control-k3-r0.3"]
    return (s["clean"] - s["edits"][key]) - (c["clean"] - c["edits"][key])


def to_null(arm: str, key: tuple) -> float:
    """The net drop on `difference` as a share of the way from the clean score to the target null."""
    s = SUPP[arm]
    return float(net_drop(arm, key)[D16] / (s["clean"][D16] - s["null"][D16]))


def worst_other(arm: str, key: tuple) -> float:
    return float(np.delete(net_drop(arm, key), D16).max())


QUERY_START = len(ROLE_LABELS) - 6
SLICE_NAMES = ["emb", *(f"block {i}" for i in range(1, ALIGN16["anchor-whole"].shape[1]))]
STACK_GAP = 1.1


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


def profiles(arm: str) -> dict:
    """Per slice, the alignment on `difference` contexts and the mean over the other ops."""
    a = ALIGN16[arm]
    return {
        "anchored": a[D16].tolist(),
        "other": np.delete(a, D16, axis=0).mean(axis=0).tolist(),
    }


def preview_figure() -> str:
    data = {a: profiles(a) for a in PREVIEW_ARMS}
    w = data["anchor-whole"]["anchored"]
    h = data["anchor-hinge"]
    ex_mid = [w[2][i] for i in EX_ANS]
    alt = f"""
        Three panels side by side, one per arm of ex-2.2.16 (the control, the whole-line arm, and the hinge arm). Each
        stacks five step-shaped traces, one per slice from the embedding at the bottom to the last block at the top,
        plotting the alignment with the anchored axis against the 24 positions of a three-example context: a solid
        trace for `difference` contexts and a dashed one for the mean of the other ops, with the other two arms as
        faint lines. A vertical line marks where the query starts. The control is flat near zero at every slice. On
        the whole-line arm the `difference` trace peaks on the answers: the example answers reach
        {min(ex_mid):.2f} to {max(ex_mid):.2f} at block 2, and the query answer climbs from {w[1][Q_ANS]:.2f} at
        block 1 to {w[-1][Q_ANS]:.2f} at the last block, while the query `=` stays below {max(r[Q_EQ] for r in w):.2f}
        at every slice. On the hinge arm the query `=` holds about {h["anchored"][-1][Q_EQ]:.2f} at the last block,
        and so does the dashed trace for the other ops ({h["other"][-1][Q_EQ]:.2f}). At the embedding the solid and
        dashed traces coincide: the whole-line arm lifts `?`, `,`, and the line break to about {w[0][Q_Q]:.2f}, and
        the hinge arm lifts `=` and `,` to about {h["anchored"][0][Q_EQ]:.2f}. A bar at the bottom right spans an
        alignment of 0 to 1.
    """
    return preview_draw(data, alt)


@memo
def preview_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="preview-alignment-stack",
        alt_text=alt_text,
        caption=f"""
            **Where ex-2.2.16 put the anchor.** Alignment with e₁ by position in a context of three examples, one
            trace per slice, from the embedding at the bottom to the last block at the top; seed means over three
            runs on held-out contexts. Solid: `{ex.ANCHORED_OP}` contexts. Dashed: the mean over the other ten ops.
            Faint: the `{ex.ANCHORED_OP}` trace of the other two arms. The shaded column is the query `=`, whose state
            predicts the answer; the vertical line marks the start of the query. The labels are the roles: a, b, and
            y for the operands and the answer, numbered by example and bare for the query. Each trace's baseline is a
            hairline at zero, and the bar at bottom right spans an alignment of 0 to 1.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 5.2), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        titles = {"control-k3-r0.3": "control", "anchor-whole": "whole-line", "anchor-hinge": "hinge"}
        n_slices = len(data["anchor-whole"]["anchored"])
        for ax, (arm, d) in zip(axes, data.items(), strict=True):
            stack_panel(ax, d, [od["anchored"] for other, od in data.items() if other != arm])
            ax.set_title(titles[arm], fontsize=9)
        stack_frame(axes[0], axes[-1], n_slices)
        handles = stack_handles()
        handles.append(plt.Line2D([], [], color=stack_ghost(), lw=0.6, label="the other arms"))
        fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


def stack_accent() -> str:
    return light_dark("#c0392b", "#ff8a76")


def stack_ghost() -> str:
    return light_dark("#999", "#777")


def stack_panel(ax, d: dict, ghosts: list, *, ticks: bool = True) -> None:
    """One stacked alignment panel: per slice, a hairline baseline, faint *ghosts*, the mean over the other ops
    (dashed), and the anchored op (solid), with the query `=` shaded and the start of the query marked.
    """
    ink = rule_color()
    n_slices = len(d["anchored"])
    ax.axvspan(Q_EQ - 0.5, Q_EQ + 0.5, facecolor=light_dark("#000", "#fff"), alpha=0.07, lw=0)
    ax.axvline(QUERY_START - 0.5, color=ink, lw=0.8, ls=(0, (2, 2)))
    for sl in range(n_slices):
        base = sl * STACK_GAP
        ax.axhline(base, color=ink, lw=0.3, alpha=0.5)
        for g in ghosts:
            ax.plot(*_lift(smooth_step(g[sl]), base), color=stack_ghost(), lw=0.6, alpha=0.6)
        ax.plot(*_lift(smooth_step(d["other"][sl], flat=0.8), base), "--", color=ink, lw=0.9)
        ax.plot(*_lift(smooth_step(d["anchored"][sl], flat=0.8), base), color=stack_accent(), lw=1.3)
    if ticks:
        ax.set_xticks(np.arange(len(ROLE_LABELS)), ROLE_LABELS, fontsize=6)
    ax.set_xlim(-0.6, len(ROLE_LABELS) - 0.4)
    ax.text(QUERY_START - 0.3, (n_slices - 0.05) * STACK_GAP, "query", fontsize=6, color=ink, va="top")


def stack_frame(first, last, n_slices: int) -> None:
    """The slice names on the *first* panel of a row, and the 0-to-1 scale bar beside the *last*."""
    ink = rule_color()
    bar_x = len(ROLE_LABELS) - 0.1
    last.plot([bar_x, bar_x], [0, 1], color=ink, lw=1.2, clip_on=False)
    for v in (0, 1):
        last.text(bar_x + 0.35, v, f"{v}", fontsize=6, color=ink, va="center", clip_on=False)
    first.set_yticks([sl * STACK_GAP for sl in range(n_slices)], SLICE_NAMES, fontsize=7)
    first.set_ylim(-0.15, n_slices * STACK_GAP)


def stack_handles() -> list:
    return [
        plt.Line2D([], [], color=stack_accent(), lw=1.3, label=ex.ANCHORED_OP),
        plt.Line2D([], [], color=rule_color(), lw=0.9, ls="--", label="other ops"),
    ]


def _lift(xy: tuple[np.ndarray, np.ndarray], base: float) -> tuple[np.ndarray, np.ndarray]:
    return xy[0], xy[1] + base


SUPP_SITES = ("query ?", "query =", "every position")
SUPP_EDITS = (
    *(("projection", g, f"γ {g:g}") for g in (0.25, 0.5, 0.75, 1.0)),
    ("repulsion", 0.25, "rep. 0.25"),
    ("repulsion", 0.0, "rep. 0"),
    ("reflection", 2.0, "refl."),
)


def suppression_figure() -> str:
    data: dict[str, dict] = {
        arm: {
            "half": float(ex.GRADING_MIN_DAMAGE * (SUPP[arm]["clean"][D16] - SUPP[arm]["null"][D16])),
            "sites": {
                site: {
                    "anchored": [float(net_drop(arm, (o, g, site))[D16]) for o, g, _ in SUPP_EDITS],
                    "worst": [worst_other(arm, (o, g, site)) for o, g, _ in SUPP_EDITS],
                }
                for site in SUPP_SITES
            },
        }
        for arm in ("anchor-whole", "anchor-hinge")
    }
    wf = data["anchor-whole"]["sites"]["every position"]
    he = data["anchor-hinge"]["sites"]["query ="]
    alt = f"""
        A grid of six panels: rows for the whole-line arm and the hinge arm of ex-2.2.16, columns for the three edit
        sites (the query `?`, the query `=`, and every position). Each panel plots the net drop in expected exact
        match against the edit: the projection at four doses, the repulsion at two, and the reflection, with a solid
        trace for `difference` contexts and a dashed one for the worst other op, a rule at the selectivity gate of
        {ex.SELECTIVITY_GATE:g}, and a rule halfway to the target null. At the query `?` both arms stay at zero. At
        the query `=` the whole-line arm stays at zero, and on the hinge arm the drop on `difference`
        ({min(he["anchored"][:4]):.2f} to {max(he["anchored"][:4]):.2f} under the projection) is matched by the worst
        other op. At every position the whole-line projection rises to {max(wf["anchored"][:4]):.2f}, below the
        halfway rule at {data["anchor-whole"]["half"]:.2f}, and the worst other op passes the gate from γ = 0.75.
    """
    caption = f"""
        **Ex-2.2.16's suppression pass on its two scored anchored arms.** The net drop in held-out expected exact
        match under each edit: the seed-mean drop, less the drop the control shows under the same edit. Solid:
        on `{ex.ANCHORED_OP}` contexts. Dashed: the worst of the other ten ops. The dotted rule is the selectivity
        gate ({ex.SELECTIVITY_GATE:g}), and the dash-dot rule is {ex.GRADING_MIN_DAMAGE:.0%} of the way from the
        clean score on `{ex.ANCHORED_OP}` to the target null. The projection runs from γ = 0.25 to full removal;
        the repulsion lands states at 0.25 and then 0; the reflection has one dose.
    """
    return suppression_draw(data, alt, "preview-suppression", caption, (8.4, 4.6))


@memo
def suppression_draw(data: dict, alt_text: str, name: str, caption: str, figsize: tuple[float, float]) -> str:
    """Net drop against the edit: one row per arm in *data*, one column per site in each arm's `sites`."""
    sites = list(next(iter(data.values()))["sites"])

    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(len(data), len(sites), figsize=figsize, layout="constrained", sharey=True, sharex=True)
        ink = rule_color()
        accent = light_dark("#c0392b", "#ff8a76")
        x = np.arange(len(SUPP_EDITS))
        groups = (slice(0, 4), slice(4, 6), slice(6, 7))
        for r, (arm, d) in enumerate(data.items()):
            for c, site in enumerate(sites):
                ax = axes[r, c]
                ax.axhline(ex.SELECTIVITY_GATE, color=ink, lw=0.8, ls=":")
                ax.axhline(d["half"], color=ink, lw=0.8, ls="-.")
                ax.axhline(0, color=ink, lw=0.3, alpha=0.5)
                for g in groups:
                    ax.plot(x[g], d["sites"][site]["anchored"][g], "-o", color=accent, lw=1.3, ms=3)
                    ax.plot(x[g], d["sites"][site]["worst"][g], "--o", color=ink, lw=0.9, ms=2.5, mfc="none")
                if site in d.get("meets", ()):
                    for spine in ax.spines.values():
                        spine.set(color=accent, linewidth=2.2, visible=True)
                if r == 0:
                    ax.set_title(f"at {site}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{arm_name(arm)}\nnet drop", fontsize=8)
                ax.set_xticks(x, [e[2] for e in SUPP_EDITS], fontsize=6, rotation=45, ha="right")
        handles = [
            plt.Line2D([], [], color=accent, lw=1.3, marker="o", ms=3, label=ex.ANCHORED_OP),
            plt.Line2D([], [], color=ink, lw=0.9, ls="--", marker="o", ms=2.5, mfc="none", label="worst other op"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


PV = {"whole_full": ("projection", 1.0, "every position")}
WHOLE = profiles("anchor-whole")["anchored"]
HINGE = profiles("anchor-hinge")
EX_EQ = [i for i, t in enumerate(ROLE_LABELS[:18]) if t == "="]
COMMA = ROLE_LABELS.index(",")

# --- The posterior on the seven-op table -------------------------------------------------------------------------


@memo
def tables() -> tuple:
    return P.build_table(ex.ex2216.TABLE), P.build_table(ex.ex2218.table_of(ex.OP_NAMES))


TABLE11, TABLE7 = tables()


@memo
def posterior_on_anchored(table, names: tuple, k: int, rho: float, kappa: float, n: int, seed: int) -> dict:
    """At one condition: the posterior on the anchored op over contexts of that op, and the ceiling and floor over
    contexts of every op.
    """
    d = list(names).index(ex.ANCHORED_OP)
    rng = np.random.default_rng([seed, k, round(rho * 100), 0])
    ctx = P.sample_contexts(table, n, k, rho, kappa, rng, true_op=d)
    post = P.posterior(table, ctx, rho, kappa)
    rng = np.random.default_rng([seed, k, round(rho * 100), 1])
    allc = P.sample_contexts(table, n, k, rho, kappa, rng)
    return {
        "post": post[:, d].astype(np.float32),
        "ceiling": float(P.ceiling(table, allc, rho, kappa).mean()),
        "floor": float(P.floor(table, allc).mean()),
    }


GRID_K = (2, 3, 4)
GRID_RHO = (0.2, 0.3, 0.4)
N_POST = 20_000
POST_SEED = 2221

POST7 = {
    (k, r): posterior_on_anchored(TABLE7, ex.OP_NAMES, k, r, ex.CUBE_RATE, N_POST, POST_SEED)
    for k in GRID_K
    for r in GRID_RHO
}
POST11 = posterior_on_anchored(TABLE11, ex.ex2216.OP_NAMES, *ex.CENTRE, ex.CUBE_RATE, N_POST, POST_SEED)


def band_shares(post: np.ndarray) -> tuple[float, float, float]:
    lo, hi = ex.MIDDLE_BAND
    return float((post < lo).mean()), float(((post >= lo) & (post <= hi)).mean()), float((post > hi).mean())


C7 = POST7[ex.CENTRE]
BANDS7 = band_shares(C7["post"])
BANDS11 = band_shares(POST11["post"])
STEP7 = float(((C7["post"] >= 0.85) & (C7["post"] < 0.95)).mean() / BANDS7[1])


def posterior_figure() -> str:
    alt = f"""
        A cumulative distribution of the posterior on `{ex.ANCHORED_OP}` over `{ex.ANCHORED_OP}` contexts at three
        examples and ρ = {ex.CENTRE[1]:g}, one curve for the eleven-op table and one for the seven-op table. Both
        rise across the middle band from {ex.MIDDLE_BAND[0]:g} to {ex.MIDDLE_BAND[1]:g}; on the seven-op table
        {BANDS7[1]:.0%} of contexts sit inside it, against {BANDS11[1]:.0%} on eleven ops.
    """
    return posterior_draw(C7["post"], POST11["post"], alt)


@memo
def posterior_draw(post7: np.ndarray, post11: np.ndarray, alt_text: str) -> str:
    @themed(
        name="posterior-anchored",
        alt_text=alt_text,
        caption=f"""
            **The stimulus on the seven-op table.** The share of `{ex.ANCHORED_OP}` contexts whose posterior on
            `{ex.ANCHORED_OP}` is at most the value on the x-axis, at three examples, ρ = {ex.CENTRE[1]:g}, and
            cube noise at κ = {ex.CUBE_RATE:g}. The shaded band is the middle band. {N_POST:,} sampled contexts per
            curve.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
        ink = rule_color()
        accent = light_dark("#2b6cb0", "#7fb3ff")
        lo, hi = ex.MIDDLE_BAND
        ax.axvspan(lo, hi, facecolor=light_dark("#000", "#fff"), alpha=0.06, lw=0)
        x = np.linspace(0, 1, 401)
        for post, color, label in ((post11, ink, "eleven ops"), (post7, accent, "seven ops")):
            s = np.sort(post)
            ax.plot(x, np.searchsorted(s, x, side="right") / len(s), color=color, lw=1.3, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"posterior on {ex.ANCHORED_OP}")
        ax.set_ylabel("share of contexts")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        return fig

    return _plot()


def grid_table() -> str:
    head = ["examples", *(f"ρ = {r:g}" for r in GRID_RHO)]
    rows = [
        [str(k), *(f"{POST7[(k, r)]['ceiling']:.3f} · {band_shares(POST7[(k, r)]['post'])[1]:.0%}" for r in GRID_RHO)]
        for k in GRID_K
    ]
    return table_html(
        head,
        rows,
        f"The seven-op table near the center condition, at κ = {ex.CUBE_RATE:g}: the Bayes ceiling in expected exact "
        f"match over contexts of every op, then the share of `{ex.ANCHORED_OP}` contexts in the middle band of the "
        f"posterior on `{ex.ANCHORED_OP}`.",
    )


# --- The design tables --------------------------------------------------------------------------------------


def label_cell(a: ex.Arm) -> str:
    """The label an arm pulls with; `no-emb` is the whole-line label with the embedding slice left out."""
    if not a.anchored:
        return "—"
    if a.label == "no-emb":
        return "`whole` + no-emb"
    return f"`{a.label}`" + (" + hinge" if a.hinge else "")


def arms_table() -> str:
    head = ["arm", "group", "anchored", "label", "corpus", "what it asks", "seeds"]
    rows = [
        [
            f"`{a.name}`",
            a.group,
            "yes" if a.anchored else "—",
            label_cell(a),
            "verification" if a.verify else "—",
            a.note,
            str(a.seeds),
        ]
        for a in ex.ARMS
    ]
    return table_html(
        head,
        rows,
        f"The {len(ex.ARMS)} arms, {ex.N_RUNS} runs. Every arm trains on the seven-op set at three examples and "
        f"ρ = {ex.CENTRE[1]:g}, with the newline mask, for {ex.EPOCHS} epochs; every anchored arm anchors "
        f"`{ex.ANCHORED_OP}` on e₁. `{ex.CONTROL}` and `{ex.PRIMARY}` are the references.",
        ref_rows=frozenset({i for i, a in enumerate(ex.ARMS) if a.name in (ex.CONTROL, ex.PRIMARY)}),
        text_cols=6,
    )


# --- This experiment: shared helpers ------------------------------------------------------------------------------

OPS: tuple[str, ...] = tuple(EVAL["runs"][0]["ops"])
D = OPS.index(ex.ANCHORED_OP)
ANCHORED_ARMS = [a.name for a in ex.ARMS if a.anchored and not a.verify]
E2_ARMS = [*ex.SITE_ARMS, *ex.SITE_REFERENCES]
SLOW_SEED = 2  # model seed 702, the third seed of every arm


def arm_name(arm: str) -> str:
    """The short name a figure or table gives an arm."""
    if arm == ex.PRIMARY:
        return "whole-line"
    return arm if arm.endswith("verify") else arm.removeprefix("anchor-")


def runs(arm: str) -> list[dict]:
    return [r for r in EVAL["runs"] if r["arm"] == arm]


def eem(r: dict) -> float:
    return r["task"]["eem"]["all"]


def skill(r: dict) -> float:
    t = r["task"]
    return (t["eem"]["all"] - t["floor"]["all"]) / (t["ceiling"]["all"] - t["floor"]["all"])


def per_seed(arm: str, f=eem) -> np.ndarray:
    return np.array([f(r) for r in runs(arm)])


def band(x: np.ndarray, y: np.ndarray) -> float:
    """The seed band between two arms, with the seed standard deviation pooled over the two."""
    n1, n2 = len(x), len(y)
    var = ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2)
    return float(ex.SEED_BAND_SD * np.sqrt(var) * np.sqrt(1 / n1 + 1 / n2))


def mean(arm: str, f=eem) -> float:
    return float(per_seed(arm, f).mean())


def gate_line(ax: Axes, y: float, *, fail: str, xs: tuple[float, float] | None = None, ls: str = "--") -> None:
    """A gate rule with its failing side hatched; across the panel, or over the x-range *xs* only."""
    ink = rule_color()
    lo, hi = ax.get_ylim()
    span = (lo, y) if fail == "below" else (y, hi)
    edge = light_dark("#000", "#fff")
    if xs is None:
        ax.axhline(y, color=ink, lw=0.9, ls=ls, zorder=2)
        ax.axhspan(*span, facecolor="none", edgecolor=edge, hatch="//", lw=0, zorder=0, alpha=0.1)
    else:
        ax.plot(xs, [y, y], color=ink, lw=0.9, ls=ls, zorder=2)
        ax.fill_between(xs, *span, facecolor="none", edgecolor=edge, hatch="//", lw=0, zorder=0, alpha=0.1)
    ax.set_ylim(lo, hi)


def dots(ax: Axes, x: float, v, color, *, rng, ms: float = 5.0) -> None:
    """One column of per-seed dots with the seed mean on top; a thin bar behind spans the seed range."""
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.08, 0.08, len(v)), v, "o", ms=2.6, color=color, alpha=0.55, zorder=3, mew=0)
    ax.plot(x, v.mean(), "o", ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def arm_color(arm: str) -> str:
    if arm in (ex.CONTROL, "control-verify", "ex-2.2.19"):
        return light_dark("#555", "#bbb")
    return light_dark("#c0392b", "#ff8a76")


def span(values, fmt: str = ".3f") -> str:
    return f"{min(values):{fmt}} to {max(values):{fmt}}"


def bold_if(ok, text: str) -> str:
    return f"<b>{text}</b>" if bool(ok) else text


# --- H1 ----------------------------------------------------------------------------------------------------------

CTRL_EEM = per_seed(ex.CONTROL)
WHOLE_EEM = per_seed(ex.PRIMARY)
H1: dict[str, Any] = {
    "ctrl": float(CTRL_EEM.mean()),
    "ctrl_sd": float(np.std(CTRL_EEM, ddof=1)),
    "whole": float(WHOLE_EEM.mean()),
    "a_diff": float(abs(CTRL_EEM.mean() - REF19_MEAN)),
    "b_short": float(CTRL_EEM.mean() - WHOLE_EEM.mean()),
    "b_band": band(CTRL_EEM, WHOLE_EEM),
}
H1["a_holds"] = H1["a_diff"] <= ex.REGRESSION_TOL
H1["b_holds"] = H1["b_short"] <= ex.TASK_COST_TOL
KL = {arm: per_seed(arm, lambda r: r["task"]["kl"]["all"]) for arm in (ex.CONTROL, ex.PRIMARY)}
SKILL = {arm: mean(arm, skill) for arm in (ex.CONTROL, ex.PRIMARY)}
REF19_KLS = [r["task"]["kl"]["all"] for r in REF19]


def h1_figure() -> str:
    data = {
        "eem": {"ex-2.2.19": REF19_EEM, ex.CONTROL: CTRL_EEM.tolist(), ex.PRIMARY: WHOLE_EEM.tolist()},
        "kl": {"ex-2.2.19": REF19_KLS, ex.CONTROL: KL[ex.CONTROL].tolist(), ex.PRIMARY: KL[ex.PRIMARY].tolist()},
        "ceiling": REF19_CEIL,
        "floor": REF19_FLOOR,
        "ref": REF19_MEAN,
        "ctrl": H1["ctrl"],
        "band": H1["b_band"],
    }
    alt = f"""
        Two panels. Left: held-out expected exact match per seed in three columns, ex-2.2.19's four runs, the
        control's five, and the whole-line arm's five, each with its seed mean as a larger dot. A shaded band of
        ±{ex.REGRESSION_TOL:g} around ex-2.2.19's mean of {REF19_MEAN:.3f} contains the control's mean of
        {H1["ctrl"]:.3f}. Over the whole-line column a dashed gate at {H1["ctrl"] - ex.TASK_COST_TOL:.3f}, hatched
        below, sits above its mean of {H1["whole"]:.3f}, and a dotted mark at the seed band,
        {H1["ctrl"] - H1["b_band"]:.3f}, sits below it. Whole-line seeds run from {WHOLE_EEM.min():.3f} to
        {WHOLE_EEM.max():.3f}, the control from {CTRL_EEM.min():.3f} to {CTRL_EEM.max():.3f}. The Bayes ceiling,
        {REF19_CEIL:.3f}, is a dashed rule above every run. Right: calibration KL for the same runs, seed means
        {np.mean(REF19_KLS):.2f}, {KL[ex.CONTROL].mean():.2f}, and {KL[ex.PRIMARY].mean():.2f} nats.
    """
    return h1_draw(data, alt)


@memo
def h1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="h1-recipe",
        alt_text=alt_text,
        caption=f"""
            **The control against ex-2.2.19, and the whole-line arm against the control.** Left: held-out expected
            exact match, one small dot per run and the seed mean on top; the right axis gives the same scale as skill,
            from the floor (0, the bottom of the axis) to the Bayes ceiling (1, the dashed rule at the top). The shaded band is criterion
            **(a)**, ex-2.2.19's mean ±{ex.REGRESSION_TOL:g}. Over the whole-line column, the dashed rule is criterion
            **(b)**, the control mean less {ex.TASK_COST_TOL:g}, with the failing side hatched; the dotted rule is the
            control mean less the seed band of five seeds against five. Right: calibration KL for the same runs.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained", width_ratios=[1.5, 1])
        rng = np.random.default_rng(0)
        ink = rule_color()
        cols = list(data["eem"])
        a.axhspan(data["ref"] - ex.REGRESSION_TOL, data["ref"] + ex.REGRESSION_TOL, color=ink, alpha=0.08, lw=0)
        a.axhline(data["ceiling"], color=ink, lw=0.9, ls="--")
        for i, arm in enumerate(cols):
            dots(a, i, data["eem"][arm], arm_color(arm), rng=rng)
            dots(b, i, data["kl"][arm], arm_color(arm), rng=rng)
        a.set_ylim(data["floor"], data["ceiling"] + 0.02)
        i = cols.index(ex.PRIMARY)
        a.plot([i - 0.3, i + 0.3], [data["ctrl"] - data["band"]] * 2, ":", color=ink, lw=0.9)
        gate_line(a, data["ctrl"] - ex.TASK_COST_TOL, fail="below", xs=(i - 0.3, i + 0.3))
        floor, ceil = data["floor"], data["ceiling"]
        a.secondary_yaxis(
            "right", functions=(lambda y: (y - floor) / (ceil - floor), lambda s: floor + s * (ceil - floor))
        ).set_ylabel("skill")
        a.set_ylabel("held-out EEM")
        b.set_ylabel("calibration KL (nats)")
        b.set_ylim(0, None)
        for ax in (a, b):
            ax.set_xticks(range(len(cols)), [arm_name(c) for c in cols], fontsize=8)
            ax.set_xlim(-0.6, len(cols) - 0.4)
        return fig

    return _plot()


def h1_table() -> str:
    rows = [
        [
            "<b>(a)</b> control vs. ex-2.2.19",
            f"{H1['ctrl']:.4f} vs. {REF19_MEAN:.4f}",
            f"{H1['a_diff']:.4f}",
            f"≤ {ex.REGRESSION_TOL:g}",
            "—",
            bold_if(H1["a_holds"], "holds" if H1["a_holds"] else "misses"),
        ],
        [
            "<b>(b)</b> whole-line short of control",
            f"{H1['whole']:.4f} vs. {H1['ctrl']:.4f}",
            f"{H1['b_short']:.4f}",
            f"≤ {ex.TASK_COST_TOL:g}",
            f"{H1['b_band']:.4f}",
            bold_if(H1["b_holds"], "holds" if H1["b_holds"] else "misses"),
        ],
    ]
    return table_html(
        ["criterion", "seed means", "difference ↓", "tolerance", "seed band", "verdict"],
        rows,
        "**The two criteria of H1.** Seed means of held-out expected exact match. The seed band of (b) is for five "
        "seeds against five, with the seed standard deviation pooled over the two arms; no criterion uses it.",
        text_cols=2,
    )


def traj_points(r: dict) -> tuple[list[float], list[float]]:
    t = r["traj"]
    pts = [(x, e) for x, e in zip(t["epoch"], t["eem"], strict=True) if e is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


TRAJ_GROUPS = {
    "control": [ex.CONTROL],
    "label": [a.name for a in ex.ARMS if a.group == "label"],
    "site": [a.name for a in ex.ARMS if a.group == "site"],
    "verify": [a.name for a in ex.ARMS if a.group == "verify"],
}
SLOW_LOWEST = [arm for arm in (a.name for a in ex.ARMS) if int(np.argmin(per_seed(arm))) == SLOW_SEED]


def late_rise(arm: str) -> float:
    """Over the last 20 epochs, the rise in trajectory EEM of the run with the lowest held-out EEM in an arm."""
    seed = int(np.argmin(per_seed(arm)))
    xs, ys = traj_points(next(r for r in TRAJ.values() if r["arm"] == arm and r["seed"] == seed))
    return ys[-1] - ys[next(i for i, x in enumerate(xs) if x >= ex.EPOCHS - 20)]


LOWEST_RISE = [late_rise(a.name) for a in ex.ARMS]


def traj_figure() -> str:
    data = {
        g: [{"arm": r["arm"], "seed": r["seed"], "xy": traj_points(r)} for r in TRAJ.values() if r["arm"] in arms]
        for g, arms in TRAJ_GROUPS.items()
    }
    alt = f"""
        Four panels side by side, one per group of arms (control, label, site, verify), each plotting the expected
        exact match on a held-out subset against the training epoch, 0 to {ex.EPOCHS}, one thin line per run, all
        {ex.N_RUNS} runs in all. The runs at model seed {ex.SEED_OFFSET + SLOW_SEED} are drawn in a contrasting color.
        Every run rises steeply over the first 50 epochs and then more slowly. In most arms the highlighted run lies
        below the others over the second half of training. The lowest run of each arm is still rising over the last
        20 epochs, by {span(LOWEST_RISE)}.
    """
    return traj_draw(data, alt)


@memo
def traj_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="h1-trajectories",
        alt_text=alt_text,
        caption=f"""
            **Expected exact match through training, every run (post hoc).** Measured on the first
            {ex.N_TRAJ_EEM_PER_OP} held-out contexts of each op, one line per run, panels by the group of the arm. Colored: model seed {ex.SEED_OFFSET + SLOW_SEED}, the third
            seed of every arm; grey: the other seeds.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.8), layout="constrained", sharey=True, sharex=True)
        axes = cast(AxesRow, axes)
        grey = light_dark("#aaa", "#666")
        hi = light_dark("#d97706", "#fbbf24")
        for ax, (g, rs) in zip(axes, data.items(), strict=True):
            for r in sorted(rs, key=lambda r: r["seed"] == SLOW_SEED):
                slow = r["seed"] == SLOW_SEED
                ax.plot(*r["xy"], color=hi if slow else grey, lw=0.9 if slow else 0.6, alpha=0.9 if slow else 0.7)
            ax.set_title(g, fontsize=9)
            ax.set_xlabel("epoch", fontsize=8)
            ax.set_xlim(0, ex.EPOCHS)
        axes[0].set_ylabel("held-out EEM (subset)", fontsize=8)
        handles = [
            plt.Line2D([], [], color=hi, lw=0.9, label=f"model seed {ex.SEED_OFFSET + SLOW_SEED}"),
            plt.Line2D([], [], color=grey, lw=0.6, label="other seeds"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


# --- S1 ----------------------------------------------------------------------------------------------------------

S1_ARMS = [a.name for a in ex.ARMS if a.group == "label"]
S1_CANDIDATES = ["anchor-latter", "anchor-prefix", "anchor-no-emb"]
WHOLE_MARGIN = mean(ex.PRIMARY, lambda r: r["margin"]["value"])
S1: dict[str, dict[str, Any]] = {
    arm: {
        "eem": mean(arm),
        "diff": mean(arm) - H1["whole"],
        "band": band(per_seed(arm), WHOLE_EEM),
        "margin": mean(arm, lambda r: r["margin"]["value"]),
        "share": mean(arm, lambda r: r["margin"]["value"]) / WHOLE_MARGIN,
    }
    for arm in S1_ARMS
    if arm != ex.PRIMARY
}
for v in S1.values():
    v["qualifies"] = v["diff"] > v["band"] and v["share"] >= ex.MARGIN_KEEP
S1_PROMOTED = [a for a in S1_CANDIDATES if S1[a]["qualifies"]]


def s1_figure() -> str:
    data = {
        arm: {
            "eem": per_seed(arm).tolist(),
            "share": (per_seed(arm, lambda r: r["margin"]["value"]) / WHOLE_MARGIN).tolist(),
            "gate": H1["whole"] + S1[arm]["band"] if arm in S1_CANDIDATES else None,
        }
        for arm in S1_ARMS
    }
    alt = f"""
        Two stacked panels with one column per label arm: whole-line, then
        {", ".join(arm_name(a) for a in S1_ARMS[1:])}. Top: held-out expected exact match per seed and seed mean,
        with a faint rule at the whole-line mean of {H1["whole"]:.3f} and, over each candidate column, a dashed gate
        at the whole-line mean plus the seed band, hatched below. The candidate means
        ({", ".join(f"{arm_name(a)} {S1[a]['eem']:.3f}" for a in S1_CANDIDATES)}) all sit above the whole-line mean
        and below their gates ({", ".join(f"{H1['whole'] + S1[a]['band']:.3f}" for a in S1_CANDIDATES)}). Bottom:
        the op margin as a share of the whole-line mean margin, with a dashed rule at {ex.MARGIN_KEEP:.0%} hatched
        below; seed-mean shares are {", ".join(f"{arm_name(a)} {S1[a]['share']:.2f}" for a in S1)}.
    """
    return s1_draw(data, H1["whole"], alt)


@memo
def s1_draw(data: dict, whole: float, alt_text: str) -> str:
    @themed(
        name="s1-label",
        alt_text=alt_text,
        caption=f"""
            **The label rule.** One column per label arm; small dots are runs, large dots seed means. Top: held-out
            expected exact match. The faint rule is the whole-line mean; over each candidate the dashed rule is the
            whole-line mean plus the seed band of that pair, which a candidate has to clear, with the side that does
            not qualify hatched. `sampled` is reported and not a candidate. Bottom: the op margin as a share of the
            whole-line seed-mean margin, with the {ex.MARGIN_KEEP:.0%} floor.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 4.6), layout="constrained", sharex=True)
        rng = np.random.default_rng(0)
        ink = rule_color()
        arms = list(data)
        for i, arm in enumerate(arms):
            color = light_dark("#555", "#bbb") if arm == ex.PRIMARY else arm_color(arm)
            dots(a, i, data[arm]["eem"], color, rng=rng)
            dots(b, i, data[arm]["share"], color, rng=rng)
        a.axhline(whole, color=ink, lw=0.6, alpha=0.5)
        a.set_ylim(0.4, 0.62)
        for i, arm in enumerate(arms):
            if data[arm]["gate"] is not None:
                gate_line(a, data[arm]["gate"], fail="below", xs=(i - 0.35, i + 0.35))
        b.set_ylim(0, 1.5)
        gate_line(b, ex.MARGIN_KEEP, fail="below")
        a.set_ylabel("held-out EEM", fontsize=8)
        b.set_ylabel("share of whole-line\nop margin", fontsize=8)
        b.set_xticks(range(len(arms)), [arm_name(x) for x in arms], fontsize=8)
        b.set_xlim(-0.6, len(arms) - 0.4)
        return fig

    return _plot()


def s1_table() -> str:
    rows = [
        [
            f"`{arm_name(ex.PRIMARY)}`",
            str(len(WHOLE_EEM)),
            f"{H1['whole']:.4f}",
            "—",
            "—",
            f"{WHOLE_MARGIN:.3f}",
            "1.00",
            "the primary",
        ],
        *(
            [
                f"`{arm_name(a)}`",
                str(len(runs(a))),
                f"{S1[a]['eem']:.4f}",
                f"{S1[a]['diff']:+.4f}",
                f"{S1[a]['band']:.4f}",
                f"{S1[a]['margin']:.3f}",
                bold_if(S1[a]["share"] >= ex.MARGIN_KEEP, f"{S1[a]['share']:.2f}"),
                ("yes" if S1[a]["qualifies"] else "no") if a in S1_CANDIDATES else "not a candidate",
            ]
            for a in S1_ARMS[1:]
        ),
    ]
    return table_html(
        ["arm", "seeds", "EEM ↑", "vs. whole-line", "seed band", "op margin", "share ↑", "qualifies"],
        rows,
        f"**The label rule.** Seed means. A candidate qualifies if its EEM beats the whole-line arm by more than the "
        f"seed band of the pair and its op margin is at least {ex.MARGIN_KEEP:.0%} of the whole-line margin; bold "
        f"marks a share that keeps the margin.",
    )


# --- E1 ----------------------------------------------------------------------------------------------------------


def align(arm: str) -> np.ndarray:
    """Seed-mean alignment, ops × slices × positions."""
    return np.mean([r["alignment"] for r in runs(arm)], axis=0)


ALIGN = {a: align(a) for a in ANCHORED_ARMS}


def profiles21(arm: str) -> dict:
    a = ALIGN[arm]
    return {"anchored": a[D].tolist(), "other": np.delete(a, D, axis=0).mean(axis=0).tolist()}


def last_slice(arm: str) -> dict[str, tuple[float, float]]:
    """At the last slice: the alignment on `difference` contexts and the mean over the other ops, at four roles."""
    a = ALIGN[arm][:, -1, :]
    d, o = a[D], np.delete(a, D, axis=0).mean(axis=0)
    return {
        "query ?": (float(d[Q_Q]), float(o[Q_Q])),
        "query =": (float(d[Q_EQ]), float(o[Q_EQ])),
        "query answer": (float(d[Q_ANS]), float(o[Q_ANS])),
        "example answers": (float(d[EX_ANS].mean()), float(o[EX_ANS].mean())),
    }


LAST = {a: last_slice(a) for a in ANCHORED_ARMS}
SYNTAX = {a: {t: mean(a, lambda r, t=t: r["syntax_embeddings"][t]) for t, _ in ex.SYNTAX_TOKENS} for a in ANCHORED_ARMS}
SYNTAX_CTRL = {t: mean(ex.CONTROL, lambda r, t=t: r["syntax_embeddings"][t]) for t, _ in ex.SYNTAX_TOKENS}
NL_ARMS = [a for a in ANCHORED_ARMS if SYNTAX[a]["⏎"] > 0.5]
OTHER_EQ = [a for a in ANCHORED_ARMS if a not in ("anchor-query-eq", "anchor-every-eq")]


def unpulled(arm: str) -> list[list[int]]:
    """The (slice, position) pairs *arm* never pulls on a labelled context, for the E1 hatching. `prefix` pulls a
    position once the posterior given the tokens before it clears its threshold, which varies by context, so only
    the first example (no evidence before it) is certain to be left out; `sampled` and the hinge can pull anywhere.
    """
    n_pos, n_slices = len(ROLE_LABELS), len(SLICE_NAMES)
    label = ex.arm(arm).label
    if label == "no-emb":
        return [[0, p] for p in range(n_pos)]
    pulled = {
        "latter": range(n_pos // 2, n_pos),
        "prefix": range(ex.example_answer_roles(ex.K)[0] + 1, n_pos),
        **ex.ROLE_PULLS,
    }.get(label, range(n_pos))
    return [[sl, p] for sl in range(n_slices) for p in range(n_pos) if p not in pulled]


def hatch_unpulled(ax: Axes, cells: list[list[int]]) -> None:
    from matplotlib.patches import Rectangle

    for sl, p in cells:
        ax.add_patch(
            Rectangle(
                (p - 0.5, sl * STACK_GAP - 0.05), 1, STACK_GAP, hatch="////", fill=False, lw=0,
                edgecolor=light_dark("#000", "#fff"), alpha=0.18, zorder=0,
            )
        )  # fmt: skip


def e1_figure() -> str:
    data = {a: profiles21(a) | {"unpulled": unpulled(a)} for a in ANCHORED_ARMS}
    alt = f"""
        A grid of {len(ANCHORED_ARMS)} panels, three per row, one per anchored arm
        ({", ".join(arm_name(a) for a in ANCHORED_ARMS)}). Each stacks five step-shaped traces, one per slice from
        the embedding at the bottom to the last block at the top, of the alignment with the anchored axis across
        the 24 positions of a three-example context: solid for `{ex.ANCHORED_OP}` contexts, dashed for the mean of
        the other ops. At the last block, the `{ex.ANCHORED_OP}` trace at the query `=` is
        {", ".join(f"{arm_name(a)} {LAST[a]['query ='][0]:.2f}" for a in ANCHORED_ARMS)}; at the query answer it is
        {", ".join(f"{arm_name(a)} {LAST[a]['query answer'][0]:.2f}" for a in ANCHORED_ARMS)}. On query-eq and
        every-eq the dashed trace at the query `=` reaches
        {LAST["anchor-query-eq"]["query ="][1]:.2f} and {LAST["anchor-every-eq"]["query ="][1]:.2f}.
    """
    return e1_draw(data, alt)


@memo
def e1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e1-alignment-stack",
        alt_text=alt_text,
        caption=f"""
            **Where each anchored arm puts the anchor.** As the preview figure: alignment with e₁ by position, one
            trace per slice from the embedding (bottom) to the last block (top); seed means on held-out contexts.
            Solid: `{ex.ANCHORED_OP}` contexts. Dashed: the mean over the other six ops. The shaded column is the
            query `=`; the vertical line marks the start of the query. Hatched: the positions and slices the arm
            never pulls (for prefix, the first example, which has no evidence before it; elsewhere in prefix the pull
            depends on the context). The bar at the right of each row spans an alignment of 0 to 1.
        """,
    )
    def _plot() -> plt.Figure:
        n = len(data)
        n_rows = -(-n // 3)
        fig, axes = plt.subplots(n_rows, 3, figsize=(8.4, 3.5 * n_rows), layout="constrained", sharey=True)
        n_slices = len(next(iter(data.values()))["anchored"])
        for i, (arm, d) in enumerate(data.items()):
            ax = axes[i // 3, i % 3]
            stack_panel(ax, d, [])
            hatch_unpulled(ax, d["unpulled"])
            ax.set_title(arm_name(arm), fontsize=9)
        for r in range(n_rows):
            stack_frame(axes[r, 0], axes[r, -1], n_slices)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        fig.legend(handles=stack_handles(), loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


def e1_table() -> str:
    roles = ("query ?", "query =", "query answer", "example answers")
    labels = {"query ?": "`?`", "query =": "`=`", "query answer": "y", "example answers": "example y"}
    head = ["arm", *(h for r in roles for h in (labels[r], f"{labels[r]} other"))]
    rows = [[arm_name(a), *(f"{v:.2f}" for r in roles for v in LAST[a][r])] for a in ANCHORED_ARMS]
    return table_html(
        head,
        rows,
        f"**Alignment at the last block, by role in the query.** Seed means on held-out contexts. For each role, the "
        f"first column is on `{ex.ANCHORED_OP}` contexts and the second the mean over the other six ops; example y "
        f"is the mean over the three example answers.",
    )


def syntax_table() -> str:
    head = ["arm", *(f"`{t}`" if t != "⏎" else "⏎" for t, _ in ex.SYNTAX_TOKENS)]
    rows = [
        [arm_name(ex.CONTROL), *(f"{SYNTAX_CTRL[t]:+.2f}" for t, _ in ex.SYNTAX_TOKENS)],
        *([arm_name(a), *(f"{SYNTAX[a][t]:+.2f}" for t, _ in ex.SYNTAX_TOKENS)] for a in ANCHORED_ARMS),
    ]
    return table_html(
        head,
        rows,
        "**The syntax embeddings at the embedding slice.** Alignment of each token embedding with e₁, seed means; "
        "the control for reference.",
        ref_rows=frozenset({0}),
    )


# --- E2 ----------------------------------------------------------------------------------------------------------

SUPP_E2 = {a: supp_mean(SUPP21, a) for a in (*E2_ARMS, *ex.POST_HOC_SUPPRESSION_ARMS, ex.CONTROL)}
E2_SITES = ex.EDIT_SITES
OPERATORS = ("projection", "repulsion", "reflection")


def drop21(arm: str, key: tuple) -> np.ndarray:
    """Per op, the net drop in expected exact match under an edit: the arm's drop less the control's."""
    s, c = SUPP_E2[arm], SUPP_E2[ex.CONTROL]
    return (s["clean"] - s["edits"][key]) - (c["clean"] - c["edits"][key])


def gap21(arm: str) -> float:
    s = SUPP_E2[arm]
    return float(s["clean"][D] - s["null"][D])


def criteria(arm: str, site: str, operator: str) -> dict:
    """The two criteria of E2 for one operator at one site on one arm."""
    keys = [(o, g, site) for o, g, _ in SUPP_EDITS if o == operator]
    drops = [float(drop21(arm, k)[D]) for k in keys]
    worst = max(float(np.delete(drop21(arm, k), D).max()) for k in keys)
    rises = len(keys) > 1 and all(b >= a - ex.GRADE_DIP for a, b in zip(drops, drops[1:], strict=False))
    full = drops[-1] / gap21(arm)
    return {
        "grades": rises and full >= ex.GRADING_MIN_DAMAGE,
        "selective": worst <= ex.SELECTIVITY_GATE,
        "share": max(d / gap21(arm) for d in drops),
        "full": full,
        "worst": worst,
        "doses": len(keys),
    }


CRIT = {(a, s, o): criteria(a, s, o) for a in E2_ARMS for s in ex.SCORED_SITES for o in OPERATORS}
MEETS = [k for k, v in CRIT.items() if v["grades"] and v["selective"]]


def shares(arm: str, site: str, operator: str) -> list[float]:
    """The net drop on the anchored op as a share of the way to the target null, at each dose of one operator."""
    return [float(drop21(arm, (o, g, site))[D]) / gap21(arm) for o, g, _ in SUPP_EDITS if o == operator]


FULL_PROJ = ("projection", 1.0, "every position")


def e2_figure() -> str:
    data = {
        arm: {
            "half": ex.GRADING_MIN_DAMAGE * gap21(arm),
            "meets": [site for site in E2_SITES if any(k[:2] == (arm, site) for k in MEETS)],
            "sites": {
                site: {
                    "anchored": [float(drop21(arm, (o, g, site))[D]) for o, g, _ in SUPP_EDITS],
                    "worst": [float(np.delete(drop21(arm, (o, g, site)), D).max()) for o, g, _ in SUPP_EDITS],
                }
                for site in E2_SITES
            },
        }
        for arm in E2_ARMS
    }
    hp = CRIT[("anchor-hinge", "every position", "projection")]
    wp = CRIT[(ex.PRIMARY, "every position", "projection")]
    alt = f"""
        A grid of {len(E2_ARMS) * len(E2_SITES)} panels: rows for the {len(E2_ARMS)} arms
        ({", ".join(arm_name(a) for a in E2_ARMS)}), columns for the four edit sites
        ({", ".join(E2_SITES)}). Each panel plots the net drop in expected exact match against the edit (the
        projection at four doses, the repulsion at two, the reflection), solid for `{ex.ANCHORED_OP}` contexts and
        dashed for the worst other op, with a dotted rule at the selectivity gate of {ex.SELECTIVITY_GATE:g} and a
        dash-dot rule halfway to the target null, about {data[ex.PRIMARY]["half"]:.2f}. The hinge panel at every
        position has a heavy red border: the only panel where an operator meets both criteria. At the query `?` every arm
        stays near zero except prompt, which reaches {float(drop21("anchor-prompt", ("projection", 1.0, "query ?"))[D]):.2f}
        under the full projection. On whole-line and hinge, the query `=` stays near zero too, while every position and the
        example answers rise with the projection dose past the halfway rule; on hinge at every position the dashed
        trace stays at the gate or below (worst {hp["worst"]:.3f}), and on whole-line it reaches {wp["worst"]:.3f}
        at full dose. On prompt the drop at every position rises with the worst other op close behind, up to
        {CRIT[("anchor-prompt", "every position", "projection")]["worst"]:.2f}. On query-eq and every-eq, the query
        `=` and every position rise steeply on both traces, the worst other op reaching
        {CRIT[("anchor-query-eq", "query =", "projection")]["worst"]:.2f} and
        {CRIT[("anchor-every-eq", "query =", "projection")]["worst"]:.2f} under the full projection, and the
        example answers stay at zero.
    """
    caption = f"""
        **The suppression pass on the candidates and references of E2.** As the preview figure: the net drop in
        held-out expected exact match under each edit, the seed-mean drop less the control's drop under the same
        edit. Solid: on `{ex.ANCHORED_OP}` contexts. Dashed: the worst of the other six ops. The dotted rule is the
        selectivity gate ({ex.SELECTIVITY_GATE:g}), and the dash-dot rule is {ex.GRADING_MIN_DAMAGE:.0%} of the way
        from the clean score on `{ex.ANCHORED_OP}` to the target null. The query `=` and every position are the
        scored sites; a heavy border marks a panel where an operator meets both criteria.
    """
    return suppression_draw(data, alt, "e2-suppression", caption, (8.4, 9.0))


def e2_table() -> str:
    rows = [
        [
            arm_name(a),
            s,
            o,
            bold_if(v["grades"], "yes" if v["grades"] else "no") if v["doses"] > 1 else "—",
            bold_if(v["selective"], "yes" if v["selective"] else "no"),
            f"{v['share']:.2f}",
            f"{v['worst']:+.3f}",
        ]
        for (a, s, o), v in CRIT.items()
    ]
    return table_html(
        ["arm", "site", "operator", "grades", "selective", "max share to null ↑", "worst other ↓"],
        rows,
        f"**The two criteria of E2 at the two scored sites.** Grades: the net drop on `{ex.ANCHORED_OP}` rises with "
        f"dose (dips of at most {ex.GRADE_DIP:g}) and reaches {ex.GRADING_MIN_DAMAGE:.0%} of the way to the target "
        f"null at full dose; the reflection has one dose and cannot grade. Selective: the net drop on every other op "
        f"is at most {ex.SELECTIVITY_GATE:g} at every dose. The last two columns are the largest share of the way to "
        f"the null over the doses, and the largest net drop on any other op at any dose.",
        text_cols=3,
    )


NO_EMB = "anchor-no-emb"
PH_ARMS = [ex.PRIMARY, "anchor-hinge", NO_EMB]
CRIT_PH = {(a, "every position", o): criteria(a, "every position", o) for a in PH_ARMS for o in OPERATORS}
EX_ANS = list(ex.example_answer_roles(ex.K))


def other_at_examples(arm: str) -> np.ndarray:
    """Per slice, the alignment at the example answers, mean over the other ops."""
    return np.delete(ALIGN[arm], D, axis=0)[:, :, EX_ANS].mean(axis=(0, 2))


def selective_max(arm: str, operator: str) -> tuple[float, float]:
    """The largest share of the way to the null over the doses of *operator* at every position whose worst other
    op stays within the selectivity gate, with that dose; (0, 0) if no dose does.
    """
    keys = [(o, g, "every position") for o, g, _ in SUPP_EDITS if o == operator]
    ok = [
        (float(drop21(arm, k)[D]) / gap21(arm), k[1])
        for k in keys
        if float(np.delete(drop21(arm, k), D).max()) <= ex.SELECTIVITY_GATE
    ]
    return max(ok, default=(0.0, 0.0))


def no_emb_figure() -> str:
    sites = ("every position", "example answers")
    data = {
        arm: {
            "half": ex.GRADING_MIN_DAMAGE * gap21(arm),
            "sites": {
                site: {
                    "anchored": [float(drop21(arm, (o, g, site))[D]) for o, g, _ in SUPP_EDITS],
                    "worst": [float(np.delete(drop21(arm, (o, g, site)), D).max()) for o, g, _ in SUPP_EDITS],
                }
                for site in sites
            },
        }
        for arm in PH_ARMS
    }
    ne = data[NO_EMB]["sites"]["every position"]
    alt = f"""
        A grid of six panels: rows for the whole-line, hinge, and no-emb arms, columns for two edit sites (every
        position and the example answers). Each panel plots the net drop in expected exact match against the edit,
        as in the E2 figure, solid for `{ex.ANCHORED_OP}` and dashed for the worst other op. The first two rows
        repeat E2. On no-emb at every position the solid trace rises faster with the projection dose
        ({", ".join(f"{v:.2f}" for v in ne["anchored"][:4])}), and the dashed trace rises with it, to
        {ne["worst"][3]:.3f} under the full projection and {ne["worst"][5]:.3f} under the deeper repulsion, well
        over the gate of {ex.SELECTIVITY_GATE:g}.
    """
    caption = f"""
        **Post hoc: the suppression pass on no-emb, beside the two candidates.** As the E2 figure, at two of its
        sites. Solid: on `{ex.ANCHORED_OP}` contexts. Dashed: the worst of the other six ops. Dotted: the selectivity
        gate; dash-dot: {ex.GRADING_MIN_DAMAGE:.0%} of the way to the target null.
    """
    return suppression_draw(data, alt, "e2-no-emb", caption, (6.0, 6.4))


POSTERIOR_OTHER = np.delete(SUPP_E2[ex.CONTROL]["posterior"], D)
OTHER_OPS = [o for o in OPS if o != ex.ANCHORED_OP]


def confusion_figure() -> str:
    data = {
        "posterior": POSTERIOR_OTHER.tolist(),
        "drops": {a: np.delete(drop21(a, FULL_PROJ), D).tolist() for a in E2_ARMS},
    }
    alt = f"""
        {len(E2_ARMS)} scatter panels side by side, one per arm ({", ".join(arm_name(a) for a in E2_ARMS)}), each
        with six points, one per other op, colored by op: the net drop in expected exact match under the full
        projection at every position on the y-axis, against the mean posterior on `{ex.ANCHORED_OP}` in that op's
        contexts on the x-axis. Every point sits between x = {POSTERIOR_OTHER.min():.3f} and
        {POSTERIOR_OTHER.max():.3f}, so each panel is a vertical column. The drops run
        {", ".join(f"{arm_name(a)} {span(data['drops'][a], '+.3f')}" for a in E2_ARMS)}.
    """
    return confusion_draw(data, alt)


@memo
def confusion_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e2-confusion",
        alt_text=alt_text,
        caption=f"""
            **Net drop on each other op against how plausible `{ex.ANCHORED_OP}` is in its contexts.** Under the
            full projection at every position; one point per op other than `{ex.ANCHORED_OP}`. The x-axis is the
            mean posterior on `{ex.ANCHORED_OP}` given the examples, over held-out contexts of that op. The dotted
            rule is the selectivity gate.
        """,
    )
    def _plot() -> plt.Figure:
        arms = list(data["drops"])
        fig, axes = plt.subplots(1, len(arms), figsize=(8.4, 2.6), layout="constrained", sharey=True, sharex=True)
        axes = cast(AxesRow, axes)
        ink = rule_color()
        for ax, arm in zip(axes, arms, strict=True):
            ax.axhline(ex.SELECTIVITY_GATE, color=ink, lw=0.8, ls=":")
            ax.axhline(0, color=ink, lw=0.3, alpha=0.5)
            for j, op in enumerate(OTHER_OPS):
                ax.plot(data["posterior"][j], data["drops"][arm][j], "o", ms=4, color=f"C{j}", label=op, alpha=0.85)
            ax.set_title(arm_name(arm), fontsize=9)
            ax.set_xlim(0, 0.1)
            ax.set_xticks([0, 0.05, 0.1], ["0", "0.05", "0.1"], fontsize=7)
            ax.set_xlabel(f"posterior on {ex.ANCHORED_OP}", fontsize=7)
        axes[0].set_ylabel("net drop", fontsize=8)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=6, frameon=False, fontsize=7)
        return fig

    return _plot()


# --- Post hoc: what the model answers instead ----------------------------------------------------------------------

CONF = fetch_jsons(ex.CONFUSION_REF)[0]["runs"]
CONF_ARMS = [ex.CONTROL, *E2_ARMS, *ex.POST_HOC_SUPPRESSION_ARMS]


def conf_mean(arm: str, key: str) -> np.ndarray:
    """Seed mean of one matrix of the confusion pass: row i, column j is the mass on the answers of op j in the
    contexts of op i. *key* is `bayes`, `null`, `clean`, or a scored site (the full projection there).
    """
    rs = [r for r in CONF if r["arm"] == arm]
    return np.mean([r["edits"][key] if key in r["edits"] else r[key] for r in rs], axis=0)


NULL_CONF = conf_mean(ex.CONTROL, "null")
CONF_CLEAN = {a: conf_mean(a, "clean") for a in CONF_ARMS}
CONF_EDIT = {a: conf_mean(a, "every position") for a in CONF_ARMS}


def other_diag(m: np.ndarray) -> float:
    """The mean mass on the true op over the contexts of the six other ops."""
    return float(np.delete(np.diag(m), D).mean())


def _spill_op(arm: str) -> int:
    """The other op whose mass on its own answer the full projection lowers most, net of the control."""
    net = np.diag(CONF_CLEAN[arm] - CONF_EDIT[arm]) - np.diag(CONF_CLEAN[ex.CONTROL] - CONF_EDIT[ex.CONTROL])
    net[D] = -np.inf
    return int(np.argmax(net))


NO_EMB_SPILL_I = _spill_op(ex.POST_HOC_SUPPRESSION_ARMS[0])
NO_EMB_SPILL = OPS[NO_EMB_SPILL_I]


def answers_figure() -> str:
    panels = {f"Bayes, {ex.ANCHORED_OP} removed": NULL_CONF} | {arm_name(a): CONF_EDIT[a] for a in CONF_ARMS}
    rows = {k: [f"{v:.2f}" for v in m[D]] for k, m in panels.items()}
    alt = f"""
        A grid of {len(panels)} heatmaps, each seven by seven, rows the true op of a context and columns the op whose
        answers the model gives, with the value printed in each cell. The first panel is the Bayes predictor with
        `{ex.ANCHORED_OP}` removed; the rest are the arms under the full projection at every position. In the
        `{ex.ANCHORED_OP}` row the Bayes panel reads {", ".join(rows[next(iter(panels))])};
        {"; ".join(f"{k} reads {', '.join(v)}" for k, v in list(rows.items())[1:])}. On control, whole-line, hinge,
        and no-emb the diagonal of the other ops stays bright; on query-eq and every-eq every cell is dark.
    """
    caption = f"""
        **What the model answers in place of `{ex.ANCHORED_OP}`.** Post hoc. Each panel is a matrix over held-out
        contexts: row *i*, column *j* is the mean mass on the answers of op *j* in the contexts of op *i*, seed
        mean. Ops that share an answer on a pair both take its mass, so a row can sum past 1. First panel: the Bayes
        predictor with `{ex.ANCHORED_OP}` removed from the posterior, the answer a model should give once it no
        longer knows the op. Others: each arm under the full projection at every position.
    """
    return answers_draw({k: m.tolist() for k, m in panels.items()}, alt, caption)


@memo
def answers_draw(panels: dict, alt_text: str, caption: str) -> str:
    @themed(name="e2-answers", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        cols = 4
        nrows = -(-len(panels) // cols)
        fig, axes = plt.subplots(nrows, cols, figsize=(8.4, 2.3 * nrows + 0.4), layout="constrained")
        cmap = plt.get_cmap(light_dark("Blues", "magma"))
        vmax = 0.8
        n = len(OPS)
        im = None
        for ax, (title, m) in zip(axes.flat, panels.items(), strict=False):
            m = np.asarray(m)
            # The diagonal stays on the color scale: the mass on the true op is what the prose compares.
            im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
            for i, j in np.ndindex(n, n):
                dark_cell = (m[i, j] / vmax > 0.55) == (light_dark(0, 1) == 0)
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="#fff" if dark_cell else "#000",
                )
            ax.set_title(title, fontsize=8)
            ax.set_xticks(range(n), OPS, rotation=90, fontsize=6)
            ax.set_yticks(range(n), OPS, fontsize=6)
        for ax in axes.flat[len(panels) :]:
            ax.set_visible(False)
        fig.supxlabel("answers of this op", fontsize=8)
        fig.supylabel("true op", fontsize=8)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.5, label="mass")
        return fig

    return _plot()


# --- S2 ----------------------------------------------------------------------------------------------------------

S2_PAIRS = (("control-verify", ex.CONTROL), ("anchor-verify", ex.PRIMARY))
S2: dict[str, dict[str, Any]] = {
    v: {
        "ref": ref,
        "diff": mean(v) - mean(ref),
        "band": band(per_seed(v), per_seed(ref)),
        "acc": per_seed(v, lambda r: r["verify"]["accuracy"]),
    }
    for v, ref in S2_PAIRS
}
for v in S2.values():
    v["within"] = abs(v["diff"]) <= v["band"]
VERIFY_MARGIN = mean("anchor-verify", lambda r: r["margin"]["value"])


def s2_figure() -> str:
    data = {
        "eem": {a: per_seed(a).tolist() for pair in S2_PAIRS for a in pair[::-1]},
        "bands": {v: (mean(ref), S2[v]["band"]) for v, ref in S2_PAIRS},
        "acc": {v: S2[v]["acc"].tolist() for v, _ in S2_PAIRS},
    }
    alt = f"""
        Two panels. Left: held-out expected exact match per seed in four columns, the control, control-verify,
        whole-line, and anchor-verify, with seed means as larger dots. Over each verification column a shaded band
        spans the reference mean ± the seed band of the pair, and each verification mean lies inside it:
        control-verify {mean("control-verify"):.3f} against the control's {H1["ctrl"]:.3f} (band ±
        {S2["control-verify"]["band"]:.3f}), anchor-verify {mean("anchor-verify"):.3f} against whole-line's
        {H1["whole"]:.3f} (band ± {S2["anchor-verify"]["band"]:.3f}). Right: verification accuracy per seed for the
        two verification arms, seed means {S2["control-verify"]["acc"].mean():.2f} and
        {S2["anchor-verify"]["acc"].mean():.2f}.
    """
    return s2_draw(data, alt)


@memo
def s2_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="s2-verify",
        alt_text=alt_text,
        caption="""
            **The verification rule.** Left: held-out expected exact match on completion contexts, small dots for
            runs and large dots for seed means. Over each verification arm, the shaded band is the mean of the arm
            without verification lines ± the seed band of the pair, the range the rule allows. Right: verification
            accuracy on the verification held-out set.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained", width_ratios=[2, 1])
        rng = np.random.default_rng(0)
        ink = rule_color()
        cols = list(data["eem"])
        for i, arm in enumerate(cols):
            if arm in data["bands"]:
                m, w = data["bands"][arm]
                a.fill_between([i - 0.3, i + 0.3], m - w, m + w, color=ink, alpha=0.1, lw=0)
            dots(a, i, data["eem"][arm], arm_color(arm), rng=rng)
        for i, arm in enumerate(data["acc"]):
            dots(b, i, data["acc"][arm], arm_color(arm), rng=rng)
        a.set_xticks(range(len(cols)), [arm_name(c) for c in cols], fontsize=8)
        a.set_xlim(-0.6, len(cols) - 0.4)
        a.set_ylabel("held-out EEM", fontsize=8)
        b.set_xticks(range(len(data["acc"])), list(data["acc"]), fontsize=8)
        b.set_xlim(-0.6, len(data["acc"]) - 0.4)
        b.set_ylim(0.7, 1.0)
        b.set_ylabel("verification accuracy", fontsize=8)
        return fig

    return _plot()


def s2_table() -> str:
    rows = [
        [
            f"`{v}` vs. `{arm_name(d['ref'])}`",
            f"{mean(d['ref']):.4f}",
            f"{mean(v):.4f}",
            f"{d['diff']:+.4f}",
            f"{d['band']:.4f}",
            bold_if(d["within"], "yes" if d["within"] else "no"),
            f"{d['acc'].mean():.3f}",
        ]
        for v, d in S2.items()
    ]
    return table_html(
        ["pair", "without", "with", "difference", "seed band", "within", "verification accuracy"],
        rows,
        "**The verification rule.** Seed means of held-out expected exact match on completion contexts, without and "
        "with verification lines, and the seed band of the pair (three seeds against five); verification accuracy "
        "has no gate.",
    )


# --- H2 ----------------------------------------------------------------------------------------------------------

LEAN_ARMS = [ex.CONTROL, *(a.name for a in ex.ARMS if a.anchored)]
LEANS = ("op1", "fragment")
H2: dict[str, dict[str, Any]] = {
    lean: {
        "ctrl": mean(ex.CONTROL, lambda r, k=lean: r["leans"][k]),
        "whole": mean(ex.PRIMARY, lambda r, k=lean: r["leans"][k]),
        "band": band(
            per_seed(ex.PRIMARY, lambda r, k=lean: r["leans"][k]), per_seed(ex.CONTROL, lambda r, k=lean: r["leans"][k])
        ),
    }
    for lean in LEANS
}
for v in H2.values():
    v["within"] = abs(v["whole"] - v["ctrl"]) <= v["band"]
H2_OUTSIDE = [
    (a, lean)
    for a in LEAN_ARMS[1:]
    for lean in LEANS
    if abs(mean(a, lambda r, k=lean: r["leans"][k]) - H2[lean]["ctrl"])
    > band(per_seed(a, lambda r, k=lean: r["leans"][k]), per_seed(ex.CONTROL, lambda r, k=lean: r["leans"][k]))
]


def h2_figure() -> str:
    data: dict[str, Any] = {
        lean: {
            "seeds": {a: per_seed(a, lambda r, k=lean: r["leans"][k]).tolist() for a in LEAN_ARMS},
            "ctrl": H2[lean]["ctrl"],
            "band": H2[lean]["band"],
        }
        for lean in LEANS
    }
    lo = H2["op1"]["ctrl"] - H2["op1"]["band"]
    below = [a for a in LEAN_ARMS[1:] if np.mean(data["op1"]["seeds"][a]) < lo]
    alt = f"""
        Two stacked panels, the first-operand lean above and the trailing-fragment lean below, each with one column
        per arm: the control, then {", ".join(arm_name(a) for a in LEAN_ARMS[1:])}. Small dots are runs and large
        dots seed means; a shaded band spans the control mean ± the seed band of the whole-line arm against the
        control. First-operand lean: control {H2["op1"]["ctrl"]:.3f}, band ± {H2["op1"]["band"]:.3f}; the
        whole-line mean, {H2["op1"]["whole"]:.3f}, is inside it; {", ".join(arm_name(a) for a in below)} sit below
        its lower edge. Trailing-fragment lean: control {H2["fragment"]["ctrl"]:.3f}, band ±
        {H2["fragment"]["band"]:.3f}; the whole-line mean, {H2["fragment"]["whole"]:.3f}, and every other arm sit
        inside it.
    """
    return h2_draw(data, alt)


@memo
def h2_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="h2-leans",
        alt_text=alt_text,
        caption="""
            **The two leans, per arm.** Small dots are runs, large dots seed means. The shaded band is the control
            mean ± the seed band of the whole-line arm against the control (five seeds against five), the band H2
            scores; an arm at three seeds has a band about a fifth wider.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.4), layout="constrained", sharex=True)
        rng = np.random.default_rng(0)
        ink = rule_color()
        labels = {"op1": "first-operand lean", "fragment": "fragment lean"}
        for ax, (lean, d) in zip(axes, data.items(), strict=True):
            ax.axhspan(d["ctrl"] - d["band"], d["ctrl"] + d["band"], color=ink, alpha=0.08, lw=0)
            ax.axhline(0, color=ink, lw=0.3, alpha=0.5)
            for i, (arm, v) in enumerate(d["seeds"].items()):
                dots(ax, i, v, arm_color(arm), rng=rng)
            ax.set_ylabel(labels[lean], fontsize=8)
        arms = list(data["op1"]["seeds"])
        axes[-1].set_xticks(range(len(arms)), [arm_name(a) for a in arms], fontsize=7, rotation=30, ha="right")
        axes[-1].set_xlim(-0.6, len(arms) - 0.4)
        return fig

    return _plot()


rf"""
# Ex 2.2.21: the in-context grammar pilot, on the reworked recipe

/// tip |
<!-- tl;dr -->
We trained the anchored arms of round 2 again on the reworked recipe. The recipe holds, and on the hinge arm an edit applied at every position at once takes the op out gradually, without touching the other ops.
///

A second try at round 2 of the D2.2 route, on the recipe that ex-2.2.17 to ex-2.2.20 reworked. We retrain the control and the arms that anchor `{ex.ANCHORED_OP}` under the whole-line label and its variants, then score the rules that ex-2.2.16 never reached, to decide how round 3 anchors and edits the inferred op.

## Findings

- [The recipe holds, and the anchor costs nothing (H1)](#the-recipe-holds-and-the-anchor-costs-nothing-h1) — partial. The control reproduces ex-2.2.19. The whole-line arm falls short of the control by more than the tolerance, but by less than the seeds vary.
- [The label rule (S1)](#the-label-rule-s1) — decided: the whole-line label stays. No variant beats it by more than the seeds vary.
- [Where the anchor sits (E1)](#where-the-anchor-sits-e1) — on the answers, where the context shows the most evidence about its op, and hardly at all at the query `=`.
- [Editing the op out (E2)](#editing-the-op-out-e2) — on the hinge arm, an edit at every position takes the op out gradually and leaves the other ops alone. On the whole-line arm the same edit spills just past the gate onto another op. No edit at the query `=` meets both criteria on any arm.
- [The verification rule (S2)](#the-verification-rule-s2) — decided: verification lines stay, since they leave completion unchanged.
- [The leans stay with the control (H2)](#the-leans-stay-with-the-control-h2) — pass. The anchor adds no lean toward itself.

## How to read this draft

The rules and predictions were frozen at commit `2d84d2b`, before any run of this experiment. The [preview](#what-ex-2216s-anchored-arms-already-show) scores the stored runs of ex-2.2.16; it predates this plan and shaped it.

## Why this experiment

[Ex-2.2.16](/docs/m2/ex-2.2.16/report.py) was round 2 of the [D2.2 route](/docs/m2/d2.2/design.md#quick-route). It trained a control and several arms that anchor `{ex.ANCHORED_OP}` on the in-context grammar, and froze rules to select how round 3 would anchor and edit: the label, the site, the operator, and whether verification lines stay in the corpus. Round 3 would then anchor the inferred op at fresh seeds and suppress it.

But ex-2.2.16 stopped at its first rule. Its control had learned less than half of what the examples allow, {CTRL16_SKILL:.0%} of the way from the floor to the Bayes ceiling. An anchor compared with a model that far from finished would say little about round 3.

Four reports then worked on the control alone: [ex-2.2.17](/docs/m2/ex-2.2.17/report.py) added the newline mask, a lower peak learning rate, and more steps; [ex-2.2.18](/docs/m2/ex-2.2.18/report.py) dropped four ops whose answers often coincide with another op; [ex-2.2.19](/docs/m2/ex-2.2.19/report.py) settled the length at {ex.EPOCHS} epochs; and [ex-2.2.20](/docs/m2/ex-2.2.20/report.py) kept the plain schedule.

On the seven-op set the control now scores {REF19_MEAN:.3f} against a ceiling of {REF19_CEIL:.3f}, {REF19_SKILL:.0%} of the way from the floor. This pilot reruns round 2 on that recipe.

It keeps the label, verification, and operator rules of ex-2.2.16, changing each only where the stored runs of ex-2.2.16 or the new recipe call for it, and adds three label variants as references. The [design](/docs/m2/d2.2/design.md#the-pilot) names the rules (a) to (e); here they are H1, S1, and S2, and the site and operator rules become an exploratory analysis, E2.

## What ex-2.2.16's anchored arms already show

Ex-2.2.16's anchored arms were trained and evaluated, but the eval was never scored. These numbers come from the half-trained model, so they only show what to look for.

It turns out the anchor landed. The op margin, which measures how far `{ex.ANCHORED_OP}` contexts sit along the anchored axis beyond the rest, was {MARGIN16["anchor-whole"]:.2f} on the whole-line arm against {MARGIN16["control-k3-r0.3"]:.2f} on the control. But most of it landed where an edit cannot reach the answer:

{preview_figure()}

On the whole-line arm the alignment (the cosine of a state with e₁) sits on the answers. It builds up on the example answers by block 2 ({min(WHOLE[2][i] for i in EX_ANS):.2f} to {max(WHOLE[2][i] for i in EX_ANS):.2f}) and fades a little after; on the query answer it climbs with depth, to {WHOLE[-1][Q_ANS]:.2f} at the last block. The query `=`, whose state predicts the answer, stays below {max(r[Q_EQ] for r in WHOLE):.2f} at every slice.

The pooled anchor term asks a labelled context to align somewhere in its span. The answers are where the context has shown the most evidence about its op, so the pull seems to have settled there.

For an edit, the two kinds of answer differ. The state at the query answer comes after the answer is predicted (the model reads left to right) and predicts only the line break, so an edit there cannot change the answer. The query can read the example answers through the blocks above them, so an edit there may still reach it, though not an edit at the last slice, which no block reads.

The query `?` stayed clean ({WHOLE[-1][Q_Q]:.2f}). Ex-2.2.16 worried that it would saturate, but the position that came close was the query answer. The hinge arm, which caps the pull, holds {HINGE["anchored"][-1][Q_EQ]:.2f} at the query `=` at the last block, but the other ops hold nearly as much there ({HINGE["other"][-1][Q_EQ]:.2f}), so that alignment says little about `{ex.ANCHORED_OP}`.

The suppression pass agrees:

{suppression_figure()}

At the query `=`, an edit on the whole-line arm barely touches the answer, and on the hinge arm it lowers every op about as much as `{ex.ANCHORED_OP}`.

With the edit at every position and every slice, the projection on the whole-line arm grows with dose up to γ = 0.75, then levels off {to_null("anchor-whole", PV["whole_full"]):.0%} of the way to the target null. From γ = 0.75 on, it also spills past the selectivity gate onto other ops. The hinge arm gets {to_null("anchor-hinge", PV["whole_full"]):.0%} of the way and spills more.

At the embedding slice the two arms differ. There the solid and dashed traces coincide, because a token embedding is the same whatever the op. Each anchored arm has put a few syntax embeddings partway onto e₁:

- on the whole-line arm, `?` ({WHOLE[0][Q_Q]:.2f}), `,` ({WHOLE[0][COMMA]:.2f}), and the line break;
- on the hinge arm, `=` ({HINGE["anchored"][0][Q_EQ]:.2f}) and `,`.

On the earlier grammar a leak like this came through the tied readout (the embedding table reused as the readout table), and untying it brought the leak down ([ex-2.2.7](/docs/m2/ex-2.2.7/report.py)). The readout here is already untied. So this leak seems to come from the pull itself: the pull acts on the embedding slice of every position it pulls, and some of those are tokens every context shares.

On the whole-line arm the bump at `,` sits one position after the answers. That makes the alignment look as though it moves back a token with depth, but the two are separate effects.

On the hinge arm the `=` embedding explains why the query `=` holds alignment on every op. It starts at {HINGE["anchored"][0][Q_EQ]:.2f} for every op and keeps about that much to the last block, while the example `=` positions fade to {min(HINGE["anchored"][-1][i] for i in EX_EQ):.2f} to {max(HINGE["anchored"][-1][i] for i in EX_EQ):.2f}. An edit at every position moves these syntax states in every context, which may be part of why it spills onto other ops.

The preview changes four things in this plan:

- Three new label variants move the pull toward the query `=`, as references rather than candidates for round 3. `prompt` leaves the query answer out of the pull, to see where the pooled term settles without it. `query-eq` pulls the query `=` alone: a position oracle,[^oracle] which shows what an anchor there would allow. `every-eq` pulls every `=`, the positions whose next token depends on the op, to see whether a pull spread over the examples settles at the query `=` too.
- The saturation half of ex-2.2.16's hinge rule goes, since the query `?` did not saturate, though the hinge arm stays a candidate.
- The site and operator rules become an exploratory analysis that chooses no arm. The suppression pass also edits the example answers, and the edits are scored at the query `=` as well as at every position.
- E1 reports the syntax embeddings beside the per-position alignment, and the `no-emb` arm, which leaves the embedding slice out of the pull, shows whether they stay clean without it.

[^oracle]: The term from [ex-2.1.8](/docs/m2/ex-2.1.8/report.py): a pull at a position chosen by hand, which no labeller could give. It is a reference for what the right position would allow, and not a strict ceiling.

## Glossary

Terms follow [ex-2.2.16](/docs/m2/ex-2.2.16/report.py#glossary); these are the ones the rules here use.

<dl>
<dt>Alignment</dt>
<dd>The cosine between a state and e₁, the anchored axis, per position and slice. Cosine similarity compares direction only: 1 when two vectors point the same way, 0 when they are perpendicular.</dd>
<dt>Op margin</dt>
<dd>How far the contexts of the anchored op sit along e₁ beyond the rest. At each slice, take the mean alignment over <code>{ex.ANCHORED_OP}</code> contexts less the mean over all contexts, <code>{ex.ANCHORED_OP}</code> included, at the role where that gap is largest; then average over slices. The anchor term optimizes this quantity, so it checks that the pull landed.</dd>
<dt>Expected exact match (EEM)</dt>
<dd>The probability mass the model puts on the colors the query answer can be under the true op, weighted by how often the op gives each, averaged over held-out contexts. The task measurement.</dd>
<dt>Bayes ceiling and floor</dt>
<dd>The expected exact match of a predictor that holds the posterior over ops given the examples, and of one that ignores the examples.</dd>
<dt>Target null</dt>
<dd>The answer a model that has lost <code>{ex.ANCHORED_OP}</code> and nothing else would give: the posterior-weighted answer distribution with <code>{ex.ANCHORED_OP}</code> removed and the other ops renormalized.</dd>
<dt>Seed band</dt>
<dd>The smallest difference between two seed means a comparison resolves: {ex.SEED_BAND_SD:g}σ√(1/n₁ + 1/n₂) for arms at n₁ and n₂ seeds, with σ the seed standard deviation pooled over the two. At the spread of ex-2.2.19's four runs it is about {BAND_55:.3f} for five seeds against five, {BAND_53:.3f} for five against three, and {BAND_33:.3f} for three against three.</dd>
</dl>

## Conditions

Every arm trains at the center condition of ex-2.2.16 (three examples, ρ = {ex.CENTRE[1]:g}) on the seven-op set, under ex-2.2.19's recipe: the newline mask, a peak rate of {ex.PEAK_LR:g}, {ex.EPOCHS} epochs, {ex.MODEL}. The anchored arms add ex-2.2.14's anchor recipe, each weight scheduled as a fraction of training as before, and crop policy `{ex.CROP_POLICY}`.

{arms_table()}

Ex-2.2.16's arms at other corpus conditions, its larger control, and its two newline-mask arms are gone: the condition is settled, the larger control was for a shortfall the recipe has since closed, and the mask is now in every arm.

**The new label variants.** All three keep ex-2.2.16's labeller: a `{ex.ANCHORED_OP}` context draws a label with probability {ex.LABEL_RATE:g}, keyed on the op array beside the corpus. Variant (e), `prompt`, pulls every position of a labelled context up to and including the query `=`, leaving out the query answer and the line break. Variant (f), `query-eq`, pulls the query `=` alone. Variant (g), `every-eq`, pulls the `=` of each example and of the query, the four positions that predict an answer. The first example's `=` comes before any evidence, so its pull asks for the op before the context shows it.

`prompt` keeps the example answers in the pull, because they are part of the evidence. If the pull still settles on them, that is a result about pooling.

**The seeds.** Fresh model seeds from {ex.SEED_OFFSET}. The control and the whole-line arm get {ex.SEEDS_REFERENCE} each, since every comparison runs through one of them; the others get three.

## The recipe holds, and the anchor costs nothing (H1)

**What we expect.** Round 3 starts from this recipe only if both of these criteria hold:

- **(a)** The control reproduces ex-2.2.19: its seed-mean held-out expected exact match is within {ex.REGRESSION_TOL:g} of {REF19_MEAN:.3f}, the mean of ex-2.2.19's four runs at {ex.EPOCHS} epochs.
- **(b)** The whole-line arm falls short of the control by at most {ex.TASK_COST_TOL:g}, a little wider than the seed band of five seeds against five.

The arms differ from ex-2.2.19 only in their seeds and in the anchor, so (a) checks that nothing else moved.

We expect (b) to hold too. Ex-2.2.14 found the anchor costs the task nothing when {ex.LABEL_RATE:.0%} of contexts are labelled, and the stored runs of ex-2.2.16 agree: the whole-line arm scored {EEM16["anchor-whole"]:.3f} against {EEM16["control-k3-r0.3"]:.3f} for the control, and {EEM16["anchor-mask"]:.3f} against {EEM16["control-mask"]:.3f} with the newline mask.

A miss on (b) would mean the anchor and the task compete on this recipe, and would leave E2 with no candidate, since it asks the same of each.

This replaces corpus rule (a) of ex-2.2.16, which asked the control to come within 0.03 of the calibrated ceiling. The seven-op control misses that: it sits {REF19_CEIL - REF19_MEAN:.3f} below the ceiling at {ex.EPOCHS} epochs, and still misses at 400. Ex-2.2.17 found the remaining gap on contexts whose examples settle the op, where the model keeps mass on the answers of a similar op.

We take the control as it stands. Its skill score and calibration KL[^kl] are reported beside every comparison, but no criterion depends on them. In ex-2.2.19 the KL was about {REF19_KL:.2f} nats; ex-2.2.16 called a model calibrated below 0.05.

[^kl]: A KL divergence: a non-negative measure, in nats, of how far one probability distribution sits from another, 0 when they match. Here it compares the model's answer distribution with the Bayes predictor's.

{h1_figure()}

{h1_table()}

**What we saw.** The control scores {H1["ctrl"]:.4f}, {H1["a_diff"]:.4f} from ex-2.2.19's {REF19_MEAN:.4f}, so (a) holds. The whole-line arm scores {H1["whole"]:.4f}, {H1["b_short"]:.4f} short of the control against a tolerance of {ex.TASK_COST_TOL:g}, so (b) misses. That shortfall is inside the seed band of the pair, {H1["b_band"]:.3f}: the control has a seed standard deviation of {H1["ctrl_sd"]:.3f}, against {REF19_SD:.3f} over ex-2.2.19's four runs.

The skill score is {SKILL[ex.CONTROL]:.2f} for the control and {SKILL[ex.PRIMARY]:.2f} for the whole-line arm, and the calibration KL is {KL[ex.CONTROL].mean():.2f} and {KL[ex.PRIMARY].mean():.2f} nats ({np.mean(REF19_KLS):.2f} in ex-2.2.19).

<details markdown="1"><summary>Gate arithmetic</summary>

- **(a)** |{H1["ctrl"]:.4f} − {REF19_MEAN:.4f}| = {H1["a_diff"]:.4f} ≤ {ex.REGRESSION_TOL:g}: holds.
- **(b)** {H1["ctrl"]:.4f} − {H1["whole"]:.4f} = {H1["b_short"]:.4f} > {ex.TASK_COST_TOL:g}: misses.
- Seed band of (b), five seeds against five: {ex.SEED_BAND_SD:g} × {H1["b_band"] / (ex.SEED_BAND_SD * np.sqrt(2 / 5)):.4f} × √(1/5 + 1/5) = {H1["b_band"]:.4f}, with σ pooled over the control and the whole-line arm.

</details>

The seeds also differ in how fast they learn. This figure was drawn after the results came in, to look at the spread over seeds:

{traj_figure()}

Model seed {ex.SEED_OFFSET + SLOW_SEED} has the lowest held-out EEM in {len(SLOW_LOWEST)} of the {len(ex.ARMS)} arms, and the lowest run of every arm is still rising over the last 20 epochs, by {span(LOWEST_RISE)}.

/// admonition | Partial
(a) holds: the control is within {H1["a_diff"]:.3f} of ex-2.2.19. (b) misses as written: the whole-line arm falls {H1["b_short"]:.3f} short against a tolerance of {ex.TASK_COST_TOL:g}, a gap inside the seed band of {H1["b_band"]:.3f}.
///

## The label rule (S1)

**The rule.** Ex-2.2.16's rule (b), unchanged. The whole-line label stays the primary unless a candidate variant (below) beats the whole-line arm on seed-mean held-out expected exact match by more than the seed band, while keeping an op margin of at least {ex.MARGIN_KEEP:.0%} of the whole-line margin. If several qualify, the one with the higher op margin goes forward.

The candidates are `latter`, `prefix`, and `no-emb` (the whole-line label with the embedding slice left out of the pull). Variant (d), `sampled`, trains the grading that round 3 will measure, so it is reported and not promoted.

**What we expect.** No variant clears the gate, for the reasons ex-2.2.16 gave: the label share barely moved the margin in ex-2.2.14, and the variants change less than that.

{s1_figure()}

{s1_table()}

**What we saw.** No candidate qualifies. Each scores above the whole-line arm, by less than the seed band of the pair: `no-emb` by {S1["anchor-no-emb"]["diff"]:+.3f} against {S1["anchor-no-emb"]["band"]:.3f}, `latter` by {S1["anchor-latter"]["diff"]:+.3f} against {S1["anchor-latter"]["band"]:.3f}, and `prefix` by {S1["anchor-prefix"]["diff"]:+.3f} against {S1["anchor-prefix"]["band"]:.3f}. `no-emb` keeps {S1["anchor-no-emb"]["share"]:.2f} of the whole-line op margin, `prefix` {S1["anchor-prefix"]["share"]:.2f}, and `latter` {S1["anchor-latter"]["share"]:.2f}, below the {ex.MARGIN_KEEP:.0%} floor. `sampled`, not a candidate, scores {S1["anchor-sampled"]["diff"]:+.3f} against the whole-line arm and keeps {S1["anchor-sampled"]["share"]:.2f} of its margin.

<details markdown="1"><summary>Gate arithmetic</summary>

A candidate qualifies if its EEM exceeds the whole-line arm by more than the seed band (three seeds against five, σ pooled over the pair) and its op margin is at least {ex.MARGIN_KEEP:.0%} of the whole-line margin, {WHOLE_MARGIN:.3f}.

{"".join(f"- `{arm_name(a)}`: {S1[a]['diff']:+.4f} against {S1[a]['band']:.4f}, {'clears' if S1[a]['diff'] > S1[a]['band'] else 'does not clear'}; margin share {S1[a]['share']:.2f}, {'keeps' if S1[a]['share'] >= ex.MARGIN_KEEP else 'below'} {ex.MARGIN_KEEP:g}." + chr(10) for a in S1_CANDIDATES)}
</details>

/// admonition | Decided
No candidate clears the seed band, so the whole-line label stays the primary.
///

## Where the anchor sits (E1)

Where an edit can work depends on where along the context each anchored arm puts its alignment. E1 measures, per anchored arm, the seed-mean alignment at each position and slice of a held-out context, on `{ex.ANCHORED_OP}` contexts and on the other ops.

{e1_figure()}

{e1_table()}

{syntax_table()}

**What we saw.** At the last block the query `=` holds {span([LAST[a]["query ="][0] for a in OTHER_EQ], ".2f")} on `{ex.ANCHORED_OP}` contexts on every arm but two: `query-eq` at {LAST["anchor-query-eq"]["query ="][0]:.2f} and `every-eq` at {LAST["anchor-every-eq"]["query ="][0]:.2f}, where the other ops hold {LAST["anchor-query-eq"]["query ="][1]:.2f} and {LAST["anchor-every-eq"]["query ="][1]:.2f} at the same position. The whole-line arm holds {LAST[ex.PRIMARY]["query answer"][0]:.2f} at the query answer and {LAST[ex.PRIMARY]["example answers"][0]:.2f} on the example answers; `prompt`, whose pull leaves out the query answer, holds {LAST["anchor-prompt"]["query answer"][0]:.2f} and {LAST["anchor-prompt"]["example answers"][0]:.2f}. The query `?` stays within {max(abs(LAST[a]["query ?"][0]) for a in ANCHORED_ARMS):.2f} of zero on every arm.

At the embedding slice the line-break embedding sits at {span([SYNTAX[a]["⏎"] for a in NL_ARMS], ".2f")} on {", ".join(f"`{arm_name(a)}`" for a in NL_ARMS)}. `no-emb` keeps every syntax embedding within {max(abs(v) for v in SYNTAX["anchor-no-emb"].values()):.2f} of zero. `prompt` lifts `,` to {SYNTAX["anchor-prompt"][","]:.2f} and `?` to {SYNTAX["anchor-prompt"]["?"]:.2f}, and `query-eq` and `every-eq` lift `=` to {SYNTAX["anchor-query-eq"]["="]:.2f} and {SYNTAX["anchor-every-eq"]["="]:.2f}.

## Editing the op out (E2)

E2 asks whether `{ex.ANCHORED_OP}` can be edited out of an anchored model with a dial: an edit whose effect on `{ex.ANCHORED_OP}` grows with its dose while the other ops stay as they were. It chooses no arm, so round 3 is designed after this report, and it replaces hinge rule (c) and operator rule (e) of ex-2.2.16.

The candidates (`{"`, `".join(ex.SITE_ARMS)}`) are arms round 3 could adopt as they stand. The references (`{"`, `".join(ex.SITE_REFERENCES)}`) put the pull where no labeller could.

**The edits.** The suppression pass of ex-2.2.16 runs, scoring only, on the candidates, the references, and the control. It has three operators, each applied at every slice at once:

- the projection, which removes a share γ of the component along e₁, for γ in {{{", ".join(f"{g:g}" for g in ex.DOSE_GAMMAS)}}};
- the repulsion, which moves states aligned above {ex.REPULSION_THRESHOLD:g} down to an alignment of {" and then ".join(f"{b:g}" for b in ex.REPULSION_LANDINGS)};
- the reflection, which flips the component, at one dose.

Each edit is applied at one of four sites: the query `?`, the query `=`, the example answers, or every position. Every drop listed below is a net drop: the seed-mean fall in held-out expected exact match, minus the fall the control shows under the same edit, so only the part the anchor caused counts.

**The two criteria.** Two sites are scored against the criteria below: the query `=`, and every position (the edit that would carry over most readily to another task). At the other two sites the drops are reported without criteria. An operator at a scored site meets the criteria on an arm when both of these hold:

- **It grades.** The net drop on `{ex.ANCHORED_OP}` contexts rises with the dose, allowing a dip between adjacent doses of at most {ex.GRADE_DIP:g}, and at full dose it covers at least {ex.GRADING_MIN_DAMAGE:.0%} of the distance from the clean score to the target null. The reflection has one dose, so it cannot grade.
- **It is selective.** On each of the other six ops, the net drop is at most {ex.SELECTIVITY_GATE:g} at every dose. This is the threshold [ex-2.2.11](/docs/m2/ex-2.2.11/report.py) set for a change in expected exact match too small to matter for the task. It sits well above the seed band of three seeds against three, because the worst of six ops is the largest of six noisy numbers.

A flat threshold treats every other op alike, but some are easier to mistake for `{ex.ANCHORED_OP}` than others. So beside it, E2 shows the net drop on each other op against how plausible `{ex.ANCHORED_OP}` is in that op's contexts: the mean posterior on `{ex.ANCHORED_OP}` given the examples. If the drop rises with that posterior, the edit takes out the op as the model infers it, and the cost on other ops follows the confusions the contexts allow.

**Where the op comes from.** The bypass test in the design for round 3 edits at different positions and slices, looking for a route by which the op reaches the answer around the edit. E2 is a first look at it, varying the position only. If the edit at every position works and the one at the query `=` does not, the answer must take the op from somewhere else, and the edit at the example answers shows whether that is the examples.

**What we expect.** Neither candidate meets both criteria. On the whole-line arm the projection at every position levels off about a third of the way to the null, as in the preview, and on the hinge arm the edit at every position spills onto other ops.

On `query-eq` we expect the projection at the query `=` to meet both, which would say that an anchor there can be edited with a dial, if a label could put it there. On `every-eq` we expect the same, more weakly, since a quarter of its pull lands on the query `=`. We are unsure about `prompt`: its pull may settle on the example answers, which E1 will show.

{e2_figure()}

{e2_table()}

**What we saw.** Of the {len(CRIT)} operators, sites, and arms scored, {len(MEETS)} meet both criteria, both on the hinge arm at every position: the projection, which covers {", ".join(f"{v:.2f}" for v in shares("anchor-hinge", "every position", "projection"))} of the way to the target null over its four doses with the worst other op at most {CRIT[("anchor-hinge", "every position", "projection")]["worst"]:.3f}, and the repulsion, which covers {", ".join(f"{v:.2f}" for v in shares("anchor-hinge", "every position", "repulsion"))} with the worst other op at most {CRIT[("anchor-hinge", "every position", "repulsion")]["worst"]:.3f}. On the whole-line arm the projection at every position grades the same way ({", ".join(f"{v:.2f}" for v in shares(ex.PRIMARY, "every position", "projection"))}), and the worst other op reaches {CRIT[(ex.PRIMARY, "every position", "projection")]["worst"]:.3f} at full dose, over the gate of {ex.SELECTIVITY_GATE:g}.

At the query `=`, no edit on the whole-line, hinge, or `prompt` arm covers more than {max(CRIT[(a, "query =", o)]["share"] for a in (ex.PRIMARY, "anchor-hinge", "anchor-prompt") for o in OPERATORS):.2f} of the way to the null. On `query-eq` and `every-eq` the projection there grades, to {CRIT[("anchor-query-eq", "query =", "projection")]["full"]:.2f} and {CRIT[("anchor-every-eq", "query =", "projection")]["full"]:.2f} at full dose, and the worst other op reaches {CRIT[("anchor-query-eq", "query =", "projection")]["worst"]:.2f} and {CRIT[("anchor-every-eq", "query =", "projection")]["worst"]:.2f}. On `prompt` the projection at every position grades, with the worst other op at {CRIT[("anchor-prompt", "every position", "projection")]["worst"]:.2f}.

At the example answers, the reported site, the full projection covers {", ".join(f"{arm_name(a)} {shares(a, 'example answers', 'projection')[-1]:.2f}" for a in E2_ARMS)} of the way to the null. So on the whole-line and hinge arms, where the edit at every position works and the one at the query `=` does not, editing the examples alone takes most of the op out: the query seems to take the op from the examples.

<details markdown="1"><summary>The drop on each other op against the posterior (preregistered)</summary>

The figure plots the drop on each other op under the full projection against the mean posterior on `{ex.ANCHORED_OP}` in the contexts of that op. Over the six other ops that posterior runs only from {POSTERIOR_OTHER.min():.3f} to {POSTERIOR_OTHER.max():.3f}, so the ops bunch on the x-axis and the figure cannot show whether the drop follows it. The next section asks the same question another way.

{confusion_figure()}

</details>

### Post hoc: what the model answers instead

The net drop says how much of `{ex.ANCHORED_OP}` an edit takes out, not what the model answers in its place. A pass over the same runs, added after the results were in, measures that: for each held-out context, the probability the model puts on the answer of each op, on the clean model and under the full projection at every position. Beside it is the Bayes predictor with `{ex.ANCHORED_OP}` removed from the posterior, which is what a model should answer once it no longer knows the op.

{answers_figure()}

On the whole-line and hinge arms, the edited model answers `{ex.ANCHORED_OP}` contexts much as that predictor does. The mass on the `{ex.ANCHORED_OP}` answer falls from {CONF_CLEAN[ex.PRIMARY][D, D]:.2f} to {CONF_EDIT[ex.PRIMARY][D, D]:.2f} on the whole-line arm and from {CONF_CLEAN["anchor-hinge"][D, D]:.2f} to {CONF_EDIT["anchor-hinge"][D, D]:.2f} on the hinge arm, and moves to the other ops in about the proportions of the predictor.

The other contexts hardly change: their mass on the true op moves from {other_diag(CONF_CLEAN[ex.PRIMARY]):.2f} to {other_diag(CONF_EDIT[ex.PRIMARY]):.2f} on the whole-line arm and from {other_diag(CONF_CLEAN["anchor-hinge"]):.2f} to {other_diag(CONF_EDIT["anchor-hinge"]):.2f} on the hinge arm, against {other_diag(CONF_CLEAN[ex.CONTROL]):.2f} to {other_diag(CONF_EDIT[ex.CONTROL]):.2f} on the control. So on these arms the edit seems to remove the op from the inference and leave the rest in place.

On `query-eq` and `every-eq` the same edit takes out every op: no cell of either matrix holds more than {max(CONF_EDIT["anchor-query-eq"].max(), CONF_EDIT["anchor-every-eq"].max()):.2f}, so the model stops giving an answer that any op would give. `prompt` sits between: its `{ex.ANCHORED_OP}` row spreads like the predictor, and the other contexts keep {other_diag(CONF_EDIT["anchor-prompt"]):.2f} on the true op, against {other_diag(CONF_CLEAN["anchor-prompt"]):.2f} clean.

### Post hoc: editing the no-emb arm

`no-emb` had the highest task score of any arm in S1 and kept the anchor. After the results were in, we added its three runs to the suppression pass, to see whether its anchor edits as well as the candidates. Nothing here was predicted or given a criterion in advance; the figure applies the E2 criteria for comparison.

{no_emb_figure()}

The edit takes `{ex.ANCHORED_OP}` out of `no-emb` faster than out of the candidates, and takes more of the other ops with it. At every position the projection covers {", ".join(f"{v:.2f}" for v in shares(NO_EMB, "every position", "projection"))} of the way to the target null over its four doses, against {", ".join(f"{v:.2f}" for v in shares(ex.PRIMARY, "every position", "projection"))} on the whole-line arm. It grades, but the worst other op reaches {CRIT_PH[(NO_EMB, "every position", "projection")]["worst"]:.3f} under the projection and {CRIT_PH[(NO_EMB, "every position", "repulsion")]["worst"]:.3f} under the repulsion, several times the gate of {ex.SELECTIVITY_GATE:g}.

The edit stays selective up to the half dose, where the projection covers {selective_max(NO_EMB, "projection")[0]:.2f} of the way to the null. On the candidates, a selective projection gets as far as {selective_max(ex.PRIMARY, "projection")[0]:.2f} on the whole-line arm (γ {selective_max(ex.PRIMARY, "projection")[1]:g}) and {selective_max("anchor-hinge", "projection")[0]:.2f} on the hinge arm (γ {selective_max("anchor-hinge", "projection")[1]:g}). So leaving out the embedding slice buys no edit the candidates lack.

In the matrices above, `no-emb` answers `{ex.ANCHORED_OP}` contexts much as the whole-line arm does once edited. Most of its spill is on `{NO_EMB_SPILL}` contexts, whose mass on their own answer falls from {CONF_CLEAN[NO_EMB][NO_EMB_SPILL_I, NO_EMB_SPILL_I]:.2f} to {CONF_EDIT[NO_EMB][NO_EMB_SPILL_I, NO_EMB_SPILL_I]:.2f}, against {CONF_CLEAN[ex.CONTROL][NO_EMB_SPILL_I, NO_EMB_SPILL_I]:.2f} to {CONF_EDIT[ex.CONTROL][NO_EMB_SPILL_I, NO_EMB_SPILL_I]:.2f} on the control.

The E1 alignment suggests why. At the example answers in blocks 1 and 2, the other ops sit higher on e₁ on `no-emb` ({", ".join(f"{v:.2f}" for v in other_at_examples(NO_EMB)[1:3])}) than on the whole-line arm ({", ".join(f"{v:.2f}" for v in other_at_examples(ex.PRIMARY)[1:3])}), so an edit there removes more of what the other ops use. Leaving the embedding slice out of the pull seems to make the early blocks hold the axis less selectively.

/// admonition | Note
Post hoc. `no-emb` would not meet the E2 criteria at every position: the projection and the repulsion both grade, and neither is selective.
///

## The verification rule (S2)

**The rule.** Ex-2.2.16's rule (d), unchanged. Verification lines stay in the corpus from round 3 on if they leave completion unchanged: the seed-mean held-out expected exact match on completion contexts is within the seed band of the arm without them, on both pairs (`control-verify` against `{ex.CONTROL}`, `anchor-verify` against `{ex.PRIMARY}`). The verification accuracy and the op margin of `anchor-verify` are reported with no gate.

**What we expect.** Completion is unchanged on both pairs. The verification arms see {1 - ex.VERIFY_RATE:.0%} of the completion lines the others do. On the old recipe that cost the control nothing measurable; at {ex.EPOCHS} epochs each line is seen more often, so the cost should be smaller still.

{s2_figure()}

{s2_table()}

**What we saw.** Both pairs are within their seed band. `control-verify` scores {S2["control-verify"]["diff"]:+.3f} against the control, within {S2["control-verify"]["band"]:.3f}, and `anchor-verify` scores {S2["anchor-verify"]["diff"]:+.3f} against the whole-line arm, within {S2["anchor-verify"]["band"]:.3f}. Verification accuracy is {S2["control-verify"]["acc"].mean():.2f} and {S2["anchor-verify"]["acc"].mean():.2f}, and the op margin of `anchor-verify` is {VERIFY_MARGIN:.3f}, against {WHOLE_MARGIN:.3f} for the whole-line arm.

<details markdown="1"><summary>Gate arithmetic</summary>

The seed band is for three seeds against five, with σ pooled over the pair.

{"".join(f"- `{v}` against `{arm_name(d['ref'])}`: |{d['diff']:+.4f}| {'≤' if d['within'] else '>'} {d['band']:.4f}, {'within' if d['within'] else 'outside'}." + chr(10) for v, d in S2.items())}
</details>

/// admonition | Decided
Completion is unchanged on both pairs within the seed band, so verification lines stay in the corpus.
///

## The leans stay with the control (H2)

**What we expect.** On the whole-line arm, the first-operand lean and the trailing-fragment lean stay within the seed band of the control, as ex-2.2.16 defined them. The newline mask now keeps a context from reading the one before it, which should make a cut-off fragment look less like a whole context than it did in ex-2.2.15.

{h2_figure()}

**What we saw.** On the whole-line arm the first-operand lean is {H2["op1"]["whole"]:.3f} against {H2["op1"]["ctrl"]:.3f} for the control, a difference of {H2["op1"]["whole"] - H2["op1"]["ctrl"]:+.3f} within the seed band of {H2["op1"]["band"]:.3f}. The trailing-fragment lean is {H2["fragment"]["whole"]:.3f} against {H2["fragment"]["ctrl"]:.3f}, {H2["fragment"]["whole"] - H2["fragment"]["ctrl"]:+.3f} within {H2["fragment"]["band"]:.3f}. Among the other anchored arms, compared with the control at their own seed band, only {" and ".join(f"`{arm_name(a)}`" for a, _ in H2_OUTSIDE)} fall outside, both below the control on the first-operand lean.

<details markdown="1"><summary>Gate arithmetic</summary>

Seed band for five seeds against five, σ pooled over the whole-line arm and the control.

- First-operand lean: |{H2["op1"]["whole"]:.4f} − {H2["op1"]["ctrl"]:.4f}| = {abs(H2["op1"]["whole"] - H2["op1"]["ctrl"]):.4f} ≤ {H2["op1"]["band"]:.4f}: within.
- Trailing-fragment lean: |{H2["fragment"]["whole"]:.4f} − {H2["fragment"]["ctrl"]:.4f}| = {abs(H2["fragment"]["whole"] - H2["fragment"]["ctrl"]):.4f} ≤ {H2["fragment"]["band"]:.4f}: within.

</details>

/// admonition | Pass
Both leans of the whole-line arm stay within the seed band of the control.
///

## Discussion

Most of the recipe goes forward to round 3: the control reproduces ex-2.2.19, and the whole-line label and the verification lines stay. Whether the anchor costs the task anything stays open, since the shortfall misses the tolerance but sits inside the seed band.

One seed lags on most arms and is still climbing at the last epoch, so five seeds at {ex.EPOCHS} epochs can't tell a small cost from a slow start. A longer schedule would let the slow seeds finish, and is the likeliest way to settle it.

On the whole-line and hinge arms the anchor sits on the example answers. These come before the query `=`, where the answer is predicted, so the op seems to be anchored upstream of the point we score. The edit works when applied everywhere, examples included; an edit at the query `=` alone has little to act on.

Where the edit works, the post hoc answer matrices suggest that the model answers as if it no longer knew the op but still weighed the other ops.

The arms that pull the anchor onto the query `=` also lift the `=` embedding, which every op shares. That may be why their edits spill onto the other ops.

An edit at every position on the hinge arm is the clearest lead for round 3. On the whole-line arm, whose pull is uncapped, the edit grades as well, but it spills just over the gate. So a cap on the alignment in [{ex.HINGE_CAP:g}, 1), where 1 would be no cap, may keep the selectivity with more of the alignment.

The `no-emb` arm is the open puzzle: it had the highest task score of any arm, but its anchor edits less selectively. Perhaps leaving the embedding slice out of the pull makes the early blocks overcompensate, holding the axis for the other ops too. Or the anchor weight was set with the embedding in the pull, and `no-emb` would edit more selectively at a lower weight. This experiment can't separate the two.

## Method

### The corpus and the posterior

The corpus is ex-2.2.18's seven-op corpus at the center condition, which ex-2.2.19 also trained on: one context per line, 300,000 contexts, three examples and ρ = {ex.CENTRE[1]:g}, cube noise at κ = {ex.CUBE_RATE:g}, and stochastic rounding. The held-out and probe sets are ex-2.2.18's. The verification arms train on the same generator with {ex.VERIFY_RATE:.0%} of contexts written as verification lines.

Dropping four ops leaves the posterior on `{ex.ANCHORED_OP}` much as it was: on the seven-op table {BANDS7[1]:.0%} of `{ex.ANCHORED_OP}` contexts sit in the middle band, against {BANDS11[1]:.0%} on eleven ops, with {BANDS7[0]:.0%} below it and {BANDS7[2]:.0%} above. On these sampled contexts the Bayes ceiling is {C7["ceiling"]:.3f} and the floor {C7["floor"]:.3f}; on the held-out set of ex-2.2.19, the ceiling is {REF19_CEIL:.3f}.

But the posterior takes few distinct values: {STEP7:.0%} of the contexts in the middle band sit in one step between 0.85 and 0.95, so the stimulus has about three levels. That matters for the graded stimulus of round 3, but not for the rules here.

{posterior_figure()}

<details markdown="1"><summary>The neighboring conditions</summary>

{grid_table()}

</details>

### The measurements

Ex-2.2.16's measurements, scored on its held-out sets at the end of training, with the mask on in every arm: expected exact match against the ceiling and floor, the calibration KL, the alignment by position and slice, the alignment against the posterior, the op margin, the two leans, the suppression pass, and verification accuracy. The op margin and the leans are also recorded through training on a fixed probe set. Comparisons are between seed means, with the seed band as the resolution; the seed standard deviation is pooled over the two arms compared.

The verification arms are scored for expected exact match, the op margin, the alignment, and the leans on the main held-out set (the same contexts as every other arm), and for verification accuracy on the verification held-out set.

### The new variants

`prompt`, `query-eq`, and `every-eq` are masks, by role within the context, on the positions a labelled context pulls (roles as listed under [Conditions](#conditions)). They run through the same pooled term as every other variant, which gives each labelled context the same total pull however many positions share it.

On `query-eq` the pool has one position, so the whole pull lands on the query `=`: the variant changes how hard that position is pulled as well as where. On `every-eq` the pool has four positions, so the query `=` gets a quarter of that.

E2 scores `query-eq` and `every-eq` without separating the two changes; E1 shows the alignment it reaches at the query `=` beside the other arms.

### Budget

{ex.N_RUNS} runs at about \${ex.cost_per_run():.2f} each on an L4, from ex-2.2.19's unanchored runs at {ex.EPOCHS} epochs; we expect the anchored step to cost about the same and will check on the first runs. The suppression pass is scoring only, on {sum(ex.arm(a).seeds for a in ex.SUPPRESSION_ARMS)} checkpoints. Under \$10 in all.
"""
