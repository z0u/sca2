# title: Ex 2.2.16: the in-context grammar pilot

# The design constants come from `experiment.py` beside this script and the posterior computation from
# `posterior.py` (the directory of the script is on sys.path while it runs). The method section is computed from the
# op table alone, with no training; the results of rule (a) come from the eval that stage C published to the store.
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
import posterior as P
from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed

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


def cond_name(k: int, rho: float) -> str:
    """How a corpus condition is named in tables and figures: `k3-r0.3` is three examples at ρ = 0.3."""
    return f"k{k}-r{rho:g}"


def rho_colors(n: int) -> list:
    """Ordered shades for the replacement rates, from a colormap whose ends both survive the page background."""
    lo, hi = light_dark((0.0, 0.85), (0.3, 1.0))
    return [plt.get_cmap("viridis")(t) for t in np.linspace(lo, hi, n)]


def rule_color() -> str:
    return light_dark("#333", "#ddd")


# --- The posterior scan ---------------------------------------------------------------------------------------


@memo
def answer_table() -> P.AnswerTable:
    return P.build_table(ex.TABLE)


@memo
def table_constants(table: P.AnswerTable) -> tuple[float, float, float]:
    """The told-op ceiling, its mode form, and the analytic floor; the floor is a few seconds of lookups."""
    return table.told_op(), table.mode_match(), table.floor()


TABLE = answer_table()
TOLD_OP, TOLD_OP_MODE, FLOOR = table_constants(TABLE)


@memo
def scan(table: P.AnswerTable, k_grid: tuple, rho_grid: tuple, n: int, seed: int) -> dict:
    """For every grid point: the posterior on the true op over *n* sampled contexts, and the mean Bayes ceiling,
    its hard-accuracy form, and the floor over their queries. Each point draws from its own stream, so the scan
    does not depend on the order the grid is walked.
    """
    out = {}
    for i, k in enumerate(k_grid):
        for j, rho in enumerate(rho_grid):
            rng = np.random.default_rng([seed, i, j, 0])
            ctx = P.sample_contexts(table, n, k, rho, 0.0, rng)
            post = P.posterior(table, ctx, rho, 0.0)
            out[(k, rho)] = {
                "post_true": post[np.arange(n), ctx.true_op].astype(np.float32),
                "ceiling": float(P.expected_match(table, ctx, post).mean()),
                "mode_ceiling": float(P.mode_match(table, ctx, post).mean()),
                "floor": float(P.floor(table, ctx).mean()),
            }
    return out


@memo
def cube_scan(table: P.AnswerTable, conditions: tuple, cube_grid: tuple, n: int, seed: int) -> dict:
    """The ceiling at each proposed condition under cube noise at each rate: for the posterior that knows the rate,
    and for the one that ignores it (computed as if κ were zero on the same contexts). The stream at κ = 0 is the
    one `scan` uses at the same grid point, so the two tables agree there.
    """
    out = {}
    for k, rho in conditions:
        for kappa in cube_grid:
            rng = np.random.default_rng([seed, ex.K_GRID.index(k), ex.RHO_GRID.index(rho), round(kappa * 1000)])
            ctx = P.sample_contexts(table, n, k, rho, kappa, rng)
            out[(k, rho, kappa)] = {
                "ceiling": float(P.ceiling(table, ctx, rho, kappa).mean()),
                "ignoring": float(P.ceiling(table, ctx, rho, 0.0).mean()),
            }
    return out


@memo
def per_op(table: P.AnswerTable, k: int, rho: float, n: int, seed: int) -> dict:
    """The ceiling, the told-op ceiling, the floor, and the spread of the posterior on the true op, for contexts of
    each op in turn at one grid point.
    """
    out = {}
    for o, name in enumerate(table.names):
        rng = np.random.default_rng([seed, 200, o])
        ctx = P.sample_contexts(table, n, k, rho, 0.0, rng, true_op=o)
        post = P.posterior(table, ctx, rho, 0.0)
        pt = post[:, o]
        out[name] = {
            "ceiling": float(P.expected_match(table, ctx, post).mean()),
            "told": float((table.prob[o] ** 2).sum(-1).mean()),
            "floor": float(P.floor(table, ctx).mean()),
            "sd": float(pt.std()),
            "mid": float(((pt >= ex.MIDDLE_BAND[0]) & (pt <= ex.MIDDLE_BAND[1])).mean()),
        }
    return out


@memo
def calibration_refs(table: P.AnswerTable, conditions: tuple, kappa: float, n: int, seed: int) -> dict:
    """At each proposed condition, under the cube-noise rate of record: the irreducible loss H(q), the mean KL
    divergence from the Bayes predictive to each reference predictor, and the expected exact match of the
    predictor that ignores one example (its shortfall from the ceiling is a scale for the margin of rule (a)).
    """
    out = {}
    for i, (k, rho) in enumerate(conditions):
        rng = np.random.default_rng([seed, 300, i])
        ctx = P.sample_contexts(table, n, k, rho, kappa, rng)
        refs = P.reference_predictors(table, ctx, rho, kappa)
        q = refs.pop("bayes")
        hidden = P.Contexts(ctx.true_op, ctx.ex_pair[:, :-1], ctx.ex_color[:, :-1], ctx.query_pair)
        out[(k, rho)] = {
            "entropy": float(P.entropy(q).mean()),
            "kl": {name: float(P.kl(q, p).mean()) for name, p in refs.items()},
            "ceiling": float(P.expected_match(table, ctx, P.posterior(table, ctx, rho, kappa)).mean()),
            "one_hidden": float(P.expected_match(table, hidden, P.posterior(table, hidden, rho, kappa)).mean()),
        }
    return out


SCAN = scan(TABLE, ex.K_GRID, ex.RHO_GRID, ex.N_CONTEXTS, ex.POSTERIOR_SEED)
CUBE = cube_scan(TABLE, ex.GRAMMAR_CONDITIONS, ex.CUBE_GRID, ex.N_CONTEXTS, ex.POSTERIOR_SEED)
PER_OP = per_op(TABLE, *ex.CENTRE, ex.N_CONTEXTS // 4, ex.POSTERIOR_SEED)
CALIB = calibration_refs(TABLE, ex.GRAMMAR_CONDITIONS, ex.CUBE_RATE, ex.N_CONTEXTS // 2, ex.POSTERIOR_SEED)


def record_ceiling(k: int, rho: float) -> float:
    """The ceiling of record at a proposed condition: the calibrated Bayes ceiling at the cube-noise rate of the corpus."""
    return CUBE[(k, rho, ex.CUBE_RATE)]["ceiling"]


def record_floor(k: int, rho: float) -> float:
    """The floor does not depend on the examples, so cube noise leaves it where the κ = 0 scan put it."""
    return SCAN[(k, rho)]["floor"]


def spread(k: int, rho: float) -> float:
    """The standard deviation of the posterior on the true op across contexts: the one-number spread."""
    return float(SCAN[(k, rho)]["post_true"].std())


def band_shares(k: int, rho: float) -> tuple[float, float, float]:
    """Shares of contexts below, inside, and above the middle band of the posterior on the true op."""
    pt = SCAN[(k, rho)]["post_true"]
    lo, hi = ex.MIDDLE_BAND
    return float((pt < lo).mean()), float(((pt >= lo) & (pt <= hi)).mean()), float((pt > hi).mean())


def ceiling(k: int, rho: float) -> float:
    return SCAN[(k, rho)]["ceiling"]


# --- Figures ------------------------------------------------------------------------------------------------


def posterior_figure() -> str:
    data = {kr: v["post_true"] for kr, v in SCAN.items()}
    lo3, mid3, hi3 = band_shares(*ex.CENTRE)
    alt = f"""
        Six panels, one per example count from {ex.K_GRID[0]} to {ex.K_GRID[-1]}, each showing the cumulative share
        of contexts against the posterior on the true op, one curve per replacement rate. With no noise the curves
        hug the right edge from three examples on; as the rate rises they spread across the range. At the center
        condition, {ex.CENTRE[0]} examples and ρ = {ex.CENTRE[1]:g}, {lo3:.0%} of contexts sit below the middle
        band, {mid3:.0%} inside it, and {hi3:.0%} above.
    """
    return posterior_draw(data, ex.K_GRID, ex.RHO_GRID, ex.MIDDLE_BAND, alt)


@memo
def posterior_draw(data: dict, k_grid: tuple, rho_grid: tuple, band: tuple, alt_text: str) -> str:
    @themed(
        name="posterior-ecdf",
        alt_text=alt_text,
        caption=f"""
            **The posterior on the true op, across contexts.** Each panel is one example count; each curve is one
            replacement rate ρ, shaded light to dark from 0 to {rho_grid[-1]:g}. A curve shows the share of contexts
            whose posterior on the true op is at most the value on the x-axis. The shaded band is the middle band,
            {band[0]:g} to {band[1]:g}: a curve that rises steeply inside it has many graded contexts, one that stays
            flat until the right edge has contexts that all but name the op. Sampled contexts, {ex.N_CONTEXTS:,}
            per curve.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.6), layout="constrained", sharex=True, sharey=True)
        axes = cast(np.ndarray, axes).ravel()
        colors = rho_colors(len(rho_grid))
        x = np.linspace(0, 1, 401)
        for ax, k in zip(axes, k_grid, strict=True):
            ax.axvspan(band[0], band[1], facecolor=light_dark("#000", "#fff"), alpha=0.06, lw=0, zorder=0)
            for rho, c in zip(rho_grid, colors, strict=True):
                pt = np.sort(data[(k, rho)])
                ax.plot(x, np.searchsorted(pt, x, side="right") / len(pt), color=c, lw=1.2, label=f"ρ = {rho:g}")
            ax.set_title(f"{k} example{'s' if k > 1 else ''}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        for ax in axes[3:]:
            ax.set_xlabel("posterior on the true op")
        for ax in axes[::3]:
            ax.set_ylabel("share of contexts")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def ceiling_figure() -> str:
    ceil = {kr: v["ceiling"] for kr, v in SCAN.items()}
    sd = {kr: spread(*kr) for kr in SCAN}
    best_k, best_rho = max(sd, key=lambda kr: sd[kr])
    alt = f"""
        Two line charts against the example count. Left, the Bayes ceiling in expected exact match: every curve rises
        with more examples and sits lower at a higher replacement rate, toward a dashed line at {TOLD_OP:.2f} for a
        model told the op, with the floor as a dotted line at {FLOOR:.2f}. Right, the spread of the posterior on the
        true op, which peaks at {sd[(best_k, best_rho)]:.2f} for {best_k} examples at ρ = {best_rho:g} and falls
        toward zero as the examples pin the op down.
    """
    return ceiling_draw(ceil, sd, ex.K_GRID, ex.RHO_GRID, TOLD_OP, FLOOR, ex.GRAMMAR_CONDITIONS, alt)


@memo
def ceiling_draw(
    ceil: dict, sd: dict, k_grid: tuple, rho_grid: tuple, told: float, floor: float, proposed: tuple, alt_text: str
) -> str:
    @themed(
        name="ceiling-grid",
        alt_text=alt_text,
        caption=f"""
            **The Bayes ceiling and the spread of the stimulus, over the grid.** One curve per replacement rate ρ,
            shaded as in the figure above. **Left:** the ceiling, the expected exact match of the predictor that
            answers with the posterior-weighted answer distribution. The dashed line is the same predictor told the
            op ({told:.3f}), the dotted line the floor, with no evidence ({floor:.3f}). **Right:** the standard
            deviation of the posterior on the true op across contexts. Ringed marks are the three proposed
            conditions.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        axes = cast(AxesRow, axes)
        colors = rho_colors(len(rho_grid))
        for ax, series, label in zip(
            axes, (ceil, sd), ("Bayes ceiling (EEM)", "spread of the posterior (sd)"), strict=True
        ):
            for rho, c in zip(rho_grid, colors, strict=True):
                ax.plot(k_grid, [series[(k, rho)] for k in k_grid], "-o", color=c, lw=1.2, ms=3.5, label=f"ρ = {rho:g}")
            for k, rho in proposed:
                ax.plot(k, series[(k, rho)], "o", ms=9, mfc="none", mec=rule_color(), mew=0.9, zorder=5)
            ax.set_xticks(list(k_grid))
            ax.set_xlabel("examples per context")
            ax.set_ylabel(label)
        axes[0].axhline(told, color=rule_color(), lw=0.9, ls="--", zorder=1)
        axes[0].axhline(floor, color=rule_color(), lw=0.7, ls=":", zorder=1)
        axes[0].set_ylim(0, 0.8)
        axes[1].set_ylim(0, 0.4)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def cube_figure() -> str:
    k0, r0 = ex.CENTRE
    drop = CUBE[(k0, r0, ex.CUBE_GRID[-1])]["ceiling"] - CUBE[(k0, r0, 0.0)]["ceiling"]
    alt = f"""
        A line chart of the Bayes ceiling against the cube-noise rate from 0 to {ex.CUBE_GRID[-1]:g}, one line per
        proposed condition. Every line falls gently; at the center condition the ceiling drops by {abs(drop):.3f}
        at the highest rate. Hollow marks, for a posterior that ignores the cube noise, sit a few thousandths above
        the filled ones.
    """
    return cube_draw(CUBE, ex.GRAMMAR_CONDITIONS, ex.CUBE_GRID, alt)


@memo
def cube_draw(cube: dict, conditions: tuple, cube_grid: tuple, alt_text: str) -> str:
    @themed(
        name="cube-noise",
        alt_text=alt_text,
        caption="""
            **What cube noise costs the ceiling.** One line per proposed condition. Filled marks: the ceiling for
            the posterior that knows the cube-noise rate κ. Hollow marks: the ceiling for the posterior that treats
            every odd example as replacement op noise, which is what a model that has not learned about cube noise
            would hold.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.6, 3.0), layout="constrained")
        colors = [light_dark(*c) for c in (("#2b6cb0", "#7fb3ff"), ("#c0392b", "#ff8a76"), ("#2e8b57", "#7fd8a4"))]
        for (k, rho), c in zip(conditions, colors, strict=True):
            ys = [cube[(k, rho, kap)]["ceiling"] for kap in cube_grid]
            ys_ign = [cube[(k, rho, kap)]["ignoring"] for kap in cube_grid]
            ax.plot(cube_grid, ys, "-o", color=c, lw=1.2, ms=4, label=cond_name(k, rho))
            ax.plot(cube_grid, ys_ign, "o", color=c, ms=4, mfc="none", mew=0.9)
        ax.set_xlabel("cube-noise rate κ")
        ax.set_ylabel("Bayes ceiling (EEM)")
        ax.set_xticks(list(cube_grid))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


# --- Tables -------------------------------------------------------------------------------------------------


def grid_table(stat, caption: str, fmt: str = "{:.3f}") -> str:
    head = ["examples", *(f"ρ = {rho:g}" for rho in ex.RHO_GRID)]
    rows = [[str(k), *(fmt.format(stat(k, rho)) for rho in ex.RHO_GRID)] for k in ex.K_GRID]
    return table_html(head, rows, caption)


def conditions_table() -> str:
    head = [
        "condition",
        "examples",
        "ρ",
        "tokens per line",
        "ceiling (EEM) ↑",
        "floor",
        "ceiling − floor",
        "hard-EM ceiling",
        "spread (sd)",
        "below band",
        "in band",
        "above band",
    ]
    rows = []
    for k, rho in ex.GRAMMAR_CONDITIONS:
        lo, mid, hi = band_shares(k, rho)
        c = ceiling(k, rho)
        rows.append(
            [
                f"`{cond_name(k, rho)}`",
                str(k),
                f"{rho:g}",
                str(ex.context_tokens(k)),
                f"{c:.3f}",
                f"{SCAN[(k, rho)]['floor']:.3f}",
                f"{c - SCAN[(k, rho)]['floor']:.3f}",
                f"{SCAN[(k, rho)]['mode_ceiling']:.3f}",
                f"{spread(k, rho):.3f}",
                f"{lo:.0%}",
                f"{mid:.0%}",
                f"{hi:.0%}",
            ]
        )
    return table_html(
        head,
        rows,
        f"The three proposed corpus conditions, at κ = 0. The ceiling and floor are the analytic bounds the control at each "
        f"condition is scored between (the bounds of record, at the cube-noise rate of the corpus, are under rule (a)); the hard-EM ceiling is the same predictor naming the mode, for comparison "
        f"with the pivot. The band columns are shares of contexts by posterior on the true op, below, inside, and "
        f"above {ex.MIDDLE_BAND[0]:g}–{ex.MIDDLE_BAND[1]:g}. The center condition is `{cond_name(*ex.CENTRE)}`.",
        ref_rows=frozenset({ex.GRAMMAR_CONDITIONS.index(ex.CENTRE)}),
    )


def per_op_table() -> str:
    head = ["op", "ceiling (EEM)", "told the op", "floor", "spread (sd)", "in band"]
    rows = [
        [
            f"`{name}`",
            f"{v['ceiling']:.3f}",
            f"{v['told']:.3f}",
            f"{v['floor']:.3f}",
            f"{v['sd']:.3f}",
            f"{v['mid']:.0%}",
        ]
        for name, v in PER_OP.items()
    ]
    return table_html(
        head,
        rows,
        f"Per true op at the center condition, `{cond_name(*ex.CENTRE)}`: the ceiling, the ceiling for a model told "
        f"the op (which is the rounding cap), the floor, the spread of the posterior on the op across its contexts, "
        f"and the share of its contexts in the middle band. Ops total on the grid have a told-op ceiling of 1. "
        f"`{ex.ANCHORED_OP}` is the anchored op.",
        ref_rows=frozenset({ex.OP_NAMES.index(ex.ANCHORED_OP)}),
    )


def cube_table() -> str:
    head = ["condition", *(f"κ = {kap:g}" for kap in ex.CUBE_GRID)]
    rows = [
        [f"`{cond_name(k, rho)}`", *(f"{CUBE[(k, rho, kap)]['ceiling']:.3f}" for kap in ex.CUBE_GRID)]
        for k, rho in ex.GRAMMAR_CONDITIONS
    ]
    return table_html(
        head,
        rows,
        "The Bayes ceiling at each proposed condition under cube noise at rate κ, for the posterior that knows κ.",
    )


def record_table() -> str:
    head = ["condition", "ceiling (EEM) ↑", "floor", "room", "pass line (ceiling − margin)", "margin / room"]
    rows = []
    for k, rho in ex.GRAMMAR_CONDITIONS:
        c, f = record_ceiling(k, rho), record_floor(k, rho)
        rows.append(
            [
                f"`{cond_name(k, rho)}`",
                f"{c:.3f}",
                f"{f:.3f}",
                f"{c - f:.3f}",
                f"{c - ex.CEILING_MARGIN:.3f}",
                f"{ex.CEILING_MARGIN / (c - f):.2f}",
            ]
        )
    return table_html(
        head,
        rows,
        f"The bounds of record for rule (a), at the cube-noise rate of the corpus, κ = {ex.CUBE_RATE:g}: the calibrated "
        f"Bayes ceiling, the floor, the room between them, the score a control has to reach, and the margin as a "
        f"share of the room (the pass line is a skill score of one minus this).",
        ref_rows=frozenset({ex.GRAMMAR_CONDITIONS.index(ex.CENTRE)}),
    )


def arms_table() -> str:
    head = ["arm", "group", "condition", "model", "anchored", "label", "corpus", "seeds", "what it asks"]
    rows = []
    for a in ex.ARMS:
        extras = [x for x, on in (("hinge", a.hinge), ("verification", a.verify), ("newline mask", a.mask)) if on]
        rows.append(
            [
                f"`{a.name}`",
                a.group,
                f"`{cond_name(*a.condition)}`",
                a.model,
                "yes" if a.anchored else "—",
                (f"`{a.label}`" if a.anchored else "—") + (" + hinge" if a.hinge else ""),
                ", ".join(x for x in extras if x != "hinge") or "—",
                str(a.seeds),
                a.note,
            ]
        )
    return table_html(
        head,
        rows,
        f"The {len(ex.ARMS)} arms, {ex.N_RUNS} runs in all. Every anchored arm anchors `{ex.ANCHORED_OP}` at "
        f"the center condition; `{ex.CONTROL}` is the reference for all of them, and `{ex.PRIMARY}` is the "
        f"whole-line arm the label, hinge, verification, and mask arms are compared with. Corpus: what the "
        f"corpus or attention of the arm adds; verification lines replace {ex.VERIFY_RATE:.0%} of the completion lines.",
        ref_rows=frozenset({i for i, a in enumerate(ex.ARMS) if a.name in (ex.CONTROL, ex.PRIMARY)}),
        text_cols=7,
    )


def calibration_table() -> str:
    head = [
        "condition",
        "H(q)",
        "floor",
        "commits to the MAP op",
        "one example ignored",
        "a tenth of the floor mixed in",
    ]
    rows = [
        [
            f"`{cond_name(k, rho)}`",
            f"{v['entropy']:.2f}",
            f"{v['kl']['floor']:.2f}",
            f"{v['kl']['map']:.2f}",
            f"{v['kl']['one-hidden']:.2f}",
            f"{v['kl']['mix-floor']:.3f}",
        ]
        for (k, rho), v in CALIB.items()
    ]
    return table_html(
        head,
        rows,
        f"The calibration scale, in nats, at κ = {ex.CUBE_RATE:g}: the irreducible loss H(q) of the Bayes "
        f"predictive q, then the mean KL divergence from q to four predictors that miss calibration in named ways. "
        f"A model is called calibrated under {ex.KL_CALIBRATED:g}. Sampled contexts, {ex.N_CONTEXTS // 2:,} per row.",
        ref_rows=frozenset({ex.GRAMMAR_CONDITIONS.index(ex.CENTRE)}),
    )


def context_roles(k: int) -> str:
    """The role labels of a context with *k* examples, in the notation of the method section."""
    sub = str.maketrans("123456", "₁₂₃₄₅₆")
    return ", ".join(f"a{i} ? b{i} = y{i}".translate(sub) for i in range(1, k + 1)) + ", then the query a ? b = y ⏎"


# --- The numbers the prose quotes -------------------------------------------------------------------------------

N = {
    "clean1_hi": band_shares(1, 0.0)[2],
    "clean3_hi": band_shares(3, 0.0)[2],
    "centre": dict(zip(("lo", "mid", "hi"), band_shares(*ex.CENTRE), strict=True)),
    "r35": dict(zip(("lo", "mid", "hi"), band_shares(3, 0.35), strict=True)),
    "ceil_clean3": ceiling(3, 0.0),
    "mode_clean3": SCAN[(3, 0.0)]["mode_ceiling"],
    "ceil_r35": ceiling(3, 0.35),
    "mode_r35": SCAN[(3, 0.35)]["mode_ceiling"],
    "ceil_centre": ceiling(*ex.CENTRE),
    "sd_centre": spread(*ex.CENTRE),
    "sd_max": max(spread(*kr) for kr in SCAN),
    "sd_max_at": max(SCAN, key=lambda kr: spread(*kr)),
    "floor_range": (min(v["floor"] for v in SCAN.values()), max(v["floor"] for v in SCAN.values())),
    "cube_cost": {
        kap: CUBE[(*ex.CENTRE, 0.0)]["ceiling"] - CUBE[(*ex.CENTRE, kap)]["ceiling"] for kap in ex.CUBE_GRID[1:]
    },
    "cube_ignoring": {
        kap: CUBE[(*ex.CENTRE, kap)]["ignoring"] - CUBE[(*ex.CENTRE, kap)]["ceiling"] for kap in ex.CUBE_GRID[1:]
    },
    "diff": PER_OP[ex.ANCHORED_OP],
    "record_centre": (record_ceiling(*ex.CENTRE), record_floor(*ex.CENTRE)),
    "one_hidden_shortfall": {kr: v["ceiling"] - v["one_hidden"] for kr, v in CALIB.items()},
    "calib_centre": CALIB[ex.CENTRE],
}
ROOM_CENTRE = N["record_centre"][0] - N["record_centre"][1]
COND_NAMES = [cond_name(k, rho) for k, rho in ex.GRAMMAR_CONDITIONS]
ALT_34 = (3, 0.4)
ALT_435 = (4, 0.35)


# --- The results of rule (a) ------------------------------------------------------------------------------------


def fetch_eval() -> dict:
    """The published eval JSON of stage C: one record per run, with its task scores and the arm and seed it belongs to."""
    store = project_store()
    ref = store.get_refs([ex.EVAL_REF])[ex.EVAL_REF]
    assert ref is not None, "ex-2.2.16 has not published its eval yet"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(store.get_many([(ref, Path(tmp) / "eval.json")])[0].read_text())


EVAL = fetch_eval()
RULE_A_ARMS = (*(f"control-{cond_name(k, rho)}" for k, rho in ex.GRAMMAR_CONDITIONS), "control-large")


def arm_scores(name: str, stat: str) -> list[float]:
    """Per-seed held-out *stat* ("eem" or "kl") over all contexts, in seed order."""
    runs = sorted((r for r in EVAL["runs"] if r["arm"] == name), key=lambda r: r["seed"])
    return [r["task"][stat]["all"] for r in runs]


@dataclass(frozen=True)
class Control:
    """One control arm scored under rule (a), per seed and against the bounds of record of its condition."""

    condition: tuple[int, float]
    model: str
    eem: list[float]
    kl: list[float]
    ceiling: float
    floor: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.eem))

    @property
    def pass_line(self) -> float:
        return self.ceiling - ex.CEILING_MARGIN

    @property
    def skill(self) -> float:
        return (self.mean - self.floor) / (self.ceiling - self.floor)

    @property
    def passes(self) -> bool:
        return self.mean >= self.pass_line


RULE_A = {
    name: Control(
        condition=ex.arm(name).condition,
        model=ex.arm(name).model,
        eem=arm_scores(name, "eem"),
        kl=arm_scores(name, "kl"),
        ceiling=record_ceiling(*ex.arm(name).condition),
        floor=record_floor(*ex.arm(name).condition),
    )
    for name in RULE_A_ARMS
}
N_PASS = sum(v.passes for v in RULE_A.values())
SKILL_RANGE = (min(v.skill for v in RULE_A.values()), max(v.skill for v in RULE_A.values()))
KL_RANGE = (min(min(v.kl) for v in RULE_A.values()), max(max(v.kl) for v in RULE_A.values()))
CTRL = RULE_A[ex.CONTROL]
LARGE = RULE_A["control-large"]


def rule_a_label(name: str) -> str:
    v = RULE_A[name]
    return f"{cond_name(*v.condition)}\n{v.model}"


def rule_a_figure() -> str:
    alt = f"""
        Two panels over four columns: the {ex.MODEL} control at {", ".join(COND_NAMES)}, then the {ex.LARGE_MODEL}
        control at {cond_name(*ex.CENTRE)}. Left, held-out expected exact match: every seed sits between
        {min(min(v.eem) for v in RULE_A.values()):.2f} and {max(max(v.eem) for v in RULE_A.values()):.2f}, a
        little under halfway from the floor near {FLOOR:.2f} to ceilings between
        {min(v.ceiling for v in RULE_A.values()):.2f} and {max(v.ceiling for v in RULE_A.values()):.2f}, and
        every column is inside the hatched failing region below its pass line. Right, the calibration KL: every seed
        sits between {KL_RANGE[0]:.2f} and {KL_RANGE[1]:.2f} nats, far above the threshold of {ex.KL_CALIBRATED:g}.
    """
    return rule_a_draw(
        {n: asdict(v) | {"pass_line": v.pass_line} for n, v in RULE_A.items()},
        tuple(rule_a_label(n) for n in RULE_A),
        TOLD_OP,
        alt,
    )


@memo
def rule_a_draw(data: dict, labels: tuple, told: float, alt_text: str) -> str:
    @themed(
        name="rule-a",
        alt_text=alt_text,
        caption=f"""
            **The controls against the ceiling.** One column per control, the {ex.LARGE_MODEL} control at the right
            after a gap. **Left:** held-out expected exact match, seeds as faded dots and the seed mean as a bar.
            Per column, the ceiling of record is the dashed segment, the floor the dotted one, and the region below
            the pass line (the ceiling less {ex.CEILING_MARGIN:g}) is hatched. The dash-dot rule is the ceiling for
            a model told the op ({told:.3f}). **Right:** the calibration KL in nats, with the threshold of
            {ex.KL_CALIBRATED:g} as a dashed rule.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        axes = cast(AxesRow, axes)
        xs = np.array([0.0, 1.0, 2.0, 3.4])
        ink = rule_color()
        rng = np.random.default_rng(0)
        for x, v in zip(xs, data.values(), strict=True):
            jitter = rng.uniform(-0.08, 0.08, len(v["eem"]))
            for ax, stat in zip(axes, ("eem", "kl"), strict=True):
                ax.plot(x + jitter, v[stat], "o", color=ink, alpha=0.4, ms=4, mew=0)
                ax.plot([x - 0.22, x + 0.22], [np.mean(v[stat])] * 2, color=ink, lw=2)
            a = axes[0]
            a.plot([x - 0.34, x + 0.34], [v["ceiling"]] * 2, color=ink, lw=0.9, ls="--")
            a.plot([x - 0.34, x + 0.34], [v["floor"]] * 2, color=ink, lw=0.7, ls=":")
            a.fill_between(
                [x - 0.34, x + 0.34], 0, v["pass_line"], facecolor="none", edgecolor=ink, alpha=0.25, hatch="///", lw=0
            )
        axes[0].axhline(told, color=ink, lw=0.7, ls="-.", zorder=1)
        axes[0].set_ylim(0, 0.8)
        axes[0].set_ylabel("held-out EEM")
        axes[1].axhline(ex.KL_CALIBRATED, color=ink, lw=0.9, ls="--")
        axes[1].set_ylim(0, 1.2)
        axes[1].set_ylabel("calibration KL (nats)")
        for ax in axes:
            ax.set_xticks(xs, labels, fontsize=7)
            ax.set_xlim(-0.6, 4.0)
        return fig

    return _plot()


def rule_a_table() -> str:
    head = ["control", "model", "EEM per seed", "seed mean ↑", "pass line", "skill score", "spread", "KL ↓"]
    rows = [
        [
            f"`{cond_name(*v.condition)}`",
            v.model,
            ", ".join(f"{s:.3f}" for s in v.eem),
            f"{v.mean:.3f}",
            f"{v.pass_line:.3f}",
            f"{v.skill:.2f}",
            f"{spread(*v.condition):.2f}",
            f"{np.mean(v.kl):.2f}",
        ]
        for v in RULE_A.values()
    ]
    return table_html(
        head,
        rows,
        "Rule (a), scored. Held-out expected exact match over all contexts per seed and its seed mean, the pass line "
        "of the condition, the skill score of the seed mean, the spread of the posterior on the true op (the "
        "criterion among passing conditions), and the seed-mean calibration KL in nats. No row passes.",
        ref_rows=frozenset({RULE_A_ARMS.index(ex.CONTROL)}),
        text_cols=3,
    )


rf"""
# Ex 2.2.16: the in-context grammar pilot

/// tip |
<!-- tl;dr -->
The first experiment on the grammar from the [D2.2 pivot](/docs/m2/d2.2/pivot.md): the model infers the op from a few solved examples, with no word to name it. We trained the control at three conditions of example count and replacement rate, and anchored `{ex.ANCHORED_OP}` under the whole-line label and its variants. Every control got a little under halfway from the floor to the Bayes ceiling (skill scores {SKILL_RANGE[0]:.2f} to {SKILL_RANGE[1]:.2f}), well short of the pass line, and so did the {ex.LARGE_MODEL} control. So the pilot stops at its first rule: the grammar and recipe are reworked before anything is anchored, and the other rules are not scored. [Ex-2.2.17](/docs/m2/ex-2.2.17/report.py) takes up the rework.
///

This is round 2 of the [quick route](/docs/m2/d2.2/design.md#quick-route). The method section folds in the [posterior scouting report](/todo/science/scout-posterior-in-context-grammar.md): it computes the posterior over ops and the Bayes ceiling from the op table, with no training, and uses them to pick the three corpus conditions for the control.

## Findings

Each rule was to propose a setting for round 3. Only the first was scored.

- [The corpus rule (a)](#the-corpus-rule-a): no condition passes. The {ex.MODEL} control scored {CTRL.mean:.2f} at the center against a pass line of {CTRL.pass_line:.2f}, the other two conditions and the {ex.LARGE_MODEL} control fell as far short, and none is calibrated. Under the frozen rule, this is the outcome that adds a round.
- Rules (b) to (e) and the two predictions: not reached. They compare anchored arms with a control that has learned about half of what its examples allow, so the comparison would mix the effect of the anchor with unfinished training.

## How to read this report

This was a preregistration. The conditions, the five rules with their thresholds, and the two predictions were frozen at commit `49e59eb`, before any run; later edits to them are immaterial or are marked here. The [method](#method) section on the posterior and the ceiling was computed before training.

Every arm trained and was evaluated, and the eval is in the store, but only rule (a) is scored here. The sections for the other rules keep their frozen text, each with a note that it was not reached. Terms are defined where they first appear and collected in the [glossary](#glossary).

## Why this experiment

[Ex-2.2.14](/docs/m2/ex-2.2.14/report.py) anchored an op and every gate passed, but the anchor went to the word that names the op, so the anchored concept was an attribute of one token. The [pivot](/docs/m2/d2.2/pivot.md) takes op words out of the grammar.

Each line is one context: a few solved examples of one op, written with `?` in place of the op word, then a query under the same op. The op is inferred from the examples, and the anchored concept is "the op in this context is `{ex.ANCHORED_OP}`", which no token names.

This pilot scouts the posterior (see [Method](#method) below), trains the new-grammar control, smoke-tests anchoring the inferred op, and tries the label variants.

The control is the regression check; a grammar this different needs its own, as ex-2.2.3 did. No model trained on the corpus can know more about the op than the examples say, and the posterior captures that. So instead of fixed accuracy numbers, the task gate is how far the control falls short of the ceiling the posterior sets, with an analytic floor beside it.

The crop policy starts as `{ex.CROP_POLICY}`, which pulls only labelled lines wholly inside the training window. [Ex-2.2.15](/docs/m2/ex-2.2.15/report.py) found that lines cut short by the window show the first-operand lean (a leak toward the anchor). Its review chose `whole` over `half`, which its rule had set, because a whole line always shows its evidence, which matters more once the evidence is a set of examples, and its discussion asks the pilot to watch the trailing-fragment lean as well.

## Glossary

<dl>
<dt>Context</dt>
<dd>One line of the corpus: <em>k</em> examples of one op, then a query. Its op is inferred from the examples.</dd>
<dt>Example</dt>
<dd>A solved equation inside a context, <code>a ? b = y,</code>: the evidence the op is inferred from.</dd>
<dt>Query</dt>
<dd>The last equation of a context, whose answer the model completes. It is always clean.</dd>
<dt>Replacement op noise</dt>
<dd>Showing, in some examples, the answer another op would give, at a rate ρ per example. It spreads the posterior over ops, so it grades the stimulus.</dd>
<dt>Cube noise</dt>
<dd>Showing, in some examples, a color drawn from the whole grid, at a rate κ. Usually no op produces it, so it removes the evidence of an example without pointing anywhere else.</dd>
<dt>Posterior on the true op</dt>
<dd>How strongly the examples of a context point to the op that generated it, computed from the op table under the stochastic rounding and replacement op noise the corpus uses. For <code>{ex.ANCHORED_OP}</code> contexts it is the graded stimulus.</dd>
<dt>Bayes ceiling</dt>
<dd>The expected exact match on the query of the predictor that holds the posterior over ops and answers with the posterior-weighted answer distribution: the score of a calibrated model that has learned everything the examples say, which is where cross-entropy training aims.</dd>
<dt>Floor</dt>
<dd>The same predictor with no evidence: uniform over ops, so its answer distribution is the op-marginal. It is what a model that ignores the examples can score.</dd>
<dt>Skill score</dt>
<dd>Where a score sits between the floor (0) and the ceiling (1): (score − floor) / (ceiling − floor).</dd>
<dt>Expected exact match (EEM)</dt>
<dd>The probability mass the model puts on the colors the answer of the query can be under the true op, weighted by how often the op gives each, per context, then averaged over held-out contexts. The task metric of every report since ex-2.2.9; the ceiling and the floor are in the same units.</dd>
<dt>KL divergence</dt>
<dd>How far one distribution is from another, in nats: KL(q ‖ p) = Σ<sub>y</sub> q(y) log(q(y) / p(y)), zero when they agree. Here q is the Bayes predictive on a query and p the model's answer distribution, and the mean over contexts is the model's excess loss over a calibrated one, so 0.05 nats is a twentieth of a nat of loss the model could still shed. The <a href="#the-calibration-check">calibration check</a> gives reference values.</dd>
<dt>Alignment</dt>
<dd>The cosine between a state and e₁, the anchored axis: 1 when the state points along the axis, 0 when perpendicular. Measured per position and slice.</dd>
<dt>Lean (ᾱ)</dt>
<dd>The mean alignment at a role over the held-out contexts of every op: a leak toward the anchor at a site that carries no evidence about the op. The <em>first-operand lean</em> is ᾱ at the first token of a context, at the last block; the <em>trailing-fragment lean</em> is ᾱ over every position, at the last block, of a context cut to start partway through, fed on its own. Both from ex-2.2.15.</dd>
<dt>Op margin</dt>
<dd>How far the contexts of the anchored op sit along e₁ beyond the rest: per slice, the mean alignment over the contexts of the anchored op minus the mean over the contexts of all eleven ops, at the role where that gap is largest, averaged over slices. The quantity the anchor term optimizes, so it checks that the pull landed. The statistic of ex-2.2.14, with roles running over the whole context.</dd>
<dt>Seed band</dt>
<dd>The smallest difference between two seed means that a comparison resolves: {ex.SEED_BAND_SD:g}σ√(2/n) at n seeds per arm, with σ the seed standard deviation pooled over the two arms. Rules (b) and (d) use it on expected exact match at the center condition.</dd>
</dl>

## Conditions

The pilot trains {len(ex.ARMS)} arms at {ex.SEEDS} seeds each, {ex.N_RUNS} runs. The control runs at the three corpus conditions the method section proposes, `{"`, `".join(COND_NAMES)}` (three or four examples per context, at a replacement rate of 0.2 or 0.3), at {ex.MODEL}, and once more at {ex.LARGE_MODEL} at the center. The larger control is wider rather than deeper: the part of the task we are least sure a small model can do is the lookup of each example against eleven ops, which we expect to need width more than depth, and keeping four blocks keeps the slice axis the same for every arm. It trains either way, so that a miss under rule (a) does not cost a round, and is scored only if rule (a) needs it. Everything else runs at the center condition, `{cond_name(*ex.CENTRE)}`.

The anchored arms all anchor `{ex.ANCHORED_OP}` on e₁ with the recipe from ex-2.2.14 and crop policy `{ex.CROP_POLICY}`, and differ in what the label covers: the whole context, or one of the four [label variants](/todo/science/label-variants-in-context-op.md). A hinge arm caps the whole-line pull at an alignment of {ex.HINGE_CAP:g}, so no state is asked to be all concept. Two pairs add something to the corpus or the model on both the control and the whole-line arm: verification lines, and the newline mask, under which attention does not cross a line break.

Ex-2.2.15 also carried an oracle arm, `knowable`, which pulled a whole labelled line only when its op was in view. Under `{ex.CROP_POLICY}` every pulled context is wholly in view, so on this grammar that arm is the same as the whole-line arm, and the pilot leaves it out.

{arms_table()}

**The labeller.** A context of `{ex.ANCHORED_OP}` draws a label with probability {ex.LABEL_RATE:g}, keyed on the op array beside the corpus rather than on any token; no context of another op is labelled. Under the whole-line label every token of a labelled context is pulled through the pooled anchor term, which asks the context to align somewhere in its span. Variant (a) leaves the embedding slice out of the pull; (b) pulls the latter half of the context; (c) pulls a position once the posterior on `{ex.ANCHORED_OP}` given the examples before it reaches {ex.PREFIX_THRESHOLD:g}; (d) makes the draw depend on the evidence: a context of `{ex.ANCHORED_OP}` is labelled with a probability proportional to its posterior on `{ex.ANCHORED_OP}`, with the constant set so that, on average, {ex.LABEL_RATE:.0%} of them are labelled, as in the other arms. A context whose examples barely fit `{ex.ANCHORED_OP}` is then rarely labelled, and one whose examples all but name it is labelled a little more often than in the other arms.


**The seeds.** Three per arm, all fresh; no arm is served from the store. Comparisons between arms are between seed means, with the seed band as the resolution.

## The corpus rule (a)

**The rule.** A condition *passes* when the seed-mean held-out expected exact match of the {ex.MODEL} control is at least the ceiling of record less {ex.CEILING_MARGIN:g}. Among the passing conditions, the one with the widest spread of the posterior on the true op (its standard deviation across contexts, from the method section; the middle-band share breaks a tie) goes forward. Spread is the criterion because it is a property of the corpus: it says how graded the stimulus is, and round 3 measures the anchor against that grading. The calibration KL (below) is a property of the model, and would favor the condition with the least noise, where there is least to infer and least grading. So the KL is reported beside each condition, and a control that passes without being calibrated is flagged. If none passes, the {ex.LARGE_MODEL} control at the center is scored on the same line, and goes forward with the center condition if it passes; if it also falls short, the grammar is reworked before anything is anchored. That is the one outcome that adds a round.

The margin is one-sided. The calibrated ceiling is not the most a model can score: a model that sharpens past calibration can reach the hard-EM ceiling, which is higher (the method section has both). So a control above the ceiling passes, and the calibration check below says whether it got there by sharpening, which is worth knowing before round 3 reads its answer distribution. The bounds of record at each condition, and the score the control has to reach:

{record_table()}

At the center the margin is about {ex.CEILING_MARGIN / ROOM_CENTRE:.0%} of the room between floor and ceiling. For scale, a predictor that ignores one of the three examples and is otherwise Bayes-optimal gives up {N["one_hidden_shortfall"][ex.CENTRE]:.3f} there, so a control that passes has learned to use every example, near enough.


**What we expect.** The {ex.MODEL} control passes at every condition, and `{cond_name(*ex.CENTRE)}` goes forward. The pivot expects the model to be enough because there are no variables to track; the inference is a lookup over eleven ops per example and a pool over examples. A control that passes at `{COND_NAMES[0]}` and misses at the center would say the noise, not the grammar, is what costs it, and the fallback within the same block size is already in the table.

**The calibration check.** The check is the mean over held-out contexts of the KL divergence from the Bayes predictive $q$ (the posterior-weighted answer distribution on the query) to the model's answer distribution $p$ at the query `=`, in nats. Averaged over contexts, it is the model's cross-entropy on the answer less the Bayes-optimal cross-entropy: the loss the model could still shed. A control is *calibrated* at a condition when it is under {ex.KL_CALIBRATED:g} nats. The [method](#the-calibration-check) section gives the scale: a tenth of the floor mixed into the Bayes answer costs about {N["calib_centre"]["kl"]["mix-floor"]:.3f} at the center, ignoring one example {N["calib_centre"]["kl"]["one-hidden"]:.2f}, and committing to the most probable op {N["calib_centre"]["kl"]["map"]:.2f}. We expect the control to be calibrated at every condition it passes; a control that passes and is not calibrated has sharpened, and the direction of the miss (the KL grows fastest where the model puts no mass on an answer $q$ allows, so a sharpened model shows it more than a hedging one) is reported beside it.

**The result.** No condition passes, and the {ex.LARGE_MODEL} control misses too:

{rule_a_figure()}

{rule_a_table()}

The shortfall is large and much the same everywhere. Every control has a skill score between {SKILL_RANGE[0]:.2f} and {SKILL_RANGE[1]:.2f}, a little under halfway from the floor to the ceiling, and every seed sits far below its pass line. Width did not help: the {ex.LARGE_MODEL} control at the center scored {LARGE.mean:.3f}, against {CTRL.mean:.3f} for the {ex.MODEL} control, a difference inside the spread of its own seeds ({min(LARGE.eem):.3f} to {max(LARGE.eem):.3f}). The calibration KL is {KL_RANGE[0]:.2f} to {KL_RANGE[1]:.2f} nats across the seeds, many times the threshold of {ex.KL_CALIBRATED:g}.

So the grammar is reworked before anything is anchored, as the rule says. The miss alone does not say whether the grammar or the training recipe is the limit; the [discussion](#discussion) has what came next.

## The label rule (b)

**The rule.** The whole-line label stays the primary unless a variant *clears the task gate*: its seed-mean held-out expected exact match at the center, over all contexts, exceeds that of the whole-line arm by more than the seed band, while it *holds the anchor*: its op margin is at least {ex.MARGIN_KEEP:.0%} of the margin of the whole-line arm. If more than one variant qualifies, the one with the higher op margin goes forward. Variant (d) trains the grading that the [evidence section](#the-anchor-follows-the-evidence) reads, so it is reported on the same statistics and not promoted.

The op margin is the one ex-2.2.14 used, with the contexts of the anchored op as the labelled group and roles running over the whole context (the glossary has it). The variants narrow the pull, so their margins can only be lower by construction where the narrowed positions carried it; the bar asks that most of it survives.

**What we expect.** No variant clears the gate. Ex-2.2.14 found that the label share barely moves the margin and that the task is untouched at the rate of the primary, and the variants change less than that. What they may change is where the alignment sits: (a) should leave the embeddings clean, (b) and (c) should put less alignment on the first example, and (c) leaves the middle-band contexts unlabelled. Those are read in the alignment figures of the [evidence section](#the-anchor-follows-the-evidence); this rule scores the task and the margin only.

**Not reached.** Rule (a) did not pass, so this is not scored. The arms trained, and their eval is in the store.

## The hinge rule (c)

**The rule.** The uncapped whole-line arm *saturates* at the query `?` when the seed-mean alignment at that position, on the held-out contexts of `{ex.ANCHORED_OP}` at the last block, is at least {ex.SATURATION_LEVEL:g}. If it does, the hinge arm goes forward in its place, provided the hinge arm holds the anchor ({ex.MARGIN_KEEP:.0%} of the whole-line margin) and it is within the seed band of the whole-line arm on the task. If the whole-line arm does not saturate, it stays, and the hinge arm is reported.

Saturation is the dose collapse ex-2.2.14 found at the op word: a state that is all concept has no partial dose, because projecting e₁ out of it leaves a remainder too small to mean anything. The query `?` is a constant token with nothing else to hold, so the pooled pull may put the alignment of the whole context there. The hinge counts an alignment of {ex.HINGE_CAP:g} as full: in the anchor term only, a position pays 1 − cos/{ex.HINGE_CAP:g} in place of 1 − cos, and nothing above the cap. A sharp corner at the cap would switch the pull off there while the anti-subspace term keeps pushing the state back down, so a state near the cap could flip between pulled and not pulled from step to step. So the corner is rounded with a softplus, a smooth version of max(0, x), of width {ex.HINGE_SOFTNESS:g}, and the pull fades out over a narrow band around the cap, which gives the two terms a smooth point to balance at. We expect the pulled states to settle near {ex.HINGE_CAP:g}, keeping a part off the axis for the projection to land on. Measurements use the raw cosine.

**What we expect.** We do not predict either way. The pooled term concentrates the pull where alignment comes cheapest, which argues for saturation; the anti-subspace term penalizes the squared alignment of every state, which argues against it, and *red* under the same recipe reached about 0.5 at its own token. What we do expect is that the query `?` carries more alignment than any example position on the whole-line arm, since it is the first position at which the whole context has been seen and its state is free.

**Not reached.** Rule (a) did not pass, so this is not scored. The arms trained, and their eval is in the store.

## The verification rule (d)

**The rule.** Verification lines stay in the corpus from round 3 on if they *leave completion unchanged*: the seed-mean held-out expected exact match on completion contexts is within the seed band of the arm without them, on both pairs, `control-verify` against `{ex.CONTROL}` and `anchor-verify` against `{ex.PRIMARY}`. Both pairs have to hold; a corpus that costs the anchored arm and not the control would say the anchor and the second task compete.

A verification line replaces the completion of {ex.VERIFY_RATE:.0%} of contexts with a candidate answer, a marker, and a `TRUE` or `FALSE` verdict ([the pivot](/docs/m2/d2.2/pivot.md#a-verification-line)); the loss is masked on a `FALSE` candidate (the answer token only). The verification accuracy itself (the share of verdicts right, held out) is reported with no gate, as the first number D2.3 gets; so is the op margin of `anchor-verify` against the margin of the whole-line arm.

**What we expect.** Completion is unchanged on both pairs. The verification arms see {1 - ex.VERIFY_RATE:.0%} of the completion lines the others do, over the same steps, and the calibration look in ex-2.2.9 found that the task is limited by distinct lines rather than passes over them; so the cost, if there is one, shows on the ops that round the most. A miss on the control pair would send the rate down before round 3 rather than take verification out.

**Not reached.** Rule (a) did not pass, so this is not scored. The arms trained, and their eval is in the store.

## The operator rule (e)

**The rule.** A scoring-only suppression pass runs on the checkpoints of the pilot itself, on the whole-line arm, the hinge arm, and the control at the center: the projection along its dose axis (γ in {", ".join(f"{g:g}" for g in ex.DOSE_GAMMAS)}: the fraction of the component on e₁ removed), the repulsion along its dose axis (a landing at {", ".join(f"{b:g}".replace("-", "−") for b in ex.REPULSION_LANDINGS)}, for states above an alignment of {ex.REPULSION_THRESHOLD:g}), and the reflection (the projection at γ = {ex.REFLECT_GAMMA:g}, one dose, as a reference). Each acts at every slice, at three sites: the query `?`, the query `=`, and every position. The rule is scored on the every-position edit, on the anchored arm that rule (c) sends forward: the whole-line arm, or the hinge arm if the whole-line arm saturates. The other anchored arm is reported beside it. An operator *grades with dose* when the seed-mean drop in expected exact match on the held-out `{ex.ANCHORED_OP}` contexts is non-decreasing along its dose axis and reaches, at full dose, at least {ex.GRADING_MIN_DAMAGE:.0%} of the way from the clean score to the target null; it is *selective* when on each of the other ten ops the seed-mean drop is at most {ex.SELECTIVITY_GATE:g} at every dose, with the control under the same edit as the reference. The operator that does both goes forward with its dose axis; if both do, the projection, which is the simpler; if neither does, the operator for round 3 stays open, and the pilot reports why.

The target null is the answer a model that has lost `{ex.ANCHORED_OP}` and nothing else would give: the posterior-weighted answer distribution with `{ex.ANCHORED_OP}` removed and the rest renormalized. Its expected exact match on a context is the target the full-dose damage is measured against, per context, from the posterior of the method section.

**What we expect.** The projection grades if the query `?` does not saturate, and collapses (all of its damage in the last step of γ) if it does; the repulsion grades either way, since its dose is where the state lands rather than how much is removed. Selectivity should hold for both at the query sites, which the other ops' contexts share only as syntax, and is the open question at every position, where the edit touches states that carry the other ops. The two query sites are the first bypass measurement: an edit at `?` that does less than the same edit at `=` says the model reads the op from the examples rather than from `?`.

**Not reached.** Rule (a) did not pass, so this is not scored. The arms trained, and their eval is in the store.

## The anchor follows the evidence

**What we expect.** On the whole-line arm, the alignment at the query `=` on a held-out `{ex.ANCHORED_OP}` context rises with the posterior on `{ex.ANCHORED_OP}` across the middle band ({ex.MIDDLE_BAND[0]:g} to {ex.MIDDLE_BAND[1]:g}): binned by posterior, the seed-mean alignment at the last block is higher in each bin than the one below it. That is the graded stimulus, the part redness played for *red*, and it is a result the pull did not train for: the label is binary on the true op, so a context whose examples half-fit `{ex.ANCHORED_OP}` was pulled as hard as one that names it. A shortcut, such as a characteristic answer color, would likely not follow the posterior.

Variant (c) leaves the middle-band contexts unlabelled, so grading there under (c) is the cleaner version of the same result; variant (d) trains it and is the comparison. Alignment against the posterior on the *other* ops' contexts (where the posterior on `{ex.ANCHORED_OP}` is low but not zero) is reported beside it, since a graded response should be low and flat there.

**Not reached.** Rule (a) did not pass, so this is not scored. The arms trained, and their eval is in the store.

## The leans under `{ex.CROP_POLICY}`

**What we expect.** On the whole-line arm, both leans stay within the seed band of the control. Ex-2.2.15 found that under `all`, lines cut short by the window taught the model to lean toward e₁ at the first operand of every line, and more so on a trailing fragment shown alone; under `{ex.CROP_POLICY}` both leans sat inside the band of the control. This grammar gives a cut context more to lean with, since a window cuts more contexts than lines and a fragment of a context still looks like a context, so the lean is worth reading again. The first-operand lean is ᾱ at the first token of the context at the last block; the trailing-fragment lean is ᾱ over every position, at the last block, of a context cut to start at each example boundary, fed on its own; both over the held-out contexts of every op, with the mask arms measured with the mask on.

**Not reached.** Rule (a) did not pass, so this is not scored. The arms trained, and their eval is in the store.

## Discussion

The pilot expected the {ex.MODEL} control to reach its ceiling at every condition, on the reasoning that inferring the op is a lookup over eleven ops per example and a pool over examples. It got a little under halfway instead, at every condition and at twice the width. Validation loss was still falling at the end of training, so the first question after the pilot was whether the recipe or the task was the limit.

[Ex-2.2.17](/docs/m2/ex-2.2.17/report.py) is a scout on that question, on the center control. A newline mask, a lower peak learning rate, and eight times the steps lift it to about 0.45, still short of the pass line, and no schedule, length, width, or depth moves it further. Its scoring places the remaining gap on contexts whose examples settle the op, and on one op, `hsvmix`, that the model computes poorly. Its backlog items on the op set ([similar ops](/todo/science/drop-ops-with-similar-answers.md), [`hsvmix`](/todo/science/hsvmix-hue-precision.md)) are the grammar side of the rework.

None of rules (b) to (e) was tested here, so they can carry over to the next pilot unchanged.

## Method

### The posterior over ops

Each example is a pair of operands and an answer, and under stochastic rounding the answer is a draw from up to eight colors. So the likelihood of a shown answer *y* under an op *o* is the probability that rounding the result of *o* on that pair gives *y*, written $P_o(y \mid a, b)$ (`answer_dist` in `sca.data.ops`).

The corpus rounds each channel independently, in proportion to where the raw value sits between grid levels. The posterior uses the same rounding.
"""

r"""
Replacement op noise enters the likelihood as a mixture. With rate ρ, each example shows a draw from the distribution of another op instead, the other op uniform over the ten, so

$$
L_o(y \mid a, b) = (1 - \rho - \kappa)\, P_o(y \mid a, b) + \rho \, \mathbb{E}_{o' \neq o}\big[P_{o'}(y \mid a, b)\big] + \frac{\kappa}{216},
$$

where $L_o$ is the likelihood of the shown answer under op *o* in the noisy corpus, $\mathbb{E}_{o' \neq o}$ is the mean over the ten other ops, and κ is the cube-noise rate. The posterior over the eleven ops is the product of the example likelihoods under a uniform prior. This is the posterior a model trained on the noisy corpus can reach at best. It also keeps every op above zero whenever ρ or κ is above zero, which is what lets the target null weight the other ops by how nearly they fit on a context that fits one op alone.
"""

rf"""
Of two possible readings of ρ, we take the one whose numbers match the pivot: per example, with the replacing op uniform over the other ten. A replacement is invisible when the replacing op agrees with the true op on the pair, so slightly fewer than a fraction ρ of examples mislead, and the posterior is a smooth weighting rather than a count of the ops that fit.

Under this reading, the shares of contexts in each band of the posterior reproduce the scratch simulation in the pivot: one clean example puts the posterior above {ex.MIDDLE_BAND[1]:g} on {N["clean1_hi"]:.0%} of contexts and three do so on {N["clean3_hi"]:.0%}; at three examples and ρ = 0.35, {N["r35"]["hi"]:.0%} are above the band, {N["r35"]["mid"]:.0%} inside it, and {N["r35"]["lo"]:.0%} below.

The figure covers the whole grid, {len(ex.K_GRID)} example counts by {len(ex.RHO_GRID)} replacement rates, with the true op uniform over the table.

{posterior_figure()}

Clean examples pin the op down fast, so the example count alone grades very little; replacement op noise is what spreads the posterior across the range. The spread is widest, at a standard deviation of {N["sd_max"]:.2f}, for {N["sd_max_at"][0]} examples at ρ = {N["sd_max_at"][1]:g}; at the center condition it is {N["sd_centre"]:.2f}, with {N["centre"]["mid"]:.0%} of contexts in the middle band and {N["centre"]["lo"]:.0%} below it.

### The Bayes ceiling and the floor

A model that has learned everything the examples say answers the query with the answer distribution weighted by the posterior, $q(y) = \sum_o \pi_o P_o(y \mid c, d)$ for the query pair $(c, d)$. That is the distribution cross-entropy training converges to, and a calibrated model holds it.[^calibrated]

Our reports score expected exact match: the probability mass the model puts on the answers the true op can give, weighted by how often it gives them. So the ceiling is $\sum_y q(y)\, P_t(y \mid c, d)$ under the true op *t*, averaged over contexts.

Told the op, the same predictor scores $\sum_y P_t(y)^2$, which is {TOLD_OP:.3f} over the table. The shortfall from 1 comes from stochastic rounding, and it does not depend on the examples or the noise.

The floor (no evidence, uniform over ops) is the same at every grid point, {FLOOR:.3f} (the sampled values run from {N["floor_range"][0]:.3f} to {N["floor_range"][1]:.3f}).

[^calibrated]: A calibrated model is one whose stated probabilities match how often things actually happen: of the answers it gives 30% to, about 30% are right.

{ceiling_figure()}

With three clean examples the ceiling is {N["ceil_clean3"]:.3f}, within 0.02 of a model told the op, so almost all of the shortfall is rounding. Replacement op noise adds a shortfall that comes from inference: at three examples and ρ = 0.35 the ceiling falls to {N["ceil_r35"]:.3f}, and at the center condition it is {N["ceil_centre"]:.3f}.

<details markdown="1"><summary>The grid as tables</summary>

{grid_table(ceiling, "The Bayes ceiling in expected exact match at each grid point.")}

{grid_table(spread, "The spread of the posterior on the true op across contexts (standard deviation) at each grid point.")}

</details>

**The numbers in the pivot are the hard-accuracy form.** The pivot quotes a ceiling of about 0.74 at three clean examples, 0.58 at ρ = 0.35, and 0.75 for a model told the op. Those are what the same predictor scores when it puts all its mass on the mode of $q$: {N["mode_clean3"]:.3f}, {N["mode_r35"]:.3f}, and {TOLD_OP_MODE:.3f} here.

That form is higher because expected exact match is linear in the model distribution, so the metric itself is maximized by a model that names one color. The calibrated predictor spreads its mass over every rounding of the answer, and scores about 0.05 less at three clean examples and 0.1 less at ρ = 0.35.

The ceiling of record is the calibrated one, since that is where training aims. A control scoring above it would have sharpened past calibration, and the calibration check in the measurements would show that. The tables keep the hard form as the maximum of the metric.

Per op, the ceiling is capped by how much the op rounds. `{ex.ANCHORED_OP}` is total on the grid, so its told-op ceiling is 1 and its ceiling at the center condition is {N["diff"]["ceiling"]:.3f}, with the posterior on it spread about as widely as on any op (sd {N["diff"]["sd"]:.2f}, {N["diff"]["mid"]:.0%} in the middle band). That is the graded stimulus the anchored arms are scored against.

{per_op_table()}

### Cube noise

The pivot allows cube noise at a low rate, so that the model learns to discount examples that fit nothing. At the center condition it costs {N["cube_cost"][ex.CUBE_GRID[1]]:.3f} of ceiling at κ = {ex.CUBE_GRID[1]:g}, {N["cube_cost"][ex.CUBE_GRID[2]]:.3f} at κ = {ex.CUBE_GRID[2]:g}, and {N["cube_cost"][ex.CUBE_GRID[3]]:.3f} at κ = {ex.CUBE_GRID[3]:g}.

A posterior that does not know about cube noise treats a cube color as replacement op noise. It is a little sharper, and the metric rewards sharpness (as above), so it scores {N["cube_ignoring"][ex.CUBE_GRID[3]]:.3f} higher at the highest rate. So the ceiling barely depends on whether the model has learned about cube noise. Whether the corpus carries it is left open (`CUBE_RATE`).

{cube_figure()}

{cube_table()}

### The three corpus conditions

Rule (a) picks the widest spread among the conditions where the control nears its ceiling, so the three should differ in spread and in ceiling, and bracket the working point of the pivot of three examples at ρ near 0.3. We propose `{COND_NAMES[0]}`, `{COND_NAMES[1]}`, and `{COND_NAMES[2]}`.

{conditions_table()}

`{COND_NAMES[1]}` is the center. `{COND_NAMES[0]}` steps ρ down on the same line length: it gives up spread ({spread(*ex.GRAMMAR_CONDITIONS[0]):.2f} against {spread(*ex.CENTRE):.2f}) for a higher ceiling, and is the fallback within the same block size if the control falls short at the center.

`{COND_NAMES[2]}` adds one example at the center ρ, which raises the ceiling by {ceiling(*ex.GRAMMAR_CONDITIONS[2]) - ceiling(*ex.CENTRE):.2f} and keeps most of the spread ({spread(*ex.GRAMMAR_CONDITIONS[2]):.2f}), though fewer contexts sit in the middle band ({band_shares(*ex.GRAMMAR_CONDITIONS[2])[1]:.0%} against {band_shares(*ex.CENTRE)[1]:.0%}). It asks whether more evidence buys a higher ceiling without flattening the stimulus, at {ex.context_tokens(4)} tokens per line rather than {ex.context_tokens(3)}.

Two alternatives were weighed. `{cond_name(*ALT_34)}` has the widest spread on the three-example row, but its middle-band share ({band_shares(*ALT_34)[1]:.0%}) is no larger than at the center and its ceiling is {ceiling(*ex.CENTRE) - ceiling(*ALT_34):.2f} lower, so it grades no better and leaves less room between floor and ceiling. `{cond_name(*ALT_435)}` comes closer to the spread at the center ({spread(*ALT_435):.2f}, with {band_shares(*ALT_435)[1]:.0%} in the middle band) at a ceiling of {ceiling(*ALT_435):.3f}; `{COND_NAMES[2]}` gives up a little of that spread for {ceiling(*ex.GRAMMAR_CONDITIONS[2]) - ceiling(*ALT_435):.2f} more ceiling.


### Reading a score against its ceiling

The ceiling differs between conditions, so a score is shown two ways. Figures keep raw expected exact match on the y-axis and draw the ceiling and floor of each condition as dashed lines beside its seed marks, so the reader sees without arithmetic how much of the available room a model takes.

A table that compares across conditions adds the skill score as a column (0 is a model that ignores the examples, 1 is the Bayes predictor). The figure for the control also draws the ceiling for a model told the op ({TOLD_OP:.3f}) beside the Bayes ceiling, so that the part of the gap due to inference can be seen.


### The calibration check

The check compares the model's answer distribution at the query `=`, $p$, with the Bayes predictive $q$ on the same context, through the KL divergence KL(q ‖ p) = Σ<sub>y</sub> q(y) log(q(y) / p(y)), averaged over held-out contexts. Averaged that way it is the cross-entropy of the model on the answer less the cross-entropy of the Bayes predictor, since the true answers are drawn from $q$ once the true op is marginalized under the posterior. So the number is excess loss, in nats, and zero means the model holds exactly the distribution the examples support.

The table below gives the scale, at the cube-noise rate of record. H(q) is the irreducible loss: what even the Bayes predictor pays, from rounding and from the ops the examples do not rule out. The other columns are the excess of four predictors that miss calibration in named ways: the floor predictor (no evidence), one that commits to the most probable op and answers with its distribution, one that ignores one of the examples, and one that mixes a tenth of the floor into the Bayes answer.

{calibration_table()}

Two things to take from it. Ignoring one example costs several times the threshold of {ex.KL_CALIBRATED:g}, so a calibrated model is one that uses all of its examples; and committing to one op costs more than ignoring an example, so a model that names the op instead of weighing them shows up here even where its expected exact match is high (that is the sharpening the ceiling of record allows for). The mix column is what the threshold roughly means: less miscalibrated than a Bayes predictor with a tenth of ignorance stirred in.

The ceiling of record at each condition is the Bayes ceiling at κ = {ex.CUBE_RATE:g}, from the [cube table](#cube-noise). The floor does not depend on the examples, so it is the κ = 0 value.

### The corpus and the training

One context per line, the op uniform over table A+, every pair uniform over the ordered pairs of the grid, and the query clean; the noise model is the one `posterior.py` assumes, a per-example replacement rate ρ with the replacing op uniform over the other ten and cube noise uniform over the grid at κ = {ex.CUBE_RATE:g}, so the ceiling here is the ceiling of the corpus. The generator is `sca.data.incontext`; a context of *k* examples is 6*k* + 6 tokens, so a line is {ex.context_tokens(3)} tokens at three examples and {ex.context_tokens(4)} at four, and {ex.context_tokens(3, verify=True)} at three with a verification line.

The corpus holds {ex.N_LINES:,} contexts, the line count of ex-2.2.14, so `{ex.ANCHORED_OP}` has about the same number of lines and the labeller the same number of draws (about {ex.N_LINES // ex.N_OPS * ex.LABEL_RATE:.0f} labelled contexts). The training window is {ex.BLOCK} tokens, a random crop of the packed corpus, which holds about two whole contexts at four examples and three at three, with a fragment at each end. Held out: {ex.HOLDOUT_CONTEXTS:,} contexts per op, drawn from the same generator at a seed the corpus does not use.

The line and role of each token come from the positions of `⏎` (`line_role_arrays`), since the length of a context depends on its example count and on whether it carries a verification line. The labeller draws once per context from the op array beside the corpus. Crop policy `{ex.CROP_POLICY}` pulls a labelled context only when the window holds all of it.

The recipe is that of ex-2.2.14, unchanged: λ_a = 0.1 annealed to a 0.1 floor over the last tenth of training, τ = 0.1, the anti-subspace weight from 2.5× the anchor weight to 0.3× by 90% of training, the untied readout, {ex.EPOCHS} epochs at {ex.MODEL}. A context is four times the tokens of an op-word line, so an epoch is about four times the steps of an epoch in ex-2.2.14.


### The measurements

Every arm is scored at the end of training on its held-out contexts, one context per forward pass, with the mask arms measured with the mask on.

- The task: held-out expected exact match, over all contexts and per op, against the ceiling and floor of record for the condition of the arm. Rule (a) scores the controls on it; rules (b) and (d) score the center arms on it.
- The calibration check: the mean KL divergence from the Bayes predictive to the model's answer distribution at the query `=`, as above, on every arm; every figure that plots expected exact match plots it beside.
- Alignment by role and slice: the mean cosine with e₁ at every role of the context (the operands, `?`, `=`, and the answer of each example, and of the query) at each of the five slices, on the `{ex.ANCHORED_OP}` contexts and on the other ops', with the query `?` and the query `=` read separately. Rule (c) uses the query `?` at the last block.
- Alignment against the posterior: the alignment at the query `=` and the query `?` at the last block, per held-out `{ex.ANCHORED_OP}` context, beside the posterior on `{ex.ANCHORED_OP}` for that context, from the method section. The [evidence section](#the-anchor-follows-the-evidence) bins it.
- The op margin: the statistic of ex-2.2.14 with the contexts of the anchored op as the labelled group and roles running over the whole context, on the same held-out contexts. Rules (b) and (c) take it as a share of the margin of the whole-line arm.
- The leans: the first-operand lean (ᾱ at the first token, last block) and the trailing-fragment lean (ᾱ over every position, last block, of a context cut to start at each example boundary, fed on its own), over the held-out contexts of every op, as ex-2.2.15 measured them. The op margin and both leans are also recorded every 50 training steps on a fixed probe set, as ex-2.2.15 did, so the report can say how each arm got where it landed.
- The suppression pass: on the whole-line arm, the hinge arm, and the control, each operator at each dose and site, the held-out expected exact match on `{ex.ANCHORED_OP}` contexts and on those of each other op, against the clean pass and the target null per context, and the KL divergence from the target-null predictive to the edited answer distribution. Rule (e) is scored on it.
- Verification accuracy on the verification arms, held out, with no gate.

Figures label the roles in the notation of the [posterior section](#the-posterior-over-ops): *a*, *b*, and *y* for the operands and the answer, subscripted by example, with `?` and `=` as themselves.

### Budget

{ex.N_RUNS} runs: {len(ex.ARMS)} arms at {ex.SEEDS} seeds. Ex-2.2.14 trained 50 runs of {ex.EPOCHS} epochs at {ex.MODEL} on a 300k-line corpus for about \$1.80 on Modal, and ex-2.2.15 trained 55 runs for about \$2.30 (`bin/mini cost`): about 3.5 cents and two to three minutes on an L4 per run. At this model size a training step is bound by latency, not arithmetic: about 10.5 ms on an L4 whatever the window holds ([GPU notes](/eng/gpu-efficiency.md)), and an ex-2.2.14 run took about 5,000 steps. So the cost follows the step count. A context here has four to five times the tokens of an op-word line and the window is half as long again, so an epoch is about three times the steps: some 13,000 steps at three examples and 16,500 at four, which is two and a half to three minutes of stepping, or four to six minutes per training task with compilation, start-up, and the trajectory reads, well inside the one-hour timeout. At the per-step cost of ex-2.2.14 that puts the training at \$3 to \$4, a little more for the {ex.LARGE_MODEL} control. The window of {ex.BLOCK} tokens at {ex.MODEL} fits an L4 with room to spare. Eval adds the alignment measurements on eleven held-out sets, the posterior on every held-out context (a table lookup), and the suppression pass, which is scoring only: {len(ex.EDIT_SITES)} sites × ({len(ex.DOSE_GAMMAS)} + {len(ex.REPULSION_LANDINGS)} + 1) edits on nine checkpoints. Under \$6 in all, at the pace `--max-containers 12 --budget 6h` sets. The training trajectories record the margin and the leans every 50 steps, so a corpus that saturates early shows there, and a later round can train on fewer contexts.
"""
