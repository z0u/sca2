# ruff: noqa: B018
# title: A floor in degrees for the write tolerance

# An exploratory re-analysis of stored alignment arrays, with no new training and no new measurement pass: the
# alignment with e₁ at every slice and position, clean and under the full edit, that the fixed confirmation stored for
# `τ_ℓ fixed` (seeds 712 to 719), edit-effects for `falloff`, and recipe-compare for `step, soft τ` and
# `pool 1–3, soft τ` (seeds 704 to 711).
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

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


fc = _load_sibling("fixed-confirm", "write_floor_fc")
rc = _load_sibling("recipe-compare", "write_floor_rc")
ee = rc.ee

# --- The stored arrays ----------------------------------------------------------------------------------------

FIXED, FALLOFF, STEP_SOFT, POOL = "fixed", "falloff", "step-soft", "pool-123-soft"
RECIPES = (FIXED, FALLOFF, STEP_SOFT, POOL)
SHOWN = {FIXED: "τ_ℓ fixed", FALLOFF: "falloff", STEP_SOFT: "step, soft τ", POOL: "pool 1–3, soft τ"}
SEEDS_OF = {FIXED: list(fc.SEEDS), **{n: list(rc.SEEDS) for n in (FALLOFF, STEP_SOFT, POOL)}}
ALPHA_REF = {
    FIXED: lambda s: fc.WRITE_ARRAYS_REF.format(label=fc.label_of(s, None)),
    FALLOFF: lambda s: ee.ARRAYS_REF.format(label=f"falloff-s{s}"),
    STEP_SOFT: lambda s: rc.WRITE_ARRAYS_REF.format(label=f"{STEP_SOFT}-s{s}"),
    POOL: lambda s: rc.WRITE_ARRAYS_REF.format(label=f"{POOL}-s{s}"),
}


def fetch_all() -> tuple[dict, np.ndarray]:
    """Per run, the alignment with e₁ clean and as it arrives under the full edit, keyed by (recipe, seed); and the
    fit of every example and query answer under every op.
    """
    store = project_store()
    arrays = {(n, s): ALPHA_REF[n](s) for n in RECIPES for s in SEEDS_OF[n]}
    refs = [ee.FITS_REF, *arrays.values()]
    got = store.get_refs(refs)
    have = {r: a for r, a in got.items() if a is not None}
    assert len(have) == len(refs), f"not published: {sorted(set(refs) - set(have))}"
    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        paths = dict(zip(refs, store.get_many([(have[r], into / f"{i}") for i, r in enumerate(refs)]), strict=True))
        with np.load(paths[ee.FITS_REF]) as z:
            fits = z["fits"]
        alpha = {}
        for key, ref in arrays.items():
            with np.load(paths[ref]) as z:
                alpha[key] = {k: z[k] for k in ("clean", "arriving", "op_ids")}
    return alpha, fits


ALPHA, FITS = fetch_all()
OPS: tuple[str, ...] = tuple(rc.OP_NAMES)
assert tuple(fc.OP_NAMES) == OPS
D = OPS.index(rc.ANCHORED_OP)
DIFF = rc.ANCHORED_OP
K = ee.K
OP_IDS = ALPHA[(FIXED, SEEDS_OF[FIXED][0])]["op_ids"].astype(int)
for _a in ALPHA.values():
    assert (_a["op_ids"] == OP_IDS).all(), "every run was scored on the same held-out contexts"

UNIT = 6  # tokens per equation: a ? b = y and a separator
N_POS = UNIT * (K + 1)
ANSWERS = [UNIT * e + 4 for e in range(K)] + [UNIT * K + 4]  # the example answers, then the query answer
ROLE = ("a", "?", "b", "=", "y", ",")
POS_LABEL = [ROLE[p % UNIT] if p != N_POS - 1 else "⏎" for p in range(N_POS)]
SYNTAX = np.array([POS_LABEL[p] in ("?", ",", "⏎") for p in range(N_POS)])
OPERAND = np.array([POS_LABEL[p] in ("a", "b") for p in range(N_POS)])
Q = 0.99  # the quantile the bound and the write are taken at, as in H4 of ex-2.2.1
TOLERANCE = 1.25  # the tolerance of H4 in ex-2.2.1: the write at most a quarter over the bound
N_BOOT = 200  # resamples of the held-out contexts, per run
SMALL = 10.0  # degrees: the bounds of the sites that go over the tolerance are all well under this
FLOOR_Q = 0.95  # the quantile of the seed spread of the bound that sets the floor
N_SITES = 4 * N_POS  # slice and position pairs past the embedding, per run

# The contexts each position is scored over: those of the other ops, and at the answers only the answers that do not
# fit `difference`, as in edit-effects and the fixed-effects report.
OTHER = OP_IDS != D
N_STATES = int(OTHER.sum())  # the states of the other ops at a position that is not an answer
CONTEXTS = [
    np.nonzero(OTHER & ~FITS[:, ANSWERS.index(p), D])[0] if p in ANSWERS else np.nonzero(OTHER)[0] for p in range(N_POS)
]


def angle(alpha) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(alpha, 0, 1)))


@memo
def turn_maps(clean: np.ndarray, arriving: np.ndarray, n_boot: int, seed: int) -> dict:
    """The bound and the write of one run as angles, `(slices, positions)`, and the spread of the excess of the write
    over the bound when the held-out contexts are drawn again with replacement (the same draw for both, so the excess
    is paired). The draw is fixed by *seed*, so a re-render gives the same spread.
    """
    rng = np.random.default_rng(seed)
    c, a = np.abs(clean.astype(np.float32)), np.abs(arriving.astype(np.float32))
    bound, write = np.zeros((5, N_POS)), np.zeros((5, N_POS))
    boot_sd = np.zeros((5, N_POS))
    for p, ctx in enumerate(CONTEXTS):
        cp, ap = c[:, ctx, p], a[:, ctx, p]
        bound[:, p], write[:, p] = angle(np.quantile(cp, Q, axis=1)), angle(np.quantile(ap, Q, axis=1))
        draw = rng.integers(0, len(ctx), size=(n_boot, len(ctx)))
        for sl in range(1, 5):
            excess = angle(np.quantile(ap[sl][draw], Q, axis=1)) - angle(np.quantile(cp[sl][draw], Q, axis=1))
            boot_sd[sl, p] = excess.std()
    return {"bound": bound, "write": write, "boot_sd": boot_sd}


TURN = {
    (n, s): turn_maps(ALPHA[(n, s)]["clean"], ALPHA[(n, s)]["arriving"], N_BOOT, i) for i, (n, s) in enumerate(ALPHA)
}
for _t in TURN.values():
    assert np.allclose(_t["bound"][0], _t["write"][0], atol=0.05), "the embedding sees the clean stream"

# Past the embedding, per recipe: (runs, slices 1 to 4, positions).
BOUND = {n: np.stack([TURN[(n, s)]["bound"][1:] for s in SEEDS_OF[n]]) for n in RECIPES}
WRITE = {n: np.stack([TURN[(n, s)]["write"][1:] for s in SEEDS_OF[n]]) for n in RECIPES}
BOOT_SD = {n: np.stack([TURN[(n, s)]["boot_sd"][1:] for s in SEEDS_OF[n]]) for n in RECIPES}
EXCESS = {n: WRITE[n] - BOUND[n] for n in RECIPES}
OVER = {n: np.radians(WRITE[n]) / np.radians(BOUND[n]) > TOLERANCE for n in RECIPES}

# The counts the fixed-effects report gives for its two recipes, recomputed here from the same arrays.
assert OVER[FIXED].sum() == 34 and OVER[FALLOFF].sum() == 28, "the over counts of the fixed-effects report"

# --- E1: the spread from the choice of contexts ---------------------------------------------------------------

BOOT_AT_OVER = {n: BOOT_SD[n][OVER[n]] for n in RECIPES}
BOOT_MAX = max(float(v.max()) for v in BOOT_AT_OVER.values())
BOOT_MEDIAN = float(np.median(np.concatenate(list(BOOT_AT_OVER.values()))))
EXCESS_MIN = min(float(EXCESS[n][OVER[n]].min()) for n in RECIPES)
Z = np.concatenate([EXCESS[n][OVER[n]] / BOOT_SD[n][OVER[n]] for n in RECIPES])
Z_MIN, Z_MEDIAN, N_Z_UNDER_4 = float(Z.min()), float(np.median(Z)), int((Z < 4).sum())
assert Z_MIN > 2 and N_Z_UNDER_4 <= 2, "E1 says every site over the tolerance is past the spread from the contexts"
# How far each site sits past the ratio line itself (the write less 1.25× the bound), against the same spread. A site
# near the line can cross it on another draw even where its excess over the bound is many spreads wide.
MARGIN_Z = np.concatenate(
    [(EXCESS[n][OVER[n]] - (TOLERANCE - 1) * BOUND[n][OVER[n]]) / BOOT_SD[n][OVER[n]] for n in RECIPES]
)
N_NEAR_LINE = int((MARGIN_Z < 2).sum())
# REVIEW: E1 said nearly every site over the tolerance would still be over on another draw. That holds for the excess
# over the bound (Z_MIN above 2) but not for the ratio: the margin past 1.25× the bound is under twice its spread at
# about a fifth of the sites. The text now says both. Verify: MARGIN_Z against Z.
assert 15 <= N_NEAR_LINE <= 40, "E1 says about a fifth of the sites sit near the ratio line"

# --- E2: the spread of the bound between runs -----------------------------------------------------------------

SEED_SD = {n: BOUND[n].std(axis=0, ddof=1) for n in RECIPES}  # (slices, positions)
SEED_MEAN = {n: BOUND[n].mean(axis=0) for n in RECIPES}
POOLED_SD = np.concatenate([SEED_SD[n].ravel() for n in RECIPES])
POOLED_MEAN = np.concatenate([SEED_MEAN[n].ravel() for n in RECIPES])
IS_SMALL = POOLED_MEAN < SMALL
FLOOR = float(np.quantile(POOLED_SD[IS_SMALL], FLOOR_Q))
SD_SMALL_MEDIAN = float(np.median(POOLED_SD[IS_SMALL]))
# The same quantile taken one recipe at a time, and over the operand sites alone (the symbol sites are two fifths of
# the pool and spread by much less, so they pull the pooled quantile down).
RECIPE_FLOOR = {n: float(np.quantile(SEED_SD[n].ravel()[SEED_MEAN[n].ravel() < SMALL], FLOOR_Q)) for n in RECIPES}
RECIPE_MEDIAN = {n: float(np.median(SEED_SD[n].ravel()[SEED_MEAN[n].ravel() < SMALL])) for n in RECIPES}
OPERAND_FLOOR = float(np.quantile(np.concatenate([SEED_SD[n][:, OPERAND].ravel() for n in RECIPES]), FLOOR_Q))
# REVIEW: E2 said the four recipes give about the same spread one at a time, so pooling changes little. The medians
# agree, but the 95% quantiles run from about 1.0° to 1.4° per recipe, and over the operand sites alone the pooled
# quantile is about 1.4°, so the text now gives those and the discussion names the pool composition among the choices
# the exact floor rests on. The split of E3 is unchanged by any of them, since no symbol site has an excess past
# 0.81° (SYNTAX_EXCESS_MAX), and the discussion now says that instead of "from a degree to the floor of E2".
# Verify: RECIPE_FLOOR, OPERAND_FLOOR, and the counts at floors of 0.9° to 1.5°.
assert max(RECIPE_MEDIAN.values()) - min(RECIPE_MEDIAN.values()) < 0.05, "E2: the medians agree across recipes"
assert FLOOR < OPERAND_FLOOR < 1.5, "E2: the operand-only quantile is higher than the pooled one"
SD_TINY_MEDIAN = float(np.median(POOLED_SD[POOLED_MEAN < 1]))
SD_LARGE_MEDIAN = float(np.median(POOLED_SD[~IS_SMALL]))
OVER_BOUND_MAX = max(float(BOUND[n][OVER[n]].max()) for n in RECIPES)
assert OVER_BOUND_MAX < SMALL, "every site over the tolerance has a bound under the cut"
# The same rule taken site by site: the write is over the bound by more than twice the seed spread of the bound there.
PER_SITE = {n: OVER[n] & (EXCESS[n] > 2 * SEED_SD[n][None]) for n in RECIPES}

# --- E3: what the floor keeps ---------------------------------------------------------------------------------

BOTH = {n: OVER[n] & (EXCESS[n] > FLOOR) for n in RECIPES}
RUNS_WITH = {n: [s for i, s in enumerate(SEEDS_OF[n]) if BOTH[n][i].any()] for n in RECIPES}
SLICES_WITH = {n: sorted({int(sl) + 1 for sl in np.nonzero(BOTH[n])[1]}) for n in RECIPES}
for _n in RECIPES:
    assert not (BOTH[_n] & SYNTAX[None, None]).any(), "E3 says no site at a symbol or separator clears the floor"
    assert (BOTH[_n] <= OPERAND[None, None]).all(), "E3 says every site that clears the floor is at an operand"
    assert (OVER[_n] <= (SYNTAX | OPERAND)[None, None]).all(), "E3 says no site at `=` or an answer goes over"
assert BOTH[FIXED].sum() == 1 and len(RUNS_WITH[FIXED]) == len(RUNS_WITH[FALLOFF]) == 1, "the discussion text"
PER_SITE_SHARED = {n: int((PER_SITE[n] & BOTH[n]).sum()) for n in RECIPES}
assert PER_SITE_SHARED[FIXED] == 0, "E3 says the two rules share no site on the recipe of record"
PER_SITE_SYNTAX = {n: int((PER_SITE[n] & SYNTAX[None, None]).sum()) for n in RECIPES}
PER_SITE_SYNTAX_EXCESS = max(float(EXCESS[n][PER_SITE[n] & SYNTAX[None, None]].max()) for n in RECIPES)
SYNTAX_SD_MEDIAN = float(np.median(np.concatenate([SEED_SD[n][:, SYNTAX].ravel() for n in RECIPES])))
POOL_SHARED = PER_SITE_SHARED[POOL]
N_SYNTAX_OVER = {n: int((OVER[n] & SYNTAX[None, None]).sum()) for n in RECIPES}
SYNTAX_EXCESS_MAX = max(float(EXCESS[n][OVER[n] & SYNTAX[None, None]].max()) for n in RECIPES)
SYNTAX_BOUND_MAX = max(float(BOUND[n][OVER[n] & SYNTAX[None, None]].max()) for n in RECIPES)
N_SYNTAX_OVER_DEGREE = sum(int((BOUND[n][OVER[n] & SYNTAX[None, None]] > 1).sum()) for n in RECIPES)
assert N_SYNTAX_OVER_DEGREE <= 3, "E3 says the bound at a symbol site over the tolerance is under a degree at all but a few"
OPERAND_EXCESS_MIN = min(float(EXCESS[n][OVER[n] & OPERAND[None, None]].min()) for n in RECIPES)
POOL_EXCESS_MEDIAN = float(np.median(EXCESS[POOL][BOTH[POOL]]))
SWEEP = np.linspace(0, 4.5, 91)
COUNT_AT = {n: np.array([(OVER[n] & (EXCESS[n] > f)).sum() for f in SWEEP]) for n in RECIPES}
AT_ONE = {n: int((OVER[n] & (EXCESS[n] > 1.0)).sum()) for n in RECIPES}
assert {n: AT_ONE[n] - int(BOTH[n].sum()) for n in RECIPES} == {FIXED: 2, FALLOFF: 0, STEP_SOFT: 2, POOL: 1}, "E3 text"
POOL_RATIO = np.radians(WRITE[POOL][BOTH[POOL]]) / np.radians(BOUND[POOL][BOTH[POOL]])
assert POOL_RATIO.max() < 2.1 and 3 < np.median(BOUND[POOL][BOTH[POOL]]) < 5, (
    "E3: up to about twice, bounds of about 4°"
)


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


def ink(n: str) -> str:
    return {
        FIXED: light_dark("#2a9d8f", "#7fd8c8"),
        FALLOFF: light_dark("#888", "#aaa"),
        STEP_SOFT: light_dark("#5b6fb5", "#9fb0f0"),
        POOL: light_dark("#c0392b", "#ff8a76"),
    }[n]


def rule() -> str:
    return light_dark("#444", "#bbb")


MARK = {FIXED: "D", FALLOFF: "o", STEP_SOFT: "s", POOL: "^"}

r"""
# A floor in degrees for the write tolerance

/// tip |
<!-- lede -->
The write tolerance of ex-2.2.1 counts a site as over when the write is more than a quarter over the bound. So where the bound is a fraction of a degree, a tiny turn counts. We looked for a floor in degrees that the data could set, and found one in how much the bound varies from run to run: about a degree. With that floor, every site at a symbol or separator drops out. What stays are the operands at the first slice or two, chiefly on `pool 1–3, soft τ`.
///

The paper review of the [recipe-compare report](/docs/m2/recipe-compare/report.py) asked whether the tolerance should ignore small shifts. In that report and the [fixed-effects report](/docs/m2/fixed-effects/report.py), every site over the tolerance has a bound of a few degrees or less, so a ratio over 1.25 can be a turn of a fraction of a degree. With a floor, a site would count only when the write is past both the ratio and the bound plus some number of degrees. The backlog item behind this report asked for that number to come from the measurement rather than be picked by eye.

This report needs no new runs. It uses the stored alignment with e₁ at every slice and position, clean and under the full edit, on the eight runs of each of four recipes: `τ_ℓ fixed` from the [fixed confirmation](/docs/m2/fixed-confirm/report.py), `falloff` from the [edit-effects report](/docs/m2/edit-effects/report.py), and `step, soft τ` and `pool 1–3, soft τ` from the recipe-compare report.
"""

# %%

f"""
## Observations

- [The spread from the choice of contexts (E1)](#the-spread-from-the-choice-of-contexts-e1): drawing the held-out contexts again moves the excess of the write over the bound by a few hundredths of a degree, far less than any excess past the tolerance. So that spread cannot set the floor. The excess at the sites over the tolerance is a real feature of each run.
- [The spread of the bound between runs (E2)](#the-spread-of-the-bound-between-runs-e2): where the bound is small, it differs from run to run by about half a degree, and by about a degree at the {FLOOR_Q:.0%} quantile. We take that degree as the floor.
- [What the floor keeps (E3)](#what-the-floor-keeps-e3): every site at a `?`, `,` or ⏎ falls under the floor. The sites that clear it are all at operands, at slice 1 (and slice 2 on `step, soft τ`): one to three runs of the other recipes, and every run of the pool. A rule taken site by site (an excess over twice the seed spread at that site) flags about as many sites. But some of them are at symbols, where the bound hardly varies between runs.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers {sum(len(v) for v in SEEDS_OF.values())} runs: `τ_ℓ fixed` at seeds {SEEDS_OF[FIXED][0]} to {SEEDS_OF[FIXED][-1]}, and `falloff`, `step, soft τ` and `pool 1–3, soft τ` at seeds {SEEDS_OF[FALLOFF][0]} to {SEEDS_OF[FALLOFF][-1]}. Each run is scored on the {len(OP_IDS):,} held-out contexts of ex-2.2.21, under the full edit, which takes the e₁ component out of the state at every position and every slice.

The bound and the write are computed as in the fixed-effects report, and give the same counts over the tolerance for its two recipes. A site is one slice past the embedding and one position, {N_SITES} per run. At the embedding the write equals the bound by construction, so it is left out.

We left out one source the item suggested for the floor: the write that an edit along a reference direction gives at the same sites. The stored arrays hold the alignment with e₁ only, so that would need a new measurement pass.

## The measurements

Bound
:   From the clean pass: arcsin of the {Q:.0%} quantile of |α| over the states of the other ops at a slice and position, as an angle. It is how far the edit turns those states there. At an answer position it counts only the answers that `{DIFF}` could not give from the same operands.

Write
:   The same quantile, over the states as they arrive under the full edit, after the preceding block has read the edited states of the slices below.

Excess
:   The write less the bound, in degrees. Positive when the blocks after the edit put more e₁ on the other ops than the clean model has there.

Over the tolerance
:   A site where the write is more than {TOLERANCE}× the bound, the tolerance of H4 in ex-2.2.1, each run taken on its own.

Seed spread
:   The standard deviation of the bound at one site over the eight runs of one recipe.
"""

# %%

boot_rows = [
    [
        f"`{SHOWN[n]}`",
        f"{OVER[n].sum()} of {len(SEEDS_OF[n]) * N_SITES}",
        f"{EXCESS[n][OVER[n]].min():.2f}° to {EXCESS[n][OVER[n]].max():.2f}°",
        f"{np.median(BOOT_AT_OVER[n]):.2f}°",
        f"{BOOT_AT_OVER[n].max():.2f}°",
    ]
    for n in RECIPES
]

# No figure beside this table: the comparison is one ratio per recipe, and the spread is too small to show on the scale
# of the excess.
BOOT_TABLE = table_html(
    ["recipe", "sites over", "excess there", "spread there, median", "spread there, largest"],
    boot_rows,
    f"""**The excess at the sites over the tolerance, against its spread from the choice of contexts.** The sites
    over count slice and position pairs past the embedding over the eight runs of each recipe, each run taken on
    its own. The spread is the standard deviation of the excess over {N_BOOT} draws of the held-out contexts with
    replacement, at each of those sites.""",
)

f"""
## The spread from the choice of contexts (E1)

The bound and the write are each a {Q:.0%} quantile over about {N_STATES:,} states, so they would move a little on another set of contexts. If that movement were as large as the excess at the sites over the tolerance, it would be the natural floor: an excess within it could be an accident of which contexts were held out. We drew the contexts again with replacement {N_BOOT} times per run, took the bound and the write on the same draw, and measured the spread of the excess at each site.

{BOOT_TABLE}

The spread is tiny next to the excess. At the sites over the tolerance it is a few hundredths of a degree at the median, and the excess there is about {Z_MEDIAN:.0f} times its own spread at the median. The closest case is an operand site with a bound of about 6°, where the spread is nearly a degree and the excess {Z_MIN:.1f} times that. So the excess at every site over the tolerance would survive another set of contexts drawn the same way.

Whether a site stays over the tolerance is a narrower question. The tolerance line is at an excess of a quarter of the bound, not at zero. {N_NEAR_LINE} of the {len(Z)} sites are within twice their spread of that line, and another draw could put those under it.
"""

# %%


@memo
def spread_draw(sd: dict, mean: dict, floor: float, small: float, alt_text: str) -> str:
    @themed(
        name="write-floor-spread",
        alt_text=alt_text,
        caption=f"""
            **How much the bound varies between runs, by how wide it is.** Each point is one site (a slice past the
            embedding and a position) of one recipe: the seed mean of the bound on the horizontal axis, and its
            standard deviation over the eight runs on the vertical. Both axes are logarithmic. The solid line is the floor, the {FLOOR_Q:.0%} quantile
            of the spread over the sites with a seed-mean bound under {small:.0f}° (left of the dotted line).
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.6, 3.2), layout="constrained")
        for n in RECIPES:
            ax.scatter(mean[n], sd[n], s=9, marker=MARK[n], color=ink(n), alpha=0.7, lw=0, label=SHOWN[n])
        ax.axhline(floor, color=rule(), lw=1.0)
        ax.axvline(small, color=rule(), lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("seed-mean bound (°)", fontsize=8)
        ax.set_ylabel("seed spread of the bound (°)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper left", markerscale=1.5)
        return fig

    return _plot()


def spread_figure() -> str:
    alt = (
        f"The seed spread rises with the bound across three clusters (symbols about {SD_TINY_MEDIAN:.1f}°, operands "
        f"half a degree to a degree, answers about {SD_LARGE_MEDIAN:.0f}°); the floor at {FLOOR:.1f}° is above nearly every "
        "small-bound site."
    )
    data_sd = {n: SEED_SD[n].ravel() for n in RECIPES}
    data_mean = {n: np.maximum(SEED_MEAN[n].ravel(), 1e-2) for n in RECIPES}
    return spread_draw(data_sd, data_mean, FLOOR, SMALL, alt)


f"""
## The spread of the bound between runs (E2)

The bound differs from one trained run to the next, by much more than it does from one set of contexts to another. An excess smaller than that seed spread is within what training alone moves the bound by, so the seed spread is a candidate for the floor. The figure shows it at every site, against how wide the bound is there.

{spread_figure()}

The spread grows with the bound, and the sites fall into three groups. At the symbols and separators the bound is under a degree, and it spreads by about {SD_TINY_MEDIAN:.1f}° between runs. At the operands it is a few degrees, and spreads by about half a degree to a degree. At the answers it is tens of degrees and spreads by about {SD_LARGE_MEDIAN:.0f}°, but no site there goes over the tolerance: every site that does has a bound under {OVER_BOUND_MAX:.0f}°.

Over the sites with a bound under {SMALL:.0f}°, the seed spread is {SD_SMALL_MEDIAN:.2f}° at the median and {FLOOR:.2f}° at the {FLOOR_Q:.0%} quantile. We take the second as the floor: an excess larger than that is more than the bound moves between runs at {FLOOR_Q:.0%} of the small-bound sites. The operand sites set it, since the bound at every symbol and separator spreads by less.

The floor depends a little on which sites go into the pool. The symbol sites are two fifths of it (the `=` sites, whose bound is a few degrees, are most of the rest). They are well under the floor and pull the quantile down: over the operand sites alone it is {OPERAND_FLOOR:.2f}°. Taken one recipe at a time, the median is the same on all four, and the {FLOOR_Q:.0%} quantile runs from {min(RECIPE_FLOOR.values()):.2f}° to {max(RECIPE_FLOOR.values()):.2f}°.

The cut at {SMALL:.0f}° leaves out the answers. Their spread comes from the tail of states found in the fixed-effects report, and including them would lift the floor past anything the small-bound sites need.
"""

# %%


@memo
def keep_draw(bound: dict, excess: dict, over: dict, syntax: np.ndarray, sweep: np.ndarray, count: dict,
              floor: float, alt_text: str) -> str:  # fmt: skip
    @themed(
        name="write-floor-keep",
        alt_text=alt_text,
        caption=f"""
            **Left:** the sites over the tolerance, by the bound there and the excess of the write over it, on all
            four recipes, each run taken on its own. Hollow markers are at a `?`, `,` or ⏎, filled ones at an
            operand. **Right:** how many of them stay over as the floor rises, per recipe over its eight runs. The
            solid line in both panels is the floor of E2, {floor:.2f}°.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.8), layout="constrained")
        for n in RECIPES:
            b, e, sy = bound[n][over[n]], excess[n][over[n]], np.broadcast_to(syntax, over[n].shape)[over[n]]
            left.scatter(b[sy], e[sy], s=14, marker=MARK[n], facecolor="none", edgecolor=ink(n), lw=0.8)
            left.scatter(b[~sy], e[~sy], s=14, marker=MARK[n], color=ink(n), lw=0, label=SHOWN[n])
            right.plot(sweep, count[n], color=ink(n), lw=1.3, label=SHOWN[n])
        left.axhline(floor, color=rule(), lw=1.0)
        right.axvline(floor, color=rule(), lw=1.0)
        left.set_xlabel("bound (°)", fontsize=8)
        left.set_ylabel("write less bound (°)", fontsize=8)
        left.set_xlim(0, None)
        left.set_ylim(0, None)
        right.set_xlabel("floor (°)", fontsize=8)
        right.set_ylabel("sites still over", fontsize=8)
        right.set_ylim(0, None)
        for ax in (left, right):
            ax.tick_params(labelsize=7)
        right.legend(fontsize=7, frameon=False)
        return fig

    return _plot()


def keep_figure() -> str:
    alt = (
        "Symbol sites cluster near zero under the floor; the operand sites above it are mostly on pool 1–3, soft τ, "
        f"which keeps {BOTH[POOL].sum()} past the floor against {BOTH[FIXED].sum()} to "
        f"{max(BOTH[n].sum() for n in (FALLOFF, STEP_SOFT))} for the other recipes."
    )
    return keep_draw(BOUND, EXCESS, OVER, SYNTAX, SWEEP, COUNT_AT, FLOOR, alt)


keep_rows = [
    [
        f"`{SHOWN[n]}`",
        f"{OVER[n].sum()}",
        f"{N_SYNTAX_OVER[n]}",
        f"{BOTH[n].sum()}",
        f"{len(RUNS_WITH[n])} of {len(SEEDS_OF[n])}",
        ", ".join(str(s) for s in SLICES_WITH[n]),
        f"{PER_SITE[n].sum()} ({PER_SITE_SYNTAX[n]})",
        f"{AT_ONE[n]}",
    ]
    for n in RECIPES
]

KEEP_TABLE = table_html(
    ["recipe", "over the tolerance", "at a symbol", "over both", "runs", "slices", "twice the seed spread", "1° floor"],
    keep_rows,
    f"""**The sites over the tolerance, and those over the floor as well**, per recipe over its eight runs, each run
    taken on its own. "At a symbol" counts those at a `?`, `,` or ⏎. "Over both" counts those whose excess is
    also over the floor of E2 ({FLOOR:.2f}°), with the runs and slices they fall on. The last two columns give the
    count over both under two other rules: an excess over twice the seed spread at that site (in brackets, how
    many of those are at a symbol), and a floor of a whole degree.""",
)

f"""
## What the floor keeps (E3)

The figure places each site over the tolerance by its bound and its excess, and counts how many stay over as the floor rises.

{keep_figure()}

The sites over the tolerance are of two kinds, and the floor separates them. At the symbols and separators (`?`, `,` and ⏎) the bound is under a degree at all but {N_SYNTAX_OVER_DEGREE} of them and at most {SYNTAX_BOUND_MAX:.1f}°, the excess is at most {SYNTAX_EXCESS_MAX:.1f}°, and every one of them falls under the floor. They make up most of the sites over the tolerance on `τ_ℓ fixed` and `falloff` (the table below). At the operands the bound is a few degrees, and the excess ranges from {OPERAND_EXCESS_MIN:.1f}° to several degrees. No site at `=` or at an answer goes over the tolerance. Every site that clears the floor is at an operand, at slice {" or ".join(str(s) for s in sorted(set().union(*SLICES_WITH.values())))}.

{KEEP_TABLE}

On `τ_ℓ fixed`, `falloff` and `step, soft τ` a few operand sites stay, on one to three runs each. On `pool 1–3, soft τ` every run keeps some, {BOTH[POOL].sum()} sites in all, with a median excess of {POOL_EXCESS_MEDIAN:.1f}° on bounds of about 4°. At those sites, block 1 puts e₁ back on the operands of the other ops, up to about twice what the clean model has there. That is the slice where the recipe-compare report found the pool spilling, and the one slice the pool pulls and the step leaves alone.

A floor of one whole degree keeps the same kind of sites, all at operands, with a few more of them: two more on `τ_ℓ fixed` (whose excess at the operands of one run is just over a degree), two more on `step, soft τ`, and one more on the pool. The count on the recipe of record shifts when the floor moves by a few tenths of a degree, but the split between the two kinds does not.

A rule taken site by site counts a site when the excess there is over twice the seed spread of the bound at that site. It keeps {POOL_SHARED} of the {BOTH[POOL].sum()} pool sites that the floor keeps, and about as many sites as the floor on the other recipes, though on `τ_ℓ fixed` the two rules share none.

It also keeps {sum(PER_SITE_SYNTAX.values())} sites at symbols, with excesses of up to {PER_SITE_SYNTAX_EXCESS:.1f}°. The bound at a symbol spreads by only about {SYNTAX_SD_MEDIAN:.1f}° between runs, so a small excess there is large for that site. In turn, the rule drops some operand sites, where the spread is wider. The two rules ask different questions: whether the turn is large in degrees, or unusual for that site.
"""

# %%

f"""
## Discussion

The tolerance of ex-2.2.1, taken as a ratio alone, mixes two kinds of site: symbols and separators, where the clean model puts almost no e₁ and the write adds a fraction of a degree, and operands at the first slice or two, where the write adds a degree or more to a bound of a few degrees. A floor of about a degree removes the first kind and keeps the second.

We are fairly sure of the split under an absolute floor. It holds for any floor above the largest excess at a symbol, {SYNTAX_EXCESS_MAX:.1f}°, and the two kinds are apart on the left of the E3 figure. A rule relative to each site would draw the line elsewhere, flagging a few symbol sites whose small excess is unusual for that site. Which of the two a gate wants is a choice about what the tolerance is for.

We are less sure of the exact floor. It rests on eight runs per recipe, and on the choice of the {FLOOR_Q:.0%} quantile and the cut at {SMALL:.0f}°. It also rests on taking the quantile over the symbol and operand sites together. The operand sites alone would put it a few tenths of a degree higher (E2), and a few more sites would fall under it. A reference direction, edited at the same sites, would give a second scale for it that this report could not measure from stored data.

If the floor were adopted, the reports that quote the tolerance would count one site over both on the recipe of record and {BOTH[FALLOFF].sum()} on `falloff`, all at operands of a single run on each recipe. They would also count sites on every run of `pool 1–3, soft τ`. The pool would then stand apart on the write as it already does on spill, at the same slice.
"""

# %%

r"""
## References

[recipe-compare report](/docs/m2/recipe-compare/report.py)
:   `pool 1–3, soft τ` removes about as much as `falloff` and costs the task least, but its edit spills over the criterion on two seeds, and that spill enters with the edit at block 1.

[fixed-effects report](/docs/m2/fixed-effects/report.py)
:   On `τ_ℓ fixed` the write stays within the clean bound past the embedding, as on `falloff`, with a few dozen sites over the tolerance of ex-2.2.1, all where the bound is a few degrees.

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` keeps the anchor on all eight runs at seeds 712 to 719 and is the recipe of record for D2.2.

[edit-effects report](/docs/m2/edit-effects/report.py)
:   On `falloff`, the write stays within the clean bound past the embedding, measured as the ratio of the write to the bound at the 99% quantile.

[ex-2.2.1](/docs/m2/ex-2.2.1/report.py)
:   Set the write bound of H4, with a tolerance of a quarter over the bound.
"""
