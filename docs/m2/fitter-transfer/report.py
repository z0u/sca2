# ruff: noqa: B018
# title: Fitted edits at every slice, carried to a run they were not chosen on

# An exploratory re-analysis with no new runs and no new scoring: the tuning-set and test-set scores the
# post-hoc-compare experiment stored for every configuration of its grid, restricted to the fits that need no role
# labels and edit every slice, with the configuration of each run chosen on the other seven. The rule was written
# down and committed (5530b6e) before any number in this report was computed.
import importlib.util
import json
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
    """An experiment module by path, registered while it loads (its dataclasses need it), as the reports do."""
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


ex = _load_sibling("post-hoc-compare", "fitter_transfer_ex")
fc = ex.fc

# --- The stored results ---------------------------------------------------------------------------------------


def fetch() -> tuple[dict, dict]:
    """The results of post-hoc-compare (every configuration scored on the tuning and test sets of every run), and
    the axis scoring of the anchored runs as fixed-effects publishes it.
    """
    store = project_store()
    refs = [ex.RESULTS_REF, fc.AXES_REF]
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    assert len(have) == len(refs), f"not published: {sorted(set(refs) - set(have))}"
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(have[r], Path(tmp) / f"{i}.json") for i, r in enumerate(refs)])
        return tuple(json.loads(Path(p).read_text()) for p in paths)


BODY, ANCHOR_AXES = fetch()
OPS: tuple[str, ...] = tuple(BODY["design"]["ops"])
DIFF = ex.ANCHORED_OP
D = OPS.index(DIFF)
CRITERION: float = BODY["design"]["spill_criterion"]
MARGIN = 0.01  # the choice with room under the criterion
MISS = 0.5  # a run misses when its removal is under this
SEEDS: list[int] = list(ex.SEEDS)
METHODS: tuple[str, ...] = ex.METHODS
FITTERS = ("dim", "ridge", "leace", "inlp")
RUNS = {(r["kind"], r["seed"]): r for r in BODY["runs"]}
EVERY_SLICE = list(range(ex.N_SLICES))
SHOWN = {"dim": "difference in means", "ridge": "probe", "leace": "LEACE", "inlp": "INLP",
         "optimized": "optimized direction", "anchor": "SCA (anchor)"}  # fmt: skip
SLICE_NAMES = ["emb", *(str(s) for s in range(1, ex.N_SLICES))]


def anchored(s: int, axes: list[int] | None = None) -> tuple[float, float]:
    """Removal and spill of the full edit along e₁ on the anchored run at seed *s*, net of the median over the other
    63 axes of the run, as fixed-effects publishes them, or over *axes* alone.
    """
    r = {x["label"]: x for x in ANCHOR_AXES["runs"]}[fc.label_of(s, None)]
    drop = np.asarray(r["drop"])  # (axes, doses, ops), the anchored axis first
    net = drop[0, -1] - np.median(drop[axes or slice(1, None), -1], axis=0)
    return float(net[D] / (r["clean"][D] - r["null"][D])), float(np.delete(net, D).max())


A_REMOVAL, A_SPILL = (np.array(v) for v in zip(*(anchored(s) for s in SEEDS), strict=True))
assert abs(A_REMOVAL.mean() - BODY["anchored_removal"]) < 1e-9
# The anchor net of the median over the fifteen tuning axes, the baseline the fitters are scored against here.
A_REMOVAL_15, A_SPILL_15 = (np.array(v) for v in zip(*(anchored(s, list(ex.tune_axes())) for s in SEEDS), strict=True))

# --- Scopes, scores, and the choice on other runs -------------------------------------------------------------
# A scope is the part of the grid a choice may take: "every" is the pooled fits that edit every slice (the shape of
# the anchored edit), "blocks" the pooled fits that edit slices 1 to 3 (added after E1), and "pooled" every pooled
# fit (the scope of the pooled analysis of post-hoc-compare).
SCOPE_SLICES = {"every": EVERY_SLICE, "blocks": [1, 2, 3], "pooled": None}


def in_scope(c: dict, scope: str) -> bool:
    return c["site"] == "pooled" and SCOPE_SLICES[scope] in (None, c["slices"])


def grid(s: int, m: str) -> dict:
    return RUNS[("control", s)]["test"][METHODS.index(m)]


def tuning(s: int, m: str, scope: str) -> tuple[np.ndarray, np.ndarray]:
    """Removal and spill of every configuration of *m* on the tuning set of the control at seed *s*, net of the
    median along the fifteen tuning axes (as post-hoc-compare stored them); NaN outside *scope*.
    """
    sel = RUNS[("control", s)]["selection"][m]
    r, sp = (np.array([np.nan if v is None else v for v in sel[k]], float) for k in ("removal", "spill"))
    out = ~np.array([in_scope(c, scope) for c in grid(s, m)["configs"]])
    r[out], sp[out] = np.nan, np.nan
    return r, sp


def test(s: int, m: str, scope: str) -> tuple[np.ndarray, np.ndarray]:
    """Removal and spill of every configuration of *m* on the test set of the control at seed *s*, net of the
    tuning-set axis median under its operator form; NaN outside *scope*.
    """
    run, g = RUNS[("control", s)], grid(s, m)
    clean, null = np.asarray(g["clean"]), np.asarray(g["null"])
    med = run["axes"]["median_drop"]
    base = np.array([med[str(ex.config_of(c).form)] for c in g["configs"]])
    net = clean[None] - np.asarray(g["scores"]) - base
    r, sp = net[:, D] / (clean[D] - null[D]), np.delete(net, D, axis=1).max(axis=1)
    out = ~np.array([in_scope(c, scope) for c in g["configs"]])
    r[out], sp[out] = np.nan, np.nan
    return r, sp


def frontier(r: np.ndarray, sp: np.ndarray) -> np.ndarray:
    ok = np.flatnonzero(np.isfinite(r))
    return ok[ex.frontier(r[ok], sp[ok])]


def mix(p: dict, v: np.ndarray) -> float:
    """*v* at a point of the selection rule: its bracketing configurations, mixed with their weights."""
    return float(sum(w * v[i] for i, w in zip(p["configs"], p["weights"], strict=True)))


def chosen(m: str, scope: str, at: float, loo: bool) -> list[dict]:
    """Per run, the point of the selection rule at a spill of *at* on the seed-mean tuning frontier of the other
    seven runs (*loo*) or of the run itself, and its removal and spill on the test set of that run.
    """
    out = []
    for s in SEEDS:
        per = [tuning(o, m, scope) for o in SEEDS if (o != s) == loo]
        tr, ts = np.mean([a for a, _ in per], axis=0), np.mean([b for _, b in per], axis=0)
        p = ex.read_point(tr, ts, frontier(tr, ts), at, "spill")
        r, sp = test(s, m, scope)
        out.append({"seed": s, "removal": mix(p, r), "spill": mix(p, sp), "weights": p["weights"],
                    "configs": [grid(s, m)["configs"][i] for i in p["configs"]]})  # fmt: skip
    return out


def summary(points: list[dict]) -> dict:
    r, sp = np.array([p["removal"] for p in points]), np.array([p["spill"] for p in points])
    return {"removal": r, "spill": sp, "over": int((sp > CRITERION).sum()), "miss": int((r < MISS).sum())}


EVERY = {m: summary(chosen(m, "every", CRITERION, loo=True)) for m in METHODS}
EVERY_MARGIN = {m: summary(chosen(m, "every", MARGIN, loo=True)) for m in METHODS}
EVERY_OWN = {m: summary(chosen(m, "every", CRITERION, loo=False)) for m in METHODS}
BLOCKS = {m: summary(chosen(m, "blocks", CRITERION, loo=True)) for m in METHODS}
BLOCKS_MARGIN = {m: summary(chosen(m, "blocks", MARGIN, loo=True)) for m in METHODS}
POOLED = {m: summary(chosen(m, "pooled", CRITERION, loo=True)) for m in METHODS}
EVERY_CONFIGS = {m: chosen(m, "every", CRITERION, loo=True) for m in METHODS}
A_OVER = int((A_SPILL > CRITERION).sum())
assert A_OVER == 0 and (A_REMOVAL >= MISS).all()
assert (A_SPILL_15 <= CRITERION).all() and (A_REMOVAL_15 >= MISS).all()
# The pooled choice on other runs reproduces post-hoc-compare: LEACE, INLP, and the probe keep 0.36 to 0.43 on 712.
assert all(0.35 < POOLED[m]["removal"][0] < 0.44 for m in ("ridge", "leace", "inlp"))


def mean_dose(points: list[dict]) -> float:
    """The seed mean of the dose of the configurations chosen, weighted as the rule mixed them."""
    return float(np.mean([sum(w * c["dose"] for w, c in zip(p["weights"], p["configs"], strict=True)) for p in points]))


DOSE = {m: mean_dose(EVERY_CONFIGS[m]) for m in METHODS}

# --- Presentation ---------------------------------------------------------------------------------------------

INK = {
    "anchor": ("#2a9d8f", "#7fd8c8"),
    "dim": ("#8d6a9f", "#c7a8d8"),
    "ridge": ("#d08c2a", "#f2b866"),
    "leace": ("#3a6ea5", "#8fb8e8"),
    "inlp": ("#b5485d", "#f08aa0"),
    "optimized": ("#555555", "#bbbbbb"),
}
MARKER = {"anchor": "D", "dim": "o", "ridge": "s", "leace": "^", "inlp": "v", "optimized": "*"}


def ink(m: str) -> str:
    return light_dark(*INK[m])


def rule() -> str:
    return light_dark("#333", "#ddd")


def num(v: float, spec: str = ".2f") -> str:
    return format(v, spec).replace("-", "−")


def span(v: np.ndarray) -> str:
    return f"{num(v.mean())} ({num(v.min())} to {num(v.max())})"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    code = lambda c: re.sub(r"`([^`]+)`", r"<code>\1</code>", c)  # noqa: E731
    ths = "".join(f"<th{' class=num' if i else ''}>{code(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{code(c)}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


r"""
# Fitted edits at every slice, carried to a run they were not chosen on

/// tip |
<!-- lede -->
We gave the post-hoc fitters the same shape of edit as the anchor: a pooled fit at every slice, with the configuration chosen on the other runs. In that shape no run can miss by editing the wrong slice. But the fitters do not carry over. At the earliest slices, the fitted edits cost the other ops so much that the edit can only be turned part of the way up, and three of the four fitters remove only a small part of the op on nearly every run. Leaving out the embedding helps the probe, which then carries over on most runs, with less removal than the anchor.
///

In the post-hoc-compare experiment, we chose a fitter configuration on seven runs and carried it to the eighth. It removed most of the op on seven runs. On seed 712 it missed for LEACE, INLP, and the probe: the choice made on the other runs edited one slice, and that run holds the op a slice earlier. The anchored edit acts at every slice, so it cannot miss in that way. Does a fitter given the same shape of edit carry over as well as the anchor does? This report answers from the scores that experiment stored, with no new runs and no new scoring.
"""

# %%

rf"""
## Observations

- [Carried to the run left out (E1)](#carried-to-the-run-left-out-e1): at every slice, with the choice made on the other runs, the standard fitters remove {num(min(EVERY[m]["removal"].mean() for m in FITTERS))} to {num(max(EVERY[m]["removal"].mean() for m in FITTERS))} of the op against {num(A_REMOVAL.mean())} for the anchor, and miss on most runs. The optimized direction (a ceiling, fitted against the scoring itself) comes close to the anchor.
- [Where the spill comes from (E2)](#where-the-spill-comes-from-e2): the cost to the other ops comes from the edits at the embedding and at slice 1, which add little removal. Slice 2 alone, where the single-slice choices of post-hoc-compare edited, costs about nothing.
- [Without the embedding (E3)](#without-the-embedding-e3): on slices 1 to 3, the probe carries over on all but one run, with less removal than the anchor and spill on two runs; the other three fitters still miss on most runs. This scope was added after the first results.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It follows the analysis "One configuration chosen on other runs" of post-hoc-compare, which was itself added after that run. We fixed the rule below and committed it before computing any of its numbers. So E1 and the choice on the run itself test one question, fixed in advance. We added the slice-by-slice scores of E2 and the scope of E3 after seeing E1, the second at the suggestion of a reader.

The runs are those of post-hoc-compare: the eight controls of the anneal confirmation at seeds 712 to 719, where every fitter is fitted and scored, and the eight `τ_ℓ fixed` runs at the same seeds for the anchor, at full dose, as published.

The fitters are difference in means, the ridge probe, LEACE, and INLP, with the optimized direction beside them as a ceiling. Each is restricted to the configurations that need no role labels and edit like the anchor. These are pooled fits: fitted on every position of a context, and editing every position at every slice (the embedding and each block output). The anchored edit acts at the same slices and positions. Within that scope, the grid still offers the target (binary or graded, where the fitter takes both), sequential refit (on or off), the replacement (zero or the mean), the rank of INLP, and the dose (a quarter, a half, three quarters, or all of the component).

Choice on the other seven runs
:   For each run in turn, we average the tuning scores of the other seven runs. The selection rule of post-hoc-compare then picks the point where the frontier of those scores crosses the spill criterion, interpolating between the two configurations on either side of it. The fitter is then fitted on the run left out with those configurations and scored on its test set, mixed with the same weights. This is the main measurement.

Choice with a margin
:   The same, with the point taken at a spill of 0.01 on the other seven runs, half the criterion, as a practitioner might leave room.

Choice on the run itself
:   The same rule on the tuning scores of the run itself, within the same scope. This involves no transfer; it says how much the every-slice restriction costs before anything is carried over.

Every score is net of the median drop along fifteen coordinate axes under the same operator form on the tuning set, as in the analysis this follows. The published anchored scores are net of the median over all 63 other axes. Against the same fifteen axes, the anchor removes {num(A_REMOVAL_15.mean())} and spills on no run, so the choice of baseline does not change the comparison. The permuted-label baseline was scored only at the configurations post-hoc-compare chose, so it is not available here. As a check on the code, the same rule over every pooled fit reproduces the numbers of post-hoc-compare, including the 0.36 to 0.43 that LEACE, INLP, and the probe keep on seed 712.

## Measurements

Removal
:   The drop in expected exact match on `{DIFF}` under the edit, as a share of the way from the clean score to the target null. Higher is better.

Spill
:   The largest drop on any other op under the edit. Lower is better, and the selectivity criterion is 0.02.

Over the criterion
:   A run whose spill on its test set is above 0.02.

Miss
:   A run whose removal is below 0.5, so the edit takes away less than half of the way to the target null. The anchor removes {num(A_REMOVAL.min())} or more on every run.
"""

# %%

r"""
## Carried to the run left out (E1)

Does a pooled fit at every slice, with its configuration chosen on the other seven runs, remove about as much as the anchor on every run, within the spill criterion? If it does, the miss on seed 712 was a matter of the slice, and a fitter that edits every slice would carry over as the anchor does. The figure shows the removal on each run, beside the same choice made over every pooled fit. That second choice is the setting of post-hoc-compare, where most choices edit one slice. It turns out the fitters remove much less at every slice.
"""


@memo
def e1_draw(every: dict, pooled: dict, a_removal: np.ndarray) -> str:
    @themed(
        name="fitter-transfer-e1",
        alt_text=(
            "Removal per run for the anchor and five methods, each method with two columns. In the left column of "
            "each fitter, any pooled fit, most runs sit near 0.8 to 0.95, with one run near 0.4 for the probe, LEACE, "
            "and INLP. In the right column, "
            "every slice, the four standard fitters sit between 0.1 and 0.4 with a few runs higher; the optimized "
            "direction sits near 0.84 in both columns, and the anchor near 0.88."
        ),
        caption=f"""**Removal per run with the configuration chosen on the other seven runs**, on the test set of each
        run, net. For each method the hollow column is the choice over every pooled fit (as in post-hoc-compare) and
        the filled column the choice over the pooled fits that edit every slice. A small dot per run and the seed
        mean as the marker; the anchor at full dose. The dashed line is a removal of {MISS}, under which a run
        misses.""",
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 2.8), layout="constrained")
        rng = np.random.default_rng(0)
        bg = light_dark("white", "#111")

        def column(x: float, v: np.ndarray, m: str, filled: bool) -> None:
            jit = rng.uniform(-0.06, 0.06, len(v))
            face = ink(m) if filled else "none"
            ax.plot([x, x], [v.min(), v.max()], "-", color=ink(m), lw=1.0, alpha=0.5, zorder=2)
            ax.plot(x + jit, v, "o", ms=2.6, color=ink(m), mfc=face, alpha=0.6, zorder=3, mew=0.6)
            ax.plot(x, v.mean(), MARKER[m], ms=6.5, color=ink(m), mfc=face if filled else bg, mec=ink(m) if not filled else bg, mew=0.9, zorder=4)  # fmt: skip

        column(0, a_removal, "anchor", True)
        ticks = [0.0]
        for k, m in enumerate(METHODS, start=1):
            column(k - 0.17, pooled[m]["removal"], m, False)
            column(k + 0.17, every[m]["removal"], m, True)
            ticks.append(k)
        ax.axhline(MISS, color=rule(), lw=0.9, ls="--", zorder=1)
        ax.set_xticks(ticks, [SHOWN["anchor"], *(SHOWN[m].replace(" ", "\n", 1) for m in METHODS)], fontsize=7)
        ax.set_ylabel("removal (net)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlim(-0.6, len(METHODS) + 0.6)
        ax.plot([], [], "o", color=rule(), mfc="none", ms=4, label="any pooled fit")
        ax.plot([], [], "o", color=rule(), ms=4, label="every slice")
        ax.legend(fontsize=7, frameon=False, loc="lower left", ncols=2)
        return fig

    return _plot()


e1_draw(EVERY, POOLED, A_REMOVAL)

# %%


def e1_table() -> str:
    rows = [[SHOWN["anchor"], span(A_REMOVAL), f"{A_OVER} of 8", "0 of 8", "", "", ""]]
    for m in METHODS:
        e, g = EVERY[m], EVERY_MARGIN[m]
        rows.append([SHOWN[m], span(e["removal"]), f"{e['over']} of 8", f"{e['miss']} of 8",
                     num(g["removal"].mean()), f"{g['over']} of 8", f"{g['miss']} of 8"])  # fmt: skip
    return table_html(
        ["", "removal ↑", "over ↓", "misses ↓", "removal, margin ↑", "over, margin ↓", "misses, margin ↓"],
        rows,
        f"""**Pooled fits at every slice, with the configuration chosen on the other seven runs**: removal on the test
        set of the run left out (seed mean and range, net), the runs over the spill criterion, and the runs that
        miss (removal under {MISS}), out of eight. The last three columns take the choice at a spill of {MARGIN} on
        the other runs. The anchor is the full edit along e₁ on the anchored runs.""",
    )


e1_table()

# %%

_STD_LO = min(EVERY[m]["removal"].mean() for m in FITTERS)
_STD_HI = max(EVERY[m]["removal"].mean() for m in FITTERS)
_MISS_LO = min(EVERY[m]["miss"] for m in FITTERS)
_DOSE_LO = min(DOSE[m] for m in FITTERS)
_DOSE_HI = max(DOSE[m] for m in FITTERS)
# REVIEW: "over the criterion" had one cause for all four fitters (a point chosen at the line landing on either side).
# For difference in means the least spill at every slice on the other seven runs is 0.045 to 0.054 for every run left
# out, and for LEACE it is above 0.02 for four of the eight, so there the rule has no point at the line and falls to
# the least-spilling configuration (a quarter dose). Verify: the minimum of the seed-mean tuning spill in the "every"
# scope, per run left out.

rf"""
No standard fitter carries over at every slice. With the choice made on the other runs, each removes {num(_STD_LO)} to {num(_STD_HI)} of the way to the target null on average, and each misses on {_MISS_LO} or more of the eight runs. They also land past the spill criterion on several runs. For the probe and INLP, a point chosen at the line on other runs lands on either side of it on the run left out. Difference in means has no configuration at every slice that the other runs keep within the criterion, and LEACE has none for half of the runs. In those cases the rule takes the configuration that spills least, at a quarter of the component (the lowest dose in the grid). With room under the criterion, the probe and INLP land over it on fewer runs and remove a little less, and difference in means and LEACE take the least-spilling configuration for every run. The probe with mean ablation does best. It removes about half of what the anchor removes, and still misses on five runs.

The configurations chosen show why. On average they remove {num(_DOSE_LO)} to {num(_DOSE_HI)} of the fitted component (the probe the most), where the anchored edit removes all of it. Within the criterion there is no room for more, because at every slice a fitted direction costs the other ops more than the criterion allows. Seed 712, the run that missed in post-hoc-compare, is now the best run for every standard fitter. That fits with it holding the op a slice earlier than the rest.

The optimized direction removes nearly as much as the anchor at every slice, and spills on no run. So an edit of this shape can remove the op from a control within the criterion, but the standard fitters do not find the direction that does it.
"""

# %%

r"""
## Where the spill comes from (E2)

A fitted edit at every slice costs the other ops a lot. Which slices bring that cost? And how much does the shape cost a fitter that chooses on its own run? The figure scores each slice edited alone at full dose, as seed means. It uses one form of each fitter: the binary target, no refit, rank 2 for INLP, and mean replacement (which nearly every choice of E1 used). The cost comes mostly from the embedding and slice 1.
"""


def slice_scores(m: str) -> tuple[list[float], list[float]]:
    """Seed-mean removal and spill at full dose of one pooled form of *m* (binary, mean replacement, no refit, rank 2
    for INLP), for each slice alone and for every slice together.
    """
    per = [test(s, m, "pooled") for s in SEEDS]
    r, sp = np.mean([a for a, _ in per], axis=0), np.mean([b for _, b in per], axis=0)
    cs = grid(SEEDS[0], m)["configs"]
    rank = 2 if m == "inlp" else 1
    plain = {tuple(c["slices"]): i for i, c in enumerate(cs) if c["site"] == "pooled" and c["target"] == "binary"
             and c["dose"] == 1.0 and c["replacement"] == "mean" and not c["refit"] and c["rank"] == rank}  # fmt: skip
    keys = [(s,) for s in range(ex.N_SLICES)] + [tuple(EVERY_SLICE)]
    return [float(r[plain[k]]) for k in keys], [float(sp[plain[k]]) for k in keys]


SLICE = {m: slice_scores(m) for m in FITTERS}


@memo
def e2_draw(slice_scores: dict) -> str:
    @themed(
        name="fitter-transfer-e2",
        alt_text=(
            "Two panels over the slice edited: the embedding, slices 1 to 4 alone, and every slice. Top, removal: "
            "every fitter peaks at slice 2 near 0.75 to 0.85 and is near zero at slices 3 and 4. Bottom, spill: "
            "difference in means and LEACE spill 0.2 to 0.4 at the embedding and slice 1 and about 0.4 to 0.5 at "
            "every slice; the probe and INLP stay under 0.1 at every point but sit over the criterion line at the "
            "embedding, slice 1, and every slice. At slice 2 every fitter is near zero spill."
        ),
        caption=f"""**Removal and spill for each slice edited alone, and for every slice together**, at full dose,
        seed mean over the eight controls on the test set, net, for one pooled form of each fitter (binary target,
        mean replacement, no refit, rank 2 for INLP). The
        dotted line in the lower panel is the spill criterion; the anchor at full dose sits at a removal of
        {num(A_REMOVAL.mean())} and a spill of {num(A_SPILL.mean())} at every slice.""",
    )
    def _plot() -> plt.Figure:
        fig, (top, bot) = plt.subplots(2, 1, figsize=(5.6, 3.6), sharex=True, layout="constrained")
        x = np.arange(ex.N_SLICES + 1, dtype=float)
        for k, m in enumerate(FITTERS):
            off = (k - 1.5) * 0.09
            r, sp = slice_scores[m]
            for ax, v in ((top, r), (bot, sp)):
                ax.plot(x[:-1] + off, v[:-1], "-", color=ink(m), lw=0.8, alpha=0.6, zorder=2)
                ax.plot(x + off, v, MARKER[m], color=ink(m), ms=4.5, mec=light_dark("white", "#111"), mew=0.5,
                        zorder=3, label=SHOWN[m])  # fmt: skip
        for ax, v in ((top, A_REMOVAL.mean()), (bot, A_SPILL.mean())):
            ax.plot(x[-1] + 0.3, v, MARKER["anchor"], color=ink("anchor"), ms=4.5, zorder=3, label=SHOWN["anchor"])
        bot.axhline(CRITERION, color=rule(), lw=0.8, ls=":", zorder=1)
        top.set_ylabel("removal", fontsize=8)
        bot.set_ylabel("spill", fontsize=8)
        bot.set_xticks(x, [*SLICE_NAMES, "every"], fontsize=7)
        bot.set_xlabel("slice edited", fontsize=8)
        for ax in (top, bot):
            ax.tick_params(labelsize=7)
            ax.axvline(x[-1] - 0.5, color=light_dark("#ccc", "#444"), lw=0.6, zorder=0)
        top.legend(fontsize=6.5, frameon=False, ncols=5, loc="upper left", bbox_to_anchor=(0, 1.3))
        return fig

    return _plot()


e2_draw(SLICE)

# %%

_S2_HI = max(SLICE[m][1][2] for m in FITTERS)
_EARLY_DL = min(min(SLICE[m][1][0], SLICE[m][1][1]) for m in ("dim", "leace"))
_EARLY_RI = max(max(SLICE[m][1][0], SLICE[m][1][1]) for m in ("ridge", "inlp"))
_GAIN = max(SLICE[m][0][-1] - SLICE[m][0][2] for m in FITTERS)

# REVIEW: "helps the probe and little else" became "the probe and INLP by about the same amount". Against the choice
# on the other runs (E1), the seed-mean removal rises by 0.22 for both (INLP 0.17 to 0.39, the probe 0.42 to 0.64);
# paired by run, the INLP gain is the steadier of the two. Verify: EVERY_OWN[m]["removal"] - EVERY[m]["removal"].
rf"""
At slice 2 alone, every fitter removes most of the op and spills {num(_S2_HI)} or less, within the criterion. At the embedding or at slice 1, difference in means and LEACE spill {num(_EARLY_DL)} or more (many times the criterion), and the probe and INLP spill up to {num(_EARLY_RI)} (a few times it). The slices after slice 2 hold little of the op to remove. Every slice together removes at most {num(_GAIN)} more than slice 2 alone, so the early slices add spill and little removal.

So in a control, the direction a fitter finds for the op at the early slices seems to also hold something the other ops use. The single-slice choices of post-hoc-compare worked because they avoided those slices.

With zero replacement in place of the mean, the probe and INLP spill several times more at the early slices, so mean ablation is much of why those two do better than the other fitters here.

That fits the miss on seed 712 in post-hoc-compare. That run holds the op at slice 1. A fitter spills at slice 1 on the other runs, so a rule chosen on them avoids it.

Choosing on the run itself helps the probe and INLP by about the same amount, and helps difference in means and LEACE little or not at all. With the same scope and the choice made on the tuning scores of the run itself, the probe removes {num(EVERY_OWN["ridge"]["removal"].mean())} and the other fitters {num(max(EVERY_OWN[m]["removal"].mean() for m in ("dim", "leace", "inlp")))} or less, and each lands over the criterion on most runs. For difference in means and LEACE, nearly all of the shortfall comes from the shape of the edit, before anything is carried over. For the probe it splits about evenly between the shape and the carrying, and its gain from choosing on its own run is within the seed spread.
"""


def e2_table() -> str:
    rows = []
    for m in METHODS:
        o = EVERY_OWN[m]
        rows.append([SHOWN[m], span(o["removal"]), f"{o['over']} of 8", f"{o['miss']} of 8"])
    return table_html(
        ["", "removal ↑", "over ↓", "misses ↓"],
        rows,
        """**Pooled fits at every slice, with the configuration chosen on the run itself**: removal on the test set
        (seed mean and range, net), the runs over the spill criterion, and the runs that miss, out of eight.""",
    )


e2_table()

# %%

r"""
## Without the embedding (E3)

The embedding is the slice where the fitters spill most, and a practitioner would seldom edit the token embeddings. Does a fitter carry over when it edits the blocks in the middle and leaves the embedding alone? The grid has no set of slices 1 to 4, but it has slices 1 to 3, and slice 4 holds little of the op. So the same rule picks among the pooled fits on slices 1 to 3, chosen on the other seven runs. A reader of the first draft suggested this scope after E1, so it is a second look at the same runs, not a test. The probe now carries over on all but one run.
"""


def e3_table() -> str:
    rows = [[SHOWN["anchor"], span(A_REMOVAL), f"{A_OVER} of 8", "0 of 8", "", "", ""]]
    for m in METHODS:
        b, g = BLOCKS[m], BLOCKS_MARGIN[m]
        rows.append([SHOWN[m], span(b["removal"]), f"{b['over']} of 8", f"{b['miss']} of 8",
                     num(g["removal"].mean()), f"{g['over']} of 8", f"{g['miss']} of 8"])  # fmt: skip
    return table_html(
        ["", "removal ↑", "over ↓", "misses ↓", "removal, margin ↑", "over, margin ↓", "misses, margin ↓"],
        rows,
        f"""**Pooled fits on slices 1 to 3, with the configuration chosen on the other seven runs**, laid out as the
        table of E1: removal on the test set of the run left out (seed mean and range, net), the runs over the spill
        criterion, and the runs that miss, out of eight; the last three columns take the choice at a spill of
        {MARGIN} on the other runs. Added after E1.""",
    )


e3_table()

# %%

_712 = [m for m in FITTERS if BLOCKS[m]["removal"].argmax() == 0]
_BEST712 = "every fitter" if len(_712) == len(FITTERS) else " and ".join(SHOWN[m] for m in _712) or "no fitter"

rf"""
Leaving out the embedding helps the probe most, INLP some, and LEACE and difference in means little. The probe removes {num(BLOCKS["ridge"]["removal"].mean())} on average and misses on {BLOCKS["ridge"]["miss"]} run of eight, against {EVERY["ridge"]["miss"]} at every slice, though it lands over the criterion on {BLOCKS["ridge"]["over"]} runs, one of them by a wide margin. With room under the criterion its removal falls to {num(BLOCKS_MARGIN["ridge"]["removal"].mean())}. INLP gains too, and still misses on most runs. LEACE and difference in means stay about where they were, since slice 1 still costs them many times the criterion (E2). Seed 712 is again the best run for {_BEST712}.

So one fixed recipe for the probe, chosen on other runs, does carry over on slices 1 to 3, at a lower removal than the anchor and with spill on some runs. For the other three fitters it does not.
"""

# %%

rf"""
## Discussion

In post-hoc-compare, the miss on seed 712 looked like something a fitter could avoid by editing every slice, as the anchor does. On these runs the fitters cannot take that shape. A fitted edit at the embedding and at slice 1 costs the other ops more than the criterion allows, so the selection rule turns the edit part of the way down, and three of the four fitters then remove little of the op. In a control, how much the fitted direction for the op is shared with the other ops changes with the slice, and the fitters differ in how much that matters. Leaving out the embedding, the probe with mean ablation carries over on most runs, with less removal than the anchor. LEACE and difference in means would also need a slice choice that avoids slice 1.

The anchored edit needs no such choice. Training put the op on e₁ at every slice and kept the other ops off it. So its full edit at every slice removes {num(A_REMOVAL.mean())} of the op and spills on no run. That supports the case for placing the op during training: the same recipe works on every run, with no choice of slice. A fitter with a per-run slice choice still removes about as much as the anchor here (post-hoc-compare). So the case is about carrying a recipe to a new model. And the probe on slices 1 to 3 shows that a fitter recipe can carry over part of the way.

The optimized direction shows that an edit of this shape exists in each control. It searches against the target null, which no real pipeline has, so it says what a rank-1 edit at every slice could do.

This covers one concept, one small model, and eight runs per arm. It is also a re-analysis added after the comparison it follows, with E3 added after its own first results, so it describes what we saw. The edit at every slice, embedding included, is also a design of ours, since fitters are usually applied at a chosen layer. The doses come in quarters, so a choice between a quarter and a half is a mix of the two scores, which may differ a little from the score at a dose in between. A fitter with a dose per slice (smaller at the early slices) might do better at every slice; that would be a design of ours with no published baseline behind it, and the grid does not have it.
"""

# %%

r"""
## References

[post-hoc-compare](/docs/m2/post-hoc-compare/report.py)
:   Restricted to fits that need no role labels, LEACE removes about as much of the op as the anchor; with a configuration chosen on the other seven runs, the fitters miss seed 712, which holds the op a slice earlier than the rest.

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` at fresh seeds 712 to 719 keeps every anchor; it is the recipe of record, and its checkpoints and dose scoring are stored.

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   Trained the controls at seeds 712 to 719 that the fitters are fitted on, sharing initialization, batches, and corpus with the anchored runs.
"""
