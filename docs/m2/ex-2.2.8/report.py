# title: Ex 2.2.8: a survey of the intervention operator

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import AxesGrid, AxesRow, figure_html, light_dark, themed

SLICE_NAMES = ["emb", "1", "2", "3", "4"]
OPS = list(ex.OP_NAMES)
PRIMARY_OP = "mix"
FAMILIES = ("projection", "shaped", "repulsion-linear")
FAMILY_TITLE = {
    "projection": "projection",
    "shaped": "shaped suppression",
    "repulsion-linear": "repulsion",
}
INK = {
    "projection": ("#333", "#ddd"),
    "shaped": ("#d0461b", "#f07a50"),
    "repulsion-linear": ("#1f6fb4", "#5fa8dd"),
}
# One ink per operator family, as (light, dark) pairs.


def load_jsons(refs: Sequence[str]) -> dict[str, dict | None]:
    """Each published JSON result as a dict, or None before it exists.

    One `get_refs` and one `get_many` for the lot: resolving refs one at a time pays the bucket's per-call latency each time.
    """
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    with tempfile.TemporaryDirectory() as d:
        paths = store.get_many([(a, Path(d) / f"{i}.json") for i, a in enumerate(have.values())])
        return dict.fromkeys(refs) | {r: json.loads(p.read_text()) for r, p in zip(have, paths, strict=True)}


def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:{fmt}}"
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


def ink(family: str):
    return light_dark(*INK[family])


def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


@dataclass(frozen=True)
class Results:
    """The survey's published scores, and ex-2.2.3's metrics for the reference rows."""

    metrics: dict
    prod: dict

    @property
    def trials(self) -> list[dict]:
        return self.metrics["design"]["trials"]

    def trial(self, name: str) -> dict:
        return next(t for t in self.trials if t["name"] == name)

    def runs(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def stat(self, cond: str, op: str, trial: str | None, key: str, group: str) -> np.ndarray:
        """One statistic over the seeds of a condition: the clean value when *trial* is None."""
        out = []
        for r in self.runs(cond):
            s = r["ops"][op]
            v = s["clean"][key] if trial is None else s["trials"][trial][key]
            out.append(v[group])
        return np.array(out, float)

    def per_slice(self, cond: str, op: str, trial: str | None, key: str) -> np.ndarray:
        """(seeds, slices) of a per-slice statistic."""
        out = []
        for r in self.runs(cond):
            s = r["ops"][op]
            out.append(s["clean"][key] if trial is None else s["trials"][trial][key])
        return np.array(out, float)

    def prod_stat(self, cond: str, op: str, iv: str, key: str, group: str) -> np.ndarray:
        """Ex-2.2.3's stored value of the same statistic under one of its interventions, over the same seeds."""
        names = (cond, f"{cond}-more")
        runs = sorted((r for r in self.prod["scores"] if r["condition"] in names), key=lambda r: r["seed"])
        return np.array([r["ops"][op]["interventions"][iv][key][group] for r in runs], float)


@dataclass(frozen=True)
class Row:
    """One trial's seed-mean reads on one condition, rounded to the third decimal (the survey's reporting precision), so
    that trials the report cannot tell apart tie rather than being ranked on rounding.
    """

    name: str
    family: str
    a: float
    b: float
    p: float
    operands: bool
    red_acc: dict[str, float]
    """Seed-mean accuracy on the red lines, per op."""
    deficit: dict[str, float]
    """Seed-mean non-red deficit, per op."""
    red_acc_sd: dict[str, float]
    deficit_sd: dict[str, float]

    @property
    def worst_deficit(self) -> float:
        return max(self.deficit.values())

    @property
    def worst_red_acc(self) -> float:
        return max(self.red_acc.values())

    @property
    def feasible(self) -> bool:
        """The frozen constraint: non-red deficit within the gate on every op."""
        return self.worst_deficit <= ex.NONRED_DEFICIT_GATE

    @property
    def feasible_mix(self) -> bool:
        """The looser read ex-2.2.3's H4 gated: within the gate on `mix` alone."""
        return self.deficit[PRIMARY_OP] <= ex.NONRED_DEFICIT_GATE

    @property
    def removes(self) -> bool:
        return self.worst_red_acc <= ex.RED_ACC_GATE

    @property
    def margin(self) -> float:
        """Distance to the constraint, positive inside it."""
        return ex.NONRED_DEFICIT_GATE - self.worst_deficit


@dataclass(frozen=True)
class Summary:
    """The derived counts and picks the tl;dr and observations quote, computed once."""

    n_feasible: int
    n_whole: int
    n_whole_feasible: int
    n_op: int
    n_op_feasible: int
    front: list[Row]
    ref: Row
    ops: Row
    shaped_ref: Row
    costly: list[Row]
    best_whole: Row | None
    n_whole_free: int
    best_free: Row | None
    least_whole: Row
    split: list[Row]
    q99: dict[str, float]
    t00_ref: Row
    t00_ops: Row
    t00_whole: tuple[float, float]
    t00_step: tuple[float, float]
    t00_whole_feasible: int
    t00_op_feasible: int


def rows_for(res: Results, cond: str) -> list[Row]:
    rows_out = []
    for t in res.trials:
        red = {op: res.stat(cond, op, t["name"], "acc", "red") for op in OPS}
        dfc = {op: res.stat(cond, op, t["name"], "deficit", "nonred") for op in OPS}
        rows_out.append(
            Row(
                t["name"],
                t["family"],
                t["a"],
                t["b"],
                t["p"],
                t["positions"] is not None,
                {op: round(float(v.mean()), 3) + 0.0 for op, v in red.items()},
                {op: round(float(v.mean()), 3) + 0.0 for op, v in dfc.items()},
                {op: float(v.std(ddof=1)) if len(v) > 1 else float("nan") for op, v in red.items()},
                {op: float(v.std(ddof=1)) if len(v) > 1 else float("nan") for op, v in dfc.items()},
            )
        )
    return rows_out


def proposal(rows_in: list[Row]) -> Row | None:
    """The frozen rule: among feasible trials, the lowest red accuracy on `mix`."""
    feasible = [r for r in rows_in if r.feasible]
    return min(feasible, key=lambda r: r.red_acc[PRIMARY_OP]) if feasible else None


def describe(r: Row) -> str:
    """A trial in words: its family and parameters, and where it edits."""
    where = "at the operand positions" if r.operands else "at every position"
    if r.family == "projection":
        return f"the plain projection {where}"
    params = f"threshold {r.a:g}, " + (f"landing {r.b:g}" if r.family == "repulsion-linear" else f"ramp p = {r.p:g}")
    return f"{FAMILY_TITLE[r.family]} at {params}, {where}"


def front(rows_in: list[Row], op: str = PRIMARY_OP) -> list[Row]:
    """The trials no other trial beats on both reads at once (lower red accuracy and lower deficit on *op*)."""
    keep = []
    for r in rows_in:
        dominated = any(
            (o.red_acc[op] <= r.red_acc[op] and o.deficit[op] <= r.deficit[op])
            and (o.red_acc[op] < r.red_acc[op] or o.deficit[op] < r.deficit[op])
            for o in rows_in
        )
        if not dominated:
            keep.append(r)
    return sorted(keep, key=lambda r: r.deficit[op])


def family_lines(ax, rows_in: list[Row], x: str, key: str, color, label: str) -> None:
    """One family's trials on `mix` against one parameter: solid at every position, dashed at the operands."""
    for operands, ls in ((False, "-"), (True, "--")):
        pts = sorted((getattr(r, x), getattr(r, key)[PRIMARY_OP]) for r in rows_in if r.operands == operands)
        if pts:
            ax.plot(
                *zip(*pts, strict=True), ls=ls, marker="o", ms=3, lw=1, color=color, label=None if operands else label
            )


def trial_mark(ax, x: float, y: float, row: Row, ringed: bool) -> None:
    """A trial's mark: $×$ for a reference projection; otherwise the family's ink, $●$ at every position and $▲$ at
    the operands, filled when feasible on every op and open otherwise; a ring around the proposed trial.
    """
    m = "^" if row.operands else "o"
    if row.family == "projection":
        ax.scatter(x, y, marker="x", s=36, color=ink("projection"), zorder=4)
    elif row.feasible:
        ax.scatter(x, y, marker=m, s=22, color=ink(row.family), linewidths=0.9, zorder=3)
    else:
        ax.scatter(
            x, y, marker=m, s=22, facecolors="none", edgecolors=ink(row.family), alpha=0.7, linewidths=0.9, zorder=3
        )
    if ringed:
        ax.scatter(x, y, marker="o", s=120, facecolors="none", edgecolors=ink(row.family), linewidths=1.2, zorder=5)


loaded = load_jsons([ex.METRICS_REF, ex.EX223_METRICS_REF])
metrics, prod = loaded[ex.METRICS_REF], loaded[ex.EX223_METRICS_REF]
if metrics is None:
    stop("_Results are not published yet; the result cells render once they are._")
if prod is None:
    stop("ex-2.2.3's metrics are missing from the store")
res: Results = Results(metrics, prod)
rows: dict[str, list[Row]] = {c: rows_for(res, c) for c in ex.CONDITIONS}

# The noise floors: per-run σ of each objective under the reference projection, at the adopted point's seeds.
noise_cond = "recipe-short"
noise_n = len(res.runs(noise_cond))
sd_red = res.stat(noise_cond, PRIMARY_OP, "projection", "acc", "red").std(ddof=1)
sd_def = res.stat(noise_cond, PRIMARY_OP, "projection", "deficit", "nonred").std(ddof=1)
band = {"red_acc": 2 * sd_red * np.sqrt(2 / noise_n), "deficit": 2 * sd_def * np.sqrt(2 / noise_n)}
# How small a difference between two seed means at twenty seeds the survey can resolve, per objective.
recipe_rows = rows[noise_cond]
recipe_by_name = {r.name: r for r in recipe_rows}
prop = proposal(recipe_rows)
prop_t00 = proposal(rows["t00"])
ref_row, ops_row = recipe_by_name["projection"], recipe_by_name["operands"]
whole_rows = [r for r in recipe_rows if not r.operands and r.family != "projection"]
op_rows = [r for r in recipe_rows if r.operands and r.family != "projection"]
# Whole-sequence trials that cost the non-red `mix` lines more than projecting everything does.
costly_rows = sorted(
    (r for r in whole_rows if r.deficit[PRIMARY_OP] > ref_row.deficit[PRIMARY_OP]), key=lambda r: -r.deficit[PRIMARY_OP]
)
whole_feasible_rows = [r for r in whole_rows if r.feasible]
best_whole = min(whole_feasible_rows, key=lambda r: r.red_acc[PRIMARY_OP]) if whole_feasible_rows else None
# ...and of those, the ones whose cost is within a band of zero, and the one that removes the most.
whole_free_rows = [r for r in whole_rows if r.worst_deficit <= band["deficit"]]
best_free = min(whole_free_rows, key=lambda r: r.red_acc[PRIMARY_OP]) if whole_free_rows else None
least_whole = max(whole_rows, key=lambda r: r.red_acc[PRIMARY_OP])
# Whole-sequence trials whose operand-only copy differs by more than a band on either `mix` read.
split_rows = [
    r
    for r in whole_rows
    if abs(r.red_acc[PRIMARY_OP] - recipe_by_name[r.name + "-operands"].red_acc[PRIMARY_OP]) > band["red_acc"]
    or abs(r.deficit[PRIMARY_OP] - recipe_by_name[r.name + "-operands"].deficit[PRIMARY_OP]) > band["deficit"]
]
# The non-red `mix` lines' clean 99th-percentile alignment at its highest slice and position, per condition:
# where a threshold starts to catch them.
q99 = {c: float(res.per_slice(c, PRIMARY_OP, None, "alpha_q99_nonred").mean(0).max()) for c in rows}
# The `t00` side: the whole-sequence cost's range, and the steps' cost against projecting everything.
t00_rows = rows["t00"]
t00_by_name = {r.name: r for r in t00_rows}
t00_whole_rows = [r for r in t00_rows if not r.operands and r.family != "projection"]
t00_step_rows = [r for r in t00_whole_rows if r.family == "shaped" and r.p == 0]
summary = Summary(
    n_feasible=sum(r.feasible for r in recipe_rows),
    n_whole=len(whole_rows),
    n_whole_feasible=len(whole_feasible_rows),
    n_op=len(op_rows),
    n_op_feasible=sum(r.feasible for r in op_rows),
    front=front(recipe_rows),
    ref=ref_row,
    ops=ops_row,
    shaped_ref=recipe_by_name["shaped-a0.5-p1"],
    costly=costly_rows,
    best_whole=best_whole,
    n_whole_free=len(whole_free_rows),
    best_free=best_free,
    least_whole=least_whole,
    split=split_rows,
    q99=q99,
    t00_ref=t00_by_name["projection"],
    t00_ops=t00_by_name["operands"],
    t00_whole=(min(r.deficit[PRIMARY_OP] for r in t00_whole_rows), max(r.deficit[PRIMARY_OP] for r in t00_whole_rows)),
    t00_step=(min(r.deficit[PRIMARY_OP] for r in t00_step_rows), max(r.deficit[PRIMARY_OP] for r in t00_step_rows)),
    t00_whole_feasible=sum(r.feasible for r in t00_whole_rows),
    t00_op_feasible=sum(r.feasible for r in t00_rows if r.operands and r.family != "projection"),
)

n_trials = len(rows["recipe-short"])
summary_ref, summary_ops, summary_shaped = summary.ref, summary.ops, summary.shaped_ref
assert summary.best_free is not None, "the tl;dr assumes a free trial exists"
summary_best_free, summary_least_whole = summary.best_free, summary.least_whole
summary_costly, summary_split = summary.costly, summary.split
summary_t00_ref, summary_t00_ops = summary.t00_ref, summary.t00_ops


def worst_op(row: Row) -> str:
    return max(row.deficit, key=lambda op: row.deficit[op])


prop_line = (
    f"**Proposed operator: `{prop.name}`,** {describe(prop)}. Its worst deficit over the six ops is "
    f"{prop.worst_deficit:.3f}, on `{worst_op(prop)}`, a margin of {prop.margin:+.3f} "
    f"([proposal](#the-proposal))."
    if prop is not None
    else "**No trial is feasible on every op.** The infeasibility map is the finding; nothing is proposed."
)
t00_line = (
    f"On `t00` the same rule picks `{prop_t00.name}` (red accuracy {prop_t00.red_acc[PRIMARY_OP]:.3f}, deficit "
    f"{prop_t00.deficit[PRIMARY_OP]:.3f} on `mix`)."
    if prop_t00 is not None
    else "On `t00` no trial is feasible on every op."
)
costly_txt = ", ".join(f"`{r.name}` {r.deficit[PRIMARY_OP]:.3f}" for r in summary_costly)
costly_red_txt = ", ".join(f"{r.red_acc[PRIMARY_OP]:.3f}" for r in summary_costly)
split_txt = ", ".join(f"`{r.name}`" for r in summary_split)

rf"""
# Ex 2.2.8: a survey of the intervention operator on the stored ex-2.2.3 checkpoints

/// tip |
<!-- tl;dr -->
A survey of {n_trials} intervention operators on stored ex-2.2.3 checkpoints, with no training, looking for one that removes *red* as fully as the plain projection and is as selective as the operand-only one. On the adopted point the plain projection is inside the selectivity gate on every op, so the frozen rule proposes it; a threshold on the alignment removes less *red* and is no more selective that we could measure. On `t00` the syntax embeddings carry the axis, and only the operand-only edits are inside the gate.
///

## Observations

- **The reference rows reproduce.** The `projection` and `operands` rows match their stored ex-2.2.3 values on every seed, op, and group ([table](#the-reference-rows-reproduce)).
- **Noise floors.** Under `projection` at the twenty seeds of `recipe-short`, two seed means can be told apart when they differ by more than {band["red_acc"]:.3f} in red accuracy or {band["deficit"]:.3f} in non-red deficit. We cannot resolve smaller differences.
- **On the adopted point, the plain projection is inside the gate.** Its worst op is `mix`: non-red deficit {summary_ref.deficit[PRIMARY_OP]:.3f} at red accuracy {summary_ref.red_acc[PRIMARY_OP]:.3f}. The operand-only projection is at {summary_ops.deficit[PRIMARY_OP]:.3f} and {summary_ops.red_acc[PRIMARY_OP]:.3f}. We expected to see the whole-sequence cost that ex-2.2.3 found on the first points its rule proposed, but at twenty seeds the adopted point does not show it. Both projections are feasible, and the plain one removes more *red*, so the rule proposes it. It clears the gate by {summary_ref.margin:+.3f}, less than one band.
- **A threshold inside the non-red range costs more than projecting everything.** On the non-red `mix` lines the clean alignment reaches {q99["recipe-short"]:.2f} (99th percentile over slices and positions). Applied at every position, the steps at a ≤ 0.3 cost {costly_txt}. That is above the {summary_ref.deficit[PRIMARY_OP]:.3f} of the projection, and they remove about as much *red* ({costly_red_txt}). So zeroing the axis on some states of a line while leaving their neighbors alone costs that line more than zeroing all of them. These {len(summary_costly)} trials are the only ones that cost more than the projection ([figure](#the-landscape)).
- **Above the non-red range the cost vanishes, and what is left of *red* follows the landing.** {summary.n_whole_feasible} of the {summary.n_whole} tuned trials applied at every position are inside the gate on every op, and {summary.n_whole_free} of those cost within a band of zero. Of those {summary.n_whole_free}, `{summary_best_free.name}` removes the most, leaving red accuracy {summary_best_free.red_acc[PRIMARY_OP]:.3f}. Red accuracy rises with the threshold, the ramp, and the landing, up to {summary_least_whole.red_acc[PRIMARY_OP]:.3f} at `{summary_least_whole.name}` ([figure](#the-marginals)). The shaped suppression scored in ex-2.2.1 (`{summary_shaped.name}`) is at {summary_shaped.red_acc[PRIMARY_OP]:.3f} and {summary_shaped.deficit[PRIMARY_OP]:.3f}. A state left part-way along the axis keeps part of the color it held: at the last slice the alignment of the red operand settles at the threshold or the landing, as designed ([figure](#where-the-operators-leave-the-state)).
- **The position mask only matters where the threshold is low.** For {summary.n_whole - len(summary_split)} of the {summary.n_whole} tuned trials, the whole-sequence row and its operand-only copy agree on both `mix` reads to within a band. The {len(summary_split)} that differ are {split_txt}. Where a threshold already leaves the non-red states alone, the position mask changes nothing.
- **On `t00` only operand-only edits are feasible.** There the syntax embeddings hold the axis, and the clean alignment of the non-red lines reaches {q99["t00"]:.2f}. Projecting everything costs {summary_t00_ref.deficit[PRIMARY_OP]:.3f} on `mix`. Every tuned trial applied at every position costs between {summary.t00_whole[0]:.3f} and {summary.t00_whole[1]:.3f}, and the steps cost {summary.t00_step[0]:.3f} to {summary.t00_step[1]:.3f}. So partial removal costs more than full removal there too. All {summary.t00_op_feasible} tuned operand-only trials are inside the gate, as is `operands`, at {summary_t00_ops.red_acc[PRIMARY_OP]:.3f} and {summary_t00_ops.deficit[PRIMARY_OP]:.3f}. At five seeds we cannot tell the operand-only rows near it apart.
- **The front is short.** {summary.n_feasible} of {n_trials} trials are feasible on every op: both projections, {summary.n_whole_feasible} of the {summary.n_whole} tuned trials at every position, and {summary.n_op_feasible} of the {summary.n_op} at the operands. The `mix` front has {len(summary.front)} trials, from the gentlest edit to the most complete ([table](#every-trial)).
- {prop_line} {t00_line}

"""

r"""
/// admonition | How to read this report
This is a survey, so it scores no hypothesis. What it does preregister is a search plan: the trial list, the objective, its constraint, and the noise trials. All of those were frozen in `experiment.py` before the run.

Every trial is published, and nothing here may be quoted as a result. The anchored-op prereg adopts the proposed operator and re-measures it at fresh seeds, reporting the survey value beside the confirmed one. The checkpoints are stored, so these are the same seeds ex-2.2.3 scored, and the operator choice is the part that is not fresh: this survey makes it after seeing those seeds.

The plan was rehearsed once on the dev storage pair before this production run. That rehearsal also scored a Bézier-mapped repulsion; the production plan leaves it out ([search plan](#search-plan)).
///
"""

rf"""
## Why, and what we ran

Ex-2.2.1 left three operators, none of them both complete and selective. The plain projection removes *red* fully, but it costs the non-red lines. Ex-2.2.3 saw that cost on the syntax embeddings of the first points its selection rule proposed, where the `mix` deficits were well above the {ex.NONRED_DEFICIT_GATE:g} gate, so it amended the rule with the selectivity gate and adopted `recipe-short`. On that point the projection's cost is inside the gate, with little to spare.

The other two each give something up. The operand-only projection avoids the cost, but it needs to know the syntax of the line. The shaped suppression at the threshold used in M1 (a = 0.5, b = 1, p = 1) removes about half of *red* at no non-red cost.

So the [design](../d2.2/design.md) called for this scoring-only pass on stored runs before the anchored-op prereg, and two backlog items name it as their closing move ([shaped suppression](https://github.com/z0u/sca2/blob/main/todo/science/shaped-suppression-rather-than-projecting-whole-axis.md), [repulsion](https://github.com/z0u/sca2/blob/main/todo/science/repulsion-sets-the-landing-alignment.md)). Does any operator remove as much as the plain projection while keeping the margin of the operand-only edit? And does one of them do that with no position mask?

The design named the nine checkpoints from ex-2.2.1. D2.2 has since moved to the six-op grammar and adopted `recipe-short`, so this pass scores the stored runs of ex-2.2.3 instead: the adopted point at all twenty seeds, and `t00` at five. `t00` was the first point the rule proposed, and its syntax embeddings carry the axis at more than twice the level the recipe does.

Each checkpoint is scored on the six-op probe set from ex-2.2.3 through `sca.intervention.apply`. The lines of all six ops are concatenated, so each operator is one forward pass. The results are then read per op with the readout from ex-2.2.3, so every statistic means what it meant there.

The `repulsion` operator is new to the contract. The shaped suppression removes a fraction of the axis component; repulsion instead sets where the state lands, moving every state at or above the threshold *a* to alignment *b*. That is a step at the threshold, except when a = b, where it becomes the ceiling $\min(\alpha, b)$. The write is the angle between the arriving alignment and the landing one, and the scorer checks it against the measured rotation on every state.
"""

rf"""
/// details | Glossary
- **trial** — one operator: a family, its parameters, and the positions it edits. Every trial is scored on every stored seed, so each read is a seed mean over twenty runs on `recipe-short` and five on `t00`.
- **residual stream** — the vector the transformer carries from layer to layer, which each layer reads from and writes back to. Here it is unit-norm.
- **α** — how well a state lines up with the anchor axis $e_1$. Since the stream is unit-norm, that is just the first coordinate of the state. **landing** — where an operator leaves a state that it edits, as an alignment.
- **shaped suppression** — one of the operators from M1: above a threshold *a*, remove a fraction $h(\alpha) = b \cdot ((\alpha - a)/(1 - a))^p$ of the axis component, then re-normalize. `p = 0` is a step, meaning full removal above the threshold.
- **repulsion** — the other operator from M1: above the threshold *a*, land the state at alignment *b* on the axis, keeping its off-axis direction.
- **red lines** — probe lines whose dose (the redness of the redder operand) is at least {ex.RED_DOSE:g}; **non-red lines** — dose at most {ex.NONRED_DOSE:g}. **red accuracy** is exact match on the red lines, so a lower value means more of *red* was removed. The **non-red deficit** is the drop in P(answer) on the non-red lines, so a lower value means the edit was more selective. Both are as in H4 of ex-2.2.3.
- **feasible** — seed-mean non-red deficit within {ex.NONRED_DEFICIT_GATE:g} on every op. **band** — 2σ·√(2/n), built from the per-run σ of a statistic under `projection` at twenty seeds. It is the smallest difference between two seed means the survey can resolve.
- **front** — the trials on `mix` that no other trial beats on both reads at once.
///
"""

search_trials = ex.TRIALS

rf"""
## Search plan

Frozen in `experiment.py` before the run. The space is a grid: two families whose parameters mean something, so every corner is worth a look. At {len(search_trials)} operators on stored checkpoints, the whole grid costs less than one training run.

**The space.** {len(ex.REFERENCE)} reference rows: the `projection` from ex-2.2.3, applied at every position and again at the operand positions. Then two families, each at every position and again at the operand positions only:

| family | grid | trials |
| --- | --- | ---: |
| shaped suppression | a ∈ {{{", ".join(f"{a:g}" for a in ex.SHAPED_A)}}} × p ∈ {{{", ".join(f"{p:g}" for p in ex.SHAPED_P)}}}, b = 1 | {len(ex.SHAPED)} |
| repulsion | (a, b) ∈ {{{", ".join(f"({a:g}, {b:g})" for a, b in ex.LINEAR_AB)}}} | {len(ex.LINEAR)} |

`shaped` at p = 0 is a thresholded projection, and repulsion at b = 0 does the same thing, so that edge is not repeated. Counting the operand-only copies, that is {len(search_trials)} trials in all.

**What the rehearsal changed.** The plan first ran on the dev storage pair, so it was run again here on production. The rehearsal also included the Bézier-mapped repulsion from M1, at eight points; that version is continuous at the threshold. Each of those eight matched the linear row at the same landing on both reads, so the production plan drops that family. Nothing else changed.

**The objective.** {ex.OBJECTIVE}

**Noise floors.** The per-run σ of each objective is read under `{ex.NOISE_TRIALS[0]}` at the twenty seeds of `recipe-short`. A difference between two seed means smaller than 2σ·√(2/20) is unresolved. Every trial is read on the same twenty seeds, so this band is conservative for a paired comparison.

**What is checked, per trial and seed.** The assertions in the contract run on every state: the clean pass matches, the edit stays within the named positions, and where the write has a closed form (projection and repulsion) the measured rotation matches it to 2 × 10⁻³ rad. A trial whose write did not match would have failed the scoring task rather than being scored.

**Not in the plan.** We did not try positions other than the operands and all, and we did not edit only some of the slices. Operators fitted to the data (LEACE, diff-in-means) are left to the anchor-versus-fitted comparison in the D2.2 design. The readout still reports the redder-than-both lines, but this pass does not rank on them.
"""

r"""
## The reference rows reproduce

The two projection rows from ex-2.2.3, re-scored by this pass on the same checkpoints and probe lines, beside their stored values.
"""

check_pairs = {"projection": ex.ex223.PROJECTION.name, "operands": "operands"}
check_table_rows = []
check_worst = 0.0
for check_cond in ex.CONDITIONS:
    for check_here, check_there in check_pairs.items():
        check_diffs = {}
        for check_key, check_grp in (("acc", "red"), ("acc", "nonred"), ("deficit", "nonred"), ("p_ans", "all")):
            check_diff = max(
                float(
                    np.abs(
                        res.stat(check_cond, op, check_here, check_key, check_grp)
                        - res.prod_stat(check_cond, op, check_there, check_key, check_grp)
                    ).max()
                )
                for op in OPS
            )
            check_diffs[f"{check_key}/{check_grp}"] = check_diff
            check_worst = max(check_worst, check_diff)
        check_table_rows.append(
            [f"`{check_cond}`", f"`{check_here}`", *(f"{v:.1e}" if v else "0" for v in check_diffs.values())]
        )

table_html(
    ["condition", "row", "red acc", "non-red acc", "non-red deficit", "P(ans), all"],
    check_table_rows,
    "**Largest |re-scored − stored| per statistic.** Over the seeds of the condition, the six ops, and the group named. The floating-point path differs only in batch composition (the six ops are scored in one pass here).",
)

# %%

noise_table_rows = []
for noise_op in OPS:
    noise_red = res.stat(noise_cond, noise_op, "projection", "acc", "red")
    noise_deficit = res.stat(noise_cond, noise_op, "projection", "deficit", "nonred")
    noise_table_rows.append(
        [
            f"`{noise_op}`",
            span2(noise_red),
            f"{noise_red.std(ddof=1):.3f}",
            span2(noise_deficit),
            f"{noise_deficit.std(ddof=1):.3f}",
        ]
    )

rf"""
## Noise floors

Measured under `projection` at the twenty seeds of `recipe-short`. The rule ranks on `mix`, and the band there is {band["red_acc"]:.3f} on red accuracy and {band["deficit"]:.3f} on the non-red deficit.
""" + table_html(
    ["op", "red accuracy", "σ", "non-red deficit", "σ"],
    noise_table_rows,
    "**Per-run spread of the two objectives, by op.** Seed mean ± half the seed range, and the per-run standard deviation the bands are built from.",
)

# %%

landscape_rows = rows["recipe-short"]
landscape_xlim, landscape_ylim = (-0.01, 0.3), (-0.02, 1.0)


@memo
@themed(
    name="landscape",
    alt_text="""
        Six scatter panels, one per op, each with the non-red deficit on the horizontal axis, zoomed to the first third of its range, and red-line accuracy on the vertical, dashed gate lines near the origin. On every op nearly all marks stand in a vertical column at zero deficit, spanning red accuracy from near zero to about 0.8, circles and triangles together; the two projection crosses sit at the foot of the column just inside the deficit gate, with a ring around the plain one in every panel; and three open circles trail to the right of the gate at low red accuracy, the low-threshold steps.
    """,
    caption=f"""
        **The landscape: removal against selectivity, per op.** One mark per trial, seed means over the twenty seeds of `recipe-short`: $●$ at every position, $▲$ at the operand positions, filled when the trial is inside the deficit gate on every op and open otherwise, in the family's ink. $×$ marks the two reference projections. Dashed lines are ex-2.2.3's gates ({ex.NONRED_DEFICIT_GATE:g} on the deficit, {ex.RED_ACC_GATE:g} on red accuracy); the corner they enclose is where an operator is both selective and complete. The ring is the proposed trial. The deficit axis is zoomed to the range the adopted point uses; the figure below shows the full range beside `t00`.
    """,
)
def landscape_plot() -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesGrid, axes)
    grey = light_dark("#888", "#aaa")
    for ax, op in zip((a for row in axes for a in row), OPS, strict=True):
        ax.axvline(ex.NONRED_DEFICIT_GATE, ls="--", lw=0.7, color=grey, zorder=0)
        ax.axhline(ex.RED_ACC_GATE, ls="--", lw=0.7, color=grey, zorder=0)
        for r in landscape_rows:
            x, y = r.deficit[op], r.red_acc[op]
            trial_mark(ax, x, y, r, prop is not None and r.name == prop.name)
        ax.set_title(op, fontsize=9)
        ax.set_xlim(*landscape_xlim)
        ax.set_ylim(*landscape_ylim)
    for ax in axes[1]:
        ax.set_xlabel("non-red deficit ↓")
    for row in axes:
        row[0].set_ylabel("red accuracy ↓")
    # A small legend for the families, in grey marker shapes plus the inks.
    for fam in FAMILIES[1:]:
        axes[0][0].scatter([], [], color=ink(fam), marker="o", s=22, label=FAMILY_TITLE[fam])
    axes[0][0].scatter([], [], color=light_dark("#666", "#bbb"), marker="^", s=22, label="operand positions")
    axes[0][0].legend(fontsize=7, loc="upper right", frameon=False)
    return fig


r"""
## The landscape

Every trial on every op, on the two reads the objective uses. On the adopted point nearly every trial stands at zero deficit, and the position mask makes no difference; what separates the trials is how much red they leave. Two exceptions: the steps set inside the non-red range trail off to the right, and the plain projection sits just inside the gate.
""" + landscape_plot()

# %%

t00_landscape_conds = ("recipe-short", "t00")
t00_landscape_props = {"recipe-short": prop, "t00": prop_t00}


@memo
@themed(
    name="landscape-t00",
    alt_text="""
        Two scatter panels sharing both axes, non-red deficit horizontal and red accuracy vertical, both zero to one. Left, the adopted point: every mark sits within a quarter of the way along the deficit axis, most of them inside the gate. Right, t00: the whole-sequence circles are spread far to the right, between a third and the far end of the deficit axis, all at red accuracy near zero, while the operand-only triangles sit against the left edge inside the gate, ringed at the lowest of them.
    """,
    caption=f"""
        **The same landscape on `mix`, at the adopted point and at `t00`.** Seed means over twenty seeds (left) and five (right), the marks as above; the ring is the frozen rule's pick on each point. The axes run the full range on both panels so the two points can be compared; the per-op figure above zooms into the corner. Dashed lines are the gates ({ex.NONRED_DEFICIT_GATE:g}, {ex.RED_ACC_GATE:g}).
    """,
)
def landscape_t00_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    grey = light_dark("#888", "#aaa")
    for ax, c in zip(axes, t00_landscape_conds, strict=True):
        ax.axvline(ex.NONRED_DEFICIT_GATE, ls="--", lw=0.7, color=grey, zorder=0)
        ax.axhline(ex.RED_ACC_GATE, ls="--", lw=0.7, color=grey, zorder=0)
        p = t00_landscape_props[c]
        for r in rows[c]:
            trial_mark(ax, r.deficit[PRIMARY_OP], r.red_acc[PRIMARY_OP], r, p is not None and r.name == p.name)
        ax.set_title(c, fontsize=9)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlabel("non-red deficit ↓")
    axes[0].set_ylabel("red accuracy ↓")
    return fig


r"""
Which operator is best depends on the point. On `t00` the syntax embeddings carry the axis, and there every edit applied at every position costs most of the non-red lines, whatever its threshold, ramp, or landing. The operand-only edits are the only ones inside the gate.
""" + landscape_t00_plot()

# %%

marginal_rows = rows["recipe-short"]
shaped_rows = [r for r in marginal_rows if r.family == "shaped"]
linear_rows = [r for r in marginal_rows if r.family == "repulsion-linear"]
shaped_ps = sorted({r.p for r in shaped_rows})
linear_as = sorted({r.a for r in linear_rows})


@memo
@themed(
    name="marginals",
    alt_text="""
        Four panels in a two-by-two grid. Top row: red accuracy on mix; bottom row: non-red deficit on mix. Left column: shaped suppression against its threshold a, one line per ramp p, solid for every position and dashed for the operand positions; red accuracy rises with both the threshold and the ramp, from near zero to about 0.7, the solid and dashed lines nearly on top of each other; the deficit is flat at zero except for the p = 0 line at every position, which starts at 0.23 at a = 0.1 and falls to zero by a = 0.4. Right column: repulsion against its landing b, one line per threshold a; red accuracy rises with the landing from about 0.1 to 0.6, and the deficit is flat at zero.
    """,
    caption="""
        **The marginals on `mix`.** Seed means over twenty seeds. Left: shaped suppression against its threshold *a*, one shade per ramp *p*. Right: repulsion against its landing *b*, one shade per threshold *a*. Both panels are keyed by their legends. Solid lines are the whole-sequence trials, dashed the operand-only ones. The dashed grey rule is the gate.
    """,
)
def marginals_plot() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex="col", layout="constrained")
    axes = cast(AxesGrid, axes)
    grey = light_dark("#888", "#aaa")
    cmap_s = plt.get_cmap(light_dark("Oranges", "Oranges_r"))
    cmap_l = plt.get_cmap(light_dark("Blues", "Blues_r"))
    stops = light_dark((0.35, 0.95), (0.05, 0.65))

    def shade(cmap, i, n):
        return cmap(stops[0] + (stops[1] - stops[0]) * (i / max(n - 1, 1)))

    for row_i, key in enumerate(("red_acc", "deficit")):
        axl, axr = axes[row_i]
        for i, p in enumerate(shaped_ps):
            family_lines(
                axl, [r for r in shaped_rows if r.p == p], "a", key, shade(cmap_s, i, len(shaped_ps)), f"p = {p:g}"
            )
        for i, a in enumerate(linear_as):
            family_lines(
                axr, [r for r in linear_rows if r.a == a], "b", key, shade(cmap_l, i, len(linear_as)), f"a = {a:g}"
            )
        gate = ex.RED_ACC_GATE if key == "red_acc" else ex.NONRED_DEFICIT_GATE
        for ax in (axl, axr):
            ax.axhline(gate, ls="--", lw=0.7, color=grey, zorder=0)
            ax.set_ylim(-0.02, 1.0)
        axl.set_ylabel("red accuracy ↓" if key == "red_acc" else "non-red deficit ↓")
    axes[1][0].set_xlabel("threshold a (shaped)")
    axes[1][1].set_xlabel("landing b (repulsion)")
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    axes[0][1].legend(fontsize=7, frameon=False, loc="upper left")
    return fig


r"""
## The marginals

The same two reads on `mix`, now plotted against the parameters of each family. For the shaped suppression, the threshold and the ramp both set how much *red* is left. Its cost is zero everywhere except for the step at low thresholds applied at every position. For repulsion what matters is the landing: red accuracy follows *b*, and the threshold adds a little on top.
""" + marginals_plot()

# %%

landing_rows = [r for r in rows["recipe-short"] if not r.operands]
landing_clean = res.per_slice("recipe-short", PRIMARY_OP, None, "alpha_red_operand").mean(0)
landing_by_name = {
    r.name: res.per_slice("recipe-short", PRIMARY_OP, r.name, "alpha_red_operand").mean(0) for r in landing_rows
}
landing_x = np.arange(len(SLICE_NAMES))


@memo
@themed(
    name="landing",
    alt_text="""
        Two panels, one per family, each with the five residual-stream slices on the horizontal axis and the red operand's alignment with the anchor axis on the vertical. A bold grey line near 0.9 is the clean value. Under shaped suppression the lines fan out between zero and 0.75 by threshold and ramp, most rising a little from the embedding to slice 3 and dipping at slice 4 as the clean line does. Under repulsion each line sits flat at its landing through slice 3 and dips at the last slice.
    """,
    caption="""
        **Where the operators leave the red operand, by slice, on the red `mix` lines.** Mean alignment of the dose-carrying operand's state after the edit, over lines and twenty seeds, for every whole-sequence trial; the bold grey line is the clean value. Shaped rows shade by threshold (faintest to boldest: 0.1 to 0.7), repulsion rows by landing *b* (faintest to boldest: 0.2 to 0.7).
    """,
)
def landing_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    stops = light_dark((0.35, 0.95), (0.05, 0.65))
    for ax, fam, cmap_name in zip(axes, FAMILIES[1:], ("Oranges", "Blues"), strict=True):
        cmap = plt.get_cmap(light_dark(cmap_name, cmap_name + "_r"))
        fam_rows = [r for r in landing_rows if r.family == fam]
        keys = sorted({r.a if fam == "shaped" else r.b for r in fam_rows})
        for r in fam_rows:
            k = r.a if fam == "shaped" else r.b
            i = keys.index(k)
            ax.plot(
                landing_x,
                landing_by_name[r.name],
                lw=0.9,
                alpha=0.85,
                color=cmap(stops[0] + (stops[1] - stops[0]) * i / max(len(keys) - 1, 1)),
            )
        ax.plot(landing_x, landing_clean, lw=2, color=light_dark("#666", "#bbb"), zorder=0)
        ax.set_xticks(landing_x, SLICE_NAMES)
        ax.set_title(FAMILY_TITLE[fam], fontsize=9)
        ax.set_ylim(-0.05, 1.0)
    axes[0].set_ylabel("α of the red operand")
    for ax in axes:
        ax.set_xlabel("slice")
    return fig


r"""
## Where the operators leave the state

In both families, how much *red* an operator leaves tracks where it leaves the state of the red operand. The shaped suppression lands each state wherever its ramp puts it, so the landing varies with where the state arrived. Repulsion lands every state it touches at one alignment, and the stream holds it there through slice 3. The last slice pulls every landing down, including the clean state.
""" + landing_plot()

# %%

write_rows = rows["recipe-short"]
write_by_name = {
    r.name: float(res.per_slice("recipe-short", PRIMARY_OP, r.name, "q99_write_nonred").mean(0).max())
    for r in write_rows
}


@memo
@themed(
    name="write-cost",
    alt_text="""
        One scatter panel: the largest 99th-percentile write on non-red mix lines across the slices on the horizontal axis, in radians, against the non-red deficit on the vertical, zoomed to the first third of its range. Most marks sit at zero deficit with writes under 0.15 radians. The plain projection's cross is at the far right, near 0.27 radians, at a deficit just inside the gate, and an open circle sits directly above it at the same write and several times the deficit; the three open circles for the low-threshold steps are the only marks above the gate, at deficits of 0.07 to 0.23.
    """,
    caption="""
        **What the non-red lines pay for the write they receive, on `mix`.** The 99th-percentile write of each trial on the non-red lines (the largest over the five slices, seed mean) against its non-red deficit. Same marks as the landscape.
    """,
)
def write_cost_plot() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
    grey = light_dark("#888", "#aaa")
    ax.axhline(ex.NONRED_DEFICIT_GATE, ls="--", lw=0.7, color=grey, zorder=0)
    for r in write_rows:
        x, y = write_by_name[r.name], r.deficit[PRIMARY_OP]
        trial_mark(ax, x, y, r, prop is not None and r.name == prop.name)
    ax.set_xlabel("q99 write on non-red lines (rad)")
    ax.set_ylabel("non-red deficit ↓")
    ax.set_ylim(-0.01, 0.3)
    return fig


r"""
## The write and its cost

The write is the angle an operator turns a state through. M1 argued for the shaped operators on the grounds that keeping the write small keeps the side-effect small. That only holds if the two move together, and on the non-red lines of this point they do not. The plain projection and the step at a = 0.1 turn the non-red lines through the same angle, yet the step costs them several times as much. A step at a = 0.2 turns them less and still costs four times what the projection does. What the non-red lines pay depends more on which of their states are turned than on how far.
""" + write_cost_plot()

# %%

every_trial_rows = rows["recipe-short"]
front_names = {r.name for r in summary.front}
every_table_rows = []
every_ref_idx = set()
for i, r in enumerate(sorted(every_trial_rows, key=lambda r: (r.operands, FAMILIES.index(r.family), r.a, r.b, r.p))):
    if r.family == "projection":
        every_ref_idx.add(i)
    params = (
        "—"
        if r.family == "projection"
        else (f"a {r.a:g}, p {r.p:g}" if r.family == "shaped" else f"a {r.a:g}, b {r.b:g}")
    )

    def bold(v: float, ok: bool) -> str:
        return f"<b>{v:.3f}</b>" if ok else f"{v:.3f}"

    marks = (
        ("●" if not r.operands else "▲")
        + (" ⦿" if prop is not None and r.name == prop.name else "")
        + (" ⋆" if r.name in front_names else "")
    )
    every_table_rows.append(
        [
            f"`{r.name}` {marks}",
            FAMILY_TITLE[r.family],
            params,
            "operands" if r.operands else "all",
            bold(r.red_acc[PRIMARY_OP], r.red_acc[PRIMARY_OP] <= ex.RED_ACC_GATE),
            bold(r.deficit[PRIMARY_OP], r.deficit[PRIMARY_OP] <= ex.NONRED_DEFICIT_GATE),
            bold(r.worst_red_acc, r.removes),
            bold(r.worst_deficit, r.feasible),
            f"{r.margin:+.3f}",
            "✓" if r.feasible else "",
        ]
    )

r"""
## Every trial

No omissions. Seed means over the twenty seeds of `recipe-short`. ⦿ is the proposed trial and ⋆ marks the `mix` front; bold values pass their gate. *Margin* is the gate minus the worst non-red deficit over the six ops, which is what the survey ranks on alongside the objective.
""" + table_html(
    [
        "trial",
        "family",
        "parameters",
        "positions",
        "red acc `mix` ↓",
        "deficit `mix` ↓",
        "worst red acc ↓",
        "worst deficit ↓",
        "margin ↑",
        "feasible",
    ],
    every_table_rows,
    "**Every trial on `recipe-short`.** Reference rows are shaded.",
    ref_rows=frozenset(every_ref_idx),
)

# %%

proposal_rows = rows["recipe-short"]
proposal_by_name = {r.name: r for r in proposal_rows}
assert summary.best_free is not None, "the proposal prose assumes a free trial exists"
proposal_ops, proposal_best_free = proposal_by_name["operands"], summary.best_free
proposal_front = summary.front
proposal_front_txt = ", ".join(
    f"`{r.name}` ({r.red_acc[PRIMARY_OP]:.3f}, {r.deficit[PRIMARY_OP]:.3f})" for r in proposal_front
)
if prop is None:
    proposal_body = (
        "No trial is feasible on every op, so the rule proposes nothing. The infeasibility map above is the finding."
    )
else:
    proposal_near = [
        r
        for r in proposal_rows
        if r.feasible
        and r.name != prop.name
        and abs(r.red_acc[PRIMARY_OP] - prop.red_acc[PRIMARY_OP]) <= band["red_acc"]
    ]
    proposal_t00_rows = {r.name: r for r in rows["t00"]}
    proposal_t00_prop = proposal_t00_rows[prop.name]
    proposal_body = rf"""
The frozen rule proposes **`{prop.name}`**, {describe(prop)}. Its margin to the gate is {prop.margin:+.3f}, less than the deficit band of {band["deficit"]:.3f}. The margin of the operand-only projection is {proposal_ops.margin:+.3f}. {len(proposal_near)} other feasible {"trial sits" if len(proposal_near) == 1 else "trials sit"} within one red-accuracy band of the proposal{": " + ", ".join(f"`{r.name}`" for r in proposal_near) if proposal_near else ""}. The `mix` front, from the gentlest edit to the most complete, as (red accuracy, deficit): {proposal_front_txt}.

On `t00` the same trial has a worst deficit of {proposal_t00_prop.worst_deficit:.3f}, outside the gate, and the rule picks {f"`{prop_t00.name}`, {describe(prop_t00)}" if prop_t00 is not None else "nothing"} there.
"""

rf"""
## The proposal

{proposal_body}

**What the prereg should carry.** The rule picks the plain projection, which is the row the anchored-op experiments already score, so on the adopted point this pass changes nothing about the primary intervention. What it adds is the margin: the projection is inside the gate by less than a band. So the prereg should keep the operand-only projection beside it as the selective reference.

Two things the rule did not rank are for the prereg to settle before its seeds are drawn. First, the answer depends on the point: on `t00` the same rule picks an operand-only step. A prereg that may run on a syntax-heavy point should name the operand-only operator as its intervention there, rather than choosing it after the read.

Second, there is a candidate that needs no syntax: applied at every position, `{proposal_best_free.name}` leaves red accuracy {proposal_best_free.red_acc[PRIMARY_OP]:.3f} at no cost the survey can resolve. That is close to the {proposal_ops.red_acc[PRIMARY_OP]:.3f} of the operand-only projection, and it uses no position mask, which is what a prompt with unknown operand positions will need. Choosing it here would be post hoc, so it is recorded for the prereg to carry as a third row, re-scored at fresh seeds, if the syntax-free question is worth one.
"""

rf"""
## Post hoc

Nothing was added after the run. The reference rows reproduced {"to the bit" if check_worst == 0 else f"to {check_worst:.1e}"}, and every contract check passed on every trial and seed; a failed check fails the scoring task, and none did. The plan allowed for the rule picking a reference row, since the references are trials and the objective ranks them with the rest.

One observation was not anticipated by the plan, and is recorded as exploratory. The plan expected a threshold to trade removal against cost monotonically, with the plain projection at the costly end. Instead, a step set inside the alignment range of the non-red lines costs more than the projection does, on the adopted point and on `t00` alike. A line whose states are zeroed on some tokens and kept on others is decoded worse than one zeroed throughout. That suggests a prediction the anchored-op prereg can carry: above the non-red range the whole-sequence cost is zero, and below it the cost rises as the threshold falls.
"""
