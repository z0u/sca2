import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Ex 2.2.8: a survey of the intervention operator",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from dataclasses import dataclass
    from pathlib import Path
    from typing import cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import AxesGrid, AxesRow, figure_html, light_dark, themed

    use_publisher(report_bundle(__file__))

    SLICE_NAMES = ["emb", "1", "2", "3", "4"]
    OPS = list(ex.OP_NAMES)
    PRIMARY_OP = "mix"
    FAMILIES = ("projection", "shaped", "repulsion-linear", "repulsion-bezier")
    FAMILY_TITLE = {
        "projection": "projection",
        "shaped": "shaped suppression",
        "repulsion-linear": "repulsion, linear",
        "repulsion-bezier": "repulsion, Bézier",
    }
    INK = {
        "projection": ("#333", "#ddd"),
        "shaped": ("#d0461b", "#f07a50"),
        "repulsion-linear": ("#1f6fb4", "#5fa8dd"),
        "repulsion-bezier": ("#2a8f5a", "#5fc98b"),
    }
    """One ink per operator family, as (light, dark) pairs."""

    None


@app.function(hide_code=True)
def load_json(ref: str) -> dict | None:
    """A published JSON result as a dict, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "data.json")])
        return json.loads(path.read_text())


@app.function(hide_code=True)
def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:{fmt}}"
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


@app.function(hide_code=True)
def ink(family: str):
    return light_dark(*INK[family])


@app.function(hide_code=True)
def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<div class="report-table-scroll"><table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>'
    return figure_html(table, caption=mo.md(caption).text, class_="report-figure")


@app.class_definition(hide_code=True)
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


@app.class_definition(hide_code=True)
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


@app.function(hide_code=True)
def rows_for(res: Results, cond: str) -> list[Row]:
    rows = []
    for t in res.trials:
        red = {op: res.stat(cond, op, t["name"], "acc", "red") for op in OPS}
        dfc = {op: res.stat(cond, op, t["name"], "deficit", "nonred") for op in OPS}
        rows.append(
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
    return rows


@app.function(hide_code=True)
def proposal(rows: list[Row]) -> Row | None:
    """The frozen rule: among feasible trials, the lowest red accuracy on `mix`."""
    feasible = [r for r in rows if r.feasible]
    return min(feasible, key=lambda r: r.red_acc[PRIMARY_OP]) if feasible else None


@app.function(hide_code=True)
def describe(r: Row) -> str:
    """A trial in words: its family and parameters, and where it edits."""
    where = "at the operand positions" if r.operands else "at every position"
    if r.family == "projection":
        return f"the plain projection {where}"
    params = f"threshold {r.a:g}, " + (f"landing {r.b:g}" if r.family.startswith("repulsion") else f"ramp p = {r.p:g}")
    return f"{FAMILY_TITLE[r.family]} at {params}, {where}"


@app.function(hide_code=True)
def front(rows: list[Row], op: str = PRIMARY_OP) -> list[Row]:
    """The trials no other trial beats on both reads at once (lower red accuracy and lower deficit on *op*)."""
    keep = []
    for r in rows:
        dominated = any(
            (o.red_acc[op] <= r.red_acc[op] and o.deficit[op] <= r.deficit[op])
            and (o.red_acc[op] < r.red_acc[op] or o.deficit[op] < r.deficit[op])
            for o in rows
        )
        if not dominated:
            keep.append(r)
    return sorted(keep, key=lambda r: r.deficit[op])


@app.function(hide_code=True)
def family_lines(ax, rows: list[Row], x: str, key: str, color, label: str) -> None:
    """One family's trials on `mix` against one parameter: solid at every position, dashed at the operands."""
    for operands, ls in ((False, "-"), (True, "--")):
        pts = sorted((getattr(r, x), getattr(r, key)[PRIMARY_OP]) for r in rows if r.operands == operands)
        if pts:
            ax.plot(
                *zip(*pts, strict=True), ls=ls, marker="o", ms=3, lw=1, color=color, label=None if operands else label
            )


@app.function(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    _metrics = load_json(ex.METRICS_REF)
    mo.stop(_metrics is None, mo.md("_Results are not published yet; the result cells render once they are._"))
    assert _metrics is not None
    _prod = load_json(ex.EX223_METRICS_REF)
    assert _prod is not None, "ex-2.2.3's metrics are missing from the store"
    res: Results = Results(_metrics, _prod)
    rows: dict[str, list[Row]] = {c: rows_for(res, c) for c in ex.CONDITIONS}
    return res, rows


@app.cell(hide_code=True)
def _(res: Results, rows: dict[str, list[Row]]):
    # The noise floors: per-run σ of each objective under the reference projection, at the adopted point's seeds.
    _c = "recipe-short"
    _n = len(res.runs(_c))
    _sd_red = res.stat(_c, PRIMARY_OP, "projection", "acc", "red").std(ddof=1)
    _sd_def = res.stat(_c, PRIMARY_OP, "projection", "deficit", "nonred").std(ddof=1)
    band = {"red_acc": 2 * _sd_red * np.sqrt(2 / _n), "deficit": 2 * _sd_def * np.sqrt(2 / _n)}
    """How small a difference between two seed means at twenty seeds the survey can resolve, per objective."""
    _rs = rows[_c]
    _by = {r.name: r for r in _rs}
    prop = proposal(_rs)
    prop_t00 = proposal(rows["t00"])
    _ref, _ops = _by["projection"], _by["operands"]
    _whole = [r for r in _rs if not r.operands and r.family != "projection"]
    _op_rows = [r for r in _rs if r.operands and r.family != "projection"]
    # Whole-sequence trials that cost the non-red `mix` lines more than projecting everything does.
    _costly = sorted(
        (r for r in _whole if r.deficit[PRIMARY_OP] > _ref.deficit[PRIMARY_OP]), key=lambda r: -r.deficit[PRIMARY_OP]
    )
    _whole_feas = [r for r in _whole if r.feasible]
    _best_whole = min(_whole_feas, key=lambda r: r.red_acc[PRIMARY_OP]) if _whole_feas else None
    # ...and of those, the ones whose cost is within a band of zero, and the one that removes the most.
    _whole_free = [r for r in _whole if r.worst_deficit <= band["deficit"]]
    _best_free = min(_whole_free, key=lambda r: r.red_acc[PRIMARY_OP]) if _whole_free else None
    _least_whole = max(_whole, key=lambda r: r.red_acc[PRIMARY_OP])
    # Whole-sequence trials whose operand-only copy differs by more than a band on either `mix` read.
    _split = [
        r
        for r in _whole
        if abs(r.red_acc[PRIMARY_OP] - _by[r.name + "-operands"].red_acc[PRIMARY_OP]) > band["red_acc"]
        or abs(r.deficit[PRIMARY_OP] - _by[r.name + "-operands"].deficit[PRIMARY_OP]) > band["deficit"]
    ]
    # The non-red `mix` lines' clean 99th-percentile alignment at its highest slice and position, per condition:
    # where a threshold starts to catch them.
    _q99 = {c: float(res.per_slice(c, PRIMARY_OP, None, "alpha_q99_nonred").mean(0).max()) for c in rows}
    # The `t00` side: the whole-sequence cost's range, and the steps' cost against projecting everything.
    _t = rows["t00"]
    _t_by = {r.name: r for r in _t}
    _t_whole = [r for r in _t if not r.operands and r.family != "projection"]
    _t_step = [r for r in _t_whole if r.family == "shaped" and r.p == 0]
    summary = dict(
        n_feasible=sum(r.feasible for r in _rs),
        n_whole=len(_whole),
        n_whole_feasible=len(_whole_feas),
        n_op=len(_op_rows),
        n_op_feasible=sum(r.feasible for r in _op_rows),
        front=front(_rs),
        ref=_ref,
        ops=_ops,
        shaped_ref=_by["shaped-a0.5-p1"],
        costly=_costly,
        best_whole=_best_whole,
        n_whole_free=len(_whole_free),
        best_free=_best_free,
        least_whole=_least_whole,
        split=_split,
        q99=_q99,
        t00_ref=_t_by["projection"],
        t00_ops=_t_by["operands"],
        t00_whole=(min(r.deficit[PRIMARY_OP] for r in _t_whole), max(r.deficit[PRIMARY_OP] for r in _t_whole)),
        t00_step=(min(r.deficit[PRIMARY_OP] for r in _t_step), max(r.deficit[PRIMARY_OP] for r in _t_step)),
        t00_whole_feasible=sum(r.feasible for r in _t_whole),
        t00_op_feasible=sum(r.feasible for r in _t if r.operands and r.family != "projection"),
    )
    return band, prop, prop_t00, summary


@app.cell(hide_code=True)
def _(band, prop, prop_t00, rows: dict[str, list[Row]], summary):
    _s = summary
    n_trials = len(rows["recipe-short"])
    _ref, _ops, _sh = _s["ref"], _s["ops"], _s["shaped_ref"]
    _bw, _bf, _lw = _s["best_whole"], _s["best_free"], _s["least_whole"]
    _costly, _split = _s["costly"], _s["split"]
    _t_ref, _t_ops = _s["t00_ref"], _s["t00_ops"]
    _p = prop
    _prop_line = (
        f"**Proposed operator: `{_p.name}`,** {describe(_p)}. On `mix` it leaves red accuracy {_p.red_acc[PRIMARY_OP]:.3f} "
        f"at a non-red deficit of {_p.deficit[PRIMARY_OP]:.3f}, where the operand-only projection from ex-2.2.3 is at "
        f"{_ops.red_acc[PRIMARY_OP]:.3f} and {_ops.deficit[PRIMARY_OP]:.3f}. Its worst deficit over the six ops is "
        f"{_p.worst_deficit:.3f}, on `{max(_p.deficit, key=_p.deficit.get)}`, a margin of {_p.margin:+.3f}."
        if _p is not None
        else "**No trial is feasible on every op.** The infeasibility map is the finding; nothing is proposed."
    )
    _t00_line = (
        f"On `t00` the same rule picks `{prop_t00.name}` (red accuracy {prop_t00.red_acc[PRIMARY_OP]:.3f}, deficit "
        f"{prop_t00.deficit[PRIMARY_OP]:.3f} on `mix`)."
        if prop_t00 is not None
        else "On `t00` no trial is feasible on every op."
    )
    _costly_txt = ", ".join(f"`{r.name}` {r.deficit[PRIMARY_OP]:.3f}" for r in _costly)
    _costly_red = ", ".join(f"{r.red_acc[PRIMARY_OP]:.3f}" for r in _costly)
    _split_txt = ", ".join(f"`{r.name}`" for r in _split)
    mo.md(rf"""
    # Ex 2.2.8: a survey of the intervention operator on the stored ex-2.2.3 checkpoints

    /// tip |
    <!-- tl;dr -->
    This is a survey: no training, and no hypothesis gates. We scored a hundred operators through the eval contract on stored checkpoints, namely ex-2.2.3's adopted point (twenty seeds) and `t00`, the first proposal its selection rule made (five seeds). The job was to tune the threshold and ramp of the shaped suppression, and to try the repulsion form from M1, looking for an operator that removes *red* as fully as the plain projection while staying as selective as the operand-only one.

    On the adopted point the plain projection is inside the gate on every op, and the frozen rule proposes it (red accuracy {_ref.red_acc[PRIMARY_OP]:.3f} on `mix`, non-red deficit {_ref.deficit[PRIMARY_OP]:.3f}, against a gate of {ex.NONRED_DEFICIT_GATE:g}). Adding a threshold on the anchor alignment moves that cost in either direction. Set it inside the range the non-red lines occupy (a ≤ 0.3; their clean alignment reaches {_s["q99"]["recipe-short"]:.2f}), and a step costs more than projecting everything. Set it above that range, and every operator applied at every position costs nothing the survey can resolve, but also removes less red; `{_bf.name}` removes the most of them ({_bf.red_acc[PRIMARY_OP]:.3f}).

    On `t00` the syntax rows carry the axis, and there no whole-sequence edit is inside the gate while every operand-only one is. The rule picks `{prop_t00.name if prop_t00 else "nothing"}` on that point.
    ///

    ## Observations

    - **The reference rows reproduce.** Re-scored here, the `projection` and `operands` rows from ex-2.2.3 match their stored values on every seed, op, and group to four decimals ([table](#the-reference-rows-reproduce)). The scorer is the same code.
    - **Noise floors.** Under `projection` at the twenty seeds of `recipe-short`, the per-run σ gives a band of {band["red_acc"]:.3f} on red accuracy and {band["deficit"]:.3f} on the non-red deficit: the smallest difference between two seed means the survey can tell apart.
    - **On the adopted point, the plain projection is inside the gate.** Its worst op is `mix`, where the non-red deficit is {_ref.deficit[PRIMARY_OP]:.3f} at {_ref.red_acc[PRIMARY_OP]:.3f} red accuracy. The operand-only projection is at {_ops.deficit[PRIMARY_OP]:.3f} and {_ops.red_acc[PRIMARY_OP]:.3f}. We expected the whole-sequence cost that ex-2.2.3 saw on the first proposals of its rule; at twenty seeds, the point adopted after the rule was amended does not show it. Both projections are feasible and the plain one removes more, so the rule proposes it, with a margin of {_ref.margin:+.3f}, which is less than a band.
    - **A threshold inside the non-red range costs more than projecting everything.** On the non-red `mix` lines the clean alignment reaches a 99th percentile of {_s["q99"]["recipe-short"]:.2f} over slices and positions. Applied at every position, the steps at a ≤ 0.3 cost {_costly_txt}, above the {_ref.deficit[PRIMARY_OP]:.3f} of the plain projection, while removing about as much red ({_costly_red}). So a step that zeroes the axis on some of the states in a line and leaves their neighbors alone costs that line more than zeroing them all. {len(_costly)} trials in all cost more than the projection, and all of them are whole-sequence steps below a = 0.4 ([figure](#the-landscape)).
    - **Above the non-red range, the whole-sequence cost vanishes and removal tracks the landing.** {_s["n_whole_feasible"]} of the {_s["n_whole"]} tuned trials applied at every position are inside the gate on every op, and {_s["n_whole_free"]} of those are within a band of zero cost. `{_bf.name}` removes the most of them, leaving {_bf.red_acc[PRIMARY_OP]:.3f}; `{_bw.name}`, a gentle ramp at a low threshold, is at {_bw.red_acc[PRIMARY_OP]:.3f} and {_bw.deficit[PRIMARY_OP]:.3f}. Red accuracy climbs with the threshold, the ramp, and the landing, up to {_lw.red_acc[PRIMARY_OP]:.3f} at `{_lw.name}` ([figure](#the-marginals)). The shaped suppression that ex-2.2.1 scored (`{_sh.name}`) sits at {_sh.red_acc[PRIMARY_OP]:.3f} and {_sh.deficit[PRIMARY_OP]:.3f}. A state left part-way along the axis keeps part of the color it carried: at the last slice the alignment of the red operand settles at the threshold or the landing, as designed ([figure](#where-the-operators-leave-the-state)).
    - **The position mask only matters where the threshold is low.** For {_s["n_whole"] - len(_split)} of the {_s["n_whole"]} tuned trials, the whole-sequence row and its operand-only copy agree on both `mix` reads to within a band. The {len(_split)} that differ are {_split_txt}. Where a threshold already leaves the non-red states alone, restricting the edit to the operand positions changes nothing.
    - **On `t00` only operand-only edits are feasible.** There the syntax rows carry the axis: on the non-red lines the clean alignment reaches {_s["q99"]["t00"]:.2f}. Projecting everything costs {_t_ref.deficit[PRIMARY_OP]:.3f} on `mix`, and every tuned trial applied at every position costs between {_s["t00_whole"][0]:.3f} and {_s["t00_whole"][1]:.3f}, with the steps between {_s["t00_step"][0]:.3f} and {_s["t00_step"][1]:.3f}. Partial removal costs more than full removal there too. All {_s["t00_op_feasible"]} tuned operand-only trials are inside the gate, and so is `operands`, at {_t_ops.red_acc[PRIMARY_OP]:.3f} and {_t_ops.deficit[PRIMARY_OP]:.3f}. With only five seeds, the operand-only rows near it cannot be told apart.
    - **The front is short.** {_s["n_feasible"]} of {n_trials} trials are feasible on every op: both projections, {_s["n_whole_feasible"]} of the {_s["n_whole"]} tuned trials at every position, and {_s["n_op_feasible"]} of the {_s["n_op"]} at the operands. The `mix` front has {len(_s["front"])} trials, running from the gentlest edit to the most complete ([table](#every-trial)).
    - {_prop_line} {_t00_line}

    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this report
    This is a survey, so it scores no hypothesis. What it does preregister is a search plan: the trial list, the objective, its constraint, and the noise trials, all frozen in `experiment.py` before the run.

    Every trial is published, and nothing here may be quoted as a result. The anchored-op prereg adopts the proposed operator and re-measures it at fresh seeds, reporting the survey value beside the confirmed one. The checkpoints are stored, so these are the same seeds ex-2.2.3 scored; the model is what will be fresh in the prereg. The choice of operator is what is not fresh here, since this survey makes it after seeing these seeds.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Why, and what we ran

    Ex-2.2.1 left three operators, none of them both complete and selective. The plain projection removes *red* fully, but it costs the non-red lines. Ex-2.2.3 found that cost on the syntax rows of the first proposals its selection rule made, with `mix` deficits well above the {ex.NONRED_DEFICIT_GATE:g} gate, and adopted `recipe-short` after amending the rule with the selectivity gate; on that point the cost of the projection sits inside the gate with little to spare.

    The other two each give something up. The operand-only projection avoids the cost, but it needs to know the syntax of the line. The shaped suppression at the threshold used in M1 (a = 0.5, b = 1, p = 1) removes about half of red at no non-red cost.

    So the [design](../d2.2/design.md) called for this scoring-only pass on stored runs before the anchored-op prereg, and two backlog items name it as their closing move ([shaped suppression](https://github.com/z0u/sca2/blob/main/todo/science/shaped-suppression-rather-than-projecting-whole-axis.md), [repulsion](https://github.com/z0u/sca2/blob/main/todo/science/repulsion-sets-the-landing-alignment.md)). Does any operator remove as much as the plain projection while keeping the margin of the operand-only edit, and does one do that with no position mask?

    The design named the nine checkpoints from ex-2.2.1. D2.2 has since moved to the six-op grammar and adopted `recipe-short`, so this pass scores the stored runs of ex-2.2.3 instead: the adopted point at all twenty seeds, and `t00` at five. `t00` was the first proposal of the rule, and its syntax rows carry the axis at more than twice the level the recipe does.

    Each checkpoint is scored on the six-op probe set from ex-2.2.3 through `sca.intervention.apply`, with the lines of all six ops concatenated so that each operator is one forward pass. The results are then read per op with the readout from ex-2.2.3, so every statistic means what it meant there.

    The `repulsion` operator is new to the contract, added for this pass. The landing map $m(\alpha)$ from M1 sets where a state above the threshold lands on the axis. The write is the angle between the arriving and the landing alignment, and the scorer checks it against the measured rotation on every state.

    The linear mapper puts every state at or above the threshold *a* at the landing *b*. That is a step, except when a = b, where it becomes the ceiling $\min(\alpha, b)$. The Bézier mapper is continuous at the threshold: it leaves $(a, a)$ with unit slope and arrives flat at $(1, b)$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    /// details | Glossary
    - **trial** — one operator: a family, its parameters, and the positions it edits. Every trial is scored on every stored seed, so each read is a seed mean over twenty runs on `recipe-short` and five on `t00`.
    - **residual stream** — the vector the transformer carries from layer to layer, which each layer reads from and writes back to. Here it is unit-norm.
    - **α** — how well a state lines up with the anchor axis $e_1$. Since the stream is unit-norm, that is just the first coordinate of the state. **landing** — where an operator leaves a state that it edits, as an alignment.
    - **shaped suppression** — one of the operators from M1: above a threshold *a*, remove a fraction $h(\alpha) = b \cdot ((\alpha - a)/(1 - a))^p$ of the axis component, then re-normalize. `p = 0` is a step, meaning full removal above the threshold.
    - **repulsion** — the other operator from M1: above the threshold, land the state at $m(\alpha)$ on the axis, keeping its off-axis direction. Linear: land at *b*. Bézier: a smooth map from $(a, a)$ to $(1, b)$.
    - **red lines** — probe lines whose dose (the redness of the redder operand) is at least {ex.RED_DOSE:g}; **non-red lines** — dose at most {ex.NONRED_DOSE:g}. **red accuracy** is exact match on the red lines, so a lower value means more of *red* was removed. The **non-red deficit** is the drop in P(answer) on the non-red lines, so a lower value means the edit was more selective. Both are as in H4 of ex-2.2.3.
    - **feasible** — seed-mean non-red deficit within {ex.NONRED_DEFICIT_GATE:g} on every op. **band** — 2σ·√(2/n), built from the per-run σ of a statistic under `projection` at twenty seeds. It is the smallest difference between two seed means the survey can resolve.
    - **front** — the trials on `mix` that no other trial beats on both reads at once.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    _t = ex.TRIALS
    mo.md(rf"""
    ## Search plan

    Frozen in `experiment.py` before the run. The space is a grid rather than a sample. There are only a few families and their parameters mean something, so every corner is worth a look; and at a hundred operators on stored checkpoints the whole grid costs less than one training run.

    **The space.** {len(ex.REFERENCE)} reference rows, which are the `projection` from ex-2.2.3 at every position and at the operand positions. Then three families, each at every position and again at the operand positions only:

    | family | grid | trials |
    | --- | --- | ---: |
    | shaped suppression | a ∈ {{{", ".join(f"{a:g}" for a in ex.SHAPED_A)}}} × p ∈ {{{", ".join(f"{p:g}" for p in ex.SHAPED_P)}}}, b = 1 | {len(ex.SHAPED)} |
    | repulsion, linear | (a, b) ∈ {{{", ".join(f"({a:g}, {b:g})" for a, b in ex.LINEAR_AB)}}} | {len(ex.LINEAR)} |
    | repulsion, Bézier | (a, b) ∈ {{{", ".join(f"({a:g}, {b:g})" for a, b in ex.BEZIER_AB)}}} | {len(ex.BEZIER)} |

    `shaped` at p = 0 is a thresholded projection, and `repulsion-linear` at b = 0 does the same thing, so that edge is not repeated. The Bézier map is monotone only when b ≥ a + (1 − a)/3; below that it rises before it settles. The grid keeps two such points (a = 0.4 with b = 0.4 and 0.6) to see whether that matters.

    The Bézier grid has {len(ex.BEZIER)} points and the linear one {len(ex.LINEAR)}, so the two mappers are not compared point for point. The linear family is where the landing can be read cleanly; the Bézier rows are there to say whether continuity at the threshold buys anything. Counting the operand-only copies, that is {len(_t)} trials in all.

    **The objective.** {ex.OBJECTIVE}

    **Noise floors.** The per-run σ of each objective is read under `{ex.NOISE_TRIALS[0]}` at the twenty seeds of `recipe-short`. A difference between two seed means smaller than 2σ·√(2/20) is unresolved. Every trial is read on the same twenty seeds, so this band is conservative for a paired comparison.

    **What is checked, per trial and seed.** The assertions in the contract run on every state: the clean pass matches, the edit stays within the named positions, and where the write has a closed form (projection and both repulsions) the measured rotation matches it to 2 × 10⁻³ rad. A trial whose write did not match would have failed the scoring task rather than being scored.

    **Not in the plan.** Positions other than the operands and all; edits at a subset of the slices; operators fitted to the data (LEACE, diff-in-means), which the D2.2 design keeps for the anchor-versus-fitted comparison; and the redder-than-both lines, which the readout still reports but this pass does not rank on.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    mo.md(r"""
    ## The reference rows reproduce

    The two projection rows from ex-2.2.3, re-scored by this pass on the same checkpoints and probe lines, beside their stored values. The table gives the largest absolute difference over every seed, op, and group.
    """)
    _pairs = {"projection": ex.ex223.PROJECTION.name, "operands": "operands"}
    _rows = []
    _worst = 0.0
    for _c in ex.CONDITIONS:
        for _here, _there in _pairs.items():
            _d = {}
            for _key, _grp in (("acc", "red"), ("acc", "nonred"), ("deficit", "nonred"), ("p_ans", "all")):
                _diff = max(
                    float(np.abs(res.stat(_c, op, _here, _key, _grp) - res.prod_stat(_c, op, _there, _key, _grp)).max())
                    for op in OPS
                )
                _d[f"{_key}/{_grp}"] = _diff
                _worst = max(_worst, _diff)
            _rows.append([f"`{_c}`", f"`{_here}`", *(f"{v:.1e}" if v else "0" for v in _d.values())])
    check_worst = _worst
    mo.md(
        table_html(
            ["condition", "row", "red acc", "non-red acc", "non-red deficit", "P(ans), all"],
            _rows,
            "**Largest |re-scored − stored| per statistic.** Over the seeds of the condition, the six ops, and the group named. The floating-point path differs only in batch composition (the six ops are scored in one pass here).",
        )
    )
    return (check_worst,)


@app.cell(hide_code=True)
def _(band, res: Results):
    _c = "recipe-short"
    _rows = []
    for _op in OPS:
        _red = res.stat(_c, _op, "projection", "acc", "red")
        _def = res.stat(_c, _op, "projection", "deficit", "nonred")
        _rows.append([f"`{_op}`", span2(_red), f"{_red.std(ddof=1):.3f}", span2(_def), f"{_def.std(ddof=1):.3f}"])
    mo.md(
        rf"""
    ## Noise floors

    Measured under `projection` at the twenty seeds of `recipe-short`. The rule ranks on `mix`, and the band there is {band["red_acc"]:.3f} on red accuracy and {band["deficit"]:.3f} on the non-red deficit.
    """
        + table_html(
            ["op", "red accuracy", "σ", "non-red deficit", "σ"],
            _rows,
            "**Per-run spread of the two objectives, by op.** Seed mean ± half the seed range, and the per-run standard deviation the bands are built from.",
        )
    )
    return


@app.cell(hide_code=True)
def _(prop, rows: dict[str, list[Row]]):
    _rs = rows["recipe-short"]
    _xlim, _ylim = (-0.01, 0.3), (-0.02, 1.0)

    @themed(
        name="landscape",
        alt_text="""
            Six scatter panels, one per op, each with the non-red deficit on the horizontal axis, zoomed to the first third of its range, and red-line accuracy on the vertical, dashed gate lines near the origin. On every op nearly all marks stand in a vertical column at zero deficit, spanning red accuracy from near zero to about 0.8, circles and triangles together; the two projection crosses sit at the foot of the column just inside the deficit gate, the ring around the plain one on the mix panel; and three open circles trail to the right of the gate at low red accuracy, the low-threshold steps.
        """,
        caption=f"""
            **The landscape: removal against selectivity, per op.** One mark per trial, seed means over the twenty seeds of `recipe-short`: $●$ at every position, $▲$ at the operand positions, filled when the trial is inside the deficit gate on every op and open otherwise, in the family's ink. $×$ marks the two reference projections. Dashed lines are ex-2.2.3's gates ({ex.NONRED_DEFICIT_GATE:g} on the deficit, {ex.RED_ACC_GATE:g} on red accuracy); the corner they enclose is where an operator is both selective and complete. The ring is the proposed trial. The deficit axis is zoomed to the range the adopted point uses; the figure below shows the full range beside `t00`.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesGrid, axes)
        grey = light_dark("#888", "#aaa")
        for ax, op in zip((a for row in axes for a in row), OPS, strict=True):
            ax.axvline(ex.NONRED_DEFICIT_GATE, ls="--", lw=0.7, color=grey, zorder=0)
            ax.axhline(ex.RED_ACC_GATE, ls="--", lw=0.7, color=grey, zorder=0)
            for r in _rs:
                x, y = r.deficit[op], r.red_acc[op]
                trial_mark(ax, x, y, r, prop is not None and r.name == prop.name)
            ax.set_title(op, fontsize=9)
            ax.set_xlim(*_xlim)
            ax.set_ylim(*_ylim)
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

    mo.md(
        r"""
    ## The landscape

    Every trial on every op, on the two reads the objective uses. On the adopted point nearly every trial stands at zero deficit, with the position mask making no difference, and what separates them is how much red they leave. Two exceptions: the steps set inside the non-red range, which trail off to the right, and the plain projection, just inside the gate.
    """
        + _plot()
    )
    return


@app.cell(hide_code=True)
def _(prop, prop_t00, rows: dict[str, list[Row]]):
    _conds = ("recipe-short", "t00")
    _props = {"recipe-short": prop, "t00": prop_t00}

    @themed(
        name="landscape-t00",
        alt_text="""
            Two scatter panels sharing both axes, non-red deficit horizontal and red accuracy vertical, both zero to one. Left, the adopted point: every mark sits within a quarter of the way along the deficit axis, most of them inside the gate. Right, t00: the whole-sequence circles are spread far to the right, between a third and the far end of the deficit axis, all at red accuracy near zero, while the operand-only triangles sit against the left edge inside the gate, ringed at the lowest of them.
        """,
        caption=f"""
            **The same landscape on `mix`, at the adopted point and at `t00`.** Seed means over twenty seeds (left) and five (right), the marks as above; the ring is the frozen rule's pick on each point. The axes run the full range on both panels so the two points can be compared; the per-op figure above zooms into the corner. Dashed lines are the gates ({ex.NONRED_DEFICIT_GATE:g}, {ex.RED_ACC_GATE:g}).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        grey = light_dark("#888", "#aaa")
        for ax, c in zip(axes, _conds, strict=True):
            ax.axvline(ex.NONRED_DEFICIT_GATE, ls="--", lw=0.7, color=grey, zorder=0)
            ax.axhline(ex.RED_ACC_GATE, ls="--", lw=0.7, color=grey, zorder=0)
            p = _props[c]
            for r in rows[c]:
                trial_mark(ax, r.deficit[PRIMARY_OP], r.red_acc[PRIMARY_OP], r, p is not None and r.name == p.name)
            ax.set_title(c, fontsize=9)
            ax.set_xlim(-0.02, 1.0)
            ax.set_ylim(-0.02, 1.0)
            ax.set_xlabel("non-red deficit ↓")
        axes[0].set_ylabel("red accuracy ↓")
        return fig

    mo.md(
        r"""
    Which operator is best depends on the point. On `t00` the syntax rows carry the axis, and there every edit applied at every position costs most of the non-red lines, whatever its threshold, ramp, or landing. The operand-only edits are the only ones inside the gate.
    """
        + _plot()
    )
    return


@app.cell(hide_code=True)
def _(rows: dict[str, list[Row]]):
    _rs = rows["recipe-short"]
    _sh = [r for r in _rs if r.family == "shaped"]
    _li = [r for r in _rs if r.family == "repulsion-linear"]
    _be = [r for r in _rs if r.family == "repulsion-bezier"]
    _ps = sorted({r.p for r in _sh})
    _as = sorted({r.a for r in _li})

    @themed(
        name="marginals",
        alt_text="""
            Four panels in a two-by-two grid. Top row: red accuracy on mix; bottom row: non-red deficit on mix. Left column: shaped suppression against its threshold a, one line per ramp p, solid for every position and dashed for the operand positions; red accuracy rises with both the threshold and the ramp, from near zero to about 0.7, the solid and dashed lines nearly on top of each other; the deficit is flat at zero except for the p = 0 line at every position, which starts at 0.23 at a = 0.1 and falls to zero by a = 0.4. Right column: repulsion against its landing b, one line per threshold a, with Bézier rows as diamonds; red accuracy rises with the landing from about 0.1 to 0.6, and the deficit is flat at zero.
        """,
        caption="""
            **The marginals on `mix`.** Seed means over twenty seeds. Left: shaped suppression against its threshold *a*, one shade per ramp *p* (light to dark: 0, 0.5, 1, 2). Right: linear repulsion against its landing *b*, one shade per threshold *a*, with the Bézier rows as $◆$. Solid lines are the whole-sequence trials, dashed the operand-only ones. The dashed grey rule is the gate.
        """,
    )
    def _plot() -> plt.Figure:
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
            for i, p in enumerate(_ps):
                family_lines(axl, [r for r in _sh if r.p == p], "a", key, shade(cmap_s, i, len(_ps)), f"p = {p:g}")
            for i, a in enumerate(_as):
                family_lines(axr, [r for r in _li if r.a == a], "b", key, shade(cmap_l, i, len(_as)), f"a = {a:g}")
            for r in _be:
                axr.scatter(
                    r.b,
                    getattr(r, key)[PRIMARY_OP],
                    marker="D",
                    s=14,
                    color=ink("repulsion-bezier"),
                    alpha=0.9 if not r.operands else 0.5,
                    zorder=4,
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

    mo.md(
        r"""
    ## The marginals

    The same two reads on `mix`, now plotted against the parameters of each family. For the shaped suppression, the threshold and the ramp both set how much red is left, and the cost is zero everywhere except for the step at low thresholds applied at every position. For the repulsions what matters is the landing: red accuracy follows *b*, and the threshold adds a little on top. The Bézier rows sit with the linear ones at the same landing.
    """
        + _plot()
    )
    return


@app.cell(hide_code=True)
def _(res: Results, rows: dict[str, list[Row]]):
    _c = "recipe-short"
    _rs = [r for r in rows[_c] if not r.operands]
    _clean = res.per_slice(_c, PRIMARY_OP, None, "alpha_red_operand").mean(0)
    _land = {r.name: res.per_slice(_c, PRIMARY_OP, r.name, "alpha_red_operand").mean(0) for r in _rs}
    _x = np.arange(len(SLICE_NAMES))

    @themed(
        name="landing",
        alt_text="""
            Three panels, one per family, each with the five residual-stream slices on the horizontal axis and the red operand's alignment with the anchor axis on the vertical. A bold grey line near 0.9 is the clean value. Under shaped suppression the lines fan out between zero and 0.75 by threshold and ramp, most rising a little from the embedding to slice 3 and dipping at slice 4 as the clean line does. Under the linear repulsion each line sits flat at its landing through slice 3 and dips at the last slice; under the Bézier map the lines start between 0.2 and 0.6 and slope gently down across the slices.
        """,
        caption="""
            **Where the operators leave the red operand, by slice, on the red `mix` lines.** Mean alignment of the dose-carrying operand's state after the edit, over lines and twenty seeds, for every whole-sequence trial; the bold grey line is the clean value. Shaped rows shade by threshold (light to dark: 0.1 to 0.7), repulsion rows by landing *b* (light to dark: 0.2 to 0.7).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        stops = light_dark((0.35, 0.95), (0.05, 0.65))
        for ax, fam, cmap_name in zip(axes, FAMILIES[1:], ("Oranges", "Blues", "Greens"), strict=True):
            cmap = plt.get_cmap(light_dark(cmap_name, cmap_name + "_r"))
            fam_rows = [r for r in _rs if r.family == fam]
            keys = sorted({r.a if fam == "shaped" else r.b for r in fam_rows})
            for r in fam_rows:
                k = r.a if fam == "shaped" else r.b
                i = keys.index(k)
                ax.plot(
                    _x,
                    _land[r.name],
                    lw=0.9,
                    alpha=0.85,
                    color=cmap(stops[0] + (stops[1] - stops[0]) * i / max(len(keys) - 1, 1)),
                )
            ax.plot(_x, _clean, lw=2, color=light_dark("#666", "#bbb"), zorder=0)
            ax.set_xticks(_x, SLICE_NAMES)
            ax.set_title(FAMILY_TITLE[fam], fontsize=9)
            ax.set_ylim(-0.05, 1.0)
        axes[0].set_ylabel("α of the red operand")
        axes[1].set_xlabel("slice")
        return fig

    mo.md(
        r"""
    ## Where the operators leave the state

    In every family, how much red an operator leaves tracks where it leaves the state of the red operand; the marginals above are that same trend plotted against each parameter. The shaped suppression lands each state wherever its ramp puts it, so the landing varies with where the state arrived. A linear repulsion lands every state it touches at one alignment, and the stream holds it there through slice 3. The last slice pulls every landing down, including the clean state.
    """
        + _plot()
    )
    return


@app.cell(hide_code=True)
def _(prop, res: Results, rows: dict[str, list[Row]]):
    _c = "recipe-short"
    _rs = rows[_c]
    _w = {r.name: float(res.per_slice(_c, PRIMARY_OP, r.name, "q99_write_nonred").mean(0).max()) for r in _rs}

    @themed(
        name="write-cost",
        alt_text="""
            One scatter panel: the largest 99th-percentile write on non-red mix lines across the slices on the horizontal axis, in radians, against the non-red deficit on the vertical, zoomed to the first third of its range. Most marks sit at zero deficit with writes under 0.15 radians. The plain projection's cross has the largest write, near 0.27 radians, at a deficit just inside the gate; the operand-only cross sits at 0.2 radians and near-zero deficit; the three open circles for the low-threshold steps have writes between 0.1 and 0.23 radians and deficits of 0.07 to 0.23, above the gate.
        """,
        caption="""
            **What the non-red lines pay for the write they receive, on `mix`.** The 99th-percentile write of each trial on the non-red lines (the largest over the five slices, seed mean) against its non-red deficit. Same marks as the landscape.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        grey = light_dark("#888", "#aaa")
        ax.axhline(ex.NONRED_DEFICIT_GATE, ls="--", lw=0.7, color=grey, zorder=0)
        for r in _rs:
            x, y = _w[r.name], r.deficit[PRIMARY_OP]
            trial_mark(ax, x, y, r, prop is not None and r.name == prop.name)
        ax.set_xlabel("q99 write on non-red lines (rad) ↓")
        ax.set_ylabel("non-red deficit ↓")
        ax.set_ylim(-0.01, 0.3)
        return fig

    mo.md(
        r"""
    ## The write and its cost

    The write is the angle an operator turns a state through. The argument in M1 for the shaped operators is that bounding the write bounds the side-effect, which holds only if the two move together; on the non-red lines of this point they do not. The plain projection writes the non-red lines the most and costs them little, while a step at a = 0.2 writes them less and costs them four times as much. What the non-red lines pay depends more on which of their states are turned than on how far.
    """
        + _plot()
    )
    return


@app.cell(hide_code=True)
def _(prop, rows: dict[str, list[Row]], summary):
    _rs = rows["recipe-short"]
    _front = {r.name for r in summary["front"]}
    _rows = []
    _ref_idx = set()
    for i, r in enumerate(sorted(_rs, key=lambda r: (r.operands, FAMILIES.index(r.family), r.a, r.b, r.p))):
        if r.family == "projection":
            _ref_idx.add(i)
        params = (
            "—"
            if r.family == "projection"
            else (f"a {r.a:g}, p {r.p:g}" if r.family == "shaped" else f"a {r.a:g}, b {r.b:g}")
        )
        bold = lambda v, ok: f"<b>{v:.3f}</b>" if ok else f"{v:.3f}"  # noqa: E731
        marks = (
            ("●" if not r.operands else "▲")
            + (" ⦿" if prop is not None and r.name == prop.name else "")
            + (" ⋆" if r.name in _front else "")
        )
        _rows.append(
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
    mo.md(
        r"""
    ## Every trial

    No omissions. Seed means over the twenty seeds of `recipe-short`. ⦿ is the proposed trial and ⋆ marks the `mix` front; bold values pass their gate. *Margin* is the gate minus the worst non-red deficit over the six ops, which is what the survey ranks on alongside the objective.
    """
        + table_html(
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
            _rows,
            "**Every trial on `recipe-short`.** Reference rows are shaded.",
            ref_rows=frozenset(_ref_idx),
        )
    )
    return


@app.cell(hide_code=True)
def _(band, prop, prop_t00, rows: dict[str, list[Row]], summary):
    _rs = rows["recipe-short"]
    _by = {r.name: r for r in _rs}
    _ops, _bf = _by["operands"], summary["best_free"]
    _fr = summary["front"]
    _front_txt = ", ".join(f"`{r.name}` ({r.red_acc[PRIMARY_OP]:.3f}, {r.deficit[PRIMARY_OP]:.3f})" for r in _fr)
    if prop is None:
        _body = "No trial is feasible on every op, so the rule proposes nothing. The infeasibility map above is the finding."
    else:
        _near = [
            r
            for r in _rs
            if r.feasible
            and r.name != prop.name
            and abs(r.red_acc[PRIMARY_OP] - prop.red_acc[PRIMARY_OP]) <= band["red_acc"]
        ]
        _t00_rows = {r.name: r for r in rows["t00"]}
        _p_t00 = _t00_rows[prop.name]
        _body = rf"""
    The frozen rule proposes **`{prop.name}`**, {describe(prop)}. On `mix` it is at {prop.red_acc[PRIMARY_OP]:.3f} red accuracy and a non-red deficit of {prop.deficit[PRIMARY_OP]:.3f}. Its worst deficit over the ops is {prop.worst_deficit:.3f}, a margin of {prop.margin:+.3f}, which is less than the deficit band of {band["deficit"]:.3f}. The operand-only projection from ex-2.2.3 is at {_ops.red_acc[PRIMARY_OP]:.3f} and {_ops.deficit[PRIMARY_OP]:.3f}, a margin of {_ops.margin:+.3f}.

    {len(_near)} other feasible trial(s) sit within one red-accuracy band ({band["red_acc"]:.3f}) of the proposal{": " + ", ".join(f"`{r.name}`" for r in _near) if _near else ""}. The `mix` front, from the gentlest edit to the most complete, is {_front_txt}.

    On `t00` the same trial is at {_p_t00.red_acc[PRIMARY_OP]:.3f} and {_p_t00.deficit[PRIMARY_OP]:.3f}, with a worst deficit of {_p_t00.worst_deficit:.3f}, outside the gate. The rule picks {f"`{prop_t00.name}`, {describe(prop_t00)}" if prop_t00 is not None else "nothing"} there.
    """
    mo.md(
        rf"""
    ## The proposal

    {_body}

    **What the prereg should carry.** The rule picks the plain projection, which is the row the anchored-op experiments already score, so on the adopted point this pass changes nothing about the primary intervention. What it adds is the margin: the projection is inside the gate by less than a band. The prereg should therefore keep the operand-only projection beside it as the selective reference.

    Two things the rule did not rank are for the prereg to settle before its seeds are drawn. The first is that the answer depends on the point: on `t00` the same rule picks an operand-only step. A prereg that may run on a syntax-heavy point should name the operand-only operator as its intervention there, rather than choosing it after the read.

    The second is a candidate that needs no syntax. Applied at every position, `{_bf.name}` is at {_bf.red_acc[PRIMARY_OP]:.3f} red accuracy at no cost the survey can resolve, close to the {_ops.red_acc[PRIMARY_OP]:.3f} of the operand-only projection but with no position mask, which is what a prompt with unknown operand positions will need. Choosing it here would be post hoc, so it is recorded for the prereg to carry as a third row, re-scored at fresh seeds, if the syntax-free question is worth one.
    """
    )
    return


@app.cell(hide_code=True)
def _(check_worst):
    mo.md(rf"""
    ## Post hoc

    Nothing was added after the run. The reference rows reproduced {"to the bit" if check_worst == 0 else f"to {check_worst:.1e}"}, and every contract check passed on every trial and seed; a failed check fails the scoring task, and none did. The plan allowed for the rule picking a reference row, since the references are trials and the objective ranks them with the rest.

    Two observations were not anticipated by the plan, and are recorded as exploratory. The plan expected a threshold to trade removal against cost monotonically, with the plain projection at the costly end. Instead, on the adopted point a step set inside the alignment range of the non-red lines costs more than the projection does, and on `t00` every whole-sequence step costs more than projecting everything.

    Both point the same way: a line whose states are zeroed on some tokens and kept on others is decoded worse than one zeroed throughout. That suggests a prediction the anchored-op prereg can carry: above the non-red range the whole-sequence cost is zero, and below it the cost rises as the threshold falls.
    """)
    return


if __name__ == "__main__":
    app.run()
