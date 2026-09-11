import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Ex 2.2.7: a pilot of the syntax rows",
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
    from mini.vis import AxesGrid, figure_html, light_dark, themed

    use_publisher(report_bundle(__file__))

    ROLES = ["op1", "op", "op2", "=", "ans", "⏎"]
    PRED = {2: "=", 3: "ans", 4: "⏎"}
    """The next-token predictions that carry information: from op2 the `=`, from `=` the answer, from the
    answer the newline. The op word after op1 and op2 after the op word are unpredictable by construction."""
    STORED = list(ex.STORED)
    ARMS = [a.name for a in ex.ARMS] + [ex.REFERENCE]
    """Part B's arms, then the production reference."""
    ROW_ARMS = STORED + [a.name for a in ex.ARMS]
    """Every arm the row table is read on: the stored checkpoints, then the pilot's."""
    FIXES = {a.name: a.fix for a in ex.ARMS} | {ex.REFERENCE: "none"}
    MIX = ex.ex223.PRIMARY_OP.name
    PROJECTION = ex.ex223.PROJECTION.name
    OPERANDS = "operands"
    OPS = list(ex.ex223.OP_NAMES)
    SYNTAX = list(ex.SYNTAX_WORDS)
    EQ, NL = "=", "\n"

    INK = {
        "blocks-only": ("#1f6fb4", "#5fa8dd"),
        "untied": ("#1a8f7a", "#4fc3ac"),
        "rows-clean": ("#d0461b", "#f07a50"),
        "blocks-only-line": ("#6a5acd", "#a394f0"),
        "untied-line": ("#b8860b", "#e0b040"),
        "recipe-short": ("#9a9a9a", "#7a7a7a"),
        "t00": ("#555555", "#bbbbbb"),
        "control-short": ("#c0c0c0", "#555555"),
    }
    """One ink per arm, as (light, dark) pairs; the production arms draw grey."""
    STRIP_LABEL = {
        "clean": "clean",
        "input": "input",
        "output": "output",
        "both": "both",
        "input-control": "input\n(control)",
    }

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
def load_npz(ref: str) -> dict[str, np.ndarray] | None:
    """A published npz as a dict of arrays, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "arrays.npz")])
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


@app.function(hide_code=True)
def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:{fmt}}"
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


@app.function(hide_code=True)
def ink(cond: str):
    return light_dark(*INK[cond])


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
    return figure_html(table, caption=caption, class_="report-figure")


@app.class_definition(hide_code=True)
@dataclass(frozen=True)
class Results:
    """The pilot's arms, the stored checkpoints Part A scored, and the production reference they are read
    against, with one accessor per shape the cells need.
    """

    metrics: dict
    arrays: dict[str, np.ndarray]
    prod: dict
    prod_arrays: dict[str, np.ndarray]

    def is_prod(self, cond: str) -> bool:
        return cond in (ex.REFERENCE, *ex.STORED)

    def _records(self, key: str, cond: str) -> list[dict]:
        if self.is_prod(cond):
            names = (cond, f"{cond}-more") if cond == ex.REFERENCE else (cond,)
            return sorted((r for r in self.prod[key] if r["condition"] in names), key=lambda r: r["seed"])
        return sorted((r for r in self.metrics[key] if r["condition"] == cond), key=lambda r: r["seed"])

    def runs(self, cond: str) -> list[dict]:
        """The eval records of an arm in seed order; the production reference pools its addendum seeds."""
        return self._records("runs", cond)

    def scored(self, cond: str) -> list[dict]:
        return self._records("scores", cond)

    def rows(self, cond: str) -> list[dict]:
        """Part A's records for an arm, in seed order (stored arms and pilot arms alike)."""
        return sorted((r for r in self.metrics["rows"] if r["condition"] == cond), key=lambda r: r["seed"])

    def stat(self, cond: str, key: str, op: str | None = None) -> np.ndarray:
        return np.array([r[key] if op is None else r["per_op"][op][key] for r in self.runs(cond)], float)

    def em(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_em"][op] for r in self.runs(cond)], float)

    def score(self, cond: str, op: str, iv: str | None, key: str, group: str) -> np.ndarray:
        out = []
        for r in self.scored(cond):
            s = r["ops"][op]
            v = s["clean"][key] if iv is None else s["interventions"][iv][key]
            out.append(v[group])
        return np.array(out, float)

    def deficit(self, cond: str, op: str, iv: str, group: str = "nonred") -> np.ndarray:
        return self.score(cond, op, None, "acc", group) - self.score(cond, op, iv, "acc", group)

    def component(self, cond: str, word: str, table: str = "rows") -> np.ndarray:
        """The axis component of one row, per seed, on the embedding (`rows`) or the readout (`rows_readout`)."""
        return np.array([r[table][word] for r in self.rows(cond) if r[table] is not None], float)

    def color_abs(self, cond: str, table: str = "rows") -> np.ndarray:
        """Per seed, the mean |component| over the color rows of one table."""
        out = []
        for r in self.rows(cond):
            if r[table] is None:
                continue
            out.append(np.mean([abs(v) for w, v in r[table].items() if w not in SYNTAX]))
        return np.array(out, float)

    def strip_acc(self, cond: str, strip: str, op: str, group: str, pos: int) -> np.ndarray:
        """Next-token accuracy at one predicting position, per seed, under one strip condition."""
        return np.array([r["strip"][strip][op]["acc"][group][pos] for r in self.rows(cond)], float)

    def strip_p(self, cond: str, strip: str, op: str, group: str, pos: int) -> np.ndarray:
        return np.array([r["strip"][strip][op]["p_next"][group][pos] for r in self.rows(cond)], float)

    def strip_proj(self, cond: str, strip: str, op: str, key: str, group: str) -> np.ndarray:
        """The full-position projection read on the stripped model: `acc` or the P(answer) `deficit`."""
        return np.array([r["strip"][strip][op]["projection"][key][group] for r in self.rows(cond)], float)

    def tied(self, cond: str) -> bool:
        return all(r["tied"] for r in self.rows(cond))


@app.cell(hide_code=True)
def _(res: Results):
    _ans = lambda c, s, g: res.strip_acc(c, s, MIX, g, 3).mean()  # noqa: E731
    _eq = lambda c, s: res.strip_acc(c, s, MIX, "red", 2).mean()  # noqa: E731
    _row = {c: res.component(c, EQ).mean() for c in ROW_ARMS}
    _ops = {c: np.mean([res.component(c, w).mean() for w in OPS]) for c in ROW_ARMS}
    _head = {c: res.component(c, EQ, "rows_readout").mean() for c in ROW_ARMS if not res.tied(c)}
    _nl = {c: res.component(c, NL).mean() for c in ROW_ARMS}
    _em_floor = min(res.em(c, op).mean() for c in ARMS for op in OPS)
    _ml = {c: res.stat(c, "m_line") for c in ARMS}
    _pilot_ml = [_ml[c].mean() for c in ARMS if c != ex.REFERENCE]
    _ref = _ml[ex.REFERENCE]
    _proj_def = {c: res.deficit(c, MIX, PROJECTION) for c in ARMS}
    _opnd_def = res.deficit(ex.REFERENCE, MIX, OPERANDS).mean()
    _proj_red = {c: res.score(c, MIX, PROJECTION, "acc", "red") for c in ARMS}
    _bo_worst = _proj_red["blocks-only"].max()
    mo.md(rf"""
    # Ex 2.2.7: a pilot of the syntax rows

    /// tip |
    <!-- tl;dr -->
    A scouting run, with no gates. Every anchored model carries the anchor axis on the embedding rows of the op words and `=`, and a full-position projection pays for that on the non-red lines. We asked where that component does its work, by stripping it from the stored checkpoints on the input side or the output side of the tied table, and then retrained the recipe three ways that each remove one candidate mechanism: anchoring the blocks only, untying the readout, and holding the rows clean by a hard constraint.

    The logit path puts it there. With a readout table of its own the model moves the component onto that table and the embedding rows come mostly clean, while leaving the embedding out of the anchor does not remove it and makes the projection less complete. Holding the rows at zero after every step costs nothing we can see, and the full-position projection's non-red cost comes down to the operand-only edit's. The pilot proposes carrying that constraint into the handover.
    ///

    ## Observations

    - **Where the component works.** On the stored `{ex.REFERENCE}` checkpoints, stripping it from the readout side costs the `=` prediction on the red lines ({_eq("recipe-short", "clean"):.2f} to {_eq("recipe-short", "output"):.2f}; {_eq("t00", "output"):.2f} on `t00`) and nothing else. Stripping it from the embedding side costs the answer instead ({_ans("recipe-short", "clean", "red"):.2f} to {_ans("recipe-short", "input", "red"):.2f} on the red lines, {_ans("recipe-short", "input", "nonred"):.2f} on the non-red), but turning the rows by the same angle in a random direction costs at least as much ({_ans("recipe-short", "input-control", "red"):.2f}), so that side reads as the blocks having learned where the `=` row sits ([figure](#where-the-component-works)).
    - **The row table.** The `=` row carries {_row["recipe-short"]:.2f} on `{ex.REFERENCE}` and {_row["t00"]:.2f} on `t00`, the op words {_ops["recipe-short"]:.2f} and {_ops["t00"]:.2f}. Anchoring the blocks only leaves it at {_row["blocks-only"]:.2f}. Untying the readout brings the embedding's `=` row to {_row["untied"]:.2f} and puts {_head["untied"]:.2f} on the readout's; under the whole-line labeller the `⏎` row takes the axis as well ({_nl["blocks-only-line"]:.2f} on `blocks-only-line`, {_head["untied-line"]:.2f} on `untied-line`'s readout). `rows-clean` holds every syntax row at zero by construction ([figure](#the-row-table)).
    - **Task cost.** Held-out exact match is at least {_em_floor:.3f} on every op of every arm ([table](#task-cost)).
    - **Placement.** m_line runs {min(_pilot_ml):.3f}–{max(_pilot_ml):.3f} across the pilot arms, against {_ref.mean():.3f} <span class='range'>±{(_ref.max() - _ref.min()) / 2:.3f}</span> on `{ex.REFERENCE}`; `blocks-only` is the low end ([table](#where-the-pull-lands)).
    - **Suppression and selectivity.** Under the full-position projection the non-red `mix` deficit is {span2(_proj_def["recipe-short"])} on `{ex.REFERENCE}` (the operand-only edit reads {_opnd_def:.3f} there), and {span2(_proj_def["untied"])} on `untied`, {span2(_proj_def["rows-clean"])} on `rows-clean` and {span2(_proj_def["blocks-only"])} on `blocks-only`. Red-line `mix` accuracy under the projection is {_proj_red["recipe-short"].mean():.2f} on the reference, {_proj_red["untied"].mean():.2f} on `untied` and {_proj_red["rows-clean"].mean():.2f} on `rows-clean`; on `blocks-only` it is {_proj_red["blocks-only"].mean():.2f}, with one seed at {_bo_worst:.2f} ([table](#suppression-and-selectivity)).
    """)
    return


@app.cell(hide_code=True)
def _():
    _arms = ex.ARMS
    _stored = ex.STORED
    mo.md(
        rf"""
    ## Why, and what we ran

    Ex-2.2.2's E8 found the anchor axis on the embedding rows of the op words and `=`, at a few tenths in every anchored model, and ex-2.2.3 found more of it at the heavier adopted point. That component is what a full-position projection pays for on the non-red lines, and the operand-only edit routes around it only because this grammar tells us where the operands are. In M3 there is no operand position, so the M3-shaped operator is full-position and needs the rows clean. The [tied-readout item](https://github.com/z0u/sca2/blob/main/todo/science/syntax-rows-carry-the-axis-via-tied-readout.md) and the design's Prep C each name a mechanism; this pilot runs both, with a third beside them.

    nGPT ties the readout to the embedding, so one row does two jobs: it is the residual stream's starting state when its token is read, and it is the logit of its token at every position that predicts it. Three mechanisms could put the axis on it.

    1. **The direct pull.** The anchor term acts at every residual slice, the embedding included, and the labelled span covers the op word and `=`. Their slice-0 states are the rows themselves.
    2. **The tied readout.** After a red operand the stream sits near e₁; the cheapest way to raise the logit of the token that follows is to lean that token's row the same way.
    3. **The blocks.** A block that reads the component from the stream at a syntax position has a reason to keep it there, whichever table put it in.

    **Part A** prices the component without training anything. On ex-2.2.3's stored checkpoints ({", ".join(f"`{c}`, {n} seeds" for c, n in _stored.items())}) we strip the axis component from the syntax rows on the *input* side only (the embedding, with the original table kept as the readout), the *output* side only, or both, and read the next-token accuracy at each position on the red and non-red probe lines, and the full-position projection's cost on the stripped model. A control strips the same amount from the embedding rows along a random direction off the axis, so an input-side cost is the axis component's and not any disturbance of a syntax row's. Whichever side loses the syntax predictions is where the component works.

    **Part B** retrains ex-2.2.3's `{ex.REFERENCE}` (λ = {ex.ex223.SCORING_LAMBDA:g}, {ex.EPOCHS} epochs) three ways, each removing one mechanism, and reads Part A's table, the E8 row table, and ex-2.2.3's placement and suppression statistics on every run. The first two also run under ex-2.2.6's whole-line labeller, which the handover proposes to adopt and which pulls the syntax positions by design.

    | arm | seeds | readout | anchored slices | rows held clean | labeller | |
    | --- | ---: | --- | --- | :---: | --- | --- |
    {chr(10).join(f"| `{a.name}` | {a.seeds} | {'tied' if a.tie else 'untied'} | {'blocks only' if a.slices else 'all'} | {'yes' if a.clean else 'no'} | {'whole line' if a.keying == 'line' else 'operands'} | {a.title} |" for a in _arms)}
    | `{ex.REFERENCE}` | 20 | tied | all | no | operands | production, ex-2.2.3's recipe |

    `rows-clean` is the simplest fix that keeps the table shared: after every optimizer step, the same projection that keeps nGPT's rows at unit length also zeroes the axis component of every non-color row. `untied` gives the readout a table of its own, initialised as a copy of the embedding, so the anchor and anti-subspace terms keep acting on the embedding and the logits are free. `blocks-only` is Prep C: both anchoring terms skip slice 0.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    _metrics = load_json(ex.METRICS_REF)
    _arrays = load_npz(ex.ARRAYS_REF)
    mo.stop(
        _metrics is None or _arrays is None,
        mo.callout(
            mo.md("No results published yet. Run the experiment (see `experiment.py`) and re-open this report."),
            kind="warn",
        ),
    )
    _prod = load_json(ex.EX223_METRICS_REF)
    _prod_arrays = load_npz(ex.EX223_ARRAYS_REF)
    mo.stop(
        _prod is None or _prod_arrays is None,
        mo.callout(mo.md("Ex-2.2.3's production results are not reachable."), kind="warn"),
    )
    assert _metrics is not None and _arrays is not None
    assert _prod is not None and _prod_arrays is not None
    res: Results = Results(_metrics, _arrays, _prod, _prod_arrays)
    return (res,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where the component works

    Part A on the stored checkpoints. Each panel is one of the three informative next-token predictions on the `mix` probe lines: `=` from op2, the answer from `=`, and the newline from the answer. The x axis is the strip condition. A filled marker is the red lines (both operands' redness at least 0.8), a hollow one the non-red lines; bars are the seed range over five seeds.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _strips = list(ex.STRIPS)
    _x = np.arange(len(_strips))

    @themed(
        name="strip-accuracy",
        alt_text="""
            A grid of small panels, three rows by three columns. Rows are the three next-token predictions: the equals sign, the answer, and the newline. Columns are the stored arms recipe-short, t00, and control-short. In each panel the five strip conditions run along the x axis (clean, input, output, both, and the input control) and next-token accuracy along the y axis, with filled markers for the red lines and hollow ones for the non-red lines, and a bar for the seed range.
        """,
        caption=r"""
            **Next-token accuracy on the `mix` probe lines under each strip, on the stored checkpoints.** Rows are the predictions of `=`, the answer, and the newline; columns are the stored arms. Filled markers are the red lines, hollow the non-red; bars span the five seeds. `input` strips the axis component from the syntax rows of the embedding and keeps the original table as the readout; `output` the reverse; `both` strips it from the shared table; the control moves the embedding rows the same distance along a random direction off the axis.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, len(STORED), figsize=(5.6, 4.2), sharex=True, sharey="row")
        axes = cast(AxesGrid, axes)
        for col, c in enumerate(STORED):
            color = ink(c)
            for row, (pos, name) in enumerate(PRED.items()):
                ax = axes[row][col]
                for group, fill in (("red", color), ("nonred", "none")):
                    m = np.array([res.strip_acc(c, s, MIX, group, pos).mean() for s in _strips])
                    lo = np.array([res.strip_acc(c, s, MIX, group, pos).min() for s in _strips])
                    hi = np.array([res.strip_acc(c, s, MIX, group, pos).max() for s in _strips])
                    ax.errorbar(_x, m, yerr=[m - lo, hi - m], fmt="o", ms=3.5, lw=0.8, color=color, mfc=fill)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", c="#888", alpha=0.2)
                ax.set_ylim(-0.05, 1.05)
                if col == 0:
                    ax.set_ylabel(f"acc, predicting {name}", fontsize=7.5)
            axes[0][col].set_title(c, fontsize=8, color=color)
            axes[2][col].set_xticks(_x, [STRIP_LABEL[s] for s in _strips], fontsize=6.5)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(res: Results):
    _rows = []
    for _c in STORED + [a.name for a in ex.ARMS]:
        for _s in ex.STRIPS:
            _rows.append(
                [f"<code>{_c}</code>" if _s == "clean" else "", STRIP_LABEL[_s].replace("\n", " ")]
                + [span2(res.strip_acc(_c, _s, MIX, g, 3), ".2f") for g in ("red", "nonred")]
                + [span2(res.strip_p(_c, _s, MIX, g, 3), ".2f") for g in ("red", "nonred")]
                + [span2(res.strip_proj(_c, _s, MIX, "acc", "red"), ".2f")]
                + [span2(res.strip_proj(_c, _s, MIX, "acc", "nonred"), ".3f")]
            )
    _head = [
        "arm",
        "strip",
        "answer acc, red",
        "answer acc, non-red",
        "P(answer), red",
        "P(answer), non-red",
        "projection: red acc",
        "projection: non-red acc",
    ]
    _caption = """
    The answer prediction on the <code>mix</code> probe lines under each strip, per arm, and the full-position projection read on the stripped model: exact-match accuracy on the red lines (the removal read) and on the non-red lines (the selectivity read, as an accuracy rather than a deficit, since a strip moves the clean baseline too). Seed means with half the seed range. The pilot arms follow the stored ones; a fix that has already cleaned its rows shows no difference across its strips.
    """
    _n = len(ex.STRIPS)
    mo.Html(table_html(_head, _rows, _caption, ref_rows=frozenset(range(0, 3 * _n))))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The row table

    Ex-2.2.2's E8, read on every arm: the axis component of each syntax row of the embedding table, with the mean absolute component over the 216 color rows beside it. nGPT's rows are unit vectors, so a component is a cosine. The untied arms carry a second table; its rows are drawn hollow.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _x = np.arange(len(ROW_ARMS))

    @themed(
        name="row-components",
        alt_text="""
            A single panel. The arms run along the x axis, from the stored checkpoints on the left to the pilot arms on the right, and the axis component of an embedding row along the y axis. At each arm there is a marker for the equals sign, one for the newline, a cluster for the six op words, and a grey marker for the mean absolute component of the color rows; bars span the seeds. The untied arms show a second, hollow set for their readout table.
        """,
        caption=r"""
            **Axis component per syntax row, by arm.** Each marker is the seed mean of one row's component on e₁ (a cosine, since the rows are unit vectors), with the seed range as a bar; the op words are drawn small and the color rows' mean absolute component in grey. Hollow markers are the readout table of the untied arms. The stored arms are ex-2.2.3's; every pilot arm is Part B's.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        grey = light_dark("#999", "#777")
        ax.axhline(0, color=grey, lw=0.6)
        for i, c in enumerate(ROW_ARMS):
            color = ink(c)
            tables = ["rows"] + ([] if res.tied(c) else ["rows_readout"])
            for k, table in enumerate(tables):
                fill = color if table == "rows" else "none"
                off = -0.12 if len(tables) == 2 and k == 0 else (0.12 if len(tables) == 2 else 0.0)
                for word, marker, ms in ((EQ, "s", 4.5), (NL, "^", 4.5)):
                    v = res.component(c, word, table)
                    ax.errorbar(
                        i + off,
                        v.mean(),
                        yerr=[[v.mean() - v.min()], [v.max() - v.mean()]],
                        fmt=marker,
                        ms=ms,
                        lw=0.8,
                        color=color,
                        mfc=fill,
                    )
                for j, word in enumerate(OPS):
                    v = res.component(c, word, table)
                    ax.plot(i + off + (j - 2.5) * 0.03, v.mean(), "o", ms=2, color=color, mfc=fill)
                v = res.color_abs(c, table)
                ax.plot(i + off, v.mean(), "D", ms=3, color=grey, mfc=grey if table == "rows" else "none")
        ax.set_xticks(_x, ROW_ARMS, fontsize=7, rotation=25, ha="right")
        ax.set_ylabel("component on e₁", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", c="#888", alpha=0.2)
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="s", ls="", color=grey, ms=4.5, label="="),
            Line2D([], [], marker="^", ls="", color=grey, ms=4.5, label="⏎"),
            Line2D([], [], marker="o", ls="", color=grey, ms=2, label="op words"),
            Line2D([], [], marker="D", ls="", color=grey, ms=3, label="colors, mean |·|"),
        ]
        ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(res: Results):
    _rows = []
    for _c in ROW_ARMS:
        _rows.append(
            [
                f"<code>{_c}</code>",
                "embedding",
                span2(res.component(_c, EQ), ".2f"),
                span2(res.component(_c, NL), ".2f"),
            ]
            + [span2(res.component(_c, w), ".2f") for w in OPS]
            + [span2(res.color_abs(_c), ".3f")]
        )
        if not res.tied(_c):
            _rows.append(
                [
                    "",
                    "readout",
                    span2(res.component(_c, EQ, "rows_readout"), ".2f"),
                    span2(res.component(_c, NL, "rows_readout"), ".2f"),
                ]
                + [span2(res.component(_c, w, "rows_readout"), ".2f") for w in OPS]
                + [span2(res.color_abs(_c, "rows_readout"), ".3f")]
            )
    _head = ["arm", "table", "=", "⏎"] + [f"<code>{w}</code>" for w in OPS] + ["colors, mean |·|"]
    _caption = "The row table as numbers: the signed axis component of each syntax row, per arm and table, seed mean with half the seed range; the last column is the mean absolute component over the color rows."
    mo.Html(table_html(_head, _rows, _caption, ref_rows=frozenset(range(len(STORED)))))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost

    Exact match on the held-out pairs of each op. The reference is production's twenty seeds; the pilot arms have two or three each, so the half-ranges are rough.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _rows = [
        [f"<code>{c}</code>", f"{len(res.runs(c))}"]
        + [span2(res.em(c, op), ".3f") for op in OPS]
        + [f"{res.em(c, MIX).mean() - res.em(ex.REFERENCE, MIX).mean():+.3f}"]
        for c in ARMS
    ]
    _caption = "Held-out exact match per op, seed mean with half the seed range; the last column is the `mix` gap from the production reference. Ex-2.2.3's task gate was a `mix` gap within 0.02 of its own control."
    mo.Html(
        table_html(
            ["arm", "seeds"] + [f"<code>{op}</code>" for op in OPS] + ["Δ mix"],
            _rows,
            _caption,
            ref_rows=frozenset({len(ARMS) - 1}),
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where the pull lands

    Ex-2.2.3's placement statistics on the `mix` probe lines, per arm. The line arms' m_line is under their own labeller's weighting, as in ex-2.2.6.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _rows = []
    for _c in ARMS:
        _rows.append(
            [
                f"<code>{_c}</code>",
                span2(res.stat(_c, "m_line")),
                span2(res.stat(_c, "alpha_op1")),
                span2(res.stat(_c, "lead_emb"), ".2f"),
                span2(res.stat(_c, "contrast"), ".2f"),
                span2(res.stat(_c, "r2_sim")),
                span2(res.stat(_c, "latch_pi"), ".2f"),
                span2(res.stat(_c, "retention"), ".2f"),
            ]
        )
    _head = ["arm", "m_line", "ᾱ op1", "lead (emb)", "contrast", "r² sim", "latch π", "retention"]
    _caption = """
    Placement on the <code>mix</code> probe lines, seed means with half the seed range. m_line is the per-line margin under the arm's own labeller; ᾱ op1 the mean alignment at op1 over every slice and color (ex-2.2.3's containment read); lead the G1 group's softmin weight on op1 at the embedding; contrast the deep-slice op2 weight of G2 minus G1; r² sim the grading of the op1 response against the similarity target; latch π the larger of the non-red group's deep-slice weights on the op word and on <code>=</code>; retention the final m_line over its running peak. On <code>blocks-only</code> the embedding slice is not pulled, so its lead is what the blocks' pull leaves there.
    """
    mo.Html(table_html(_head, _rows, _caption, ref_rows=frozenset({len(ARMS) - 1})))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Suppression and selectivity

    Ex-2.2.3's H4 statistics per arm: red-line accuracy under the full-position `projection` on each op (the removal read, lower is more complete), and the non-red `mix` deficit under `projection` and under the operand-only edit (the selectivity read). The question for each fix is whether the full-position deficit comes down to the operand-only one.
    """)
    return


@app.cell(hide_code=True)
def _(res: Results):
    _rows = []
    for _c in ARMS:
        _rows.append(
            [f"<code>{_c}</code>"]
            + [span2(res.score(_c, op, PROJECTION, "acc", "red"), ".2f") for op in OPS]
            + [span2(res.deficit(_c, MIX, PROJECTION), ".3f"), span2(res.deficit(_c, MIX, OPERANDS), ".3f")]
            + [span2(res.score(_c, MIX, PROJECTION, "acc", "nonred"), ".3f")]
        )
    _head = (
        ["arm"]
        + [f"<code>{op}</code>" for op in OPS]
        + ["deficit, projection", "deficit, operands", "non-red acc, projection"]
    )
    _caption = f"""
    Exact-match accuracy under the full-position <code>projection</code> on the red lines (dose ≥ {ex.ex223.RED_DOSE:g}) of each op, then on <code>mix</code>: the non-red (dose ≤ {ex.ex223.NONRED_DOSE:g}) accuracy deficit under the full-position projection and under the operand-only edit, and the non-red accuracy under the projection. Ex-2.2.3's gates were red accuracy at most {ex.ex223.RED_ACC_GATE:g} on every op and a deficit at most {ex.ex223.NONRED_DEFICIT_GATE:g}.
    """
    mo.Html(table_html(_head, _rows, _caption, ref_rows=frozenset({len(ARMS) - 1})))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What we make of it

    **Which mechanism.** Each of the three arms removed one candidate, and two of the three results point the same way. Leaving the embedding out of the anchor did not clean the rows, so the direct pull at slice 0 was not the cause. Giving the readout a table of its own moved the component onto that table, at about the reference's size, and left the embedding rows mostly clean, so the logit path is what places it. Part A reads the same from the stored checkpoints: the readout side's component is what the `=` prediction after a red op2 leans on, which is the [tied-readout item](https://github.com/z0u/sca2/blob/main/todo/science/syntax-rows-carry-the-axis-via-tied-readout.md)'s mechanism as written, and that is the only thing the readout side does. What the embedding-side strips show is that the blocks learn to read the rows where the logit path leaves them: the random-direction control costs as much, and on `t00`, where the `=` row sits about 70° off a clean one, a stripped model is a different model. So the component is not vestigial either; a stronger anti-subspace term on those rows would be working against the logits.

    **Prep C.** Of the design's three hypotheses, (a) holds in the blocks, with the contrast and the similarity grading at the reference's values and m_line a little lower; (b) holds, since the color rows still lead the pull at the embedding even though nothing pulls them there; and (c) does not hold, because the rows keep the axis. The cost Prep C did not anticipate is completeness: with the embedding un-anchored and the anti-subspace term skipping it too, the red operand's information stays readable off the axis at slice 0, and on one of three seeds most red `mix` lines survive the full-position projection.

    **The fix to carry.** Holding the syntax rows at zero after every step costs nothing this pilot can see: task, placement, and removal all read as the reference on three seeds, and the projection's non-red cost sits at the operand-only edit's level. That comparison is the one to hold loosely, since the reference's twenty seeds spread across most of that range and three seeds cannot resolve it; the handover's twenty can. The constraint keeps the table shared, so it is the arm that carries to tied models, and the untied arm stays a diagnostic. Under the whole-line labeller the `⏎` row takes the axis the same way, since it follows the newly pulled answer; the constraint covers every non-color row, so it should hold there too, but that arm did not run here. The `untied-line` arm read a larger non-red cost on one of its two seeds under both edits, which points at the labeller rather than the rows, and two seeds is too few to say more.

    **What it changes.** The handover can adopt the row constraint and read the full-position projection beside the operand-only edit as its removal operator, which is what the M3-shaped operator needs. The operator pass in the D2.2 design chooses between plain projection and the shaped forms on that footing.

    ## Method notes

    - Part A's strips edit the tables of a loaded checkpoint and nothing else; `input` and `output` give the tied model a readout table of its own for the scoring pass (`NGPT.with_tables`), so the two sides can differ. The stripped rows are re-normalized to unit length, as nGPT keeps them.
    - The control strip replaces each row's axis component with one of the same size along a random unit direction orthogonal to e₁ (one fixed draw), so the row turns by about the same angle as under `input` and ends up off the axis as well. It reads how much of the `input` cost is the turn itself.
    - Part B's arms share ex-2.2.3's corpus, eval sets and probe lines (the same seeds through its `prepare_corpus`); the line arms train against ex-2.2.6's line-keyed probe table.
    - `rows-clean` applies `sca.anchoring.clean_embedding_rows` after every optimizer step, to every non-color row but the pad; `blocks-only` passes `anchor_slices=(1, 2, 3, 4)` to `train_anchored`, so both the anchor and the anti-subspace term skip the embedding; `untied` sets `tie_embeddings=False` on the model config, and its checkpoints carry the second table.
    - The eval and score tasks are ex-2.2.3's, unchanged; on the untied arms the eval contract reads logits through the readout table and `ablate_weights` projects both tables.
    - This is a pilot: five arms with two or three seeds each, five seeds of each stored arm, and production's twenty for the reference. Nothing here is gated, and nothing in it should be quoted as a result.
    """)
    return


if __name__ == "__main__":
    app.run()
