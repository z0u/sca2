# title: Ex 2.2.7: a pilot of the syntax embeddings

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

# The document's directory is on sys.path while it runs, so the conditions and the
# ref come from the definition module beside this one.
import experiment as ex
from mini.lit import memo, read_npz, stop
from mini.store import project_store
from mini.vis import AxesGrid, figure_html, light_dark, themed

ROLES = ["op1", "op", "op2", "=", "ans", "⏎"]
PRED = {2: "=", 3: "ans", 4: "⏎"}
# The next-token predictions that carry information: from op2 the `=`, from `=` the answer, from the
# answer the newline. The op word after op1 and op2 after the op word are unpredictable by construction.
STORED = list(ex.STORED)
ARMS = [a.name for a in ex.ARMS] + [ex.REFERENCE]
# Part B's arms, then the production reference.
ROW_ARMS = STORED + [a.name for a in ex.ARMS]
# Every arm the embedding-component table is read on: the stored checkpoints, then the pilot's.
SHOWN = {"rows-clean": "syntax-off-axis"}
# Display names for arms whose stored name is kept for its memo key (the `style-terms` skill).


def show(arm: str) -> str:
    return SHOWN.get(arm, arm)


FIXES = {a.name: a.fix for a in ex.ARMS} | {ex.REFERENCE: "none"}
MIX = ex.ex223.PRIMARY_OP.name
PROJECTION = ex.ex223.PROJECTION.name
OPERANDS = "operands"
OPS = list(ex.ex223.OP_NAMES)
SYNTAX = list(ex.SYNTAX_WORDS)
EQ, NL = "=", "\n"
TAIL = 0.07
# A non-red `mix` deficit above this counts as a tail seed: the reference's twenty seeds run to about this value.

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
# One ink per arm, as (light, dark) pairs; the production arms draw grey.
STRIP_LABEL = {
    "clean": "clean",
    "input": "input",
    "output": "output",
    "both": "both",
    "input-control": "input\n(control)",
}


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many` for the lot."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:{fmt}}"
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


def ink(cond: str):
    return light_dark(*INK[cond])


def dots(
    ax,
    x: float,
    v,
    color,
    *,
    marker: str = "o",
    mfc=None,
    ms: float = 4.5,
    jitter: float = 0.08,
    clip: tuple[float, float] | None = None,
) -> None:
    """One seed per small dot, spread a little in x, with the seed mean as a larger marker on top. With `clip`,
    seeds beyond the panel are drawn as small hollow triangles at its edge (the mean is still of every seed).
    """
    v = np.asarray(v, float)
    if len(v) == 0:
        return
    xs = x + (np.linspace(-jitter, jitter, len(v)) if len(v) > 1 else np.zeros(1))
    if clip is not None:
        lo, hi = clip
        for beyond, edge, tri in ((v > hi, hi, "^"), (v < lo, lo, "v")):
            if beyond.any():
                ax.plot(xs[beyond], np.full(beyond.sum(), edge), tri, ms=3, color=color, mfc="none", zorder=2, ls="")
        v_in = np.clip(v, lo, hi)
        shown = (v >= lo) & (v <= hi)
        ax.plot(xs[shown], v_in[shown], ".", ms=2.2, color=color, alpha=0.45, zorder=2, ls="")
    else:
        ax.plot(xs, v, ".", ms=2.2, color=color, alpha=0.45, zorder=2, ls="")
    ax.plot([x], [v.mean()], marker, ms=ms, color=color, mfc=color if mfc is None else mfc, zorder=3, ls="")


def bare(ax, *, ylabel: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", c="#888", alpha=0.2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=7.5)


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
    """The pilot's arms, the stored checkpoints Part A scored, and the production reference they are read
    against, with one accessor per shape the cells need.
    """

    metrics: dict
    arrays: Mapping[str, np.ndarray]
    prod: dict
    prod_arrays: Mapping[str, np.ndarray]

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

    def part_a(self, cond: str) -> list[dict]:
        """Part A's records for an arm (the strips and the embedding components), in seed order; stored and pilot arms alike."""
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
        """The axis component of one embedding, per seed, on the embedding (`rows`) or the readout (`rows_readout`)."""
        return np.array([r[table][word] for r in self.part_a(cond) if r[table] is not None], float)

    def color_abs(self, cond: str, table: str = "rows") -> np.ndarray:
        """Per seed, the mean |component| over the color embeddings of one table."""
        out = []
        for r in self.part_a(cond):
            if r[table] is None:
                continue
            out.append(np.mean([abs(v) for w, v in r[table].items() if w not in SYNTAX]))
        return np.array(out, float)

    def red_mean(self, cond: str, table: str = "rows") -> np.ndarray:
        """Per seed, the mean signed component over the red color embeddings (redness at least the red dose)."""
        words = [w for w, r in zip(ex.PALETTE, ex.REDNESS, strict=True) if r >= ex.ex223.RED_DOSE]
        return np.array([np.mean([r[table][w] for w in words]) for r in self.part_a(cond) if r[table] is not None])

    def strip_acc(self, cond: str, strip: str, op: str, group: str, pos: int) -> np.ndarray:
        """Next-token accuracy at one predicting position, per seed, under one strip condition."""
        return np.array([r["strip"][strip][op]["acc"][group][pos] for r in self.part_a(cond)], float)

    def strip_p(self, cond: str, strip: str, op: str, group: str, pos: int) -> np.ndarray:
        return np.array([r["strip"][strip][op]["p_next"][group][pos] for r in self.part_a(cond)], float)

    def strip_proj(self, cond: str, strip: str, op: str, key: str, group: str) -> np.ndarray:
        """The full-position projection read on the stripped model: `acc` or the P(answer) `deficit`."""
        return np.array([r["strip"][strip][op]["projection"][key][group] for r in self.part_a(cond)], float)

    def tied(self, cond: str) -> bool:
        return all(r["tied"] for r in self.part_a(cond))


with tempfile.TemporaryDirectory() as _tmp:
    files = fetch([ex.METRICS_REF, ex.ARRAYS_REF, ex.EX223_METRICS_REF, ex.EX223_ARRAYS_REF], Path(_tmp))
    metrics = read_json(files[ex.METRICS_REF])
    arrays_ = read_npz(files[ex.ARRAYS_REF])
    prod = read_json(files[ex.EX223_METRICS_REF])
    prod_arrays = read_npz(files[ex.EX223_ARRAYS_REF])
if metrics is None or arrays_ is None:
    stop("No results published yet. Run the experiment (see `experiment.py`) and re-open this report.")
if prod is None or prod_arrays is None:
    stop("Ex-2.2.3's production results are not reachable.")
res: Results = Results(metrics, arrays_, prod, prod_arrays)

# Everything the tl;dr and Observations paragraph below quotes, computed once against `res`.
ans_acc = lambda c, s, g: res.strip_acc(c, s, MIX, g, 3).mean()  # noqa: E731
eq_acc = lambda c, s: res.strip_acc(c, s, MIX, "red", 2).mean()  # noqa: E731
row_component = {c: res.component(c, EQ).mean() for c in ROW_ARMS}
ops_component = {c: np.mean([res.component(c, w).mean() for w in OPS]) for c in ROW_ARMS}
head_component = {c: res.component(c, EQ, "rows_readout").mean() for c in ROW_ARMS if not res.tied(c)}
nl_component = {c: res.component(c, NL).mean() for c in ROW_ARMS}
em_floor = min(res.em(c, op).mean() for c in ARMS for op in OPS)
ml_stat = {c: res.stat(c, "m_line") for c in ARMS}
pilot_ml = [ml_stat[c].mean() for c in ARMS if c != ex.REFERENCE]
ref_ml = ml_stat[ex.REFERENCE]
proj_deficit = {c: res.deficit(c, MIX, PROJECTION) for c in ARMS}
opnd_deficit = res.deficit(ex.REFERENCE, MIX, OPERANDS).mean()
proj_red = {c: res.score(c, MIX, PROJECTION, "acc", "red") for c in ARMS}
bo_worst = proj_red["blocks-only"].max()
op_red = {c: [res.score(c, op, PROJECTION, "acc", "red").mean() for op in OPS] for c in ARMS}
red_component = {c: res.red_mean(c).mean() for c in ROW_ARMS}
alpha_op1 = {c: res.stat(c, "alpha_op1").mean() for c in ARMS}
retention_untied_line = res.stat("untied-line", "retention")
bol_ml = ml_stat["blocks-only-line"]
tail_count = {c: int((proj_deficit[c] > TAIL).sum()) for c in ("blocks-only-line", "untied-line")}
tail_ref = int((proj_deficit[ex.REFERENCE] > TAIL).sum())
count_words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
line_opnd_deficit = res.deficit("untied-line", MIX, OPERANDS)

rf"""
# Ex 2.2.7: a pilot of the syntax embeddings

/// tip |
<!-- tl;dr -->
A scouting run into why the embeddings of the op words and `=` hold part of the anchor axis, which is what a full-position projection pays for on the non-red lines.

The readout puts it there, to predict `=` after a red operand, and the tied table passes it to the embedding. Given a readout table of its own, the model keeps the component on that table and the syntax embeddings come mostly clean. Leaving the embeddings unanchored does not clean them.

Untying costs nothing on task or placement, and brings the non-red cost of the projection down toward the cost of the operand-only edit. The pilot proposes it for the handover, since it needs nothing from the grammar. Under the whole-line labeller a few seeds lose selectivity, so that labeller should go in with a check rather than by default.
///

/// note | A later finding
The controls here move the embeddings along a random direction. Under Adam the coordinate axes of the residual stream are a privileged basis, with the syntax tokens one-hot on one of them, and a random direction touches that axis, so a random-direction control may cost more than a quiet direction would. A reference direction compared with e₁ is better taken as another coordinate axis off the syntax axes; the [rotated-basis](/docs/m2/rotated-basis/report.py) experiment has the measurements.
///

## Observations

- **Where the component works.** On the stored `{ex.REFERENCE}` checkpoints, stripping it from the readout side costs the `=` prediction on the red lines ({eq_acc("recipe-short", "clean"):.2f} to {eq_acc("recipe-short", "output"):.2f}; {eq_acc("t00", "output"):.2f} on `t00`) and nothing else. Stripping it from the embedding side costs the answer instead ({ans_acc("recipe-short", "clean", "red"):.2f} to {ans_acc("recipe-short", "input", "red"):.2f} on the red lines, {ans_acc("recipe-short", "input", "nonred"):.2f} on the non-red). But turning the embeddings by the same angle in a random direction costs at least as much ({ans_acc("recipe-short", "input-control", "red"):.2f}), so on that side the blocks seem to have learned where the `=` embedding sits. The pilot arms repeat the pattern. On `untied`, stripping the readout side costs the `=` prediction ({eq_acc("untied", "output"):.2f}) and stripping the embedding side costs little ({ans_acc("untied", "input", "red"):.2f} on the answer). On `syntax-off-axis` and `untied-line`, no strip changes anything ([figures](#where-the-component-works)).
- **The embedding-component table.** The `=` embedding is at {row_component["recipe-short"]:.2f} on `{ex.REFERENCE}` and {row_component["t00"]:.2f} on `t00`; the op words are at {ops_component["recipe-short"]:.2f} and {ops_component["t00"]:.2f}. Anchoring the blocks only leaves the `=` embedding at {row_component["blocks-only"]:.2f}. Untying the readout brings the `=` embedding down to {row_component["untied"]:.2f}, and puts {head_component["untied"]:.2f} on the readout vector. Under the whole-line labeller the `⏎` embedding takes the axis as well ({nl_component["blocks-only-line"]:.2f} on `blocks-only-line`, {head_component["untied-line"]:.2f} on the readout of `untied-line`). `syntax-off-axis` holds every syntax embedding at zero by construction. The red color embeddings themselves sit at {red_component["recipe-short"]:.2f} on the reference, {red_component["rows-clean"]:.2f} on `syntax-off-axis` and {red_component["untied"]:.2f} on `untied`, but at {red_component["blocks-only"]:.2f} on `blocks-only` and {red_component["blocks-only-line"]:.2f} on `blocks-only-line`, where nothing pulls the embedding ([figure](#the-embedding-component-table)).
- **Task cost.** Held-out exact match is at least {em_floor:.3f} on every op of every arm ([table](#task-cost)).
- **Placement.** m_line runs {min(pilot_ml):.3f}–{max(pilot_ml):.3f} across the pilot arms, against {ref_ml.mean():.3f} <span class='range'>±{(ref_ml.max() - ref_ml.min()) / 2:.3f}</span> on `{ex.REFERENCE}`; `blocks-only` is the low end, and one seed of `blocks-only-line` did not place (m_line {bol_ml.min():.2f}, against {np.sort(bol_ml)[1]:.2f} for its next seed). ᾱ at op1, the containment read, is {alpha_op1["untied"]:.2f} on `untied` and {alpha_op1["untied-line"]:.2f} on `untied-line` against {alpha_op1[ex.REFERENCE]:.2f} on the reference, and retention on `untied-line` is {retention_untied_line.mean():.2f} <span class='range'>±{(retention_untied_line.max() - retention_untied_line.min()) / 2:.2f}</span> ([figure](#where-the-pull-lands)).
- **Suppression and selectivity.** Under the full-position projection the non-red `mix` deficit is {span2(proj_deficit["recipe-short"])} on `{ex.REFERENCE}`, where the operand-only edit gives {opnd_deficit:.3f}. On the other arms it is {span2(proj_deficit["untied"])} on `untied`, {span2(proj_deficit["rows-clean"])} on `syntax-off-axis` and {span2(proj_deficit["blocks-only"])} on `blocks-only`. Red-line `mix` accuracy under the projection is {proj_red["recipe-short"].mean():.2f} on the reference, {proj_red["untied"].mean():.2f} on `untied` and {proj_red["rows-clean"].mean():.2f} on `syntax-off-axis`. On `blocks-only` it is {proj_red["blocks-only"].mean():.2f}, with one seed at {bo_worst:.2f}, and across the six ops the two blocks-only arms sit at {min(op_red["blocks-only"] + op_red["blocks-only-line"]):.2f}–{max(op_red["blocks-only"] + op_red["blocks-only-line"]):.2f} where the reference sits at {min(op_red[ex.REFERENCE]):.2f}–{max(op_red[ex.REFERENCE]):.2f}. Under the whole-line labeller the non-red cost has a tail: {count_words[tail_count["untied-line"]]} of nine `untied-line` seeds and {count_words[tail_count["blocks-only-line"]]} of nine `blocks-only-line` seeds lose more than {TAIL:g} of the non-red `mix` lines under the projection, against {count_words[tail_ref]} of the twenty reference seeds, and `untied-line` pays {span2(line_opnd_deficit)} under the operand-only edit as well ([figure](#suppression-and-selectivity)).
"""

_arms = ex.ARMS
_stored = ex.STORED

rf"""
## Why, and what we ran

Ex-2.2.2's E8 found the anchor axis on the embeddings of the op words and `=`, at a few tenths in every anchored model, and ex-2.2.3 found more of it at the heavier adopted point. That component is what a full-position projection pays for on the non-red lines, and the operand-only edit routes around it only because this grammar tells us where the operands are. In M3 there is no operand position, so the M3-shaped operator is full-position and needs the syntax embeddings clean. The [tied-readout item](https://github.com/z0u/sca2/blob/main/todo/science/syntax-embeddings-carry-the-axis-via-tied-readout.md) and the design's Prep C each name a mechanism; this pilot runs both, with a third beside them.

nGPT ties the readout to the embedding, so one vector does two jobs: it is the residual stream's starting state when its token is read, and it is the logit of its token at every position that predicts it. Three mechanisms could put the axis on it.

1. **The direct pull.** The anchor term acts at every residual slice, the embedding included, and the labelled span covers the op word and `=`. Their slice-0 states are the embeddings themselves.
2. **The tied readout.** After a red operand the stream sits near e₁; the cheapest way to raise the logit of the token that follows is to lean that token's embedding the same way.
3. **The blocks.** A block that reads the component from the stream at a syntax position has a reason to keep it there, whichever table put it in.

**Part A** prices the component without training anything. On ex-2.2.3's stored checkpoints ({", ".join(f"`{c}`, {n} seeds" for c, n in _stored.items())}) we strip the axis component from the syntax embeddings on the *input* side only (the embedding, with the original table kept as the readout), the *output* side only, or both, and read the next-token accuracy at each position on the red and non-red probe lines, and the full-position projection's cost on the stripped model. A control turns the same embeddings by the same angle along a random direction off the axis, to read how much of an input-side cost is the turn itself. Whichever side loses the syntax predictions is where the component works.

**Part B** retrains ex-2.2.3's `{ex.REFERENCE}` (λ = {ex.ex223.SCORING_LAMBDA:g}, {ex.EPOCHS} epochs) three ways, each removing one mechanism, and reads Part A's table, E8's embedding-component table, and ex-2.2.3's placement and suppression statistics on every run. The first two also run under ex-2.2.6's whole-line labeller, which the handover proposes to adopt and which pulls the syntax positions by design.

| arm | seeds | readout | anchored slices | syntax held off axis | labeller | |
| --- | ---: | --- | --- | :---: | --- | --- |
{chr(10).join(f"| `{show(a.name)}` | {a.seeds} | {'tied' if a.tie else 'untied'} | {'blocks only' if a.slices else 'all'} | {'yes' if a.clean else 'no'} | {'whole line' if a.keying == 'line' else 'operands'} | {a.title} |" for a in _arms)}
| `{ex.REFERENCE}` | 20 | tied | all | no | operands | production, ex-2.2.3's recipe |

`syntax-off-axis` (stored as `rows-clean`) is the simplest fix that keeps the table shared: after every optimizer step, the same projection that keeps nGPT's embeddings at unit length also zeroes the axis component of every non-color embedding. `untied` gives the readout a table of its own, initialised as a copy of the embedding, so the anchor and anti-subspace terms keep acting on the embedding and the logits are free. `blocks-only` is Prep C: both anchoring terms skip slice 0.
"""

r"""
## Where the component works

Part A on the stored checkpoints. Each panel is one of the three informative next-token predictions on the `mix` probe lines: `=` from op2, the answer from `=`, and the newline from the answer. The x axis is the strip condition. A filled marker is the red lines (both operands' redness at least 0.8), a hollow one the non-red lines; the small dots are the seeds and the larger marker their mean.
"""


@memo
def strip_accuracy_stored(stack: dict[str, dict[str, np.ndarray]], alt_text: str, caption: str) -> str:
    @themed(name="strip-accuracy", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        strips = list(ex.STRIPS)
        x = np.arange(len(strips))
        fig, axes = plt.subplots(3, len(STORED), figsize=(5.6, 4.2), sharex=True, sharey="row")
        axes = cast(AxesGrid, axes)
        for col, c in enumerate(STORED):
            color = ink(c)
            for row, (pos, name) in enumerate(PRED.items()):
                ax = axes[row][col]
                for group, fill, dx in (("red", color, -0.14), ("nonred", "none", 0.14)):
                    for j, strip in enumerate(strips):
                        dots(ax, j + dx, stack[c][f"{strip}/{group}/{pos}"], color, mfc=fill, ms=3.5, jitter=0.06)
                bare(ax, ylabel=f"acc, predicting {name}" if col == 0 else None)
                ax.set_ylim(-0.05, 1.05)
            axes[0][col].set_title(show(c), fontsize=8, color=color)
            axes[2][col].set_xticks(x, [STRIP_LABEL[s] for s in strips], fontsize=6.5)
        return fig

    return plot()


strip_accuracy_stored(
    {
        c: {
            f"{strip}/{group}/{pos}": res.strip_acc(c, strip, MIX, group, pos)
            for strip in ex.STRIPS
            for group in ("red", "nonred")
            for pos in PRED
        }
        for c in STORED
    },
    """
        A grid of small panels, three rows by three columns. Rows are the three next-token predictions: the equals sign, the answer, and the newline. Columns are the stored arms recipe-short, t00, and control-short. In each panel the five strip conditions run along the x axis (clean, input, output, both, and the input control) and next-token accuracy along the y axis, with filled markers for the red lines and hollow ones for the non-red lines; each seed is a small dot beside the mean.
    """,
    r"""
        **Next-token accuracy on the `mix` probe lines under each strip, on the stored checkpoints.** Rows are the predictions of `=`, the answer, and the newline; columns are the stored arms. Filled markers are the seed means on the red lines, hollow on the non-red, with one small dot per seed. `input` strips the axis component from the syntax embeddings and keeps the original table as the readout; `output` the reverse; `both` strips it from the shared table; the control moves the embeddings the same distance along a random direction off the axis.
    """,
)

r"""
The same strips on the pilot arms. An arm whose syntax embeddings are already clean should show no difference across its strips; the untied arms split the two tables, so `input` and `output` strip different tables there.
"""


@memo
def strip_accuracy_arms(stack: dict[str, dict[str, np.ndarray]], alt_text: str, caption: str) -> str:
    @themed(name="strip-accuracy-arms", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        strips = list(ex.STRIPS)
        x = np.arange(len(strips))
        arms = [a.name for a in ex.ARMS]
        fig, axes = plt.subplots(3, len(arms), figsize=(8.4, 4.2), sharex=True, sharey="row")
        axes = cast(AxesGrid, axes)
        for col, c in enumerate(arms):
            color = ink(c)
            for row, (pos, name) in enumerate(PRED.items()):
                ax = axes[row][col]
                for group, fill, dx in (("red", color, -0.14), ("nonred", "none", 0.14)):
                    for j, strip in enumerate(strips):
                        dots(ax, j + dx, stack[c][f"{strip}/{group}/{pos}"], color, mfc=fill, ms=3.5, jitter=0.06)
                bare(ax, ylabel=f"acc, predicting {name}" if col == 0 else None)
                ax.set_ylim(-0.05, 1.05)
            axes[0][col].set_title(show(c), fontsize=8, color=color)
            axes[2][col].set_xticks(x, [STRIP_LABEL[s] for s in strips], fontsize=6.5)
        return fig

    return plot()


strip_accuracy_arms(
    {
        a.name: {
            f"{strip}/{group}/{pos}": res.strip_acc(a.name, strip, MIX, group, pos)
            for strip in ex.STRIPS
            for group in ("red", "nonred")
            for pos in PRED
        }
        for a in ex.ARMS
    },
    """
        A grid of small panels, three rows by five columns. Rows are the three next-token predictions: the equals sign, the answer, and the newline. Columns are the pilot arms blocks-only, untied, syntax-off-axis, blocks-only-line, and untied-line. In each panel the five strip conditions run along the x axis and next-token accuracy along the y axis, with filled markers for the red lines and hollow ones for the non-red lines; each seed is a small dot beside the mean.
    """,
    r"""
        **Next-token accuracy on the `mix` probe lines under each strip, on the pilot arms.** Same layout as the stored figure: rows are the predictions of `=`, the answer, and the newline; filled markers are the seed means on the red lines, hollow on the non-red, with one small dot per seed.
    """,
)

# %%
strip_rows = []
for strip_c in STORED + [a.name for a in ex.ARMS]:
    for strip_s in ex.STRIPS:
        strip_rows.append(
            [f"<code>{show(strip_c)}</code>" if strip_s == "clean" else "", STRIP_LABEL[strip_s].replace("\n", " ")]
            + [span2(res.strip_acc(strip_c, strip_s, MIX, g, 3), ".2f") for g in ("red", "nonred")]
            + [span2(res.strip_p(strip_c, strip_s, MIX, g, 3), ".2f") for g in ("red", "nonred")]
            + [span2(res.strip_proj(strip_c, strip_s, MIX, "acc", "red"), ".2f")]
            + [span2(res.strip_proj(strip_c, strip_s, MIX, "acc", "nonred"), ".3f")]
        )
strip_head = [
    "arm",
    "strip",
    "answer acc, red",
    "answer acc, non-red",
    "P(answer), red",
    "P(answer), non-red",
    "projection: red acc",
    "projection: non-red acc",
]
strip_caption = """
The answer prediction on the <code>mix</code> probe lines under each strip, per arm, and the full-position projection read on the stripped model: exact-match accuracy on the red lines (the removal read) and on the non-red lines (the selectivity read, as an accuracy rather than a deficit, since a strip moves the clean baseline too). Seed means with half the seed range. The pilot arms follow the stored ones; a fix that has already cleaned its syntax embeddings shows no difference across its strips.
"""
strip_n = len(ex.STRIPS)
table_html(strip_head, strip_rows, strip_caption, ref_rows=frozenset(range(0, 3 * strip_n)))

r"""
## The embedding-component table

Ex-2.2.2's E8, read on every arm: the axis component of each syntax embedding, with the mean absolute component over the 216 color embeddings beside it, and the mean signed component of the red color embeddings (redness at least 0.8), which is how far the red operands themselves sit along the axis before any block runs. nGPT's embeddings are unit vectors, so a component is a cosine. The untied arms carry a second table; its readout vectors are drawn hollow.
"""


@memo
def embedding_components(
    row_arms: list[str],
    tied: dict[str, bool],
    eq: dict[str, np.ndarray],
    eq_readout: dict[str, np.ndarray],
    nl: dict[str, np.ndarray],
    nl_readout: dict[str, np.ndarray],
    ops: dict[str, dict[str, np.ndarray]],
    ops_readout: dict[str, dict[str, np.ndarray]],
    color_abs: dict[str, np.ndarray],
    color_abs_readout: dict[str, np.ndarray],
    red_mean: dict[str, np.ndarray],
    red_mean_readout: dict[str, np.ndarray],
    alt_text: str,
    caption: str,
) -> str:
    @themed(name="embedding-components", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        x = np.arange(len(row_arms))
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        grey = light_dark("#999", "#777")
        ax.axhline(0, color=grey, lw=0.6)
        for i, c in enumerate(row_arms):
            color = ink(c)
            tables = ["rows"] + ([] if tied[c] else ["rows_readout"])
            for k, table in enumerate(tables):
                fill = color if table == "rows" else "none"
                off = -0.12 if len(tables) == 2 and k == 0 else (0.12 if len(tables) == 2 else 0.0)
                for marker, ms, vals in (
                    ("s", 4.5, eq if table == "rows" else eq_readout),
                    ("^", 4.5, nl if table == "rows" else nl_readout),
                ):
                    v = vals[c]
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
                    v = (ops if table == "rows" else ops_readout)[c][word]
                    ax.plot(i + off + (j - 2.5) * 0.03, v.mean(), "o", ms=2, color=color, mfc=fill)
                v = (color_abs if table == "rows" else color_abs_readout)[c]
                ax.plot(i + off, v.mean(), "D", ms=3, color=grey, mfc=grey if table == "rows" else "none")
                v = (red_mean if table == "rows" else red_mean_readout)[c]
                ax.plot(i + off, v.mean(), "v", ms=4, color=grey, mfc=grey if table == "rows" else "none")
        ax.set_xticks(x, [show(a) for a in row_arms], fontsize=7, rotation=25, ha="right")
        ax.set_ylabel("component on e₁", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", c="#888", alpha=0.2)
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="s", ls="", color=grey, ms=4.5, label="="),
            Line2D([], [], marker="^", ls="", color=grey, ms=4.5, label="⏎"),
            Line2D([], [], marker="o", ls="", color=grey, ms=2, label="op words"),
            Line2D([], [], marker="D", ls="", color=grey, ms=3, label="colors, mean |·|"),
            Line2D([], [], marker="v", ls="", color=grey, ms=4, label="red colors, mean"),
        ]
        ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return fig

    return plot()


embedding_components(
    ROW_ARMS,
    {c: res.tied(c) for c in ROW_ARMS},
    {c: res.component(c, EQ) for c in ROW_ARMS},
    {c: res.component(c, EQ, "rows_readout") for c in ROW_ARMS if not res.tied(c)},
    {c: res.component(c, NL) for c in ROW_ARMS},
    {c: res.component(c, NL, "rows_readout") for c in ROW_ARMS if not res.tied(c)},
    {c: {w: res.component(c, w) for w in OPS} for c in ROW_ARMS},
    {c: {w: res.component(c, w, "rows_readout") for w in OPS} for c in ROW_ARMS if not res.tied(c)},
    {c: res.color_abs(c) for c in ROW_ARMS},
    {c: res.color_abs(c, "rows_readout") for c in ROW_ARMS if not res.tied(c)},
    {c: res.red_mean(c) for c in ROW_ARMS},
    {c: res.red_mean(c, "rows_readout") for c in ROW_ARMS if not res.tied(c)},
    """
        A single panel. The arms run along the x axis, from the stored checkpoints on the left to the pilot arms on the right, and the axis component of an embedding along the y axis. At each arm there is a marker for the equals sign, one for the newline, a cluster for the six op words, a grey diamond for the mean absolute component of the color embeddings, and a grey triangle for the mean component of the red color embeddings; bars span the seeds. The untied arms show a second, hollow set for their readout table.
    """,
    r"""
        **Axis component per syntax embedding, by arm.** Each marker is the seed mean of one embedding's component on e₁ (a cosine, since the embeddings are unit vectors), with the seed range as a bar; the op words are drawn small, and in grey are the color embeddings' mean absolute component (diamond) and the red color embeddings' mean component (triangle). Hollow markers are the readout table of the untied arms. The stored arms are ex-2.2.3's; every pilot arm is Part B's.
    """,
)

# %%
component_rows = []
for row_c in ROW_ARMS:
    component_rows.append(
        [
            f"<code>{show(row_c)}</code>",
            "embedding",
            span2(res.component(row_c, EQ), ".2f"),
            span2(res.component(row_c, NL), ".2f"),
        ]
        + [span2(res.component(row_c, w), ".2f") for w in OPS]
        + [span2(res.color_abs(row_c), ".3f"), span2(res.red_mean(row_c), ".2f")]
    )
    if not res.tied(row_c):
        component_rows.append(
            [
                "",
                "readout",
                span2(res.component(row_c, EQ, "rows_readout"), ".2f"),
                span2(res.component(row_c, NL, "rows_readout"), ".2f"),
            ]
            + [span2(res.component(row_c, w, "rows_readout"), ".2f") for w in OPS]
            + [span2(res.color_abs(row_c, "rows_readout"), ".3f"), span2(res.red_mean(row_c, "rows_readout"), ".2f")]
        )
component_head = (
    ["arm", "table", "=", "⏎"] + [f"<code>{w}</code>" for w in OPS] + ["colors, mean |·|", "red colors, mean"]
)
component_caption = "The embedding-component table as numbers: the signed axis component of each syntax embedding, per arm and table, seed mean with half the seed range; the last two columns are the mean absolute component over the color embeddings and the mean signed component over the red color embeddings."
table_html(component_head, component_rows, component_caption, ref_rows=frozenset(range(len(STORED))))

r"""
## Task cost

Exact match on the held-out pairs of each op. The reference is production's twenty seeds; the pilot arms have nine each.
"""


@memo
def task_cost(em: dict[str, dict[str, np.ndarray]], alt_text: str, caption: str) -> str:
    @themed(name="task-cost", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        markers = ["o", "s", "^", "D", "v", "P"]
        fig, ax = plt.subplots(figsize=(5.6, 2.4))
        for i, c in enumerate(ARMS):
            color = ink(c)
            for j, op in enumerate(OPS):
                dots(ax, i + (j - 2.5) * 0.11, em[c][op], color, marker=markers[j], ms=3.5, jitter=0.03)
        ax.set_xticks(np.arange(len(ARMS)), [show(a) for a in ARMS], fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0.98, 1.002)
        bare(ax, ylabel="held-out exact match")
        from matplotlib.lines import Line2D

        grey = light_dark("#999", "#777")
        handles = [
            Line2D([], [], marker=m, ls="", color=grey, ms=3.5, label=op) for m, op in zip(markers, OPS, strict=True)
        ]
        ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return fig

    return plot()


task_cost(
    {c: {op: res.em(c, op) for op in OPS} for c in ARMS},
    """
        A single panel. The arms run along the x axis, the production reference last, and held-out exact match along the y axis, from 0.98 to 1. At each arm there is one marker per op, drawn in the arm's colour, with the seeds as small dots beside each mean.
    """,
    r"""
        **Held-out exact match per op, by arm.** One marker shape per op, the seed mean, with one small dot per seed; the y axis starts at 0.98. The reference is ex-2.2.3's production recipe.
    """,
)

# %%
em_rows = [
    [f"<code>{show(c)}</code>", f"{len(res.runs(c))}"]
    + [span2(res.em(c, op), ".3f") for op in OPS]
    + [f"{res.em(c, MIX).mean() - res.em(ex.REFERENCE, MIX).mean():+.3f}"]
    for c in ARMS
]
em_caption = "Held-out exact match per op, seed mean with half the seed range; the last column is the `mix` gap from the production reference. Ex-2.2.3's task gate was a `mix` gap within 0.02 of its own control."
table_html(
    ["arm", "seeds"] + [f"<code>{op}</code>" for op in OPS] + ["Δ mix"],
    em_rows,
    em_caption,
    ref_rows=frozenset({len(ARMS) - 1}),
)

r"""
## Where the pull lands

Ex-2.2.3's placement statistics on the `mix` probe lines, per arm. The line arms' m_line is under their own labeller's weighting, as in ex-2.2.6.
"""

PLACEMENT_STATS = [
    ("m_line", "m_line"),
    ("alpha_op1", "ᾱ op1"),
    ("lead_emb", "lead (emb)"),
    ("contrast", "contrast"),
    ("r2_sim", "r² sim"),
    ("latch_pi", "latch π"),
    ("retention", "retention"),
]


@memo
def placement(stats: dict[str, dict[str, np.ndarray]], alt_text: str, caption: str) -> str:
    @themed(name="placement", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(8.4, 3.6), sharex=True)
        axes = cast(AxesGrid, axes)
        flat = [axes[r][c] for r in range(2) for c in range(4)]
        for ax, (key, label) in zip(flat, PLACEMENT_STATS, strict=False):
            pooled = np.concatenate([stats[c][key] for c in ARMS])
            lo, hi = np.percentile(pooled, [3, 97])
            pad = 0.15 * (hi - lo) + 1e-6
            for i, c in enumerate(ARMS):
                dots(ax, i, stats[c][key], ink(c), ms=4, clip=(lo - pad, hi + pad))
            ax.set_ylim(lo - 1.6 * pad, hi + 1.6 * pad)
            bare(ax, ylabel=label)
            ax.set_xticks(np.arange(len(ARMS)), [show(a) for a in ARMS], fontsize=6.5, rotation=35, ha="right")
            ax.tick_params(labelbottom=True)
        flat[-1].axis("off")
        return fig

    return plot()


placement(
    {c: {key: res.stat(c, key) for key, _ in PLACEMENT_STATS} for c in ARMS},
    """
        A grid of small panels, two rows by four columns, one per placement statistic: m_line, alpha at op1, lead at the embedding, contrast, r squared of the similarity grading, latch pi, and retention; the last cell is empty. In each panel the arms run along the x axis with the production reference last, and the statistic along the y axis, with one small dot per seed and a larger marker at the seed mean, in the arm's colour; a seed beyond a panel's range is a hollow triangle at its edge.
    """,
    r"""
        **Placement on the `mix` probe lines, by arm.** One panel per statistic of the table below; the larger marker is the seed mean and the small dots are the seeds. Each panel spans the bulk of the seeds, and a seed beyond it is drawn as a hollow triangle at the edge (one `blocks-only-line` seed did not place, and sits off most panels; the table has it). The reference is ex-2.2.3's production recipe.
    """,
)

# %%
placement_rows = []
for placement_c in ARMS:
    placement_rows.append(
        [
            f"<code>{show(placement_c)}</code>",
            span2(res.stat(placement_c, "m_line")),
            span2(res.stat(placement_c, "alpha_op1")),
            span2(res.stat(placement_c, "lead_emb"), ".2f"),
            span2(res.stat(placement_c, "contrast"), ".2f"),
            span2(res.stat(placement_c, "r2_sim")),
            span2(res.stat(placement_c, "latch_pi"), ".2f"),
            span2(res.stat(placement_c, "retention"), ".2f"),
        ]
    )
placement_head = ["arm", "m_line", "ᾱ op1", "lead (emb)", "contrast", "r² sim", "latch π", "retention"]
placement_caption = """
Placement on the <code>mix</code> probe lines, seed means with half the seed range. m_line is the per-line margin under the arm's own labeller; ᾱ op1 the mean alignment at op1 over every slice and color (ex-2.2.3's containment read); lead the G1 group's softmin weight on op1 at the embedding; contrast the deep-slice op2 weight of G2 minus G1; r² sim the grading of the op1 response against the similarity target; latch π the larger of the non-red group's deep-slice weights on the op word and on <code>=</code>; retention the final m_line over its running peak. On <code>blocks-only</code> the embedding slice is not pulled, so its lead is what the blocks' pull leaves there.
"""
table_html(placement_head, placement_rows, placement_caption, ref_rows=frozenset({len(ARMS) - 1}))

r"""
## Suppression and selectivity

Ex-2.2.3's H4 statistics per arm: red-line accuracy under the full-position `projection` on each op (the removal read, lower is more complete), and the non-red `mix` deficit under `projection` and under the operand-only edit (the selectivity read). The question for each fix is whether the full-position deficit comes down to the operand-only one.
"""


@memo
def suppression(
    red_acc: dict[str, dict[str, np.ndarray]],
    proj_def: dict[str, np.ndarray],
    opnd_def: dict[str, np.ndarray],
    alt_text: str,
    caption: str,
) -> str:
    @themed(name="suppression", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        markers = ["o", "s", "^", "D", "v", "P"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.6), width_ratios=[1.3, 1])
        for i, c in enumerate(ARMS):
            color = ink(c)
            for j, op in enumerate(OPS):
                dots(ax1, i + (j - 2.5) * 0.11, red_acc[c][op], color, marker=markers[j], ms=3.5, jitter=0.03)
            dots(ax2, i - 0.14, proj_def[c], color, ms=4.5, jitter=0.06, clip=(-0.02, 0.12))
            dots(ax2, i + 0.14, opnd_def[c], color, mfc="none", ms=4.5, jitter=0.06, clip=(-0.02, 0.12))
        for ax, label in ((ax1, "red acc, projection"), (ax2, "non-red mix deficit")):
            ax.set_xticks(np.arange(len(ARMS)), [show(a) for a in ARMS], fontsize=7, rotation=25, ha="right")
            bare(ax, ylabel=label)
        ax1.set_ylim(-0.05, 1.05)
        ax2.set_ylim(-0.025, 0.125)
        from matplotlib.lines import Line2D

        grey = light_dark("#999", "#777")
        handles = [
            Line2D([], [], marker=m, ls="", color=grey, ms=3.5, label=op) for m, op in zip(markers, OPS, strict=True)
        ]
        ax1.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper right")
        handles = [
            Line2D([], [], marker="o", ls="", color=grey, ms=4.5, label="projection"),
            Line2D([], [], marker="o", ls="", color=grey, mfc="none", ms=4.5, label="operands"),
        ]
        ax2.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper right")
        return fig

    return plot()


suppression(
    {c: {op: res.score(c, op, PROJECTION, "acc", "red") for op in OPS} for c in ARMS},
    {c: res.deficit(c, MIX, PROJECTION) for c in ARMS},
    {c: res.deficit(c, MIX, OPERANDS) for c in ARMS},
    """
        Two panels side by side. Left: red-line accuracy under the full-position projection, with the arms along the x axis, the production reference last, and one marker per op at each arm; lower is a more complete removal. Right: the non-red mix accuracy deficit, filled markers under the projection and hollow ones under the operand-only edit, with seeds above the panel drawn as hollow triangles at its top edge. In both, each seed is a small dot beside the mean, in the arm's colour.
    """,
    r"""
        **Suppression and selectivity, by arm.** Left, exact-match accuracy on the red lines under the full-position projection, one marker shape per op (the removal read; lower is more complete). Right, the non-red `mix` deficit under the projection (filled) and under the operand-only edit (hollow); the panel stops at 0.12, and a seed above it is a hollow triangle at the top edge (the table has the values). Larger markers are seed means, small dots the seeds.
    """,
)

# %%
suppression_rows = []
for supp_c in ARMS:
    suppression_rows.append(
        [f"<code>{show(supp_c)}</code>"]
        + [span2(res.score(supp_c, op, PROJECTION, "acc", "red"), ".2f") for op in OPS]
        + [span2(res.deficit(supp_c, MIX, PROJECTION), ".3f"), span2(res.deficit(supp_c, MIX, OPERANDS), ".3f")]
        + [span2(res.score(supp_c, MIX, PROJECTION, "acc", "nonred"), ".3f")]
    )
suppression_head = (
    ["arm"]
    + [f"<code>{op}</code>" for op in OPS]
    + ["deficit, projection", "deficit, operands", "non-red acc, projection"]
)
suppression_caption = f"""
Exact-match accuracy under the full-position <code>projection</code> on the red lines (dose ≥ {ex.ex223.RED_DOSE:g}) of each op, then on <code>mix</code>: the non-red (dose ≤ {ex.ex223.NONRED_DOSE:g}) accuracy deficit under the full-position projection and under the operand-only edit, and the non-red accuracy under the projection. Ex-2.2.3's gates were red accuracy at most {ex.ex223.RED_ACC_GATE:g} on every op and a deficit at most {ex.ex223.NONRED_DEFICIT_GATE:g}.
"""
table_html(suppression_head, suppression_rows, suppression_caption, ref_rows=frozenset({len(ARMS) - 1}))

r"""
## What we make of it

**Which mechanism.** Each of the three arms removed one candidate, and two of the three results point the same way. Leaving the embedding out of the anchor did not clean the syntax embeddings, so the direct pull at slice 0 was not the cause. Giving the readout a table of its own moved the component onto that table, at about the size it has on the reference, and left the syntax embeddings mostly clean.

So the logit path is what places it. Part A says the same from the stored checkpoints: after a red op2, the `=` prediction leans on the component on the readout side, and that is the only thing the readout side does. The untied arm shows it from the other side: stripping the readout table costs the same prediction, and stripping the embedding costs almost nothing. This is the mechanism as written in the [tied-readout item](https://github.com/z0u/sca2/blob/main/todo/science/syntax-embeddings-carry-the-axis-via-tied-readout.md).

The embedding-side strips show something else: the blocks learn to read the embeddings where the logit path leaves them. The random-direction control costs as much, and on `t00`, where the `=` embedding sits about 70° off a clean one, a stripped model is a different model. So the component is doing work, and a stronger anti-subspace term on those embeddings would pull against the logits.

**Prep C.** Of the three hypotheses in the design, (a) holds in the blocks: the contrast and the similarity grading are at the values of the reference, with m_line a little lower. (b) holds, since the color embeddings still lead the pull at slice 0 even though nothing pulls them there. (c) does not hold, because the syntax embeddings keep the axis.

The cost Prep C did not anticipate is completeness. On both blocks-only arms the full-position projection leaves more red lines with their answer, on every op, and on one seed it leaves most of them. The embedding-component table says why: the projection removes the e₁ component of the stream at every slice and nothing else, so anything that survives it sits off the axis.

In the reference, the anchor at slice 0 and the anti-subspace term together put the redness of a red operand onto the axis before any block runs, with the red color embeddings at about 0.8 on e₁. With nothing pulling the embedding they sit at about half that, so half of the redness enters the stream off the axis.

The blocks are as well aligned as the reference's, since the contrast and the similarity grading match. They read the redness from the stream and write it onto the axis, and the projection removes what they wrote. The off-axis half is still in the residual stream at the last slice, where the readout can use it. So "the states at the later slices are aligned" describes what the blocks add, and how well that part is aligned does not decide whether the projection is complete.

**What transfers.** `syntax-off-axis` chooses its embeddings by token class: the non-color embeddings are the ones held off the axis. That is a rule about which vocabulary entries may hold the axis, the same kind of built-in position knowledge that the mellowmax pooling was adopted to avoid, and natural language has no such class. So the constraint as run here is a fix for this grammar, and what it measures is the ceiling: what a fully clean shared table buys.

The other two arms need nothing from the grammar. Untying is available on any model, as a copy of the table, and many models already ship untied. `blocks-only` keeps the table shared but pays in completeness, above.

A tied-table variant we did not run would hold every embedding off the axis, so the embedding has no axis component at all. It would need no token class, but the redness would then enter off the axis by construction, the completeness cost again. That question belongs with the operator pass, where the reflection and the shaped forms are also up for decision.

**The fix to carry.** On nine seeds, `untied` and `syntax-off-axis` match each other and the reference on task, placement and removal. Both have a lower non-red cost under the projection on most seeds, with a two-seed tail at the level of the upper seeds of the reference. The pilot proposes the untied readout for the handover, with the full-position projection read beside the operand-only edit, and `syntax-off-axis` as the in-grammar ceiling it should match.

The concern from the discussion in ex-2.2.3 still stands, that a method needing an untied readout is a harder sell where the tables are tied; the all-embeddings-off-axis variant above is the tied option to test if that becomes the target. Two things to watch on the untied arms: ᾱ at op1, the containment read, is higher than on the reference, and the `=` embedding is lower rather than at zero.

**The whole-line labeller.** Under it the `⏎` embedding takes the axis the same way, since it follows the newly pulled answer, and on the untied arm it lands on the readout table like the rest.

The larger finding is the tail. On both line arms a few seeds lose a large share of the non-red lines under the projection, `untied-line` pays under the operand-only edit as well, and the margin on `untied-line` drifts down over training (the retention read). One `blocks-only-line` seed did not place at all.

There is no plain-recipe line arm at nine seeds here, and ex-2.2.6 has two or three, so the pilot cannot say whether the tail comes from the labeller alone or from pairing it with a fix. The handover should read the selectivity of the line labeller at more seeds before adopting it.

**What it changes.** The handover can adopt the untied readout, and use the full-position projection alongside the operand-only edit as its removal operator, which is what the M3-shaped operator needs. The operator pass in the D2.2 design chooses between plain projection and the shaped forms on that footing, and the whole-line labeller goes in with a selectivity check rather than by default.

## Method notes

- Part A's strips edit the tables of a loaded checkpoint and nothing else; `input` and `output` give the tied model a readout table of its own for the scoring pass (`NGPT.with_tables`), so the two sides can differ. The stripped embeddings are re-normalized to unit length, as nGPT keeps them.
- The control strip replaces the axis component of each syntax embedding with one of the same size along a random unit direction orthogonal to e₁ (one fixed draw). The embedding then turns by about the same angle as under `input` and ends up off the axis as well.
- Part B's arms share ex-2.2.3's corpus, eval sets and probe lines (the same seeds through its `prepare_corpus`); the line arms train against ex-2.2.6's line-keyed probe table.
- `syntax-off-axis` (stored as `rows-clean`) applies `sca.anchoring.clean_embedding_rows` after every optimizer step, to every non-color embedding but the pad; `blocks-only` passes `anchor_slices=(1, 2, 3, 4)` to `train_anchored`, so both the anchor and the anti-subspace term skip the embedding; `untied` sets `tie_embeddings=False` on the model config, and its checkpoints carry the second table.
- The eval and score tasks are ex-2.2.3's, unchanged; on the untied arms the eval contract reads logits through the readout table and `ablate_weights` projects both tables.
- This is a pilot: five arms with nine seeds each (the first run had three, or two on the line arms; the rest were added on request, and the earlier seeds are memoized), Part A on five seeds of each frozen stored arm and on all twenty of the reference, and production's twenty for the reference elsewhere. Nothing here is gated, and nothing in it should be quoted as a result.
"""
