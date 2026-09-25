# title: How far the answer moves: an RGB-distance readout beside expected exact match

r"""
# How far the answer moves: an RGB-distance readout beside expected exact match

/// tip |
<!-- tl;dr -->
We re-scored ex-2.2.11's 54 checkpoints with a distance on the color grid beside expected exact match, to see where the answers go when the anchored direction is removed. On `handover` the removed answers are near misses, two to three grid steps from the right color and well short of a random guess, and the non-red lines move less than the control's own lines do under the same operator.
///

## Observations

- [The removal lines, by distance](#the-removal-lines-by-distance): under `projection`, the greedy guess on `handover` moves from the floor (0 to 0.5 steps, a perfect answer) to 1.8 steps on `mix` and 3.4 on `value-hsv`, between 44% and 77% of the way to chance. The control stays at the floor. The mean, median, and expected distances agree with the greedy one within 0.3 steps.
- [Kept and lost](#kept-and-lost): the operator loses 77% to 99% of the removal lines. The kept lines stay at the floor; the lost lines sit 1.8 (`mix`) to 3.7 (`value-hsv`) steps from the answer, and reach chance only on `value-hsv`.
- [The lost answers on the cube](#the-lost-answers-on-the-cube): on `mix`, nine in ten lost answers are one or two steps off. On `hue-hsv` the guesses keep their saturation and move in hue; on `darken` they land darker than the answer.
- [A counterfactual answer](#a-counterfactual-answer): the lost answers are not the op applied to a de-reddened operand. The nearest such answer is a little closer to the guess than the true one on most ops, but the guess lands in its box on under a quarter of the lines on all ops but `lighten`.
- [The non-red lines](#the-non-red-lines): on `handover` the exact match deficit stays under 0.01 and the mean of the distribution moves under 0.03 steps on every op. The control's own lines move as much or more under the same operator.
- [Between seeds](#between-seeds): the seed spread of the distances is 6% to 20% of the mean, against 33% to 127% for the kept share of exact match.

## Scope

This is a re-score of stored checkpoints, planned in the [backlog item](/todo/science/rgb-distance-readout-beside-exact-match.md). There is no training, no gates, and no verdicts.

Every removal and selectivity number in D2.2 is *expected exact match*: the probability mass the model puts on the colors that a line's answer can be. It gives no partial credit, so a guess one grid step from the answer scores the same as a guess of black.

So when `handover` keeps 24% of exact match on the `hue-hsv` removal lines, that could mean a quarter of the lines answered perfectly and the rest sent anywhere, or every line nudged one step over. Likewise, a non-red deficit of 0.01 could be one line in a hundred lost outright, or the mass of every line leaning a little.

We add a distance on the color grid, in grid steps. One step is the spacing between the six levels of each channel, and the far corner of the cube is 5√3 ≈ 8.7 steps from black. The target of a line is its *raw answer* (the unrounded value of the op), which is also the expected value of the answer distribution under stochastic rounding.[^stochastic]

Four quantities are measured from the raw answer. Three are central points of the model's distribution over the 216 grid colors: the *greedy* guess (the most likely color), the *mean*, and the channel-wise *median*. The fourth is the *expected* distance of a color drawn from the distribution.

The greedy and expected distances are the ones to compare with exact match. The mean is the one that moves when mass leans toward a neighbor while the guess stays the same.

[^stochastic]: Stochastic rounding picks the grid level above or below a value at random, weighted so that the average equals the unrounded value.

Every distance comes with two references. The *floor* is the distance from the raw answer to the best possible grid answer (the mode of the answer distribution): zero when the answer lands on the grid, and up to half a step per channel otherwise. A distance at the floor is a perfect answer.

*Chance* is the mean distance of a color drawn uniformly from the grid: about 3 steps for a target near the middle of the cube, and more for one near a corner. A distance at chance means the answer no longer depends on the line.

The distances and exact match are measured under the same three passes as ex-2.2.11:
(a) the clean forward pass,
(b) `projection`, with e₁ removed at every slice and position, and
(c) `operands`, with e₁ removed at the operand positions only.
Each is read on the line groups of ex-2.2.11: the removal lines (a red operand whose hue the op carries into the answer), the non-red lines, and the rest. As in the ex-2.2.11 report, the shaped operator is left out.

## The runs

The runs are the 54 final checkpoints of ex-2.2.11: the un-anchored control (5 seeds), `handover` (20), and two arms that each undo one of the changes in `handover`, `handover-slot` (20) and `handover-tied` (9). The probe lines are the ones ex-2.2.11 scored: 5,832 per op, and twice that for the three order-sensitive ops.

The two scorers agree: the re-score reproduces ex-2.2.11's kept share of expected exact match on every op (24% on `hue-hsv`, 12% to 18% on the next four).
"""

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import experiment as ex
from mini.lit import memo, read_npz, stop
from mini.store import Artifact, project_store
from mini.vis import figure_html, light_dark, themed
from sca.answer_distance import GRID

# The conditions in the order the columns of each figure take: the control, then the arms, then the candidate.
ORDER: tuple[str, ...] = ("control", "handover-tied", "handover-slot", "handover")
ANCHORED = ORDER[1:]
OPS = list(ex.OP_NAMES)
CENTER = np.full(3, 2.5)


def fetch(refs: Sequence[str], into: Path) -> dict[str, tuple[Artifact, Path] | None]:
    """Each ref's artifact and published file under *into*, or None before it exists: one `get_refs` and one `get_many` for the lot."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | {r: (a, p) for (r, a), p in zip(have.items(), paths, strict=True)}


@dataclass
class Results:
    metrics: dict
    arrays: Mapping[str, np.ndarray]
    sources: tuple[Artifact | None, ...]

    def __memo_key__(self) -> list[str | None]:
        """The figure cache keys a `Results` by the hashes of its sources rather than digesting every array it holds."""
        return [a.sha256 if a is not None else None for a in self.sources]

    def runs(self, cond: str) -> list[dict]:
        return [r for r in self.metrics["runs"] if r["cond"] == cond]

    def idx(self, axis: str, name: str) -> int:
        return list(self.metrics[axis]).index(name)

    def stat(self, cond: str, op: str, pass_: str, group: str, stat: str) -> np.ndarray:
        """(run,): one group mean of one statistic under one pass, per run of *cond*."""
        i = (self.idx("ops", op), self.idx("passes", pass_), self.idx("groups", group), self.idx("stats", stat))
        return np.array([self.arrays[f"{r['label']}/summary"][i] for r in self.runs(cond)])

    def kept(self, cond: str, op: str, group: str = "removal", pass_: str = "projection") -> np.ndarray:
        """(run,): the share of the clean expected exact match kept under a pass, as ex-2.2.11 reads it."""
        return self.stat(cond, op, pass_, group, "eem") / self.stat(cond, op, "clean", group, "eem")

    def split(self, cond: str, op: str, pass_: str, group: str, split: str, stat: str) -> np.ndarray:
        """(run,): a split's line count or mean distance under one pass, per run of *cond*."""
        i = (
            self.idx("ops", op),
            self.idx("passes", pass_),
            self.idx("groups", group),
            self.idx("splits", split),
            self.idx("split_stats", stat),
        )
        return np.array([self.arrays[f"{r['label']}/split"][i] for r in self.runs(cond)])

    def hist(self, cond: str, op: str, pass_: str, group: str) -> np.ndarray:
        """(run, bin): the histogram of the greedy distance rounded to a step."""
        i = (self.idx("ops", op), self.idx("passes", pass_), self.idx("groups", group))
        return np.array([self.arrays[f"{r['label']}/hist"][i] for r in self.runs(cond)])

    def lines(self, op: str, name: str) -> np.ndarray:
        return self.arrays[f"lines/{op}/{name}"]

    def group(self, op: str, group: str) -> np.ndarray:
        return self.lines(op, f"group/{group}")

    def reference(self, op: str, group: str, name: str) -> float:
        """The mean over a group's lines of a per-line reference: `floor_mode`, `floor_draw`, `chance`, or
        `center` (the distance of the raw answer from the middle of the cube, where a uniform distribution's mean sits).
        """
        m = self.group(op, group)
        v = np.linalg.norm(self.lines(op, "raw") - CENTER, axis=1) if name == "center" else self.lines(op, name)
        return float(v[m].mean())

    def per_line(self, label: str, op: str, pass_: str, name: str) -> np.ndarray:
        return self.arrays[f"{label}/{op}/{pass_}/{name}"]

    def n_lines(self, op: str, group: str) -> int:
        return self.metrics["n_lines"][op][group]


def load_results() -> Results | None:
    with tempfile.TemporaryDirectory() as tmp:
        got = fetch([ex.METRICS_REF, ex.ARRAYS_REF], Path(tmp))
        gm, ga = got[ex.METRICS_REF], got[ex.ARRAYS_REF]
        arrays = None if ga is None else read_npz(ga[1])
        if gm is None or ga is None or arrays is None:
            return None
        return Results(json.loads(gm[1].read_text()), arrays, (gm[0], ga[0]))


# --- Figure style --------------------------------------------------------------------------------


def ink(cond: str) -> str:
    """One ink per condition, as ex-2.2.9 draws them."""
    inks = {
        "control": ("#6b6b6b", "#b0b0b0"),
        "handover": ("#c0392b", "#ff8a76"),
        "handover-slot": ("#2b6cb0", "#7fb3ff"),
        "handover-tied": ("#7b3fa0", "#cfa3ff"),
    }
    return light_dark(*inks[cond])


def marker(cond: str) -> str:
    return {"control": "s", "handover": "o", "handover-slot": "^", "handover-tied": "D"}[cond]


def dots(ax: Axes, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 5.0, width: float = 0.05, label=None):
    """One column of per-seed dots with the seed mean drawn on top, in the condition's ink and marker. A run
    with no line in the group (NaN) draws nothing.
    """
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return
    color, m = ink(cond), marker(cond)
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def fig_legend(fig: plt.Figure, ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)


def offsets(n: int) -> np.ndarray:
    """The x offsets of *n* condition columns at one tick: 0.15 apart, up to a spread of ±0.3."""
    half = min(0.075 * (n - 1), 0.3)
    return np.linspace(-half, half, n) if n > 1 else np.zeros(1)


def reference_ticks(ax: Axes, xs: np.ndarray, values: Sequence[float], *, ls: str, label: str | None = None) -> None:
    """A short horizontal rule at each op's column for a per-op reference level (the floor, chance, the center)."""
    color = light_dark("#333", "#ddd")
    for i, (x, v) in enumerate(zip(xs, values, strict=True)):
        ax.plot([x - 0.42, x + 0.42], [v, v], ls=ls, color=color, lw=0.8, zorder=1, label=label if i == 0 else None)


def columns_panel(ax: Axes, values, ticks: Sequence[str], *, rng, conds=ORDER, legend: bool = False) -> None:
    """Seed-dot columns per condition at each tick. *values(cond, tick)* gives a (run,) array."""
    xs = np.arange(len(ticks))
    off = offsets(len(conds))
    for x, t in zip(xs, ticks, strict=True):
        for o, c in zip(off, conds, strict=True):
            dots(ax, x + o, values(c, t), c, rng=rng, width=0.03, label=c if legend and x == 0 else None)
    ax.set_xticks(xs, ticks, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    """An authored result table in the shared report style; the first column is text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def span2(v: np.ndarray, digits: int = 2) -> str:
    """A seed mean with its range, as `0.51 (0.4–0.6)`."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "—"
    return f"{v.mean():.{digits}f} ({v.min():.{max(digits - 1, 1)}f}–{v.max():.{max(digits - 1, 1)}f})"


# %%
res = load_results()
if res is None:
    stop("The re-score has not been published yet: `bin/mini run docs/m2/answer-distance/experiment.py --app local`.")
assert res is not None

r"""
## The removal lines, by distance

On the removal lines under `projection`, the answers behind the exact match that `handover` keeps are mostly near misses, rather than right answers among wrong ones. The figure puts the distance beside the kept share, per op.

The clean pass sits at the floor on every op and condition (the table below the figure gives it), so any distance above the floor under `projection` is caused by the operator.
"""


def removal_figure(res: Results) -> str:
    kept = {(c, op): res.kept(c, op) for c in ORDER for op in OPS}
    greedy = {(c, op): res.stat(c, op, "projection", "removal", "greedy") for c in ORDER for op in OPS}
    mean = {(c, op): res.stat(c, op, "projection", "removal", "mean") for c in ORDER for op in OPS}
    refs = {k: [res.reference(op, "removal", k) for op in OPS] for k in ("floor_mode", "chance", "center")}
    return removal_draw(kept, greedy, mean, refs)


@memo
def removal_draw(kept: dict, greedy: dict, mean: dict, refs: dict) -> str:
    @themed(
        name="removal-distance",
        alt_text="""
            Under projection, handover's greedy distance on the removal lines rises from the floor to about halfway to chance on every op, while the control stays at the floor.
        """,
        caption=f"""
            **The removal lines under `projection`, per op and condition: exact match kept, and how far the answer moves.** Top: the share of the clean expected exact match kept, as ex-2.2.11 reads it, dashed at its {ex.ex2211.RED_KEPT_GATE:.0%} gate. Middle: the distance in grid steps of the greedy guess from the raw answer; the dotted rule at each op is the floor (the best grid answer's distance) and the dashed rule is chance (a uniform draw from the grid). Bottom: the distance of the mean of the model's distribution from the raw answer; the dashed rule is the center of the cube, where a uniform distribution's mean sits. Each small dot is one seed, the larger mark the seed mean, and the thin bar the seed range.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.6), layout="constrained", sharex=True)
        top, mid, bot = axes
        rng = np.random.default_rng(2)
        xs = np.arange(len(OPS))
        columns_panel(top, lambda c, op: kept[c, op], OPS, rng=rng, legend=True)
        columns_panel(mid, lambda c, op: greedy[c, op], OPS, rng=rng)
        columns_panel(bot, lambda c, op: mean[c, op], OPS, rng=rng)
        reference_ticks(mid, xs, refs["floor_mode"], ls=":", label="floor")
        reference_ticks(mid, xs, refs["chance"], ls="--", label="chance")
        reference_ticks(bot, xs, refs["floor_mode"], ls=":")
        reference_ticks(bot, xs, refs["center"], ls="--", label="cube center")
        top.axhline(ex.ex2211.RED_KEPT_GATE, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
        top.set_ylim(-0.02, 1.05)
        for ax in (mid, bot):
            ax.set_ylim(-0.1, max(ax.get_ylim()[1], 4.0))
        top.set_ylabel("exact match kept")
        mid.set_ylabel("greedy distance (steps)")
        bot.set_ylabel("mean-of-distribution\ndistance (steps)")
        fig_legend(fig, top)
        mid.legend(loc="upper right", fontsize=7, frameon=False, ncols=2)
        bot.legend(loc="upper right", fontsize=7, frameon=False)
        return fig

    return _plot()


removal_figure(res)

"""
The table gives the same measurements on `handover`, adds the clean and `operands` passes, and adds the hit rate (the share of lines whose greedy guess is one of the line's possible answers).
"""


def removal_table(res: Results) -> str:
    head = [
        "op",
        "removal lines",
        "exact match kept",
        "hit rate, clean",
        "hit rate, `projection`",
        "greedy distance, clean",
        "greedy distance, `projection`",
        "greedy distance, `operands`",
        "floor",
        "chance",
    ]
    rows = []
    for op in OPS:
        rows.append(
            [
                f"`{op}`",
                f"{res.n_lines(op, 'removal'):,}",
                span2(res.kept("handover", op)),
                span2(res.stat("handover", op, "clean", "removal", "hit")),
                span2(res.stat("handover", op, "projection", "removal", "hit")),
                span2(res.stat("handover", op, "clean", "removal", "greedy")),
                span2(res.stat("handover", op, "projection", "removal", "greedy")),
                span2(res.stat("handover", op, "operands", "removal", "greedy")),
                f"{res.reference(op, 'removal', 'floor_mode'):.2f}",
                f"{res.reference(op, 'removal', 'chance'):.2f}",
            ]
        )
    return table_html(
        head,
        rows,
        "**The removal lines on `handover`, per op: seed mean (seed range).** Distances in grid steps from the raw answer. The floor is the best grid answer's distance and chance a uniform draw's, both averaged over the op's removal lines.",
    )


removal_table(res)

r"""
## Kept and lost

A distance averaged over a group mixes lines the operator leaves answered with lines it does not, so we split the lines by the greedy guess. Under a given pass, a line is:
(a) *kept* when the guess is one of the possible answers of the line,
(b) *lost* when the clean guess was a possible answer and the guess under the pass is not, and
(c) *never* when the clean guess already missed.
The distance of the lost lines tells us where a removed answer goes.
"""


def lost_figure(res: Results) -> str:
    share = {
        (c, op): res.split(c, op, "projection", "removal", "lost", "n") / res.n_lines(op, "removal")
        for c in ORDER
        for op in OPS
    }
    lost = {(c, op): res.split(c, op, "projection", "removal", "lost", "greedy") for c in ORDER for op in OPS}
    kept = {(c, op): res.split(c, op, "projection", "removal", "kept", "greedy") for c in ORDER for op in OPS}
    refs = {k: [res.reference(op, "removal", k) for op in OPS] for k in ("floor_mode", "chance")}
    return lost_draw(share, lost, kept, refs)


@memo
def lost_draw(share: dict, lost: dict, kept: dict, refs: dict) -> str:
    @themed(
        name="lost-lines",
        alt_text="""
            Most of handover's removal lines are lost under projection; the lost lines sit two to three steps from the answer, short of chance on every op but value-hsv, while the kept lines stay at the floor.
        """,
        caption="""
            **The removal lines under `projection`, split by whether the greedy guess survives.** Top: the share of each op's removal lines that are lost (answered on the clean pass, not under the operator). Middle: the greedy distance of the lost lines from the raw answer, in grid steps, with chance dashed. Bottom: the greedy distance of the kept lines, with the floor dotted. Each small dot is one seed, the larger mark the seed mean, and the thin bar the seed range; a condition with no lost line on a seed draws nothing for it.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.6), layout="constrained", sharex=True)
        top, mid, bot = axes
        rng = np.random.default_rng(3)
        xs = np.arange(len(OPS))
        columns_panel(top, lambda c, op: share[c, op], OPS, rng=rng, legend=True)
        columns_panel(mid, lambda c, op: lost[c, op], OPS, rng=rng)
        columns_panel(bot, lambda c, op: kept[c, op], OPS, rng=rng)
        reference_ticks(mid, xs, refs["chance"], ls="--", label="chance")
        reference_ticks(bot, xs, refs["floor_mode"], ls=":", label="floor")
        top.set_ylim(-0.02, 1.05)
        mid.set_ylim(-0.1, max(mid.get_ylim()[1], 4.0))
        bot.set_ylim(-0.05, max(bot.get_ylim()[1], 1.0))
        top.set_ylabel("share of removal lines lost")
        mid.set_ylabel("lost lines: greedy\ndistance (steps)")
        bot.set_ylabel("kept lines: greedy\ndistance (steps)")
        fig_legend(fig, top)
        mid.legend(loc="upper right", fontsize=7, frameon=False)
        bot.legend(loc="upper right", fontsize=7, frameon=False)
        return fig

    return _plot()


lost_figure(res)

"""
Per line, as a distribution: in the histograms below, the bin at zero holds the kept lines and the other bins show where the lost answers go.

On `mix`, nine in ten of the moved answers sit one or two steps off, which is what treating the red channel as absent would give (on `mix` that shift is about 2.5 steps). `darken` decays from one step, `hue-hsv` spreads over two to four, and `value-hsv` is close to the shape a uniform guess gives.
"""


def chance_hist(res: Results, op: str, group: str = "removal") -> np.ndarray:
    """The distribution over rounded steps of a uniform guess's distance from the raw answer on a group's lines."""
    raw = res.lines(op, "raw")[res.group(op, group)]
    d = np.linalg.norm(GRID[None] - raw[:, None], axis=2)
    return np.bincount(np.clip(np.rint(d).astype(int).ravel(), 0, ex.N_BINS - 1), minlength=ex.N_BINS) / d.size


def hist_figure(res: Results) -> str:
    lost = {op: res.hist("handover", op, "projection", "removal").sum(0) for op in OPS}
    kept_clean = {op: res.hist("handover", op, "clean", "removal").sum(0) for op in OPS}
    chance = {op: chance_hist(res, op) for op in OPS}
    return hist_draw(lost, kept_clean, chance)


@memo
def hist_draw(hist: dict, clean: dict, chance: dict) -> str:
    @themed(
        name="lost-histograms",
        alt_text="""
            Per-op histograms of guess distance: mix peaks at one to two steps, darken decays from one step, hue-hsv spreads over two to four, and value-hsv is close to the uniform-guess shape.
        """,
        caption="""
            **Where the removal-line answers land under `projection` on `handover`, per op.** Bars: the share of the op's removal lines, pooled over twenty seeds, whose greedy guess sits each number of grid steps from the raw answer (rounded), under `projection`; in outline, the same under the clean pass. The dashed line is the distribution a uniform guess from the grid would give on the same lines.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 4, figsize=(8.4, 5.6), layout="constrained", sharex=True, sharey=True)
        axes = np.asarray(axes)
        bins = np.arange(ex.N_BINS)
        for ax, op in zip(axes.ravel(), OPS, strict=False):
            h = hist[op] / hist[op].sum()
            c = clean[op] / clean[op].sum()
            ax.bar(bins, h, width=0.8, color=ink("handover"), alpha=0.85, label="projection")
            ax.step(bins, c, where="mid", color=light_dark("#333", "#ddd"), lw=0.9, label="clean")
            ax.plot(bins, chance[op], "--", color=light_dark("#333", "#ddd"), lw=0.8, label="uniform guess")
            ax.set_title(op, fontsize=8)
            ax.grid(axis="y", alpha=0.2)
        axes.ravel()[-1].axis("off")
        axes.ravel()[0].legend(loc="upper right", fontsize=6.5, frameon=False)
        for ax in axes[-1]:
            ax.set_xlabel("steps from raw answer", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("share of lines", fontsize=8)
        return fig

    return _plot()


hist_figure(res)

r"""
## The lost answers on the cube

The figure draws the lost removal lines of the two ops the backlog item names: `hue-hsv`, whose removal lines keep the most exact match, and `darken`, the near miss of ex-2.2.11.
"""


def cube_figure(res: Results, seed: int = 0, n_max: int = 240) -> str:
    label = f"handover-s{seed}"
    panels = {}
    for op in ex.LINE_OPS:
        m = res.group(op, "removal")
        clean = res.per_line(label, op, "clean", "guess").astype(int)
        proj = res.per_line(label, op, "projection", "guess").astype(int)
        raw = res.lines(op, "raw")
        # A guess is a hit when every channel is the raw value's floor or ceiling: the possible answers are the
        # corners of the box the raw answer sits in, and one color when it sits on the grid.
        support = (np.abs(GRID[None] - raw[:, None]) < 1 - 1e-5).all(axis=2)
        hit_clean = support[np.arange(len(raw)), clean]
        hit_proj = support[np.arange(len(raw)), proj]
        lost = m & hit_clean & ~hit_proj
        idx = np.flatnonzero(lost)
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=min(n_max, len(idx)), replace=False) if len(idx) else idx
        panels[op] = (GRID[proj[idx]] / 5.0, raw[idx] / 5.0, int(lost.sum()))
    return cube_draw(panels, label)


@memo
def cube_draw(panels: dict, label: str) -> str:
    from sca.vis import plot_rgb_cube

    @themed(
        name="lost-cube",
        alt_text="""
            On the color wheel, hue-hsv guesses fan out from the red-hued answers toward yellow-green and blue-purple at the same saturation, and darken guesses land darker than their answers.
        """,
        caption=f"""
            **Where the lost removal lines land, on `{label}` under `projection`.** One panel per op, on the color wheel view of the cube. Each mark is the greedy guess of one lost line, in the guessed color; the ring is the line's raw answer and the stub joins the two. A random sample of the lost lines is drawn where there are more than fit.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(panels), figsize=(8.4, 4.2), layout="constrained")
        for ax, (op, (guess, raw, n)) in zip(np.atleast_1d(axes), panels.items(), strict=True):
            plot_rgb_cube(ax, guess, guess, truth=raw, view="wheel", s=14)
            ax.set_title(f"{op}: {n:,} lost lines", fontsize=9)
        return fig

    return _plot()


cube_figure(res)

r"""
## A counterfactual answer

This is a post hoc check on the near misses. If the operator took the redness out of the red operand and left the op intact, the guess would be the op applied to a de-reddened operand. We try two versions of that operand:
(a) red channel set to zero, and
(b) red channel lowered to the larger of its green and blue, which keeps the other two channels and drops the saturation.
"""


@memo
def counterfactual_rows(res: Results) -> list[list[str]]:
    from sca.answer_distance import STEP, distance
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME

    rows = []
    for op in OPS:
        rule = OP_BY_NAME.get(op) or CANDIDATE_BY_NAME[op]
        idx = np.flatnonzero(res.group(op, "removal"))
        red = np.where(res.group(op, "removal_op1")[idx], 0, 1)  # which operand is the red one
        ab = res.lines(op, "operands")[idx].astype(int) * STEP  # (n, 2, 3) on the 0..15 scale
        raw = res.lines(op, "raw")[idx]
        cfs = []
        for grey in (False, True):
            ops = ab.copy()
            rows_ = np.arange(len(idx))
            ops[rows_, red, 0] = ops[rows_, red, 1:].max(axis=1) if grey else 0
            cfs.append(np.array([rule.raw(tuple(a), tuple(b)) for a, b in ops.tolist()]) / STEP)
        box = (np.abs(GRID[None] - cfs[1][:, None]) < 1 - 1e-5).all(axis=2)
        d = np.zeros(4)
        for r in res.runs("handover"):
            g = res.per_line(r["label"], op, "projection", "guess").astype(int)[idx]
            d += [
                distance(GRID[g], raw).mean(),
                *(distance(GRID[g], c).mean() for c in cfs),
                box[np.arange(len(idx)), g].mean(),
            ]
        d /= len(res.runs("handover"))
        rows.append([f"`{op}`", f"{d[0]:.2f}", f"{d[1]:.2f}", f"{d[2]:.2f}", f"{d[3]:.0%}"])
    return rows


def counterfactual_table(res: Results) -> str:
    head = ["op", "to the answer", "to the zero-red answer", "to the grey-red answer", "in the grey-red box"]
    return table_html(
        head,
        counterfactual_rows(res),
        "**Distance of the greedy guess on the removal lines, `handover` under `projection`, seed mean.** The answer is the line's raw answer; the zero-red and grey-red answers are the op applied with the red operand's red channel set to zero, or to the larger of its green and blue. The last column is the share of lines whose guess is one of the grey-red answer's possible roundings.",
    )


counterfactual_table(res)

"""
Neither substitute is where the guesses go. The zero-red answer is farther from the guess than the true answer on every op but `lighten` and `hsvmix`. The grey-red answer is a little nearer on most ops (2.0 steps against 2.5 on `hue-hsv`), but the guess lands in its box on under a quarter of the lines on every op but `lighten` (36%).

So the lost answers sit between the answer and its de-reddened version, and no single substitute operand describes them.
"""

r"""
## The non-red lines

The non-red lines are the other side of selectivity: the operator should leave them alone. Ex-2.2.11 gates the exact match deficit on `mix` at {ex.ex2211.NONRED_DEFICIT_GATE:g}, and finds it near zero on every op. The distance shows whether the mass moves at all on those lines.
"""


def nonred_figure(res: Results) -> str:
    g = "nonred_excl"
    deficit = {
        (c, op): res.stat(c, op, "clean", g, "eem") - res.stat(c, op, "projection", g, "eem")
        for c in ORDER
        for op in OPS
    }
    greedy = {
        (c, op): res.stat(c, op, "projection", g, "greedy") - res.stat(c, op, "clean", g, "greedy")
        for c in ORDER
        for op in OPS
    }
    mean = {
        (c, op): res.stat(c, op, "projection", g, "mean") - res.stat(c, op, "clean", g, "mean")
        for c in ORDER
        for op in OPS
    }
    return nonred_draw(deficit, greedy, mean)


@memo
def nonred_draw(deficit: dict, greedy: dict, mean: dict) -> str:
    @themed(
        name="nonred-distance",
        alt_text="""
            On the non-red lines every rise is small; handover and handover-slot rise less than the control on every op, and handover-tied rises above it on difference.
        """,
        caption=f"""
            **The non-red lines (without a red answer) under `projection`, per op and condition: what the operator changes.** Top: the deficit in expected exact match (clean minus `projection`), dashed at ex-2.2.11's {ex.ex2211.NONRED_DEFICIT_GATE:g} gate, which that report reads on `mix` alone. Middle: the rise in the greedy distance from the raw answer, in grid steps. Bottom: the rise in the distance of the mean of the model's distribution. Each small dot is one seed, the larger mark the seed mean, and the thin bar the seed range.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.6), layout="constrained", sharex=True)
        top, mid, bot = axes
        rng = np.random.default_rng(4)
        columns_panel(top, lambda c, op: deficit[c, op], OPS, rng=rng, legend=True)
        columns_panel(mid, lambda c, op: greedy[c, op], OPS, rng=rng)
        columns_panel(bot, lambda c, op: mean[c, op], OPS, rng=rng)
        top.axhline(ex.ex2211.NONRED_DEFICIT_GATE, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
        for ax in (top, mid, bot):
            ax.axhline(0, color=light_dark("#333", "#ddd"), lw=0.6, ls=":", zorder=1)
        top.set_ylabel("exact match deficit")
        mid.set_ylabel("greedy distance\nrise (steps)")
        bot.set_ylabel("mean-of-distribution\ndistance rise (steps)")
        fig_legend(fig, top)
        return fig

    return _plot()


nonred_figure(res)

r"""
## Between seeds

The distances vary far less between seeds than exact match does. A measurement used for a gate needs a seed spread small relative to the effect it measures; the table gives each removal measurement's spread across the twenty `handover` seeds as a share of its mean (the coefficient of variation, CV).
"""


def spread_table(res: Results) -> str:
    head = [
        "op",
        "exact match kept: mean",
        "seed CV",
        "greedy distance: mean",
        "seed CV",
        "mean-of-distribution distance: mean",
        "seed CV",
    ]
    rows = []
    for op in OPS:
        vals = [
            res.kept("handover", op),
            res.stat("handover", op, "projection", "removal", "greedy"),
            res.stat("handover", op, "projection", "removal", "mean"),
        ]
        rows.append([f"`{op}`", *(s for v in vals for s in (f"{v.mean():.2f}", f"{v.std(ddof=1) / v.mean():.0%}"))])
    return table_html(
        head,
        rows,
        "**Seed spread of the removal measurements on `handover` under `projection`, per op.** CV is the standard deviation over the twenty seeds divided by the seed mean.",
    )


spread_table(res)

r"""
## What we make of it

The kept exact match on the removal lines is not a quarter of the lines answered and the rest sent anywhere. It is closer to the other reading: nearly every line moves, and moves a short way. On `mix`, the op the D2.2 gates are scored on, the guess ends one or two steps from the answer nine times in ten.

The answers move toward what the line would give without its red, but not all the way: the counterfactual check says the model is not computing the op on a de-reddened operand.

That is the kind of side-effect this program wants to be able to describe. The intervention changes the answer on the lines that use the concept, by an amount that scales with how much the op relies on the red channel, and leaves the other lines alone.

The distance gives the removal measurement partial credit and a scale. On `mix`, exact match on `handover` sits near zero with a seed spread larger than its mean, while the distance on the same lines has a seed spread under a tenth of its mean.

That comparison flatters the distance: the CV favors a measurement with a large mean, and the distances have means an order of magnitude above the kept shares. In units of the seed spread, the two separate the conditions about equally well. The advantage of the distance is that it stays informative where exact match is near zero.

The distance also separates ops that exact match does not: `mix` and `value-hsv` both keep under a tenth of their exact match, but their guesses sit 1.8 and 3.4 steps away.

On the non-red lines, the mean of the distribution is the sensitive measurement, since it moves while the guess stays put. On `handover` every such move is under 0.03 steps, and the control's lines move as much or more, so the operator's cost on the lines it should leave alone is about what removing any direction costs.

For use in a gate, the backlog item asks for a preregistered direction before the measurement is adopted. This re-score supports two measurements:
(a) removal, as the normalized distance of the mean of the distribution on the removal lines, (distance − floor) / (chance − floor), rising under the operator, with the greedy distance beside it as the number a user of exact match would expect; and
(b) selectivity, as the non-red rise of the same mean staying at or under the rise of the control under the same operator.
The next experiment should set the thresholds from its own control runs. Until one of the distances has been used in a gate, the greedy and mean distances should be quoted together.

The distance does not say whether an answer is right, except at the floor, so a gate that wants "answered" should keep the hit rate or exact match beside it.

## Method

The distances are in `sca.answer_distance`, and the re-score in [`experiment.py`](experiment.py) beside this script. Each of the 54 final checkpoints of ex-2.2.11 is run over the same probe lines under the clean pass, `projection`, and `operands`, with one compiled forward pass per operator. The answer distribution at the pre-answer position is restricted to the 216 color tokens.

For each line, we compute against its raw answer: the greedy guess, the mean and channel-wise median of that distribution, its expected exact match and expected distance, and whether the guess is a hit. For each group and pass, these are averaged, split by kept, lost, and never, and binned by rounded greedy distance. On `handover` and the control, the per-line guess is published for every op, and the mean beside it for `hue-hsv` and `darken`; every other per-line array stays in the run's memo record.
"""
