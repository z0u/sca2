# title: Ex 2.2.4: a scouting round before the anchored-op experiments

r"""
# Ex 2.2.4: a scouting round before the anchored-op experiments

/// tip |
<!-- tl;dr -->
A scouting pass over a more varied set of ops, on the grid alone. The hue, saturation, and value ops spread their answers through the color cube, and the answer moves furthest there once the red operand loses its red, though a small change in the red operand reaches their answer no more often than it does for the current ops. Three commutative ops spread and are sensitive to both operands: `difference`, `exclusion`, and `mix` done in HSV. We propose a table with all six added and `add` dropped.
///

## Observations

Each line below is a read on the grid, together with the statistic it rests on. None of them is a result. The next preregistration will adopt what it needs from here and check it on trained models.

- [The op set](#the-op-set): every candidate spreads its answers more evenly than the current ops do. A one-level change in the red operand reaches the answer on 61% to 100% of red lines for the commutative candidates and on 37% to 55% for the one-attribute HSV ops, the range `screen` and `multiply` sit in. Zeroing the red operand's R runs the other way: the answer moves furthest on the HSV trio. We propose table A+: drop `add`, add `difference`, `exclusion`, and `hsvmix`, and carry `hue-hsv`, `sat-hsv`, and `value-hsv` as a marked subset, the first ops in the grammar that read operand order.
- The one-level rule of E8 responds to rounding as much as to the op: `mix` counts as dependent on 61% of its own red probe lines and 49% of the shared draw. Under stochastic rounding the rule is graded and reads 0.50 on both sets, and no op moves by as much as 0.1, so the ranking of the ops is the same either way.
- [What we make of it](#what-we-make-of-it): the removal statistic should be a distance rather than exact match, scored on lines where zeroing the red operand's R moves the answer far, and the grammar should adopt stochastic rounding and the whole-line labeller from the two pilots, for comparability with M3.

## How to read this

This is the scouting round planned in the [backlog item](/todo/science/scouting-round-before-the-anchored-op-experiments.md): one notebook, the cheapest version of each question, no gates and no verdicts. Each section says what we ran, shows one figure or table, and says whether it changes the design. Sections land as their questions are run, and this one covers the op set. The tied-readout diagnostic and the τ against λ_a trade are still to come. The answer-drawing labeller ran as a pilot of its own, beside a pilot of stochastic rounding, and the closing section reads both.

Ex-2.2.3 anchored *red* on a grammar of six operations, then measured removal: does the model still answer correctly once the *red* axis is projected out of the residual stream?[^stream] Removal came out partial on four of the six ops. E8 traced that to the ops themselves: on the saturating ops, most red lines have an answer that would be the same even if the red operand were a little less red, so a model that has lost *red* can still answer them.

The [diverse-op todo item](/todo/science/diverse-operator-set-hue-saturation-brightness.md) proposes adding ops whose answers spread through the cube and depend on both operands, such as the hue, saturation, and brightness blend modes, alongside some of the saturating ones, so that the removal tests have both kinds to read.

[^stream]: The *residual stream* is the running vector of activations that each transformer layer reads from and writes back to; it is where we place the anchor.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

from mini.lit import memo
from mini.vis import figure_html, light_dark, themed
from sca.data.colors import Rgb, redness
from sca.data.ops import (
    CANDIDATES,
    LEVELS,
    OPS,
    TOP,
    Op,
    answer_dist,
    colors,
    commutativity,
    lines,
    mix_probe_lines,
    on_grid,
    probe_partners,
    unordered_pairs,
)
from sca.vis import CUBE_VIEWS, draw_cube_bound, grid_diameter, project_cube

ALL_OPS: tuple[Op, ...] = OPS + CANDIDATES
# The six ops of ex-2.2.3, then the nine candidates, in the order every table and figure uses.

RED_DOSE = 0.8
# Ex-2.2.3's red line: the redder operand has redness ≥ 0.8.

N_PROBE, PROBE_SEED = 27, 0
# Ex-2.2.3's probe draw: 27 partners per color, shared across the non-`mix` ops.

STEP = LEVELS[1] - LEVELS[0]
# One grid level: the drop the dependence rule of E8 applies to the R of the red operand.

GRID_RGB = np.asarray(colors(), dtype=float) / TOP
# The 216 colors as unit-cube RGB, in palette order.

HSV_TRIO: tuple[str, ...] = ("hue-hsv", "sat-hsv", "value-hsv")
# The one-attribute HSV ops: the non-commutative subset of table A+.

TABLES: dict[str, tuple[str, ...]] = {
    "current": ("mix", "add", "screen", "multiply", "lighten", "darken"),
    "A": ("mix", "screen", "multiply", "lighten", "darken", "difference", "exclusion", "hsvmix"),
}
TABLES["A+"] = TABLES["A"] + HSV_TRIO
# Three tables the relevance section reads: ex-2.2.3's, a commutative widening (A), and A with the
# HSV trio as a marked subset (A+).

r"""
## The op set

**What we ran.** We took nine candidate ops, each defined on the same 216-color grid and snapped to it the way the six ops of ex-2.2.3 are, so every line still has an answer in the vocabulary.

Three of them work channel by channel and are commutative: `difference` and `exclusion`, which are the blend modes of those names, and `hsvmix`, which is `mix` done in HSV (the average hue, saturation, and value). The other six take one attribute from the second operand and the rest from the first: the hue, saturation, and luminosity modes of the W3C compositing spec, as in Photoshop, and the same three in HSV proper, as in Krita.
"""


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    """A result table in the shared classes: numeric cells right-aligned."""
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def lowered(a: Rgb, b: Rgb, drop: int) -> tuple[Rgb, Rgb]:
    """The line with the redder operand's R lowered by *drop*, floored at 0."""
    if redness(a) >= redness(b):
        return (max(a[0] - drop, 0), a[1], a[2]), b
    return a, (max(b[0] - drop, 0), b[1], b[2])


@dataclass(frozen=True)
class Scout:
    """One op's answers on every ordered pair, and the reads the section takes on them."""

    op: Op

    # Each read below is a pure function of the op (and its arguments), so it is memoized on disk: a warm render
    # never evaluates an op on the grammar's lines. The answer table itself stays lazy, since the reads that use it
    # are the cached ones, and pickling fifteen tables of 46,000 lines would cost more than it saves.

    @cached_property
    def table(self) -> dict[tuple[Rgb, Rgb], Rgb]:
        return {(a, b): self.op(a, b) for a, b in lines()}

    @cached_property
    @memo
    def counts(self) -> np.ndarray:
        """Unordered pairs whose answer is each grid color, in palette order."""
        index = {c: i for i, c in enumerate(colors())}
        n = np.zeros(len(index))
        for c, k in Counter(self.table[(a, b)] for a, b in unordered_pairs()).items():
            n[index[c]] = k
        return n

    @cached_property
    def spread(self) -> float:
        """Entropy of the answer distribution over the grid, in bits; log2(216) ≈ 7.75 is uniform."""
        p = self.counts / self.counts.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    @cached_property
    @memo
    def marginals(self) -> np.ndarray:
        """(channel, level): the share of unordered pairs whose answer has that level in that channel."""
        ans = np.array([self.table[(a, b)] for a, b in unordered_pairs()])
        return np.stack([[np.mean(ans[:, c] == lv) for lv in LEVELS] for c in range(3)])

    @cached_property
    @memo
    def commutative(self) -> float:
        return commutativity(self.op)

    @cached_property
    @memo
    def redder(self) -> int:
        """Lines whose answer is redder than either operand."""
        return sum(redness(ans) > max(redness(a), redness(b)) + 1e-9 for (a, b), ans in self.table.items())

    @memo
    def dependence(self, pairs: list[tuple[Rgb, Rgb]], drop: int) -> float:
        """The share of *pairs* (ordered) whose answer changes when the redder operand's R falls by *drop*."""
        return float(np.mean([self.op(*lowered(a, b, drop)) != self.table[(a, b)] for a, b in pairs]))

    @memo
    def dependence_stochastic(self, pairs: list[tuple[Rgb, Rgb]], drop: int) -> float:
        """The same read under stochastic rounding: the share of a line's answer mass that moves when the
        redder operand's R falls by *drop* (the total variation distance), averaged over *pairs*.
        """
        moved = []
        for a, b in pairs:
            p, q = answer_dist(self.op, a, b), answer_dist(self.op, *lowered(a, b, drop))
            moved.append(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in p.keys() | q.keys()) / 2)
        return float(np.mean(moved))

    @memo
    def displacement(self, pairs: list[tuple[Rgb, Rgb]], drop: int) -> float:
        """How far the answer moves, in unit-cube RGB, when the redder operand's R falls by *drop*: the mean
        over *pairs*, where √3 is corner to corner.
        """
        return float(
            np.mean(
                [np.linalg.norm(np.subtract(self.op(*lowered(a, b, drop)), self.table[(a, b)]) / TOP) for a, b in pairs]
            )
        )

    @memo
    def sensitivity(self, role: int, seed: int = 0) -> float:
        """The share of lines whose answer changes when the operand in *role* (0 or 1) is replaced by a
        random other color: does the op read that operand at all?
        """
        cs = colors()
        rng = np.random.default_rng(seed)
        changed = 0
        sample = lines()[::5]
        for a, b in sample:
            other = cs[rng.integers(len(cs))]
            swapped = (other, b) if role == 0 else (a, other)
            changed += self.table.get(swapped, self.op(*swapped)) != self.table[(a, b)]
        return changed / len(sample)


scouts: dict[str, Scout] = {op.name: Scout(op) for op in ALL_OPS}
red_lines: list[tuple[Rgb, Rgb]] = [(a, b) for a, b in lines() if max(redness(a), redness(b)) >= RED_DOSE - 1e-9]
# Every red line of the grammar, in both operand orders.
partners = probe_partners(N_PROBE, PROBE_SEED)[1]
probe_red: list[tuple[Rgb, Rgb]] = [
    (c, b)
    for c, ps in zip(colors(), partners, strict=True)
    for b in ps
    if max(redness(c), redness(b)) >= RED_DOSE - 1e-9
]
# The red lines of ex-2.2.3's shared probe draw, which reads every non-`mix` op.
mix_red: list[tuple[Rgb, Rgb]] = [
    (a, b) for a, b in mix_probe_lines() if max(redness(a), redness(b)) >= RED_DOSE - 1e-9
]
# `mix`'s own probe set: D2.1's closed pairs.

rows = []
for op in ALL_OPS:
    s = scouts[op.name]
    rows.append(
        [
            f"<code>{op.name}</code>",
            op.rule,
            "yes" if s.commutative > 0.999 else f"{s.commutative:.0%}",
            f"{on_grid(op):.0%}",
            f"{s.redder:,} ({s.redder / len(lines()):.0%})",
        ]
    )
caption = """
<b>The fifteen ops.</b> The six ops of ex-2.2.3, then the nine candidates. <em>Commutative</em> is the share of unordered pairs whose answer is the same in both operand orders. <em>On grid</em> is the share of pairs the rule answers without rounding. <em>Redder than both</em> counts the lines whose answer is redder than either operand, the case the labeller never sees.
"""
table_html(["op", "rule", "commutative", "on grid", "redder than both"], rows, caption)

r"""
### Where the answers land

Below is the answer-cloud figure from ex-2.2.3, extended to the candidates. Each panel shows the color cube seen from the gray diagonal, with one mark per grid color, sized by how many pairs answer there. An op that spreads its answers puts small marks everywhere. A saturating op piles them onto a face, an edge, or a corner.
"""

counts_by_op: dict[str, np.ndarray] = {op.name: scouts[op.name].counts for op in ALL_OPS}
answer_clouds_caption = f"""
    **Where the answers of each op land.** One mark per grid color, with area proportional to the number of unordered pairs whose answer is that color. All panels share one scale: a mark that fills its grid cell stands for 600 pairs, out of {len(unordered_pairs()):,}. The top row and the first panel of the second row are the ops of ex-2.2.3; the rest are the candidates.
"""


@memo
def answer_clouds_figure(counts: dict[str, np.ndarray], alt_text: str, caption: str) -> str:
    """The answer-cloud grid, one panel per op. Memoized on the per-op counts and the text."""
    order = np.argsort(GRID_RGB @ CUBE_VIEWS["solid"].toward, kind="stable")
    xy = project_cube(GRID_RGB[order])
    full_cell = 600

    @themed(name="answer-clouds", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 5, figsize=(11.5, 8.4))
        for ax, op in zip(axes.flat, ALL_OPS, strict=True):
            draw_cube_bound(ax)
            dia = grid_diameter(len(LEVELS)) * np.sqrt(counts[op.name][order] / full_cell)
            ax.add_collection(
                EllipseCollection(
                    widths=dia,
                    heights=dia,
                    angles=0,
                    units="xy",
                    offsets=xy,
                    offset_transform=ax.transData,
                    facecolors=GRID_RGB[order],
                    edgecolors=light_dark("#00000033", "#ffffff55"),
                    linewidths=0.5,
                    zorder=3,
                    clip_on=False,
                )
            )
            ax.set_title(op.name, y=1.12)
        return fig

    return plot()


answer_clouds_figure(
    counts_by_op,
    """
        Fifteen color-cube panels, one per op, with a mark on each grid color sized by how many pairs answer there. The six current ops crowd a face or a corner, add and multiply most of all; the nine candidates spread small marks through the cube, with difference and the hue ops leaning dark and luminosity leaning light.
    """,
    answer_clouds_caption,
)

r"""
### How evenly the answers spread

We measure evenness two ways. The first is per channel: for each op, the share of answers at each of the six levels of R, G, and B. A flat line at one sixth means the answers cover that channel evenly, and a spike at 0 or 15 means the op saturates. The second is a single number per op, the entropy of the answer distribution over the 216 colors.[^ent]

[^ent]: Entropy in bits, so a higher number means the answers are spread over more colors.
"""

marginals_by_op: dict[str, np.ndarray] = {op.name: scouts[op.name].marginals for op in ALL_OPS}


@memo
def channel_marginals_figure(marginals: dict[str, np.ndarray], alt_text: str, caption: str) -> str:
    """The per-channel answer-level shares, one panel per op. Memoized on the marginals and the text."""

    @themed(name="channel-marginals", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 5, figsize=(11.5, 6.4), sharex=True, sharey=True)
        inks = light_dark(["#c33", "#2a2", "#33c"], ["#f66", "#5d5", "#77f"])
        for ax, op in zip(axes.flat, ALL_OPS, strict=True):
            m = marginals[op.name]
            ax.axhline(1 / len(LEVELS), ls="--", lw=0.8, color=light_dark("#888", "#aaa"))
            for c in range(3):
                ax.plot(LEVELS, m[c], "o-", ms=3, lw=1.4, color=inks[c])
            ax.set_title(op.name, fontsize=10)
            ax.set_xticks([0, TOP])
            ax.grid(alpha=0.25)
        for ax in axes[-1]:
            ax.set_xlabel("answer level")
        for ax in axes[:, 0]:
            ax.set_ylabel("share of pairs")
        return fig

    return plot()


channel_marginals_figure(
    marginals_by_op,
    """
        Fifteen small panels, one per op, each with three lines (red, green, blue) showing the share of answers at each of the six grid levels. The current ops spike at 15 (add, screen, lighten) or at 0 (multiply, darken); the candidates run close to the flat one-sixth line, with difference sloping toward 0.
    """,
    """
        **Answer level shares per channel.** For each op, the share of unordered pairs whose answer has each grid level in R, G, and B, drawn in the color of that channel. The dashed line is one sixth, which is what an op with even coverage of the channel would show. For the per-channel ops and `hsvmix` the three lines coincide, because the rule treats the channels alike. Panels share the vertical scale.
    """,
)

r"""
### Does the answer need the red operand to be red?

E8 of ex-2.2.3 calls a red line *dependent* when lowering the R of the red operand by one grid level changes the snapped answer. On the dependent lines of every op, removal came out complete. We apply that rule to every op here, both on every red line and on the shared probe draw. We also add a second rule beside it: drop the R of the red operand to zero instead of by one level.

The one-level rule asks whether the op notices a small change in the red operand. The to-zero rule asks whether the answer needs *red* at all. Two more reads go with them. The first is the one-level rule under stochastic rounding, where a line's answer is a distribution over the grid colors nearest its raw value, and the read is the share of that mass that moves.[^sto] The second is how far the answer moves, in the unit cube, when R drops to zero: for a removal that is meant to make the model fail, the distance is what "fails badly" would be scored against.

*Red* is the concept D2.1 chose as a stand-in for one a model should not be able to use, and ex-2.2.3 kept it for comparability. Any other color could be anchored the same way, so an op's behavior on red lines is one consideration here, beside spread and commutativity, rather than the deciding one.

[^sto]: Stochastic rounding is the grammar variant of the [stochastic-rounding pilot](/docs/m2/ex-2.2.5/report.py): a raw channel value between two grid levels rounds to the upper one with probability equal to how far along it sits, drawn once per line at corpus build. The answer of a line is then a distribution over up to eight colors, and `answer_dist` in `sca.data.ops` computes it. The share of mass that moves between two distributions is their total variation distance.
"""

dep: dict[str, dict[str, float]] = {
    op.name: {
        "step": scouts[op.name].dependence(red_lines, STEP),
        "step_probe": scouts[op.name].dependence(probe_red, STEP),
        "step_sto": scouts[op.name].dependence_stochastic(red_lines, STEP),
        "zero": scouts[op.name].dependence(red_lines, TOP),
        "zero_move": scouts[op.name].displacement(red_lines, TOP),
    }
    for op in ALL_OPS
}
# Per op: one-level dependence on all red lines, on the shared probe draw, and under stochastic rounding;
# to-zero dependence and how far the answer moves under it.
mix_own: dict[str, float] = {
    "step": scouts["mix"].dependence(mix_red, STEP),
    "step_sto": scouts["mix"].dependence_stochastic(mix_red, STEP),
    "zero": scouts["mix"].dependence(mix_red, TOP),
}
# `mix` on its own probe set, which is what E8 read.
sens: dict[str, tuple[float, float]] = {
    op.name: (scouts[op.name].sensitivity(0), scouts[op.name].sensitivity(1)) for op in ALL_OPS
}
# Per op: the share of lines whose answer changes when op1, or op2, is replaced at random.

rows = []
for op in ALL_OPS:
    s, d = scouts[op.name], dep[op.name]
    rows.append(
        [
            f"<code>{op.name}</code>",
            f"{s.spread:.2f}",
            f"{d['step']:.2f}",
            f"{d['step_probe']:.2f}",
            f"{d['step_sto']:.2f}",
            f"{d['zero']:.2f}",
            f"{d['zero_move']:.2f}",
            f"{sens[op.name][0]:.2f} / {sens[op.name][1]:.2f}",
        ]
    )
caption = f"""
<b>Spread and dependence, per op.</b> <em>Spread</em> is the entropy of the answer distribution in bits, where 7.75 is uniform over the grid. <em>One level</em> is the rule from E8: the share of red lines (dose ≥ {RED_DOSE:g}) whose answer changes when the R of the red operand drops one grid level, given over all {len(red_lines):,} red lines, over the {len(probe_red):,} red lines of the shared probe draw, and under stochastic rounding, where it is the share of answer mass that moves. <em>To zero</em> drops R to 0 instead, and <em>moves by</em> is how far the answer moves in the unit cube when it does, where √3 ≈ 1.73 is corner to corner. <em>Reads op1 / op2</em> is the share of lines whose answer changes when that operand is replaced at random. On its own probe set of {len(mix_red)} red lines, <code>mix</code> is {mix_own["step"]:.2f} by the one-level rule, {mix_own["step_sto"]:.2f} under stochastic rounding, and {mix_own["zero"]:.2f} to zero.
"""
table_html(
    [
        "op",
        "spread ↑",
        "one level ↑",
        "one level, probe",
        "one level, stochastic",
        "to zero ↑",
        "moves by ↑",
        "reads op1 / op2",
    ],
    rows,
    caption,
)

# %%
spread_by_op: dict[str, float] = {op.name: scouts[op.name].spread for op in ALL_OPS}
commutative_by_op: dict[str, float] = {op.name: scouts[op.name].commutative for op in ALL_OPS}


@memo
def spread_vs_dependence_figure(
    spread: dict[str, float],
    commutative: dict[str, float],
    dep: dict[str, dict[str, float]],
    mix_step: float,
    alt_text: str,
    caption: str,
) -> str:
    """Spread against the two dependence reads. Memoized on the per-op spread, commutativity, and dependence
    reads, `mix`'s own-probe-set dependence, and the text.
    """

    @themed(name="spread-vs-dependence", alt_text=alt_text, caption=caption)
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
        inks = {True: light_dark("#666", "#aaa"), False: light_dark("#2557b8", "#7fa6ff")}
        nudges = {
            "step": {
                "hue": (-8, -13),
                "luminosity": (5, 4),
                "value-hsv": (-52, -1),
                "sat-hsv": (5, -10),
                "saturation": (5, 7),
                "multiply": (5, 0),
                "mix": (5, -8),
            },
            "zero_move": {
                "screen": (5, -9),
                "multiply": (-52, -1),
                "darken": (5, -9),
                "lighten": (-46, 4),
                "mix": (-26, 4),
                "difference": (5, -12),
                "luminosity": (5, -9),
                "hue": (-24, 4),
                "saturation": (5, 5),
                "hue-hsv": (5, 4),
                "sat-hsv": (5, -9),
                "hsvmix": (-46, 6),
            },
        }
        labels = {"step": "dependence on the red operand (one level)", "zero_move": "answer moves by (R to zero)"}
        for ax, key in zip(axes, ("step", "zero_move"), strict=True):
            if key == "step":
                ax.axhline(mix_step, ls=":", lw=1, color=light_dark("#999", "#888"))
            for op in ALL_OPS:
                x, y = spread[op.name], dep[op.name][key]
                ink = inks[op in OPS]
                comm = commutative[op.name] > 0.98
                ax.plot(x, y, "o", ms=7, mfc=ink if comm else "none", mec=ink, mew=1.4)
                ax.annotate(
                    op.name,
                    (x, y),
                    xytext=nudges[key].get(op.name, (5, 4)),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=ink,
                )
            ax.set(xlabel="spread of answers (bits)", ylabel=labels[key])
            ax.grid(alpha=0.25)
        axes[0].set_ylim(0, 1.05)
        axes[1].set_ylim(0.4, 1.05)
        return fig

    return plot()


spread_vs_dependence_figure(
    spread_by_op,
    commutative_by_op,
    dep,
    mix_own["step"],
    """
        Two scatter panels with answer spread on the horizontal axis and one labelled point per op. Left, one-level dependence: the current ops sit at the left, low spread, with dependence from 0.2 to 0.8; the candidates sit to the right at higher spread, the non-commutative HSV ops at 0.4 to 0.55 and the commutative ones higher, with difference at the top right. Right, how far the answer moves when R drops to zero: the current ops and the commutative candidates sit at 0.5 to 0.65, and the one-attribute HSV ops above them, value-hsv highest at 1.0.
    """,
    f"""
        **Spread against the two dependence reads.** For each op, the entropy of its answers in bits against, left, the share of its red lines whose answer changes when the R of the red operand drops one level, and, right, how far the answer moves in the unit cube when R drops to zero. Filled marks are commutative ops and open marks are not. Gray marks are the ops of ex-2.2.3, blue the candidates. The dotted line on the left is `mix` on its own probe set ({mix_own["step"]:.2f}), the level E8 compared the other ops against.
    """,
)

# %%
cand = [op.name for op in CANDIDATES]
cur = [op.name for op in OPS]
hsv = ["hue", "saturation", "luminosity", *HSV_TRIO]
hi_spread = min(scouts[n].spread for n in cand if n != "exclusion")
cur_max = max(scouts[n].spread for n in cur)
mix_m = scouts["mix"].marginals[0]
sto_shift = max(abs(d["step_sto"] - d["step"]) for d in dep.values())
cur_move = [dep[n]["zero_move"] for n in cur]

rf"""
**What we saw.**

*The candidates spread; the current ops crowd.* Every candidate except `exclusion` has an answer entropy above {hi_spread:.1f} bits, and no current op reaches {cur_max:.1f}. `exclusion` sits between the two groups, with a third of its answers at each middle level.

The channel marginals say why: `add`, `screen`, and `lighten` pile up at 15, `multiply` and `darken` at 0, and the candidates run near the flat line. So on evenness the proposed hue, saturation, and brightness ops do what the item hoped, and so do the three commutative candidates.

`mix` is uneven for a different reason: rounding. Half of its channel sums fall half-way between two levels, and the snap sends every such tie to the even level index, so its answers pile up at 6 and 12 ({mix_m[2]:.0%} and {mix_m[4]:.0%} of pairs, per channel) and only {mix_m[5]:.0%} reach 15. Stochastic rounding would split each tie evenly and take that out; the rule itself is as even as `hsvmix`.

*The dependence rule from E8 responds to rounding as much as to the op.* By the one-level rule, `mix` itself is dependent on {mix_own["step"]:.0%} of its own red probe lines, and {dep["mix"]["step_probe"]:.0%} on the shared draw used for the other ops. That is because a one-level drop in one operand moves the mean by half a level, and the snap sends half of those cases back to the original answer. Under stochastic rounding the same read is graded, the share of answer mass that moves, and `mix` comes out at {mix_own["step_sto"]:.2f} on both probe sets: the half-level shift is read as half. No op moves by more than {sto_shift:.2f} between the two roundings, so the ranking of the ops does not depend on the rounding; what the stochastic read takes out is the probe-set artifact.

The hue, saturation, and brightness ops sit at {min(dep[n]["step"] for n in hsv):.2f} to {max(dep[n]["step"] for n in hsv):.2f}, no better than `screen` or `multiply`, for a structural reason: each takes one attribute from one operand and the rest from the other, and scaling the R of a pure red changes only its value. So whichever role reads hue or saturation from the red operand never sees the drop. Sensitivity to a small change is where the commutative candidates stand out: `difference` at {dep["difference"]["step"]:.2f}, `hsvmix` at {dep["hsvmix"]["step"]:.2f}, and `exclusion` at {dep["exclusion"]["step"]:.2f}.

*The to-zero read runs the other way.* Every op, current and candidate, changes its answer on at least {min(d["zero"] for d in dep.values()):.0%} of red lines once the red operand's R is zeroed. What separates the ops is how far the answer moves. The current ops all move by {min(cur_move):.2f} to {max(cur_move):.2f}, about half a channel's range, since the per-channel rules pass part of the drop through to R and leave G and B alone. The one-attribute HSV ops move furthest: {dep["value-hsv"]["zero_move"]:.2f} on `value-hsv`, {dep["hue-hsv"]["zero_move"]:.2f} on `hue-hsv`, and {dep["sat-hsv"]["zero_move"]:.2f} on `sat-hsv`. An op that copies one attribute of the red operand answers with something far from red once that attribute is gone: a red operand at op2 gives `value-hsv` its full value, and with its R zeroed it gives almost none. So if the removal question is "can the model still mix red", the HSV ops are where a model that has lost *red* would fail hardest, and the one-level rule would not have said so.

*The blend modes are the first ops where operand order matters.* All six hue, saturation, and brightness variants agree with their own reverse on under {max(scouts[n].commutative for n in hsv):.0%} of pairs. The saturation ones read op2 on only {min(scouts[n].sensitivity(1) for n in ("saturation", "sat-hsv")):.0%} of lines, since many partners share a saturation. The grammar of ex-2.2.3 is commutative throughout: the probe sets walk every color as op1, and the labeller pools both operands the same way. A non-commutative op puts role information into the grammar for the first time. Natural language has that everywhere, so it is a change we want before M3, and the grid side of it is small: a probe draw that also walks every color as op2, and per-op reads kept separate for the subset, so that anything odd in their behavior can be set aside without touching the rest of the table.

### How often the op word is worth reading

The D2.2 design uses a quantity called *op-relevance*: for each line of the anchored op, how many other ops $k$ in the table give the same answer. At 0 the answer identifies the op on its own. At $k$ the op word only rules out the other $k$. The suppression experiments predict per-line damage from this number, so widening the table changes the prediction for every op in it.

We compare three tables: the one from ex-2.2.3, table A (drop `add`, add the three commutative candidates), and table A+ (table A with the HSV trio, `hue-hsv`, `sat-hsv`, and `value-hsv`). Three ops of A+ read operand order, so the tables count ordered pairs, which moves ex-2.2.3's figures by under a percent.
"""

by_name = {op.name: op for op in ALL_OPS}


@memo
def relevance_tables(tables: dict[str, tuple[str, ...]], ops: dict[str, Op]) -> dict[str, dict[str, dict[int, float]]]:
    """Per table and anchored op, `sca.data.ops.relevance` over the ordered lines, with every op's answers computed once.

    The library function evaluates every op of the table on every line for each anchored op; here that is fifteen ops on 46,000 lines, so the answers are tabulated first and the count is a comparison per line.
    """
    ls = lines()
    answers = {n: np.array([ops[n](a, b) for a, b in ls]) for n in ops}
    out: dict[str, dict[str, dict[int, float]]] = {}
    for label, names in tables.items():
        out[label] = {}
        for n in names:
            same = sum((answers[m] == answers[n]).all(axis=1).astype(int) for m in names if m != n)
            counts = np.bincount(same, minlength=len(names))
            out[label][n] = {k: float(c) / len(ls) for k, c in enumerate(counts) if c}
    return out


relevance_by_table = relevance_tables(TABLES, by_name)
blocks = []
for label, dists in relevance_by_table.items():
    dist_levels = sorted({k for dd in dists.values() for k, v in dd.items() if v >= 0.005})
    head = ["anchored op", *(str(k) for k in dist_levels)]
    table_rows = [
        [f"<code>{n}</code>", *(f"{dd[k]:.0%}" if dd.get(k, 0) >= 0.005 else "·" for k in dist_levels)]
        for n, dd in dists.items()
    ]
    blocks.append(table_html(head, table_rows, f"**Table {label}**"))

figure_html(
    "<br>".join(blocks),
    caption="""
        **Op-relevance tables.**
        For each anchored op, the share of its lines on which $k$ other ops of the table give the same answer. A dot means under half a percent.
    """,
)

# %%
alone = {label: {n: relevance_by_table[label][n][0] for n in names} for label, names in TABLES.items()}
trio_alone = [alone["A+"][n] for n in HSV_TRIO]

rf"""
Widening the table makes `mix` much less distinctive. Today it is alone in its answer on {alone["current"]["mix"]:.0%} of lines; under table A that falls to {alone["A"]["mix"]:.0%}, because `hsvmix` agrees with it on a third of lines, the ones whose operands are close in hue. We want that: the per-line relevance prediction needs more than one level to read, and `hsvmix` sharing the answers of `mix` on a known set of lines is what supplies it.

`difference` is the most distinctive op in table A. It also has the most redder-than-both lines ({scouts["difference"].redder:,}, over a quarter of its lines), so it is where the blind-span question (E3 of ex-2.2.3) would get the most lines.

Adding the HSV trio costs the rest of the table little. Each of the three is alone in its answer on {min(trio_alone):.0%} to {max(trio_alone):.0%} of its lines, with `difference` the most distinctive ops in A+, and the op that loses most is `lighten`, from {alone["A"]["lighten"]:.0%} alone under A to {alone["A+"]["lighten"]:.0%}.

### What we make of it

**Does it change the design?** Yes, in three places.

The table should be A+: keep `mix`, `screen`, `multiply`, `lighten`, and `darken`, drop `add`, add `difference`, `exclusion`, and `hsvmix`, and carry `hue-hsv`, `sat-hsv`, and `value-hsv` as a marked subset. We drop `add` because a fifth of its pairs go to white and it is the least sensitive op by either rule. The three commutative additions give the E8 split both kinds of op to read, as the item asks. The HSV trio are there for two other reasons: they are the first ops in the grammar that read operand order, which natural language does everywhere, and they are where the answer moves furthest once the red operand loses its red. Their reads should be reported as a subset, so that any behavior of their own can be set aside without touching the rest of the table, and the probe draw should walk every color as op2 as well as op1 for them.

The removal statistic should be a distance, scored on the lines that can show it. E8's one-level rule reads sensitivity: whether a small change in the red operand reaches the answer, which is what an intervention of that size would show. The question the anchored-op experiments ask is larger: once *red* is gone from the stream, can the model still mix red? On that question the line to read is one where zeroing the red operand's R moves the answer far, and the statistic is how far the model's answer moves under the intervention, against that distance, rather than whether it is exactly right. A model that answers almost-purple for red and blue has kept most of *red*; one that answers gray has not, and exact match cannot tell the two apart. So the prereg should carry the to-zero distance per line as its prediction, and score removal as the answer's distance from the correct one, with exact match beside it for continuity with ex-2.2.3. What "the operand with *red* removed" means in color terms is itself a choice; R to zero is the simplest, and the one E8 used.

The corpus should round stochastically, and the labeller should read the whole line. The two pilots, [stochastic rounding](/docs/m2/ex-2.2.5/report.py) and [the whole-span labeller](/docs/m2/ex-2.2.6/report.py), found that neither costs anything on the anchoring side: placement stays in production's band under both, and the whole-line pull has no task cost. Each pilot recommended keeping ex-2.2.3's setting, on the ground that it keeps every read simple. The ground we now prefer is comparability with M3, where a document-level label says nothing about position and a language target is a distribution that no model matches exactly, so exact match is never the statistic. Both changes bring the grammar closer to that, and they fit the reads above: under stochastic rounding an answer is a distribution, so the removal statistic is how much of its mass moves, and the dependence reads are graded the same way. The cost is exact match as the main statistic, and the stochastic pilot names the reads to use instead: expected exact match against the holdout ceiling, and the calibration of the answer mass against the rule.

The next preregistered experiment is then a handover. It runs ex-2.2.3's recipe on table A+ with the new corpus and labeller, against ex-2.2.3's twenty seeds as the reference, and keeps `mix` as the reference op with `hsvmix` beside it, so that a later experiment can make `hsvmix` the reference if it behaves. One arm each with only the corpus or only the labeller changed would say which change moved what, if anything moves.

**What we would do differently.** The hue, saturation, and brightness ops were the headline of the item, and the one-level rule counts them with the saturating ops: they read one attribute of one operand, whereas *red* as the labeller defines it, r·(1 − g/2 − b/2), is a magnitude that hue and saturation do not see. Read by the to-zero distance they are the ops that need *red* most. We would have started from the to-zero rule and the distance, which are the reads the removal question asks for, and treated the one-level rule as a check that the intervention is large enough to reach the answer.

**What this does not settle.** Whether the model learns eleven ops as well as it learned six is a training question. E4 of ex-2.2.3 found the operand cube less linearly decodable at six ops than at three, with lines per op as a confound. Eleven ops at the same corpus size would sharpen that confound, so the prereg should decide whether to hold lines per op fixed instead. Nor does it say how a model treats an op that reads operand order, which is why the HSV trio are a marked subset. The scouting questions still to run may move the operating point, but they do not move the table.

## Method

Everything in this section is computed on the grid when the notebook runs, using [`sca.data.ops`](/src/sca/data/ops.py): the six ops of ex-2.2.3 as `OPS` and the nine candidates as `CANDIDATES`, both snapped to the 216-color grid by the same `snap`.

**Lines.** There are 46,656 ordered pairs of grid colors, each one an (op1, op2) line, and 23,436 unordered pairs, which we use where a statistic is symmetric. A *red line* has dose ≥ 0.8, where dose is the larger of the two operand rednesses, as in ex-2.2.3. The shared probe draw is the one from ex-2.2.3: every color as op1 against 27 partners drawn once from the grid, with the same partners for every op other than `mix`. The probe set for `mix` is the 5,832 closed pairs from D2.1.

**Spread** is the Shannon entropy, in bits, of the distribution of answers over the 216 grid colors, taken over unordered pairs. **Channel marginals** are the share of unordered pairs whose answer has each grid level in each channel.

**Dependence** follows E8. For a red line, take the redder operand, lower its R by one grid level (3) or to zero, leave everything else alone, and ask whether the snapped answer changes. We report it as a share of the lines in the set named. Under **stochastic rounding** the answer of a line is the distribution `answer_dist` gives, the product over channels of a two-point distribution on the levels either side of the raw value, and the read is the total variation distance between the distributions of the line and of its lowered version, averaged over the set. **Moves by** is the Euclidean distance in the unit cube between the snapped answers of the line and of its lowered version, averaged the same way. **Reads op1 / op2** replaces that operand with a color drawn uniformly from the grid (seed 0, one draw per line, every fifth line) and asks whether the answer changes.

**Commutative** is the share of unordered pairs on which the op gives the same answer in both orders. `hsvmix` falls short of 1 on the pairs whose hues are opposite, where the circular mean is undefined and the rule keeps the hue of op1.

**Op-relevance** is `relevance` from the same module, now over an arbitrary table: for each line, the number of other ops in the table that give the same answer as the anchored op. Ex-2.2.3 counted unordered pairs; the tables here count ordered pairs, so that the ops that read operand order are counted on both orders.
"""
