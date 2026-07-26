"""
Figures for the landmark probe maps.

:mod:`sca.compute.geometry` scores a leave-one-out probe at every depth × grammar
landmark, per surface form and per target (the two operands and their mix). The
heatmap view of a map averages its three RGB channels into one number per cell;
the traces here keep the channels apart — one step-line each — and stack the
depths vertically, so a column of panels reads as a column of the heatmap and the
mean of a panel's three lines is the matching heatmap cell.

The x axis is ordinal: each landmark is one character position, and consecutive
landmarks are usually *not* consecutive characters. Two things follow, and the
drawing conventions here exist for them. Values are drawn as plateaus joined by
risers (:func:`~mini.vis.smooth_step`), because a straight line between landmarks
would claim measurements that were never made. And risers that jump a stretch of
unprobed text are *elided* — drawn in lighter ink — so the reader can tell an
interpolation from a step between neighbours.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.layout_engine import ConstrainedLayoutEngine

from mini.vis import light_dark, smooth_step, smooth_step_area, smooth_step_band
from sca.data.mixed_vocab import GAP_RISERS, LANDMARKS, OPERATORS, SPAN_RISERS

FORMS = ("named", "hex")
TARGETS = ("op1", "op2", "mix")

LANDMARK_LABELS = {
    "o1s0": "$a_{1}$", "o1s1": "$a_{2}$", "o1e1": "$a_{n-1}$", "o1e0": "$a_{n}$",
    "o2s0": "$b_{1}$", "o2s1": "$b_{2}$", "o2e1": "$b_{n-1}$", "o2e0": "$b_{n}$",
    "as0": "$r_{1}$", "as1": "$r_{2}$", "ae1": "$r_{n-1}$", "ae0": "$r_{n}$",
    "plus": "+", "eq": "=", "pre": " ",
}  # fmt: skip
"""Tick labels for :data:`~sca.data.mixed_vocab.LANDMARKS`: operand 1 is $a$, operand 2
is $b$, the answer is $r$, subscripted by character position from whichever end the
landmark is anchored to. The pre-answer space is left blank — it is a space."""

MAJOR_LANDMARKS = tuple(lm for lm in LANDMARKS if lm in OPERATORS and lm != "pre")
"""The landmarks that divide the equation into its parts: ``+`` and ``=``. They get major
ticks and a vertical rule, so every panel is read against the same three-part grammar.
``pre`` is an operator landmark but not a divider, so it stays minor."""

MINOR_LANDMARKS = tuple(lm for lm in LANDMARKS if lm not in MAJOR_LANDMARKS)

ELIDED_RISERS = {
    "named": frozenset(GAP_RISERS | SPAN_RISERS),
    "hex": frozenset(GAP_RISERS),
}
"""Per form, the risers that cross characters no landmark measures. Both forms share the
fixed grammar's unprobed spaces (``GAP_RISERS``). Only the named form also skips word
middles (``SPAN_RISERS``): its words are variable-length, so the landmark scheme samples
two characters from each end and passes over whatever sits between, while hex's fixed-width
digits leave nothing in the middle to miss."""


def channel_colors() -> tuple[str, str, str]:
    """Line color per probe channel. Color is data here: each channel gets its own hue."""
    return light_dark(("#d1495b", "#2a9d5c", "#3b6fd4"), ("#ff6b7d", "#4fd07a", "#6ea3ff"))


def mean_color() -> str:
    """Ink for the channel mean.

    Deliberately hueless: the channels own the hues, and the mean is a summary of them
    rather than a fourth thing measured alongside them.
    """
    return light_dark("#444", "#ccc")


def aliased_risers(stack: np.ndarray) -> frozenset[int]:
    """Risers between landmarks that resolved to the same character, so their columns match bit for bit.

    A riser there would span zero real distance and read as a wide plateau, inflating how
    much of the equation looks flat — so break the line instead. Pass the result as
    ``breaks``. The current landmark scheme has no such pair (the answer is sampled like
    an operand, two characters in from each end), so this is normally empty; it is a
    safety net for a scheme that reintroduces one.
    """
    return frozenset(i for i in range(stack.shape[1] - 1) if np.array_equal(stack[:, i], stack[:, i + 1]))


def draw_traces(
    ax: Axes,
    values: np.ndarray,
    *,
    elide: Iterable[int] = (),
    breaks: Iterable[int] = (),
    colors: Sequence[str] | None = None,
    lw: float | Sequence[float] = 1.0,
    ramp: float = 0.5,
    fade: float = 0.3,
    fill: Literal["mean", "channel"] | None = "mean",
    fill_alpha: float | None = None,
    spread: np.ndarray | None = None,
    band: Literal["mean", "channel"] | None = "mean",
    band_alpha: float | None = None,
) -> None:
    """Draw one panel: a step-line per channel of *values*, shaped ``(landmark, channel)``.

    *ramp* below 1 leaves a flat shoulder at each landmark, which is what makes the
    discrete character positions legible. *elide* and *breaks* are the two ways a pair of
    landmarks can fail to be neighbours — see :func:`~mini.vis.smooth_step`.

    *spread* holds the replicates *values* summarises, shaped ``(replicate, landmark,
    channel)`` — the per-seed maps behind a seed mean. Their min–max is drawn as a band
    behind the lines, turning "and the other seeds agree" from a claim in the prose into
    something the reader can see. *band* chooses whose spread: ``"mean"`` bands the channel
    mean alone, which is the quantity the summary fill and the heatmap both report, and
    keeps the panel to one extra shape; ``"channel"`` bands each channel in its own hue,
    which answers whether a *particular* channel's position is stable at the cost of three
    overlapping areas.

    *fill* shades the area under the traces, which gives the panel a height a reader can
    take in without tracing a line. ``"mean"`` fills once, in neutral grey, under the mean
    of the channels — the quantity a probe heatmap of the same data would show as one
    cell's brightness, so the panel carries both readings at once. ``"channel"`` fills
    under each channel in its own hue instead; the areas then overlap, and their combined
    silhouette is the channel *maximum* rather than the mean, so the summary reading is
    lost wherever the channels disagree.

    *fill_alpha* defaults per theme. A pale fill lightens a white page but has to lighten a
    near-black one by more to travel the same visual distance, so the dark figure needs a
    stronger value to read as the same shade of quiet.
    """
    fill_alpha = light_dark(0.18, 0.22) if fill_alpha is None else fill_alpha
    band_alpha = light_dark(0.3, 0.32) if band_alpha is None else band_alpha
    colors = colors or channel_colors()
    lws = np.broadcast_to(lw, values.shape[1])
    breaks, elide = set(breaks), set(elide)
    x = range(len(values))

    if fill == "mean":
        smooth_step_area(ax, x, values.mean(1), ramp=ramp, breaks=breaks, color=mean_color(), alpha=fill_alpha)
    if spread is not None and band == "mean":
        means = spread.mean(2)  # (replicate, landmark)
        lo, hi = means.min(0), means.max(0)
        smooth_step_band(ax, x, lo, hi, ramp=ramp, breaks=breaks, color=mean_color(), alpha=band_alpha)
    for c in range(values.shape[1]):
        if fill == "channel":
            smooth_step_area(ax, x, values[:, c], ramp=ramp, breaks=breaks, color=colors[c], alpha=fill_alpha)
        if spread is not None and band == "channel":
            lo, hi = spread[:, :, c].min(0), spread[:, :, c].max(0)
            smooth_step_band(ax, x, lo, hi, ramp=ramp, breaks=breaks, color=colors[c], alpha=band_alpha)
        smooth_step(
            ax, x, values[:, c],
            ramp=ramp, breaks=breaks, elide=elide, fade=fade, color=colors[c], lw=lws[c],
        )  # fmt: skip


def style_trace_panel(ax: Axes, *, ylim: tuple[float, float] = (-0.2, 1.2)) -> None:
    """Panel furniture for one depth's traces: rules at the dividers, no frame, ticks turned inward.

    The panels sit flush in a vertical stack, so a frame around each would draw a ladder of
    doubled lines up the figure. The unit gridlines and the inward y ticks carry the scale
    instead, and the dividers continue across the whole stack.
    """
    ax.vlines([LANDMARKS.index(lm) for lm in MAJOR_LANDMARKS], -1, 2, "#8882", lw=0.5)
    ax.set(ylim=ylim, xlim=(-0.5, len(LANDMARKS) - 0.5), yticks=[0, 1], yticklabels=[])
    ax.tick_params(axis="x", length=0, which="both")
    ax.tick_params(axis="y", left=True, right=True, direction="in")
    ax.spines[:].set_visible(False)
    ax.grid(which="major", c="#888", alpha=0.2)


def label_landmarks(ax: Axes, *, sparse: bool = False, labels: bool = True) -> None:
    """Put the landmark names on *ax*'s x axis, as major ticks for the dividers and minor for the rest.

    *sparse* drops the inner characters of each token ($a_2$, $a_{n-1}$, …), keeping only
    its first and last. Use it when panels are narrow enough that the full set collides:
    the dropped labels sit one slot from a label that survives, so the axis still reads,
    and the traces are still drawn at every landmark. *labels* off keeps the tick
    positions but writes nothing, which is what a panel above the bottom row wants —
    matplotlib would otherwise fall back to numbering the landmarks.

    The dividers hang on a second line below the characters. They sit one slot from their
    neighbours ($a_n$, ``+``, $b_1$ are consecutive landmarks), so on one line they would
    touch; on two, the axis reads as characters over the grammar that groups them.
    """

    def ticks(lms: Sequence[str]) -> tuple[list[int], list[str]]:
        return [LANDMARKS.index(lm) for lm in lms], [LANDMARK_LABELS[lm] if labels else "" for lm in lms]

    minor = [lm for lm in MINOR_LANDMARKS if not sparse or lm.endswith("0")]
    ax.set_xticks(*ticks(MAJOR_LANDMARKS), fontsize="x-small")
    ax.set_xticks(*ticks(minor), minor=True, fontsize="xx-small")
    ax.tick_params(axis="x", which="major", pad=10)
    ax.tick_params(axis="x", which="minor", pad=1)


def draw_depth_stack(
    sfig: SubFigure,
    stack: np.ndarray,
    *,
    elide: Iterable[int] = (),
    title: str | None = None,
    depth_labels: bool = True,
    tick_labels: bool = True,
    sparse_ticks: bool = False,
    **trace_kwargs,
) -> None:
    """Fill *sfig* with one panel per depth of *stack*, shaped ``(depth, landmark, channel)``.

    A four-dimensional *stack* is read as ``(replicate, depth, landmark, channel)`` — the
    per-seed maps rather than their average. The panels then plot the mean over replicates
    and band their spread, so nothing upstream has to decide which of the two to keep.

    Depth 0 (the embedding) goes at the bottom and the last layer at the top, matching the
    heatmaps, so height on the page means depth in the network in both figures.

    The panels are pushed flush against each other: a stack is one picture of how a probe
    fares through the network, and a gap between rows would break it into five pictures.
    Blocks stay apart because each is its own sub-figure.
    """
    spread = stack if stack.ndim == 4 else None
    means = stack.mean(axis=0) if spread is not None else stack
    # Alias detection runs on the raw scores, before the floor at 0. A probe that does worse
    # than predicting the mean scores below zero, and clipping would make every such landmark
    # identically 0 — indistinguishable from two landmarks that really are one character.
    breaks = aliased_risers(means)

    engine = sfig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(hspace=0, h_pad=0.01, wspace=0)
    axes = sfig.subplots(len(means), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for row, ax in enumerate(axes):
        depth = len(means) - 1 - row
        draw_traces(
            ax,
            np.clip(means[depth], 0, 1),
            elide=elide,
            breaks=breaks,
            spread=None if spread is None else np.clip(spread[:, depth], 0, 1),
            **trace_kwargs,
        )
        style_trace_panel(ax)
        if depth_labels:
            ax.set_ylabel(f"{depth}", fontsize=8)
    if title:
        axes[0].set_title(title, fontsize=9)
    label_landmarks(axes[-1], sparse=sparse_ticks, labels=tick_labels)


def probe_trace_grid(
    stacks: Mapping[tuple[str, str], np.ndarray],
    *,
    forms: Sequence[str] = FORMS,
    targets: Sequence[str] = TARGETS,
    form_axis: Literal["row", "col"] = "col",
    elided: Mapping[str, Iterable[int]] | None = None,
    width: float = 8.0,
    panel_height: float = 0.6,
    ylabel: str | None = None,
    sparse_ticks: bool | None = None,
    **trace_kwargs,
) -> Figure:
    """Lay out a depth stack per (form, target) pair, as a grid of blocks.

    *stacks* maps ``(form, target)`` to an array of shape ``(depth, landmark, channel)``, or
    ``(replicate, depth, landmark, channel)`` to band the spread across replicates.
    *form_axis* picks which way the grid runs: ``"row"`` puts the forms on separate rows
    and the targets across the columns, which is the heatmap's own arrangement — the two
    forms then sit one above the other over a shared x axis, which is the comparison the
    figure exists to invite. ``"col"`` transposes that, giving each form a column; with a
    single target it is the wide, one-target-per-figure layout.

    *sparse_ticks* defaults to whether the grid is more than two blocks wide, since that is
    when the full landmark labels start to collide.
    """
    rows, cols = (forms, targets) if form_axis == "row" else (targets, forms)
    pair = (lambda r, c: (r, c)) if form_axis == "row" else (lambda r, c: (c, r))
    elided = ELIDED_RISERS if elided is None else elided
    n_depth = next(iter(stacks.values())).shape[-3]  # from the right: depth, landmark, channel
    sparse_ticks = len(cols) > 2 if sparse_ticks is None else sparse_ticks

    fig = plt.figure(figsize=(width, len(rows) * (n_depth * panel_height + 0.3) + 0.3))
    sfigs = fig.subfigures(len(rows), len(cols), squeeze=False)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            form, target = pair(r, c)
            draw_depth_stack(
                sfigs[i, j],
                stacks[form, target],
                elide=elided.get(form, ()),
                title=f"{form} · {target}",
                depth_labels=j == 0,
                tick_labels=i == len(rows) - 1,
                sparse_ticks=sparse_ticks,
                **trace_kwargs,
            )
    if ylabel:
        fig.supylabel(ylabel, fontsize=9)
    return fig
