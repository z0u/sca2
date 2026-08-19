"""
Figures for the landmark probe maps.

:mod:`sca.compute.geometry` scores a leave-one-out probe at every depth × grammar landmark, per surface form and per target (the two operands and their mix). The heatmap view of a map averages its three RGB channels into one number per cell; the traces here keep the channels apart — one step-line each — and stack the depths vertically, so a column of panels reads as a column of the heatmap and the mean of a panel's three lines is the matching heatmap cell.

The x axis is ordinal: each landmark is one character position, and consecutive landmarks are usually *not* consecutive characters. Two things follow, and the drawing conventions here exist for them. Values are drawn as plateaus joined by risers (:func:`~mini.vis.smooth_step`), because a straight line between landmarks would claim measurements that were never made. And risers that jump a stretch of unprobed text are *elided* — drawn in lighter ink — so the reader can tell an interpolation from a step between neighbours.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.transforms import Transform

from mini.vis import light_dark, mix, page_color, smooth_step, smooth_step_area
from sca.data.mixed_vocab import GAP_RISERS, LANDMARKS, OPERATORS, SPAN_RISERS

FORMS = ("named", "hex")
TARGETS = ("op1", "op2", "mix")

LANDMARK_LABELS = {
    "o1s0": "$a_{1}$", "o1s1": "$a_{2}$", "o1e1": "$a_{n-1}$", "o1e0": "$a_{n}$",
    "o2s0": "$b_{1}$", "o2s1": "$b_{2}$", "o2e1": "$b_{n-1}$", "o2e0": "$b_{n}$",
    "as0": "$r_{1}$", "as1": "$r_{2}$", "ae1": "$r_{n-1}$", "ae0": "$r_{n}$",
    "plus": "+", "eq": "=",
    "o1sp": " ", "plussp": " ", "o2sp": " ", "pre": " ",
}  # fmt: skip
"""Tick labels for :data:`~sca.data.mixed_vocab.LANDMARKS`: operand 1 is $a$, operand 2
is $b$, the answer is $r$, subscripted by character position from whichever end the
landmark is anchored to. The four delimiter spaces are left blank — they are spaces, and
a tick mark alone places them, sitting as they do either side of ``+`` and ``=``."""

MAJOR_LANDMARKS = tuple(lm for lm in LANDMARKS if lm in ("plus", "eq"))
"""The landmarks that divide the equation into its parts: ``+`` and ``=``. They get major
ticks and a vertical rule, so every panel is read against the same three-part grammar.
The delimiter spaces are operator landmarks but not dividers, so they stay minor."""

MINOR_LANDMARKS = tuple(lm for lm in LANDMARKS if lm not in MAJOR_LANDMARKS)

ELIDED_RISERS = {
    "named": frozenset(GAP_RISERS | SPAN_RISERS),
    "hex": frozenset(GAP_RISERS),
}
"""Per form, the risers that cross characters no landmark measures. ``GAP_RISERS`` is now
empty — the fixed grammar, delimiter spaces included, is measured character by character —
so hex elides nothing at all: its four-character operands and answers are landmarks
throughout. Only the named form skips anything (``SPAN_RISERS``): its words are
variable-length, so the landmark scheme samples two characters from each end and passes
over whatever sits between, anywhere from 0 to 22 characters depending on the line."""


def channel_colors() -> tuple[str, str, str]:
    """Line color per probe channel. Color is data here: each channel gets its own hue."""
    return light_dark(("#d1495b", "#2a9d5c", "#3b6fd4"), ("#ff6b7d", "#4fd07a", "#6ea3ff"))


def mean_color() -> str:
    """Ink for the channel mean.

    Deliberately hueless: the channels own the hues, and the mean is a summary of them rather than a fourth thing measured alongside them.
    """
    return light_dark("#444", "#ccc")


def transfer_color() -> str:
    """Ink for a zero-shot cross-form reading, drawn over the within-form area.

    Amber, which is outside the R/G/B hues :func:`channel_colors` spends: a cross-form trace is a whole-value reading, and a red line here would invite the channel reading the rest of the report has trained.
    """
    return light_dark("#b06a00", "#e0a458")


def aliased_risers(stack: np.ndarray) -> frozenset[int]:
    """Risers between landmarks that resolved to the same character, so their columns match bit for bit.

    A riser there would span zero real distance and read as a wide plateau, inflating how much of the equation looks flat — so break the line instead. Pass the result as ``breaks``. The current landmark scheme has no such pair (the answer is sampled like an operand, two characters in from each end), so this is normally empty; it is a safety net for a scheme that reintroduces one.
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
    edge_alpha: float = 0.55,
    spread: np.ndarray | None = None,
    transform: Transform | None = None,
    clip_on: bool = True,
) -> None:
    """Draw one panel: a step-line per channel of *values*, shaped ``(landmark, channel)``.

    *ramp* below 1 leaves a flat shoulder at each landmark, which is what makes the discrete character positions legible. *elide* and *breaks* are the two ways a pair of landmarks can fail to be neighbours — see :func:`~mini.vis.smooth_step`.

    *fill* shades the area under the traces, which gives the panel a height a reader can take in without tracing a line. ``"mean"`` fills once, in neutral grey, under the mean of the channels — the quantity a probe heatmap of the same data would show as one cell's brightness, so the panel carries both readings at once. ``"channel"`` fills under each channel in its own hue instead; the areas then overlap, and their combined silhouette is the channel *maximum* rather than the mean, so the summary reading is lost wherever the channels disagree.

    *spread* holds the replicates *values* summarises, shaped ``(replicate, landmark, channel)`` — the per-seed maps behind a seed mean. It turns "and the other seeds agree" from a claim in the prose into something the reader can see: the fill gets an outline at *edge_alpha*, and the replicate minimum and maximum are drawn as fainter hairlines. Where the replicates agree all three coincide, so the panel reads as a crisply outlined area; where they part, two hairlines drift off a silhouette that stays where the summary is.

    That hierarchy is the point of drawing the spread as lines rather than as more shading. Three stacked tones (under the minimum, the summary, the maximum) put only a few points of opacity between neighbouring levels, which is below what a 0.45in panel can show; and the more obvious arrangement — a min–max ribbon over a summary fill — makes their *overlap* the darkest part of the panel, so a reader sees a dark stripe floating between two lighter ones with nothing in the data to match it.

    *transform* places the panel somewhere other than the Axes' own data area. Pass an offset (``Affine2D().translate(0, i) + ax.transData``) to stack several panels in one Axes, each on its own 0–1 row, and turn *clip_on* off so a trace that runs along a row's boundary keeps its full stroke instead of half of it.

    *fill_alpha* defaults per theme. A pale fill lightens a white page but has to lighten a near-black one by more to travel the same visual distance, so the dark figure needs a stronger value to read as the same shade of quiet.
    """
    fill_alpha = light_dark(0.18, 0.22) if fill_alpha is None else fill_alpha
    colors = colors or channel_colors()
    lws = np.broadcast_to(lw, values.shape[1])
    breaks, elide = set(breaks), set(elide)
    x = range(len(values))
    # Matplotlib reads an explicit transform=None as "already set", which would cost the
    # artist its default of ax.transData, so an absent transform has to be absent.
    place: dict[str, Any] = {"clip_on": clip_on} | ({} if transform is None else {"transform": transform})

    def shade(summary: np.ndarray, replicates: np.ndarray | None, color: str) -> None:
        # Opaque, not translucent: these three lines coincide wherever the replicates agree,
        # which is most of the panel, and stacked translucent copies would put the heaviest
        # ink exactly where the least is happening. The price of pre-computing a color is
        # having to say what it sits on, and the three lines sit on different things — the
        # fill runs from the floor to the summary, so the minimum is always inside it, the
        # maximum always above it, and the summary rides its edge with half of the stroke
        # either side. Tint each against its own surround or a line that fades toward the
        # page reads as a pale streak scratched across the fill.
        page = page_color()
        floor = mix(page, color, fill_alpha)  # the shade the fill paints, once composited

        def edge(y: np.ndarray, weight: float, over: str) -> None:
            # Elided like the traces: a summary line that crossed an unprobed stretch solidly
            # would claim a neighbourhood the measurements it summarises decline to claim.
            ink = mix(over, color, weight)
            smooth_step(
                ax, x, y,
                ramp=ramp, breaks=breaks, elide=elide, fade=mix(over, ink, fade),
                color=ink, lw=0.5, **place,
            )  # fmt: skip

        smooth_step_area(ax, x, summary, ramp=ramp, breaks=breaks, color=color, alpha=fill_alpha, **place)
        if replicates is None:
            return
        # Envelope first, so where the three coincide the summary's own edge is what survives.
        edge(replicates.min(0), edge_alpha * 0.55, floor)
        edge(replicates.max(0), edge_alpha * 0.55, page)
        edge(summary, edge_alpha, mix(page, floor, 0.5))

    if fill == "mean":
        shade(values.mean(1), None if spread is None else spread.mean(2), mean_color())
    for c in range(values.shape[1]):
        if fill == "channel":
            shade(values[:, c], None if spread is None else spread[:, :, c], colors[c])
        smooth_step(
            ax, x, values[:, c],
            ramp=ramp, breaks=breaks, elide=elide, fade=fade, color=colors[c], lw=lws[c], **place,
        )  # fmt: skip


def style_trace_panel(ax: Axes, *, ylim: tuple[float, float] = (-0.2, 1.2)) -> None:
    """Panel furniture for one depth's traces: rules at the dividers, no frame, ticks turned inward.

    The panels sit flush in a vertical stack, so a frame around each would draw a ladder of doubled lines up the figure. The unit gridlines and the inward y ticks carry the scale instead, and the dividers continue across the whole stack.
    """
    ax.vlines([LANDMARKS.index(lm) for lm in MAJOR_LANDMARKS], -1, 2, "#8882", lw=0.5)
    ax.set(ylim=ylim, xlim=(-0.5, len(LANDMARKS) - 0.5), yticks=[0, 1], yticklabels=[])
    ax.tick_params(axis="x", length=0, which="both")
    ax.tick_params(axis="y", left=True, right=True, direction="in")
    ax.spines[:].set_visible(False)
    ax.grid(which="major", c="#888", alpha=0.2)


def label_scale(axes: "Sequence[Axes] | np.ndarray", *, show: bool = True) -> None:
    """Number the 0 and 1 gridlines on the right of the bottom panel of a stack.

    Every panel in the figure is on the same scale, so one of them can say what it is and the rest stay clean: numbering all thirty spends a lot of ink on a fact stated once. It goes on the right because the left margin already carries the depth numbers.

    *show* off still writes the numbers and still lets the layout engine measure them, but paints them in no color. Tick labels take width from the block they sit in, so a block that alone carried real ones would end up narrower than the block above it and the two landmark axes would stop lining up — which is the comparison the stacked layout is for.
    """
    for ax in axes:
        ax.tick_params(axis="y", labelleft=False, labelright=False)
    key = axes[-1]
    key.set_yticks([0, 1], ["0", "1"])  # the panels share a y axis, so this reaches them all
    key.tick_params(axis="y", labelright=True, labelsize=7, pad=2, **({} if show else {"labelcolor": "none"}))


def label_landmarks(ax: Axes, *, sparse: bool = False, labels: bool = True) -> None:
    """Put the landmark names on *ax*'s x axis, as major ticks for the dividers and minor for the rest.

    *sparse* drops the inner characters of each token ($a_2$, $a_{n-1}$, …), keeping only its first and last. Use it when panels are narrow enough that the full set collides: the dropped labels sit one slot from a label that survives, so the axis still reads, and the traces are still drawn at every landmark. *labels* off keeps the tick positions but writes nothing, which is what a panel above the bottom row wants — matplotlib would otherwise fall back to numbering the landmarks.

    The dividers hang on a second line below the characters. They sit one slot from their neighbours ($a_n$, ``+``, $b_1$ are consecutive landmarks), so on one line they would touch; on two, the axis reads as characters over the grammar that groups them.
    """

    def ticks(lms: Sequence[str]) -> tuple[list[int], list[str]]:
        return [LANDMARKS.index(lm) for lm in lms], [LANDMARK_LABELS[lm] if labels else "" for lm in lms]

    # Sparse keeps a token's outer characters (…0) and always the delimiter spaces: those
    # carry no label — a space has nothing to print — so a tick mark is the only thing
    # placing them, and they are where a named operand's value actually reads.
    minor = [lm for lm in MINOR_LANDMARKS if not sparse or lm.endswith("0") or lm in OPERATORS]
    ax.set_xticks(*ticks(MAJOR_LANDMARKS), fontsize="x-small")
    ax.set_xticks(*ticks(minor), minor=True, fontsize="xx-small")
    ax.tick_params(axis="x", which="major", pad=10)
    # A mark under each labelled landmark, and only there — the panels above carry tick
    # positions without furniture (`style_trace_panel` zeroes the length). The delimiter
    # spaces need it most: their label is blank, so without a mark nothing places them.
    ax.tick_params(axis="x", which="minor", pad=1, length=2 if labels else 0)


def draw_depth_stack(
    sfig: SubFigure,
    stack: np.ndarray,
    *,
    elide: Iterable[int] = (),
    title: str | None = None,
    depth_labels: bool = True,
    tick_labels: bool = True,
    sparse_ticks: bool = False,
    scale_key: bool = False,
    **trace_kwargs,
) -> None:
    """Fill *sfig* with one panel per depth of *stack*, shaped ``(depth, landmark, channel)``.

    *scale_key* numbers this block's y axis — see :func:`label_scale`. Exactly one block in a figure should carry it, and every other block still reserves the room for it.

    A four-dimensional *stack* is read as ``(replicate, depth, landmark, channel)`` — the per-seed maps rather than their average. The panels then plot the mean over replicates and shade their spread, so nothing upstream has to decide which of the two to keep.

    Depth 0 (the embedding) goes at the bottom and the last layer at the top, matching the heatmaps, so height on the page means depth in the network in both figures.

    The panels are pushed flush against each other: a stack is one picture of how a probe fares through the network, and a gap between rows would break it into five pictures. Blocks stay apart because each is its own sub-figure.
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
    label_scale(axes, show=scale_key)


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

    *stacks* maps ``(form, target)`` to an array of shape ``(depth, landmark, channel)``, or ``(replicate, depth, landmark, channel)`` to shade the spread across replicates. *form_axis* picks which way the grid runs: ``"row"`` puts the forms on separate rows and the targets across the columns, which is the heatmap's own arrangement — the two forms then sit one above the other over a shared x axis, which is the comparison the figure exists to invite. ``"col"`` transposes that, giving each form a column; with a single target it is the wide, one-target-per-figure layout.

    *sparse_ticks* defaults to whether the grid is more than two blocks wide, since that is when the full landmark labels start to collide.

    The bottom-right block carries the scale key — the only 0 and 1 numbered in the figure, on the right where nothing else is written. Every panel is on the same scale, so the reader needs it once, near the corner they end up in after reading the landmark axis.
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
                scale_key=(i, j) == (len(rows) - 1, len(cols) - 1),
                **trace_kwargs,
            )
    if ylabel:
        fig.supylabel(ylabel, fontsize=9)
    return fig


def draw_transfer_stack(
    sfig: SubFigure,
    within: np.ndarray,
    cross: np.ndarray,
    *,
    elide: Iterable[int] = (),
    title: str | None = None,
    depth_labels: bool = True,
    tick_labels: bool = True,
    sparse_ticks: bool = False,
    scale_key: bool = False,
) -> None:
    """One block of :func:`transfer_trace_grid`: within-form area with the cross-form area over it.

    Both arrays are shaped ``(replicate, depth, landmark)`` — the channel-mean maps, since the question is whether a value transfers at all rather than which channel does. The within-form reading is the grey area (the same ink the per-channel figure gives a channel mean) and the cross-form reading is drawn over it in amber, so the amber fraction of the grey *is* the transfer ratio ρ, read off the panel rather than tabulated.

    Negative scores clip to the floor, as everywhere in these figures: a probe that does worse than predicting the mean has failed, and how much worse is not a finer grade of failure. Where that matters — a cross-form reading pinned at zero could be −0.01 or −30 — the caption has to say so.
    """
    means = (within.mean(0), cross.mean(0))
    breaks = aliased_risers(means[0])

    engine = sfig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(hspace=0, h_pad=0.01, wspace=0)
    axes = sfig.subplots(len(means[0]), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for row, ax in enumerate(axes):
        depth = len(means[0]) - 1 - row
        for series, color, lw, replicates in (
            (means[0], mean_color(), 0.9, within),
            (means[1], transfer_color(), 1.3, cross),
        ):
            draw_traces(
                ax,
                np.clip(series[depth], 0, 1)[:, None],
                elide=elide,
                breaks=breaks,
                colors=[color],
                lw=lw,
                fill="channel",
                spread=np.clip(replicates[:, depth], 0, 1)[:, :, None],
            )
        style_trace_panel(ax)
        if depth_labels:
            ax.set_ylabel(f"{depth}", fontsize=8)
    if title:
        axes[0].set_title(title, fontsize=9)
    label_landmarks(axes[-1], sparse=sparse_ticks, labels=tick_labels)
    label_scale(axes, show=scale_key)


def transfer_trace_grid(
    within: Mapping[tuple[str, str], np.ndarray],
    cross: Mapping[tuple[str, str], np.ndarray],
    *,
    forms: Sequence[str] = FORMS,
    targets: Sequence[str] = TARGETS,
    titles: Mapping[str, str] | None = None,
    elided: Mapping[str, Iterable[int]] | None = None,
    width: float = 8.0,
    panel_height: float = 0.45,
    ylabel: str | None = None,
) -> Figure:
    """A block per (form, target): what the form's own probe reads, and what the other form's does.

    Both mappings are keyed by the form the probes are *applied* to — which is also the form whose landmarks the x axis belongs to, and whose within-form score is ρ's denominator. Rows are forms, columns are targets, matching the per-channel figure's arrangement so the two can be read against each other.

    *titles* overrides a form's block heading; the default names the direction, e.g. ``named ← hex``, since a block's subject is the pair and not either form alone.
    """
    elided = ELIDED_RISERS if elided is None else elided
    other = {f: next(g for g in forms if g != f) for f in forms} if len(forms) == 2 else {}
    n_depth = next(iter(within.values())).shape[-2]  # (replicate, depth, landmark)

    fig = plt.figure(figsize=(width, len(forms) * (n_depth * panel_height + 0.3) + 0.3))
    sfigs = fig.subfigures(len(forms), len(targets), squeeze=False)
    for i, form in enumerate(forms):
        head = (titles or {}).get(form) or f"{form} ← {other.get(form, 'other')}"
        for j, target in enumerate(targets):
            draw_transfer_stack(
                sfigs[i, j],
                within[form, target],
                cross[form, target],
                elide=elided.get(form, ()),
                title=f"{head} · {target}",
                depth_labels=j == 0,
                tick_labels=i == len(forms) - 1,
                sparse_ticks=len(targets) > 2,
                scale_key=(i, j) == (len(forms) - 1, len(targets) - 1),
            )
    if ylabel:
        fig.supylabel(ylabel, fontsize=9)
    return fig
