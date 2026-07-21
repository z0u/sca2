"""Figure helpers for the RGB-cube domain, shared by the ex-2.1.x reports.

The conventions themselves are written down in the figure-style skill; this is
where the cube half of them is implemented, so panels stay comparable across
reports. The hypersphere counterpart is
:func:`sca.colorcube.plot_latent_disc`.

Reports import this, experiments don't. Keep it that way: mi-ni fingerprints
project source transitively, so an experiment that imported a plotting helper
would re-run every time a figure got tweaked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from jaxtyping import Float

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["CUBE_VIEWS", "CubeView", "project_cube", "grid_diameter", "plot_rgb_cube", "align_to_cube"]

type ViewName = Literal["solid", "solid-back", "wheel"]


class CubeView(NamedTuple):
    """One 2D view of the cube: how to project it, what its silhouette is, and which way it faces.

    Any flat view of a solid has to collapse one direction, and the two views
    differ in which one they give up. Pick by what the figure is claiming.
    """

    basis: Float[np.ndarray, "2 3"]
    """Projection from centered RGB onto the panel."""

    rim: Float[np.ndarray, "6 3"]
    """The six corners on the silhouette, counter-clockwise — winding-sensitive APIs care."""

    toward: Float[np.ndarray, " 3"]
    """The hidden direction, pointing at the reader; sets which of two coincident marks wins."""


def _solid(front: Float[np.ndarray, " 3"]) -> CubeView:
    """The cube stood on its black corner: white up, black down, *front* facing the reader."""
    up = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)  # the grey diagonal, vertical in the panel
    toward = front - (front @ up) * up  # the horizontal component of the facing color
    right = np.cross(toward / np.linalg.norm(toward), up)
    # Six corners either way — the facing color and its opposite fall inside — but a mirrored
    # view reverses the winding, and the rim is documented counter-clockwise.
    rim = np.array([(1, 1, 1), (1, 1, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1), (1, 0, 1)], dtype=float)
    return CubeView(
        basis=np.stack([right, up]) * (2 / np.sqrt(3)),  # white lands at +1, black at −1
        rim=rim if front[0] > 0 else np.roll(rim[::-1], 1, axis=0),
        toward=toward * 3,  # (2, −1, −1) facing red, so integer color steps stay exact
    )


def _wheel() -> CubeView:
    """Viewed down the grey diagonal, so the chromatic corners make a regular hexagon, red up."""
    e = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    e *= np.sqrt(1.5)  # the six chromatic corners then land on the unit circle
    t = np.radians(60.0)  # ...and this puts red at the top
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return CubeView(
        basis=rot @ e,
        rim=np.array([(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)], dtype=float),
        toward=np.array([1.0, 1.0, 1.0]),
    )


CUBE_VIEWS: dict[str, CubeView] = {
    "solid": _solid(np.array([1.0, 0.0, 0.0])),
    "solid-back": _solid(np.array([0.0, 1.0, 1.0])),
    "wheel": _wheel(),
}
"""The house views of the cube.

``solid`` is the default and the one to reach for when showing colors as data —
a whole grid, a dataset, a palette. Lightness runs up the panel and hue around
it, giving the familiar color solid; the red–cyan axis is the view direction, so
colors a step of (2, −1, −1) apart coincide. That occlusion is the point: it is
what makes a filled grid read as a solid object rather than a flat wheel, and
what keeps it from being mistaken for a hypersphere disc. ``solid-back`` is the
same view from the far side, cyan toward the reader; a front and a back panel
together show all six outer faces of a grid.

``wheel`` looks down the grey diagonal instead, so the six chromatic corners
form a regular hexagon with red at the top and lightness is what collapses
(black, mid-gray and white all land on the origin). Prefer it for analysis
panels — a probe projection, a recovered cube — where occluding a hue axis
would hide exactly the errors the panel exists to show.
"""


def project_cube(rgb: np.ndarray, view: ViewName = "solid") -> np.ndarray:
    """Project unit-cube RGB coordinates onto the panel. See :data:`CUBE_VIEWS` for the views."""
    return (np.asarray(rgb, dtype=float) - 0.5) @ CUBE_VIEWS[view].basis.T


def grid_diameter(levels: int, view: ViewName = "solid") -> float:
    """Mark diameter, in panel units, at which a full *levels*-per-channel grid covers with no gaps.

    Twice the covering radius of the projected lattice — the largest distance
    any point in the panel can sit from the nearest mark. Neighbours overlap a
    little, because the projection squashes the lattice unevenly (in the solid
    view a red step is shorter on the panel than a green or blue one) and marks
    sized to merely touch along the short direction would leave gaps along the
    long one.
    """
    # One red and one green step generate the projected lattice; a blue step is a combination.
    a, b = (np.eye(3)[:2] @ CUBE_VIEWS[view].basis.T) / max(levels - 1, 1)
    while True:  # Gauss reduction: shrink to the lattice's two shortest independent vectors
        if b @ b < a @ a:
            a, b = b, a
        if (m := round((a @ b) / (a @ a))) == 0:
            break
        b = b - m * a
    if a @ b < 0:
        b = -b  # pick the sign giving the acute triangle, whose circumcenter lies inside it
    area = 0.5 * abs(float(a[0] * b[1] - a[1] * b[0]))
    return float(np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(a - b) / (2 * area))


def align_to_cube(x: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Best rotation + uniform scale + shift taking *x* onto the true RGB positions.

    Returns the mapped coordinates and the leftover residual as a fraction of
    the true positions' variance. Constraining the fit to rigid motions is the
    point: a free linear map would absorb real shape mismatch into a shear, so
    the residual is comparable across grids, seeds, and layers. *x* may live in
    any dimension ≥ 3 that has already been reduced to 3 (e.g. embeddings
    projected onto a probe's read-out subspace).
    """
    xc, yc = x - x.mean(0), rgb - rgb.mean(0)
    u, s, vt = np.linalg.svd(xc.T @ yc)
    mapped = xc @ (u @ vt) * (s.sum() / max((xc**2).sum(), 1e-12)) + rgb.mean(0)
    return mapped, float(((mapped - rgb) ** 2).sum() / max((yc**2).sum(), 1e-12))


def draw_cube_bound(ax: Axes, view: ViewName = "solid", *, fill: bool = True) -> None:
    """The cube's silhouette and the geometry-panel conventions, without any data.

    :func:`plot_rgb_cube` calls this; call it directly when a panel draws its own
    marks — a lattice with edges between them, say, where the caller has to
    interleave zorders itself. The silhouette goes behind everything (zorder −10)
    and its rim in front (zorder 10), so marks in between are framed either way.
    """
    from matplotlib.patches import Polygon

    from mini.vis import light_dark

    hull = project_cube(CUBE_VIEWS[view].rim, view)
    if fill:
        ax.add_patch(Polygon(hull, closed=True, facecolor=light_dark("#eee", "#111"), lw=0, zorder=-10))
        ax.add_patch(
            Polygon(hull, closed=True, facecolor="none", edgecolor=light_dark("#0005", "#fff4"), lw=1, zorder=10)
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_axis_off()


def plot_rgb_cube(
    ax: Axes,
    rgb: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    truth: np.ndarray | None = None,
    s: float = 20,
    diameter: float | None = None,
    view: ViewName = "solid",
    bound: bool = True,
) -> None:
    """One color-cube panel, per the repo's figure conventions (see the figure-style skill).

    The cube bound as a background hexagon, data-colored points, fixed domain
    limits, no axes. Pass *truth* (the same points' true RGB) to also draw each
    point's target as an open ring with a stub to where it actually landed, so
    positional error reads off the panel. Titles and annotations stay with the
    caller. See :data:`CUBE_VIEWS` for *view*; analysis panels usually want
    ``"wheel"``.

    Marks are sized in points by *s*, which is what a scatter of arbitrary
    points wants — an embedding projection shouldn't grow its dots just because
    the vocabulary did. Pass *diameter* instead to size them in panel units,
    where the cube spans 2 from black to white: marks then hold their size
    relative to the cube under any figure resize, and a plot of a whole grid can
    ask for `grid_diameter(levels)` and tile it with no trial and error.
    """
    from matplotlib.collections import EllipseCollection
    from matplotlib.colors import to_rgba_array

    from mini.vis import light_dark

    draw_cube_bound(ax, view, fill=bound)
    v = CUBE_VIEWS[view]
    # Nearer the reader draws last: every flat view of the cube hides one axis, so without this
    # the back of a filled grid paints over its front.
    order = np.argsort(np.asarray(rgb, dtype=float) @ v.toward, kind="stable")
    rgb = np.asarray(rgb, dtype=float)[order]
    xy = project_cube(rgb, view)
    # Normalize to an RGBA list so hex strings and (N, 3) arrays behave alike as edge colors.
    rgba = to_rgba_array(np.clip(rgb, 0, 1) if colors is None else np.asarray(colors)[order]).tolist()
    # Marks stay unclipped, as the disc panels' rim annotations do: a mark centered on the
    # silhouette overhangs it by half its width, and the limits are the cube's, not the ink's.
    if truth is not None:
        tru = project_cube(np.asarray(truth, dtype=float)[order], view)
        for p, t, c in zip(xy, tru, rgba, strict=True):
            ax.plot([t[0], p[0]], [t[1], p[1]], "-", color=c, lw=0.7, alpha=0.5, zorder=1, clip_on=False)
        ax.scatter(tru[:, 0], tru[:, 1], c="none", edgecolors=rgba, s=s * 1.3, lw=0.7, zorder=2, clip_on=False)
    if diameter is not None:
        # Sized in data units, so the collection rescales with the axes rather than the figure.
        # No edge here: at tiling sizes it would draw a mesh of outlines over the solid.
        ax.add_collection(
            EllipseCollection(
                widths=diameter,
                heights=diameter,
                angles=0,
                units="xy",
                offsets=xy,
                offset_transform=ax.transData,
                facecolors=rgba,
                linewidths=0,
                zorder=3,
                clip_on=False,
            )
        )
    else:
        # A faint contrasting edge keeps white (light mode) and black (dark) marks visible on the fill.
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=rgba,
            s=s,
            edgecolors=light_dark("#00000033", "#ffffff55"),
            lw=min(0.5, s / 40),
            zorder=3,
            clip_on=False,
        )
