"""Orthographic RGB-cube views of the color domain (ported from M1's ex-1.1.1).

The cube is rotated so the black→white diagonal stands vertical — that axis
reads as *value* — with hue running horizontally. A single orthographic view
hides the three faces pointing away from the camera, so a ``front`` and a
``back`` view together show all six outer faces of the grid.

Points are colored by the color they represent: color *is* the data here, so
there is no colorbar (see the figure-style skill). The draw helper is a
geometry primitive — it takes an ``Axes`` and leaves titles, panel layout, and
theming to the caller, the same division of labor as
:func:`sca.colorcube.plot_latent_disc`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from sca.data.colors import N_LEVELS, PALETTE

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection

type Side = Literal["front", "back"]

_DIAG = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)  # black→white, the vertical axis


def _basis(side: Side) -> np.ndarray:
    """Orthonormal basis (columns e1, e2, e3) with the value diagonal as e3.

    ``e1`` spans hue horizontally; ``e2`` points into the screen (depth, used
    only for occlusion ordering); ``e3`` is the black→white diagonal.
    """
    e1 = np.array([0.0, -1.0, 1.0] if side == "front" else [0.0, 1.0, -1.0])
    e1 = e1 - (e1 @ _DIAG) * _DIAG
    e1 = e1 / np.linalg.norm(e1)
    return np.stack([e1, np.cross(_DIAG, e1), _DIAG], axis=1)


def project(rgb01: np.ndarray, side: Side) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project unit-cube RGB coordinates to ``(horizontal, value, depth)``."""
    p = np.asarray(rgb01, dtype=float) @ _basis(side)
    return p[:, 0], p[:, 2], p[:, 1]


def grid(levels: int = N_LEVELS) -> np.ndarray:
    """The full RGB grid at *levels* steps per channel, as ``[levels³, 3]`` in [0, 1]."""
    c = np.linspace(0, 1, levels)
    return np.stack(np.meshgrid(c, c, c, indexing="ij"), axis=-1).reshape(-1, 3)


def named() -> np.ndarray:
    """The 27 named colors as ``[27, 3]`` in [0, 1] — the 3×3×3 sub-lattice."""
    return np.array(list(PALETTE.values()), dtype=float) / (N_LEVELS - 1)


def style_cube_axes(ax: Axes, *, labels: bool = True) -> None:
    """Apply the geometry-panel conventions: equal aspect, no ticks or spines.

    The projection axes carry no meaningful scale, so the labels name the
    directions instead (see the figure-style skill). Shared by the dense-grid
    view and any lattice drawn in the same projection.
    """
    ax.set_aspect("equal")
    ax.margins(0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if labels:
        ax.set_xlabel("Hue (⊥ to value)")
        ax.set_ylabel("Value")


def draw_rgb_cube(
    ax: Axes,
    rgb01: np.ndarray,
    *,
    side: Side = "front",
    s: float = 195,
    edgecolors: str | None = "none",
    lw: float = 0,
    alpha: float = 1.0,
    labels: bool = True,
    zorder: float = 1,
) -> PathCollection:
    """Scatter one orthographic view of *rgb01*, colored by the colors themselves.

    Points are drawn back-to-front so nearer ones win overlaps; at the default
    size a full grid's points just touch — a gapless solid that still shows the
    grid resolution, which reads far better than a dotty scatter. Returns the
    collection so the caller can tweak it.
    """
    rgb01 = np.asarray(rgb01, dtype=float)
    x, y, depth = project(rgb01, side)
    order = np.argsort(depth)  # back (small depth) first, so front points sit on top
    pc = ax.scatter(
        x[order], y[order], c=rgb01[order], s=s, edgecolors=edgecolors, linewidths=lw, alpha=alpha, zorder=zorder
    )
    style_cube_axes(ax, labels=labels)
    return pc
