"""
Utilities for working with matplotlib stylesheets, and drawing primitives it lacks.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike


Stylesheet = Literal["base", "light", "dark", "transparent"] | Mapping[str, str]


@contextmanager
def use_style(*styles: Stylesheet):
    """Apply matplotlib styles.

    When *theme* is given, :func:`light_dark` and :func:`current_theme`
    will resolve against it inside the block.
    """
    with mpl.rc_context():
        stylesheet_dir = Path(__file__).parent / "mplstyles"
        for style in styles:
            if isinstance(style, Mapping):
                plt.style.use(dict(style))
            else:
                plt.style.use(stylesheet_dir / f"{style}.mplstyle")
        yield


def smooth_step(ax: "Axes", x: "ArrayLike", y: "ArrayLike", *, ramp: float = 1.0, **kwargs) -> PathPatch:
    """Draw a step plot whose risers are S-curves rather than vertical jumps.

    Each riser is a cubic whose control points both sit at its midpoint, level with
    their respective endpoints. That gives horizontal tangents at every sample, long
    flat shoulders, and the steepest slope at the crossover — twice the straight-line
    slope — so the curve reads as flowing while each value still gets a moment of rest.
    Because the control points never leave the [y_i, y_{i+1}] band, the curve is
    monotone between samples: no spline overshoot. By default each riser spans the
    whole gap between samples; *ramp* < 1 narrows it to that fraction of the gap,
    inserting flat plateaus like a classic step plot. Compare ``ax.step``, whose
    vertical risers carry no rate information and pile up exactly on top of each other
    where two series agree.

    Use it for point measurements on a regular ordinal axis — character positions,
    epochs, layer depths — where a straight line between samples would wrongly suggest
    the value passes through the intermediate values. *x* must be evenly spaced.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        raise ValueError("smooth_step needs at least two samples")
    dx = (x[-1] - x[0]) / (len(x) - 1)
    h = ramp * dx / 2

    verts, codes = [(x[0] - dx / 2, y[0])], [MplPath.MOVETO]
    for i in range(len(x) - 1):
        m = (x[i] + x[i + 1]) / 2
        verts += [(m - h, y[i]), (m, y[i]), (m, y[i + 1]), (m + h, y[i + 1])]
        codes += [MplPath.LINETO, *[MplPath.CURVE4] * 3]
    verts.append((x[-1] + dx / 2, y[-1]))
    codes.append(MplPath.LINETO)

    patch = PathPatch(MplPath(np.array(verts), codes), fill=False, capstyle="round", joinstyle="round", **kwargs)
    ax.add_patch(patch)
    return patch
