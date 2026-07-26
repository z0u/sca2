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
    from collections.abc import Iterable

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


def _step_path(x: np.ndarray, y: np.ndarray, half_widths: np.ndarray, breaks: set[int]) -> MplPath:
    """Plateau-and-riser path through *(x, y)*, each riser a cubic of the given half-width."""
    dx = (x[-1] - x[0]) / (len(x) - 1)
    verts, codes = [(x[0] - dx / 2, y[0])], [MplPath.MOVETO]
    for i in range(len(x) - 1):
        m, h = (x[i] + x[i + 1]) / 2, half_widths[i]
        if i in breaks:  # close this plateau at its right shoulder, restart at the next's left — a gap, no riser
            verts += [(m - h, y[i]), (m + h, y[i + 1])]
            codes += [MplPath.LINETO, MplPath.MOVETO]
        else:
            verts += [(m - h, y[i]), (m, y[i]), (m, y[i + 1]), (m + h, y[i + 1])]
            codes += [MplPath.LINETO, *[MplPath.CURVE4] * 3]
    verts.append((x[-1] + dx / 2, y[-1]))
    codes.append(MplPath.LINETO)
    return MplPath(np.array(verts), codes)


def _strokes(path: MplPath) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a path into its connected strokes, one (vertices, codes) pair per MOVETO."""
    verts, codes = np.asarray(path.vertices, float), np.asarray(path.codes, int)
    starts = [*np.flatnonzero(codes == MplPath.MOVETO), len(codes)]
    return [(verts[a:b], codes[a:b]) for a, b in zip(starts[:-1], starts[1:], strict=True)]


def _band_path(upper: MplPath, lower: MplPath) -> MplPath:
    """Close each of *upper*'s strokes onto the matching one of *lower*, giving filled ribbons.

    The lower edge is walked backwards. Reversing a stroke's vertices reverses its segments,
    and each segment's codes are the same read either way (a lone LINETO, or a CURVE4 triple),
    so reversing the code list after the leading MOVETO re-describes the same curve.
    """
    verts, codes = [], []
    for (uv, uc), (lv, lc) in zip(_strokes(upper), _strokes(lower), strict=True):
        verts += [*uv, *lv[::-1], uv[0]]
        codes += [*uc, MplPath.LINETO, *lc[:0:-1], MplPath.CLOSEPOLY]
    return MplPath(np.array(verts), codes)


def smooth_step_band(
    ax: "Axes",
    x: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    *,
    ramp: "float | ArrayLike" = 1.0,
    breaks: "Iterable[int] | None" = None,
    **kwargs,
) -> PathPatch:
    """Fill the ribbon between two :func:`smooth_step` curves.

    Both edges are stepped the same way, so the band keeps the plateau-and-riser shape of
    the lines it belongs to and never disagrees with them about where a value starts and
    stops. Use it for a spread the reader should take in as a thickness rather than read
    off — a min–max over seeds, a quantile range — behind the series it belongs to. Pass a
    scalar *lower* for the common case of an area down to a baseline.
    """
    x = np.asarray(x, float)
    if len(x) < 2:
        raise ValueError("smooth_step_band needs at least two samples")
    lo, hi = (np.broadcast_to(np.asarray(v, float), x.shape) for v in (lower, upper))
    dx = (x[-1] - x[0]) / (len(x) - 1)
    hs = np.broadcast_to(ramp, len(x) - 1) * dx / 2
    brk = set(breaks or ())
    patch = PathPatch(_band_path(_step_path(x, hi, hs, brk), _step_path(x, lo, hs, brk)), lw=0, **kwargs)
    ax.add_patch(patch)
    return patch


def smooth_step_area(
    ax: "Axes",
    x: "ArrayLike",
    y: "ArrayLike",
    *,
    baseline: float = 0.0,
    **kwargs,
) -> PathPatch:
    """Fill between a :func:`smooth_step` curve and *baseline* — :func:`smooth_step_band` with a flat floor.

    Use it for a summary the reader should be able to take in without tracing anything —
    the total, the mean of the series drawn over it — in a neutral fill, so it reads as
    ground rather than as one more series.
    """
    return smooth_step_band(ax, x, baseline, y, **kwargs)


def smooth_step(
    ax: "Axes",
    x: "ArrayLike",
    y: "ArrayLike",
    *,
    ramp: "float | ArrayLike" = 1.0,
    breaks: "Iterable[int] | None" = None,
    elide: "Iterable[int] | None" = None,
    fade: "float | str" = 0.3,
    **kwargs,
) -> PathPatch:
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

    *ramp* may be a scalar or one value per riser (length ``len(x) - 1``), so individual
    transitions flatten or square off independently: 1 is a pure ramp with no rest, 0 is a
    vertical step. Use a per-riser ramp when neighbours differ in kind — draw the jump
    across a run of unsampled positions as a smooth slide (→1) while keeping genuine
    neighbours as discrete plateaus (< 1). Any two adjacent ramps must sum to ≤ 2, which
    holds for values in [0, 1].

    *breaks* holds indices *i* after which to leave a gap instead of a riser: sample
    *i*'s plateau ends at its right shoulder and sample *i+1*'s begins at its left one,
    with nothing between. Use it where consecutive samples are the same underlying point
    (e.g. two ordinal slots that alias onto one character) so the connecting riser would
    span zero real distance. The gap is the riser's footprint, so it's widest when the
    plateaus are narrow — pair breaks with *ramp* < 1 to keep a visible flat on each side.

    *elide* holds indices *i* whose riser spans ground the samples never covered — a run
    of unmeasured positions between two measured ones. Those risers are drawn at *fade*
    times the line's opacity, so interpolated stretches carry visibly less ink than the
    measurements they join, and the eye stops reading them as another step. The faded
    riser is a second, lighter artist beneath the returned one (which is broken at those
    gaps), so the effect is real transparency: it needs no opaque mask matched to the
    background, and survives a transparent figure over any page color. Where *breaks* and
    *elide* name the same riser, the break wins and nothing is drawn.

    Pass *fade* as a color instead to draw those risers opaque in that color — see
    :func:`~mini.vis.mix` for computing one. Worth it where elided risers from several
    lines land on top of each other: translucent copies accumulate into something darker
    than any of them, which reads as emphasis where the data only agreed.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        raise ValueError("smooth_step needs at least two samples")
    brk, eli = set(breaks or ()), set(elide or ())
    dx = (x[-1] - x[0]) / (len(x) - 1)
    hs = np.broadcast_to(ramp, len(x) - 1) * dx / 2  # riser half-widths (one per gap)
    style = dict(fill=False, capstyle="round", joinstyle="round") | kwargs

    if eli - brk:
        ghost = _step_path(x, y, hs, brk)  # keeps the elided risers; the solid path drops them
        quiet = (
            {"color": fade, "alpha": None} if isinstance(fade, str) else {"alpha": fade * (kwargs.get("alpha") or 1.0)}
        )
        ax.add_patch(PathPatch(ghost, **style | quiet))
    patch = PathPatch(_step_path(x, y, hs, brk | eli), **style)
    ax.add_patch(patch)
    return patch
