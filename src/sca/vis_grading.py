"""
The grading cloud: a dithered fill showing a per-color response against redness.

A grading figure asks how a response over the v216 color grid — α per color, say —
varies with redness. Drawing all 216 colors as marks is too much ink to tile across
(slice, position) panels, so the cloud oversamples instead: the response, measured
only at the grid colors, is extended to the whole RGB cube by trilinear
interpolation, and a dense stratified sample of the cube is binned to pixels. Each
pixel takes one *measured* grid color, chosen uniformly among the samples landing in
it — a dither, so no averaging invents colors the data never produced — and, at
`dpr` > 1, a supersampled box filter turns subpixel coverage into opacity, so sparse
regions of the cloud read as translucent rather than grainy.

Draw with :class:`GradingCloud`, an artist that rasters itself at the axes' device
size at draw time, so the dither maps 1:1 onto output pixels at any figure size and
export dpi. Almost nothing depends on the response — the samples, their
redness (the x pixel), their palette color, and the interpolation weights are all
fixed by cube geometry and cached per sample-lattice size — so each panel costs one
sparse matvec and a re-bin, a few ms. The cloud is data-colored, so it draws the
same in either theme with no :func:`~mini.vis.light_dark`, and it is cheap enough
to run inside ``@themed``.
"""

from functools import lru_cache
from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage
from scipy import sparse

from sca.colorcube import redness
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS

LEVELS = np.asarray(GRIDS["v216"], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
I_RED = int(np.argmax(REDNESS))  # pure red's row in GRID_RGB

REF_PX = (240, 180)  # the canvas size k is calibrated against, ~a grid panel at 150 dpi
XSPAN = (float(REDNESS.min()), float(REDNESS.max()))


def strata(k: int, seed: int = 0) -> np.ndarray:
    """A jittered k³ stratification of the unit cube: one point per sub-cell, placed at
    random within it.

    Even coverage without the moiré a plain lattice would beat against redness, and
    deterministic — the seed fixes the picture rather than merely the noise.
    """
    g = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    return (g + np.random.default_rng(seed).random(g.shape)) / k


def _lattice_k(k: int, px: tuple[int, int], dpr: int) -> int:
    """The stratification size that holds samples-per-pixel at the look `k` names.

    Pixels grow as px·dpr², samples as (lattice k)³; the cube root holds their
    ratio, so the same k gives the same texture at any raster size. It also
    quantizes coarsely, so nearby raster sizes land on the same lattice and share
    a cached :func:`_geometry`.
    """
    return round(k * (px[0] * px[1] / (REF_PX[0] * REF_PX[1])) ** (1 / 3) * dpr ** (2 / 3))


class _Geometry(NamedTuple):
    """The response-independent parts of the cloud — everything cube geometry fixes."""

    u: np.ndarray
    """Each sample's coordinate along the loft (see :class:`GradingCloud`)."""
    rgba: np.ndarray
    """Each sample's palette color: nearest of the 216, opaque."""
    xs: np.ndarray
    """Each sample's redness — its x position, awaiting a pixel scale."""
    W: sparse.csr_array
    """Trilinear interpolation: sample i's response is ``(W @ y)[i]``."""


@lru_cache(maxsize=4)
def _geometry(kk: int, seed: int) -> _Geometry:
    """The sample set for one lattice size, shared by every cloud that lands on it.

    An entry is tens of MB at panel sizes (`W` dominates), hence the small cache;
    like-sized panels and both themes of a `@themed` render all hit one entry.
    """
    rgb = strata(kk, seed)
    # The lottery: shuffled once, so per pixel the last write is a uniform winner.
    rgb = rgb[np.random.default_rng(seed + 2).permutation(len(rgb))]
    # Fixed like the rest of the geometry so panels dither identically across conditions.
    u = np.random.default_rng(seed + 1).random(len(rgb)).astype(np.float32)
    n = len(LEVELS)
    pack = np.array([n * n, n, 1])
    palette = np.rint(rgb * (n - 1)).astype(np.int64) @ pack  # nearest of the 216
    rgba = np.concatenate([GRID_RGB[palette], np.ones((len(rgb), 1))], axis=1).astype(np.float32)
    # Trilinear interpolation as a sparse matrix.
    t = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
    f = t - i0
    cols = np.empty((len(rgb), 8), np.int32)
    data = np.empty((len(rgb), 8), np.float32)
    for j, corner in enumerate(np.ndindex(2, 2, 2)):
        cols[:, j] = (i0 + corner) @ pack
        data[:, j] = np.prod([f[:, c] if d else 1 - f[:, c] for c, d in enumerate(corner)], axis=0)
    W = sparse.csr_array((data.ravel(), cols.ravel(), np.arange(len(rgb) + 1) * 8), shape=(len(rgb), n**3))
    return _Geometry(u, rgba, redness(rgb), W)


def _raster(
    geom: _Geometry,
    y: np.ndarray,
    ylim: tuple[float, float],
    px: tuple[int, int],
    dpr: int,
    lerp: bool = False,
    color: str | tuple[float, float, float] | None = None,
) -> np.ndarray:
    """One response's RGBA raster at `px`, ready for imshow with ``origin="lower"``."""
    nx, ny = px[0] * dpr, px[1] * dpr
    if y.ndim == 1:
        a = geom.W @ y.astype(np.float32)
    else:
        rows = np.arange(len(geom.u))
        aa = geom.W @ y.astype(np.float32).T  # (sample, layer)
        if lerp:
            t = geom.u * (len(y) - 1)
            s0 = np.minimum(t.astype(np.int64), len(y) - 2)
            f = t - s0
            a = aa[rows, s0] * (1 - f) + aa[rows, s0 + 1] * f
        else:
            a = aa[rows, np.minimum((geom.u * len(y)).astype(np.int64), len(y) - 1)]
    x0, x1 = XSPAN
    ix = np.rint((geom.xs - x0) / (x1 - x0) * (nx - 1)).astype(np.int32)
    ylo, yhi = ylim
    iy = np.rint((a - ylo) / (yhi - ylo) * (ny - 1)).astype(np.int32)
    ok = (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]
    canvas = np.zeros((ny * nx, 4), np.float32)
    canvas[flat] = geom.rgba[ok]  # duplicate pixels: last write wins
    # Box-downsample premultiplied: winners are opaque and voids transparent black, so
    # the block mean is (premultiplied color, coverage); dividing restores straight RGBA.
    img = canvas.reshape(px[1], dpr, px[0], dpr, 4).mean(axis=(1, 3))
    img[..., :3] /= np.maximum(img[..., 3:], 1e-6)
    if color is not None:
        *rgb, a = to_rgba(color)
        img[..., :3] = rgb
        img[..., 3] *= a
    return img


def quantile_stack(values: np.ndarray, color: np.ndarray, q: int = 17) -> np.ndarray:
    """Per-color quantiles of a per-sample response, as a layer stack for :class:`GradingCloud`.

    `values` is one response per sample and `color` that sample's grid index, so
    each grid color has a distribution of responses rather than one. The result,
    shape ``(q, n³)``, holds each color's quantile function at `q` evenly spaced
    levels from 0 to 1, min to max. Handed to :class:`GradingCloud` with ``lerp``,
    the loft coordinate `u` becomes the probability level and each sample's y is
    the interpolated quantile function at `u`: an inverse-transform draw from that
    color's distribution. The trilinear step then blends the quantile functions of
    neighbouring grid colors, which is the natural interpolation between 1-D
    distributions and keeps two bands as two bands rather than a mean between them.
    The cloud's vertical extent is the spread, and a color measured on few samples
    reads as a band interpolated between them.

    Every grid color needs at least one sample.
    """
    out = np.full((q, len(GRID_RGB)), np.nan, np.float32)
    levels = np.linspace(0, 1, q)
    for c in np.unique(color):
        out[:, c] = np.quantile(values[color == c], levels)
    assert not np.isnan(out).any(), "every grid color needs at least one sample"
    return out


class GradingCloud(AxesImage):
    """A grading cloud that rasters itself at the axes' device size, at draw time.

    A raster sized up front gets nearest-neighbor-rescaled to wherever the axes
    actually lands — figure size, layout, and savefig dpi all move it — and any
    non-integer ratio duplicates some dither rows and drops others: moiré. By draw
    time the transforms are final, so this artist measures its extent in device
    pixels, re-rasters at that size when it changed, and hands a 1:1 image to the
    normal ``AxesImage`` path, where nearest-neighbor resampling is the identity.
    Exports come out clean at any dpi, since matplotlib applies the export dpi
    before drawing.

    `k` sets the look — how much of the cloud reads as solid versus translucent —
    and is size-invariant: the sample count follows the device size to hold
    samples-per-pixel (:func:`_lattice_k`), with the geometry cached per lattice
    size, so like-sized panels share it and a redraw costs one sparse matvec and a
    re-bin. `dpr` supersamples: the per-pixel lottery runs on a dpr× canvas and a
    box filter brings it back down, so a display pixel averages at most dpr²
    palette colors — blending stays confined below the one-pixel scale — and its
    alpha is the fraction of subpixels covered.

    *y* is the response per grid color, or a stack of them, shape ``(S, n³)`` —
    one per residual slice, or per seed. The cube is then lofted over the stack:
    each sample belongs to one layer (its share of the fixed loft coordinate `u`),
    takes its response from that layer alone, and the per-pixel lottery weighs the
    layers fairly, so the cloud shows the union of the layers' responses rather
    than the response of their mean. With `lerp` the stack is treated as a
    continuum instead, each sample blending its two nearest layers — smoother, but
    the blends are responses no layer produced.

    *color* flattens the palette to one hue, multiplying its own alpha into the
    coverage: for a lofted envelope drawn as a muted band behind a full-color
    summary cloud, where fading the palette instead would misstate the sample
    colors. Remaining kwargs are Artist properties — ``alpha``, ``clip_on``,
    ``zorder``. The zorder defaults to 1.5, between matplotlib's own defaults for
    patches (1) and lines (2), so a cloud draws over the fill it sits on and under
    any overlay; the image default of 0 would put it under both.

    *span* compresses redness into ``(x0, x1)`` in data coordinates, placing the
    cloud as one mark in a slot of a shared axis. For a per-(slice, position)
    figure, prefer that row layout to a grid of panels: one slot of width ``sw``
    per position via ``span=(p - sw/2, p + sw/2)``, with overlay series drawn by
    :func:`~mini.vis.smooth_step` at ``ramp=1 - sw`` so plateaus span the slots and
    risers the spaces between. Clouds and overlays then share one coordinate
    system, with no per-panel Axes or frozen-layout overlay tricks. Stack the rows
    in that same Axes by handing each row's artists a
    ``Affine2D().translate(0, row) + ax.transData`` — clouds included, since the
    extent goes through the artist's transform — rather than one Axes per row. A
    cloud overhangs the row it belongs to, and within one Axes that tail draws over
    the row beneath instead of being covered by its background; each row's zero
    line doubles as the rule beneath it. Label x with position names; redness gets
    no ticks — say once in the caption that it runs left to right within each slot.
    Reference implementation: the grading-grid figure in
    ``docs/m2/d2.1/report.py``.
    """

    def __init__(
        self,
        ax: Axes,
        y: np.ndarray,
        ylim: tuple[float, float] = (-0.3, 1.1),
        *,
        k: int = 30,
        seed: int = 0,
        dpr: int = 2,
        span: tuple[float, float] | None = None,
        color: str | tuple[float, float, float] | None = None,
        lerp: bool = False,
        **kwargs,
    ):
        super().__init__(ax, origin="lower", interpolation="nearest")
        self._y = np.asarray(y)
        self._ylim = ylim
        self._k, self._seed, self._dpr = k, seed, dpr
        self._color, self._lerp = color, lerp
        self._px: tuple[int, int] | None = None
        self.set_extent((*(span if span is not None else XSPAN), *ylim))
        self.set_data(np.zeros((1, 1, 4), np.float32))
        kwargs.setdefault("zorder", 1.5)
        self.set(**kwargs)
        ax.add_image(self)

    def draw(self, renderer) -> None:
        ax = self.axes
        assert ax is not None
        x0, x1, y0, y1 = self.get_extent()
        (px0, py0), (px1, py1) = ax.transData.transform([(x0, y0), (x1, y1)])
        px = max(2, round(abs(px1 - px0))), max(2, round(abs(py1 - py0)))
        if px != self._px:
            geom = _geometry(_lattice_k(self._k, px, self._dpr), self._seed)
            self.set_data(_raster(geom, self._y, self._ylim, px, self._dpr, self._lerp, self._color))
            self._px = px
        super().draw(renderer)
