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

Almost none of that depends on the response: the samples, their redness (the x
pixel), their palette color, and the interpolation weights are all fixed by cube
geometry. :class:`GradingField` computes them once; each panel then costs one sparse
matvec and a re-bin, ~2 ms. Draw into a whole Axes (redness on x), or pass ``span``
to place the cloud as a self-contained mark in a slot of a wider axis — e.g. one
slot per token position, aligned with :func:`~mini.vis.smooth_step` plateaus.
"""

from functools import lru_cache

import numpy as np
from matplotlib.axes import Axes
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


def strata(k: int, seed: int = 0) -> np.ndarray:
    """A jittered k³ stratification of the unit cube: one point per sub-cell, placed at
    random within it.

    Even coverage without the moiré a plain lattice would beat against redness, and
    deterministic — the seed fixes the picture rather than merely the noise.
    """
    g = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    return (g + np.random.default_rng(seed).random(g.shape)) / k


class GradingField:
    """The response-independent parts of the grading cloud, computed once per figure.

    `k` sets the look — how much of the cloud reads as solid versus translucent — and
    is calibrated at :data:`REF_PX`: the sample count scales with `px` and `dpr` to
    hold samples-per-pixel, so the same k gives the same texture at any size. `dpr`
    supersamples: the per-pixel lottery runs on a dpr× canvas and a box filter brings
    it back to `px` before imshow ever sees it, so the result is independent of figure
    size and savefig dpi. A display pixel then averages at most dpr² palette colors,
    all drawn from its own samples — blending stays confined below the one-pixel
    scale — and its alpha is the fraction of subpixels covered.
    """

    def __init__(
        self,
        k: int = 30,
        px: tuple[int, int] = REF_PX,
        ylim: tuple[float, float] = (-0.3, 1.1),
        seed: int = 0,
        dpr: int = 2,
    ):
        self.nx, self.ny = px
        self.dpr = dpr
        self.ylo, self.yhi = ylim
        self.xspan = float(REDNESS.min()), float(REDNESS.max())
        # Pixels grow as px·dpr², samples as k³; this holds their ratio, and the look.
        kk = round(k * (self.nx * self.ny / (REF_PX[0] * REF_PX[1])) ** (1 / 3) * dpr ** (2 / 3))
        rgb = strata(kk, seed)
        # The lottery: shuffled once, so per pixel the last write is a uniform winner.
        rgb = rgb[np.random.default_rng(seed + 2).permutation(len(rgb))]
        n = len(LEVELS)
        pack = np.array([n * n, n, 1])
        palette = np.rint(rgb * (n - 1)).astype(np.int64) @ pack  # nearest of the 216
        self.rgba = np.concatenate([GRID_RGB[palette], np.ones((len(rgb), 1))], axis=1).astype(np.float32)
        x0, x1 = self.xspan
        self.ix = np.rint((redness(rgb) - x0) / (x1 - x0) * (self.nx * dpr - 1)).astype(np.int32)
        # Trilinear interpolation as a sparse matrix: sample i's response is (W @ y)[i].
        t = np.clip(rgb, 0, 1) * (n - 1)
        i0 = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
        f = t - i0
        cols = np.empty((len(rgb), 8), np.int32)
        data = np.empty((len(rgb), 8), np.float32)
        for j, corner in enumerate(np.ndindex(2, 2, 2)):
            cols[:, j] = (i0 + corner) @ pack
            data[:, j] = np.prod([f[:, c] if d else 1 - f[:, c] for c, d in enumerate(corner)], axis=0)
        self.W = sparse.csr_array((data.ravel(), cols.ravel(), np.arange(len(rgb) + 1) * 8), shape=(len(rgb), n**3))

    def draw(self, ax: Axes, y: np.ndarray, span: tuple[float, float] | None = None, zorder: float = 2) -> AxesImage:
        """The cloud for one response vector *y* (per grid color), on the Axes' y scale.

        By default redness spans its own range on x; pass *span* to compress it into
        ``(x0, x1)`` in data coordinates instead, placing the cloud as one mark in a
        slot of a shared axis.
        """
        nx, ny = self.nx * self.dpr, self.ny * self.dpr
        a = self.W @ y.astype(np.float32)
        iy = np.rint((a - self.ylo) / (self.yhi - self.ylo) * (ny - 1)).astype(np.int32)
        ok = (iy >= 0) & (iy < ny)
        flat = iy[ok] * nx + self.ix[ok]
        canvas = np.zeros((ny * nx, 4), np.float32)
        canvas[flat] = self.rgba[ok]  # duplicate pixels: last write wins
        # Box-downsample premultiplied: winners are opaque and voids transparent black, so
        # the block mean is (premultiplied color, coverage); dividing restores straight RGBA.
        img = canvas.reshape(self.ny, self.dpr, self.nx, self.dpr, 4).mean(axis=(1, 3))
        img[..., :3] /= np.maximum(img[..., 3:], 1e-6)
        extent = (*(span if span is not None else self.xspan), self.ylo, self.yhi)
        return ax.imshow(img, extent=extent, origin="lower", aspect="auto", zorder=zorder, interpolation="nearest")


@lru_cache
def grading_field(
    k: int = 30,
    px: tuple[int, int] = REF_PX,
    ylim: tuple[float, float] = (-0.3, 1.1),
    seed: int = 0,
    dpr: int = 2,
) -> GradingField:
    """A shared :class:`GradingField` per parameter set. The precompute is only ~20 ms,
    but caching makes repeated notebook cells and re-renders free.
    """
    return GradingField(k, px, ylim, seed, dpr)
