"""Sketch: a faster dither fill for the grading grids.

`grading_sketch.draw_dither` rebuilds the whole sample cloud for every panel — k³
stratified samples, their redness, their palette color, trilinear weights, and a
per-panel argsort — but of all that, only the *interpolated response* depends on
the panel. `DitherField` hoists the rest into a one-time precomputation. Two ideas
carry it:

- Trilinear interpolation is linear in the 216 grid values, so all k³ sample
  responses are one sparse matvec: `W @ y`, with 8 weights per row fixed by geometry.
- The per-pixel lottery doesn't need a sort. Pre-shuffle the samples once; then
  "last write wins" under fancy-index assignment picks the winner at a fixed
  position of a uniform permutation, which is itself a uniform draw.

It also adds `dpr`: supersampled anti-aliasing, box-downsampled in the draw itself
so the result is independent of figure size and savefig dpi.

The script writes the grid figure at dpr 1-3 plus a magnified single panel to
compare them. Data comes from ex-2.1.10's published arrays, cached locally after
the first run so iteration skips the network. `--bench` adds the stage profile and
the panel/figure timing comparison against the original implementation. Not a
notebook, so the site build ignores it. Run from the root:

    uv run python docs/m2/grading_dither.py [--out DIR] [--bench]
"""

import argparse
from pathlib import Path
from time import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

import grading_sketch as gs
from mini.vis import AxesRow, use_style

CACHE = Path(".mini/sketches/grading/ex-2.1.10-alpha.npz")


def load_cached() -> np.ndarray:
    """Seed-mean (slice, color, position) α for the primary condition, cached beside the sketches."""
    if not CACHE.exists():
        seeds, alphas = gs.load()
        a = np.mean([alphas[f"either-t100-s{s}/alpha"] for s in range(seeds["either-t100"])], axis=0)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **{"either-t100": a})
        print(f"cached {CACHE}")
    with np.load(CACHE) as z:
        return z["either-t100"]


class DitherField:
    """The y-independent parts of `draw_dither`, computed once and shared by every panel.

    Same picture as the original (`grading_sketch.draw_dither` documents the encoding);
    only the winning color at each pixel is a different draw of the same lottery.

    `dpr` anti-aliases by supersampling: the lottery runs on a dpr× canvas — with k
    scaled by dpr^(2/3) so samples-per-pixel, and hence the look, stay put — and a box
    filter brings it back to `px` before imshow ever sees it. A display pixel then
    averages at most dpr² palette colors, all drawn from its own samples, so blending
    stays confined below the one-pixel scale; its alpha is the fraction of subpixels
    covered, so sparse regions of the cloud read as translucent rather than grainy.
    """

    def __init__(
        self,
        k: int = 140,
        px: tuple[int, int] = (480, 360),
        ylim: tuple[float, float] = (-0.3, 1.1),
        seed: int = 0,
        dpr: int = 1,
    ):
        self.nx, self.ny = px
        self.dpr = dpr
        self.ylo, self.yhi = ylim
        x0, x1 = float(gs.REDNESS.min()), float(gs.REDNESS.max())
        self.extent = (x0, x1, *ylim)
        kk = round(k * dpr ** (2 / 3))  # pixels grow as dpr², samples as k³; this holds their ratio
        rgb = gs.strata(kk, seed)
        # The lottery: shuffled once, so per pixel the last write is a uniform winner.
        rgb = rgb[np.random.default_rng(seed + 2).permutation(len(rgb))]
        n = len(gs.LEVELS)
        pack = np.array([n * n, n, 1])
        palette = np.rint(rgb * (n - 1)).astype(np.int64) @ pack  # nearest of the 216
        self.rgba = np.concatenate([gs.GRID_RGB[palette], np.ones((len(rgb), 1))], axis=1).astype(np.float32)
        self.ix = np.rint((gs.redness(rgb) - x0) / (x1 - x0) * (self.nx * dpr - 1)).astype(np.int32)
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

    def draw(self, ax: plt.Axes, y: np.ndarray):
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
        ax.imshow(img, extent=self.extent, origin="lower", aspect="auto", zorder=2, interpolation="nearest")


def fig_detail(a: np.ndarray, out: Path, k: int, px: tuple[int, int], dprs: tuple[int, ...]):
    """One panel per dpr, magnified: saved well above the canvas resolution, so each canvas
    pixel spans several image pixels and the anti-aliasing itself is what varies.
    """
    y = a[4, :, 0]  # slice 4 at op1: a dense shoulder and a sparse tail in one panel
    fig, axes = plt.subplots(1, len(dprs), figsize=(2.6 * len(dprs), 2.4), sharey=True, layout="constrained")
    for ax, dpr in zip(cast(AxesRow, axes), dprs, strict=True):
        DitherField(k=k, px=px, dpr=dpr).draw(ax, y)
        ax.set_title(f"dpr={dpr}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.3, 1.1)
    fig.savefig(out / "dither-detail.png", dpi=600)
    plt.close(fig)


def profile_baseline(y: np.ndarray, k: int, px: tuple[int, int]):
    """One baseline panel, stage by stage, to show where the time goes."""
    nx, ny = px
    x0, x1 = float(gs.REDNESS.min()), float(gs.REDNESS.max())
    ylo, yhi = -0.3, 1.1
    t0, stages = time(), []

    def mark(label: str):
        nonlocal t0
        stages.append((label, time() - t0))
        t0 = time()

    rgb = gs.strata(k, 0)
    mark("strata")
    ix = np.rint((gs.redness(rgb) - x0) / (x1 - x0) * (nx - 1)).astype(np.int64)
    mark("redness → ix")
    iy = np.rint((gs.interp_alpha(y, rgb) - ylo) / (yhi - ylo) * (ny - 1)).astype(np.int64)
    mark("interp_alpha → iy")
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]
    n = len(gs.LEVELS)
    np.rint(rgb[ok] * (n - 1)).astype(np.int64) @ np.array([n * n, n, 1])
    mark("palette")
    key = np.random.default_rng(2).integers(0, 2**31, len(flat))
    order = np.argsort(flat * (2**31) + key, kind="stable")
    order[np.flatnonzero(np.diff(flat[order], prepend=-1))]
    mark("sort lottery")
    for label, dt in stages:
        print(f"  {label:<20}{dt * 1e3:6.1f} ms")


def bench(a: np.ndarray, out: Path, k: int, px: tuple[int, int]):
    """Stage profile, per-panel timing, and the full grid, original vs DitherField at dpr=1
    (where the two run the same lottery on the same samples, so coverage must agree).
    """
    panels = [a[si, :, pi] for si in range(a.shape[0]) for pi in range(a.shape[2])]
    base = lambda ax, y: gs.draw_dither(ax, y, k=k, px=px)  # noqa: E731

    print(f"baseline stages, one panel (k={k}, px={px}):")
    profile_baseline(panels[0], k, px)

    t = time()
    field = DitherField(k=k, px=px)
    print(f"DitherField precompute: {time() - t:.2f}s (once per figure)")

    fig, ax = plt.subplots()
    times: dict[str, float] = {"base": 0.0, "fast": 0.0}
    for y in panels:
        covers = {}
        for name, draw in [("base", base), ("fast", field.draw)]:
            t = time()
            draw(ax, y)
            times[name] += time() - t
            covers[name] = np.asarray(ax.images[-1].get_array())[..., 3] > 0
            ax.clear()
        assert (covers["base"] == covers["fast"]).all(), "the fills disagree on which pixels are covered"
    plt.close(fig)
    n = len(panels)
    print(f"per panel over {n}: base {times['base'] / n * 1e3:.0f} ms, fast {times['fast'] / n * 1e3:.0f} ms")

    alpha_map = lambda cond: a  # noqa: E731 — fig_grid only asks for the primary condition
    with use_style("base", "light"):
        for label, draw in [("dither-base", base), ("dither-fast", field.draw)]:
            t = time()
            gs.fig_grid(alpha_map, out, label, draw)
            print(f"wrote {out}/grid-{label}.png ({time() - t:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".mini/sketches/grading"))
    parser.add_argument("--bench", action="store_true", help="also time against grading_sketch.draw_dither")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    a = load_cached()  # (slices, colors, positions)
    k, px, dprs = 30, (240, 180), (1, 2, 3)  # the grid figure's settings

    if args.bench:
        bench(a, out, k, px)

    alpha_map = lambda cond: a  # noqa: E731
    with use_style("base", "light"):
        for dpr in dprs:
            t = time()
            field = DitherField(k=k, px=px, dpr=dpr)
            gs.fig_grid(alpha_map, out, f"dither-dpr{dpr}", field.draw)
            print(f"wrote {out}/grid-dither-dpr{dpr}.png ({time() - t:.1f}s)")
        fig_detail(a, out, k, px, dprs)
        print(f"wrote {out}/dither-detail.png")


if __name__ == "__main__":
    main()
