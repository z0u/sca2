"""Sketch: simplified grading panels for per-(slice, position) grids.

The published grading figure draws all 216 colors as marks, which is too much
ink to tile 5 slices x 6 positions. This sketch prototypes lighter encodings on
ex-2.1.10's published arrays and writes PNGs for comparison:

- variants.png: one column per encoding, on a clean and a noisy condition
- grid-corners.png / grid-cloud.png: the full (slice, position) grid, small

Not a notebook, so the site build ignores it. Run from the repo root:

    uv run python docs/m2/grading_sketch.py [--out DIR]
"""

import argparse
import json
import tempfile
from time import time
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from mini.store import project_store
from mini.vis import AxesGrid, smooth_step, use_style
from sca.colorcube import redness, sim_to_red
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS

LEVELS = np.asarray(GRIDS["v216"], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
RLEVELS = np.unique(np.round(REDNESS, 6))  # 29 distinct redness values
SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
I_RED = int(np.argmax(REDNESS))  # pure red's row in GRID_RGB

SLICE_NAMES = ["emb", "1", "2", "3", "4"]
POS_NAMES = ["op1", "+", "op2", "=", "ans", r"\n"]


def load() -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Seed counts and the per-run alpha maps from the published results."""
    store = project_store()
    arts = store.get_refs(["reports/m2/ex-2.1.10/metrics", "reports/m2/ex-2.1.10/arrays"])
    m_art, a_art = arts["reports/m2/ex-2.1.10/metrics"], arts["reports/m2/ex-2.1.10/arrays"]
    assert m_art is not None and a_art is not None, "ex-2.1.10's results are not published"
    with tempfile.TemporaryDirectory() as d:
        m_path, a_path = store.get_many([(m_art, Path(d) / "metrics.json"), (a_art, Path(d) / "arrays.npz")])
        metrics = json.loads(m_path.read_text())
        with np.load(a_path) as z:
            alphas = {k: z[k] for k in z.files if k.endswith("/alpha")}
    seeds: dict[str, int] = {}
    for c in metrics["cells"]:
        name, _, s = c["label"].rpartition("-s")
        seeds[name] = max(seeds.get(name, 0), int(s) + 1)
    return seeds, alphas


# --- Encodings --------------------------------------------------------------


def level_stat(y: np.ndarray, fn) -> np.ndarray:
    return np.array([fn(y[np.round(REDNESS, 6) == lv]) for lv in RLEVELS])


def draw_base(ax: plt.Axes, ctrl_y: np.ndarray):
    """Zero line and the control's conditional mean, as in the published figure."""
    ax.axhline(0, color="#bbb", lw=0.8, zorder=0)
    ax.plot(RLEVELS, level_stat(ctrl_y, np.mean), color="#999", lw=3, alpha=0.5, zorder=1, solid_capstyle="round")


def draw_full_scatter(ax: plt.Axes, y: np.ndarray):
    """The current encoding: one mark per color, plus the conditional mean."""
    ax.scatter(REDNESS, y, c=GRID_RGB, s=18, lw=0.4, zorder=3, edgecolors="#00000033")
    ax.plot(RLEVELS, level_stat(y, np.mean), color="#222", lw=1.2, zorder=4)


def windowed(v: np.ndarray, fn) -> np.ndarray:
    """Apply fn over each value and its two neighboring groups; calms single-group notches."""
    stack = np.stack([np.r_[v[:1], v[:-1]], v, np.r_[v[1:], v[-1:]]])
    return fn(stack, axis=0)


def envelope(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Windowed min/max of the per-redness-group extremes."""
    return windowed(level_stat(y, np.min), np.min), windowed(level_stat(y, np.max), np.max)


def draw_silhouette(ax: plt.Axes, y: np.ndarray, mean_line: bool = True):
    """The scatter's envelope as a filled band: windowed min to max at each redness level."""
    lo, hi = envelope(y)
    ax.fill_between(RLEVELS, lo, hi, color="#777", alpha=0.28, lw=0, zorder=2)
    if mean_line:
        ax.plot(RLEVELS, level_stat(y, np.mean), color="#222", lw=1.2, zorder=4)


def draw_voronoi(ax: plt.Axes, y: np.ndarray, alpha: float = 0.85, ylim: tuple[float, float] = (-0.3, 1.1)):
    """The envelope filled with color: each pixel takes the color of the nearest mark.

    Nearest in (redness, α) with each axis normalized to its span, so the cells match what
    the eye would attribute the region to. Clipped to the windowed envelope band.
    """
    lo, hi = envelope(y)
    x0, x1 = float(RLEVELS.min()), float(RLEVELS.max())
    y0, y1 = ylim
    nx, ny = 330, 240
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    px = np.stack([np.repeat((xs - x0) / (x1 - x0), ny), np.tile((ys - y0) / (y1 - y0), nx)], axis=-1)
    marks = np.stack([(REDNESS - x0) / (x1 - x0), (y - y0) / (y1 - y0)], axis=-1)
    nearest = np.argmin(((px[:, None, :] - marks[None, :, :]) ** 2).sum(-1), axis=1).reshape(nx, ny).T
    inside = (ys[:, None] >= np.interp(xs, RLEVELS, lo)[None, :]) & (ys[:, None] <= np.interp(xs, RLEVELS, hi)[None, :])
    rgba = np.dstack([GRID_RGB[nearest], np.where(inside, alpha, 0.0)])
    ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="lower", aspect="auto", zorder=2, interpolation="nearest")


def subgrid(levels: list[float]) -> np.ndarray:
    """Mask of GRID_RGB restricted to the given channel levels."""
    return np.isin(np.round(GRID_RGB, 6), np.round(np.array(levels), 6)).all(axis=1)


CORNERS = subgrid([0.0, 1.0])  # the 8 cube corners
CORNERS_NO_RED = CORNERS.copy()
CORNERS_NO_RED[I_RED] = False  # the α_red smooth-step overlay stands in for the red corner mark
MID27 = subgrid([0.0, 9 / 15, 1.0])  # 3x3x3 subgrid; 9/15 is the level nearest the midpoint


def draw_points(ax: plt.Axes, y: np.ndarray, mask: np.ndarray, s: float = 24):
    ax.scatter(REDNESS[mask], y[mask], c=GRID_RGB[mask], s=s, lw=0.5, zorder=5, edgecolors="#00000055")


def cube_edges() -> list[np.ndarray]:
    """Indices into GRID_RGB for the 12 cube edges, ordered along the varying channel."""
    n = len(LEVELS)
    idx = np.arange(n**3).reshape(n, n, n)
    edges = []
    for axis in range(3):
        # for a in (0, n - 1):
        #     for b in (0, n - 1):
        for a in range(n):
            for b in range(n):
                sel: list = [a, b]
                sel.insert(axis, slice(None))
                edges.append(idx[tuple(sel)])
    return edges


def draw_edges(ax: plt.Axes, y: np.ndarray, lw: float = 2.4):
    """The cube's edges as polylines, each data-level segment colored by its midpoint color."""
    segs, cols = [], []
    for e in cube_edges():
        pts = np.column_stack([REDNESS[e], y[e]])
        segs += [pts[i : i + 2] for i in range(len(e) - 1)]
        cols += [np.clip((GRID_RGB[e[i]] + GRID_RGB[e[i + 1]]) / 2, 0, 1) for i in range(len(e) - 1)]
    ax.add_collection(LineCollection(segs, colors=cols, lw=lw, capstyle="round", zorder=5, alpha=0.5))


def interp_alpha(y: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a per-color response onto arbitrary unit-cube points.

    The response is only *measured* at the 216 vocabulary colors; off-grid colors have no
    token, so these values are an interpolant, not data. Same move `draw_edges` makes along
    each cube edge, extended into the interior.
    """
    n = len(LEVELS)  # LEVELS is uniform on [0, 1], so the cell index is just rgb * (n - 1)
    grid = y.reshape(n, n, n)
    t = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
    f = t - i0
    out = np.zeros(len(rgb))
    for corner in np.ndindex(2, 2, 2):
        w = np.prod([f[:, c] if d else 1 - f[:, c] for c, d in enumerate(corner)], axis=0)
        out += w * grid[tuple(i0[:, c] + d for c, d in enumerate(corner))]
    return out


def lattice(levels: int, chunk: int = 1_000_000):
    """The full levels³ RGB lattice, in blocks small enough to interpolate at once."""
    axis = np.arange(levels) / (levels - 1)
    for start in range(0, levels**3, chunk):
        i = np.arange(start, min(start + chunk, levels**3))
        yield np.stack([axis[i // levels**2], axis[i // levels % levels], axis[i % levels]], axis=-1)


def draw_cloud_pts(
    ax: plt.Axes,
    y: np.ndarray,
    levels: int = 256,
    n: int = 60_000,
    s: float = 1.2,
    alpha: float = 0.3,
    mean_line: bool = True,
    seed: int = 0,
):
    """A random sample of the oversampled lattice, as marks.

    Mark size and opacity are set per figure: the grid's panels are a third the width of the
    comparison figure's, so the same cloud lands in a ninth of the area.
    """
    rgb = np.random.default_rng(seed).integers(0, levels, (n, 3)) / (levels - 1)
    ax.scatter(redness(rgb), interp_alpha(y, rgb), c=rgb, s=s, lw=0, alpha=alpha, zorder=3)
    if mean_line:
        ax.plot(RLEVELS, level_stat(y, np.mean), color="#222", lw=1.2, zorder=4)


def draw_cloud(ax: plt.Axes, y: np.ndarray, levels: int = 256, ylim: tuple[float, float] = (-0.3, 1.1)):
    """The whole levels³ lattice, rendered by accumulation.

    Every pixel takes the mean color of the points landing in it, with opacity following how
    many did. Too many marks to draw one by one, so we bin them; the picture is the same, it
    just skips the overdraw.
    """
    nx, ny = 480, 360
    x0, x1 = float(REDNESS.min()), float(REDNESS.max())
    y0, y1 = ylim
    sums, cnt = np.zeros((nx * ny, 3)), np.zeros(nx * ny)
    for rgb in lattice(levels):
        ix = np.rint((redness(rgb) - x0) / (x1 - x0) * (nx - 1)).astype(int)
        iy = np.rint((interp_alpha(y, rgb) - y0) / (y1 - y0) * (ny - 1)).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        flat = (iy[ok] * nx + ix[ok]).astype(int)
        cnt += np.bincount(flat, minlength=nx * ny)
        for c in range(3):
            sums[:, c] += np.bincount(flat, weights=rgb[ok, c], minlength=nx * ny)
    ref = np.percentile(cnt[cnt > 0], 55)  # the bulk reads as solid; the sparse fringe fades
    rgba = np.dstack([
        np.divide(sums, np.maximum(cnt, 1)[:, None]).reshape(ny, nx, 3),
        np.clip(cnt / ref, 0, 1).reshape(ny, nx) ** 0.6 * 0.9,
    ])  # fmt: skip
    ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="lower", aspect="auto", zorder=2, interpolation="nearest")
    ax.plot(RLEVELS, level_stat(y, np.mean), color="#222", lw=1.2, zorder=4)


def strata(k: int, seed: int = 0) -> np.ndarray:
    """A jittered k³ stratification of the unit cube: one point per sub-cell, placed at random
    within it.

    Even coverage without the moiré a plain lattice would beat against redness, and
    deterministic — the seed fixes the picture rather than merely the noise.
    """
    g = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    return (g + np.random.default_rng(seed).random(g.shape)) / k


def draw_cloud_strat(ax: plt.Axes, y: np.ndarray, k: int = 39, s: float = 1.2, alpha: float = 0.3, seed: int = 0):
    """The cloud, stratified rather than sampled at random. k³ marks (39³ ≈ 59k)."""
    rgb = strata(k, seed)
    order = np.random.default_rng(seed + 1).permutation(len(rgb))  # fixed overdraw order
    ax.scatter(redness(rgb)[order], interp_alpha(y, rgb)[order], c=rgb[order], s=s, lw=0, alpha=alpha, zorder=3)


def cells() -> np.ndarray:
    """The (n-1)³ lattice cells, as index arrays into GRID_RGB: 125 cells of 8 corners."""
    n = len(LEVELS)
    idx = np.arange(n**3).reshape(n, n, n)
    return np.array([
        idx[i : i + 2, j : j + 2, k : k + 2].ravel()
        for i in range(n - 1) for j in range(n - 1) for k in range(n - 1)
    ])  # fmt: skip


def draw_sweeps(ax: plt.Axes, y: np.ndarray, k: int = 64, lw: float = 0.6, alpha: float = 0.05):
    """`draw_edges` oversampled: the r-sweep at k² intermediate (g, b), each segment exact.

    Both plotted coordinates are multilinear in rgb — redness is r(1 - g/2 - b/2), and trilinear
    interpolation is multilinear by construction — so along a line of constant (g, b) each is
    *linear* in r. A cell-crossing is therefore a straight segment between two interpolated
    endpoints, with no error at any point along it, and the region the cube maps to is ruled by
    these lines. Sweeping r alone covers the cube: every color sits on exactly one such line.

    Lines rasterize without holes, which is what point sampling could not manage, and the ink
    piles up where many colors map to the same place — the density shows without averaging
    hues together.
    """
    n = len(LEVELS)
    cell = np.stack(np.meshgrid(np.arange(k), np.arange(k), indexing="ij"), axis=-1).reshape(-1, 2)
    gb = (cell + np.random.default_rng(1).random(cell.shape)) / k  # jittered, so no lattice moiré
    rgb = np.empty((len(gb), n, 3))
    rgb[..., 0], rgb[..., 1:] = LEVELS[None, :], gb[:, None, :]
    flat = rgb.reshape(-1, 3)
    pts = np.stack([redness(flat), interp_alpha(y, flat)], axis=-1).reshape(len(gb), n, 2)
    segs = np.concatenate([pts[:, i : i + 2] for i in range(n - 1)])
    mids = np.concatenate([(rgb[:, i] + rgb[:, i + 1]) / 2 for i in range(n - 1)])
    ax.add_collection(LineCollection(list(segs), colors=np.clip(mids, 0, 1), lw=lw, alpha=alpha, zorder=3))


def draw_dither(
    ax: plt.Axes,
    y: np.ndarray,
    k: int = 140,
    px: tuple[int, int] = (480, 360),
    ylim: tuple[float, float] = (-0.3, 1.1),
    seed: int = 0,
):
    """One grid color per pixel, opaque, chosen by lot from everything that maps there.

    Blending is what greys the fills out: where a red cell and a green cell overlap, an average
    paints an olive that no color in the data produces. So each pixel instead picks a single
    winner among the samples landing in it, uniformly at random, and takes its color at full
    opacity. Every sample is quantized to its nearest grid color first, so the whole image is
    painted from the 216 measured colors and nothing else. Over a small neighborhood the mix of
    winners follows the mix of colors, which is the property mean-blending was there to provide.

    Pixels the cube never reaches stay transparent, so the silhouette is the honest one: filled
    where colors land, empty where none do.
    """
    nx, ny = px
    x0, x1 = float(REDNESS.min()), float(REDNESS.max())
    ylo, yhi = ylim
    rgb = strata(k, seed)
    ix = np.rint((redness(rgb) - x0) / (x1 - x0) * (nx - 1)).astype(np.int64)
    iy = np.rint((interp_alpha(y, rgb) - ylo) / (yhi - ylo) * (ny - 1)).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]
    n = len(LEVELS)
    palette = np.rint(rgb[ok] * (n - 1)).astype(np.int64) @ np.array([n * n, n, 1])  # nearest of the 216

    # Sort by pixel, breaking ties at random, then keep the first of each run: a uniform draw
    # among that pixel's samples, without materializing per-pixel lists.
    key = np.random.default_rng(seed + 2).integers(0, 2**31, len(flat))
    order = np.argsort(flat * (2**31) + key, kind="stable")
    sflat = flat[order]
    first = order[np.flatnonzero(np.diff(sflat, prepend=-1))]

    canvas = np.zeros((nx * ny, 4))
    canvas[flat[first], :3] = GRID_RGB[palette[first]]
    canvas[flat[first], 3] = 1.0
    ax.imshow(canvas.reshape(ny, nx, 4), extent=(x0, x1, ylo, yhi), origin="lower", aspect="auto",
              zorder=2, interpolation="nearest")  # fmt: skip


def draw_patches(ax, y: np.ndarray, m: int = 24, ylim: tuple[float, float] = (-0.3, 1.1), a0: float = 0.85):
    """One filled patch per lattice cell, composited rather than pooled.

    A cell's 8 corners are measured colors one grid step apart, so averaging *within* a cell
    blends colors that are already neighbors. Averaging *across* cells is what turns the binned
    version grey, so cells are composited instead: each lays down its own color over what is
    already there, in order of increasing redness, so the red-relevant cells finish on top.
    """
    nx, ny = 480, 360
    x0, x1 = float(REDNESS.min()), float(REDNESS.max())
    ylo, yhi = ylim
    canvas = np.zeros((nx * ny, 4))
    u = strata(m) / (len(LEVELS) - 1)  # sample offsets within one cell, shared by all cells
    cell = cells()
    for c in cell[np.argsort(REDNESS[cell].mean(axis=1))]:
        rgb = GRID_RGB[c[0]] + u
        ix = np.rint((redness(rgb) - x0) / (x1 - x0) * (nx - 1)).astype(int)
        iy = np.rint((interp_alpha(y, rgb) - ylo) / (yhi - ylo) * (ny - 1)).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        flat = iy[ok] * nx + ix[ok]
        cnt = np.bincount(flat, minlength=nx * ny)
        hit = cnt > 0
        src = np.stack([np.bincount(flat, weights=rgb[ok, ch], minlength=nx * ny) for ch in range(3)], axis=-1)
        src[hit] /= cnt[hit, None]
        a = (np.clip(cnt / np.percentile(cnt[hit], 70), 0, 1) * a0)[:, None]  # thin edges stay soft
        canvas[:, :3] = canvas[:, :3] * (1 - a) + src * a  # the over operator, cell by cell
        canvas[:, 3:] = canvas[:, 3:] * (1 - a) + a
    canvas[:, :3] /= np.maximum(canvas[:, 3:], 1e-6)  # accumulated premultiplied; undo for imshow
    ax.imshow(np.clip(canvas, 0, 1).reshape(ny, nx, 4), extent=(x0, x1, ylo, yhi), origin="lower",
              aspect="auto", zorder=2, interpolation="nearest")  # fmt: skip


VARIANTS = {
    "current (216 pts)": draw_full_scatter,
    "silhouette": draw_silhouette,
    "sil. + 8 corners": lambda ax, y: (draw_silhouette(ax, y), draw_points(ax, y, CORNERS)),
    # "sil. + 27 pts": lambda ax, y: (draw_silhouette(ax, y), draw_points(ax, y, MID27, s=16)),
    "cube edges": draw_edges,
    "cloud (60k random)": draw_cloud_pts,
    "cloud (39³ stratified)": draw_cloud_strat,
    "cloud (256³, binned)": draw_cloud,
    "sweeps (64² lines)": draw_sweeps,
    "dither (216 palette)": draw_dither,
    "cell patches": draw_patches,
    # "sil. + edges": lambda ax, y: (draw_silhouette(ax, y, mean_line=False), draw_edges(ax, y, lw=1.8)),
    # "voronoi fill": lambda ax, y: (
    #     draw_voronoi(ax, y),
    #     ax.plot(RLEVELS, level_stat(y, np.mean), color="#222", lw=1.2, zorder=4),
    # ),
}


def r2_sim(y: np.ndarray) -> float:
    """Shape r² against sim¹·⁵, as the grading statistic; NaN where the response is constant."""
    if np.std(y) < 1e-6:
        return float("nan")
    return float(np.corrcoef(y, SIM_TARGET)[0, 1] ** 2)


def overlay_row_steps(fig: plt.Figure, row: list[plt.Axes], values: np.ndarray, **kwargs):
    """A smooth-step across a row of panels: one plateau per panel, at that panel's value.

    The overlay is a transparent axis spanning the row, sharing the panels' y scale, so a
    plateau's height reads off the panels' own axis. Risers land in the gaps between panels.
    NaN values get no plateau; an isolated finite value between NaNs draws as a flat dash.
    """
    boxes = [ax.get_position() for ax in row]
    w, s = boxes[0].width, boxes[1].x0 - boxes[0].x0  # panel width and center spacing
    y0, h = boxes[0].y0, boxes[0].height
    ax_o = fig.add_axes((boxes[0].x0, y0, boxes[-1].x1 - boxes[0].x0, h))
    ax_o.set_axis_off()
    ax_o.set_xlim(-w / 2 / s, len(row) - 1 + w / 2 / s)
    ax_o.set_ylim(*row[0].get_ylim())
    ramp = (s - w) / s  # risers exactly span the gaps, plateaus the panels
    # Split at NaNs: smooth_step per finite run, a plain dash for a lone finite value.
    finite = np.isfinite(values)
    i = 0
    while i < len(values):
        if not finite[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(values) and finite[j + 1]:
            j += 1
        if j > i:
            smooth_step(ax_o, np.arange(i, j + 1), values[i : j + 1], ramp=ramp, **kwargs)
        else:
            ax_o.plot([i - w / 2 / s, i + w / 2 / s], [values[i]] * 2, **kwargs)
        i = j + 1
    return ax_o


# --- Figures ----------------------------------------------------------------


def fig_variants(alpha_map, ctrl: np.ndarray, out: Path):
    """Variant comparison: slice-mean response at op1, clean vs noisy condition."""
    conds = ["either-t100", "either-t2500"]
    fig, axes = plt.subplots(
        len(conds), len(VARIANTS), figsize=(2.05 * len(VARIANTS), 2.1 * len(conds)),
        sharex=True, sharey=True, layout="constrained",
    )  # fmt: skip
    axes = cast(AxesGrid, axes)
    for row, cond in zip(axes, conds, strict=True):
        resp = alpha_map(cond)[:, :, 0].mean(axis=0)
        for ax, (name, draw) in zip(row, VARIANTS.items(), strict=True):
            draw_base(ax, ctrl[:, :, 0].mean(axis=0))
            start = time()
            draw(ax, resp)
            end = time()
            print(f"{name}: {end - start:0.2f}s")
            if cond == conds[0]:
                ax.set_title(name, fontsize=9)
        row[0].set_ylabel(f"{cond}\nα at op1", fontsize=8)
    axes[0][0].set_ylim(-0.3, 1.1)
    for ax in axes[-1]:
        ax.set_xlabel("redness of op1", fontsize=8)
    fig.savefig(out / "variants.png", dpi=150)
    plt.close(fig)


def fig_grid(alpha_map, out: Path, label: str, draw):
    """The full (slice, position) grid at target size, with one candidate encoding.

    Overlaid per row: a red smooth-step at each panel's α for pure red (standing in for the
    red corner mark), and a grey dashed one at each panel's shape r² against sim¹·⁵.
    """
    a = alpha_map("either-t100")
    fig, axes = plt.subplots(5, 6, figsize=(7.2, 6.4), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesGrid, axes)
    for si in range(5):
        r = 4 - si  # embedding at the bottom, as the profile figures do
        for pi in range(6):
            ax = axes[r][pi]
            ax.axhline(0, color="#ccc", lw=0.6, zorder=0)
            draw(ax, a[si, :, pi])
            ax.set_frame_on(False)
            ax.tick_params(labelsize=7, left=False, labelleft=False, bottom=(r == 4), labelbottom=(r == 4))
            if r == 0:
                ax.set_title(POS_NAMES[pi], fontsize=9)
        axes[r][0].set_ylabel(SLICE_NAMES[si], fontsize=8, rotation=0, ha="right", va="center")
    axes[0][0].set_ylim(-0.3, 1.1)
    # One y scale for the whole grid, on the right of the bottom-right panel.
    axes[-1][-1].yaxis.tick_right()
    axes[-1][-1].tick_params(labelsize=7, right=True, labelright=True)
    for ax in axes[-1]:
        ax.set_xlabel("redness", fontsize=7)
    fig.suptitle(f"either-t100 (primary), per (slice, position) — {label}", fontsize=10)
    fig.legend(
        [plt.Line2D([], [], color="#d40000", lw=0.9, alpha=0.75), plt.Line2D([], [], color="#888", lw=0.75, alpha=0.75, ls=(0, (3, 2)))],
        ["α at pure red", "shape r² vs sim¹·⁵"],
        loc="outside lower center", ncols=2, fontsize=8, frameon=False,
    )  # fmt: skip

    # Freeze the layout, then span each row with the two smooth-step overlays. Run the layout
    # engine directly rather than canvas.draw(): both settle the panel boxes identically, but
    # a full draw also paints every mark, and this figure is repainted by savefig anyway.
    engine = fig.get_layout_engine()
    assert engine is not None
    engine.execute(fig)
    fig.set_layout_engine("none")
    for si in range(5):
        row = list(axes[4 - si])
        overlay_row_steps(fig, row, a[si, I_RED, :], color="#d40000", lw=0.5, alpha=0.75)
        overlay_row_steps(fig, row, np.array([r2_sim(a[si, :, pi]) for pi in range(6)]),
                          color="#888", lw=0.75, alpha=0.75, ls=(0, (3, 2)))  # fmt: skip

    fig.savefig(out / f"grid-{label}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".mini/sketches/grading"))
    out: Path = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    seeds, alphas = load()

    def alpha_map(cond: str) -> np.ndarray:
        """Seed-mean (slices, colors, positions)."""
        return np.mean([alphas[f"{cond}-s{s}/alpha"] for s in range(seeds[cond])], axis=0)

    ctrl = alpha_map("lam0")
    corners = lambda ax, y: (draw_silhouette(ax, y), draw_points(ax, y, CORNERS_NO_RED))  # noqa: E731
    cloud = lambda ax, y: draw_cloud_pts(ax, y, s=0.5, alpha=0.22, mean_line=False)  # noqa: E731
    sweeps = lambda ax, y: draw_sweeps(ax, y, lw=0.4, alpha=0.09)  # noqa: E731
    # Panels are ~180px wide at 150dpi, so the canvas and the sample count both come down.
    dither = lambda ax, y: draw_dither(ax, y, k=90, px=(240, 180))  # noqa: E731
    figures = [
        ("variants.png", lambda: fig_variants(alpha_map, ctrl, out)),
        ("grid-corners.png", lambda: fig_grid(alpha_map, out, "corners", corners)),
        ("grid-cloud.png", lambda: fig_grid(alpha_map, out, "cloud", cloud)),
        ("grid-sweeps.png", lambda: fig_grid(alpha_map, out, "sweeps", sweeps)),
        ("grid-dither.png", lambda: fig_grid(alpha_map, out, "dither", dither)),
        # ("grid-edges.png", lambda: fig_grid(alpha_map, out, "edges", VARIANTS["sil. + edges"])),
    ]
    with use_style("base", "light"):
        for name, draw in figures:
            t = time()
            draw()
            print(f"wrote {out}/{name} ({time() - t:.1f}s)")


if __name__ == "__main__":
    main()
