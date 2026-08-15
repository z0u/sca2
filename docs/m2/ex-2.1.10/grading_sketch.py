"""Sketch: simplified grading panels for per-(slice, position) grids.

The published grading figure draws all 216 colors as marks, which is too much
ink to tile 5 slices x 6 positions. This sketch prototypes lighter encodings on
ex-2.1.10's published arrays and writes PNGs for comparison:

- variants.png: one column per encoding, on a clean and a noisy condition
- grid-corners.png / grid-edges.png: the full (slice, position) grid, small

Not a notebook, so the site build ignores it. Run from the repo root:

    uv run python docs/m2/ex-2.1.10/grading_sketch.py [--out DIR]
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from mini.store import project_store
from mini.vis import AxesGrid, use_style
from sca.colorcube import redness
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS

LEVELS = np.asarray(GRIDS["v216"], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
RLEVELS = np.unique(np.round(REDNESS, 6))  # 29 distinct redness values

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


def draw_silhouette(ax: plt.Axes, y: np.ndarray, mean_line: bool = True):
    """The scatter's envelope as a filled band: min to max at each redness level."""
    ax.fill_between(RLEVELS, level_stat(y, np.min), level_stat(y, np.max), color="#777", alpha=0.28, lw=0, zorder=2)
    if mean_line:
        ax.plot(RLEVELS, level_stat(y, np.mean), color="#222", lw=1.2, zorder=4)


def subgrid(levels: list[float]) -> np.ndarray:
    """Mask of GRID_RGB restricted to the given channel levels."""
    return np.isin(np.round(GRID_RGB, 6), np.round(np.array(levels), 6)).all(axis=1)


CORNERS = subgrid([0.0, 1.0])  # the 8 cube corners
MID27 = subgrid([0.0, 9 / 15, 1.0])  # 3x3x3 subgrid; 9/15 is the level nearest the midpoint


def draw_points(ax: plt.Axes, y: np.ndarray, mask: np.ndarray, s: float = 24):
    ax.scatter(REDNESS[mask], y[mask], c=GRID_RGB[mask], s=s, lw=0.5, zorder=5, edgecolors="#00000055")


def cube_edges() -> list[np.ndarray]:
    """Indices into GRID_RGB for the 12 cube edges, ordered along the varying channel."""
    n = len(LEVELS)
    idx = np.arange(n**3).reshape(n, n, n)
    edges = []
    for axis in range(3):
        for a in (0, n - 1):
            for b in (0, n - 1):
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
    ax.add_collection(LineCollection(segs, colors=cols, lw=lw, capstyle="round", zorder=5))


VARIANTS = {
    "current (216 pts)": draw_full_scatter,
    "silhouette": draw_silhouette,
    "sil. + 8 corners": lambda ax, y: (draw_silhouette(ax, y), draw_points(ax, y, CORNERS)),
    "sil. + 27 pts": lambda ax, y: (draw_silhouette(ax, y), draw_points(ax, y, MID27, s=16)),
    "cube edges": draw_edges,
    "sil. + edges": lambda ax, y: (draw_silhouette(ax, y, mean_line=False), draw_edges(ax, y, lw=1.8)),
}


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
            draw(ax, resp)
            if cond == conds[0]:
                ax.set_title(name, fontsize=9)
        row[0].set_ylabel(f"{cond}\nα at op1", fontsize=8)
    axes[0][0].set_ylim(-0.3, 1.1)
    for ax in axes[-1]:
        ax.set_xlabel("redness of op1", fontsize=8)
    fig.savefig(out / "variants.png", dpi=150)
    plt.close(fig)


def fig_grid(alpha_map, ctrl: np.ndarray, out: Path, label: str, draw):
    """The full (slice, position) grid at target size, with one candidate encoding."""
    a = alpha_map("either-t100")
    fig, axes = plt.subplots(5, 6, figsize=(7.2, 6.4), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesGrid, axes)
    for si in range(5):
        r = 4 - si  # embedding at the bottom, as the profile figures do
        for pi in range(6):
            ax = axes[r][pi]
            draw_base(ax, ctrl[si, :, pi])
            draw(ax, a[si, :, pi])
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(POS_NAMES[pi], fontsize=9)
        axes[r][0].set_ylabel(SLICE_NAMES[si], fontsize=8, rotation=0, ha="right", va="center")
    axes[0][0].set_ylim(-0.3, 1.1)
    for ax in axes[-1]:
        ax.set_xlabel("redness", fontsize=7)
    fig.suptitle(f"either-t100 (primary), per (slice, position) — {label}", fontsize=10)
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
    with use_style("base", "light"):
        fig_variants(alpha_map, ctrl, out)
        fig_grid(alpha_map, ctrl, out, "corners", VARIANTS["sil. + 8 corners"])
        fig_grid(alpha_map, ctrl, out, "edges", VARIANTS["sil. + edges"])
    print(f"wrote {out}/variants.png, grid-corners.png, grid-edges.png")


if __name__ == "__main__":
    main()
