"""Sketch: grading-cloud figures drawn with `sca.vis_grading.GradingField`.

The dithered fill was prototyped here (see git history) and now lives in
`sca.vis_grading`; this script exercises it on ex-2.1.10's published arrays.
It writes the per-(slice, position) grid at dpr 1-3, a magnified single panel
comparing the dpr levels, and the per-slice-row variant: one Axes per slice,
each position a slot on x sized to match the smooth-step plateaus, so the
overlays draw natively in the same coordinates as the clouds — no frozen-layout
overlay axes — and the redness tick clutter goes away.

Data is cached locally after the first run so iteration skips the network.
`--bench` adds a stage profile of the original `grading_sketch.draw_dither` and
a timing comparison against it, asserting both fills cover identical pixels.
Not a notebook, so the site build ignores it. Run from the repo root:

    uv run python docs/m2/grading_dither.py [--out DIR] [--bench]
"""

import argparse
from pathlib import Path
from time import time
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

import grading_sketch as gs
from mini.vis import AxesRow, smooth_step, use_style
from sca.vis_grading import GradingField, I_RED

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


OVERLAYS: list[dict[str, Any]] = [
    dict(color="#d40000", lw=0.5, alpha=0.75),  # α at pure red
    dict(color="#888", lw=0.75, alpha=0.75, ls=(0, (3, 2))),  # shape r² vs sim¹·⁵
]
OVERLAY_NAMES = ["α at pure red", "shape r² vs sim¹·⁵"]


def stepped(ax: plt.Axes, values, sw: float, **kwargs):
    """A smooth-step over position slots: plateaus span each slot's width *sw*, risers the
    gaps. NaN values get no plateau; an isolated finite value between NaNs draws as a dash.
    """
    v = np.asarray(values, float)
    finite = np.isfinite(v)
    i = 0
    while i < len(v):
        if not finite[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(v) and finite[j + 1]:
            j += 1
        if j > i:
            smooth_step(ax, np.arange(i, j + 1), v[i : j + 1], ramp=1 - sw, **kwargs)
        else:
            ax.plot([i - sw / 2, i + sw / 2], [v[i]] * 2, **kwargs)
        i = j + 1


def fig_rows(a: np.ndarray, out: Path, field: GradingField, sw: float = 0.72):
    """The per-slice-row layout: one Axes per slice, positions as slots on x.

    Collapsing each row of panels into a single Axes puts the clouds and the overlay
    steps in one coordinate system, so `overlay_row_steps`' frozen-layout machinery
    isn't needed, and each panel's redness axis goes (stated once in the caption
    instead: redness runs left to right within a slot).
    """
    n_pos = a.shape[2]
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 6.4), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    for si in range(5):
        ax = axes[4 - si]  # embedding at the bottom, as the profile figures do
        ax.axhline(0, color="#ccc", lw=0.6, zorder=0)
        for pi in range(n_pos):
            field.draw(ax, a[si, :, pi], span=(pi - sw / 2, pi + sw / 2))
        stepped(ax, a[si, I_RED, :], sw, **OVERLAYS[0])
        stepped(ax, [gs.r2_sim(a[si, :, pi]) for pi in range(n_pos)], sw, **OVERLAYS[1])
        ax.set_ylabel(gs.SLICE_NAMES[si], fontsize=8, rotation=0, ha="right", va="center")
        ax.set_frame_on(False)
        ax.tick_params(labelsize=7, left=False, labelleft=False, bottom=False)
    axes[0].set_xlim(-0.5, n_pos - 0.5)
    axes[0].set_ylim(-0.3, 1.1)
    axes[-1].set_xticks(range(n_pos), gs.POS_NAMES)
    axes[-1].tick_params(labelbottom=True)
    # One y scale for the whole figure, on the right of the bottom row.
    axes[-1].yaxis.tick_right()
    axes[-1].tick_params(labelsize=7, right=True, labelright=True)
    fig.suptitle("either-t100 (primary), per (slice, position) — rows", fontsize=10)
    handles: list[dict[str, Any]] = [kw | {"lw": 1.2} for kw in OVERLAYS]  # legible at legend scale
    fig.legend(
        [plt.Line2D([], [], **kw) for kw in handles], OVERLAY_NAMES,
        loc="outside lower center", ncols=2, fontsize=8, frameon=False,
    )  # fmt: skip
    fig.savefig(out / "grid-rows.png", dpi=150)
    plt.close(fig)


def fig_detail(a: np.ndarray, out: Path, k: int, px: tuple[int, int], dprs: tuple[int, ...]):
    """One panel per dpr, magnified: saved well above the canvas resolution, so each canvas
    pixel spans several image pixels and the anti-aliasing itself is what varies.
    """
    y = a[4, :, 0]  # slice 4 at op1: a dense shoulder and a sparse tail in one panel
    fig, axes = plt.subplots(1, len(dprs), figsize=(2.6 * len(dprs), 2.4), sharey=True, layout="constrained")
    for ax, dpr in zip(cast(AxesRow, axes), dprs, strict=True):
        GradingField(k=k, px=px, dpr=dpr).draw(ax, y)
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
    """Stage profile, per-panel timing, and the full grid, original vs GradingField at dpr=1
    (where the two run the same lottery on the same samples, so coverage must agree).
    """
    panels = [a[si, :, pi] for si in range(a.shape[0]) for pi in range(a.shape[2])]
    base = lambda ax, y: gs.draw_dither(ax, y, k=k, px=px)  # noqa: E731

    print(f"baseline stages, one panel (k={k}, px={px}):")
    profile_baseline(panels[0], k, px)

    t = time()
    field = GradingField(k=k, px=px, dpr=1)
    print(f"GradingField precompute: {time() - t:.2f}s (once per figure)")

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
            field = GradingField(k=k, px=px, dpr=dpr)
            gs.fig_grid(alpha_map, out, f"dither-dpr{dpr}", field.draw)
            print(f"wrote {out}/grid-dither-dpr{dpr}.png ({time() - t:.1f}s)")
        fig_detail(a, out, k, px, dprs)
        print(f"wrote {out}/dither-detail.png")
        t = time()
        # Slots render at ~0.78 x 1.0 in at 150 dpi; k scales to the smaller canvas by itself.
        fig_rows(a, out, GradingField(k=k, px=(120, 150)))
        print(f"wrote {out}/grid-rows.png ({time() - t:.1f}s)")


if __name__ == "__main__":
    main()
