---
name: figure-style
description: Figure conventions for experiment reports, carried over from M1 (ex-preppy). Fixed domain limits and hidden axes for latent-space plots, hypersphere bounds as background discs, data-colored marks, theming. Use when drawing or revising any figure in a report notebook.
---

The M1 reports and the GRaM workshop poster set the house style. Match them:
a reader who has seen one SCA figure should be able to read the next one
without relearning the encoding.

## Geometry panels vs. charts

Decide which kind of panel you are drawing.

A geometry panel shows a space (latent scatter, embedding projection). The
space is the message, so draw the domain rather than the chart furniture:

- Fix the limits from the domain, not the data. Latent plots of a unit
  hypersphere use (-1, 1) with a margin of about ±0.1, i.e.
  `ax.set_xlim(-1.1, 1.1)`. Never let autoscale infer limits from data
  bounds: panels must be comparable across conditions, and a collapsed
  dimension should *look* collapsed.
- Hide the axes entirely (`ax.set_axis_off()`). Represent the hypersphere
  bound instead: a background disc behind the data
  (`light_dark('#eee', '#111')`; `#8888` if a theme-neutral value is needed),
  plus a semi-transparent stroke drawn *over* the data (`#0005`, lw 1) so the
  bound stays legible where points cover it.
- Equal aspect. For 3D projections: orthographic, viewed top-down, so the
  panel reads as a 2D slice — `ax.view_init(elev=90, azim=-90)`,
  `ax.set_proj_type('ortho')`, and set the view margin to 0.
- Since the axes are hidden, name the panel (`ax.set_title('ablated')`) and
  annotate meaningful directions: the anchor as a coordinate label like
  `(1, 0, 0, 0)` with a small marker at the rim, intervention directions as
  cones or dashed lines.

A chart (loss curve, score sweep, schedule) keeps its axes. Use the
stylesheet defaults from `mini.vis` and prefer meaningful ticks: a hue axis
gets named ticks (Red, Green, Blue), not 0–1.

For an *ordinal* series (depth, size), encode order as ordered shades of one
colormap rather than categorical hues — but pick the stops with `light_dark`:
a colormap's dark end vanishes on a dark background (e.g. viridis
`[0.75, 0.45, 0.1]` in light mode, `[0.8, 0.5, 0.25]` in dark). Judge the
dark variant properly: exported figures have transparent backgrounds, so
composite `_assets/<name>-dark.png` over `#111` first — a viewer's default
matte hides both real problems and false alarms.

A 2D geometry panel in full:

```python
from matplotlib.patches import Circle
from mini.vis import light_dark

def draw_latent_panel(ax, z, facecolors, edgecolors=None):
    ax.add_patch(Circle((0, 0), 1, facecolor=light_dark("#eee", "#111"), zorder=-10))
    ax.scatter(z[:, 0], z[:, 1], c=facecolors, s=22, edgecolors=edgecolors, lw=0.5)
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="#0005", lw=1, zorder=10))
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_axis_off()
```

## Color is data

In the color domain, color the marks with the colors they represent; a
legend or colorbar is almost always the wrong tool. Encode comparisons in
the mark itself: facecolor shows the model output (reconstruction),
edgecolor (or an inset patch, for grids) shows the true input, so damage
reads as a face/edge mismatch. Loss-vs-hue lines are drawn as segments
colored by the color at each x (round capstyle to avoid gaps).

## Theming and annotation

Every figure goes through `@themed` (see `mini.vis`); inside the plot
function pick theme-dependent values with `light_dark(light, dark)`.
Overlay lines that must survive a busy background use `gapcolor` — e.g.
black dashes with a light gap color in light mode, white with dark in dark
mode — rather than a heavier stroke.

Give every figure alt text (see the alt-text skill), and set titles on the
figure, not in surrounding Markdown, so exported PNGs are self-contained.

## Prior art

M1's figure code lives in
[ex-preppy `src/ex_color/vis/`](https://github.com/z0u/ex-preppy/tree/main/src/ex_color/vis).
[references/ex-preppy-vis.md](references/ex-preppy-vis.md) reviews it
module by module: which helpers are worth porting when the M2 experiments
need them, and which parts (the notebook wrapper, the cone solver) to leave.
