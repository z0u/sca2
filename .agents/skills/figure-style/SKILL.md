---
name: figure-style
description: Figure conventions for experiment reports. Fixed domain limits and hidden axes for latent-space plots, hypersphere bounds as background discs and RGB-cube bounds as hexagons, data-colored marks, theming, captions and nested sub-figures, plus HTML result-table and color-swatch conventions. Use when drawing or revising any figure, writing a figure or table caption, or building a results table, in a notebook.
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
  bound stays legible where points cover it. For 2D latent panels this whole
  recipe is packaged as `sca.colorcube.plot_latent_disc(ax, z, colors)` —
  use it instead of re-inlining the disc/scatter/rim block.
- The RGB cube gets the same treatment with a hexagon in place of the disc:
  `sca.vis.plot_rgb_cube(ax, rgb, colors)`, at the same (-1.1, 1.1) limits.
  Any flat view of a solid collapses one direction, so there are two, and
  they differ in which one they give up. The default `view='solid'` stands
  the cube on its black corner — white up, black down, red toward the reader
  — so lightness runs up the panel and the silhouette reads as the familiar
  color solid. Use it whenever colors are the data: a grid, a dataset, a
  palette. It deliberately looks unlike the hypersphere disc, so readers
  don't take one for the other. `view='wheel'` looks down the grey diagonal
  instead, putting the six chromatic corners on a regular hexagon with red
  up and collapsing lightness. Prefer it for analysis panels — a probe
  projection, a recovered cube — where hiding a hue axis would occlude the
  errors the panel exists to show.
  A panel that draws its own marks — a lattice with edges between vertices,
  where the caller has to interleave zorders — calls
  `sca.vis.draw_cube_bound(ax, view)` for the silhouette and the panel
  conventions, then draws on top. The silhouette sits at zorder −10 and its
  rim at +10, so hand-placed marks are framed wherever they land.
- Size marks in points (`s=`) for a scatter of arbitrary points, so an
  embedding projection doesn't grow its dots when the vocabulary does. Size
  them in panel units (`diameter=`) when the marks stand for grid cells:
  they then hold their size relative to the cube through any resize, and
  `sca.vis.grid_diameter(levels, view)` gives the value at which a full grid
  tiles with no gaps — no trial and error. Marks sized this way carry no
  edge, since at tiling density the edges become a mesh over the solid.
- Data marks and rim annotations draw with `clip_on=False`, as the disc
  panels' rim markers do. The limits describe the domain, not the ink: a
  mark centered on the silhouette overhangs it by half its width, and
  cropping that turns a circle into a flat-sided blob. Overhang into a
  neighbouring panel is the lesser problem.
- Pass `truth=` (the same points' true RGB) to draw each target as an open
  ring with a stub to where the point actually landed, which is how
  positional error should read on a cube panel. `sca.vis.align_to_cube`
  supplies the coordinates for a recovered cube: it Procrustes-fits a
  rotation, uniform scale and shift onto the true positions and returns the
  leftover residual. Keep that fit rigid — a free linear map absorbs shape
  mismatch into a shear, and then the residual stops being comparable across
  grids, seeds and layers.
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

The same rule holds in prose and HTML tables: name a palette color with an
inline swatch, not words alone. `sca.data.colors.swatch(name)` emits a
`<span class="sw" style="--sw:#rrggbb">` square followed by the name (and
falls back to `<code>` for non-palette text, e.g. a stray hex completion).

## Result tables and swatches

Authored HTML tables (built by hand and wrapped in `mo.Html`, not marimo's
`mo.ui.table`) use the shared classes in `docs/report.css` rather than inline
`style=` — central edits then restyle every report at once:

- `class="report-table"` on the `<table>`. Numeric columns right-align with
  tabular figures when *both* the `<th>` and its `<td>`s carry `class="num"`;
  headers are left-aligned otherwise.
- Wrap a wide grid (e.g. the per-seed completions table) in
  `<div class="report-table-scroll">` so it scrolls inside its own box
  instead of wrapping cells. This allows reports to be viewed on small screens,
  so you'll usually want to use this for tabular data.
- When a column's cells hold `swatch(...)` squares, give its header a ghost
  swatch — `swatch(None)`, a transparent `.sw-ghost` placeholder — so the
  header text starts at the same indent as the swatched values below it.
- Caption the table the same way you would a figure (see below), by wrapping it
  in `figure_html(..., caption=..., class_="report-figure")`.

## Theming and annotation

Every figure goes through `@themed` (see `mini.vis`); inside the plot
function pick theme-dependent values with `light_dark(light, dark)`.
Overlay lines that must survive a busy background use `gapcolor` — e.g.
black dashes with a light gap color in light mode, white with dark in dark
mode — rather than a heavier stroke.

Give every figure alt text (see the alt-text skill).

## Captions

The title goes in the caption, as its opening phrase — not drawn inside the
figure with `fig.suptitle`. A caption then decodes the ink: what the rows,
columns, marks, and shading mean, and how to read an unusual encoding.
Findings, controls, and interpretation belong in prose cells near the figure,
in paragraph form — a result quoted only in a caption is easy to miss and hard
to cross-reference. When an encoding needs a word of interpretation to be
readable at all (say, what a step across an elided region signifies), keep one
clause in the caption and put the supporting evidence in prose.

Tables get a caption on the same terms — see the wrapper above.

`ax.set_title` is still how you name a panel *within* one figure — see the
hidden-axes note above. What moves to the caption is the figure-level title.

## Sub-figures

Panels only need to live in one matplotlib figure when they share axes, a
colorbar, or a scale the reader is meant to compare across. Otherwise prefer
separate nested figures: render each panel as its own `themed` figure with a
short caption naming just that panel, then wrap the group in
`figure_html(body, caption=..., aria_label=...)`, whose outer `<figcaption>`
carries the shared decoding. Each panel then keeps its own size and the row
reflows to a stack on a narrow viewport, rather than every panel shrinking
together. `report.css` styles `figure:has(> figure)` for exactly this; use
`aria_label` rather than `role="img"` on the outer figure, or the sub-captions
stop being navigable.

Sequential/heatmap palettes must be theme-adaptive too: `@themed` renders a
light and a dark variant, so pick the colormap itself with `light_dark(...)`
rather than hard-coding a light-only map like `"Blues"`, whose pale low end
disappears on a dark background. Build one with
`LinearSegmentedColormap.from_list` running from a near-background low to a
theme accent high, e.g.
`LinearSegmentedColormap.from_list("leak", light_dark(["#eef3f7", "#1a5f8a"], ["#20242a", "#6ab0d4"]))`.
Diverging maps swap the same way but a named pair usually suffices: `RdBu_r`
reads well in light mode but its white midpoint and pale ends wash out on a
dark background, so pair it with a dark-centered perceptually-uniform map —
`cmap=light_dark("RdBu_r", "berlin")` (`berlin` ships with matplotlib ≥3.11).
Cell text over such a matrix flips on *both* axes: theme and cell saturation,
e.g. `color=light_dark("#fff", "#000") if saturated else light_dark("#000", "#fff")`.

Marks drawn over a variable or heatmap background (text ×, scatter dots)
need a contrasting halo so they read on any cell, whatever color sits
underneath: `path_effects=[pe.withStroke(linewidth=2, foreground=light_dark("#ffffff", "#000000"))]`
(`import matplotlib.patheffects as pe`), or the draw-twice halo technique the
color-matrix figure uses.

## What a figure costs to render

`@themed` calls the plot function **twice** — once per theme — so everything
inside it is paid twice, including work that has nothing to do with color.
Only the drawing genuinely differs between the two passes. Compute the data
in the cell, outside the decorated function, and let the plot function receive
it and draw:

```python
_null = {w: random_angle_null(w) for w in _widths}  # once, in the cell

@themed(name="angles", ...)
def _plot():
    ...  # draws _null; no fitting, sampling or bootstrapping in here
```

This matters most for anything sampled or fitted — a null distribution, a
bootstrap, a bisection. A 5-second resample inside a plot function is a
10-second one.

Reach for vectorized NumPy over Python comprehensions in that computation.
A list comprehension calling a NumPy function per item pays interpreter and
dispatch overhead on every element, and `np.linalg`'s decompositions
(`qr`, `svd`, `solve`, `inv`) all accept leading batch dimensions, so a stack
is one LAPACK call instead of *n*. The batch form is the same arithmetic — it
gives bit-identical results, so it's safe to apply to a published figure:

```python
# 2000 QR/SVD pairs, one call each, ~4x faster than the comprehension
pairs = rng.normal(size=(n, 2, width, 3))
angles = gm.principal_angles(pairs[:, 0], pairs[:, 1])
```

Likewise for nearest-neighbour work: `argmin` over distances doesn't need the
distances themselves. Expanding the squared distance to `|v|^2 - 2 v.x` turns
a materialized (N, V, K) difference into one (N, V) matmul — see
`sca.baselines.precision_limited_acc`.

Comprehensions are fine for assembling a handful of panels or labels; the rule
is about per-element numerics.

## Prior art

M1's figure code lives in
[ex-preppy `src/ex_color/vis/`](https://github.com/z0u/ex-preppy/tree/main/src/ex_color/vis).
[references/ex-preppy-vis.md](references/ex-preppy-vis.md) reviews it
module by module: which helpers are worth porting when the M2 experiments
need them, and which parts (the notebook wrapper, the cone solver) to leave.
