---
name: style-fig
description: Figure conventions for experiment reports. Configuration for latent-space plots, how to draw hyperspheres and RGB-cubes, data-colored marks, smooth-step token sequence charts, grading clouds, sublines (per-character series drawn under the text), theming, captions and nested sub-figures, plus HTML result-table and color-swatch conventions. Use when drawing or revising any figure, writing a figure or table caption, or building a results table, in a notebook.
---

The M1 reports and the GRaM workshop poster set the house style. Match them: a reader who has seen one SCA figure should be able to read the next one without relearning the encoding. The recurring panel types are packaged as helpers whose docstrings hold the mechanics; this file says which to use when.

## Geometry panels

A geometry panel shows a space (latent scatter, embedding projection, color cube). The space is the message, so draw the domain rather than chart furniture: limits fixed from the domain (never autoscaled, since panels must be comparable across conditions and a collapsed dimension should _look_ collapsed), axes hidden, and the bound drawn instead. The helpers package all of this: `sca.colorcube.plot_latent_disc` for spherical latents; `sca.vis.plot_rgb_cube` for cubes, with `CUBE_VIEWS` explaining the choice of view, `truth=` + `align_to_cube` for recovered cubes, and `s=`/`diameter=` for mark sizing; `sca.vis.draw_cube_bound` when a panel draws its own marks.

Hand-drawn panels follow the same conventions: equal aspect, marks and rim annotations with `clip_on=False` (see `draw_cube_bound`), and 3D projections orthographic and top-down (`ax.view_init(elev=90, azim=-90)`, `ax.set_proj_type('ortho')`, view margin 0) so the panel is a 2D slice.

## Charts

A chart (loss curve, score sweep, schedule) keeps its axes. Use the stylesheet defaults from `mini.vis` and prefer meaningful ticks: a hue axis gets named ticks (Red, Green, Blue) instead of 0–1.

- Draw range bands (`fill_between`) before any summary line, or give the bands a lower `zorder`.
- Encode an _ordinal_ series (depth, size) as ordered shades of one colormap rather than categorical hues, with stops picked via `light_dark` — a colormap's dark end vanishes on a dark background.
- For per-token series, draw plateaus joined by S-curve risers with `mini.vis.smooth_step` and its band/area/marks companions (`smooth_step_marks` puts the weight on the plateaus, for a handful of discrete sites). The docstrings cover `ramp`, `breaks`, and `elide`; `sca.vis_probes` is the reference implementation.
- For all other ordinal series, use a regular line chart.
- We never use heat maps for sequences. Where the series runs over the characters of one specific string, use a subline (below) rather than either.
- Decide `sharex`/`sharey` from the units: panels measuring the same quantity share; panels measuring different quantities get their own scale, however close the numbers. Two panels with nearly-but-not-quite equal limits look like a bug.

## Color is data

Color the marks with the colors they represent; a legend or colorbar is almost always the wrong tool. Encode comparisons in the mark itself: facecolor shows the model output, edgecolor (or an inset patch, for grids) shows the true input, so damage shows as a face/edge mismatch. Loss-vs-hue lines draw as segments colored by the color at each x (round capstyle to avoid gaps). The same rule holds in prose and HTML tables: name a palette color with an inline swatch, `sca.data.colors.swatch`.

## Grading clouds

A grading figure shows how a response measured per grid color varies with redness. A mean line or envelope hides too much of the structure, and drawing grid vertices is too visually heavy. Use `sca.vis_grading.GradingCloud` to draw a dithered cloud instead. You can use it to draw single charts, or align it with `smooth_step` overlays.

## Sublines

A subline is the text itself with one sparkline per series running underneath, each knot aligned to a character: `subline.subline.Subline(chars_per_line=…, css=…).plot(text, series)`, returning an SVG string. Reach for it when the reader needs to see _which_ character a value lands on — per-character surprisal and predictive entropy over a single prompt is the standing case (ex-2.1.1, ex-2.1.2). A matplotlib chart of the same series gives up the alignment with the glyphs, and a heatmap gives up the rate of change.

The library is vendored from [z0u/subline](https://github.com/z0u/subline) and its docstrings are thin, so the mechanics live here:

- **Values are fractions of the band**: 0 sits on the token baseline, 1 at the top. Scale each series into that range — `nll / log |V|` reads as "fraction of a uniform guess" — and clip it there, because a negative value is dropped silently while the band renders up to 2.
- **`NaN` breaks the path**, which is how a gap is drawn. Position 0 has no prediction, so prepend one and the series lines up with the text.
- Series take `--col-series-1` through `-5` in order, and only those five are defined. Give each a `dasharray` as well as a color: on a 20px band they cross constantly.
- The SVG carries **its own light/dark theme**, independent of `@themed`. Its light background already matches ours; its dark one is `#2a2a2a`, a lighter grey that reads as a box on the notebook. So pass `css="svg { --bg-color: light-dark(#fff, #181c1a); }"`, which is what ex-2.1.1 and ex-2.1.2 both do. Coincident series stay legible because the series paths carry `mix-blend-mode` (multiply on light, screen on dark) — the one thing matplotlib has no answer for.

Wrap the result with `figure_html` and externalize the group on the same terms as any other figure.

## Result tables

Authored HTML tables (built by hand and wrapped in `mo.md`) use the shared classes in `docs/report.css` rather than inline `style=`, so central edits restyle every report at once: `report-table` on the `<table>`, `num` on numeric `<th>`s and their `<td>`s, a `report-table-scroll` wrapper for wide data, and a caption via `figure_html(..., class_="report-figure")` on the same terms as a figure. In a scored table, make it visible at a glance what counts as good: mark each column's desired direction (↑ or ↓, matching the report's glossary) in its header, and bold the values that pass their gate.

## Theming

Every figure goes through `@themed` (see `mini.vis`), which renders the plot function once per theme — its docstring explains why data gets computed outside it. Inside, pick theme-dependent values with `light_dark(light, dark)`. That includes colormaps: a light-only map's pale end disappears on dark, so pick the map itself per theme — `light_dark("RdBu_r", "berlin")` for diverging (`berlin` ships with matplotlib ≥3.11), or a `LinearSegmentedColormap.from_list` running near-background → theme accent for sequential.

Judge dark variants by compositing `_assets/<name>-dark.png` over `#111`: dark exports are transparent, and your Read tool's default matte hides both real problems and false alarms.

## Captions and sub-figures

The title goes in the caption, as its opening phrase — never in `fig.suptitle` (`ax.set_title` still names a panel _within_ a figure). A caption guides decoding ("Each column shows…") and may keep one clause of interpretation where an encoding needs it; findings and their evidence belong in prose cells near the figure. Tables get a caption on the same terms, via `figure_html`.

Panels share one matplotlib figure only when they share axes, a colorbar, or a scale the reader compares across. Otherwise render each as its own `@themed` figure with a short caption, and wrap the group in `figure_html(body, caption=..., aria_label=...)`, whose outer caption holds the shared decoding — each panel then keeps its own size and the row reflows on a narrow viewport. `report.css` styles the nesting; the docstring explains `aria_label`.

Give every figure alt text (see the alt-text skill).

## Prior art

M1's figure code lives in [ex-preppy `src/ex_color/vis/`](https://github.com/z0u/ex-preppy/tree/main/src/ex_color/vis); [references/ex-preppy-vis.md](references/ex-preppy-vis.md) reviews it module by module.
