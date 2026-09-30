---
name: style-fig
description: Figure conventions for experiment reports. Configuration for latent-space plots, how to draw hyperspheres and RGB-cubes, data-colored marks, smooth-step token sequence charts, grading clouds, sublines (per-token series drawn under the text), theming, captions and nested sub-figures, plus HTML result-table and color-swatch conventions. Use when drawing or revising any figure, writing a figure or table caption, or building a results table, in a report.
---

The M1 reports and the GRaM workshop poster set the house style. Match them: a reader who has seen one SCA figure should be able to read the next one without relearning the encoding. The recurring panel types are packaged as helpers whose docstrings hold the mechanics; this file says which to use when.

## Geometry panels

A geometry panel shows a space (latent scatter, embedding projection, color cube). The space is the message, so draw the domain rather than chart furniture: limits fixed from the domain (never autoscaled, since panels must be comparable across conditions and a collapsed dimension should _look_ collapsed), axes hidden, and the bound drawn instead. The helpers package all of this: `sca.colorcube.plot_latent_disc` for spherical latents; `sca.vis.plot_rgb_cube` for cubes, with `CUBE_VIEWS` explaining the choice of view, `truth=` + `align_to_cube` for recovered cubes, and `s=`/`diameter=` for mark sizing; `sca.vis.draw_cube_bound` when a panel draws its own marks.

Hand-drawn panels follow the same conventions: equal aspect, marks and rim annotations with `clip_on=False` (see `draw_cube_bound`), and 3D projections orthographic and top-down (`ax.view_init(elev=90, azim=-90)`, `ax.set_proj_type('ortho')`, view margin 0) so the panel is a 2D slice.

## Charts

A chart (loss curve, score sweep, schedule) keeps its axes. Use the stylesheet defaults from `mini.vis` and prefer meaningful ticks: a hue axis gets named ticks (Red, Green, Blue) instead of 0–1.

- Draw range bands (`fill_between`) before any summary line, or give the bands a lower `zorder`.
- Encode an _ordinal_ series (depth, size) as ordered shades of one colormap rather than categorical hues, with stops picked via `light_dark` — a colormap's dark end vanishes on a dark background.
- Smooth steps are for the token axis only: positions through a sequence. On any other axis (training steps, layer depths or slices, conditions), join the points with straight segments or leave them as markers, so a smooth step always means a sequence.
- For per-token series, draw plateaus joined by S-curve risers with `mini.vis.smooth_step` and its band/area/marks companions (`smooth_step_marks` puts the weight on the plateaus, for a handful of discrete sites). The docstrings cover `ramp`, `breaks`, `elide`, and `fillet` (straight risers with circular corners of a given radius in points, for when the slope carries rate information); `sca.vis_probes` is the reference implementation.
- For all other ordinal series, use a regular line chart.
- We never use heat maps for sequences. Where the series runs over the tokens of one specific piece of text, use a subline (below) rather than either.
- For a measurement repeated over seeds, draw the seeds: one column per condition, a thin bar behind it spanning the seed range, the individual seeds jittered and faded, and the seed mean on top in the condition's marker. A bar chart of means hides the one seed that behaved differently, which is usually the interesting one. `dots` in `docs/m2/ex-2.2.9/report.py` is the reference.
- Label the roles of an equation in figures with the symbols of $P_o(y \mid a, b)$: *a* and *b* for the operands, *y* for the answer, *o* for the op, and `?` and `=` as printed. Keep op1, op2, and ans for code and data columns. New figures follow this; older reports keep their labels.
- Decide `sharex`/`sharey` from the units: panels measuring the same quantity share; panels measuring different quantities get their own scale, however close the numbers. Two panels with nearly-but-not-quite equal limits look like a bug.

## Confusion matrices

To show where a model puts mass it shouldn't, draw a confusion matrix: rows are the true class (an op, a concept), columns are where the mass lands, and each square counts mass on answers of the column class that the true class cannot give. Blank the diagonal and outline it, since it is a different quantity on a different scale, and give the off-diagonal squares a sequential map that runs from the page color (`light_dark("Blues", "magma")`), so it prints. Print values only above a floor, so the few squares that matter stand out.

Compare conditions with one small matrix each, in a grid with shared axes and one colorbar, so a square can be followed across conditions. Where a condition removes a class (a dropped op, an ablated concept), hatch its row densely, since it has no data, and hatch its column faintly and sparsely over the values, which stay readable. An outline around the column is harder to follow, because its edges fall on the boundaries with the neighboring columns. The column counts mass on the answers of a class the model never learned, so it is the background level for the other squares. Reference: `confusion_draw` in `docs/m2/ex-2.2.18/report.py`; the pattern should carry over to intervention experiments, with the intervention in place of the dropped class.

## Gates and thresholds

Where a hypothesis is scored against a gate, draw the gate in the figure so the reader can see the verdict rather than compute it: a dashed rule at the gate level, a dotted rule for a secondary level under it (a partial-credit bar, a reference value), and the **failing side hatched** — `axhspan(..., facecolor="none", edgecolor=..., hatch="//", lw=0, alpha=0.1, zorder=0)`. Hatching rather than a tint, because a tint would compete with the marks' own color, and color is data. A miss then reads as a region a mark has strayed into, and the eye needs no arithmetic to tell which side is which.

Two mechanics to get right. `axhspan` reads the current y-limits to size itself and then counts as data for autoscaling, so draw it after the marks and put the limits back (`lo, hi = ax.get_ylim()` … `ax.set_ylim(lo, hi)`); everything else the panel draws should come *before* it, or the frozen limits will clip it. And when the same gate appears in two sections, draw it the same way both times — a reader who has decoded one panel should not have to decode it again. `gate_line` in `docs/m2/ex-2.2.9/report.py` packages all of this.

## Legends

A legend belongs to the figure, not to one axes: `fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False)` under `layout="constrained"`, taking the handles from whichever axes carries the full set. Inside the axes it competes with the data for space and lands on top of a hatched gate region; above the panels it reads as a key to the whole figure, which is what it is. The `fig_legend` helper in `docs/m2/ex-2.2.9/report.py` is two lines and worth copying.

Better still is no legend: where the marks carry the encoding themselves (see *Color is data*), a legend is a second copy of the information.

## Color is data

Color the marks with the colors they represent; a legend or colorbar is almost always the wrong tool. Encode comparisons in the mark itself: facecolor shows the model output, edgecolor (or an inset patch, for grids) shows the true input, so damage shows as a face/edge mismatch. Loss-vs-hue lines draw as segments colored by the color at each x (round capstyle to avoid gaps). The same rule holds in prose and HTML tables: name a palette color with an inline swatch, `sca.data.colors.swatch`.

## Grading clouds

A grading figure shows how a response measured per grid color varies with redness. A mean line or envelope hides too much of the structure, and drawing grid vertices is too visually heavy. Use `sca.vis_grading.GradingCloud` to draw a dithered cloud instead. You can use it to draw single charts, or align it with `smooth_step` overlays.

When each color has a distribution of responses rather than one value (per-line damage grouped by the line's dose-carrying color, say), hand the cloud `quantile_stack(values, color)` with `lerp=True`: each sample then draws its height from its color's quantile function, so the cloud's vertical extent is the spread and a bimodal response shows as two bands. The dither gives a rare line almost no ink, so where single unusual lines matter, overlay them: per x level, a thin rule from the least to the most extreme line, capped by a dot in the line's own color, plus a mean line with a halo (`withStroke`) so it survives crossing a dense cloud. Reference: the damage-by-dose figure in `docs/m2/ex-2.2.1/report.py`.

## Sublines

A subline is the text itself with one sparkline per series running underneath, aligned to the tokens: `subline.subline.Subline(…).plot(tokens, series)`, whose docstring holds the mechanics. Tokens may be any width — a wide one draws as a plateau across its glyphs, the same grammar as `smooth_step`. Reach for it when the reader needs to see _which_ token a value lands on; per-character surprisal and predictive entropy over one prompt is the standing case (ex-2.1.1, ex-2.1.2). A matplotlib chart of the same series gives up the alignment with the glyphs, and a heatmap gives up the rate of change.

Its theme reads the page's tokens (`--bg`, `--fg-muted`, `--font-mono`), so it needs no styling from the report; a one-off override goes in `vars=` (custom properties on that SVG alone), never in `css=`, which reaches every subline on the page. Wrap the SVG with `figure_html` and externalize the group, on the same terms as any other figure. Give that wrapper an `aria_label`: an inlined strip has no alt text of its own, and this is what a screen reader announces and what a Markdown render puts in place of the markup — see the `alt-text` skill for what to write.

## Result tables

Authored HTML tables use the shared classes in `docs/report.css` rather than inline `style=`, so central edits restyle every report at once: `report-table` on the `<table>`, `num` on numeric `<th>`s and their `<td>`s, `range` on a spread quoted beside a value, and a caption via `figure_html(..., class_="report-figure")` on the same terms as a figure. Those classes cover the *content* — alignment, tabular figures, and the horizontal scroll box a very wide table falls back to. Layout needs none of them: `mini.lit`'s stylesheet gives every table and figure in a report the full width of the page, with the caption held to the reading measure, by element rather than by class. Tables read as booktabs would print them — three horizontal rules, no cell boxes — so don't draw your own. In a scored table, make it visible at a glance what counts as good: mark each column's desired direction (↑ or ↓, matching the report's glossary) in its header, and bold the values that pass their gate.

Tabular data is great for precision, but it requires a lot of effort to read and interpret. Tables should almost always be accompanied by at least one chart.

## Theming

Every figure goes through `@themed` (see `mini.vis`), which renders the plot function once per theme — its docstring explains why data gets computed outside it. Inside, pick theme-dependent values with `light_dark(light, dark)`. That includes colormaps: a light-only map's pale end disappears on dark, so pick the map itself per theme — `light_dark("RdBu_r", "berlin")` for diverging (`berlin` ships with matplotlib ≥3.11), or a `LinearSegmentedColormap.from_list` running near-background → theme accent for sequential.

Judge dark variants by compositing `_assets/<name>-dark.png` over `#111`: dark exports are transparent, and your Read tool's default matte hides both real problems and false alarms.

## Captions and sub-figures

The title goes in the caption, as its opening phrase — never in `fig.suptitle` (`ax.set_title` still names a panel _within_ a figure). A caption guides decoding ("Each column shows…") and may keep one clause of interpretation where an encoding needs it; findings and their evidence belong in prose cells near the figure. Tables get a caption on the same terms, via `figure_html`.

Panels share one matplotlib figure only when they share axes, a colorbar, or a scale the reader compares across. Otherwise render each as its own `@themed` figure with a short caption, and wrap the group in `figure_html(body, caption=..., aria_label=...)`, whose outer caption holds the shared decoding — each panel then keeps its own size and the row reflows on a narrow viewport. `mini.lit`'s stylesheet styles the nesting; the docstring explains `aria_label`.

Give every figure alt text (see the alt-text skill).

## Prior art

M1's figure code lives in [ex-preppy `src/ex_color/vis/`](https://github.com/z0u/ex-preppy/tree/main/src/ex_color/vis); [references/ex-preppy-vis.md](references/ex-preppy-vis.md) reviews it module by module.
