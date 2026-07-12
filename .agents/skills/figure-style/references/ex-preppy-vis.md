# M1 figure code: what to port from ex-preppy

A review of [ex-preppy `src/ex_color/vis/`](https://github.com/z0u/ex-preppy/tree/main/src/ex_color/vis)
(~1900 lines), the source of the M1 figure style. The overall design — small
`draw_*(ax, …)` helpers that compose into `plot_*` figure functions — is good
matplotlib idiom and worth copying. The pieces below are ranked by how much
we want them in sca2. Everything leans on two ex-preppy types that sca2
replaces: `ColorCube` (a labeled HSV/RGB grid; port alongside whichever
JAX/NumPy equivalent M2 grows) and `Theme` (`theme.val(default, light=…,
dark=…)` maps directly onto `mini.vis.light_dark`).

## Port when needed

- `plot_latent_slices.py` — the heart of the style. `draw_latent_panel`
  establishes the geometry-panel conventions (background disc, stroke
  overlay, axis-off, orthographic top-down view, fixed limits). The 2D
  version is ported: `sca.colorcube.plot_latent_disc` (used by the ex-2.9.1
  and ex-2.9.2 reports) — extend it rather than inlining the panel again.
  `draw_circle_3d` (degenerates cleanly to a line when a dimension is
  projected out) is still unported; bring it over when M2 draws its first
  3D latent grid.
- `stacked_fig.py` — the composite headline figure (two latent panels over a
  color slice over a loss series). Small, and it is the layout the poster
  and report leads with.
- `plot_color_loss.py` — loss-vs-color as `LineCollection` segments colored
  by the color at each x, with round caps to hide the joins, plus named hue
  ticks (`hues3/6/12`). The tick logic is the reusable part.
- `plot_cube.py` — `draw_color_slice` + `annotate_cells`: reconstruction as
  the cell, true color as a centered inset patch. This is the face/edge
  comparison idiom for grids; the half-pixel `_coord_edges` alignment is
  easy to get wrong from scratch.
- `plot_cube_scatter.py` — small and self-contained: data-colored scatter
  with an alpha ramp, dashed regression line using `gapcolor`, and R² in the
  legend label. `plot_similarity.py` is a thin wrapper over it.

## Leave behind

- `helpers.py` (`NbViz`) — a notebook-orchestration wrapper binding plots to
  asset filenames and alt text. sca2's `@themed` + report bundle publisher
  already covers this; porting it would duplicate `mini.vis.nb`. Its alt-text
  strings are still useful as templates.
- `draw_cone_3d` (in `plot_latent_slices.py`) — ~230 lines of root-finding
  to draw a wireframe cone silhouette. This is the "possibly a bit heavy"
  part. Since our views are axis-aligned orthographic, a fixed-view
  approximation (ellipse + two tangent lines computed analytically) would do;
  defer until an experiment needs conical annotations at all.
- `plot_dopesheet.py` — superseded by `mini.temporal.vis.plot_timeline`,
  which sca2 already uses.
- `prettify.py` — sympy-powered tick prettification (`1/3` instead of
  `0.333`). Charming, but not worth a sympy dependency; inline a small
  fraction table if wanted.
- `tabular.py` — HTML/LaTeX result tables with inline color swatches. Not
  figure style; revisit when paper-writing starts (the LaTeX formatter
  produced the M1 paper's tables).
