---
status: done
closed: 2026-09-02
tags: [figures]
opened: 2026-09-02
---
# Grading cloud from a per-color distribution rather than a scalar

`GradingCloud` takes one response value per grid color and dithers the colors over the cube, so each pixel already carries one sampled color. The y coordinate could be sampled the same way: per color, draw y from a distribution rather than a point. Two distributions suggest themselves from ex-2.2.1: the damage over that color's lines, and the model's output distribution over answers on those lines. The spread then reads as the vertical extent of the cloud, and a bimodal response, with some lines removed and some intact, shows as two bands rather than a mean between them.

The raster interpolates a per-color value trilinearly across the cube, so a distribution mode needs a per-color quantile vector in its place, interpolated the same way, with y drawn from the interpolated quantile function per sample. Raised while planning the damage-against-dose figures for ex-2.2.1 (H2/H3, E3); until it exists those are per-line scatters, and the per-color panel under E3 draws the mean.

## Notes

**2026-09-02, Claude** — Implemented as `quantile_stack` in `sca.vis_grading`: the per-color quantile function as a layer stack, handed to `GradingCloud` with `lerp=True`, so the existing loft coordinate is the probability level and each sample is an inverse-transform draw. No raster change was needed. The ex-2.2.1 damage-against-dose figure now uses it, with a mean line and per-dose min-to-max rules overlaid so single unusual lines stay visible; two fresh-eyes reviews compared it with the per-line scatter, and the second preferred the cloud once those overlays were in. In the probe set the darkest colors carry the dose on one to three lines, so their part of the cloud is a band interpolated between few points.
