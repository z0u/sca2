---
status: open
tags: [figures]
opened: 2026-08-18
---
# Migrate the grading-grid slot figure to `GradingCloud`

The progression figure in `docs/m2/d2.1/report.py` now draws with `sca.vis_grading.GradingCloud`, an artist that re-rasters itself at the axes' device size at draw time, so nearest-neighbor resampling is the identity and the dither cannot moiré against the output pixel grid. The grading-grid figure (`grading_grid` in the same report) still uses the fixed-raster `GradingField`, whose `px` is a guess at a size the axes may never render at.

The artist already supports `span`, so the migration is mechanical: one `GradingCloud(ax, a[si, :, pi], span=...)` per slot in place of `field.draw(...)`, dropping the `px` argument, then re-render both keyings of all four conditions and check the slot texture. Each slot rasters at its own (small) device size, so the per-slot geometry is cheap and shared across slots of like size. Once nothing uses `GradingField`, it and `grading_field` can go, and the slot-layout recipe in its `draw` docstring moves to `GradingCloud`.
