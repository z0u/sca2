---
status: open
tags: [vis]
opened: 2026-07-21
---
# Dark-mode rim on `plot_latent_disc`

The disc's over-the-data rim is a hard-coded `#0005`, which over the `#111` dark fill is effectively invisible — it only reads where data covers it. The new `sca.vis.plot_rgb_cube` uses `light_dark("#0005", "#fff4")` instead, so the two bounds are now drawn differently. Worth unifying, but changing `plot_latent_disc` restyles the published ex-2.9.x figures, so it wants a deliberate pass over those rather than a drive-by edit.
