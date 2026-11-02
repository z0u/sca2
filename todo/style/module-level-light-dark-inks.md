---
status: done
tags: [vis]
opened: 2026-10-21
closed: 2026-11-01
---
# Module-level `light_dark` inks draw light colors in dark figures

`light_dark()` picks its value from the theme active when it is called. Several reports build their ink dicts at module level (`INK = {"falloff": light_dark(...), ...}`), so the call runs once at import, under the light theme, and the dark render of every `@themed` figure reuses the light inks: a ghost meant to be dark gray comes out near-white on the dark page. The style review of the edit-effects report found it there and turned the inks into small functions called inside the plot.

Reports with a module-level `INK = {` and `light_dark` calls, as of this item: anchor-ramp, slice-terms, soft-pool, split-anti-seeds, and ex-2.2.2 to ex-2.2.8 (`rg -l '^INK = \{' docs/`). Each needs its inks resolved inside the plot function, then a check of its dark PNGs over `#111`, and a republish. A lint in `./go lint` that flags a module-level `light_dark(` call in a report would stop it coming back.

## Notes

**2026-11-01, fix** — By then fifteen reports had the pattern, most of them newer than this item: anchor-ramp, anneal-confirm, ex-2.1.10, filtered-corpus, fixed-confirm, late-pull, late-spill-scout, loss-rate, rotated-basis, slice-terms, soft-pool, soft-pool-scout, spill-by-position, spill-timing, and split-anti-seeds. ex-2.2.2 to ex-2.2.8 were fine: their `INK` dicts hold plain pairs, resolved inside the plot. Each report now keeps its inks as (light, dark) pairs and calls `light_dark(*pair)` where the figure draws, and `./go lint` runs `scripts/import_time_inks.py`, which flags a `light_dark` call that no function encloses.
