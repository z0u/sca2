---
status: open
tags: [vis]
opened: 2026-10-21
---
# Module-level `light_dark` inks draw light colors in dark figures

`light_dark()` picks its value from the theme active when it is called. Several reports build their ink dicts at module level (`INK = {"falloff": light_dark(...), ...}`), so the call runs once at import, under the light theme, and the dark render of every `@themed` figure reuses the light inks: a ghost meant to be dark gray comes out near-white on the dark page. The style review of the edit-effects report found it there and turned the inks into small functions called inside the plot.

Reports with a module-level `INK = {` and `light_dark` calls, as of this item: anchor-ramp, slice-terms, soft-pool, split-anti-seeds, and ex-2.2.2 to ex-2.2.8 (`rg -l '^INK = \{' docs/`). Each needs its inks resolved inside the plot function, then a check of its dark PNGs over `#111`, and a republish. A lint in `./go lint` that flags a module-level `light_dark(` call in a report would stop it coming back.
