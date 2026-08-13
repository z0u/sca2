---
status: open
tags: [vis]
opened: 2026-07-21
---
# Slope-capped sublines

`Sparkline._create_path_data` takes its curve knots from glyph ink bounds, so a ramp is always one inter-glyph gap wide however big the jump is. A large step in surprisal therefore renders near-vertical, which reads as a discontinuity and gives up the rate-of-change cue the smooth step exists for. Deriving the ramp width from the jump height instead (cap the on-screen angle, then shrink adjacent ramps so a plateau survives) fixes it, but it restyles the published ex-2.1.1/2.1.2 figures, so it wants an opt-in parameter and a deliberate pass rather than a drive-by edit.
