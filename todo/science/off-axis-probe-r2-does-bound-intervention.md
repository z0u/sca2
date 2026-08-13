---
status: open
tags: [methodology, anchoring, ex-2.1.6]
---
# An off-axis probe R² does not bound an intervention

Ex-2.1.6 reported redness still readable at R² ≈ 0.83 with the anchor direction removed, and the post-hoc check showed greenness and blueness score the same: it is the color cube the task needs, not a red-specific copy. So probe recoverability answers "can a linear readout still find it", while the intervention question is what the model's own computation does under an edit — which is what M1 measured as damage. Worth a methodology note wherever we quote leakage: the number sizes the concentration achieved, not the intervention's headroom.

Partly settled in ex-2.1.7: the probe's floor is now computed rather than guessed. `redness_rgb_floor()` fits the same ridge to the raw RGB values and gets R² = 0.863, so the with-anchor-removed probe cannot fall far below that while the task keeps the cube. Every condition lands there, the report says so next to the figure, and the number no longer reads as a concentration measurement. It does *not* size the concentration achieved either — see the item below.
