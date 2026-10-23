---
status: open
tags: [M2, anchoring, architecture]
opened: 2026-10-24
---
# Replicate the anchor on a standard transformer

The nGPT model here keeps the stream on the unit sphere and is equivariant to a rotation of the residual coordinates, so its privileged basis comes from Adam alone ([rotated-basis](/docs/m2/rotated-basis/report.py)). A standard pre-LayerNorm transformer has per-coordinate gains and biases in its norms, which privilege the axes in the forward pass as well. The anchor held on an ordinary direction under nGPT; whether it holds as well, and whether the baseline needs the same care, on an architecture whose axes are privileged twice over is open. The recipe of record on a LayerNorm model at a few seeds, with the basis measurement of rotated-basis, would say.
