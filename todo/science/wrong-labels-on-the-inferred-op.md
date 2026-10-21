---
status: open
tags: [D2.2, anchoring, labeller]
opened: 2026-10-21
---
# Wrong labels on the inferred op

Claim (e) of the [D2.2 design doc](/docs/m2/d2.2/design.md) is that scarce, noisy labels are enough. On the op word ([ex-2.2.14](/docs/m2/ex-2.2.14/report.py)), the scarce labeller with a fifth of its labels on contexts of the wrong op placed the op as well as the clean one. That has not been tried on the inferred op. The labels there are already wrong in a milder sense, since a label marks the op that generated a context even when its examples favor another.

One arm on the recipe of record (`falloff`) with a fifth of the labels moved to contexts of other ops, at the split-anti-seeds seeds, scored with removal, spill at every dose, task change and lost anchors beside the `falloff` runs at the same seeds. A wrong label pulls a context of another op toward e₁, so the spill is the measurement to watch.
