---
status: open
tags: [D2.2, anchoring, schedules]
opened: 2026-10-13
---
# An anchor weight that ramps up with depth

The recipe of record pulls `difference` toward e₁ at slices 2 to 4 with one weight, and not at all on the embedding or slice 1: a step in depth. The split-anti-seeds experiment found that a straight-line falloff of the anti weight did as well as the split it replaced, and was chosen as the recipe of record. The mirror of that shape for the pull is an anchor weight that rises with depth, low on the early slices and full at slice 4, so no slice boundary has to be chosen for either term.

`AnchorSpec` has one weight for every pulled slice, so this needs a per-slice weight, like the `AntiSpec.hold` tuple. It would be an arm against `falloff` at the split-anti-seeds seeds (704 to 711), scored with removal and spill at every dose. One open choice is whether the ramp starts at zero on the embedding or above zero on slice 1, since the late-pull experiment moved the pull off slice 1 to keep the lean off the embedding table.

Raised in the review of the split-anti-seeds results.
