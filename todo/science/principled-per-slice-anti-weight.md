---
status: open
tags: [D2.2, anchoring, schedules]
opened: 2026-10-12
---
# A principled per-slice anti weight

The slice-terms experiment set the anti-subspace weight by hand: a hold of 0.2 on the embedding and block 1, and 0.03 on blocks 2 to 4 (`AntiSpec.hold` as a tuple). It kept most of the selectivity of the high hold everywhere and gave back the removal, but the split point and the two levels are a guess fitted to one recipe.

Two more principled shapes to try:

- A weight that falls off linearly with depth, from the embedding to block 4, so no slice boundary has to be chosen.
- An anti term pooled over slices with something like mellowmax, so the weight goes where the states are most aligned with e₁. It is not clear how this would work without labels, since the anti term acts on every position and line, labeled or not.

Either would be a new arm against `split-anti` at the same seeds, scored with removal and spill at every dose as in slice-terms.

## Notes

**2026-10-13, split-anti-seeds** — The linear falloff has been tried: in [the split-anti-seeds experiment](/docs/m2/split-anti-seeds/report.py) it removed about as much as `split-anti` and kept every run within the spill criterion, and it is now the recipe of record. It missed the task gate on one op at one seed. The pooled anti term is still open, and so is tuning the falloff itself, for example a lower anti weight on the earliest slices.
