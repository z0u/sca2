---
status: open
tags: [D2.2, anchoring, seeds]
opened: 2026-10-21
---
# `falloff` at more seeds

The recipe of record (`falloff`, from [split-anti-seeds](/docs/m2/split-anti-seeds/report.py)) has run only at seeds 704 to 711, and kept the anchor on all eight. That fits a loss rate anywhere from none to about three in ten (the 95% upper bound for no losses in eight). Its close variants lost the anchor on one to three seeds of eight: the uniform anti holds, the ramps of [anchor-ramp](/docs/m2/anchor-ramp/report.py), and the ramp over slices 1 to 3 of [soft-pool](/docs/m2/soft-pool/report.py). With the recipe known this loosely, an arm that loses one seed in eight cannot be told apart from it.

Eight more `falloff` runs at fresh seeds, each with a control at the same seed, scored as in split-anti-seeds (removal and spill at every dose, task change, lost anchors). At the cost of the last scouts (about $11 for 32 runs), that is about $6. A few of the runs could also train to 600 epochs for [the longer-training item](./spill-at-600-epochs.md), though mixing the two would leave fewer seeds for the loss rate.
