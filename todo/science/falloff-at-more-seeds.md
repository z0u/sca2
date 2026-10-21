---
status: open
tags: [D2.2, anchoring, seeds]
opened: 2026-10-21
---
# How often the anchor is lost

The recipe of record (`falloff`, from [split-anti-seeds](/docs/m2/split-anti-seeds/report.py)) has run only at seeds 704 to 711, and kept the anchor on all eight. That fits a loss rate as high as three in ten (the 95% upper bound for no losses in eight). Its close variants lost the anchor on one to three seeds of eight: the uniform anti holds, the ramps of [anchor-ramp](/docs/m2/anchor-ramp/report.py), and the ramp over slices 1 to 3 of [soft-pool](/docs/m2/soft-pool/report.py). The soft start of soft-pool kept it on all 24 of its runs. With the recipe known this loosely, an arm that loses one seed in eight cannot be told apart from it.

With no losses, the 95% upper bound falls to about 17% at 16 seeds, 12% at 24, 9% at 32, and 5% at about 60. Full runs at that count are costly, but the outcome is seemingly set in the first forty or so epochs (anchor-ramp E2, soft-pool E2), so a screen could stop early:

- Check the call first on the stored trajectories of the split-anti-seeds, anchor-ramp and soft-pool runs (about a hundred, outcomes known): does "the example answers are not on e₁ by epoch 60" agree with "the full edit removes less than half"? It will miss the late kind of loss (soft-pool, `ramp 1–3`, seed 708).
- Train fresh seeds on the full-length schedule, stopping at about epoch 60, for `falloff` and for the soft start (`step, soft τ`), about 32 seeds each. That is about ten full runs of compute (under $5 at the rate of the last scouts), needs no controls, and compares the two loss rates as well as bounding them.

Training that stops early on an unchanged schedule may need a small change to the experiment code. A short exploratory report of its own, as asked in review of the design doc (2026-10-21).
