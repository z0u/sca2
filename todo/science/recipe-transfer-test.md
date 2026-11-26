---
status: open
tags: [D2.2, post-hoc]
opened: 2026-11-04
---
# Does a post-hoc recipe transfer to fresh seeds?

In the post-hoc-compare experiment, a fitter configuration chosen on seven runs and applied to the eighth worked on most runs, but on seed 712 it edited slice 2 while that run held the op at slice 1, and removed almost nothing. The anchored recipe was settled at other seeds and carried over to 712 to 719 unchanged. One run in eight cannot say how often a fixed post-hoc recipe misses, and the finding came from a look at the data, so it needs a preregistered test.

A sketch: train 16 fresh controls and 16 fresh anchored runs; fix each fitter at one configuration chosen on 712 to 719 (with a spill margin, since choosing at the criterion puts about half the runs over it), plus a second arm that picks the slice per run from where a probe peaks; count the runs where the fixed slice misses, against the anchored runs that lose the anchor. A third arm could edit every slice with a pooled fit at each and a dose per slice set by how much of the concept that slice's direction accounts for. That arm is a design of ours rather than a published baseline, and should be labelled as such.

## Notes

**2026-11-22, fitter-transfer** — the third arm has a first look on the stored scores of 712 to 719 (the fitter-transfer report). A pooled fit at every slice, with a uniform dose chosen on the other seven runs, removes 0.14 to 0.42 of the op against 0.88 for the anchor, because the fitted edits at the embedding and slice 1 spill many times the criterion. So a dose per slice (or a slice choice per run) is what that arm would need; a uniform every-slice arm would mostly measure that spill.

**2026-11-26, D2.2 wrap-up** — The D2.2 post makes its claim on what post-hoc-compare and fitter-transfer show: with one configuration chosen on other runs, the fitters carry over less well than the anchor. That rests on one miss in eight for the single-slice edit and on the stored runs of 712 to 719, so the preregistered test sketched above is still the way to put a rate on it. The item stays open.
