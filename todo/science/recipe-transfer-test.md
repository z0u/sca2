---
status: open
tags: [D2.2, post-hoc]
opened: 2026-11-04
---
# Does a post-hoc recipe transfer to fresh seeds?

In the post-hoc-compare experiment, a fitter configuration chosen on seven runs and applied to the eighth worked on most runs, but on seed 712 it edited slice 2 while that run held the op at slice 1, and removed almost nothing. The anchored recipe was settled at other seeds and carried over to 712 to 719 unchanged. One run in eight cannot say how often a fixed post-hoc recipe misses, and the finding came from a look at the data, so it needs a preregistered test.

A sketch: train 16 fresh controls and 16 fresh anchored runs; fix each fitter at one configuration chosen on 712 to 719 (with a spill margin, since choosing at the criterion puts about half the runs over it), plus a second arm that picks the slice per run from where a probe peaks; count the runs where the fixed slice misses, against the anchored runs that lose the anchor. A third arm could edit every slice with a pooled fit at each and a dose per slice set by how much of the concept that slice's direction accounts for. That arm is a design of ours rather than a published baseline, and should be labelled as such.
