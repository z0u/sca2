---
status: open
tags: [D2.2, anchoring, schedules, sobol-search]
opened: 2026-10-09
---
# A full sweep of the recipe, once the lean is fixed

The recipe of record has changed piece by piece since the D2.1 survey ([ex-2.1.12](/docs/m2/ex-2.1.12/report.py)): the grammar, the op set, the model, the training length, the pull and its pooling. Its schedules and weights have not been tuned together since. Once the pull is kept off the early slices (see [pull-slice-restrictions](./pull-slice-restrictions-at-more-seeds.md)), the optimum moves again, so that is the time to sweep.

The shape that worked in D2.1: ablate, then search. First replace each schedule with flat levels that straddle its range (the anchor warm-up and anneal, the anti hold and anneal, the learning-rate cosine), which deletes whatever dimension a flat level matches. Then a Sobol survey over what is left (anchor weight, an absolute anti weight, τ, the peak rate, perhaps epochs), about 32 trials at one seed, scored on task score, removal, and spill, with the best few confirmed at three seeds. One seed per trial is enough for the survey because the spread we care about is across trials; the confirmation covers the seeds. Proposed by the reviewer on the lean review.
