---
status: open
tags: [D2.2, anchoring, schedules, sobol-search]
opened: 2026-10-09
priority: high
---
# A full sweep of the recipe around `falloff`

The recipe of record has changed piece by piece since the D2.1 survey ([ex-2.1.12](/docs/m2/ex-2.1.12/report.py)): the grammar, the op set, the model, the training length, the pull and its pooling. Its schedules and weights have not been tuned together since. The lean is now fixed: the recipe of record is `falloff` ([split-anti-seeds](/docs/m2/split-anti-seeds/report.py)), with the pull on blocks 2 to 4 and an anti hold that falls in a straight line from 0.2 at the embedding to 0.03 at block 4. The sweep is centered on it.

The shape that worked in D2.1: ablate, then search. First replace each schedule with flat levels that straddle its range (the anchor warm-up and anneal, the anti hold and anneal, the learning-rate cosine), which deletes whatever dimension a flat level matches. Then a Sobol survey over what is left (anchor weight, the two end points of the anti falloff, τ, the peak rate, perhaps epochs), about 32 trials at one seed, scored on task score, removal, and spill, with the best few confirmed at three seeds. One seed per trial is enough for the survey because the spread we care about is across trials; the confirmation covers the seeds. Proposed by the reviewer on the lean review.

## Notes

**2026-10-09, Claude** — [Late pull](/docs/m2/late-pull/report.py) is the lean fix this item waited on: the pull on blocks 2 to 4, the anti term on every slice, and the anti weight held at 2× the anchor weight in place of the recipe 0.3×, which brought every run within the criterion at some cost in removal. The review of that report asked for the schedule part of this sweep in a specific shape: the start and end weight of the anchor schedule crossed with the start and end weight of the anti schedule, compared against the schedule shapes of M1 (one weight high early and easing off as the other ramps up and holds; the two crossing about a third of the way in). It also suggested an anti weight that is lower at the deeper blocks, since the anti term acts on the pulled blocks too and E4 of that report suggests the hold presses on the pull there. A new recipe of record is to be proposed first, and this sweep tunes around it.

**2026-10-13, holiday tidy-up** — The new recipe of record came first, as the note above asked: `falloff`, chosen in split-anti-seeds. Shortlisted as one of the two candidates for the next experiment after the [anchor ramp](./anchor-weight-that-ramps-up-with-depth.md), beside [the write bound and landing](./write-bound-and-landing-on-the-inferred-op.md). The ramp result may change which schedule shapes the sweep should cover.

**2026-10-20, housekeeping** — The [anchor ramp](/docs/m2/anchor-ramp/report.py) is in, and the step pull stays. Its one lead for this sweep: lost anchors followed the pull on slices 1 to 3 (a post hoc sum, eight seeds per arm), so the pull on the early slices may be worth a dimension of its own if the sweep goes past a single anchor weight.

**2026-10-24, backlog refresh** — Three results make the sweep cheaper than planned. The D2.2 baseline is now the median over the other coordinate axes taken in the run itself ([reference-baseline](/docs/m2/reference-baseline/report.py)), so a trial needs no control of its own, and controls can run now and then to check the agreement. The early call of [the loss-rate screen](/docs/m2/loss-rate/report.py) settles a lost anchor by about epoch 64. And in [spill-timing](/docs/m2/spill-timing/report.py) the spill averaged over a condition at epoch 64 separated the condition that spills early from the two that don't, so the survey could stop trials that spill early at about a sixth of the cost; the late kind of spill still needs the confirmation runs to go to the end. Which condition the sweep centers on may change first: [step, soft τ as the recipe of record](./step-soft-as-recipe-of-record.md).
