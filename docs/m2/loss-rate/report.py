# ruff: noqa: B018
# title: How often the anchor is lost

# An exploratory report: an early call on whether a run kept the anchor, checked on the stored runs of split-anti-seeds,
# anchor-ramp and soft-pool, then used to screen 32 new seeds each of `falloff` and the soft start, stopped at epoch 64.
# `experiment.py` beside this script trains the screen runs.

r"""
# How often the anchor is lost

/// tip |
<!-- lede -->
TODO: the question (how often `falloff` and the soft start lose the anchor), and which way it came out.
///

The recipe of record, `falloff` from [the split-anti-seeds experiment](/docs/m2/split-anti-seeds/report.py), has run at eight seeds and kept the anchor on all of them. That fits a loss rate as high as three in ten. Its close variants lost the anchor on one to three seeds of eight, in split-anti-seeds, [anchor-ramp](/docs/m2/anchor-ramp/report.py) and [soft-pool](/docs/m2/soft-pool/report.py), and the soft start of soft-pool (`step, soft τ`) kept it on all eight of its own. With the recipe known this loosely, an arm that loses one seed in eight can't be told apart from it.

Full runs at the seed counts that would narrow this are costly. But in those three experiments the runs that lost the anchor had parted from the kept ones within the first sixty or so epochs. So we first check, on the 96 stored runs whose outcome we know, whether an early call made from the alignment agrees with the full edit. Then we train 32 new seeds each of `falloff` and the soft start on the full-length schedule, stop them at epoch 64, and call each one.
"""

# %%

r"""
## Observations

- [The early call on the stored runs (E1)](#the-early-call-on-the-stored-runs-e1): TODO.
- [How often each arm loses the anchor (E2)](#how-often-each-arm-loses-the-anchor-e2): TODO.
- [When the example answers come onto e₁ (E3)](#when-the-example-answers-come-onto-e1-e3): TODO.

## Scope

This is an exploratory study, with no preregistration and no gate. It covers two arms, both with the step pull on slices 2 to 4 and the `falloff` anti weight on every slice: `falloff` itself, at a fixed pooling temperature $\tau_p$ = 0.1, and the soft start, where $\tau_p$ sharpens from 1 to 0.1 over the 40-epoch warm-up of the anchor weight. Everything else is the training of split-anti-seeds.

The new runs are at seeds 712 to 743, the same 32 seeds for both arms, so the two are paired by initialization, batches and label draws. Each trains on the full 400-epoch schedule (the learning rate and both term weights as in a full run) and stops at epoch 64. With the stored runs at seeds 704 to 711, each arm has 40 seeds.

Two seeds of each arm (704 and 705) are also trained again and stopped at epoch 64, as a check that stopping early leaves the trajectory up to the stop as it was in the stored full run.

The call is calibrated on the same stored runs it is then checked on, so E1 also holds out each experiment in turn. A screen run whose call falls between the stored lost and kept runs is trained again to the end and scored with the full edit, so no call rests on the window alone where the stored runs could not settle it. These runs have no control at their seed, so their removal is netted against the seed mean of the eight controls of ex-2.2.23; the controls differ little, and the call sits far from the half-removal line on every stored run.

We don't measure removal, spill or the task score on the screen runs: the question is the loss rate, and those are known for both arms from the earlier experiments.

## The measurements

- Kept or lost (the outcome, on stored and finished runs): a run lost the anchor when the full edit, along e₁ at every position and slice, removes less than half of `difference`, net of the control, as in split-anti-seeds.
- The window mean (the call): α (the cosine of the state with e₁) at the example answers of the `difference` probe contexts at slice 3, averaged over the records at epochs 44 to 64. α is recorded every four epochs, so the window holds six records. Slice 3 is pulled by both arms, and is the deepest slice every stored arm pulls. A run is called lost below 0.22, the middle of the gap between the stored lost runs (all below 0.15) and the kept ones (all above 0.28).
- The loss rate: losses over seeds, with the 95% upper bound from the binomial (Clopper–Pearson, one-sided). The two arms are compared seed by seed, on the seeds where one kept the anchor and the other lost it.
"""

# %%

r"""
## The early call on the stored runs (E1)

Does the window mean tell the runs that lost the anchor from those that kept it, on the 96 stored runs, and does the cut chosen on two experiments sort the third? A figure plots each run's window mean against its full-edit removal, marked by experiment.

TODO: the figure, the separation, and the held-out check. The stopped reproductions against their stored full runs.
"""

# %%

r"""
## How often each arm loses the anchor (E2)

How many of the 40 seeds of each arm lost the anchor, how high could the loss rate be, and do the two arms differ? A figure shows the α at the example answers through epoch 64 for every screen run, against the window and the cut.

TODO: the counts, the bounds, and the paired comparison.
"""

# %%

r"""
## When the example answers come onto e₁ (E3)

In soft-pool, the soft start held every position back through the first part of the warm-up, and the example answers came onto e₁ four to twelve epochs later than under `falloff`, closer together. Does that hold at 40 seeds, and how close does each arm come to the cut? The measure is the first recorded epoch at which α at the example answers goes over 0.4, as in soft-pool.

TODO.
"""

# %%

r"""
## Discussion

TODO.
"""
