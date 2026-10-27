# ruff: noqa: B018
# title: Anneal confirmation

# An exploratory report: the soft pool scout's candidate recipe with τ_ℓ annealing from 0.1 to 0.025 over the whole
# schedule, at fresh seeds 712 to 719 and in a loss-rate count at forty seeds. `experiment.py` beside this script
# trains and scores the runs.

r"""
# Anneal confirmation

/// tip |
<!-- lede -->
TODO: whether the pool over slices 1 to 3, with its slice temperature annealing slowly from 0.1 to 0.025, holds up at fresh seeds and keeps the anchor as reliably as the step recipes.
///

The [soft pool scout](/docs/m2/soft-pool-scout/report.py) found a candidate for the recipe of record: the pull pooled over slices 1 to 3, with the slice temperature $\tau_\ell$ held at 0.1. It keeps the anchor and stays within the spill criterion at seeds 704 to 711, but on one or two seeds the spill rises late and ends near the criterion. The [late spill scout](/docs/m2/late-spill-scout/report.py) then found that sharpening $\tau_\ell$ again late, to 0.025, lowered that late rise at no measured cost. Here the two moves become one: $\tau_\ell$ goes from 0.1 at the start to 0.025 at epoch 400, smoothly in log, so it is still close to 0.1 while the anchor settles and halfway (0.05) at epoch 200. We call it `τ_ℓ anneal`.

This is the confirmation run for adopting `τ_ℓ anneal` as the recipe of record. It trains the recipe at eight fresh seeds (712 to 719) to the end, with a control at each, and at forty seeds (704 to 743) to epoch 64 for a count of lost anchors.
"""

# %%

r"""
## Observations

- [The anchor at fresh seeds (E1)](#the-anchor-at-fresh-seeds-e1): TODO.
- [Spill at the end and along training (E2)](#spill-at-the-end-and-along-training-e2): TODO.
- [Lost anchors at forty seeds (E3)](#lost-anchors-at-forty-seeds-e3): TODO.

## Scope and method

This is an exploratory study, with no preregistration. The recipe would be adopted if four things hold, fixed before the runs:

1. Every run at seeds 712 to 719 keeps the anchor (removal at least 0.5).
2. No run at those seeds ends over the spill criterion of 0.02.
3. Removal is no lower than that of `falloff` at seeds 704 to 711, and the task passes the gate of the D2.2 design against the control at the seed (a mean change over ops of at least −0.02, and at least −0.05 on any one op).
4. At most one run of the forty loses the anchor: about as few as `step, soft τ` (none of forty) and `falloff` (one) lost in the loss-rate screen, which forty seeds can't tell apart.

`τ_ℓ anneal` is the soft pool scout's `τ_ℓ 0.1, τ_ℓ fixed` with one change: $\tau_\ell$ moves from 0.1 to 0.025 over epochs 0 to 400, minimum-jerk in log $\tau_\ell$. The position temperature $\tau_p$ still sharpens from 1 to 0.1 by epoch 40, the anchor weight still warms up over the first 40 epochs, and the anti weight still anneals to `falloff`'s per-slice holds by epoch 360.

The forty runs of the count share their seeds with the loss-rate screen's calls of `falloff` and `step, soft τ`, so each call pairs with theirs by initialization, batches and label draws. Each stops at epoch 64 and is called from α at the example answers over epochs 44 to 64, as that screen called them. The runs at 712 to 719 carry on from those stops to epoch 400, keeping a checkpoint every 32 epochs, and so does any run the call is unsure of. Each control trains in full with no pull and no anti term, at the same seed. All runs train and are measured on L4 GPUs.

TODO: how far a difference between runs can be trusted, from the seed range.

## The measurements

Every edit below is the e₁ edit at every position and slice, net of the median drop along the other 63 coordinate axes of the same run (the rule of the D2.2 design).

Removal
:   The drop on `difference` under the full edit, as a share of the way from the clean score to the target null. Higher is better.

Spill
:   The largest drop on any other op, at its largest over doses. Lower is better, and a run goes over the criterion above 0.02.

Task change
:   The clean score per op, less that of the control at the seed.

Call
:   Mean α at the example answers at slice 3, over epochs 44 to 64: below 0.15 a run is lost, from 0.28 it is kept, and a run in between is finished and measured.
"""

# %%

r"""
## The anchor at fresh seeds (E1)

Does every run at the fresh seeds keep the anchor, and is removal and the task as good as under `falloff`?

TODO
"""

# %%

r"""
## Spill at the end and along training (E2)

Does any run end over the spill criterion, and does the late rise of the soft pool scout show up along training? Where an op spills, does it spill while that op is being learned?

TODO
"""

# %%

r"""
## Lost anchors at forty seeds (E3)

How many of the forty runs lose the anchor by the call at epoch 64, and how does that compare with `falloff` and `step, soft τ` at the same seeds?

TODO
"""

# %%

r"""
## Discussion

TODO
"""

# %%

r"""
## References

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The pool over slices 1 to 3 with $\tau_\ell$ held at 0.1 keeps the anchor on all eight seeds with no early spill, and one seed rises late to near the criterion.

[late spill scout](/docs/m2/late-spill-scout/report.py)
:   The late spill follows the anti anneal, and sharpening $\tau_\ell$ to 0.025 over epochs 64 to 360 lowers it with no measured cost to removal or the task.

[loss-rate screen](/docs/m2/loss-rate/report.py)
:   A call from α at the example answers over epochs 44 to 64 sorts lost anchors from kept ones; at forty seeds `falloff` lost one and `step, soft τ` none.
"""
