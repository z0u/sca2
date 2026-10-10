# ruff: noqa: B018
# title: Setting the terms by slice

# Eight new runs: the `deep` pull with the anti-subspace weight set by slice, and the same with the pull pooled over
# slices. `experiment.py` beside this script trains, measures and dose-scores every run; the twins at the same seeds
# come from the late-pull experiment, and the controls from ex-2.2.23.

r"""
# Setting the terms by slice

/// tip |
<!-- lede -->
TODO
///

[The late-pull experiment](/docs/m2/late-pull/report.py) found that holding the anti-subspace weight at 0.2 for the whole of training kept the stand-in off e₁ and brought the spill within the criterion on every seed, at the cost of about a tenth of removal. A second review of the D2.2 runs traced part of that cost to the hold pressing on the pull at the blocks it pulls. So this report tries two ways to set the terms by slice: the anti weight held high on the embedding and block 1 only, and the pull pooled over slices as well as positions. It has eight new runs at four seeds, each paired with the late-pull runs and the controls at the same seeds.
"""

# %%

r"""
## Observations

- TODO (E1)
- TODO (E2)
- TODO (E3)

## Scope

This is an exploratory study, with no preregistration and no gate. Two new arms, each at model seeds 700 to 703 and 400 epochs; everything else is the late-pull recipe: the in-context grammar with `difference` anchored, anchor weight 0.1, τ 0.1, the anti-subspace term on every slice.

- `split-anti`: the pull on blocks 2 to 4, as late-pull's `deep`. The anti weight starts at 0.25 on every slice, as in the recipe, and anneals to a hold of 0.2 on the embedding and block 1 and of 0.03 (the recipe hold) on blocks 2 to 4.
- `pool-slices`: the same anti weights, with the pull on blocks 1 to 4 pooled by one mellowmax over every slice and position of a labeled context, so the pull may choose the slice as well as the position. The embedding stays out of the pool.

The comparisons are late-pull's `deep` at the recipe hold and at a hold of 0.2 on every slice, and ex-2.2.23's controls, all at the same seeds, initialization, batches and label draws. Four seeds per arm, so a difference smaller than the seed range of `deep` is not one we can trust.

## The measurements

TODO: removal, spill, the lean, α at fitting and non-fitting example answers, the dose response and landing, the edit by slice set, α along training by slice.
"""

# %%

r"""
## Removal and spill with the anti weight set by slice (E1)

Does a high anti weight on the embedding and block 1 alone keep the spill within the criterion at every dose, and does the removal come back to what `deep` removes at the recipe hold?

TODO
"""

# %%

r"""
## What the anti weight presses on (E2)

Does block 1 stay off e₁ on the other ops, and does the alignment at the fitting example answers of the pulled blocks come back up?

TODO
"""

# %%

r"""
## Where the pooled pull settles (E3)

When the pull may choose among blocks 1 to 4, which slice does it settle on, by seed, and does the edit at every slice still remove the op?

TODO
"""

# %%

r"""
## Discussion

TODO
"""
