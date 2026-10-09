# ruff: noqa: B018
# title: Pulling the later blocks only

# Twelve new runs: two arms that pull blocks 2 to 4 only, with the anti-subspace term on every slice, and the recipe
# trained again at the same four seeds for its trajectory. `experiment.py` beside this script trains and measures every
# run; the twins and controls at the same seeds come from ex-2.2.23, as embedding-lean measured them.

r"""
# Pulling the later blocks only

/// tip |
<!-- lede -->
The embedding-lean report traced the spill of the edit to the pull on the first two slices, which loads lightness onto e₁ in the color embedding table. Here we pull only blocks 2 to 4 and keep the anti-subspace term on every slice, with and without a hard constraint that holds the whole embedding table off e₁. TODO: which way it came out.
///

[Embedding-lean](/docs/m2/embedding-lean/report.py) found that every anchored run of ex-2.2.23 leans lightness onto e₁ in its color embedding table, since at the embedding nothing contextual exists for the pull to use, and darker colors are the nearest token-level stand-in for `difference` answers. Editing only blocks 2 to 4 of those runs removed most of `difference` with almost no spill, while editing the embedding alone spilled as much as editing every slice. Pulling every slice but the embedding has been tried before (ex-2.2.21 and the τ × λ_a sweep), but with the anti-subspace term left off the embedding too, and the table leaned all the same.

So this experiment trains twelve runs at 400 epochs and four seeds:

- `late`: the pull on blocks 2 to 4 only, and the anti-subspace term on every slice, so the first two slices are asked to stay off e₁ rather than left alone.
- `late-clean`: the same, and after every step the e₁ component of every embedding is set to zero, so the table cannot lean at all.
- `whole`: the recipe of record, every slice pulled, trained again to record the alignment at every slice and position along training.

Each run shares its seed, and so its initialization, batches and label draws, with one anchored run and one control of ex-2.2.23. The anchor weight, the pool temperature τ and both schedules are the recipe values.
"""

# %%

r"""
## Observations

- [The color table (E1)](#the-color-table-e1): TODO.
- [Removal and spill (E2)](#removal-and-spill-e2): TODO.
- [When the alignment appears (E3)](#when-the-alignment-appears-e3): TODO.

## Scope

This is an exploratory study, with no preregistration and no gate. Four seeds per arm can show a large change in the lean or the spill, against the seed range of the twelve every-slice runs in embedding-lean, but not a small one. The `whole` arm should reproduce its ex-2.2.23 twins; where it does not, the difference between them is the scale of run-to-run noise on identical inputs.

The anti-subspace weight keeps its recipe schedule, a multiple of the anchor weight. Giving it a weight of its own, and sweeping the pull per slice, wait on what this run shows.

## The measurements

The measurements are those of embedding-lean, taken on every run:

- The lean of the color table: the correlation, over the 24 colors, between the lightness of a color and the e₁ component of its embedding. A run with no lean scores near zero; the anchored runs of ex-2.2.23 score near −0.7, and their controls anywhere from −0.6 to +0.6 by seed.
- The lean in the states: the same correlation at each slice, over the color positions of contexts of other ops.
- Removal and spill of the edit, which projects e₁ out of the state at every position on a set of slices. Removal is the share of the way from the clean score of `difference` to its target null that the edit gets, and spill is the largest drop on another op. Both are net of the control at the same seed. The edit is scored on every slice and on blocks 2 to 4.
- The task score of each op, as expected exact match on held-out contexts, so that a cost of the new pull to the task shows.
- Along training, every four epochs: the mean alignment at every slice and position of the probe contexts, for `difference` and for the other ops, and the e₁ component of every color embedding.
"""

# %%

r"""
## The color table (E1)

Does the color table still lean when the pull leaves the first two slices and the anti-subspace term stays on them? If the lean was the pull meeting the embedding with a stand-in, it should fade in `late`. If it persists, something else loads lightness onto e₁, such as the pull at block 2 reaching back through the residual stream. In `late-clean` the table is held at zero by construction, so the question there is whether the lean moves into the states at block 1.

TODO
"""

# %%

r"""
## Removal and spill (E2)

Does the edit remove `difference` without spill, on every slice and on blocks 2 to 4? Editing blocks 2 to 4 of the recipe runs already spilled little, so the comparison that matters is the edit on every slice, which is the goal. A `late` run that removes as much with as little spill on every slice as on blocks 2 to 4 has no lean left for the edit to find.

TODO
"""

# %%

r"""
## When the alignment appears (E3)

Along training, at which slices and roles does the alignment with e₁ grow, and when, under each pull? The `whole` arm shows the recipe; the two late arms show whether the alignment builds up at block 2 from the start or arrives once block 1 has something contextual to offer.

TODO
"""

# %%

r"""
## Discussion

TODO
"""
