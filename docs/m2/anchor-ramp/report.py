# ruff: noqa: B018
# title: An anchor weight that ramps with depth

# An exploratory report: two arms that ramp the pull up with depth, beside the `falloff` recipe of record, at the
# split-anti-seeds seeds. `experiment.py` beside this script trains, measures and dose-scores the new runs; the
# `falloff` runs and the controls come from split-anti-seeds and ex-2.2.23.

r"""
# An anchor weight that ramps with depth

/// tip |
<!-- lede -->
TODO: what we tried (a pull that rises in a straight line with depth, in place of the step onto slices 2 to 4), and which way it came out.
///

The recipe of record ([`falloff`](/docs/m2/split-anti-seeds/report.py)) sets the anti weight $\lambda_{\bar s}$ by slice, falling in a straight line from the embedding to slice 4, so no slice boundary has to be chosen for it. The pull still has one: it acts on slices 2 to 4 at the full anchor weight $\lambda_a$, and not at all on the embedding or slice 1. This report tries the mirror of the falloff for the pull, a weight that rises in a straight line with depth. Two new arms, eight runs each, at the seeds of the split-anti-seeds experiment, so the `falloff` runs there are the baseline with no new training.
"""

# %%

r"""
## Observations

- [Removal and spill (E1)](#removal-and-spill-e1): TODO.
- [Where the anchor sits by depth (E2)](#where-the-anchor-sits-by-depth-e2): TODO.
- [Which slices the edit needs (E3)](#which-slices-the-edit-needs-e3): TODO.

## Scope

This is an exploratory study, with no preregistration and no gate. The two arms differ from `falloff` only in the pull weight by slice. Both ramps reach the weight of the step at slice 4, and differ in where they start from zero:

- `ramp from emb`: zero at the embedding, so slice 1 is pulled at a quarter of the slice-4 weight, slice 2 at a half and slice 3 at three quarters.
- `ramp from 1`: zero at slice 1, so the pull starts on slice 2 as the step does, at a third of the slice-4 weight, and slice 3 at two thirds.

TODO: the table of pull and anti weights by slice for the three arms.

Matching the step at slice 4 means each ramp pulls less in total than the step: about five sixths as much for `ramp from emb`, and two thirds for `ramp from 1`. The other choice, matching the total, would pull slice 4 harder than the recipe does. The τ×λ sweep found removal changed little with $\lambda_a$, so we expect the shape to matter more than this difference in total, though that sweep pulled every slice after the embedding with one weight.

The anti weight is the falloff in every arm, so `ramp from emb` pulls slice 1 against a hold of about 0.16. [The late-pull experiment](/docs/m2/late-pull/report.py) moved the pull off slice 1 to keep the lightness lean off the embedding table, so a pull there may bring some of it back; E2 looks for that.

Every run trains at model seeds 704 to 711 for 400 epochs, and is paired by seed with its `falloff` twin and with ex-2.2.23's control, which share the initialization, batches and label draws. With eight seeds, a difference in removal of a few hundredths is within what seeds alone gave between the split and the falloff, so we take a difference as real only where it shows on most seeds.

## The measurements

The measurements and dose scoring are those of the split-anti-seeds experiment.

- Removal: the net drop in the score on `difference` under the edit, as a share of the way from the clean score to the target null. The edit projects e₁ out of the state at every position, on a set of slices, at a dose of a quarter, a half, three quarters, or all of the e₁ component. Higher is better.
- Spill: the largest net drop on another op under the edit. A run is within the criterion when its spill is at most 0.02 at every dose.
- Both are net of the control at the same seed.
- Task change: the mean score over ops, less that of the control. Zero means no damage.
- The alignment α with e₁ at the example answers, at every slice, at the end of training, and the lightness slope of the color embedding table (how much further along e₁ black sits than white).
"""

# %%

r"""
## Removal and spill (E1)

Does a pull that ramps with depth remove as much of the op as the step, and keep the edit as selective? The main measure is the removal at the full dose of the edit on every slice, paired by seed with `falloff`, with the spill at any dose beside it.

TODO: the dose figure and the summary table.

TODO: what we saw, and what we make of it.
"""

# %%

r"""
## Where the anchor sits by depth (E2)

The step pulls slices 2 to 4 equally; a ramp pulls the later slices harder than the earlier ones. Does the alignment by slice follow the weight, and does the pull on slice 1 under `ramp from emb` bring back the drift onto e₁ in the contexts of the other ops, or the lean in the embedding table?

TODO: alignment by slice on `difference` and on the other ops, with the pull and anti weights by slice underneath; the lightness slope by arm.

TODO: what we saw, and what we make of it.
"""

# %%

r"""
## Which slices the edit needs (E3)

In the split-anti-seeds experiment, the edit on slices 2 to 4 alone fell short on one seed (710) in every arm, and the edit on slice 1 alone removed different amounts in different arms. A ramp moves some of the pull onto slices 1 and 2. Does it change how much each partial edit removes, and does seed 710 still miss?

TODO: removal by the edit on slice 1 alone and on slices 2 to 4 alone, by arm and seed.

TODO: what we saw, and what we make of it.
"""

# %%

r"""
## Discussion

TODO.
"""
