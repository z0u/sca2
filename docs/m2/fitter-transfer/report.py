# ruff: noqa: B018
# title: Fitted edits at every slice, carried to a run they were not chosen on

# An exploratory re-analysis with no new runs and no new scoring: the tuning-set and test-set scores the
# post-hoc-compare experiment stored for every configuration of its grid, restricted to the fits that need no role
# labels and edit every slice, with the configuration of each run chosen on the other seven. The rule was written
# down and committed before any number in this report was computed.

r"""
# Fitted edits at every slice, carried to a run they were not chosen on

/// tip |
<!-- lede -->
TODO: what we tried and which way it came out.
///

In the post-hoc-compare experiment, a fitter configuration chosen on seven runs and carried to the eighth removed most of the op on seven runs, and on seed 712 it missed: the choice made on the other runs edited one slice, and that run holds the op a slice earlier. The anchored edit acts at every slice, so it cannot miss in that way. This report asks whether a fitter given the same shape of edit, a pooled fit at every slice with the configuration chosen on the other runs, carries over as well as the anchor does. It reads the scores that experiment stored, with no new runs and no new scoring.
"""

# %%

r"""
## Observations

- TODO (E1)
- TODO (E2)

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It follows the analysis "One configuration chosen on other runs" of post-hoc-compare, which was itself added after that run. We fixed the rule below before computing any of its numbers, so that this check reads as a test of one question rather than a fresh search over the grid.

The runs are those of post-hoc-compare: the eight controls of the anneal confirmation at seeds 712 to 719, where every fitter is fitted and scored, and the eight `τ_ℓ fixed` runs at the same seeds for the anchor, at full dose, as published.

The fitters are difference in means, the ridge probe, LEACE, and INLP, with the optimized direction beside them as a ceiling. Each is restricted to the configurations that need no role labels and edit like the anchor: fitted on every position of a context, pooled, editing every position at every slice (the embedding and each block output). Within that scope the grid still offers the target (binary or graded, where the fitter takes both), sequential refit (on or off), the replacement (zero or the mean), the rank of INLP, and the dose.

Choice on the other seven runs
:   For each run in turn, the selection rule of post-hoc-compare picks a point on the frontier of the seed-mean tuning scores of the other seven runs, where it crosses the spill criterion, interpolating between the two configurations that bracket it. The fitter is then fitted on the run left out with those configurations and scored on its test set, mixed with the same weights. This is the main measurement.

Choice with a margin
:   The same, with the point taken at a spill of 0.01 on the other seven runs, half the criterion, as a practitioner might leave room.

Choice on the run itself
:   The same rule on the tuning scores of the run itself, within the same scope. It is not a transfer; it says how much the every-slice restriction costs before anything is carried over.

Every score is net of the median drop along fifteen coordinate axes under the same operator form on the tuning set, as in the analysis this follows; the permuted-label baseline was scored only at the configurations post-hoc-compare chose, so it is not available here.

## Measurements

Removal
:   The drop in expected exact match on `difference` under the edit, as a share of the way from the clean score to the target null. Higher is better.

Spill
:   The largest drop on any other op under the edit. Lower is better, and the selectivity criterion is 0.02.

Over the criterion
:   A run whose spill on its test set is above 0.02.

Miss
:   A run whose removal is below 0.5, so the edit takes away less than half of the way to the target null. The anchor removes 0.83 or more on every run.
"""

# %%

r"""
## Carried to the run left out (E1)

Does a pooled fit at every slice, with its configuration chosen on the other seven runs, remove about as much as the anchor on every run, within the spill criterion?

TODO
"""

# %%

r"""
## What the restriction costs (E2)

How much of the removal does the every-slice shape give up against the fitter choosing on its own run, and against the single-slice choices of post-hoc-compare?

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

[post-hoc-compare](/docs/m2/post-hoc-compare/report.py)
:   Restricted to fits that need no role labels, LEACE removes about as much of the op as the anchor; with a configuration chosen on the other seven runs, the fitters miss seed 712, which holds the op a slice earlier than the rest.

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` at fresh seeds 712 to 719 keeps every anchor; it is the recipe of record, and its checkpoints and dose scoring are stored.

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   Trained the controls at seeds 712 to 719 that the fitters are fitted on, sharing initialization, batches, and corpus with the anchored runs.
"""
