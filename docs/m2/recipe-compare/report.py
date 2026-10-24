# ruff: noqa: B018
# title: Three recipes side by side

# An exploratory re-analysis of the stored runs of `falloff`, `step, soft τ` and `pool 1–3, soft τ` at seeds 704 to
# 711, with no new training. `experiment.py` beside this script adds the write bound on the two soft conditions and
# the axis scoring on the loss-rate runs that were finished at 400 epochs; everything else was stored by
# split-anti-seeds, soft-pool, edit-effects and reference-baseline.

r"""
# Three recipes side by side

/// tip |
<!-- lede -->
TODO
///

The recipe of record, `falloff`, was chosen in [split-anti-seeds](/docs/m2/split-anti-seeds/report.py) before the pooling temperature could start soft. Two recipes from the [soft-pool experiment](/docs/m2/soft-pool/report.py) have since looked close to it. `step, soft τ` is the same recipe with a soft start: it kept the anchor on every seed of the [loss-rate screen](/docs/m2/loss-rate/report.py), and against the median baseline of [reference-baseline](/docs/m2/reference-baseline/report.py) its spill is within the criterion on every run. `pool 1–3, soft τ` pools the pull over slices 1 to 3 as well as over positions, so nobody chooses which blocks the op goes on. That makes it the easier recipe to explain and to carry to another model, and a candidate even if it scores a little lower, as long as we know how much lower.

So the question is how the three compare when every measurement the D2.2 design asks for is taken the same way on each. This report trains nothing. It scores the stored runs of all three at seeds 704 to 711, with two small new passes: the write bound on the two soft conditions, and the median baseline on the seven loss-rate runs that were trained to the end.
"""

# %%

r"""
## Observations

- [Removal, spill and task (E1)](#removal-spill-and-task-e1): TODO.
- [What the edit does (E2)](#what-the-edit-does-e2): TODO.
- [Where the op sits (E3)](#where-the-op-sits-e3): TODO.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers three conditions at model seeds 704 to 711, 400 epochs each, all with the `falloff` anti weight:

- `falloff`, the recipe of record: the step pull on slices 2 to 4, pooled over positions at a fixed temperature $\tau_p$ of 0.1 (the runs of split-anti-seeds).
- `step, soft τ`: the same step, with $\tau_p$ starting at 1 and sharpening to 0.1 by the end of the pull warm-up at epoch 40 (soft-pool).
- `pool 1–3, soft τ`: one pull pooled over the slice and position pairs of slices 1 to 3, with both temperatures starting soft ($\tau_p$ 1 → 0.1, and the slice temperature $\tau_\ell$ 4 → 0.4) and nothing pulling slice 4 (soft-pool).

The runs at a seed share their initialization, batches and label draws, and so does ex-2.2.23's control at that seed, which trained without the anchor. So each comparison is paired by seed, and a difference between two conditions is set against the spread of the paired differences over the eight seeds, rather than against the seed range of either condition.

Eight seeds are too few to compare how often a recipe loses the anchor. None of the 24 runs here lost it. The loss-rate screen counted that at 40 seeds each for `falloff` (one loss) and `step, soft τ` (none), and bounded both rates at about a tenth; the pool has no count beyond these eight. This report quotes those counts and does not add to them.

The loss-rate screen trained seven of its runs to 400 epochs: the four `falloff` runs it could not call early (one of them the lost run) and three `step, soft τ` runs it finished as a check. They have no control at their seeds, so they are scored against the median baseline alone, and since they were picked as slow or doubtful seeds, they are a check on the slow end and not a sample of either condition.

## The measurements

Everything is measured on the 14,000 held-out contexts of ex-2.2.21, 2,000 per op. The edit takes a share of the e₁ component (the dose: a quarter, a half, three quarters or all of it) out of the state at every position and every slice.

- Removal: the drop on `difference` under the full edit, as a share of the way from the clean score to the target null. Higher is better.
- Spill: the largest drop on any other op at any dose. A run is within the criterion when it is at most 0.02.
- Baseline: both drops are net of the median drop over the same edit along every other coordinate axis of the same run, per op and dose (the rule of the D2.2 design). Removal with nothing subtracted is shown beside it.
- Task change: the clean score per op, less that of the control at the same seed. The task gate of the D2.2 design fails if the mean over ops falls below −0.02, or any one op below −0.05.
- Write bound, grading and landing, as in [edit-effects](/docs/m2/edit-effects/report.py). The write is how much e₁ the blocks after an edit put back on the states of the other ops, against the clean model (the bound). The grading is the share of the way to the target null the edit takes on `difference` contexts, by bin of the posterior on `difference`. The landing is the share of the distance to the target null that the full edit covers, over whole answer distributions. The grading is a context-by-context measure, and the median baseline exists only per op, so it is netted against the control at the same seed, as in edit-effects, with the share before netting beside it.
- Where the op sits: the alignment α with e₁ at the example answers by slice, on `difference` contexts at the end of training, and the removal by the full edit restricted to block 1, to blocks 2 to 4, and to block 4.
"""

# %%

r"""
## Removal, spill and task (E1)

Do the three recipes remove as much, spill as little, and cost the task as little, at each seed? The main measure is removal at the full dose, with the spill at the worst dose beside it and the task change after that. The paired difference from `falloff` at each seed says whether a gap is larger than the seeds make by themselves. The seven finished loss-rate runs come last, as a check on slow seeds.

TODO
"""

# %%

r"""
## What the edit does (E2)

The measurements of E1 are means over all the contexts of an op. Edit-effects showed on `falloff` that the edit also behaves well context by context: the blocks downstream put back less e₁ on the other ops than the clean model has, the drop on each `difference` context follows how much it needed the op, and the edited answers land about half of the way to the target null. Do the soft start and the pool keep those properties? The main measure is the write as a ratio of the bound, with the grading and the landing beside it.

TODO
"""

# %%

r"""
## Where the op sits (E3)

The step puts the pull on slices 2 to 4, where the pool lets the pull choose among slices 1 to 3. In [slice-terms](/docs/m2/slice-terms/report.py), a pull pooled over blocks 1 to 4 at a fixed temperature settled on block 4 on most seeds and on the query answer on one, where the op could not be removed. Soft-pool found the pool over slices 1 to 3 peaking at slice 3 on average. Does it settle on the same slice at every seed, and does the edit find it there? The main measure is α at the example answers by slice, per seed, with the removal by the edit restricted to a set of slices beside it.

TODO
"""

# %%

r"""
## Discussion

TODO
"""
