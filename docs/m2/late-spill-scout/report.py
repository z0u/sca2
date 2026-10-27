# ruff: noqa: B018
# title: Late spill scout

# An exploratory report: three late-schedule changes to the soft pool scout's `τ_ℓ 0.1, τ_ℓ fixed`, each carried on
# from that condition's stored epoch-64 states at seeds 704 to 711. `experiment.py` beside this script trains and
# scores the runs.

r"""
# Late spill scout

/// tip |
<!-- lede -->
TODO
///

The [soft pool scout](/docs/m2/soft-pool-scout/report.py) left a candidate recipe, the pool over slices 1 to 3 with the slice temperature τ_ℓ held at 0.1, that has no early spill. But on one of its eight seeds the spill rises after epoch 128 and ends at the criterion, and with more seeds some runs would likely go over it. The rise comes while the anti weight λ_s̄ anneals from its peak to the per-slice holds, the only loss term still moving at that point (the learning rate's cosine decay runs alongside it, which is a reason to move the anti anneal on its own). Spill-timing found the late kind of spill lands on ops the run is still learning, so one reading is that the anti term keeps e₁ closed while an op is learned, and the anneal opens it.

So this report asks whether a late change to the schedules keeps the late spill down, and what it costs. It changes one schedule at a time, only after epoch 64, so each run carries on from the stored epoch-64 state of the same seed: four sets of eight runs, one of them the condition unchanged, each trained for the last 336 of 400 epochs. Since only one seed rises under this condition, the same four also carry on the soft pool scout's other finished condition, `τ_ℓ 0.1, by 40`, at the two seeds that rise under it.
"""

# %%

r"""
## Observations

- TODO (E1)
- TODO (E2)
- TODO (E3)

## Scope and method

This is an exploratory study, with no preregistration and no gate. It covers one condition of the soft pool scout, `τ_ℓ 0.1, τ_ℓ fixed`, at its eight seeds (704 to 711), and four ways of carrying it on from epoch 64. The same four carry on `τ_ℓ 0.1, by 40` at 705 and 707, the seeds that rise late under it:

`base`
:   The condition unchanged. These runs repeat the soft pool scout's finished runs (the resume reproduces a straight run bit for bit), and keep a checkpoint every 32 epochs as the arms do.

`anti late`
:   The anti anneal pauses at epoch 64, holding the weight it has reached (close to its peak: the minimum-jerk anneal has gone about 4% of the way to the holds by then), and finishes from epoch 200 to 360. So the anti weight stays high while the late ops are learned (every seed of the base had learned `value-hsv` by epoch 128, and the quiet recipes by 180, in spill-timing) and reaches the same per-slice holds by the same epoch. Under `base` the anneal is a quarter of the way to the holds at epoch 128 and two-fifths at 160, where spill-timing saw the late kind begin, so the two arms differ modestly there and widely from epoch 200 on.

`anti held`
:   The pause runs to epoch 360, so the anti weight never anneals to the holds and only comes down with the anchor in the last 40 epochs. This is the largest change to the anti term that keeps its first 64 epochs, and shows what the anti term can do about the late spill at most, and what it costs.

`τ_ℓ late`
:   The slice temperature sharpens again, from 0.1 to 0.025 over epochs 64 to 360, alongside the anti anneal. Stage 1 of the soft pool scout found a sharper τ_ℓ lowers the alignment at slice 1 and the early spill; this asks whether sharpening further does the same for the late kind. At 0.025 the pool over three slices is close to a max, so the pull lands almost wholly on the slice where the evidence peaks (slice 3 on every seed), and the arm is also a narrowing of which slices the pull reaches.

Every arm uses only schedules and temperatures, nothing keyed to a token role, so each would carry to real text as the recipe does.

Since every run of a seed shares its first 64 epochs, a difference between an arm and `base` at the same seed comes from the change after epoch 64 alone. The late rise is small, though: on the seed it reaches, it ends at the criterion, while quiet runs scatter by about 0.006 (one standard deviation, over the stored runs of quiet conditions in spill-timing) around a value a little below zero, and a run settles after epoch 200 so that successive checkpoints differ by about half that. So a change at one seed means little, and the comparison that counts is the paired one on the seed that rises (707, on `mix`) and the mean over the eight. One rising seed cannot tell an arm that keeps the late spill down from one that only re-rolls it, so E1 also asks whether the rise at 707 stops when the anti weight is held, and whether the quiet seeds move together.
<!-- REVIEW: was "the seeds that rise (705 and 707)". Under `τ_ℓ 0.1, τ_ℓ fixed` only 707 rises (0.020, on `mix`); 705 rises under `τ_ℓ 0.1, by 40`, the other finished condition, and sits at −0.005 here. Verify: the by-seed table of E4 and the along-training figure of E5 in the soft pool scout. -->
<!-- REVIEW: the noise statement now quotes spill-timing (Scope and method: 0.006 over 27 quiet runs; E2: a standard deviation of about 0.003 after epoch 300, and quiet seeds at about −0.01) in place of "about a third of the criterion either side of zero". -->

The spill along training uses the same eight reference axes for every run, so a paired comparison at a seed shares them; on 704, 705 and 708 the draw includes the separator coordinate (spill-timing), which the median absorbs.

The measurements are those of the soft pool scout's stage 2:

Spill
:   The largest drop on another op under the full edit at epoch 400, against the median of the same edit along every other coordinate axis of the run (the D2.2 baseline). The criterion is 0.02.

Spill along training
:   The same edit on each checkpoint kept from epoch 64 on (every 32 epochs, and the end), against the median of eight reference axes, as spill-timing scored it.

Removal
:   How much of the anchored op the full edit removes, at epoch 400.

Alignment at the answers
:   The mean alignment α with e₁ at the example answers of the anchored op, at slice 3, along training. A stronger anti term should lower it before removal shows a cost, so it checks that `anti held` acts where the plan expects.

Task change
:   What the anchoring costs the model on the other ops, against the control at the seed.
"""

# %%

r"""
## Spill along training (E1)

Does any arm keep the late rise down on the seeds where it appears, and does the rise follow the anti weight in time?

TODO: spill at every kept checkpoint, one panel per arm, with the anti weight (per slice) and τ_ℓ in panels underneath on a shared epoch axis.
"""

# %%

r"""
## Spill and removal at the end (E2)

At epoch 400, how does each arm compare with `base` at the same seed, on spill and on removal?

TODO: paired differences per seed, and a table of the means.
"""

# %%

r"""
## What the changes cost (E3)

Does an arm that keeps the late spill down pay for it, in removal, in task cost, or by losing the anchor (a run that removes less than 0.5, as the soft pool scout counts it)? `anti held` trains to the end under an anti weight at slice 3 about eight times that of `base`, so it is the arm most likely to.

TODO
"""

# %%

r"""
## Discussion

TODO
"""
