# ruff: noqa: B018
# title: A model trained without difference

# An exploratory study, at the plan stage: the question, the runs, the measurements, and the question each section
# asks, with the results left as TODO. Nothing has run yet; `experiment.py` follows once the plan is agreed.

r"""
# A model trained without `difference`

/// tip |
<!-- lede -->
On the recipe of record the full edit takes the model about half of the way to the target null on `difference` contexts. We train the same model on the same corpus with the `difference` contexts left out, to see how close a model that never learned the op comes to the target null, and whether it misses it in the same way as the edited model.
///

The [backlog item](/todo/science/reference-model-without-the-anchored-op.md) asks how much of the distance the edit leaves could be closed at all. The edit-effects experiment found that the edited model sends the mass to the right ops overall but weighs them differently from the target null context by context, and its Discussion offers two readings: the edit blurs evidence the model uses to weigh the other ops, or no model of this kind learns those weights on contexts that look like `difference`. A model trained without `difference` has had nothing removed, so how close it comes is a reference for the edit. It is also the filtered-corpus comparison of the D2.2 design. This study trains eight new runs, one per seed at seeds 704 to 711, and reads the stored `falloff` runs and controls at the same seeds.
"""

# %%

r"""
## Observations

- [How close it comes (E1)](#how-close-it-comes-e1): TODO.
- [Whether it misses in the same way (E2)](#whether-it-misses-in-the-same-way-e2): TODO.
- [What leaving the op out does to the other ops (E3)](#what-leaving-the-op-out-does-to-the-other-ops-e3): TODO.

## Scope

This is an exploratory study, with no preregistration and no gate.

The new condition, `filtered`, is the 400-epoch control of ex-2.2.23 (no anchor, the runs the split-anti-seeds experiment paired its arms with) trained on the corpus of record with every `difference` context taken out. Nothing else in the corpus changes: the contexts of the other six ops are the same lines, and they keep their replacement op noise, so about one example in twenty of theirs still shows the answer `difference` would give. That keeps the target null exact for this corpus. A context of another op draws its noise from the other six ops, `difference` among them, which is the noise model the target null keeps when it sets `difference` to zero. So the Bayes predictor of the filtered corpus is the target null, on every context (the cube noise too, which the posterior of record models as a term of its own), and `filtered` is scored against the predictor that is ideal for what it was trained on.
<!-- REVIEW: added the cube-noise aside (assumptions review). The corpus of record carries cube noise at 0.02; `sca.data.incontext.posterior_over_ops` leaves it out, but the posterior the dose scoring uses (`docs/m2/ex-2.2.16/posterior.py`) includes it, so the claim holds for the stored arrays. Verify: the likelihood in that module's `posterior`. -->

Contexts like the held-out `difference` contexts are rare under that corpus: a context of another op shows the answer of `difference` only on an example that is noise of that one kind, so a context whose examples mostly point at `difference` seldom occurs. So on those contexts `filtered` answers from what it learned on other contexts, as the edited model has to.
<!-- REVIEW: the held-out `difference` contexts carry the same replacement noise as the corpus, so most show `difference` on two or three examples rather than all three; the earlier sentence described only the all-three case (assumptions review). Verify: the posterior bins of edit-effects (E2 there), where 695 of 2,000 held-out `difference` contexts have a posterior of 0.95 or more. -->

The corpus is the stored one of record with the `difference` lines dropped and the rest packed as before; with no anchor, the labels are unused. Each run is measured at its final checkpoint, on the schedule of the control, whose warmup and peak rate are set in epochs. Each `filtered` run trains at the seed of a `falloff` run and of a 400-epoch control, so all three share the initialization. The batches differ, since the corpus does. The stored arrays of the split-anti-seeds dose scoring hold the answer distributions of the `falloff` runs, clean and under the full edit, and of the controls, on the same held-out contexts. `filtered` is scored with the same pass, so every distance below is taken on the same contexts and in the same way. With eight runs per condition, a difference between conditions is trusted when it is clear of the seed range of both.

/// admonition | Open decision
Taking out `difference` leaves six-sevenths of the tokens, so at 400 epochs `filtered` trains for about six-sevenths of the steps of the control, while every remaining context is seen as often. The alternative is to match the steps (about 467 epochs), which shows each remaining context more often than the control does. Matching the passes keeps the contexts of the other ops paired with the control one for one, and keeps the shape of the schedule, since an epoch is sized by the tokens of the corpus and the warmup is set in epochs; matching the steps makes the warmup a seventh shorter in steps than the control had. E3 checks the task score of the other ops against the control either way.
///

## The measurements

Each measurement is taken on the answer distribution at the query `=`, over the colors of the palette, and averaged over held-out contexts.

Distance from the target null
:   Total variation between the answer distribution of the model and the target null, with any mass the model puts off the palette counted as distance; 0 is a perfect match and 1 shares nothing. Lower is closer. On `difference` contexts for `falloff` it is taken under the full edit, as in edit-effects; for `filtered` and the control it is taken on the clean pass.

Bayes gap
:   The same distance, between the clean control and the Bayes predictor: how far an un-anchored model of this size falls short of the ideal on contexts like the ones it trained on. It sets the scale for the other distances. Taken on `difference` contexts, where edit-effects took its yardstick, and on the contexts of the other ops.
<!-- REVIEW: the gap was on the contexts of the other ops only; edit-effects quotes its 2.4× against the gap on `difference` contexts (0.16), so E1 needs that one to compare with, and the stored arrays hold both (assumptions review). Verify: `clean_to_bayes` and `other_clean_to_bayes` in the edit-effects report. -->

Share of the way
:   One minus the distance of a model from the target null over the distance of the control at the same seed, each a mean over `difference` contexts: how much of the way from the control to the target null that model has come. For the edited `falloff` runs, edit-effects took this against their own clean pass, and it is reported both ways.

Distance between the models
:   Total variation between `filtered` and the edited `falloff` run at the same seed, context by context, on `difference` contexts.

Task score
:   Expected exact match, per op, on the clean pass.
"""

# %%

r"""
## How close it comes (E1)

How far is `filtered` from the target null on `difference` contexts, beside the edited `falloff` runs and the Bayes gap? If it comes about as close as the edited model, then landing about half of the way is near what a model of this kind does on these contexts, and the shortfall need not come from the edit (E2 asks whether the two miss in the same way). If it comes much closer, the edit leaves a distance that a model can close, and that distance belongs to the edit. If it is about as far as the Bayes gap, the target null is reached as well as the ideal is on ordinary contexts.

The figure will show the three distances per seed, with the share of the way beside them in a table, and the same split by the posterior on `difference` as in edit-effects (E2 there). The contexts where the posterior on `difference` is lower are the ones whose examples partly fit other ops, which are closer to what `filtered` saw in training.

TODO.
"""

# %%

r"""
## Whether it misses in the same way (E2)

Wherever `filtered` lands, do the edited model and `filtered` miss the target null in the same way? If the distance between the two models is small beside the distance of each from the target null, they weigh the other ops alike on a context, and the part of the shortfall they share is a property of these contexts rather than of the edit. If the two models are as far from each other as from the target null, they miss in different ways.

The breakdown from edit-effects (E3 there) will be repeated for `filtered`: the mass on answers only `difference` gives, on answers another op could give, and on colors no op gives; the distance from the nearest mixture of the answers of the ops; and whether a target null with flatter or sharper weights on the other ops fits it better.

TODO.
"""

# %%

r"""
## What leaving the op out does to the other ops (E3)

On contexts of the other ops, does `filtered` score as well as the control, and does it gain where the target null says a model without `difference` should? The target null predicts a gain on contexts of other ops whose examples point partly at `difference`, since a model that cannot answer `difference` puts that mass on the right op. edit-effects found the edited model gains most of the way the target null predicts there, and this is the same comparison for a model that never learned the op, with the edited model beside it.

It also checks the training: if `filtered` scored well below the control on the other ops overall, its distance in E1 would partly be a weaker model rather than a missing op.

TODO.
"""

# %%

r"""
## Discussion

TODO.
"""

# %%

r"""
## References

[edit-effects](/docs/m2/edit-effects/report.py)
:   On the recipe of record the full edit lands about half of the way to the target null; the distance left is about 2.4 times the distance from the clean model to the Bayes predictor, nearly all of it in how the other ops are weighed context by context.

[split-anti-seeds](/docs/m2/split-anti-seeds/report.py)
:   The `falloff` condition, with the anti weight falling off with depth, kept the anchor on all eight seeds and stayed within the selectivity criterion; it became the recipe of record.

"""
