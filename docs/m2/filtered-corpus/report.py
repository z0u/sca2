# ruff: noqa: B018
# title: A model trained without difference

# An exploratory study, at the plan stage: the question, the runs, the measurements, and the question each section
# asks, with the results left as TODO. Nothing has run yet; `experiment.py` follows once the plan is agreed.

r"""
# A model trained without `difference`

/// tip |
<!-- lede -->
On the earlier recipe the full edit took the model about half of the way to the target null on `difference` contexts. We train the same model on the same corpus with no loss on the `difference` contexts, to see how close a model that never learned the op comes to the target null, and whether it misses it in the same way as the edited model of the recipe of record.
///

The edit-effects experiment found that the edited model sends the mass to the right ops overall, but shares it among them differently from the target null context by context. Its Discussion offers two readings: the edit blurs evidence the model uses to share the mass among the other ops, or no model with this architecture, grammar, and set of other ops learns that sharing on contexts that look like `difference`. A model trained without `difference` has had nothing removed, so how close it comes is a reference for the edit. It is also the filtered-corpus comparison of the D2.2 design, and its figures are meant to go into the D2.2 post. This study trains eight new runs, one per seed at seeds 712 to 719, and scores the stored `τ_ℓ fixed` runs and controls at the same seeds.
"""

# %%

r"""
## Observations

- [Along training (E1)](#along-training-e1): TODO.
- [How close it comes (E2)](#how-close-it-comes-e2): TODO.
- [Whether it misses in the same way (E3)](#whether-it-misses-in-the-same-way-e3): TODO.
- [What leaving the op out does to the other ops (E4)](#what-leaving-the-op-out-does-to-the-other-ops-e4): TODO.

## Scope and method

This is an exploratory study, with no preregistration and no gate.

The new condition, `sans-diff`, is the 400-epoch control (no anchor; the controls the anneal-confirm experiment trained at these seeds) with the language-model loss switched off on every `difference` context. The corpus, the batches, and the schedule are those of the control at the same seed: the windows still hold `difference` contexts, and the model reads them as input, but no target inside one is charged, so nothing in training asks the model to answer one. The loss is the mean over the targets that are charged, so a step weighs the other six ops as the control does. Each `sans-diff` run starts from the initialization of the control and of the `τ_ℓ fixed` run at its seed and sees the same windows in the same order, so the difference between `sans-diff` and the control at a seed comes from the masked loss alone.
<!-- REVIEW: the condition was a corpus with the `difference` lines dropped (`filtered`), which changed the batches and left a choice between matching passes (400 epochs) and matching steps (about 467). The paper review asked whether the loss could be masked instead, keeping everything else the same; the training code already has a target mask (`loss_mask` in `sca.anchoring.sample_anchored_batches`, used for the `FALSE` verification answers), so the open decision went away. Verify: the loss in `sca.training.loop` is the mean over non-padding targets, and a masked target is set to padding. -->

The contexts of the other six ops keep their replacement op noise, so about one example in twenty of theirs still shows the answer `difference` would give. That keeps the target null exact for this model. A context of another op draws its noise from the other six ops, `difference` among them, which is the noise model the target null keeps when it sets `difference` to zero. So the Bayes predictor of the contexts `sans-diff` is trained on is the target null, on every context (the cube noise too, which the posterior of record models as a term of its own), and `sans-diff` is scored against the predictor that is ideal for what it was trained on.
<!-- REVIEW: added the cube-noise aside (assumptions review). The corpus of record carries cube noise at 0.02; `sca.data.incontext.posterior_over_ops` leaves it out, but the posterior the dose scoring uses (`docs/m2/ex-2.2.16/posterior.py`) includes it, so the claim holds for the stored arrays. Verify: the likelihood in that module's `posterior`. -->

Contexts like the held-out `difference` contexts are rare among the ones `sans-diff` is trained on: a context of another op shows the answer of `difference` only on an example that is noise of that one kind, so a context whose examples mostly point at `difference` seldom occurs. So on those contexts `sans-diff` answers from what it learned on other contexts, as the edited model has to.
<!-- REVIEW: the held-out `difference` contexts carry the same replacement noise as the corpus, so most show `difference` on two or three examples rather than all three (assumptions review). Verify: the posterior bins of edit-effects (E2 there), where 695 of 2,000 held-out `difference` contexts have a posterior of 0.95 or more. -->

The reference runs are those of the fixed-confirm experiment: `τ_ℓ fixed`, the recipe of record, at seeds 712 to 719, and the controls at the same seeds. Their dose scoring is stored, with the answer distributions on the same held-out contexts, clean and under the full edit, beside the posterior over ops. `sans-diff` is scored with the same pass, so every distance below is taken on the same contexts and in the same way. Each run is measured at its final checkpoint. With eight runs per condition, a difference between conditions is trusted when it is clear of the seed range of both. Seed 718 learns the HSV ops late under `τ_ℓ fixed` and misses the task criterion of fixed-confirm on two of them; E4 shows whether the control and `sans-diff` do the same at that seed.
<!-- REVIEW: moved from the `falloff` runs and ex-2.2.23 controls at 704–711 to the recipe of record, `τ_ℓ fixed` at 712–719, with the anneal-confirm controls (paper review). Verify: `DOSE_ARRAYS_REF` in `docs/m2/fixed-confirm/experiment.py` and `docs/m2/anneal-confirm/experiment.py`; the control arrays are under the latter. -->

A check arm, `dropped`, trains two seeds (712 and 713) on the corpus with the `difference` contexts taken out, for 400 epochs, which is about six-sevenths of the steps. Under the mask, `sans-diff` still reads `difference` contexts in its window; under `dropped` it never sees one. If the two land in about the same place, reading the contexts without answering them makes little difference to what the model learns.

Every measurement here is taken on the answer distribution, which a rotation of the residual stream leaves unchanged, so the coordinate axes that Adam favors (the rotated-basis experiment) do not enter, except through the edit itself, which acts along e₁.

## The measurements

Each measurement is taken on the answer distribution at the query `=`, over the colors of the palette, and averaged over held-out contexts.

Distance from the target null
:   Total variation between the answer distribution of the model and the target null, with any mass the model puts off the palette counted as distance; 0 is a perfect match and 1 shares nothing. Lower is closer. On `difference` contexts for `τ_ℓ fixed` it is taken under the full edit, as in edit-effects; for `sans-diff` and the control it is taken on the clean pass. On the contexts of the other ops, for `sans-diff`, it is how far the model falls short of the predictor that is ideal for what it was trained on.

Bayes gap
:   The same distance, between the clean control and the Bayes predictor: how far a model of this size, trained without an anchor, falls short of the ideal on contexts like the ones it trained on. It sets the scale for the other distances. Taken on `difference` contexts, where edit-effects took its reference, and on the contexts of the other ops. edit-effects called it a comparison point rather than a floor for the edited model, since the two are distances between different pairs. For `sans-diff` the two pairs are the same: its Bayes predictor is the target null on every context, so its own gap on the contexts of the other ops is the floor that its distance on `difference` contexts can be measured against.

Share of the way
:   One minus the distance of a model from the target null over the distance of the control at the same seed, each a mean over `difference` contexts: how much of the way from the control to the target null that model has come. The control is the start of the way (0), and the target null its end (1). For the edited `τ_ℓ fixed` runs, edit-effects took this against their own clean pass, and it is reported both ways.

Distance between the models
:   Total variation between `sans-diff` and the edited `τ_ℓ fixed` run at the same seed, context by context, on `difference` contexts.

Seed-to-seed spread
:   The same distance, between two `sans-diff` runs at different seeds, over all 28 pairs: how far apart two models trained the same way land. A distance between the models near the spread means they are as alike as a model is to itself at another seed.
<!-- REVIEW: added the seed-to-seed spread and the own-gap reading of the distance on other-op contexts (assumptions review, agreed in the paper review). The paper review asked whether "the way from the control" is the floor; it is not, and the share of the way now says where the way starts and ends. Named a spread rather than a floor, since "floor" already means the no-evidence predictor (ex-2.2.21, ex-2.2.22) and edit-effects calls the Bayes gap "a comparison point rather than a floor". -->

Task score
:   Expected exact match, per op, on the clean pass.
"""

# %%

r"""
## Along training (E1)

How does `sans-diff` learn beside the control, which sees the same windows in the same order? The trajectories every run records hold the task score of each op on held-out contexts along training, so the six ops `sans-diff` trains on can be compared with the control and with `τ_ℓ fixed`, step for step. Because the batches match, a gap between `sans-diff` and the control on one of those ops is caused by the masked loss.

The `τ_ℓ fixed` runs kept checkpoints along training (at epochs 16, 32, and 64, then every 32 epochs), and `sans-diff` keeps the same ones. Scoring both at each kept epoch gives the distance from the target null on `difference` contexts along training, for `sans-diff` on the clean pass and for `τ_ℓ fixed` under the full edit, so we can see whether the two settle early or are still closing when training ends, and whether they close at the same pace. The controls kept no checkpoints, so this curve is not available for them; their task score on `difference` along training is.

The figure will show the task score per op along training, one panel per op, with the three conditions drawn per seed, and the two distance curves in a panel under it.

TODO.
"""

# %%

r"""
## How close it comes (E2)

How far is `sans-diff` from the target null on `difference` contexts, beside the edited `τ_ℓ fixed` runs and the Bayes gap? If it comes about as close as the edited model, then landing short of the target null is near what a model of this kind does on these contexts, and the shortfall need not come from the anchor and the edit (E3 asks whether the two miss in the same way). If it comes much closer, the anchor and the edit leave a distance that a model can close. If it is about as far as the Bayes gap, the target null is reached as well as the ideal is on ordinary contexts.

The figure will show the three distances per seed, with the share of the way in a table, and the same split by the posterior on `difference` as in edit-effects (E2 there). The contexts where the posterior on `difference` is lower are the ones whose examples partly fit other ops, which are closer to what `sans-diff` saw in training. Beside them, the distance of `sans-diff` from the target null on the contexts of the other ops gives its own Bayes gap, the shortfall it has on contexts like the ones it trained on.

TODO.
"""

# %%

r"""
## Whether it misses in the same way (E3)

Wherever `sans-diff` lands, do the edited model and `sans-diff` miss the target null in the same way? If the distance between the two models is small beside the distance of each from the target null, and near the seed-to-seed spread, they share the mass among the other ops alike on a context, and the part of the shortfall they share is a property of these contexts rather than of the anchor and the edit. If the two models are as far from each other as from the target null, they miss in different ways.

The two differ in two ways: `sans-diff` never learned the op, and it trained without the anchor. The clean `τ_ℓ fixed` run and the clean control at the same seed differ only in the anchor, so their distance on the contexts of the other ops bounds how much the anchor alone moves the answers there.

Confusion matrices will show where the mass goes, one small matrix per condition in a grid: the clean control, `τ_ℓ fixed` under the full edit, `sans-diff`, and the target null. Rows are the op of the context and columns the op whose answer gets the mass, as in ex-2.2.22, with the `difference` column hatched, since `sans-diff` never learned that op. The matrices pool over contexts, which can make a model look closer to the target null than it is context by context (ex-2.2.22 found this for the edited model), so the distance stays the main measure. The breakdown from edit-effects (E3 there) will also be repeated for `sans-diff`: the mass on answers only `difference` gives, on answers another op could give, and on colors no op gives; the distance from the nearest mixture of the answers of the ops; and whether a target null that spreads the mass more or less evenly over the other ops fits it better.

TODO.
"""

# %%

r"""
## What leaving the op out does to the other ops (E4)

On contexts of the other ops, does `sans-diff` score as well as the control, and does it gain where the target null says a model without `difference` should? The target null predicts a gain on contexts of other ops whose examples point partly at `difference`, since a model that cannot answer `difference` puts that mass on the right op. edit-effects found the edited model gains most of the way the target null predicts there, and this is the same comparison for a model that never learned the op, with the edited model beside it.

It also checks the training: if `sans-diff` scored well below the control on the other ops overall, its distance in E2 would partly be a weaker model rather than a missing op. The two `dropped` runs are compared with `sans-diff` at their seeds here and in E2.

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

[anneal-confirm](/docs/m2/anneal-confirm/report.py)
:   Trained the 400-epoch control, with no anchor, at seeds 712 to 719, beside the annealed recipe, and stored the dose scoring of both.

[edit-effects](/docs/m2/edit-effects/report.py)
:   On the `falloff` recipe the full edit lands about half of the way to the target null; the distance left is about 2.4 times the distance from the clean model to the Bayes predictor, nearly all of it in how the mass is shared among the other ops context by context.

[ex-2.2.22](/docs/m2/ex-2.2.22/report.py)
:   Confusion matrices of the clean and edited models against the target null, by how sure the context is about its op; pooled over contexts, they make the edited model look closer to the target null than the distance taken context by context.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` kept the anchor on all eight fresh seeds, 712 to 719, and stayed within the spill criterion; it became the recipe of record.

[rotated-basis](/docs/m2/rotated-basis/report.py)
:   Adam puts nearly half of the energy of the residual stream on one coordinate axis per seed; in a randomly rotated basis the anchor still holds on all eight seeds, and the energy moves to the rotated axes.

"""
