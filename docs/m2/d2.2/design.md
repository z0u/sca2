# D2.2 design: anchoring an operation the model infers

D2.2 asks whether Sparse Concept Anchoring works for a concept that is never written down. Each line of the corpus is one context: three solved examples of one hidden operation, then a query under the same op, written `op1 ? op2 = answer` with a constant `?` where the op word would be. The model has to infer the op from the examples to answer the query. We anchor one op, `difference`, to the first axis of the residual stream (e₁), and then remove it by editing that axis out.

The [pivot](pivot.md) explains why we took the op words out of the grammar, and the [index](/docs/index.md#d22-anchoring-an-operation) lists every experiment along the way.

## Where it stands

The recipe of record keeps the anchor on every run we have trained, at little cost to the task. The full edit removes most of `difference`, and the other ops stay within the selectivity criterion on every run. The edited model sends the mass to the other ops in about the right overall proportions, but context by context it gets only about half of the way to where an ideal predictor would.

Two pieces of work remain before the blog post: a model trained without `difference`, to show how far any model could land, and the full set of measurements on the recipe of record, confusion matrices included. Everything else is future work.

## The recipe of record

The recipe is `τ_ℓ fixed`, adopted after the fixed confirmation. Its pull is pooled over slices 1 to 3, so the recipe does not have to say which block holds the op.

Task {.manual}
:   Seven ops: `mix`, `lighten`, `darken`, `difference`, and three HSV blend modes (`hue-hsv`, `sat-hsv`, `value-hsv`). Answers between grid levels round up or down at random, so a model is scored on the probability it puts on the right answer. Three examples in ten show the answer of another op (replacement op noise), so some contexts leave the op in doubt.

Model and training {.manual}
:   A d64-L4 simplified nGPT with a readout table of its own and attention that stops at the line break, trained for 400 epochs on a cosine schedule.

Labels {.manual}
:   About one `difference` context in fifty carries a label saying only that it uses `difference`. The model never sees the label, and no other op is labelled.

Pull {.manual}
:   On a labelled context, the anchor term pulls the states at slices 1 to 3 toward e₁, pooled over every slice and position. The pool over positions starts soft and sharpens to a temperature $\tau_p$ of 0.1 by epoch 40, and the pool over slices stays at a temperature $\tau_\ell$ of 0.1 throughout. The anchor weight $\lambda_a$ warms up to 0.1 over the same 40 epochs and eases off at the end. The embedding and block 4 are not pulled.

Anti-subspace term
:   A light push on every state, labelled or not, away from e₁, so the pull has to earn its alignment. Its weight starts at 0.25 on every slice and anneals by epoch 360 to a hold that falls off with depth, from 0.2 at the embedding to 0.03 at block 4. The high hold on the first slices keeps the color embeddings from standing in for the op.

Edit {.manual}
:   Remove a share of the e₁ component (the dose, from a quarter to all of it) from the state at every position and every slice.

## Scoring

Every run is scored on held-out contexts, and each drop under the edit is net of a baseline: the median drop under the same edit along each of the other 63 coordinate axes of the run. A control trained without the anchor at the same seed gives the task change, and checks now and then that the baseline still agrees with it.

Task change
:   The clean score per op, less that of the control. The task gate fails below −0.02 on the mean over ops, or below −0.05 on any one op.

Removal
:   The drop on `difference` under the full edit, as a share of the way to the target null. A run that removes less than half has lost the anchor.

Spill
:   The largest drop on any other op, at any dose. The selectivity criterion is 0.02.

Landing
:   The share of the distance to the target null that the full edit covers, context by context.

Write bound
:   How much e₁ the blocks after an edit put back on the states of the other ops, against what the clean model has there.

The baseline leans on Adam giving the coordinate axes a privileged role: the model parks its syntax tokens on a few axes and leaves the rest quiet, so the other axes make a fair reference. The anchor itself does not need this; it held as before with the basis of the optimizer rotated.

## The claims

For the narrow domain of color mixing, we want to be able to say five things.

### (a) An inferred op can be anchored on every run, without costing the task

The anchor sits on the example answers, where the context shows the most about its op, and its alignment there rises with the posterior on `difference`. That fits an anchor that holds the inferred op, though it follows the evidence of each example on its own more than the evidence of the whole context.

Losing the anchor was the largest gap in earlier recipes. The recipe of record kept it on all eight runs at fresh seeds and on all forty runs of an early count, which puts the loss rate below about one in fourteen. Seven of the eight fresh runs passed the task gate; the eighth learned two HSV ops late and was still catching up at the end.

### (b) Removing the op is graded and selective

A stronger edit takes out more of `difference`, and the other ops are left as they were at every dose. On the recipe of record the full edit removes 0.88 of the way to the target null on the seed mean (0.83 to 0.91), and no run ends near the spill criterion or rises toward it late in training. The drop on each context follows how much it needed `difference`, which was measured on the scout seeds and is still to be measured at the fresh ones.

The spill of earlier recipes came from a stand-in for the op in the color embedding table, and later from block 1. Holding the anti weight high on the first slices and keeping the pool over slices sharp took both away.

### (c) The edited model answers in a predictable way

Once `difference` is removed, the model should answer as an ideal predictor that never knew `difference` but still weighs the other ops by how well they fit the examples: the target null. The edited answers go to the other ops in about the right overall proportions, but within a context they weigh those ops differently, and the full edit covers only about half of the distance.

We don't know whether half is short. A model trained without any `difference` contexts would show how close a model can come, and that is the first piece of work before the blog post.

### (d) The side-effects are bounded in advance

Because we placed the axis, the clean model tells us how far an edit can turn the states of other ops. The claim to show is that the blocks after the edit put no more e₁ back on those states than the clean model has there. On `falloff` the write stayed within that bound past the embedding. Under the pool it stays close to the bound, with a few sites over it at slice 1, where the bound is only a few degrees.

On the recipe of record the evidence for the op peaks at slice 3 on every seed. Why editing block 1 alone removes most of the op under the older step recipe, which never pulled block 1, is still open.

### (e) Scarce, noisy labels are enough

About one `difference` context in fifty is labelled, which is the shape of label we expect for a concept in natural language. The labels are already noisy in one sense: a label marks the op that generated the context, even when its examples favor another op. On the op word, a fifth of the labels on the wrong op still placed the op; that is untested for the inferred op.

## Before the blog post

A reference model without `difference`
:   The control recipe on a corpus with no `difference` contexts, at a few seeds, scored on the held-out `difference` contexts with the landing measurements. It tells us whether half of the way to the target null is short (claim c), and it is the filtered-corpus comparison too. In the [backlog](/todo/science/reference-model-without-the-anchored-op.md).

The full scoring suite on the recipe of record
:   The measurements of the edit-effects report (removal by dose, grading, landing, and the write bound) at the fresh seeds 712 to 719, and the confusion matrices of ex-2.2.22 (by posterior bin, the clean model, the edited model, and the target null), with the reference model beside them. The runs and some of the measurements are already stored, so this is mostly a re-analysis. In the [backlog](/todo/science/full-scoring-suite-on-the-recipe-of-record.md).

## Future work

These are leads, roughly in the order we would take them up.

- Why block 1 alone holds most of the op on the step recipe, and a map of what e₁ holds along the context ([backlog](/todo/science/what-e1-holds-beside-the-op.md)).
- Whether the soft start of $\tau_p$ is still needed now that $\tau_\ell$ is sharp.
- Reference models without each op in turn, with that op anchored in place of `difference` ([backlog](/todo/science/reference-models-for-each-anchored-op.md)).
- Comparisons with methods that do not place the concept: post-hoc directions on the controls, and gradient routing ([backlog](/todo/science/baseline-comparisons-sca-plan-related-work-delta.md)).
- Wrong labels on the inferred op ([backlog](/todo/science/wrong-labels-on-the-inferred-op.md)).
- Whether the rest of the representation (the color geometry and the states of the other ops) keeps its shape under the anchor.
- A sweep of the recipe weights and schedules together ([backlog](/todo/science/full-recipe-sweep-after-the-lean-fix.md)), and a run at [600 epochs](/todo/science/spill-at-600-epochs.md).
- An interactive view of α, sliced by op, role, slice, seed and epoch ([backlog](/todo/style/interactive-alpha-explorer.md)).

## Out of scope

The verification measurements (D2.3), though verification lines are allowed in the corpus. Topic markers, and anchoring at fine-tune time, both open questions in the [pivot](pivot.md#open-questions). Several ops on separate axes.

## References

[ex-2.2.14](/docs/m2/ex-2.2.14/report.py)
:   Anchoring the op word: the anchor landed on the word itself, which led to the pivot. Scarce and partly wrong labels placed it as well as full ones.

[ex-2.2.22](/docs/m2/ex-2.2.22/report.py)
:   Pooled over contexts, the edited model answers about as the target null would, though context by context it gets less than half of the way there.

[embedding-lean](/docs/m2/embedding-lean/report.py)
:   The early spill came from lightness loaded onto e₁ in the color embedding table, a stand-in for `difference` before the model has read the context.

[edit-effects](/docs/m2/edit-effects/report.py)
:   On `falloff`, the write stays within the bound, the drop on each context follows how much it needed `difference`, and the full edit lands about half of the way to the target null.

[reference-baseline](/docs/m2/reference-baseline/report.py)
:   The median over the other coordinate axes of the run replaces the paired control as the baseline, and the numbers barely move.

[rotated-basis](/docs/m2/rotated-basis/report.py)
:   With the basis of the optimizer rotated, the anchor holds and removes as before; only the paired-control baseline suffers.

[recipe-compare](/docs/m2/recipe-compare/report.py)
:   The pool over slices 1 to 3 removes about as much as the step recipes, and its spill enters with the edit at block 1.

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The final slice temperature decides the early spill of the pool; at $\tau_\ell$ 0.1 no run spills, and the edit at block 1 costs about what it does under the step.

[late spill scout](/docs/m2/late-spill-scout/report.py)
:   The late spill follows the anti anneal, and sharpening $\tau_\ell$ late lowers it.

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` at fresh seeds keeps every anchor, removes about as much as the annealed version, and has no late spill; adopted as the recipe of record.

<details markdown="1"><summary>About this page</summary>

Rewritten 2026-10-28 for the recipe of record `τ_ℓ fixed`, and simplified: the provenance of each setting moved to the reports, the claims say where they stand, and the open work is split into what comes before the blog post and what is future work. The longer version, with its history of the recipe, is in git (as of `a299178`), and the plan through ex-2.2.22 is [archived](design-2026-10.md).

</details>
