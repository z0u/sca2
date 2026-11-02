# D2.2 post: outline

A working outline for the LessWrong post on D2.2, the next in the sequence after *Anchoring one concept in a transformer* ([D2.1 post](/references/d2.1-anchored-transformer.md)). It collects what we would like the post to say, in the order a reader would meet it, and names where the evidence for each point lives, so the post can be written from the reports. Open points are marked **open**, each with a recommendation. The [design doc](design.md) holds the claims and the recipe in full; this page is about telling them.

## The post at a glance

Reader {.manual}
:   Someone who may not have read the earlier posts. The D2.1 post opened with a hand-drawn picture of the residual stream and built the recipe up one piece at a time, in about 2,200 words with ten figures. This one can lean on it for the basics, and should stay near that length, with a longer method appendix linked from the site.

Headline {.manual}
:   We anchored a concept the model has to infer, an operation that is never written down, onto an axis we chose. Editing that axis out removes most of the op, in proportion to how much each context needed it, and leaves the other ops alone. The edited model ends up a little short of a model that never learned the op, and we can say where the rest of the op is kept.

What D2.1 promised {.manual}
:   The D2.1 post said we would try intervening on the anchored models, add more ops and anchor them, and compare with other methods. This post delivers the first two (with one op anchored of seven) and part of the third (directions found after training). Gradient routing and SAE steering stay future work.

## Decisions to make first

**Open:** which runs the body of the post reports on. Every measurement in the design doc comes from the recipe of record with its labels redrawn at every visit, so over 400 epochs about 34,000 of the 42,891 `difference` contexts carry a label at some point. A real pipeline labels a fixed set once. With a fixed draw the anchor held on every run (24 of 24), but removal fell to 0.81 on the seed mean at 1/50, with two or three runs in eight ending low ([fixed-labels, PR #323](https://github.com/z0u/sca2/pull/323)). Recommendation: keep the body on the per-visit runs, since the reference model, the confusion matrices and the write bound all come from them, and say plainly in the setup how the labels were drawn. Then give the fixed draw its own section beside the low runs, as the regime closest to production. Re-running the full suite on the fixed draw would delay the post for a result we can already describe.

**Open:** whether to wait for the post-hoc comparison. It is preregistered but has not run, and its anchored arm waits on the same label question ([post-hoc compare, PR #318](https://github.com/z0u/sca2/pull/318)). Recommendation: write the post around a section for it, and run it once the arm is chosen, since a comparison is the first question a reader will ask.

**Open:** how much of the path to the recipe to show. Recommendation: one short section with one figure and two problems (below), since the path is where most of the work went but little of what a reader needs.

Settled: the low runs get a section of their own, and the post goes ahead without fixing them.

## 1. From a word to an operation

- D2.1 anchored *red*, a concept with its own token. Most concepts worth controlling are never named in the text; the model works them out. So D2.2 asks whether anchoring works for a concept that has to be inferred ([pivot](pivot.md)).
- The task in one picture: a context of three solved examples and a query, with `?` where the op word would be (`red ? blue = …`). The model infers the op from the examples. A hand-drawn figure in the style of the D2.1 opener would carry this.
- Seven ops, `difference` anchored to e₁ of the residual stream of the same small nGPT as before (four blocks, 64 wide).

## 2. When the evidence is mixed

- Some examples show the answer of another op (replacement op noise), so a context can leave the op in doubt. The posterior over ops says how much each context needs `difference`, and it grades every measurement.
- The target null: what an ideal predictor that never knew `difference` would answer, weighing the other ops by how well they fit. It is the right target for a removed op, and it is what makes "removal" a graded, checkable number ([design § scoring](design.md#scoring)).
- Figure candidate: one context with its posterior bars, before and after the op is removed.

## 3. Anchoring an op, briefly

- What carried over from D2.1: the anchor and anti-subspace terms, labels on whole contexts, pooling that lets the term choose its states.
- What changed: the pull is pooled over slices as well as positions, so we do not have to say which block holds the op, and the anti term falls off with depth ([design § recipe](design.md#the-recipe-of-record)).
- The two problems along the way, in a sentence each: a color embedding standing in for the op (embedding-lean), then block 1 writing e₁ on every op (late-pull, slice-terms, soft-pool-scout). One figure of spill falling across the recipes, perhaps from recipe-compare.
- **Open:** whether the mellowmax pooling picture from D2.1 is worth redrawing with the slice axis added.

## 4. Did it land? (claim a)

- The anchor sits on the example answers, where a context shows the most about its op, and its alignment there rises with the posterior on `difference` (fixed-effects; example-evidence).
- It held on every run: 8 of 8 fresh seeds and 40 of 40 runs called early at epoch 64 (fixed-confirm, loss-rate). Seven of eight passed the task gate; the eighth learned two HSV ops late.
- Figure candidates: α by role and slice; the task change per op.

## 5. Taking it out (claim b)

- The full edit removes 0.88 of the way to the target null on the seed mean (0.83 to 0.91), and spill stays under the criterion on every run at every dose (fixed-effects, reference-baseline).
- The dose curve: removal is complete by three quarters of the full edit, and the drop on each context follows how much it needed the op.
- Why we edit every slice: an edit at fewer slices has to land between the block that writes the op and the block that reads it, and the reader moves from seed to seed. A larger model has readers at depths we don't choose, so the full edit is the one that carries over (block1-edit).
- Figure candidates: dose curve with spill underneath; share of the way to the target null across posterior bins.

## 6. Where the answers go (claim c)

- The edited model sends the mass to the other ops in about the right overall proportions, but context by context it covers about half the distance to the target null (fixed-effects confusion matrices).
- A model trained with no loss on `difference` contexts (sans-diff) comes 0.58 of the way where the edited model comes 0.50, and misses in the same way. Measured against a second sans-diff run, the edited model is about 85% of the way from the control to sans-diff. Name that endpoint whenever the 85% is quoted, since the plain share is about half (filtered-corpus).
- The surprise worth a paragraph: on contexts of other ops whose examples point partly at `difference`, the edited model gets better, because the mass the op took now goes to the right op. The gain grows with the posterior on `difference`; in the top bin the edit gets about 75% of the gain the target null predicts, and sans-diff about 94% (filtered-corpus E4).
- Figure candidates: confusion matrices (control, edited, sans-diff); the gain on other ops against the posterior.

## 7. Inside the model (claim d)

- Because we chose the axis, the clean model tells us in advance how much e₁ the states of the other ops carry, and so how far an edit can move them. The blocks after the edit stay within that bound (write-floor).
- Where the op lives: block 1 writes the evidence of each example onto e₁, one later block reads it at the query `=`, and an edit works when it sits between them (block1-edit).
- **Open:** whether this section stays in the body or moves to the appendix. Recommendation: keep the writer and reader picture in the body, since section 9 builds on it, and move the write bound to the appendix.

## 8. How few labels? (claim e)

- One `difference` context in fifty carries a label, the shape of label we expect for a concept in natural language, and about a quarter of the labelled contexts have a posterior below one half on `difference`, so many labels mark an op the context does not show ([design § e](design.md#e-scarce-noisy-labels-are-enough)).
- The fixed draw: the anchor held at 1/50, 1/25 and 1/10 on all 24 runs; removal at 1/25 matched the per-visit draw (with one late spill), and at 1/50 and 1/10 it averaged 0.81 with a low tail. No trend with label count between about 900 and 4,300 labelled contexts ([fixed-labels, PR #323](https://github.com/z0u/sca2/pull/323)).
- Retraining the same labels and seed moved which runs ended low, so whether a run ends low comes down to small numerical differences along training (fixed-labels E5).

## 9. What the edit misses, and why some runs end low

- Every run writes the evidence that an example fits `difference` as one vector with two parts: along e₁, where the anchor put it, and along the direction a non-fitting answer has, which the loss made on its own. The readers (attention, at block 3 or 4) take some of the evidence from each part, and the edit reaches only the first ([low-runs, PR #332](https://github.com/z0u/sca2/pull/332)).
- Anchoring as an interpretability tool: fixing one end makes the other visible by contrast. The evidence is one arc on the unit sphere, from the non-fitting direction to e₁, and each example sits along it by how well it fits. Without e₁ in place these would be two unnamed directions. Figure candidate: the answer states of one run projected onto the plane of e₁ and the non-fitting direction, fitting and non-fitting examples in two colors, slices 2 and 3 side by side, with the arc drawn.
- A cheap check before writing: whether the non-fitting direction is shared across ops or differs by op (stored checkpoints, no training).
- Which copy the readers lean on is invisible to training: twins with the same labels, seed and batches match in loss and α for the whole run and part only in removal.
- What this says about anchoring: it controls where one copy of the evidence goes, and the edit is bounded by how much the readers use that copy. The fixed-label low runs are the stress case; under the per-visit draw the same thing costs a few hundredths.
- **Open:** whether to lead the post with "removes most of the op" and keep this as its own section after the claims, or to bring it into the headline. Recommendation: its own section, with one sentence in the headline pointing to it.

## 10. Against directions found after training

- The difference in means, a ridge probe and LEACE (a linear eraser), fitted on the controls with every label, and edited through the same scorer ([plan](post-hoc-comparison.md), [PR #318](https://github.com/z0u/sca2/pull/318)).
- **Open:** the result. One framing point is settled already: since the anchor saw a fresh draw at every visit, the fitters and the anchor saw about as many distinct labelled contexts, so "fifty times the labels" no longer holds for the per-visit arm.

## 11. What we would do next

- An edit-aware term that makes the readers depend on e₁ (proposed in [PR #332](https://github.com/z0u/sca2/pull/332)); the anchor and anti terms at the [mid-block states](/todo/science/anchor-and-anti-at-mid-block-states.md); a [two-block model](/todo/science/two-block-model-for-the-writer-and-reader.md), where the reader can only be block 2; more ops anchored at once; gradient routing as a training-time comparison; the D2.3 verification measurements ([design § future work](design.md#future-work)).

## Appendix, linked from the site

- The reference baseline (median over the quiet axes) and why Adam makes it fair; the absolute tolerance on the write bound; comparing conditions as two samples with a Welch t interval; the early α screen and why it cannot see a low run ([eng/determinism.md](/eng/determinism.md)).

## Figures

The D2.1 post took its figures from a page of their own ([D2.1 figures](/docs/m2/d2.1/report.py)), drawn from published results and uploaded from the bundle. A D2.2 figures page in the same shape is the natural next step once the sections above settle, with hand-drawn diagrams for sections 1, 2 and 7, and the arc figure of section 9 drawn from stored checkpoints.

## Sources still moving

- Fixed labels ([PR #323](https://github.com/z0u/sca2/pull/323)) and low runs ([PR #332](https://github.com/z0u/sca2/pull/332)) are in review.
- The post-hoc comparison ([PR #318](https://github.com/z0u/sca2/pull/318)) is preregistered and waits on the choice of anchored arm.
