# D2.2 post: outline

A working outline for the LessWrong post on D2.2, the next in the sequence after *Anchoring one concept in a transformer* ([D2.1 post](/references/d2.1-anchored-transformer.md)). It collects what we would like the post to say, in the order a reader would meet it, and names where the evidence for each point lives, so the post can be written from the reports. Open points are marked **open**, each with a recommendation. The [design doc](design.md) holds the claims and the recipe in full; this page is about telling them.

## The post at a glance

Reader {.manual}
:   Someone who may not have read the earlier posts. The D2.1 post opened with a hand-drawn picture of the residual stream and built the recipe up one piece at a time, in about 2,200 words with ten figures. This one can lean on it for the basics, and should stay near that length. Method detail goes in LessWrong footnotes where a sentence or two will do, and in an appendix linked from the site where it won't.

Headline {.manual}
:   We anchored a concept the model has to infer, an operation that is never written down, onto an axis we chose. Editing that axis out removes most of what the model does with the op, in proportion to how much each context needed it, and leaves the other ops alone. The edited model ends up a little short of a model that never learned the op, and we have a lead on where the rest of the evidence for the op is kept.

What D2.1 promised {.manual}
:   The D2.1 post said we would try intervening on the anchored models, add more ops and anchor them, and compare with other methods. This post delivers the first two (with one op anchored of seven), and compares with three of the other methods (directions found after training). Gradient routing and SAE steering stay future work.

## Decisions

Settled: the body of the post reports on the per-visit runs, since the reference model, the confusion matrices and the write bound all come from them, and the setup says plainly how the labels were drawn. Over 400 epochs about 34,000 of the 42,891 `difference` contexts carry a label at some point. Each label scheme stands in for something real. The per-visit draw is like a very large corpus, where most contexts are seen once and a fixed share of them is labelled. The fixed draw is like a small labelled set reused over many epochs. The fixed draw gets its own part of the labels section (section 7).

Settled: the post waits for the post-hoc comparison, since a comparison is the first question a reader will ask. Its anchored arm is likely `τ_ℓ fixed` with the per-visit draw, which a Fable second opinion supports (section 6).

Settled: the low runs get a part of their own, and the post goes ahead without fixing them.

**Open:** how much of the path to the recipe to show. Recommendation: one short section with one figure and the two problems and their solutions (section 2), since the path is where most of the work went but little of what a reader needs.

## 1. From a word to an operation

- D2.1 anchored *red*, a concept with its own token. Most concepts worth controlling are never named in the text; the model works them out. So D2.2 asks whether anchoring works for a concept that has to be inferred ([pivot](pivot.md)).
- The task in one picture: a context of three solved examples and a query, with `?` where the op word would be. The model infers the op from the examples. This is where the grammar illustration goes: each example drawn as `a ? b = y` with `a` and `b` as small images and `y` their combination under the hidden op, then the query with `y` as the model answers it pixel by pixel ([backlog](/todo/style/equations-as-images-under-blend-modes.md)).
- Seven ops, `difference` anchored to e₁ of the residual stream of the same small nGPT as before (four blocks, 64 wide).
- When the evidence is mixed: some examples show the answer of another op (replacement op noise), so a context can leave the op in doubt. The posterior over ops says how much each context needs `difference`. The target null is what an ideal predictor that never knew `difference` would answer, weighing the other ops by how well they fit, and removal is measured as a share of the way to it ([design § scoring](design.md#scoring)). This needs only a short paragraph here, since section 4 measures against it; the picture of it waits for section 5.

## 2. Anchoring an op, briefly

- What carried over from D2.1: the anchor and anti-subspace terms, labels on whole contexts, pooling that lets the term choose its states.
- What changed: the pull is pooled over slices as well as positions, so we do not have to say which block holds the op, and the anti term falls off with depth ([design § recipe](design.md#the-recipe-of-record)).
- The two problems along the way and their solutions, in a sentence each: a color embedding standing in for the op, solved by holding the anti weight high on the first slices (embedding-lean, late-pull); then block 1 writing e₁ on every op, solved by keeping the pool over slices sharp (slice-terms, soft-pool-scout). One figure of spill falling across the recipes, perhaps from recipe-compare.
- The mellowmax pooling picture from D2.1, redrawn with the slice axis added. Its SVG in the repo ([pooling.svg](/docs/m2/ex-2.1.9/public/pooling.svg)) is small and hand-written, with light and dark colors, so adding an axis there is easy; editing the Figma original works too.

## 3. Did it land? (claim a)

- We did not know where the model would hold the op, only that it had to be somewhere. Pooling over slices and positions let the anchor term find it.
- The anchor sits on the example answers, where a context shows the most about its op, and its alignment there rises with the posterior on `difference` (fixed-effects; example-evidence).
- It held on every run: 8 of 8 fresh seeds and 40 of 40 runs called early at epoch 64 (fixed-confirm, loss-rate). Seven of eight passed the task gate; the eighth learned two HSV ops late.
- Figure candidates: α by role and slice, which shows where the term found the op; the task change per op.

## 4. Taking it out (claim b)

- The full edit removes 0.88 of the way to the target null on the seed mean (0.83 to 0.91), and spill stays under the criterion on every run at every dose (fixed-effects, reference-baseline).
- The dose curve: removal is complete by three quarters of the full edit, and the drop on each context follows how much it needed the op.
- Why we edit every slice: an edit at fewer slices has to land between the block that writes the op and the block that reads it, and the reader moves from seed to seed. A larger model has readers at depths we don't choose, so the full edit is the one that carries over (block1-edit).
- Where the op lives, briefly: block 1 writes the evidence of each example onto e₁, one later block reads it at the query `=`, and an edit works when it sits between them (block1-edit). Section 7 builds on this picture.
- Figure candidates: dose curve with spill underneath; share of the way to the target null across posterior bins.

## 5. Where the answers go (claim c)

- The target null in a picture: one context with its posterior bars, before and after the op is removed, beside what the edited model answers.
- The edited model sends the mass to the other ops in about the right overall proportions, but context by context it covers about half the distance to the target null (fixed-effects confusion matrices).
- A model trained with no loss on `difference` contexts (sans-diff) comes 0.58 of the way where the edited model comes 0.50, and misses in the same way. Measured against a second sans-diff run, the edited model is about 85% of the way from the control to sans-diff. Name that endpoint whenever the 85% is quoted, since the plain share is about half (filtered-corpus).
- The surprise worth a paragraph: on contexts of other ops whose examples point partly at `difference`, the edited model gets better, because the mass the op took now goes to the right op. The gain grows with the posterior on `difference`; in the top bin the edit gets about 75% of the gain the target null predicts, and sans-diff about 94% (filtered-corpus E4).
- Figure candidates: confusion matrices (control, edited, sans-diff); the gain on other ops against the posterior.

## 6. Against directions found after training

- The difference in means, a ridge probe and LEACE (a linear eraser), fitted on the controls with every label, and edited through the same scorer ([plan](post-hoc-comparison.md), [PR #318](https://github.com/z0u/sca2/pull/318)).
- How to describe the labels, for the comparison: the fitters get every `difference` label at once, about 43,000 contexts, while the anchor sees one in fifty per visit, which comes to about 34,000 distinct contexts over training. So the two are about level on distinct labelled contexts, and the comparison is fair in that sense. The fitters still have it easier in one way, since they see all their labels together on a model that has finished training. Whether the per-visit draw is realistic is a separate question, which section 7 answers with the fixed draw.
- The anchored arm, from a second opinion in the low-runs thread: `τ_ℓ fixed` with the per-visit draw as the main arm, since one pass over a large corpus is per-visit by construction and no per-visit run showed the low tail. The fixed 1/50 draw can be one robustness row below the main table, since its runs already exist; it asks whether the ranking survives with the anchor at its weakest.
- Say once that the fitters see every label at once while the anchor pull sees one in fifty per pass, so a reader doesn't take parity on distinct labels as a tie.
- Expect a small removal edge for the fitters: the e₁ edit leaves a few hundredths of the op in normal runs (the off-axis part of section 7), and a fitted direction can pick up both parts. That edge is no mark against the anchor, and the post should say why before showing the numbers.
- **Open:** the result.

## 7. How few labels? (claim e)

- One `difference` context in fifty carries a label, the shape of label we expect for a concept in natural language, and about a quarter of the labelled contexts have a posterior below one half on `difference`, so many labels mark an op the context does not show ([design § e](design.md#e-scarce-noisy-labels-are-enough)).
- The fixed draw: the anchor held at 1/50, 1/25 and 1/10 on all 24 runs, and most runs removed the op about as well as under the per-visit draw. At 1/25 removal matched (with one late spill); at 1/50 and 1/10 it averaged 0.81, pulled down by a low tail of two or three runs in eight. No trend with label count between about 900 and 4,300 labelled contexts ([fixed-labels, PR #323](https://github.com/z0u/sca2/pull/323)).
- Retraining the same labels and seed moved which runs ended low, so whether a run ends low comes down to small numerical differences along training (fixed-labels E5).

### What the edit misses, and why some runs end low

- Every run writes the evidence that an example fits `difference` as one vector with two parts: along e₁, where the anchor put it, and along the direction a non-fitting answer has, which the loss made on its own. The readers (attention, at block 3 or 4) take some of the evidence from each part, and the edit reaches only the first ([low-runs, PR #332](https://github.com/z0u/sca2/pull/332)).
- Anchoring as an interpretability tool: fixing one end makes the other visible by contrast. The evidence is one arc on the unit sphere, from the non-fitting direction to e₁, and each example sits along it by how well it fits. Without e₁ in place these would be two unnamed directions. Figure candidate: the answer states of one run projected onto the plane of e₁ and the non-fitting direction, fitting and non-fitting examples in two colors, slices 2 and 3 side by side, with the arc drawn.
- A cheap check before writing: whether the non-fitting direction is shared across ops or differs by op (stored checkpoints, no training).
- Which aspect of the evidence the readers lean on is invisible to training: twins with the same labels, seed and batches match in loss and α for the whole run and part only in removal.
- What this says about anchoring: it controls where one aspect of the evidence goes, and the edit is bounded by how much the readers use that aspect. The fixed-label low runs are the stress case; under the per-visit draw the same thing costs a few hundredths.
- The headline says "most of what the model does with the op" and points here for the rest.

## 8. What we would do next

- The ones to name: gradient routing, a training-time method to compare with, and the D2.3 verification measurements.
- Briefly, if there is room: an edit-aware term that makes the readers depend on e₁ (proposed in [PR #332](https://github.com/z0u/sca2/pull/332)); the anchor and anti terms at the [mid-block states](/todo/science/anchor-and-anti-at-mid-block-states.md); more ops anchored at once ([design § future work](design.md#future-work)).

## Appendix, linked from the site

- The write bound, in plain terms. In the autoencoder the bound was geometric: removing the e₁ part of a state moves it by no more than that part. In a transformer the edited states feed later blocks, which could write e₁ back onto the states of other ops and move them further than the edit did. The write bound says how much e₁ the later blocks may put on those states: no more than the clean model already has there. Because we chose the axis, that limit can be read off the clean model before any edit, and the measured writes stay within it past the embedding, apart from a few sites at slice 1 on one run in eight (write-floor, fixed-effects).
- The reference baseline (median over the quiet axes) and why Adam makes it fair; the absolute tolerance on the write bound; comparing conditions as two samples with a Welch t interval; perhaps the early α screen and why it cannot see a low run ([eng/determinism.md](/eng/determinism.md)).

## Figures

The D2.1 post took its figures from a page of their own ([D2.1 figures](/docs/m2/d2.1/report.py)), drawn from published results and uploaded from the bundle. A D2.2 figures page in the same shape is the natural next step once the sections above settle. It would hold the grammar images of section 1, the pooling picture of section 2, and the arc figure of section 7 drawn from stored checkpoints, beside the figures taken from the reports.

## Sources still moving

- Fixed labels ([PR #323](https://github.com/z0u/sca2/pull/323)) and low runs ([PR #332](https://github.com/z0u/sca2/pull/332)) are in review.
- The post-hoc comparison ([PR #318](https://github.com/z0u/sca2/pull/318)) is preregistered; Fable-san recommended `τ_ℓ fixed` with the per-visit draw as its anchored arm, with the fixed draw as an optional robustness row; the choice is still to be confirmed in that thread.
