# D2.2 post: outline

A working outline for the LessWrong post on D2.2, the next in the sequence after *Anchoring one concept in a transformer* ([D2.1 post](/references/d2.1-anchored-transformer.md)). It collects what we would like the post to say, in the order a reader would meet it, and names where the evidence for each point lives, so the post can be written from the reports. Open points are marked **open**, each with a recommendation. The [design doc](design.md) holds the claims and the recipe in full; this page is about telling them.

## The post at a glance

Reader {.manual}
:   Someone who may not have read the earlier posts. The D2.1 post opened with a hand-drawn picture of the residual stream and built the recipe up one piece at a time, in about 2,200 words with ten figures. This one can lean on it for the basics, and should stay near that length. Method detail goes in LessWrong footnotes where a sentence or two will do, and in an appendix linked from the site where it won't.

Headline {.manual}
:   We anchored a concept the model has to infer, an operation that is never written down, onto an axis we chose. Editing that axis out removes most of what the model does with the op, in proportion to how much each context needed it, and leaves the other ops alone. On this model a direction fitted after training (LEACE, INLP) removes the op about as well, and the two edits are hard to tell apart. What the anchor adds is a location known before training: no search over where and how hard to edit, an edit that stayed within the spill criterion on every run without tuning, and a fixed geometry to analyse. The edited model ends up a little short of a model that never learned the op, and we have a lead on where the rest of the evidence for the op is kept.

    The sentence on fitted directions belongs in the opening summary at the top of the post, since it answers the first question a reader will ask ([second opinion on the comparison](https://claude.ai/artifact/W8KpG2fcAEjr4uhdbHXp6z), agreed in review).

What D2.1 promised {.manual}
:   The D2.1 post said we would try intervening on the anchored models, add more ops and anchor them, and compare with other methods. This post delivers the first two (with one op anchored of seven), and compares with several of the other methods (directions found after training). Gradient routing and SAE steering stay future work.

## Decisions

Settled: the body of the post reports on the per-visit runs, since the reference model, the confusion matrices and the write bound all come from them, and the setup says plainly how the labels were drawn. Over 400 epochs about 34,000 of the 42,891 `difference` contexts carry a label at some point. Each label scheme stands in for something real. The per-visit draw is like a very large corpus, where most contexts are seen once and a fixed share of them is labelled. The fixed draw is like a small labelled set reused over many epochs. The fixed draw gets its own part of the labels section (section 7).

Settled: the post includes the post-hoc comparison, which has reported, with `τ_ℓ fixed` and the per-visit draw as its anchored arm (section 6).

Leaning: no further experiments before the post. The comparison carries one caveat instead, that its reading without role labels is a lower bound on the fitters until a position gate is tried (section 6). The decision is being settled in the thread that weighed the comparison.

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

- Open with the same sentence as the summary: on this model a direction fitted after training removes the op about as well as the anchored edit, and the two edits are hard to tell apart on grading, landing and where the answers go. Then what the anchor still offers ([post-hoc compare, PR #318](https://github.com/z0u/sca2/pull/318); [second opinion](https://claude.ai/artifact/W8KpG2fcAEjr4uhdbHXp6z)).
- What was compared: the difference in means, a ridge probe, LEACE and INLP (linear erasers), and a direction found by gradient search, each fitted on the controls with every label and tuned per run over slice, site and dose, and edited through the same scorer as the anchor.
- The numbers, in two tables side by side. With role labels and a per-run grid, LEACE and INLP remove 0.94 to 0.97 of the way within the spill criterion, against 0.88 for the anchor, and land a little closer to the target null (0.54 to 0.55 against 0.50) with the same confusion pattern. Pooled over positions, which needs no role labels, LEACE removes 0.91 and INLP 0.89, about level with the anchor; at matched removal they spill past the criterion on 2 runs of 8 against 0 of 8, within chance of each other. The difference in means, the usual choice for steering, falls behind once role labels are taken away.
- So claims (b) through (e) describe what removing one direction does in this model. They hold for the anchored edit and for a fitted one, and the post should present them that way, in sections 4 and 5 as well as here.
- The case for the anchor is a different cost profile, as an alternative to an eraser:
    - The cost: an extra training term, a task score 0.003 lower, and a recipe that took some thirty experiments to settle. Post-hoc methods apply to a model that already exists, which is why most people use them. Say this plainly.
    - No search: the location is known before training, so there is no sweep over slice, site and dose for each model, and the full edit stayed within the criterion on all 8 runs without tuning. A fitted configuration chosen on the other seven runs missed the slice on one run in eight (seed 712), which kept 0.36 to 0.43 of its removal. Tell that in a sentence with the sample size beside it.
    - A fixed geometry to analyse: because e₁ was chosen, the write bound could be read off the clean model before any edit, and the arc of section 7 was visible only because one end of it was pinned. This is where the program is heading (several concepts, designed subspaces), and the post can name it as the direction of travel.
- What to leave out: label efficiency as an edge, since the fitters stay flat down to one label in fifty; the ceiling hypotheses (H3 and the gradient-searched direction), since they turn on the baseline and invite a question a footnote can't answer; and the verdict words of the report. Say what happened in plain words and link the preregistration.
- What to put in: which analyses were not preregistered (the pooled reading, the configuration chosen on other runs, the axis-baseline reading of H4); the pooled table beside the main one; and above it the caveat that the pooled reading is a lower bound on the fitters, since a per-position gate from the fitted direction's own score might recover much of the role-label edge without role labels. Two smaller caveats to carry: the seed 712 story is one run in eight, and the LEACE probe result is an open puzzle in the report. If the free re-analysis of the configuration chosen on other runs (picked at a small spill margin in place of at the criterion) runs before the post, the no-search point takes its number.
- Figures: one frontier panel per method, as seed-mean frontiers with a sans-diff mark; the confusion matrices with the anchored row set off by a bold label.

## 7. How few labels? (claim e)

- One `difference` context in fifty carries a label, the shape of label we expect for a concept in natural language, and about a quarter of the labelled contexts have a posterior below one half on `difference`, so many labels mark an op the context does not show ([design § e](design.md#e-scarce-noisy-labels-are-enough)).
- The fixed draw: the anchor held at 1/50, 1/25 and 1/10 on all 24 runs, and most runs removed the op about as well as under the per-visit draw. At 1/25 removal matched (with one late spill); at 1/50 and 1/10 it averaged 0.81, pulled down by a low tail of two or three runs in eight. No trend with label count between about 900 and 4,300 labelled contexts ([fixed-labels](/docs/m2/fixed-labels/report.py)).
- The fitters in section 6 were as flat down to one label in fifty, so scarce labels are enough for both, and the post shouldn't present them as an edge for the anchor.
- Retraining the same labels and seed moved which runs ended low, so whether a run ends low comes down to small numerical differences along training (fixed-labels E5).

### What the edit misses, and why some runs end low

- Every run writes the evidence that an example fits `difference` as one vector with two parts: along e₁, where the anchor put it, and along the direction a non-fitting answer has, which the loss made on its own. The readers (attention, at block 3 or 4) take some of the evidence from each part, and the edit reaches only the first ([low-runs](/docs/m2/low-runs/report.py)).
- Anchoring as an interpretability tool: fixing one end makes the other visible by contrast. The evidence is one arc on the unit sphere, from the non-fitting direction to e₁, and each example sits along it by how well it fits. Without e₁ in place these would be two unnamed directions. Figure candidate: the answer states of one run projected onto the plane of e₁ and the non-fitting direction, fitting and non-fitting examples in two colors, slices 2 and 3 side by side, with the arc drawn.
- A cheap check before writing: whether the non-fitting direction is shared across ops or differs by op (stored checkpoints, no training).
- Which aspect of the evidence the readers lean on is invisible to training: twins with the same labels, seed and batches match in loss and α for the whole run and part only in removal.
- What this says about anchoring: it controls where one aspect of the evidence goes, and the edit is bounded by how much the readers use that aspect. The fixed-label low runs are the stress case; under the per-visit draw the same thing costs a few hundredths.
- The headline says "most of what the model does with the op" and points here for the rest.

## 8. What we would do next

- The ones to name: gradient routing, a training-time method to compare with, and the D2.3 verification measurements.
- The comparison that tests what the anchor claims: a recipe transfer on fresh seeds, counting how often a fitted configuration chosen on earlier runs misses the slice against how often the recipe of record misses, preregistered after the post. The post can say it is planned.
- Briefly, if there is room: the position gate for the fitters; the same comparison for *red* on the D2.1 checkpoints; an edit-aware term that makes the readers depend on e₁ ([backlog](/todo/science/make-the-reader-depend-on-e1.md)); the anchor and anti terms at the [mid-block states](/todo/science/anchor-and-anti-at-mid-block-states.md); more ops anchored at once ([design § future work](design.md#future-work)).

## Appendix, linked from the site

- The write bound, in plain terms. In the autoencoder the bound was geometric: removing the e₁ part of a state moves it by no more than that part. In a transformer the edited states feed later blocks, which could write e₁ back onto the states of other ops and move them further than the edit did. The write bound says how much e₁ the later blocks may put on those states: no more than the clean model already has there. Because we chose the axis, that limit can be read off the clean model before any edit, and the measured writes stay within it past the embedding, apart from a few sites at slice 1 on one run in eight (write-floor, fixed-effects).
- The reference baseline (median over the quiet axes) and why Adam makes it fair; the absolute tolerance on the write bound; comparing conditions as two samples with a Welch t interval; perhaps the early α screen and why it cannot see a low run ([eng/determinism.md](/eng/determinism.md)).

- Compute overhead, a question readers will ask. If an anchored model learns the task in as many steps as a control, the overhead is the per-step cost of the anchor and anti terms, which can be counted analytically and checked with a timing probe. Both are being checked in a thread of their own.

## Figures

The D2.1 post took its figures from a page of their own ([D2.1 figures](/docs/m2/d2.1/report.py)), drawn from published results and uploaded from the bundle. A D2.2 figures page in the same shape is the natural next step once the sections above settle. It would hold the grammar images of section 1, the pooling picture of section 2, and the arc figure of section 7 drawn from stored checkpoints, beside the figures taken from the reports.

## Sources still moving

- The post-hoc comparison ([PR #318](https://github.com/z0u/sca2/pull/318)) has reported and is in review; the [second opinion](https://claude.ai/artifact/W8KpG2fcAEjr4uhdbHXp6z) on how to present it is agreed.
