# ruff: noqa: B018
# title: Directions found after training, beside the anchor

# A preregistered comparison: the anchored edit of the recipe of record against edits along directions fitted after
# training on the controls at the same seeds, each method tuned on a split it is never scored on. The plan is
# docs/m2/d2.2/post-hoc-comparison.md. Nothing has run; the placeholders are filled in place once it has.

r"""
# Directions found after training, beside the anchor

/// tip |
<!-- lede -->
We compare the anchored edit with edits along directions found after training: four standard fitters and one direction optimized against the scoring itself, each fitted on a control model with every label, tuned per run on data it is never scored on, and scored by the code that scores the anchor. <The outcome line, written when results land.>
///

Placing a concept during training costs a training term and some task score. This report asks what that buys over searching for the concept afterwards, on the same model and with the same scoring, so that the D2.2 blog post can say how anchoring compares with the methods a reader already knows.
"""

# %%

r"""
## Findings

- [The trade-off (E1)](#the-trade-off-e1) — <what we saw, in one sentence>.
- [Spill at matched removal (H1)](#spill-at-matched-removal-h1) — **<verdict>**. <One sentence.>
- [Removal at the spill criterion (H2)](#removal-at-the-spill-criterion-h2) — **<verdict>**. <One sentence.>
- [The optimized direction (H3)](#the-optimized-direction-h3) — **<verdict>**. <One sentence.>
- [The fitters find a placed concept (H4)](#the-fitters-find-a-placed-concept-h4) — **<verdict>**. <One sentence.>
- [Grading, landing, and where the answers go (E2)](#grading-landing-and-where-the-answers-go-e2) — <one sentence>.
- [How many labels a fitter needs (E3)](#how-many-labels-a-fitter-needs-e3) — <one sentence>.
- [The write against each bound (E4)](#the-write-against-each-bound-e4) — <one sentence>.
- [What a fresh probe can still find (E5)](#what-a-fresh-probe-can-still-find-e5) — <one sentence>.
- [The two baseline rules (E6)](#the-two-baseline-rules-e6) — <one sentence>.

/// admonition | How to read this report
This report was preregistered: the predictions, the grid, and the rule that picks a configuration from it were frozen at commit `<hash>`, before any run of this experiment. Each section opens with what we expect, and the results replaced the placeholders in place.
///
"""

# %%

r"""
## Why this experiment

The anchored pipeline trains the model with a pull toward the first basis vector e₁ on labelled `difference` contexts, and removes the op afterwards by projecting e₁ out of the residual stream at every slice and position. On the recipe of record that edit removes most of the op, costs the other ops less than the criterion, drops each context by about as much as it needed `difference`, and leaves a write the blocks after it keep within a bound set before the edit runs (the fixed-effects experiment).

A reader who knows the steering and erasure literature will ask whether a direction found after training does as well. The usual pipeline trains with no pull, fits a direction for the concept from the states and the labels, and projects it out, or replaces its component with a mean, or erases it with LEACE. If one of those matched the anchor on this model, the post would have to say that the placement bought nothing a search could not find.

Ex-2.2.1 asked this once, for *red* on the D2.1 checkpoints, and every fitted direction that removed at least half of *red* cost the lines with no red far more than the anchored edit did. That run gave the fitters little: three control seeds, labels from the token alone, one fit per slice, and no tuning. So this report gives the fitters every advantage we can, scores them the same way as the anchor, and asks where they land on the trade between removal and spill.
"""

# %%

r"""
## Runs and methods

No run is trained. The anchored runs are the eight `τ_ℓ fixed` runs of the fixed confirmation at seeds 712 to 719, which are the recipe of record. The controls are the eight runs of the anneal confirmation at the same seeds, trained with no pull, with the same initialization, batches, and corpus. Every post-hoc method is fitted on a control and scored on it, so each method has eight runs, and each comparison with the anchor is between two samples of eight runs.

Anchored
:   The full edit along e₁, as published in fixed-effects. Nothing is refitted or retuned.

Difference in means
:   The unit vector from the mean state of the other contexts to the mean state of `difference` contexts. The direction most steering work uses.

Ridge probe
:   The normalized weight vector of a linear readout of the target, fitted by least squares with an ℓ₂ penalty. It finds what can be read off the state linearly.

LEACE
:   Least-squares concept erasure ([Belrose et al., 2023](https://arxiv.org/abs/2306.03819)): the smallest affine edit, measured in the covariance-whitened norm, after which no linear probe can read the target. It removes one direction and measures how much to remove along another, so the edit is oblique.

Iterated probe (INLP)
:   The ridge probe fitted, projected out, and fitted again on what is left, to a rank of 2, 4, or 8; the edit removes the whole span. It answers the objection that a concept held across several directions can't be removed along one.

Optimized direction
:   A unit vector per edited slice, fitted by gradient descent on the control so that the edited answers on `difference` contexts move toward the target null while the answers on the other contexts stay where they were. It uses the target null, which no real pipeline has, so it is a diagnostic ceiling: about the best a rank-1 edit of this control could do with the scoring in hand. [COAST](https://arxiv.org/abs/2605.01167) is its nearest published relative.

The first four are the standard fitters that H1 and H2 judge. Each is also fitted on the anchored runs, where the concept sits on e₁ by construction, as a check that the fitters can find a concept that is there (H4). The sans-diff model, trained with no loss on `difference`, stays the reference for how close to the target null a model of this size comes.
"""

# %%

r"""
## What each method gets

Three sets of contexts, which never share a context: a fit set the directions are fitted on, a tuning set the configuration is chosen on, and the test set every published D2.2 score is taken on.

Fit set
:   280,000 fresh contexts, 40,000 per op, from the training generator of the run with its replacement op noise, each with its generating op and its posterior over ops: about the size of the training corpus, and drawn the way it was. The states are those of the clean pass of the run at every slice and position, and the fits are made from means and covariances accumulated over the set, so the states need not be held at once.

Tuning set
:   Another 14,000 fresh contexts from the same generator, drawn the same way.

Test set
:   The 14,000 held-out contexts of ex-2.2.21, 2,000 per op, so the anchored numbers need no recomputing.

Every method is fitted with a label on every `difference` context. The anchor had a label on a fiftieth of them at each visit, but the draw is made again at every visit, so over 400 epochs about three quarters of the `difference` contexts were labelled at some point, about one and a half times each. So in distinct labelled contexts the fit set and the anchor are about level, though the anchor met its labels a few hundred per pass over the corpus, through a stochastic term, and the fitters see all of theirs at once, in closed form. What the fitters lack is a term that shaped the whole representation around the concept, and that is the thing being measured; how much the label count itself matters to them is what the ladder of E3 measures.
<!-- REVIEW: the plan said the fitters get "fifty times the labels the anchor saw", and the first draft said "about level" for a fit set of 2,000 `difference` contexts. Under the context labeller (`sca.anchoring._context_mask`) the 1/50 draw is fresh at every visit; at 300k contexts of 24 tokens, a window of 96, an epoch of a quarter of the corpus, and 400 epochs, a context is wholly in view about 70 times, so about 0.75 of the 43k `difference` contexts draw a label at least once (34k distinct, about 650 per pass over the corpus). The fit set was enlarged to 40k `difference` contexts so the claim holds and the bottom rung of E3 (800 contexts) is not a sample-size test. Verify: `(1 - 0.02) ** 70`, and the fit set size above. -->

The fitters also get a graded target, the posterior on `difference` given the examples, which takes away the noise of a label that marks the generating op even where the examples point elsewhere.

Each method then gets a grid over how it is fitted, where it acts, and how much it removes:

| Factor | Levels |
| --- | --- |
| Target | binary (the generating op); graded (the posterior on `difference`), for the probe, LEACE, and the optimized direction |
| Sites fitted | every position of a context, pooled; the example answers; the query `=` |
| Slices edited | each slice alone; slices 1 to 3; every slice |
| Sequential refit | when several slices are edited, the fit at each slice may be made on the states as they arrive under the edits before it |
| Positions edited | every position; or the sites fitted |
| Dose | 0.25, 0.5, 0.75, and 1 of the component removed |
| Replacement | zero (the projection); or the mean coefficient over the contexts of the other ops (mean ablation) |
| Rank | 2, 4, or 8, for INLP, on the binary target |

The fits at the example answers and the query `=` use role labels a real corpus does not have, so they are generous diagnostics rather than candidate recipes, and the tables mark which sites each selected configuration used. The anchor gets no grid. Its one configuration is the published edit, reached through some thirty experiments at other seeds, and the fitters arrive with their own literatures behind them and a per-run grid on top.

Selection rule (frozen)
:   Per method and per run, every configuration is scored on the tuning set, net of the median drop under the same operator form along fifteen fixed coordinate axes other than e₁, and the frontier is taken: the configurations that no other beats on both removal and spill. Two points are read off the frontier. *Removal at the criterion* is where the frontier crosses a spill of 0.02, interpolated linearly between the two frontier configurations that bracket it, or the configuration with the least spill if none is within the criterion. *Spill at matched removal* is where the frontier crosses the seed mean of the anchored removal, interpolated the same way; a frontier that never reaches that removal has no such point, and the run is recorded as not matched. The bracketing configurations are then scored on the test set, and each point is their test scores mixed with the weights found on the tuning set, so the test set chooses nothing, and the point can sit a little off the line on the test set; the table shows where it sits.
<!-- REVIEW: the first draft took the tuning frontier on raw drops. A typical coordinate axis of a control costs about 0.03 per op at full dose (reference-baseline), so raw 0.02 on the tuning set would have meant about −0.01 net, and the selection would have sat inside the criterion it is judged against, biased against the fitters. The axis median is fitter-independent, so fifteen axes per operator form cost about 2,500 passes per run. Verify: the axis subsample is fixed before the run, and the test-set baselines are the full ones of the Measurements. -->

Removal at the criterion is what a practitioner with one of these methods would do: sweep the layer and the strength on a validation set, keep the side effects within a tolerance, and take the strongest edit that does.
"""

# %%

r"""
## Measurements

Every measurement is the one the D2.2 design and fixed-effects define, taken through `sca.intervention.apply`, so any method that produces a subspace and an operator is scored by the code that scores the anchor.

Removal
:   The drop in expected exact match on `difference` under the edit, as a share of the way from the clean score to the target null. Higher is better.

Spill
:   The largest drop on any other op under the edit. Lower is better, and the selectivity criterion is 0.02.

Net and raw
:   Published anchored scores are net of the median drop under the same edit along each of the other 63 coordinate axes of the run. A fitted direction is not an axis, so each post-hoc configuration gets a baseline in its own terms: the median drop over fifteen fits of the same fitter with the labels permuted across contexts, which gives a direction with no concept in it and the same kind of operator. On the controls the coordinate-axis median is computed too, and each hypothesis is judged on whichever of the two baselines favors the post-hoc method for that measure (the smaller for removal, the larger for spill). The anchor keeps its published baseline. Raw drops are shown beside net ones throughout.

Task change
:   The change in task score from the mean of the controls. Zero by construction for every post-hoc method, since the model is the control; for the anchor it is the published change, and it sits in the summary table as the price of anchoring.
"""

# %%

r"""
## The trade-off (E1)

This section comes first because it shows the shape of the result that the hypotheses then take points from. Each post-hoc configuration is one mark at its seed-mean removal and spill on the test set, with the frontier of each method drawn as a line, and the anchored edit as a point with its dose curve. A table beside it gives, per method, the two frontier points of the selection rule, raw and net, the task change, and which sites, slices, and replacement the bracketing configurations used.

/// admonition | TODO
Removal (y) against spill (x), one panel; a mark per configuration per method, the frontier per method as a line, the anchor as a point and its dose curve; the criterion as a vertical rule. Table: removal at the criterion, spill at matched removal, raw and net, task change, and the selected configurations, per method.
///
"""

# %%

r"""
## Spill at matched removal (H1)

**What we expect.** At the removal the anchor reaches, each standard fitter spills past the criterion on most runs. We will look at the count of runs, out of eight, whose spill at matched removal is over 0.02 net, with a run that is not matched counted as over, and compare it with the count of anchored runs over 0.02 at full dose by a one-sided Fisher exact test. A fitter passes if its count is five or more and the test gives p below 0.05; it misses if its count is under five or the test does not separate it from the anchor. H1 passes if every fitter that passes H4 passes, is partial if some do, and misses if none do. A fitter that matches the anchor would mean that, on this model, a searched direction serves as well as a placed one.

/// admonition | TODO
Spill at matched removal per run, per fitter, with the anchor at full dose beside it; runs not matched marked at the top of the axis. Table: the count over the criterion per method and the Fisher p.
///
"""

# %%

r"""
## Removal at the spill criterion (H2)

**What we expect.** Kept within the criterion, each standard fitter removes less than the anchor. We will look at the difference of seed means between the anchored removal (at full dose, or where its own dose curve crosses the criterion if it spills past it) and the removal at the criterion of each fitter, with a 95% Welch t interval. A fitter passes if the interval lies above zero, and misses if it does not. H2 passes if every fitter that passes H4 passes, is partial if some do, and misses if none do. With eight runs a side, gaps under about 0.05 are hard to see, so a miss with a narrow gap says the two are close, and a miss with a wide interval says the seeds could not tell.

/// admonition | TODO
Removal at the criterion per run, per fitter and for the anchor, with seed means. Table: the difference of seed means and its interval per fitter.
///
"""

# %%

r"""
## The optimized direction (H3)

**What we expect.** The optimized direction comes closer to the anchor than any standard fitter does, on both measures: its removal at the criterion is at least the highest seed mean among the standard fitters, and its count over the criterion at matched removal is at most the lowest. It passes if both hold, and misses if neither does; one without the other would be outside the plan, and the verdict would be Unresolved. Beside the verdict we say how it stands against the anchor. If its Welch interval on removal at the criterion contains zero and the Fisher test does not separate its count from the anchor, it matches the anchor, and the advantage of placing the concept is that no search is needed. If the interval lies above zero, it falls short too, and the control holds the op in a way no rank-1 edit removes cleanly, which is the stronger claim for the anchor.

/// admonition | Open decision
The verdict here sits on the clause that changes the post least (the ceiling beats the standard fitters), while the clause that decides how the post is framed (whether the ceiling matches the anchor) carries none. The alternative is to split it: H3a, the ceiling against the standard fitters as above; H3b, the ceiling against the anchor, with "matches" (interval contains zero and Fisher does not separate) and "falls short" (interval above zero) as the two named outcomes, and anything else Unresolved. Proposed: split, so a reader of the Findings meets the outcome that matters.
///

/// admonition | TODO
The two measures for the optimized direction beside the standard fitters and the anchor, per run.
///
"""

# %%

r"""
## The fitters find a placed concept (H4)

**What we expect.** On the anchored runs, each standard fitter finds e₁. Restricted to fits on the pooled positions or at the example answers, take the configuration selected for removal at the criterion. Its target and site give a clean-state fit at each of slices 1 to 3, whichever slices the configuration edits, and each of those directions should be within a cosine of 0.9 of e₁ (the seed mean of its absolute value; for INLP, the largest cosine between e₁ and the span). Its removal at the criterion should be within 0.1 of what the edit along e₁ removes at full dose on the same runs. A fitter passes if both hold, and misses if either does not. A fitter that finds e₁ but has to stop short of it to stay within the criterion, as INLP may when its extra directions cost spill, misses: it would not be a fair stand-in for its method on the controls either. The query `=` is left out: the clean alignment there is near zero in the anchored runs (the mid-block-edit experiment), so a fit at that site is not expected to find e₁.

The rule for a miss, frozen here: a fitter that misses H4 is left out of the verdicts of H1 to H3, its rows stay in the tables marked as such, and the report names it as a method the comparison could not fairly score.

/// admonition | TODO
Cosine with e₁ per slice and per run, per fitter; removal at the criterion on the anchored runs against the edit along e₁.
///
"""

# %%

r"""
## Grading, landing, and where the answers go (E2)

At the configuration each method selects for removal at the criterion, the scoring suite of fixed-effects: the share of the way to the target null removed in each bin of the posterior on `difference`, and the gain on the contexts of other ops by bin; how far the edit moves the answers toward the target null in total variation, and how far from the control toward the sans-diff model; and the confusion matrices of what the edited model gives on `difference` contexts.

/// admonition | TODO
Grading by posterior bin per method beside the anchor; landing on both scales; confusion matrices per method.
///
"""

# %%

r"""
## How many labels a fitter needs (E3)

At the configuration each method selects, the fit is repeated with labels on every `difference` context of the fit set, on a fixed half, a tenth, and a fiftieth of them, with the unlabelled `difference` contexts counted among the others, as an unlabelled context is for the anchor. A fixed fiftieth is 800 contexts, about what the anchor met in one pass over the corpus and far fewer than it met over training, so the lowest rung is harsher than the anchor had it.

/// admonition | TODO
Removal at the criterion (y) against the label rate (x, log scale), per method, with the anchor as a horizontal line.
///
"""

# %%

r"""
## The write against each bound (E4)

For a direction that is not an axis, or an oblique eraser, the write bound of the anchor does not carry over as it is. So each method gets the bound in its own terms: the turn its operator makes on the clean states of the other ops (the 99% quantile, per slice and role), against the turn it makes on the states as they arrive under the edits at the slices before. A method whose blocks put the concept back shows a write over its own bound.

/// admonition | TODO
The write over the bound per slice and role, per method, at the selected configuration.
///
"""

# %%

r"""
## What a fresh probe can still find (E5)

LEACE promises that no linear probe can read the concept after its edit, and the anchor promises nothing about that. So a fresh ridge probe for the binary target is fitted on the edited states of the test set, with cross-validation over contexts, per slice and per method. The evidence for the op can be recomputed from the operand colors (the block1-edit experiment), so a method may leave the op readable even where the answer no longer uses it, and a readable concept bounds no intervention.

/// admonition | TODO
Probe accuracy (or R²) on the edited states per slice, per method, with chance marked.
///
"""

# %%

r"""
## The two baseline rules (E6)

On the controls, the permuted-label baseline and the coordinate-axis median are compared, configuration by configuration, for the configurations scored on the test set. Adam parks energy on some axes of a control, so the two can differ, and this says by how much and in which direction.

/// admonition | TODO
Permuted-label baseline (y) against coordinate-axis baseline (x) per configuration, removal and spill in two panels.
///
"""

# %%

r"""
## Discussion

<!-- Written after a paper round on the result sections. -->
"""

# %%

r"""
## Method

**The fits.** Difference in means takes the binary target only. The ridge penalty is fixed at the value ex-2.2.1 used. LEACE takes a small ridge on the covariance, since the stream is unit-norm and its covariance is close to rank-deficient. INLP fits the ridge probe, projects its direction out of the fit states, and repeats. A fit at a slice set of several slices is one fit per slice; e₁ is shared across slices by construction, and a fitted direction need not be. With sequential refit, the fit at slice ℓ is made on the fit set as it arrives under the edits at the slices before ℓ, which keeps an oblique eraser on the states it will see (ex-2.2.1 found it extrapolates poorly off them). Mean ablation replaces the coefficient with its mean over the fit contexts of the other ops.

**The optimized direction.** A unit vector per edited slice, initialized at the difference in means and fitted by Adam for a few hundred steps on the fit set: the KL divergence from the target null to the edited answers on `difference` contexts, plus a weight times the KL divergence from the clean answers to the edited ones on the other contexts, at full dose. The weight takes four values, and the dose and slice sets are crossed as for the others. Its configurations join the frontier like any other.

**Grid size.** About 360 configurations for difference in means, 720 for the probe and for LEACE, and about 1,080 for INLP, counting doses; each is one scoring pass, reading the logits at the query answer only. The whole grid is scored on the tuning set, and again on the test set for the figure of E1, where nothing is chosen. A fit is made once per (target, site, slice, refit) and reused across doses and positions.

**The baseline.** On the tuning set, the median over fifteen coordinate axes other than e₁, drawn once with a fixed seed before the run and shared by every run. On the test set, fifteen permuted fits per configuration, at the bracketing configurations of each point. LEACE is homogeneous in its covariance vector, so a permuted fit is a full-strength oblique erasure along a reference direction. The coordinate-axis median applies the same operator form (slices, positions, dose, replacement) along each of the 63 axes other than e₁; it does not depend on the fitter, so it is computed once per form.

**Comparisons.** Each method against the anchor as two samples of eight runs: the difference of seed means with a 95% Welch t interval, and Fisher's exact test for counts of runs over the criterion.

**Smoke test.** Before the full run, the pipeline runs end to end on two stored runs outside the comparison: the control of ex-2.2.23 at seed 704 and the `falloff` run at the same seed. It checks H4 on the anchored run, so that a fitter that can't find a placed concept is found before the full run, it shows where a probe for the op scores best on the control, and its timing gives the cost of the full run.

/// admonition | Open decision
Whether the control holds the evidence for the op at the example answers, as the anchored model does, is not known. The block1-edit experiment found that the control gathers the operand colors at each example answer in block 1 as the anchored runs do, but it fitted no probe for the op on a control. The smoke test fits one at every role and slice of the control at seed 704; if it scores best at a role other than the example answers or the query `=`, that role joins the sites fitted before the freeze.
///

/// admonition | Open decision
Whether to run the same comparison for *red* on the D2.1 checkpoints, where ex-2.2.1 gave the fitters far less. The code would carry over as it is. Proposed: not in this report, which is about the inferred op; a backlog item, to take up if the post or the paper shows the *red* result beside this one.
///

**Cost.** No training. Each run takes about 6,000 scoring passes: the grid on the tuning and test sets, the axis baseline on the tuning set, the permuted fits, and the ladder. The anchored runs take the same for H4, so sixteen runs come to about 100,000 passes, plus the state passes over the fit set and for sequential refit, and the optimized direction. The plan estimates eight to ten L4-hours as eight containers in parallel, about \$15, with \$25 as the ceiling; the smoke test gives the figure for the full run before it starts.
"""

# %%

r"""
## References

[ex-2.2.1](/docs/m2/ex-2.2.1/report.py)
:   On the D2.1 checkpoints, every fitted direction that removed at least half of *red* from the control cost the lines with no red 0.37 or more, an order of magnitude above the anchored edit.

[fixed-effects](/docs/m2/fixed-effects/report.py)
:   The full scoring suite on the recipe of record at seeds 712 to 719: removal 0.88 at the full dose, spill below the criterion at every dose, graded drops, a landing of 0.50, and a write within the bound.

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` at fresh seeds 712 to 719 keeps every anchor; it is the recipe of record, and its checkpoints and dose scoring are stored.

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   Trained the controls at seeds 712 to 719 that this comparison fits on, sharing initialization, batches, and corpus with the anchored runs.

[block1-edit](/docs/m2/block1-edit/report.py)
:   Block 1 writes the evidence for the op onto e₁ at the example answers, and a later block reads it at the query `=`.

[mid-block-edit](/docs/m2/mid-block-edit/report.py)
:   Editing the mid-block states as well gains nothing and spills; in passing it shows the query `=` near zero alignment throughout.

[reference-baseline](/docs/m2/reference-baseline/report.py)
:   The median drop along the other coordinate axes of the run replaces the paired control as the baseline, and the published numbers barely move.

[sans-diff model](/docs/m2/filtered-corpus/report.py)
:   A model trained with no loss on `difference` comes 0.58 of the way from the control to the target null, against 0.50 for the edited recipe of record, and misses in the same kind of way.

[ex-2.2.21](/docs/m2/ex-2.2.21/report.py)
:   The 14,000 held-out contexts, 2,000 per op, that every D2.2 score is taken on.
"""
