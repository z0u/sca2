# ruff: noqa: B018
# title: Making the reader depend on e₁

# A plan for an exploratory study: the recipe of record at a fixed 1/50 label draw, with a term that scores the
# model under the e₁ edit on the labelled contexts, against the 24 stored fixed-draw runs at 1/50. Nothing has run;
# every result is a TODO. The experiment module (`experiment.py`) comes once the plan is agreed.

r"""
# Making the reader depend on e₁

/// tip |
<!-- lede -->
On a fixed label draw, a few runs end with the e₁ edit leaving much of `difference` in place, because the blocks that read the op take part of its evidence from a direction off the axis. Nothing in training sees that choice. Here we add a term that does: on each labelled context, the model also answers with e₁ removed, and that answer is trained toward what the context would give without `difference`.
///

The [low-runs](/docs/m2/low-runs/report.py) re-analysis found that every run writes the evidence that an example fits `difference` in two parts: along e₁, where the anchor put it, and along one other direction, which the loss made on its own. The low runs are the ones whose reader blocks take enough from the off-axis part for the edit to leave about a fifth of the op in place. A run and its twin (same labels, seed, and batches, different last bits of arithmetic) match in loss and in alignment for the whole run, and part only under the edit. So the loss is flat along the choice the edit depends on.

This plan makes the loss see that choice. It trains 16 new runs at the eight seeds of the [fixed-labels](/docs/m2/fixed-labels/report.py) experiment, in two forms of the term, and compares them with the 24 stored runs that trained the recipe of record on a fixed 1/50 draw. The question is whether a term that scores the edited model removes the low runs, and whether it does so by changing what the reader uses or by a shortcut that only holds at the edit it was trained on.
"""

# %%

r"""
## Observations

- [The low tail (E1)](#the-low-tail-e1): TODO.
- [What it costs (E2)](#what-it-costs-e2): TODO.
- [Writer or reader (E3)](#writer-or-reader-e3): TODO.
- [Graded, or a cue (E4)](#graded-or-a-cue-e4): TODO.

## Scope and method

This is an exploratory study, with no preregistration and no gate. Every run is the recipe of record, `τ_ℓ fixed`, trained for 400 epochs on a fixed 1/50 label draw, as the fixed-labels experiment trained it. The draw at each seed is the one stored by that experiment, so a new run and the stored run at its seed share their labelled contexts, initialization, and batches. They are still compared as two samples rather than as pairs, since two runs of one seed under slightly different recipes end about as far apart as runs at different seeds.

Three conditions:

`fixed` (stored)
:   The comparison: the eight fixed 1/50 runs of the fixed-labels experiment and its 16 crossed runs, 24 in all, with no new training. Seven of the 24 end low (removal under 0.8). Over the 24, removal has a mean of 0.81 and a standard deviation of 0.10, against 0.88 and 0.03 over the eight per-visit runs of the [fixed confirmation](/docs/m2/fixed-confirm/report.py).

`edit-zero`
:   The recipe with the edit-aware term at full dose: on each labelled context, a second forward pass removes e₁ at every slice and position (the full edit, as the scorer applies it), and its distribution over the query answer is trained toward the target null of that context. Eight runs, seeds 712 to 719.

`edit-dose`
:   The same term with a random dose: each labelled context draws a dose $u$ uniformly between 0 and 1, the edited pass removes that share of the e₁ component, and the target is the mixture $(1-u)\,p_\text{with} + u\,p_\text{null}$, where $p_\text{with}$ is the ideal answer distribution with `difference` among the ops and $p_\text{null}$ the target null. Eight runs, seeds 712 to 719.

Both forms come from the backlog item that followed low-runs. The full-dose form is the most direct version of the idea. Its weak point is that a state with e₁ fully removed but the off-axis evidence intact never occurs in clean training, so the model could learn to answer the null on that combination alone and go on reading the off-axis part everywhere else. A random dose with a graded target leaves no single state to detect. E4 compares the two on that.

The term sees only the query answer of labelled contexts that lie whole inside the training window. Its weight $\lambda_e$ follows the anchor weight schedule: it warms up with the pull over the first 40 epochs and eases off with it at the end, so the term is on while the anchor is placing e₁. Before e₁ holds the op, the edited pass is close to the clean one and the term pulls the answer of a labelled context toward the null, against the task loss. That gives the model a reason to put the evidence on e₁, which is what the anchor asks for, but it may also cost the task on `difference` while it lasts, so E2 looks at the task change along training as well as at the end.

The edited pass carries gradients through every block. The term can then lower its loss in two ways: by moving the off-axis evidence onto e₁ (the writer), or by having the later blocks stop using it (the reader). E3 asks which happened.

/// admonition | Open decision
The fallback term of [ex-2.2.2](/docs/m2/ex-2.2.2/report.py) detached the edited state, so only the blocks after the edit learned from it. Here the edit acts at every slice, so a detach would have to stop the gradient at each slice in turn, and the term could then train only each block on its edited input. That is closer to "train the reader" in a narrow sense, but it leaves the writer free to keep the off-axis part, which is the part the edit misses. The plan lets gradients through. To check: if E3 finds the off-axis evidence gone in the new runs, the term worked on the writer, and a detached arm would say whether a reader-only term does as well.
///

The weight comes from a scout at seed 712, trained to epoch 64 in `edit-zero` at $\lambda_e$ of 0.01, 0.05, and 0.25 (the bracket of ex-2.2.2 around its 0.05, also a cross-entropy). The rule, fixed before the scout: the largest weight whose validation loss, averaged over epochs 24 to 64, is no more than 0.014 above that of the stored fixed-labels run at that seed over the same epochs (the largest gap between twins in a 40-epoch mean of validation loss, in low-runs; at a single record the twins part by up to 0.02), and whose run is called kept. If no weight passes, the plan stops there. Both forms then use that weight. The scout runs are not scored with the rest.
<!-- REVIEW: the rule compared validation loss at the single record at epoch 64 against 0.014, which low-runs derived
from 40-epoch means. On the stored trajectories (records every 4 epochs) the twin pair at seed 719 parts by 0.018 at
epoch 64 and by up to 0.019 before it, so the single-record rule would have rejected a run that is no further from
its reference than a twin is. Over epochs 24 to 64 the twin means differ by at most 0.010 (seed 719), under the
threshold. Verify: val_loss in fixed-labels' finished and cross trajectories, fixed-labels-s{seed} against
fixed-labels-cross-d{seed}-s{seed} at seeds 712, 715, 718, 719. -->


Removal compares as two samples, the difference of seed means with a 95% Welch t interval, and the count of low runs with Fisher's exact test. At eight runs against the 24 stored ones, a new condition as tight as the per-visit runs would sit about 0.07 above `fixed`, against an interval half-width of about 0.045, so a gap of that size can be seen. The low-run count is weaker: zero low runs in eight against seven in 24 gives a one-sided p of about 0.10, so a low count alone could not tell the term from chance. The stored runs trained on two card models (the fixed-labels runs on L4s, the crossed runs on L40S cards). The report gives a table of runs per card per condition, and checks any close comparison on one card alone.

Removal on the held-out contexts is close to what the term optimizes, so a rise there is the expected direction rather than a test. It is the manipulation check. The parts the term does not touch by construction are the cost on the other ops and on the task (E2), where the evidence sits afterwards (E3), and the response at doses the term never trained (E4).

## The measurements

Every edit is the e₁ edit, scored on the 14,000 held-out contexts of [ex-2.2.21](/docs/m2/ex-2.2.21/report.py), with the scorer and baseline of the fixed confirmation.

Removal
:   The drop on `difference` under the full edit, as a share of the way from the clean score to the target null, less the median drop under the same edit along each of the other 63 axes. Higher is better. A run below 0.8 is low.

Edit loss
:   The term itself: the cross-entropy of the edited answer against its target, on the labelled training contexts, recorded along training. Lower means the edited model answers as the term asks.

Spill
:   The largest drop on any other op under the edit, at any dose, net of the same baseline. The selectivity criterion is 0.02.

Task change
:   The clean score per op, less the mean of the controls. The gate of the D2.2 design fails below −0.02 on the mean over ops, or below −0.05 on any one op.

Dose response
:   Removal at each dose of the scorer (a quarter to all of the e₁ component), compared with the per-visit runs, which no term trained at any dose.

Off-axis evidence
:   Whether an example fits `difference`, decoded by a linear probe from the state at its answer with e₁ zeroed, scored by AUC at slice 3, as in low-runs. 0.5 is chance and 1 is perfect.

Single-slice edit
:   The share of the way removed by the e₁ edit at one slice alone, with no baseline. It locates the reader: at slice 2 for a reader at block 3, at slice 3 for one at block 4.
"""

# %%

r"""
## The low tail (E1)

Does the term remove the low runs, and does what it learns on fewer than a thousand labelled contexts carry over to contexts it never saw? If the reader comes to depend on e₁, removal on the held-out contexts should rise toward the per-visit runs and the spread should shrink. If the term only fits the labelled set, the edit loss would fall while removal on the held-out contexts stays where `fixed` is.

<!-- Figure: removal at epoch 400 by condition (fixed, edit-zero, edit-dose, with the per-visit runs for scale), one dot per run, low runs marked; beside it, edit loss on the labelled contexts against removal on the held-out ones, one mark per run. Table: removal (mean and range), low count, Welch interval against fixed, Fisher p. -->

TODO.
"""

# %%

r"""
## What it costs (E2)

What does the term cost the other ops and the task? Spill is outside what the term trains, since it scores only `difference` contexts. A term that taught the model to answer the null whenever e₁ is low would show up here, because the other ops already sit at low e₁. The task change is the other cost: on a labelled context the term and the task loss ask for different answers until e₁ holds the op.

<!-- Figure: spill and task change at epoch 400 by condition, as in E2 of fixed-labels; task change along training for the first 100 epochs. Table: counts over the criterion and the gate. -->

TODO.
"""

# %%

r"""
## Writer or reader (E3)

Where did the change happen? If the term worked on the writer, the off-axis evidence should be weaker in the new runs, and the probe with e₁ zeroed would fall from its level of about 0.99 in every stored run. If it worked on the reader, the evidence would still be there off the axis, but the reader would no longer use it: the single-slice edit at the slice before the reader would remove about as much as the full edit, and the two shapes of low run in low-runs (a reader at block 4, or block 3 taking part of the evidence off the axis) would be absent.

<!-- Figure: off-axis evidence probe AUC by condition; single-slice edits at slices 1 to 4 by condition, one mark per run, as in E3 of low-runs. -->

TODO.
"""

# %%

r"""
## Graded, or a cue (E4)

Does the edited model respond to how much e₁ is removed, or only to its being gone? The full-dose term trains one state, and the model could learn to detect it. That would show as removal near zero at the lower doses and a jump at the full dose, where the per-visit runs climb steadily with the dose. The random-dose term trains every dose, so the comparison between the two forms says whether a cue was there to learn.

<!-- Figure: removal against dose, seed mean and range per condition, with the per-visit runs. -->

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

[low-runs](/docs/m2/low-runs/report.py)
:   The evidence that an example fits `difference` lies partly on e₁ and partly along one direction off it in every run, and the low runs are the ones whose readers take more of it from off the axis, which training does not see.

[fixed-labels](/docs/m2/fixed-labels/report.py)
:   With one label draw held for the run, the anchor settles at every rate, but a few runs in eight end well below the rest in removal, and which runs do so moves when a run is trained again with slightly different arithmetic.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed`, with the per-visit label draw, keeps the anchor on all eight runs at seeds 712 to 719 with removal near 0.9, and is the recipe of record for D2.2.

[ex-2.2.2](/docs/m2/ex-2.2.2/report.py)
:   A term that trained the model on its answer under a reflection of the anchor axis gave the designed answer under that edit on every seed, and carried over only partly to the plain projection.

[ex-2.2.21](/docs/m2/ex-2.2.21/report.py)
:   The in-context grammar with three examples and its held-out set of 14,000 contexts, 2,000 per op.
"""
