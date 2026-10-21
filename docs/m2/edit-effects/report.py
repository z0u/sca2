# ruff: noqa: B018
# title: What the edit does on the recipe of record

# An exploratory re-analysis of split-anti-seeds' `falloff` runs at seeds 704 to 711, with no new training.
# `experiment.py` beside this script stores the alignment with e₁ at every slice and position, clean and under the
# full edit; the grading and the landing come from the dose arrays split-anti-seeds published.

r"""
# What the edit does on the recipe of record

/// tip |
<!-- lede -->
TODO: the question (does the edit on the recipe of record do what the placed axis says it should: a bounded turn on the states of other ops, a drop on each context that follows how much it needed `difference`, and answers that land where the target null puts them), and which way it came out.
///

The [D2.2 design doc](/docs/m2/d2.2/design.md) lists three claims that the recipe of record has no measurement for yet: the write bound of claim (d), the grading context by context of claim (b), and where the edited model lands of claim (c). None needs new training. This page measures all three on the eight `falloff` runs of [split-anti-seeds](/docs/m2/split-anti-seeds/report.py), from the dose scoring that experiment stored and one new pass over the held-out contexts that records the alignment with e₁ at every slice and position, clean and under the edit.
"""

# %%

r"""
## Observations

- [How far the edit turns the other ops (E1)](#how-far-the-edit-turns-the-other-ops-e1): TODO.
- [Whether the drop follows what each context needed (E2)](#whether-the-drop-follows-what-each-context-needed-e2): TODO.
- [Where the edited answers fall short (E3)](#where-the-edited-answers-fall-short-e3): TODO.

## Scope

This is an exploratory re-analysis, with no preregistration and no gate. It covers the `falloff` arm of split-anti-seeds (the recipe of record) at model seeds 704 to 711, and the control at each of those seeds from ex-2.2.23, trained without the anchor at the same initialization, batches and label draws. Every measurement is on the 14,000 held-out contexts of ex-2.2.21, 2,000 per op, each three examples and a query.

The edit is the full edit of the recipe: the whole e₁ component taken out of the state at every position and every slice. E2 and E3 use the full dose only; split-anti-seeds already showed that the removal grows with the dose. With eight runs, a pattern seen on every seed is one we can lean on; a difference between seeds is noted and not explained.

## The measurements

- The turn, or *write*: the angle by which the edit turns a state. The states are unit vectors and the edit renormalizes, so the turn is arcsin of the alignment α the state arrives with. At the embedding that is the clean α; at a later slice it is the α left after the blocks have read the edited states of the slices before.
- The *bound* at a slice and position: arcsin of the 99th percentile of the clean |α| over the contexts of the six other ops, as in H4 of [ex-2.2.1](/docs/m2/ex-2.2.1/report.py). It is set before any edit, from the clean geometry. The write stays within it when the 99th-percentile turn under the edit is no larger. Ex-2.2.1 allowed a quarter over (a ratio of 1.25) as a tolerance; we quote the ratio.
- Whether an example *fits* `difference`: its answer is one that `difference` gives on its operands. In a context of another op, a fitting example is usually one where the replacement op noise showed the answer of `difference`.
- *Dependence*: how much a context needed `difference`, as the expected exact match of the Bayes predictor less that of the target null. It comes from the examples and the posterior alone, so it does not depend on the model.
- The *drop* on a context: its expected exact match on the clean pass less that under the edit, net of the drop the same edit causes on the control at the same context.
- The *landing* and the remaining distance: the total variation distance between the edited answers and the target null, on `difference` contexts, and the share of the clean distance that the edit covers.
"""

# %%

r"""
## How far the edit turns the other ops (E1)

The anti-subspace term keeps the states of other ops off e₁, so we expect the edit to turn them only a little, and to turn them least at the pulled blocks, where the anti weight is low but the pull has room. The question is whether the turn under the edit stays within the bound the clean geometry sets, at every slice and role, or whether a later block puts e₁ back onto the other ops after the edit has taken it out upstream.

TODO: figure: the 99th-percentile |α| on other-op contexts by position, one trace per slice, clean (the bound) against arriving under the edit (the write); example answers split by whether they fit `difference`. The control as a ghost.

TODO: table: the largest ratio of write to bound per slice, over all positions and over the positions before the query answer, seed mean and range.

TODO: what we saw and what we make of it.
"""

# %%

r"""
## Whether the drop follows what each context needed (E2)

The target null predicts the drop context by context: a context whose examples all fit `difference` should lose most, and one whose examples point elsewhere should lose little. If the anchor holds the inferred op, the drop under the edit should follow the dependence; if it holds something coarser, such as the label or a color feature of `difference` answers, it would drop by about the same on every `difference` context.

The target null also predicts a change on the contexts of other ops: where an example showed the answer of `difference`, removing the op should move the answer back toward the true op, a gain. We look at both.

TODO: figure: the drop against the dependence, binned, one hairline per run; `difference` contexts and the other ops.

TODO: table: per seed, the correlation of drop with dependence on `difference` contexts, and the share of the dependence removed, by posterior bin.

TODO: what we saw and what we make of it.
"""

# %%

r"""
## Where the edited answers fall short (E3)

The full edit covers about half of the distance to the target null on `difference` contexts. The rest could be on the answers of `difference` the edit leaves behind, on colors no op gives, or on the answers of the other ops, weighed differently from the target null. Each would point somewhere different: a weak removal, damage to the answer, or a model that has lost `difference` and re-weighs what is left in its own way.

A yardstick for what "half" means: even on the clean pass, the model sits some distance from the Bayes predictor it is trying to match. An edited model can hardly be closer to the target null than the clean model is to the predictor it learned from.

TODO: table: the mass on colors only `difference` gives, colors another op gives, colors no op gives, under the target null, the clean pass and the edit; and the remaining distance split the same way.

TODO: figure: the remaining distance against the yardstick, per seed.

TODO: whether the edited answers are a mixture of the other ops and how far its weights per context are from those of the target null.

TODO: what we saw and what we make of it.
"""

# %%

r"""
## Discussion

TODO.
"""
