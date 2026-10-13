# ruff: noqa: B018
# title: The split anti weight at fresh seeds

# Preregistration skeleton. The arms, seeds and thresholds below are the plan under review; the experiment code
# lands after the freeze.

r"""
# The split anti weight at fresh seeds

/// tip |
<!-- lede -->
We train the candidate recipes from [the slice-terms experiment](/docs/m2/slice-terms/report.py) at eight fresh seeds, to ask whether holding the anti-subspace weight high on the first slices only keeps the op removable and the edit selective. Beside it we try an anti weight that falls off linearly with depth, so that no slice boundary has to be chosen.
///
"""

# %%

r"""
## Findings

- [The split keeps both removal and selectivity (H1)](#the-split-keeps-both-removal-and-selectivity-h1) — **TODO**.
- [A linear falloff does as well as the split (H2)](#a-linear-falloff-does-as-well-as-the-split-h2) — **TODO**.
- [Which slices the edit needs (E1)](#which-slices-the-edit-needs-e1) — TODO.
- [Decision](#decision) — TODO.

/// admonition | How to read this report
This report was preregistered: the predictions and the criteria for the decision were frozen at commit `ff44a17`, before any run of this experiment. Each section opens with what we expect, and the results replaced the placeholders in place.
///
"""

# %%

r"""
## Why this experiment

The pull indirectly asks the states at the example answers of `difference` contexts to align with e₁, and the anti-subspace term asks every other state to stay off it. When the anti weight anneals low, the color embedding table and slice 1 drift onto e₁ on the other ops, and the edit that projects e₁ out takes some of those ops with it: the spill. [The late-pull experiment](/docs/m2/late-pull/report.py) found that a higher hold everywhere keeps the spill within the criterion, at some cost in removal. [The slice-terms experiment](/docs/m2/slice-terms/report.py) traced that cost to the hold countering the pull at the pulled slices, and held the weight high on the embedding and slice 1 only. On its four seeds this gave the removal back and kept the spill within the criterion on three.

Those four seeds are the ones the arms were chosen on, so the result may be partly fitted to them. Before the split becomes the recipe a deliverable rests on, it should hold at fresh seeds, beside the arms it was chosen over.

The split also sets a boundary by hand: high on the first two slices, low after. If a weight that falls off smoothly with depth does as well, the recipe has one less choice in it.
"""

# %%

r"""
## Parameters

Every arm pulls `difference` toward e₁ at slices 2 to 4, pooled over the positions of each labeled context, so that most of the pull lands on the example answers. The anti-subspace term acts on every slice (the `deep` pull of the late-pull experiment). The anchor weight $\lambda_a$ (0.1), the pooling temperature $\tau_p$ (0.1), the label, the corpus and the schedules are the recipe values, at 400 epochs. The anti weight $\lambda_{\bar s}$ starts at 0.25 on every slice and anneals to its *hold* by epoch 360, as in the slice-terms experiment. The arms differ only in that hold:

<figure class="report-figure"><table class="report-table dense"><thead><tr><th>arm</th><th class=num>emb</th><th class=num>1</th><th class=num>2</th><th class=num>3</th><th class=num>4</th></tr></thead><tbody>
<tr><td><code>deep, hold 0.2</code></td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.2</td></tr>
<tr><td><code>deep, hold 0.1</code></td><td class=num>0.1</td><td class=num>0.1</td><td class=num>0.1</td><td class=num>0.1</td><td class=num>0.1</td></tr>
<tr><td><code>split-anti</code></td><td class=num>0.2</td><td class=num>0.2</td><td class=num>0.03</td><td class=num>0.03</td><td class=num>0.03</td></tr>
<tr><td><code>falloff</code></td><td class=num>0.2</td><td class=num>0.16</td><td class=num>0.12</td><td class=num>0.07</td><td class=num>0.03</td></tr>
</tbody></table><figcaption><p><strong>The hold of the anti weight by slice</strong>, for each arm. Slice 0 is the embedding. The <code>falloff</code> holds are a straight line from 0.2 to 0.03, rounded here to two places.</p></figcaption></figure>

`falloff` runs in a straight line between the two holds of `split-anti`, so it matches the split at the embedding and slice 4 and differs between them. `deep, hold 0.1` is the gentler uniform hold, a candidate for the recipe in its own right.

Each arm trains at model seeds 704 to 711, 32 runs in all. Ex-2.2.23 trained 400-epoch controls (no pull, no anti term) at these seeds, so every run has a control with the same initialization, batches and label draws. No arm has been trained at these seeds before.

`deep` at the recipe hold (0.03 everywhere) is left out, since it spilled over the criterion on every seed so far.
"""

# %%

r"""
## Measurements

The measurements and dose scoring are those of the slice-terms experiment.

- Removal: the net drop in the score on `difference` under the edit, as a share of the way from the clean score to the target null. The edit projects e₁ out of the state at every position, on a set of slices, at a dose of a quarter, a half, three quarters, or all of the e₁ component. Higher is better.
- Spill: the largest net drop on another op under the edit. A run is within the criterion when its spill is at most 0.02 at every dose.
- Both are net of the control at the same seed.
- Landing: the share of the distance to the target null that the full edit covers, as total variation on `difference` contexts.
- Task change: the mean score over ops, less that of the control.
- The alignment α with e₁ at the example answers, at every slice, at the end of training, and the slope of the color embedding table (the e₁ component of the color embeddings against lightness).
"""

# %%

r"""
## The split keeps both removal and selectivity (H1)

**What we expect.** On the slice-terms seeds, `split-anti` removed more than `deep, hold 0.2` at every seed and stayed within the criterion on three of four. We expect the same at the fresh seeds. The main number is the removal at the full dose, paired by seed against `deep, hold 0.2`.

A pass needs both removal and spill to hold:

- A pass would be `split-anti` removing more than `deep, hold 0.2` on at least 6 of 8 seeds, by at least 0.05 on average, with at least 7 of 8 runs within the criterion.
- A partial pass would be the removal gain holding with 5 or 6 runs within the criterion.
- A miss would be no removal gain (fewer than 5 seeds ahead, or an average gain under 0.02), or 4 or fewer runs within the criterion.

If the removal gain lands between 0.02 and 0.05 on average, or the seeds split evenly on its direction, the result would be outside the plan, and the verdict would be Unresolved.

/// admonition | TODO
Removal and spill by dose for every arm, one hairline per run and the seed mean bold, with the criterion as a dashed rule. A table beside it: removal, runs within the criterion, largest spill at any dose, landing and task change, as seed mean and range.
///
"""

# %%

r"""
## A linear falloff does as well as the split (H2)

**What we expect.** `falloff` holds slice 1 a little lower than `split-anti` and slices 2 and 3 higher. The lower hold at slice 1 may let a little more spill through, and the higher holds at slices 2 and 3 may counter the pull there and cost a little removal. We expect both effects to be small. The main number is the removal at the full dose, paired by seed against `split-anti`.

As in H1, a pass needs both removal and spill to hold:

- A pass would be `falloff` removing within 0.03 of `split-anti` on average, with at least 7 of 8 runs within the criterion.
- A miss would be `falloff` removing 0.03 or more less than `split-anti` on average, or 6 or fewer runs within the criterion.
- If `falloff` removes 0.03 or more than `split-anti` on average, that would also be outside what we expect, and we would report it as a pass with the gain named.

If the seeds split evenly on the direction of the removal difference and the average is near the 0.03 edge, the verdict would be Unresolved.

/// admonition | TODO
Removal and largest spill against `split-anti` at the same seed, one point per seed, with the line of equality and a band of ±0.03 for removal and the criterion for spill. Beside it, α at the end of training by slice at the example answers, in `difference` contexts and in the contexts of the other ops, for `falloff` and `split-anti`, one hairline per run and the seed mean bold, with the hold of each arm by slice in a shared-x panel underneath.
///
"""

# %%

r"""
## Which slices the edit needs (E1)

On the slice-terms seeds and the late-pull seeds, editing slice 1 alone removed most of the op on every seed, while editing slices 2 to 4 alone removed nearly everything on some seeds and little on others. One reading is that the evidence for the op is carried from the example answers to the `=` sites by attention, in some other direction than e₁, and that the block whose heads make that move reads the example answers as they are at its input. Slice 1 is the input to every block after it, so clearing e₁ there removes the evidence before it moves. Clearing slices 2 to 4 leaves the copy at slice 1 for block 2 to read. This reading is not tested here.

We repeat the edit on slice 1 alone and on slices 2 to 4 alone for every run, to see whether the split by seed holds at the fresh seeds and in each arm.

/// admonition | TODO
Removal under the edit on slice 1 alone and on slices 2 to 4 alone, by arm and seed.
///
"""

# %%

r"""
## Decision

We choose the recipe of record for D2.2 among the four arms, with the human, once the results are in. The criteria, reported for every arm:

- Runs within the criterion, out of 8.
- Removal at the full dose, seed mean and range.
- Landing.
- Task change. This is a hard gate: an arm whose mean task change is below −0.02, or which loses more than 0.05 on any op at any seed against its control, is not a candidate.

These are fresh seeds for every arm, so the chosen arm's numbers here can be quoted as a result without a further confirmation.

/// admonition | TODO
Every criterion for every arm, in a table; what we chose and why.
///
"""

# %%

r"""
## Discussion

TODO, once the results are in.
"""

# %%

r"""
## Method

Training, measurement and dose scoring follow the slice-terms experiment, and its experiment module is reused with new arms and seeds. The anti weight per slice is the `AntiSpec.hold` tuple added there, so `falloff` needs no new code. The controls are ex-2.2.23's 400-epoch runs at seeds 704 to 711.

Cost: 32 runs at about \$0.3 each with their measurements, and the dose scoring of 40 checkpoints, about \$11 in all.
"""
