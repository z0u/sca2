import marimo

__generated_with = "0.23.15"
app = marimo.App(
    app_title="Ex 2.1.6: anchoring red in a transformer",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.6: anchoring *red* in a transformer

    This is the first anchored transformer. During training we add one
    regularizer term that pulls the residual stream activations of *red-labeled*
    <!-- In the previous experiment, we used $a$ to mean "operand 1". Let's pick something consistent; updating the terms used in old experiments is acceptable. -->
    equations toward a fixed direction $a$. Then we ask two questions: did the
    concept land where we put it, and what did that cost the task?

    As the first anchored experiment, it opens D2.1 proper. M1 established
    anchoring in autoencoders, and everything since M2/Ex-2.1.1 has been
    un-anchored groundwork: finding a testbed where the task is solved and the
    concept geometry is legible, so that the effects of an anchor can be
    attributed to it.

    The testbed is the word-level `v216` cell from Ex-2.1.3: 216 grid colors,
    **one token** each, `name + name = name` equations, d64-L4. Held-out
    baseline accuracy is ≈ 1.0 and the color cube is linearly decodable from the
    embeddings, so there is headroom to lose and a known un-anchored geometry to
    compare against.

    The labels are sparse, noisy, and binary, following M1/Ex-1.7 via
    M1/Ex-2.9.1. Each line is labeled *red* with probability
    <!-- We should probably visualize this distribution, as a line chart and on the color cube (using marker size, perhaps). Let's do that before running the experiment. -->
    `redness(op1)⁸ × 0.08`. Only strongly red first operands are ever labeled,
    and even they usually aren't.

    Samples were atomic in the autoencoder experiments, but now that each sample
    is a sequence, we need to decide _what_ to label. For this experiment,
    labels will be attached to sequences, and the anchor term applies uniformly
    to every position and layer of a labeled line. That matches the shape of
    natural language, where a document is labeled rather than a token. It also
    leaves an interesting question for the training dynamics to answer. The pull
    is undifferentiated, but the task loss resists it unevenly. So does the
    concept condense at the position that computes it, or smear into a broadcast
    "this line mentions red" feature?

    The schedule comes from [issue #10] and M1/Ex-2.9.3: regularizer protection
    has to outlive the heat of the optimizer. So the *anchor* weight holds at peak
    until the LR decay is essentially done, then anneals to a floor above zero.

    The architecture helps too. Our simplified nGPT keeps every residual-stream
    state on the unit hypersphere, so the cosine-geometry anchor term from M1
    <!-- β-calibration needs a definition -->
    transfers unmodified, and there is no β-calibration problem during training
    because norms are pinned at 1.

    Several things are left out: the *anti-anchor*, *anti-subspace*, and
    *separate* terms, and any intervention (suppression, ablation, or redirect).
    One anchor term only. We measure whether the axis is reserved for *red*
    rather than enforcing it, and defer intervention to a later experiment,
    <!-- what's this trying to say? -->
    where $-a$ would first have a job to do.

    [issue #10]: https://github.com/z0u/sca2/issues/10
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistered skeleton. The method, hypotheses, and decision
    thresholds below were written and frozen before the experiment ran.
    Results will replace the `TODO` placeholders in place, so reviewing it
    reads as a prediction → observation diff. Anything conceived after seeing
    the data goes under *Exploratory analyses*, marked post hoc.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    ### Task and corpus

    Identical to the `v216` cell of ex-2.1.3: the six-level sub-grid `(0, 3, 6, 9,
    12, 15)` of the 16-level cube (216 colors, one opaque token each),
    equations `name + name = name` with exact integer mixing, 100 k lines,
    20 % of distinct closed pairs held out. The eval sets are the same three:
    `named_seen`, `named_holdout`, and `open` (pairs whose mix has no name).

    We add one thing: an alignment probe set of all 216 colors as operand 1,
    each mixed with 8 random second operands. That way per-color alignment
    statistics average over op2 rather than conditioning on it.

    ### Labels

    <!-- Bernoulli draw needs definition -->
    A line is labeled *red* by a Bernoulli draw with
    `p = redness(op1)⁸ × 0.08`, where `redness = r·(1 − g/2 − b/2)`. This is
    the `RED_PROB` of M1, applied to the color of the first operand only. A
    red op2 or a red answer never triggers a label. That keeps the ground
    truth unambiguous for the question of *where* the concept should condense.
    An either-slot labeling scheme is queued as a follow-up.

    Labels are redrawn per visit, as in M1, i.e. the labels will differ from epoch to epoch. So the expected pull on a color is
    proportional to its label affinity, and that is where we expect the graded
    response of H2 to come from.

    <!-- Why the space before the % symbols? I thought SI had an exception for this -->
    Under a plain Bernoulli draw only ≈ 0.1 % of lines would be labeled, and
    ~93 % of batches would carry none at all. So batches are label-balanced,
    as issue #10 describes. Each batch of 64 includes 4 lines drawn with
    probability proportional to their label affinity and treated as labeled,
    and the anchor term is normalized by the labeled count (the
    label-affinity-weighted mean from M1). This reproduces the Bernoulli
    expectation while fixing the per-batch count, so the gradient variance of
    the term does not inherit the label sparsity.

    /// details | A more realistic labeling variant
    Labels could instead be drawn once at corpus build, making them a property
    of a line, much like labeled internet text. That variant is queued beside
    either-slot labeling. Both replace expected pull with whatever the model
    can infer from sparse fixed evidence, so both would be easier to read once
    the baseline mechanism is understood.
    ///

    ### Model and anchor term

    <!-- e_0 needs definition -->
    The d64-L4 simplified nGPT as used from ex-2.1.1 onward, unchanged. The
    anchor direction is $a = e_0$, matching M1. The residual operations of
    this architecture (LERP toward normalized sub-module output, scalar gains)
    are rotation-equivariant, so choosing a basis vector is a convenience
    rather than a hint to the model.

    For labeled lines, the anchor term is

    $$\mathcal{L}_{\text{anchor}} = \frac{1}{|\mathcal{P}|\,(L{+}1)}
    \sum_{t \in \mathcal{P}} \sum_{\ell=0}^{L}
    \bigl(1 - \cos(h_{\ell,t},\, a)\bigr)$$

    <!-- Is this \ell a layer? IIRC we used it to mean "label" in M1; please check for consistency. -->

    That is the mean over every non-padding position $\mathcal{P}$ of the line
    and all $L{+}1 = 5$ residual-stream slices (the embedding plus four block
    outputs). Unlabeled lines contribute nothing, and no other term is added.

    ### Schedule and arms

    <!-- it would be nice to visualize the schedule; if the LR schedule is not done with a dopesheet we should materialize both and plot them together. -->

    The LR schedule is the one from ex-2.1.3: 100 epochs, 10 warmup, cosine to
    1 %. The anchor weight ramps to its peak alongside the LR warmup, holds at
    peak through epoch 90 (by which point the LR has decayed to a few percent
    of peak), then anneals linearly to a 10 % floor at epoch 100. It never
    reaches zero, which is the lesson of ex-2.9.3 applied as issue #10
    prescribes.

    <!-- consistency check against M1 (see .tex source): w or \omega? -->

    The sweep arms are peak anchor weight $w \in \{0,\ 0.03,\ 0.1,\ 0.3\}$ crossed
    with seeds $\{0, 1, 2\}$, so 12 cells. The $w{=}0$ arm is the
    in-experiment control: same corpus, labels ignored. We score every
    <!-- I wonder if we need to specify this in advance? Our hypothesis is that "anchoring works"; should it resolve as "unsupported" just because we didn't predict the value of this coefficient? (honest question; I'm unsure) -->
    hypothesis on the middle rung $w{=}0.1$, prespecified here so the ladder
    can't be cherry-picked afterward. The other rungs chart the
    dose–response.

    ### Measurements

    <!-- define NLL; and assume the reader won't navigate to sca.baselines to look -->

    Behavioral: exact match, NLL, and the distance metrics of ex-2.1.3 on all
    eval sets, with the usual nulls from `sca.baselines`.

    Alignment maps: $\cos(h_{\ell,t}, a)$ on the alignment probe set, per
    layer × position. We group lines by the similarity of op1 to red, using
    <!--
    - I don't remember that grouping from M1; maybe it was introduced in the M1 _continuation_ in this project (M2). And I think we need a brief definition of SIM3 (is that the right symbol?). Any why the gap between 0.01 and 0.5?
    - Do we need to group them at all, or can we set a pass threshold on some scalar?
    -->
    the grouping from M1 (`SIM3 > 0.5` is red, `< 0.01` is other). The
    contrast $C(\ell, t)$ is then the red-group mean minus the other-group
    mean.

    <!-- can you please explain this equation? And why did we pick the symbol A? -->
    The headline statistic is the layer-mean contrast at the position of op1,
    $A = \tfrac{1}{L+1}\sum_\ell C(\ell, \text{op1})$. We average over
    anchored sites because the anchor applies to all of them equally, so there
    is no site selection to make. Alongside it we report the min and max over
    layers as unscored diagnostics. A wide min–max gap tells us which depths
    resist the anchor.

    Grading: for each op1 color, the layer-mean $\cos(h, a)$ at the position
    of op1, plotted against the redness of that color.

    Leakage (secondary, no threshold): a ridge probe for redness, fit on the
    residual stream at op1 with the $a$-component projected out. High off-axis
    R² means red is also encoded elsewhere; we expect this to be low.

    Trajectories: every ~50 steps we record $A$ and validation loss, along
    with the LR and anchor-weight schedules. This is the apparatus from
    ex-2.9.3, carried over as the first queue item of issue #10. A flat loss
    curve does not mean learning is done, so the alignment trajectory is the
    evidence to consult before any decision to shorten the schedule.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Scored on the $w{=}0.1$ arm (seed-averaged) unless stated; the other
    rungs are context.

    **H1.** Anchoring does not harm the task. Every anchored arm has
    `named_holdout` exact match within 0.02 (absolute) of the seed mean of the
    $w{=}0$ control. Partial: the gate passes at $w \le 0.1$ but fails at
    $0.3$.

    <!-- define Spearman rho -->
    <!-- (b) is hard to understand -->

    **H2.** The concept lands on the anchor, and does so in a graded way.
    (a) $A \ge 0.5$, using the layer mean. (b) Spearman
    $\rho \ge 0.8$ between op1 redness and the per-color layer-mean
    $\cos(h, a)$ at op1, over the 216 colors. (c) The control arm shows
    $|A| \le 0.1$. Partial: (a) and (c) hold, but the response is a step
    rather than a grade, with ρ high only because red separates from non-red
    and no ordering within the reds.

    **H3.** The concept condenses at the position that carries it. The
    layer-mean $C(\cdot, \text{op1})$ is at least 2× the layer-mean contrast
    at each of `+`, op2, and `=`. We exclude the answer position from this
    comparison, because a red op1 drags the mix toward red and confounds the
    contrast there by construction.

    We also anticipate a specific alternative outcome: broadcast, where all
    positions of a labeled line move toward $a$ roughly equally. That would
    fail H3 while H2 passes. It would tell us something about sequence-level
    labeling rather than about anchoring itself, and a logsumexp
    position-pooling variant is the queued response.

    **H4.** The late-instability mechanism of ex-2.9.3 does not reappear under
    the anneal-to-floor schedule. For every anchored run, the end-of-training
    $A$ is at least 0.8× its running maximum over training. Partial:
    violations confined to the $w{=}0.3$ arm.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost (H1)

    /// admonition | TODO
    Table: exact match by arm ($w$ rows) × eval set (`named_seen`,
    `named_holdout`, `open` distance), seed mean ± range. The $w{=}0$ control
    row goes on top, with the published `v216` numbers from ex-2.1.3 quoted
    beside it as an external reference. Include the `sca.baselines` nulls for
    the `open` distance row.

    Expected: a flat column, within 0.02 of control at every rung. Contrary: a
    monotone drop with $w$, which we would read together with the
    dose–response of H2 to see what alignment level the lost accuracy bought.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Alignment and grading (H2)

    /// admonition | TODO
    Two figures.

    <!-- I think 1. might need to be a scatter plot with a trend line, or a windowed average over redness? Otherwise this could be a very noisy chart. -->

    1. Grading curve: per-color layer-mean $\cos(h, a)$ at the position of op1
    against op1 redness. One point per color, color-swatch marks per the
    figure conventions, control arm greyed underneath, one panel per rung,
    Spearman ρ annotated. Expected: monotone and graded, with the labeled-red
    corner near $\cos = 1$ and the far side of the cube near the control
    baseline. Contrary: flat, meaning the anchor didn't take; or a binary step
    at the label threshold, meaning it took but without grading.

    <!-- Suggestion: a stacked smooth_step plot for this like in 2.1.5. Have a look and see if you think it's suitable. Or if we were going to use layer depth as the x-axis, maybe a basic line chart is more appropriate. -->

    2. Per-layer decomposition of $A$: contrast by layer at op1, all rungs,
    control band behind. Expected: contrast at every depth. A sag in the last
    layer would suggest next-token pressure competing at op1, though the
    target at op1 is always `+`, so there should be slack to spare.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Condensation vs broadcast (H3)

    /// admonition | TODO
    <!-- stacked smooth_step plot, with the mean shaded under the line and the min and max as hairlines. Gives heatmap info plus horizontal comparability. -->
    Heat map: contrast $C(\ell, t)$, layers × the six line positions (op1,
    `+`, op2, `=`, answer, newline), primary arm, seed mean. Annotate the
    layer-mean row with the 2× threshold verdict against `+`, op2, and `=`.

    Expected: a dominant op1 column. Contrary: a uniformly warm map, meaning
    broadcast. The answer column is shown but excluded from scoring, since a
    red op1 gives a reddish mix. Note in the caption whatever it does show,
    since that previews the either-slot follow-up.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training stability (H4)

    /// admonition | TODO
    Trajectory figure: $A$ vs step for every anchored run (rows are rungs,
    lines are seeds), with the LR and anchor-weight schedules as background
    bands, and the running-max ratio reported for each run.

    Expected: a rise early (either during or shortly after LR warmup), a hold through the plateau, and no sag
    after the anneal below 0.8× the running max. Contrary: the signature from
    ex-2.9.3, where alignment forms early and then collapses, either while the
    LR is still hot or once the anneal begins. If collapses appear, note when
    they happen relative to both schedules. That timing was the useful
    diagnostic in M1.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Secondary measurements

    /// admonition | TODO
    Leakage: off-axis redness probe R² at op1, per layer, primary arm against
    control, reported without a threshold. Low values would mean the anchor
    concentrated redness onto $a$. Values matching the control would mean it
    added an aligned copy while leaving the natural encoding in place. Either
    reading is useful for designing an intervention experiment later, since
    this number is effectively the power analysis for one.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Reserved for post-hoc work. Anything here was conceived after seeing the
    data and is marked as such. A few candidates we can anticipate now:
    per-seed variability in condensation, the behavior of the answer-position
    column, alignment on `open`-pair lines, and whether over-representing
    labeled lines through balanced batching leaves a trace in the task loss
    on those lines.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    Written after the analysis round, with the human. Questions to address:
    which of H1–H4 cleared; what the dose–response says about where the
    capacity cost begins; whether we saw condensation or broadcast, and what
    that implies for sequence-level labeling in natural language; and what the
    leakage number says about how to design an intervention experiment, in
    particular whether an anti-subspace term is needed.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    /// admonition | Open decisions for review (pre-freeze)
    These are defaults chosen for drafting, so flag disagreement before the
    freeze.

    1. H2 scores the layer-mean alignment, with min and max reported as
    diagnostics only.
    2. Labels are redrawn per visit, as in M1, via balanced batching at 4
    affinity-sampled lines per batch of 64.
    3. The weight ladder is $\{0, 0.03, 0.1, 0.3\}$, with $w{=}0.1$ as the
    prespecified scoring rung.
    4. Three seeds.
    5. The grading groups in H2 reuse the `SIM3` red/other split from M1.
    6. The anneal window is epochs 90–100, down to a 10 % floor. <!-- I actually think we could start annealing much earlier, like from epoch 20 or 50. Starting the anneal late might increase the chance of leakage due to the persistent pull toward $a$. I guess we could add this as a sweep arm if the initial sweep fails with high leakage? -->
    7. The anchor term averages over all non-padding positions, including the
    answer and newline.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
