import marimo

__generated_with = "0.23.15"
app = marimo.App(
    app_title="Ex 2.1.6: anchoring red in a transformer",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo puts the notebook directory on sys.path, so the design constants come
    # from the experiment module rather than a copy of them in the prose.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.vis import light_dark, themed
    from sca import vis as sv

    use_publisher(report_bundle(__file__))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.6: anchoring *red* in a transformer

    This is the first anchored transformer. During training we add one
    regularizer term that pulls the residual stream activations of *red-labeled*
    equations toward a fixed direction $\hat v_{\text{red}}$. Then we ask two
    questions: did the concept land where we put it, and what did that cost the
    task?

    /// details | Notation
    Equations are `a + b = mix(a, b)` throughout M2, so $a$ and $b$ stay
    reserved for the operands, as in ex-2.1.3 and ex-2.1.4. The anchor
    direction takes M1's symbol for it, $\hat{v}_c$ for concept $c$, written
    here as $\hat v_{\text{red}}$ since *red* is the only concept in play. The
    regularizer weight is $\lambda$, also from M1. One symbol does change
    meaning: M1 wrote the label indicator $\ell^{(i)}_{c}$, but M2 is
    transformer work and layer subscripts are everywhere, so from here $\ell$
    indexes layers, and a label indicator — should a later experiment need to
    write one, e.g. for multiple concepts or per-token labels — is
    $y^{(i)}_{c}$.

    For the sweep, ex-2.1.5's three-tiered vocabulary carries over. A
    **condition** is one setting of the swept hyperparameter; a **cell** is one
    run, a condition crossed with a seed, which is what `metrics["cells"]`
    holds; an **arm** is a named group of conditions with a purpose of its own.
    Since $\lambda$ here is a ladder, its conditions are also called **rungs**.
    ///

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
    transfers unmodified. In an unnormalized residual stream a cosine term
    constrains direction and leaves magnitude free, so it would need a
    companion term with its own weight to keep activation scale in hand — one
    more coefficient to tune. Norms pinned at 1 remove that question.

    Several things are left out: the *anti-anchor*, *anti-subspace*, and
    *separate* terms, and any intervention (suppression, ablation, or redirect).
    One anchor term only. So this experiment asks whether the anchor axis
    *happens* to carry red alone, having done nothing to keep other concepts
    off it — the repulsive terms are what would reserve it, and they earn their
    place once we intervene and a stray concept sharing the axis would show up
    as a side-effect.

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
    20% of distinct closed pairs held out. The eval sets are the same three:
    `named_seen`, `named_holdout`, and `open` (pairs whose mix has no name).

    We add one thing: an alignment probe set of all 216 colors as operand 1,
    each mixed with 8 random second operands. That way per-color alignment
    statistics average over op2 rather than conditioning on it.

    ### Labels

    A line is labeled *red* by a coin flip weighted by the redness of its first
    operand: `p = redness(op1)⁸ × 0.08`, where `redness = r·(1 − g/2 − b/2)`
    (a Bernoulli draw, i.e. one biased coin flip per line, independent of every
    other line). This is the `RED_PROB` of M1, applied to the color of the
    first operand only. A red op2 or a red answer never triggers a label. That keeps the ground
    truth unambiguous for the question of *where* the concept should condense.
    An either-slot labeling scheme is queued as a follow-up.

    Labels are redrawn per visit, as in M1, i.e. the labels will differ from epoch to epoch. So the expected pull on a color is
    proportional to its label affinity, and that is where we expect the graded
    response of H2 to come from.

    Under a plain Bernoulli draw only ≈ 0.1% of lines would be labeled, and
    ~93% of batches would carry none at all. So batches are label-balanced,
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
    """)
    return


@app.cell(hide_code=True)
def _():
    # Mark area reads the share of labels, so the floor is what keeps a color that is
    # never labeled on the panel at all.
    _dia = np.maximum(sv.grid_diameter(6) * np.sqrt(ex.LABEL_W / ex.LABEL_W.max()), 0.035)
    _by_redness = np.argsort(ex.REDNESS)
    # Closed with a final (x, 1.0) so the last step — pure red, a third of the mass on its own —
    # is drawn rather than trailing off the right edge.
    _cum_x = np.append(ex.REDNESS[_by_redness], 1.03)
    _cum_y = np.append(np.cumsum(ex.LABEL_W[_by_redness]), 1.0)

    @themed(
        name="labels",
        alt_text="""
            The labels concentrate hard on the red corner: on the color cube only a
            handful of marks near pure red are large, and label probability climbs
            four orders of magnitude between redness 0.3 and 1.
        """,
        caption="""
            The label distribution over the 216 colors. Left: the cube with mark
            area proportional to a color's share of all labeled lines, floored so
            that never-labeled colors still show. Right: per-color label
            probability against redness (log scale, the eighth-power curve
            behind it), with the dashed line at the corpus mean and the grey
            area giving the cumulative share of labels below each redness.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
        sv.plot_rgb_cube(axes[0], ex.GRID_RGB, diameter=_dia)
        axes[0].set_title("share of labels (mark area)")

        _grey = light_dark("#888", "#888")
        ax = axes[1]
        _x = np.linspace(0, 1, 400)
        ax.plot(_x, ex.RED_RATE * _x**8, color=_grey, lw=1, zorder=1)
        ax.axhline(ex.LABEL_P.mean(), ls=(0, (4, 3)), lw=1, color=_grey, zorder=1)
        ax.scatter(
            ex.REDNESS,
            np.maximum(ex.LABEL_P, 1e-9),  # keep the exact zeros off the log axis
            c=ex.GRID_RGB,
            s=26,
            edgecolors=light_dark("#00000033", "#ffffff55"),
            lw=0.5,
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 0.3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("redness of op1")
        ax.set_ylabel("P(labeled)")
        ax.set_title("per-color label probability")
        ax.annotate(f"{int((ex.LABEL_P < 1e-6).sum())} colors below", (0.02, 1.4e-6), fontsize=7, color=_grey)
        ax.annotate("corpus mean", (1.0, ex.LABEL_P.mean() * 1.3), fontsize=7, ha="right", color=_grey)

        cum = ax.twinx()
        cum.fill_between(_cum_x, _cum_y, step="post", lw=0, color=light_dark("#0000000f", "#ffffff0f"), zorder=0)
        cum.set_ylim(0, 1.02)
        cum.set_xlim(-0.03, 1.03)
        cum.set_ylabel("cumulative share", color=_grey)
        cum.tick_params(colors=_grey)
        fig.suptitle("Label distribution over the 216 colors")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    The eighth power is what makes this a *concept* label rather than a redness
    readout. Pure red alone takes {ex.LABEL_W.max():.0%} of all labeled lines,
    the ten reddest colors take {np.sort(ex.LABEL_W)[-10:].sum():.0%} between
    them, and {int((ex.LABEL_P < 1e-6).sum())} of the 216 colors are labeled
    less than once in a million lines. Averaged over the cube the rate is
    {ex.LABEL_P.mean():.2%} of lines, which is where the empty-batch problem
    above comes from.

    The right panel is the shape H2's grading prediction rests on. The pull a
    color feels is graded over four orders of magnitude, so if alignment tracks
    label affinity we should see a curve rather than a step — but the model
    only ever sees Bernoulli draws from this, and nothing stops it from
    treating "red enough to be labeled sometimes" as a threshold.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model and anchor term

    The d64-L4 simplified nGPT as used from ex-2.1.1 onward, unchanged. The
    anchor direction is $\hat v_{\text{red}} = e_0$, the first basis vector of
    the 64-dimensional residual stream — the vector $(1, 0, 0, \ldots)$,
    matching M1. The residual operations of this architecture (LERP toward
    normalized sub-module output, scalar gains) are rotation-equivariant, so
    choosing a basis vector is a convenience rather than a hint to the model.

    For labeled lines, the anchor term is

    $$\mathcal{L}_{\text{anchor}} = \frac{1}{|\mathcal{P}|\,(L{+}1)}
    \sum_{t \in \mathcal{P}} \sum_{\ell=0}^{L}
    \bigl(1 - \cos(h_{\ell,t},\, \hat v_{\text{red}})\bigr)$$

    where $h_{\ell,t}$ is the residual-stream state at layer $\ell$, position
    $t$. That is the mean over every non-padding position $\mathcal{P}$ of the
    line and all $L{+}1 = 5$ residual-stream slices (the embedding plus four
    block outputs). Unlabeled lines contribute nothing, and no other term is
    added.

    ### Schedule and conditions

    The LR schedule is the one from ex-2.1.3: 100 epochs, 10 warmup, cosine to
    1%. The anchor weight ramps to its peak alongside the LR warmup, holds at
    peak through epoch 90 (by which point the LR has decayed to a few percent
    of peak), then anneals linearly to a 10% floor at epoch 100. It never
    reaches zero, which is the lesson of ex-2.9.3 applied as issue #10
    prescribes.

    The sweep conditions are peak anchor weight
    $\lambda \in \{0,\ 0.03,\ 0.1,\ 0.3\}$ crossed with seeds $\{0, 1, 2\}$, so
    12 cells. ($\lambda$ is M1's symbol for a regularizer weight, $\Omega$
    being the regularizer itself.) The $\lambda{=}0$ condition is the
    in-experiment control: same corpus, labels ignored. The other rungs chart
    the dose–response.

    One **star arm** joins them: $\lambda{=}0.1^{*}$, identical except that the
    anneal starts at epoch 50 rather than 90. Star arm is our shorthand, not a
    term of art — it borrows the *star point* of a central composite design,
    where one factor moves off the design centre while everything else holds.
    It varies one thing off the scoring rung instead of crossing the whole
    grid, so it costs 3 cells rather than 12 and answers a single question.
    Here the question is whether
    a long hold at peak buys alignment or just leaks: the anchor keeps pulling
    every labeled line for 90 epochs, and leakage is the secondary measurement
    we most expect to be sensitive to that. Against it stands the lesson of
    ex-2.9.3, that withdrawing protection while the optimizer is still hot lets
    the task loss reclaim the axis, and epoch 50 puts this condition much
    closer to that edge than the default does — see the figure below. Annealing
    to a floor rather than to zero is what makes it worth trying at all. It is
    a diagnostic, not a scored condition; if it shows lower leakage at equal
    alignment, the anneal window becomes a design question for the next
    experiment.
    """)
    return


@app.cell(hide_code=True)
def _():
    _epochs = np.linspace(0, ex.SCHEDULER.epochs, 2001)
    _lr = ex.learning_rate(_epochs) / ex.PEAK_LR
    _weight = ex.anchor_weight(_epochs)
    _star = ex.anchor_weight(_epochs, anneal_start=ex.STAR_ANNEAL_START)

    @themed(
        name="schedules",
        alt_text="""
            The anchor weight holds at peak until epoch 90, by which point the
            learning rate has decayed to 4% of peak; the star arm lets go at
            epoch 50, while the learning rate is still around 60%.
        """,
        caption="""
            The two schedules on one axis, each as a fraction of its own peak —
            the LR peak is fixed at 1e-2, the anchor weight's is the swept
            $\\lambda$. Both anneal windows are 10 epochs long and both stop at
            a floor of 10% rather than zero. Dots mark the learning rate at the
            moment each anneal begins.
        """,
    )
    def _plot() -> plt.Figure:
        _lr_c, _an_c = light_dark("#888", "#999"), light_dark("#c1332a", "#f0665a")
        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        ax.plot(_epochs, _lr, color=_lr_c, lw=1.5, label="learning rate")
        ax.plot(_epochs, _weight, color=_an_c, lw=2.0, label=r"anchor weight $\lambda$")
        ax.plot(_epochs, _star, color=_an_c, lw=1.5, ls=(0, (4, 3)), label=r"$\lambda{=}0.1^{*}$ (star arm)")
        for _start, _text_at in ((ex.STAR_ANNEAL_START, (40, 0.40)), (ex.ANNEAL_START, (82, 0.26))):
            _at = float(ex.learning_rate(_start) / ex.PEAK_LR)
            ax.plot([_start], [_at], "o", ms=4, color=_lr_c)
            ax.annotate(
                f"LR {_at:.0%} of peak\nwhen the anneal starts",
                (_start, _at),
                xytext=_text_at,
                fontsize=7,
                ha="right",
                va="top",
                color=light_dark("#666", "#999"),
                arrowprops=dict(arrowstyle="-", lw=0.6, color=light_dark("#aaa", "#666")),
            )
        ax.set_xlim(0, ex.SCHEDULER.epochs)
        ax.set_ylim(0, 1.28)
        ax.set_xlabel("epoch")
        ax.set_ylabel("fraction of peak")
        ax.set_title("Learning rate and anchor weight over training")
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=3)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    Drawing the two together settles something the prose was vague about, and
    not in the direction we assumed. A cosine decay spends most of its range
    late, so at epoch {ex.STAR_ANNEAL_START:.0f} the LR is still
    {ex.learning_rate(ex.STAR_ANNEAL_START) / ex.PEAK_LR:.0%} of peak — over
    half of it, so "well down" was wrong. By epoch {ex.ANNEAL_START:.0f} it is
    {ex.learning_rate(ex.ANNEAL_START) / ex.PEAK_LR:.0%}. The star arm is
    therefore a sharper test than we intended: it withdraws most of the anchor
    into an optimizer still moving quickly, which is close to the setting that
    produced the late collapses of M1/Ex-2.9.3. That makes it a good probe of
    the mechanism and a poor candidate for a default schedule, and it is why
    the star arm stays a diagnostic.

    It anneals over the same 10-epoch window as the default, reaching the floor
    at epoch {ex.STAR_ANNEAL_START + ex.ANNEAL_EPOCHS:.0f} and holding there.
    The alternative reading — stretch the anneal from epoch
    {ex.STAR_ANNEAL_START:.0f} to the end of training — would move the rate of
    the anneal along with its start, and the point of a star arm is to move one
    thing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Scoring

    Hypotheses resolve on the middle rung $\lambda{=}0.1$, which is where the
    numbers in the results sections come from. But H2's claim is that anchoring
    works, and that claim shouldn't fail because we guessed a coefficient
    wrong, so its gates read *some anchored rung clears the threshold while
    also clearing H1*. H1 is what keeps this from being cherry-picking: a rung
    earns nothing by reaching alignment if it broke the task getting there. The
    ladder is only four rungs, so the multiple-comparisons cost of reading
    across it is small. If the passing rung is not $\lambda{=}0.1$, we say so
    and report both.

    ### Measurements

    Behavioral: exact match, NLL, and the distance metrics of ex-2.1.3 on all
    eval sets, with the usual nulls from `sca.baselines`. NLL is the negative
    log-likelihood the model assigns to the correct answer token — how
    surprised it is by the truth, in nats, with 0 meaning certain and correct.
    Exact match is the coarser scoring of the two: it counts the argmax, so a
    run can hold accuracy while NLL drifts, and that drift is the earlier
    warning of capacity being spent elsewhere. The nulls are the reference
    scores from `sca.baselines`: what the same metric gives for degenerate
    predictors (the corpus-mean color, a uniform draw, the identity of op1),
    which is what makes a distance number readable as good or bad.

    Alignment maps: $\cos(h_{\ell,t}, \hat v_{\text{red}})$ on the alignment
    probe set, per layer × position. We group lines by how red op1 is, reusing
    `SIM3` from `sca.colorcube` — an angular hue similarity to pure red,
    weighted by vibrancy and cubed, so it lands near 1 only for colors that
    read as red to a person and near 0 for everything else. Red is
    `SIM3 > 0.5`, other is `SIM3 < 0.01`.
    The gap between the two drops the ambiguous middle (pinks, oranges, dark
    reds) at the cost of scoring on fewer colors. The contrast $C(\ell, t)$ is
    then the red-group mean minus the other-group mean.

    Grouping earns its place where we need one number per layer × position
    cell, which is what H3's 2× ratio compares and what H4 tracks over
    training. The layer-mean of it at op1 is

    $$\bar{C}_{\text{op1}} = \frac{1}{L+1}\sum_{\ell=0}^{L} C(\ell,
    \text{op1})$$

    Reading it from the inside out: at one layer and one position, take the
    mean $\cos(h, \hat v_{\text{red}})$ over red-op1 lines and subtract the
    same over other-op1 lines, which is $C$; then average that over the five
    residual-stream slices. It is a difference of cosines, so it runs from
    $-2$ to $2$, and 0 means the anchor axis treats red and non-red lines
    alike. We average over anchored sites because the anchor applies to all of
    them equally, so there is no site selection to make. Alongside it we report
    the min and max over layers as unscored diagnostics. A wide min–max gap
    tells us which depths resist the anchor.

    For H2 there is no such constraint — the question is about colors, not
    cells — so we score it group-free, over all 216 colors. Write $\alpha_c$
    for the per-color layer-mean $\cos(h, \hat v_{\text{red}})$ at op1 and
    $u_c \propto \texttt{redness}(c)^8$ for label affinity, normalized to sum
    to 1. Then

    $$\Delta\alpha_{\text{op1}} = \sum_c u_c\,\alpha_c \;-\;
    \frac{1}{216}\sum_c \alpha_c$$

    the affinity-weighted mean alignment minus the plain one: how much closer
    to the anchor a color sits for being the kind of color the anchor pulls.
    The weights are the label probabilities themselves, so this asks about
    precisely the lines that felt the term, with no threshold to place and no
    color discarded. It reduces to the group difference when the weights are
    indicator-shaped and the cube is mostly non-red, so it reads on a
    comparable scale, and 0 still means the axis is indifferent to redness.

    Grading: for each op1 color, the layer-mean
    $\cos(h, \hat v_{\text{red}})$ at the position of op1, plotted against the
    redness of that color.

    Leakage (secondary, no threshold): a ridge probe for redness, fit on the
    residual stream at op1 with the $\hat v_{\text{red}}$-component projected
    out. High off-axis R² means red is also encoded elsewhere; we expect this
    to be low.

    Trajectories: every ~50 steps we record $\bar{C}_{\text{op1}}$ and
    validation loss, along
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

    Reported on the $\lambda{=}0.1$ condition (seed-averaged) unless stated; the
    other rungs are context. H2's gates read *some anchored rung*, per the
    scoring rule above.

    **H1.** Anchoring does not harm the task. Every anchored condition has
    `named_holdout` exact match within 0.02 (absolute) of the seed mean of the
    $\lambda{=}0$ control. Partial: the gate passes at $\lambda \le 0.1$ but
    fails at $0.3$.

    **H2.** The concept lands on the anchor, and does so in a graded way, at
    some rung that also clears H1. (a) $\Delta\alpha_{\text{op1}} \ge 0.5$.
    (b) Redder colors sit closer to the anchor: over the 216 colors, $\alpha_c$
    rises monotonically with the redness of $c$, at Spearman $\rho \ge 0.8$.
    (Spearman ρ is a correlation coefficient computed on ranks rather than
    values, so it measures whether the ordering agrees and doesn't care about
    the shape of the curve.) (c) The control condition shows
    $|\Delta\alpha_{\text{op1}}| \le 0.1$.

    Partial: (a) and (c) hold, but the response turns out to be a step rather
    than a grade — the reds all sit at one alignment and the non-reds at
    another, with no ordering within either. Spearman ρ can clear 0.8 on that
    shape, since two well-separated clumps rank correctly, so the grading
    figure is what distinguishes the two readings.

    **H3.** The concept condenses at the position that carries it. The
    layer-mean $C(\cdot, \text{op1})$ is at least 2× the layer-mean contrast
    at each of `+`, op2, and `=`. We exclude the answer position from this
    comparison, because a red op1 drags the mix toward red and confounds the
    contrast there by construction.

    We also anticipate a specific alternative outcome: broadcast, where all
    positions of a labeled line move toward $\hat v_{\text{red}}$ roughly
    equally. That would
    fail H3 while H2 passes. It would tell us something about sequence-level
    labeling rather than about anchoring itself, and a logsumexp
    position-pooling variant is the queued response.

    **H4.** The late-instability mechanism of ex-2.9.3 does not reappear under
    the anneal-to-floor schedule. For every anchored run, the end-of-training
    $\bar{C}_{\text{op1}}$ is at least 0.8× its running maximum over training.
    Partial: violations confined to the $\lambda{=}0.3$ condition.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost (H1)

    /// admonition | TODO
    Table: exact match by condition ($\lambda$ rows, star arm last) × eval set (`named_seen`,
    `named_holdout`, `open` distance), seed mean ± range. The $\lambda{=}0$ control
    row goes on top, with the published `v216` numbers from ex-2.1.3 quoted
    beside it as an external reference. Include the `sca.baselines` nulls for
    the `open` distance row.

    Expected: a flat column, within 0.02 of control at every rung. Contrary: a
    monotone drop with $\lambda$, which we would read together with the
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

    1. Grading curve: $\alpha_c$ against the redness of $c$. A scatter — one point per color,
    color-swatch marks per the figure conventions — with a windowed mean over
    redness drawn through it, since 216 points with op2 averaged out will
    carry real spread and the eye should be reading the trend rather than the
    scatter. Control condition greyed underneath, one panel per rung, Spearman ρ
    annotated, and $\Delta\alpha_{\text{op1}}$ with it. Expected: monotone and
    graded, with the labeled-red corner near
    $\cos = 1$ and the far side of the cube near the control baseline.
    Contrary: flat, meaning the anchor didn't take; or a binary step at the
    label threshold, meaning it took but without grading.

    2. Per-layer decomposition of $\bar{C}_{\text{op1}}$: contrast by layer at
    op1, all rungs, control band behind. A plain line chart, layer depth on x —
    depths are evenly spaced and every one is measured, so there is nothing for
    the step-plateau convention of ex-2.1.5 to guard against here (it exists
    because grammar landmarks are *not* evenly spaced). Expected: contrast at
    every depth. A sag in the last layer would suggest next-token pressure
    competing at op1, though the target at op1 is always `+`, so there should
    be slack to spare.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Condensation vs broadcast (H3)

    /// admonition | TODO
    Contrast $C(\ell, t)$ over the six line positions (op1, `+`, op2, `=`,
    answer, newline), primary condition, seed mean. Drawn as stacked `smooth_step`
    panels in the manner of ex-2.1.5 rather than a heat map: positions on x,
    one panel per layer, the layer mean shaded under the line and the min and
    max over seeds as hairlines. That keeps what a heat map gives — a column
    of panels reads as a column of cells — and adds a shared vertical scale, so
    the 2× comparison between op1 and its neighbours is something the reader
    can see rather than infer from color. The step-plateau form is right here:
    positions are ordinal and the tokens between landmarks are not measured.
    Annotate the layer-mean panel with the 2× threshold verdict against `+`,
    op2, and `=`.

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
    Trajectory figure: $\bar{C}_{\text{op1}}$ vs step for every anchored run (rows are rungs,
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
    Leakage: off-axis redness probe R² at op1, per layer, primary condition against
    control, reported without a threshold. Low values would mean the anchor
    concentrated redness onto $\hat v_{\text{red}}$. Values matching the
    control would mean it added an aligned copy while leaving the natural
    encoding in place. Either reading is useful for designing an intervention
    experiment later, since this number is effectively the power analysis for
    one.

    The $\lambda{=}0.1^{*}$ star arm goes on the same axes, since the shorter
    hold at peak was proposed as the lower-leakage schedule and this is where
    that would show. Report its $\Delta\alpha_{\text{op1}}$ beside it, because
    less leakage at less alignment is no bargain.
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

    1. H2 scores the group-free $\Delta\alpha_{\text{op1}}$ over all 216
    colors; the `SIM3` group contrast $C$ is kept for H3 and H4, which need one
    number per layer × position cell, with its min and max over layers reported
    as diagnostics only.
    2. Labels are redrawn per visit, as in M1, via balanced batching at 4
    affinity-sampled lines per batch of 64.
    3. The weight ladder is $\{0, 0.03, 0.1, 0.3\}$. Results are reported at
    $\lambda{=}0.1$, but H2 passes on any rung that also clears H1, so the
    claim doesn't rest on having guessed the coefficient.
    4. Three seeds.
    5. The anneal window is epochs 90–100, down to a 10% floor, with the
    $\lambda{=}0.1^{*}$ star arm running the same 10-epoch window from epoch 50
    to test whether the long hold at peak costs leakage. The schedule figure
    shows what that costs: the star arm lets go while the LR is still near 60%
    of peak.
    6. The anchor term averages over all non-padding positions, including the
    answer and newline.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
