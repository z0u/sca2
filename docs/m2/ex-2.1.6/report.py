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
    regularizer weight is $\lambda$, also from M1. One symbol changes meaning:
    M1 wrote the label indicator $\ell^{(i)}_{c}$, but layer subscripts are
    everywhere in transformer work, so from here $\ell$ indexes layers. If a
    later experiment needs a label indicator (say, for multiple concepts or
    per-token labels), it is $y^{(i)}_{c}$. New here: $m$ is reserved for the
    alignment margin defined under Measurements, so a counter alongside $n$
    should be spelled some other way.

    For the sweep, the three-tiered vocabulary of ex-2.1.5 carries over. A
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
    is a sequence, we need to decide _what_ to label. Here, labels attach to
    equations, and the anchor term pulls every position of a labeled equation's
    prompt equally, at every layer. That matches the shape of natural language,
    where a span of text is labeled rather than a token. It also leaves a
    question for the training dynamics to answer: the pull is undifferentiated
    but the task loss resists it unevenly, so does the concept condense at the
    position that computes it, or smear into a broadcast "this line mentions
    red" feature?

    The schedule comes from [issue #10] and M1/Ex-2.9.3: regularizer protection
    has to outlive the heat of the optimizer. So the *anchor* weight holds at peak
    until the LR decay is essentially done, then anneals to a floor above zero.

    The architecture helps too. Our simplified nGPT keeps every residual-stream
    state on the unit hypersphere, so the cosine-geometry anchor term from M1
    transfers unmodified. In an unnormalized residual stream a cosine term
    constrains direction and leaves magnitude free, so it would need a
    companion term with its own weight to keep activation scale in hand — one
    more coefficient to tune. Norms pinned at 1 remove that question.

    We leave out the *anti-anchor*, *anti-subspace*, and *separate* terms, and
    any intervention (suppression, ablation, or redirect): one anchor term
    only. Nothing reserves the axis, so this experiment asks whether it
    *happens* to carry red alone. The repulsive terms matter once we intervene,
    when a stray concept sharing the axis would show up as a side-effect.

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
    operand: `p = redness(op1)⁸ × 0.08`, where `redness = r·(1 − g/2 − b/2)`,
    one independent draw per line.[^bern] This is the `RED_PROB` of M1, applied to the
    color of the first operand only. A red op2 or a red answer never triggers a
    label, which keeps the ground truth unambiguous for the question of *where*
    the concept should condense. An either-slot labeling scheme is queued as a
    follow-up.

    Labels are redrawn per visit, as in M1, so they differ from epoch to
    epoch. The expected pull on a color is therefore proportional to its label
    affinity, which is where we expect the graded response of H2 to come from.

    [^bern]: A Bernoulli draw: one biased coin flip per line, independent of
    every other line.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ### Batch composition

    Samples are packed token blocks, not single equations. An equation is
    {ex.LINE_TOKENS} tokens at word level and a block is {ex.BLOCK} tokens,
    so a batch of {ex.BATCH} blocks carries about {ex.LINES_PER_BATCH:.0f}
    equations. The corpus-mean label rate is {ex.LABEL_P.mean():.2%}, which
    works out to {ex.LABELED_PER_BATCH:.2f} labeled lines per batch, with
    {ex.BATCHES_WITH_A_LABEL:.0%} of batches carrying at least one. That is
    denser supervision than M1 ran with, where the term fired on roughly 6%
    of batches.

    At that density we skip the label balancing that issue #10 recommends.
    Labels are plain independent draws, one per line. The anchor term is
    normalized by however many labels a batch happens to hold; that
    normalization is inherited from M1. Both the original implementation and
    our reproduction (`sca.colorcube.loss_terms`) take a label-weighted mean
    with a small ε in the denominator. So the
    {1 - ex.BATCHES_WITH_A_LABEL:.0%} of batches holding no label contribute
    nothing rather than a 0/0. The gradient of the term is therefore noisy
    between batches, but that noise is inherent to sparse labels. If the H4
    trajectories show the term failing to take hold, balancing is still
    available as a variance-reduction step.

    Skipping it also keeps the comparison clean. Balanced batching
    over-represents red first operands, so the anchored conditions would
    train on a different corpus from the control, and every accuracy
    difference in H1 would carry an alternative explanation. As it stands,
    the control and the anchored conditions see identical batches in
    identical order for a given seed; the only difference between them is
    $\lambda$.

    /// details | A more realistic labeling variant
    Labels could instead be drawn once when the corpus is built, making them
    a fixed property of a line, much like labeled internet text. That variant
    is queued beside either-slot labeling. Both replace the expected pull
    with whatever the model can infer from sparse fixed evidence, so both
    would be easier to read once the baseline mechanism is understood.
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

    The right panel is the shape the grading prediction of H2 rests on. The
    pull a color feels is graded over four orders of magnitude, so if alignment
    tracks label affinity we should see a curve rather than a step. But the
    model only ever sees Bernoulli draws from this, and nothing stops it from
    treating "red enough to be labeled sometimes" as a threshold.

    /// admonition | The concentration is a risk we are choosing to accept
    The weights behave like about {ex.EFFECTIVE_COLORS:.0f} distinct colors
    rather than 216, and pure red alone takes {ex.LABEL_W.max():.0%} of the
    labels. So the anchor's evidence is a dozen or so first operands, repeated
    for 100 epochs. An axis could learn *this token* instead of *red* and
    still clear H2(a) and (c).

    We keep M1's exponent anyway. Ex-2.1.6 asks whether the M1 result transfers
    to a transformer, and moving the label distribution at the same time as the
    architecture would leave nothing to attribute a difference to. It is also
    the closer analogue of a document-level label in natural language, which
    fires on a small and unrepresentative slice of the corpus. H2(b), the
    grading test, separates a memorized exemplar from a generalized concept,
    which is a good part of why it is a gate rather than a diagnostic. A
    flatter exponent is queued in `todo-science.md` as the follow-up that
    would settle it.
    ///
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
    $t$, averaged over all $L{+}1 = 5$ residual-stream slices (the embedding
    plus four block outputs) and over $\mathcal{P}$, the positions of the
    labeled equation's **prompt**: `op1`, `+`, `op2`, `=`. Unlabeled lines
    contribute nothing, and no other term is added.

    The answer token and the newline are inside the labeled equation but
    outside $\mathcal{P}$. We exclude them for two reasons, both about
    keeping results readable. First, the answer of a red-labeled line is
    itself reddish, since a red op1 drags the mix. An anchored answer
    position would therefore align for two reasons at once, and we could not
    tell them apart. Second, the answer position is where the task is
    decided. Pulling it directly would mix the capacity cost that H1 is
    trying to measure (what it costs to carry an anchored concept) with the
    much blunter cost of distorting the state that emits the answer.
    Excluding both costs nothing that we are measuring: the four pulled
    positions still receive an identical, undifferentiated pull, so the
    condensation question of H3 stands, and the answer position becomes a
    clean read of how far alignment spreads on its own.

    In a full language model, the analogue of this label would pull the
    whole span between document boundaries, answer-like positions included,
    since nothing there marks a position as safe to exclude. So the
    prompt-only pull is a measurement instrument for this testbed, where the
    answer has a known confound; it is not a claim that the exclusion is
    needed. The unpulled answer position doubles as a first read on what a
    whole-span pull would change, and a whole-span variant is queued with
    the other labeling follow-ups.

    ### Schedule and conditions

    The LR schedule is the one from ex-2.1.3: 100 epochs, 10 warmup, cosine to
    1%. The anchor weight ramps to its peak alongside the LR warmup, holds at
    peak through epoch 90 (by which point the LR has decayed to a few percent
    of peak), then anneals to a 10% floor at epoch 100. It never
    reaches zero, which is the lesson of ex-2.9.3 applied as issue #10
    prescribes.

    Both moves are minimum-jerk — the quintic that starts and ends at rest —
    rather than linear. That is what M1 used: the ex-2.9.x dopesheets take
    mini's default interpolator, which is `minjerk`, so every regularizer ramp
    and anneal in those experiments had this shape.

    Whether the smoothness earns anything is untested; M1 never compared
    linear against smooth. It did test a *stepped* anneal, which worked only
    when the LR warmup restarted from zero at each step, so the one
    discontinuity anyone measured needed an accommodation to survive. A kink
    is the milder version of the same event (a jump in the derivative of
    $\lambda$ rather than in $\lambda$ itself) and may well cost nothing. We
    take the M1 default because this experiment has no budget to find out, and
    record the assumption here rather than in the code.

    The sweep conditions are peak anchor weight
    $\lambda \in \{0,\ 0.03,\ 0.1,\ 0.3\}$ crossed with seeds $\{0, 1, 2\}$, so
    12 cells. ($\lambda$ is M1's symbol for a regularizer weight, $\Omega$
    being the regularizer itself.) The $\lambda{=}0$ condition is the
    in-experiment control: same corpus, labels ignored. The other rungs chart
    the dose–response.

    One **star arm** joins them: $\lambda{=}0.1^{*}$, identical except that the
    anneal starts at epoch 50 rather than 90, still reaching the floor at 100.
    Star arm is our shorthand, borrowed from the *star points* of a central
    composite design: it moves one factor off the scoring rung instead of
    crossing the whole grid, so it costs 3 cells rather than 12 and answers a
    single question. The question here is whether the long hold at peak buys
    alignment or just leaks: the anchor keeps pulling every labeled line for
    90 epochs, and leakage is the secondary measurement we most expect to be
    sensitive to that. Against it stands the lesson of ex-2.9.3, that
    withdrawing protection while the optimizer is still hot lets the task loss
    reclaim the axis, and epoch 50 is a good deal hotter than epoch 90 (see
    the figure below). Spreading the withdrawal over the remaining 50 epochs,
    and stopping at a floor rather than zero, is what makes it worth trying.
    It is an unscored diagnostic; if it shows lower leakage at equal
    alignment, the anneal window becomes a design question for a later
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
            learning rate has decayed to 4% of peak; the star arm starts down
            at epoch 50, while the learning rate is still around 60%, and takes
            the rest of training to reach the same floor.
        """,
        caption="""
            The two schedules on one axis, each as a fraction of its own peak —
            the LR peak is fixed at 1e-2, the anchor weight's is the swept
            $\\lambda$. The anchor weight's ramp and anneal are minimum-jerk;
            both anneals end at epoch 100 on a floor of 10% rather than zero,
            so the star arm's is the longer and gentler of the two. Dots mark
            the learning rate at the moment each anneal begins.
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
    late, so at epoch {ex.STAR_ANNEAL_START:.0f}, where the star arm begins
    coming down, the LR is still
    {ex.learning_rate(ex.STAR_ANNEAL_START) / ex.PEAK_LR:.0%} of peak — over
    half of it. By epoch {ex.ANNEAL_START:.0f}, where the default begins, it is
    {ex.learning_rate(ex.ANNEAL_START) / ex.PEAK_LR:.0%}. The two conditions
    are therefore not withdrawing protection into comparable optimizers, and
    that is the thing to keep in view when reading the star-arm result.

    It is also why the anneal stretches rather than slides. Both conditions
    reach the floor at epoch {ex.ANNEAL_END:.0f}, so the star-arm anneal
    takes {ex.ANNEAL_END - ex.STAR_ANNEAL_START:.0f} epochs against the
    default's {ex.ANNEAL_END - ex.ANNEAL_START:.0f}, and λ comes off roughly in
    step with the cooling LR instead of dropping away underneath a hot one.
    The alternative — a fixed 10-epoch window sliding to 50–60 — reaches the
    floor while the LR is still around
    {ex.learning_rate(ex.STAR_ANNEAL_START + 10) / ex.PEAK_LR:.0%} of peak,
    which is close to the setting that produced M1/Ex-2.9.3's late collapses.
    That would be a fine stress test of the mechanism and a poor candidate for
    a schedule we might adopt, and the star arm is here to propose a schedule.

    The moved factor is the *start* of the anneal, with the endpoint pinned to
    the end of training. The rate follows from those two, so the design has
    one knob.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Scoring

    Hypotheses resolve on the middle rung $\lambda{=}0.1$, which is where the
    numbers in the results sections come from. Call a rung **task-clean** if
    its `named_holdout` exact match is within 0.02 (absolute) of the seed mean
    of the $\lambda{=}0$ control. H1 asks whether *every* anchored rung is
    task-clean. H2 claims that anchoring works, and that claim shouldn't fail
    just because we guessed a coefficient wrong. So its gates read *some
    task-clean rung clears the threshold*: a rung can carry H2 even if H1
    fails because a different rung broke the task. The task-clean requirement
    is what keeps this from being cherry-picking, because a rung earns nothing
    by reaching alignment if it broke the task getting there. The ladder is
    only four rungs, so the multiple-comparisons cost of reading across it is
    small. If the rung H2 resolves on is not $\lambda{=}0.1$, we say so and
    report both.

    H3 follows H2: it is scored on the rung H2 resolves on, and only if H2(a)
    passed there. That restriction matters because the H3 gate is a ratio of
    margins; a 2× ratio between noise-level margins would read as condensation
    where nothing condensed. H4 applies to every anchored run, but it is read
    alongside H2: a run whose alignment never rises keeps its running maximum
    trivially, so H4 can only be *informative* where the anchor took.

    ### Measurements

    Behavioral: exact match, NLL,[^nll] and the distance metrics of ex-2.1.3
    on all eval sets, with the usual nulls from `sca.baselines`.[^nulls] Exact
    match is the coarser scoring of the two: it only counts the argmax. So a
    run can hold accuracy while NLL drifts, and that drift is the earlier
    warning that capacity is being spent elsewhere.

    Alignment maps: $\cos(h_{\ell,t}, \hat v_{\text{red}})$ on the alignment
    probe set, per layer × position. Write $\alpha_c(\ell, t)$ for the mean of
    that cosine over probe lines whose first operand is color $c$. H2, H3,
    and H4 all read one statistic off this map, the alignment margin

    $$m(\ell, t) = \sum_c u_c\,\alpha_c(\ell, t) \;-\;
    \frac{1}{216}\sum_c \alpha_c(\ell, t),
    \qquad u_c \propto \texttt{redness}(c)^8,\ \ \textstyle\sum_c u_c = 1$$

    That is the affinity-weighted mean alignment minus the plain mean: how
    much closer to the anchor a color sits for being the kind of color the
    anchor pulls. The weights $u_c$ are the label probabilities themselves,
    normalized, so the statistic asks about precisely the lines that felt the
    term. A margin of 0 means the anchor axis is indifferent to redness. As a
    difference of cosines it runs from $-2$ to $2$, but the ends are out of
    reach: $2$ would need the red-weighted colors to sit exactly on the
    anchor while the cube average sits at the antipode. In a 64-dimensional
    stream, unrelated directions sit near $\cos = 0$, so the unweighted mean
    should stay close to zero, and complete success looks like $m \approx 1$.

    /// details | Why weight by redness rather than group colors
    An alternative we considered groups colors into red (`SIM3 > 0.5`) and
    other (`SIM3 < 0.01`, both from `sca.colorcube`) and scores a
    red-minus-other contrast. We weight by `redness` instead, for one reason:
    `redness` is what generates the labels, so weighting by it means the
    measurement and the supervision use the same notion of red. Under the
    grouping they are two different notions, and a disagreement between them
    would show up as a weak result with no way to attribute it. Weighting
    also sets no threshold and discards no color. The grouped version throws
    away the ambiguous middle (pinks, oranges, dark reds), which is exactly
    the range where a graded response would be visible.
    ///

    H2 scores the layer mean at op1,

    $$m_{\text{op1}} = \frac{1}{L+1}\sum_{\ell=0}^{L}
    m(\ell, \text{op1})$$

    We average over anchored sites because the anchor applies to all of them
    equally, so there is no site selection to make. Alongside it we report the
    min and max over layers as unscored diagnostics; a wide min–max gap tells
    us which depths resist the anchor. H3 compares the same layer mean across
    positions, and H4 tracks $m_{\text{op1}}$ over training.

    Grading: for each op1 color, the layer mean of $\alpha_c$ at the op1
    position, plotted against the redness of that color. This is the same
    quantity that $m_{\text{op1}}$ summarizes, shown per color instead of
    contracted to a number.

    Leakage (secondary, no threshold): a ridge probe for redness, fit on the
    residual stream at op1 with the $\hat v_{\text{red}}$-component projected
    out. High off-axis R² means red is also encoded elsewhere; we expect this
    to be low.

    Trajectories: every ~50 steps we record $m_{\text{op1}}$ and
    validation loss, along
    with the LR and anchor-weight schedules. This is the apparatus from
    ex-2.9.3, carried over as the first queue item of issue #10. A flat loss
    curve does not mean learning is done, so the alignment trajectory is the
    evidence to consult before any decision to shorten the schedule.

    [^nll]: Negative log-likelihood: the surprise, in nats, of the probability
    the model assigns to the correct answer token, with 0 meaning certain and
    correct.

    [^nulls]: Reference scores for degenerate predictors (the corpus-mean
    color, a uniform draw, the identity of op1), which make a distance number
    readable as good or bad.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Unless stated otherwise, results are reported on the $\lambda{=}0.1$
    condition (seed-averaged); the other rungs are context. Per the scoring
    rule above, the H2 gates read *some task-clean rung*, and H3 follows H2.

    **H1.** Anchoring does not harm the task. Every anchored condition is
    task-clean: `named_holdout` exact match within 0.02 (absolute) of the seed
    mean of the $\lambda{=}0$ control. Partial: the gate passes at
    $\lambda \le 0.1$ but fails at $0.3$.

    **H2.** The concept lands on the anchor, and does so in a graded way, at
    some task-clean rung. (a) $m_{\text{op1}} \ge 0.5$.
    (b) Redder colors sit closer to the anchor: over the 216 colors, $\alpha_c$
    increases with the redness of $c$, at Spearman
    $\rho \ge 0.8$.[^spearman] (c) The control condition shows
    $|m_{\text{op1}}| \le 0.1$.

    [^spearman]: Spearman ρ is a correlation coefficient computed on ranks
    rather than values, so it measures whether the ordering agrees and doesn't
    care about the shape of the curve.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    Partial: (a) and (c) hold but (b) fails, with the grading figure showing a
    step rather than a grade. In a step, the reds all sit at one alignment,
    the non-reds at another, and there is no ordering within either group. We
    checked how the ρ gate behaves on that shape. The measured alignment is
    continuous, so a pure step orders the colors within each of its two levels
    arbitrarily. Its expected ρ against this redness grid is at most
    {ex.STEP_RHO_CEILING:.2f}, for a step placed near the middle of the cube;
    a step at redness 0.5, the shape a memorized red exemplar would produce,
    scores {ex.step_rho(0.5):.2f}. So a step cannot clear the 0.8 gate on its
    own. What *would* carry a steppy response over the gate is residual
    ordering within the levels, which is some grading after all. That is why
    the figure, and the windowed mean drawn through it, is the arbiter for
    shapes in between.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **H3.** The concept condenses at the position that carries it.
    $m_{\text{op1}}$ is at least 2× the same layer mean at each of
    `+`, op2, and `=`. Those four positions are pulled identically, so the
    comparison is between sites that differ only in what the task does with
    them. The answer and newline are outside the pull; they are reported but
    not scored. Per the scoring rule, H3 resolves on the rung H2 does, and
    only if H2(a) passed there.

    We also anticipate a specific alternative outcome: broadcast, where all
    positions of a labeled line move toward $\hat v_{\text{red}}$ roughly
    equally. That would fail H3 while H2 passes. It would tell us something
    about sequence-level labeling rather than about anchoring itself; a
    logsumexp position-pooling variant is the queued response.

    **H4.** The late-instability mechanism of ex-2.9.3 does not reappear under
    the anneal-to-floor schedule. For every anchored run, the end-of-training
    $m_{\text{op1}}$ is at least 0.8× its running maximum over
    training.
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
    annotated, and $m_{\text{op1}}$ with it. Expected: monotone and
    graded, with the labeled-red corner near
    $\cos = 1$ and the far side of the cube near the control baseline.
    Contrary: flat, meaning the anchor didn't take; or a binary step at the
    label threshold, meaning it took but without grading.

    2. Per-layer decomposition of $m_{\text{op1}}$: $m$
    by layer at op1, all rungs, control band behind. A plain line chart, layer depth on x —
    depths are evenly spaced and every one is measured, so there is nothing for
    the step-plateau convention of ex-2.1.5 to guard against here (it exists
    because grammar landmarks are *not* evenly spaced). Expected:
    $m$ above zero at every depth. A sag in the last layer would suggest next-token pressure
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
    $m(\ell, t)$ over the six line positions (op1, `+`, op2, `=`,
    answer, newline), on the rung H2 resolves on ($\lambda{=}0.1$ unless the
    scoring rule moved it), seed mean, with the four pulled
    positions marked off from the two that are only measured. Drawn as stacked `smooth_step`
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
    broadcast. The answer column is shown but excluded from scoring, since it is
    unpulled and a red op1 gives a reddish mix regardless. Note in the caption
    (or below) whatever it does show: alignment there is spillover, which
    previews both the either-slot follow-up and how far an anchor reaches on its
    own.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training stability (H4)

    /// admonition | TODO
    Trajectory figure: $m_{\text{op1}}$ vs step for every anchored run (rows are rungs,
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
    that would show. Report its $m_{\text{op1}}$ beside it, because
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
    per-seed variability in condensation, how far alignment spills into the
    two unpulled positions, alignment on `open`-pair lines, and whether the
    batch-to-batch noise in the anchor gradient (no balancing) shows up in the
    trajectories of H4.
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


if __name__ == "__main__":
    app.run()
