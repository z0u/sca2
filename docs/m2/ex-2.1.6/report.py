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

    We add one thing: an alignment probe set. For each of the 216 colors, it
    builds one probe line per closed partner of that color, meaning the 27
    pairs whose mix has a name, with the color as the first operand. Closed
    pairs guarantee every probe line has a named answer, so the measured
    answer position corresponds to a real token. Using all 27 makes the probe
    set exhaustive: no sampling, no seed, and every cell measured on
    identical lines, drawn from seen and held-out pairs alike. Per-color
    alignment statistics average over op2 rather than conditioning on it.

    ### Labels

    A line is labeled *red* by a coin flip weighted by the redness of its first
    operand: `p = redness(op1)⁸ × 0.08`, where `redness = r·(1 − g/2 − b/2)`,
    one independent draw per line.[^bern] This is the `RED_PROB` of M1, applied to the
    color of the first operand only. A red op2 or a red answer never triggers a
    label, which keeps the ground truth unambiguous for the question of *where*
    the concept should condense. An either-slot labeling scheme is queued as a
    follow-up.

    Labels are redrawn per visit, as in M1, so they differ from epoch to
    epoch, and the expected pull on a color is proportional to its label
    affinity. The graded response of H2 is a further claim on top of that.
    The affinity spans four orders of magnitude, so a response that stayed
    proportional to it would be measurable on only the reddest few dozen
    colors. The grading H2(b) predicts comes instead from generalization:
    colors partway red moving partway. M1 is the precedent. With labels
    drawn from this same affinity, the damage it measured under intervention
    graded as low powers of its HSV similarity-to-red — `sim³` for ablation,
    `sim²` for suppression — far flatter than `redness⁸`. The ceilings under
    the hypotheses make this quantitative.

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

    The right panel also bounds what the pull alone can produce.
    {int((ex.LABEL_P < 1e-6).sum())} colors are labeled less than once in a
    million lines, so a response that tracked label affinity would sit under
    any achievable measurement floor across most of the cube. It would read
    as a step, and it clears neither gate of H2(b). So the curve H2(b) predicts
    is a claim about generalization: the anchor catching *red* rather than
    the labeled tokens, with colors partway red moving partway whether or
    not they were ever labeled. The model only ever sees Bernoulli draws
    from this distribution, and nothing stops it from treating "red enough
    to be labeled sometimes" as a threshold. That outcome is what (b)
    scores against.

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
    outside $\mathcal{P}$. We exclude them because of a confound: the
    answer of a red-labeled line is itself reddish, since a red op1 drags
    the mix. An anchored answer position would therefore align for two
    reasons at once, and we could not tell them apart. The exclusion costs
    nothing that we are measuring: the four pulled positions still receive
    an identical, undifferentiated pull, so the condensation question of
    H3 stands, and the answer position becomes a clean read of how far
    alignment spreads on its own.

    What the exclusion does not do is spare the state that emits the
    answer. The model is causal, so the answer token is predicted from the
    state at `=`, a position inside $\mathcal{P}$ that is pulled at every
    layer, including the final one the logits are read from. The task cost
    H1 measures therefore includes pressure on the answer logits
    themselves, alongside the cost of carrying an anchored concept through
    the rest of the prompt. Any pull that covers the prompt has this
    property; we state it here so H1 is read as measuring both together.

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

    Hypotheses resolve on the middle rung, $\lambda{=}0.1$; the numbers in
    the results sections come from that rung. Call a rung **task-clean** if
    its `named_holdout` exact match is within 0.02 (absolute) of the seed mean
    of the $\lambda{=}0$ control. H1 asks whether *every* anchored rung is
    task-clean.

    H2 claims that anchoring works, and that claim shouldn't fail just
    because we guessed a coefficient wrong. So its gates ask whether *some*
    task-clean rung clears the threshold: a rung can carry H2 even if H1
    fails because a different rung broke the task. The task-clean requirement
    keeps this from being cherry-picking: a rung earns nothing by reaching
    alignment if it broke the task getting there. And the ladder has only
    four rungs, so the multiple-comparisons cost of reading across it is
    small. If the rung H2 resolves on is not $\lambda{=}0.1$, we say so and
    report both. If more than one other task-clean rung clears the gates, H2
    resolves on the one with the highest $m_{\text{op1}}$; the choice is
    mechanical rather than ours. If no rung is task-clean, H2 fails on the
    gate it is scored against, and we report the alignment numbers anyway.
    The star arm is not a rung: it is reported alongside the ladder, but it
    takes no part in the "every anchored rung" test of H1 or in the search
    for a rung for H2 to resolve on.

    Every hypothesis except H4 is scored on the final checkpoint of each run;
    H4 is the one that reads the trajectory, and its end-of-training value is
    that same final checkpoint.

    H3 is scored on the same rung H2 resolves on, and only if H2(a) passed
    there. That restriction matters because the H3 gate is a ratio of
    margins. If both margins are at noise level, a 2× ratio between them
    would look like condensation where nothing condensed. H4 applies to every
    anchored run whose alignment ever clears a floor of 0.2. Where the
    anchor never took, the running maximum is itself at noise level, and a
    ratio of noise-level margins says nothing about stability. Runs below
    the floor are reported but not scored.

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

    In words: take the mean alignment weighted by label affinity, and subtract the
    plain mean over all colors. The margin measures how much closer a color
    sits to the anchor for being the kind of color the anchor pulls. The
    weights $u_c$ are the label probabilities themselves, normalized, so the
    statistic asks about precisely the lines that felt the anchor term. A
    margin of 0 means the anchor axis is indifferent to redness. Since $m$
    is a difference of cosines, it runs from $-2$ to $2$, but the ends are
    out of reach: $2$ would need the red-weighted colors to sit exactly on
    the anchor while the cube average sits at the exact opposite direction.
    In a 64-dimensional stream, unrelated directions sit near $\cos = 0$, so
    the unweighted mean should stay close to zero, and complete success
    looks like $m \approx 1$.

    /// details | Why weight by redness rather than group colors
    We considered an alternative: group colors into red (`SIM3 > 0.5`) and
    other (`SIM3 < 0.01`, both from `sca.colorcube`), and score a
    red-minus-other contrast. We weight by `redness` instead, for one
    reason: `redness` is what generates the labels, so weighting by it means
    the measurement and the supervision use the same notion of red. The
    grouped version uses two different notions, and a disagreement between
    them would show up as a weak result with no way to tell which notion was
    responsible. Weighting also sets no threshold and discards no color. The
    grouped version throws away the ambiguous middle (pinks, oranges, dark
    reds), which is exactly the range where a graded response would be
    visible. Both notions of red do appear in the H2(b) grading gate, but as
    separate, named tracks; there, a disagreement between them can be traced
    to one notion or the other, which it couldn't inside a single blended
    statistic.
    ///

    H2 scores the layer mean at op1,

    $$m_{\text{op1}} = \frac{1}{L+1}\sum_{\ell=0}^{L}
    m(\ell, \text{op1})$$

    We average over anchored sites because the anchor applies to all of them
    equally, so there is no site selection to make. Alongside the mean we
    report the min and max over layers as unscored diagnostics; a wide
    min–max gap tells us which depths resist the anchor. H3 compares the
    same layer mean across positions, and H4 tracks $m_{\text{op1}}$ over
    training.

    Grading: for each op1 color, we take the layer mean of $\alpha_c$ at the
    op1 position and plot it against the redness of that color. This is the
    same quantity that $m_{\text{op1}}$ summarizes, shown per color instead
    of contracted to a number. We read two statistics off this plot:
    Spearman ρ against `redness`, and Pearson R² against `sim^1.5`. Here
    `sim` is the angular similarity-to-red used by the intervention scoring
    in M1 (`sca.colorcube.sim_to_red`, restricted to this grid). The
    exponent comes from mapping the M1 ablation result from damage to
    alignment: damage graded as `sim³` and is quadratic in alignment, so
    alignment should grade as `sim^1.5`. H2(b) reads both statistics.

    Leakage (secondary, no threshold): we fit a ridge probe for redness on
    the residual stream at op1, one probe per layer, after projecting out
    the $\hat v_{\text{red}}$ component. A high R² here would mean red is
    also encoded somewhere other than the anchor axis; we expect it to be
    low. R² is read on held-out colors, using a 5-fold split over the 216
    op1 colors, so no color is scored by a probe that saw it. The ridge
    penalty is chosen within each training fold by an inner
    cross-validation, so no setting is picked by looking at the number we
    report.

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

    **H1.** Anchoring does not harm the task. Every anchored rung is
    task-clean: `named_holdout` exact match within 0.02 (absolute) of the seed
    mean of the $\lambda{=}0$ control. Partial: the gate passes at
    $\lambda \le 0.1$ but fails at $0.3$. The star arm is reported in the
    same table but scored only descriptively, per the scoring rule.

    **H2.** The concept lands on the anchor, and does so in a graded way, at
    some task-clean rung. (a) $m_{\text{op1}} \ge 0.5$.
    (b) Redder colors sit closer to the anchor, on either of two pre-named
    notions of *redder*. The grading statistic for a color $c$ is the layer
    mean of $\alpha_c$ at op1, averaged over seeds before anything is ranked
    or correlated; over the 216 colors, it clears at least one of
    Spearman $\rho \ge 0.8$ against `redness`,[^spearman] or Pearson
    $R^2 \ge 0.8$ against `sim^1.5`.[^pearson] (c) The control condition shows
    $|m_{\text{op1}}| \le 0.1$.

    [^spearman]: Spearman ρ is a correlation coefficient computed on ranks
    rather than values, so it measures whether the ordering agrees and doesn't
    care about the shape of the curve.

    [^pearson]: The squared Pearson correlation: the share of variance in the
    response explained by a linear function of `sim^1.5`. It is scale- and
    offset-invariant, so it tests proportionality to the shape M1 implies
    for alignment with no tuned constant.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    Partial: (a) and (c) hold but (b) fails on both tracks, with the grading
    figure showing a step rather than a grade. In a step, the reds all sit at
    one alignment, the non-reds at another, and there is no ordering within
    either group.

    /// admonition | Why (b) is two tracks, and how the gates were placed
    This gate went through several review rounds before freezing, so the
    reasoning is recorded here in full, with the response shapes we checked
    and what each scores.

    *The shape of the pull itself fails both tracks, and that is the
    intended reading.* The direct pull is proportional to `redness⁸`, and a
    response that stayed proportional to it cannot clear a rank gate.
    Unrelated directions in a 64-dimensional stream put a floor of about
    0.011 on the measured $\alpha_c$ (a spread of $1/\sqrt{{64}}$ per
    cosine, averaged over 27 probe lines × 5 residual slices), or ≈0.006
    after the three-seed mean the gate is scored on, and
    {int((ex.LABEL_P < 1e-6).sum())} colors have expected pull below
    $10^{{-6}}$. So most of the cube ranks at random; simulation at that
    floor puts ρ near 0.4, and its R² against `sim^1.5` is
    {ex.r2_sim(ex.REDNESS**8):.2f}, far under the gate on both tracks. A
    response confined to the pull means the anchor caught the labeled tokens
    rather than *red*, which is the memorized-exemplar outcome (b) exists to
    catch. M1 sets the expectation for success: with labels drawn from this
    same affinity, its measured damage graded as low powers of `sim`.

    *The two tracks name two families a generalized response could take*,
    one from each side of the milestone. Track 1 is ρ against `redness`. It
    accepts any response that rises monotonically with the label-generating
    variable: a linear `redness` response scores ≈0.99 at the seed-mean
    noise floor, and `redness³` scores ≈0.89. Track 2 is R² against
    `sim^1.5`, M1's ablation result mapped from damage to alignment
    (damage ∝ `sim³` and damage quadratic in alignment give alignment ∝
    `sim^1.5`; the suppression result maps to `sim¹` the same way). The 3/2
    power also maximizes coverage of the family: R² ≥ 0.86 against every
    power of `sim` from 1 to 3, and `redness³` scores
    {ex.r2_sim(ex.REDNESS**3):.2f}. (These simulated scores assume a
    unit-amplitude response; at half amplitude, roughly the least H2(a)
    accepts, `redness³` scores ≈0.83 on track 1 and ≈0.90 on track 2.)
    Each track is robust where the other is marginal. `sim` orders the cube
    by hue rather than by the `redness` formula, and it takes only 23
    distinct values on this grid, so a sim-family response of any power
    scores ρ ≈ 0.71 on track 1 at the seed-mean floor, under the gate; its
    noise-free ceiling of {ex.SIM_RHO_CEILING:.2f} sits just over it,
    because midranks credit the ties. In the other direction, a linear
    `redness` response scores R² = {ex.r2_sim(ex.REDNESS):.2f} on track 2,
    right at the gate, while clearing track 1 with room to spare. Passing
    on either track accepts every graded shape above; requiring both tracks
    would fail most of them.

    *A pure step clears neither gate.* The measured alignment is continuous,
    so a step orders the colors within each of its two levels arbitrarily.
    Its expected ρ is at most {ex.STEP_RHO_CEILING:.2f} over all placements
    (a step at redness 0.5, the memorized-exemplar shape, scores
    {ex.step_rho(0.5):.2f}), and its R² against `sim^1.5` is at most
    {ex.STEP_R2_CEILING:.2f} over placements in either variable. The margins
    between those ceilings and the 0.8 gates are a few hundredths, which is
    one more reason the figure stays in charge. What would carry a steppy
    response over a gate is residual ordering within the levels, which is
    some grading after all; for shapes in between, the arbiter is the
    windowed mean drawn through the scatter.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **H3.** The concept condenses at the position that carries it: that is,
    $m_{\text{op1}}$ is at least 2× the same layer mean at each of `+`, op2,
    and `=`. If the margin at a comparison position is zero or negative, the
    2× test has no content, so we treat it as met. (H3 is scored only when
    H2(a) has already put $m_{\text{op1}}$ above 0.5, so a comparison site
    at or below zero is dominated outright.) The anchor pulls those four
    positions identically, so the comparison is between sites that differ
    only in what the task does with them. The answer and newline positions
    are outside the pull; they are reported but not scored. Per the scoring
    rule, H3 resolves on the rung H2 does, and only if H2(a) passed there.

    We also anticipate a specific alternative outcome: broadcast, where all
    positions of a labeled line move toward $\hat v_{\text{red}}$ roughly
    equally. That would fail H3 while H2 passes. It would tell us something
    about sequence-level labeling rather than about anchoring itself; a
    logsumexp position-pooling variant is the queued response.

    **H4.** The late-instability mechanism of ex-2.9.3 does not reappear under
    the anneal-to-floor schedule. Concretely: for every anchored run whose
    running maximum of $m_{\text{op1}}$ reaches 0.2, the end-of-training
    value is at least 0.8× that maximum. The 0.2 floor keeps the gate from
    resolving on noise. In a simulation of unrelated 64-d states, the
    per-checkpoint noise in the margin is near 0.004, the maximum of a
    noise-only trajectory is near 0.014, and a ratio of noise-level margins
    fails the 0.8× gate about as often as not.
    Runs below the floor are reported but not scored.
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
    scatter. Control condition greyed underneath, one panel per rung, both
    grading statistics annotated (Spearman ρ vs `redness`, R² vs `sim^1.5`),
    and $m_{\text{op1}}$ with them. Expected: monotone and
    graded, with the labeled-red corner near
    $\cos = 1$ and the far side of the cube near the control baseline.
    Contrary: flat, meaning the anchor didn't take; or a binary step at the
    label threshold, meaning it took but without grading; or a response
    confined to the handful of labeled colors, meaning the pull was
    memorized rather than generalized.

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
    op2, and `=`. The last-layer panel is also where the pressure on answer
    emission noted in the method would surface: at `=` the pull and the
    answer logits share a state, so a dip there relative to earlier layers
    is worth a caption note if it appears.

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
    Trajectory figure: $m_{\text{op1}}$ vs step for every anchored run.
    Rows are rungs; lines are seeds. The LR and anchor-weight schedules
    appear as background bands. Each run is annotated with its running-max
    ratio, and runs under the 0.2 floor are marked unscored.

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
