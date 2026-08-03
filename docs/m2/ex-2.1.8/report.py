import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.8: when the repulsion acts, and how much is left",
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

    use_publisher(report_bundle(__file__))


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Ex 2.1.8: when the repulsion acts, and how much is left

    /// tip |
    <!-- tl;dr -->
    Ex-2.1.7 found that stretching one anneal was worth more than adding the
    repulsive term in the first place. Here we cross **when** that anneal
    finishes against **the level it finishes at**, to find an operating point
    that contains the cube-wide drift without costing selectivity.
    ///

    Ex-2.1.7 gave the first selective anchor in a transformer, but one gate
    went unmet. The anti-subspace term is defined on $\bar\alpha$, the mean
    alignment between the anchor direction and the whole color cube. With the
    term on, $\bar\alpha$ came down from 0.53 to {A50}, never reaching
    {AGATE}, the level that would count as containing the drift rather than
    just weakening it.

    The trajectories showed when containment gave way. While the repulsion
    outweighs the pull, $\bar\alpha$ stays near the control; once the
    anneal[^anneal] brings $\lambda_{\bar{\text{s}}}$ down to about a tenth of
    $\lambda_\text{a}$, it starts to climb. Each condition broke at a time set
    by its own schedule: around epoch 40 on the M1 anneal, epoch 75 on the arm
    that stretched it.

    [^anneal]: An anneal here is a schedule that ramps a loss weight down
        gradually over training, rather than switching it off at once.

    That arm, `span-anti-late`, is the reason this experiment exists. It
    changed one number: the epoch at which the repulsion finishes annealing to
    its hold ratio moved from 50 to 90. That improved every statistic the H2
    and H4 hypotheses of that report scored: margin {M50} → {M90},
    $\bar\alpha$ {A50} → {A90}, retention across the anneal, and grading.

    The M1 keyframes came from a 5-dimensional autoencoder bottleneck, mapped
    onto our 100 epochs by fraction of training, so there was never a reason to
    expect them to suit a 64-dimensional residual stream. But one arm cannot
    say why it worked, and two readings fit it equally well.

    **Timing.** The repulsion has to still be there when the pull is strong
    enough to move the cube, and a later anneal keeps it there.

    **Level.** Containment is lost at a threshold, around
    $\lambda_{\bar{\text{s}}}/\lambda_\text{a} \approx 0.1$, and a later
    anneal helps only because it postpones the crossing. On that reading the
    endpoint is the wrong parameter to move, and a floor above the threshold
    would hold containment whenever the anneal finished.

    They separate cleanly, because the second reading predicts an interaction
    the first does not: if a high enough floor never crosses the threshold,
    then at that floor the endpoint should stop mattering.

    /// details | Notation
    Carried from ex-2.1.7 unchanged.

    | symbol | meaning |
    | --- | --- |
    | $\hat v_{\text{red}}$ | the anchor direction |
    | $\lambda_\text{a}$ | anchor weight, fixed at {SCORING} throughout |
    | $\lambda_{\bar{\text{s}}}$ | anti-subspace weight, always given as the ratio $\lambda_{\bar{\text{s}}}/\lambda_\text{a}$ |
    | $m$, $m_{\text{op1}}$ | alignment margin; its layer mean at op1 |
    | $\bar\alpha$ | mean alignment with $\hat v_{\text{red}}$ over all 216 colors, layer-averaged at op1 |
    | **cell** | a **condition** crossed with a seed |
    ///
    """.replace("{SCORING}", f"{ex.SCORING_LAMBDA:g}")
        .replace("{AGATE}", f"{ex.MEAN_ALIGN_GATE:g}")
        .replace("{M50}", f"{ex.EX217_REFERENCE['end50-hold03']['m_op1']:.2f}")
        .replace("{M90}", f"{ex.EX217_REFERENCE['end90-hold03']['m_op1']:.2f}")
        .replace("{A50}", f"{ex.EX217_REFERENCE['end50-hold03']['alpha']:.2f}")
        .replace("{A90}", f"{ex.EX217_REFERENCE['end90-hold03']['alpha']:.2f}")
    )
    # REVIEW: the four ex-2.1.7 numbers in the intro now interpolate from
    # `ex.EX217_REFERENCE` instead of being written out, so the report and the
    # reproduction check cannot round them differently. Values verified against
    # ex-2.1.7's published metrics: span-anti m_op1 0.457 / ᾱ 0.326,
    # span-anti-late 0.598 / 0.132, span-bare ᾱ 0.526. Also narrowed "every
    # statistic ex-2.1.7 scored" to that report's H2 and H4 statistics: its H1
    # (task cost) was flat, not improved.
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | TODO
    Filled in when results land. Every preregistered hypothesis, its verdict,
    and the one number that decides it, with its gate inline. Under 200 words.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration. The method, hypotheses and decision thresholds
    below were written and frozen in Git before the experiment ran. Results
    replace the `TODO` placeholders in place, so a later reader can diff the
    prediction against the observation. Anything conceived after seeing the
    data goes under *Exploratory analyses*, marked post hoc.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Method

    ### Testbed

    Everything ex-2.1.7 held is held here: the word-level `v216` testbed from
    ex-2.1.3 (216 colors, one token each, d64-L4), the same corpus seed and
    splits, the same three training seeds, the same sequence-level labels
    driven by the first operand's redness, and the same measurements. The
    anchor schedule and its weight are unchanged. Only the anti-subspace
    schedule moves.

    Two choices constrain what this experiment can give the next one.

    **The pull stays on the four-position prompt span.** The op1-only pull in
    ex-2.1.7 scored higher, but that option exists only because this synthetic
    language has a known position carrying the concept; a document-level label
    gives no such position. Once we retire that position oracle we have to pull
    on a span, so an operating point is only useful to us if it was measured on
    one.

    **The anchor weight stays at $\lambda_\text{a} = {SCORING}$.** The ceiling
    arm in ex-2.1.7 showed selectivity falling at ten times this weight, with
    no task cost even there, so the scoring rung remains the right place to
    tune the repulsion against.

    ### The grid

    Two factors, fully crossed, at $\lambda_\text{a} = {SCORING}$ on the span
    pull.

    **Factor A, the anneal endpoint** — the epoch at which
    $\lambda_{\bar{\text{s}}}/\lambda_\text{a}$ reaches its hold ratio:
    {ENDS}. 50 is the M1 keyframe mapped by fraction of training, and 90 is
    the timing arm from ex-2.1.7, so two cells reproduce measured conditions.

    **Factor B, the hold ratio** — the level it settles at: {HOLDS}. 0.03 is
    the M1 level. 0.30 is three times the level at which ex-2.1.7 saw
    containment give way, so on the level reading it should never cross the
    threshold at all.

    Endpoint 100 is excluded. With no hold window left after the anneal, the
    hold ratio barely matters there, so that row would cross a factor against
    nothing and dilute its main effect.

    An un-anchored control ($\lambda_\text{a} = 0$) runs through the same code,
    giving {N_COND} conditions and {N_CELL} cells in total.
    """.replace("{SCORING}", f"{ex.SCORING_LAMBDA:g}")
        .replace("{ENDS}", ", ".join(f"{e:g}" for e in ex.ANTI_ANNEAL_ENDS))
        .replace("{HOLDS}", ", ".join(f"{h:g}" for h in ex.ANTI_HOLD_RATIOS))
        .replace("{N_COND}", str(len(ex.CONDITIONS)))
        .replace("{N_CELL}", str(len(ex.CONDITIONS) * len(ex.SEEDS)))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The confound, and the arm that separates it

    The two factors are not symmetric. Changing the hold ratio barely changes
    the total repulsion delivered. Changing the endpoint does: a later anneal
    spends longer at a high weight. So along factor A, *when* the repulsion
    acts and *how much* of it there is move together.

    The dose below is $\int \lambda_{\bar{\text{s}}}(e)\cdot\text{lr}(e)\,de$
    over training. The learning rate belongs in the integrand because it
    multiplies the gradient of every term, and it is itself warming and
    annealing across the same window. A weight-only integral would call two
    schedules equal even when they deliver visibly different pushes.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Computed from CONDITIONS through the same schedule functions the training
    # loop will call, so the table cannot describe runs that did not happen.
    # REVIEW: three changes to the prose under this table. (1) The endpoint's
    # dose span is 1.73x, not "nearly doubles", and the hold ratio's is 1.17x at
    # endpoint 50 but 1.05x at endpoint 90, so "at most a sixth". (2) The dose
    # arm sits below `end90-hold03` at every epoch, not below `end50-hold03`
    # ("M1's") — it exceeds the latter through mid-training by up to 0.07, which
    # is what dose-matching downward at a later endpoint requires. (3) Named the
    # weak-early-repulsion reading of a null, since scaling the opening ratio to
    # 1.443 changes when the repulsion acts as well as how much there is.
    # Verify: `ex.anti_dose` over CONDITIONS, and `ex.anti_subspace_weight`
    # sampled over epochs for the three schedules in the right-hand panel.
    _ref = ex.anti_dose(next(c for c in ex.CONDITIONS if c["name"] == ex.DOSE_ARM_REFERENCE))
    _rows = "\n".join(
        f"| `{c['name']}` | {c['anti_anneal_end']:.0f} | {c['anti_hold_ratio']:.2f} | "
        f"{c['anti_peak_ratio']:.3f} | {ex.anti_dose(c):.5f} | {ex.anti_dose(c) / _ref:.2f} |"
        for c in ex.CONDITIONS
        if c["lam"] > 0
    )
    mo.md(f"""
    <div class="report-table">

    | condition | `anneal_end` | `hold_ratio` | `peak_ratio` | dose | ÷ `{ex.DOSE_ARM_REFERENCE}` |
    | --- | ---: | ---: | ---: | ---: | ---: |
    {_rows}

    </div>

    Across the hold ratio, dose moves by at most a sixth; across the endpoint
    it rises by about three quarters. So the two levels of factor B need no
    dose-matching, and the three levels of factor A cannot be matched with the
    hold ratio: reaching the endpoint-90 dose from endpoint 50 would need
    $\\lambda_{{\\bar{{\\text{{s}}}}}}/\\lambda_\\text{{a}} \\approx 1.18$,
    holding the repulsion above the anchor weight for half of training.

    The achievable match runs the other way. `{ex.DOSE_ARM}` is
    `end90-hold03` with the opening ratio scaled
    {ex.ANTI_PEAK_RATIO:g} → {ex.DOSE_MATCHED_PEAK_RATIO:g}, which delivers
    the dose of `{ex.DOSE_ARM_REFERENCE}` on the late schedule. It is an
    interpolation rather than an extrapolation: the schedule stays at or below
    `end90-hold03` at every epoch.

    One asymmetry carries into that reading: matching the dose downward means
    the arm opens weaker than every other condition, so it holds less repulsion
    through the epochs where the anchor is first ramping in. If the arm tracks
    `{ex.DOSE_ARM_REFERENCE}` rather than `end90-hold03`, weak early repulsion
    is a second explanation alongside the dose account, and the trajectory
    figure is where the two would separate.
    """)
    return


@app.cell(hide_code=True)
def _():
    # The schedules the runs will actually train under, drawn from CONDITIONS
    # through the library functions. Sampled finely; the recorded trajectories
    # will be coarse and start after the warmup.
    E = np.linspace(0.0, float(ex.SCHEDULER.epochs), 1001)

    def _mu(c: dict) -> np.ndarray:
        return ex.anti_subspace_weight(
            E,
            lam=c["lam"],
            anneal_end=c["anti_anneal_end"],
            hold_ratio=c["anti_hold_ratio"],
            peak_ratio=c["anti_peak_ratio"],
        )

    @themed(
        name="schedules",
        alt_text="""
            Two log-scale line charts of anti-subspace weight against epoch, 0 to 100.
            In both, every curve starts high near 0.25 and descends. Left panel: six
            curves in two colors and three dash patterns, separating into a lower group
            that settles near 0.003 and an upper group that settles near 0.03, with the
            descent finishing at three different epochs. Right panel: three curves, two
            grey and one dashed blue that opens lower than the others and descends
            later. A faint line in both panels holds flat at 0.1 before falling at
            epoch 90.
        """,
        caption=r"""
            **The schedules under test.** Anti-subspace weight
            $\lambda_{\bar{\mathrm{s}}}$ over training, log scale, with the anchor
            weight $\lambda_\mathrm{a}$ in ghost ink for reference. **Left:** the six
            factorial cells — color keys the hold ratio (blue 0.03, orange 0.30), dash
            length keys the anneal endpoint (dotted 50, dashed 70, solid 90).
            **Right:** the dose arm (dashed) against `end50-hold03` and `end90-hold03`
            (grey), reaching the late endpoint from a lower opening ratio so the area
            under the curve matches the early one.
        """,
    )
    def _plot():
        _grey = light_dark("#666", "#999")
        _hold = {0.03: light_dark("#1f6fb4", "#5fa8dd"), 0.30: light_dark("#b4531f", "#dd8f5f")}
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, layout="constrained")

        # Left: the factorial. Color keys the hold ratio, dash keys the endpoint,
        # so the two factors stay visually separable.
        for c in ex.CONDITIONS:
            if c["name"] not in ex.FACTORIAL_NAMES:
                continue
            dash = {50.0: (0, (1, 1.6)), 70.0: (0, (4, 1.6)), 90.0: "-"}[c["anti_anneal_end"]]
            axs[0].plot(E, _mu(c), color=_hold[c["anti_hold_ratio"]], ls=dash, lw=1.4)
        axs[0].set_title("the factorial", fontsize="small", color=_grey)

        # Right: the dose arm against the two cells it sits between.
        for name, style in ((ex.DOSE_ARM_REFERENCE, "-"), ("end90-hold03", "-"), (ex.DOSE_ARM, (0, (4, 1.6)))):
            c = next(k for k in ex.CONDITIONS if k["name"] == name)
            axs[1].plot(E, _mu(c), lw=1.4, ls=style, color=_grey if name != ex.DOSE_ARM else _hold[0.03])
        axs[1].set_title("the dose arm", fontsize="small", color=_grey)

        # The anchor weight in ghost ink in both panels: the repulsion is only
        # ever meaningful relative to the pull it works against.
        for ax in axs:
            ax.plot(E, ex.anchor_weight(E), color=_grey, lw=1.0, alpha=0.45)
            ax.set_yscale("log")
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_ylim(1e-3, 1)
            ax.minorticks_off()
            ax.set_xlabel("epoch", fontsize="x-small")
            ax.tick_params(labelsize=7.5)
        axs[0].set_ylabel("weight", fontsize="x-small")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Measurements

    All carried from ex-2.1.7, computed by the same eval step on the same
    probe suite. No new measurement code.

    - `named_holdout` exact match, against the in-experiment control (H1).
    - $m_{\text{op1}}$: the label-affinity-weighted alignment at op1 minus the
      cube mean, averaged over layers (H2, H3).
    - $\bar\alpha$: mean alignment over all 216 colors at op1, layer-averaged
      (H2, H3). This is the quantity the repulsive term is defined on.
    - Grading: Spearman $\rho$ against `redness` and Pearson $R^2$ against
      `sim¹·⁵` over the 216 colors (H2).
    - Per-run margin trajectories, for retention (H2).

    Ex-2.1.7 quoted *retention* two ways under one name: the minimum
    final-over-peak margin across seeds in its Findings, the mean across seeds
    in its Arms section. This report uses the **{RSTAT}**, and every later
    report should. The gate of H4(b) is worded per run ("for every anchored
    run…"), so the minimum is the statistic it asks for; a three-seed mean can
    hide a single sliding run, the failure the gate exists to catch.
    """.replace("{RSTAT}", ex.RETENTION_STAT)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Hypotheses

    Gates and noise floors carry over from ex-2.1.7 with their thresholds and
    rationale: same architecture, same measurement, same statistics.

    **H1 — task cost.** Every condition is task-clean, meaning `named_holdout`
    exact match within {TASK} (absolute) of the seed mean of the in-experiment
    control. Ex-2.1.7 found no task cost anywhere, including at ten times this
    anchor weight, so this is a standing gate rather than an open question;
    it is stated because a sustained repulsive floor is the one thing in this
    grid that has not been tried. Partial: the `hold03` cells and the dose arm
    are clean and a `hold30` cell is not.

    **H2 — an operating point exists.** Some task-clean factorial cell meets
    all four of ex-2.1.7's selectivity and containment conditions at once,
    which no condition in that experiment did:
    (a) containment, $\bar\alpha \le {ALPHA}$;
    (b) margin, $m_{\text{op1}} \ge {MARGIN}$;
    (c) grading, Spearman $\rho \ge {RHO}$ against `redness` or Pearson
    $R^2 \ge {R2}$ against `sim¹·⁵`;
    (d) retention, every run whose running maximum of $m_{\text{op1}}$ reaches
    {FLOOR} ending at $\ge {RET}\times$ that maximum.
    Partial, in decreasing strength: (a) is met at the looser {ALPHAP} (the
    halving level from ex-2.1.7) with (b)–(d) held; or (a) is met at {ALPHA}
    but the margin falls between the ex-2.1.7 value of {M50} and the gate.

    **H3 — the level contributes at least as much containment as the timing.**
    Scored on two main effects[^main] on $\bar\alpha$, both taken as
    differences of means and reported with their sign. The hold-ratio effect is
    the nine runs at 0.03 minus the nine at 0.30. The anneal-endpoint effect is
    the six runs at endpoint 50 minus the six at endpoint 90. H3 holds when the
    hold-ratio effect is at least {AGATE} and at least as large as the endpoint
    effect.[^effect]

    Two outcomes that fail H3 are named in advance, so a miss stays readable.
    First, the endpoint effect clears {AGATE} and is the larger. That reads as
    the timing account: the repulsion has to be present while the pull is
    strong, and no floor substitutes for that. Second, `{PRIMARY}` alone
    contains the drift while neither main effect clears {AGATE}. That is an
    interaction, saying each factor is blocked by the other. The second outcome
    is only reachable if the `hold30` row reverses the endpoint ordering.
    Ex-2.1.7 measured the `hold03` row moving from {A50} to {A90} across the
    endpoint on its own, and half of that span already falls inside the
    endpoint effect before the other row is counted.

    A supporting observation, the interaction term of the same factorial read
    for its sign rather than scored: the level account predicts the endpoint
    effect **shrinks at the high hold ratio**, because a floor that never
    crosses the threshold makes the crossing time irrelevant. An endpoint
    effect of similar size at both hold ratios fits the timing account instead.
    The discussion should lean on this if the H3 main effects come out close.

    [^main]: A main effect is the average change from moving one factor,
        averaged over the levels of the other.

    [^effect]: Taking the ex-2.1.6 control spread of about 0.02 for
        $\bar\alpha$, the noise on the hold-ratio effect is near
        $0.02\sqrt{2}/\sqrt{9} \approx 0.009$, and on the endpoint effect,
        with six-run means, near $0.02\sqrt{2}/\sqrt{6} \approx 0.012$; so
        {AGATE} is four to five times either. Whether the anchored conditions
        hold that spread is checked from this experiment's own seeds before
        the effects are read. If they do not, the observed spread replaces the
        control spread and the gate moves with it; it is the only threshold in
        this report allowed to move.
    """.replace("{TASK}", f"{ex.TASK_GATE:g}")
        .replace("{ALPHAP}", f"{ex.MEAN_ALIGN_PARTIAL:g}")
        .replace("{ALPHA}", f"{ex.MEAN_ALIGN_GATE:g}")
        .replace("{MARGIN}", f"{ex.MARGIN_GATE:g}")
        .replace("{RHO}", f"{ex.GRADE_RHO_GATE:g}")
        .replace("{R2}", f"{ex.GRADE_R2_GATE:g}")
        .replace("{FLOOR}", f"{ex.H4_FLOOR:g}")
        .replace("{RET}", f"{ex.H4_RETENTION:g}")
        .replace("{AGATE}", f"{ex.ALPHA_EFFECT_GATE:g}")
        .replace("{PRIMARY}", ex.PRIMARY)
        .replace("{A50}", f"{ex.EX217_REFERENCE['end50-hold03']['alpha']:.2f}")
        .replace("{A90}", f"{ex.EX217_REFERENCE['end90-hold03']['alpha']:.2f}")
        .replace("{M50}", f"{ex.EX217_REFERENCE['end50-hold03']['m_op1']:.2f}")
    )
    # REVIEW: three changes to H1 and H3, none to a gate value.
    # (1) H1's partial named the dose arm as the likely offender; the analysis
    # section names the `hold30` row, which is the arm holding a repulsive
    # weight no earlier experiment sustained, so the partial now matches it.
    # (2) H3's endpoint factor has three levels, so "the main effect" did not
    # name a contrast. It is now the end50 − end90 difference of six-run means:
    # those are the two levels ex-2.1.7 measured, so the effect is comparable to
    # the published pair, and endpoint 70 stays available for the trajectory
    # reading without entering the score. The footnote's noise figure was
    # computed for nine-run means only; the six-run figure is added.
    # (3) The H3 heading claimed "the level ... not the timing", which the gate
    # (an ordering, not an exclusion) does not test; narrowed to the ordering.
    # The second named miss is kept but qualified: the 0.33 → 0.13 span that
    # ex-2.1.7 measured in the `hold03` row already puts ~0.10 into the endpoint
    # effect, so that miss needs the `hold30` row to reverse. Verify: if the
    # reproduction check finds `end50-hold03` and `end90-hold03` disagreeing
    # with ex-2.1.7, this qualification goes with it.
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Reproduction check

    /// admonition | TODO
    A two-row table: `end50-hold03` and `end90-hold03` against `span-anti`
    and `span-anti-late` from ex-2.1.7, on $m_{\text{op1}}$ and $\bar\alpha$
    (seed means; references in `ex.EX217_REFERENCE`). These conditions are
    identical in every parameter, so the two experiments should agree within
    the seed-mean noise floor of {NOISE} on the margin. Ex-2.1.6 carries no
    such floor for $\bar\alpha$, so the tolerance there is two seed-mean
    spreads computed from this experiment's own seeds, the same construction
    the H3 footnote uses.

    Expected: agreement. A gap wider than about two noise units means
    something moved between the experiments that neither report accounts for,
    and it invalidates reading this grid against the ex-2.1.7 numbers, which
    the H2 partial and the H3 framing both do.
    ///
    """.replace("{NOISE}", f"{ex.NOISE_SEED_MEAN:g}")
    )
    # REVIEW: the check compared two statistics but quoted a tolerance for one.
    # Rather than import a floor for ᾱ that no earlier experiment measured, the
    # tolerance is derived in-experiment, as H3's footnote already does for the
    # effect gate. This matters because H2 turns on ᾱ moving 0.13 → 0.1, a
    # distance of the same order as the seed spread. Verify: ex-2.1.7's own
    # per-seed ᾱ spread on these two conditions was about 0.011–0.015.
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Task cost (H1)

    /// admonition | TODO
    A table over all {N} conditions: `named_holdout` exact match (seed mean ±
    spread), and the absolute gap from the control, with the {TASK} gate
    marked.

    Expected: every gap well inside the gate, as in ex-2.1.7 where the largest
    was 0.0013 across seven conditions. A contrary result would most likely
    appear in the `hold30` column, the only place this grid sustains a
    repulsive weight the earlier experiments never held.
    ///
    """.replace("{N}", str(len(ex.CONDITIONS))).replace("{TASK}", f"{ex.TASK_GATE:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## An operating point (H2)

    /// admonition | TODO
    The scoring table: one row per factorial cell, columns $\bar\alpha$,
    $m_{\text{op1}}$, grading ($\rho$ and $R^2$), and retention (minimum
    across seeds), each marked against its gate, with a final column saying
    whether the cell meets all four. The ex-2.1.7 conditions `span-anti` and
    `span-anti-late` appear in the table as `end50-hold03` and `end90-hold03`.

    Expected under the level account: `{PRIMARY}` meets all four, and the
    `hold30` row generally beats the `hold03` row on containment. Under the
    timing account the `end90` column meets them instead and `end50-hold30`
    fails containment.

    A contrary result worth naming: containment improves and the margin falls
    with it, so no cell meets both. That would say the repulsion gets its
    containment by flattening the response rather than by clearing the axis.
    It is the same trade the ceiling arm in ex-2.1.7 found on the anchor
    weight, arriving on the repulsive side. The grading columns tell those
    apart, so read them before the verdict.
    ///
    """.replace("{PRIMARY}", ex.PRIMARY)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | TODO
    A trajectory figure: $\bar\alpha$ and $m_{\text{op1}}$ against epoch, one
    panel per factorial cell in a 3 × 2 mosaic (endpoint across, hold ratio
    down), with each cell's weight schedule drawn beneath it and the control
    in grey in every panel. The figure of this shape in ex-2.1.7 is the model.

    Expected under the level account: in the `hold03` row, $\bar\alpha$ leaves
    the control at an epoch that tracks each cell's anneal endpoint; in the
    `hold30` row it does not leave at all. Under the timing account the
    `hold30` row departs too, just later.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Attribution (H3)

    /// admonition | TODO
    The factorial table on $\bar\alpha$: the six cell means, the two main
    effects with the {AGATE} gate marked, and the interaction. Repeat on
    $m_{\text{op1}}$ beside it, since a factor that contains the drift without
    holding the margin has not given us an operating point.

    Also state the seed spread this experiment actually measures for
    $\bar\alpha$, as the footnote to H3 requires, before reading either effect.

    Expected: the hold-ratio effect larger, and a negative interaction — the
    endpoint mattering less at the high floor. A same-sized endpoint effect in
    both rows is the timing account, and is one of the two named misses.
    ///
    """.replace("{AGATE}", f"{ex.ALPHA_EFFECT_GATE:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## The dose arm

    /// admonition | TODO
    Three rows: `{REF}`, `end90-hold03`, and `{ARM}`, on $\bar\alpha$,
    $m_{\text{op1}}$, grading and retention — with the dose column from the
    Method table repeated, since it is the whole point of the comparison.

    The arm delivers the repulsion of `{REF}` on the timing of
    `end90-hold03`. If it tracks `end90-hold03`, the endpoint effect is timing
    and the extra dose is incidental. If it tracks `{REF}`, the endpoint effect
    is dose, and "anneal later" is a roundabout way of saying "use more".

    Scores no hypothesis: one condition, three seeds, and the peak ratio moves
    alongside the endpoint, so it separates the two accounts without
    quantifying either.
    ///
    """.replace("{REF}", ex.DOSE_ARM_REFERENCE).replace("{ARM}", ex.DOSE_ARM)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Secondary measurements

    /// admonition | TODO
    Carried from ex-2.1.7 for continuity, scoring nothing: redness read from
    the other 63 directions against the computed RGB floor of 0.863, and the
    extent of the color cube through depth. The timing arm in ex-2.1.7 was the
    highest of any condition on the first of these while being the best on
    every scored statistic, so the grid is worth checking for a trend the
    single arm could not show.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    /// admonition | TODO
    Post hoc, planned after seeing the results; scores no hypothesis. Left
    empty in the preregistration.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    Written when the results land. Three questions it should answer, and
    nothing beyond what the data supports:

    1. Which operating point should the next experiments use, and on what
       evidence? Anything that pulls without a position oracle needs one, and
       so would an intervention experiment — $\bar\alpha$ is the number that
       predicts how much of the color cube an edit to the anchor axis would
       move.
    2. Does the schedule finding generalize past this architecture, or is it
       another set of keyframes fitted to one testbed? The M1 keyframes did
       not transfer; stating what would make ours transfer is the useful move.
    3. What is still missing. Containment is not exclusivity: ex-2.1.7 found
       redness as readable off the anchor axis as in the control, and nothing
       in this grid changes that measurement's meaning.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
