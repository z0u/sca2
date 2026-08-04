import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.8: Repulsion tuning",
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
    mo.md(r"""
    # Ex 2.1.8: Repulsion tuning

    /// tip |
    <!-- tl;dr -->
    We test anti-subspace schedules (trailing timing and strength) to find an operating point that contains the cube-wide drift without reducing selectivity.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    In ex-2.1.6 we aligned *red* with a direction in the residual stream, but it dragged the whole color space with it.
    Then in ex-2.1.7 we added a repulsive *anti-subspace* term that pushed everything away from the _red_ axis $\hat v_{\text{red}}$. It helped, but not enough: the mean *red*-alignment over all colors fell from $\bar\alpha$ = 0.53 to {A50}, and the target was &lt; {AGATE}.

    The training trajectories show what happens over the course of a run. While the repulsion outweighs the pull, $\bar\alpha$ stays near the control, which is what we want. Then the anneal[^anneal] brings $\lambda_{\bar{\text{s}}}$ down to about a tenth of $\lambda_\text{a}$, and $\bar\alpha$ starts to climb again. Each condition turned at its own point in the schedule: around epoch 40 on the M1 schedule, and epoch 75 on `span-anti-late`, the arm that stretched it.

    The M1 keyframes came from a 5-dimensional autoencoder bottleneck, mapped onto our 100 epochs by fraction of training. So we never had much reason to expect them to suit a 64-dimensional residual stream.

    [^anneal]: An anneal here is a schedule that ramps a loss weight down gradually over training.

    All that `span-anti-late` did was push the end of the anneal from epoch 50 to epoch 90. That improved every statistic the H2 and H4 hypotheses scored: margin {M50} → {M90}, $\bar\alpha$ {A50} → {A90}, retention across the anneal, and grading. For future experiments, then, should we stretch the anneal, or end it at a higher weight? Each option is a reading of the ex-2.1.7 result, and the rest of this report refers to them by name:

    - The timing account: the repulsion has to still be there while the pull is strong enough to move the cube, and a later anneal keeps it there. Stretch the anneal.
    - The level account: containment is lost when $\lambda_{\bar{\text{s}}}/\lambda_\text{a}$ falls past the threshold ex-2.1.7 observed, and a later anneal helps only by postponing that. End the anneal at a higher weight.

    We test both. The two separate cleanly, because only the level account predicts an interaction: a floor that stays above the threshold never crosses it, so at that floor the timing of the endpoint should stop mattering.

    /// details | Notation and glossary
    Carried from ex-2.1.7.

    | term | meaning | target |
    |:---:|:--- |:---:|
    | $\hat v_{\text{red}}$ | the anchor direction | |
    | $\lambda_\text{a}$ | anchor weight, fixed at {SCORING} throughout | |
    | $\lambda_{\bar{\text{s}}}$ | anti-subspace weight, always given as the ratio $\lambda_{\bar{\text{s}}}/\lambda_\text{a}$ | |
    | cell | a sweep condition crossed with a seed | |
    | $\alpha_c$, $\bar\alpha$ | the alignment (cosine) of color $c$ with $\hat v_{\text{red}}$ at op1, averaged over layers; and the mean of that over all 216 colors | ↓ |
    | $m$, $m_{\text{op1}}$ | alignment margin $\sum_c w_c \alpha_c - \bar\alpha$, where $w_c$ are the *red*-label affinities normalized to sum to 1; $m_{\text{op1}}$ is the layer mean of $m$ at op1 | ↑ |
    | contained | of a condition: $\bar\alpha \le {AGATE}$, so the drift is held near the control rather than just weakened | |
    | retention | per run, the final $m_{\text{op1}}$ divided by its running peak (scored once the peak reaches {FLOOR}); reported as the minimum across seeds | ↑ |
    | leak | $R^2$ of redness decoded from the 63 non-anchor directions at op1; unscored, and cannot fall below the RGB-geometry floor of 0.863 | — |

    | measurement | interpretation |
    |:---:|:--- |
    | $m \rightarrow 1$ | *red* colors are aligned with the anchor, and other colors are not |
    | $m \rightarrow 0$ | *red* colors are unaligned, or they are but so is every other color |
    | $\bar\alpha \rightarrow 1$ | the whole cube is anchored, not just *red* |
    | $\bar\alpha \rightarrow 0$ | most colors are orthogonal to the anchor |
    ///
    """.replace("{SCORING}", f"{ex.SCORING_LAMBDA:g}")
        .replace("{AGATE}", f"{ex.MEAN_ALIGN_GATE:g}")
        .replace("{FLOOR}", f"{ex.H4_FLOOR:g}")
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
    This is a preregistration. The method, hypotheses and decision thresholds below were written and frozen in Git before the experiment ran. Anything conceived after seeing the data goes under *Exploratory analyses*.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    Like ex-2.1.7: the word-level `v216` corpus from ex-2.1.3 (216 colors, one token each, d64-L4), and the same corpus, seeds, sequence-level labels, and measurements.

    We use the same *anchor* schedule, but vary the *anti-subspace* schedule.

    Since we're using sequence-level labels, this experiment is like the `span-*` conditions in ex-2.1.7.
    The op1-only pull in ex-2.1.7 scored higher, but that option exists only because this synthetic language has a known position carrying the concept. Once we retire that position oracle we have to pull on a span.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The anchor weight stays at $\lambda_\text{a} = {SCORING}$. The ceiling arm in ex-2.1.7 showed selectivity falling at ten times this weight, with no task cost even there. So the previous scoring rung is still the right place to tune the repulsion against.
    """.replace("{SCORING}", f"{ex.SCORING_LAMBDA:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The sweep grid

    Factors:
    """)
    return


@app.cell(hide_code=True)
def _():
    def _and_join(items) -> str:
        """Prose list: a sealing "and" before the last item, so the list cannot leak into the next sentence."""
        *head, last = list(items)
        return f"{', '.join(head)} and {last}" if head else last

    mo.md(
        r"""
    - **A.** Anneal endpoint

        The epoch at which $\lambda_{\bar{\text{s}}}/\lambda_\text{a}$ reaches its hold ratio.
        Levels: {ENDS}.
        Endpoint 50 is the M1 keyframe mapped by fraction of training, and
        90 is the timing arm from ex-2.1.7, so two cells reproduce conditions we have already measured.

    - **B.** Hold ratio

        The level it settles at. Levels: {HOLDS}.
        Ratio 0.03 is the M1 level; 0.30 is three times the level at which ex-2.1.7 saw
        containment give way, so on the level reading it should never cross the
        threshold at all.

    We leave out endpoint 100. It leaves no hold window after the anneal, so
    the hold ratio barely matters there; that row would cross one factor
    against nothing and dilute its main effect.

    An un-anchored control ($\lambda_\text{a} = 0$) runs through the same code,
    giving {N_COND} conditions and {N_CELL} cells in total.
    """.replace("{ENDS}", _and_join(f"{e:g}" for e in ex.ANTI_ANNEAL_ENDS))
        .replace("{HOLDS}", _and_join(f"{h:.2f}" for h in ex.ANTI_HOLD_RATIOS))
        .replace("{N_COND}", str(len(ex.CONDITIONS)))
        .replace("{N_CELL}", str(len(ex.CONDITIONS) * len(ex.SEEDS)))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### A confound, and its mitigation

    The two factors are not symmetric. Changing the hold ratio barely changes
    the total repulsion delivered. Changing the endpoint does, because a later
    anneal spends longer at a high weight. So along factor A, *when* the
    repulsion acts and *how much* of it there is move together.

    We call that total the *dose*, and measure it as
    $\int \lambda_{\bar{\text{s}}}(e)\cdot\text{lr}(e)\,de$ over training. The
    learning rate belongs inside the integral because it multiplies the
    gradient of every term, and it is itself warming up and annealing across
    the same window. A weight-only integral would call two schedules equal even
    when they deliver visibly different pushes.
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
    | condition | `anneal_end` | `hold_ratio` | `peak_ratio` | dose | ÷ `{ex.DOSE_ARM_REFERENCE}` |
    | --- | ---: | ---: | ---: | ---: | ---: |
    {_rows}

    Across the hold ratio, dose moves by at most a sixth; across the endpoint it rises by about three quarters. So we don't need to dose-match the two levels of factor B. And we can't dose-match the three levels of factor A with the hold ratio without extending past the tested range of ex-2.1.7.[^cant]

    [^cant]: Reaching the endpoint-90 dose from endpoint 50 would need $\\lambda_{{\\bar{{\\text{{s}}}}}}/\\lambda_\\text{{a}} \\approx 1.18$, which would hold the repulsion above the anchor weight for half of training. We could do that, but that's a different experiment.

    Instead, we add an arm: `{ex.DOSE_ARM}` is `end90-hold03` with the opening ratio scaled {ex.ANTI_PEAK_RATIO:g} → {ex.DOSE_MATCHED_PEAK_RATIO:g}, so it delivers the dose of `{ex.DOSE_ARM_REFERENCE}` on the late schedule.
    That's still asymmetrical: this arm has less repulsion through the epochs where the anchor is first ramping in. If it tracks `{ex.DOSE_ARM_REFERENCE}` rather than `end90-hold03`, weak early repulsion may be the cause. This should be visible in the trajectory figure.
    """)
    return


@app.cell(hide_code=True)
def _():
    # The schedules the runs will actually train under, drawn from CONDITIONS
    # through the library functions. Sampled finely; the recorded trajectories
    # will be coarse and start after the warmup.
    E = np.linspace(0.0, float(ex.SCHEDULER.epochs), 1001)
    # The dose integrand's other factor, computed once rather than per theme pass.
    LR = ex.learning_rate(E)

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
            Two line charts against epoch, 0 to 100, side by side. Left: six regularizer weight curves,
            all starting near 0.25 and descending, separating into a lower group that
            settles near 0.003 and an upper group that settles near 0.03, with the
            descent finishing at three different epochs. A seventh
            curve starts lower, near 0.14, crosses the others around epoch 25, and
            joins the lower group.
            Right: the same weight schedule, but with the dose shown as shaded areas.
            Two humps that rise from zero, peak in the first third of training and
            decay to zero by about epoch 80.
            The two crescents where they differ are hatched — grey where the dose arm falls short early, purple
            where it runs higher late. The two crescents are about the same size.
            Faint reference curves show the anchor weight in both panels, and the
            unscaled late schedule on the right.
        """,
        caption=r"""
            **The schedules under test.** **Left:** anti-subspace weight
            $\lambda_{\bar{\mathrm{s}}}$ over training, for the six factorial conditions and the dose arm.
            **Right:** the dose itself, as its integrand
            $\lambda_{\bar{\mathrm{s}}}\cdot$lr, so area on the page
            is repulsion delivered. Hatching shows where the dose arm
            falls short of the reference early and where it makes that back late.
            Ghost ink in both panels: the anchor weight $\lambda_\mathrm{a}$; on
            the right, also `end90-hold03`, the schedule the arm was scaled down
            from.
        """,
    )
    def _plot():
        _grey = light_dark("#666", "#999")
        _hold = {0.03: light_dark("#1f6fb4", "#5fa8dd"), 0.30: light_dark("#b4531f", "#dd8f5f")}
        _dashes = {50.0: "-", 70.0: (0, (1, 1.6)), 90.0: (0, (4, 1.6))}
        # The arm gets its own color in both panels. Coloring it by hold ratio
        # would make it a second blue dashed line, and its opening ratio — the
        # one thing that sets it apart — is the hardest difference to see.
        _arm_color = light_dark("#7b3fa0", "#b98ce0")
        _by_name = {c["name"]: c for c in ex.CONDITIONS}
        # Not sharey: the panels carry different quantities. Left is the weight
        # itself on a log axis, where shape is readable; right is the integrand
        # of the dose on a linear axis, where area is readable.
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")

        # Left: the factorial, plus the arm, so the schedule the runs train under
        # is comparable with the six it sits among. Color keys the hold ratio,
        # dash keys the endpoint, so the two factors stay visually separable.
        for c in ex.CONDITIONS:
            if c["name"] not in ex.FACTORIAL_NAMES:
                continue
            dash = _dashes[c["anti_anneal_end"]]
            axs[0].plot(E, _mu(c), color=_hold[c["anti_hold_ratio"]], ls=dash, lw=1.4)
        _arm = _by_name[ex.DOSE_ARM]
        axs[0].plot(E, _mu(_arm), color=_arm_color, ls="-.", lw=1)
        axs[0].plot(E, ex.anchor_weight(E), color=_grey, lw=0.5, alpha=0.45, zorder=-1)
        axs[0].set_yscale("log")
        axs[0].set_ylim(1e-3, 1)
        axs[0].minorticks_off()
        axs[0].set_ylabel("weight", fontsize="x-small")
        axs[0].set_title("the factorial", fontsize="small", color=_grey)

        # The key, in the corner the schedules leave empty. Two columns because
        # the encoding has two independent parts: dash for the endpoint, color
        # for the hold ratio. The arm sits under the colors as the one curve that
        # is keyed by neither — a blank handle heads each column in place of a
        # section title, which matplotlib legends have no room for.
        def _key(label: str, ls="none", color=_grey, lw=1.4) -> plt.Line2D:
            return plt.Line2D([], [], label=label, ls=ls, color=color, lw=lw)

        axs[0].legend(
            handles=[
                _key("anneal end"),
                *(_key(f"{e:g}", ls=d) for e, d in _dashes.items()),
                _key("hold ratio"),
                *(_key(f"{h:.2f}", ls="-", color=col) for h, col in _hold.items()),
                _key("dose arm", ls="-.", color=_arm_color, lw=1),
            ],
            loc="upper right",
            ncols=2,
            fontsize=6.5,
            labelcolor=_grey,
            frameon=False,
            borderaxespad=0.4,
            handlelength=1.8,
            handletextpad=0.5,
            labelspacing=0.35,
            columnspacing=1.2,
        )

        # Right: μ·lr, whose integral is the dose, so the equal areas the arm was
        # built for are the thing the panel draws. Hatched: where the arm falls
        # short of `end50-hold03` early, and where it makes that back late. Those
        # two lobes have equal area — that equality *is* the dose match, and it
        # is what the log-weight panel cannot show.
        ax, ref = axs[1], _by_name[ex.DOSE_ARM_REFERENCE]
        _d_arm, _d_ref = _mu(_arm) * LR * 1e3, _mu(ref) * LR * 1e3
        _hatch = dict(facecolor="none", lw=0, hatch_linewidth=0.5)
        ax.fill_between(E, _d_ref, _d_arm, where=_d_arm < _d_ref, hatch="///", edgecolor=_grey, **_hatch)
        ax.fill_between(E, _d_ref, _d_arm, where=_d_arm > _d_ref, hatch="\\\\\\", edgecolor=_arm_color, **_hatch)
        # The un-matched late cell and the anchor, both in ghost ink: the arm's
        # dose is meaningful against the schedule it was scaled down from, and
        # the repulsion against the pull it works against.
        ax.plot(E, _mu(_by_name["end90-hold03"]) * LR * 1e3, color=_grey, lw=0.5, alpha=0.45, ls=_dashes[90.0])
        ax.plot(E, ex.anchor_weight(E) * LR * 1e3, color=_grey, lw=0.5, alpha=0.45, zorder=-1)
        ax.plot(E, _d_ref, color=_grey, lw=1.0, ls=_dashes[50.0])
        ax.plot(E, _d_arm, color=_arm_color, lw=1, ls="-.")
        ax.set_ylim(0, None)
        ax.set_ylabel("weight × lr / $10^{-3}$", fontsize="x-small")
        ax.set_title("the dose arm", fontsize="small", color=_grey)

        for ax in axs:
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_xlabel("epoch", fontsize="x-small")
            ax.tick_params(labelsize=7.5)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Measurements

    Like ex-2.1.7:

    - `named_holdout` exact match, against the in-experiment control (H1).
    - $m_{\text{op1}}$: the label-affinity-weighted alignment at op1 minus the
      cube mean, averaged over layers (H2, H3).
    - $\bar\alpha$: mean alignment over all 216 colors at op1, layer-averaged
      (H2, H3). This is the quantity the repulsive term is defined on.
    - Grading: Spearman $\rho$ against `redness` and Pearson $R^2$ against
      `sim¹·⁵` over the 216 colors (H2).
    - Per-run margin trajectories, for retention (H2).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Ex-2.1.7 quoted *retention* two ways under one name: the minimum
    final-over-peak margin across seeds in its Findings, and the mean across
    seeds in its Arms section. This report uses the **{RSTAT}**, and every
    later report should. The gate of H4(b) is worded per run ("for every
    anchored run…"), so the minimum is what it needs.
    """.replace("{RSTAT}", ex.RETENTION_STAT)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Gates and noise floors carry over from ex-2.1.7 with their thresholds and
    rationale, since we have the same architecture, the same measurement, and
    the same statistics.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **H1: Task cost.** Every condition is task-clean, meaning `named_holdout`
    exact match within {TASK} (absolute) of the seed mean of the in-experiment
    control.[^h1-standing] Partial: the `hold03` cells and the dose arm are
    clean and a `hold30` cell is not.

    **H2: An operating point exists.** Some task-clean factorial cell meets
    all four of the selectivity and containment conditions from ex-2.1.7 at
    once, which no condition in that experiment did:
    (a) containment, $\bar\alpha \le {ALPHA}$;
    (b) margin, $m_{\text{op1}} \ge {MARGIN}$;
    (c) grading, Spearman $\rho \ge {RHO}$ against `redness` or Pearson
    $R^2 \ge {R2}$ against `sim¹·⁵`;
    (d) retention, every run whose running maximum of $m_{\text{op1}}$ reaches
    {FLOOR} ending at $\ge {RET}\times$ that maximum.
    Partial, in decreasing strength: (a) is met at the looser {ALPHAP} (the
    halving level from ex-2.1.7) with (b)–(d) held; or (a) is met at {ALPHA}
    but the margin falls between the ex-2.1.7 value of {M50} and the gate.

    **H3: The level contributes at least as much containment as the timing.**
    We score this on two main effects[^main] on $\bar\alpha$, both taken as
    differences of means and reported with their sign. The hold-ratio effect is
    the nine runs at 0.03 minus the nine at 0.30. The anneal-endpoint effect is
    the six runs at endpoint 50 minus the six at endpoint 90. H3 holds when the
    hold-ratio effect is at least {AGATE} and at least as large as the endpoint
    effect.[^effect]

    Explicit outcomes that would fail H3:
    **1.** If the endpoint effect clears {AGATE} and is the larger, that would suggest the repulsion has to be present while the pull is strong, and no floor substitutes for that.
    **2.** If `{PRIMARY}` alone contains the drift while neither main effect clears {AGATE}, then each factor is blocked by the other.[^h3-fail-2]

    The sign of the interaction term of the factorial is considered but not scored. The level account predicts the endpoint effect shrinks at the high hold ratio, because a floor that never crosses the threshold makes the crossing time irrelevant. An endpoint effect of similar size at both hold ratios fits the timing account instead. The discussion should mention this if the H3 main effects come out close.

    [^h1-standing]: Ex-2.1.7 found no task cost anywhere, including at ten times this
        anchor weight, so this is a standing gate rather than an open question. We
        state it because a sustained repulsive floor is the one thing in this grid
        that has not been tried.

    [^h3-fail-2]: The second outcome is only reachable if the `hold30` row reverses the endpoint ordering. Ex-2.1.7 measured the `hold03` row moving from {A50} to {A90} across the endpoint on its own, and half of that span already falls inside the endpoint effect before the other row is counted.

    [^main]: A main effect is the average change from moving one factor,
        averaged over the levels of the other.

    [^effect]: Taking the ex-2.1.6 control spread of about 0.02 for
        $\bar\alpha$, the noise on the hold-ratio effect is near
        $0.02\sqrt{2}/\sqrt{9} \approx 0.009$, and on the endpoint effect,
        with six-run means, near $0.02\sqrt{2}/\sqrt{6} \approx 0.012$; so
        {AGATE} is four to five times either. Before reading the effects, we
        compute the spread from this experiment's own seeds and check that the
        anchored conditions hold it. If they do not, the observed spread
        replaces the control spread and the gate moves with it. It is the only
        threshold in this report allowed to move.
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
    mo.md(r"""
    ## Reproduction check
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// admonition | TODO
    A two-row table: `end50-hold03` and `end90-hold03` against `span-anti`
    and `span-anti-late` from ex-2.1.7, on $m_{\text{op1}}$ and $\bar\alpha$
    (seed means; references in `ex.EX217_REFERENCE`). These conditions are
    identical in every parameter, so the two experiments should agree on the
    margin within the seed-mean noise floor of {NOISE}. Ex-2.1.6 carries no
    such floor for $\bar\alpha$. The tolerance there is two seed-mean spreads
    computed from this experiment's own seeds, the same construction the H3
    footnote uses.

    Expected: agreement. A gap wider than about two noise units would mean
    something moved between the experiments that neither report accounts for.
    That would rule out reading this grid against the ex-2.1.7 numbers, which
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
    mo.md(r"""
    ## Task cost (H1)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
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
    mo.md(r"""
    ## An operating point (H2)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// admonition | TODO
    The scoring table: one row per factorial cell, columns $\bar\alpha$,
    $m_{\text{op1}}$, grading ($\rho$ and $R^2$), and retention (minimum
    across seeds), each marked against its gate, with a final column saying
    whether the cell meets all four. The ex-2.1.7 conditions `span-anti` and
    `span-anti-late` appear in the table as `end50-hold03` and `end90-hold03`.
    Column headers carry the desired direction (↑/↓, as in the glossary), and
    values that pass their gate are bolded.

    Expected under the level account: `{PRIMARY}` meets all four, and the
    `hold30` row generally beats the `hold03` row on containment. Under the
    timing account the `end90` column meets them instead and `end50-hold30`
    fails containment.

    A contrary result worth naming: containment improves and the margin falls
    with it, so no cell meets both. That would say the repulsion gets its
    containment by flattening the response rather than by clearing the axis.
    It is the same trade the ceiling arm in ex-2.1.7 found on the anchor
    weight, arriving from the repulsive side. The grading columns tell the two
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
    down), with the weight schedule of each cell drawn beneath it and the
    control in grey in every panel. The figure of this shape in ex-2.1.7 is
    the model.

    Expected under the level account: in the `hold03` row, $\bar\alpha$ leaves
    the control at an epoch that tracks the anneal endpoint of each cell; in
    the `hold30` row it does not leave at all. Under the timing account the
    `hold30` row departs too, just later.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Attribution (H3)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// admonition | TODO
    The factorial table on $\bar\alpha$: the six cell means, the two main
    effects with the {AGATE} gate marked, and the interaction. Repeat on
    $m_{\text{op1}}$ beside it, since a factor that contains the drift without
    holding the margin has not given us an operating point.

    Also state the seed spread this experiment actually measures for
    $\bar\alpha$, as the footnote to H3 requires, before reading either effect.

    Expected: the hold-ratio effect larger, and a negative interaction, with
    the endpoint mattering less at the high floor. A same-sized endpoint effect
    in both rows is the timing account, and is one of the two named misses.
    ///
    """.replace("{AGATE}", f"{ex.ALPHA_EFFECT_GATE:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The dose arm
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// admonition | TODO
    Three rows: `{REF}`, `end90-hold03`, and `{ARM}`, on $\bar\alpha$,
    $m_{\text{op1}}$, grading and retention, with the dose column from the
    Method table repeated, since it is the whole point of the comparison.

    The arm delivers the repulsion of `{REF}` on the timing of
    `end90-hold03`. If it tracks `end90-hold03`, the endpoint effect is timing
    and the extra dose is incidental. If it tracks `{REF}`, the endpoint effect
    is dose, and "anneal later" is a roundabout way of saying "use more".

    This arm scores no hypothesis. It is one condition with three seeds, and
    the peak ratio moves alongside the endpoint, so it separates the two
    accounts without quantifying either.
    ///
    """.replace("{REF}", ex.DOSE_ARM_REFERENCE).replace("{ARM}", ex.DOSE_ARM)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Secondary measurements

    /// admonition | TODO
    Carried from ex-2.1.7 for continuity; these score nothing. Redness read from
    the other 63 directions against the computed RGB floor of 0.863, and the
    extent of the color cube through depth. The timing arm in ex-2.1.7 was the
    highest of any condition on the first of these, while also being the
    best on every scored statistic. So the grid is worth checking for a trend
    that the single arm could not show.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    /// admonition | TODO
    Post hoc, planned after seeing the results; scores no hypothesis. We leave
    this empty in the preregistration.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    Written when the results land. It should answer these questions, and go no
    further than the data supports:

    1. Which operating point should the next experiments use, and on what
       evidence? Anything that pulls without a position oracle needs one, and
       so would an intervention experiment. $\bar\alpha$ is the number that
       predicts how much of the color cube an edit to the anchor axis would
       move.
    2. Does the schedule finding generalize past this architecture, or is it
       another set of keyframes fitted to one testbed? The M1 keyframes did
       not transfer, so it is worth stating what would make ours transfer.
    3. What is still missing. Containment is not exclusivity: ex-2.1.7 found
       redness as readable off the anchor axis as in the control, and nothing
       in this grid changes what that measurement means.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
