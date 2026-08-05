import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.8: Repulsion tuning",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo puts the notebook directory on sys.path, so the design constants come
    # from the experiment module rather than a copy of them in the prose.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import figure_html, light_dark, themed

    use_publisher(report_bundle(__file__))

    def load_results() -> tuple[dict, dict[str, np.ndarray]] | None:
        """Resolve metrics and the stacked per-cell arrays from the store, or None if unpublished."""
        store = project_store()
        arts = store.get_refs([ex.METRICS_REF, ex.ARRAYS_REF])
        m_art, a_art = arts[ex.METRICS_REF], arts[ex.ARRAYS_REF]
        if m_art is None or a_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            m_path, a_path = store.get_many([(m_art, Path(d) / "metrics.json"), (a_art, Path(d) / "arrays.npz")])
            with np.load(a_path) as z:
                arrays = {k: z[k] for k in z.files}
            metrics = json.loads(m_path.read_text())
        return metrics, arrays

    def load_geometry() -> dict | None:
        """The per-cell geometry pass, published under its own ref."""
        store = project_store()
        art = store.get_refs([ex.GEOMETRY_REF])[ex.GEOMETRY_REF]
        if art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            path = store.get_many([(art, Path(d) / "geometry.json")])[0]
            return json.loads(path.read_text())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.8: Repulsion tuning

    /// tip |
    <!-- tl;dr -->
    We tested anti-subspace schedules (trailing timing and strength) to find an operating point that contains the cube-wide drift without reducing selectivity.
    Holding the repulsion high for longer contains the drift and keeps the margin, which nothing before this did. The response still isn't graded enough to call it an operating point.
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
    | $\rho$ | *rank* grading: the Spearman correlation between the per-color $\alpha_c$ and `redness`. Asks whether the response puts the colors in the same order redness does, ignoring how large it is | ≥ {RHO} |
    | $R^2$ | *proportionality* grading: the squared Pearson correlation between $\alpha_c$ and `sim¹·⁵`, each color's cosine similarity to *red* raised to 1.5. Asks whether the response has that shape, ignoring its scale | ≥ {R2} |
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
        .replace("{RHO}", f"{ex.GRADE_RHO_GATE:g}")
        .replace("{R2}", f"{ex.GRADE_R2_GATE:g}")
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
def _(
    CONDS: list[str],
    CONTROL_ACC,
    EFFECT_GATE,
    HOLD_EFFECT,
    INTERACTION,
    INTERACTION_GATE,
    acc,
    alpha_bar,
    grading,
    m_op1,
    retention,
):
    _p = ex.PRIMARY
    _best_r2 = max(ex.FACTORIAL_NAMES, key=lambda c: grading(c)[1])
    mo.md(rf"""
    **H1 (task cost) — holds.** Largest `named_holdout` exact-match gap from the
    control, across all {len(CONDS)} conditions:
    {max(abs(acc(c).mean() - CONTROL_ACC) for c in CONDS):.4f}. Gate:
    {ex.TASK_GATE:g}.

    **H2 (an operating point exists) — fails, on grading.** `{_p}`, the
    preregistered primary cell, meets containment
    ($\bar\alpha = {alpha_bar(_p).mean():.3f}$, gate {ex.MEAN_ALIGN_GATE:g}),
    margin ({m_op1(_p).mean():.3f}, gate {ex.MARGIN_GATE:g}) and retention
    ({retention(_p):.2f}, gate {ex.H4_RETENTION:g}). No cell in the grid reaches
    the grading gate of {ex.GRADE_R2_GATE:g} on either track; the best R² is
    {grading(_best_r2)[1]:.2f}. Neither named partial applies.

    **H3 (a sustained floor makes the anneal endpoint redundant) — holds.**
    Hold-ratio main effect on $\bar\alpha$: {HOLD_EFFECT:+.3f}, gate
    {EFFECT_GATE:.3f}. Interaction: {INTERACTION:+.3f}, gate
    {INTERACTION_GATE:.3f} — a pass by
    {(INTERACTION - INTERACTION_GATE) / INTERACTION_GATE * 100:.0f}%. Both gates
    are the preregistered values rescaled by this experiment's own seed spread,
    as the H3 footnote directs.

    **Containment is now newly reachable.** `{_p}` is the first
    condition in M2 to hold $\bar\alpha$ near the control while keeping the
    margin. The best in ex-2.1.7 was
    {ex.EX217_REFERENCE["end90-hold03"]["alpha"]:.2f} and its gate went unmet
    everywhere.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this report
    The method, hypotheses and decision thresholds were written and frozen in Git before the experiment ran, and the results below replace the placeholders that stood in for them. Anything conceived after seeing the data goes under *Exploratory analyses*.
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
        90 is the timing arm from ex-2.1.7. So two cells repeat conditions we
        have already measured.

    - **B.** Hold ratio

        The level it settles at. Levels: {HOLDS}.
        Ratio 0.03 is the M1 level. Ratio 0.30 is three times the level at
        which ex-2.1.7 saw containment give way, so on the level account it
        should never cross the threshold at all.

    We leave out endpoint 100: the anneal would run to the end of training,
    leaving no hold window for the ratio to act on, so that row would cross one
    factor against nothing and dilute its main effect.

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
    halving level from ex-2.1.7) while (b)–(d) all hold; or (a) is met at
    {ALPHA} but the margin falls between the ex-2.1.7 value of {M50} and the
    gate.

    **H3: A sustained floor makes the anneal endpoint redundant.** This is
    the prediction that sets the level account apart. If containment is lost at
    a threshold, then the endpoint only has leverage in the row whose floor
    crosses that threshold. We score H3 on $\bar\alpha$ with two conditions,
    each a difference of means reported with its sign.
    (a) The hold-ratio main effect,[^main] the nine runs at 0.03 minus the nine
    at 0.30, is at least {AGATE}. That says the floor does something at all.
    (b) The interaction is at least {IGATE}: the end50 − end90 difference
    within the 0.03 row, minus the same difference within the 0.30
    row.[^effect]
    Condition (a) keeps (b) from passing without any containment behind it: a
    *negative* endpoint effect in the 0.30 row would inflate the interaction on
    its own.

    Two outcomes would fail H3.
    **1.** The endpoint effect within a row is the three runs at endpoint 50
    minus the three at endpoint 90. If it clears {AGATE} in both rows while the
    interaction stays under {IGATE}, that is the timing account: the repulsion
    has to be present while the pull is strong, and no floor stands in for it.
    **2.** If `{PRIMARY}` alone contains the drift while (a) fails, then each
    factor is blocked by the other.[^h3-fail-2]

    We also report, without scoring it, whether the hold-ratio effect is at
    least as large as the endpoint effect. In the clean outcomes this ordering
    agrees with the interaction. In the partial ones it can disagree, because a
    floor that slows the climb without preventing it can come out ahead on the
    ordering while the interaction stays small. Where they disagree, we trust
    the interaction.

    [^h1-standing]: Ex-2.1.7 found no task cost anywhere, including at ten times this
        anchor weight, so this is a standing gate rather than an open question. We
        state it because a sustained repulsive floor is the one thing in this grid
        that has not been tried.

    [^h3-fail-2]: Suppose the `hold03` row repeats its ex-2.1.7 span of {A50}
        to {A90}. Then (a) can only fail if the mean of the 0.30 row is nearly
        as high, and with `{PRIMARY}` contained that in turn requires a
        wide endpoint spread inside the 0.30 row itself, of the same sign as
        the spread in the 0.03 row. Such a spread pulls the interaction toward
        zero, so this miss would take (b) down with (a).

    [^main]: A main effect is the average change from moving one factor,
        averaged over the levels of the other.

    [^effect]: Start from the ex-2.1.6 control spread of about 0.02 for
        $\bar\alpha$. The hold-ratio effect compares nine-run means, so its
        noise is near $0.02\sqrt{2}/\sqrt{9} \approx 0.009$, and {AGATE} is
        about five times that. The interaction is a difference of two
        differences of three-run cell means, with noise near
        $0.02 \cdot 2/\sqrt{3} \approx 0.023$, so {IGATE} is about four times
        that. The endpoint effect within one row is a difference of three-run
        means, with noise near $0.02\sqrt{2}/\sqrt{3} \approx 0.016$, so
        {AGATE} is about three times that where named miss 1 reads it.
        What the level account expects for the interaction is about 0.20,
        twice the gate: the full {A50} to {A90} span in the 0.03 row and none
        of it in the 0.30 row. Before reading either statistic, we compute the
        spread from the seeds of this experiment and check that the anchored
        conditions hold it. If they do not, the observed spread replaces the
        control spread and the gates move with it. They are the only
        thresholds in this report allowed to move.
    """.replace("{TASK}", f"{ex.TASK_GATE:g}")
        .replace("{ALPHAP}", f"{ex.MEAN_ALIGN_PARTIAL:g}")
        .replace("{ALPHA}", f"{ex.MEAN_ALIGN_GATE:g}")
        .replace("{MARGIN}", f"{ex.MARGIN_GATE:g}")
        .replace("{RHO}", f"{ex.GRADE_RHO_GATE:g}")
        .replace("{R2}", f"{ex.GRADE_R2_GATE:g}")
        .replace("{FLOOR}", f"{ex.H4_FLOOR:g}")
        .replace("{RET}", f"{ex.H4_RETENTION:g}")
        .replace("{AGATE}", f"{ex.ALPHA_EFFECT_GATE:g}")
        .replace("{IGATE}", f"{ex.ALPHA_INTERACTION_GATE:g}")
        .replace("{PRIMARY}", ex.PRIMARY)
        .replace("{A50}", f"{ex.EX217_REFERENCE['end50-hold03']['alpha']:.2f}")
        .replace("{A90}", f"{ex.EX217_REFERENCE['end90-hold03']['alpha']:.2f}")
        .replace("{M50}", f"{ex.EX217_REFERENCE['end50-hold03']['m_op1']:.2f}")
    )
    # REVIEW: two corrections to the H3 rewrite below, neither a gate value.
    # (1) Named miss 1 defined the endpoint effect as six-run means but then
    # read it "in both rows separately", which is three runs against three. It
    # now says three, and the footnote gains that statistic's noise
    # (0.02·√2/√3 ≈ 0.016, so 0.05 is ~3× rather than the ~5× that holds for
    # the nine-run main effect). The gate is unchanged and still clears noise.
    # (2) The `h3-fail-2` footnote called miss 2 and condition (a)'s guard "the
    # same geometry from two sides". They sit on opposite sides: the guard
    # covers a *negative* endpoint effect in the 0.30 row, which inflates the
    # interaction, while miss 2 needs a large *positive* one (PRIMARY contained
    # forces end50/end70 high), which deflates it. The footnote now says miss 2
    # takes (b) down with (a).
    # REVIEW: H3's gate moved from the main-effect ordering to the interaction,
    # with the hold-ratio main effect kept as a guard. The ordering compares two
    # marginal sizes and can pass in a world where the floor only slows the
    # climb (hold effect large, interaction small), which is not the level
    # account; the interaction is that account's distinguishing prediction, and
    # a review round had already licensed the discussion to lean on it whenever
    # the main effects came out close — a decisive statistic should carry a
    # gate. Power: predicted signal ~0.20 against ~0.023 noise, gate 0.1. The
    # ordering demotes to a reported observation. Verify: the level-account
    # expectation derives from EX217_REFERENCE (0.33 → 0.13 in the hold03 row);
    # if the reproduction check fails, the power argument goes with it.
    # REVIEW: three earlier changes to H1 and H3, none to a gate value.
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
    _res = load_results()
    mo.stop(_res is None, mo.md("_Results are not published yet; the analysis cells below render once they are._"))
    assert _res is not None
    metrics, arrays = _res
    cells = {c["label"]: c for c in metrics["cells"]}

    CONDS: list[str] = [c["name"] for c in ex.CONDITIONS]
    ANCHORED = [c for c in CONDS if c != "lam0"]
    BY_NAME = {c["name"]: c for c in ex.CONDITIONS}
    # The factorial as a grid, so a row or column can be addressed by its level
    # rather than by re-deriving names from the two constant lists each time.
    ROW = {h: [f"end{e:.0f}-hold{h * 100:02.0f}" for e in ex.ANTI_ANNEAL_ENDS] for h in ex.ANTI_HOLD_RATIOS}
    HOLD_LO, HOLD_HI = ex.ANTI_HOLD_RATIOS
    END_LO, END_HI = ex.ANTI_ANNEAL_ENDS[0], ex.ANTI_ANNEAL_ENDS[-1]

    def over_seeds(cond: str, fn) -> np.ndarray:
        """One value per seed for a condition, in seed order."""
        return np.array([fn(cells[f"{cond}-s{s}"]) for s in ex.SEEDS])

    def acc(cond: str, set_name: str = "named_holdout", key: str = "accuracy") -> np.ndarray:
        return over_seeds(cond, lambda c: c["sets"][set_name][key])

    def alpha_map(cond: str) -> np.ndarray:
        """Seed-mean alignment map for a condition: (layers, colors, positions)."""
        return np.mean([arrays[f"{cond}-s{s}/alpha"] for s in ex.SEEDS], axis=0)

    def m_op1(cond: str) -> np.ndarray:
        """Per-seed layer-mean margin at op1 — the statistic H2(b) reads."""
        return over_seeds(cond, lambda c: c["m_op1"])

    def alpha_bar(cond: str) -> np.ndarray:
        """Per-seed mean alignment over all 216 colors at op1 — the containment statistic."""
        return over_seeds(cond, lambda c: c["alpha_mean_op1"])

    def traj(cond: str, seed: int, key: str) -> np.ndarray:
        return np.asarray(cells[f"{cond}-s{seed}"]["traj"][key], dtype=float)

    def grading(cond: str) -> tuple[float, float]:
        """The two H2(c) statistics, read off the seed-mean per-color response at op1.

        Graded on the seed mean rather than per seed and averaged, which is what
        ex-2.1.7 did — so the reproduction check compares like with like.
        """
        a = alpha_map(cond)[:, :, 0].mean(axis=0)
        return ex.spearman(a, ex.REDNESS), ex.r2_sim(a)

    def retention(cond: str) -> float:
        """Lowest final-over-peak margin across the seeds that reached the floor; 0 if none did.

        The minimum, per `ex.RETENTION_STAT`: H2(d)'s gate is worded per run.
        """
        r = [t[-1] / t.max() for s in ex.SEEDS if (t := traj(cond, s, "m_op1")).max() >= ex.H4_FLOOR]
        return min(r) if r else 0.0

    geometry = load_geometry() or {}
    return (
        ANCHORED,
        BY_NAME,
        CONDS,
        END_HI,
        END_LO,
        HOLD_HI,
        HOLD_LO,
        ROW,
        acc,
        alpha_bar,
        alpha_map,
        cells,
        geometry,
        grading,
        m_op1,
        retention,
        traj,
    )


@app.cell(hide_code=True)
def _(
    CONDS: list[str],
    END_HI,
    END_LO,
    HOLD_HI,
    HOLD_LO,
    ROW,
    acc,
    alpha_bar,
    grading,
    m_op1,
    retention,
):
    CONTROL_ACC = float(acc("lam0").mean())
    TASK_CLEAN = {c: bool(abs(acc(c).mean() - CONTROL_ACC) <= ex.TASK_GATE) for c in CONDS}

    # The seed spread H3's footnote asks for, pooled over the six factorial
    # conditions, and the factor the gates move by if the anchored cells do not
    # hold the ex-2.1.6 control spread of 0.02. This is the one rescaling the
    # preregistration allows, so it is computed before either H3 statistic.
    CONTROL_SPREAD = 0.02
    ALPHA_SPREAD = float(np.sqrt(np.mean([alpha_bar(c).var(ddof=1) for c in ex.FACTORIAL_NAMES])))
    GATE_SCALE = max(1.0, ALPHA_SPREAD / CONTROL_SPREAD)
    EFFECT_GATE = ex.ALPHA_EFFECT_GATE * GATE_SCALE
    INTERACTION_GATE = ex.ALPHA_INTERACTION_GATE * GATE_SCALE

    def end_effect(hold: float) -> float:
        """The end50 − end90 difference of three-run means within one row."""
        lo, hi = (f"end{e:.0f}-hold{hold * 100:02.0f}" for e in (END_LO, END_HI))
        return float(alpha_bar(lo).mean() - alpha_bar(hi).mean())

    def h2_conditions(cond: str) -> dict[str, bool]:
        """The four H2 conditions for one cell, each under the name the hypothesis gives it."""
        rho, r2 = grading(cond)
        return {
            "containment": bool(alpha_bar(cond).mean() <= ex.MEAN_ALIGN_GATE),
            "margin": bool(m_op1(cond).mean() >= ex.MARGIN_GATE),
            "grading": bool(rho >= ex.GRADE_RHO_GATE or r2 >= ex.GRADE_R2_GATE),
            "retention": bool(retention(cond) >= ex.H4_RETENTION),
        }

    # H3(a): the nine runs at the low hold ratio minus the nine at the high one.
    HOLD_EFFECT = float(
        np.mean([alpha_bar(c).mean() for c in ROW[HOLD_LO]]) - np.mean([alpha_bar(c).mean() for c in ROW[HOLD_HI]])
    )
    INTERACTION = end_effect(HOLD_LO) - end_effect(HOLD_HI)
    # H2 is scored on the factorial only; the dose arm and the control are not
    # candidates for an operating point.
    MEETS = {c: TASK_CLEAN[c] and all(h2_conditions(c).values()) for c in ex.FACTORIAL_NAMES}
    return (
        ALPHA_SPREAD,
        CONTROL_ACC,
        CONTROL_SPREAD,
        EFFECT_GATE,
        GATE_SCALE,
        HOLD_EFFECT,
        INTERACTION,
        INTERACTION_GATE,
        MEETS,
        TASK_CLEAN,
        end_effect,
        h2_conditions,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reproduction check
    """)
    return


@app.cell(hide_code=True)
def _(alpha_bar, m_op1):
    # REVIEW: the check compared two statistics but quoted a tolerance for one.
    # Rather than import a floor for ᾱ that no earlier experiment measured, the
    # tolerance is derived in-experiment, as H3's footnote already does for the
    # effect gate. This matters because H2 turns on ᾱ moving 0.13 → 0.1, a
    # distance of the same order as the seed spread. Verify: ex-2.1.7's own
    # per-seed ᾱ spread on these two conditions was about 0.011–0.015.
    def _tol(cond: str) -> float:
        """Two seed-mean spreads for ᾱ, from this experiment's own seeds."""
        return 2 * float(alpha_bar(cond).std(ddof=1)) / np.sqrt(len(ex.SEEDS))

    _rows = "".join(
        f"<tr><th><code>{c}</code></th>"
        f"<td class='num'>{m_op1(c).mean():.3f}</td><td class='num'>{r['m_op1']:.2f}</td>"
        f"<td class='num'>{m_op1(c).mean() - r['m_op1']:+.3f}</td>"
        f"<td class='num'>{alpha_bar(c).mean():.3f}</td><td class='num'>{r['alpha']:.2f}</td>"
        f"<td class='num'>{alpha_bar(c).mean() - r['alpha']:+.3f}</td>"
        f"<td class='num'>±{_tol(c):.3f}</td></tr>"
        for c, r in ex.EX217_REFERENCE.items()
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th>
        <th class="num">m<sub>op1</sub> here</th><th class="num">ex-2.1.7</th><th class="num">Δ</th>
        <th class="num">ᾱ here</th><th class="num">ex-2.1.7</th><th class="num">Δ</th>
        <th class="num">ᾱ tolerance</th>
      </tr></thead>
      <tbody>{_rows}</tbody>
    </table></div>
    """
    _caption = f"""
    The two conditions this grid shares with ex-2.1.7, against that report's published
    seed means. <code>end50-hold03</code> repeats its <code>span-anti</code> and
    <code>end90-hold03</code> its <code>span-anti-late</code>. The margin
    tolerance is the carried seed-mean noise floor, {ex.NOISE_SEED_MEAN:g}; the
    ᾱ tolerance is two seed-mean spreads from this experiment's own seeds, in
    the last column.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(alpha_bar, grading, m_op1):
    _dm = max(abs(m_op1(c).mean() - r["m_op1"]) for c, r in ex.EX217_REFERENCE.items())
    _da = max(abs(alpha_bar(c).mean() - r["alpha"]) for c, r in ex.EX217_REFERENCE.items())
    mo.md(rf"""
    Both conditions reproduce. The margins land {_dm:.3f} from the published values
    against a floor of {ex.NOISE_SEED_MEAN:g}, and ᾱ within {_da:.3f}, inside
    the tolerance in either row. Grading agrees as well: `end90-hold03` scores
    R² = {grading("end90-hold03")[1]:.2f} here, the value ex-2.1.7 published for
    the same cell.

    So the two experiments measure the same thing, and this grid can be read
    against the ex-2.1.7 numbers — which the H2 partials and the H3 power
    argument both rely on.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost (H1)
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], CONTROL_ACC, TASK_CLEAN, acc):
    def _cell(cond: str, set_name: str, key: str, fmt: str = "{:.3f}") -> str:
        v = acc(cond, set_name, key)
        return f"{fmt.format(v.mean())} <span class='range'>±{(v.max() - v.min()) / 2:.3f}</span>"

    _rows = "".join(
        f"<tr><th><code>{c}</code></th>"
        f"<td class='num'>{_cell(c, 'named_seen', 'accuracy')}</td>"
        f"<td class='num'>{_cell(c, 'named_holdout', 'accuracy')}</td>"
        f"<td class='num'>{_cell(c, 'named_holdout', 'nll')}</td>"
        f"<td class='num'>{_cell(c, 'open', 'guess_dist')}</td>"
        f"<td class='num'>{acc(c).mean() - CONTROL_ACC:+.4f}</td>"
        f"<td>{'clean' if TASK_CLEAN[c] else 'not clean'}</td></tr>"
        for c in CONDS
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">seen EM ↑</th><th class="num">holdout EM ↑</th>
        <th class="num">holdout NLL ↓</th><th class="num"><code>open</code> dist ↓</th>
        <th class="num">Δ holdout EM</th><th>H1 gate</th>
      </tr></thead>
      <tbody>{_rows}</tbody>
    </table></div>
    """
    _caption = f"""
    Behavior by condition. Each cell is the seed mean, with half the seed range
    beside it. EM is exact match on the answer token, NLL its surprise in nats,
    and <code>open</code> dist the RGB distance from the guessed color to the
    true mix on pairs with no named answer. Δ holdout EM is the signed gap to
    the in-experiment control, which the H1 gate scores at {ex.TASK_GATE:g}.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], CONTROL_ACC, HOLD_HI, ROW, acc):
    _worst = max(CONDS, key=lambda c: abs(acc(c).mean() - CONTROL_ACC))
    _hi = max(ROW[HOLD_HI], key=lambda c: abs(acc(c).mean() - CONTROL_ACC))
    mo.md(rf"""
    **H1 holds.** The largest gap from the control is
    {abs(acc(_worst).mean() - CONTROL_ACC):.4f}, at `{_worst}`, against a gate
    of {ex.TASK_GATE:g}. Every condition solves the task.

    The preregistration named the `hold{HOLD_HI * 100:02.0f}` row as the place a
    task cost would show up, since it sustains a repulsive weight no earlier
    experiment held. It does not: the worst cell in that row is `{_hi}` at
    {abs(acc(_hi).mean() - CONTROL_ACC):.4f}, the same size as the gaps
    elsewhere. Holding the repulsion an order of magnitude above the M1 floor for
    the rest of training costs nothing the task loss can see.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## An operating point (H2)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Four conditions have to hold at once. The table below scores them together, with values that pass their gate in bold.
    """)
    return


@app.cell(hide_code=True)
def _(BY_NAME, MEETS, alpha_bar, grading, h2_conditions, m_op1, retention):
    def _mark(v: float, gate: float, up: bool, fmt: str = "{:.3f}") -> str:
        ok = v >= gate if up else v <= gate
        return f"<strong>{fmt.format(v)}</strong>" if ok else fmt.format(v)

    def _row(c: str) -> str:
        rho, r2 = grading(c)
        cell = BY_NAME[c]
        # Name every condition the cell misses, not just the first: a row that
        # fails containment and grading alike should say so.
        missed = ", ".join(k for k, ok in h2_conditions(c).items() if not ok)
        return (
            f"<tr><th><code>{c}</code></th>"
            f"<td class='num'>{cell['anti_anneal_end']:.0f}</td>"
            f"<td class='num'>{cell['anti_hold_ratio']:.2f}</td>"
            f"<td class='num'>{_mark(alpha_bar(c).mean(), ex.MEAN_ALIGN_GATE, up=False)}"
            f" <span class='range'>±{alpha_bar(c).std(ddof=1):.3f}</span></td>"
            f"<td class='num'>{_mark(m_op1(c).mean(), ex.MARGIN_GATE, up=True)}"
            f" <span class='range'>±{m_op1(c).std(ddof=1):.3f}</span></td>"
            f"<td class='num'>{_mark(rho, ex.GRADE_RHO_GATE, up=True, fmt='{:.2f}')}</td>"
            f"<td class='num'>{_mark(r2, ex.GRADE_R2_GATE, up=True, fmt='{:.2f}')}</td>"
            f"<td class='num'>{_mark(retention(c), ex.H4_RETENTION, up=True, fmt='{:.2f}')}</td>"
            f"<td>{'all four' if MEETS[c] else missed}</td></tr>"
        )

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>cell</th><th class="num">end</th><th class="num">hold</th>
        <th class="num">ᾱ ↓</th><th class="num">m<sub>op1</sub> ↑</th>
        <th class="num">ρ ↑</th><th class="num">R² ↑</th><th class="num">retention ↑</th>
        <th>H2</th>
      </tr></thead>
      <tbody>{"".join(_row(c) for c in ex.FACTORIAL_NAMES)}</tbody>
    </table></div>
    """
    _caption = f"""
    The H2 scoring table, one row per factorial cell. <strong>Bold</strong>
    marks a value that passes its gate: ᾱ ≤ {ex.MEAN_ALIGN_GATE:g},
    m<sub>op1</sub> ≥ {ex.MARGIN_GATE:g}, ρ ≥ {ex.GRADE_RHO_GATE:g} or R² ≥
    {ex.GRADE_R2_GATE:g}, retention ≥ {ex.H4_RETENTION:g}. ᾱ and m<sub>op1</sub>
    carry the seed standard deviation; grading is read off the seed-mean
    response, and retention is the minimum across seeds. The last column names
    every condition a cell misses, or reports that it meets all four.
    <code>end50-hold03</code> and <code>end90-hold03</code> are ex-2.1.7's
    <code>span-anti</code> and <code>span-anti-late</code>.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(HOLD_HI, HOLD_LO, ROW, alpha_bar, grading, m_op1, retention):
    _best_r2 = max(ex.FACTORIAL_NAMES, key=lambda c: grading(c)[1])
    _best_rho = max(ex.FACTORIAL_NAMES, key=lambda c: grading(c)[0])
    _contained = [c for c in ex.FACTORIAL_NAMES if alpha_bar(c).mean() <= ex.MEAN_ALIGN_GATE]
    _p = ex.PRIMARY
    mo.md(rf"""
    **H2 fails, on grading.** The cell the preregistration named,
    `{_p}`, meets three of the four conditions: it contains the drift at
    ᾱ = {alpha_bar(_p).mean():.3f} against a gate of {ex.MEAN_ALIGN_GATE:g},
    holds the margin at {m_op1(_p).mean():.3f} against {ex.MARGIN_GATE:g}, and
    retains {retention(_p):.2f}× its peak against {ex.H4_RETENTION:g}. But its
    response is graded at ρ = {grading(_p)[0]:.2f} and R² = {grading(_p)[1]:.2f},
    and the gate is {ex.GRADE_R2_GATE:g} on either track.

    Every cell misses on grading. The best R² in the grid is
    {grading(_best_r2)[1]:.2f} (`{_best_r2}`) and the best ρ is
    {grading(_best_rho)[0]:.2f} (`{_best_rho}`), so no cell comes within
    {ex.GRADE_R2_GATE - grading(_best_r2)[1]:.2f} of passing. Neither named
    partial applies, since both assume grading holds and vary the containment
    or margin around it.

    The contrary result the preregistration named — containment bought by
    flattening the response — is not what happened. As the anneal is stretched, ᾱ falls
    and the margin rises together: `{ROW[HOLD_LO][0]}` sits at
    ᾱ {alpha_bar(ROW[HOLD_LO][0]).mean():.3f} with margin
    {m_op1(ROW[HOLD_LO][0]).mean():.3f}, and `{ROW[HOLD_LO][-1]}` at
    {alpha_bar(ROW[HOLD_LO][-1]).mean():.3f} with {m_op1(ROW[HOLD_LO][-1]).mean():.3f}.
    Grading rises with them, from {grading(ROW[HOLD_LO][0])[1]:.2f} to
    {grading(ROW[HOLD_LO][-1])[1]:.2f}. So stretching the anneal buys
    selectivity, and just stops short of the level H2 asked for.

    Containment on its own is now reachable, which it was not in ex-2.1.7.
    One cell of the six falls to ᾱ ≤ {ex.MEAN_ALIGN_GATE:g}
    ({", ".join(f"`{c}`" for c in _contained)}), and a second comes within
    {min(alpha_bar(c).mean() for c in ROW[HOLD_HI] if c not in _contained) - ex.MEAN_ALIGN_GATE:.3f}
    of it. That experiment's best was
    {ex.EX217_REFERENCE["end90-hold03"]["alpha"]:.2f}, and its gate went unmet
    everywhere. Both of the near cells are in the
    `hold{HOLD_HI * 100:02.0f}` row.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The endpoint table says where each run finished. The trajectories say how it
    got there, which separates the two accounts H3 is built on.
    """)
    return


@app.cell(hide_code=True)
def _(BY_NAME, HOLD_HI, HOLD_LO, ROW, traj):
    _KEYS = ("alpha_op1", "m_op1")
    # Seed means once, in the cell: @themed calls the plot function twice.
    _MEAN = {c: {k: np.mean([traj(c, s, k) for s in ex.SEEDS], axis=0) for k in _KEYS} for c in ex.FACTORIAL_NAMES}
    _MEAN["lam0"] = {k: np.mean([traj("lam0", s, k) for s in ex.SEEDS], axis=0) for k in _KEYS}
    _EPOCHS = np.mean([traj(ex.PRIMARY, s, "epoch") for s in ex.SEEDS], axis=0)
    # The schedules on a finer grid than the recorded trajectory, since the
    # anneal's shape is the thing the lower panels exist to show.
    _SE = np.linspace(0.0, float(ex.SCHEDULER.epochs), 601)
    _ANCHOR = ex.anchor_weight(_SE)
    _ANTI = {
        c: ex.anti_subspace_weight(
            _SE,
            lam=BY_NAME[c]["lam"],
            anneal_end=BY_NAME[c]["anti_anneal_end"],
            hold_ratio=BY_NAME[c]["anti_hold_ratio"],
            peak_ratio=BY_NAME[c]["anti_peak_ratio"],
        )
        for c in ex.FACTORIAL_NAMES
    }
    # Endpoint across, hold ratio down, as the design reads. Each cell gets a
    # trajectory panel with its own weight schedule directly beneath, and the
    # spacer row keeps the two hold-ratio blocks apart.
    _GRID = [
        ROW[HOLD_LO],
        [f"w-{c}" for c in ROW[HOLD_LO]],
        [".", ".", "."],
        ROW[HOLD_HI],
        [f"w-{c}" for c in ROW[HOLD_HI]],
    ]
    _PANELS = [*ROW[HOLD_LO], *ROW[HOLD_HI]]
    _LEFT = (ROW[HOLD_LO][0], ROW[HOLD_HI][0])

    @themed(
        name="trajectories",
        alt_text="""
            Six pairs of panels in a three-by-two grid, sharing the epoch axis from 0 to 100.
            Columns are anneal endpoints 50, 70 and 90; the upper row of pairs is hold
            ratio 0.03 and the lower is 0.30. In each pair the upper panel plots mean
            alignment as a solid line and margin as a dashed line, over a flat grey
            control pair near zero; the shorter log-scale panel beneath plots that
            cell's two weight schedules, anchor in red and repulsion in blue.
            In the upper row of pairs, the solid line stays low while the blue
            repulsion curve is above the red anchor curve, then climbs steeply once the
            blue curve drops below it — later in each successive column, and reaching a
            lower final height. In the lower row the blue curve settles only a little
            below the red one rather than far below it, and the solid line rises
            gently in all three columns without ever climbing steeply. The dashed
            margin line climbs early and stays up in every panel.
        """,
        caption=r"""
            **Training dynamics across the grid.** Columns are the anneal
            endpoint, rows the hold ratio. Above each pair: seed means of the
            two cosines on the anchor axis, on one shared scale. Solid is
            $\bar\alpha$, the mean alignment over all 216 colors at op1; dashed
            is the margin $m_{\text{op1}}$; the grey pair is the un-anchored
            control. The caret on the left spine is the containment gate, the
            one on the right the margin floor that retention is scored from.
            Below each: the weight schedule that cell trained under, anchor
            $\lambda_\mathrm{a}$ in red and repulsion
            $\lambda_{\bar{\mathrm{s}}}$ in blue, on a log scale shared by all
            six so schedules compare across the grid as well as down a pair.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axd = plt.subplot_mosaic(
            _GRID, figsize=(7.5, 6), height_ratios=[3, 1.5, 0.45, 3, 1.5], sharex=True,
        )  # fmt: skip
        _grey = light_dark("#999", "#777")
        _gate = light_dark("#555", "#bbb")
        _term = {"anchor": light_dark("#c33", "#e66"), "anti": light_dark("#36c", "#7af")}
        # Color keys the hold ratio, as in the schedules figure at the top, so a
        # row of this grid and a curve of that one read as the same condition.
        _hold = {HOLD_LO: light_dark("#1f6fb4", "#5fa8dd"), HOLD_HI: light_dark("#b4531f", "#dd8f5f")}
        _dash = (0, (6, 1))
        # One y-scale everywhere, so a line's height means the same in every panel.
        top = max(max(np.max(_MEAN[c][k]) for k in _KEYS) for c in ex.FACTORIAL_NAMES)
        for cond in _PANELS:
            ax, ink = axd[cond], _hold[BY_NAME[cond]["anti_hold_ratio"]]
            # Carets on the spines rather than rules across the panel: the
            # containment gate belongs to ᾱ and the floor to m, and putting them
            # on opposite spines tells them apart without a second color.
            ax.plot(0, ex.MEAN_ALIGN_GATE, marker=9, ms=3, color=_gate, clip_on=False, zorder=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
            ax.plot(1, ex.H4_FLOOR, marker=8, ms=3, color=_gate, clip_on=False, zorder=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
            ax.plot(_EPOCHS, _MEAN["lam0"]["alpha_op1"], color=_grey, lw=1.1, zorder=3)
            ax.plot(_EPOCHS, _MEAN["lam0"]["m_op1"], color=_grey, lw=1.1, ls=_dash, zorder=3)
            ax.plot(_EPOCHS, _MEAN[cond]["m_op1"], color=ink, lw=1.4, ls=_dash, zorder=4)
            ax.plot(_EPOCHS, _MEAN[cond]["alpha_op1"], color=ink, lw=1.7, zorder=5)
            ax.set_title(f"end {BY_NAME[cond]['anti_anneal_end']:.0f} · hold {BY_NAME[cond]['anti_hold_ratio']:.2f}",
                         fontsize=9.5, color=ink)  # fmt: skip
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_ylim(-0.08, top * 1.08)
            ax.tick_params(labelbottom=False, labelleft=cond in _LEFT)
            ax.set_ylabel("cosine", fontsize="x-small") if cond in _LEFT else None

        import matplotlib.patheffects as pe

        _halo = [pe.withStroke(linewidth=2.5, foreground=light_dark("#ffffff", "#000000"))]
        axd[_LEFT[0]].annotate(
            "control", (58, _MEAN["lam0"]["alpha_op1"][-1]), fontsize=7.5, color=_grey, va="center", ha="center",
            path_effects=_halo,
        )  # fmt: skip
        # Key the two line styles once, in neutral ink: within a panel, color
        # carries the hold ratio and nothing else.
        fig.legend(
            [plt.Line2D([], [], color=_grey, lw=1.7), plt.Line2D([], [], color=_grey, lw=1.5, ls=_dash)],
            [
                rf"$\bar\alpha$, contained at the left caret ({ex.MEAN_ALIGN_GATE:g})",
                rf"$m_{{\text{{op1}}}}$, floored at the right caret ({ex.H4_FLOOR:g})",
            ],
            fontsize=8.5, frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            labelcolor=_grey, handlelength=2.4,
        )  # fmt: skip

        for cond in _PANELS:
            ax = axd[f"w-{cond}"]
            ax.plot(_SE, _ANCHOR, color=_term["anchor"], lw=1.4)
            ax.plot(_SE, _ANTI[cond], color=_term["anti"], lw=1.4)
            ax.set_yscale("log")
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_ylim(2e-4, 4)
            ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
            ax.minorticks_off()  # a decade of unlabelled minor ticks is a smear at this height
            ax.tick_params(labelbottom=True, labelleft=cond in _LEFT, labelsize=7.5)
            ax.set_ylabel("weight", fontsize="x-small") if cond in _LEFT else None
        for _c in _GRID[-1]:
            axd[_c].set_xlabel("epoch", fontsize="x-small")
        # Name the two terms once, on the panel where they are furthest apart.
        for _x, _y, _txt, _k in ((64, 0.13, r"$\lambda_\mathrm{a}$", "anchor"), (30, 0.5, r"$\lambda_{\bar{\mathrm{s}}}$", "anti")):  # fmt: skip
            axd[f"w-{ROW[HOLD_LO][0]}"].annotate(_txt, (_x, _y), fontsize=9, color=_term[_k], va="bottom")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(BY_NAME, HOLD_HI, HOLD_LO, ROW, alpha_bar, traj):
    # Two epochs per cell, both read the same way for every condition. The
    # crossing comes from the schedule functions and the departure from the
    # recorded trajectory, so neither is fitted.
    _E = np.linspace(0.0, float(ex.SCHEDULER.epochs), 4001)
    _A = ex.anchor_weight(_E)

    def _crossing(cond: str) -> float:
        """First epoch at which the repulsion falls below the pull."""
        c = BY_NAME[cond]
        mu = ex.anti_subspace_weight(
            _E, lam=c["lam"], anneal_end=c["anti_anneal_end"],
            hold_ratio=c["anti_hold_ratio"], peak_ratio=c["anti_peak_ratio"],
        )  # fmt: skip
        return float(_E[mu < _A][0])

    _ep = np.mean([traj("lam0", s, "epoch") for s in ex.SEEDS], axis=0)
    _ctl = np.mean([traj("lam0", s, "alpha_op1") for s in ex.SEEDS], axis=0)

    def _departure(cond: str) -> float:
        """Last epoch at which ᾱ is still within 0.05 of the control — where it leaves for good."""
        a = np.mean([traj(cond, s, "alpha_op1") for s in ex.SEEDS], axis=0)
        near = np.flatnonzero(a <= _ctl + 0.05)
        return float(_ep[near[-1]]) if len(near) else float("nan")

    _rows = "".join(
        f"<tr><th><code>{c}</code></th>"
        f"<td class='num'>{_crossing(c):.0f}</td><td class='num'>{_departure(c):.0f}</td>"
        f"<td class='num'>{_departure(c) - _crossing(c):+.0f}</td>"
        f"<td class='num'>{ex.SCHEDULER.epochs - _departure(c):.0f}</td>"
        f"<td class='num'>{alpha_bar(c).mean():.3f}</td></tr>"
        for c in (*ROW[HOLD_LO], *ROW[HOLD_HI])
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>cell</th><th class="num">crossing</th><th class="num">departure</th>
        <th class="num">lag</th><th class="num">runway</th><th class="num">final ᾱ</th>
      </tr></thead>
      <tbody>{_rows}</tbody>
    </table></div>
    """
    _caption = """
    Two epochs read off each cell. Crossing is the first epoch at which the
    repulsion falls below the pull, computed from the schedules. Departure is
    the last epoch at which ᾱ is still within 0.05 of the control, read off the
    recorded trajectory. Runway is the training left after departure.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(HOLD_HI, HOLD_LO, ROW, alpha_bar):
    mo.md(rf"""
    Every cell departs, including the three at the sustained floor. So the
    strong form of the level account — a floor high enough that the crossing
    never happens — is not what the trajectories show. All six cross, and all
    six climb afterwards.

    What the floor does is move the crossing later and flatten what follows.
    Departure tracks the crossing in every cell, and raising the hold ratio
    pushes both later, the departure by more than the crossing. That is the
    delay. The flattening shows up when the departure epoch is held roughly
    fixed and the row changes: `{ROW[HOLD_HI][1]}` and `{ROW[HOLD_LO][2]}` leave
    the control within two epochs of each other, and finish at
    ᾱ = {alpha_bar(ROW[HOLD_HI][1]).mean():.3f} against
    {alpha_bar(ROW[HOLD_LO][2]).mean():.3f}. Same runway, different climb rate.

    Both effects run the same way, which is why the endpoint and the floor look
    interchangeable in the endpoint table and are not. The margin, dashed in
    every panel, climbs early while the repulsion is strong and then holds
    across the whole anneal; nothing in the grid costs it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Attribution (H3)
    """)
    return


@app.cell(hide_code=True)
def _(ALPHA_SPREAD, CONTROL_SPREAD, EFFECT_GATE, GATE_SCALE, INTERACTION_GATE):
    _moved = GATE_SCALE > 1.0
    mo.md(rf"""
    H3 has a footnote that asks for the seed spread before either statistic is used, and
    allows the gates to move with it. Pooled over the six factorial cells, the seed spread of ᾱ
    is {ALPHA_SPREAD:.4f}, against the ex-2.1.6 control value of
    {CONTROL_SPREAD:g} the gates were sized on. The anchored cells do not hold
    it, so the gates move by {GATE_SCALE:.2f}×: the main effect is scored at
    {EFFECT_GATE:.3f} and the interaction at {INTERACTION_GATE:.3f}. Those are
    the numbers used below{"" if _moved else " (unmoved)"}.

    One cell accounts for most of that spread. The rest sit between 0.006 and 0.027,
    so the rescaling is driven by a single condition rather than by the grid
    being noisy throughout.
    """)
    return


@app.cell(hide_code=True)
def _(HOLD_HI, HOLD_LO, ROW, alpha_bar, m_op1):
    def _cells(fn, fmt: str = "{:.3f}") -> str:
        return "".join(
            f"<tr><th>hold {h:.2f}</th>"
            + "".join(f"<td class='num'>{fmt.format(fn(c))}</td>" for c in ROW[h])
            + f"<td class='num'>{fmt.format(np.mean([fn(c) for c in ROW[h]]))}</td></tr>"
            for h in (HOLD_LO, HOLD_HI)
        )

    _head = "".join(f"<th class='num'>end {e:.0f}</th>" for e in ex.ANTI_ANNEAL_ENDS)
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr><th>ᾱ ↓</th>{_head}<th class="num">row mean</th></tr></thead>
      <tbody>{_cells(lambda c: float(alpha_bar(c).mean()))}</tbody>
      <thead><tr><th>m<sub>op1</sub> ↑</th>{_head}<th class="num">row mean</th></tr></thead>
      <tbody>{_cells(lambda c: float(m_op1(c).mean()))}</tbody>
    </table></div>
    """
    _caption = """
    The factorial, as cell means on both statistics. Above: ᾱ, which H3 is
    scored on. Below: the margin, repeated because a factor that contains the
    drift while losing the margin does not give an operating point.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(
    EFFECT_GATE,
    END_HI,
    END_LO,
    HOLD_EFFECT,
    HOLD_HI,
    HOLD_LO,
    INTERACTION,
    INTERACTION_GATE,
    end_effect,
):
    mo.md(rf"""
    **H3 holds, on both conditions.** The hold-ratio main effect is
    {HOLD_EFFECT:+.3f} against a gate of {EFFECT_GATE:.3f}, so the floor does
    something. The interaction is {INTERACTION:+.3f} against
    {INTERACTION_GATE:.3f}: the end{END_LO:.0f} − end{END_HI:.0f} difference is
    {end_effect(HOLD_LO):+.3f} in the `hold{HOLD_LO * 100:02.0f}` row and
    {end_effect(HOLD_HI):+.3f} in the `hold{HOLD_HI * 100:02.0f}` row, so the
    endpoint has about {end_effect(HOLD_LO) / end_effect(HOLD_HI):.1f} times as
    much leverage at the low floor as at the high one.

    Neither named miss occurred. Miss 1 needed the endpoint effect to clear
    {EFFECT_GATE:.3f} in both rows with the interaction under its gate; the
    interaction is over it. Miss 2 needed the main effect to fail, and it
    passes by a factor of {HOLD_EFFECT / EFFECT_GATE:.1f}.

    However, the interaction only clears its gate by
    {INTERACTION - INTERACTION_GATE:.3f}, about
    {(INTERACTION - INTERACTION_GATE) / INTERACTION_GATE * 100:.0f}% — on a gate that moved because of this experiment's own
    seed spread. And the level account predicted about +0.20, with the whole
    endpoint effect inside the low row; the observed {INTERACTION:+.3f} is
    below that, because the endpoint still moves ᾱ by
    {end_effect(HOLD_HI):+.3f} in the high row, where that account expects
    nothing.

    So the endpoint keeps some leverage even at the sustained floor. The
    trajectories show that the floor postpones the crossing and slows the climb
    after it, but it still crosses. The statistic H3 was
    built on separates the accounts in the direction it was meant to, and the
    mechanism behind it is milder than the account it favors.

    We also said we would report the ordering of the main effects without
    scoring it. The hold-ratio effect ({HOLD_EFFECT:+.3f}) is larger than the
    endpoint effect averaged over the two rows
    ({(end_effect(HOLD_LO) + end_effect(HOLD_HI)) / 2:+.3f}), which agrees with
    the interaction.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The dose arm
    """)
    return


@app.cell(hide_code=True)
def _(BY_NAME, alpha_bar, grading, m_op1, retention):
    _ORDER = [ex.DOSE_ARM_REFERENCE, "end90-hold03", ex.DOSE_ARM]
    _ref_dose = ex.anti_dose(BY_NAME[ex.DOSE_ARM_REFERENCE])
    _rows = "".join(
        f"<tr><th><code>{c}</code></th>"
        f"<td class='num'>{BY_NAME[c]['anti_anneal_end']:.0f}</td>"
        f"<td class='num'>{BY_NAME[c]['anti_peak_ratio']:.3f}</td>"
        f"<td class='num'>{ex.anti_dose(BY_NAME[c]) / _ref_dose:.2f}</td>"
        f"<td class='num'>{alpha_bar(c).mean():.3f}"
        f" <span class='range'>±{alpha_bar(c).std(ddof=1):.3f}</span></td>"
        f"<td class='num'>{m_op1(c).mean():.3f}</td>"
        f"<td class='num'>{grading(c)[1]:.2f}</td>"
        f"<td class='num'>{retention(c):.2f}</td></tr>"
        for c in _ORDER
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">end</th><th class="num">peak ratio</th>
        <th class="num">dose</th><th class="num">ᾱ ↓</th><th class="num">m<sub>op1</sub> ↑</th>
        <th class="num">R² ↑</th><th class="num">retention ↑</th>
      </tr></thead>
      <tbody>{_rows}</tbody>
    </table></div>
    """
    _caption = f"""
    The dose arm against the two cells it is built from. Dose is relative to
    <code>{ex.DOSE_ARM_REFERENCE}</code>, as in the Method table. The arm has
    that cell's dose and the <code>end90-hold03</code> timing, so its position
    between them is the comparison.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(alpha_bar):
    _ref, _late, _arm = (float(alpha_bar(c).mean()) for c in (ex.DOSE_ARM_REFERENCE, "end90-hold03", ex.DOSE_ARM))
    mo.md(rf"""
    The arm lands at ᾱ = {_arm:.3f}, between `{ex.DOSE_ARM_REFERENCE}`
    ({_ref:.3f}) and `end90-hold03` ({_late:.3f}), and about
    {(_ref - _arm) / (_ref - _late) * 100:.0f}% of the way across. So most of
    what the endpoint buys is timing, and some of it is dose.

    That should be taken loosely: this is one condition on three seeds, it scores nothing, and scaling the opening ratio to
    {ex.DOSE_MATCHED_PEAK_RATIO:g} weakens the repulsion through the epochs
    where the anchor first ramps in, which the Method section named as a
    confound of its own. The reading it does support is that the
    endpoint effect is not *merely* a dose effect,
    because holding the dose fixed leaves most of the gap intact.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Secondary measurements

    Carried from ex-2.1.7 without a gate. Two probes read redness from
    complementary parts of the residual stream at op1: one from the anchor
    coordinate alone, and one from the other 63 directions. The first starts
    empty and measures how much redness anchoring put there. The second has an
    RGB floor it cannot go far below, since redness is a function of color and
    the task needs the color cube, so it reports whether the cube survived
    rather than whether the concept moved.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], cells, geometry):
    _axis = {c: float(np.mean([cells[f"{c}-s{s}"]["axis_r2"] for s in ex.SEEDS])) for c in CONDS}
    _leak = {c: np.mean([cells[f"{c}-s{s}"]["leak_r2"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _spread = {c: np.mean([geometry[f"{c}-s{s}"]["spread"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _floor = ex.redness_rgb_floor()
    _rows = "".join(
        f"<tr><th><code>{c}</code></th>"
        f"<td class='num'>{_axis[c]:.3f}</td>"
        f"<td class='num'>{_leak[c][0]:.3f}</td><td class='num'>{_leak[c][-1]:.3f}</td>"
        f"<td class='num'>{_leak[c][-1] - _floor:+.3f}</td>"
        f"<td class='num'>{_spread[c][-1]:.3f}</td></tr>"
        for c in CONDS
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">on-axis R²</th>
        <th class="num">off-axis, embed</th><th class="num">off-axis, last</th>
        <th class="num">vs floor</th><th class="num">cube spread</th>
      </tr></thead>
      <tbody>{_rows}</tbody>
    </table></div>
    """
    _caption = f"""
    Where redness is readable at op1, and what the color cloud looks like there.
    On-axis R² is redness read from the anchor coordinate alone, averaged over
    layers. The two off-axis columns are the ridge probe on the other 63
    directions, at the token embedding and at the last layer, and the next
    column is the last-layer value against the RGB floor of {_floor:.3f}. Cube
    spread is the mean squared distance of the 216 op1 states from their
    centroid at the last layer: how much extent the cloud keeps. All are seed
    means, and none is scored.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(ANCHORED, CONDS: list[str], cells, geometry):
    _axis = {c: float(np.mean([cells[f"{c}-s{s}"]["axis_r2"] for s in ex.SEEDS])) for c in CONDS}
    _leak = {c: np.mean([cells[f"{c}-s{s}"]["leak_r2"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _spread = {c: np.mean([geometry[f"{c}-s{s}"]["spread"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _floor = ex.redness_rgb_floor()
    _hi = max(ANCHORED, key=lambda c: _axis[c])
    mo.md(rf"""
    The anchor coordinate carries redness in every anchored condition, from
    R² = {min(_axis[c] for c in ANCHORED):.2f} to
    {_axis[_hi]:.2f} (`{_hi}`), against {_axis["lam0"]:.2f} in the control. The
    spread across the grid is narrow, and it does not order the conditions the
    way ᾱ does, so this is not a second containment measurement.

    Off the axis, every condition sits within
    {max(abs(v[-1] - _floor) for v in _leak.values()):.2f} of the RGB floor at
    the last layer, and the anchored conditions all sit slightly above it
    ({min(_leak[c][-1] for c in ANCHORED):.2f}–{max(_leak[c][-1] for c in ANCHORED):.2f})
    where the control sits just below ({_leak["lam0"][-1]:.2f}). Above the floor
    is reachable because the residual stream encodes color nonlinearly and
    redness is quadratic in RGB, so a linear probe does better there than on raw
    channels. Nothing here reads as a second copy of the concept living off the
    axis, and nothing reads as the cube being lost: spread at the last layer
    runs {min(_spread[c][-1] for c in CONDS):.3f}–{max(_spread[c][-1] for c in CONDS):.3f},
    with the control at {_spread["lam0"][-1]:.3f} in the middle of that range.

    This is the point ex-2.1.7 made and this grid does not change: containment
    is not exclusivity. Holding ᾱ near the control keeps the cube off the anchor
    direction; it does not stop redness being readable from everywhere else,
    because the task needs the color cube either way.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exploratory analyses

    Post hoc, and scoring nothing. H2 failed on grading in every cell, so the
    question is what the response actually looks like.

    ### The grading metrics disagree

    The grading metrics are meant to rule out a *step*, in which very *red* colors would be aligned but everything else orthogonal. Both metrics cap out on a step, at the best it can score over all step locations: ρ can reach
    {SRHO} and R² {SR2}, both under the {GATE} gates, so no step passes.
    """.replace("{SRHO}", f"{ex.STEP_RHO_CEILING:.2f}")
        .replace("{SR2}", f"{ex.STEP_R2_CEILING:.2f}")
        .replace("{GATE}", f"{ex.GRADE_R2_GATE:g}")
    )
    return


@app.cell(hide_code=True)
def _(grading):
    _above = [c for c in ex.FACTORIAL_NAMES if grading(c)[1] > ex.STEP_R2_CEILING]
    _best_rho = max(grading(c)[0] for c in ex.FACTORIAL_NAMES)
    _best_r2 = max(grading(c)[1] for c in ex.FACTORIAL_NAMES)
    mo.md(rf"""
    Considering those ceilings, the two tracks say opposite things.

    - **ρ** ranks colors by `redness`. Every cell scores below what a step would: the best ρ in the grid is {_best_rho:.2f} against a step ceiling of {ex.STEP_RHO_CEILING:.2f}.
    - **R²** fits values against `sim¹·⁵` (proportionality). {len(_above)} of the six score above the ceiling, up to R² = {_best_r2:.2f} against {ex.STEP_R2_CEILING:.2f}.

    Neither actually says whether the response is a grade or a step, so we visualize it below.
    """)
    return


@app.cell(hide_code=True)
def _(BY_NAME, HOLD_HI, HOLD_LO, ROW, alpha_map, grading, m_op1):
    _RESP = {c: alpha_map(c)[:, :, 0].mean(axis=0) for c in [*ex.FACTORIAL_NAMES, "lam0"]}
    _order = np.argsort(ex.REDNESS)

    def _windowed(y: np.ndarray, k: int = 25) -> np.ndarray:
        """Mean of *y* over a sliding window of the redness ordering."""
        pad = np.pad(y[_order], (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    def _fitted_target(y: np.ndarray) -> np.ndarray:
        """`sim¹·⁵` rescaled by least squares onto the response — the shape R² scores against.

        R² is scale-invariant, so the target has no natural height; fitting it
        puts the curve where the statistic is actually comparing shapes.
        """
        t = ex.SIM_TARGET
        b = np.polyfit(t, y, 1)
        return np.polyval(b, t)

    _CTRL = _windowed(_RESP["lam0"])
    _GRID = [ROW[HOLD_LO], ROW[HOLD_HI]]

    @themed(
        name="grading",
        alt_text="""
            Six panels in a three-by-two grid sharing both axes: per-color alignment at
            the first operand, from about -0.1 to 1, against the redness of that color,
            from 0 to 1. Columns are anneal endpoints 50, 70 and 90; the upper row is
            hold ratio 0.03 and the lower 0.30. Each panel scatters 216 marks drawn in
            the color each stands for, with a heavy sliding-mean line through them, a
            flat grey control band near zero, and a dashed straight reference line.
            In every panel the marks form a dense flat cluster of greys, greens and
            blues at low redness, then a sparse rising tail of oranges and reds. The
            flat cluster sits higher in the upper row than the lower. The sliding mean
            is close to flat across the left two-thirds of every panel and rises
            steeply only in the right third, staying below the dashed line at low
            redness and crossing it near the top.
        """,
        caption=r"""
            **Alignment at op1, per color.** $\alpha_c$, the layer mean of
            $\cos(h, \hat v_{\text{red}})$ averaged over seeds, against the
            redness of the color. Columns are the anneal endpoint, rows the hold
            ratio, as in the trajectory figure. One mark per color, drawn in that
            color; the heavy line is a 25-color sliding mean over the redness
            ordering, and the flat grey band is the same sliding mean for the
            un-anchored control. The dashed line is `sim¹·⁵` rescaled onto the
            response by least squares — the shape the $R^2$ track scores
            against, so the marks' distance from it is what that statistic
            measures. Per-panel statistics: both grading gates sit at 0.8, and
            a pure step response can reach at most ρ = 0.75 or $R^2$ = 0.73.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesGrid

        fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), sharey=True, sharex=True)
        axes = cast(AxesGrid, axes)
        for row, conds in zip(axes, _GRID, strict=True):
            for ax, cond in zip(row, conds, strict=True):
                a, (rho, r2) = _RESP[cond], grading(cond)
                ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
                ax.plot(
                    ex.REDNESS[_order], _CTRL, color=light_dark("#999", "#777"), lw=3, alpha=0.5, zorder=1,
                    solid_capstyle="round",
                )  # fmt: skip
                # The R² target, so a reader can see the fit the statistic reports
                # rather than taking the number on trust. Smoothed the same way
                # as the response: several colors share a redness, so connecting
                # the target in redness order would zigzag between them.
                ax.plot(ex.REDNESS[_order], _windowed(_fitted_target(a)), color=light_dark("#888", "#888"),
                        lw=1.0, ls=(0, (4, 2)), zorder=2)  # fmt: skip
                ax.scatter(ex.REDNESS, a, c=ex.GRID_RGB, s=18, lw=0.4, zorder=3,
                           edgecolors=light_dark("#00000033", "#ffffff55"))  # fmt: skip
                ax.plot(ex.REDNESS[_order], _windowed(a), color=light_dark("#222", "#eee"), lw=1.6, zorder=4)
                ax.set_title(
                    f"end {BY_NAME[cond]['anti_anneal_end']:.0f} · hold {BY_NAME[cond]['anti_hold_ratio']:.2f}",
                    fontsize=9.5,
                )
                ax.annotate(
                    f"ρ = {rho:.2f}\nR² = {r2:.2f}\n$m$ = {m_op1(cond).mean():.2f}",
                    (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                    color=light_dark("#444", "#bbb"),
                )  # fmt: skip
        for ax in axes[1]:
            ax.set_xlabel("redness of op1", fontsize="small")
        for row in axes:
            row[0].set_ylabel(r"$\alpha_c$ at op1", fontsize="small")
        axes[0][0].set_ylim(-0.25, 1.25)  # headroom for the per-panel statistics
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The response is not a step. It has a knee: flat from the control level up to
    a redness of about 0.4, then a clean rise that tracks the dashed target
    closely. Raising the hold ratio lowers the flat part toward the control
    without changing the rise, which is containment doing what it is meant to.

    But most of the color cube lives in that flat part. {LO} of the {N} colors
    fall below redness 0.4, and inside a flat band their alignments carry no
    ordering. A rank statistic over all {N} colors is therefore dominated by them.
    """.replace("{LO}", f"{(ex.REDNESS < 0.4).sum()}").replace("{N}", f"{ex.REDNESS.size}")
    )
    return


@app.cell(hide_code=True)
def _(alpha_map):
    _KNEES = (0.2, 0.3, 0.4, 0.5)
    _R = ex.REDNESS
    _resp = {c: alpha_map(c)[:, :, 0].mean(axis=0) for c in ex.FACTORIAL_NAMES}
    _rows = "".join(
        f"<tr><th><code>{c}</code></th>"
        f"<td class='num'>{ex.spearman(_resp[c], _R):.3f}</td>"
        + "".join(f"<td class='num'>{ex.spearman(_resp[c][_R >= k], _R[_R >= k]):.3f}</td>" for k in _KNEES)
        + "</tr>"
        for c in ex.FACTORIAL_NAMES
    )
    _head = "".join(f"<th class='num'>≥ {k:g} <span class='range'>({(_R >= k).sum()})</span></th>" for k in _KNEES)
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead>
        <tr><th>cell</th><th class="num">all 216</th>{_head}</tr></thead>
      <tbody>{_rows}</tbody>
    </table></div>
    """
    _caption = f"""
    Spearman ρ against <code>redness</code>, computed over all 216 colors and
    then over the colors above each redness cut, with the number of colors
    remaining in each header. The gate is {ex.GRADE_RHO_GATE:g}.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(alpha_map):
    _R = ex.REDNESS
    _hi = _R >= 0.4
    _rho_hi = [float(ex.spearman(a[_hi], _R[_hi])) for a in (alpha_map(c)[:, :, 0].mean(axis=0) for c in ex.FACTORIAL_NAMES)]  # fmt: skip
    mo.md(rf"""
    Above the knee the ordering is good in every cell — ρ runs
    {min(_rho_hi):.2f} to {max(_rho_hi):.2f} on the {_hi.sum()} colors with redness ≥
    0.4, against a gate of {ex.GRADE_RHO_GATE:g} — and it climbs monotonically
    as the flat colors are excluded. So the graded part of the response is
    graded, but the full-set statistic is dominated by the flat part.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What a contained response can score

    That raises a question about the gate. If containment collapses the low-redness colors together, and the grading statistic reads their ordering, then they must be in tension. How high could the statistic be, under those conditions?

    We build the best response either statistic could hope for: the target shape exactly, above a knee, and contained to near zero below it. This is a property of the two statistics and the 216-color grid, not a measurement of any run.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Monte Carlo over the contained block's arbitrary ordering, which is what
    # makes this an expectation rather than a value. Seeded per call so the
    # numbers don't depend on which cell asks first, and small enough that the
    # draw count costs nothing.
    def ceiling(knee: float, scale: float = 1.0, draws: int = 600) -> tuple[float, float, float]:
        """Expected (ρ, R², ᾱ) for a response that is *scale* × `sim¹·⁵` above *knee*, contained below it."""
        R, T = ex.REDNESS, ex.SIM_TARGET
        rng = np.random.default_rng(0)
        hi = R >= knee
        out = np.array(
            [
                (ex.spearman(a, R), ex.r2_sim(a), a.mean())
                for a in (np.where(hi, scale * T, rng.normal(0, 0.02, R.size)) for _ in range(draws))
            ]
        )
        rho, r2, alpha = out.mean(axis=0)
        return float(rho), float(r2), float(alpha)

    _rows = "".join(
        f"<tr><th>{k:g}</th><td class='num'>{(ex.REDNESS < k).sum()}</td>"
        f"<td class='num'>{r:.2f}</td><td class='num'>{r2:.2f}</td><td class='num'>{a:.3f}</td></tr>"
        for k in (0.0, 0.2, 0.3, 0.4, 0.5)
        for r, r2, a in [ceiling(k)]
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>knee</th><th class="num">colors contained</th>
        <th class="num">best ρ</th><th class="num">best R²</th><th class="num">ᾱ</th>
      </tr></thead>
      <tbody>{_rows}</tbody>
    </table></div>
    """
    _caption = f"""
    The best either grading track can score, for a response that reproduces
    <code>sim¹·⁵</code> exactly above the knee and sits at the control below it,
    with the ᾱ that response implies. A knee of 0 is no containment at all.
    Both grading gates are {ex.GRADE_R2_GATE:g} and the containment gate is
    ᾱ ≤ {ex.MEAN_ALIGN_GATE:g}. Expectations over the contained block's
    arbitrary ordering; a pure step reaches ρ = {ex.STEP_RHO_CEILING:.2f} and
    R² = {ex.STEP_R2_CEILING:.2f} for comparison.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return (ceiling,)


@app.cell(hide_code=True)
def _(ceiling, grading):
    _best_r2 = max(grading(c)[1] for c in ex.FACTORIAL_NAMES)
    _corr = float(np.corrcoef(ex.SIM_TARGET, ex.REDNESS)[0, 1])
    _open, _mid, _knee, _deep = (ceiling(k) for k in (0.0, 0.3, 0.4, 0.5))
    _half = ceiling(0.2, scale=0.5)
    mo.md(rf"""
    The two tracks come apart. Even with no containment at all, the best ρ a
    perfect `sim¹·⁵` response reaches is {_open[0]:.2f}, barely over the gate, because
    `sim¹·⁵` is not a monotone function of `redness` (they correlate at {_corr:.2f}, not 1).
    With the {(ex.REDNESS < 0.4).sum()} colors below `redness` 0.4 contained, the ceiling falls
    to {_knee[0]:.2f}, under the observed values. So past a light knee the ρ track
    penalizes containment, and at the containment this grid achieves it cannot reach
    {ex.GRADE_RHO_GATE:g} however well the response is graded.

    R² behaves the other way. Its target is near zero for the colors
    containment collapses, so containment moves the response toward the target
    rather than away: the ceiling is {_knee[1]:.2f} at a knee of 0.4 and
    {_deep[1]:.2f} even at 0.5. That track is reachable, and the best cell in
    this grid gets {_best_r2:.2f} of it.[^m1r2]

    Within that family — `sim¹·⁵` at full amplitude — the ρ gate and the
    containment H2 also asks for cannot both be met. ᾱ crosses under
    {ex.MEAN_ALIGN_GATE:g} somewhere between a knee of 0.3 and 0.4, and by
    there ρ has fallen from {_mid[0]:.2f} to {_knee[0]:.2f}.

    That family is narrow, though. ρ ranks colors, so it is blind to the
    amplitude the ᾱ gate measures: scale the above-knee response to half of
    `sim¹·⁵` at a knee of 0.2, and both gates clear at once, at ρ =
    {_half[0]:.2f} and ᾱ = {_half[2]:.3f}. What the two gates jointly exclude
    is a response that is both steeply graded and full-amplitude on the reds,
    which is narrower than the gates being incompatible. Either way the ρ gate
    is doing something the preregistration did not say it was doing, and a
    future design should settle the amplitude question before reusing it.

    The R² shortfall is in the response itself: against an achievable
    {_knee[1]:.2f}, {_best_r2:.2f} is a real gap. Setting a realistic R² gate
    for future work would take more than this one grid.

    [^m1r2]: M1 scored proportionality with a Pearson R² as well, but of a
    different quantity: reconstruction error under intervention against
    $\cos^2$ similarity to *red* (D1.3), where the square is what the
    projection predicts. Here it is training-time alignment against
    `sim¹·⁵`, an exponent chosen as a shape between linear and quadratic rather
    than derived. Same statistic, different response variable and target, so
    M1's values are not a benchmark for this gate.
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
