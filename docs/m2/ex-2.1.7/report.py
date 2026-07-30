import marimo

__generated_with = "0.23.14"
app = marimo.App(
    app_title="Ex 2.1.7: a repulsive term and a narrower pull",
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.7: a repulsive term and a narrower pull

    Ex-2.1.6 anchored *red* with a single attractive term and got alignment
    without selectivity. The term moved the whole color cube onto the anchor
    direction, and the red-selective margin settled at about half its
    threshold, flat across a tenfold range of the anchor weight. Two
    candidate mechanisms could explain that, and the experiment could not
    tell them apart.

    First, nothing repelled the rest of the cube. M1 never ran its anchor
    bare: it also used an *anti-subspace* term, at a few percent of the
    anchor weight, which constrained the mean-square alignment of every
    point. That is exactly the quantity that ran away in ex-2.1.6.

    Second, most of the pull was blind. The anchor pulled four prompt
    positions of each labeled equation, and three of them (`+`, op2, `=`)
    carry no information about which color sits at op1. At those positions
    the label is an unobservable coin flip. Only the expected pull is
    visible to the optimizer, and the expected pull is nearly
    color-independent: a drift toward the anchor by construction.

    This experiment separates the two with a 2×2 factorial at the ex-2.1.6
    scoring rung ($\lambda = 0.1$): {bare anchor, anchor + anti-subspace} ×
    {four-position span pull, op1-only pull}. If the anti-subspace factor
    restores the margin, missing repulsion was the problem; if the op1-only
    factor does, the blind span was; if only the combination does, they
    compound. Two extra arms ride along: a schedule-timing variant of the
    anti-subspace anneal, and a weight-ceiling rung at $\lambda = 1$.

    /// admonition | TODO: tl;dr
    One paragraph after the results land: which factor moved the margin, by
    how much, and what it cost.
    ///

    /// details | Notation
    Notation carries over from ex-2.1.6: $\hat v_{\text{red}}$ is the anchor
    direction, $\lambda$ the anchor weight, $\ell$ indexes layers, $m$ is the
    alignment margin, and a **condition** crossed with a seed is a **cell**.
    New here: $\mu$ is the anti-subspace weight, always given relative to
    $\lambda$ as the ratio $\mu / \lambda$.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration skeleton: the method, hypotheses, and decision
    thresholds are written before the experiment runs. Placeholders marked
    `TODO` say what each figure or table will show; results will replace them
    in place, so the finished report reads as a prediction → observation
    diff. The hypotheses freeze once this skeleton is agreed; anything
    conceived after seeing the data goes under *Exploratory analyses*,
    marked post hoc.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    ### Testbed, labels, and measurements

    Everything ex-2.1.6 held is held here: the word-level `v216` testbed from
    ex-2.1.3 (216 grid colors, one token each, d64-L4), the same corpus seed
    and eval sets, labels drawn per visit with probability
    `redness(op1)⁸ × 0.08`, and the exhaustive 27-partner alignment probe
    set. The measurements are also the same: the alignment map over layers ×
    positions, the label-affinity-weighted margin, off-axis leakage probes,
    and the margin trajectory every 50 steps. The [ex-2.1.6
    report](../ex-2.1.6/report.py) documents each; only what changes is
    described below.

    One measurement is promoted rather than new. The exploratory geometry
    pass from ex-2.1.6 (per layer: the centroid of the 216 op1 states, its
    alignment with the anchor, and the extent of the cloud) is preregistered
    here, because H4 scores part of it. The mean alignment over all colors,
    $\bar\alpha$, is likewise recorded at every trajectory checkpoint beside
    the margin. In ex-2.1.6 it was added as an unscored diagnostic and
    turned out to be what the result was about.

    ### The factorial

    All four cells run at $\lambda = 0.1$, the scoring rung of ex-2.1.6,
    chosen there because every rung was task-clean and 0.1 sat in the middle
    of the swept range. The control ($\lambda = 0$) and the `span-bare` cell
    reproduce the ex-2.1.6 control and its λ=0.1 condition. So every
    comparison in this report is between cells that went through identical
    code, and the published ex-2.1.6 numbers double as a replication check.

    **Factor one: the anti-subspace term.** The `anti` cells add the M1
    repulsive term to the loss at weight $\mu$:

    $$L_{\text{anti}} = \operatorname{mean}\,\cos^2(h, \hat v_{\text{red}})$$

    where the mean runs over every residual-stream slice and every non-pad
    position of the batch: every line, labeled or not. The term penalizes
    the mean-square alignment of everything, so the labeled pull has to buy
    alignment against a headwind. That is the selectivity pressure ex-2.1.6
    lacked. Note what the term does not do: it never asks any particular
    point to leave the axis, only that the cloud as a whole not sit on it.

    Capacity is worth a sentence. In the 4- and 5-dimensional bottlenecks of
    M1, keeping the cloud off one axis surrendered a fifth or a quarter of
    the space, and the harder M1 conditions (ex-2.9.1) needed the dimension
    cleared outright. Our stream is 64-dimensional: an isotropic cloud has a
    mean $\cos^2$ of 1/64 per slice, and the three color dimensions of the
    task fit in the remaining 63 with room to spare. So the term should be
    cheap here. The live question is whether, at 3% of the anchor weight,
    it is strong enough to matter, not whether the task can afford it.

    **Factor two: the pull span.** The `span` cells pull the four prompt
    positions (op1, `+`, op2, `=`) of a labeled equation, as ex-2.1.6 did.
    The `op1` cells pull op1 alone, so every pulled state belongs to the
    token that carries the labeled color. If the color-independent drift
    came from the three blind positions, narrowing the pull removes it with
    no repulsive term at all. Note that the span pull is still the shape
    natural language forces, because a document label marks no position as
    the relevant one. So if op1-only wins, the lesson is about what
    sequence-level labeling costs, not a design we can keep for M3.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Schedule

    The anchor schedule is unchanged from ex-2.1.6: ramp over the 10-epoch
    LR warmup, hold at $\lambda$, anneal from epoch 90 to a floor of
    $0.1\lambda$ at 100.

    The anti-subspace schedule is copied from M1/ex-2.9.1 and stretched to
    our 100 epochs by fraction of training. M1 balanced the attractive and
    repulsive terms in time rather than by a single ratio. Anti-subspace
    opened at 2.5× the anchor peak weight, meaning it was at full strength
    from step 0, before the anchor had ramped in. It then annealed to 3% of
    the anchor by the halfway point and held there. Mapped onto our frame:
    $\mu / \lambda$ starts at 2.5 at epoch 0, anneals (minimum-jerk) to
    0.03 by epoch 50, and holds; from epoch 90 both terms share the
    end-of-training anneal, so their ratio is constant from the midpoint on.

    Nobody has measured how robust that timing is. The M1 schedules varied
    by experiment: ex-2.4.1, a gentler condition that confined vibrant
    colors to a two-axis subspace rather than clearing an axis, ran its
    terms *up* over training instead. So one arm probes the timing:
    `span-anti-late` stretches the anneal endpoint from epoch 50 to 90,
    holding the repulsion near peak through most of training. If default
    and late disagree by more than the noise floor, the timing matters and
    a dedicated sweep is warranted; if they agree, the recipe is forgiving
    on this axis and we can stop worrying about it.

    The second arm, `span-anti-hi`, runs the full recipe at $\lambda = 1$,
    one rung above the maximum ex-2.1.6 swept, with $\mu$ scaled in
    proportion. Ex-2.1.6 never found the weight at which the task pushes
    back, so this is the power analysis for how much anchor weight is
    available to spend. Both terms scale together, so it probes the
    headroom of the recipe, not of the bare anchor.
    """)
    return


@app.cell(hide_code=True)
def _():
    _conds = {c["name"]: c for c in ex.CONDITIONS}

    def _row(c: dict) -> str:
        anti = f"2.5 → 0.03 by ep {c.get('anti_anneal_end', ex.ANTI_ANNEAL_END):g}" if c["anti"] else "—"
        pulled = "op1 + op2 =" if c["span"] == ex.SPAN_FULL else "op1"
        span = "—" if c["lam"] == 0 else f"<code>{pulled}</code>"  # no anchor, so nothing is pulled
        role = {
            "lam0": "control",
            "span-bare": "the ex-2.1.6 λ=0.1 cell, re-run",
            "span-anti": "primary scoring cell",
            "op1-bare": "factorial",
            "op1-anti": "factorial",
            "span-anti-late": "timing arm",
            "span-anti-hi": "ceiling arm",
        }[c["name"]]
        return (
            f"<tr><th><code>{c['name']}</code></th><td class='num'>{c['lam']:g}</td>"
            f"<td>{span}</td><td class='num'>{anti}</td><td>{role}</td></tr>"
        )

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">λ</th><th>pulled positions</th>
        <th class="num">μ/λ schedule</th><th>role</th>
      </tr></thead>
      <tbody>{"".join(_row(c) for c in ex.CONDITIONS)}</tbody>
    </table></div>
    """
    _caption = f"""
    The seven conditions, each run at seeds {ex.SEEDS} — {len(ex.CONDITIONS) * len(ex.SEEDS)}
    cells. The four factorial cells share λ = {ex.SCORING_LAMBDA:g}; the two arms are
    reported descriptively rather than gated. μ/λ is the anti-subspace weight relative
    to the anchor peak.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _():
    _epochs = np.linspace(0, 100, 1001)

    @themed(
        name="schedules",
        alt_text="""
            Weight schedules over 100 epochs on a log axis. The learning rate warms up
            by epoch 10 and decays smoothly. The anchor weight ramps up with the warmup,
            holds at 0.1, and anneals to a floor after epoch 90. The anti-subspace
            weight starts at 0.25 — above everything else — and descends to 0.003 by
            epoch 50, where it holds; a dashed variant descends to the same level by
            epoch 90 instead.
        """,
        caption=f"""
            The three schedules at the scoring rung (λ = {ex.SCORING_LAMBDA:g}), log
            scale. The anti-subspace term opens at {ex.ANTI_PEAK_RATIO:g}λ, dominant
            before the anchor arrives, and anneals to {ex.ANTI_HOLD_RATIO:g}λ by epoch
            {ex.ANTI_ANNEAL_END:g} (the M1/ex-2.9.1 keyframes, mapped by fraction of
            training). The dashed line is the <code>span-anti-late</code> arm, which
            reaches the hold ratio at epoch {ex.ANTI_ANNEAL_END_LATE:g} instead. From
            epoch {ex.ANNEAL_START:g} the anchor and anti-subspace weights share the
            end-of-training anneal to a floor of {ex.ANNEAL_FLOOR:g}× their held values.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        lam = ex.SCORING_LAMBDA
        _grey = light_dark("#888", "#888")
        ax.plot(_epochs, ex.learning_rate(_epochs), color=_grey, lw=1, label="learning rate")
        ax.plot(
            _epochs, ex.anchor_weight(_epochs, peak=lam), color=light_dark("#c33", "#e66"), lw=1.5, label="anchor λ"
        )
        ax.plot(
            _epochs,
            ex.anti_subspace_weight(_epochs, lam=lam),
            color=light_dark("#36c", "#7af"),
            lw=1.5,
            label="anti-subspace μ",
        )
        ax.plot(
            _epochs,
            ex.anti_subspace_weight(_epochs, lam=lam, anneal_end=ex.ANTI_ANNEAL_END_LATE),
            color=light_dark("#36c", "#7af"),
            lw=1.2,
            ls=(0, (4, 3)),
            label="μ, late arm",
        )
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_ylim(1e-4, 0.4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("weight")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: H4(a)'s low-end reference was "≈ 1/8 for an isotropic cloud". 1/8 is
    # the spread of a single cosine in d64 (ex-2.1.6 uses it that way to derive the
    # 0.056 noise floor), not the value this statistic takes without an anchor:
    # alpha-bar averages over 216 colors and 5 layers, and ex-2.1.6's control
    # measured 0.008 (seeds: 0.019, -0.015, 0.021). Replaced with the measured
    # control value. Verify: ex-2.1.6 metrics, `alpha_mean_op1` on the lam0 cells.
    # The 0.25 gate itself is unchanged and still stated relative to the bare
    # term's 0.53.
    #
    # REVIEW: H3's "about 4× the three-seed noise floor" is 3×: 0.1 / 0.032. A main
    # effect is a difference of two two-cell means, so its noise is about the same
    # 0.032 as a single seed-mean (0.032/sqrt(2) per side, combined in quadrature).
    # Corrected in the text and in `MAIN_EFFECT_GATE`; the gate value is unchanged.
    #
    # REVIEW: added the both-effects-clear ordering to H3's partials. The gate is a
    # conjunction (anti effect >= 0.1 and >= the op1 effect), so the case where both
    # clear 0.1 with op1-only larger failed H3 without a named reading, and the
    # analysis section claimed all four orderings were named.
    mo.md(r"""
    ## Hypotheses

    Gates carried over from ex-2.1.6 keep their thresholds and their
    rationale. The noise floors quoted there (margin ≈ 0.056 per seed,
    ≈ 0.032 on a three-seed mean) apply unchanged, since the architecture
    and the measurement are the same. Unless stated otherwise, results are
    reported on the `span-anti` cell (seed-averaged), the condition the
    ex-2.1.6 discussion named as the next experiment. The other factorial
    cells are what H3 reads, and the two arms are descriptive.

    **H1.** The added term and the narrower pull are as free as the bare
    anchor was: every $\lambda = 0.1$ condition (the four factorial cells
    and the timing arm) is task-clean, meaning `named_holdout` exact match
    within 0.02 (absolute) of the seed mean of the in-experiment control.
    The ceiling arm is reported in the same table but scored only
    descriptively; finding its cost is what it is for. Partial: the
    factorial is clean but the timing arm is not, which would itself be
    evidence that sustained repulsion has a price.

    **H2.** With the missing ingredient restored, the concept lands
    selectively. On some task-clean factorial cell: (a) the layer-mean
    margin at op1 reaches $m_{\text{op1}} \ge 0.5$; (b) the response is
    graded, meaning that over the 216 colors, the seed-mean layer-mean
    alignment at op1 clears Spearman $\rho \ge 0.8$ against `redness` or
    Pearson $R^2 \ge 0.8$ against `sim¹·⁵` (both tracks, thresholds, and
    step ceilings as justified at length in ex-2.1.6); (c) the control
    shows $|m_{\text{op1}}| \le 0.1$. Partial: some cell clears 0.35
    without reaching 0.5; 0.35 is above the ex-2.1.6 value of 0.27 by more
    than the seed-mean noise. One caution on the partial: "some cell" takes
    a maximum over the four factorial cells, and the maximum of four noisy
    means sits about one noise unit above any single one. So a bare-partial
    result resting on exactly one cell at 0.35 is weaker than the same
    number on the pre-named primary cell.

    **H3.** The margin failure in ex-2.1.6 was mostly the missing
    repulsion, not the blind span. This is scored on the main effects of
    the factorial on $m_{\text{op1}}$ (for each factor: the mean over its
    two cells, differenced): the anti-subspace main effect is at least 0.1
    and at least as large as the op1-only main effect. A main effect is a
    difference of two-cell means, so its noise is about the three-seed
    floor of 0.032, and 0.1 is about 3× that. Partials: the op1-only effect
    clears 0.1 and is the larger of the two, which reads as the blind-span
    account winning, whether or not the anti-subspace effect also clears
    it; or neither main effect clears 0.1 but the `op1-anti` cell alone
    does (an interaction: each mechanism blocks the margin on its own).

    **H4.** The dynamics behave as the term dictates. (a) Containment: in
    every anti cell at $\lambda = 0.1$, the end-of-training mean alignment
    over all colors at op1 satisfies $\bar\alpha \le 0.25$. For reference,
    the bare term in ex-2.1.6 ended at 0.53, and the control in that
    experiment at 0.008. (b) Retention: the ex-2.1.6 trajectory gate,
    unchanged: for every anchored run whose running maximum of
    $m_{\text{op1}}$ reaches 0.2, the final value is at least 0.8× that
    maximum. In ex-2.1.6 the margin peaked near epoch 10 and slid back a
    quarter under steady pressure; if the slide was the rest of the cube
    catching up to a selective early response, repulsion should hold the
    peak. Partial: (a) holds and (b) fails only in the timing arm, whose
    sustained repulsion changes the trajectory most.
    """)
    return


@app.cell(hide_code=True)
def _():
    _res = load_results()
    mo.stop(_res is None, mo.md("_Results are not published yet; the analysis cells below render once they are._"))
    assert _res is not None
    metrics, arrays = _res
    return metrics, arrays


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost (H1)

    /// admonition | TODO: behavior table
    The H1 table from ex-2.1.6, one row per condition plus the ex-2.1.3
    `v216` reference and the published ex-2.1.6 λ=0.1 row: seen and
    held-out exact match, held-out NLL, `open` guess distance, the signed
    gap to control, and the gate verdict. Expected: every λ=0.1 row within
    0.02 of control, replicating H1. The ceiling arm is where a cost would
    first appear, and a clean ceiling arm would say the task tolerates the
    recipe at 10× the scoring weight. Contrary: a cost in the anti cells
    but not `span-bare` would mean the repulsive term, not the anchor, is
    what the task feels.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selectivity (H2)

    /// admonition | TODO: margin by condition
    A dot-and-range plot of $m_{\text{op1}}$ per condition (per-seed dots,
    seed-mean bar), with the 0.5 gate, the 0.35 partial line, and the
    published ex-2.1.6 value of 0.27 as a reference line. Expected: `span-anti` at
    or above the gate; `span-bare` reproducing 0.27. Contrary: all four
    cells in a band around 0.27, meaning neither mechanism was the
    bottleneck.
    ///

    /// admonition | TODO: grading scatter
    Per-color response at op1 against redness for the passing cell (or the
    best cell, if none passes), colored by the colors themselves, with the
    windowed mean and the two track statistics (ρ, R²) beside their gates.
    Expected: a graded rise, not a step at the label threshold. Contrary: a
    step would say the anchor caught the labeled exemplars rather than
    *red*; in that case the flattened-affinity follow-up already queued in
    `todo-science.md` becomes the next experiment.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Attribution (H3)

    /// admonition | TODO: the 2×2
    The factorial as a 2×2 table of $m_{\text{op1}}$ (seed mean ± half
    range per cell) with the two main effects and the interaction in the
    margins. Expected: the anti-subspace main effect ≥ 0.1 and ≥ the
    op1-only effect. H3 names a reading for the pass and for each partial;
    state which one obtained and carry it to the discussion. If the outcome
    is none of them (both effects below 0.1 with no interaction), that is
    the null: neither mechanism was the bottleneck.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Containment and dynamics (H4)

    /// admonition | TODO: mean alignment and margin trajectories
    Two panels per condition family, x = epoch. Left: $\bar\alpha$ at op1
    with the 0.25 containment gate and the ex-2.1.6 bare-term endpoint
    (0.53) marked. Right: $m_{\text{op1}}$ with its running maximum, the
    0.8× retention band, and both weight schedules behind it. Expected: anti
    cells hold $\bar\alpha$ low from the start (the term opens at 2.5λ),
    and the margin holds its peak instead of sliding. Contrary: $\bar\alpha$
    climbing after epoch 50 would say the 0.03λ hold ratio is too weak once
    the anneal ends. The timing arm is positioned to confirm exactly that,
    since it keeps the weight up through epoch 90.
    ///

    /// admonition | TODO: per-layer and geometry read
    The margin by layer × position for `span-anti` beside `span-bare` (the
    ex-2.1.6 figure, with one new column), and the preregistered geometry:
    centroid·anchor and extent per layer. Expected: centroid alignment near
    the control value (0.02 in ex-2.1.6) rather than the bare-term 0.89 at
    mid-stack. No gate; this is the mechanism picture behind H4(a).
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Arms

    /// admonition | TODO: timing arm
    `span-anti-late` against `span-anti`, on the H2 margin and the H4
    trajectories. If they agree within noise, the M1 recipe is robust to a
    2× stretch of its anneal and the timing question can rest. A gap in
    either direction says schedule timing matters, and its size would size
    the dedicated sweep. Report descriptively; no gate.
    ///

    /// admonition | TODO: ceiling arm
    `span-anti-hi` (λ=1, μ scaled with it) in the H1 table and on the
    margin plot. The first rung at which the task pushes back sets the
    weight budget for the D2.2 operation anchors. If this rung is still
    free, note that the ceiling remains unfound and leave chasing it to a
    dedicated sweep.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Anything conceived after the results were seen lands here, marked post
    hoc. Reserved.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO: discussion
    To write after the results: what the factorial attributed, what that
    means for the D2.1 claim ("SCA transfers to transformers" needs a
    selective anchor, not just a free one), and which queued follow-up is
    next: the flattened label affinity if selectivity appeared, or the
    remaining repulsive terms (anti-anchor, separate) or position pooling
    if it did not. Keep interpretation within what the four cells and two
    arms measured. One trap to avoid: if the margin moves nowhere, the H3
    null and the H2 contrary reading are the same finding reached twice;
    report it once, not as two lines of evidence.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
