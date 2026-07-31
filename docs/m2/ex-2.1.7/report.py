import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
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
    # Ex 2.1.7: a repulsive term and a narrower pull

    Ex-2.1.6 anchored *red* with a single attractive term and got alignment
    without selectivity. The term moved the whole color cube onto the anchor
    direction, and the red-selective margin settled at about half its
    threshold, flat across a tenfold range of the anchor weight. Two
    candidate mechanisms could explain that, and the experiment could not
    tell them apart.

    First, nothing repelled the rest of the cube. M1 never ran its anchor
    bare: it also used an indiscriminate *anti-subspace* term, at a few
    percent of the anchor weight, which penalized the average squared
    alignment between the anchor direction and every representation in the
    batch — a gentle, global pressure keeping the cloud as a whole off the
    axis. That average is exactly the quantity that ran away in ex-2.1.6.

    Second, most of the pull was blind. The anchor pulled four prompt
    positions of each labeled equation, and three of them (`+`, op2, `=`)
    carry no information about which color sits at op1. At those positions
    the label is an unobservable coin flip. Only the expected pull is
    visible to the optimizer, and the expected pull is nearly
    color-independent.

    This experiment separates the two with a 2×2 factorial at the ex-2.1.6
    scoring rung ($\lambda_\text{a} = 0.1$): {bare anchor, anchor + anti-subspace} ×
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
    Notation carries over from ex-2.1.6, with loss weights now subscripted
    by their term. New here: the bar in $\lambda_{\bar{\text{s}}}$ marks a
    repulsive term, whose weight is always given relative to the anchor as
    the ratio $\lambda_{\bar{\text{s}}} / \lambda_\text{a}$.

    | symbol | meaning |
    | --- | --- |
    | $\hat v_{\text{red}}$ | the anchor direction |
    | $\lambda_\text{a}$ | anchor weight (ex-2.1.6's bare $\lambda$) |
    | $\lambda_{\bar{\text{s}}}$ | anti-subspace weight |
    | $\ell$ | layer index |
    | $m$, $m_{\text{op1}}$ | alignment margin; its layer mean at op1 |
    | $\bar\alpha$ | mean alignment with $\hat v_{\text{red}}$ over all 216 colors, layer-averaged at op1 |
    | **cell** | a **condition** crossed with a seed |
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
    report](../ex-2.1.6/report.py) documents each; only the changes are
    described below.

    One measurement is promoted: The exploratory geometry pass from ex-2.1.6
    (per layer: the centroid of the 216 op1 states, its alignment with the
    anchor, and the extent of the cloud) is preregistered here, because H4
    scores part of it. The mean alignment over all colors, $\bar\alpha$, is
    likewise recorded at every trajectory checkpoint beside the margin. In
    ex-2.1.6 it was added as an unscored diagnostic, and it turned out to be
    important.

    ### The sweep

    All four conditions run at $\lambda_\text{a} = 0.1$, the scoring rung
    of ex-2.1.6, reused here by default: every rung was task-clean and 0.1
    sat in the middle of the swept range. The control
    ($\lambda_\text{a} = 0$) and the `span-bare` condition reproduce the
    ex-2.1.6 control and its $\lambda_\text{a} = 0.1$ condition. So every
    comparison in this report is between conditions that went through
    identical code, and the published ex-2.1.6 numbers double as a
    replication check.

    **Factor one: the anti-subspace term.** The `anti` conditions add the
    M1 repulsive term to the loss at weight $\lambda_{\bar{\text{s}}}$:

    $$\lambda_{\bar{\text{s}}}\, \mathcal{L}_{\bar{\text{s}}}, \qquad
    \mathcal{L}_{\bar{\text{s}}} = \operatorname{mean}\,\cos^2(h, \hat v_{\text{red}})$$

    where the mean runs over every residual-stream slice and every non-pad
    position of the batch: every line, labeled or not. This is M1's term with
    the reserved coordinate axis replaced by an arbitrary unit direction: M1
    penalized $\sum_{i} \hat z_i^2$ over the reserved axes of the *normalized*
    representation, and each $\hat z_i^2$ is a $\cos^2$ against a basis vector.
    A higher-dimensional reserved subspace would sum $\cos^2$ over an
    orthonormal basis of it; here the subspace is the one-dimensional span of
    the anchor. The term penalizes the mean-square alignment of everything, so
    the labeled pull has to buy alignment against a headwind. That is, we
    expect, the selectivity pressure ex-2.1.6 lacked. Note what the term does
    not do: it never asks any particular point to leave the axis, only that the
    cloud as a whole not sit on it.

    Capacity is worth a sentence. In the 4- and 5-dimensional bottlenecks of
    M1, keeping the cloud off one axis surrendered a fifth or a quarter of
    the space, and the harder M1 conditions (ex-2.9.1) needed the dimension
    cleared outright. Our stream is 64-dimensional: an isotropic cloud has a
    mean $\cos^2$ of 1/64 per slice, and the three color dimensions of the
    task fit in the remaining 63 with room to spare. So the term should be
    cheap here. The live question is whether, at 3% of the anchor weight,
    it is strong enough to matter, not whether the task can afford it.

    **Factor two: the pull span.** The `span` conditions pull the four prompt
    positions (op1, `+`, op2, `=`) of a labeled equation, as ex-2.1.6 did. The
    `op1` conditions pull op1 alone, so every pulled state belongs to the token
    that carries the labeled color. If the color-independent drift came from the
    three blind positions, narrowing the pull removes it with no repulsive term
    at all. Note that the span pull is still the shape natural language forces,
    because a document label marks no position as the relevant one. So if
    op1-only wins, the lesson is about what sequence-level labeling costs, not
    necessarily a design we can keep for M3.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Schedule

    The anchor schedule is unchanged from ex-2.1.6: ramp over the 10-epoch
    LR warmup, hold at $\lambda_\text{a}$, anneal from epoch 90 to a floor
    of $0.1\lambda_\text{a}$ at 100.

    The anti-subspace schedule is copied from M1/Ex-2.9.1 and stretched to
    our 100 epochs by fraction of training. M1 balanced the attractive and
    repulsive terms in time rather than by a single ratio. Anti-subspace
    opened at 2.5× the anchor peak weight, meaning it was at full weight
    from step 0 (its effective push still scaled by the warming-up learning
    rate, like every term), before the anchor had ramped in. It then
    annealed to 3% of the anchor by the halfway point and held there.
    Mapped onto our frame: $\lambda_{\bar{\text{s}}} / \lambda_\text{a}$
    starts at 2.5 at epoch 0, anneals (minimum-jerk) to 0.03 by epoch 50,
    and holds; from epoch 90 both terms share the end-of-training anneal,
    so their ratio is constant from the midpoint on.

    Nobody has measured how robust that timing is. The M1 schedules varied
    by experiment: Ex-2.4.1, a gentler condition that confined vibrant
    colors to a two-axis subspace rather than clearing an axis, ran its
    terms *up* over training instead. So one arm probes the timing:
    `span-anti-late` stretches the anneal endpoint from epoch 50 to 90,
    holding the repulsion near peak through most of training. If default
    and late disagree by more than the noise floor, the timing matters and
    a dedicated sweep is warranted; if they agree, the recipe is forgiving
    on this axis and we can stop worrying about it.

    The second arm, `span-anti-hi`, runs the full recipe at
    $\lambda_\text{a} = 1$, one rung above the maximum Ex-2.1.6 swept, with
    $\lambda_{\bar{\text{s}}}$ scaled in proportion. Ex-2.1.6 never found the
    weight at which the task pushes back, so this is the power analysis for how
    much anchor weight is available to spend. Both terms scale together, so it
    probes the headroom of the recipe, not of the bare anchor.
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
            "span-bare": "the Ex-2.1.6 λ<sub>a</sub>=0.1 condition, re-run",
            "span-anti": "primary scoring condition",
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
        <th>condition</th><th class="num">λ<sub>a</sub></th><th>pulled positions</th>
        <th class="num">λ<sub>s̄</sub>/λ<sub>a</sub> schedule</th><th>role</th>
      </tr></thead>
      <tbody>{"".join(_row(c) for c in ex.CONDITIONS)}</tbody>
    </table></div>
    """
    _caption = f"""
    The seven conditions, each run at seeds {ex.SEEDS} — {len(ex.CONDITIONS) * len(ex.SEEDS)}
    cells. The four factorial conditions share λ<sub>a</sub> = {ex.SCORING_LAMBDA:g} and carry the H2
    and H3 gates; the two arms add none of their own, though H1 and H4 reach them by their
    stated scopes — the timing arm is a λ<sub>a</sub> = {ex.SCORING_LAMBDA:g} anti condition,
    and H4(b) covers every anchored run. λ<sub>s̄</sub>/λ<sub>a</sub> is
    the anti-subspace weight relative to the anchor peak.
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
            The three schedules at the scoring rung (λ<sub>a</sub> = {ex.SCORING_LAMBDA:g}),
            log scale. The anti-subspace term opens at {ex.ANTI_PEAK_RATIO:g}λ<sub>a</sub>,
            dominant before the anchor arrives, and anneals to
            {ex.ANTI_HOLD_RATIO:g}λ<sub>a</sub> by epoch {ex.ANTI_ANNEAL_END:g}
            (the M1/ex-2.9.1 keyframes, mapped by fraction of training). The
            dashed line is the `span-anti-late` arm, which reaches the hold
            ratio at epoch {ex.ANTI_ANNEAL_END_LATE:g} instead. From epoch
            {ex.ANNEAL_START:g} the anchor and anti-subspace weights share the
            end-of-training anneal to a floor of {ex.ANNEAL_FLOOR:g}× their held
            values.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        lam = ex.SCORING_LAMBDA
        _grey = light_dark("#888", "#888")
        ax.plot(_epochs, ex.learning_rate(_epochs), color=_grey, lw=1, label="learning rate")
        ax.plot(
            _epochs,
            ex.anchor_weight(_epochs, peak=lam),
            color=light_dark("#c33", "#e66"),
            lw=1.5,
            label=r"anchor $\lambda_\mathrm{a}$",
        )
        ax.plot(
            _epochs,
            ex.anti_subspace_weight(_epochs, lam=lam),
            color=light_dark("#36c", "#7af"),
            lw=1.5,
            label=r"anti-subspace $\lambda_{\bar{\mathrm{s}}}$",
        )
        ax.plot(
            _epochs,
            ex.anti_subspace_weight(_epochs, lam=lam, anneal_end=ex.ANTI_ANNEAL_END_LATE),
            color=light_dark("#36c", "#7af"),
            lw=1.2,
            ls=(0, (4, 3)),
            label=r"$\lambda_{\bar{\mathrm{s}}}$, late arm",
        )
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_ylim(1e-4, 0.4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("weight")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: H4(a)'s low-end reference was "≈ 1/8 for an isotropic cloud". 1/8 is
    # the spread of a single cosine in d64 (ex-2.1.6 uses it that way to derive the
    # 0.056 noise floor), not the value this statistic takes without an anchor:
    # alpha-bar averages over 216 colors and 5 layers, and ex-2.1.6's control
    # measured 0.008 (seeds: 0.019, -0.015, 0.021). Replaced with the measured
    # control value. Verify: ex-2.1.6 metrics, `alpha_mean_op1` on the lam0 cells.
    # The gate was subsequently tightened to 0.1 (near-control containment),
    # with the old 0.25 halving level kept as a named partial.
    #
    # REVIEW: H3's "about 4× the three-seed noise floor" is 3×: 0.1 / 0.032. A main
    # effect is a difference of two two-cell means, so its noise is about the same
    # 0.032 as a single seed-mean (0.032/sqrt(2) per side, combined in quadrature).
    # Corrected in the text and in `MAIN_EFFECT_GATE`; the gate value is unchanged.
    #
    # REVIEW: added the both-effects-clear ordering to H3's contrary readings.
    # The gate is a conjunction (anti effect >= 0.1 and >= the op1 effect), so
    # the case where both clear 0.1 with op1-only larger failed H3 without a
    # named reading, and the analysis section claimed all four orderings were
    # named.
    mo.md(r"""
    ## Hypotheses

    Gates carried over from ex-2.1.6 keep their thresholds and their
    rationale. The noise floors quoted there (margin ≈ 0.056 per seed,
    ≈ 0.032 on a three-seed mean) apply unchanged, since the architecture
    and the measurement are the same. Unless stated otherwise, results are
    reported on the `span-anti` condition (seed-averaged), the one the
    ex-2.1.6 discussion named as the next experiment. The other factorial
    conditions are used by H3, and the two arms are just descriptive.

    **H1.** The added term and the resulting narrower pull are as free as the
    lone anchor was: every $\lambda_\text{a} = 0.1$ condition (the four
    factorial conditions and the timing arm) is task-clean, meaning
    `named_holdout` exact match within 0.02 (absolute) of the seed mean of the
    in-experiment control. Partial: the factorial is clean but the timing arm is
    not, which would itself be evidence that sustained repulsion has a price.

    **H2.** With the missing ingredient restored, the concept lands
    selectively. On some task-clean factorial condition: (a) the layer-mean
    margin at op1 reaches $m_{\text{op1}} \ge 0.5$; (b) the response is
    graded, meaning that over the 216 colors, the seed-mean layer-mean
    alignment at op1 clears Spearman $\rho \ge 0.8$ against `redness` or
    Pearson $R^2 \ge 0.8$ against `sim¹·⁵` (both tracks, thresholds, and
    step ceilings as justified at length in ex-2.1.6); (c) the control
    shows $|m_{\text{op1}}| \le 0.1$. Partial: some condition clears 0.35
    without reaching 0.5; 0.35 is above the ex-2.1.6 value of 0.27 by more
    than the seed-mean noise.[^h2max]

    **H3.** The margin failure in ex-2.1.6 was mostly the missing
    repulsion, not the blind span. This is scored on the main effects of
    the factorial on $m_{\text{op1}}$ (for each factor: the mean over its
    two conditions, differenced): the anti-subspace main effect is at least 0.1
    and at least as large as the op1-only main effect.[^h3noise]
    Two outcomes that fail H3 are
    named in advance so a miss stays readable; unlike the partials elsewhere,
    they are contrary readings rather than weaker passes. The op1-only effect
    clears 0.1 and is the larger of the two, which reads as the blind-span
    interpretation of ex-2.1.6 was correct (whether or not the anti-subspace
    effect also clears it); or neither main effect clears 0.1 but the `op1-anti`
    condition alone does (an interaction: each mechanism blocks the margin on
    its own).

    **H4.** The repulsive term visibly does its two jobs: it keeps the cube
    as a whole off the axis, and the margin holds its early peak instead of
    sliding back.
    (a) Containment: in every anti condition at $\lambda_\text{a} = 0.1$, the
    end-of-training mean alignment over all colors at op1 satisfies
    $\bar\alpha \le 0.1$.[^h4gate]
    (b) Retention: the ex-2.1.6 trajectory gate, unchanged: for every anchored
    run whose running maximum of $m_{\text{op1}}$ reaches 0.2, the final value
    is at least 0.8× that maximum.[^h4slide]
    Partials, in decreasing strength: every anti condition clears the old
    halving level $\bar\alpha \le 0.25$ without all reaching 0.1, which reads as
    repulsion weakening the runaway without containing it; or (a) holds and (b)
    fails only in the timing arm, whose sustained repulsion changes the
    trajectory most.

    [^h2max]: One caution: "some condition" takes a maximum over the four
        factorial conditions, and the maximum of four noisy means sits about
        one noise unit above any single one. So a bare-partial result resting
        on exactly one condition at 0.35 is weaker than the same number on
        the pre-named primary condition.

    [^h3noise]: A main effect is a difference of two-condition means, so its
        noise is about the three-seed floor of 0.032, and 0.1 is about 3×
        that.

    [^h4gate]: Why 0.1: the ex-2.1.6 reference points are 0.53 for the bare
        term and 0.008 for the control, with a seed spread of about 0.02.
        Working repulsion should push $\bar\alpha$ to near-control levels;
        0.1 is a fifth of the bare term and still five noise units above
        control, so a working mechanism passes comfortably and a
        merely-weakened runaway does not.

    [^h4slide]: In ex-2.1.6 the margin peaked near epoch 10 and slid back a
        quarter under steady pressure. If the slide was the rest of the cube
        catching up to a selective early response, repulsion should hold the
        peak.
    """)
    return


@app.cell(hide_code=True)
def _():
    _res = load_results()
    mo.stop(_res is None, mo.md("_Results are not published yet; the analysis cells below render once they are._"))
    assert _res is not None
    metrics, arrays = _res
    cells = {c["label"]: c for c in metrics["cells"]}
    stats = metrics["corpus_stats"]

    CONDS: list[str] = [c["name"] for c in ex.CONDITIONS]
    ANCHORED = [c for c in CONDS if c != "lam0"]
    # Matplotlib titles render LaTeX; an authored HTML table does not, so each
    # condition carries a plain-text label as well as a marked-up one.
    LABELS = {
        "lam0": r"control",
        "span-bare": r"span $\cdot$ bare",
        "span-anti": r"span $\cdot$ anti",
        "op1-bare": r"op1 $\cdot$ bare",
        "op1-anti": r"op1 $\cdot$ anti",
        "span-anti-late": r"late arm",
        "span-anti-hi": r"ceiling arm",
    }
    LABELS_TXT = {k: v.replace(r"$\cdot$", "·") for k, v in LABELS.items()}

    def over_seeds(cond: str, fn) -> np.ndarray:
        """One value per seed for a condition, in seed order."""
        return np.array([fn(cells[f"{cond}-s{s}"]) for s in ex.SEEDS])

    def acc(cond: str, set_name: str = "named_holdout", key: str = "accuracy") -> np.ndarray:
        return over_seeds(cond, lambda c: c["sets"][set_name][key])

    def alpha_map(cond: str) -> np.ndarray:
        """Seed-mean alignment map for a condition: (layers, colors, positions)."""
        return np.mean([arrays[f"{cond}-s{s}/alpha"] for s in ex.SEEDS], axis=0)

    def margin_map(cond: str) -> np.ndarray:
        """Seed-mean margin map: (layers, positions)."""
        return np.mean([arrays[f"{cond}-s{s}/margin"] for s in ex.SEEDS], axis=0)

    def m_op1(cond: str) -> np.ndarray:
        """Per-seed layer-mean margin at op1 — the statistic H2(a), H3 and H4 read."""
        return over_seeds(cond, lambda c: c["m_op1"])

    def alpha_bar(cond: str) -> np.ndarray:
        """Per-seed mean alignment over all 216 colors at op1 — H4(a)'s containment statistic."""
        return over_seeds(cond, lambda c: c["alpha_mean_op1"])

    def traj(cond: str, seed: int, key: str) -> np.ndarray:
        return np.asarray(cells[f"{cond}-s{seed}"]["traj"][key], dtype=float)

    def grading(cond: str) -> tuple[np.ndarray, float, float]:
        """The per-color response at op1 (seed-mean, layer-mean) with its two H2(b) statistics."""
        a = alpha_map(cond)[:, :, 0].mean(axis=0)
        return a, ex.spearman(a, ex.REDNESS), ex.r2_sim(a)

    CONTROL_ACC = float(acc("lam0").mean())
    TASK_CLEAN = {c: bool(abs(acc(c).mean() - CONTROL_ACC) <= ex.TASK_GATE) for c in CONDS}
    geometry = load_geometry() or {}
    return (
        ANCHORED,
        CONDS,
        CONTROL_ACC,
        LABELS,
        LABELS_TXT,
        TASK_CLEAN,
        acc,
        alpha_bar,
        arrays,
        cells,
        geometry,
        grading,
        m_op1,
        margin_map,
        stats,
        traj,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost (H1)
    """)
    return


@app.cell(hide_code=True)
def _(CONDS, CONTROL_ACC, LABELS_TXT, TASK_CLEAN, acc):
    # Two external references, both published: ex-2.1.3's v216 cell (the un-anchored
    # testbed this is all built on) and ex-2.1.6's λ=0.1 condition, which `span-bare`
    # re-runs. Neither is a condition of this sweep.
    _REFS = [
        ("ex-2.1.3 <code>v216</code>", {"seen": 1.000, "hold": 0.9948, "nll": 0.0245, "open": 0.1414}),
        ("ex-2.1.6 λ<sub>a</sub> = 0.1", {"seen": 1.000, "hold": 0.9974, "nll": 0.0167, "open": 0.1295}),
    ]

    def _cell(cond: str, set_name: str, key: str, fmt: str = "{:.3f}") -> str:
        v = acc(cond, set_name, key)
        return f"{fmt.format(v.mean())} <span class='range'>±{(v.max() - v.min()) / 2:.3f}</span>"

    _rows = "".join(
        f"<tr><th>{LABELS_TXT[c]}</th>"
        f"<td class='num'>{_cell(c, 'named_seen', 'accuracy')}</td>"
        f"<td class='num'>{_cell(c, 'named_holdout', 'accuracy')}</td>"
        f"<td class='num'>{_cell(c, 'named_holdout', 'nll')}</td>"
        f"<td class='num'>{_cell(c, 'open', 'guess_dist')}</td>"
        f"<td class='num'>{acc(c).mean() - CONTROL_ACC:+.3f}</td>"
        f"<td>{'clean' if TASK_CLEAN[c] else 'not clean'}</td></tr>"
        for c in CONDS
    )
    _ref_rows = "".join(
        f"<tr class='ref'><th>{name}</th>"
        f"<td class='num'>{r['seen']:.3f}</td><td class='num'>{r['hold']:.3f}</td>"
        f"<td class='num'>{r['nll']:.3f}</td><td class='num'>{r['open']:.3f}</td>"
        f"<td class='num'>—</td><td>reference</td></tr>"
        for name, r in _REFS
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">seen EM</th><th class="num">holdout EM</th>
        <th class="num">holdout NLL</th><th class="num"><code>open</code> dist</th>
        <th class="num">Δ holdout EM</th><th>H1 gate</th>
      </tr></thead>
      <tbody>{_ref_rows}{_rows}</tbody>
    </table></div>
    """
    _caption = f"""
    Behavior by condition. Each cell is the seed mean, with half the seed range
    beside it. EM is exact match on the answer token. NLL is the surprise of
    that token in nats. <code>open</code> dist is the RGB distance from the
    guessed color to the true mix, on pairs with no named answer. Δ holdout EM
    is the signed gap to the in-experiment control, and the last column reports
    the H1 gate, which passes when that gap is within {ex.TASK_GATE:g}. The two
    top rows are published cells from earlier experiments, shown for comparison.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selectivity (H2)
    """)
    return


@app.cell(hide_code=True)
def _(CONDS, LABELS, m_op1):
    _EX216_MARGIN = 0.2732  # the published ex-2.1.6 λ=0.1 value `span-bare` re-runs
    _m: dict[str, np.ndarray] = {c: m_op1(c) for c in CONDS}

    @themed(
        name="margin-by-condition",
        alt_text="""
            Per-seed margin at op1 for each of the seven conditions, with the
            seed mean drawn as a bar. Reference rules mark the 0.5 gate, the
            0.35 partial level, and the published ex-2.1.6 value of 0.27.
        """,
        caption=rf"""
            The statistic H2(a) scores: $m_{{\text{{op1}}}}$, the layer-mean
            alignment margin at the first operand. One dot per seed, with the
            seed mean as the heavy bar. The solid rule is the H2(a) gate of
            {ex.MARGIN_GATE:g} and the dashed rule the {0.35:g} partial; the dotted
            rule is the published ex-2.1.6 value of {_EX216_MARGIN:.2f}, which the
            <code>span-bare</code> condition re-runs. The four factorial
            conditions are drawn solid, the two arms hollow, and the control grey.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        _ink = dict(zip(CONDS, light_dark(
            ["#999", "#f2b134", "#c1332a", "#e08a2e", "#7a2320", "#3d7ea6", "#5c3d8f"],
            ["#888", "#ffcc66", "#f0665a", "#ffab5e", "#b8564d", "#6ab0d4", "#a78bd6"],
        ), strict=True))  # fmt: skip
        for i, cond in enumerate(CONDS):
            v, color = _m[cond], _ink[cond]
            arm = cond in ("span-anti-late", "span-anti-hi")
            ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=color, lw=3, solid_capstyle="round", zorder=3)
            ax.scatter(
                np.full(len(v), i), v, s=26, zorder=4, color="none" if arm else color,
                edgecolors=color, lw=1.4,
            )  # fmt: skip
        _grey = light_dark("#999", "#777")
        ax.axhline(ex.MARGIN_GATE, color=_grey, lw=1.0)
        ax.axhline(0.35, color=_grey, lw=0.9, ls=(0, (4, 3)))
        ax.axhline(_EX216_MARGIN, color=_grey, lw=0.9, ls=(0, (1, 2)))
        ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
        ax.annotate("H2(a) gate", (len(CONDS) - 0.4, ex.MARGIN_GATE), fontsize=7.5, va="bottom", ha="right",
                    color=light_dark("#444", "#bbb"))  # fmt: skip
        ax.annotate("ex-2.1.6", (len(CONDS) - 0.4, _EX216_MARGIN), fontsize=7.5, va="bottom", ha="right",
                    color=light_dark("#444", "#bbb"))  # fmt: skip
        ax.set_xticks(range(len(CONDS)), [LABELS[c] for c in CONDS], fontsize=8.5)
        ax.set_xlim(-0.6, len(CONDS) - 0.4)
        ax.set_ylabel(r"$m_{\text{op1}}$")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(LABELS, grading, m_op1):
    _factorial: list[str] = list(ex.FACTORIAL)
    _resp: dict[str, tuple[np.ndarray, float, float]] = {c: grading(c) for c in [*_factorial, "lam0"]}
    _order = np.argsort(ex.REDNESS)

    def _windowed(y: np.ndarray, k: int = 25) -> np.ndarray:
        """Mean of *y* over a sliding window of the redness ordering."""
        pad = np.pad(y[_order], (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    @themed(
        name="grading",
        alt_text="""
            Four panels, one per factorial condition, showing the per-color
            alignment at the first operand against the redness of that color.
            The two bare conditions lift the whole cube well above the control;
            the two anti conditions hold most colors near the control and raise
            only the reddest ones.
        """,
        caption=r"""
            Alignment at op1, per color: $\alpha_c$, the layer mean of
            $\cos(h, \hat v_{\text{red}})$ averaged over seeds, against the
            redness of the color. One mark per color, drawn in that color; the
            heavy line is a 25-color sliding mean over the redness ordering, and
            the flat grey band underneath is the same sliding mean for the
            control. The grading statistics and $m_{\text{op1}}$ are quoted per
            panel; both H2(b) gates sit at 0.8. The panels are the 2×2: pull
            span across, repulsive term down.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesGrid

        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharey=True, sharex=True)
        axes = cast(AxesGrid, axes)
        _ctrl = _windowed(_resp["lam0"][0])
        _grid = [["span-bare", "span-anti"], ["op1-bare", "op1-anti"]]
        for row, conds in zip(axes, _grid, strict=True):
            for ax, cond in zip(row, conds, strict=True):
                a, rho, r2 = _resp[cond]
                ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
                ax.plot(
                    ex.REDNESS[_order], _ctrl, color=light_dark("#999", "#777"), lw=3, alpha=0.5, zorder=1,
                    solid_capstyle="round",
                )  # fmt: skip
                ax.scatter(ex.REDNESS, a, c=ex.GRID_RGB, s=18, lw=0.4, zorder=3,
                           edgecolors=light_dark("#00000033", "#ffffff55"))  # fmt: skip
                ax.plot(ex.REDNESS[_order], _windowed(a), color=light_dark("#222", "#eee"), lw=1.6, zorder=4)
                ax.set_title(LABELS[cond], fontsize=10)
                ax.annotate(
                    f"ρ = {rho:.2f}\nR² = {r2:.2f}\n$m$ = {m_op1(cond).mean():.2f}",
                    (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                    color=light_dark("#444", "#bbb"),
                )  # fmt: skip
        for ax in axes[1]:
            ax.set_xlabel("redness of op1")
        for row in axes:
            row[0].set_ylabel(r"$\alpha_c$ at op1")
        axes[0][0].set_ylim(-0.25, 1.25)  # headroom for the per-panel statistics
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Attribution (H3)
    """)
    return


@app.cell(hide_code=True)
def _(m_op1):
    def _effects(stat) -> dict[str, float]:
        """The 2×2 read as main effects and an interaction, on any per-condition statistic."""
        sb, sa, ob, oa = (float(np.mean(stat(c))) for c in ("span-bare", "span-anti", "op1-bare", "op1-anti"))
        return {
            "span-bare": sb, "span-anti": sa, "op1-bare": ob, "op1-anti": oa,
            "anti": (sa + oa) / 2 - (sb + ob) / 2,   # adding the repulsive term, averaged over spans
            "op1": (ob + oa) / 2 - (sb + sa) / 2,    # narrowing the pull, averaged over terms
            "interaction": (sb + oa) - (sa + ob),
        }  # fmt: skip

    _e: dict[str, float] = _effects(m_op1)

    def _c(cond: str) -> str:
        v = m_op1(cond)
        return f"{v.mean():.3f} <span class='range'>±{(v.max() - v.min()) / 2:.3f}</span>"

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th></th><th class="num">bare</th><th class="num">+ anti-subspace</th>
        <th class="num">span effect</th>
      </tr></thead>
      <tbody>
        <tr><th>span pull</th><td class="num">{_c("span-bare")}</td>
            <td class="num">{_c("span-anti")}</td><td class="num">—</td></tr>
        <tr><th>op1-only pull</th><td class="num">{_c("op1-bare")}</td>
            <td class="num">{_c("op1-anti")}</td><td class="num">—</td></tr>
        <tr class="ref"><th>anti effect</th><td class="num">—</td><td class="num">—</td>
            <td class="num">interaction</td></tr>
        <tr class="ref"><th>main effects</th><td class="num" colspan="2">
            anti {_e["anti"]:+.3f} &nbsp;&nbsp; op1-only {_e["op1"]:+.3f}</td>
            <td class="num">{_e["interaction"]:+.3f}</td></tr>
      </tbody>
    </table></div>
    """
    _caption = f"""
    The factorial on $m_{{\\text{{op1}}}}$. Body cells are seed means with half
    the seed range beside them. A main effect is the mean of the two conditions
    carrying a factor minus the mean of the two without it, so the anti effect
    is how much the repulsive term is worth averaged over both pull spans, and
    the op1-only effect is how much narrowing the pull is worth averaged over
    both terms. The interaction is what the two together buy beyond their sum.
    H3 asks the anti effect to reach {ex.MAIN_EFFECT_GATE:g} and to be at least
    as large as the op1-only effect; the noise on a main effect is about
    {ex.NOISE_SEED_MEAN:g}.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Containment and dynamics (H4)
    """)
    return


@app.cell(hide_code=True)
def _(ANCHORED, CONDS, LABELS, traj):
    _EX216_ALPHA = 0.5257  # the published ex-2.1.6 λ=0.1 endpoint for ᾱ

    def _mean_traj(cond: str, key: str) -> np.ndarray:
        return np.mean([traj(cond, s, key) for s in ex.SEEDS], axis=0)

    @themed(
        name="trajectories",
        alt_text="""
            Two panels against epoch. Left: the mean alignment over all colors,
            with the containment gate at 0.1, the partial level at 0.25, and the
            ex-2.1.6 bare-term endpoint at 0.53. Right: the alignment margin,
            with the 0.2 scoring floor. One line per condition, averaged over
            seeds.
        """,
        caption=rf"""
            How the two quantities move during training, measured every
            {ex.TRAJ_STRIDE} steps and averaged over seeds. Left:
            $\bar\alpha$, the mean alignment over all 216 colors at op1 — what ran
            away in ex-2.1.6, and what H4(a) asks the repulsive term to contain
            below {ex.MEAN_ALIGN_GATE:g} (solid rule). The dashed rule is the
            {ex.MEAN_ALIGN_PARTIAL:g} partial and the dotted one the published
            ex-2.1.6 endpoint of {_EX216_ALPHA:.2f}. Right:
            $m_{{\text{{op1}}}}$, with the H4(b) scoring floor of {ex.H4_FLOOR:g}
            dashed; a run whose running maximum never reaches it is reported but
            not scored. The pale band behind both panels is the anchor weight as
            a fraction of its peak, so its descent marks the end-of-training
            anneal.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesRow

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
        axes = cast(AxesRow, axes)
        _ink = dict(zip(CONDS, light_dark(
            ["#999", "#f2b134", "#c1332a", "#e08a2e", "#7a2320", "#3d7ea6", "#5c3d8f"],
            ["#888", "#ffcc66", "#f0665a", "#ffab5e", "#b8564d", "#6ab0d4", "#a78bd6"],
        ), strict=True))  # fmt: skip
        _grey = light_dark("#999", "#777")
        epochs = _mean_traj("span-anti", "epoch")
        weight = _mean_traj("span-anti", "weight")
        for ax, key, ylabel in (
            (axes[0], "alpha_op1", r"$\bar\alpha$ at op1"),
            (axes[1], "m_op1", r"$m_{\text{op1}}$"),
        ):
            twin = ax.twinx()
            twin.fill_between(epochs, weight / weight.max(), color=light_dark("#0000000d", "#ffffff12"), lw=0)
            twin.set_ylim(0, 1.05)
            twin.set_yticks([])
            for cond in CONDS:
                ax.plot(
                    epochs, _mean_traj(cond, key), color=_ink[cond], lw=1.6 if cond in ex.FACTORIAL else 1.2,
                    ls="-" if cond in ex.FACTORIAL or cond == "lam0" else (0, (4, 3)), label=LABELS[cond], zorder=3,
                )  # fmt: skip
            ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, ex.SCHEDULER.epochs)
        axes[0].axhline(ex.MEAN_ALIGN_GATE, color=_grey, lw=1.0)
        axes[0].axhline(ex.MEAN_ALIGN_PARTIAL, color=_grey, lw=0.9, ls=(0, (4, 3)))
        axes[0].axhline(_EX216_ALPHA, color=_grey, lw=0.9, ls=(0, (1, 2)))
        axes[1].axhline(ex.H4_FLOOR, color=_grey, lw=0.9, ls=(0, (4, 3)))
        axes[0].legend(fontsize=7.5, frameon=False, ncols=2, loc="upper left")
        _top = max(np.max(_mean_traj(c, "alpha_op1")) for c in ANCHORED)
        axes[0].set_ylim(-0.1, max(0.75, _top * 1.35))  # headroom for the legend
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(ANCHORED, LABELS_TXT, alpha_bar, traj):
    # H4 as a scored table: containment per condition, retention per run. The
    # retention gate is per *run*, so its column counts runs rather than averaging.
    def _retention(cond: str) -> tuple[list[float], int]:
        ratios, below = [], 0
        for s in ex.SEEDS:
            m = traj(cond, s, "m_op1")
            if m.max() >= ex.H4_FLOOR:
                ratios.append(float(m[-1] / m.max()))
            else:
                below += 1
        return ratios, below

    def _row(cond: str) -> str:
        a = alpha_bar(cond)
        ratios, below = _retention(cond)
        anti = any(c["name"] == cond and c["anti"] for c in ex.CONDITIONS)
        scored = anti and cond != "span-anti-hi"  # H4(a) scopes to the λ_a = 0.1 anti conditions
        gate = "—" if not scored else ("pass" if a.mean() <= ex.MEAN_ALIGN_GATE else "fail")
        if scored and gate == "fail" and a.mean() <= ex.MEAN_ALIGN_PARTIAL:
            gate = "partial"
        r_txt = ", ".join(f"{r:.2f}" for r in ratios) if ratios else f"below floor ({below}/{len(ex.SEEDS)})"
        r_gate = "—" if not ratios else ("pass" if min(ratios) >= ex.H4_RETENTION else "fail")
        return (
            f"<tr><th>{LABELS_TXT[cond]}</th>"
            f"<td class='num'>{a.mean():.3f} <span class='range'>±{(a.max() - a.min()) / 2:.3f}</span></td>"
            f"<td>{gate}</td><td class='num'>{r_txt}</td><td>{r_gate}</td></tr>"
        )

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">ᾱ at op1</th><th>H4(a)</th>
        <th class="num">retention, per seed</th><th>H4(b)</th>
      </tr></thead>
      <tbody>{"".join(_row(c) for c in ANCHORED)}</tbody>
    </table></div>
    """
    _caption = f"""
    The two H4 gates. ᾱ is the end-of-training mean alignment over all 216
    colors at op1, seed mean with half the seed range. H4(a) scores the
    λ<sub>a</sub> = {ex.SCORING_LAMBDA:g} anti conditions against a ceiling of
    {ex.MEAN_ALIGN_GATE:g}, with <em>partial</em> marking a condition that clears
    the {ex.MEAN_ALIGN_PARTIAL:g} level without reaching it. Retention is each
    run's final margin over its running maximum; runs whose maximum never
    reaches {ex.H4_FLOOR:g} are reported but not scored, and H4(b) passes when
    every scored run holds {ex.H4_RETENTION:g}× its peak.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(CONDS, LABELS, arrays, geometry, margin_map):
    _by_layer = {c: margin_map(c)[:, 0] for c in CONDS}
    _seeds = {c: np.array([arrays[f"{c}-s{s}/margin"][:, 0] for s in ex.SEEDS]) for c in CONDS}
    _dot = {c: np.mean([geometry[f"{c}-s{s}"]["centre_dot_anchor"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _spread = {c: np.mean([geometry[f"{c}-s{s}"]["spread"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _depths = np.arange(len(_by_layer["lam0"]))

    @themed(
        name="by-layer",
        alt_text="""
            Three panels against layer depth. Left: the margin at the first
            operand by layer, one line per condition. Middle: the cosine between
            the centre of the 216-color cloud and the anchor direction. Right:
            the extent of that cloud.
        """,
        caption=r"""
            The mechanism picture behind H4(a), per layer, seed means. Depth 0 is
            the token embedding and depth 4 the last block's output. Left: the
            margin at op1 by layer — the per-depth terms whose mean is
            $m_{\text{op1}}$; the shaded band around the control is its seed
            min–max, as the scale of a null. Middle: the cosine between the
            centre of the 216 op1 states and the anchor direction — where the
            cloud sits. Right: the extent of that cloud, the mean squared
            distance of a color from the centre (states are unit-norm, so this
            runs from 0 for a collapsed cloud to 1 for a spread one). The
            repulsive term acts on the middle panel by construction; the right
            panel is what it costs.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesRow

        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
        axes = cast(AxesRow, axes)
        _ink = dict(zip(CONDS, light_dark(
            ["#999", "#f2b134", "#c1332a", "#e08a2e", "#7a2320", "#3d7ea6", "#5c3d8f"],
            ["#888", "#ffcc66", "#f0665a", "#ffab5e", "#b8564d", "#6ab0d4", "#a78bd6"],
        ), strict=True))  # fmt: skip
        band = _seeds["lam0"]
        axes[0].fill_between(_depths, band.min(0), band.max(0), color=_ink["lam0"], alpha=0.25, lw=0)
        panels = ((axes[0], _by_layer, r"$m$ at op1"), (axes[1], _dot, "centre · anchor"),
                  (axes[2], _spread, "extent of the cloud"))  # fmt: skip
        for ax, data, title in panels:
            for cond in CONDS:
                ax.plot(
                    _depths, data[cond], color=_ink[cond], marker="o", ms=3.5,
                    lw=1.8 if cond in ex.FACTORIAL else 1.3,
                    ls="-" if cond in ex.FACTORIAL or cond == "lam0" else (0, (4, 3)), label=LABELS[cond],
                )  # fmt: skip
            ax.set_title(title, fontsize=10)
            ax.set_xticks(_depths)
            ax.set_xlabel("layer (0 = embedding)")
            ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
        axes[2].set_ylim(0, 1.0)
        axes[0].legend(fontsize=7, frameon=False, ncols=2, loc="upper right")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Secondary measurements

    Where redness is readable is carried over from ex-2.1.6 without a gate.
    The question it answers is whether the anchor axis holds a *copy* of the
    concept or the concept itself: in ex-2.1.6, projecting the anchor direction
    out left redness as readable from the other 63 directions as in the control,
    which is what a copy looks like. The repulsive term pushes unlabeled colors
    off the axis, so it acts on exactly this quantity.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS, LABELS, cells):
    _leak = {c: np.mean([cells[f"{c}-s{s}"]["leak_r2"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _axis = {c: np.mean([cells[f"{c}-s{s}"]["axis_r2"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _depths = np.arange(len(_leak["lam0"]))

    @themed(
        name="leakage",
        alt_text="""
            Two panels against layer depth. Left: how well redness can be read
            from the residual stream once the anchor direction is projected out,
            for each condition. Right: how much of the redness variance the
            anchor component carries on its own.
        """,
        caption=r"""
            Where redness is readable at op1. Left: held-out R² for a ridge probe
            that predicts redness from the residual stream at op1, with the
            $\hat v_{\text{red}}$ component removed first — how well *red* can
            still be read from the other 63 directions. Right: the same target
            predicted from the anchor component alone, as squared correlation.
            Per-layer seed means. The sample is the 216 op1 colors, with a 5-fold
            split so no color is scored by a probe that saw it during fitting.
            Neither panel has a pass/fail threshold.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesRow

        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
        axes = cast(AxesRow, axes)
        _ink = dict(zip(CONDS, light_dark(
            ["#999", "#f2b134", "#c1332a", "#e08a2e", "#7a2320", "#3d7ea6", "#5c3d8f"],
            ["#888", "#ffcc66", "#f0665a", "#ffab5e", "#b8564d", "#6ab0d4", "#a78bd6"],
        ), strict=True))  # fmt: skip
        for ax, data, title in ((axes[0], _leak, "off the anchor axis"), (axes[1], _axis, "on the anchor axis")):
            for cond in CONDS:
                ax.plot(
                    _depths, data[cond], color=_ink[cond], marker="o", ms=3.5,
                    lw=1.8 if cond in ex.FACTORIAL else 1.3,
                    ls="-" if cond in ex.FACTORIAL or cond == "lam0" else (0, (4, 3)), label=LABELS[cond],
                )  # fmt: skip
            ax.set_title(title, fontsize=10)
            ax.set_xticks(_depths)
            ax.set_xlabel("layer (0 = embedding)")
            ax.set_ylim(-0.05, 1.0)
        axes[0].set_ylabel("R² for redness")
        axes[0].legend(fontsize=7, frameon=False, ncols=2, loc="lower left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Arms
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
    prioritized: the flattened label affinity if selectivity appeared, or the
    remaining repulsive terms (anti-anchor, separate) or position pooling
    if it did not. Keep interpretation within what the four conditions and
    two arms measured. One trap to avoid: if the margin moves nowhere, the H3
    null and the H2 contrary reading are the same finding reached twice;
    report it once, not as two lines of evidence.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
