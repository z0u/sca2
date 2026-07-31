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

    /// admonition | tl;dr
    Both mechanisms are real, and the blind span is the larger of the two.
    Narrowing the pull to op1 is worth +0.22 of margin and adding the
    repulsive term +0.14, against ex-2.1.6's 0.27 baseline; the two together
    reach 0.64, which clears the 0.5 selectivity gate with a graded response
    (R² = 0.88). None of it cost the task anything measurable, even at ten
    times the anchor weight. So H2 passes, and H3 lands on its preregistered
    contrary reading rather than its hypothesis. H4 fails on both parts: at
    M1's weight ratio the repulsive term reduces the cube-wide drift without
    containing it, and half the anchored conditions still give back more than
    a fifth of their peak margin. The unexpected result is in an arm — holding the
    repulsion near peak through training, instead of annealing it at the
    halfway point as M1 did, beats the M1 schedule on every measurement.
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
    /// admonition | How to read this report
    This was a preregistered skeleton. The method, hypotheses, and decision
    thresholds below were written and frozen before the experiment ran, and
    the results replaced the placeholders in place, so reading it is a
    prediction → observation diff. No threshold was amended after the freeze.
    Anything conceived after seeing the data is under *Exploratory analyses*,
    marked post hoc.

    One thing to keep in view while reading H3 and H4 together. The
    anti-subspace term is, by construction, a penalty on the mean-square
    alignment of the whole cloud — which is the quantity H4(a) scores. So
    "the repulsive term lowers $\bar\alpha$" is close to a restatement of
    what the term is, and it is not on its own evidence that the mechanism
    explains ex-2.1.6. The load-bearing question is H3: whether lowering
    $\bar\alpha$ *buys margin*, a quantity neither factor acts on directly.
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
def _(CONDS, CONTROL_ACC, acc, stats):
    mo.md(rf"""
    **H1 holds everywhere, with two orders of magnitude to spare.** The largest
    gap to the control on `named_holdout` exact match is
    {max(abs(acc(c).mean() - CONTROL_ACC) for c in CONDS):.4f}, against a gate of
    {ex.TASK_GATE:g} — a single example out of {stats["eval_n"]["named_holdout"]}.
    Seen pairs are at 1.000 in every condition. So neither the repulsive term
    nor the narrower pull costs anything the task can feel, and the partial
    outcome H1 named (the factorial clean but the timing arm not) does not
    arise: the timing arm is one of the conditions tied with the control.

    The finer scoring agrees. Answer NLL stays between
    {min(acc(c, "named_holdout", "nll").mean() for c in CONDS):.3f} and
    {max(acc(c, "named_holdout", "nll").mean() for c in CONDS):.3f} nats with no
    ordering by condition, and the `open`-pair distances sit together near
    {np.mean([acc(c, "open", "guess_dist").mean() for c in CONDS]):.3f}, spanning
    {max(acc(c, "open", "guess_dist").mean() for c in CONDS) - min(acc(c, "open", "guess_dist").mean() for c in CONDS):.3f}
    from end to end.

    Where that sits against the nulls needs stating carefully, because it is not
    flattering and it is not about the anchor. The floor is
    {stats["nulls"]["open"]["floor_dist"]:.3f} and a guesser that flips a coin
    between the two names bracketing the true mix scores
    {stats["nulls"]["open"]["k2"]["dist"]:.3f} — so at
    {np.mean([acc(c, "open", "guess_dist").mean() for c in CONDS]):.3f} these
    models are *worse* than that bracketing guesser, picking the nearest name
    about {np.mean([acc(c, "open", "nearest_acc").mean() for c in CONDS]):.0%} of
    the time against its 50%. They are far better than the
    {stats["nulls"]["open"]["blind"]["dist"]:.3f} of a guesser that ignores the
    prompt, so the color is being read; it is the last step of resolving between
    two adjacent names that is imperfect. That is a property of the `v216`
    testbed, not of anchoring: the control sits at
    {acc("lam0", "open", "guess_dist").mean():.3f} with everything else. H1 asks
    whether the anchor moved this number, and it did not.

    The ceiling arm is the interesting row. At $\lambda_\text{{a}} = 1$, ten
    times the scoring rung and three times the largest weight ex-2.1.6 swept,
    holdout accuracy is {acc("span-anti-hi").mean():.4f} — indistinguishable
    from control. So the weight ceiling is still unfound on the task side, and
    the budget available for the D2.2 operation anchors is at least this large.
    Whether more weight is *useful* is a different question, and the answer
    below is no.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selectivity (H2)

    Both factors move the margin, and they move it a long way. The figure
    below puts every condition on the statistic H2(a) scores, against
    ex-2.1.6's value for the same pull.
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
            The panels grade progressively from top left to bottom right. Span
            with a bare anchor is a nearly flat band near 0.5 across every
            color, barely rising with redness. Adding the repulsive term drops
            the grey and green end to about 0.25 while the red end stays near
            0.9. Narrowing the pull to op1 drops the low end further, to about
            0.15 bare and 0.05 with the repulsive term, giving a clean rise from
            near the control baseline up to about 0.9 at the reddest colors.
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
def _(alpha_bar, grading, m_op1):
    _EX216_M, _EX216_A = 0.2732, 0.5257
    mo.md(rf"""
    **H2 passes, on the op1 conditions.** (a) The margin reaches
    $m_{{\text{{op1}}}} = {m_op1("op1-anti").mean():.3f}$ on `op1-anti` and
    {m_op1("op1-bare").mean():.3f} on `op1-bare`, both task-clean, against a gate
    of {ex.MARGIN_GATE:g}. Two conditions clearing it rather than one matters
    here, because the hypothesis takes a maximum over four noisy means and its
    own footnote warns that the maximum of four sits about a noise unit above
    any single one; `op1-anti` clears the gate by
    {m_op1("op1-anti").mean() - ex.MARGIN_GATE:.2f}, roughly four seed-mean noise
    units, so the pass does not rest on that maximum. (b) The response is graded
    on both conditions — but each clears a different one of the two tracks, and
    neither clears both. `op1-bare` reaches ρ = {grading("op1-bare")[1]:.3f}
    against `redness` with R² = {grading("op1-bare")[2]:.3f}; `op1-anti` reaches
    R² = {grading("op1-anti")[2]:.3f} against `sim¹·⁵` with
    ρ = {grading("op1-anti")[1]:.3f}. The gate is a disjunction, so one track is
    what it asks for and both conditions pass as written. Still, `op1-bare`
    clears the rank track by {grading("op1-bare")[1] - ex.GRADE_RHO_GATE:.3f},
    which is thin enough that the pass should be read as resting on `op1-anti`'s
    R², which clears by {grading("op1-anti")[2] - ex.GRADE_R2_GATE:.2f}.

    The step ceilings are the check that makes this a claim about gradedness
    rather than magnitude: a pure step response can score at most
    {ex.STEP_RHO_CEILING:.2f} on the rank track and {ex.STEP_R2_CEILING:.2f} on
    the proportionality track. `op1-anti`'s R² clears its ceiling by
    {grading("op1-anti")[2] - ex.STEP_R2_CEILING:.2f}, `op1-bare`'s ρ by only
    {grading("op1-bare")[1] - ex.STEP_RHO_CEILING:.2f}. The scatter is the more
    convincing evidence in any case: a rise across the whole cube rather than a
    step at the label threshold, so the anchor caught *red* and not merely the
    exemplars that happened to be labeled. (c) The control sits
    at $|m_{{\text{{op1}}}}| = {abs(m_op1("lam0").mean()):.3f}$, inside its 0.1
    ceiling.

    The preregistered primary condition is not among the passes.
    `span-anti` — ex-2.1.6's pull with M1's repulsive company, the condition the
    ex-2.1.6 discussion named as the next experiment — reaches
    {m_op1("span-anti").mean():.3f}. That is comfortably into the partial band
    (0.35) and well clear of ex-2.1.6's {_EX216_M:.2f}, but short of the gate.
    Naming it in advance is what makes that readable: the repulsive term alone
    was the predicted fix, and alone it gets most of the way rather than all of
    it.

    **The replication is bit-identical, which is the useful kind here.**
    `span-bare` re-runs ex-2.1.6's $\lambda = 0.1$ condition through this
    experiment's code, which now carries a second regularizer term at weight
    zero. It returns $m_{{\text{{op1}}}} = {m_op1("span-bare").mean():.3f}$
    against a published {_EX216_M:.4f} and
    $\bar\alpha = {alpha_bar("span-bare").mean():.3f}$ against
    {_EX216_A:.4f} — and the agreement is digit-for-digit per seed, not merely
    within noise. That is worth reading as what it is. It does not independently
    corroborate ex-2.1.6, since nothing stochastic was re-rolled; what it
    verifies is that adding a zero-weighted term perturbed neither the numerics
    nor the RNG stream. So the ex-2.1.6 comparisons this report leans on are
    reproduced rather than assumed, and the four factorial conditions differ from
    each other only in the two factors.
    """)
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
def _(cells, m_op1):
    _sb, _sa, _ob, _oa = (float(m_op1(c).mean()) for c in ("span-bare", "span-anti", "op1-bare", "op1-anti"))
    _anti = (_sa + _oa) / 2 - (_sb + _ob) / 2
    _op1 = (_ob + _oa) / 2 - (_sb + _sa) / 2
    _int = (_sb + _oa) - (_sa + _ob)

    def _per_seed(seed: int) -> tuple[float, float]:
        """(anti, op1) main effects computed within one seed, so the seed cancels."""
        sb, sa, ob, oa = (cells[f"{c}-s{seed}"]["m_op1"] for c in ("span-bare", "span-anti", "op1-bare", "op1-anti"))
        return (sa + oa) / 2 - (sb + ob) / 2, (ob + oa) / 2 - (sb + sa) / 2

    mo.md(rf"""
    **H3 fails, on the first of its two named contrary readings.** Both main
    effects clear {ex.MAIN_EFFECT_GATE:g}: the repulsive term is worth
    {_anti:+.3f} of margin and the narrower pull {_op1:+.3f}, against a noise
    floor of about {ex.NOISE_SEED_MEAN:g} on a main effect. But H3 asked for the
    anti-subspace effect to be *at least as large*, and it is about
    {_op1 / _anti:.1f} times smaller. That is the reading the preregistration
    named: the blind-span interpretation of ex-2.1.6 was correct. Both
    mechanisms were real, and the one the ex-2.1.6 discussion put second is the
    one that mattered more.

    Some care about what "larger" is worth here. The gap between the two effects
    is {abs(_op1 - _anti):.3f}, about {abs(_op1 - _anti) / ex.NOISE_SEED_MEAN:.0f}
    noise units on the generic seed-mean floor — but that floor understates the
    case, because both effects can be computed within a seed and then compared
    pairwise, which cancels the seed. Done that way the op1-only effect is the
    larger on every seed, with no overlap between the two sets
    ({", ".join(f"{_per_seed(s)[1]:.2f} vs {_per_seed(s)[0]:.2f}" for s in ex.SEEDS)}),
    so the ordering is not carried by one lucky run. But the two factors are not
    on a common scale — one changes how many positions are pulled, the other adds
    a term at a weight copied from another experiment — so this says the
    *particular* settings tried favor the span, and not that repulsion is
    intrinsically the weaker lever. The timing arm below makes that concrete: the
    same term on a different schedule is worth considerably more.

    The interaction is {_int:+.3f}, so the two are mildly sub-additive: together
    they buy about {abs(_int):.2f} less than their separate effects would
    suggest. That is the expected shape if they are two routes to one place —
    both ultimately reduce how much of the pull is satisfiable by a
    color-independent shift, so the second one applied has less left to do. It
    also means the interaction outcome H3 named (neither main effect clearing
    while `op1-anti` alone does) is the opposite of what happened.
    """)
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
def _(ANCHORED, alpha_bar, traj):
    _EX216_A = 0.5257

    def _ratios(cond: str) -> list[float]:
        return [float(t[-1] / t.max()) for s in ex.SEEDS if (t := traj(cond, s, "m_op1")).max() >= ex.H4_FLOOR]

    def _peak(cond: str) -> list[float]:
        """The epoch at which each seed's margin trajectory peaks."""
        return [float(traj(cond, s, "epoch")[traj(cond, s, "m_op1").argmax()]) for s in ex.SEEDS]

    _held = [c for c in ANCHORED if _ratios(c) and min(_ratios(c)) >= ex.H4_RETENTION]
    _slid = [c for c in ANCHORED if _ratios(c) and min(_ratios(c)) < ex.H4_RETENTION]
    mo.md(rf"""
    **H4 fails on both parts, and the pattern of failure is the finding.**

    (a) Containment. No anti condition at the scoring rung reaches
    $\bar\alpha \le {ex.MEAN_ALIGN_GATE:g}$:
    `span-anti` ends at {alpha_bar("span-anti").mean():.3f},
    `op1-anti` at {alpha_bar("op1-anti").mean():.3f},
    and the timing arm at {alpha_bar("span-anti-late").mean():.3f}. The named
    partial asked all of them to clear {ex.MEAN_ALIGN_PARTIAL:g}, and `span-anti`
    misses that too, so H4(a) fails outright rather than partially. The term does
    move the quantity it is defined on — from {alpha_bar("span-bare").mean():.2f}
    in the bare condition down to {alpha_bar("span-anti").mean():.2f}, about a
    {1 - alpha_bar("span-anti").mean() / alpha_bar("span-bare").mean():.0%}
    reduction — but at M1's ratio it weakens the cube-wide drift rather than
    containing it. The trajectory shows when the containment is lost:
    $\bar\alpha$ sits near the control for the first forty epochs, while the
    repulsion is still strong, and climbs once the anneal has delivered it to
    {ex.ANTI_HOLD_RATIO:g}$\lambda_\text{{a}}$. That is the contrary outcome the
    preregistration flagged for this figure, and the timing arm was positioned to
    test it.

    (b) Retention. {len(_held)} of the six anchored conditions hold
    {ex.H4_RETENTION:g}× their peak
    ({", ".join(f"`{c}`" for c in _held)}), and {len(_slid)} slide below it
    ({", ".join(f"`{c}`" for c in _slid)}). The gate is written over every
    anchored run, so H4(b) fails.

    The split is informative, but it does not run purely along H3's seam. Both
    op1 conditions hold, at
    {min(min(_ratios(c)) for c in ("op1-bare", "op1-anti")):.2f} or better across
    all six runs — and so does `span-anti-late`, which pulls the full span. What
    the three holding conditions have in common is that something kept the
    color-independent shift down for most of training: a narrow pull in two of
    them, a repulsion held near peak through epoch
    {ex.ANTI_ANNEAL_END_LATE:g} in the third. The three that slide are the ones
    where $\bar\alpha$ was free to climb. The peak moves later wherever that
    happens too — epochs
    {min(_peak("span-bare")):.0f}–{max(_peak("span-bare")):.0f} in `span-bare`,
    {min(_peak("span-anti")):.0f}–{max(_peak("span-anti")):.0f} once the
    repulsive term is added, and
    {min(_peak("span-anti-late")):.0f}–{max(_peak("span-anti-late")):.0f} in the
    timing arm — a slower climb to a higher place rather than an early peak that
    erodes.

    Read with H2 and H3, that is one story rather than three. In ex-2.1.6 the
    margin rose early and then gave back a quarter of itself, and the reading
    offered there was that the rest of the cube was catching up. This experiment
    supports that reading and identifies the shift as the thing doing the
    catching up. Either way of suppressing it — removing the blind positions, or
    keeping the repulsion strong — stops the slide, and `op1-bare` does it
    carrying no repulsive term at all.
    """)
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
def _(geometry, margin_map):
    def _at(cond: str, key: str, layer: int) -> float:
        return float(np.mean([geometry[f"{cond}-s{s}"][key][layer] for s in ex.SEEDS]))

    def _eff(key: str, factor: str = "op1", layer: int = 2) -> float:
        """A factorial main effect read on a geometry statistic rather than on the margin."""
        sb, sa, ob, oa = (_at(c, key, layer) for c in ("span-bare", "span-anti", "op1-bare", "op1-anti"))
        return (ob + oa) / 2 - (sb + sa) / 2 if factor == "op1" else (sa + oa) / 2 - (sb + ob) / 2

    mo.md(rf"""
    The per-layer view says where the improvement lives, and it is not a uniform
    lift. In ex-2.1.6 the margin peaked in the embedding and decayed through the
    stack — {margin_map("span-bare")[0, 0]:.2f} → {margin_map("span-bare")[-1, 0]:.2f}
    here, reproducing it. Both factors flatten that decay rather than raising the
    starting point: `op1-anti` runs {margin_map("op1-anti")[0, 0]:.2f} →
    {margin_map("op1-anti")[-1, 0]:.2f}, so it ends with
    {margin_map("op1-anti")[-1, 0] / margin_map("span-bare")[-1, 0]:.0f}× the
    margin of the bare span condition at the last layer while starting only
    {margin_map("op1-anti")[0, 0] / margin_map("span-bare")[0, 0]:.1f}× as high.
    The timing arm is the clearest case: it leaves the embedding level with the
    bare span condition ({margin_map("span-anti-late")[0, 0]:.2f} against
    {margin_map("span-bare")[0, 0]:.2f}) and finishes highest of any condition
    ({margin_map("span-anti-late")[-1, 0]:.2f}), gaining through the stack where
    every other condition loses. Depth is where the selectivity was being lost,
    and depth is where these interventions act.

    The middle and right panels are the mechanism behind H4(a). In the control
    the centre of the color cloud sits {_at("lam0", "centre_dot_anchor", 2):.2f}
    of the way onto the anchor, the scale of an unrelated direction in 64
    dimensions. The bare span pull swings it to
    {_at("span-bare", "centre_dot_anchor", 2):.2f} — the whole-cube movement
    ex-2.1.6 diagnosed. Each factor pulls that back:
    {_at("span-anti", "centre_dot_anchor", 2):.2f} with the repulsive term,
    {_at("op1-anti", "centre_dot_anchor", 2):.2f} with both, and
    {_at("span-anti-late", "centre_dot_anchor", 2):.2f} in the timing arm, which
    is nearest the control of any anchored condition.

    The extent panel answers a question ex-2.1.6 left open. It found the bare
    anchor compressing the cube mid-stack, to
    {_at("span-bare", "spread", 2) / _at("lam0", "spread", 2):.0%} of the control
    extent, and suggested a `separate`-style term might eventually be needed here
    as it was in M1. On this evidence it is not. Every other factorial condition
    recovers most of the extent, and reading the same 2×2 on spread gives the
    same ordering H3 found on the margin: narrowing the pull is worth
    {_eff("spread"):+.2f} and the repulsive term {_eff("spread", "anti"):+.2f}.
    So this is not a job the repulsion does specifically — `op1-bare` reaches
    {_at("op1-bare", "spread", 2) / _at("lam0", "spread", 2):.0%} of the control
    extent carrying no repulsive term at all, and `op1-anti`
    {_at("op1-anti", "spread", 2) / _at("lam0", "spread", 2):.0%}. Whatever was
    compressing the cube, it was the color-independent shift, and anything that
    reduces the shift restores the extent along with it.

    The ceiling arm is where compression does show up: at
    $\lambda_\text{{a}} = 1$ the cube is narrower even in the token embedding
    ({_at("span-anti-hi", "spread", 0):.2f} against
    {_at("lam0", "spread", 0):.2f} everywhere else), which the pull reaches
    directly. So a `separate`-style term would start to matter past the
    selectivity optimum, not at it.
    """)
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
def _(CONDS, cells):
    _leak = {c: float(np.mean([cells[f"{c}-s{s}"]["leak_r2"] for s in ex.SEEDS])) for c in CONDS}
    _axis = {c: float(np.mean([cells[f"{c}-s{s}"]["axis_r2"] for s in ex.SEEDS])) for c in CONDS}
    mo.md(rf"""
    **The axis became much more readable; everywhere else is unchanged.** On the
    anchor component alone, redness goes from R² = {_axis["lam0"]:.2f} in the
    control to {_axis["op1-bare"]:.2f} and {_axis["op1-anti"]:.2f} in the op1
    conditions — double what the bare span pull achieved
    ({_axis["span-bare"]:.2f}), and tracking the margin ordering closely. So the
    conditions that score well on selectivity are also the ones that put the most
    redness on the axis, which is the consistency check one would want.

    Off the axis, nothing moved. Project the anchor direction out and redness is
    still readable from the remaining 63 directions at
    R² ≈ {min(_leak.values()):.2f}–{max(_leak.values()):.2f} in every condition
    including the control ({_leak["lam0"]:.2f}). This is the same picture
    ex-2.1.6 reported, and neither new factor changes it. The reading offered
    there still stands: redness is a function of color, the task needs the color
    cube, so a probe reading redness off the cube is not evidence of a second
    copy of the concept. What the number does establish is that anchoring
    *concentrated* redness onto the axis without removing it from anywhere else.
    For an intervention that is the relevant caveat — suppressing the axis will
    not remove the model's access to redness — and it is the question D2.2 is for.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Arms
    """)
    return


@app.cell(hide_code=True)
def _(CONTROL_ACC, acc, alpha_bar, cells, grading, m_op1, traj):
    def _ret(cond: str) -> float:
        return float(np.mean([traj(cond, s, "m_op1")[-1] / traj(cond, s, "m_op1").max() for s in ex.SEEDS]))

    def _leak(cond: str) -> float:
        return float(np.mean([cells[f"{cond}-s{s}"]["leak_r2"] for s in ex.SEEDS]))

    _dm = m_op1("span-anti-late").mean() - m_op1("span-anti").mean()
    mo.md(rf"""
    ### The timing arm, which is where the surprise is

    `span-anti-late` differs from `span-anti` in one number: the epoch at which
    the repulsive term finishes annealing down to its hold ratio, moved from
    {ex.ANTI_ANNEAL_END:g} to {ex.ANTI_ANNEAL_END_LATE:g}. Nothing else changes —
    same anchor, same span, same peak and hold ratios, same seeds. It beats the
    M1 schedule on every statistic H2 and H4 score:

    - margin {m_op1("span-anti-late").mean():.3f} against
      {m_op1("span-anti").mean():.3f}, a gap of {_dm:.2f} — about
      {_dm / ex.NOISE_SEED_MEAN:.0f} noise units, and about the size of the whole
      anti-subspace main effect;
    - retention {_ret("span-anti-late"):.2f} against {_ret("span-anti"):.2f},
      moving it from the sliding group to the holding one;
    - grading R² {grading("span-anti-late")[2]:.2f} against
      {grading("span-anti")[2]:.2f};
    - containment {alpha_bar("span-anti-late").mean():.3f} against
      {alpha_bar("span-anti").mean():.3f}, the lowest of any anchored condition —
      listed last because it is the least surprising of the four. More repulsion
      for longer lowering the quantity the repulsion penalizes is close to
      definitional; the margin and retention rows are what make it a result.

    The one measurement that does not improve is off-axis leakage, where the arm
    is the highest of any condition ({_leak("span-anti-late"):.2f} against
    {_leak("span-anti"):.2f}) — a direction this report treats elsewhere as one
    you would rather not see grow. It is a small difference on a measurement with
    no gate, but it is the reason to say "every scored statistic" rather than
    "everything".

    The arm was written to ask whether the M1 recipe is forgiving on this axis,
    with either answer acceptable and no gate attached. It is not forgiving: a
    2× stretch of one anneal is worth more than adding the term was. Nobody had
    measured this before — the M1 schedules varied by experiment, and ex-2.4.1
    ran its terms *up* over training rather than down — so the value was
    inherited rather than chosen. It now looks worth choosing deliberately, and
    a dedicated sweep over the anneal endpoint is the obvious next step. Two
    honest limits on that claim: this is one alternative timing rather than a
    curve, and it is confounded with total repulsive strength, since holding
    near peak for longer also delivers more of it. Separating "when" from "how
    much" needs the sweep, not this arm.

    ### The ceiling arm, where more weight is not better

    `span-anti-hi` runs the full recipe at $\lambda_\text{{a}} =
    {ex.CEILING_LAMBDA:g}$, ten times the scoring rung, with the repulsive term
    scaled in proportion. Two things follow.

    The task ceiling remains unfound. Holdout accuracy is within
    {abs(acc("span-anti-hi").mean() - CONTROL_ACC):.4f} of control, so at ten
    times the scoring weight the task still pays nothing. Whatever bounds the anchor
    weight for D2.2, it is not the task loss at this scale, and chasing the real
    ceiling would need a dedicated sweep well above $\lambda_\text{{a}} = 1$.

    But the selectivity ceiling has been passed. At ten times the weight the
    margin is {m_op1("span-anti-hi").mean():.3f}, level with `span-anti`'s
    {m_op1("span-anti").mean():.3f} at a tenth of it — the gap is
    {abs(m_op1("span-anti-hi").mean() - m_op1("span-anti").mean()):.3f}, inside
    the seed-mean noise floor, so read it as "no better" rather than "worse". The
    other three measurements are outside the noise and all point one way:
    $\bar\alpha$ back up to {alpha_bar("span-anti-hi").mean():.3f}, retention
    down to {_ret("span-anti-hi"):.2f}, and the cube compressed even in the token
    embedding. So a heavier pull buys more alignment and no more selectivity: it
    reaches the point where moving everything is the cheapest way to satisfy the
    term, which is ex-2.1.6's failure mode arrived at from a different direction.
    The practical consequence is that the anchor weight is not the knob to turn
    for selectivity, and the two factors this experiment tested are.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Post hoc: conceived after seeing the results, scoring no hypothesis. It reads
    the preregistered measurements rather than anything new.

    ### Why doesn't the repulsive term push *red* off the anchor too? (post hoc)

    The anti-subspace term is indiscriminate by construction — it penalizes
    $\cos^2(h, \hat v_{\text{red}})$ over every position of every line, labeled
    or not. So the timing arm should be alarming: holding that penalty near peak
    for another forty epochs ought to push the anchored concept off the anchor
    along with everything else, and instead it produces the best margin in the
    experiment. Either the term is somehow sparing the labeled lines, or the two
    effects are not the same size.

    It is the second. Decomposing the margin into its two halves shows the term
    does reach *red* — it just loses the race.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS, arrays):
    _top = np.argsort(ex.REDNESS)[-10:]  # the ten reddest colors, carrying 85% of the label mass

    def _alpha(cond: str) -> np.ndarray:
        return np.mean([arrays[f"{cond}-s{s}/alpha"] for s in ex.SEEDS], axis=0)[:, :, 0].mean(axis=0)

    def _row(cond: str) -> str:
        a = _alpha(cond)
        return (
            f"<tr><th>{cond}</th><td class='num'>{a[_top].mean():.3f}</td>"
            f"<td class='num'>{a.mean():.3f}</td><td class='num'>{float(ex.LABEL_W @ a) - a.mean():.3f}</td></tr>"
        )

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr><th>condition</th><th class="num">ᾱ, ten reddest</th>
      <th class="num">ᾱ, all 216</th><th class="num">margin</th></tr></thead>
      <tbody>{"".join(_row(c) for c in CONDS)}</tbody>
    </table></div>
    """
    _caption = f"""
    The margin split into the two quantities it differences. The first column is
    the mean alignment of the ten reddest colors — the ones carrying
    {ex.LABEL_W[_top].sum():.0%} of the label mass, so effectively "where the
    anchored concept sits". The second is the mean over all 216, the quantity the
    repulsive term penalizes. Comparing <code>span-anti</code> with
    <code>span-anti-late</code> is the case of interest: sustained repulsion costs
    the reddest colors a little and the rest of the cube a great deal.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(arrays):
    _top = np.argsort(ex.REDNESS)[-10:]

    def _alpha(cond: str) -> np.ndarray:
        return np.mean([arrays[f"{cond}-s{s}/alpha"] for s in ex.SEEDS], axis=0)[:, :, 0].mean(axis=0)

    _dred = _alpha("span-anti")[_top].mean() - _alpha("span-anti-late")[_top].mean()
    _dall = _alpha("span-anti").mean() - _alpha("span-anti-late").mean()
    _span_ratio = ex.LIVE_PER_BATCH / ex.PULLED_PER_BATCH[ex.SPAN_FULL]
    _op1_ratio = ex.LIVE_PER_BATCH / ex.PULLED_PER_BATCH[ex.SPAN_OP1]
    mo.md(rf"""
    Stretching the anneal costs the reddest colors
    {_dred:.3f} of alignment and the cube as a whole {_dall:.3f} —
    {_dall / _dred:.0f} times as much. So the term is doing exactly what its
    definition says, *red* included; the margin improves because the same
    indiscriminate push is far more consequential for the unlabeled bulk than
    for the labeled reds.

    The reason is in how the two terms are normalized, which is a design detail
    rather than a fact about concepts. Neither weight depends on the labels —
    $\lambda_\text{{a}}$ and $\lambda_{{\bar{{\text{{s}}}}}}$ are functions of
    the epoch alone, identical in every batch. What differs is the denominator
    *inside* each term. Both are means; the anchor averages over the positions
    it pulls, the repulsion over every live position. Measured on the real
    sampler, a batch has {ex.LIVE_PER_BATCH:,.0f} live positions, while the
    anchor fires in {ex.FIRING_RATE[ex.SPAN_FULL]:.0%} of batches and averages
    {ex.PULLED_PER_BATCH[ex.SPAN_FULL]:.1f} pulled positions when it does. So a
    labeled state feels a pull scaled by
    $1/{ex.PULLED_PER_BATCH[ex.SPAN_FULL]:.1f}$ against a push scaled by
    $1/{ex.LIVE_PER_BATCH:,.0f}$ — about {_span_ratio:,.0f}:1 before weights.
    Even at the opening $\lambda_{{\bar{{\text{{s}}}}}} =
    {ex.ANTI_PEAK_RATIO:g}\lambda_\text{{a}}$, and after the $2\cos$ factor from
    differentiating $\cos^2$, the pull wins at a pulled position by a couple of
    hundred to one. An *unlabeled* state has no pull at all, so it feels only the
    push and goes wherever the task will let it.

    Because the anchor is a mean rather than a sum, the per-position pull is also
    insensitive to *how many* labels a batch happens to carry: a batch with one
    labeled line pulls its positions as hard as a batch with three. What the
    label rate sets is how often the term fires at all.

    ### A confound in H3's second factor, and why the ceiling arm resolves it (post hoc)

    That same normalization means the op1-only pull is not purely a narrowing.
    Dividing the same $\lambda_\text{{a}}$ among
    {ex.PULLED_PER_BATCH[ex.SPAN_OP1]:.1f} positions instead of
    {ex.PULLED_PER_BATCH[ex.SPAN_FULL]:.1f} makes each surviving position's pull
    about {ex.PULLED_PER_BATCH[ex.SPAN_FULL] / ex.PULLED_PER_BATCH[ex.SPAN_OP1]:.1f}×
    stronger ({_op1_ratio:,.0f}:1 against the repulsion, rather than
    {_span_ratio:,.0f}:1). So H3's op1 factor changes two things at once: which
    positions are pulled, and how hard each one is pulled. Read on its own, its
    main effect could be either.

    The ceiling arm separates them, which is not what it was put there for. It
    multiplies $\lambda_\text{{a}}$ by ten — every pulled position's gradient
    scaled by 10, well past the ~4× that narrowing delivers — and selectivity
    got *worse*, not better. If per-position pull strength were what buys margin,
    the ceiling arm should be the best condition in the experiment and it is
    among the worst. So the op1 effect is carried by which positions are pulled,
    not by how hard, and H3's reading stands. A cleaner design would still fix
    this: normalizing by a fixed count rather than by the realized mask would
    make span a pure narrowing, and that is worth doing before the factor is
    used again.

    Two things to carry forward. The repulsive term's strength *relative to the
    anchored concept* is set by the label rate as much as by
    $\lambda_{{\bar{{\text{{s}}}}}}$ — sparser labels make the same weight
    gentler on the concept and no gentler on everything else, an interaction
    neither this experiment nor M1 has mapped. And it explains why the timing arm
    had room to improve at all: at these ratios the term is nowhere near strong
    enough to threaten the concept, so "more repulsion for longer" was nearly
    free. That predicts a limit — push $\lambda_{{\bar{{\text{{s}}}}}}$ far
    enough and the reddest colors should come off the axis in earnest, and the
    $2\cos$ gradient means it bites hardest on the states nearest the axis, which
    are exactly the anchored ones. Finding that point is what the queued anneal
    sweep should look for.
    """)
    return


@app.cell(hide_code=True)
def _(alpha_bar, grading, m_op1):
    mo.md(rf"""
    ## Discussion

    **A selective anchor in a transformer.** Ex-2.1.6 could show that an anchor
    was free but not that it was selective, which left D2.1 short of its claim:
    "SCA transfers to transformers" needs a concept that lands where it was put,
    not just a regularizer the task tolerates. This experiment supplies that.
    `op1-anti` reaches a margin of {m_op1("op1-anti").mean():.2f} against a
    preregistered gate of {ex.MARGIN_GATE:g}, with a graded response
    (R² = {grading("op1-anti")[2]:.2f} against the M1-derived shape) and no
    measurable task cost. That is the missing piece of D2.1, and it arrived
    without either of the two things one might have reached for next — a
    heavier pull, which the ceiling arm shows makes matters worse, or the
    remaining M1 repulsive terms, which were not needed.

    **What the factorial attributed.** Both candidate mechanisms from the
    ex-2.1.6 discussion are real, and their sizes are the opposite of the order
    that discussion put them in. Narrowing the pull to the position that carries
    the labeled color is worth more than adding M1's repulsive term, and the
    conditions that pull the whole span are exactly the conditions whose margin
    still slides during training. The interpretation the preregistration named
    for this outcome is that the blind-span reading of ex-2.1.6 was correct: when
    three of four pulled positions cannot see which color they are being pulled
    for, the optimizer supplies a color-independent shift, and that shift is what
    dilutes the margin.

    **A caution about how far that generalizes.** The span pull is not an
    arbitrary choice we can simply drop. It is the shape sequence-level labeling
    forces, because a document label marks no position as the relevant one. Our
    op1-only pull is available only because this synthetic language has a known
    position that carries the concept — which is exactly what a real corpus does
    not give you. So the honest statement of the finding is about what
    sequence-level labeling costs, not about a design M3 can keep. What transfers
    is the diagnosis; the remedy needs a mechanism for locating the concept
    within a labeled span, and the margin-by-position profile suggests the model
    itself knows where it is.

    **The repulsive term is worth keeping, on a different schedule.** At M1's
    weight ratio it fails H4(a): it reduces the cube-wide drift by about
    {1 - alpha_bar("span-anti").mean() / alpha_bar("span-bare").mean():.0%}
    without containing it, and $\bar\alpha$ climbs once the anneal hands it down
    to {ex.ANTI_HOLD_RATIO:g}$\lambda_\text{{a}}$. But the timing arm — one
    number changed, the anneal endpoint stretched from epoch
    {ex.ANTI_ANNEAL_END:g} to {ex.ANTI_ANNEAL_END_LATE:g} — improves margin,
    containment, retention and grading together, by more than adding the term
    was worth in the first place. The M1 keyframes were inherited from a
    5-dimensional autoencoder bottleneck and mapped onto our 100 epochs by
    fraction of training; there was never a reason to expect them to be right
    for a 64-dimensional residual stream. That makes the anneal endpoint the
    best-supported next sweep.

    **Where this leaves the open questions.** The best-supported next
    measurement is a sweep of the anti-subspace anneal endpoint against its hold
    ratio, since the timing arm confounds "when the repulsion acts" with "how
    much of it there is" and a grid separates them. The position question is
    harder and more interesting: it wants a pull that finds the concept inside a
    labeled span rather than being told where it is — pooling over the span, or
    weighting positions by an attention-derived estimate. And the flattened
    label-affinity question raised in earlier experiments reads differently now
    that a graded response exists to compare against, since the grading here is
    measured under a `redness⁸` distribution that puts
    {ex.LABEL_W[np.argsort(ex.REDNESS)[-10:]].sum():.0%} of its mass on ten
    colors. The exploratory section adds a fourth: the repulsive term's strength
    relative to the anchored concept turns out to be set by the label rate as
    much as by its weight, and that interaction is unmapped.

    One thing this experiment does *not* license. The off-axis leakage is
    unchanged from ex-2.1.6 in every condition: redness stays as readable from
    the other 63 directions as it is in the control. A selective anchor is not
    the same as an exclusive one, and nothing here bounds what suppressing the
    axis would do to the model's access to *red*. That is D2.2's question, and
    this experiment supplies the anchored models to ask it of rather than an
    answer to it.
    """)
    return


if __name__ == "__main__":
    app.run()
