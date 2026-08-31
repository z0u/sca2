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
        """Resolve metrics and the stacked per-run arrays from the store, or None if unpublished."""
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
        """The per-run geometry pass, published under its own ref."""
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

    /// tip |
    <!-- tl;dr -->
    We tested two mechanisms to improve anchor selectivity: **1.** Apply the anchor term only to operand 1 (no other tokens), and **2.** Add a repulsive term to clear the target subspace. Both work, but 1. worked better, and their effects stack somewhat.
    ///

    Ex-2.1.6 anchored *red* with a single attractive term and got alignment without selectivity. The term moved the whole color cube onto the anchor direction, and the red-selective margin settled at about half its threshold, flat across a tenfold range of the anchor weight. Two candidate mechanisms could explain that, and the experiment could not tell them apart.

    First, nothing repelled the rest of the cube. M1 never ran its anchor bare: it added an indiscriminate *anti-subspace* term, at a few percent of the anchor weight, penalizing the average squared alignment between the anchor direction and every representation in the batch. That is a gentle, global pressure that keeps the cloud as a whole off the axis, and that average is the quantity that ran away in ex-2.1.6.

    Second, most of the pull was blind. The anchor pulled four prompt positions of each labeled equation, and three of them (`+`, op2, `=`) carry no information about which color sits at op1. There the label is an unobservable coin flip, so only the expected pull is visible to the optimizer, and that is nearly color-independent.

    This experiment separates the two with a 2×2 factorial[^factorial] at the ex-2.1.6 scoring rung ($\lambda_\text{a} = 0.1$): {bare anchor, anchor + anti-subspace} × {four-position span pull, op1-only pull}. If the anti-subspace factor restores the margin, missing repulsion was the problem. If the op1-only factor does, the blind span was. If only the combination does, they compound.

    Two extra arms ride along: a schedule-timing variant of the anti-subspace anneal, and a weight-ceiling rung at $\lambda = 1$.

    [^factorial]: A design that runs every combination of two on/off
    factors, four conditions here, so each factor's effect can be measured on its own.

    /// details | Notation
    Notation carries over from ex-2.1.6, with loss weights now subscripted by their term. New here: the bar in $\lambda_{\bar{\text{s}}}$ marks a repulsive term, whose weight is always given relative to the anchor as the ratio $\lambda_{\bar{\text{s}}} / \lambda_\text{a}$.

    | symbol | meaning |
    | --- | --- |
    | $\hat v_{\text{red}}$ | the anchor direction |
    | $\lambda_\text{a}$ | anchor weight (ex-2.1.6's bare $\lambda$) |
    | $\lambda_{\bar{\text{s}}}$ | anti-subspace weight |
    | $\ell$ | layer index |
    | $m$, $m_{\text{op1}}$ | alignment margin; its layer mean at op1 |
    | $\bar\alpha$ | mean alignment with $\hat v_{\text{red}}$ over all 216 colors, layer-averaged at op1 |
    | **run** | a **condition** crossed with a seed: one training run |
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], CONTROL_ACC, acc, alpha_bar, grading, m_op1, traj):
    # REVIEW: added the Findings section. Every verdict and its deciding number
    # already appeared in an analysis section; nothing here is new, and no gate
    # or verdict changed. Verify against the H1 table, the margin figure, the H3
    # factorial table, and the H4 table, in that order.
    _sb, _sa, _ob, _oa, _la = (
        float(m_op1(c).mean()) for c in ("span-bare", "span-anti", "op1-bare", "op1-anti", "span-anti-late")
    )

    def _ret(cond: str) -> float:
        """Lowest final-over-peak margin across the seeds that reached the floor; 0 if none did."""
        r = [t[-1] / t.max() for s in ex.SEEDS if (t := traj(cond, s, "m_op1")).max() >= ex.H4_FLOOR]
        return min(r) if r else 0.0

    _held = sum(1 for c in CONDS if c != "lam0" and _ret(c) >= ex.H4_RETENTION)
    mo.md(rf"""
    **H1 (task cost) — holds.** Largest `named_holdout` exact-match gap from the control, across all seven conditions: {max(abs(acc(c).mean() - CONTROL_ACC) for c in CONDS):.4f}. Gate: {ex.TASK_GATE:g}.

    **H2 (selectivity) — holds, on the op1 conditions.** `op1-anti` reaches $m_{{\text{{op1}}}} = {_oa:.3f}$ and `op1-bare` {_ob:.3f}, against a gate of {ex.MARGIN_GATE:g}. Both are graded (`op1-anti` R² = {grading("op1-anti")[2]:.2f}; `op1-bare` ρ = {grading("op1-bare")[1]:.2f}, gates {ex.GRADE_R2_GATE:g}), and the control sits at $|m_{{\text{{op1}}}}| = {abs(m_op1("lam0").mean()):.3f}$ against a ceiling of {ex.CONTROL_MARGIN_GATE:g}. The preregistered primary condition `span-anti` reaches {_sa:.3f}, a partial.

    **H3 (attribution) — fails.** Both main effects clear {ex.MAIN_EFFECT_GATE:g}, but the anti-subspace effect ({(_sa + _oa) / 2 - (_sb + _ob) / 2:+.3f}) is smaller than the op1-only effect ({(_ob + _oa) / 2 - (_sb + _sa) / 2:+.3f}), not larger; the ordering holds within every seed.

    **H4 (containment and dynamics) — fails on both parts.** (a) No anti condition at the scoring rung falls to $\bar\alpha \le {ex.MEAN_ALIGN_GATE:g}$; `span-anti` ends at {alpha_bar("span-anti").mean():.3f}, missing the {ex.MEAN_ALIGN_PARTIAL:g} partial too. (b) {_held} of the six anchored conditions hold {ex.H4_RETENTION:g}× their peak margin; the gate asks for all of them.

    **The timing arm scores nothing and did best on containment.** `span-anti-late` stretches the anti-subspace anneal from epoch {ex.ANTI_ANNEAL_END:g} to {ex.ANTI_ANNEAL_END_LATE:g} and changes nothing else. It improves on `span-anti`, the schedule it varies, on every statistic H2 and H4 score: margin {_sa:.3f} → {_la:.3f}, $\bar\alpha$ {alpha_bar("span-anti").mean():.3f} → {alpha_bar("span-anti-late").mean():.3f} (the lowest of any anchored condition), and on retention it crosses from the sliding group to the holding one. Its margin still sits below `op1-anti`'s {_oa:.3f}. See *Arms*.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this report
    This report was preregistered in Git. We wrote and froze the method, hypotheses, and decision thresholds before running the experiment, then replaced the placeholders with results in place. No threshold was amended after the freeze. Anything conceived after seeing the data is under *Exploratory analyses*, marked post hoc.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    ### Testbed, labels, and measurements

    Everything ex-2.1.6 held is held here: the word-level `v216` testbed from ex-2.1.3 (216 grid colors, one token each, d64-L4), the same corpus seed and eval sets, labels drawn per visit with probability `redness(op1)⁸ × 0.08`, and the exhaustive 27-partner alignment probe set. The measurements are also the same: the alignment map over layers × positions, the label-affinity-weighted margin, the pair of redness probes, and the margin trajectory every 50 steps. The [ex-2.1.6 report](../ex-2.1.6/report.py) documents each; only the changes are described below.

    One measurement is promoted: the exploratory geometry pass from ex-2.1.6 (per layer: the centroid of the 216 op1 states, its alignment with the anchor, and the extent of the cloud) is preregistered here, because H4 scores part of it. The mean alignment over all colors, $\bar\alpha$, is likewise recorded at every trajectory checkpoint beside the margin. In ex-2.1.6 it was added as an unscored diagnostic, and it turned out to be important.

    ### The sweep

    All four conditions run at $\lambda_\text{a} = 0.1$, the scoring rung of ex-2.1.6, reused here by default: every rung was task-clean and 0.1 sat in the middle of the swept range. The control ($\lambda_\text{a} = 0$) and the `span-bare` condition reproduce the ex-2.1.6 control and its $\lambda_\text{a} = 0.1$ condition. So every comparison in this report is between conditions that went through identical code.

    **Factor one: the anti-subspace term.** The `anti` conditions add the M1 repulsive term to the loss at weight $\lambda_{\bar{\text{s}}}$:

    $$\lambda_{\bar{\text{s}}}\, \mathcal{L}_{\bar{\text{s}}}, \qquad \mathcal{L}_{\bar{\text{s}}} = \operatorname{mean}\,\cos^2(h, \hat v_{\text{red}})$$

    where the mean runs over every residual-stream slice and every non-pad position of the batch: every line, labeled or not. This is M1's term with the reserved coordinate axis replaced by an arbitrary unit direction: M1 penalized $\sum_{i} \hat z_i^2$ over the reserved axes of the *normalized* representation, and each $\hat z_i^2$ is a $\cos^2$ against a basis vector. A higher-dimensional reserved subspace would sum $\cos^2$ over an orthonormal basis of it; here the subspace is the one-dimensional span of the anchor. The term penalizes the mean-square alignment of everything, so the labeled pull has to buy alignment against a headwind. That is, we expect, the selectivity pressure ex-2.1.6 lacked. It never asks any particular point to leave the axis, only that the cloud as a whole not sit on it.

    The term should be cheap in a 64-dimensional stream: an isotropic cloud (one with no preferred direction) has a mean $\cos^2$ of 1/64 per slice, and the task's three color dimensions fit in the remaining 63 with room to spare. In M1's 4- and 5-dimensional bottlenecks it cost a fifth or a quarter of the space, and its harder conditions (ex-2.9.1) needed the dimension cleared outright. The question is whether, at 3% of the anchor weight, it is strong enough to matter.

    **Factor two: the pull span.** The `span` conditions pull the four prompt positions (op1, `+`, op2, `=`) of a labeled equation, as ex-2.1.6 did. The `op1` conditions pull op1 alone, so every pulled state belongs to the token that carries the labeled color. If the color-independent drift came from the three blind positions, narrowing the pull removes it with no repulsive term at all.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Schedule

    The anchor schedule is unchanged from ex-2.1.6: ramp over the 10-epoch LR warmup, hold at $\lambda_\text{a}$, anneal from epoch 90 to a floor of $0.1\lambda_\text{a}$ at 100.

    The anti-subspace schedule is copied from M1/Ex-2.9.1 and stretched to our 100 epochs by fraction of training. M1 balanced the attractive and repulsive terms in time rather than by a single ratio. Anti-subspace opened at 2.5× the anchor peak weight, meaning it was at full weight from step 0 (its effective push still scaled by the warming-up learning rate, like every term), before the anchor had ramped in. It then annealed to 3% of the anchor by the halfway point and held there. Mapped onto our frame: $\lambda_{\bar{\text{s}}} / \lambda_\text{a}$ starts at 2.5 at epoch 0, anneals (minimum-jerk) to 0.03 by epoch 50, and holds; from epoch 90 both terms share the end-of-training anneal, so their ratio is constant from the midpoint on.

    One arm probes the timing: `span-anti-late` stretches the anneal endpoint from epoch 50 to 90, holding the repulsion near peak through most of training.

    The second arm, `span-anti-hi`, runs the full recipe at $\lambda_\text{a} = 1$, one rung above the maximum Ex-2.1.6 swept, with $\lambda_{\bar{\text{s}}}$ scaled in proportion. Both terms scale together, so it probes the headroom of the recipe, not of the bare anchor.
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
    The seven conditions, each run at seeds {ex.SEEDS} — {len(ex.CONDITIONS) * len(ex.SEEDS)} runs. The four factorial conditions share λ<sub>a</sub> = {ex.SCORING_LAMBDA:g} and carry the H2 and H3 gates; the two arms add none of their own, though H1 and H4 reach them by their stated scopes — the timing arm is a λ<sub>a</sub> = {ex.SCORING_LAMBDA:g} anti condition, and H4(b) covers every anchored run. λ<sub>s̄</sub>/λ<sub>a</sub> is the anti-subspace weight relative to the anchor peak.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _():
    _epochs = np.linspace(0, 100, 1001)

    @themed(
        name="schedules",
        alt_text="""
            Weight schedules over 100 epochs on a log axis. The learning rate warms up by epoch 10 and decays smoothly. The anchor weight ramps up with the warmup, holds at 0.1, and anneals to a floor after epoch 90. The anti-subspace weight starts at 0.25 — above everything else — and descends to 0.003 by epoch 50, where it holds; a dashed variant descends to the same level by epoch 90 instead.
        """,
        caption=f"""
            The three schedules at the scoring rung (λ<sub>a</sub> = {ex.SCORING_LAMBDA:g}), log scale. The anti-subspace term opens at {ex.ANTI_PEAK_RATIO:g}λ<sub>a</sub>, dominant before the anchor arrives, and anneals to {ex.ANTI_HOLD_RATIO:g}λ<sub>a</sub> by epoch {ex.ANTI_ANNEAL_END:g} (the M1/ex-2.9.1 keyframes, mapped by fraction of training). The dashed line is the `span-anti-late` arm, which reaches the hold ratio at epoch {ex.ANTI_ANNEAL_END_LATE:g} instead. From epoch {ex.ANNEAL_START:g} the anchor and anti-subspace weights share the end-of-training anneal to a floor of {ex.ANNEAL_FLOOR:g}× their held values.
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
    # effect is a difference of two two-condition means, so its noise is about the same
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

    Gates carry over from ex-2.1.6 with their thresholds and rationale, as do its noise floors (margin ≈ 0.056 per seed, ≈ 0.032 on a three-seed mean): same architecture, same measurement. Unless stated otherwise, results are reported on `span-anti` (seed-averaged), the condition the ex-2.1.6 discussion named as the next experiment.

    **H1.** The added term and the resulting narrower pull are as free as the lone anchor was: every $\lambda_\text{a} = 0.1$ condition (the four factorial conditions and the timing arm) is task-clean, meaning `named_holdout` exact match within 0.02 (absolute) of the seed mean of the in-experiment control. Partial: the factorial is clean but the timing arm is not, which would itself be evidence that sustained repulsion has a price.

    **H2.** With the missing ingredient restored, the concept lands selectively. On some task-clean factorial condition: (a) the layer-mean margin at op1 reaches $m_{\text{op1}} \ge 0.5$; (b) the response is graded, meaning that over the 216 colors, the seed-mean layer-mean alignment at op1 clears Spearman $\rho \ge 0.8$ against `redness` or Pearson $R^2 \ge 0.8$ against `sim¹·⁵` (both tracks, thresholds, and step ceilings as justified at length in ex-2.1.6); (c) the control shows $|m_{\text{op1}}| \le 0.1$. Partial: some condition clears 0.35 without reaching 0.5; 0.35 is above the ex-2.1.6 value of 0.27 by more than the seed-mean noise.[^h2max]

    **H3.** The margin failure in ex-2.1.6 was mostly the missing repulsion, not the blind span. This is scored on the main effects of the factorial on $m_{\text{op1}}$ (for each factor: the mean over its two conditions, differenced): the anti-subspace main effect is at least 0.1 and at least as large as the op1-only main effect.[^h3noise] Two outcomes that fail H3 are named in advance so a miss stays readable; unlike the partials elsewhere, they are contrary readings rather than weaker passes. The op1-only effect clears 0.1 and is the larger of the two, which reads as the blind-span interpretation of ex-2.1.6 was correct (whether or not the anti-subspace effect also clears it); or neither main effect clears 0.1 but the `op1-anti` condition alone does (an interaction: each mechanism blocks the margin on its own).

    **H4.** The repulsive term visibly does its two jobs: it keeps the cube as a whole off the axis, and the margin holds its early peak instead of sliding back.
    (a) Containment: in every anti condition at $\lambda_\text{a} = 0.1$, the end-of-training mean alignment over all colors at op1 satisfies $\bar\alpha \le 0.1$.[^h4gate]
    (b) Retention: the ex-2.1.6 trajectory gate, unchanged: for every anchored run whose running maximum of $m_{\text{op1}}$ reaches 0.2, the final value is at least 0.8× that maximum.[^h4slide] Partials, in decreasing strength: every anti condition clears the old halving level $\bar\alpha \le 0.25$ without all reaching 0.1, which reads as repulsion weakening the runaway without containing it; or (a) holds and (b) fails only in the timing arm, whose sustained repulsion changes the trajectory most.

    [^h2max]: One caution: "some condition" takes a maximum over the four factorial conditions, and the maximum of four noisy means sits about one noise unit above any single one. So a bare-partial result resting on exactly one condition at 0.35 is weaker than the same number on the pre-named primary condition.

    [^h3noise]: A main effect is a difference of two-condition means, so its noise is about the three-seed floor of 0.032, and 0.1 is about 3× that.

    [^h4gate]: Why 0.1: the ex-2.1.6 reference points are 0.53 for the bare term and 0.008 for the control, with a seed spread of about 0.02. Working repulsion should push $\bar\alpha$ to near-control levels; 0.1 is a fifth of the bare term and still five noise units above control, so a working mechanism passes comfortably and a merely-weakened runaway does not.

    [^h4slide]: In ex-2.1.6 the margin peaked near epoch 10 and slid back a quarter under steady pressure. If the slide was the rest of the cube catching up to a selective early response, repulsion should hold the peak.
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
def _(CONDS: list[str], CONTROL_ACC, LABELS_TXT, TASK_CLEAN, acc):
    # Two external references, both published: ex-2.1.3's v216 condition (the
    # un-anchored testbed this is all built on) and ex-2.1.6's λ=0.1 condition, which
    # `span-bare` re-runs. Neither belongs to this sweep.
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
    Behavior by condition. Each cell is the seed mean, with half the seed range beside it. EM is exact match on the answer token. NLL is the surprise of that token in nats. <code>open</code> dist is the RGB distance from the guessed color to the true mix, on pairs with no named answer. Δ holdout EM is the signed gap to the in-experiment control, and the last column reports the H1 gate, which passes when that gap is within {ex.TASK_GATE:g}. The two top rows are published conditions from earlier experiments, shown for comparison.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], CONTROL_ACC, acc, stats):
    mo.md(rf"""
    **H1 holds everywhere, with two orders of magnitude to spare.** On `named_holdout` exact match, the largest gap between any condition and the control is {max(abs(acc(c).mean() - CONTROL_ACC) for c in CONDS):.4f}. The gate allows up to {ex.TASK_GATE:g}; the observed gap amounts to a single example out of {stats["eval_n"]["named_holdout"]}. Seen pairs are at 1.000 in every condition. So neither the repulsive term nor the narrower pull has any measurable cost to the task. The partial outcome we planned for, where the factorial arms pass but the timing arm does not, did not arise: the timing arm is one of the conditions tied with the control.

    The finer scoring agrees. Answer NLL (the model's surprise at the correct token, in nats) stays between {min(acc(c, "named_holdout", "nll").mean() for c in CONDS):.3f} and {max(acc(c, "named_holdout", "nll").mean() for c in CONDS):.3f} with no ordering by condition. The `open`-pair distances cluster near {np.mean([acc(c, "open", "guess_dist").mean() for c in CONDS]):.3f}, and the spread from best to worst condition is only {max(acc(c, "open", "guess_dist").mean() for c in CONDS) - min(acc(c, "open", "guess_dist").mean() for c in CONDS):.3f}.[^open-good]

    [^open-good]: How good is that `open`-pair number in absolute terms? Not very, though the
    shortfall has nothing to do with the anchor. The floor (the best score any guesser could reach) is {stats["nulls"]["open"]["floor_dist"]:.3f}, and a guesser that flips a coin between the two names bracketing the true mix scores {stats["nulls"]["open"]["k2"]["dist"]:.3f}. At {np.mean([acc(c, "open", "guess_dist").mean() for c in CONDS]):.3f}, these models are *worse* than that coin-flip guesser: they pick the nearest name about {np.mean([acc(c, "open", "nearest_acc").mean() for c in CONDS]):.0%} of the time, versus its 50%. But they are far better than the {stats["nulls"]["open"]["blind"]["dist"]:.3f} of a guesser that ignores the prompt. So the models do read the color; what is imperfect is the last step, choosing between two adjacent names. That is a property of the `v216` testbed, not of anchoring: the control sits at {acc("lam0", "open", "guess_dist").mean():.3f}, right alongside the anchored conditions.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selectivity (H2)

    Both factors (the repulsive term and the narrower pull) increase the margin substantially. The figure below shows the statistic that H2(a) scores, for every condition, alongside the published ex-2.1.6 value for the same pull.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], LABELS, m_op1):
    _EX216_MARGIN = 0.2732  # the published ex-2.1.6 λ=0.1 value `span-bare` re-runs
    _m: dict[str, np.ndarray] = {c: m_op1(c) for c in CONDS}

    @themed(
        name="margin-by-condition",
        alt_text="""
            Per-seed margin at op1 for each of the seven conditions, with the seed mean drawn as a bar. Reference rules mark the 0.5 gate, the 0.35 partial level, and the published ex-2.1.6 value of 0.27.
        """,
        caption=rf"""
            The statistic H2(a) scores: $m_{{\text{{op1}}}}$, the layer-mean alignment margin at the first operand. One dot per seed, with the seed mean as the heavy bar. The solid rule is the H2(a) gate of {ex.MARGIN_GATE:g} and the dashed rule the {0.35:g} partial; the dotted rule is the published ex-2.1.6 value of {_EX216_MARGIN:.2f}, which the
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
            Four panels, one per factorial condition, showing the per-color alignment at the first operand against the redness of that color. The panels grade progressively from top left to bottom right. Span with a bare anchor is a nearly flat band near 0.5 across every color, barely rising with redness. Adding the repulsive term drops the grey and green end to about 0.25 while the red end stays near 0.9. Narrowing the pull to op1 drops the low end further, to about 0.15 bare and 0.05 with the repulsive term, giving a clean rise from near the control baseline up to about 0.9 at the reddest colors.
        """,
        caption=r"""
            Alignment at op1, per color: $\alpha_c$, the layer mean of $\cos(h, \hat v_{\text{red}})$ averaged over seeds, against the redness of the color. One mark per color, drawn in that color; the heavy line is a 25-color sliding mean over the redness ordering, and the flat grey band underneath is the same sliding mean for the control. The grading statistics and $m_{\text{op1}}$ are quoted per panel; both H2(b) gates sit at 0.8. The panels are the 2×2: pull span across, repulsive term down.
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
    **H2 passes, on the op1 conditions.** (a) The margin reaches $m_{{\text{{op1}}}} = {m_op1("op1-anti").mean():.3f}$ on `op1-anti` and {m_op1("op1-bare").mean():.3f} on `op1-bare`, both task-clean, against a gate of {ex.MARGIN_GATE:g}. The hypothesis takes a maximum over four noisy means, but `op1-anti` clears the gate by {m_op1("op1-anti").mean() - ex.MARGIN_GATE:.2f}, roughly four seed-mean noise units, so the pass does not rest on that maximum. (b) The response is graded on both conditions, but each clears a different one of the two tracks, and neither clears both. `op1-bare` reaches ρ = {grading("op1-bare")[1]:.3f} against `redness` with R² = {grading("op1-bare")[2]:.3f}; `op1-anti` reaches R² = {grading("op1-anti")[2]:.3f} against `sim¹·⁵` with ρ = {grading("op1-anti")[1]:.3f}. The gate asks for either track, so both conditions pass. Still, `op1-bare` clears the rank track by only {grading("op1-bare")[1] - ex.GRADE_RHO_GATE:.3f}. That is thin, so the pass is better read as resting on the R² of `op1-anti`, which clears by {grading("op1-anti")[2] - ex.GRADE_R2_GATE:.2f}.

    The step-ceilings check qualifies this. A pure step response (one that jumps at the label threshold instead of rising smoothly) can score at most {ex.STEP_RHO_CEILING:.2f} on the rank track and {ex.STEP_R2_CEILING:.2f} on the proportionality track. The R² of `op1-anti` clears its ceiling by {grading("op1-anti")[2] - ex.STEP_R2_CEILING:.2f}; the ρ of `op1-bare` by only {grading("op1-bare")[1] - ex.STEP_RHO_CEILING:.2f}. The scatter is the more convincing evidence in any case: it rises across the whole cube rather than stepping at the label threshold, so the anchor caught *red* and not merely the exemplars that happened to be labeled. (c) The control sits at $|m_{{\text{{op1}}}}| = {abs(m_op1("lam0").mean()):.3f}$, inside its 0.1 ceiling.

    The preregistered primary condition `span-anti` is not among the passes. It reaches {m_op1("span-anti").mean():.3f}: comfortably into the partial band (0.35) and well clear of the ex-2.1.6 result of {_EX216_M:.2f}, but short of the gate. The repulsive term alone was the predicted fix, and it gets most (but not all) of the way there.

    /// details | The replication is bit-identical
    `span-bare` re-runs the $\lambda = 0.1$ condition of ex-2.1.6 through this experiment's code, which now carries a second regularizer term at weight zero. It returns $m_{{\text{{op1}}}} = {m_op1("span-bare").mean():.3f}$ against a published {_EX216_M:.4f}, and $\bar\alpha = {alpha_bar("span-bare").mean():.3f}$ against {_EX216_A:.4f}. The agreement is digit-for-digit per seed. This does not independently corroborate ex-2.1.6, since nothing stochastic was re-rolled. What it verifies is that adding a zero-weighted term perturbed neither the numerics nor the RNG stream. So the ex-2.1.6 comparisons this report leans on are reproduced rather than assumed, and the four factorial conditions differ from each other only in the two factors.
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
    _caption = mo.md(f"""
    The factorial on $m_{{\\text{{op1}}}}$. Body cells are seed means, with half the seed range beside them. Anti effect: margin gained by adding the repulsive term, averaged over both pull spans; op1-only effect: margin gained by narrowing the pull, averaged over both terms; interaction: what the two together buy beyond their sum. H3 asks the anti effect to reach {ex.MAIN_EFFECT_GATE:g} and to be at least as large as the op1-only effect; the noise on a main effect is about {ex.NOISE_SEED_MEAN:g}.
    """)
    mo.Html(figure_html(_table, caption=_caption.text, class_="report-figure"))
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
    **H3 fails, and implies the first of its two named contrary readings.** Both main effects clear {ex.MAIN_EFFECT_GATE:g}: the repulsive term is worth {_anti:+.3f} of margin, and the narrower pull is worth {_op1:+.3f}, against a noise floor of about {ex.NOISE_SEED_MEAN:g} on a main effect. But H3 asked for the anti-subspace effect to be *at least as large*, and instead the op1-only effect is the larger, by a factor of about {_op1 / _anti:.1f}. That is the reading the preregistration named: the blind-span interpretation of ex-2.1.6 was correct.

    The gap between the two effects is {abs(_op1 - _anti):.3f}, about {abs(_op1 - _anti) / ex.NOISE_SEED_MEAN:.0f} noise units on the generic seed-mean floor, and that floor understates the case. Computed within a seed, which cancels seed-to-seed variation, the op1-only effect is the larger on every seed, with no overlap between the two sets ({", ".join(f"{_per_seed(s)[1]:.2f} vs {_per_seed(s)[0]:.2f}" for s in ex.SEEDS)}), so the ordering is not due to one lucky run.

    Still, the two factors are not on a common scale: one changes how many positions are pulled, and the other adds a term at a weight copied from another experiment. So this says the *particular* settings tried favor the span, not that repulsion is intrinsically the weaker lever.

    The interaction is {_int:+.3f}: mildly sub-additive, the expected shape if the two are routes to one place. Both reduce how much of the pull can be satisfied by a color-independent shift, so whichever is applied second has less left to do. It also means the interaction outcome H3 named (neither main effect clearing while `op1-anti` alone does) is the opposite of what happened.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Containment and dynamics (H4)

    The anti-subspace term penalizes the mean-square alignment of the whole cloud, which is the quantity H4(a) scores. So "the repulsive term lowers $\bar\alpha$" is close to a tautology, and on its own not evidence that this mechanism explains ex-2.1.6. H3 carries the load: does lowering $\bar\alpha$ gain margin, a quantity neither factor acts on directly?
    """)
    return


@app.cell(hide_code=True)
def _():
    # Weight schedules for the figure, computed from `CONDITIONS` via the same
    # functions the training loop calls. That way the figure cannot show a
    # schedule the runs did not actually train under. These are sampled finely;
    # the recorded trajectories, by contrast, are coarse and start after the
    # warmup.
    SCHED_E = np.linspace(0, ex.SCHEDULER.epochs, 1001)

    def _curves(c: dict) -> list[tuple[str, np.ndarray]]:
        curves = [("anchor", ex.anchor_weight(SCHED_E, peak=c["lam"]))]
        if c["anti"]:
            curves.append(("anti", ex.anti_subspace_weight(
                SCHED_E, lam=c["lam"], anneal_end=c.get("anti_anneal_end", ex.ANTI_ANNEAL_END),
            )))  # fmt: skip
        return curves

    SCHEDULES = {c["name"]: _curves(c) for c in ex.CONDITIONS}
    """(term, weight) curves per condition, `term` being one of anchor, anti, ghost."""

    # Each arm is a variation on the primary schedule, so wherever an arm is
    # drawn, the primary schedule is also drawn in ghost ink. This mirrors the
    # faint traces in the trajectory panels.
    _GHOST = [("ghost", w) for _, w in SCHEDULES[ex.PRIMARY]]
    SCHEDULES |= {c: _GHOST + SCHEDULES[c] for c in ("span-anti-late", "span-anti-hi")}
    return SCHEDULES, SCHED_E


@app.cell(hide_code=True)
def _(ANCHORED, CONDS: list[str], LABELS, SCHEDULES, SCHED_E, traj):
    _EX216_ALPHA = 0.5257  # the published ex-2.1.6 λ=0.1 endpoint for ᾱ
    _KEYS = ("alpha_op1", "m_op1")
    # Compute seed means once, outside the plot function: @themed calls it
    # twice (light and dark variants).
    _MEAN = {c: {k: np.mean([traj(c, s, k) for s in ex.SEEDS], axis=0) for k in _KEYS} for c in CONDS}
    _EPOCHS = np.mean([traj("span-anti", s, "epoch") for s in ex.SEEDS], axis=0)
    # Each condition gets a pair of axes: the trajectory on top, and directly
    # under it the weight schedule that condition trained under. The spacer row
    # separates the two blocks while keeping each pair together. In the first
    # two columns, rows are the two anchor widths; the third column holds the
    # two arms. The control has no column of its own; instead it is drawn into
    # every panel as a grey reference pair.
    _GRID = [
        ["span-bare", "span-anti", "span-anti-late"],
        ["w-span-bare", "w-span-anti", "w-span-anti-late"],
        [".", ".", "."],
        ["op1-bare", "op1-anti", "span-anti-hi"],
        ["w-op1-bare", "w-op1-anti", "w-span-anti-hi"],
    ]
    _PANELS = [c for r in (0, 3) for c in _GRID[r]]
    _LEFT = ("span-bare", "op1-bare")

    @themed(
        name="trajectories",
        alt_text="""
            Six pairs of panels in a three-by-two grid, one pair per condition, sharing the epoch axis. In each pair, the upper panel shows mean alignment (solid) and margin (dashed) over a grey control pair and faint traces of the other conditions; the shorter log-scale panel below shows that condition's weight schedule, anchor in red and repulsion in blue. Carets on the left spine mark the containment gates at 0.1 and 0.25; one on the right marks the margin floor at 0.2. A dotted rule at the ex-2.1.6 endpoint of 0.53, labeled in the first panel, meets the solid line there. In the anti conditions the margin climbs early, while the repulsion is strong; mean alignment stays near the control until the anneal lowers the repulsion, then climbs, latest in the late arm, whose blue curve descends latest.
        """,
        caption=rf"""
            Training dynamics, one pair of panels per condition. Above: seed means of the two cosines on the anchor axis, on one shared scale. Solid is $\bar\alpha$, the mean alignment over all 216 colors at op1; dashed is the margin $m_{{\text{{op1}}}}$. The grey pair is the control; the faint lines are the other conditions. The carets on the left spine are the H4(a) grading levels for $\bar\alpha$, and the one on the right is the H4(b) floor for $m_{{\text{{op1}}}}$. The dotted rule is the published ex-2.1.6 endpoint of {_EX216_ALPHA:.2f}. Since span-bare re-runs that condition, its solid line landing on the rule is the reproduction check. Seed spread is given in the H4 table below. Below each: the weight schedule that condition trained under, in the colors of the schedules figure above: anchor $\lambda_\mathrm{{a}}$ in red, repulsion $\lambda_{{\bar{{\mathrm{{s}}}}}}$ in blue (absent in the bare conditions). The two arms in the third column also show the primary schedule in ghost ink, since each is a variation on it.
        """,
    )
    def _plot() -> plt.Figure:
        # A tall mosaic: trajectory row, schedule row, spacer, then both again.
        fig, axd = plt.subplot_mosaic(
            _GRID, figsize=(7.5, 6), height_ratios=[3, 1.5, 0.45, 3, 1.5], sharex=True,
        )  # fmt: skip
        _ink = dict(zip(CONDS, light_dark(
            ["#999", "#f2b134", "#c1332a", "#e08a2e", "#7a2320", "#3d7ea6", "#5c3d8f"],
            ["#888", "#ffcc66", "#f0665a", "#ffab5e", "#b8564d", "#6ab0d4", "#a78bd6"],
        ), strict=True))  # fmt: skip
        _grey = light_dark("#999", "#777")
        _ghost = light_dark("#e2e2e2", "#333")
        # The H4 levels, in a pale steel blue: quiet against the data, and
        # distinct from the grey of the control and the ex-2.1.6 rule.
        # _gate = light_dark("#a4bdd1", "#41607a")
        _gate = light_dark("#555", "#bbb")
        _term = {"anchor": light_dark("#c33", "#e66"), "anti": light_dark("#36c", "#7af"), "ghost": _ghost}
        _dash = (0, (6, 1))
        # One y-scale for every panel, so a line's height means the same thing everywhere.
        top = max(max(np.max(_MEAN[c]["alpha_op1"]), np.max(_MEAN[c]["m_op1"])) for c in ANCHORED)
        for cond in _PANELS:
            ax = axd[cond]
            for other in _PANELS:
                if other != cond:
                    ax.plot(_EPOCHS, _MEAN[other]["alpha_op1"], color=_ghost, lw=0.4, zorder=1)
                    ax.plot(_EPOCHS, _MEAN[other]["m_op1"], color=_ghost, lw=0.4, zorder=1)
            # Mark each level with a caret on the spine, rather than a rule
            # across the whole panel. Under this transform, x is in axes
            # coordinates: 0 is the left spine and 1 the right. Markers 9 and 8
            # have their base at the point, so each protrudes inward. The gates
            # for ᾱ go on the left spine and the floor for m on the right,
            # which tells them apart without needing a second color.
            for _level in (ex.MEAN_ALIGN_GATE, ex.MEAN_ALIGN_PARTIAL):
                ax.plot(0, _level, marker=9, ms=3, color=_gate, clip_on=False, zorder=6,
                        transform=ax.get_yaxis_transform())  # fmt: skip
            ax.plot(1, ex.H4_FLOOR, marker=8, ms=3, color=_gate, clip_on=False, zorder=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
            # The one full-width rule: it marks an outside result, not a level
            # on our own axis, so a caret would be the wrong device.
            ax.axhline(_EX216_ALPHA, color=_grey, lw=0.8, ls=(0, (1, 2.5)), zorder=2)
            ax.plot(_EPOCHS, _MEAN["lam0"]["alpha_op1"], color=_grey, lw=1.1, zorder=3)
            ax.plot(_EPOCHS, _MEAN["lam0"]["m_op1"], color=_grey, lw=1.1, ls=_dash, zorder=3)
            ax.plot(_EPOCHS, _MEAN[cond]["m_op1"], color=_ink[cond], lw=1.4, ls=_dash, zorder=4)
            ax.plot(_EPOCHS, _MEAN[cond]["alpha_op1"], color=_ink[cond], lw=1.7, zorder=5)
            ax.set_title(LABELS[cond], fontsize=9.5, color=_ink[cond])
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_ylim(-0.08, top * 1.08)
            ax.tick_params(labelbottom=False, labelleft=cond in _LEFT)
        axd["span-bare"].set_ylabel("cosine", fontsize="x-small")
        axd["op1-bare"].set_ylabel("cosine", fontsize="x-small")
        # Label the control once, next to its line; elsewhere the grey pair is
        # recognizable on its own. The label sits over other ink, so a white
        # halo keeps it legible.
        import matplotlib.patheffects as pe

        _halo = [pe.withStroke(linewidth=2.5, foreground=light_dark("#ffffff", "#000000"))]
        axd["span-bare"].annotate(
            "control", (55, _MEAN["lam0"]["alpha_op1"][-1]), fontsize=7.5, color=_grey, va="center", ha="center",
            path_effects=_halo,
        )  # fmt: skip
        # Label the dotted rule in the panel that re-runs it. span-bare repeats
        # the λ=0.1 condition of ex-2.1.6, so its solid line landing on the rule is
        # the reproduction check; labeling it there makes that point without a
        # legend entry.
        axd["span-bare"].annotate(
            "ex-2.1.6", (72, _EX216_ALPHA), fontsize=7.5, color=_grey, va="bottom", ha="center",
            path_effects=_halo,
        )  # fmt: skip
        # Key the two line styles once, above the grid, in neutral ink. Within
        # a panel, color only distinguishes conditions.
        fig.legend(
            [plt.Line2D([], [], color=_grey, lw=1.7), plt.Line2D([], [], color=_grey, lw=1.5, ls=_dash)],
            [
                rf"$\bar\alpha$, graded at the left carets ({ex.MEAN_ALIGN_GATE:g}, {ex.MEAN_ALIGN_PARTIAL:g})",
                rf"$m_{{\text{{op1}}}}$, floored at the right caret ({ex.H4_FLOOR:g})",
            ],
            fontsize=8.5, frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            labelcolor=_grey, handlelength=2.4,
        )  # fmt: skip

        # Weight schedules, one under each condition, all on the same log scale
        # so schedules can be compared across columns as well as down a pair.
        for cond in _PANELS:
            ax = axd[f"w-{cond}"]
            for term, w in SCHEDULES[cond]:
                ax.plot(SCHED_E, w, color=_term[term], lw=1.0 if term == "ghost" else 1.4)
            ax.set_yscale("log")
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_ylim(2e-4, 4)
            ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
            ax.minorticks_off()  # a decade of unlabelled minor ticks is a smear at this height
            ax.tick_params(labelbottom=True, labelleft=cond in _LEFT, labelsize=7.5)
        axd["w-span-bare"].set_ylabel("weight", fontsize="x-small")
        axd["w-op1-bare"].set_ylabel("weight", fontsize="x-small")
        for _c in _GRID[-1]:
            axd[_c].set_xlabel("epoch", fontsize="x-small")
        # Label the two terms once, on the panel where both are at full
        # strength. The bare panels get a note that the repulsion is zero,
        # so it doesn't read as merely off-scale.
        _zero = r"$\lambda_{\bar{\mathrm{s}}} = 0$"
        for _x, _y, _txt, _k in ((62, 0.13, r"$\lambda_\mathrm{a}$", "anchor"), (36, 0.4, r"$\lambda_{\bar{\mathrm{s}}}$", "anti")):  # fmt: skip
            axd["w-span-anti"].annotate(_txt, (_x, _y), fontsize=9, color=_term[_k], va="bottom")
        axd["w-span-bare"].annotate(_zero, (50, 4e-4), fontsize=8, color=_term["anti"], va="bottom")
        axd["w-op1-bare"].annotate(_zero, (50, 4e-4), fontsize=8, color=_term["anti"], va="bottom")
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
    The two H4 gates. ᾱ is the end-of-training mean alignment over all 216 colors at op1, seed mean with half the seed range. H4(a) scores the λ<sub>a</sub> = {ex.SCORING_LAMBDA:g} anti conditions against a ceiling of {ex.MEAN_ALIGN_GATE:g}, with <em>partial</em> marking a condition that clears the {ex.MEAN_ALIGN_PARTIAL:g} level without reaching it. Retention is each run's final margin over its running maximum; runs whose maximum never reaches {ex.H4_FLOOR:g} are reported but not scored, and H4(b) passes when every scored run holds {ex.H4_RETENTION:g}× its peak.
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
    **H4 fails on both parts.**

    (a) Containment. No *anti* condition at the scoring rung falls to $\bar\alpha \le {ex.MEAN_ALIGN_GATE:g}$: `span-anti` ends at {alpha_bar("span-anti").mean():.3f}, `op1-anti` at {alpha_bar("op1-anti").mean():.3f}, and the timing arm at {alpha_bar("span-anti-late").mean():.3f}. The named partial asked all of them to clear the looser bar of {ex.MEAN_ALIGN_PARTIAL:g}, and `span-anti` misses that too, so H4(a) fails outright rather than partially.

    The repulsive term does reduce the quantity it is defined on: $\bar\alpha$ ends at {alpha_bar("span-bare").mean():.2f} in the bare condition and {alpha_bar("span-anti").mean():.2f} with the term. But at the weight ratio carried over from M1, the term only weakens the cube-wide drift; it does not contain it.

    The trajectory shows when containment is lost. While the repulsion outweighs the pull, $\bar\alpha$ sits near the control. It climbs once the anneal has brought $\lambda_{{\bar{{\mathrm{{s}}}}}}$ down to about a tenth of $\lambda_\text{{a}}$. So each condition breaks at a time set by its own schedule: around epoch 40 on the default anneal, and epoch 75 on the late one. The preregistration flagged this contrary outcome as a possibility for this figure, and the timing arm was positioned to test it.

    (b) Retention. {len(_held)} of the six anchored conditions hold {ex.H4_RETENTION:g}× their peak ({", ".join(f"`{c}`" for c in _held)}), and {len(_slid)} slide below it ({", ".join(f"`{c}`" for c in _slid)}). The gate applies to every anchored run, so H4(b) fails.

    The split does not fall purely along the same line as H3. Both op1 conditions hold, at {min(min(_ratios(c)) for c in ("op1-bare", "op1-anti")):.2f} or better across all six runs. So does `span-anti-late`, which pulls the full span.

    What the three holding conditions have in common is that something held the color-independent shift down. In two of them a narrow pull lowers the level $\bar\alpha$ settles at; in the third, a repulsion held near peak through epoch {ex.ANTI_ANNEAL_END_LATE:g} delays the climb. The three that slide are the ones where $\bar\alpha$ was free to climb. Wherever it climbs, the peak also moves later: epochs {min(_peak("span-bare")):.0f}–{max(_peak("span-bare")):.0f} in `span-bare`, {min(_peak("span-anti")):.0f}–{max(_peak("span-anti")):.0f} once the repulsive term is added, and {min(_peak("span-anti-late")):.0f}–{max(_peak("span-anti-late")):.0f} in the timing arm. That looks like a slower climb to a higher place, rather than an early peak that erodes.

    In ex-2.1.6 the margin rose early and then gave back a quarter of itself, and the reading offered there was that the rest of the cube was catching up. This experiment supports that reading, and points to the color-independent shift as the thing doing the catching up. `op1-bare` stops the slide while carrying no repulsive term at all.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], LABELS, arrays, geometry, margin_map):
    _by_layer = {c: margin_map(c)[:, 0] for c in CONDS}
    _seeds = {c: np.array([arrays[f"{c}-s{s}/margin"][:, 0] for s in ex.SEEDS]) for c in CONDS}
    _dot = {c: np.mean([geometry[f"{c}-s{s}"]["centre_dot_anchor"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _spread = {c: np.mean([geometry[f"{c}-s{s}"]["spread"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _depths = np.arange(len(_by_layer["lam0"]))

    @themed(
        name="by-layer",
        alt_text="""
            Three panels against layer depth, one line per condition. The margin decays with depth in every condition, least in the timing arm; the cloud centre swings far onto the anchor direction under the bare span pull and much less under every other condition; and the cloud's extent shrinks with depth, with the bare span pull and the ceiling arm the narrowest.
        """,
        caption=r"""
            Margin and cloud geometry by layer, seed means. Depth 0 is the token embedding and depth 4 the last block's output. **Left:** the margin at op1 by layer — the per-depth terms whose mean is $m_{\text{op1}}$; the shaded band around the control is its seed min–max, as the scale of a null. **Middle:** the cosine between the centre of the 216 op1 states and the anchor direction — where the cloud sits. **Right:** the extent of that cloud, the mean squared distance of a color from the centre (states are unit-norm, so this runs from 0 for a collapsed cloud to 1 for a spread one). The repulsive term acts on the middle panel by construction; the right panel is what it costs.
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
            ax.set_xticks(_depths, ["emb", *map(str, range(1, len(_depths)))])
            ax.set_xlabel("slice")
            ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
        axes[2].set_ylim(0, 1.0)
        axes[0].legend(fontsize=7, frameon=False, ncols=2, loc="upper right")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], LABELS_TXT, geometry, margin_map):
    def _at(cond: str, key: str, layer: int) -> float:
        return float(np.mean([geometry[f"{cond}-s{s}"][key][layer] for s in ex.SEEDS]))

    _ctrl_spread = _at("lam0", "spread", 2)

    def _row(cond: str) -> str:
        cls = ' class="ref"' if cond == "lam0" else ""
        return (
            f"<tr{cls}><th>{LABELS_TXT[cond]}</th>"
            f'<td class="num">{margin_map(cond)[0, 0]:.2f}</td>'
            f'<td class="num">{margin_map(cond)[-1, 0]:.2f}</td>'
            f'<td class="num">{_at(cond, "centre_dot_anchor", 2):.2f}</td>'
            f'<td class="num">{_at(cond, "spread", 2) / _ctrl_spread:.0%}</td></tr>'
        )

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th></th><th class="num">m, embedding</th><th class="num">m, last layer</th>
        <th class="num">centre · anchor</th><th class="num">extent</th>
      </tr></thead>
      <tbody>{"".join(_row(c) for c in CONDS)}</tbody>
    </table></div>
    """
    _caption = mo.md(r"""
    Per-layer geometry at op1, seed-averaged, decoding the three panels above. First two columns: the margin at the token embedding and at the last layer. Last two, read mid-stack at layer 2 where the control cloud is still broad: the cosine between the centre of the color cloud and the anchor direction, and the extent of the cloud as a fraction of the control's.
    """)
    mo.Html(figure_html(_table, caption=_caption.text, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(geometry, margin_map):
    def _at(cond: str, key: str, layer: int) -> float:
        return float(np.mean([geometry[f"{cond}-s{s}"][key][layer] for s in ex.SEEDS]))

    def _eff(key: str, factor: str = "op1", layer: int = 2) -> float:
        """A factorial main effect read on a geometry statistic rather than on the margin."""
        sb, sa, ob, oa = (_at(c, key, layer) for c in ("span-bare", "span-anti", "op1-bare", "op1-anti"))
        return (ob + oa) / 2 - (sb + sa) / 2 if factor == "op1" else (sa + oa) / 2 - (sb + ob) / 2

    # REVIEW: three changes in the prose below, all narrowing claims to what the
    # per-layer numbers show.
    # (1) "Both factors flatten that decay rather than raising the starting
    #     point" -> "act mostly on that decay rather than on the starting
    #     point": the op1 factor does raise layer 0, from 0.448 (span-bare) to
    #     0.740 (op1-bare), which the next sentence then quotes as 1.6x. The
    #     depth claim survives because the end-of-stack ratio is 4x.
    # (2) The timing arm "gaining through the stack where every other condition
    #     loses" -> rises mid-stack and finishes about where it started: it runs
    #     0.450 -> 0.428, a slight net loss, with the gain confined to layers
    #     1-2. Verify: `margin_map("span-anti-late")[:, 0]`.
    # (3) The control's centre-anchor cosine of 0.02 was glossed as "about what
    #     an unrelated direction gives in 64 dimensions". A random direction in
    #     d64 gives |cos| ~ 1/8, so 0.02 is well below that, not typical of it;
    #     replaced with a plain statement of what the number means. (Same class
    #     of slip as the H4(a) note in the Hypotheses cell.) The layer these
    #     geometry numbers are read at (2) was also unstated, and is now named.

    mo.md(rf"""
    **Increased selectivity at deeper layers.** The margin in Ex-2.1.6 peaked in the embedding and decayed through the stack (reproduced here). Both new factors act mostly on the decay rather than on the starting point: `op1-anti` starts {margin_map("op1-anti")[0, 0] / margin_map("span-bare")[0, 0]:.1f}× the bare span condition and ends the stack at {margin_map("op1-anti")[-1, 0] / margin_map("span-bare")[-1, 0]:.0f}× its margin.

    The timing arm makes that plainest. It leaves the embedding approx. equal to `span-bare`, rises through the middle of the stack, and finishes highest of any condition — about where it started, while every other condition ends well below its own embedding value.

    **Both factors independently reduce the whole-cube shift.** The anchor axis carries essentially nothing about where the control cloud sits, and the bare span pull (ex-2.1.6) swings it most of the way over. The timing arm resists the whole-cube shift best out of the tested conditions.

    **A *separate* term may not be needed.** In ex-2.1.6, the bare anchor compressed the cube mid-stack, suggesting that the *anti-subspace* and pairwise *separation* terms from M1 may be needed. But in this experiment, most of the extent was recovered by *anti-subspace* alone.

    Restoring the extent is not something the repulsion does specifically, yet the spread improves over the factorial conditions in the same order as the margin $m$. It seems the color-independent shift was what compressed the cube, so anything that reduces the shift restores it.

    **Compression recurs at high term weight (ceiling arm).** At $\lambda_\text{{a}} = 1$ the cube is narrower even in the token embedding ({_at("span-anti-hi", "spread", 0):.2f} against {_at("lam0", "spread", 0):.2f} everywhere else). So a `separate`-style term may yet be needed in some configurations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Secondary measurements

    We reuse the "where is redness readable" measurement from ex-2.1.6, without a gate. Two probes read the same target from complementary parts of the residual stream: one from the anchor coordinate alone, and one from everything else. The repulsive term pushes unlabeled colors off the axis, so it acts on both.

    The two have different reference points. The anchor coordinate starts empty and can reach 1, so that probe measures how much redness anchoring put there. The other 63 directions encode the color cube regardless of what the anchor does. Redness is a function of color, so that probe has an RGB floor it cannot go far below: it reports whether the cube survived rather than whether the concept moved.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], LABELS, cells):
    _leak = {c: np.mean([cells[f"{c}-s{s}"]["leak_r2"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _axis = {c: np.mean([cells[f"{c}-s{s}"]["axis_r2"] for s in ex.SEEDS], axis=0) for c in CONDS}
    _depths = np.arange(len(_leak["lam0"]))
    _floor = ex.redness_rgb_floor()

    @themed(
        name="leakage",
        alt_text=f"""
            Two panels against layer depth, sharing a vertical R² scale. Left: how well redness reads from the other 63 directions once the anchor coordinate is removed, for each condition. Every line sits in a narrow band near the top, at the caret marking the R² = {_floor:.2f} floor. Right: the same target read from the anchor coordinate alone, where the conditions separate widely and the control sits near zero.
        """,
        caption=rf"""
            How strongly redness is encoded at op1, read two ways on one R² scale. **Left:** held-out R² for a ridge probe predicting redness from the residual stream at op1 with the $\hat v_{{\text{{red}}}}$ coordinate deleted. The caret marks R² = {_floor:.2f}, the score the same probe gets from the raw RGB values, which no condition can fall far below while the cube is intact. **Right:** the same target read from the anchor coordinate alone, as squared correlation, which starts at zero and has room to reach 1. Per-layer seed means. The sample is the 216 op1 colors, with a 5-fold split so no color is scored by a probe that saw it during fitting. Neither panel has a pass/fail threshold.
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
        for ax, data, title in (
            (axes[0], _leak, "from the other 63 directions"),
            (axes[1], _axis, "from the anchor alone"),
        ):
            for cond in CONDS:
                ax.plot(
                    _depths, data[cond], color=_ink[cond], marker="o", ms=3.5,
                    lw=1.8 if cond in ex.FACTORIAL else 1.3,
                    ls="-" if cond in ex.FACTORIAL or cond == "lam0" else (0, (4, 3)), label=LABELS[cond],
                )  # fmt: skip
            ax.set_title(title, fontsize=10)
            ax.set_xticks(_depths, ["emb", *map(str, range(1, len(_depths)))])
            ax.set_xlabel("slice")
            ax.set_ylim(-0.05, 1.0)
        # The floor is a property of the target, not of any run: a caret on the
        # axis says where it is without drawing a rule across the conditions.
        axes[0].plot(
            0, _floor, marker=5, ms=6, color=light_dark("#555", "#aaa"),
            transform=axes[0].get_yaxis_transform(), clip_on=False, zorder=5,
        )  # fmt: skip
        axes[0].set_ylabel("R² for redness")
        axes[0].legend(fontsize=7, frameon=False, ncols=2, loc="lower left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], cells):
    _axis = {c: float(np.mean([cells[f"{c}-s{s}"]["axis_r2"] for s in ex.SEEDS])) for c in CONDS}
    _floor = ex.redness_rgb_floor()
    # Per-layer seed means, for the depth trend: averaging over layers hides it,
    # since the conditions cross the floor in opposite directions.
    _by_depth = {c: np.mean([cells[f"{c}-s{s}"]["leak_r2"] for s in ex.SEEDS], axis=0) for c in CONDS}
    mo.md(rf"""
    **The axis became much more readable; everywhere else sits at the RGB floor.** On the anchor coordinate alone, redness goes from R² = {_axis["lam0"]:.2f} in the control to {_axis["op1-bare"]:.2f} and {_axis["op1-anti"]:.2f} in the op1 conditions. That is double what the bare span pull achieved ({_axis["span-bare"]:.2f}), and it tracks the margin ordering closely. So the conditions that score well on selectivity are also the ones that put the most redness on the axis.

    Read from the other 63 directions, every condition at every depth stays within {max(abs(v - _floor).max() for v in _by_depth.values()):.2f} of R² = {_floor:.2f}, which is what the same probe gets from the raw RGB values. Call that the RGB floor: redness is a function of color and the task needs the color cube, so a linear readout recovers about this much wherever the anchor put it. Redness found outside the anchor is the cube being read, not a second copy of the concept. The control is flat at {_by_depth["lam0"].min():.2f}–{_by_depth["lam0"].max():.2f} at every depth, just under the floor, because its color code is marginally lossier than the raw channels.

    The other conditions drift away from the floor with depth, in both directions. The op1 conditions start *below* it at the token embedding ({_by_depth["op1-anti"][0]:.2f}) and finish *above* it at the last layer ({_by_depth["op1-anti"][-1]:.2f}). Above the floor is reachable because the residual stream encodes color nonlinearly and redness is quadratic in RGB, so a linear probe does better there than on the raw channels. `span-bare` runs the other way, from {_by_depth["span-bare"][0]:.2f} down to {_by_depth["span-bare"][-1]:.2f}. Falling below the floor means color itself was lost, the same cube compression its spread reports.

    This measurement deletes 1 direction of 64, so it has almost no room to move, and where it sits is set by what the task requires rather than by where the anchor put anything. It is a check that the color cube survived, beside task accuracy, rather than evidence about the anchor.
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
    ### The timing arm

    `span-anti-late` differs from `span-anti` in one way: the epoch at which the repulsive term finishes annealing down to its hold ratio, moved from {ex.ANTI_ANNEAL_END:g} to {ex.ANTI_ANNEAL_END_LATE:g}. That one change improves on the M1 schedule on every statistic that H2 and H4 score, which we did not expect:

    - margin {m_op1("span-anti-late").mean():.3f} against {m_op1("span-anti").mean():.3f}. The difference of {_dm:.2f} is about {_dm / ex.NOISE_SEED_MEAN:.0f} noise units, and about the size of the whole anti-subspace main effect;
    - retention {_ret("span-anti-late"):.2f} against {_ret("span-anti"):.2f}, moving it from the sliding group to the holding one;
    - grading R² {grading("span-anti-late")[2]:.2f} against {grading("span-anti")[2]:.2f};
    - containment {alpha_bar("span-anti-late").mean():.3f} against {alpha_bar("span-anti").mean():.3f}, the lowest of any anchored condition. Least surprising of the four: more repulsion for longer lowers the very quantity it penalizes.

    Redness read from the other 63 directions is the one measurement that does not follow: this arm is the highest of any condition ({_leak("span-anti-late"):.2f} against {_leak("span-anti"):.2f}). That is a weak observation, and it is why the claim above is scoped to the scored statistics. A value above the floor says the residual stream is a more probe-friendly encoding of color, which says nothing about the anchor either way.

    So stretching one anneal by 2× was worth more than adding the term in the first place. It looks worth a dedicated sweep over the anneal endpoint. Note that we only measured one alternative timing, and it is confounded with total repulsive strength, since holding near peak for longer also delivers more repulsion overall.

    ### The ceiling arm

    More weight turns out not to be better. `span-anti-hi` runs the full recipe at $\lambda_\text{{a}} = {ex.CEILING_LAMBDA:g}$, ten times the scoring rung, with the repulsive term scaled in proportion.

    We still have not found the task ceiling. Holdout accuracy is within {abs(acc("span-anti-hi").mean() - CONTROL_ACC):.4f} of control, so even at ten times the scoring weight the task shows no cost. Whatever bounds the anchor weight for D2.2, it is not the task loss at this scale, and finding the real ceiling would need a dedicated sweep well above $\lambda_\text{{a}} = 1$.

    But we have passed the selectivity ceiling. At ten times the weight the margin is {m_op1("span-anti-hi").mean():.3f}, level with the {m_op1("span-anti").mean():.3f} of `span-anti` at a tenth of the weight. The other three measurements are all worse: $\bar\alpha$ back up to {alpha_bar("span-anti-hi").mean():.3f}, retention down to {_ret("span-anti-hi"):.2f}, and the cube compressed even in the token embedding. So a heavier pull gives more alignment but no more selectivity: at this weight, moving everything becomes the cheapest way to satisfy the term.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    This section is post hoc: we planned it after seeing the results, and it scores no hypothesis.

    ### Why doesn't the repulsive term push *red* off the anchor too?

    The anti-subspace term does not distinguish between concepts: it penalizes $\cos^2(h, \hat v_{\text{red}})$ at every position of every line, labeled or not. That makes the timing arm look puzzling. Holding that penalty near its peak for another forty epochs ought to push the anchored concept off the anchor along with everything else. Instead, that arm has the highest margin of any `span` condition.

    Splitting the margin into its two halves shows the term does act on *red*; the pull toward the anchor is simply stronger.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], arrays):
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
    The margin split into the two quantities it differences. The first column is the mean alignment of the ten reddest colors — the ones carrying {ex.LABEL_W[_top].sum():.0%} of the label mass, so effectively "where the anchored concept sits". The second is the mean over all 216, the quantity the repulsive term penalizes. Comparing <code>span-anti</code> with
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

    # For the balance subsection: the first post-warmup epoch at which the
    # anti/anchor weight ratio falls below a level, from the run's own schedules.
    _sched_e = np.linspace(0, ex.SCHEDULER.epochs, 4001)

    def _ratio_below(level: float, anneal_end: float = ex.ANTI_ANNEAL_END) -> float:
        r = ex.anti_subspace_weight(_sched_e, anneal_end=anneal_end) / np.maximum(ex.anchor_weight(_sched_e), 1e-9)
        past = _sched_e > ex.SCHEDULER.warmup_epochs
        return float(_sched_e[past][np.argmax(r[past] < level)])

    mo.md(rf"""
    Delaying the $\lambda_{{\bar{{\text{{s}}}}}}$ anneal costs the reddest colors {_dred:.3f} of alignment. It costs the cube as a whole {_dall:.3f}, which is {_dall / _dred:.0f} times as much. So the term is doing what its definition says, *red* included. The margin improves because the same indiscriminate push matters far more for the unlabeled bulk than for the labeled reds.

    Why is the push so much gentler on the reds? It comes down to how the two terms are normalized. Neither weight depends on the labels: $\lambda_\text{{a}}$ and $\lambda_{{\bar{{\text{{s}}}}}}$ are functions of the epoch alone, identical in every batch. What differs is the denominator *inside* each term. Both terms are means, but over different sets: the anchor averages over the positions it pulls, and the repulsion averages over every live position. Measured on the real sampler, a batch has {ex.LIVE_PER_BATCH:,.0f} live positions. The anchor fires in {ex.FIRING_RATE[ex.SPAN_FULL]:.0%} of batches, and when it does, it averages {ex.PULLED_PER_BATCH[ex.SPAN_FULL]:.1f} pulled positions. So a labeled state feels a pull scaled by $1/{ex.PULLED_PER_BATCH[ex.SPAN_FULL]:.1f}$ against a push scaled by $1/{ex.LIVE_PER_BATCH:,.0f}$, about {_span_ratio:,.0f}:1 before weights. Even at the opening $\lambda_{{\bar{{\text{{s}}}}}} = {ex.ANTI_PEAK_RATIO:g}\lambda_\text{{a}}$, and after the $2\cos$ factor from differentiating $\cos^2$, the pull at a pulled position is still a couple of hundred times stronger than the push. An *unlabeled* state has no pull at all. It feels only the push, and goes wherever the task will let it.

    Because the anchor is a mean rather than a sum, the per-position pull is also insensitive to *how many* labels a batch happens to carry: a batch with one labeled line pulls its positions as hard as a batch with three. The label rate only sets how often the term fires at all.
    """)
    return


@app.cell(hide_code=True)
def _(alpha_bar):
    # REVIEW: two narrowings in the discussion below.
    # (1) "That fills the gap in D2.1" now names its scope (three seeds, one
    #     synthetic testbed), since every margin number in this report comes
    #     from `v216` at d64-L4.
    # (2) Dropped "the margin-by-position profile suggests the model itself
    #     knows where it is": `m_by_position` is recorded by `eval_one` but no
    #     figure or table in this report shows it, so a reader cannot check the
    #     claim. Verify: if a per-position panel is added, this can come back.
    mo.md(r"""
    ## Discussion

    This experiment demonstrated a selective anchor in a transformer: `op1-anti` clears the margin gate with a graded response against the M1-derived shape and no measurable task cost. That fills the gap in D2.1 for this testbed, and it did so without either of the two obvious next steps — a heavier pull (the ceiling arm shows that makes matters worse) or the remaining M1 repulsive terms.

    Both candidate mechanisms from the ex-2.1.6 discussion turn out to be real, but their relative sizes are the reverse of what that discussion expected, with the tested hyperparameters. Narrowing the pull to the one position that carries the labeled color helps more than adding the M1 repulsive term, and the conditions that pull the whole span are the ones whose margin slides later in training.

    However, the span pull is not an arbitrary choice we can simply drop. Sequence-level labeling forces it, because a document label marks no position as the relevant one. Our op1-only pull is possible only because this synthetic language has a known position that carries the concept, which is what a real corpus does not give you.

    The repulsive term is worth keeping, even though it failed H4(a). It reduced the cube-wide drift but did not contain it, and $\bar\alpha$ climbed again once the anneal brought the weight down. But using a later anneal improved margin, containment, retention, and grading, by more than adding the term was worth in the first place. The M1 keyframes were inherited from a 5-dimensional autoencoder bottleneck and mapped onto our 100 epochs by fraction of training, so there was never a reason to expect them to be right for this architecture.

    Possible follow-ups:
    - A sweep over weight ratios and anneal scheduling
    - Try different pooling over the span to allow uneven pull without having to specify positions up-front

    One thing this experiment does *not* license. Redness stays as readable from the other 63 directions as it is in the control, unchanged from ex-2.1.6 in every condition. A selective anchor is not the same as an exclusive one, so nothing here bounds what suppressing the axis would do to the model's access to *red*. That is the question for the intervention experiments.
    """)
    return


if __name__ == "__main__":
    app.run()
