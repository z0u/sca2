import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.10: A label that doesn't point",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path
    from typing import cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo puts the notebook directory on sys.path, so the design constants come
    # from the experiment module rather than a copy of them in the prose.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import AxesGrid, figure_html, light_dark, smooth_step, smooth_step_area, themed

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

    def load_ex219() -> tuple[dict, dict[str, np.ndarray]] | None:
        """Ex-2.1.9's published results: the carried baselines, the τ calibration,
        and the reproduction targets.
        """
        store = project_store()
        arts = store.get_refs([ex.EX219_METRICS_REF, ex.EX219_ARRAYS_REF])
        m_art, a_art = arts[ex.EX219_METRICS_REF], arts[ex.EX219_ARRAYS_REF]
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
    # Ex 2.1.10: A label that doesn't point

    /// tip |
    <!-- tl;dr -->
    The mellowmax-pooled pull found the concept position on its own, but every
    label was keyed on op1, so only one position ever carried the evidence. Now
    a line is labeled when *either* operand draws the label, and the pull has to
    find the red operand line by line. It does: the pull in each label group
    lands on its own operand, and the selectivity delivered matches that of the
    slot oracle.
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
def _(a_op1, acc, contrast, grading_r2, m_line, pi_group_mean, retention):
    _P = ex.PRIMARY
    _ctrl = acc("lam0").mean()
    _lead = pi_group_mean(_P)[0, 0, 0]  # G1's op1 weight at the embedding; G2's op2 weight equals it
    _frac = (m_line(_P).mean() - m_line("lam0").mean()) / (m_line(ex.ORACLE_ARM).mean() - m_line("lam0").mean())
    _ret = retention(_P)
    assert _ret is not None
    mo.md(rf"""
    **H1 (task cost) — holds.** The largest `named_holdout` exact-match gap
    from the control across the five anchored conditions is
    {max(abs(acc(c).mean() - _ctrl) for c in [ex.REFERENCE_ARM, *ex.EITHER_NAMES, ex.ORACLE_ARM]):.4f},
    against a gate of {ex.TASK_GATE:g}.

    **H2 (the pull selects the labelled operand) — holds.** On the primary
    (`{_P}`): (a) at the embedding the leading role in each group is its own
    operand, at weight {_lead:.2f} (gate ≥ {ex.LEAD_GATE:g}; uniform is
    0.25), in all nine runs individually; (b) the deep-slice between-group
    op2 contrast is {contrast(_P).mean():+.2f} (gate ≥
    {ex.CONTRAST_GATE:g}).

    **H3 (the operating point survives) — holds.** On `{_P}`: containment
    $\bar\alpha$ at op1 = {a_op1(_P).mean():.3f} (gate ≤
    {ex.MEAN_ALIGN_GATE:g}); retention {_ret:.2f} (gate ≥
    {ex.RETENTION_GATE:g}, minimum across nine seeds); grading $r^2$ =
    {grading_r2(_P):.2f} against {grading_r2(ex.REFERENCE_ARM):.2f} for the
    reference arm, a drop of
    {grading_r2(ex.REFERENCE_ARM) - grading_r2(_P):.3f} (gate ≤
    {ex.GRADE_R2_DROP:g}).

    **H4 (selectivity without the pointer) — holds.** Control-subtracted
    m_line of the primary is {_frac:.0%} of the value for the slot oracle
    (gate ≥ {ex.ORACLE_FRAC_GATE:.0%}).

    The reference arm reproduced `pool-t100` from ex-2.1.9 through the new
    labelling code path, matching every carried statistic to three decimals
    (*Reproduction*, in the H2 section).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this report
    This report was preregistered: the method, hypotheses and decision
    thresholds were frozen before the experiment ran (commit `b36a590`).
    Results replaced the `TODO` placeholders in place; everything conceived
    after seeing the data is under *Exploratory analyses*, marked as post hoc.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Ex-2.1.9 took away the position oracle: the pooled (mellowmax) anchor term
    asks each labeled line to align *somewhere* in its span, and the pull put
    {LEAD} of its weight on op1 at the embedding without being told. But the
    labels still pointed. A label is a per-visit Bernoulli draw: each time a
    line is visited in training, it *draws* a label with a small probability.
    In ex-2.1.9 that probability depended only on the redness of the first
    operand, so op1 was the only position whose evidence ever justified a label,
    and a pull that settled on op1 *by position* could never be wrong. RoPE
    never writes position into the stream, but causal attention gives each role
    a different view of the line (e.g. op1 attends only to itself), so from the
    first block on, the roles are distinguishable states a pull could latch.

    Labels that don't point are a step toward M3, which will target natural
    text. There, labels arrive at the document level ("this review is
    negative", "this tweet is obscene"), since nobody marks which tokens carry
    the concept. A document-level label says the concept occurs, but not where.
    This experiment widens our labels by one step in that direction: either
    operand can draw the label, so a line can be labeled because of a red op2
    while its op1 is blue. The localization then has to vary per-line: on
    op1-triggered lines the red state is at position 0, on op2-triggered
    lines at position 2, and a pool that commits to one position for the whole
    run would pull blue states onto the axis on the lines it got wrong.

    Whether that failure would even show up in the scores is not obvious,
    because the span positions share token embeddings. Once red embeddings sit
    on the axis, a red op2 line has an aligned position for free: the
    labeller's slot-symmetry is satisfied by the lexicon before any per-line
    choice happens. It only really matters after attention: in ex-2.1.9, we
    found the pool's deep-slice choice is winner-take-all *per run*, committed
    by epoch 8. If that winner is global, the two label groups will show the
    same weight profile; if the pool localizes per line, a red op2 should carry
    the weight on the lines where it is the evidence (H2). The selectivity
    delivered to labeled lines is scored separately (H4), since the embedding
    give-away means the second can pass while the first fails.

    One scope limit: our mixing operation can't make the
    answer redder than both operands, so "the concept is only at a position
    the labeller never sees" doesn't exist in this language. Two candidate
    positions, correlated with the answer, is as blind as this testbed gets;
    enough to test per-line localization; a concept at a position the
    labeller gave no hint of is out of reach here.
    """.replace("{LEAD}", f"{ex.EX219_REFERENCE['pool-t100']['lead_emb']:.2f}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | Glossary
    - **line** — one equation, `c1 + c2 = answer`. Training samples lines
      from this grammar; the probe set enumerates all 5832 of them.
    - **span** — a line's four prompt positions, the ones the pooled pull
      chooses among.
    - **role** — what a span position holds: op1, `+`, op2, `=`. The grammar
      is fixed in this experiment, so role and position name the same thing here.
    - **slice** ($\ell$) — a depth at which the residual stream is read: the
      embedding, plus the stream after each of the four blocks.
    - **alignment** ($\alpha$) — $\cos(h, \hat v)$, a state's cosine to the
      anchor axis; $\bar\alpha$ is its mean over colors or lines, called
      *drift* when the mean runs over the unlabeled bulk.
    - **margin** ($m$) — label-weighted minus unweighted mean alignment: how
      much closer the labeled-ish material is to the axis than everything
      else.
    - **softmin temperature** ($\tau$) — how sharply the pooled pull
      concentrates on its best-aligned position: small τ approaches a hard
      min, large τ approaches the span mean.
    - **softmin weight** ($\pi$) — the share of the pooled pull a span
      position receives; sums to 1 over the span.
    - **G1**, **G2** — the label groups: probe lines weighted by P(only op1) and
      P(only op2) respectively.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    Everything follows ex-2.1.9 except the labelling rule: the word-level
    `v216` corpus from ex-2.1.3 (216 colors, one token each, d64-L4), the same
    seeds, the mellowmax-pooled anchor term, the anchor schedule, and the
    anti-subspace schedule fixed at ex-2.1.8's operating point
    (`end90-hold30`) in every anchored cell.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### The labelling rule

    Labels are as they have been since ex-2.1.6 (binary, drawn per visit,
    with probability graded in redness), except that now both operands draw.
    Each operand of a line draws independently at redness⁸ × {SLOTRATE}, and
    the line is labeled when either draws. The per-slot rate is half of the
    op1 labeller's {REDRATE}, which keeps the expected labeled-line rate where
    it was, up to the tiny both-draw overlap; the label-budget check below
    computes both. A labeled line's pulled positions are its prompt
    span, as before: the label never says which operand drew.

    The answer position stays outside the span: the model has to *produce*
    the mix there, so pulling that state toward red would oppose the task on
    every labeled line whose answer isn't red, entangling the anchor with the
    output rather than with the concept. The span will have to widen
    eventually: in M3 the span is the whole document, answer-analogues
    included. That widening is its own variable, so it stays out of this
    experiment; a full-line arm (answer included, a position where the
    concept is diluted) is queued in todo-science as the intermediate step.
    Lines whose operands are *both* reddish exist too; their weight maps are
    on the exploratory list.

    The `slot-oracle` condition is the exception that measures what the label
    withholds: it pulls only the operand(s) that actually drew, no pooling
    needed. And the `op1-labels` condition keeps the old labeller entirely
    (op1-keyed at {REDRATE}), reproducing ex-2.1.9's `pool-t100`
    through the new code path.

    One bookkeeping consequence: label draws are consumed per line whatever
    the anchor weight, so conditions sharing a labeller see identical batches
    and masks at a given seed. Conditions with *different* labellers consume
    different draws, so comparisons across the labelling rules carry
    corpus-draw noise on top of seed noise. The control runs the either-slot
    labeller, keeping it batch-identical with the primary.
    """.replace("{SLOTRATE}", f"{ex.PER_SLOT_RATE:g}").replace("{REDRATE}", f"{ex.RED_RATE:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    # Joined with the block's own indent so every row dedents evenly under mo.md.
    _rows = "\n    ".join(
        f"| `{c['name']}`{' **(primary)**' if c['name'] == ex.PRIMARY else ''} | {c['lam']:g} | "
        f"{'∞' if np.isinf(c['tau']) else f'{c["tau"]:g}'} | "
        f"{c['labels']} | {'the operand that drew' if c['pull'] == 'slot' else 'prompt span, pooled'} | "
        f"{c['seeds']} |"
        for c in ex.CONDITIONS
    )
    mo.md(
        r"""
    ### Conditions

    Six conditions, most run with three seeds each. The primary condition runs
    with nine seeds: the deep-slice mellowmax in ex-2.1.9 was one-hot per *run*,
    and we can measure the failure rate better with more seeds. {NRUNS} runs in
    all. Every anchored cell runs the anti-subspace term at the operating point;
    only the labelling rule and τ vary.

    <!-- tau renders as T in this font in table headers, so using mathmode -->

    | condition | $\lambda_a$ | $τ$ | labels | pulled positions | seeds |
    |---|---|---|---|---|---|
    {ROWS}

    `either-t100` is the primary: the either-slot labeller at τ = 0.1. That
    temperature was the only rung ex-2.1.9 found graded in every seed (latch
    rate 0/3). The τ = 0.5 and 2.5 arms are softer,[^tau-geo] reported but not
    gated.

    [^tau-geo]: τ is a temperature (the weight ratio between two positions is
        $e^{\Delta x/\tau}$), so the rungs step geometrically, ×5. At τ = 2.5
        the ratios on realistic alignment spreads are ~1.2: a pull with only a
        tiny bias toward its best-aligned position, nearly the span mean that
        ex-2.1.9's τ = ∞ arm pinned, which is why no span-mean arm runs here.

    `op1-labels` re-runs ex-2.1.9's `pool-t100` verbatim: the reproduction
    check for the new labelling code path, and the baseline the widened
    labels are judged against. `slot-oracle` shows what knowing the triggering slot is worth,
    measured through identical code: a reference rather than a strict
    ceiling, for the same reason ex-2.1.9's `op1-oracle` was.
    """.replace("{ROWS}", _rows).replace("{NRUNS}", str(sum(c["seeds"] for c in ex.CONDITIONS)))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration

    A few checks before anything runs, computed from the design constants and
    the published arrays from ex-2.1.9: the label budget, the placement of the
    soft-τ rungs, and the reproduction targets for the reference arm.
    """)
    return


@app.cell(hide_code=True)
def _():
    from sca.colorcube import redness as _redness
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS as _N_LEVELS

    # The ordered closed-pair universe the corpus draws from: both orders of
    # every distinct closed pair, plus the self-pairs (always closed, always
    # trained), matching the probe set's 216 × 27 lines.
    _levels = nc.GRIDS[ex.GRID]
    _distinct = nc.closed_pairs(_levels)
    _colors = list(nc.grid_palette(_levels).values())
    _ordered = np.asarray([*_distinct, *[(b, a) for a, b in _distinct], *[(c, c) for c in _colors]], dtype=float)
    _r1 = _redness(_ordered[:, 0] / (_N_LEVELS - 1))
    _r2 = _redness(_ordered[:, 1] / (_N_LEVELS - 1))

    _p_old = float(np.mean(_r1**8 * ex.RED_RATE))
    _p_new = float(np.mean(ex.line_label_p(_r1, _r2)))
    _g1 = ex.p_slot(_r1) * (1 - ex.p_slot(_r2))
    _g2 = (1 - ex.p_slot(_r1)) * ex.p_slot(_r2)
    _both = ex.p_slot(_r1) * ex.p_slot(_r2)
    _tot = float(np.mean(_g1 + _g2 + _both))
    _shares = [float(np.mean(x)) / _tot for x in (_g1, _g2, _both)]

    mo.md(rf"""
    **The label budget.** Over the closed-pair set the corpus draws from,
    the op1 labeller labels {_p_old:.4f} of lines per visit and the either-slot
    labeller at the halved rate labels {_p_new:.4f}, a
    {abs(_p_new / _p_old - 1) * 100:.1f}% difference. The halving does keep
    the budget, so a margin change is attributable to the labelling rule
    rather than to more or less total pull. Of the either-slot label mass,
    {_shares[0]:.0%} is op1-only, {_shares[1]:.0%} op2-only and
    {_shares[2]:.1%} both-draw. The near-even split between the one-slot
    groups is what H2's group weighting and H4's latch arithmetic lean on.
    """)
    return


@app.cell(hide_code=True)
def _():
    _e219 = load_ex219()
    mo.stop(
        _e219 is None,
        mo.md("_Ex-2.1.9's published results aren't reachable from here; the calibration cells need them._"),
    )
    assert _e219 is not None
    _metrics219, arrays219 = _e219
    return (arrays219,)


@app.cell(hide_code=True)
def _(arrays219):
    def _ex219_alpha(cond: str) -> np.ndarray:
        return np.mean([arrays219[f"{cond}-s{s}/alpha"] for s in (0, 1, 2)], axis=0)

    _ALPHA219 = {c: _ex219_alpha(c) for c in ["lam0", "pool-t100"]}

    def _leading_weight(alpha: np.ndarray, tau: float) -> np.ndarray:
        """Label-weighted mean of the leading softmin weight, per residual slice."""
        w = ex.softmin_weights(1.0 - alpha[:, :, : ex.SPAN], tau)
        return np.einsum("lc,c->l", w.max(axis=-1), ex.LABEL_W)

    _taus = np.geomspace(0.003, 8.0, 70)
    _lead = {c: np.array([_leading_weight(a, t) for t in _taus]) for c, a in _ALPHA219.items()}

    @themed(
        name="tau-ladder",
        alt_text="""
            Two line charts of leading softmin weight against temperature tau
            on a log scale from 0.003 to 8. Left panel: the un-anchored
            control's alignment profile from ex-2.1.9; right panel: its
            pool-t100 condition. Each panel has five curves, one per residual
            slice, descending from near 1 at the left edge to 0.25 at the
            right edge. Three vertical dashed lines mark tau = 0.1, 0.5 and
            2.5. At tau = 2.5 every curve has flattened to within a few
            hundredths of 0.25, the uniform value marked by a faint
            horizontal line.
        """,
        caption=r"""
            **Where the soft-τ rungs land.** The check the ladder was chosen
            by (the rungs are not derivable from first principles): the share
            of softmin weight held by the leading span position, as a
            function of τ, computed on the stored alignments from ex-2.1.9
            under the un-anchored control (**left**) and under `pool-t100`
            (**right**). Dashed verticals mark the rungs chosen here, and the
            curves say what each will do. τ = 0.1 keeps a clear leader on the
            trained profile; τ = 2.5 sits within a few hundredths of uniform
            (0.25) on both, so it *is* the tiny-bias arm and a τ = ∞ arm
            would add nothing; τ = 0.5 splits the gap. Had a rung landed on
            the flat shoulder next to another, we would have moved it.
        """,
    )
    def _plot():
        _grey = light_dark("#666", "#999")
        _slice_colors = [
            light_dark("#7b3fa0", "#b98ce0"),
            light_dark("#1f6fb4", "#5fa8dd"),
            light_dark("#2a9d8f", "#5fc9bd"),
            light_dark("#b4531f", "#dd8f5f"),
            light_dark("#a02040", "#d06080"),
        ]
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained", sharey=True)
        for ax, (cond, title) in zip(
            axs, [("lam0", "control profile"), ("pool-t100", "pooled-pull profile")], strict=True
        ):
            for li in range(5):
                ax.plot(_taus, _lead[cond][:, li], color=_slice_colors[li], lw=1.4, label=f"slice {li}")
            for t in ex.TAUS:
                ax.axvline(t, color=_grey, ls=(0, (4, 3)), lw=0.8, alpha=0.7)
            ax.axhline(1 / ex.SPAN, color=_grey, lw=0.5, alpha=0.45, zorder=-1)
            ax.set_xscale("log")
            ax.set_xlabel("τ", fontsize="x-small")
            ax.set_title(title, fontsize="small", color=_grey)
        axs[0].set_ylabel("leading weight", fontsize="x-small")
        axs[0].set_ylim(0.2, 1.02)
        axs[1].legend(fontsize="x-small", loc="upper right", frameon=False)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays219):
    def _seed_mean(stat, cond: str) -> float:
        """Mean of the per-run statistic — the convention every result is scored under."""
        return float(np.mean([stat(arrays219[f"{cond}-s{s}/alpha"]) for s in (0, 1, 2)]))

    _m9 = _seed_mean(lambda a: ex.pooled_margin(a, ex.LABEL_W), "pool-t100")
    _a9 = _seed_mean(ex.alpha_span, "pool-t100")
    _ao9 = _seed_mean(lambda a: float(a[:, :, 0].mean()), "pool-t100")
    mo.md(rf"""
    The ex-2.1.9 arrays give the reproduction targets for the `op1-labels`
    arm, as means of per-run statistics: m_span = {_m9:.2f},
    ᾱ_span = {_a9:.2f}, ᾱ at op1 = {_ao9:.2f}. A miss there means the new labelling code path moved the numbers, and
    gets settled before any either-slot condition is read.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Measurements

    Task evals, the readability profile, grading, and the trajectory
    instrument are from ex-2.1.9. Two measurements change and two are new,
    all on the same probe set: all 5832 ordered closed pairs, so every color
    appears both as op1 and as op2. (Terms like *line*, *role*, *slice* and
    *alignment* are pinned down in the Glossary above; alignment throughout
    is $\cos(h, \hat v)$, a state's cosine to the anchor axis.)

    **Per-line alignment maps.** The eval step now stores alignment per probe
    *line* (slices, lines, positions) rather than averaged into per-color rows.
    The per-color maps that earlier statistics used are recovered by averaging
    over partners, so the line-keyed statistics below are computed from those.
    Per-line softmin weights are stored the same way.[^data-volume]

    [^data-volume]: The volume stays small: 5 slices × 5832 lines × ~6 positions
        in float32 is ≈ 0.7 MB per array, two arrays per run, so ≈ 35 MB across
        all {NRUNS} runs. That is an order of magnitude over ex-2.1.9's
        per-color arrays but nowhere near needing different processing, and
        the per-role trajectory adds well under a megabyte per run.

    **m_line**, the selectivity this experiment scores: m_span, generalized
    from colors to lines, to measure "how much closer does a line sit to the
    axis for being the kind of line the labeller flags".[^m_span] The max over
    roles is applied to every condition and the control alike, so its
    maximum-of-noise inflation cancels in the control-subtracted comparisons, as
    before. The trajectory records m_line, so retention reads off it. m_span and
    $m_\text{op1}$ are reported beside it for continuity.

    [^m_span]: To recap m_span (from ex-2.1.9): on the per-color maps, take the
        affinity-weighted mean alignment minus the plain mean, giving a margin
        per (slice, role) saying how much closer red-ish material is to the axis
        than average. Keep the best span role per slice, and average over
        slices. m_line is the same recipe on the per-line maps: weight *lines*
        instead of colors, using the labeller's own P(labeled) per line,
        normalized. m_span keyed on op1 alone, whereas m_line keys on both
        operands.

    **Per-group weight profiles.** The localization instrument. Probe lines
    are weighted into two groups by the labeller's own one-slot-only
    probabilities (G1 by P(op1 drew and op2 didn't), G2 by the reverse), so
    no redness threshold has to be invented, and both-red lines (where either
    answer is defensible) get almost no say. Within each group,
    $\bar\pi_G(\ell, t)$ is the group-weighted mean softmin weight per slice
    and role. Under per-line localization the two profiles should differ:
    G1 stepping at op1, G2 at op2. Under a global winner they are identical
    no matter what: if every line carries the same weight profile, the two
    group means, which differ only in how lines are weighted, both return
    that one profile.

    **Per-role drift in the trajectory.** Ex-2.1.9 couldn't say *when* the
    `+` embedding climbed to 0.30 alignment because the trajectory recorded
    op1 drift only; it now records $\bar\alpha(\ell, t)$ over all span roles
    at each measurement, which is the data that question needs. Report-side
    only; nothing is gated on it.
    """.replace("{NRUNS}", str(sum(c["seeds"] for c in ex.CONDITIONS)))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Hypotheses

    Gates and noise floors are from ex-2.1.6 through ex-2.1.9 wherever the
    statistic is unchanged; derivations for the new ones are in
    [experiment.py](./experiment.py), next to their constants.

    Throughout, "the primary" is `{PRIMARY}` (fixed in advance, marked in the
    *Conditions* table). The τ = 0.5 and 2.5 arms are reported beside it but
    not gated. If the primary turns out task-dirty, H2–H4 go unscored and the
    refuted H1 is the finding.

    **H1: Task cost.** Every condition is task-clean: `named_holdout` exact
    match within {TASK} (absolute) of the seed mean of the in-experiment
    control. No partial credit; nothing in this design adds task pressure,
    and no anchored condition on this testbed has ever shown any.

    **H2: The pull selects the labelled operand.** On the primary's per-group
    weight profiles:
    **(a)** at the embedding slice, each group's leading role is its own
    operand (op1 for G1, op2 for G2) with weight $\ge {LEAD}$ in both
    (uniform is 0.25);
    **(b)** across the four post-attention slices, the between-group contrast in
    op2 weight, $\mathrm{mean}_\ell\,[\bar\pi_{G2}(\ell, \text{op2}) -
    \bar\pi_{G1}(\ell, \text{op2})]$, is $\ge {CONTRAST}$.
    Partial: (a) holds and (b) lands in $[{CONTRASTP}, {CONTRAST})$.
    (a) is close to forced by the shared embeddings *if* the pull aligns red
    tokens at all, so on its own it mostly rules out the syntax latch. (b) is
    the real question: ex-2.1.9's deep-slice winner-take-all was per *run*,
    and (b) asks whether the winner can vary per line when the label demands it.
    Contrary outcomes:
    *global latch* — both groups lead at the same role (op1, or worse, `+`), the
    contrast sits near 0, and on op2-triggered lines the pull is dragging
    non-red states onto the axis, which H3(a) should catch as ᾱ contamination;
    *uniform* — the primary reaches {LEAD} in neither group at any slice
    (τ = 0.1 is the sharpest rung, so if it can't concentrate, the softer
    arms won't either), meaning the pool can't tell the positions apart once
    the label stops pointing.

    **H3: The operating point survives the wider labeller.** On the primary:
    **(a)** containment, $\bar\alpha \le {ALPHA}$ at op1 (also the contamination
    detector for H2's global-latch outcome);
    **(b)** retention, every run whose running maximum of m_line reaches
    {FLOOR} ending at $\ge {RET}\times$ that maximum (quoted as the minimum
    across seeds);
    **(c)** grading, the primary's seed-mean op1 response $r^2$ against
    `sim¹·⁵` no more than {DROP} below the `op1-labels` arm's (higher is
    fine).
    Partial: (a) and (b) hold, (c) misses.

    **H4: Selectivity holds without the pointer.** Control-subtracted
    m_line of the primary reaches $\ge {FRAC}$ of the control-subtracted
    m_line of `slot-oracle`. Partial: $[{FRACP}, {FRAC})$.
    The latch account predicts roughly the op1-triggered share of the label
    mass (~half) plus whatever the shared embeddings give away, so a pass has
    to clear that; for scale, ex-2.1.9's pooled arms recovered 91–98% of
    their oracle.

    Note that H4 can pass while H2(b) fails: the shared embeddings can deliver
    selectivity to labeled lines with no per-line localization at depth. That
    would suggest selectivity from the lexicon rather than from localization.
    """.replace("{TASK}", f"{ex.TASK_GATE:g}")
        .replace("{PRIMARY}", ex.PRIMARY)
        .replace("{LEAD}", f"{ex.LEAD_GATE:g}")
        .replace("{CONTRASTP}", f"{ex.CONTRAST_PARTIAL:g}")
        .replace("{CONTRAST}", f"{ex.CONTRAST_GATE:g}")
        .replace("{ALPHA}", f"{ex.MEAN_ALIGN_GATE:g}")
        .replace("{FLOOR}", f"{ex.RETENTION_FLOOR:g}")
        .replace("{RET}", f"{ex.RETENTION_GATE:g}")
        .replace("{DROP}", f"{ex.GRADE_R2_DROP:g}")
        .replace("{FRAC}", f"{ex.ORACLE_FRAC_GATE:g}")
        .replace("{FRACP}", f"{ex.ORACLE_FRAC_PARTIAL:g}")
    )
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
    N_SEEDS = {c["name"]: int(c["seeds"]) for c in ex.CONDITIONS}

    def over_seeds(cond: str, fn) -> np.ndarray:
        """One value per run of a condition, in seed order (nine on the primary)."""
        return np.array([fn(cells[f"{cond}-s{s}"]) for s in range(N_SEEDS[cond])])

    def acc(cond: str, set_name: str = "named_holdout", key: str = "accuracy") -> np.ndarray:
        return over_seeds(cond, lambda c: c["sets"][set_name][key])

    def alpha_map(cond: str) -> np.ndarray:
        """Seed-mean per-color alignment map for a condition: (slices, colors, roles)."""
        return np.mean([arrays[f"{cond}-s{s}/alpha"] for s in range(N_SEEDS[cond])], axis=0)

    def m_line(cond: str) -> np.ndarray:
        """Per-run m_line — the selectivity H4 scores."""
        return over_seeds(cond, lambda c: c["m_line"])

    def m_span(cond: str) -> np.ndarray:
        return over_seeds(cond, lambda c: c["m_span"])

    def a_span(cond: str) -> np.ndarray:
        return over_seeds(cond, lambda c: c["alpha_span"])

    def m_op1(cond: str) -> np.ndarray:
        return over_seeds(cond, lambda c: c["m_op1"])

    def a_op1(cond: str) -> np.ndarray:
        """Per-run ᾱ at op1 — the containment statistic H3(a) reads."""
        return over_seeds(cond, lambda c: c["alpha_mean_op1"])

    def traj(cond: str, seed: int, key: str) -> np.ndarray:
        return np.asarray(cells[f"{cond}-s{seed}"]["traj"][key], dtype=float)

    return (
        ANCHORED,
        CONDS,
        N_SEEDS,
        a_op1,
        a_span,
        acc,
        alpha_map,
        arrays,
        cells,
        m_line,
        m_op1,
        m_span,
        metrics,
        traj,
    )


@app.cell(hide_code=True)
def _(N_SEEDS: dict, alpha_map, arrays, metrics, traj):
    # The probe lines' operand rednesses, recomputed from the palette by the same
    # loop the prep step runs — what the group weighting and m_line key on.
    from sca.colorcube import redness as _redness
    from sca.data import named_colors as _nc
    from sca.data.colors import N_LEVELS as _NL, mix as _mix

    _levels = _nc.GRIDS[ex.GRID]
    _palette = _nc.grid_palette(_levels)
    _colors = list(_palette.values())
    _on_grid = set(_levels)
    _pairs = [(c, b) for c in _colors for b in _colors if all(v in _on_grid for v in _mix(c, b))]
    R1 = _redness(np.asarray([p[0] for p in _pairs], dtype=float) / (_NL - 1))
    R2 = _redness(np.asarray([p[1] for p in _pairs], dtype=float) / (_NL - 1))
    G1W, G2W = ex.group_weights(R1, R2)

    def pi_group(cond: str, seed: int) -> np.ndarray:
        """One run's per-group weight profiles π̄_G(l, t): (group, slices, span roles)."""
        w = arrays[f"{cond}-s{seed}/w_line"]
        return np.stack([np.einsum("n,lnt->lt", G1W, w), np.einsum("n,lnt->lt", G2W, w)])

    def pi_group_mean(cond: str) -> np.ndarray:
        return np.mean([pi_group(cond, s) for s in range(N_SEEDS[cond])], axis=0)

    def contrast(cond: str) -> np.ndarray:
        """Per-run H2(b) statistic: the between-group op2 contrast over the four
        post-attention slices, mean over slices of π̄_G2(l, op2) − π̄_G1(l, op2).
        """
        pgs = [pi_group(cond, s) for s in range(N_SEEDS[cond])]
        return np.array([float((pg[1, 1:, 2] - pg[0, 1:, 2]).mean()) for pg in pgs])

    READ: np.ndarray = np.mean(
        [np.asarray(metrics["readability"][f"lam0-s{s}"], dtype=float) for s in range(N_SEEDS["lam0"])], axis=0
    )
    # The control's readability profile R²(l, t) for op1's redness, mean over the
    # three un-anchored seeds — the dashed reference behind the G1 profiles.

    def grading_r2(cond: str) -> float:
        """r² of the seed-mean per-color op1 response against sim¹·⁵ — H3(c)'s statistic."""
        return ex.r2_sim(alpha_map(cond)[:, :, 0].mean(axis=0))

    def retention(cond: str) -> float | None:
        """Lowest final-over-peak m_line across the runs that reached the floor.

        The minimum, per `ex.RETENTION_STAT`; None when no run reached the
        floor (the control), leaving the criterion vacuous rather than failed.
        """
        r = [
            t[-1] / t.max() for s in range(N_SEEDS[cond]) if (t := traj(cond, s, "m_line")).max() >= ex.RETENTION_FLOOR
        ]
        return min(r) if r else None

    return READ, R1, R2, contrast, grading_r2, pi_group, pi_group_mean, retention


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost (H1)
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], acc):
    _CTRL = float(acc("lam0").mean())

    def _cell(cond: str, set_name: str, key: str, fmt: str = "{:.3f}") -> str:
        v = acc(cond, set_name, key)
        return f"{fmt.format(v.mean())} <span class='range'>±{(v.max() - v.min()) / 2:.3f}</span>"

    _rows = "".join(
        f"<tr><th><code>{c}</code></th>"
        f"<td class='num'>{_cell(c, 'named_seen', 'accuracy')}</td>"
        f"<td class='num'>{_cell(c, 'named_holdout', 'accuracy')}</td>"
        f"<td class='num'>{_cell(c, 'named_holdout', 'nll')}</td>"
        f"<td class='num'>{_cell(c, 'open', 'guess_dist')}</td>"
        f"<td class='num'>{acc(c).mean() - _CTRL:+.4f}</td>"
        f"<td>{'clean' if abs(acc(c).mean() - _CTRL) <= ex.TASK_GATE else 'not clean'}</td></tr>"
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
    Behavior by condition. Each value is the seed mean, with half the seed
    range beside it. EM is exact match on the answer token, NLL its surprise
    in nats, and <code>open</code> dist the RGB distance from the guessed
    color to the true mix on pairs with no named answer. Δ holdout EM is the
    signed gap to the in-experiment control, which the H1 gate scores at
    {ex.TASK_GATE:g}.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], acc):
    _ctrl = float(acc("lam0").mean())
    _worst = max((c for c in CONDS if c != "lam0"), key=lambda c: abs(acc(c).mean() - _ctrl))
    mo.md(rf"""
    **H1 holds.** The largest gap from the control is
    {abs(acc(_worst).mean() - _ctrl):.4f}, at `{_worst}`, against a gate of
    {ex.TASK_GATE:g}, so H2–H4 are all scored. Labels that no longer say
    which operand drew change nothing the task loss can see, as with every
    anchored condition on this testbed so far.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where the weight went, by label group (H2)

    The `op1-labels` arm re-runs `pool-t100` from ex-2.1.9 verbatim (same
    labels, same seeds, same draws) through the widened sampler, so any
    drift from the new labelling code path is visible before we read an
    either-slot condition.
    """)
    return


@app.cell(hide_code=True)
def _(a_op1, a_span, m_op1, m_span):
    _stats = {"m_span": m_span, "alpha_span": a_span, "m_op1": m_op1, "alpha_op1": a_op1}
    _names = {"m_span": "m<sub>span</sub>", "alpha_span": "ᾱ<sub>span</sub>", "m_op1": "m<sub>op1</sub>",
              "alpha_op1": "ᾱ (op1)"}  # fmt: skip

    def _row(cond: str, ref_key: str) -> str:
        ref = ex.EX219_REFERENCE[ref_key]
        tds = ""
        for k, fn in _stats.items():
            v = fn(cond)
            tds += (
                f"<td class='num'>{v.mean():.3f} <span class='range'>±{v.std(ddof=1):.3f}</span></td>"
                f"<td class='num'>{ref[k]:.3f}</td><td class='num'>{v.mean() - ref[k]:+.3f}</td>"
            )
        return f"<tr><th><code>{cond}</code></th>{tds}</tr>"

    _head = "".join(
        f"<th class='num'>{_names[k]}</th><th class='num'>ex-2.1.9</th><th class='num'>Δ</th>" for k in _stats
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr><th>condition</th>{_head}</tr></thead>
      <tbody>{_row("op1-labels", "pool-t100")}{_row("lam0", "lam0")}</tbody>
    </table></div>
    """
    _caption = """
    The reproduction check. Each statistic is this experiment's seed mean (±
    seed standard deviation) beside the same statistic recomputed from
    ex-2.1.9's published arrays under the same per-run convention, and the
    difference. <code>op1-labels</code> re-runs <code>pool-t100</code>
    through the widened labelling code; <code>lam0</code> runs the either-slot
    labeller, so it is a new draw of the control rather than a re-run.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(pi_group_mean):
    _lead_emb = pi_group_mean("op1-labels")[0, 0, 0]
    mo.md(rf"""
    The reproduction is as close as the rounding in the store allows: every
    carried statistic lands within ±0.001 of its target, and the embedding
    lead weight ({_lead_emb:.2f}) matches the
    {ex.EX219_REFERENCE["pool-t100"]["lead_emb"]:g} of ex-2.1.9. The
    op1-keyed draw consumes the rng as the old sampler did, so this is the
    same training trajectory replayed, and the either-slot conditions can be
    read against `op1-labels` as a like-for-like baseline.

    The gaps in the control are the size of the seed spread, a reminder that
    it consumes *different* draws (two per line from the either-slot
    labeller), so that comparison carries corpus-draw noise, as the method
    warned.

    Both groups weight the same 5832 probe lines by the one-slot-only
    probabilities of the labeller itself, G1 by P(only op1 drew) and G2 by
    P(only op2 drew). The profiles below are the group-weighted mean softmin
    weights, per slice and role. Under a global winner the two columns of
    each pair would be identical; under per-line localization they mirror
    each other.
    """)
    return


@app.cell(hide_code=True)
def _(N_SEEDS: dict, READ: np.ndarray, contrast, pi_group, pi_group_mean):
    _x = np.arange(ex.SPAN)
    _INK = {
        "either-t100": light_dark("#1f6fb4", "#5fa8dd"),
        "either-t500": light_dark("#7fb3d8", "#4f7ea6"),
        "either-t2500": light_dark("#9ab5c8", "#41586b"),
    }
    _ALT = {
        "either-t100": """
            Ten small panels in two columns labeled G1, op1 drew, and G2, op2
            drew: five residual slices per column with the embedding at the
            bottom, each spanning the roles op1, plus, op2 and equals. In the
            G1 column every slice shows a tall block at op1 — 0.77 at the
            embedding, near 0.9 at depth — with a small step at equals in the
            embedding row only. The G2 column is the mirror image: the block
            sits at op2 at every slice, 0.77 at the embedding rising to 0.95
            at depth. Nine per-seed hairlines hug the mean everywhere. A
            dashed readability line in the G1 column is high at op1 on every
            slice; at the embedding it falls to near zero at the other three
            roles, while at depth it dips at plus and rises again at equals.
        """,
        "either-t500": """
            The same ten-panel layout for tau 0.5. Both columns are nearly
            flat around 0.25 at the embedding with a slight op1 tilt in G1
            and op2 tilt in G2; at depth the G2 column's op2 weight grows to
            about 0.44 while the G1 column stays diffuse. Hairlines stay
            close to the mean.
        """,
        "either-t2500": """
            The same ten-panel layout for tau 2.5. Every panel in both
            columns is flat at about 0.25 with tilts of a few hundredths, the
            span mean in all but name.
        """,
    }

    def _profile_fig(cond: str, title: str) -> str:
        _pg = pi_group_mean(cond)  # (group, slices, roles)
        _seeds = [pi_group(cond, s) for s in range(N_SEEDS[cond])]
        _con = contrast(cond)

        @themed(
            name=f"group-profiles-{cond}",
            alt_text=_ALT[cond],
            caption=rf"""
                **{title}** — τ = {dict(zip(ex.EITHER_NAMES, ex.TAUS, strict=True))[cond]:g};
                deep-slice op2 contrast {_con.mean():+.2f}
                <span class='range'>±{_con.std(ddof=1):.2f}</span>.
            """,
        )
        def _plot() -> plt.Figure:
            from matplotlib.layout_engine import ConstrainedLayoutEngine

            ink = _INK[cond]
            _grey = light_dark("#666", "#999")
            _read_ink = light_dark("#555", "#aaa")
            _hair = light_dark("#00000055", "#ffffff55")
            _gate = light_dark("#555", "#bbb")
            fig, axes = plt.subplots(5, 2, figsize=(3.1, 3.9), sharex=True, sharey=True)
            _engine = fig.get_layout_engine()
            assert isinstance(_engine, ConstrainedLayoutEngine)
            _engine.set(hspace=0, h_pad=0.01, wspace=0.05)
            for col, gname in enumerate(("G1: op1 drew", "G2: op2 drew")):
                for row in range(5):
                    sl = 4 - row  # last slice on top, the embedding at the bottom
                    ax = axes[row][col]
                    smooth_step_area(ax, _x, _pg[col, sl], ramp=0.5, color=ink, alpha=light_dark(0.22, 0.28))
                    smooth_step(ax, _x, _pg[col, sl], ramp=0.5, color=ink, lw=1.5)
                    for sp in _seeds:
                        smooth_step(ax, _x, sp[col, sl], ramp=0.5, color=_hair, lw=0.5)
                    if col == 0:
                        # The stored readability profile reads op1's redness, so it
                        # belongs behind G1 alone; G2's analogue was not measured.
                        smooth_step(ax, _x, READ[sl], ramp=0.5, color=_read_ink, lw=1.0, ls=(0, (4, 2)))
                    ax.set(ylim=(-0.08, 1.08), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0.0, 0.5, 1.0])
                    ax.spines[:].set_visible(False)
                    ax.grid(axis="y", which="major", c="#888", alpha=0.2)
                    ax.tick_params(axis="x", length=0, labelbottom=False)
                    ax.tick_params(axis="y", left=True, right=True, direction="in", labelleft=False, labelright=False)
                    if col == 0:
                        ax.set_ylabel("emb" if sl == 0 else f"{sl}", fontsize=7.5)
                axes[0][col].set_title(gname, fontsize=8.5, color=ink)
                # H2(a) reads exactly here: the group's own-operand weight at the embedding.
                own = 0 if col == 0 else 2
                # REVIEW: the H2(a) carets were drawn under get_yaxis_transform(),
                # which reads x in axes fraction — so G1's landed on the left spine
                # and G2's outside the panel, contradicting the caption ("at each
                # group's own operand"). Now plain data coords. Verify: G2's caret
                # should sit under the op2 tick, beside the 0.77 annotation.
                axes[4][col].plot(own, ex.LEAD_GATE, marker=9, ms=3, color=_gate, clip_on=False, zorder=6)
                axes[4][col].annotate(f"{_pg[col, 0, own]:.2f}", (own, min(_pg[col, 0, own] + 0.14, 0.92)),
                                      ha="center", fontsize=7, color=light_dark("#444", "#bbb"))  # fmt: skip
                axes[4][col].set_xticks(_x, ex.ROLES)
                axes[4][col].tick_params(axis="x", labelbottom=True, labelsize=7.5)
            axes[4][1].tick_params(axis="y", labelright=True, labelsize=6.5, pad=2)
            return fig

        return _plot()

    _titles = {"either-t100": "The primary", "either-t500": "Softer", "either-t2500": "Near the span mean"}
    _body = "".join(_profile_fig(c, _titles[c]) for c in ex.EITHER_NAMES)
    _caption = rf"""
    **Per-group weight profiles under the either-slot labeller.** Within each
    sub-figure: columns are the label groups (G1 weights probe lines by
    P(only op1 drew), G2 by P(only op2 drew)); rows are residual slices with
    the embedding at the bottom; the shaded smooth-step is the seed-mean
    softmin weight $\bar\pi_G(\ell, t)$ over the span roles, with per-seed
    hairlines (nine on the primary). The dashed line behind the G1 columns
    is the un-anchored control's readability profile $R^2(\ell, t)$ for
    op1's redness. Carets on the embedding row mark the H2(a) gate
    ({ex.LEAD_GATE:g}; uniform is 0.25) at each group's own operand, with
    the seed-mean weight beside it. Each sub-caption quotes the H2(b)
    deep-slice op2 contrast (gate ≥ {ex.CONTRAST_GATE:g}, scored on the
    primary only).
    """
    mo.Html(figure_html(_body, caption=_caption, aria_label="Per-group softmin weight profiles, one sub-figure per either-slot condition"))  # fmt: skip
    return


@app.cell(hide_code=True)
def _(contrast, pi_group_mean):
    _P = ex.PRIMARY
    _pg = pi_group_mean(_P)
    _con = {c: contrast(c) for c in [*ex.EITHER_NAMES, "op1-labels"]}
    mo.md(rf"""
    **H2 holds, and the profiles mirror each other.** (a) At the embedding
    the leading role in each group is its own operand, at weight
    {_pg[0, 0, 0]:.2f} (gate ≥ {ex.LEAD_GATE:g}), in all nine runs
    individually. The two groups take equal values there by construction,
    since at the embedding the state of a token cannot depend on its
    line.[^emb-sym]

    (b) The between-group op2 contrast over the post-attention slices
    is {_con[_P].mean():+.2f}
    <span class='range'>±{_con[_P].std(ddof=1):.2f}</span> against the
    {ex.CONTRAST_GATE:g} gate. The op2 weight of G2 at the last slice is
    {_pg[1, 4, 2]:.2f}, so on the lines where op2 was the evidence the pull
    puts nearly everything there. The winner-take-all behavior of ex-2.1.9
    is still in force, and the winner now varies line by line: the columns
    are far from identical, and they do not collapse to uniform at the τ of
    the primary.

    [^emb-sym]: The probe set contains both orders of every pair, and the
        span positions share token embeddings, so the G1 and G2 profiles are
        exact mirrors at the embedding slice. What the gate tests there is
        concentration: that the pool has moved away from uniform, and that
        the concentration sits on operands rather than syntax. Depth is
        where the groups can differ, which is why H2(b) reads only the
        post-attention slices.

    The unscored arms behave as the τ ladder predicted. At τ = 0.5 the
    embedding is nearly flat (lead {pi_group_mean("either-t500")[0, 0, 0]:.2f})
    and the deep contrast reaches only
    {_con["either-t500"].mean():+.2f}; at τ = 2.5 the profile is the span
    mean in all but name (contrast {_con["either-t2500"].mean():+.2f}). A
    pull that cannot concentrate cannot localize, however good the labels
    are.

    The op1-keyed reference arm has a *measured* contrast of
    {_con["op1-labels"].mean():+.2f}, worth holding onto for M3. The shared
    lexicon aligns red op2 tokens for free, so a softmin read at τ = 0.1
    finds them even though the pull never targeted op2. But that free
    contrast decays with depth: the op2 weight of G2 falls from
    {pi_group_mean("op1-labels")[1, 1, 2]:.2f} at slice 1 to
    {pi_group_mean("op1-labels")[1, 4, 2]:.2f} at slice 4, where the
    primary holds ({_pg[1, 1, 2]:.2f} → {_pg[1, 4, 2]:.2f}). Training on
    the wider labels is what keeps the localization alive where the states
    are contextual.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The operating point under the wider labeller (H3)

    The endpoint numbers say where each run finished; the trajectories say
    how it got there — in particular whether a pull that has to localize per
    line buys its margin later, or holds it less firmly once the repulsion
    anneals away, than the op1-keyed reference.
    """)
    return


@app.cell(hide_code=True)
def _(ANCHORED, N_SEEDS: dict, retention, traj):
    _KEYS = ("alpha_op1", "m_line")
    # Seed means once, in the cell: @themed calls the plot function twice.
    _MEAN = {
        c: {k: np.mean([traj(c, s, k) for s in range(N_SEEDS[c])], axis=0) for k in _KEYS} for c in [*ANCHORED, "lam0"]
    }
    _EPOCHS = np.mean([traj(ex.PRIMARY, s, "epoch") for s in range(N_SEEDS[ex.PRIMARY])], axis=0)
    _RET = {c: retention(c) for c in ANCHORED}
    # The schedules on a finer grid than the recorded trajectory. All anchored
    # conditions share them at the operating point, so one panel serves five.
    _SE = np.linspace(0.0, float(ex.SCHEDULER.epochs), 601)
    _ANCHOR = ex.anchor_weight(_SE)
    _ANTI = ex.anti_subspace_weight(_SE)
    _GRID = [["op1-labels", "either-t100", "either-t500"], ["either-t2500", "slot-oracle", "sched"]]
    _LEFT = ("op1-labels", "either-t2500")

    @themed(
        name="trajectories",
        alt_text="""
            Five trajectory panels and one schedule panel in a two-by-three
            grid sharing the epoch axis from 0 to 100. Each trajectory panel
            plots mean alignment at op1 as a solid line staying below 0.1,
            and the line margin as a dashed line climbing to about 0.6 by
            epoch 30 and on to roughly 0.7 by the end, over a flat grey
            control pair near zero. The dashed line dips slightly after epoch
            90 in every panel, and every panel's retention annotation reads
            0.96 or higher. The five trajectory panels are near copies of one
            another. The schedule panel, bottom right, shows the anchor
            weight ramping to a plateau and annealing after epoch 90, and the
            repulsion weight starting high, crossing below the anchor around
            epoch 60 and settling at about a third of it.
        """,
        caption=rf"""
            **Training dynamics by condition.** Seed means of the two cosines
            on the anchor axis, on one shared scale: solid is $\bar\alpha$ at
            op1 (contained at the left caret, {ex.MEAN_ALIGN_GATE:g}); dashed
            is m_line (its retention floor of {ex.RETENTION_FLOOR:g} at the
            right caret). The grey pair is the un-anchored control, and the
            number in each panel is that condition's retention (final over
            peak m_line, minimum across seeds; gate ≥ {ex.RETENTION_GATE:g},
            scored on the primary). The trajectory instrument reads one line
            per color, so its level sits above the endpoint statistic's,
            which averages all 27 partners; retention compares within the
            curve, where the instrument is constant. Bottom right: the weight
            schedule every anchored condition trained under — anchor
            $\lambda_\mathrm{{a}}$ in red, repulsion
            $\lambda_{{\bar{{\mathrm{{s}}}}}}$ in blue, log scale — shared
            because the anti-subspace term is fixed at the operating point.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axd = plt.subplot_mosaic(_GRID, figsize=(7.5, 4.2), sharex=True, layout="constrained")
        _grey = light_dark("#999", "#777")
        _gate = light_dark("#555", "#bbb")
        _ink = light_dark("#1f6fb4", "#5fa8dd")
        _term = {"anchor": light_dark("#c33", "#e66"), "anti": light_dark("#36c", "#7af")}
        _dash = (0, (6, 1))
        _top = max(np.max(_MEAN[c]["m_line"]) for c in ANCHORED)
        for cond in ANCHORED:
            ax = axd[cond]
            ax.plot(0, ex.MEAN_ALIGN_GATE, marker=9, ms=3, color=_gate, clip_on=False, zorder=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
            ax.plot(1, ex.RETENTION_FLOOR, marker=8, ms=3, color=_gate, clip_on=False, zorder=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
            ax.plot(_EPOCHS, _MEAN["lam0"]["alpha_op1"], color=_grey, lw=1.1, zorder=3)
            ax.plot(_EPOCHS, _MEAN["lam0"]["m_line"], color=_grey, lw=1.1, ls=_dash, zorder=3)
            ax.plot(_EPOCHS, _MEAN[cond]["m_line"], color=_ink, lw=1.4, ls=_dash, zorder=4)
            ax.plot(_EPOCHS, _MEAN[cond]["alpha_op1"], color=_ink, lw=1.7, zorder=5)
            _r = _RET[cond]
            assert _r is not None
            ax.annotate(f"{_r:.2f}", (0.97, 0.6), xycoords="axes fraction", ha="right", fontsize=8,
                        color=light_dark("#444", "#bbb"))  # fmt: skip
            ax.set_title("either-t100 (primary)" if cond == ex.PRIMARY else cond, fontsize=9.5, color=_ink,
                         fontweight="bold" if cond == ex.PRIMARY else "normal")  # fmt: skip
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_ylim(-0.08, _top * 1.08)
            ax.tick_params(labelbottom=cond in _GRID[1], labelleft=cond in _LEFT, labelsize=8)
            ax.set_ylabel("cosine", fontsize="x-small") if cond in _LEFT else None
            ax.set_xlabel("epoch", fontsize="x-small") if cond in _GRID[1] else None
        _ax = axd["sched"]
        _ax.plot(_SE, _ANCHOR, color=_term["anchor"], lw=1.4)
        _ax.plot(_SE, _ANTI, color=_term["anti"], lw=1.4)
        _ax.set_yscale("log")
        _ax.set_xlim(0, ex.SCHEDULER.epochs)
        _ax.set_ylim(2e-4, 4)
        _ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
        _ax.minorticks_off()
        _ax.tick_params(labelsize=7.5)
        _ax.set_xlabel("epoch", fontsize="x-small")
        _ax.set_title("weights (all anchored)", fontsize=9.5, color=_grey)
        for _xx, _yy, _txt, _k in ((64, 0.13, r"$\lambda_\mathrm{a}$", "anchor"), (16, 0.5, r"$\lambda_{\bar{\mathrm{s}}}$", "anti")):  # fmt: skip
            _ax.annotate(_txt, (_xx, _yy), fontsize=9, color=_term[_k], va="bottom")
        fig.legend(
            [plt.Line2D([], [], color=_grey, lw=1.7), plt.Line2D([], [], color=_grey, lw=1.5, ls=_dash)],
            [r"$\bar\alpha$ at op1", "m_line"],
            fontsize=8.5, frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            labelcolor=_grey, handlelength=2.4,
        )  # fmt: skip
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(ANCHORED, a_op1, retention):
    _P = ex.PRIMARY
    _ret = retention(_P)
    assert _ret is not None
    _worst = min((r for c in ANCHORED if (r := retention(c)) is not None))
    mo.md(rf"""
    **H3(a) and (b) hold.** On the primary, $\bar\alpha$ at op1 ends at
    {a_op1(_P).mean():.3f} <span class='range'>±{a_op1(_P).std(ddof=1):.3f}</span>
    against the {ex.MEAN_ALIGN_GATE:g} gate. That is *below* the
    {a_op1(ex.REFERENCE_ARM).mean():.3f} of the op1-keyed reference, so the
    contamination a global latch would leave (non-red op1 states dragged
    onto the axis on op2-triggered lines) is absent, which agrees with the
    H2 profiles. With half the pull budget now landing on op2, op1 simply
    hosts less of the drift.

    Retention is {_ret:.2f} against {ex.RETENTION_GATE:g}, taken as the
    minimum over nine seeds, a stricter
    read than for the three-seed conditions; the worst retention anywhere in
    the sweep is {_worst:.2f}. The five anchored panels are near copies, so
    localizing per line neither delays the margin nor loosens its hold.
    """)
    return


@app.cell(hide_code=True)
def _(CONDS: list[str], alpha_map, grading_r2, m_line):
    _RESP = {c: alpha_map(c)[:, :, 0].mean(axis=0) for c in CONDS}

    def _fitted_target(y: np.ndarray) -> np.ndarray:
        """`sim¹·⁵` rescaled by least squares onto the response — the shape r² scores against."""
        b = np.polyfit(ex.SIM_TARGET, y, 1)
        return np.polyval(b, ex.SIM_TARGET)

    def _level_mean(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The conditional mean of *y* at each distinct redness level (29 of them)."""
        levels = np.unique(np.round(ex.REDNESS, 6))
        means = np.array([y[np.round(ex.REDNESS, 6) == lv].mean() for lv in levels])
        return levels, means

    _CTRL = _level_mean(_RESP["lam0"])
    _GRID = [CONDS[:3], CONDS[3:]]

    @themed(
        name="grading",
        alt_text="""
            Six panels in a two-by-three grid sharing both axes: per-color
            alignment at the first operand, from about -0.2 to 1.2, against
            the redness of that color from 0 to 1. Panels are the control,
            op1-labels, the primary either-t100, the two softer arms and the
            slot oracle. The control panel is flat near zero. The op1-labels,
            either-t100 and slot-oracle panels each scatter 216 marks drawn
            in the color they stand for, forming a flat cluster at low
            redness and a rising tail of oranges and reds that tracks the
            dashed reference curve closely. In the either-t500 and
            either-t2500 panels the low-redness cluster scatters much more
            widely, from about -0.2 to 0.5, and the conditional-mean line
            wanders around a slightly raised level before rising, hugging
            the reference less tightly.
        """,
        caption=rf"""
            **Grading: alignment at op1, per color.** $\alpha_c$, the mean
            over slices of $\cos(h, \hat v_{{\text{{red}}}})$ at op1, seed
            mean, against the redness of the color. One mark per color, drawn
            in that color; the thin dark line is the mean response at each of
            the 29 distinct redness levels, the flat grey band the same line
            for the control, and the dashed line `sim¹·⁵` rescaled onto the
            response by least squares — the shape the $r^2$ statistic scores
            against. H3(c) gates the primary's $r^2$ at no more than
            {ex.GRADE_R2_DROP:g} below `op1-labels`'s, whose own $r^2$
            ({grading_r2(ex.REFERENCE_ARM):.2f}) reproduces ex-2.1.9's
            `pool-t100` ({ex.EX219_REFERENCE["pool-t100"]["r2"]:g}).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), sharey=True, sharex=True, layout="constrained")
        axes = cast(AxesGrid, axes)
        for row, conds in zip(axes, _GRID, strict=True):
            for ax, cond in zip(row, conds, strict=True):
                a = _RESP[cond]
                ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
                ax.plot(*_CTRL, color=light_dark("#999", "#777"), lw=3, alpha=0.5, zorder=1,
                        solid_capstyle="round")  # fmt: skip
                ax.scatter(ex.REDNESS, a, c=ex.GRID_RGB, s=18, lw=0.4, zorder=3,
                           edgecolors=light_dark("#00000033", "#ffffff55"))  # fmt: skip
                if cond != "lam0":
                    ax.plot(*_level_mean(_fitted_target(a)), color=light_dark("#888", "#888"),
                            lw=1.0, ls=(0, (4, 2)), zorder=2)  # fmt: skip
                    ax.plot(*_level_mean(a), color=light_dark("#222", "#eee"), lw=1.2, zorder=4)
                ax.set_title("either-t100 (primary)" if cond == ex.PRIMARY else ("control" if cond == "lam0" else cond),
                             fontsize=9.5)  # fmt: skip
                if cond != "lam0":
                    ax.annotate(
                        f"r² = {grading_r2(cond):.2f}\nm_line = {m_line(cond).mean():.2f}",
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
def _(grading_r2):
    _P = ex.PRIMARY
    mo.md(rf"""
    **H3(c) holds.** The primary grades at $r^2$ = {grading_r2(_P):.2f}
    against {grading_r2(ex.REFERENCE_ARM):.2f} for the reference arm, a drop
    of {grading_r2(ex.REFERENCE_ARM) - grading_r2(_P):.3f} against the
    {ex.GRADE_R2_DROP:g} tolerated. So widening the labels cost the response
    shape essentially nothing at the τ of the primary.

    The softer arms fare worse:
    {grading_r2("either-t500"):.2f} at τ = 0.5 and
    {grading_r2("either-t2500"):.2f} at τ = 2.5. The grading panels show
    where it goes; the low-redness response turns noisy and drifts off zero,
    since a diffuse pull drags every span position of every labeled line,
    red op1 or not. The slot oracle grades best of all
    ({grading_r2(ex.ORACLE_ARM):.2f}), as the analogue in ex-2.1.9 did.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selectivity against the slot oracle (H4)

    m_line is the selectivity delivered to the lines the labeller flags, and
    the slot oracle — trained on the same label draws, told which operand
    drew — is the reference for what the withheld information is worth.
    """)
    return


@app.cell(hide_code=True)
def _(ANCHORED, m_line):
    _CTRL = float(m_line("lam0").mean())
    # Control-subtracted per-run values, computed once — @themed draws twice.
    _M = {c: m_line(c) - _CTRL for c in ANCHORED}
    _ORACLE = float(_M[ex.ORACLE_ARM].mean())

    @themed(
        name="h4-dots",
        alt_text="""
            A dot plot with five condition rows against control-subtracted
            line margin from 0 to about 0.5. From top: op1-labels at 0.36,
            either-t100 with nine tightly clustered dots at 0.40, either-t500
            and either-t2500 both near 0.46, and slot-oracle at 0.39. Small
            dots are individual runs and a vertical tick marks each mean; the
            seed spread within any row is a few thousandths. A dashed gate
            line at 0.29 and a dotted partial line at 0.19 sit well left of
            every row.
        """,
        caption=rf"""
            **Control-subtracted m_line by condition.** Small dots are runs
            (nine on the primary), the tick their mean; the reference arms
            are greyed. Dashed caret: the H4 gate,
            {ex.ORACLE_FRAC_GATE:.0%} of the slot oracle's control-subtracted
            mean ({_ORACLE:.3f}); dotted, the {ex.ORACLE_FRAC_PARTIAL:.0%}
            partial. The latch account — a pull stuck on op1 serving only the
            op1-triggered half of the label mass — predicts landing near the
            partial line.
        """,
    )
    def _plot() -> plt.Figure:
        # Reads top-to-bottom: the op1-keyed reference, the sweep, the oracle.
        _rows = [ex.ORACLE_ARM, "either-t2500", "either-t500", ex.PRIMARY, ex.REFERENCE_ARM]
        _grey = light_dark("#999", "#777")
        _gate = light_dark("#555", "#bbb")
        _ink = {
            ex.REFERENCE_ARM: _grey,
            ex.ORACLE_ARM: _grey,
            "either-t100": light_dark("#1f6fb4", "#5fa8dd"),
            "either-t500": light_dark("#7fb3d8", "#4f7ea6"),
            "either-t2500": light_dark("#9ab5c8", "#41586b"),
        }
        fig, ax = plt.subplots(figsize=(7.2, 2.2), layout="constrained")
        for yi, cond in enumerate(_rows):
            v = _M[cond]
            ax.plot(v, [yi] * len(v), "o", ms=3.5, color=_ink[cond], alpha=0.55, mew=0)
            ax.plot(v.mean(), yi, "|", ms=18, color=_ink[cond])
        ax.set_title(r"m_line − control", fontsize="small", color=_grey)
        ax.set_yticks(range(len(_rows)), ["slot-oracle", "either-t2500", "either-t500", "either-t100 (primary)",
                                          "op1-labels"], fontsize=8)  # fmt: skip
        ax.set_xlim(0, 0.52)
        ax.tick_params(axis="x", labelsize=8)
        ax.axvline(_ORACLE * ex.ORACLE_FRAC_GATE, color=_gate, lw=0.9, ls=(0, (4, 3)))
        ax.axvline(_ORACLE * ex.ORACLE_FRAC_PARTIAL, color=_gate, lw=0.9, ls=(0, (1, 2)))
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(ANCHORED, a_op1, a_span, grading_r2, m_line, m_op1, m_span):
    _CTRL = float(m_line("lam0").mean())
    _ORACLE = float(m_line(ex.ORACLE_ARM).mean()) - _CTRL

    def _row(c: str) -> str:
        frac = f"{(m_line(c).mean() - _CTRL) / _ORACLE:.0%}" if c != ex.ORACLE_ARM else "—"
        bold = "<strong>{}</strong>".format if c == ex.PRIMARY else "{}".format
        return (
            f"<tr><th>{bold(f'<code>{c}</code>')}</th>"
            f"<td class='num'>{m_line(c).mean():.3f} <span class='range'>±{m_line(c).std(ddof=1):.3f}</span></td>"
            f"<td class='num'>{m_line(c).mean() - _CTRL:.3f}</td>"
            f"<td class='num'>{frac}</td>"
            f"<td class='num'>{m_span(c).mean():.3f}</td>"
            f"<td class='num'>{m_op1(c).mean():.3f}</td>"
            f"<td class='num'>{a_span(c).mean():.3f}</td>"
            f"<td class='num'>{a_op1(c).mean():.3f}</td>"
            f"<td class='num'>{grading_r2(c):.2f}</td></tr>"
        )

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">m<sub>line</sub> ↑</th>
        <th class="num">− control</th><th class="num">oracle frac</th>
        <th class="num">m<sub>span</sub></th><th class="num">m<sub>op1</sub></th>
        <th class="num">ᾱ<sub>span</sub> ↓</th><th class="num">ᾱ (op1) ↓</th>
        <th class="num">r²</th>
      </tr></thead>
      <tbody>{"".join(_row(c) for c in ANCHORED)}</tbody>
    </table></div>
    """
    _caption = f"""
    The anchored conditions, seed means (± seed standard deviation on the
    scored statistic). Only the primary (<strong>bold</strong>) is gated;
    "oracle frac" is control-subtracted m<sub>line</sub> as a fraction of the
    slot oracle's {_ORACLE:.3f}. m<sub>span</sub> and m<sub>op1</sub> are
    carried for continuity with ex-2.1.9 (both key their line weights on op1
    alone), ᾱ<sub>span</sub> and ᾱ (op1) are the drift statistics, and r²
    repeats the grading track scored under H3.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(a_span, grading_r2, m_line):
    _P = ex.PRIMARY
    _CTRL = float(m_line("lam0").mean())
    _frac = (m_line(_P).mean() - _CTRL) / (m_line(ex.ORACLE_ARM).mean() - _CTRL)
    _ref_frac = (m_line(ex.REFERENCE_ARM).mean() - _CTRL) / (m_line(ex.ORACLE_ARM).mean() - _CTRL)
    mo.md(rf"""
    **H4 holds.** The primary delivers {_frac:.0%} of the
    control-subtracted m_line of the oracle, against the
    {ex.ORACLE_FRAC_GATE:.0%} gate. That is level with the oracle, within
    seed noise (spreads of {m_line(_P).std(ddof=1):.3f} and
    {m_line(ex.ORACLE_ARM).std(ddof=1):.3f} on conditions sharing the same
    label draws). The latch account predicted roughly the op1-triggered half
    plus the lexicon give-away, and it is ruled out twice over: by this
    fraction, and by the H2 profiles.

    The prereg allowed for landing at or above the oracle in principle,
    since the oracle is bound to the operand that drew even where the
    concept has migrated at depth. At this margin the defensible reading is
    parity: knowing which slot triggered is worth nothing here that the pool
    cannot recover on its own.

    The op1-keyed arm reaches only {_ref_frac:.0%} of the oracle on the
    same statistic, because its pull never serves the op2-triggered half of
    the label mass. Widening the labels is what closed the gap.

    The soft-τ arms *exceed* everything on m_line
    ({m_line("either-t500").mean():.2f} at τ = 0.5) while failing every
    other instrument: contrast near zero (H2), grading down by ~0.2
    (H3(c)), and span drift ᾱ_span at {a_span("either-t500").mean():.2f}
    against {a_span(_P).mean():.2f} for the primary. A diffuse pull aligns
    the whole span of labeled-ish lines, and a margin that takes the best
    role per slice rewards that spread. The number is real, but what it
    measures there is a smeared, poorly graded alignment rather than a
    localized one. The exploratory section returns to this trade.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Nothing here is preregistered; everything in this section is post hoc.
    The four questions below are the candidates the preregistration queued,
    in its order.
    """)
    return


@app.cell(hide_code=True)
def _(N_SEEDS: dict, traj):
    _P = ex.PRIMARY
    _AR = np.mean([traj(_P, s, "alpha_roles") for s in range(N_SEEDS[_P])], axis=0)  # (K, slices, roles)
    _EP = np.mean([traj(_P, s, "epoch") for s in range(N_SEEDS[_P])], axis=0)
    _SE = np.linspace(0.0, float(ex.SCHEDULER.epochs), 601)
    _ANTI = ex.anti_subspace_weight(_SE)

    @themed(
        name="role-drift",
        alt_text="""
            Two panels sharing an epoch axis from 0 to 100 and an alignment
            axis from about -0.1 to 0.45, with the repulsion schedule shaded
            faintly behind each. Left panel, the embedding slice: the equals
            and plus curves rise to about 0.13 by epoch 15, hold flat to
            epoch 35, then climb to 0.40 and 0.28 respectively by epoch 90;
            op1 and op2 dip to about -0.12 near epoch 8 and return to within
            a few hundredths of zero. Right panel, the last
            slice: op1 and op2 rise gently to 0.15 and 0.12 while plus and
            equals stay near zero.
        """,
        caption=r"""
            **Unweighted drift by role, over training (primary, seed mean).**
            $\bar\alpha(\ell, t)$ over all 216 colors at the embedding slice
            (**left**) and the last slice (**right**), one curve per span
            role; the shaded band is the repulsion weight
            $\lambda_{\bar{\mathrm{s}}}$ on its own scale, for timing.
            The trajectory instrument reads one line per color, as in the
            dynamics figure.
        """,
    )
    def _plot() -> plt.Figure:
        from mini.vis import AxesRow

        _role_ink = [
            light_dark("#a02040", "#d06080"),
            light_dark("#2a9d8f", "#5fc9bd"),
            light_dark("#b4531f", "#dd8f5f"),
            light_dark("#7b3fa0", "#b98ce0"),
        ]
        _grey = light_dark("#666", "#999")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, sl, title in zip(axes, (0, 4), ("embedding slice", "last slice"), strict=True):
            ax.fill_between(_SE, _ANTI / _ANTI.max() * 0.45, color=_grey, alpha=0.10, lw=0, zorder=0)
            for t in range(ex.SPAN):
                ax.plot(_EP, _AR[:, sl, t], color=_role_ink[t], lw=1.4, label=ex.ROLES[t])
            ax.axhline(0, color=_grey, lw=0.5, alpha=0.4, zorder=1)
            ax.set_title(title, fontsize="small", color=_grey)
            ax.set_xlabel("epoch", fontsize="x-small")
            ax.tick_params(labelsize=8)
        axes[0].set_ylabel(r"$\bar\alpha$ (all colors)", fontsize="x-small")
        axes[1].legend(fontsize="x-small", frameon=False, ncols=2, loc="upper left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(N_SEEDS: dict, R1: np.ndarray, R2: np.ndarray, arrays, contrast, grading_r2, traj):
    _P = ex.PRIMARY
    _AR = np.mean([traj(_P, s, "alpha_roles") for s in range(N_SEEDS[_P])], axis=0)
    _EP = np.mean([traj(_P, s, "epoch") for s in range(N_SEEDS[_P])], axis=0)
    _at = lambda e: _AR[int(np.searchsorted(_EP, e))]  # noqa: E731

    # Per-line weight concentration at the last slice, and the both-red profile.
    _line_p = np.asarray(ex.line_label_p(R1, R2))
    _LW = _line_p / _line_p.sum()
    _pb = np.asarray(ex.p_slot(R1)) * np.asarray(ex.p_slot(R2))
    _BOTH = _pb / _pb.sum()
    _lead_mass, _split_mass, _both4 = [], [], []
    for _s in range(N_SEEDS[_P]):
        _w4 = arrays[f"{_P}-s{_s}/w_line"][4]  # (lines, roles)
        _lead = _w4.max(axis=1)
        _lead_mass.append(float(np.einsum("n,n->", _LW, (_lead > 0.9).astype(float))))
        _ops = _w4[:, [0, 2]]
        _split_mass.append(float(np.einsum("n,n->", _BOTH, (np.abs(_ops[:, 0] - _ops[:, 1]) < 0.2).astype(float))))
        _both4.append(np.einsum("n,nt->t", _BOTH, _w4))
    _both_emb = np.mean(
        [np.einsum("n,nt->t", _BOTH, arrays[f"{_P}-s{_s}/w_line"][0]) for _s in range(N_SEEDS[_P])], axis=0
    )
    _both_deep = np.mean(_both4, axis=0)

    mo.md(rf"""
    **When the syntax embeddings drift (queued by ex-2.1.9).** Ex-2.1.9
    could see that its `+` embedding ended near 0.30 alignment; it could not
    see when. The per-role trajectory answers: late. At epoch 30 the two
    syntax embeddings sit at {_at(30)[0, 1]:.2f} (`+`) and
    {_at(30)[0, 3]:.2f} (`=`). The drift accrues through epochs 30–90 as
    the repulsion anneals toward its hold ratio, reaching
    {_at(90)[0, 1]:.2f} and {_at(90)[0, 3]:.2f}, and the end-of-training
    anneal of the anchor recovers none of it. So the syntax drift is bought
    against the *weakening* repulsion, and in this experiment `=` is the
    larger of the two; `=` sits outside the labelled operands but inside the
    pulled span.

    At depth the picture inverts. The operand roles carry the residual drift
    ({_at(100)[4, 0]:.2f} at op1, {_at(100)[4, 2]:.2f} at op2) and the
    syntax roles carry none, and that is where the containment gate of H3(a)
    reads and passes.

    **Is the deep choice still one-hot? (per-line weight entropy.)** Yes,
    per line. On the label-weighted probe mass, the leading role at the
    last slice holds over 0.9 of the weight of a line on
    {np.mean(_lead_mass):.0%} of the mass (seed range
    {min(_lead_mass):.0%}–{max(_lead_mass):.0%}). The un-labelled bulk
    stays near uniform, since a flat alignment profile gives the softmin
    nothing to concentrate on. The winner-take-all behavior of ex-2.1.9
    survives the wider labels in the form H2 requires: committed within a
    line, varying across lines.

    **The both-red lines.** Where both operands are red, either answer is
    defensible, and the pool declines to choose: at the embedding the
    both-draw-weighted profile splits {_both_emb[0]:.2f}/{_both_emb[2]:.2f}
    between the operands (forced by the shared lexicon), and at the last
    slice it still spreads: op1 {_both_deep[0]:.2f}, op2
    {_both_deep[2]:.2f}, `=` {_both_deep[3]:.2f}, with
    {np.mean(_split_mass):.0%} of the both-draw mass within 0.2 of an even
    operand split. These lines are {_pb.sum() / _line_p.sum():.1%}
    of the label mass, so nothing above turns on them. They preview the M3
    case where a document holds the concept in several places at once.

    **The soft-τ trade.** Across the ladder, localization and grading fall
    together while m_line rises: contrast
    {" → ".join(f"{contrast(c).mean():.2f}" for c in ex.EITHER_NAMES)} and
    $r^2$ {" → ".join(f"{grading_r2(c):.2f}" for c in ex.EITHER_NAMES)} as
    τ goes {" → ".join(f"{t:g}" for t in ex.TAUS)}, with m_line highest at
    the diffuse end (the H4 table). A nearly-mean pull does still buy
    line-keyed *margin*, in that the spans of labeled lines align more than
    the bulk. But it localizes barely at all once the label stops pointing,
    and what it aligns is poorly graded. For M3, a margin statistic alone
    cannot distinguish localized selectivity from smeared selectivity, so it
    needs the group contrast and the grading track beside it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    After the results land, and after a discussion round: what the outcome
    means for document-level labels in M3, in particular whether
    localization has to come from the loss side (pooling) or needs label-side
    help; and what it adds to the D2.1 close-out claim ("anchoring transfers
    under labels that don't say where the concept sits"). Not to be written
    in advance.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
