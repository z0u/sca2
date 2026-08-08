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

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo puts the notebook directory on sys.path, so the design constants come
    # from the experiment module rather than a copy of them in the prose.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import light_dark, themed

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
    Mellowmax-pooled pull found the concept position unaided, but every label
    was keyed on op1, so only one position ever carried the evidence. Now a line
    is labeled when *either* operand draws the label, so the pull has to find
    the red operand line by line.
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
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration: the method, hypotheses and decision thresholds
    below are frozen before the experiment runs (immaterial edits aside).
    Results will replace the `TODO` placeholders in place, and anything
    conceived after seeing the data goes under *Exploratory analyses*, marked
    as post hoc.
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
    labels still pointed. A label is a per-visit Bernoulli draw — each time a
    line is visited in training, it *draws* a label with a small probability —
    and in ex-2.1.9 that probability depended on the *first* operand's redness
    alone, so op1 was the only position whose evidence ever justified a label,
    and a pull that settled on op1 globally could never be wrong. Settling
    globally is possible, too: RoPE never writes position into the stream, but
    causal attention gives each role a different view of the line — op1
    attends only to itself — so from the first block on, the roles are
    distinguishable states a pull could latch.

    Why push toward labels that don't point? Because M3's targets are natural
    text, and there labels arrive at the document level — "this review is
    negative", "this tweet is obscene" — since nobody marks which tokens carry
    the concept. A document-level label says the concept occurs *somewhere* —
    one phrase, or smeared across the whole text — and nothing else.
    This experiment widens our labels by one step in that direction: either
    operand can draw the label, so a line can be labeled because of a red op2
    while its op1 is blue. The localization then has to vary per-line: on
    op1-triggered lines the red state is at position 0, on op2-triggered
    lines at position 2, and a pool that commits to one position for the whole
    run would pull blue states onto the axis on the lines it got wrong.

    Whether that failure would even show up in the scores is not obvious,
    because of a mechanism ex-2.1.9 already ran into: the span positions share token
    embeddings. At the embedding slice a color's state is its token embedding
    wherever the token appears (position enters only through attention, so it
    can't show up before the first block), so once red embeddings sit on the
    axis, a red op2 line has an aligned
    position *for free* — the labeller's slot-symmetry is satisfied by the
    lexicon before any per-line choice happens. The post-attention slices are
    where the question is real: there, ex-2.1.9 found the pool's deep-slice
    choice is winner-take-all *per run*, committed by epoch 8. If that
    winner is global, the two label groups will show the same weight profile;
    if the pool localizes per line, a red op2 should carry the weight on the
    lines where it is the evidence (H2). The selectivity delivered to labeled lines
    is scored separately (H4), since the embedding give-away means the second
    can pass while the first fails.

    One scope limit: our mixing operation can't make the
    answer redder than both operands, so "the concept sits only at a position
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
    - **line** — one probe equation, `c1 + c2 = answer`; the probe set has
      5832 of them.
    - **span** — a line's four prompt positions, the ones the pooled pull
      chooses among.
    - **role** — what a span position holds: op1, `+`, op2, `=`. The grammar
      is fixed, so role and position name the same thing here.
    - **slice** ($\ell$) — a depth at which the residual stream is read: the
      embedding, plus the stream after each of the four blocks.
    - **alignment** ($\alpha$) — $\cos(h, \hat v)$, a state's cosine to the
      anchor axis; $\bar\alpha$ is its mean over colors or lines, called
      *drift* when the mean runs over the unlabeled bulk.
    - **margin** ($m$) — label-weighted minus unweighted mean alignment: how
      much closer the labeled-ish material is to the axis than everything
      else.
    - **softmin weight** ($\pi$) — the share of the pooled pull a span
      position receives; sums to 1 over the span.
    - **G1 / G2** — the label groups: probe lines weighted by P(only op1
      drew) and P(only op2 drew) respectively.
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

    Labels are as they have been since ex-2.1.6 — binary, drawn per visit,
    with probability graded in redness — except that now both operands draw.
    Each operand of a line draws independently at redness⁸ × {SLOTRATE}, and
    the line is labeled when either draws. The per-slot rate is half of the
    op1 labeller's {REDRATE}, which keeps the expected labeled-line rate where
    it was, up to the tiny both-draw overlap; the label-budget check below
    computes both. A labeled line's pulled positions are its prompt
    span, as before — the label never says which operand drew.

    The answer position stays outside the span, as it has since the pull
    became pooled in ex-2.1.9: the model has to *produce* the mix there, so
    pulling that state toward red would oppose the task on every labeled line
    whose answer isn't red — it would entangle the anchor with the output
    rather than with the concept. (A full-line arm, answer included, would be
    a fair robustness probe — a position where the concept is diluted — and
    is queued in todo-science rather than run here.) Lines whose operands are
    *both* reddish exist too; their weight maps are on the exploratory list.

    The `slot-oracle` condition is the exception that measures what the label
    withholds: it pulls only the operand(s) that actually drew, no pooling
    needed. And the `op1-labels` condition keeps the old labeller entirely —
    op1-keyed at {REDRATE} — reproducing ex-2.1.9's `pool-t100`
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

    Six conditions. Three seeds each, except the primary, which runs nine:
    ex-2.1.9's deep-slice choice was one-hot per *run*, so the failure of
    interest is a rate across runs, and three seeds can't tell "latched once
    in three" from bad luck — τ = 0.1's clean 0/3 there may itself have been
    luck. Nine seeds on the one gated condition gives the latch rate a usable
    denominator and the per-seed hairlines in the H2 figure something to
    show; {NRUNS} runs in all. Every anchored cell runs the anti-subspace
    term at the operating point; only the labelling rule and τ vary.

    <!-- tau renders as T in this font in table headers, so using mathmode -->

    | condition | $\lambda_a$ | $τ$ | labels | pulled positions | seeds |
    |---|---|---|---|---|---|
    {ROWS}

    `either-t100` is the primary: the either-slot labeller at τ = 0.1,
    the only rung ex-2.1.9 found graded in every seed (latch rate 0/3) — a
    criterion from the previous experiment's published result, independent of
    everything gated here. The τ = 0.5 and 2.5 arms are the soft ladder,
    reported but not gated: τ is a temperature (the weight ratio between two
    positions is $e^{\Delta x/\tau}$), so the rungs step geometrically, ×5.
    At τ = 2.5 the ratios on realistic alignment spreads are ~1.2 — a pull
    with only a tiny bias toward its best-aligned position, nearly the span
    mean that ex-2.1.9's τ = ∞ arm pinned, which is why no span-mean arm runs
    here.

    `op1-labels` re-runs ex-2.1.9's `pool-t100` verbatim: the reproduction
    check for the new labelling code path, and the baseline the widened
    labels are judged against. `slot-oracle` shows what knowing the triggering slot is worth,
    measured through identical code — a reference rather than a strict
    ceiling, for the same reason ex-2.1.9's `op1-oracle` was.
    """.replace("{ROWS}", _rows).replace("{NRUNS}", str(sum(c["seeds"] for c in ex.CONDITIONS)))
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration

    A few checks before anything runs, computed from the design constants and
    ex-2.1.9's published arrays: the label budget, the placement of the
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
    labeller at the halved rate labels {_p_new:.4f} — a
    {abs(_p_new / _p_old - 1) * 100:.1f}% difference, so the halving does keep
    the budget, and a margin change is attributable to the labelling rule
    rather than to more or less total pull. Of the either-slot label mass,
    {_shares[0]:.0%} is op1-only, {_shares[1]:.0%} op2-only and
    {_shares[2]:.1%} both-draw — the near-even split between the one-slot
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
            **Where the soft-τ rungs land** — the check the ladder was chosen
            by, not derivable from first principles: the leading span
            position's share of the softmin weight, as a function of τ, if
            the weights were computed on ex-2.1.9's stored alignments — the
            un-anchored control's profile (**left**) or the `pool-t100`
            profile (**right**). Dashed verticals are this experiment's
            rungs, and the curves say what each will do: τ = 0.1 keeps a
            clear leader on the trained profile, τ = 2.5 is within a few
            hundredths of uniform (0.25) on both — so it *is* the tiny-bias
            arm, and a τ = ∞ arm would add nothing — and τ = 0.5 splits the
            gap. Had a rung landed on the flat shoulder next to another, we'd
            have moved it.
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

    **Per-line alignment maps.** The eval step now stores alignment
    per probe *line* — (slices, 5832 lines, positions) — rather than averaged
    into per-color rows. The per-color maps every earlier statistic uses are
    recovered by averaging over partners, so every earlier statistic is unchanged, and the
    line-keyed statistics below become computable. Per-line softmin weights
    are stored the same way. The volume stays small: 5 slices × 5832 lines ×
    ~6 positions in float32 is ≈ 0.7 MB per array, two arrays per run, so
    ≈ 35 MB across all {NRUNS} runs — an order of magnitude over ex-2.1.9's
    per-color arrays but nowhere near needing different processing, and the
    per-role trajectory adds well under a megabyte per run.

    **m_line**, the selectivity this experiment scores — m_span, generalized
    from colors to lines. To recap m_span (ex-2.1.9's scored statistic): on
    the per-color maps, take the affinity-weighted mean alignment minus the
    plain mean — a margin per (slice, role) saying how much closer red-ish
    material is to the axis than average — keep the best span role per
    slice, and average over slices. m_line is the same recipe on the
    per-line maps: weight *lines* instead of colors, using the labeller's
    own P(labeled) per line, normalized — "how much closer does a line sit
    to the axis for being the kind of line the labeller flags" — which keys
    on op1 alone in the old scheme and on both operands now. The
    max over roles is applied to every condition and the control alike, so
    its maximum-of-noise inflation cancels in the control-subtracted
    comparisons, as before. The trajectory records m_line, so retention reads
    off it. m_span and $m_\text{op1}$ are reported beside it for continuity.

    **Per-group weight profiles.** The localization instrument. Probe lines
    are weighted into two groups by the labeller's own one-slot-only
    probabilities — G1 by P(op1 drew and op2 didn't), G2 by the reverse — so
    no redness threshold has to be invented, and both-red lines (where either
    answer is defensible) get almost no say. Within each group,
    $\bar\pi_G(\ell, t)$ is the group-weighted mean softmin weight per slice
    and role. Under per-line localization the two profiles should differ —
    G1 stepping at op1, G2 at op2. Under a global winner they are identical
    no matter what: if every line carries the same weight profile, the two
    group means — which differ only in how lines are weighted — both return
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
    **(a)** at the embedding slice, each group's leading role is its own operand —
    op1 for G1, op2 for G2 — with weight $\ge {LEAD}$ in both (uniform is
    0.25);
    **(b)** across the four post-attention slices, the between-group contrast in
    op2 weight, $\mathrm{mean}_\ell\,[\bar\pi_{G2}(\ell, \text{op2}) -
    \bar\pi_{G1}(\ell, \text{op2})]$, is $\ge {CONTRAST}$.
    Partial: (a) holds and (b) lands in $[{CONTRASTP}, {CONTRAST})$.
    (a) is close to forced by the shared embeddings *if* the pull aligns red
    tokens at all, so on its own it mostly rules out the syntax latch; (b) is
    the real question — ex-2.1.9's deep-slice winner-take-all was per *run*,
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
    across seeds — nine of them here, a stricter read than ex-2.1.9's three);
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
    their oracle. Note that H4 can pass while H2(b) fails — the shared
    embeddings can deliver selectivity to labeled lines with no per-line
    localization at depth — and that pair of verdicts would itself be a
    finding: selectivity from the lexicon rather than from localization.
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
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Task cost (H1)

    /// admonition | TODO
    The usual table: `named_holdout` exact match by condition (seed mean ±
    spread) against the control, and the gap to the {TASK} gate. Expected:
    everything clean. A dirty condition would be news; a dirty primary leaves
    H2–H4 unscored, per the Hypotheses.
    ///
    """.replace("{TASK}", f"{ex.TASK_GATE:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where the weight went, by label group (H2)

    /// admonition | TODO
    The headline figure: per-group weight profiles $\bar\pi_{G1}(\ell,t)$ and
    $\bar\pi_{G2}(\ell,t)$ for the primary, in ex-2.1.9's smooth-step
    position-profile form (five slice rows with the embedding at the bottom,
    roles on the x-axis, the control's readability profile dashed behind —
    the preferred format for these results) — two columns (G1: op1 drew; G2:
    op2 drew), the
    embedding row annotated with the H2(a) verdicts and the deep-slice op2
    contrast (the H2(b) number) quoted in the caption. The same figure for
    the soft-τ arms beside it (same size, reported not gated), and per-seed
    hairlines since ex-2.1.9's deep-slice choices were per-run one-hot —
    seed-mean profiles alone could read as graded when the runs are not.
    With nine seeds on the primary, the hairlines double as the latch-rate
    count.

    Expected under per-line localization: G1 steps at op1, G2 at op2, at the
    embedding and (more diffusely) at depth. Global latch: the two columns
    look identical — then check where the latch sits (op1 would mean the old
    habit won; `+` would mean the repulsion lost the race it won in
    ex-2.1.9). Uniform: no step anywhere; the pool can't discriminate once
    the label is symmetric, and label-side mechanisms (per-line reweighting,
    EM-style responsibilities) move up the queue for M3.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## The operating point under the wider labeller (H3)

    /// admonition | TODO
    Two figures, all conditions with the primary called out. Fig 1:
    trajectories of
    m_line and ᾱ (op1) over training in the usual panel-grid form, schedules
    drawn beneath, the retention statistic annotated per panel. Fig 2: the
    grading scatter (per-color op1 response against `sim¹·⁵`), one panel per
    condition, each with its $r^2$; `op1-labels` is the reference the {DROP}
    drop is read against, and its own reproduction against ex-2.1.9's
    `pool-t100` (targets under *Calibration*) is quoted in
    the caption. Layout note: six conditions make a clean 2 × 3 in both
    figures, with the schedules drawn inside each trajectory panel rather
    than given a seventh — ex-2.1.9's five-plus-schedules grid was lopsided,
    and this count shouldn't be.

    Expected: the primary tracks `op1-labels` on all three statistics — the pull
    budget is unchanged, only its assignment to lines moved. Contrary worth
    watching for: ᾱ at op1 rising above the gate with the margin intact,
    which is the global-latch contamination signature (non-red op1 states
    pulled on op2-triggered lines) arriving before the H2 figure says so
    explicitly.
    ///
    """.replace("{DROP}", f"{ex.GRADE_R2_DROP:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selectivity against the slot oracle (H4)

    /// admonition | TODO
    Dot chart of control-subtracted m_line by condition (per-seed dots and
    their mean; nine dots on the primary), with `op1-labels` and
    `slot-oracle` as reference bands and
    the oracle fraction quoted beside the primary. m_span and $m_\text{op1}$
    reported
    alongside for continuity with ex-2.1.9. Expected: the primary between the
    references, nearer the oracle, as ex-2.1.9's pooled arms were to theirs.
    Contrary: ~half the oracle is the latch account's signature;
    landing *above* the oracle is possible in principle (the oracle pulls one slot
    per line even where the concept has migrated to `=` at depth) and would
    say the pool's freedom is worth more than the label's pointer.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Nothing here is preregistered; everything in this section is marked post
    hoc. Candidates we suspect we'll want: the per-role drift trajectory
    (when does the `+` embedding pay for its 0.30 alignment — early, against
    the repulsion's peak, or late as the anneal decays? — the question
    ex-2.1.9 queued and this run's trajectory finally records); per-line
    weight entropy at depth (is the deep-slice choice still one-hot per line
    once it varies per line?); the both-red lines (where either slot
    justifies the label, does the pool split or pick?); and how the soft-τ
    arms trade the H2 contrast against grading — whether a nearly-mean pull
    localizes at all once the label stops pointing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    After the results land, and after a discussion round: what the outcome
    means for document-level labels in M3 — in particular whether
    localization has to come from the loss side (pooling) or needs label-side
    help — and what it adds to the D2.1 close-out claim ("anchoring transfers
    under labels that don't say where the concept sits"). Not to be written
    in advance.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
