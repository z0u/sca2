import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.9: A pull that finds its own position",
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

    def load_ex218() -> tuple[dict, dict[str, np.ndarray]] | None:
        """Ex-2.1.8's published results, for the τ calibration and the reproduction targets."""
        store = project_store()
        arts = store.get_refs([ex.EX218_METRICS_REF, ex.EX218_ARRAYS_REF])
        m_art, a_art = arts[ex.EX218_METRICS_REF], arts[ex.EX218_ARRAYS_REF]
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
    # Ex 2.1.9: A pull that finds its own position

    /// tip |
    <!-- tl;dr -->
    Pulling on op1 alone worked best so far, but it relies on us knowing which
    token carries the concept, and natural language won't tell us that. So we
    replace the mean over the span in the anchor term with a soft minimum
    (mellowmax). The pull can then concentrate on whichever position aligns
    best, and we check where it chooses to concentrate.
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
    Ex-2.1.6 pulled the whole prompt span of *red*-labeled lines toward the
    anchor axis. The term was mostly satisfied by moving the whole color cube,
    so we got alignment without selectivity. Ex-2.1.7 found that two fixes
    stack:
    **(a)** a repulsive anti-subspace term, and
    **(b)** narrowing the pull to op1 alone.
    Ex-2.1.8 tuned the schedule of the repulsion (a) and landed on an operating
    point (`end90-hold30`) where the span pull holds the cube near the control
    (ᾱ = {ALPHA}) while keeping the margin ({MOP1}).

    That leaves the narrowing fix (b), which we can't keep. Pulling only op1
    works because this language has exactly one position that names the
    labeled concept, and we know which one it is, whereas a document-level label on
    natural text gives us a span and nothing else. Can the pull *find* the
    right position on its own, given the freedom to spread its budget unevenly?

    The pooled term (below) asks only that each labeled line align *somewhere*
    in its span, so the gradient concentrates on whichever position is already
    cheapest to align. The cheapest position is a shared syntax token: `+` and
    `=` appear in every line, so one nudge to their embedding satisfies every
    labeled line at once.

    The anti-subspace term should make that option unattractive. Every line,
    labeled or not, feels repulsion on a syntax token that sits on the axis,
    while repulsion on an alignment *specific to red operands* is countered by
    the pull where red actually occurs. If that balances out, the pull should
    settle on op1 at the embedding, the only span position whose state varies
    with the labeled color, and then follow the concept as it migrates across
    positions with depth, which ex-2.1.6 saw the un-anchored model do.

    It might not work out. The ex-2.1.8 span pull actually aligned the syntax
    tokens the most: at the embedding its best-aligned span roles (positions
    named by their job in the line: op1, `+`, op2, `=`) are `+` and `=`, and
    the cube-wide drift is there too. ᾱ measured at op1 is {ALPHA}, but the
    same statistic maxed over the span roles is {ASPAN}.[^aspan] The pooled
    term would happily settle on that. Call that outcome the *syntax latch* —
    a latch because the softmin's concentration is self-reinforcing (the
    leading position gets the pull, which extends its lead), so whatever it
    closes on first, it holds. The question is whether the repulsion redirects
    it before it closes.

    [^aspan]: Ex-2.1.8 gated ᾱ at op1 only; the maxed-over-roles number is
        computed here from its published arrays. Note the repulsion already
        acts on every position — the syntax tokens climbed the axis despite
        it, because the span-mean pull itself pushes them there (three of the
        four pulled positions). Raising the repulsion floor would set up a
        tug-of-war on those same states, and ex-2.1.8 left that lever untested
        (its best floor was its highest). Pooling goes after the cause
        instead: it lets the pull withdraw from positions it doesn't need.
    """.replace("{ALPHA}", f"{ex.EX218_REFERENCE['end90-hold30']['alpha_op1']:.2f}")
        .replace("{MOP1}", f"{ex.EX218_REFERENCE['end90-hold30']['m_op1']:.2f}")
        .replace("{ASPAN}", f"{ex.EX218_REFERENCE['end90-hold30']['alpha_span']:.2f}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    Everything follows ex-2.1.8 except the pooling: the word-level `v216`
    corpus from ex-2.1.3 (216 colors, one token each, d64-L4), the same seeds,
    sequence-level labels, measurements, anchor schedule, and the
    anti-subspace schedule fixed at the operating point of that experiment
    (`end90-hold30`) in every anchored cell.

    ### The pooled term

    Previously, the anchor term averaged $x_t = 1 - \cos(h_t, \hat v)$ over every
    pulled position. We replace that mean, per labeled line, with a soft
    minimum over the span of the line, the *mellowmax*[^mm]

    $$\mathrm{mm}_\tau(x) = -\tau \log \frac{1}{n} \sum_{t \in \text{span}} e^{-x_t / \tau},$$

    averaged over labeled lines and residual-stream slices as before.[^rs] The
    pooling happens *within* each slice, so the pull can choose different
    positions at different depths, following the concept as it migrates.

    Three properties made us pick this form over the other things called
    "softmin".
    First, its gradient is the softmin weights
    $\pi_t = \mathrm{softmax}(-x/\tau)_t$, which are non-negative and sum to 1,
    so the pull budget of each line is conserved and $\lambda_a$ means the same
    thing at every τ.
    Second, the $1/n$ inside the log makes $\tau = \infty$
    the span mean (we'll pin that with a unit test), so the τ = ∞ arm
    reproduces the ex-2.1.8 pull and the τ sweep interpolates from there down
    toward a hard min at τ = 0.
    Third, it never pushes.

    That last property is why we avoid the other common "softmin", the
    softmax-*weighted average* $\sum_t \pi_t x_t$. Its gradient terms flip sign
    once the cost of a position is τ past the pooled value, so poorly-aligned
    positions would be actively pushed off the axis (on a two-position toy at
    τ = 0.4, about 10% of the gradient budget). That would confound this
    experiment: we would be pulling the best position and pushing the rest.

    A training crop can truncate a line, so the pool runs over the *visible*
    span positions of the line, and each labeled line counts once. At τ = ∞
    that's a per-line mean of means rather than the flat mean over positions
    used in ex-2.1.8. The two differ only on truncated lines, and the
    reproduction check below will say whether that (or anything else in the new
    code path) moved the numbers.

    [^rs]: The *residual stream* is the running vector each token carries
        through the network; a slice is that vector read off at one depth.

    [^mm]: [Asadi & Littman, 2017](https://arxiv.org/abs/1612.05628). A
        temperature-controlled interpolation between min (τ → 0) and mean
        (τ → ∞); "softmin" is ambiguous between this and the softmax-weighted
        average, hence the care.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Joined with the block's own indent so every row dedents evenly under mo.md.
    _rows = "\n    ".join(
        f"| `{c['name']}` | {c['lam']:g} | {'∞' if np.isinf(c['tau']) else f'{c["tau"]:g}'} | "
        f"{'op1 only' if c['span'] == 1 else 'prompt span'} |"
        for c in ex.CONDITIONS
    )
    mo.md(
        r"""
    ### Conditions

    Six conditions, three seeds each. Every anchored cell runs the
    anti-subspace term at the ex-2.1.8 operating point; only the pooling
    varies.

    | condition | $\lambda_a$ | τ | pulled positions |
    |---|---|---|---|
    {ROWS}

    `span-mean` re-runs the `end90-hold30` condition of ex-2.1.8 through the
    pooled code path at τ = ∞: the baseline the pooled conditions have to beat,
    differing from them only in τ. `op1-oracle` is the op1-only pull of
    ex-2.1.7 at the new operating point: what knowing the right position buys,
    measured through identical code, so we can say what fraction of it pooling
    recovers. It is a reference rather than a strict ceiling — the oracle
    pulls op1 at every slice, and ex-2.1.6 saw the concept migrate to later
    roles with depth, so a per-slice pool could in principle beat it.
    """.replace("{ROWS}", _rows)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibrating τ

    The interesting τ range depends on how much $x_t$ varies across the span,
    which we can read off the stored alignment maps of ex-2.1.8 before running
    anything. The figure below computes what the softmin weights *would* be on
    two reference profiles: a fresh run (the un-anchored control) and a trained
    one (the span pull at the operating point).
    """)
    return


@app.cell(hide_code=True)
def _():
    _e218 = load_ex218()
    mo.stop(
        _e218 is None,
        mo.md("_Ex-2.1.8's published results aren't reachable from here; the calibration figure needs them._"),
    )
    assert _e218 is not None
    _metrics218, arrays218 = _e218

    def ex218_alpha(cond: str) -> np.ndarray:
        """Ex-2.1.8's seed-mean alignment map for a condition: (slices, colors, positions)."""
        return np.mean([arrays218[f"{cond}-s{s}/alpha"] for s in [0, 1, 2]], axis=0)

    def leading_weight(alpha: np.ndarray, tau: float) -> np.ndarray:
        """Label-weighted mean of the leading softmin weight, per residual slice."""
        w = ex.softmin_weights(1.0 - alpha[:, :, : ex.SPAN], tau)
        return np.einsum("lc,c->l", w.max(axis=-1), ex.LABEL_W)

    ALPHA218 = {c: ex218_alpha(c) for c in ["lam0", "end90-hold30"]}
    return ALPHA218, leading_weight


@app.cell(hide_code=True)
def _(ALPHA218, leading_weight):
    _taus = np.geomspace(0.003, 1.0, 60)
    _lead = {c: np.array([leading_weight(a, t) for t in _taus]) for c, a in ALPHA218.items()}

    @themed(
        name="tau-calibration",
        alt_text="""
            Two line charts of leading softmin weight against temperature tau on a log
            scale from 0.003 to 1. Left panel: the un-anchored control's alignment
            profile; right panel: the ex-2.1.8 span pull at the operating point. Each
            panel has five curves, one per residual slice, all starting near 1 at the
            left edge and descending to 0.25 at the right edge. A horizontal band from
            0.6 to 0.8 is shaded, and three vertical dashed lines mark tau = 0.01, 0.03
            and 0.1. Most curves pass through the shaded band between tau = 0.005 and
            0.03; in the right panel the last slice's curve sits noticeably higher,
            crossing the band nearer 0.06 to 0.09.
        """,
        caption=r"""
            **Calibrating the τ grid from the ex-2.1.8 stored alignments.** The
            leading span position's share of the softmin weight, by residual
            slice, if the weights were computed from the alignment profile of
            the un-anchored control (**left**) or the span pull at the
            operating point (**right**). The shaded band is the 0.6–0.8 target
            from the design note; dashed verticals are the chosen grid. Both
            profiles put the band at τ ≈ 0.01–0.03, so the grid brackets it
            with one rung toward the mean.
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
            axs, [("lam0", "control profile"), ("end90-hold30", "span-pull profile")], strict=True
        ):
            ax.axhspan(*ex.LEAD_BAND, color=_grey, alpha=0.15, lw=0)
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
def _(ALPHA218):
    _x0 = np.einsum("ct,c->t", 1.0 - ALPHA218["end90-hold30"][0, :, : ex.SPAN], ex.LABEL_W)
    _m218 = ex.pooled_margin(ALPHA218["end90-hold30"], ex.LABEL_W)
    _m218c = ex.pooled_margin(ALPHA218["lam0"], ex.LABEL_W)
    _as218 = ex.alpha_span(ALPHA218["end90-hold30"])
    _as218c = ex.alpha_span(ALPHA218["lam0"])
    mo.md(
        rf"""
    The design note targets a leading position taking 0.6–0.8 of the weight, which
    lands at τ ≈ 0.01–0.03 on both profiles.

    So the grid is $\tau \in \{{{", ".join(f"{t:g}" for t in ex.TAUS)}\}}$: the
    sharp edge of the band, the soft edge, and one rung toward the mean end of
    the τ axis (τ = 0.1, weights nearer uniform); the mean itself is the
    τ = ∞ arm, which closes the family. τ = 0.01 sits below the per-seed margin noise
    ({ex.NOISE_PER_SEED:g}), so at that rung the pool will commit hard to early differences
    that may just be noise. That is the winner-take-most end of the sweep.

    The ex-2.1.8 arrays give the reproduction targets for the `span-mean` arm,
    as seed means: pooled margin m_span = {_m218:.2f} (control {_m218c:.2f})
    and ᾱ_span = {_as218:.2f} (control {_as218c:.2f}). They also put numbers
    on the syntax latch: at the embedding slice, the alignment
    $\cos(h, \hat v)$ of the span-pull run's states, averaged over colors with
    label-affinity weights, is `+` {1 - _x0[1]:.2f} and `=` {1 - _x0[3]:.2f},
    against op1 {1 - _x0[0]:.2f} and op2 {1 - _x0[2]:.2f} — a pool applied to
    *this* profile would close on the syntax tokens.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Measurements

    Alignment maps ($\cos(h, \hat v)$ at every slice, color, and role, each
    color averaged over its 27 probe lines), task evals, trajectories and
    probes are from ex-2.1.8 unchanged. Labels also still key on the color of
    op1 alone, as in every experiment since ex-2.1.6: pooling moves where the
    *pull* acts, never which lines are labeled. Three statistics are new or
    newly scored, all computed from the same maps:

    **The pooled margin m_span** is the mean over slices of the max-over-span
    margin: for each slice, take the margin at each span role and keep the
    largest, then average over slices. The margin at a role is selectivity
    rather than raw alignment: over the 216 colors, the affinity-weighted mean
    alignment of their states at that (slice, role), minus the unweighted
    mean — how much closer a line sits to the axis for having a red op1. At
    depth the syntax roles' states vary with the color of op1 too (they attend
    back to it), so the margin means the same thing at every role; there is no
    separate notion of how aligned a role *should* be. We can't score the
    margin at op1 alone, because one condition pulls exactly where we would be
    scoring, and op1 may not be where the concept belongs at depth anyway.

    The max is applied identically to every condition and to the control, so
    the maximum-of-noise inflation it invites cancels out in the
    control-subtracted comparisons the hypotheses make. We report
    $m_\text{op1}$ beside it for continuity with ex-2.1.7 and ex-2.1.8.

    **ᾱ_span** is the counterpart for containment, with the same max
    treatment: for each slice, the unweighted mean alignment over the 216
    colors' states at each span role, keeping the largest role; then the mean
    over slices. Color enters only through that mean — this is drift of the
    whole cloud, not selectivity. The usual ᾱ reads at op1 only, which the
    ex-2.1.8 operating point shows isn't enough here: its cube-wide drift sits
    on the syntax roles, where the op1 statistic can't see it.

    **The softmin weight profile** $\bar\pi(l, t)$ records what the pull
    actually chose. (π, since $w$ collides with model weights.) At the end of
    training we compute the weights $\mathrm{softmax}(-x/\tau)$ on the
    alignment probe set (every color as op1 against all 27 closed partners),
    per slice and span role, weighted by label affinity over colors.

    Its reference is **the readability profile of the control** $R^2(l, t)$: a
    ridge probe predicting the redness of op1 from the state of the
    *un-anchored control* at each (slice, role), with one op1 color held out
    at a time. This $R^2$ is the probe's held-out coefficient of
    determination (negative on a role that carries nothing), a different
    quantity from the squared Pearson correlation that grading uses; earlier
    reports wrote both as $R^2$, so here they are $R^2$ (probes) and $r^2$
    (grading). At the embedding we know this profile in advance, since op1 is
    the only span position whose state varies with the color of op1. H4(a)
    relies on this for its prediction.
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: H2(a)'s gate of 0.15 was described as "about a third" of ex-2.1.7's
    # op1-oracle gain; that gain is 0.22, so it is about two thirds. Corrected the
    # prose and interpolated the number rather than restating it. The gate itself
    # is unchanged — its noise-based derivation (~3.3× the seed-mean difference
    # noise) is the binding one. Verify: `EX217_OP1_GAIN` in experiment.py.
    # REVIEW: H4(b) said "the four position-aware slices" without saying which of
    # the five they are; fixed to the four after the embedding, since at the
    # embedding a token's state cannot depend on any other position, which is the
    # same fact H4(a) relies on. Verify: the R² profile's embedding row should be
    # ~0 off op1.
    # REVIEW: P was "the task-clean finite-τ condition with the largest seed-mean
    # m_span", which selects by the statistic H2(a) then gates — a maximum over
    # three arms scored against a single-comparison gate. P is now fixed in
    # advance to `pool-t030` by the calibration-band criterion (its τ sits inside
    # the 0.6–0.8 band, at its soft edge, on both reference profiles), which is
    # independent of every gated statistic; the other pooled arms demote to a
    # reported sweep. Verify: the calibration figure — if the band moves off
    # τ = 0.03 when recomputed, the choice of P should move with it before the
    # freeze.
    # REVIEW: H3(c) said "within 0.1" of the span-mean arm's grading, which reads
    # two-sided; the intent is a floor, so it now says "no more than 0.1 below
    # (higher is fine)". Verify: `GRADE_R2_DROP` names a drop, not a distance.
    mo.md(
        r"""
    ## Hypotheses

    Gates and noise floors are from ex-2.1.6 through ex-2.1.8 wherever the
    statistic is unchanged. Derivations for the new gates are in
    [experiment.py](./experiment.py), next to their constants.

    Throughout, *P* names the primary pooled condition, fixed in advance as
    `{PRIMARY}`: its τ sits at the calibration band's soft edge on both
    reference profiles — the conservative end of the target range — a
    criterion independent of every gated statistic. H2–H4 all score P. The
    other two pooled arms are the τ sweep, reported beside P but not gated.
    If P turns out task-dirty, H2–H4 go unscored and the refuted H1 is the
    finding.

    **H1: Task cost.** Every condition is task-clean: `named_holdout` exact
    match within {TASK} (absolute) of the seed mean of the in-experiment
    control. No partial credit: ex-2.1.7 found no task cost even at ten times
    this anchor weight, and nothing in this design should add task pressure.

    **H2: Pooling increases selectivity over the span mean.**
    (a) $m_\text{span}(P) - m_\text{span}(\text{span-mean}) \ge {GAIN}$,
    about two thirds of what the op1 oracle gave at op1 in ex-2.1.7
    ({OP1GAIN}), and roughly 3× the noise of a difference of three-seed
    means.
    (b) The syntax-role drift falls: $\bar\alpha_\text{span}(P) \le
    \bar\alpha_\text{span}(\text{span-mean}) - {SDRIFT}$.
    Partial: (b) holds and (a) lands in $[{GAINP}, {GAIN})$; or exactly one
    of (a), (b) holds outright. (b) is a prediction of the mechanism: a
    pull that concentrates its budget should stop spending alignment on
    positions that don't help it. A condition that gains margin while leaving
    the syntax drift in place has found some other mechanism.

    **H3: The operating point survives pooling.** On P:
    (a) containment, $\bar\alpha \le {ALPHA}$ at op1;
    (b) retention, every run whose running maximum of $m_\text{span}$ reaches
    {FLOOR} ending at $\ge {RET}\times$ that maximum (quoted as the minimum
    across seeds);
    (c) grading, the squared Pearson correlation $r^2$ against `sim¹·⁵` of
    P's seed-mean op1 response no more than {DROP} below the `span-mean`
    arm's (higher is fine).[^grading-note]
    Partial: (a) and (b) hold, (c) misses.

    **H4: The weight lands where the concept lives.** On the weight profile of P:
    (a) at the embedding slice, the leading role is op1 with weight
    $\ge {LEAD}$ (uniform is 0.25; this gates the *trained* weights, where
    the 0.6–0.8 calibration band described weights recomputed on reference
    profiles);
    (b) across the four slices after the embedding, the weighted readability gain —
    $\sum_t \bar\pi(l,t) R^2(l,t) - \mathrm{mean}_t R^2(l,t)$, averaged over
    slices — is at least {RGAIN}.
    Partial: (a) holds and (b) lands in $[0, {RGAIN})$.
    Two contrary outcomes, both leading to the same conclusion: span labels
    need help from the label side (per-line reweighting), not just the loss
    side. *Latch*: the syntax latch from the intro arrives — the embedding
    weight concentrates on `+` or `=`, meaning the pull took the cheap shared
    tokens and the repulsion failed to make them unattractive; a latch would
    take H2(b) down with it — one mechanism, seen from the loss side there
    and the weight side here. *Uniform*: no slice reaches {LEAD} in any
    pooled condition, meaning the pool can't tell the positions apart at any
    τ short of collapse.

    [^grading-note]: The Spearman ρ grading track is retired as of this
        experiment. The ceiling analysis in ex-2.1.8 showed that ρ against
        `redness` can't reach its 0.8 gate alongside full-amplitude containment
        (best reachable 0.59–0.82), because the response target isn't monotone
        in redness. The $r^2$ track survives, but as a relative gate; ex-2.1.8
        left the question of where a realistic absolute bar sits to a later
        grid.
    """.replace("{TASK}", f"{ex.TASK_GATE:g}")
        .replace("{PRIMARY}", ex.PRIMARY)
        .replace("{GAIN}", f"{ex.POOL_GAIN_GATE:g}")
        .replace("{GAINP}", f"{ex.POOL_GAIN_PARTIAL:g}")
        .replace("{OP1GAIN}", f"{ex.EX217_OP1_GAIN:g}")
        .replace("{SDRIFT}", f"{ex.SYNTAX_DRIFT_GATE:g}")
        .replace("{ALPHA}", f"{ex.MEAN_ALIGN_GATE:g}")
        .replace("{FLOOR}", f"{ex.RETENTION_FLOOR:g}")
        .replace("{RET}", f"{ex.RETENTION_GATE:g}")
        .replace("{DROP}", f"{ex.GRADE_R2_DROP:g}")
        .replace("{LEAD}", f"{ex.LEAD_GATE:g}")
        .replace("{RGAIN}", f"{ex.READ_GAIN_GATE:g}")
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
    everything clean, as in every anchored experiment on this testbed so far. A
    dirty condition would be news in itself; a dirty P leaves H2–H4 unscored,
    per the Hypotheses.
    ///
    """.replace("{TASK}", f"{ex.TASK_GATE:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: the method promises a reproduction check on the `span-mean` arm but
    # no section scored it; folded it into this placeholder, where its targets
    # already live. It gates nothing on its own — a miss would be read as the new
    # code path having moved the numbers, which changes how H2 is read.
    mo.md(r"""
    ## Selectivity against the span mean (H2)

    /// admonition | TODO
    Dot chart of control-subtracted $m_\text{span}$ by condition (three seeds
    and their mean), with `span-mean` and `op1-oracle` as reference bands, and
    the fraction of the oracle gap recovered,
    $(m_\text{span}(P) - m_\text{span}(\text{mean})) /
    (m_\text{span}(\text{oracle}) - m_\text{span}(\text{mean}))$, quoted
    beside it. Second panel: ᾱ_span the same way. $m_\text{op1}$ reported
    beside $m_\text{span}$ for continuity. An accompanying table carries the
    reproduction check the method promises: `span-mean`'s $m_\text{span}$ and
    ᾱ_span against the ex-2.1.8 targets quoted under *Calibrating τ*, so a
    drift introduced by the new code path is visible before any pooled
    condition is read.

    Expected under the mechanism: pooled conditions between the two references
    on $m_\text{span}$ — though above `op1-oracle` is possible, since the
    oracle pulls op1 at every slice while the concept migrates to later roles
    with depth — and *below* `span-mean` on ᾱ_span. Contrary: pooled
    $m_\text{span}$ at or under the span mean would say pooling bought
    nothing; pooled ᾱ_span unchanged while $m_\text{span}$ rises would say
    the margin gain came from something other than withdrawing the pull on
    syntax tokens. If the syntax drift stays, the next lever is the repulsion
    floor itself, which ex-2.1.8 left untested upward (its best floor was its
    highest) — queued, not gated here.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## The operating point under pooling (H3)

    /// admonition | TODO
    Two figures, both covering *all* conditions with P called out (scored
    numbers quoted for P only; showing the rest is reporting the sweep, not
    an exploratory analysis). Fig 1: trajectories of $m_\text{span}$ and ᾱ
    (op1) over training, in the format of the earlier trajectory figures — a
    grid with one panel per condition, the other conditions ghosted for
    reference, and the anchor/anti schedules drawn beneath; the retention
    statistic annotated on each panel and quoted in the caption. Fig 2: the
    grading scatter (per-color op1 response against `sim¹·⁵`), one panel per
    condition, each with its $r^2$.

    Expected: P tracks `span-mean` on containment and retention, since pooling
    redirects the budget of the pull rather than adding to it, and grading
    within {DROP}. Contrary: a pooled pull that concentrates early and hard
    could slide after the anti anneal ends, showing up as a retention miss; or
    concentrating the pull could sharpen the response into a step, showing up
    as a grading miss.
    ///
    """.replace("{DROP}", f"{ex.GRADE_R2_DROP:g}")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where the weight went (H4)

    /// admonition | TODO
    Illustration of the primary effect: $\bar\pi(l, t)$ as a smooth-step
    stacked line chart over slices, one series per role (the ex-2.1.4 and
    ex-2.1.6 form), one figure per pooled condition with P called out — a
    grid is optional. The readability profile of the control $R^2(l, t)$
    beside it on the same axes; the stacked form leaves room for both where
    a heatmap wouldn't. Annotate the embedding slice, where H4(a) is
    decided, and quote the four-slice mean readability gain (the H4(b)
    number; per-slice values only if the chart leaves room).
    Expected under the mechanism: weight on op1 at the embedding, migrating
    toward `+`/`=`/op2 with depth in step with readability. Latch outcome:
    embedding weight on `+`/`=`, where readability is ~0. Uniform outcome:
    every slice's weights near 0.25. The weights may track readability
    better than the margin gains suggest, meaning the pull found the right
    place without any payoff; the discussion should then treat finding the
    mechanism and benefiting from it separately — the mechanism may work
    while buying nothing at this scale or on this testbed.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Nothing here is preregistered; everything in this section is marked post
    hoc. Candidates we suspect we'll want: the trajectory of the weight
    profile over training (does concentration happen during the anchor ramp,
    or only once the repulsion anneals? — the trajectory instrument will
    record the small per-slice-and-role weight profile at each measurement,
    so this is a stored summary rather than something to recompute), per-color
    weight maps (do strongly-red lines concentrate differently from weakly-red
    ones?), and how sensitive the H4 profile is to τ across the grid.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    After the results land, and after a discussion round: what the outcome
    implies for the document-level labels in M3, and whether label-side
    mechanisms (per-line reweighting, EM-flavored schemes) are needed at all.
    Not to be written in advance.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
