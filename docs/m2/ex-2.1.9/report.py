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
    (mellowmax), and check where the pull chooses to concentrate. It finds the
    right position on its own — at the embedding the weight lands on op1, the
    only position that can carry the concept there — and the operating point
    stays healthy, with grading improving. What it doesn't do is buy margin:
    the span mean's max-over-roles margin already sits near the oracle's, so
    there was almost no headroom for the pooled pull to claim. At depth the
    concentration is winner-take-all run by run — at the two sharper τ some
    runs latch onto `+` in the first few epochs — and only the softest rung
    keeps a graded profile in every run.
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
def _(a_op1, a_span, acc, grading_r2, m_span, pi_bar, read_gain, retention):
    _P = ex.PRIMARY
    _ctrl = acc("lam0").mean()
    _gain = m_span(_P).mean() - m_span(ex.MEAN_ARM).mean()
    _headroom = m_span(ex.ORACLE_ARM).mean() - m_span(ex.MEAN_ARM).mean()
    _drop = a_span(ex.MEAN_ARM).mean() - a_span(_P).mean()
    _ret = retention(_P)
    assert _ret is not None
    mo.md(rf"""
    **H1 (task cost) — holds.** The largest `named_holdout` exact-match gap
    from the control across all five anchored conditions is
    {max(abs(acc(c).mean() - _ctrl) for c in ["span-mean", *ex.POOLED_NAMES, ex.ORACLE_ARM]):.4f},
    against a gate of {ex.TASK_GATE:g}.

    **H2 (pooling increases selectivity) — unmet.** On the primary condition
    (`{_P}`):
    (a) the margin gain over the span mean is {_gain:+.3f} against a gate of
    {ex.POOL_GAIN_GATE:g} (partial {ex.POOL_GAIN_PARTIAL:g}), and
    (b) the syntax-role drift falls by {_drop:.3f} against a required
    {ex.SYNTAX_DRIFT_GATE:g} — the predicted direction on both, short of both
    gates, and neither partial applies. One piece of context, unpacked in the
    selectivity section: the op1 oracle itself beats the span mean by only
    {_headroom:.3f} on $m_\text{{span}}$ here, so H2(a)'s gate was out of
    reach for any pull; `{_P}` recovered {_gain / _headroom:.0%} of the
    headroom that existed.

    **H3 (the operating point survives pooling) — holds.** On `{_P}`: containment
    $\bar\alpha$ = {a_op1(_P).mean():.3f} (gate ≤ {ex.MEAN_ALIGN_GATE:g});
    retention {_ret:.2f} (gate ≥ {ex.RETENTION_GATE:g}, minimum across seeds);
    and grading $r^2$ = {grading_r2(_P):.2f} against the span mean's
    {grading_r2(ex.MEAN_ARM):.2f} — the gate tolerated a drop of
    {ex.GRADE_R2_DROP:g} and instead the response got better graded.

    **H4 (the weight lands where the concept lives) — holds.** At the
    embedding slice `{_P}` puts {pi_bar(_P)[0, 0]:.2f} of the softmin weight
    on op1 (gate ≥ {ex.LEAD_GATE:g}; uniform is 0.25) — the only span
    position whose state can carry the concept there — and the weighted
    readability gain is {read_gain(_P):+.2f} (gate ≥ {ex.READ_GAIN_GATE:g}).
    At depth the concentration is winner-take-all seed by seed: at the two
    sharper τ some seeds latch onto `+` in the first few epochs, and only
    τ = 0.1 keeps a graded profile on op1 in every seed (*Where the weight
    went*).

    So the pull finds the concept-bearing position without the oracle, and
    holding the operating point costs nothing — but finding it buys almost no
    additional margin here. What that means for M3's document-level labels is
    for the discussion round.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this report
    This report was preregistered: the method, hypotheses and decision
    thresholds were frozen before the experiment ran (commit `0abebb8`, with
    pre-launch review corrections noted in comments). Results replaced the
    `TODO` placeholders in place; everything conceived after seeing the data
    is under *Exploratory analyses*, marked as post hoc.
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
    the span mean (we'll pin that with a unit test), so the τ = ∞ condition
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
    return ALPHA218, arrays218, leading_weight


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
def _(ALPHA218, arrays218):
    _x0 = np.einsum("ct,c->t", 1.0 - ALPHA218["end90-hold30"][0, :, : ex.SPAN], ex.LABEL_W)

    def _seed_mean(stat, cond: str) -> float:
        """Mean of the per-run statistic — the convention every result is scored under.

        Both pooled statistics take a max over roles, so computing them on the
        seed-mean map instead would suppress the max-of-noise inflation that
        the control subtraction is there to absorb, and understate the
        control's ᾱ_span in particular.
        """
        return float(np.mean([stat(arrays218[f"{cond}-s{s}/alpha"]) for s in (0, 1, 2)]))

    _m218 = _seed_mean(lambda a: ex.pooled_margin(a, ex.LABEL_W), "end90-hold30")
    _m218c = _seed_mean(lambda a: ex.pooled_margin(a, ex.LABEL_W), "lam0")
    _as218 = _seed_mean(ex.alpha_span, "end90-hold30")
    _as218c = _seed_mean(ex.alpha_span, "lam0")
    mo.md(
        rf"""
    The design note targets a leading position taking 0.6–0.8 of the weight, which
    lands at τ ≈ 0.01–0.03 on both profiles.

    So the grid is $\tau \in \{{{", ".join(f"{t:g}" for t in ex.TAUS)}\}}$: the
    sharp edge of the band, the soft edge, and one rung toward the mean end of
    the τ axis (τ = 0.1, weights nearer uniform); the mean itself is the
    τ = ∞ condition, which closes the family. τ = 0.01 sits below the per-seed margin noise
    ({ex.NOISE_PER_SEED:g}), so at that rung the pool will commit hard to early differences
    that may just be noise. That is the winner-take-most end of the sweep.

    The ex-2.1.8 arrays give the reproduction targets for the `span-mean` condition,
    as means of per-run statistics — the convention the results are scored
    under: pooled margin m_span = {_m218:.2f} (control {_m218c:.2f})
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
    # REVIEW (pre-launch): recomputing the calibration puts τ = 0.03's leading
    # weight at the band's soft edge on the two shallowest slices and 0.4–0.5
    # on the deeper three, so "sits at the soft edge" over-claimed; P and its
    # criterion are unchanged, the wording now says "at or just below". The
    # reproduction targets are likewise now means of per-run statistics
    # (per-run max over roles, then mean), matching how results are scored —
    # on the seed-mean map the max-of-noise inflation cancels and the
    # control's ᾱ_span reads ~2.6× lower than the scoring convention's.
    mo.md(
        r"""
    ## Hypotheses

    Gates and noise floors are from ex-2.1.6 through ex-2.1.8 wherever the
    statistic is unchanged. Derivations for the new gates are in
    [experiment.py](./experiment.py), next to their constants.

    In the hypothesis statements, *P* names the primary pooled condition,
    fixed in advance as `{PRIMARY}`: on both reference profiles its τ sits at
    the calibration band's soft edge on the shallow slices, and below the
    band on the deeper ones — the conservative end of the target range — a
    criterion independent of every gated statistic. H2–H4 all score P; the
    results prose calls it `{PRIMARY}` or "the primary condition". The other
    two pooled conditions are the τ sweep, reported beside P but not gated.
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
    condition's (higher is fine).[^grading-note]
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
    cells = {c["label"]: c for c in metrics["cells"]}

    CONDS: list[str] = [c["name"] for c in ex.CONDITIONS]
    ANCHORED = [c for c in CONDS if c != "lam0"]

    def over_seeds(cond: str, fn) -> np.ndarray:
        """One value per seed for a condition, in seed order."""
        return np.array([fn(cells[f"{cond}-s{s}"]) for s in ex.SEEDS])

    def acc(cond: str, set_name: str = "named_holdout", key: str = "accuracy") -> np.ndarray:
        return over_seeds(cond, lambda c: c["sets"][set_name][key])

    def alpha_map(cond: str) -> np.ndarray:
        """Seed-mean alignment map for a condition: (slices, colors, roles)."""
        return np.mean([arrays[f"{cond}-s{s}/alpha"] for s in ex.SEEDS], axis=0)

    def m_span(cond: str) -> np.ndarray:
        """Per-seed m_span — the statistic H2(a) reads."""
        return over_seeds(cond, lambda c: c["m_span"])

    def a_span(cond: str) -> np.ndarray:
        """Per-seed ᾱ_span — the drift statistic H2(b) reads."""
        return over_seeds(cond, lambda c: c["alpha_span"])

    def m_op1(cond: str) -> np.ndarray:
        return over_seeds(cond, lambda c: c["m_op1"])

    def a_op1(cond: str) -> np.ndarray:
        """Per-seed ᾱ at op1 — the containment statistic H3(a) reads."""
        return over_seeds(cond, lambda c: c["alpha_mean_op1"])

    def traj(cond: str, seed: int, key: str) -> np.ndarray:
        return np.asarray(cells[f"{cond}-s{seed}"]["traj"][key], dtype=float)

    return (
        ANCHORED,
        CONDS,
        a_op1,
        a_span,
        acc,
        alpha_map,
        arrays,
        cells,
        m_op1,
        m_span,
        metrics,
        traj,
    )


@app.cell(hide_code=True)
def _(alpha_map, cells, metrics, traj):
    def retention(cond: str) -> float | None:
        """Lowest final-over-peak m_span across the seeds that reached the floor.

        The minimum, per `ex.RETENTION_STAT`; None when no run reached the
        floor (the control), leaving the criterion vacuous rather than failed.
        """
        r = [t[-1] / t.max() for s in ex.SEEDS if (t := traj(cond, s, "m_span")).max() >= ex.RETENTION_FLOOR]
        return min(r) if r else None

    def pi_bar(cond: str) -> np.ndarray:
        """Seed-mean weight profile π̄(l, t): (slices, span roles)."""
        return np.mean([np.asarray(cells[f"{cond}-s{s}"]["pi"], dtype=float) for s in ex.SEEDS], axis=0)

    READ: np.ndarray = np.mean([np.asarray(v, dtype=float) for v in metrics["readability"].values()], axis=0)
    # The control's readability profile R²(l, t), mean over the three un-anchored seeds.

    def grading_r2(cond: str) -> float:
        """r² of the seed-mean per-color op1 response against sim¹·⁵ — H3(c)'s statistic."""
        return ex.r2_sim(alpha_map(cond)[:, :, 0].mean(axis=0))

    def read_gain(cond: str) -> float:
        """H4(b): mean over the four slices after the embedding of Σ_t π̄·R² − mean_t R²."""
        g = (pi_bar(cond) * READ).sum(axis=1) - READ.mean(axis=1)
        return float(g[1:].mean())

    return READ, grading_r2, pi_bar, read_gain, retention


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
    {ex.TASK_GATE:g}. Every condition solves the task, so H2–H4 are all
    scored. Concentrating the whole pull budget of a line onto one position —
    which at τ = 0.01 is nearly a hard min — costs nothing the task loss can
    see, consistent with every anchored experiment on this testbed so far.
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: the method promises a reproduction check on the `span-mean` arm but
    # no section scored it; folded it into this placeholder, where its targets
    # already live. It gates nothing on its own — a miss would be read as the new
    # code path having moved the numbers, which changes how H2 is read.
    mo.md(r"""
    ## Selectivity against the span mean (H2)

    First the reproduction check the method promises, so any drift introduced
    by the new code path is visible before a pooled condition is read: the
    `span-mean` condition re-runs ex-2.1.8's operating point through the pooled
    code at τ = ∞, so it should land on that experiment's numbers.
    """)
    return


@app.cell(hide_code=True)
def _(a_op1, a_span, m_op1, m_span):
    _stats = {
        "m_span": m_span,
        "alpha_span": a_span,
        "m_op1": m_op1,
        "alpha_op1": a_op1,
    }
    _names = {
        "m_span": "m<sub>span</sub>",
        "alpha_span": "ᾱ<sub>span</sub>",
        "m_op1": "m<sub>op1</sub>",
        "alpha_op1": "ᾱ (op1)",
    }

    def _row(cond: str, ref_key: str) -> str:
        ref = ex.EX218_REFERENCE[ref_key]
        tds = ""
        for k, fn in _stats.items():
            v = fn(cond)
            tds += (
                f"<td class='num'>{v.mean():.3f} <span class='range'>±{v.std(ddof=1):.3f}</span></td>"
                f"<td class='num'>{ref[k]:.3f}</td><td class='num'>{v.mean() - ref[k]:+.3f}</td>"
            )
        return f"<tr><th><code>{cond}</code></th>{tds}</tr>"

    _head = "".join(
        f"<th class='num'>{_names[k]}</th><th class='num'>ex-2.1.8</th><th class='num'>Δ</th>" for k in _stats
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr><th>condition</th>{_head}</tr></thead>
      <tbody>{_row("span-mean", "end90-hold30")}{_row("lam0", "lam0")}</tbody>
    </table></div>
    """
    _caption = """
    The reproduction check. Each statistic is this experiment's seed mean (±
    seed standard deviation) beside the same statistic recomputed from
    ex-2.1.8's published arrays under the same per-run convention, and the
    difference. <code>span-mean</code> re-runs <code>end90-hold30</code>
    through the pooled code path; <code>lam0</code> re-runs its control.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(a_span, m_span):
    _dm = m_span("span-mean").mean() - ex.EX218_REFERENCE["end90-hold30"]["m_span"]
    _da = a_span("span-mean").mean() - ex.EX218_REFERENCE["end90-hold30"]["alpha_span"]
    mo.md(rf"""
    The largest differences — {_dm:+.3f} on $m_\text{{span}}$ and {_da:+.3f}
    on ᾱ_span — sit within a seed spread of the targets, and the same is true
    of the control. So the per-line mean-of-means (which differs from
    ex-2.1.8's flat mean only on crop-truncated lines) reproduces the
    operating point, and the pooled conditions can be read against
    `span-mean` as a like-for-like baseline.
    """)
    return


@app.cell(hide_code=True)
def _(ANCHORED, a_span, m_span):
    _CTRL_M = float(m_span("lam0").mean())
    _CTRL_A = float(a_span("lam0").mean())
    # Control-subtracted per-seed values, computed once — @themed draws twice.
    _M = {c: m_span(c) - _CTRL_M for c in ANCHORED}
    _A = {c: a_span(c) - _CTRL_A for c in ANCHORED}
    _GAIN_AT = float(_M["span-mean"].mean())
    _DRIFT_AT = float(_A["span-mean"].mean())

    @themed(
        name="h2-dots",
        alt_text="""
            Two dot-plot panels sharing a horizontal cosine axis from 0 to about
            0.75, with five condition rows: span-mean, three pooled conditions at
            increasing tau, and op1-oracle. Each row shows three small seed dots
            and a vertical tick at their mean. Left panel, margin: all five rows cluster
            between 0.62 and 0.69, with a dotted partial line at 0.71 and a
            dashed gate line at 0.78, both to the right of every row; the
            pooled rows sit slightly right of span-mean, below the oracle row.
            Right panel, drift: the rows walk steadily
            left from span-mean at 0.34 down to op1-oracle at 0.13, with the
            dashed gate at 0.24 sitting left of the pool-t030 row.
        """,
        caption=rf"""
            **The H2 statistics, control-subtracted.** Left:
            $m_\text{{span}}$; right: ᾱ_span. Small dots are seeds, the tick
            their mean; the τ = ∞ and oracle reference rows are greyed.
            Dashed carets mark the H2 gates for P: a margin gain of
            {ex.POOL_GAIN_GATE:g} over `span-mean` (dotted: the
            {ex.POOL_GAIN_PARTIAL:g} partial) and a drift drop of
            {ex.SYNTAX_DRIFT_GATE:g} below it. The two panels share their
            scale, so the sizes of margin and drift compare directly.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesRow

        _rows = ANCHORED[::-1]  # top-to-bottom reading order after inversion
        _grey = light_dark("#999", "#777")
        _gate = light_dark("#555", "#bbb")
        _ink = {
            "span-mean": _grey,
            "op1-oracle": _grey,
            "pool-t010": light_dark("#7fb3d8", "#4f7ea6"),
            "pool-t030": light_dark("#1f6fb4", "#5fa8dd"),
            "pool-t100": light_dark("#12497c", "#8ec4ea"),
        }
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, data, title in zip(
            axes, (_M, _A), (r"$m_\mathrm{span}$ − control", "ᾱ$_\\mathrm{span}$ − control"), strict=True
        ):
            for yi, cond in enumerate(_rows):
                v = data[cond]
                ax.plot(v, [yi] * len(v), "o", ms=3.5, color=_ink[cond], alpha=0.55, mew=0)
                ax.plot(v.mean(), yi, "|", ms=18, color=_ink[cond])
            ax.set_title(title, fontsize="small", color=_grey)
            ax.set_yticks(range(len(_rows)), _rows, fontsize=8)
            ax.set_xlim(0, 0.82)
            ax.tick_params(axis="x", labelsize=8)
        for x, ls in ((_GAIN_AT + ex.POOL_GAIN_GATE, (0, (4, 3))), (_GAIN_AT + ex.POOL_GAIN_PARTIAL, (0, (1, 2)))):
            axes[0].axvline(x, color=_gate, lw=0.9, ls=ls)
        axes[1].axvline(_DRIFT_AT - ex.SYNTAX_DRIFT_GATE, color=_gate, lw=0.9, ls=(0, (4, 3)))
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(ANCHORED, a_span, grading_r2, m_op1, m_span):
    _CTRL_M = float(m_span("lam0").mean())
    _headroom = m_span(ex.ORACLE_ARM).mean() - m_span(ex.MEAN_ARM).mean()

    def _row(c: str) -> str:
        gain = m_span(c).mean() - m_span(ex.MEAN_ARM).mean()
        frac = f"{gain / _headroom:.0%}" if c in ex.POOLED_NAMES else "—"
        bold = "<strong>{}</strong>".format if c == ex.PRIMARY else "{}".format
        return (
            f"<tr><th>{bold(f'<code>{c}</code>')}</th>"
            f"<td class='num'>{m_span(c).mean():.3f} <span class='range'>±{m_span(c).std(ddof=1):.3f}</span></td>"
            f"<td class='num'>{m_span(c).mean() - _CTRL_M:.3f}</td>"
            f"<td class='num'>{gain:+.3f}</td>"
            f"<td class='num'>{frac}</td>"
            f"<td class='num'>{a_span(c).mean():.3f} <span class='range'>±{a_span(c).std(ddof=1):.3f}</span></td>"
            f"<td class='num'>{m_op1(c).mean():.3f}</td>"
            f"<td class='num'>{grading_r2(c):.2f}</td></tr>"
        )

    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
      <thead><tr>
        <th>condition</th><th class="num">m<sub>span</sub> ↑</th>
        <th class="num">− control</th><th class="num">− span-mean</th>
        <th class="num">oracle frac</th><th class="num">ᾱ<sub>span</sub> ↓</th>
        <th class="num">m<sub>op1</sub></th><th class="num">r²</th>
      </tr></thead>
      <tbody>{"".join(_row(c) for c in ANCHORED)}</tbody>
    </table></div>
    """
    _caption = f"""
    The τ sweep beside its references, seed means (± seed standard deviation
    where quoted). Only the primary condition (<strong>bold</strong>) is
    gated; "oracle frac" is
    the fraction of the span-mean-to-oracle gap recovered, whose denominator
    is {_headroom:.3f}. m<sub>op1</sub> is carried for continuity with
    ex-2.1.7/8, and r² previews the grading track scored under H3.
    """
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(a_span, cells, m_span, pi_bar):
    _P = ex.PRIMARY
    _gain = m_span(_P).mean() - m_span(ex.MEAN_ARM).mean()
    _drop = a_span(ex.MEAN_ARM).mean() - a_span(_P).mean()
    _headroom = m_span(ex.ORACLE_ARM).mean() - m_span(ex.MEAN_ARM).mean()
    # op1's share of the pull budget, mean over slices; the references by construction.
    _share = {c: pi_bar(c)[:, 0].mean() for c in ex.POOLED_NAMES}

    def _latched(cond: str) -> list[int]:
        """Seeds whose slice-4 softmin weight left op1 (weight < 0.5)."""
        pis = {s: np.asarray(cells[f"{cond}-s{s}"]["pi"], dtype=float) for s in ex.SEEDS}
        return [s for s, pi in pis.items() if pi[4, 0] < 0.5]

    _l30 = _latched(_P)[0]
    mo.md(rf"""
    **H2 is unmet.** On the primary condition `{_P}`, (a) the margin gain is
    {_gain:+.3f} against a gate of {ex.POOL_GAIN_GATE:g} (partial
    {ex.POOL_GAIN_PARTIAL:g}), and (b) the syntax-role drift falls by
    {_drop:.3f} against a required {ex.SYNTAX_DRIFT_GATE:g}. Both moved the
    way the mechanism predicts and neither reaches its gate, so neither
    partial applies.

    Three things the gate arithmetic hides:

    1. **The margin headroom collapsed before the sweep began.** The gate was
       sized from the op1 oracle's +{ex.EX217_OP1_GAIN:g} margin gain in
       ex-2.1.7 — measured at op1, under the old schedule. Here, scored as
       $m_\text{{span}}$ at the ex-2.1.8 operating point, the oracle's whole
       advantage over the span mean is {_headroom:.3f}: less than half the
       gate. No condition, oracle included, could have passed H2(a) as
       preregistered. The primary recovered {_gain / _headroom:.0%} of the
       headroom that existed.

    2. **The sweep is ordered by where the pull's budget went, and that is
       non-monotone in τ.** In the dot chart, both statistics walk steadily
       from `span-mean` to the oracle even though τ runs (∞, 0.01, 0.03,
       0.1, and the oracle is no τ at all). The hidden axis is op1's share
       of the pull: 0.25 for the span mean by construction, then
       {_share["pool-t010"]:.2f} → {_share[_P]:.2f} →
       {_share["pool-t100"]:.2f} across the sweep (mean over slices of the
       weight profiles under H4), and 1.0 for the oracle. A sharper τ
       concentrates the budget harder but risks putting it on the wrong
       role: at depth, each seed of the two sharper rungs commits everything
       to a single role, and at τ = 0.01 two seeds of three commit to `+`
       (at τ = 0.03, one; at τ = 0.1, none). The same decomposition holds
       within a condition — `{_P}`'s committed-to-`+` seed (s{_l30}) scores
       $m_\text{{span}}$ {m_span(_P)[_l30]:.2f} and ᾱ_span
       {a_span(_P)[_l30]:.2f} where its op1-keeping seeds score
       ≈{np.mean(np.delete(m_span(_P), _l30)):.2f} and
       ≈{np.mean(np.delete(a_span(_P), _l30)):.2f}. So the finite-τ means
       differ mostly through how many seeds latched, and the sweep's message
       is a latch-rate trend rather than a smooth dose response; with three
       seeds per rung the rates themselves are coarse, so carrying a τ
       forward should rest on the latch mechanism (*Where the weight went*),
       with more seeds at the chosen rung in the next design.

    3. **Why the span mean scores near the oracle.** The role profiles below
       show it: at depth, every span role of the `span-mean` condition
       carries margin (its syntax-role states attend back to op1, so they
       become selective too), and a max over roles only needs one good role.
       Pooling *did* change the shape — margin and drift both withdraw from
       op2 and `=` almost entirely — but op1's own margin is nearly the same
       in every anchored condition, so the max barely moves.
    """)
    return


@app.cell(hide_code=True)
def _(alpha_map):
    _CONDS3 = [ex.MEAN_ARM, ex.PRIMARY, ex.ORACLE_ARM]
    _prof = {}
    for _c in _CONDS3:
        _a = alpha_map(_c)
        _m = np.einsum("c,lct->lt", ex.LABEL_W, _a) - _a.mean(axis=1)
        _prof[_c] = {"margin": _m[:, : ex.SPAN], "drift": _a[:, :, : ex.SPAN].mean(axis=1)}

    @themed(
        name="role-profiles",
        alt_text="""
            Six panels in a two-by-three grid: columns are span-mean, pool-t030
            and op1-oracle; the top row plots margin by role against slice, the
            bottom row drift by role, both from about -0.1 to 0.9. Four
            marked lines per panel, one per span role. Top left: all
            four roles rise together to between 0.5 and 0.8 after the
            embedding. Top middle and top right: the op1 line alone stays high,
            about 0.8 at slice 1 falling to 0.4 (middle) or 0.5 (right) at slice 4, while
            the plus, op2 and equals lines sit below 0.3. Bottom left: plus and
            equals start near 0.55 at the embedding and fall with depth while
            op2 rises to about 0.28. Bottom middle: the plus line starts at
            0.49 and decays to near zero; op2 and equals stay low. Bottom
            right: the same shape with plus starting lower, at 0.3.
        """,
        caption=r"""
            **Where the margin and the drift live.** Margin (top) and drift
            (bottom) by span role, per slice, seed means: $m(\ell, t)$ and the
            unweighted mean alignment $\bar\alpha(\ell, t)$. Columns:
            the span mean, the primary, and the op1 oracle. The scored
            statistics take the per-slice max over these lines
            ($m_\text{span}$ from the top row, ᾱ_span from the bottom), so a
            single high line is all either statistic sees.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesGrid

        _role_ink = {
            "op1": light_dark("#c1332a", "#f0665a"),
            "+": light_dark("#1f6fb4", "#5fa8dd"),
            "op2": light_dark("#2a9d8f", "#5fc9bd"),
            "=": light_dark("#7b3fa0", "#b98ce0"),
        }
        _grey = light_dark("#666", "#999")
        fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.2), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesGrid, axes)
        _x = np.arange(5)
        for row, quantity in zip(axes, ("margin", "drift"), strict=True):
            for ax, cond in zip(row, _CONDS3, strict=True):
                ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
                for ti, ink in enumerate(_role_ink.values()):
                    ax.plot(_x, _prof[cond][quantity][:, ti], color=ink, lw=1.5, marker="o", ms=3, mew=0)
                if quantity == "margin":
                    ax.set_title("pool-t030 (primary)" if cond == ex.PRIMARY else cond, fontsize=9.5)
                ax.set_ylim(-0.12, 0.92)
                ax.tick_params(labelsize=8)
        for ax in axes[1]:
            ax.set_xticks(_x, ["emb", "1", "2", "3", "4"])
            ax.set_xlabel("slice", fontsize="x-small")
        axes[0][0].set_ylabel("margin", fontsize="small")
        axes[1][0].set_ylabel("drift", fontsize="small")
        _leg = [plt.Line2D([], [], color=ink, lw=1.5) for ink in _role_ink.values()]
        axes[0][2].legend(_leg, list(_role_ink), fontsize=8, frameon=False, loc="upper right", labelcolor=_grey)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(alpha_map):
    _CONDS3 = [ex.MEAN_ARM, ex.PRIMARY, ex.ORACLE_ARM]
    _prof = {}
    for _c in _CONDS3:
        _a = alpha_map(_c)
        _m = np.einsum("c,lct->lt", ex.LABEL_W, _a) - _a.mean(axis=1)
        _prof[_c] = {"margin": _m[:, : ex.SPAN], "drift": _a[:, :, : ex.SPAN].mean(axis=1)}
    _x = np.arange(ex.SPAN)

    @themed(
        name="position-profiles",
        alt_text="""
            A five-by-three grid of small flush panels: the same margin and
            drift data as the previous figure, transposed. Columns are
            span-mean, pool-t030 and op1-oracle; rows are residual slices
            with the last slice on top and the embedding at the bottom; each
            panel spans the four span roles op1, plus, op2 and equals, from 0
            to about 0.95 vertically. A shaded smooth-step area shows the
            margin and a solid line the drift. In the span-mean column every
            row above the embedding is a wide raised block: margin between
            0.5 and 0.8 at all four roles. In the pool-t030 and oracle
            columns the block narrows to a single tall step at op1, with the
            other roles below 0.3. The drift line roughly shadows the margin
            at about half its height in every panel; on the embedding row it
            is a step at plus and equals for span-mean, at plus alone for
            pool-t030, and lower, about 0.3 at plus, for the oracle.
        """,
        caption=r"""
            **The same profiles, position-major.** Margin $m(\ell, t)$
            (shaded area) and drift $\bar\alpha(\ell, t)$ (line) over the
            four span roles, one panel per residual slice with the embedding
            at the bottom — the transpose of the figure above, in the layout
            the weight profiles under H4 use. Columns: the span mean, the
            primary, and the op1 oracle. Roles are ordinal, so the risers
            carry no data.
        """,
    )
    def _plot() -> plt.Figure:
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        from mini.vis import smooth_step, smooth_step_area

        _margin_ink = light_dark("#b4531f", "#dd8f5f")
        _drift_ink = light_dark("#1f6fb4", "#5fa8dd")
        _grey = light_dark("#666", "#999")
        fig, axes = plt.subplots(5, 3, figsize=(7.0, 3.7), sharex=True, sharey=True)
        _engine = fig.get_layout_engine()
        assert isinstance(_engine, ConstrainedLayoutEngine)
        _engine.set(hspace=0, h_pad=0.01, wspace=0.05)
        for col, cond in enumerate(_CONDS3):
            for row in range(5):
                sl = 4 - row  # last slice on top, the embedding at the bottom
                ax = axes[row][col]
                smooth_step_area(ax, _x, _prof[cond]["margin"][sl], ramp=0.5, color=_margin_ink,
                                 alpha=light_dark(0.25, 0.3))  # fmt: skip
                smooth_step(ax, _x, _prof[cond]["margin"][sl], ramp=0.5, color=_margin_ink, lw=1.2)
                smooth_step(ax, _x, _prof[cond]["drift"][sl], ramp=0.5, color=_drift_ink, lw=1.5)
                ax.set(ylim=(-0.08, 0.98), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0.0, 0.4, 0.8])
                ax.spines[:].set_visible(False)
                ax.grid(axis="y", which="major", c="#888", alpha=0.2)
                ax.tick_params(axis="x", length=0, labelbottom=False)
                ax.tick_params(axis="y", left=True, right=True, direction="in", labelleft=False, labelright=False)
                if col == 0:
                    ax.set_ylabel("emb" if sl == 0 else f"{sl}", fontsize=8)
            axes[0][col].set_title("pool-t030 (primary)" if cond == ex.PRIMARY else cond, fontsize=9.5)
            axes[4][col].set_xticks(_x, ex.ROLES)
            axes[4][col].tick_params(axis="x", labelbottom=True, labelsize=8.5)
        axes[4][2].tick_params(axis="y", labelright=True, labelsize=7, pad=2)
        fig.supylabel("slice", fontsize=9)
        _m_key = plt.Rectangle((0, 0), 1, 1, fc=_margin_ink, alpha=0.35, ec="none")
        _d_key = plt.Line2D([], [], color=_drift_ink, lw=1.5)
        fig.legend([_m_key, _d_key], [r"margin $m(\ell, t)$", r"drift $\bar\alpha(\ell, t)$"],
                   fontsize=8.5, frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   labelcolor=_grey)  # fmt: skip
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(alpha_map):
    _emb = {c: alpha_map(c)[0, :, : ex.SPAN].mean(axis=0) for c in [ex.MEAN_ARM, ex.PRIMARY, ex.ORACLE_ARM]}
    mo.md(rf"""
    The drift that pooling leaves behind has an address: the `+` embedding.
    On the embedding row, the primary's pooling released `=` almost entirely
    (drift {_emb[ex.MEAN_ARM][3]:.2f} under the span mean →
    {_emb[ex.PRIMARY][3]:.2f}) but left {_emb[ex.PRIMARY][1]:.2f} on `+`,
    barely below the span mean's {_emb[ex.MEAN_ARM][1]:.2f} — consistent
    with the ~0.2 of softmin weight the pull still spends there (H4 below).
    The op1 oracle complicates that account: it never pulls `+` at any
    slice, and its op1 states cannot even read the `+` embedding (position 0
    attends only to itself), yet its `+` embedding still ends at
    {_emb[ex.ORACLE_ARM][1]:.2f}. So about half of the `+` drift is paid for
    by something other than the pull — and whatever pays for it does so
    against the repulsion, which runs from the first epoch and still holds
    at the end. The trajectory records no per-role drift, so this run can't
    say whether that alignment climbs early (against the repulsion's peak)
    or late (as the anneal decays); the candidate mechanism and the timing
    are both queued as exploratory questions rather than interpreted here.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The operating point under pooling (H3)

    The endpoint numbers say where each run finished; the trajectories say
    how it got there — in particular whether a pull that concentrated early
    slid once the repulsion annealed away, which was the contrary outcome H3
    named.
    """)
    return


@app.cell(hide_code=True)
def _(ANCHORED, retention, traj):
    _KEYS = ("alpha_op1", "m_span")
    # Seed means once, in the cell: @themed calls the plot function twice.
    _MEAN = {c: {k: np.mean([traj(c, s, k) for s in ex.SEEDS], axis=0) for k in _KEYS} for c in [*ANCHORED, "lam0"]}
    _EPOCHS = np.mean([traj(ex.PRIMARY, s, "epoch") for s in ex.SEEDS], axis=0)
    _RET = {c: retention(c) for c in ANCHORED}
    # The schedules on a finer grid than the recorded trajectory. All anchored
    # conditions share them at the operating point, so one panel serves five.
    _SE = np.linspace(0.0, float(ex.SCHEDULER.epochs), 601)
    _ANCHOR = ex.anchor_weight(_SE)
    _ANTI = ex.anti_subspace_weight(_SE, anneal_end=ex.ANTI_ANNEAL_END, hold_ratio=ex.ANTI_HOLD_RATIO)
    _GRID = [["span-mean", "pool-t010", "pool-t030"], ["pool-t100", "op1-oracle", "sched"]]
    _LEFT = ("span-mean", "pool-t100")

    @themed(
        name="trajectories",
        alt_text="""
            Five trajectory panels and one schedule panel in a two-by-three
            grid sharing the epoch axis from 0 to 100. Each trajectory panel
            plots mean alignment at op1 as a solid line staying below 0.15,
            and the span margin as a dashed line climbing to between 0.6 and
            0.75 by epoch 40 and holding roughly level to the end, over a flat
            grey control pair near zero. The dashed line dips slightly after
            epoch 90 in every panel. The schedule panel, bottom right, shows
            the anchor weight ramping up to a plateau and annealing after
            epoch 90, and the repulsion weight starting high, crossing below
            the anchor around epoch 60 and settling at about a third of it.
        """,
        caption=rf"""
            **Training dynamics by condition.** Seed means of the two cosines
            on the anchor axis, on one shared scale: solid is $\bar\alpha$ at
            op1 (contained at the left caret, {ex.MEAN_ALIGN_GATE:g}); dashed
            is $m_\text{{span}}$ (its retention floor of
            {ex.RETENTION_FLOOR:g} at the right caret). The grey pair is the
            un-anchored control, and the number in each panel is that
            condition's retention (final over peak $m_\text{{span}}$, minimum
            across seeds; gate ≥ {ex.RETENTION_GATE:g}). Bottom right: the
            weight schedule every anchored condition trained under — anchor
            $\lambda_\mathrm{{a}}$ in red, repulsion
            $\lambda_{{\bar{{\mathrm{{s}}}}}}$ in blue, log scale — shared
            here because the anti-subspace term is fixed at the ex-2.1.8
            operating point in every anchored condition.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axd = plt.subplot_mosaic(_GRID, figsize=(7.5, 4.2), sharex=True, layout="constrained")
        _grey = light_dark("#999", "#777")
        _gate = light_dark("#555", "#bbb")
        _ink = light_dark("#1f6fb4", "#5fa8dd")
        _term = {"anchor": light_dark("#c33", "#e66"), "anti": light_dark("#36c", "#7af")}
        _dash = (0, (6, 1))
        _top = max(np.max(_MEAN[c]["m_span"]) for c in ANCHORED)
        for cond in ANCHORED:
            ax = axd[cond]
            ax.plot(0, ex.MEAN_ALIGN_GATE, marker=9, ms=3, color=_gate, clip_on=False, zorder=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
            ax.plot(1, ex.RETENTION_FLOOR, marker=8, ms=3, color=_gate, clip_on=False, zorder=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
            ax.plot(_EPOCHS, _MEAN["lam0"]["alpha_op1"], color=_grey, lw=1.1, zorder=3)
            ax.plot(_EPOCHS, _MEAN["lam0"]["m_span"], color=_grey, lw=1.1, ls=_dash, zorder=3)
            ax.plot(_EPOCHS, _MEAN[cond]["m_span"], color=_ink, lw=1.4, ls=_dash, zorder=4)
            ax.plot(_EPOCHS, _MEAN[cond]["alpha_op1"], color=_ink, lw=1.7, zorder=5)
            _r = _RET[cond]
            assert _r is not None
            ax.annotate(f"{_r:.2f}", (0.97, 0.6), xycoords="axes fraction", ha="right", fontsize=8,
                        color=light_dark("#444", "#bbb"))  # fmt: skip
            ax.set_title("pool-t030 (primary)" if cond == ex.PRIMARY else cond, fontsize=9.5, color=_ink)
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
        for _x, _y, _txt, _k in ((64, 0.13, r"$\lambda_\mathrm{a}$", "anchor"), (16, 0.5, r"$\lambda_{\bar{\mathrm{s}}}$", "anti")):  # fmt: skip
            _ax.annotate(_txt, (_x, _y), fontsize=9, color=_term[_k], va="bottom")
        fig.legend(
            [plt.Line2D([], [], color=_grey, lw=1.7), plt.Line2D([], [], color=_grey, lw=1.5, ls=_dash)],
            [r"$\bar\alpha$ at op1", r"$m_{\text{span}}$"],
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
    **H3(a) and (b) hold.** On the primary condition, $\bar\alpha$ at op1 ends at
    {a_op1(_P).mean():.3f} <span class='range'>±{a_op1(_P).std(ddof=1):.3f}</span>
    against the {ex.MEAN_ALIGN_GATE:g} gate, and retention is {_ret:.2f}
    against {ex.RETENTION_GATE:g}. The contrary outcome — a concentrated pull
    sliding once the anti anneal ends at epoch {ex.ANTI_ANNEAL_END:g} — shows
    up only as the small end-of-training dip every condition shares (the
    anchor's own anneal), and the worst retention anywhere in the sweep is
    {_worst:.2f}. Pooling redirects the budget of the pull without
    destabilizing what ex-2.1.8 found.
    """)
    return


@app.cell(hide_code=True)
def _(ANCHORED, alpha_map, grading_r2, m_span):
    _ALL = ["lam0", *ANCHORED]
    _RESP = {c: alpha_map(c)[:, :, 0].mean(axis=0) for c in _ALL}

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
    _GRID = [_ALL[:3], _ALL[3:]]

    @themed(
        name="grading",
        alt_text="""
            Six panels in a two-by-three grid sharing both axes: per-color
            alignment at the first operand, from about -0.2 to 1.2, against
            the redness of that color from 0 to 1. Panels are the control,
            span-mean, and the three pooled conditions and the oracle. The control
            panel is flat near zero. Every other panel scatters 216 marks
            drawn in the color each stands for, forming a flat cluster at low
            redness and a rising tail of oranges and reds, with a thin
            conditional-mean line and a dashed reference curve through it.
            The flat cluster sits slightly lower and the rise slightly
            straighter in the pooled and oracle panels than in span-mean; in
            the pool-t010 panel the rise has a visible shelf between redness
            0.7 and 0.8.
        """,
        caption=r"""
            **Grading: alignment at op1, per color.** $\alpha_c$, the mean
            over slices of $\cos(h, \hat v_{\text{red}})$ at op1, seed mean,
            against the redness of the color. One mark per color, drawn in
            that color; the thin dark line is the mean response at each of
            the 29 distinct redness levels, the flat grey band the same line
            for the control, and the dashed line `sim¹·⁵` rescaled onto the
            response by least squares — the shape the $r^2$ statistic scores
            against. H3(c) gates the primary's $r^2$ relative to
            `span-mean`'s.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesGrid

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
                ax.set_title("pool-t030 (primary)" if cond == ex.PRIMARY else ("control" if cond == "lam0" else cond),
                             fontsize=9.5)  # fmt: skip
                if cond != "lam0":
                    ax.annotate(
                        f"r² = {grading_r2(cond):.2f}\n$m_\\mathrm{{span}}$ = {m_span(cond).mean():.2f}",
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
    **H3(c) holds, with the sign reversed.** The gate would have tolerated $r^2$
    on the primary falling {ex.GRADE_R2_DROP:g} below the span mean's;
    instead it lands {grading_r2(_P) - grading_r2(ex.MEAN_ARM):+.2f} *above*
    it ({grading_r2(_P):.2f} against {grading_r2(ex.MEAN_ARM):.2f}). The
    contrary expectation — that concentrating the pull would sharpen the
    response into a step — mostly has it backwards: every pooled condition
    grades at or above the span mean, and the oracle best of all
    ({grading_r2(ex.ORACLE_ARM):.2f}). Withdrawing the pull from positions
    that don't carry the concept apparently cleans the response shape at the
    one that does. The exception is visible in the `pool-t010` panel: a
    shelf in the rise near redness 0.8, which turns out to belong to the
    latched seeds (*Exploratory analyses*). As a footnote to ex-2.1.8's open
    question about where a realistic absolute grading bar sits: the primary
    and the oracle both clear the 0.8 that experiment couldn't reach.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where the weight went (H4)

    The weight profile is the pull's own account of where it spent each
    line's budget; the control's readability profile is the reference for
    where the concept actually lives. H4 asks whether they agree — at the
    embedding, where only op1 can carry the concept, and across depth, where
    the syntax roles pick it up through attention.
    """)
    return


@app.cell(hide_code=True)
def _(READ: np.ndarray, cells, pi_bar, read_gain):
    _PI = {c: pi_bar(c) for c in ex.POOLED_NAMES}
    _PI_SEEDS = {
        c: np.array([np.asarray(cells[f"{c}-s{s}"]["pi"], dtype=float) for s in ex.SEEDS]) for c in ex.POOLED_NAMES
    }
    _GAIN = {c: read_gain(c) for c in ex.POOLED_NAMES}
    _x = np.arange(ex.SPAN)

    @themed(
        name="weight-profiles",
        alt_text="""
            A five-by-three grid of small flush panels. Columns are the pooled
            conditions at tau 0.01, 0.03 and 0.1; rows are residual slices with the
            last slice on top and the embedding at the bottom; each panel
            spans the four span roles op1, plus, op2 and equals on its
            horizontal axis, from 0 to 1 vertically. A shaded smooth-step
            area shows the seed-mean softmin weight, three hairlines the
            individual seeds, and a dashed line the control's
            readability, which is identical across the columns of a row: op1
            near 0.85 everywhere, plus and equals near 0.8 on all rows but
            the embedding, where both drop to zero, and op2 never above 0.2.
            In the bottom (embedding) row of every column, the weight area is
            a tall block at op1, about 0.77, with a small step at plus, and
            the hairlines hug the mean. On the deeper rows of the tau 0.01
            and 0.03 columns the hairlines separate completely: each seed's
            profile is a full-height block at a single role — plus in two
            seeds of three at tau 0.01, one of three at 0.03 — so the mean
            area, op1 at about a third or two thirds with plus taking the
            rest, is a mixture of one-hot seeds. In the tau 0.1 column op1
            grows toward 0.95 with the other roles near zero and the
            hairlines stay together.
        """,
        caption=rf"""
            **What the pull chose, against where the concept is readable.**
            The trained softmin weight profile of each pooled condition
            (shaded area, seed mean $\bar\pi(\ell, t)$; hairlines, the three
            seeds individually), over the four span roles, one panel per
            residual slice with the embedding at the bottom. Where the
            hairlines separate — the deep slices at τ ≤ 0.03 — each seed's
            profile is one-hot on a single role, and the mean is a mixture
            of seeds rather than a shape any run has. The dashed line
            is the readability profile of the *un-anchored control*,
            $R^2(\ell, t)$ — a ridge probe predicting op1's redness from the
            state at each (slice, role), one color held out at a time — the
            same reference in every column of a row (its own seed spread is
            ≤ 0.09 everywhere except op2 at depth, ≈ 0.2). The caret on the
            embedding row marks the H4(a) gate: op1's weight ≥
            {ex.LEAD_GATE:g} (uniform is 0.25). The number atop each column
            is its H4(b) weighted readability gain (gate ≥
            {ex.READ_GAIN_GATE:g}, scored on the primary). Roles are
            ordinal, so the risers carry no data.
        """,
    )
    def _plot() -> plt.Figure:
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        from mini.vis import smooth_step, smooth_step_area

        _cond_ink = {
            "pool-t010": light_dark("#7fb3d8", "#4f7ea6"),
            "pool-t030": light_dark("#1f6fb4", "#5fa8dd"),
            "pool-t100": light_dark("#12497c", "#8ec4ea"),
        }
        _grey = light_dark("#666", "#999")
        _read_ink = light_dark("#555", "#aaa")
        _hair = light_dark("#00000055", "#ffffff55")
        _gate = light_dark("#555", "#bbb")
        fig, axes = plt.subplots(5, 3, figsize=(7.0, 3.7), sharex=True, sharey=True)
        _engine = fig.get_layout_engine()
        assert isinstance(_engine, ConstrainedLayoutEngine)
        _engine.set(hspace=0, h_pad=0.01, wspace=0.05)
        for col, cond in enumerate(ex.POOLED_NAMES):
            ink = _cond_ink[cond]
            for row in range(5):
                sl = 4 - row  # last slice on top, the embedding at the bottom
                ax = axes[row][col]
                smooth_step_area(ax, _x, _PI[cond][sl], ramp=0.5, color=ink, alpha=light_dark(0.22, 0.28))
                smooth_step(ax, _x, _PI[cond][sl], ramp=0.5, color=ink, lw=1.5)
                for seed_prof in _PI_SEEDS[cond][:, sl]:
                    smooth_step(ax, _x, seed_prof, ramp=0.5, color=_hair, lw=0.5)
                smooth_step(ax, _x, READ[sl], ramp=0.5, color=_read_ink, lw=1.0, ls=(0, (4, 2)))
                ax.set(ylim=(-0.08, 1.08), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0.0, 0.5, 1.0])
                ax.spines[:].set_visible(False)
                ax.grid(axis="y", which="major", c="#888", alpha=0.2)
                ax.tick_params(axis="x", length=0, labelbottom=False)
                ax.tick_params(axis="y", left=True, right=True, direction="in", labelleft=False, labelright=False)
                if col == 0:
                    ax.set_ylabel("emb" if sl == 0 else f"{sl}", fontsize=8)
            axes[0][col].set_title("pool-t030 (primary)" if cond == ex.PRIMARY else cond, fontsize=9.5, color=ink,
                                   fontweight="bold" if cond == ex.PRIMARY else "normal")  # fmt: skip
            axes[0][col].annotate(f"gain {_GAIN[cond]:+.2f}", (0.97, 0.88), xycoords="axes fraction", ha="right",
                                  va="top", fontsize=8, color=light_dark("#444", "#bbb"))  # fmt: skip
            # The H4(a) gate reads exactly here: op1's weight at the embedding.
            axes[4][col].plot(0, ex.LEAD_GATE, marker=9, ms=3, color=_gate, clip_on=False, zorder=6,
                              transform=axes[4][col].get_yaxis_transform())  # fmt: skip
            axes[4][col].set_xticks(_x, ex.ROLES)
            axes[4][col].tick_params(axis="x", labelbottom=True, labelsize=8.5)
        # One panel says what the scale is; numbering all fifteen spends ink on a fact stated once.
        axes[4][2].tick_params(axis="y", labelright=True, labelsize=7, pad=2)
        fig.supylabel("slice", fontsize=9)
        _pi_key = plt.Rectangle((0, 0), 1, 1, fc=_grey, alpha=0.35, ec="none")
        _read_key = plt.Line2D([], [], color=_read_ink, lw=1.0, ls=(0, (4, 2)))
        fig.legend([_pi_key, _read_key], [r"softmin weight $\bar\pi$", "control readability $R^2$"],
                   fontsize=8.5, frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   labelcolor=_grey)  # fmt: skip
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(READ: np.ndarray, cells, pi_bar, read_gain):
    _P = ex.PRIMARY
    _emb = {c: pi_bar(c)[0] for c in ex.POOLED_NAMES}
    # Deep-slice choices per seed: which role holds the slice-4 weight in each run.
    _s4 = {
        c: [int(np.argmax(np.asarray(cells[f"{c}-s{s}"]["pi"], dtype=float)[4])) for s in ex.SEEDS]
        for c in ex.POOLED_NAMES
    }
    _n_plus = {c: sum(1 for r in roles if r == 1) for c, roles in _s4.items()}
    mo.md(rf"""
    **H4(a) holds.** At the embedding, the primary's leading role is op1
    with weight {_emb[_P][0]:.2f} against the {ex.LEAD_GATE:g} gate — and
    the same is true at every τ ({_emb["pool-t010"][0]:.2f} at 0.01,
    {_emb["pool-t100"][0]:.2f} at 0.1). Neither contrary outcome occurred
    where the gate reads: no condition latched the embedding onto `+` or `=`
    (readability ≈ 0 there), and none stayed uniform. The trained weights
    concentrate harder than the 0.6–0.8 calibration band predicted from the
    reference profiles, which is the softmin's self-reinforcement doing what
    the introduction said it would — just in the right place.

    **H4(b) holds.** The primary's weighted readability gain is
    {read_gain(_P):+.2f}
    against the {ex.READ_GAIN_GATE:g} gate (pool-t010
    {read_gain("pool-t010"):+.2f}, pool-t100 {read_gain("pool-t100"):+.2f}).
    Most of the gain comes from avoiding op2 — the one span role that stays
    hard to read at depth ($R^2$ {READ[1:, 2].mean():.2f} averaged over the
    four deep slices, against {READ[1:, [0, 1, 3]].mean():.2f} for the other
    three roles).

    Across depth the conditions diverge, and seed by seed the divergence is
    all-or-nothing: the hairlines show each run of the two sharper rungs
    committing its entire deep-slice weight to a single role. At τ = 0.01
    that role is `+` in {_n_plus["pool-t010"]} seeds of three and op1 in the
    other; at τ = 0.03, `+` in {_n_plus["pool-t030"]}; at τ = 0.1 none
    commit to `+`, and the profile is the sweep's only genuinely graded one
    at depth (op1 at 0.87–0.97 per seed, with real weight left elsewhere).
    So the seed-mean's apparent `+` minority at τ = 0.03 is a mixture of
    one-hot seeds, and the τ trend is a latch-rate trend: 2/3 → 1/3 → 0/3.
    A `+` latch is a cheap early choice rather than a ruinous one, since `+`
    at depth reads op1's redness at $R^2$ ≈ {READ[1:, 1].mean():.2f}. The
    prereg expected migration *toward* the syntax roles with depth in step
    with readability; what actually distinguishes the healthy runs is that
    they keep op1 — which stays the *most* readable role at every slice —
    and the pull only leaves it where τ is sharp enough to lock in an early
    accident.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Nothing here is preregistered; everything in this section is post hoc.
    """)
    return


@app.cell(hide_code=True)
def _(traj):
    # Post hoc. Per-seed trajectories: the seed mean at depth would average
    # one-hot runs into a shape no run has, so every run is drawn.
    _PI_T = {c: [traj(c, s, "pi") for s in ex.SEEDS] for c in ex.POOLED_NAMES}
    _EPOCHS = np.mean([traj(ex.PRIMARY, s, "epoch") for s in ex.SEEDS], axis=0)

    @themed(
        name="concentration-timing",
        alt_text="""
            Two line panels sharing an epoch axis from 0 to 100, each with
            nine thin curves — one per run, three per pooled condition at tau
            0.01, 0.03 and 0.1 — and a horizontal reference line at 0.25 for
            uniform weights. Left panel, op1's weight at the embedding: the
            nine curves scatter over 0 to 0.8 for the first few epochs, then
            converge to about 0.8 by epoch 20 and stay flat to the end.
            Right panel, op1's
            weight at the last slice: every curve saturates by epoch 8 and
            never moves again. Two of the three tau 0.01 curves fall to 0 by
            epoch 3 and the third rises to 1; one tau 0.03 curve falls to 0
            while the other two rise to 1; all three tau 0.1 curves rise to
            about 0.9 and hold there, short of saturation.
        """,
        caption=r"""
            **When the pull decides.** op1's share of the softmin weight
            over training, at the embedding (**left**) and at the last slice
            (**right**), one line per run; the thin rule is uniform (0.25).
            The vertical band is the anchor's warmup window (epochs 0–10);
            the repulsion anneal runs to epoch 90.
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesRow

        _ink = {
            "pool-t010": light_dark("#7fb3d8", "#4f7ea6"),
            "pool-t030": light_dark("#1f6fb4", "#5fa8dd"),
            "pool-t100": light_dark("#12497c", "#8ec4ea"),
        }
        _grey = light_dark("#666", "#999")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, sl, title in zip(axes, (0, 4), ("op1 weight at the embedding", "op1 weight at slice 4"), strict=True):
            ax.axvspan(0, ex.SCHEDULER.warmup_epochs, color=_grey, alpha=0.12, lw=0)
            ax.axhline(1 / ex.SPAN, color=_grey, lw=0.6, alpha=0.5, zorder=0)
            for cond, ink in _ink.items():
                for si, pi_t in enumerate(_PI_T[cond]):
                    ax.plot(_EPOCHS, pi_t[:, sl, 0], color=ink, lw=1.1, alpha=0.9,
                            label=cond if si == 0 else "_")  # fmt: skip
            ax.set_title(title, fontsize="small", color=_grey)
            ax.set_xlim(0, ex.SCHEDULER.epochs)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("epoch", fontsize="x-small")
            ax.tick_params(labelsize=8)
        axes[0].set_ylabel(r"$\pi_\mathrm{op1}$", fontsize="x-small")
        axes[1].legend(fontsize=8, frameon=False, loc="center right", labelcolor=_grey)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The choice is made during the anchor ramp, run by run. Every run's
    embedding weight settles by about epoch 20, and every run's deep-slice
    choice is committed by epoch 8 — the τ = 0.01 latches by epoch 2–3,
    before the anchor has even finished warming up — and no choice changes
    for the rest of training. Nor is there much reason it should: once the
    softmin has concentrated, the losing roles receive no pull, and late
    training supplies no perturbation of the needed size — the learning
    rate is decaying, the regularizer weights are annealing, and 100 epochs
    is generous for this corpus, so the task loss has long converged. Two
    consequences: the long repulsion anneal that ex-2.1.8 tuned matters for
    containment rather than for position choice (whatever the repulsion
    contributes to the choice, it contributes in the first few epochs,
    before the softmin closes), and any intervention on the choice — a τ
    schedule that starts soft and sharpens, say — has to land in that same
    window. The latch, then, is a race at initialization: whichever role
    happens to align cheapest in the first few hundred steps keeps the
    weight, which is why the winner-take-most rung latches most often.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, cells):
    # Post hoc: the H3(c) scatter's pool-t010 shelf, split by each run's
    # deep-slice choice.
    _levels = np.unique(np.round(ex.REDNESS, 6))

    def _level_mean(y: np.ndarray) -> np.ndarray:
        return np.array([y[np.round(ex.REDNESS, 6) == lv].mean() for lv in _levels])

    def _resp(cond: str, s: int) -> np.ndarray:
        """One run's per-color op1 response: the mean over slices of alignment at op1."""
        return arrays[f"{cond}-s{s}/alpha"][:, :, 0].mean(axis=0)

    def _kept_op1(cond: str, s: int) -> bool:
        return bool(np.asarray(cells[f"{cond}-s{s}"]["pi"], dtype=float)[4, 0] >= 0.5)

    @themed(
        name="grading-by-seed",
        alt_text="""
            Three line panels, one per pooled condition at tau 0.01, 0.03 and
            0.1, sharing axes: mean response against redness of op1 from 0 to
            1, three lines per panel, one per run, colored red if the run's
            deep slices kept op1 and blue if they latched onto plus. Red
            lines rise smoothly from near 0 at low redness to about 0.9 at
            redness 1 in every panel. Blue lines — two in the tau 0.01 panel,
            one in the tau 0.03 panel, none at tau 0.1 — track near zero all
            the way to redness 0.72, then jump to about 0.5 at redness 0.8
            and 0.7 by redness 1: a step where the red lines have a slope.
        """,
        caption=r"""
            **The shelf, split by each run's deep-slice choice.** The
            grading response of every pooled run — the mean response at each
            of the 29 redness levels, as in the H3(c) figure but one line
            per run — colored by where that run's slice-4 softmin weight
            ended: op1 kept (red) or latched onto `+` (blue).
        """,
    )
    def _plot() -> plt.Figure:
        from typing import cast

        from mini.vis import AxesRow

        _ink = {True: light_dark("#c1332a", "#f0665a"), False: light_dark("#1f6fb4", "#5fa8dd")}
        _grey = light_dark("#666", "#999")
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, cond in zip(axes, ex.POOLED_NAMES, strict=True):
            ax.axhline(0, color=light_dark("#bbb", "#555"), lw=0.8, zorder=0)
            for s in ex.SEEDS:
                ax.plot(_levels, _level_mean(_resp(cond, s)), color=_ink[_kept_op1(cond, s)], lw=1.3)
            ax.set_title("pool-t030 (primary)" if cond == ex.PRIMARY else cond, fontsize=9.5)
            ax.set_xlabel("redness of op1", fontsize="x-small")
            ax.tick_params(labelsize=8)
        axes[0].set_ylabel(r"$\alpha_c$ at op1", fontsize="x-small")
        _keys = [plt.Line2D([], [], color=_ink[k], lw=1.3) for k in (True, False)]
        axes[2].legend(_keys, ["deep pull kept op1", "latched onto +"], fontsize=8, frameon=False,
                       loc="upper left", labelcolor=_grey)  # fmt: skip
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, cells):
    def _resp_at(cond: str, s: int, level: float) -> float:
        sel = np.round(ex.REDNESS, 2) == level
        return float(arrays[f"{cond}-s{s}/alpha"][:, sel, 0].mean())

    _latched = [s for s in ex.SEEDS if np.asarray(cells[f"pool-t010-s{s}"]["pi"], dtype=float)[4, 0] < 0.5]
    _kept = [s for s in ex.SEEDS if s not in _latched][0]
    _hi = np.mean([_resp_at("pool-t010", s, 0.8) for s in _latched])
    _lo = np.mean([_resp_at("pool-t010", s, 0.72) for s in _latched])
    mo.md(rf"""
    The shelf belongs to the latched runs. In the runs whose deep slices
    kept op1, the response grades smoothly with redness at every τ. In the
    runs that latched onto `+`, it holds up only at the red end and drops
    sharply below redness 0.8 — `pool-t010`'s two latched runs average
    {_hi:.2f} at redness 0.8 and {_lo:.2f} at 0.72, where its op1-keeping
    run reads {_resp_at("pool-t010", _kept, 0.72):.2f} — which is also where
    the label affinity falls away steeply. That reading makes sense of the
    mechanism: in a latched run only the embedding slice still pulls op1, so
    a color keeps its alignment only to the extent that its lines keep
    getting labeled, and the graded contribution the deep slices add in
    healthy runs is missing. The seed-mean scatter under H3(c) blends the
    two shapes, which is what the `pool-t010` panel's shelf is.

    Still queued, from the prereg's candidate list and this round: per-color
    weight maps (do strongly-red lines concentrate differently from
    weakly-red ones?); the source and timing of the `+` embedding drift that
    persists even under the op1 oracle (noted under H2 — the trajectory
    should record per-role drift next time so the timing is answerable); a τ
    schedule that starts soft and sharpens as a latch remedy; and more seeds
    at the chosen rung, since three per condition makes a latch rate a
    coarse estimate.
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
