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
    Ex-2.1.9's pooled pull found the concept's position unaided — but every
    label still keyed on op1, so only one position ever carried the evidence.
    Now a line is labeled when *either* operand draws, with the per-slot rate
    halved to keep the label budget where it was, and the pull has to find
    the red operand line by line. Two soft-τ arms ride along.
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
    labels still pointed. A line drew a label as a function of its *first*
    operand's redness, so op1 was the only position whose evidence ever
    justified a label, and a pull that settled on op1 globally could never be
    wrong.

    A document-level label on natural text — "this review is negative", "this
    tweet is obscene" — says the concept occurs *somewhere* and nothing else.
    This experiment widens our labels by one step in that direction: either
    operand can draw the label, so a line can be labeled because of a red op2
    while its op1 is blue. The pull then has to do per-line work: on
    op1-triggered lines the red state is at position 0, on op2-triggered
    lines at position 2, and a pool that commits to one position for the whole
    run will pull blue states onto the axis on the lines it got wrong.

    Whether that failure would even show up in the scores is not obvious,
    because of a mechanism ex-2.1.9 already ran into: the span positions share token
    embeddings. At the embedding slice a color's state is its token embedding
    wherever the token appears (RoPE adds no positional component at depth 0),
    so once red embeddings sit on the axis, a red op2 line has an aligned
    position *for free* — the labeller's slot-symmetry is satisfied by the
    lexicon before any per-line choice happens. The position-aware slices are
    where the question is real: there, ex-2.1.9 found the pool's deep-slice
    choice is winner-take-all *per run*, committed by epoch 8. If that
    winner is global, the two label groups will show the same weight profile;
    if the pool localizes per line, a red op2 should carry the weight on the
    lines where it is the evidence. That contrast is the discriminator this
    design gates on (H2), and the selectivity delivered to labeled lines
    (H4) is scored separately, since the embedding give-away means the second
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
    mo.md(
        r"""
    ## Method

    Everything follows ex-2.1.9 except the labelling rule: the word-level
    `v216` corpus from ex-2.1.3 (216 colors, one token each, d64-L4), the same
    seeds, the mellowmax-pooled anchor term, the anchor schedule, and the
    anti-subspace schedule fixed at ex-2.1.8's operating point
    (`end90-hold30`) in every anchored cell.

    ### The labelling rule

    Labels stay what they have been since ex-2.1.6 — binary, drawn per visit,
    with probability graded in redness — except that now both operands draw.
    Each operand of a line draws independently at redness⁸ × {SLOTRATE}, and
    the line is labeled when either draws. The per-slot rate is half of the
    op1 labeller's {REDRATE}, which keeps the expected labeled-line rate where
    it was, up to the tiny both-draw overlap; the label-budget check below
    computes both. A labeled line's pulled positions are its prompt
    span, as before — the label never says which operand drew.

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
        f"| `{c['name']}` | {c['lam']:g} | {'∞' if np.isinf(c['tau']) else f'{c["tau"]:g}'} | "
        f"{c['labels']} | {'the operand that drew' if c['pull'] == 'slot' else 'prompt span, pooled'} |"
        for c in ex.CONDITIONS
    )
    mo.md(
        r"""
    ### Conditions

    Six conditions, three seeds each. Every anchored cell runs the
    anti-subspace term at the operating point; only the labelling rule and τ
    vary.

    | condition | $\lambda_a$ | τ | labels | pulled positions |
    |---|---|---|---|---|
    {ROWS}

    `either-t100` is the primary, *P*: the either-slot labeller at τ = 0.1,
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
    """.replace("{ROWS}", _rows)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration

    Three checks before anything runs, computed from the design constants and
    ex-2.1.9's published arrays: the label budget, the baseline profile shape,
    and the placement of the soft-τ rungs.
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
    **The label budget.** Over the closed-pair universe the corpus draws from,
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
        mo.md("_Ex-2.1.9's published results aren't reachable from here; the carried figures need them._"),
    )
    assert _e219 is not None
    _metrics219, arrays219 = _e219

    def ex219_mean(key: str, conds: list[str]) -> dict[str, np.ndarray]:
        """Seed-mean of a per-run array from ex-2.1.9, per condition."""
        return {c: np.mean([arrays219[f"{c}-s{s}/{key}"] for s in (0, 1, 2)], axis=0) for c in conds}

    READ219 = np.mean([arrays219[f"lam0-s{s}/readability"] for s in (0, 1, 2)], axis=0)
    return READ219, arrays219, ex219_mean


@app.cell(hide_code=True)
def _(ex219_mean):
    _CONDS3 = ["span-mean", "pool-t100", "op1-oracle"]
    _alpha3 = ex219_mean("alpha", _CONDS3)
    _prof = {}
    for _c, _a in _alpha3.items():
        _m = np.einsum("c,lct->lt", ex.LABEL_W, _a) - _a.mean(axis=1)
        _prof[_c] = {"margin": _m[:, : ex.SPAN], "drift": _a[:, :, : ex.SPAN].mean(axis=1)}
    _x = np.arange(ex.SPAN)

    @themed(
        name="ex219-position-profiles",
        alt_text="""
            A five-by-three grid of small flush panels. Columns are the
            ex-2.1.9 conditions span-mean, pool-t100 and op1-oracle; rows are
            residual slices with the last slice on top and the embedding at
            the bottom; each panel spans the four span roles op1, plus, op2
            and equals, from 0 to about 0.95 vertically. A shaded smooth-step
            area shows the margin and a solid line the drift. In the
            span-mean column every row above the embedding is a wide raised
            block with margin between 0.5 and 0.8 at all four roles. In the
            pool-t100 and oracle columns the block narrows to a single tall
            step at op1, with the other roles below 0.3. The drift line
            roughly shadows the margin at about half its height; on the
            embedding row it steps up at plus and equals for span-mean and at
            plus alone for pool-t100.
        """,
        caption=r"""
            **The baseline shape, from ex-2.1.9.** Margin $m(\ell,t)$
            (shaded area) and drift $\bar\alpha(\ell,t)$ (line) over the four
            span roles, one panel per residual slice with the embedding at the
            bottom, from ex-2.1.9's published arrays (seed means). Under
            op1-keyed labels the pooled pull (`pool-t100`, middle) narrows the
            span-mean's four-role block to a step at op1, nearly matching the
            oracle. This experiment's H2 figure will draw the same panels
            *twice per condition* — once per label group — and the question is
            whether the step moves to op2 on the lines where op2 drew.
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
            axes[0][col].set_title(cond, fontsize=9.5)
            axes[4][col].set_xticks(_x, ex.ROLES)
            axes[4][col].tick_params(axis="x", labelbottom=True, labelsize=8.5)
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
def _(READ219, ex219_mean):
    _pi9 = ex219_mean("pi", ["pool-t100"])["pool-t100"]
    _x = np.arange(ex.SPAN)

    @themed(
        name="ex219-weight-profile",
        alt_text="""
            A single column of five small flush panels, residual slices with
            the last slice on top and the embedding at the bottom, each
            spanning the four span roles op1, plus, op2 and equals from 0 to
            1. A shaded smooth-step area shows the softmin weight of
            ex-2.1.9's pool-t100 condition and a dashed line shows the
            un-anchored control's readability profile. At the embedding the
            weight is a tall step at op1 near 0.8 with about 0.2 at plus. In
            the deeper slices the weight stays almost entirely on op1, with a
            sliver at equals in the last slice, while the readability line
            stays high at op1 and plus, dips at op2 and rises again at
            equals.
        """,
        caption=r"""
            **Where the pull went under op1-keyed labels** (ex-2.1.9,
            `pool-t100`, seed mean): the softmin weight profile
            $\bar\pi(\ell,t)$ (shaded) against the un-anchored control's
            readability profile $R^2(\ell,t)$ (dashed) — a ridge probe's
            held-out score for op1's redness at each site. The H2 figure
            reads its per-group weight profiles against these same axes: on
            op2-triggered lines, weight following the *evidence* would step
            at op2, where this profile has almost none.
        """,
    )
    def _plot() -> plt.Figure:
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        from mini.vis import smooth_step, smooth_step_area

        _pi_ink = light_dark("#7b3fa0", "#b98ce0")
        _read_ink = light_dark("#2a9d8f", "#5fc9bd")
        _grey = light_dark("#666", "#999")
        fig, axes = plt.subplots(5, 1, figsize=(3.4, 3.7), sharex=True, sharey=True)
        _engine = fig.get_layout_engine()
        assert isinstance(_engine, ConstrainedLayoutEngine)
        _engine.set(hspace=0, h_pad=0.01)
        for row in range(5):
            sl = 4 - row
            ax = axes[row]
            smooth_step_area(ax, _x, _pi9[sl], ramp=0.5, color=_pi_ink, alpha=light_dark(0.22, 0.28))
            smooth_step(ax, _x, _pi9[sl], ramp=0.5, color=_pi_ink, lw=1.5)
            smooth_step(ax, _x, READ219[sl], ramp=0.5, color=_read_ink, lw=1.0, ls=(0, (4, 2)))
            ax.set(ylim=(-0.08, 1.05), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0.0, 0.5, 1.0])
            ax.spines[:].set_visible(False)
            ax.grid(axis="y", which="major", c="#888", alpha=0.2)
            ax.tick_params(axis="x", length=0, labelbottom=False)
            ax.tick_params(axis="y", left=True, right=True, direction="in", labelleft=False, labelright=False)
            ax.set_ylabel("emb" if sl == 0 else f"{sl}", fontsize=8)
        axes[4].set_xticks(_x, ex.ROLES)
        axes[4].tick_params(axis="x", labelbottom=True, labelsize=8.5)
        fig.supylabel("slice", fontsize=9)
        _p_key = plt.Rectangle((0, 0), 1, 1, fc=_pi_ink, alpha=0.35, ec="none")
        _r_key = plt.Line2D([], [], color=_read_ink, lw=1.0, ls=(0, (4, 2)))
        fig.legend([_p_key, _r_key], [r"softmin weight $\bar\pi(\ell,t)$", r"control readability $R^2(\ell,t)$"],
                   fontsize=8.5, frameon=False, ncols=1, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   labelcolor=_grey)  # fmt: skip
        return fig

    mo.Html(_plot())
    return


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
            **Placing the soft-τ ladder,** recomputed from ex-2.1.9's stored
            alignments as in that experiment's calibration: the leading span
            position's share of the softmin weight if the weights were
            computed on the un-anchored control's profile (**left**) or the
            `pool-t100` profile (**right**). Dashed verticals are this
            experiment's rungs. τ = 0.1 keeps a clear leader on the trained
            profile; τ = 2.5 is within a few hundredths of uniform (0.25) on
            both, confirming it as the tiny-bias arm and the reason no
            τ = ∞ arm runs here.
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
    mo.md(r"""
    ### Measurements

    Task evals, the readability profile, grading, and the trajectory
    instrument are ex-2.1.9's. Two things change shape and two are new, all on
    the same probe set (every color as op1 against its 27 closed partners —
    which, read symmetrically, is every closed pair):

    **Per-line alignment maps.** The eval step now stores $\cos(h, \hat v)$
    per probe *line* — (slices, 5832 lines, positions) — rather than averaged
    into per-color rows. The per-color maps every earlier statistic uses are
    recovered by averaging over partners, so every earlier statistic is unchanged, and the
    line-keyed statistics below become computable. Per-line softmin weights
    are stored the same way.

    **m_line**, the selectivity this experiment scores: the per-line
    generalization of ex-2.1.9's m_span. Per slice, the line-weighted mean
    alignment at each span role minus the unweighted mean over lines, keeping
    the largest role; then the mean over slices. The weights are the
    labeller's own P(labeled) per line, normalized — "how much closer does a
    line sit to the axis for being the kind of line the labeller flags" —
    which keys on op1 alone in the old scheme and on both operands now. The
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
    G1 stepping at op1, G2 at op2; under a global winner they are identical
    by construction.

    **Per-role drift in the trajectory.** Ex-2.1.9 couldn't say *when* the
    `+` embedding climbed to 0.30 alignment because the trajectory recorded
    op1 drift only; it now records $\bar\alpha(\ell, t)$ over all span roles
    at each measurement, which is the data that question needs. Report-side
    only; nothing is gated on it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Hypotheses

    Gates and noise floors are ex-2.1.6 through ex-2.1.9's wherever the
    statistic is unchanged; derivations for the new ones are in
    [experiment.py](./experiment.py), next to their constants.

    Throughout, *P* is `{PRIMARY}` (fixed in advance, see *Conditions*). The
    τ = 0.5 and 2.5 arms are reported beside P but not gated. If P turns out
    task-dirty, H2–H4 go unscored and the refuted H1 is the finding.

    **H1: Task cost.** Every condition is task-clean: `named_holdout` exact
    match within {TASK} (absolute) of the seed mean of the in-experiment
    control. No partial credit; nothing in this design adds task pressure,
    and no anchored condition on this testbed has ever shown any.

    **H2: The pull follows the label, line by line.** On P's per-group weight
    profiles:
    (a) at the embedding slice, each group's leading role is its own operand —
    op1 for G1, op2 for G2 — with weight $\ge {LEAD}$ in both (uniform is
    0.25);
    (b) across the four position-aware slices, the between-group contrast in
    op2 weight, $\mathrm{mean}_\ell\,[\bar\pi_{G2}(\ell, \text{op2}) -
    \bar\pi_{G1}(\ell, \text{op2})]$, is $\ge {CONTRAST}$.
    Partial: (a) holds and (b) lands in $[{CONTRASTP}, {CONTRAST})$.
    (a) is close to forced by the shared embeddings *if* the pull aligns red
    tokens at all, so on its own it mostly rules out the syntax latch; (b) is
    the real question — ex-2.1.9's deep-slice winner-take-all was per *run*,
    and (b) asks whether the winner can vary per line when the label demands
    it. Contrary outcomes: *global latch* — both groups lead at the same
    role (op1, or worse, `+`), the contrast sits near 0, and on op2-triggered
    lines the pull is dragging non-red states onto the axis, which H3(a)
    should catch as ᾱ contamination; *uniform* — no slice reaches {LEAD} in
    either group at any τ, meaning the pool can't tell the positions apart
    once the label stops pointing.

    **H3: The operating point survives the wider labeller.** On P:
    (a) containment, $\bar\alpha \le {ALPHA}$ at op1 (also the contamination
    detector for H2's global-latch outcome);
    (b) retention, every run whose running maximum of m_line reaches
    {FLOOR} ending at $\ge {RET}\times$ that maximum (quoted as the minimum
    across seeds);
    (c) grading, P's seed-mean op1 response $r^2$ against `sim¹·⁵` no more
    than {DROP} below the `op1-labels` arm's (higher is fine).
    Partial: (a) and (b) hold, (c) misses.

    **H4: Selectivity holds without the pointer.** Control-subtracted
    m_line of P reaches $\ge {FRAC}$ of the control-subtracted m_line of
    `slot-oracle`. Partial: $[{FRACP}, {FRAC})$.
    The latch account predicts roughly the op1-triggered share of the label
    mass (~half) plus whatever the shared embeddings give away, so a pass has
    to clear that; for scale, ex-2.1.9's pooled arms recovered 91–98% of
    their oracle. H4 is outcome where H2 is mechanism: the embedding
    give-away means H4 can pass while H2(b) fails, and that pair of verdicts
    would itself be a finding — selectivity delivered by the lexicon, not by
    localization.
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
    everything clean. A dirty condition would be news; a dirty P leaves H2–H4
    unscored, per the Hypotheses.
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
    $\bar\pi_{G2}(\ell,t)$ for P, in the smooth-step position-profile form of
    the carried ex-2.1.9 figures — two columns (G1: op1 drew; G2: op2 drew),
    five slice rows, the control's readability profile dashed behind, the
    embedding row annotated with the H2(a) verdicts and the deep-slice op2
    contrast (the H2(b) number) quoted in the caption. The same figure for
    the soft-τ arms beside it (smaller, reported not gated), and per-seed
    hairlines since ex-2.1.9's deep-slice choices were per-run one-hot —
    seed-mean profiles alone could read as graded when the runs are not.

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
    Two figures, all conditions with P called out. Fig 1: trajectories of
    m_line and ᾱ (op1) over training in the usual panel-grid form, schedules
    drawn beneath, the retention statistic annotated per panel. Fig 2: the
    grading scatter (per-color op1 response against `sim¹·⁵`), one panel per
    condition, each with its $r^2$; `op1-labels` is the reference the {DROP}
    drop is read against, and its own reproduction against ex-2.1.9's
    `pool-t100` (targets under *Calibration*) is quoted in
    the caption.

    Expected: P tracks `op1-labels` on all three statistics — the pull
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
    Dot chart of control-subtracted m_line by condition (three seeds and
    their mean), with `op1-labels` and `slot-oracle` as reference bands and
    the oracle fraction quoted beside P. m_span and $m_\text{op1}$ reported
    alongside for continuity with ex-2.1.9. Expected: P between the
    references, nearer the oracle, as ex-2.1.9's pooled arms were to theirs.
    Contrary: P at ~half the oracle is the latch account's signature;
    P *above* the oracle is possible in principle (the oracle pulls one slot
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
