import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.12: fitted channel probes over the anchored checkpoints",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import tempfile
    from pathlib import Path
    from typing import NamedTuple, cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.transforms import Affine2D

    # Marimo puts the notebook directory on sys.path, so the conditions and the
    # ref come from the definition module beside this one.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import AxesRow, figure_html, light_dark, themed
    from sca.colorcube import sim_to_red
    from sca.vis_grading import GRID_RGB
    from sca.vis_probes import draw_traces

    use_publisher(report_bundle(__file__))

    SLICE_NAMES = ["emb", "1", "2", "3", "4"]
    POS_NAMES = ["op1", "+", "op2", "=", "ans", r"\n"]
    TARGET_TITLES = {"op1": "The color of op1", "op2": "The color of op2", "ans": "The color of the answer"}
    """Panel titles, one per probe target: which token's color that panel's probes decode."""


@app.function(hide_code=True)
def load_npz(ref: str) -> dict[str, np.ndarray] | None:
    """A published npz as a dict of arrays, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "arrays.npz")])
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


@app.function(hide_code=True)
def draw_probe_grid(
    ax: plt.Axes, stack: np.ndarray, *, own: int | None = None, row_names: bool = True, scale: bool = True
) -> None:
    """One row per residual slice, the six token positions across, drawn into *ax*.

    *stack* is one condition's (seed, slice, position, channel) strict R² for one target. Each row draws the seed mean as a step-line per RGB channel over the grey area of their mean, with the seed envelope as hairlines — see :func:`~sca.vis_probes.draw_traces`, which the landmark maps of ex-2.1.5 use the same way.

    The rows share one Axes, each offset by a translate transform, so a row spans y ∈ [si, si+1] and keeps a fixed 0–1 scale: a site that decodes nothing looks like it decodes nothing, here and in every other panel. Each row's zero line is also the rule between it and the row below, so the rules mark the boundaries and no separators are needed. That is the layout of the grading grids on [the figures page](../d2.1/report.py), and the two figures cover the same sites, so a reader who has read one can read the other.

    Negative scores clip to the floor first. A probe that does worse than predicting the mean has failed, and how much worse is not a finer grade of failure; the rows also draw unclipped, so a value left below zero would run down into the row beneath.

    *own* is the position of the token whose color the panel decodes; its x label draws in bold, since what a site holds *before* that token is the question H1 asks. *row_names* draws the slice names on the left and *scale* the 0–1 scale on the right; a panel in a row of them wants each only on its own side of the figure (see :func:`probe_grids`).
    """
    sw = 0.7  # smooth-step plateau width, as the grading grids use
    _, n_slices, n_pos, _ = stack.shape
    stack = np.clip(stack, 0, 1)
    for si in range(n_slices):
        shift = Affine2D().translate(0, si) + ax.transData  # embedding at the bottom, as the profile figures do
        draw_traces(
            ax, stack.mean(0)[si], spread=stack[:, si], ramp=1 - sw, faint_risers=True, transform=shift, clip_on=False
        )
        ax.axhline(si, color=light_dark("#aaaa", "#333a"), lw=0.5)
    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(0, n_slices)
    ax.set_xticks(range(n_pos), POS_NAMES)
    ax.set_yticks(np.arange(n_slices) + 0.5, SLICE_NAMES if row_names else [""] * n_slices)
    ax.tick_params(axis="x", labelsize=6, bottom=False)
    ax.tick_params(axis="y", labelsize=7, left=False)
    if own is not None:
        # Weight, not color: the three channel hues are data here, and a red tick label would
        # read as the R channel rather than as a slot.
        ax.get_xticklabels()[own].set(fontweight="bold")
    if scale:
        # One y scale for the whole figure, against the bottom row, whose offset is zero.
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(np.arange(n_slices + 1), ["0", "1"] + [""] * (n_slices - 1))
        sec.tick_params(labelsize=6, direction="out")
        sec.set_ylabel("R², 0–1 per row", fontsize=7)


@app.function(hide_code=True)
def probe_grids(
    panels: dict[str, tuple[np.ndarray, int | None]], figsize: tuple[float, float] = (8.6, 2.7)
) -> plt.Figure:
    """A row of probe grids sharing one set of rows and one R² scale, titled by *panels*' keys; each value pairs the R² array with the position of the token whose color it decodes (see :func:`draw_probe_grid`).

    The panels measure the same quantity on the same scale and the reader compares across them, so they belong in one figure — one image per condition, which is also how the recipe reads: each condition is a picture, and the progression is the four of them in order. The row names sit on the left of the first panel and the scale on the right of the last, since both are shared.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    row = cast(AxesRow, axes[0])
    for i, (ax, (title, (stack, own))) in enumerate(zip(row, panels.items(), strict=True)):
        draw_probe_grid(ax, stack, own=own, row_names=i == 0, scale=i == len(row) - 1)
        ax.set_title(title, fontsize=8, pad=6)
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.12: fitted channel probes over the anchored checkpoints

    /// tip |
    <!-- tl;dr -->
    Ridge probes fitted on the published checkpoints of the four D2.1 conditions, one per RGB channel, for each operand and for the answer, at every (slice, position) site. The probes find nothing before op2 in any condition, anchored or not, so the off-key tilt in the grading figures is a property of the probe set. No decodability cost of anchoring resolves. The maps also show the bare anchor giving up decodability in the late slices, and the rest of the recipe restoring it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings

    **H1 (pre-op2 ceiling) — holds.** Across all conditions, slices, and channels, the largest pre-op2 seed-mean strict R² for the op2 target is −0.56. That is well under the gate of 0.12. Every condition reads at or below the no-information floor that the fold design sets.

    <!-- REVIEW: results round changed H1's summary line from "sits at the no-information floor" to "at or below" — the op1 sites read −1.59 to −1.34, below the −0.56 floor, which the H1 analysis attributes to within-fold extrapolation. Verify: the pre-op2 minimum quoted in the H1 section. -->

    **H2 (decodability) — not resolved.** The R channel misses the one-sided 0.10 gate at all three sites, by 0.12 to 0.18, and every miss falls inside the preregistered resolution band, which runs from 0.53 to 0.96 (twice the pooled seed spread; both groups vary a lot from seed to seed, the control somewhat more). G and B pass the gate outright at every site, reading higher in the primary condition than in the scoring seeds of the control.

    <!-- REVIEW: results round changed H2's verdict from "holds" to "not resolved" — the frozen rule turns an in-band miss into "not resolved", never into a pass, and R missed at all three sites, so the universal per-channel claim is neither established nor refuted. Verify: the H2 section's verdict line and the discussion say the same. -->
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this report
    Preregistered: the hypotheses and their gates were frozen at the commit that added this file (`cdd18b46`), before the run. Results replaced the placeholders in place, and analyses conceived after seeing the data sit under [Exploratory analyses](#exploratory-analyses), marked post hoc.
    ///

    ## Why this experiment

    The α maps of [the D2.1 figures page](../d2.1/report.py) key one measurement by either operand's color, and wherever a condition responds at all, the slot that is *not* the key tilts too.[^alpha] For the slots that precede op2 that tilt cannot be a response: attention is causal, so the state above the first token is a function of that token alone, and the tilt is the on-key response composed through the probe set — a pair is on the set only if its mix lands back on the color grid, which correlates a color with its partners. The α statistic cannot show this cleanly, because it reads alignment with the anchor axis, which the un-anchored control ignores; its panels are flat for lack of any response to compose, and a reader cannot tell the composition story from the alignment story by looking.

    [^alpha]: α is the cosine between the residual stream (the running vector each layer reads from and writes back to) and the anchor axis, the direction the concept is trained toward.

    Fitted probes remove that asymmetry. A probe fit at a site reads whatever color representation the model keeps there, whether that is the anchored one or the model's own. That puts every condition on the same footing: whatever any of them shows at the sites that precede op2 is capped by the information ceiling of the pairing, computable in advance (H1), and how much of that ceiling a condition reaches is a property of its embedding geometry rather than of anchoring. [Ex-2.1.5](../ex-2.1.5/report.py) ran the same kind of scan on an un-anchored two-form model, and there the op2 probes read zero at op1 sites. That experiment sampled its lines over distinct operand pairs, so the operands were all but uncorrelated and the ceiling was effectively zero. Comparing the two probe sets is what makes the case: the tilt follows the pairing, and the model has no say at these sites.

    The same scan answers a question the D2.1 experiments left open. Anchoring reshapes the stream — that is what it is for — and the task gates only showed that *accuracy* survives. Whether the ordinary, linearly readable color geometry survives too is the bounded-side-effects claim read through a probe, and per-channel R² maps against the un-anchored control measure it (H2).

    ## Method

    No training. Every run is a published checkpoint of the four D2.1 conditions, probed on its own experiment's published probe set (the four sets are identical, which the pipeline asserts): 5,832 lines, every color as op1 against each of the 27 partners whose mix stays on the grid.

    **Sites and targets.** The residual stream is captured at 5 slices (embedding, then each of the 4 layers) × 6 token positions (`op1 + op2 = ans \n`). At each site, ridge probes (ℓ₂ = 10⁻²) decode three targets: the RGB triple of op1, of op2, and of the answer. Probes are fit and scored per site independently, as in ex-2.1.5.

    **Holdout.** The strict per-value protocol of ex-2.1.5 (`sca.compute.geometry.strict_r2`): to score a channel at a level, every line holding that level in that channel of *any* role — either operand or the answer — leaves the fit together, so the probe must place an unseen level from the others. R² is reported per channel from the out-of-fold predictions, and the prediction curves the figures draw are those same out-of-fold predictions, averaged per color keyed by either operand in turn (the two margins the D2.1 figures use).

    **Teacher forcing.** The ans and `\n` positions have the answer token in the input, so readings there include what the token itself carries; the embedding row is the surface-text control, as in ex-2.1.5.

    **Noise floor.** The primary condition has nine seeds and the others three. The pooled between-seed standard deviation of per-site R², computed per condition before any comparison is read, sets the resolution: a difference smaller than twice that floor is reported as not resolved.

    ### Conditions

    The four conditions of the D2.1 progression, from their published checkpoints: the un-anchored control and the bare anchor at λ = 0.1 (ex-2.1.6, 3 seeds each), the anchor plus the anti-subspace term at its tuned operating point (ex-2.1.8 `end90-hold30`, 3 seeds), and the full recipe with pooled either-operand labels (ex-2.1.10 `either-t100`, 9 seeds).

    ## Hypotheses

    - **H1.** At the two sites that precede op2 — the op1 and `+` positions, every slice — the seed-mean strict R² of the op2-target probe stays below 0.12 in every condition and every channel. This is deliberately a protocol check rather than a discovery: the state at these sites is a function of the tokens before op2, so the best any probe can do is E[op2 | op1], and the closure rule caps that at R² = 0.25 / (35/12) ≈ 0.086 per channel, whatever the model; the gate adds margin for estimation noise. A reading at or above 0.12 would mean the probe found more op2 information before op2 than the pairing supplies — a leak in the protocol or a misread of the probe set. No lower gate is stated, because every outcome below the ceiling is consistent with the bookkeeping account: how much of the ceiling each condition reaches is a property of its embedding geometry (how linearly it exposes level parity), read descriptively under E1, and a cross-condition difference there would be an observation about geometry that does not reopen the causal question.
      <!-- REVIEW: prereg round asked for a lower or similarity gate on H1, since the upper gate alone cannot fail without a bug. Resolved the other way: H1 now says plainly that it is a protocol check, and the cross-condition comparison moved to E1 as descriptive — a floor would score how linearly each embedding exposes parity, which is not the claim under test. Verify: the H1 analysis's contrary reading matches this scope. -->
    - **H2.** Anchoring does not cost linear color decodability where the control has it. Site selection is separated from scoring so the selection noise stays off the gate: for each target, the comparison site is the (slice, position) with the best strict R² (mean over channels) in the map of the control's seed 0 alone, ties broken toward the earlier slice and then the earlier position, among targets whose selection map reaches R² ≥ 0.5 (a target below that is reported unscored). The score then compares the seed-mean of the control's remaining seeds against the seed-mean of all nine primary seeds. At each selected site, the primary's per-channel R² comes within 0.10 of the control's (one-sided: the primary may exceed it). A miss smaller than twice the pooled between-seed standard deviation at that site is reported as not resolved rather than as a failure. Partial: exactly one channel of one target misses by more than that and by no more than 0.2. The intermediate conditions are reported at the same sites without a gate.
      <!-- REVIEW: prereg round flagged the original argmax-over-30-sites on the control's own 3-seed mean as loading the one-sided gate against the primary (selection inflates the control at the chosen site). Resolved by selecting on control seed 0 and scoring on seeds 1–2; ties get a fixed break; the method's noise floor now decides "not resolved" for near-gate misses. Verify: the analysis never lets the selection seed into the scored mean. -->
    """)
    return


@app.cell(hide_code=True)
def _():
    _loaded = load_npz(ex.ARRAYS_REF)
    mo.stop(_loaded is None, mo.md("_Results are not published yet; the analysis renders once they are._"))
    assert _loaded is not None
    arrays: dict[str, np.ndarray] = _loaded
    # (seeds, slices, positions, channels) strict R², per condition × target.
    r2 = {
        (c.key, t): np.stack([arrays[f"{c.key}/s{i}/r2_strict_ch"] for i in range(c.seeds)])[:, :, :, ti, :]
        for c in ex.CONDITIONS
        for ti, t in enumerate(ex.TARGETS)
    }
    return arrays, r2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The maps

    One figure per condition carries every reading the sections below score: its three panels are the probe targets, and within a panel the slices stack upward from the embedding, as the α grids on the figures page do.
    """)
    return


@app.cell(hide_code=True)
def _(r2):
    # Takeaway first: what each figure shows, not an inventory of what is drawn.
    _ALT = {
        "ex-2.1.6/lam0": """
            Each panel decodes its own token and nothing that precedes it: op1 and op2 read about 0.7 at their own slot in every slice, and the answer's color climbs at the equals slot, with a small step at op2 in the upper slices. In the op2 panel the two slots before op2 lie flat on the floor.
        """,
        "ex-2.1.6/lam0.1": """
            The same shape, except that the op2 panel fades going up the stack: its own slot reads about 0.84 at the embedding and almost nothing by the last slice. The two slots before op2 lie flat on the floor there, as in every condition.
        """,
        "ex-2.1.8/end90-hold30": """
            About half of that late-slice fade is repaired: op2's own slot holds 0.75 at the embedding and 0.32 at the last slice, where the bare anchor gave up nearly all of it. The two slots before op2 lie flat on the floor.
        """,
        "ex-2.1.10/either-t100": """
            No fade left: op2's own slot holds 0.6 to 0.8 through every slice, as it does un-anchored, and the nine-seed envelope marks sit close to the channel marks. The two slots before op2 lie flat on the floor.
        """,
    }

    def _maps(cond: ex.Condition) -> str:
        @themed(
            name=f"probe-channel-maps-{cond.exp}-{cond.name}",
            alt_text=_ALT[cond.key],
            caption=rf"""
                **{cond.title}** ({cond.exp}, `{cond.name}`, {cond.seeds} seeds). Per-channel strict probe R² at every site: one row per residual slice, with the embedding at the bottom, and the six token positions across. Each RGB channel draws as a mark per site over the grey area of their mean, with faint risers between them, since the sites are where the measurements are; the grey marks are the seed envelope, so where the seeds disagree they stand off the silhouette. The three panels decode three targets, and the bold slot on the x axis is the token whose color that panel reads. Negative scores clip to the floor — a probe that does worse than predicting the mean has failed, and the H1 prose carries the raw values. Each row spans the same 0–1 scale, ticked once at the bottom right.
            """,
        )
        def _plot() -> plt.Figure:
            return probe_grids({TARGET_TITLES[t]: (r2[cond.key, t], p) for t, p in ex.TARGETS.items()})

        return _plot()

    for c in mo.status.progress_bar(ex.CONDITIONS, title="Rendering charts", remove_on_exit=True, show_eta=True):
        mo.output.append(mo.md(f"### {c.title}\n\n{_maps(c)}"))
    return


@app.cell(hide_code=True)
def _(r2):
    # Pre-op2 readings for the op2 target: (slices, 2 positions, channels) per condition.
    _pre = {c.key: r2[c.key, "op2"].mean(0)[:, list(ex.PRE_OP2), :] for c in ex.CONDITIONS}
    _mx = max(float(v.max()) for v in _pre.values())
    # The constant-state site: the '+' position at the embedding.
    _plus_emb = [float(v[0, 1].max()) for v in _pre.values()]
    _op1_lo = min(float(v[:, 0].min()) for v in _pre.values())
    _op1_hi = max(float(v[:, 0].max()) for v in _pre.values())
    _agree = "exactly" if max(_plus_emb) == min(_plus_emb) else f"to {max(_plus_emb) - min(_plus_emb):.1e}"
    mo.md(rf"""
    ## The off-key ceiling (H1)

    **H1 holds, and the gate is not approached.** Over every condition, slice, and channel, the largest pre-op2 reading for the op2 target is {_mx:+.2f}, against a gate of {ex.H1_GATE}. The maps clip negatives to the floor, so the raw values are quoted here; the sign is what matters.

    Under this holdout, a probe with no information cannot even reach R² = 0. Scoring a level removes every line carrying it from the fit, so such a probe predicts the training mean of each fold, and that mean sits away from the held-out level by construction. On this level structure it lands at −0.56. The fold combinatorics fix that number, not any model.

    A site whose state never varies pins the same number empirically. At the embedding, the `+` state is a single token embedding, and it is a different vector in every condition. Even so, all four conditions read {min(_plus_emb):.4f} there, agreeing {_agree}. The op1 sites read lower still, {_op1_lo:.2f} to {_op1_hi:.2f}. At those sites the probe can read the level of op1 itself and extrapolate op2 from the within-fold regression, and under this holdout that extrapolation points the wrong way. E1 traces the mechanism.

    So no condition, anchored or not, carries op2 information that a linear probe can use before op2. The off-key tilt in the α maps comes from the pairing of the probe set, as the figures page reads it.

    <!-- REVIEW: the prereg placeholder asked for a per-condition × channel table of the deciding maxima and a rule at the 0.086 ceiling on the figure. Both dropped: every value is negative, the twelve tabulated cells would all read −0.56 (the maxima land on the same constant-state site everywhere), and a ceiling rule cannot be drawn on a map that clips negatives to the floor. The prose quotes the maximum, the cross-condition identity, and the per-site ranges instead. Verify: the quoted maximum stands in for the table's every entry. -->
    """)
    return


@app.cell(hide_code=True)
def _(r2):
    _ctrl, _prim = ex.CONDITIONS[0], ex.CONDITIONS[-1]
    _inter = ex.CONDITIONS[1:3]
    _CH = "RGB"

    class _Row(NamedTuple):
        """One target's deciding comparison: per-channel readings at the site the control picked."""

        t: str
        site: str
        sel: float
        ctrl: np.ndarray
        prim: np.ndarray
        gap: np.ndarray
        band: float
        inter: dict[str, np.ndarray]

    _rows: list[_Row] = []
    for _t in ex.TARGETS:
        _m = r2[_ctrl.key, _t]
        _sel = _m[ex.H2_SELECT_SEED].mean(-1)
        _si, _pi = np.unravel_index(int(np.argmax(_sel)), _sel.shape)
        _c = _m[np.arange(len(_m)) != ex.H2_SELECT_SEED][:, _si, _pi, :]
        _p = r2[_prim.key, _t][:, _si, _pi, :]
        _band = 2 * float(np.sqrt((_c.var(0, ddof=1).mean() + _p.var(0, ddof=1).mean()) / 2))
        _rows.append(_Row(
            t=_t, site=f"slice {_si}, {POS_NAMES[_pi]}", sel=float(_sel[_si, _pi]),
            ctrl=_c.mean(0), prim=_p.mean(0), gap=_c.mean(0) - _p.mean(0), band=_band,
            inter={c.title: r2[c.key, _t].mean(0)[_si, _pi] for c in _inter},
        ))  # fmt: skip
    assert all(r.sel >= ex.H2_MIN_CONTROL for r in _rows), "an unscored target needs its row marked"

    def _num(v: float, bold: bool = False) -> str:
        s = f"{v:+.2f}"
        return f"<strong>{s}</strong>" if bold else s

    def _table() -> str:
        head = (
            "<tr><th>target</th><th>site (control's pick)</th><th>condition</th>"
            + "".join(f'<th class="num">{ch}</th>' for ch in _CH)
            + "".join(f'<th class="num">Δ{ch} ↓</th>' for ch in _CH)
            + '<th class="num">2×σ</th></tr>'
        )
        body = ""
        for r in _rows:
            n = 2 + len(r.inter)
            body += (
                f'<tr><td rowspan="{n}">{r.t}</td><td rowspan="{n}">{r.site}</td>'
                + "<td>un-anchored (seeds 1–2)</td>"
                + "".join(f'<td class="num">{_num(v)}</td>' for v in r.ctrl)
                + '<td class="num"></td>' * 3
                + '<td class="num"></td></tr>'
            )
            for title, vals in r.inter.items():
                body += (
                    f"<tr><td>{title}</td>"
                    + "".join(f'<td class="num">{_num(v)}</td>' for v in vals)
                    + '<td class="num"></td>' * 4
                    + "</tr>"
                )
            body += (
                f"<tr><td>{_prim.title} (primary)</td>"
                + "".join(f'<td class="num">{_num(v)}</td>' for v in r.prim)
                + "".join(f'<td class="num">{_num(g, bold=g <= ex.H2_GATE)}</td>' for g in r.gap)
                + f'<td class="num">{r.band:.2f}</td></tr>'
            )
        return f'<div class="report-table-scroll"><table class="report-table">{head}{body}</table></div>'

    _misses = [(r.t, ch, float(g), r.band) for r in _rows for ch, g in zip(_CH, r.gap, strict=True) if g > ex.H2_GATE]  # fmt: skip
    assert all(g < band for _, _, g, band in _misses), "a resolved miss would change the verdict"
    _gb_excess = max(float(-r.gap[k]) for r in _rows for k in (1, 2))
    _ctrl_spread = sorted(float(v) for v in r2[_ctrl.key, "op1"][:, 1, 0, :].mean(-1))
    _prim_spread = sorted(float(v) for v in r2[_prim.key, "op1"][:, 1, 0, :].mean(-1))

    mo.vstack(
        [
            mo.md(r"""
        ## Decodability under anchoring (H2)

        The deciding comparison is one row per condition per target, at the site the control's selection seed picked for that target.
        """),
            mo.md(
                figure_html(
                    _table(),
                    caption=rf"""
                **The H2 comparison at the control-chosen sites.** Δ is the control's scoring-seed mean minus the condition's seed-mean, per channel, for the gated (primary) row only; lower is better and values at or below the one-sided gate of {ex.H2_GATE} are bold. 2×σ is twice the pooled between-seed standard deviation at the site — the preregistered resolution: a positive Δ smaller than it reads as not resolved. The intermediate conditions carry no gate.
            """,
                    aria_label="H2 decodability comparison table",
                )
            ),
            mo.md(rf"""
        **H2 does not resolve.** The R channel misses the {ex.H2_GATE} gate at all three sites, by {min(g for _, _, g, _ in _misses):.2f} to {max(g for _, _, g, _ in _misses):.2f}. Every miss sits far inside its resolution band, so under the frozen rule each reads as not resolved rather than as a failure — and never as a pass, so the per-channel claim as a whole is neither established nor refuted. No other channel misses. G and B run the other way: at every site the primary condition reads above the scoring seeds of the control, by up to {_gb_excess:.2f}.

        Seed variance is what limits the comparison, and the control carries slightly more than half of it. Its three seeds disagree about how linearly color reads, with channel-mean R² at the op1 site running {", ".join(f"{v:.2f}" for v in _ctrl_spread)} across them, and its scored mean rests on two seeds; the nine primary seeds span {_prim_spread[0]:.2f} to {_prim_spread[-1]:.2f} themselves. A deficit would have had to be about half an R² unit to resolve, so only a large cost could have been caught.

        The best control site for the ans target is the ans position itself, which the selection rule was allowed to pick. Teacher forcing puts the answer token in the input there, so both models read the embedding of the token as much as anything they computed. The comparison is still like for like.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(arrays: dict[str, np.ndarray], r2):
    _TI = {t: i for i, t in enumerate(ex.TARGETS)}
    _lv = np.rint(GRID_RGB * 5).astype(int)
    _sim = sim_to_red(GRID_RGB, power=1.5)
    _zd = load_npz("reports/m2/d2.1/arrays")

    def _pred(c: ex.Condition) -> np.ndarray:
        """Seed-mean out-of-fold prediction margins keyed by op1: (slices, colors, positions, targets, ch)."""
        return np.mean([arrays[f"{c.key}/s{i}/pred_op1"] for i in range(c.seeds)], axis=0)

    # E1: the off-key prediction curve at the op1 site, in level units, against op1's R level.
    _curves = {c.key: _pred(c)[0, :, 0, _TI["op2"], 0] * 5 for c in ex.CONDITIONS}
    _by_level = {k: np.array([v[_lv[:, 0] == i].mean() for i in range(6)]) for k, v in _curves.items()}
    _e1_spread = max(float(np.abs(a - b).max()) for a in _by_level.values() for b in _by_level.values())
    _e1_ctrl = _by_level[ex.CONTROL]

    # E2: on-key prediction curves at the op1 site vs the α lookup, per anchored condition.
    _e2 = {}
    if _zd is not None:
        for _c in ex.CONDITIONS[1:]:
            _alpha = np.mean([_zd[f"{_c.key}/s{i}/op1"] for i in range(_c.seeds)], axis=0)[:, :, 0].mean(0)
            _p = _pred(_c)[:, :, 0, _TI["op1"], :].mean(0)
            _e2[_c.key] = [float(np.corrcoef(_p[:, k], _alpha)[0, 1]) for k in range(3)] + [
                float(np.corrcoef(_alpha, _sim)[0, 1])
            ]

    # E3: emb on-key prediction error vs the affinity and its partner-mean.
    _c1, _c2 = arrays["probe/pairs"][:, 0], arrays["probe/pairs"][:, 1]
    _pmean_sim = np.bincount(_c2, _sim[_c1]) / np.bincount(_c2)
    _e3 = {}
    for _c in ex.CONDITIONS:
        _err = _pred(_c)[0, :, 0, _TI["op1"], :] - GRID_RGB
        _e3[_c.key] = (
            max(abs(float(np.corrcoef(_err[:, k], _pmean_sim)[0, 1])) for k in range(3)),
            float(np.corrcoef(_err[:, 0], _sim)[0, 1]),
        )

    # E4 (post hoc): channel-mean op2 decodability at its own position, last slice.
    _e4 = {c.key: r2[c.key, "op2"][:, 4, 2, :].mean(-1) for c in ex.CONDITIONS}
    _fmt = lambda a: ", ".join(f"{v:+.2f}" for v in a)  # noqa: E731

    @themed(
        name="offkey-prediction-curve",
        alt_text="""
            Four nearly coincident solid lines rise from about 1.5 to about 3.5 across the six levels; a dashed line alternates between 2 and 3 at each level. The solid lines are indistinguishable from one another.
        """,
        caption=r"""
            **The off-key prediction at the op1 site.** Mean out-of-fold prediction of op2's R channel, in level units, against op1's R level, at the embedding slice: one solid line per condition, with the parity sawtooth — the ideal predictor E[op2 | op1] — dashed. The four conditions nearly coincide.
        """,
    )
    def _e1_plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.4, 2.4), layout="constrained")
        xs = np.arange(6)
        ax.plot(xs, 2 + (xs % 2), ls="--", color=light_dark("#888", "#999"), lw=1, label="ideal E[op2 | op1]")
        for i, y in enumerate(_by_level.values()):
            ax.plot(xs, y, color=light_dark("#1f77b4", "#6baed6"), lw=1, alpha=0.8, label="observed, all four conditions" if i == 0 else None)  # fmt: skip
        ax.set(xticks=xs, yticks=[1, 2, 3, 4])
        ax.set_xlabel("op1 R level", fontsize=8)
        ax.set_ylabel("predicted op2 R level", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        return fig

    mo.vstack(
        [
            mo.md(r"""
        ## Exploratory analyses

        **E1 — the strict estimator rules out the parity route.** The preregistered comparison expected the probes to trace the parity sawtooth E[op2 level | op1 level] = 2, 3, 2, 3, 2, 3. They do nothing of the kind, and in hindsight they cannot: scoring a level under the strict holdout removes it from every role, so the very lines a parity reader would generalize from leave the fit along with it. (That reading of the fold design is post hoc; E1 as preregistered expected the sawtooth.)
        """),
            mo.Html(_e1_plot()),
            mo.md(rf"""
        What the probes read instead is the level of op1, which is decodable, and they extrapolate op2 from the within-fold regression. The out-of-fold R-channel prediction of the control runs {_e1_ctrl[0]:.2f}, {", ".join(f"{v:.2f}" for v in _e1_ctrl[1:5])}, {_e1_ctrl[5]:.2f} in level units across the R level of op1, a ramp, and the curves of the four conditions agree within {_e1_spread:.2f} level units. That extrapolation is what pushes the op1-site R² below even the constant-state floor. This estimator cannot say whether any embedding exposes parity linearly; it says only that the strict off-key reading holds no op2 information in any model.

        **E2 — the R channel tracks α loosely.** Correlations over colors at the op1 site, averaged over slices: across the three anchored conditions the R channel reads {", ".join(f"{v[0]:+.2f}" for v in _e2.values())} against the α lookup, and G and B fall between {min(v[k] for v in _e2.values() for k in (1, 2)):+.2f} and {max(v[k] for v in _e2.values() for k in (1, 2)):+.2f}. α itself tracks the sim¹·⁵ statistic at {min(v[3] for v in _e2.values()):.2f}–{max(v[3] for v in _e2.values()):.2f}. As preregistered, no channel reproduces α, because α carries redness, which is a function of all three channels rather than of any one.

        **E3 — no drag signature in the readout of the model itself.** In every condition, the control included, the on-key prediction errors at the emb slice correlate with the partner-mean of the affinity at |r| ≤ {max(v[0] for v in _e3.values()):.2f}. They correlate with the affinity itself only weakly, and to almost the same degree in each condition ({min(v[1] for v in _e3.values()):+.2f} to {max(v[1] for v in _e3.values()):+.2f} for the R channel). So the drag-versus-direct ordering that shows up along the anchor axis does not reappear in RGB decodability: whatever the op1-keyed labels dragged onto the axis, they did not measurably bend the color geometry of the embeddings.

        **E4 — the bare anchor gives up late-slice decodability, and the recipe restores it** (post hoc). Channel-mean op2 decodability at its own position, last slice: control {np.mean(_e4[ex.CONTROL]):.2f}, bare anchor {np.mean(_e4["ex-2.1.6/lam0.1"]):.2f} (seeds {_fmt(sorted(_e4["ex-2.1.6/lam0.1"]))}), anti-subspace {np.mean(_e4["ex-2.1.8/end90-hold30"]):.2f}, primary {np.mean(_e4[ex.PRIMARY]):.2f}. The indiscriminate lift of the bare anchor crowds linear color readout out of the top of the stack. The anti-subspace term restores about half of it, and the full recipe matches the control. The op2 panel of each condition's map shows the whole shape.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    The ceiling result supplies the half of the demonstration that the α statistic could not. α reads an axis the un-anchored control ignores, so the control's flat panels were consistent with two stories: either there was no response to compose, or the measurement only sees anchored models. Fitted probes read the geometry of each model itself, and in all four conditions they find the same nothing before op2, down to the same fold-design floor.

    On side-effects, the gated comparison says less than we would like. The unresolved gaps point both ways, with R slightly favoring the control and G and B favoring the primary. Resolving a difference of about 0.1 would take more un-anchored seeds than the three that exist.

    The post-hoc E4 observation is the sharpest new fact. The bare anchor costs late-slice decodability outright, and the containment half of the recipe, first the anti-subspace term and then pooled labels, restores it. That agrees with what ex-2.1.8 and ex-2.1.10 measured along the anchor axis as drift and its containment, read here in the coordinates of the model instead. It is a single observation on three-seed conditions and it is marked post hoc; a preregistered version would have to fix its sites in advance.
    """)
    return


if __name__ == "__main__":
    app.run()
