import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="D2.1: figures for the anchoring post",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import tempfile
    from pathlib import Path
    from typing import cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.transforms import Affine2D

    # Marimo puts the notebook directory on sys.path, so the condition list and the ref come from the prep module beside this one rather than a copy of them here.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import AxesRow, light_dark, smooth_step, smooth_step_area, themed
    from sca.colorcube import sim_to_red
    from sca.vis_grading import GRID_RGB, I_RED, REDNESS, GradingCloud

    use_publisher(report_bundle(__file__))

    SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
    """The grading statistic's target shape, as in ex-2.1.6 – ex-2.1.10."""
    SLICE_NAMES = ["emb", "1", "2", "3", "4"]
    POS_NAMES = ["op1", "+", "op2", "=", "ans", r"\n"]
    KEYINGS = {"op1": "first", "op2": "second"}
    """Each keying, and which operand's color it puts on the color axis."""


@app.function(hide_code=True)
def load_margins() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], np.ndarray] | None:
    """α per condition and keying, plus the probe set's operand index.

    `experiment.py` publishes both operand margins of every condition, per run. `out[key]["op1"]` is seed-mean (slice, color, position) α against the first operand's color and `["op2"]` the same measurement keyed by the second, so a figure picks its color axis by choosing between them. `stacks` is the same measurement with the seed axis kept, (seed, slice, color, position) — the progression figure's spread envelope dithers over it. The pair index rides along: it says which color sits at each operand slot on each probe line, which is what the partner-composition note reads.
    """
    store = project_store()
    art = store.get_refs([ex.ARRAYS_REF])[ex.ARRAYS_REF]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "margins.npz")])
        with np.load(path) as z:
            stacks = {
                c.key: {k: np.stack([z[f"{c.key}/s{i}/{k}"] for i in range(c.seeds)]) for k in KEYINGS}
                for c in ex.CONDITIONS
            }
            out = {key: {k: v.mean(axis=0) for k, v in d.items()} for key, d in stacks.items()}
            return out, stacks, z["probe/pairs"]


@app.function(hide_code=True)
def r2_sim(y: np.ndarray) -> float:
    """Shape r² against sim¹·⁵, the grading statistic; NaN where the response is constant."""
    if np.std(y) < 1e-6:
        return float("nan")
    return float(np.corrcoef(y, SIM_TARGET)[0, 1] ** 2)


@app.function(hide_code=True)
def draw_grading_grid(ax: plt.Axes, a: np.ndarray, *, row_names: bool = True, scale: bool = True) -> None:
    """One row per residual slice, one grading-cloud slot per token position, drawn into *ax*.

    *a* is (slice, color, position) α, keyed by whichever operand's color the caller wants on the color axis; the overlay is that keying's pure-red row. Every figure below shares this, so the two views of a condition differ only in the array they are handed — and so do the four conditions.

    The rows share one Axes, each offset by a translate transform, so a row spans y ∈ [si, si+1] and its slots keep a fixed 0–1 scale: a slot that holds nothing looks like it holds nothing, here and in the panel below. Stacking them this way costs nothing in code and buys the overhang — a cloud's extent runs past the row it belongs to, and within one Axes those tails draw over the row beneath rather than being covered by its background. Each row's zero line is also the rule between it and the row below, so the overlay marks the boundaries and no separators are needed.

    *row_names* draws the slice names on the left and *scale* the 0–1 α scale on the right; a panel in a row of them wants each only on its own side of the figure, since the rows and the scale are the same for all of them (see :func:`grading_grids`).
    """
    sw = 0.72  # smooth-step plateau width
    n_slices, _, n_pos = a.shape
    red_c = light_dark("#d40000", "#f44")
    grey_c = light_dark("#eee", "#111")
    for si in range(n_slices):
        shift = Affine2D().translate(0, si) + ax.transData  # embedding at the bottom, as the profile figures do
        reds = a[si, I_RED, :]
        xs = np.arange(len(reds))
        smooth_step_area(ax, xs, reds, ramp=1 - sw, color=grey_c, transform=shift, zorder=0)
        for pi in range(n_pos):
            GradingCloud(ax, a[si, :, pi], span=(pi - sw / 2, pi + sw / 2), transform=shift, clip_on=False)
        smooth_step(ax, xs, reds, ramp=1 - sw, color=red_c, lw=1, alpha=1, transform=shift, clip_on=False, zorder=1)
        ax.axhline(si, color=light_dark("#aaaa", "#333a"), lw=0.5)
    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(0, n_slices)
    ax.set_xticks(range(n_pos), POS_NAMES)
    ax.set_yticks(np.arange(n_slices) + 0.5, SLICE_NAMES if row_names else [""] * n_slices)
    ax.tick_params(axis="x", labelsize=6, bottom=False)
    ax.tick_params(axis="y", labelsize=7, left=False)
    if scale:
        # One y scale for the whole figure, against the bottom row, whose offset is zero.
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(np.arange(n_slices + 1), ["0", "1"] + [""] * (n_slices - 1))
        sec.tick_params(labelsize=6, direction="out")


@app.function(hide_code=True)
def grading_grids(panels: dict[str, np.ndarray], figsize: tuple[float, float] = (7.4, 3.3)) -> plt.Figure:
    """A row of grading grids sharing one set of rows and one α scale, titled by *panels*' keys.

    The panels measure the same quantity on the same scale and the reader compares across them, so they belong in one figure — which is also what the post wants to upload, one image per condition. The row names sit on the left of the first panel and the α scale on the right of the last, since both are shared; a lone panel gets each on its own side.

    The default size fits a pair into the content column. A single grid wants a taller figure — closer to square per panel, as the pair is — rather than this one stretched over one panel.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    row = cast(AxesRow, axes[0])
    for i, (ax, (title, a)) in enumerate(zip(row, panels.items(), strict=True)):
        draw_grading_grid(ax, a, row_names=i == 0, scale=i == len(row) - 1)
        ax.set_title(title, fontsize=8, pad=6)  # pad clears the top row's unclipped cloud
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # D2.1: figures for the anchoring post

    Grading-cloud figures for the LessWrong post on D2.1, *Anchoring concepts in transformers*. This page is not an experiment. Every panel is drawn from the published results of [ex-2.1.6](../ex-2.1.6/report.py), [ex-2.1.8](../ex-2.1.8/report.py), and [ex-2.1.10](../ex-2.1.10/report.py), which hold the preregistered claims and their evidence. The post uploads the light `_assets/*.png` files from the bundle of this page.
    """)
    return


@app.cell(hide_code=True)
def _():
    _loaded = load_margins()
    mo.stop(_loaded is None, mo.md("_Results are not published yet; the figures render once they are._"))
    assert _loaded is not None
    alphas: dict[str, dict[str, np.ndarray]] = _loaded[0]
    alpha_stacks: dict[str, dict[str, np.ndarray]] = _loaded[1]
    probe_pairs: np.ndarray = _loaded[2]
    return alpha_stacks, alphas, probe_pairs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The recipe, one piece at a time

    Each panel below adds one piece of the anchoring recipe. The quantity plotted is the α response at the op1 token, drawn against the redness of the first operand.[^alpha] Each chart is a *grading cloud*: the response is measured at the 216 vocabulary colors, interpolated across the whole RGB cube, and drawn as a dither of the colors themselves, so the vertical spread at each redness is the range of responses among equally red colors. The grey band behind each cloud drops the averaging: it redraws the response separately for every residual slice and every seed, flattened to grey so it reads as an envelope rather than data.

    [^alpha]: α is the cosine between the residual stream (the running vector that each layer reads from and writes back to) and the anchor axis, the direction we train the concept toward. Here it is averaged over the five residual slices, the five points in the network where we read that vector.
    """)
    return


@app.cell(hide_code=True)
def _(
    alpha_stacks: dict[str, dict[str, np.ndarray]],
    alphas: dict[str, dict[str, np.ndarray]],
):
    _ys = {c.key: alphas[c.key]["op1"][:, :, 0].mean(axis=0) for c in ex.CONDITIONS}
    _envs = {c.key: alpha_stacks[c.key]["op1"][:, :, :, 0].reshape(-1, len(GRID_RGB)) for c in ex.CONDITIONS}
    # Wide enough for the envelope's full range, since the clouds draw unclipped.
    _ylim = (-0.4, 1.05)

    @themed(
        name="grading-progression",
        alt_text="""
            Four grading clouds against redness, each over a grey per-slice, per-seed envelope. The un-anchored cloud is flat near zero; adding the anchor lifts every color to at least 0.4, with the envelope fanning wide across slices; the anti-subspace term restores a graded ramp from near zero; pooled labels leave the same clean rise to about 0.9 at pure red, the envelope hugging the ramp.
        """,
        caption=r"""
            **The anchoring recipe, one piece at a time.** Each panel is a grading cloud of the α response at op1 (mean over the five residual slices) against the redness of the first operand, drawn over the whole RGB cube; the grey envelope behind it redraws the response separately for every residual slice and seed, so the cloud reads against the spread it summarizes. Left to right: the un-anchored control and the bare anchor at λ = 0.1 (ex-2.1.6); the anchor plus the anti-subspace term at its tuned operating point, with labels still keyed to op1 (ex-2.1.8, `end90-hold30`); and the full recipe, where sequence-level labels name *either* operand and mellowmax pooling finds the red token on its own (the primary condition of ex-2.1.10, `either-t100`).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            1, len(ex.CONDITIONS), figsize=(7.4, 2.2), sharex=True, sharey=True, layout="constrained"
        )
        axes = cast(AxesRow, axes)
        for ax, cond in zip(axes, ex.CONDITIONS, strict=True):
            GradingCloud(
                ax, _envs[cond.key], _ylim, k=20, color=light_dark("#0003", "#fff2"), clip_on=False, zorder=-1, dpr=1
            )
            GradingCloud(ax, _ys[cond.key], _ylim, k=35, dpr=2, clip_on=False)
            ax.set_title(cond.title, fontsize=9)
            ax.set_xlabel("redness of op1", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_aspect("equal")
        axes[0].set_xticks([0, 0.5, 1])
        axes[0].set_yticks([0, 0.5, 1])
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("α at op1, slice mean", fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, dict[str, np.ndarray]]):
    _bare = alphas["ex-2.1.6/lam0.1"]["op1"][:, :, 0].mean(axis=0)
    _primary = alphas[ex.PRIMARY]["op1"][:, :, 0].mean(axis=0)
    mo.md(rf"""
    The bare anchor aligns everything at once: the response rises for every color, and even the least red colors sit at α ≈ {_bare.min():.2f}. Adding the anti-subspace term is what makes the alignment graded, pulling non-red colors back to zero while pure red stays anchored. That grading survives the move to labels that no longer point at a particular token: in the primary condition, the slice-mean response at op1 tracks the sim¹·⁵ affinity shape with r² ≈ {r2_sim(_primary):.2f}.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where the response sits, per (slice, position)

    The clouds above average over the residual slices and look only at op1. Splitting each condition by slice and by token position shows where in the network, and where in the sequence, its response sits.
    """)
    return


@app.cell(hide_code=True)
def _(probe_pairs: np.ndarray):
    # What each color's partners look like: mean redness over the 27 lines it leads.
    _partner_red = np.bincount(probe_pairs[:, 0], REDNESS[probe_pairs[:, 1]]) / np.bincount(probe_pairs[:, 0])
    mo.md(rf"""
    Each condition gets a pair of these, side by side, because the measurement affords two views. The probe set runs every color as op1 against each of the 27 partners it mixes cleanly with, and mixing is symmetric, so every color also sits at op2 on 27 lines. Averaging each line over op2 keys the α map onto the first operand's color; averaging over op1 keys it onto the second. Both are margins of one per-line array, in the statistical sense — each drops an operand by averaging over it — so the colors are re-sorted rather than re-measured, and either way a slot's response is a mean over 27 lines.[^recompute]

    **Read the off-key slot with care.** Wherever a condition responds at all, the slot that is *not* the key tilts upward too, and part of that tilt is the shape of the probe set rather than anything the model does. A pair is on the probe set only if the mix lands back on the color grid, and that closure rule correlates a color's redness with its partners': pure red's partners average {_partner_red[I_RED]:.2f} redness against {REDNESS.mean():.2f} for the palette as a whole. So an off-key slot shows what a color's *partners* do, and only the on-key slot is a response to the key. The embedding row is the clean demonstration — there every state is a function of its own token, so whatever tilt survives in the off-key slot at `emb` is all bookkeeping. The op1 slot repeats that at every slice, because attention is causal: the state above the first token never sees the second operand, so its tilt in the op2-keyed view is partner composition all the way up the stack. The tilt scales with how much response there is to compose from, which is why the un-anchored panels show none of it. Fitted channel probes make the same point without the anchor axis: [ex-2.1.12](../ex-2.1.12/report.py) probes each condition's own color readout and finds no usable op2 information before op2 in any of them, anchored or not.

    [^recompute]: Ex-2.1.10 published its per-line array, so both of its margins are contractions of what it stored. Ex-2.1.6 and ex-2.1.8 averaged theirs over partners before storing, which is the step that drops op2's identity, so their second margin is recomputed from their published checkpoints by `experiment.py` beside this notebook. It checks the op1 margin it recovers against the one they published before publishing anything.
    """)
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, dict[str, np.ndarray]]):
    # Takeaway first: what the figure shows, not an inventory of what is drawn.
    _ALT = {
        "ex-2.1.6/lam0": """
            Nothing anywhere, in either keying: every slice and every token position holds a flat band of color at zero.
        """,
        "ex-2.1.6/lam0.1": """
            The bare anchor lifts the whole sequence: every slot in both panels sits high, with only a mild tilt, so almost nothing is graded by redness. The steepest climb in each panel is at the keyed operand's own slot — op1 on the left, op2 on the right.
        """,
        "ex-2.1.8/end90-hold30": """
            Grading returns, but it fills the whole anchored span: op1, plus, op2 and equals all carry tall red-rising clouds, while the answer and newline slots stay low. The tallest cloud sits at the keyed operand's slot in each panel, with equals close behind, so the pull is shared across the span either way round.
        """,
        "ex-2.1.10/either-t100": """
            The pull has narrowed to one token, and it follows the keying: a sigmoid cloud rises to about 0.97 at the op1 slot in every slice on the left, at the op2 slot on the right, while every other slot stays low in both.
        """,
    }

    def _grids(cond: ex.Condition) -> str:
        @themed(
            name=f"grading-grid-{cond.exp}-{cond.name}",
            alt_text=_ALT[cond.key],
            caption=rf"""
                **{cond.title}** ({cond.exp}, `{cond.name}`). One row per residual slice, with the embedding at the bottom; each slot holds the grading cloud of the α response at that token position. The two panels key the same measurement by a different operand — its redness runs left to right within every slot — and in red, α on the probe lines whose keyed operand is pure red. One y scale, 0 to 1, serves both.
            """,
        )
        def _plot() -> plt.Figure:
            return grading_grids({f"Keyed by the {op} operand": alphas[cond.key][k] for k, op in KEYINGS.items()})

        return _plot()

    for c in mo.status.progress_bar(ex.CONDITIONS, title="Rendering charts", remove_on_exit=True, show_eta=True):
        mo.output.append(mo.md(f"### {c.title}\n\n{_grids(c)}"))
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, dict[str, np.ndarray]]):
    # Position slots, as POS_NAMES orders them: op1 + op2 = ans \n.
    _OP1, _PLUS, _OP2, _EQ, _ANS, _NL = range(6)
    # Mid-stack (slices 1–3) α on the lines whose keyed operand is pure red.
    _red = lambda key, k, t: float(alphas[key][k][1:4, I_RED, t].mean())  # noqa: E731
    _r2 = lambda key, k, t: r2_sim(alphas[key][k][:, :, t].mean(axis=0))  # noqa: E731
    # Drift: the mean over *all* colors at the last slice, not the pure-red row.
    _bulk = lambda key, k, t: float(alphas[key][k][4, :, t].mean())  # noqa: E731
    _flat, _bare, _anti = "ex-2.1.6/lam0", "ex-2.1.6/lam0.1", "ex-2.1.8/end90-hold30"
    mo.md(rf"""
    Read across the eight panels and the recipe narrows *where* the pull lands, in three steps.

    Un-anchored there is nothing to place. No slot in either keying leaves the noise, and the slice-mean response at op1 scores r² = {_r2(_flat, "op1", _OP1):.2f} against the sim¹·⁵ shape.

    **The bare anchor lifts the whole sequence.** With a pure-red first operand, mid-stack α runs {_red(_bare, "op1", _OP1):.2f} at op1, {_red(_bare, "op1", _PLUS):.2f} at the `+` slot, {_red(_bare, "op1", _OP2):.2f} at op2 and {_red(_bare, "op1", _EQ):.2f} at the `=` slot — and it does not stop at the anchored span, reaching {_red(_bare, "op1", _ANS):.2f} at the answer and {_red(_bare, "op1", _NL):.2f} at the newline. Both keyings show the same indiscriminate lift.

    **The anti-subspace term confines it to the span.** Outside the span the response falls away ({_red(_anti, "op1", _ANS):.2f} at the answer, {_red(_anti, "op1", _NL):.2f} at the newline), and the drift over all colors at the last slice drops from {_bulk(_bare, "op1", _OP1):.2f} to {_bulk(_anti, "op1", _OP1):.2f} — the containment ex-2.1.8 was tuned for. Inside the span the pull is still shared: a pure-red op1 brings the op2 slot to {_red(_anti, "op1", _OP2):.2f} and the `=` slot to {_red(_anti, "op1", _EQ):.2f}, and the op2 slot grades against *op1's* redness at r² = {_r2(_anti, "op1", _OP2):.2f}. That is what op1-keyed labels buy: a red token pulls the span it sits in, without the span having to say which token it was.

    **Pooled labels narrow it to the token.** On the primary a pure-red op1 leaves the rest of the span between {min(_red(ex.PRIMARY, "op1", t) for t in (_PLUS, _OP2, _EQ)):.2f} and {max(_red(ex.PRIMARY, "op1", t) for t in (_PLUS, _OP2, _EQ)):.2f}, and a pure-red op2 leaves the op1 slot at {_red(ex.PRIMARY, "op2", _OP1):.2f}. Each operand slot answers to its own color — r² = {_r2(ex.PRIMARY, "op1", _OP1):.2f} at op1 and {_r2(ex.PRIMARY, "op2", _OP2):.2f} at op2, reaching α ≈ {_red(ex.PRIMARY, "op1", _OP1):.2f} and {_red(ex.PRIMARY, "op2", _OP2):.2f} at pure red. So the pull finds the red token in either slot, grades it the same way there, and leaves its neighbours alone, which is the claim the pooled labels were meant to buy.

    Two asymmetries survive on the primary. Op2 carries a little more of the drift at the last slice ({_bulk(ex.PRIMARY, "op2", _OP2):.2f} against {_bulk(ex.PRIMARY, "op1", _OP1):.2f} at op1), matching ex-2.1.10's reading that op1 hosts less of it once half the pull budget goes to op2. The answer slot barely moves between the two keyings (r² {_r2(ex.PRIMARY, "op2", _ANS):.2f} against {_r2(ex.PRIMARY, "op1", _ANS):.2f}): the answer is the mix, so its redness is a function of both operands and neither keying has much claim on it.
    """)
    return


if __name__ == "__main__":
    app.run()
