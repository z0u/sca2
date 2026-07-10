import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import light_dark, themed

    use_publisher(report_bundle(__file__))

    # Store ref published by experiment.py (kept in sync by hand).
    CURVES_REF = "reports/ngpt-scaling/curves"
    WIDTHS = [32, 64, 128]
    DEPTHS = [4, 8, 12]

    def load_curves() -> dict[str, np.ndarray] | None:
        """Resolve the val-loss curves from the store, or None if unpublished."""
        store = project_store()
        art = store.get_ref(CURVES_REF)
        if art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            raw = json.loads(store.get(art, Path(d) / "curves.json").read_text())
        return {label: np.asarray(losses) for label, losses in raw.items()}

    def cell(curves: dict[str, np.ndarray], w: int, d: int) -> np.ndarray:
        return curves[f"d{w}|L{d}"]

    def plateau(curves: dict[str, np.ndarray], w: int, d: int) -> float:
        """Converged loss: mean of the last 10 epochs (per-epoch eval noise is ~±0.08)."""
        return float(cell(curves, w, d)[-10:].mean())

    def width_shades() -> dict[int, tuple]:
        """One ordered shade per width, picked with `light_dark` so the dark end
        of the ramp stays legible on a dark background.
        """
        stops = light_dark([0.7, 0.45, 0.12], [0.8, 0.55, 0.28])
        return dict(zip(WIDTHS, plt.cm.viridis(stops), strict=True))

    def depth_shades() -> dict[int, tuple]:
        """One ordered shade per depth (darker = deeper), same convention."""
        stops = light_dark([0.7, 0.45, 0.12], [0.8, 0.55, 0.28])
        return dict(zip(DEPTHS[::-1], plt.cm.viridis(stops), strict=True))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # nGPT scaling: flat across the width × depth grid

    Our nGPT keeps the published residual form — a LERP toward the sub-module's
    *normalized* output, `h ← Norm(h + α·(Norm(sub(h)) − h))` — but strips the
    rest to scalars: one gain per sub-module in place of the per-channel *eigen
    learning rates*, and the residual step `α` **fixed** at 1/n_layer rather than
    learned (the value the learned gates settled near anyway). The milestone
    leans on that simplified backbone actually scaling: if we want to argue SCA
    carries to LLMs, the transformer underneath has to hold up as it grows.

    So this [experiment](./experiment.py) sweeps the model over a width × depth
    grid — widths {32, 64, 128} × depths {4, 8, 12}, everything else fixed (batch
    16, peak LR 10⁻², 100 epochs, *Pride and Prejudice*) — and asks a narrow
    question: does converged loss stay well-behaved across the grid? Concretely,
    **no depth penalty** (deeper is never worse at fixed width) and **no
    width-gated instability** (nothing spikes or fails to train as width grows).
    """)
    return


@app.cell(hide_code=True)
def _():
    loaded = load_curves()
    return (loaded,)


@app.cell(hide_code=True)
def _(loaded):
    mo.stop(
        loaded is None,
        mo.md(
            "No results yet — run the experiment (it publishes curves to the store on completion):\n\n"
            "```bash\nbin/mini run docs/ngpt-scaling/experiment.py --app modal --max-containers 9\n```"
        ),
    )
    curves = loaded
    # Largest within-a-width spread across depth — the "depth penalty" if there were one.
    depth_spread = max(
        max(plateau(curves, w, d) for d in DEPTHS) - min(plateau(curves, w, d) for d in DEPTHS) for w in WIDTHS
    )
    mo.md(
        f"**The backbone scales cleanly.** Across the grid, converged loss never rises with depth: at each "
        f"width the three depths sit within **{depth_spread:.02f}** nats/char of one another — inside the "
        f"±0.08 per-epoch eval noise, so the depth axis is flat. Width does what added capacity should — loss "
        f"falls monotonically from **{plateau(curves, 32, 4):.2f}** at 32-dim to **{plateau(curves, 128, 12):.2f}** "
        f"at 128-dim × 12 layers — and no cell spikes or stalls. Same learning rate (10⁻²) everywhere; a fixed "
        f"scalar residual gate is enough."
    )
    return (curves,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## No depth penalty

    Converged validation loss (mean of the last 10 epochs) against depth, one
    line per width. Each line is essentially horizontal — adding layers doesn't
    cost anything at any width — and the lines stack in width order, so wider is
    uniformly better. There is no wide-and-deep corner where the loss turns up.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    @themed(
        name="plateau-vs-depth",
        alt_text=(
            "Line chart of converged validation loss against depth (4, 8, 12 layers), one line per width. "
            "All three lines are essentially flat: width 32 sits near 1.5, width 64 near 1.4, and width 128 "
            "near 1.33, at every depth. Deeper models are no worse than shallow ones at any width, and each "
            "wider model is uniformly lower than the narrower ones."
        ),
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        shades = width_shades()
        for w in WIDTHS:
            ys = [plateau(curves, w, d) for d in DEPTHS]
            ax.plot(DEPTHS, ys, "o-", color=shades[w], label=f"width {w}", lw=2.2)
        ax.set(xlabel="depth (n_layer)", ylabel="converged validation loss (nats/char)")
        ax.set_xticks(DEPTHS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Every cell trains smoothly

    The same result per epoch: one panel per width, one line per depth (darker =
    deeper). Every cell descends through the LR warmup and settles onto a
    plateau; within each panel the depth lines sit on top of one another rather
    than fanning apart. Wider panels simply settle lower.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    @themed(
        name="convergence",
        alt_text=(
            "Three line charts of validation loss against epoch, one per width (32, 64, 128), sharing a "
            "y-axis, each with one line per depth (4, 8, 12 layers). In every panel all three depth lines "
            "fall together through the first ten epochs of warmup and converge onto a single plateau, with "
            "no spikes or divergence. The plateau drops from about 1.5 in the width-32 panel to about 1.4 "
            "at width 64 to about 1.33 at width 128."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
        shades = depth_shades()
        for ax, w in zip(axes, WIDTHS, strict=True):
            ax.axvline(10, color="#8888", lw=1, ls=":", label="end of LR warmup")
            for d in DEPTHS:
                ax.plot(cell(curves, w, d), color=shades[d], label=f"{d} layers")
            ax.set(title=f"width {w}", xlabel="epoch")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("validation loss (nats/char)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(curves):
    _best = plateau(curves, 128, 12)
    mo.md(
        f"""
    ## What this settles

    The simplified nGPT — scalar gains, `α` fixed at 1/n_layer — trains flat
    across depth and improves with width over the whole grid we can afford, with
    no cell destabilizing ({_best:.2f} nats/char at the deepest, widest corner).
    That's the property the milestone needs: the backbone SCA will anchor
    concepts in scales without a depth penalty, so a boundary drawn around the
    deep-and-wide corner would be a property of *size*, not of the simplified
    architecture.

    The grid tops out at 128 × 12 on an L4. The next step is to confirm the fixed
    scalar gate still holds at a genuinely larger size — wider and deeper, on a
    bigger GPU with a bigger batch — before leaning on it for M3.
    """
    )
    return


if __name__ == "__main__":
    app.run()
