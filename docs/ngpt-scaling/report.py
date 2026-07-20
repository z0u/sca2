import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="medium",
    app_title="nGPT scaling: flat across the width × depth grid",
    auto_download=["html"],
    css_file="../report.css",
)

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

    Before we build the color-mixing experiments on top of this transformer, we
    want to know that it holds its shape as it grows. This report trains the model
    at a range of sizes and checks that none of them misbehave.

    The model is a simplified version of *nGPT*. The idea behind nGPT is to keep
    the model's running state (the *residual stream*, the vector that each layer
    reads from and writes back to) on the surface of a hypersphere, by normalizing
    it after every step. We keep that residual update,
    `h ← Norm(h + α·(Norm(sub(h)) − h))`, which moves the state `h` a fraction `α`
    of the way toward a sub-module's normalized output and then renormalizes. We
    simplify two pieces of it. Each sub-module gets a single learned *gain* (one
    number that scales its output) in place of nGPT's per-channel *eigen learning
    rates*, and the residual step `α` is fixed at 1/n_layer instead of being learned,
    which is about where the learned version settled anyway.

    For the claim that SCA (the concept-anchoring method this project studies)
    carries over to language models, this pared-down backbone needs to stay
    well-behaved as it scales. So the [experiment](./experiment.py) trains it
    across a grid: three widths (how many numbers are in that state vector)
    crossed with three depths (how many layers), {32, 64, 128} × {4, 8, 12}, with
    everything else held fixed: batch size 16, peak learning rate 10⁻², 100 epochs,
    and *Pride and Prejudice* for the training text. Two outcomes would concern us.
    One is a **depth penalty**, where adding layers at a fixed width makes the
    model worse. The other is an instability that shows up only at large width,
    where a run spikes or fails to train. We hope to see neither.
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
            "No results yet. Run the experiment first; it publishes the loss curves to the store when it finishes:\n\n"
            "```bash\nbin/mini run docs/ngpt-scaling/experiment.py --app modal --max-containers 9\n```"
        ),
    )
    curves = loaded
    # Largest within-a-width spread across depth — the "depth penalty" if there were one.
    depth_spread = max(
        max(plateau(curves, w, d) for d in DEPTHS) - min(plateau(curves, w, d) for d in DEPTHS) for w in WIDTHS
    )
    mo.md(
        f"""
    **The backbone scales cleanly.** We score each run by its converged loss: the
    model's average error at predicting the next character once training has
    settled, measured in *nats per character* (natural-log units, where lower is
    better). That loss never rises as we add layers. At each width, the three
    depths land within {depth_spread:.02f} nats/char of one another, well inside
    the ±0.08 of noise we see between epochs, so the depth axis is flat. Width
    behaves the way added capacity should: loss falls monotonically from
    {plateau(curves, 32, 4):.2f} at width 32 to {plateau(curves, 128, 12):.2f} at
    width 128 with 12 layers. No cell spikes or stalls, and the same learning rate
    (10⁻²) works everywhere. So fixing the residual step to a scalar constant seems
    to be enough.
    """
    )
    return (curves,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Converged loss versus depth

    This chart shows converged validation loss (measured on held-out text and
    averaged over the last 10 epochs) against depth, with one line per width. Read
    each line from left to right: if adding layers hurt, the line would slope up.
    Instead each one is nearly horizontal, so extra depth costs nothing at any
    width. The lines also stack in width order, so a wider model is uniformly
    better, and there is no far corner, wide and deep together, where the loss
    turns back up.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    @themed(
        name="plateau-vs-depth",
        alt_text=(
            "Line chart of converged validation loss against depth (4, 8, and 12 layers), with one line "
            "per width. All three lines are close to flat: width 32 sits near 1.5, width 64 near 1.4, and "
            "width 128 near 1.33, at every depth. Deeper models are no worse than shallow ones at any "
            "width, and each wider model sits uniformly below the narrower ones."
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
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training curves

    The same runs, now shown across training: one panel per width, one line per
    depth, loss on the vertical axis and epoch on the horizontal. Every run
    descends through the learning-rate warmup (the opening stretch of training,
    where the step size ramps up from small) and settles onto a plateau. Within a
    panel, the depth lines sit on top of one another rather than fanning apart, and
    the wider panels settle lower.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    @themed(
        name="convergence",
        alt_text=(
            "Three line charts of validation loss against epoch, one per width (32, 64, and 128), sharing "
            "a y-axis, each with one line per depth (4, 8, and 12 layers). In every panel, all three depth "
            "lines fall together through the first ten epochs of warmup and converge onto a single plateau, "
            "with no spikes or divergence. The plateau drops from about 1.5 in the width-32 panel to about "
            "1.4 at width 64 to about 1.33 at width 128."
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
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(curves):
    _best = plateau(curves, 128, 12)
    mo.md(
        f"""
    ## Findings

    Across the whole grid, the simplified nGPT trains flat across depth and keeps
    improving with width, and no cell destabilizes ({_best:.2f} nats/char at the
    deepest, widest corner). That is what we were hoping for. The backbone that SCA
    will anchor concepts in scales without a depth penalty, so if a later
    experiment runs into trouble, the simplified architecture is unlikely to be the
    reason.

    The grid tops out at width 128 with 12 layers on an L4, which is probably
    fine for M2 (this milestone). For M3 (a future
    milestone), we should confirm that the fixed scalar residual step still holds
    at a genuinely larger size: wider and deeper, on a bigger GPU with a bigger
    batch.
    """
    )
    return


if __name__ == "__main__":
    app.run()
