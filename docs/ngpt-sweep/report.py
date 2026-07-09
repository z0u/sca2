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
    CURVES_REF = "reports/ngpt-sweep/curves"
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

    def cell(curves: dict[str, np.ndarray], width: int, depth: int) -> np.ndarray:
        return curves[f"d{width}|L{depth}"]

    def plateau(curves: dict[str, np.ndarray], width: int, depth: int) -> float:
        """Converged loss: mean of the last 10 epochs (per-epoch eval noise is ~±0.08)."""
        return float(cell(curves, width, depth)[-10:].mean())

    def depth_colors() -> dict[int, tuple]:
        """Depth is ordinal: ordered viridis shades (light → dark with depth).

        Theme-dependent (call inside a ``themed`` plot fn): the deepest shade is
        lifted in dark mode so it keeps contrast against a black background.
        """
        stops = light_dark([0.75, 0.45, 0.1], [0.8, 0.5, 0.25])
        return dict(zip(DEPTHS, plt.cm.viridis(stops), strict=True))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # nGPT hyperparameter sweep

    Before the M2 experiments, we simplified our nGPT variant: the per-channel
    "eigen learning rates" of the published recipe became plain scalar gains,
    and the residual step size — which the earlier architecture sweep showed
    settling near 1/n_layer when learned — is now a constant 1/n_layer. This
    sweep settles the remaining size defaults and checks that simplification:
    3 widths × 3 depths, a character-level model trained on *Pride and
    Prejudice*. Because the residual step is tied to depth by that fixed rule,
    the depth axis doubles as the test — if 1/n_layer is wrong, deeper models
    should train conspicuously worse.

    This is a **report**: it reads results the experiment already produced.
    The experiment is [`experiment.py`](./experiment.py), a `main(ctx)` DAG
    driven from the CLI — one CPU prep step, then nine training runs fanned
    out on Modal L4s:

    ```bash
    bin/mini run docs/ngpt-sweep/experiment.py --app modal --max-containers 9
    ```

    Everything else is held fixed: 8 heads × 8 dims per head, `n_ff` = 4×width,
    context 512, batch 16, 100 epochs with a 10-epoch warmup, peak LR 10⁻²
    (nGPT proved insensitive to LR across 3×10⁻³–4×10⁻² in the earlier sweep,
    so it isn't an axis).
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
            "```bash\nbin/mini run docs/ngpt-sweep/experiment.py --app modal --max-containers 9\n```"
        ),
    )
    curves = loaded
    plateaus = {(w, d): plateau(curves, w, d) for w in WIDTHS for d in DEPTHS}
    (best_w, best_d), best = min(plateaus.items(), key=lambda kv: kv[1])
    mo.md(
        f"**{len(curves)} runs completed.** Converged validation loss ranges "
        f"{min(plateaus.values()):.2f}–{max(plateaus.values()):.2f} nats/char. The best cell is "
        f"**d{best_w}|L{best_d}** at {best:.2f} nats/char ({best / np.log(2):.2f} bits/char) — "
        f"and the worst, d128|L12, is not merely worse: it failed to train (see below)."
    )
    return curves, plateaus


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Convergence

    Validation loss per epoch, one panel per width. Within a panel the three
    depths share a color scale (darker = deeper), so a depth effect shows as
    separation between the lines — and it does, dramatically, at width 128.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    @themed(
        name="curves",
        alt_text=(
            "Three line charts of validation loss against epoch, one per model width (32, 64, 128), "
            "each with three lines for depths 4, 8, and 12 drawn light to dark. At widths 32 and 64 "
            "all curves fall steeply and flatten near 1.4 to 1.6, the depth lines nearly coinciding. "
            "At width 128 they split: 4 layers reaches the lowest loss of all, 8 layers spikes near "
            "epoch 10 before recovering, and 12 layers gets stuck at about 3.1 and never comes down."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True)
        colors = depth_colors()
        for ax, width in zip(axes, WIDTHS, strict=True):
            ax.axvline(10, color="#8888", lw=1, ls=":", label="end of LR warmup")
            for depth in DEPTHS:
                ax.plot(cell(curves, width, depth), color=colors[depth], label=f"{depth} layers")
            ax.set(title=f"width {width}", xlabel="epoch")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("validation loss (nats/char)")
        axes[0].legend()
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Width and depth

    The same data reduced to endpoints: converged validation loss (mean of the
    last 10 epochs — per-epoch eval is noisy at ~±0.08) for each cell. The
    depth lines fanning apart with width is the failure signature the sweep
    was designed to expose.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    @themed(
        name="final-loss",
        alt_text=(
            "Line chart of converged validation loss against model width (32, 64, 128 on a log "
            "axis), one line per depth (4, 8, 12) drawn light to dark. The lines coincide at width "
            "32 and fan apart as width grows: 4 layers keeps improving, 8 layers turns upward at "
            "width 128, and 12 layers climbs to about 3.1 — the depth penalty grows with width."
        ),
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        colors = depth_colors()
        for depth in DEPTHS:
            ax.plot(
                WIDTHS,
                [plateau(curves, w, depth) for w in WIDTHS],
                "o-",
                color=colors[depth],
                label=f"{depth} layers",
            )
        ax.set(xlabel="width (n_embd)", ylabel="converged validation loss (nats/char)", xscale="log")
        ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(plateaus):
    _spread = {w: max(plateaus[w, d] for d in DEPTHS) - min(plateaus[w, d] for d in DEPTHS) for w in WIDTHS}
    mo.md(
        f"Two regimes. At 4 layers, width buys what it should: "
        f"{plateaus[32, 4]:.2f} → {plateaus[64, 4]:.2f} → {plateaus[128, 4]:.2f} nats/char from "
        f"width 32 → 128. But the depth spread grows with width — "
        f"{_spread[32]:.2f}, {_spread[64]:.2f}, then {_spread[128]:.2f} nats/char — ending in "
        f"outright failure: d128|L12 bottoms out at epoch 5, mid-warmup, and sits at ~{plateaus[128, 12]:.1f} "
        f"for the remaining 90 epochs. d128|L8 spikes right as the warmup ends and the LR reaches its "
        f"peak, then recovers to a worse plateau than d128|L4. The timing points at optimization, not "
        f"capacity: with the residual step pinned to 1/n_layer, deeper-and-wider models can't absorb "
        f"the peak LR of 10⁻². The earlier sweep's LR-insensitivity was measured at width 32 — the "
        f"column that is depth-flat here too — and evidently does not transfer to wide models. "
        f"Whether the cure is a lower LR or the full recipe's learnable gates, that corner of the "
        f"grid needs one before it's usable."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What this settles

    Defaults for the M2 color-mixing experiments: **shallow and as wide as
    the budget allows** — 4 layers improves monotonically with width, depth
    buys nothing at any width tried, and the best cell overall is d128|L4.
    The simplification (scalar gains, residual step fixed at 1/n_layer) is
    safe in exactly that regime, which is the regime M2 needs; the deep-wide
    corner is a known boundary to respect, not a mystery to solve now. That
    the depth axis caught a real failure mode is the check working as
    designed.

    As with 2.9.1, part of the point was the plumbing: a prep → 9-way GPU
    `ctx.map` → publish DAG on Modal, curves flowing to this report through a
    named store ref — and the recovery loop earning its keep when the largest
    cell outran its 12-minute task timeout (cancel, raise the role timeout,
    retry: one cell re-ran, eight were memo hits).
    """)
    return


if __name__ == "__main__":
    app.run()
