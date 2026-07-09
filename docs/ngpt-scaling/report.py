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
    WIDTH = 128  # the failing width; the recipe comparison is run here
    WIDTHS = [32, 64, 128]  # base & norm sweep width to expose the width-gating
    DEPTHS = [4, 8, 12]
    # (arm key, label) in reading order; `base` is the failing recipe, `norm` the fix.
    ARMS = [
        ("base", "additive (raw sub·)"),
        ("sqrt", "additive, α=1/√L"),
        ("lrn", "additive, learnable α"),
        ("lr3e3", "additive, LR 3×10⁻³"),
        ("norm", "normalized LERP (fix)"),
    ]

    def load_curves() -> dict[str, np.ndarray] | None:
        """Resolve the val-loss curves from the store, or None if unpublished."""
        store = project_store()
        art = store.get_ref(CURVES_REF)
        if art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            raw = json.loads(store.get(art, Path(d) / "curves.json").read_text())
        return {label: np.asarray(losses) for label, losses in raw.items()}

    def cell(curves: dict[str, np.ndarray], arm: str, depth: int) -> np.ndarray:
        return curves[f"{arm}|d{WIDTH}|L{depth}"]

    def plateau(curves: dict[str, np.ndarray], arm: str, depth: int) -> float:
        """Converged loss: mean of the last 10 epochs (per-epoch eval noise is ~±0.08)."""
        return float(cell(curves, arm, depth)[-10:].mean())

    def arm_colors() -> dict[str, str]:
        """One hue per recipe: the fix (`norm`) a confident teal, the failing
        additive arms warm/greyed, so the split reads at a glance.
        """
        return dict(
            base=light_dark("#d1495b", "#e06c7d"),  # red — the bug
            sqrt=light_dark("#e8963a", "#f0a860"),  # amber
            lrn=light_dark("#8a6fb0", "#a78bc9"),  # muted violet
            lr3e3=light_dark("#8d99ae", "#a7b1c2"),  # grey — the band-aid
            norm=light_dark("#1b998b", "#2ec4b6"),  # teal — the fix
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # nGPT scaling: a residual bug, found and fixed

    We simplified our nGPT residual to a fixed *additive* step —
    `h ← Norm(h + α·sub(h))` with `α = 1/n_layer` — and a width × depth sweep
    exposed a failure: at width 128 the model degrades with depth (`d128|L8`
    spikes and recovers worse, `d128|L12` never trains). It's tempting to read
    that as "the peak learning rate is too high for deep-and-wide models," but
    that's a symptom. The cause is a geometry bug in the residual step, and
    once fixed the depth penalty disappears.

    The additive step assumes `sub(h)` has norm ≈ 1. It doesn't: the MLP scales
    its pre-activations by a √n_embd baseline (to keep GELU in range), so
    `‖MLP(h)‖ ∝ √n_embd` — ≈ 7, 10, 15 at width 32, 64, 128. The *effective*
    step is `α·‖sub(h)‖`, which **grows with width**, so the fixed gate never
    controlled the rotation it was supposed to. nGPT proper avoids this by
    stepping toward the **normalized** output,
    `h ← Norm(h + α·(Norm(sub(h)) − h))`, making `α` a true interpolation
    fraction. We had stripped that normalization out.

    This report reads results the [experiment](./experiment.py) already
    produced, over two axes (everything else fixed: batch 16, 100 epochs,
    *Pride and Prejudice*):

    - **Recipe**, at the failing width 128 × depths {4, 8, 12}: the buggy
      additive step (`base`) against four candidate cures.
    - **Width**, for `base` and the fix `norm` × widths {32, 64, 128}: showing
      the failure is width-gated and the fix is width-flat.
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
            "```bash\nbin/mini run docs/ngpt-scaling/experiment.py --app modal --max-containers 27\n```"
        ),
    )
    curves = loaded
    base_l12 = plateau(curves, "base", 12)
    norm_spread = max(plateau(curves, "norm", d) for d in DEPTHS) - min(plateau(curves, "norm", d) for d in DEPTHS)
    mo.md(
        f"**The fix flattens the depth axis.** The `base` recipe fails as the sweep found — "
        f"converged loss climbs to **{base_l12:.2f}** nats/char at `d128|L12`. The normalized-LERP fix "
        f"(`norm`) converges to **{plateau(curves, 'norm', 12):.2f}** at the same depth — and its spread "
        f"across all three depths is just {norm_spread:.02f} nats/char. Same learning rate (10⁻²), same "
        f"everything else; one line of geometry."
    )
    return (curves,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The failure and the fix

    Converged validation loss (mean of the last 10 epochs) against depth, one
    line per recipe. The three *additive* arms — the raw-output step, whether
    at `α = 1/√L`, with a learnable `α`, or nothing changed — all blow up at
    12 layers. Only two things hold the line: lowering the LR (a band-aid: flat
    but worse everywhere), and normalizing the sub-module output (the fix: flat
    *and* best at every depth).
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    @themed(
        name="plateau-vs-depth",
        alt_text=(
            "Line chart of converged validation loss against depth (4, 8, 12 layers) at width 128, one "
            "line per recipe. The three additive recipes (raw step, 1/√L step, learnable α) sit near 1.4 "
            "at 4 layers and climb steeply to about 3.1 at 12 layers. The lower-LR additive recipe stays "
            "flat near 1.45–1.5. The normalized-LERP fix stays flat and lowest, near 1.33 at every depth."
        ),
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        colors = arm_colors()
        for arm, label in ARMS:
            ys = [plateau(curves, arm, d) for d in DEPTHS]
            lw = 2.6 if arm in ("norm", "base") else 1.6
            ax.plot(DEPTHS, ys, "o-", color=colors[arm], label=label, lw=lw, zorder=3 if arm == "norm" else 2)
        ax.set(xlabel="depth (n_layer)", ylabel="converged validation loss (nats/char)")
        ax.set_xticks(DEPTHS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center left")
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Convergence

    The same story per epoch, for the two recipes that matter: the failing
    `base` (left) and the `norm` fix (right), one line per depth (darker =
    deeper). Under `base`, depth 12 collapses in the warmup and sits at ~3.1
    for the rest of training; depth 8 recovers to a worse plateau. Under the
    fix, all three depths track together.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    def depth_shades(hi: str, lo: str) -> dict[int, tuple]:
        stops = light_dark([0.35, 0.6, 0.9], [0.45, 0.68, 0.95])
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("d", [lo, hi])
        return dict(zip(DEPTHS, cmap(stops), strict=True))

    @themed(
        name="convergence",
        alt_text=(
            "Two line charts of validation loss against epoch at width 128, sharing a y-axis. Left, the "
            "additive `base` recipe: depths 4 and 8 fall to about 1.4 and 1.7, but depth 12 rises during "
            "warmup and stays stuck near 3.1 for all 100 epochs. Right, the normalized-LERP fix: depths "
            "4, 8, and 12 all fall together and converge near 1.33, with no depth penalty."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
        panels = [
            ("base", "additive (raw sub·)", "#d1495b", "#f0b9c1"),
            ("norm", "normalized LERP (fix)", "#1b998b", "#a9e0d8"),
        ]
        for ax, (arm, title, hi, lo) in zip(axes, panels, strict=True):
            shades = depth_shades(hi, lo)
            ax.axvline(10, color="#8888", lw=1, ls=":", label="end of LR warmup")
            for depth in DEPTHS:
                ax.plot(cell(curves, arm, depth), color=shades[depth], label=f"{depth} layers")
            ax.set(title=title, xlabel="epoch")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("validation loss (nats/char)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The failure is width-gated

    If the diagnosis is right — the effective step is `α·‖sub(h)‖` and
    `‖MLP(h)‖ ∝ √n_embd` — then the additive recipe should be fine at small
    width and break only as width grows, while the fix should be flat at every
    width. Running `base` and `norm` across widths {32, 64, 128}, that is
    exactly what happens: `base`'s depth lines sit together until width 128,
    where they fan apart and `L12` diverges; `norm` stays flat throughout.
    """)
    return


@app.cell(hide_code=True)
def _(curves):
    def plat(arm: str, w: int, d: int) -> float:
        return float(curves[f"{arm}|d{w}|L{d}"][-10:].mean())

    @themed(
        name="width-gating",
        alt_text=(
            "Two line charts of converged validation loss against width (32, 64, 128 on a log axis), "
            "sharing a y-axis, one line per depth (4, 8, 12). Left, the additive `base` recipe: the three "
            "depth lines coincide near 1.4 at widths 32 and 64, then fan apart at width 128 where 12 "
            "layers jumps to about 3.1. Right, the normalized-LERP fix: all three depth lines stay flat "
            "and low, near 1.33, across every width."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
        stops = light_dark([0.7, 0.45, 0.12], [0.8, 0.55, 0.28])
        shades = dict(zip(DEPTHS, plt.cm.viridis(stops), strict=True))
        for ax, (arm, title) in zip(
            axes, [("base", "additive (raw sub·)"), ("norm", "normalized LERP (fix)")], strict=True
        ):
            for depth in DEPTHS:
                ax.plot(
                    WIDTHS, [plat(arm, w, depth) for w in WIDTHS], "o-", color=shades[depth], label=f"{depth} layers"
                )
            ax.set(title=title, xlabel="width (n_embd)", xscale="log")
            ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
            ax.minorticks_off()
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("converged validation loss (nats/char)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(curves):
    _b = {d: plateau(curves, "base", d) for d in DEPTHS}
    _n = {d: plateau(curves, "norm", d) for d in DEPTHS}
    mo.md(
        f"""
    ## Why the additive step fails, and why normalizing fixes it

    At initialization the recipes are already geometrically different. Summing
    the per-layer rotation of the hidden state through the stack (degrees,
    input → output of each block):

    | recipe | d32·L4 | d32·L12 | d128·L4 | d128·L12 |
    |---|--:|--:|--:|--:|
    | additive (raw sub·), α=1/L | 246° | 370° | 298° | **615°** |
    | additive, α=1/√L | 311° | 773° | 328° | **917°** |
    | normalized LERP, α=1/L | 100° | 85° | 103° | **88°** |

    The additive step's travel grows with **both** width and depth — at
    `d128|L12` the hidden state tumbles through 615°, overwriting the token
    identity many times over. Normalizing the sub-module output pins the step
    to the interpolation fraction `α`, and total travel holds near 90°
    *independent of scale*. The `1/√L` idea makes it worse, not better: a bigger
    step on an already-unbounded additive update. That matches the trained
    result — additive `d128|L12` converges to {_b[12]:.2f} nats/char, the
    normalized fix to {_n[12]:.2f}.

    Two corollaries worth stating, because they rule out the tempting
    almost-fixes:

    - **It isn't the learning rate.** Lowering the peak LR to 3×10⁻³ *stabilizes*
      the additive recipe but plateaus worse at every depth ({plateau(curves, "lr3e3", 12):.2f}
      at L12 vs the fix's {_n[12]:.2f}). nGPT's reputed LR-insensitivity holds
      at width 32 — the column that is depth-flat here too — and simply doesn't
      transfer to wide models.
    - **A learnable α doesn't rescue it.** Making the scalar step learnable
      (init 1/L) still adds the raw output, and still fails at L12
      ({plateau(curves, "lrn", 12):.2f}) — the normalization is the operative
      change, not the gate's learnability.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What this settles

    The normalized LERP (`h ← Norm(h + α·(Norm(sub(h)) − h))`) is now the model
    default; the raw-additive step survives only as an opt-in to reproduce the
    failure above. The simplifications we *do* keep — a single scalar gain in
    place of per-channel eigen learning rates, and `α` fixed at 1/n_layer
    rather than learned — are unaffected: the fix is flat *and* best at every
    depth, so a fixed scalar gate remains enough.

    This also lifts the boundary the failure seemed to draw around the
    deep-and-wide corner. That corner was a bug, not a property of the simplified
    architecture — which matters for the milestone: if we want to argue SCA
    should carry to LLMs, the transformer underneath has to actually scale, and
    now it does across the width×depth grid we can afford. The next step is to
    confirm the fixed scalar gate holds at a genuinely larger size (wider and
    deeper than 128×12, on a bigger GPU) before leaning on it for M3.
    """)
    return


if __name__ == "__main__":
    app.run()
