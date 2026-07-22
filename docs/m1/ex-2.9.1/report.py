import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="medium",
    app_title="Experiment 2.9.1 redux: deleting red, now in JAX",
    auto_download=["html"],
    css_file="../../report.css",
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from sca.colorcube import plot_latent_disc
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.temporal import Dopesheet, ParamGroup, Timeline, plot_timeline, realize_timeline
    from mini.vis import light_dark, themed

    use_publisher(report_bundle(__file__))

    # Store refs published by experiment.py (kept in sync by hand).
    METRICS_REF = "reports/ex-2.9.1/metrics"
    BEST_EVAL_REF = "reports/ex-2.9.1/best-eval"
    DOPESHEET = Path(__file__).parent / "dopesheet.csv"

    def load_results() -> tuple[list[dict], dict[str, np.ndarray]] | None:
        """Resolve per-seed metrics and the best run's eval dump from the store, or None if unpublished."""
        store = project_store()
        metrics_art, best_art = store.get_ref(METRICS_REF), store.get_ref(BEST_EVAL_REF)
        if metrics_art is None or best_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            metrics = json.loads(store.get(metrics_art, Path(d) / "metrics.json").read_text())
            with np.load(store.get(best_art, Path(d) / "best.npz")) as z:
                best = dict(z)
        return metrics, best


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Experiment 2.9.1 redux: deleting *red*, now in JAX

    This is a port of [ex-preppy](https://github.com/z0u/ex-preppy) experiment
    2.9.1 (M1, autoencoders), run as an end-to-end test of this repo's
    infrastructure before the M2 transformer experiments. It re-answers M1's
    question: if Sparse Concept Anchoring pins *red* to one latent axis during
    training, does zeroing that axis delete red, and only red?

    The model is a small autoencoder: it compresses each RGB color down to a
    5-dimensional vector (the bottleneck) and reconstructs the color from that
    vector alone. Training adds four regularizers[^regularizer] on the
    bottleneck, which is unit-normalized so every latent vector lands on the
    surface of a hypersphere: anchor pulls red-labeled samples toward one fixed
    axis (e₀), anti-anchor pushes everything away from the opposite point
    (−e₀), separate spreads samples within a batch apart from each other, and
    anti-subspace pushes everything away from axis 0 in general. The red label
    itself is sparse and noisy: even a pure red sample only gets labeled "red"
    with probability 0.08 per draw.

    After training, we ablate (zero out) axis 0 and reconstruct every color,
    then score the run by how tightly each color's post-ablation
    reconstruction error tracks its HSV similarity to red. The score is R²,
    which runs from 0 (no relationship) to 1 (a perfect linear fit): a value
    near 1 means the deletion was clean, with error scaling with redness and
    colors unlike red left untouched.

    This is a report: it reads results the experiment already produced. The
    experiment itself is [`experiment.py`](./experiment.py), a `main(ctx)` DAG
    driven from the CLI — 16 seeded runs fanned out on Modal CPU containers:

    ```bash
    bin/mini run docs/m1/ex-2.9.1/experiment.py --app modal --max-containers 8
    ```

    The port is lighter than the original: the PyTorch/Lightning machinery
    becomes a pytree and one `lax.scan`, 16 seeds replace 60, validation
    happens only at the end, and we test ablation only (no pruning or
    suppression conditions). The dopesheet is identical to the original, so
    the same `mini.temporal` machinery interprets it unchanged.

    [^regularizer]: An extra term added to the training loss, alongside the
    main reconstruction objective, that shapes the geometry of the latent
    space without changing what the model is fundamentally trying to do.
    """)
    return


@app.cell(hide_code=True)
def _():
    loaded = load_results()
    return (loaded,)


@app.cell(hide_code=True)
def _(loaded):
    mo.stop(
        loaded is None,
        mo.md(
            "No results yet — run the experiment (it publishes metrics to the store on completion):\n\n"
            "```bash\nbin/mini run docs/m1/ex-2.9.1/experiment.py --app modal --max-containers 8\n```"
        ),
    )
    metrics, best_eval = loaded
    best = max(metrics, key=lambda m: m["score"])
    scores = [m["score"] for m in metrics]
    mo.md(f"""
    {len(metrics)} runs completed. Scores range {min(scores):.2f}–{max(scores):.2f}
    (median {np.median(scores):.2f}): anchoring quality depends on the seed, as it did in
    the original. The best run is seed {best["seed"]} with score **{best["score"]:.3f}**, in
    line with the original's best of 0.985 over 60 seeds.
    """)
    return best, best_eval, metrics


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The schedule

    Training uses a [dopesheet](./dopesheet.csv): a table of keyframes for
    the regularizer weights and the learning rate, much like an animator's
    keyframe track. `mini.temporal` interpolates between keyframes with a
    minimum-jerk curve, which eases values in and out. Everything ramps to
    zero by step 1425, leaving the last 75 steps as pure reconstruction
    fine-tuning.
    """)
    return


@app.cell(hide_code=True)
def _():
    _sheet = Dopesheet.from_csv(DOPESHEET)
    _history = realize_timeline(Timeline(_sheet))

    @themed(
        name="schedule",
        alt_text="""
            Two stacked line charts of hyperparameter values against training step. Top: the four
            regularizer weights — anti-subspace starts high at 0.25 and eases down; anchor and
            anti-anchor ramp up over the first 250 steps; all four reach zero by step 1425. Bottom:
            the learning rate ramps from near zero to 0.1 by step 750, holds, then anneals to 0.05.
        """,
    )
    def _plot() -> plt.Figure:
        fig, _ = plot_timeline(
            _history,
            _sheet.as_df(),
            groups=[
                ParamGroup(name="Regularizer weights", params=["separate", "anchor", "anti-anchor", "anti-subspace"]),
                ParamGroup(name="Learning rate", params=["lr"], height_ratio=0.6),
            ],
            title="Hyperparameter schedule (from the original 2.9.1 dopesheet)",
            show_phase_labels=False,  # one phase; the label only collides with the title
        )
        fig.set_size_inches(9.5, 5)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Scores across seeds

    Each run trains from the same schedule but a different seed, and outcomes
    vary substantially with the starting point. That matches the original,
    which handled the variance with a 60-seed sweep and selection along a
    Pareto frontier (the best trade-offs across several objectives). We keep
    the same shape at smaller scale: sweep the seeds, then take the best
    scorer.
    """)
    return


@app.cell(hide_code=True)
def _(best, metrics):
    @themed(
        name="scores",
        alt_text="""
            Dot plot of ablation score by seed for 16 runs. Most seeds score between 0.7 and 1.0,
            a few fall lower, and the best seed is circled at the top of the range.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 3))
        seeds = [m["seed"] for m in metrics]
        scores = [m["score"] for m in metrics]
        ax.scatter(seeds, scores, color="tab:blue", s=30)
        ax.scatter([best["seed"]], [best["score"]], s=140, facecolors="none", edgecolors="tab:red")
        ax.annotate("best", (best["seed"], best["score"]), xytext=(0, -16), textcoords="offset points", ha="center")
        ax.set(xlabel="seed", ylabel="score (R², error vs. similarity)", ylim=(0, 1.05))
        ax.set_xticks(seeds)
        ax.grid(alpha=0.3)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Was the deletion clean?

    For the best run, we ablate latent axis 0 and reconstruct the full 8×8×8
    RGB grid. If the anchored concept was fully contained in that axis, the
    damage should be proportional to each color's similarity to red, and colors
    unlike red should be untouched.
    """)
    return


@app.cell(hide_code=True)
def _(best, best_eval):
    @themed(
        name="error-vs-similarity",
        alt_text="""
            Scatter plot of post-ablation reconstruction error against cubed HSV similarity to red,
            one point per grid color, each drawn in its own color. Points fall close to a straight
            line from the origin: gray, blue, and green points cluster at zero error, while red and
            red-adjacent points climb to the highest errors.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 4))
        edge = light_dark("#00000033", "#ffffff55")
        ax.scatter(best_eval["sim3"], best_eval["mse_abl"], c=best_eval["rgb"], s=26, edgecolors=edge, lw=0.5)
        r2 = best["score"]
        ax.text(0.05, 0.92, f"$R^2$ = {r2:.3f}", transform=ax.transAxes)
        ax.set(xlabel="similarity to red (angular HSV, cubed)", ylabel="reconstruction MSE after ablation")
        ax.grid(alpha=0.3)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(best_eval):
    _mse_base, _mse_abl = best_eval["mse_base"], best_eval["mse_abl"]
    _sim = best_eval["sim3"]
    _reds, _others = _sim > 0.5, _sim < 0.01
    mo.md(f"""
    Baseline reconstruction is near-perfect (mean MSE {_mse_base.mean():.2e}). After ablation,
    red-like colors (similarity³ > 0.5) have a mean error of {_mse_abl[_reds].mean():.3f}, while
    colors unlike red (similarity³ < 0.01, {int(_others.sum())} of {len(_sim)} grid points) sit
    at {_mse_abl[_others].mean():.2e}: deleting red barely affects them.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The latent geometry

    The same result shows up in the bottleneck geometry. Before
    ablation, position along axis 0 tracks similarity to red: pure red sits
    at 1, blues, greens, and grays sit near 0 (pushed off the axis by the
    anti-subspace term), and warm colors fall in between. Ablation zeroes
    axis 0, which collapses that one direction but leaves the arrangement of
    the other dimensions intact. That's why the effect on reconstruction is
    proportional to redness.
    """)
    return


@app.cell(hide_code=True)
def _(best_eval):
    @themed(
        name="latents",
        alt_text="""
            Two disc-shaped scatter plots of bottleneck latents, one point per grid color, each drawn
            in its own color, inside a circle marking the unit hypersphere bound. The anchored axis
            points up, labeled (1, 0, 0, 0, 0). Left, baseline: blues, greens, and grays hug the
            horizontal diameter, warm colors spread upward, and pure reds reach the top of the circle.
            Right, ablated: every point sits exactly on the horizontal diameter, the reds folded in
            among the other colors.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.6))
        fg = light_dark("#000", "#fff")
        for ax, z, title in zip(axes, (best_eval["z_base"], best_eval["z_abl"]), ("Baseline", "Ablated"), strict=True):
            plot_latent_disc(ax, z, best_eval["rgb"], s=22)
            ax.set_title(title, y=-0.12)
        axes[0].plot([0], [1.05], marker="v", color=fg, clip_on=False)
        axes[0].annotate(
            "(1, 0, 0, 0, 0)",
            xy=(0, 1.1),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            annotation_clip=False,
        )
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings

    The result reproduces M1's finding, but the point of this run was the
    plumbing: the same dopesheet driving a JAX training loop through
    `mini.temporal`; a 16-way `ctx.map` fan-out on Modal with memoized,
    resumable records; artifacts and refs flowing through the store to this
    report; and `@themed` figures published via the report bundle. Training a
    single seed takes about ten seconds, and the runs are bit-identical
    between local CPU and Modal. So, the infrastructure seems to work.

    Next: the M2 experiments proper, anchoring concepts in a small transformer's
    residual stream on the color-mixing task (see the [README](../../README.md)).
    """)
    return


if __name__ == "__main__":
    app.run()
