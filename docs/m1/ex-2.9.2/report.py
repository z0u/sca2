import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="medium",
    app_title="Experiment 2.9.2: fallback control for deleting red",
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
    from mini.vis import light_dark, themed

    use_publisher(report_bundle(__file__))

    # Store refs published by experiment.py (kept in sync by hand).
    METRICS_REF = "reports/ex-2.9.2/metrics"
    EXEMPLAR_REFS = {"base": "reports/ex-2.9.2/exemplar-base", "fallback": "reports/ex-2.9.2/exemplar-fallback"}
    INTERVENTIONS = ("zero", "oa", "oa-nontarget", "reflect", "redirect")
    VARIANTS = ("base", "fallback")

    def load_results() -> tuple[list[dict], dict[str, dict[str, np.ndarray]]] | None:
        """Resolve per-run metrics and the exemplar eval dumps from the store, or None if unpublished."""
        store = project_store()
        arts = {}
        for v, ref in EXEMPLAR_REFS.items():
            if (art := store.get_ref(ref)) is None:
                return None
            arts[v] = art
        metrics_art = store.get_ref(METRICS_REF)
        if metrics_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            metrics = json.loads(store.get(metrics_art, Path(d) / "metrics.json").read_text())
            exemplars = {}
            for v, art in arts.items():
                with np.load(store.get(art, Path(d) / f"{v}.npz")) as z:
                    exemplars[v] = dict(z)
        return metrics, exemplars


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Experiment 2.9.2: fallback control for deleting *red*

    [Ex-2.9.1](../ex-2.9.1/report.py) reproduced the main M1 result: anchor
    *red* to latent axis 0, zero the axis, and the reconstruction error
    concentrates on red-like colors. It also reproduced the main weakness: the
    outcome varies a lot with the random seed. The original study handled that
    with a 60-seed sweep and Pareto selection, but that won't scale: as models
    grow, the chance that any given seed allows clean intervention may shrink.
    So this experiment tries a different approach.

    The SCA paper suggests optimal ablation
    ([Li & Janson 2024](https://arxiv.org/abs/2409.09951)) as a fix: replace the
    removed component with an optimized constant. We test that and also a
    training-time alternative we'll call "fallback control". The plan is to
    teach the decoder, during training, what to output once the concept has been
    removed.

    ## Why weight ablation is noisy

    The bottleneck is unit-normalized, so zeroing an encoder axis renormalizes
    whatever is left. For a red-like input the remainder is small and close to
    random, just the residual geometry that this particular seed happened to
    produce, so ablated red is decoded as some other arbitrary color. Li &
    Janson call this "spoofing": the intervention removes the concept and
    inserts a random claim about the input. Two seeds with identical anchoring
    quality can then score very differently, purely on where red happens to
    land.

    Optimal ablation measures importance while minimizing spoofing. It replaces
    the removed component with the constant $a^* = \arg\min_a \mathbb{E}[\mathcal{L}]$,
    the value that affects loss the least. But for concept removal, that
    objective points the wrong way: the loss it minimizes includes the target's
    own loss, so $a^*$ gets pulled toward whatever constant best restores *red*.
    We evaluate both the literal method (`oa`) and a removal-appropriate version
    (`oa-nontarget`, which optimizes the constant over non-red colors only).

    Fallback control instead makes the post-removal behavior
    a trained property. The anti-anchor regularizer already keeps the direction
    −e₀ empty, so we add one decoder-only loss term, MSE(dec(−e₀), mid-gray).
    This pins that reserved direction to a *null* output of our choosing. A
    model that has genuinely lost the color information should fall back to a
    noncommittal guess, and the least committal guess over the RGB cube is
    mid-gray, so we pick mid-gray as *null*. An intervention can then redirect
    red to −e₀ and get a predictable response.

    ## Conditions

    We run two training variants, otherwise identical to ex-2.9.1 (same model,
    data, dopesheet), with 32 seeds each: `base` (ex-2.9.1's loss, unchanged)
    and `fallback` (the same loss plus 0.05 × the fallback term). Each trained
    model is then scored under five weight-level interventions on axis 0:

    - `zero` — zero the encoder's output row 0 and bias (the status quo).
    - `oa` — zero row 0, then set the bias to the constant that minimizes mean
      reconstruction error over the full grid (optimal ablation, taken
      literally).
    - `oa-nontarget` — the same as `oa`, but the constant is optimized over
      non-red colors only.
    - `reflect` — negate row 0 and bias, so z₀ → −z₀ and red lands exactly on
      −e₀. This is a clean redirect, though not a true deletion, since a sign
      flip undoes it.
    - `redirect` — zero row 0 and set the bias to −1. The redness computation
      is deleted, and inputs are also nudged toward −e₀ in proportion to how
      much of their pre-norm activation the deletion removed. This is
      permanent, like `zero`, but with a defined destination.

    Each run is scored like M1 Ex-2.9.1: the R² between per-color reconstruction
    error after the intervention and (HSV similarity to red)³. The experiment
    lives in [`experiment.py`](./experiment.py):

    ```bash
    bin/mini run docs/m1/ex-2.9.2/experiment.py --app modal --max-containers 8
    ```
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
            "```bash\nbin/mini run docs/m1/ex-2.9.2/experiment.py --app modal --max-containers 8\n```"
        ),
    )
    metrics, exemplars = loaded

    def stat(variant: str, intervention: str, field: str) -> np.ndarray:
        return np.array([r["interventions"][intervention][field] for r in metrics if r["variant"] == variant])

    def runs(variant: str) -> list[dict]:
        return [r for r in metrics if r["variant"] == variant]

    n_seeds = len(runs("base"))
    s_bz = stat("base", "zero", "score")
    s_fr = stat("fallback", "reflect", "score")
    rp_bz = stat("base", "zero", "red_pure")
    rp_fr = stat("fallback", "reflect", "red_pure")
    mo.md(f"""
    **{len(metrics)} runs completed** ({n_seeds} seeds × 2 variants).

    On the selectivity, `base + zero` gets {s_bz.mean():.2f} ± {s_bz.std():.2f}
    (min {s_bz.min():.2f}) and `fallback + reflect` gets {s_fr.mean():.2f} ± {s_fr.std():.2f}
    (min {s_fr.min():.2f}). That looks like an improvement, but most of the gap comes from two
    base seeds that failed badly. The clear change is in the response. Damage to pure red goes
    from {rp_bz.mean():.2f} ± {rp_bz.std():.2f} across seeds to {rp_fr.mean():.3f} ±
    {rp_fr.std():.3f}, pinned just under the analytic ¼ bound.
    """)
    return exemplars, metrics, runs, stat


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selectivity across seeds

    What matters here is what happens to the bad seeds, because the floor and
    spread of the distribution decide how many seeds you need to run. Each dot
    below is one trained model, and the bar is the condition's median.
    """)
    return


@app.cell(hide_code=True)
def _(stat):
    _scores = {(v, i): stat(v, i, "score") for v in VARIANTS for i in INTERVENTIONS}
    _med = {k: np.median(v) for k, v in _scores.items()}

    @themed(
        name="scores",
        alt_text=f"""
        Strip plot of selectivity scores (R squared) for ten conditions: five interventions,
        each under base and fallback training, with about 32 dots per condition.
        Base plus zero, the status quo, has a median of {_med[("base", "zero")]:.2f}, with dots
        scattered down to {_scores[("base", "zero")].min():.2f}. The oa condition sits lower
        (median {_med[("base", "oa")]:.2f} under base training). Reflect and redirect under
        fallback training form the tightest, highest clusters, with medians of
        {_med[("fallback", "reflect")]:.2f} and {_med[("fallback", "redirect")]:.2f} and
        no dot below {min(_scores[("fallback", "reflect")].min(), _scores[("fallback", "redirect")].min()):.2f}.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 4))
        rng = np.random.default_rng(0)
        colors = {"base": light_dark("#9aa5b1", "#6b7686"), "fallback": light_dark("#1a5f8a", "#6ab0d4")}
        for gi, intervention in enumerate(INTERVENTIONS):
            for vi, variant in enumerate(VARIANTS):
                xs = gi + (vi - 0.5) * 0.36 + rng.uniform(-0.09, 0.09, len(_scores[(variant, intervention)]))
                ax.scatter(xs, _scores[(variant, intervention)], s=14, color=colors[variant], alpha=0.75, lw=0)
                ax.plot(
                    [gi + (vi - 0.5) * 0.36 - 0.14, gi + (vi - 0.5) * 0.36 + 0.14],
                    [_med[(variant, intervention)]] * 2,
                    color=colors[variant],
                    lw=2,
                )
        ax.set_xticks(range(len(INTERVENTIONS)))
        ax.set_xticklabels(INTERVENTIONS)
        ax.set(ylabel="score (R², error vs. similarity)", ylim=(0, 1.02))
        ax.grid(alpha=0.3, axis="y")
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=v) for v, c in colors.items()]
        ax.legend(handles=handles, loc="lower right", title="training")
        ax.set_title("Selectivity by intervention and training variant")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(stat):
    _bz, _fr = stat("base", "zero", "score"), stat("fallback", "reflect", "score")
    _frd = stat("fallback", "redirect", "score")
    _n_bad = int((_bz < 0.1).sum())
    _ok = _bz > 0.1
    mo.md(f"""
    The scatter shows two things. First, the `base` variant has {_n_bad} seeds scoring
    below 0.1 under every intervention. Those are anchoring failures (in the worst one, red never
    anchored at all: a validation anchor loss of 0.61, against a median of 0.006), and no
    intervention-time trick reaches them. The fallback variant happened to produce none of these
    in 32 seeds. That is encouraging, but training is chaotic enough that we can't separate one
    extra loss term from plain luck. Second, among the seeds that did anchor, the R²
    distributions barely move: `base + zero` excluding failures scores {_bz[_ok].mean():.2f} ±
    {_bz[_ok].std():.2f}, against {_fr.mean():.2f} ± {_fr.std():.2f} for `fallback + reflect`
    and {_frd.mean():.2f} ± {_frd.std():.2f} for `redirect`. By this metric, fallback
    control adds little.

    R² measures whether error is proportional to redness, but it says nothing about whether
    the size of the response is the same from seed to seed; we'll look into that
    below. (One aside: plain zeroing scores a little lower on the fallback variant,
    median {np.median(stat("fallback", "zero", "score")):.2f} vs {np.median(_bz):.2f}.
    The fallback term presumably perturbs the decoder that zero-ablated latents still
    pass through, so a trained fallback should be paired with its redirect rather than
    zeroing.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A predictable response

    SCA aims for bounded side effects. With the decoder pinned to mid-gray at
    −e₀, redirecting pure red there should give MSE(red, gray) = ¼. Without
    fallback training there is no bound at all, and red lands wherever the
    untrained region happens to decode. Zero ablation's expectation under random
    redistribution is ⅓, but the seed-to-seed spread makes that expectation
    unhelpful.
    """)
    return


@app.cell(hide_code=True)
def _(stat):
    _conds = [("base", "zero"), ("base", "reflect"), ("fallback", "reflect"), ("fallback", "redirect")]
    _vals = {c: stat(*c, "red_pure") for c in _conds}

    @themed(
        name="red-damage",
        alt_text=f"""
        Strip plot of reconstruction error for pure red after intervention, across four conditions,
        with dashed reference lines at one quarter (the gray-fallback bound) and one third
        (the value expected for a random on-manifold direction). Base plus zero scatters widely, from
        {_vals[("base", "zero")].min():.2f} to {_vals[("base", "zero")].max():.2f}; base plus
        reflect scatters even wider and higher. Fallback plus reflect collapses to a tight
        cluster on the one-quarter line (spread {_vals[("fallback", "reflect")].std():.3f});
        fallback plus redirect clusters almost as tightly just below it.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        rng = np.random.default_rng(0)
        fg = light_dark("#1a5f8a", "#6ab0d4")
        for i, c in enumerate(_conds):
            xs = i + rng.uniform(-0.13, 0.13, len(_vals[c]))
            ax.scatter(xs, _vals[c], s=14, color=fg, alpha=0.75, lw=0)
            ax.plot([i - 0.2, i + 0.2], [np.median(_vals[c])] * 2, color=fg, lw=2)
        ref = light_dark("#000", "#fff")
        ax.axhline(0.25, ls="--", lw=1, color=ref, alpha=0.6)
        ax.axhline(1 / 3, ls=":", lw=1, color=ref, alpha=0.6)
        ax.text(3.42, 0.25, "¼ = MSE(red, gray)", va="bottom", ha="right", fontsize=9, alpha=0.8)
        ax.text(3.42, 1 / 3, "⅓ = E[MSE], random direction", va="bottom", ha="right", fontsize=9, alpha=0.8)
        ax.set_xticks(range(len(_conds)))
        ax.set_xticklabels([f"{v}\n{i}" for v, i in _conds])
        ax.set(ylabel="reconstruction MSE, pure red")
        ax.grid(alpha=0.3, axis="y")
        ax.set_title("Damage to pure red: spread vs. the analytic bound")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(stat):
    _bz = stat("base", "zero", "red_pure")
    _br = stat("base", "reflect", "red_pure")
    _fr = stat("fallback", "reflect", "red_pure")
    _fc = stat("fallback", "reflect", "collateral")
    mo.md(f"""
    Under `base + zero`, damage to pure red spans {_bz.min():.2f}–{_bz.max():.2f} across seeds.
    On some seeds the "deleted" red reconstructs almost perfectly, because renormalization
    handed it to a neighboring color. `base + reflect` produces large damage ({_br.mean():.2f} ±
    {_br.std():.2f}), but with no control over where red ends up. `fallback + reflect` lands at
    {_fr.mean():.3f} ± {_fr.std():.3f}, just under the ¼ bound predicted from the geometry, while
    collateral damage on non-red colors stays at {_fc.mean():.4f}.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Median proportionality

    The charts below show the median-scoring runs of each variant.
    """)
    return


@app.cell(hide_code=True)
def _(exemplars):
    _panels = [("base", "zero", exemplars["base"]), ("fallback", "redirect", exemplars["fallback"])]
    _r2 = {}
    for _v, _i, _e in _panels:
        _r2[(_v, _i)] = float(np.corrcoef(_e["sim3"], _e[f"mse_{_i}"])[0, 1] ** 2)

    @themed(
        name="error-vs-similarity",
        alt_text=f"""
        Two scatter plots of post-intervention reconstruction error against cubed HSV similarity
        to red, one point per grid color, each drawn in its own color. On the left, the base variant's
        median seed under zero ablation (R squared {_r2[("base", "zero")]:.2f}): error rises
        with similarity and tops out wherever this seed's redistribution happened to put red,
        here about {float(_panels[0][2]["mse_zero"].max()):.2f}. On the right, the fallback variant's
        median seed under redirect (R squared {_r2[("fallback", "redirect")]:.2f}): points rise
        from gray colors at zero error up to red points clustered together at about
        {float(_panels[1][2]["mse_redirect"].max()):.2f}.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
        edge = light_dark("#00000033", "#ffffff55")
        for ax, (v, i, e) in zip(axes, _panels, strict=True):
            ax.scatter(e["sim3"], e[f"mse_{i}"], c=e["rgb"], s=24, edgecolors=edge, lw=0.5)
            ax.text(0.05, 0.92, f"$R^2$ = {_r2[(v, i)]:.3f}", transform=ax.transAxes)
            ax.set(xlabel="similarity to red (angular HSV, cubed)", title=f"{v} + {i} (median seed)")
            ax.grid(alpha=0.3)
        axes[0].set(ylabel="reconstruction MSE after intervention")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(exemplars):
    _e = exemplars["fallback"]

    @themed(
        name="latents",
        alt_text="""
        Three disc-shaped scatter plots of bottleneck latents, one point per grid color, each
        drawn in its own color, inside a circle marking the unit hypersphere bound. The anchored
        axis points up, labeled (1, 0, 0, 0, 0); the fallback direction points down, labeled
        minus e0. On the left, baseline: blues, greens, and grays hug the horizontal diameter and reds
        reach the top of the circle. In the middle, reflected: the arrangement is mirrored, reds now at
        the bottom of the circle on the fallback direction. On the right, redirected: non-red colors
        stay near the horizontal diameter while reds sit at the bottom.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.4))
        fg = light_dark("#000", "#fff")
        panels = [("Baseline", _e["z_base"]), ("Reflected", _e["z_reflect"]), ("Redirected", _e["z_redirect"])]
        for ax, (title, z) in zip(axes, panels, strict=True):
            plot_latent_disc(ax, z, _e["rgb"])
            ax.set_title(title, y=-0.17)
        axes[0].plot([0], [1.05], marker="v", color=fg, clip_on=False)
        axes[0].annotate(
            "(1, 0, 0, 0, 0)",
            xy=(0, 1.1),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            annotation_clip=False,
        )
        axes[0].plot([0], [-1.05], marker="^", color=fg, clip_on=False)
        axes[0].annotate(
            "−e₀ (fallback)",
            xy=(0, -1.1),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            annotation_clip=False,
        )
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(runs):
    _fb = np.array([r["fallback_color"] for r in runs("fallback")])
    _bs = np.array([r["fallback_color"] for r in runs("base")])
    mo.md(f"""
    The mechanism shows up in the geometry. Both redirect-style interventions move red to −e₀, a
    region the anti-anchor regularizer kept empty, and fallback training decided what lives
    there. Across the {len(_fb)} fallback runs, dec(−e₀) is ({_fb[:, 0].mean():.2f},
    {_fb[:, 1].mean():.2f}, {_fb[:, 2].mean():.2f}), with a maximum per-channel deviation of
    {np.abs(_fb - 0.5).max():.3f}: mid-gray on every seed. In the base variant the same point
    decodes to an uncontrolled color (per-channel spread {_bs.std(axis=0).max():.2f}), which is
    why `base + reflect` damages red heavily but unpredictably.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why optimal ablation scores lower here

    Optimal ablation scored lower than plain zeroing on selectivity: OA's
    constant minimizes expected loss over the task distribution, and that
    distribution includes the concept being removed. In an anchored model the
    cheapest way to reduce expected loss is to partially restore the concept and
    shrink the error signal that the score measures. This makes sense for the
    question OA was built for, where a small loss gap is the useful outcome.

    The removal-appropriate version, which optimizes the constant over non-red
    colors only, lands close to zero. SCA's anti-subspace regularizer already
    made small constants the least disruptive for bystander colors during
    training, so this version behaves like plain zeroing. Anchoring, in other
    words, gives you most of the "optimal" part of optimal ablation for free.
    """)
    return


@app.cell(hide_code=True)
def _(metrics, stat):
    _c_oa = np.array([r["c_oa"] for r in metrics])
    _c_nt = np.array([r["c_oa_nt"] for r in metrics])
    _rp_oa = stat("base", "oa", "red_pure")
    _rp_z = stat("base", "zero", "red_pure")
    mo.md(f"""
    The literal OA constant is positive (red-ward) in {int((_c_oa > 0).sum())}
    of {len(metrics)} runs (median {np.median(_c_oa):+.2f}, except for the
    anchoring failures). With OA, the damage to pure red drops to
    {_rp_oa.mean():.3f} on the base variant, against {_rp_z.mean():.3f} for
    zeroing, so the deletion is partially undone.
    """)
    return


@app.cell(hide_code=True)
def _(stat):
    _rp = stat("fallback", "reflect", "red_pure")
    _floor = min(stat("fallback", "reflect", "score").min(), stat("fallback", "redirect", "score").min())
    mo.md(f"""
    ## Takeaways

    The seed problem from ex-2.9.1 seems to be two problems. One is intervention
    unpredictability, and in this experiment that was solved by fallback
    control: one decoder-only loss term gives the intervention a designed
    outcome, bounded above by ¼ for pure red and within {_rp.std():.2f} of that
    bound across every seed. The other is anchoring failure, where the concept
    never isolates onto its axis, and nothing applied at intervention time can
    fix that.
    """)
    return


if __name__ == "__main__":
    app.run()
