import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="D2.1: figures for the anchoring post",
    css_file="../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import re
    import tempfile
    from pathlib import Path
    from typing import cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import AxesRow, light_dark, smooth_step, themed
    from sca.colorcube import sim_to_red
    from sca.vis_grading import GRID_RGB, I_RED, REDNESS, grading_field

    use_publisher(report_bundle(__file__))

    RLEVELS = np.unique(np.round(REDNESS, 6))
    """The 29 distinct redness values among the 216 grid colors."""
    SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
    """The grading statistic's target shape, as in ex-2.1.6 – ex-2.1.10."""
    SLICE_NAMES = ["emb", "1", "2", "3", "4"]
    POS_NAMES = ["op1", "+", "op2", "=", "ans", r"\n"]

    # The four panels of the progression figure, in recipe order. The refs are
    # stringly typed (each experiment.py defines its own, but those modules are
    # not importable across directories), so the condition names are repeated here.
    EXPERIMENTS: dict[str, list[str]] = {
        "ex-2.1.6": ["lam0", "lam0.1"],
        "ex-2.1.8": ["end90-hold30"],
        "ex-2.1.10": ["either-t100"],
    }
    PANELS: list[tuple[str, str]] = [
        ("ex-2.1.6/lam0", "un-anchored"),
        ("ex-2.1.6/lam0.1", "+ anchor"),
        ("ex-2.1.8/end90-hold30", "+ anti-subspace"),
        ("ex-2.1.10/either-t100", "+ pooled labels"),
    ]
    PRIMARY = "ex-2.1.10/either-t100"

    def load_alpha() -> dict[str, np.ndarray] | None:
        """Seed-mean (slice, color, position) α per panel condition, from the published arrays."""
        refs = {exp: f"reports/m2/{exp}/arrays" for exp in EXPERIMENTS}
        store = project_store()
        arts = store.get_refs(list(refs.values()))
        if any(arts[r] is None for r in refs.values()):
            return None
        out: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory() as d:
            paths = store.get_many([(arts[refs[exp]], Path(d) / f"{exp}.npz") for exp in EXPERIMENTS])
            for exp, path in zip(EXPERIMENTS, paths, strict=True):
                with np.load(path) as z:
                    for cond in EXPERIMENTS[exp]:
                        runs = [z[k] for k in z.files if re.fullmatch(rf"{re.escape(cond)}-s\d+/alpha", k)]
                        assert runs, f"{exp} published no runs for {cond}"
                        out[f"{exp}/{cond}"] = np.mean(runs, axis=0)
        return out

    def level_mean(y: np.ndarray) -> np.ndarray:
        """The response's mean at each distinct redness level — the cloud's summary line."""
        return np.array([y[np.round(REDNESS, 6) == lv].mean() for lv in RLEVELS])

    def r2_sim(y: np.ndarray) -> float:
        """Shape r² against sim¹·⁵, the grading statistic; NaN where the response is constant."""
        if np.std(y) < 1e-6:
            return float("nan")
        return float(np.corrcoef(y, SIM_TARGET)[0, 1] ** 2)

    def stepped(ax: plt.Axes, values, sw: float, **kwargs):
        """A smooth-step over position slots: plateaus span each slot's width *sw*, risers
        the gaps. NaN values get no plateau; an isolated finite value draws as a dash.
        """
        v = np.asarray(values, float)
        finite = np.isfinite(v)
        i = 0
        while i < len(v):
            if not finite[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(v) and finite[j + 1]:
                j += 1
            if j > i:
                smooth_step(ax, np.arange(i, j + 1), v[i : j + 1], ramp=1 - sw, **kwargs)
            else:
                ax.plot([i - sw / 2, i + sw / 2], [v[i]] * 2, **kwargs)
            i = j + 1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # D2.1: figures for the anchoring post

    Grading-cloud figures for the LessWrong post on D2.1, *Anchoring concepts in transformers*. This page is not an experiment: every panel is drawn from the published results of [ex-2.1.6](./ex-2.1.6/report.py), [ex-2.1.8](./ex-2.1.8/report.py), and [ex-2.1.10](./ex-2.1.10/report.py), which hold the preregistered claims and their evidence. The post uploads the light `_assets/*.png` files from this page's bundle.
    """)
    return


@app.cell(hide_code=True)
def _():
    _loaded = load_alpha()
    mo.stop(_loaded is None, mo.md("_Results are not published yet; the figures render once they are._"))
    alphas: dict[str, np.ndarray] = _loaded
    return (alphas,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The recipe, one piece at a time

    Each panel below adds one piece of the anchoring recipe and shows the α response at op1 — the cosine between the residual stream and the anchor axis, meaned over the five residual slices — as a grading cloud against the redness of the first operand.
    """)
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, np.ndarray]):
    _ys = {key: alphas[key][:, :, 0].mean(axis=0) for key, _ in PANELS}

    @themed(
        name="grading-progression",
        alt_text="""
            Four grading clouds against redness. The un-anchored cloud is flat near zero; adding the anchor lifts every color to at least 0.4; the anti-subspace term restores a graded ramp from near zero; pooled labels leave the same clean rise to about 0.9 at pure red.
        """,
        caption=r"""
            **The anchoring recipe, one piece at a time.** Each panel is a grading cloud of the α response at op1 (mean over the five residual slices) against the redness of the first operand, drawn over the whole RGB cube; the dark line is the mean at each redness level. Left to right: the un-anchored control and the bare anchor at λ = 0.1 (ex-2.1.6); the anchor plus the anti-subspace term at its tuned operating point, labels still keyed to op1 (ex-2.1.8, `end90-hold30`); and the full recipe, where sequence-level labels name *either* operand and mellowmax pooling finds the red token on its own (ex-2.1.10's primary, `either-t100`).
        """,
    )
    def _plot() -> plt.Figure:
        field = grading_field(k=30, px=(240, 200))
        fig, axes = plt.subplots(1, len(PANELS), figsize=(7.4, 2.5), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, (key, title) in zip(axes, PANELS, strict=True):
            y = _ys[key]
            ax.axhline(0, color=light_dark("#ccc", "#444"), lw=0.6, zorder=0)
            field.draw(ax, y)
            ax.plot(RLEVELS, level_mean(y), color=light_dark("#222", "#ddd"), lw=1.0, zorder=4)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("redness of op1", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylim(-0.3, 1.1)
        axes[0].set_ylabel("α at op1, slice mean", fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, np.ndarray]):
    _bare = alphas["ex-2.1.6/lam0.1"][:, :, 0].mean(axis=0)
    _primary = alphas[PRIMARY][:, :, 0].mean(axis=0)
    mo.md(rf"""
    The bare anchor buys alignment without selectivity: every color's response rises, and the least red colors still sit at α ≈ {_bare.min():.2f}. The anti-subspace term is what makes alignment graded — non-red colors return to zero while pure red stays anchored — and the grading survives the move to labels that no longer point at a token: in the primary condition the slice-mean response at op1 tracks the sim¹·⁵ affinity shape with r² ≈ {r2_sim(_primary):.2f}.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The primary condition, per (slice, position)

    The clouds above average over the residual slices and look only at op1. Breaking the primary condition down shows where in the network, and where in the sequence, the anchored response lives.
    """)
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, np.ndarray]):
    _a = alphas[PRIMARY]

    @themed(
        name="grading-grid",
        alt_text="""
            Five rows of grading clouds, one per residual slice, with a slot per token position. In every slice the op1 slot holds a sigmoid cloud rising to about 0.9 at pure red; the other slots sit near zero, with weaker red-rising tails at op2 and the answer.
        """,
        caption=r"""
            **The primary condition per (slice, position)** (ex-2.1.10, `either-t100`). One row per residual slice, embedding at the bottom; each slot holds the grading cloud of the α response at that token position, keyed throughout by the *first* operand's color, with its redness running left to right within the slot. Overlays: α for probe lines whose first operand is pure red (red), and the slot's shape r² against sim¹·⁵ (grey dashed). One y scale for the whole figure, at the bottom right.
        """,
    )
    def _plot() -> plt.Figure:
        sw = 0.72
        n_pos = _a.shape[2]
        field = grading_field(k=30, px=(120, 150))
        red = dict(color=light_dark("#d40000", "#ff6659"), lw=0.5, alpha=0.75)
        grey = dict(color=light_dark("#888", "#999"), lw=0.75, alpha=0.75, ls=(0, (3, 2)))
        fig, axes = plt.subplots(5, 1, figsize=(7.2, 6.4), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for si in range(5):
            ax = axes[4 - si]  # embedding at the bottom, as the profile figures do
            ax.axhline(0, color=light_dark("#ccc", "#444"), lw=0.6, zorder=0)
            for pi in range(n_pos):
                field.draw(ax, _a[si, :, pi], span=(pi - sw / 2, pi + sw / 2))
            stepped(ax, _a[si, I_RED, :], sw, **red)
            stepped(ax, [r2_sim(_a[si, :, pi]) for pi in range(n_pos)], sw, **grey)
            ax.set_ylabel(SLICE_NAMES[si], fontsize=8, rotation=0, ha="right", va="center")
            ax.set_frame_on(False)
            ax.tick_params(labelsize=7, left=False, labelleft=False, bottom=False)
        axes[0].set_xlim(-0.5, n_pos - 0.5)
        axes[0].set_ylim(-0.3, 1.1)
        axes[-1].set_xticks(range(n_pos), POS_NAMES)
        axes[-1].tick_params(labelbottom=True)
        # One y scale for the whole figure, on the right of the bottom row.
        axes[-1].yaxis.tick_right()
        axes[-1].tick_params(labelsize=7, right=True, labelright=True)
        fig.legend(
            [plt.Line2D([], [], **(kw | {"lw": 1.2})) for kw in (red, grey)],  # legible at legend scale
            ["α at pure red", "shape r² vs sim¹·⁵"],
            loc="outside lower center", ncols=2, fontsize=8, frameon=False,
        )  # fmt: skip
        return fig

    mo.Html(_plot())
    return


if __name__ == "__main__":
    app.run()
