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
    from mini.vis import AxesRow, light_dark, smooth_step, smooth_step_area, themed
    from sca.colorcube import sim_to_red
    from sca.data.named_colors import GRIDS, grid_palette
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

    def _closed_pairs() -> np.ndarray:
        """The (op1, op2) color indices of ex-2.1.10's 5832 probe lines, in its line order.

        Its prep step walks every color as op1 against the partners whose mix stays on
        the grid, both in palette order; `argwhere` enumerates the same pairs in the
        same order. Mixing is a channel-wise round-half-up mean, so the relation is
        symmetric: each color sits at op2 on as many lines as it sits at op1.
        """
        levels = GRIDS["v216"]
        g = np.array(list(grid_palette(levels).values()))  # (216, 3), integer levels
        mixed = (g[:, None, :] + g[None, :, :] + 1) // 2
        return np.argwhere(np.isin(mixed, levels).all(axis=2))

    PROBE_PAIRS = _closed_pairs()
    """Which color sits at each operand slot, per probe line."""

    def keyed_by(lines: np.ndarray, slot: int) -> np.ndarray:
        """Re-key per-line α — (slices, lines, positions) — onto the color at *slot*.

        ex-2.1.10 publishes the per-line array beside its op1 margin, which it takes as
        a reshape because the lines come grouped by op1. Sorting the lines by op2 lets
        the same reshape read the other margin: the response against the color of the
        *second* operand, averaged over the first operands it appears with.
        """
        idx = PROBE_PAIRS[:, slot]
        counts = np.bincount(idx)
        assert counts.min() == counts.max(), f"probe set is ragged at slot {slot}"
        order = np.argsort(idx, kind="stable")
        return lines[:, order].reshape(lines.shape[0], len(counts), counts[0], -1).mean(axis=2)

    def load_alpha() -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        """Seed-mean α per panel condition, plus the primary's response re-keyed onto op2.

        Each panel array is (slice, color, position) against op1's color, as published.
        The second return is the primary's per-line array collapsed the other way, so
        its color axis is op2's.
        """
        refs = {exp: f"reports/m2/{exp}/arrays" for exp in EXPERIMENTS}
        store = project_store()
        arts = {r: a for r, a in store.get_refs(list(refs.values())).items() if a is not None}
        if len(arts) < len(refs):
            return None
        out: dict[str, np.ndarray] = {}
        op2: np.ndarray | None = None
        with tempfile.TemporaryDirectory() as d:
            paths = store.get_many([(arts[refs[exp]], Path(d) / f"{exp}.npz") for exp in EXPERIMENTS])
            for exp, path in zip(EXPERIMENTS, paths, strict=True):
                with np.load(path) as z:
                    for cond in EXPERIMENTS[exp]:
                        runs = [z[k] for k in z.files if re.fullmatch(rf"{re.escape(cond)}-s\d+/alpha", k)]
                        assert runs, f"{exp} published no runs for {cond}"
                        out[f"{exp}/{cond}"] = np.mean(runs, axis=0)
                        if f"{exp}/{cond}" != PRIMARY:
                            continue
                        pat = rf"{re.escape(cond)}-s\d+/alpha_lines"
                        lines = np.mean([z[k] for k in z.files if re.fullmatch(pat, k)], axis=0)
                        # The op1 margin recomputed from the lines is the published array,
                        # so the pair index is in step with the probe set that produced it.
                        np.testing.assert_allclose(keyed_by(lines, 0), out[PRIMARY], rtol=0, atol=1e-5)
                        op2 = keyed_by(lines, 1)
        assert op2 is not None, "the primary published no per-line α"
        return out, op2

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

    def grading_grid(a: np.ndarray, red_label: str) -> plt.Figure:
        """One row per residual slice, one grading-cloud slot per token position.

        *a* is (slice, color, position) α, keyed by whichever operand's color the
        caller wants on the color axis; *red_label* names the overlay, which is that
        keying's pure-red row. The two figures below share this so the op1 and op2
        keyings differ only in the array they are handed.
        """
        sw = 0.72
        n_pos = a.shape[2]
        field = grading_field(k=50, px=(120, 150))
        red_c = light_dark("#d40000", "#f44")
        grey_c = light_dark("#888", "#999")
        fig, axes = plt.subplots(5, 1, figsize=(7.2, 5.6), sharex=True, sharey=True)
        axes = cast(AxesRow, axes)
        for si in range(5):
            ax = axes[4 - si]  # embedding at the bottom, as the profile figures do
            reds = a[si, I_RED, :]
            xs = np.arange(len(reds))
            smooth_step_area(ax, xs, reds, ramp=1 - sw, color=grey_c, alpha=0.1)
            for pi in range(n_pos):
                img = field.draw(ax, a[si, :, pi], span=(pi - sw / 2, pi + sw / 2))
                img.set_clip_on(False)
            smooth_step(ax, xs, reds, ramp=1 - sw, color=red_c, lw=1, alpha=1)
            ax.set_ylabel(SLICE_NAMES[si], fontsize=8, rotation=0, ha="right", va="center")
            ax.tick_params(labelsize=7, left=False, labelleft=False, bottom=False)
        axes[0].set_xlim(-0.5, n_pos - 0.5)
        axes[0].set_ylim(0, 1)
        axes[-1].set_xticks(range(n_pos), POS_NAMES)
        axes[-1].set_yticks([0, 0.5, 1])
        axes[-1].tick_params(labelbottom=True)
        # One y scale for the whole figure, on the right of the bottom row.
        axes[-1].yaxis.tick_right()
        axes[-1].tick_params(labelsize=7, right=True, labelright=True)
        fig.legend(
            # Handle at lw 1.2, legible at legend scale.
            [plt.Line2D([], [], color=red_c, lw=1.2, alpha=0.75)],
            [red_label],
            loc="outside lower center", fontsize=8, frameon=False,
        )  # fmt: skip
        return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # D2.1: figures for the anchoring post

    Grading-cloud figures for the LessWrong post on D2.1, *Anchoring concepts in transformers*. This page is not an experiment. Every panel is drawn from the published results of [ex-2.1.6](./ex-2.1.6/report.py), [ex-2.1.8](./ex-2.1.8/report.py), and [ex-2.1.10](./ex-2.1.10/report.py), which hold the preregistered claims and their evidence. The post uploads the light `_assets/*.png` files from the bundle of this page.
    """)
    return


@app.cell(hide_code=True)
def _():
    _loaded = load_alpha()
    mo.stop(_loaded is None, mo.md("_Results are not published yet; the figures render once they are._"))
    assert _loaded is not None
    alphas: dict[str, np.ndarray] = _loaded[0]
    alpha_op2: np.ndarray = _loaded[1]
    return alpha_op2, alphas


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The recipe, one piece at a time

    Each panel below adds one piece of the anchoring recipe. The quantity plotted is the α response at the op1 token, drawn against the redness of the first operand.[^alpha] Each chart is a *grading cloud*: the response is measured at the 216 vocabulary colors, interpolated across the whole RGB cube, and drawn as a dither of the colors themselves, so the vertical spread at each redness is the range of responses among equally red colors.

    [^alpha]: α is the cosine between the residual stream (the running vector that each layer reads from and writes back to) and the anchor axis, the direction we train the concept toward. Here it is averaged over the five residual slices, the five points in the network where we read that vector.
    """)
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, np.ndarray]):
    _ys = {key: alphas[key][:, :, 0].mean(axis=0) for key, _ in PANELS}
    _field = grading_field(k=40, px=(240, 200), ylim=(-0.1, 1.1))

    @themed(
        name="grading-progression",
        alt_text="""
            Four grading clouds against redness. The un-anchored cloud is flat near zero; adding the anchor lifts every color to at least 0.4; the anti-subspace term restores a graded ramp from near zero; pooled labels leave the same clean rise to about 0.9 at pure red.
        """,
        caption=r"""
            **The anchoring recipe, one piece at a time.** Each panel is a grading cloud of the α response at op1 (mean over the five residual slices) against the redness of the first operand, drawn over the whole RGB cube. The dark line is the mean at each redness level. Left to right: the un-anchored control and the bare anchor at λ = 0.1 (ex-2.1.6); the anchor plus the anti-subspace term at its tuned operating point, with labels still keyed to op1 (ex-2.1.8, `end90-hold30`); and the full recipe, where sequence-level labels name *either* operand and mellowmax pooling finds the red token on its own (the primary condition of ex-2.1.10, `either-t100`).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(PANELS), figsize=(7.4, 2.2), sharex=True, sharey=True, layout="constrained")
        axes = cast(AxesRow, axes)
        for ax, (key, title) in zip(axes, PANELS, strict=True):
            y = _ys[key]
            # ax.axhline(0, color=light_dark("#ccc", "#444"), lw=0.6, zorder=0)
            _img = _field.draw(ax, y)
            _img.set_clip_on(False)
            # ax.plot(RLEVELS, level_mean(y), color=light_dark("#222", "#ddd"), lw=1.0, zorder=4)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("redness of op1", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_aspect("equal")
        axes[0].set_xticks([0, 0.5, 1])
        axes[0].set_yticks([0, 0.5, 1])
        # axes[0].set_ylim(-0.1, 1.1)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("α at op1, slice mean", fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, np.ndarray]):
    _bare = alphas["ex-2.1.6/lam0.1"][:, :, 0].mean(axis=0)
    _primary = alphas[PRIMARY][:, :, 0].mean(axis=0)
    mo.md(rf"""
    The bare anchor aligns everything at once: the response rises for every color, and even the least red colors sit at α ≈ {_bare.min():.2f}. Adding the anti-subspace term is what makes the alignment graded, pulling non-red colors back to zero while pure red stays anchored. That grading survives the move to labels that no longer point at a particular token: in the primary condition, the slice-mean response at op1 tracks the sim¹·⁵ affinity shape with r² ≈ {r2_sim(_primary):.2f}.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The primary condition, per (slice, position)

    The clouds above average over the residual slices and look only at op1. Splitting the primary condition by slice and by token position shows where in the network, and where in the sequence, the anchored response sits.
    """)
    return


@app.cell(hide_code=True)
def _(alphas: dict[str, np.ndarray]):
    @themed(
        name="grading-grid",
        alt_text="""
            Five rows of grading clouds, one per residual slice, with a slot per token position. In every slice the op1 slot holds a sigmoid cloud rising to about 0.9 at pure red; the other slots sit near zero, with weaker red-rising tails at op2 and the answer.
        """,
        caption=r"""
            **The primary condition per (slice, position)** (ex-2.1.10, `either-t100`). One row per residual slice, with the embedding at the bottom. Each slot holds the grading cloud of the α response at that token position, keyed throughout by the color of the *first* operand, whose redness runs left to right within the slot. Over them, in red, α on the probe lines whose first operand is pure red. One y scale serves the whole figure, at the bottom right.
        """,
    )
    def _plot() -> plt.Figure:
        return grading_grid(alphas[PRIMARY], "α where op1 is pure red")

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(alpha_op2: np.ndarray, alphas: dict[str, np.ndarray]):
    _a1, _a2 = alphas[PRIMARY], alpha_op2
    _r2 = lambda a, t: r2_sim(a[:, :, t].mean(axis=0))  # noqa: E731
    _rise = lambda a, t: float(np.diff(level_mean(a[0, :, t])[[0, -1]])[0])  # noqa: E731
    # What each color's partners look like: mean redness over the 27 lines it leads.
    _partner_red = np.bincount(PROBE_PAIRS[:, 0], REDNESS[PROBE_PAIRS[:, 1]]) / np.bincount(PROBE_PAIRS[:, 0])
    mo.md(rf"""
    ## The same measurement, keyed by op2

    Under pooled labels neither operand is privileged, so the figure above is one of two views the same measurement affords. The probe set runs every color as op1 against each of the 27 partners it mixes cleanly with; mixing is symmetric, so every color also sits at op2 on 27 lines. Averaging each line over op1 rather than over op2 re-keys the whole α map onto the *second* operand's color, with no new evaluation.[^margin]

    The result is close to a mirror image. The cloud moves to the op2 slot and grades about as well there as op1 does in its own slot — slice-mean r² against sim¹·⁵ of {_r2(_a2, 2):.2f} at op2 against {_r2(_a1, 0):.2f} at op1 — and pure red reaches α ≈ {_a2[1:4, I_RED, 2].mean():.2f} in mid-stack, against {_a1[1:4, I_RED, 0].mean():.2f} for a pure-red op1. So the pull finds the red token in either slot and grades it the same way there, which is the claim the pooled labels were meant to buy.

    Two asymmetries survive. Op2 carries a little more of the bulk drift at the last slice ({_a2[4, :, 2].mean():.2f} averaged over all colors, against {_a1[4, :, 0].mean():.2f} at op1), matching ex-2.1.10's reading that op1 hosts less of it once half the pull budget goes to op2. The answer slot, by contrast, barely moves between the two keyings (r² {_r2(_a2, 4):.2f} against {_r2(_a1, 4):.2f}): the answer is the mix, so its redness is a function of both operands and neither keying has much claim on it.

    **Read the off-key slot with care.** In each figure, the slot that is *not* the key still tilts upward — its level-mean rises by about {_rise(_a2, 0):.2f} across the redness range at the embedding, in both views alike. In the op1-keyed figure, part of that tilt is a real path: op2's state can attend back to a red op1. In the op2-keyed figure no such path exists, since op1 precedes op2 and a causal model cannot let the later token reach the earlier one. What produces it is the shape of the probe set. A pair is on it only if the mix lands back on the color grid, and that closure rule correlates a color's redness with its partners': pure red's partners average {_partner_red[I_RED]:.2f} redness against {REDNESS.mean():.2f} for the palette as a whole. So an off-key slot shows what a color's *partners* do, and only the on-key slot is a response to the key. The embedding row is the clean demonstration — there every state is a function of its own token, so the tilt left in the off-key slot is all bookkeeping.

    [^margin]: Both views are margins of one per-line array, in the statistical sense: each drops an operand by averaging over it. The op1-keyed cloud already contains every line the op2-keyed cloud draws — the colors are re-sorted, not re-measured, and either way a slot's response is a mean over 27 lines.
    """)
    return


@app.cell(hide_code=True)
def _(alpha_op2: np.ndarray):
    @themed(
        name="grading-grid-op2",
        alt_text="""
            The same five rows of grading clouds, now keyed by the second operand. The sigmoid cloud has moved to the op2 slot, rising to about 0.99 in the middle slices, while the op1 slot sits flat near zero and the answer slot keeps its weaker red-rising tail.
        """,
        caption=r"""
            **The same condition keyed by the second operand** (ex-2.1.10, `either-t100`). The layout of the figure above, with every cloud keyed by the color of the *second* operand instead: op2's redness runs left to right within each slot, and each color's response is the mean over the 27 first operands it appears with. In red, α on the probe lines whose second operand is pure red. The op1 slot is that partner mean rather than a response to op2 — op1 precedes op2, so nothing at op1 can see it.
        """,
    )
    def _plot() -> plt.Figure:
        return grading_grid(alpha_op2, "α where op2 is pure red")

    mo.Html(_plot())
    return


if __name__ == "__main__":
    app.run()
