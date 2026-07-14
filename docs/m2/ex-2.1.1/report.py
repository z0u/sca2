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

    # Marimo puts the notebook's directory on sys.path, so the experiment
    # definition is importable — refs and sweep constants can't drift.
    from experiment import (
        CORPUS_SEED,
        DEPTHS,
        HOLDOUT_FRAC,
        METRICS_REF,
        N_EXAMPLES,
        SEEDS,
        WEIGHTS_REF,
        WIDTHS,
    )
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import light_dark, themed
    from sca.data import colors
    from subline.series import Series
    from subline.subline import Subline

    use_publisher(report_bundle(__file__))

    EVAL_SETS = ["named_seen", "named_holdout", "hex_unseen", "cross_unseen"]

    def load_results() -> tuple[list[dict], dict[str, np.ndarray]] | None:
        """Resolve the metrics and probe weights from the store, or None if unpublished."""
        store = project_store()
        m_art, w_art = store.get_ref(METRICS_REF), store.get_ref(WEIGHTS_REF)
        if m_art is None or w_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            metrics = json.loads(store.get(m_art, Path(d) / "metrics.json").read_text())
            with np.load(store.get(w_art, Path(d) / "weights.npz")) as z:
                weights = {k: z[k] for k in z.files}
        return metrics, weights

    def label(w: int, d: int, s: int) -> str:
        return f"d{w}-L{d}-s{s}"

    def acc(metrics: list[dict], w: int, d: int, s: int, eval_set: str) -> float:
        (r,) = [r for r in metrics if r["label"] == label(w, d, s)]
        return r["accuracy"][eval_set]["accuracy"]

    def width_shades() -> dict[int, tuple]:
        stops = light_dark([0.7, 0.45, 0.12], [0.8, 0.55, 0.28])
        return dict(zip(WIDTHS, plt.cm.viridis(stops), strict=True))

    def pick_backbone(metrics: list[dict]) -> tuple[int, int]:
        """The smallest cell (by params ∝ width²·depth) that saturates the unseen-pair sets."""

        def unseen(w: int, d: int) -> float:
            return float(np.mean([acc(metrics, w, d, s, es) for s in SEEDS for es in ("hex_unseen", "cross_unseen")]))

        cells = sorted(((w, d) for w in WIDTHS for d in DEPTHS), key=lambda c: c[0] ** 2 * c[1])
        return next((c for c in cells if unseen(*c) >= 0.995), max(cells, key=lambda c: unseen(*c)))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.1: the color-mixing transformer, un-anchored

    M2 asks whether Sparse Concept Anchoring transfers from autoencoders to
    transformers. Before anchoring anything, D2.1 needs its baseline: a small
    transformer that demonstrably learns a task with unambiguous color
    concepts, plus the apparatus to measure what the anchored runs will be
    compared against. That is this experiment.

    The task is a character-level language of mixing equations on a 16-level
    RGB grid: `red + blue = purple`, `#e26 + #48a = #958`,
    `rose + #fe8 = #f78`, and alias lines (`red = #f00`) that tie the two
    surface forms of each concept together. Mixing is the channel-wise
    round-half-up mean, so
    every prompt has exactly one correct completion, and a *concept* (say
    *red*) is multi-token in both of its spellings — which is what D2.1.2+
    need: an anchor should capture red-the-concept, not the token `red`.

    We sweep width {16, 32, 64} × depth {2, 4} × 3 seeds ([experiment
    definition](./experiment.py)) and measure two things per cell:

    - **Completion accuracy** (greedy, exact match), on named pairs seen in
      training, *held-out* named pairs (never shown as named equations, so the
      model must compose the alias dictionary with hex arithmetic), and hex /
      cross-form operand pairs never seen together.
    - **Probes**: ridge regression from the residual stream at each depth to
      the operand color, the result color, and the result's *redness* — M1's
      graded concept label, ported to this grid.

    **Hypotheses.** (1) A small nGPT learns the task: near-perfect accuracy on
    seen forms and on unseen *hex* pairs, giving the anchored runs headroom to
    show degradation. (2) Color is linearly decodable from the residual
    stream, increasingly so with depth. (3) *Where* it is decodable is not
    consistent across seeds — the probe directions for redness should be
    essentially unrelated run to run. That last one is the point of the
    milestone: post-hoc search finds a different geometry every time, and SCA's
    job (next experiment) is to pin it in advance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What the model sees

    The corpus sampler is deterministic, so regenerating it here with the
    experiment's own constants reproduces the training data exactly. These are
    the first lines the model saw, verbatim:
    """)
    return


@app.cell(hide_code=True)
def _():
    train_pairs, holdout = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
    corpus = colors.sample_corpus(N_EXAMPLES, CORPUS_SEED, train_pairs)

    def _form(ex) -> str:
        if ex.rhs is None:
            return "alias"
        return {0: "named", 3: "hex"}.get(ex.prompt.count("#") + ex.answer.count("#"), "cross")

    _counts = {f: sum(_form(ex) == f for ex in corpus) for f in ("hex", "named", "cross", "alias")}
    _pairs = {p for ex in corpus if (p := ex.pair) is not None}
    _grid = colors.N_LEVELS**3
    _all_pairs = _grid * (_grid + 1) // 2
    _head = "".join(ex.text for ex in corpus[:10])
    mo.vstack(
        [
            mo.md(f"```\n{_head}```"),
            mo.md(
                f"{len(corpus):,} lines in total: "
                + ", ".join(f"{n:,} {f}" for f, n in _counts.items())
                + f". Between them they use {len(_pairs):,} distinct operand pairs — "
                f"**{len(_pairs) / _all_pairs:.2%}** of the grid's {_all_pairs / 1e6:.1f}M, so the unseen-pair "
                f"eval sets (sampled to avoid every one of them) test the mixing rule, not recall. "
                f"Named equations draw only from the training side of the pair split below; the "
                f"{len(holdout)} held-out pairs are the `named_holdout` eval set, never shown as a "
                f"named equation."
            ),
        ]
    )
    return corpus, holdout, train_pairs


@app.cell(hide_code=True)
def _(holdout, train_pairs):
    _vals = list(colors.PALETTE.values())
    _names = list(colors.PALETTE)

    @themed(
        name="named-pair-matrix",
        alt_text=(
            "A 27 by 27 grid of color swatches, rows and columns labeled with the palette's color names "
            "from black to white. Each cell shows the mix of its row and column colors; the diagonal is the "
            "palette itself. Small dots mark the pairs that appear as named equations in training; open "
            "rings mark the held-out named pairs, which are reserved for evaluation."
        ),
    )
    def _plot() -> plt.Figure:
        img = np.array([[colors.mix(a, b) for b in _vals] for a in _vals], dtype=float) / (colors.N_LEVELS - 1)
        fig, ax = plt.subplots(figsize=(7.4, 7.4))
        ax.imshow(img, interpolation="nearest")
        train, held = set(train_pairs), set(holdout)
        pts = {
            (i, j): pair in held
            for i, a in enumerate(_vals)
            for j, b in enumerate(_vals)
            if (pair := (min(a, b), max(a, b))) in train | held
        }
        # A contrasting halo keeps the marks legible on cells near the mark color.
        mark, halo = light_dark("#000", "#fff"), light_dark("#fffa", "#000a")
        _dots = [(x, y) for (y, x), h in pts.items() if not h]
        _rings = [(x, y) for (y, x), h in pts.items() if h]
        ax.scatter(*zip(*_dots, strict=True), s=22, color=halo)
        ax.scatter(*zip(*_dots, strict=True), s=6, color=mark)
        ax.scatter(*zip(*_rings, strict=True), s=80, facecolors="none", edgecolors=halo, lw=3.5)
        ax.scatter(*zip(*_rings, strict=True), s=80, facecolors="none", edgecolors=mark, lw=1.2)
        ax.set_xticks(range(len(_names)), _names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(_names)), _names, fontsize=7)
        ax.set_title(
            "mix(a, b) over the palette — pairs rendered as named equations:\n· in training, ○ held out for eval"
        )
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every cell of the matrix is reachable through hex and cross equations
    (those draw operands from the full 16³ grid), but only the marked pairs
    ever appear as *named* equations, with a named answer. A ringed pair like
    `red + blue` is answerable two ways: recall (impossible — that rendering
    never occurs in training) or composition — look both names up via the
    alias lines, mix in hex space, and translate the result back through the
    dictionary. That is what the `named_holdout` eval set measures. The
    `hex_unseen` and `cross_unseen` sets are sampled at eval time from the
    full grid, avoiding every operand pair the corpus used.
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
            "No results yet — run the experiment (it publishes metrics and probe weights on completion):\n\n"
            "```bash\nbin/mini run docs/m2/ex-2.1.1/experiment.py --app modal --max-containers 9\n```"
        ),
    )
    metrics, weights = loaded
    return metrics, weights


@app.cell(hide_code=True)
def _(metrics):
    _hex = [acc(metrics, w, d, s, "hex_unseen") for w in WIDTHS for d in DEPTHS for s in SEEDS]
    _hold = [acc(metrics, w, d, s, "named_holdout") for w in WIDTHS for d in DEPTHS for s in SEEDS]
    mo.md(
        f"**Headline numbers.** Accuracy on unseen hex pairs spans "
        f"**{min(_hex):.2f}–{max(_hex):.2f}** across the sweep; held-out named pairs "
        f"(the compositional test) span **{min(_hold):.2f}–{max(_hold):.2f}**. "
        f"The figures below break this down by cell and eval set."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Completion accuracy across the sweep

    One panel per eval set: accuracy against width, one line per depth (mean
    over seeds), individual seeds as faint points. The named-holdout panel is
    the interesting one — it can only be solved by composing the alias
    dictionary with the mixing arithmetic, never by recall.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="accuracy-sweep",
        alt_text=(
            "Four line charts of completion accuracy (0 to 1) against model width (16, 32, 64), one panel "
            "per eval set: named seen, named holdout, hex unseen, and cross unseen. Each panel has one line "
            "per depth (2 and 4 layers, darker is deeper), averaged over three seeds, with individual seeds "
            "as faint points."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2), sharey=True)
        _stops = light_dark([0.6, 0.2], [0.7, 0.4])
        shades = dict(zip(DEPTHS, plt.cm.viridis(_stops), strict=True))
        for ax, es in zip(axes, EVAL_SETS, strict=True):
            for d in DEPTHS:
                per_seed = np.array([[acc(metrics, w, d, s, es) for s in SEEDS] for w in WIDTHS])
                for s in range(len(SEEDS)):
                    ax.plot(WIDTHS, per_seed[:, s], "o", color=shades[d], alpha=0.3, ms=3)
                ax.plot(WIDTHS, per_seed.mean(axis=1), "o-", color=shades[d], label=f"{d} layers", lw=2)
            ax.set(title=es.replace("_", " "), xlabel="width", xscale="log", ylim=(-0.03, 1.03))
            ax.set_xticks(WIDTHS, labels=[str(w) for w in WIDTHS])
            ax.set_xticks([], minor=True)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("completion accuracy")
        axes[0].legend(fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _w, _d = pick_backbone(metrics)
    mo.md(rf"""
    ## Watching it answer, character by character

    Accuracy says *whether* a completion is right; per-character surprisal
    says *where* the model is uncertain along the way. Below, one example per
    eval set for the d{_w}-L{_d} backbone (seed {SEEDS[0]}), with the model's
    surprisal of each character drawn beneath the text (as a fraction of
    $\log |V|$, the uniform-guess ceiling). Operands are unpredictable by
    construction, so surprisal should spike at each operand's first characters
    and fall as the prefix pins down the rest. Everything after `=` is
    determined by the operands, so a model that has *computed* the mix glides
    through the answer at near-zero surprisal — even on operand pairs it has
    never seen. Where an answer is instead *guessed*, the surprisal stays
    high across the answer characters.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _w, _d = pick_backbone(metrics)
    (_cell,) = [r for r in metrics if r["label"] == label(_w, _d, SEEDS[0])]
    _rows = [(es, _cell["surprisal"][es][0]) for es in EVAL_SETS]
    _log_v = np.log(len(colors.alphabet()))
    _width = max(len(r["text"]) for _, r in _rows)

    def _subline(name: str, row: dict) -> str:
        values = np.concatenate([[np.nan], np.clip(np.array(row["nll"]) / _log_v, 0, 1)])
        svg = Subline(chars_per_line=_width).plot(row["text"], [Series(raw=values, label=name)])
        return f'<div style="max-width: 560px">{svg}</div>'

    mo.Html(
        '<div role="img" aria-label="Four short mixing equations, one per eval set, each with a sparkline '
        'of per-character surprisal drawn under the text, on a shared 0-to-log-V scale.">'
        + "".join(_subline(name, row) for name, row in _rows)
        + "</div>"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where color lives, before anchoring

    Probe R² against residual-stream depth (0 = embedding), one panel per probe
    target, one line per width (deepest models, mean over seeds). Rising R² for
    the *result* is the model visibly computing the mix before emitting it.

    The probes read the residual stream at two positions, highlighted below:
    the **first operand's last character** (by then the whole operand has been
    consumed, so its value can be represented) and the **space after `=`** —
    the last position before the answer is emitted, where the result must be
    ready.
    """)
    return


@app.cell(hide_code=True)
def _():
    _rng = np.random.default_rng(7)
    _exs = [
        colors.make_example("named", colors.PALETTE["red"], colors.PALETTE["blue"], _rng),
        colors.make_example("cross", colors.PALETTE["orange"], (2, 12, 7), _rng),
    ]

    def _mark(ex) -> str:
        p = ex.prompt
        _i, _j = len(p.split(" ")[0]) - 1, len(p) - 1
        hl = lambda ch, c, t: f'<span style="background: {c}; border-radius: 2px" title="{t}">{ch}</span>'  # noqa: E731
        chars = [
            hl(c, "#e4572e66", "operand read-out")
            if k == _i
            else hl("␣", "#4d9de066", "result read-out")
            if k == _j
            else c
            for k, c in enumerate(p)
        ]
        return "".join(chars) + f'<span style="opacity: 0.55">{ex.answer}</span>'

    mo.Html(
        '<pre style="line-height: 2.2; font-size: 1.05em">' + "<br>".join(_mark(ex) for ex in _exs) + "</pre>"
        '<p><span style="background: #e4572e66; border-radius: 2px">&nbsp;operand&nbsp;</span> probes read the '
        'first operand\'s color here; <span style="background: #4d9de066; border-radius: 2px">&nbsp;result&nbsp;'
        "</span> probes read the result's color and redness at the pre-answer space (shown as ␣). The dimmed "
        "answer is never probed.</p>"
    )
    return


@app.cell(hide_code=True)
def _(metrics):
    _probes = ["operand_rgb", "result_rgb", "result_redness"]

    @themed(
        name="probe-r2",
        alt_text=(
            "Three line charts of probe R-squared against residual-stream depth for the four-layer models, "
            "one panel per probe target: operand RGB, result RGB, and result redness. One line per width "
            "(16, 32, 64; darker is wider), averaged over seeds. R-squared for the operand rises within the "
            "first layers; the result targets rise later in depth."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2), sharey=True)
        shades = width_shades()
        d = max(DEPTHS)
        for ax, probe in zip(axes, _probes, strict=True):
            for w in WIDTHS:
                rows = [r["probe_r2"][probe] for r in metrics if r["label"].startswith(f"d{w}-L{d}-")]
                ax.plot(np.mean(rows, axis=0), "o-", color=shades[w], label=f"width {w}", lw=2)
            ax.set(title=probe.replace("_", " "), xlabel="residual depth", ylim=(-0.05, 1.05))
            ax.set_xticks(range(max(DEPTHS) + 1))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("probe R² (held-out half)")
        axes[0].legend(fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Do seeds agree on where *redness* points?

    For each pair of seeds (same architecture), the absolute cosine similarity
    between their fitted redness-probe directions, per layer. Random directions
    in n dimensions have |cos| ≈ 0.8/√n, marked as the dashed line. If the
    baseline geometry were seed-stable, anchoring would be redundant; scatter
    near the random line is the motivation for pinning the direction at
    training time.
    """)
    return


@app.cell(hide_code=True)
def _(weights):
    def _redness_cosines(w: int, d: int) -> np.ndarray:
        """Pairwise |cos| between seeds' redness probe directions: (n_pairs, depth+1)."""
        vecs = [weights[f"{label(w, d, s)}/result_redness"][:, :, 0] for s in SEEDS]  # (L+1, C) each
        unit = [v / np.linalg.norm(v, axis=1, keepdims=True) for v in vecs]
        return np.array([np.abs((unit[i] * unit[j]).sum(axis=1)) for i in range(3) for j in range(i + 1, 3)])

    @themed(
        name="probe-direction-agreement",
        alt_text=(
            "Line chart of the absolute cosine similarity between redness probe directions fitted on "
            "different seeds, against residual-stream depth, one line per width for the four-layer models. "
            "A dashed horizontal line marks the expected similarity of random directions for each width."
        ),
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        shades = width_shades()
        d = max(DEPTHS)
        for w in WIDTHS:
            cos = _redness_cosines(w, d)
            ax.plot(cos.mean(axis=0), "o-", color=shades[w], label=f"width {w}", lw=2)
            ax.axhline(0.8 / np.sqrt(w), color=shades[w], lw=1, ls="--", alpha=0.6)
        ax.set(xlabel="residual depth", ylabel="cross-seed |cos| of redness direction", ylim=(0, 1))
        ax.set_xticks(range(max(DEPTHS) + 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _best = pick_backbone(metrics)
    mo.md(
        f"""
    ## What this settles

    The backbone for the anchoring experiments: the smallest cell that
    saturates the unseen-pair eval sets —
    **width {_best[0]}, {_best[1]} layers**. D2.1.2 freezes that architecture
    and adds the anchor — pulling sequences labeled *red-ish* (by the same
    graded `redness` used for the probes here, applied as sparse noisy labels)
    toward a chosen direction at a chosen layer — then re-runs exactly these
    measurements. The comparison this report exists for: completion accuracy
    unchanged relative to the numbers above, and the redness probe direction
    landing where we put it instead of somewhere new every seed.
    """
    )
    return


if __name__ == "__main__":
    app.run()
