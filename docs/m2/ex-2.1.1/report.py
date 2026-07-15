import marimo

__generated_with = "0.23.9"
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
        CKPT_REF,
        CORPUS_SEED,
        DEPTHS,
        HOLDOUT_FRAC,
        METRICS_REF,
        N_EXAMPLES,
        SEEDS,
        WEIGHTS_REF,
        WIDTHS,
    )
    from mini.reports import externalize_html, report_bundle, use_publisher
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
    return holdout, train_pairs


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
    eval set for the d{_w}-L{_d} backbone (seed {SEEDS[0]}), with two series
    drawn beneath the text (both as fractions of $\log |V|$, the uniform-guess
    ceiling): the model's surprisal of each character, and the entropy of its
    predictive distribution — the surprisal it *expected*, before seeing the
    character. Operands are unpredictable by construction, so both should
    spike at each operand's first characters and fall as the prefix pins down
    the rest. Everything after `=` is determined by the operands, so a model
    that has *computed* the mix glides through the answer at near-zero
    surprisal — even on operand pairs it has never seen. Where an answer is
    instead *guessed*, the surprisal stays high across the answer characters.
    Where the two series track each other, the model knew how uncertain it
    was; a surprisal spike above the entropy line means it was caught out.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _w, _d = pick_backbone(metrics)
    (_cell,) = [r for r in metrics if r["label"] == label(_w, _d, SEEDS[0])]
    rows = [(es, _cell["surprisal"][es][0]) for es in EVAL_SETS]
    log_v = np.log(len(colors.alphabet()))
    sub_width = min(max(len(r["text"]) for _, r in rows), 80)
    # Match the sublines' dark background to this notebook's, rather than subline's
    # neutral default; light mode already matches. `css` overrides the library's own
    # `--bg-color` (later rule wins).
    sub_css = "svg { --bg-color: light-dark(#fff, #181c1a); }"

    def sublines(rows: list[tuple[str, dict]], series, aria_label: str, name: str) -> mo.Html:
        """Lay out one captioned subline per eval set; `series(row)` builds its series list.

        The block is inlined (so the SVGs share the page's CSS) *and* externalized as
        `_assets/<name>.html` — a plain file for tooling that can't run the frontend.
        """

        def one(name: str, row: dict) -> str:
            svg = Subline(chars_per_line=sub_width, css=sub_css).plot(row["text"], series(row))
            caption = f'<figcaption style="font-size: 11px; font-family: monospace; opacity: 0.65">{name}</figcaption>'
            return f'<figure style="display: inline-block; margin: 0 1em 0 0">{svg}{caption}</figure>'

        html = (
            f'<div role="img" aria-label="{aria_label}" style="text-wrap: balance">'
            + "".join(one(name, row) for name, row in rows)
            + "</div>"
        )
        return mo.Html(externalize_html(html, name=name))

    def pad(row: dict, key: str) -> np.ndarray:
        """Scale to fractions of log |V| and align with the text: position 0 has no prediction."""
        return np.concatenate([[np.nan], np.asarray(row[key]) / log_v])

    return pad, rows, sublines


@app.cell(hide_code=True)
def _(pad, rows, sublines):
    def _series(row: dict) -> list[Series]:
        return [
            Series(raw=np.clip(pad(row, "nll"), 0, 1), label="surprisal"),
            Series(raw=np.clip(pad(row, "entropy"), 0, 1), label="entropy", dasharray="3 2"),
        ]

    sublines(
        rows,
        _series,
        "Four short mixing equations, one per eval set, each with a sparkline of per-character "
        "surprisal (solid) and predictive entropy (dashed) drawn under the text, on a shared "
        "0-to-log-V scale. The two series track each other, spiking at operand starts and "
        "staying near zero across the answers — except named holdout, where surprisal rises "
        "well above entropy on the answer characters.",
        name="sublines-surprisal",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The gap between those two series is a signal in its own right:
    *surprise-surprise*, the surprisal in excess of what the model expected,

    $$s_2 = \frac{i - h}{\log |V|}$$

    where $i$ is the surprisal and $h$ the entropy. It is near zero where the
    model knew its own uncertainty (confident *and* right, or uncertain and
    merely unlucky), positive where it was caught out, and negative where the
    character was more predictable than the model's distribution let on. The
    sparkline clips at zero, so the negative lobe is drawn as a second,
    flipped series, $-s_2$: solid marks *caught out*, dashed marks *mundane*.
    """)
    return


@app.cell(hide_code=True)
def _(pad, rows, sublines):
    def _s2(row: dict) -> np.ndarray:
        return pad(row, "nll") - pad(row, "entropy")

    def _series(row: dict) -> list[Series]:
        return [
            Series(raw=np.clip(_s2(row), None, 1), label="s₂"),
            Series(raw=np.clip(-_s2(row), None, 1), label="−s₂", dasharray="3 2"),
        ]

    sublines(
        rows,
        _series,
        "The same four equations, now with sparklines of surprise-surprise: surprisal minus "
        "entropy as a fraction of log V. The solid series shows the positive part (more surprised "
        "than expected); the dashed series shows the negative part flipped above zero (less "
        "surprised than expected). Three sets stay close to the baseline; named holdout shows "
        "tall positive spikes across its answer characters.",
        name="sublines-surprise-surprise",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The spike is on `named_holdout` — the one set this sweep never solves
    (accuracy 0 above). The model does not hedge on those answers: entropy
    stays low while the true characters arrive as a surprise, so $s_2$ reads
    *confidently wrong*, not *uncertain*. What is it confidently wrong
    about, exactly?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why the named answers fail

    Look at where that spike lands. In `lime + black = green` above, the
    model passes `g` cheaply and gets `r` for free; the surprisal only jumps
    at the `e` — the first character that separates *green* from *gray*. It
    is fluently spelling a color name, just the wrong one. So the
    result-form rule (a named answer exactly when both operands are named)
    is not the weak link: the model commits to a name every time. The
    failure is in choosing *which* name.

    The experiment publishes its checkpoints alongside the metrics, and
    these models are small enough to query on CPU, so we can ask directly.
    Below, every held-out pair, prompted exactly as in the `named_holdout`
    eval set, one column per seed of the backbone architecture:
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    from sca.compute.evaluation import greedy_completions
    from sca.compute.model import load_checkpoint
    from sca.data.tokenizer import CharTokenizer

    backbone = pick_backbone(metrics)
    _store = project_store()
    _arts = {s: a for s in SEEDS if (a := _store.get_ref(f"{CKPT_REF}/{label(*backbone, s)}")) is not None}
    mo.stop(
        len(_arts) < len(SEEDS),
        mo.md("The checkpoints aren't in the store yet — re-run the experiment to publish them."),
    )
    _models = {}
    with tempfile.TemporaryDirectory() as _tmp:
        for _s, _art in _arts.items():
            _store.get(_art, Path(_tmp) / str(_s) / "model")
            _model, _config, _ = load_checkpoint(Path(_tmp) / str(_s))
            _models[_s] = (_model, CharTokenizer(_config.tokenizer))

    def complete(seed: int, prompts: list[str]) -> list[str]:
        """Greedy completions from the backbone cell trained with *seed*."""
        model, tok = _models[seed]
        return greedy_completions(model, tok, prompts, 12)

    return backbone, complete


@app.cell(hide_code=True)
def _(backbone, complete, holdout):
    named_holdout_exs = colors.as_named(holdout, seed=2)  # the eval set, verbatim
    _by_seed = {s: complete(s, [ex.prompt for ex in named_holdout_exs]) for s in SEEDS}

    def _swatch(text: str) -> str:
        rgb = colors.PALETTE.get(text)
        if rgb is None:
            return f"<code>{text}</code>"
        return (
            f'<span aria-hidden="true" style="background: {colors.to_hex(rgb)}; border: 1px solid #8886; '
            f'border-radius: 2px; display: inline-block; width: 0.8em; height: 0.8em"></span> {text}'
        )

    _head = "<tr><th>prompt</th><th>expected</th>" + "".join(f"<th>seed {s}</th>" for s in SEEDS) + "</tr>"
    _rows = "".join(
        f"<tr><td><code>{ex.prompt}</code></td><td>{_swatch(ex.answer)}</td>"
        + "".join(f"<td>{_swatch(_by_seed[s][i])}</td>" for s in SEEDS)
        + "</tr>"
        for i, ex in enumerate(named_holdout_exs)
    )
    _w, _d = backbone
    mo.vstack(
        [
            mo.Html(f'<table style="font-size: 0.9em">{_head}{_rows}</table>'),
            mo.md(f"*Greedy completions of the `named_holdout` prompts, d{_w}-L{_d}, all seeds.*"),
        ]
    )
    return (named_holdout_exs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Two things stand out. The answers are always names, never hex, and they
    are wrong in suggestive ways: usually a palette neighbor of the true
    mix, sometimes a bare operand echo (`olive + lavender = lavender`). And
    the seeds largely agree on the *same* wrong answers, so this is a
    systematic bias, not decoding noise — it looks like retrieval of the
    nearest memorized named equation, not a computed mix.

    The mixing arithmetic itself is not the problem, because the very same
    value pairs are solved whenever the prompt licenses a hex answer:
    """)
    return


@app.cell(hide_code=True)
def _(complete, holdout, named_holdout_exs):
    _rng = np.random.default_rng(9)
    _sets = {
        "named": named_holdout_exs,
        "cross": [colors.make_example("cross", a, b, _rng) for a, b in holdout],
        "hex": [colors.make_example("hex", a, b, _rng) for a, b in holdout],
    }
    _scores = {}
    for _form, _exs in _sets.items():
        _got = complete(SEEDS[0], [ex.prompt for ex in _exs])
        _scores[_form] = sum(g == ex.answer for g, ex in zip(_got, _exs, strict=True))
    _n = len(named_holdout_exs)
    _rev_prompts = sorted({f"{colors.to_hex(ex.result)} = " for ex in named_holdout_exs})
    _rev = complete(SEEDS[0], _rev_prompts)
    mo.md(
        f"Seed {SEEDS[0]}, the same held-out value pairs in each surface form: "
        + ", ".join(f"**{_scores[form]}/{_n}** {form}" for form in _sets)
        + ". And prompted with the *reverse* of an alias line — a frame that never occurs in training — "
        "it emits hex-shaped noise where a name should go:\n\n```\n"
        + "\n".join(f"{p}{g}" for p, g in zip(_rev_prompts, _rev, strict=True))
        + "\n```"
    )
    return


@app.cell(hide_code=True)
def _(train_pairs):
    _reps = round(N_EXAMPLES * colors.FORM_WEIGHTS["named"] / len(train_pairs))
    mo.md(rf"""
    That last block is the missing piece made visible: the model has no
    usable hex → name mapping at all. Three properties of the corpus
    conspire to keep it that way:

    1. **The named slice is memorizable.** Named equations draw from only
       {len(train_pairs)} distinct pairs, so each is seen ~{_reps} times in
       training. A lookup table suffices, and the model evidently builds one
       (`named_seen` ≈ 1); once that slice's loss is zero, nothing pushes it
       to learn the compositional route instead.
    2. **The alias dictionary is one-way.** Alias lines always read
       `name = hex`. The reverse direction is supervised nowhere except
       through those memorizable named equations — the small-scale analog of
       the *reversal curse*: training on `A = B` does not produce `B = A`.
    3. **Hex answers factorize per channel; named answers don't.** A hex
       answer is emitted digit by digit, and each digit depends on one
       channel of the operands — nothing ever requires the whole mix at one
       position. A *name's* first character depends on all three channels
       and the inverted dictionary simultaneously. The named path needs a
       readout that the (dominant) hex task never builds. The probe section
       below is consistent with this: the result's R² plateaus well below
       the operand's even in cells with perfect hex accuracy.

    The signal, then, is not too weak — it is too easy to satisfy by lookup.
    Candidate corpus fixes are queued in the repo's todo list: reverse alias
    lines (`#f00 = red`); named operands whose off-palette mix forces a hex
    answer (`red + navy = #804`), so that name + name prompts must engage
    the arithmetic rather than the lookup table; and a denser named palette,
    to make memorization the expensive strategy. Until one of those lands,
    `named_holdout` sits at zero for corpus reasons, not capacity ones.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where color lives, before anchoring

    Probe R² against residual-stream depth (0 = embedding), one panel per probe
    target, one line per width (deepest models, mean over seeds). Rising R² for
    the *result* is the mix becoming partially readable before the answer is
    emitted — but note that it plateaus well below the operand's R², even in
    cells whose hex accuracy is perfect. The full mix need never sit at any
    single position: each hex digit can be computed at the position that emits
    it, so the pre-answer probe sees at most a head start. Probing every
    answer position, per channel, would map that lazy schedule directly; a
    follow-up. For the anchoring runs the pre-answer space stays the
    comparison point, and the contrast across runs matters more than the
    absolute level.

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

    One caveat travels with the baseline: `named_holdout` sits at zero for
    corpus reasons (see *Why the named answers fail*), so it offers the
    anchored runs no headroom as a degradation canary until the corpus
    grows the fixes queued in the todo list.
    """
    )
    return


if __name__ == "__main__":
    app.run()
