import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.1: the color-mixing transformer, un-anchored",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo  # noqa: F401
    import matplotlib.patheffects as pe
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
    from mini.vis import figure_html, light_dark, themed
    from sca.data import colors, cube
    from subline.series import Series
    from subline.subline import Subline

    use_publisher(report_bundle(__file__))

    EVAL_SETS = ["named_seen", "named_holdout", "hex_unseen", "cross_unseen"]

    def load_results() -> tuple[list[dict], dict[str, np.ndarray]] | None:
        """Resolve the metrics and probe weights from the store, or None if unpublished."""
        store = project_store()
        arts = store.get_refs([METRICS_REF, WEIGHTS_REF])
        m_art, w_art = arts[METRICS_REF], arts[WEIGHTS_REF]
        if m_art is None or w_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            m_path, w_path = store.get_many([(m_art, Path(d) / "metrics.json"), (w_art, Path(d) / "weights.npz")])
            metrics = json.loads(m_path.read_text())
            with np.load(w_path) as z:
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

    M2 asks whether Sparse Concept Anchoring carries over from autoencoders to
    transformers. Before we anchor anything we need a baseline, so this report
    trains a small transformer on a task built around color concepts that leave
    no room for ambiguity.

    The task is a character-level language of mixing equations on a 16-level RGB
    grid. Here are the sample types, which come up throughout:

    | Type | Example |
    |------|---------|
    | Named pairs | `red + blue = purple` |
    | Hex pairs   | `#f00 + #00f = #808`  |
    | Cross-form  | `red + #00f = #808`   |
    | Alias       | `red = #f00`          |

    Every operand spans several characters in both of its spellings. The hope is
    that this pushes the model toward red-the-concept rather than the literal
    token `red`.

    Mixing (`+`) is the channel-wise round-half-up mean, so each prompt has one
    correct completion.

    We sweep width {16, 32, 64} × depth {2, 4} × 3 seeds ([experiment
    definition](./experiment.py)), and for each cell we measure two things.

    The first is completion accuracy: greedy decoding, scored as an exact string
    match, over four evaluation sets. Those are named pairs seen in training;
    held-out named pairs, which never appear as named equations, so the model
    has to combine the alias dictionary with hex arithmetic to answer them;
    hex-only equations; and cross-form operand pairs that were never shown
    together.

    The second is a set of probes: ridge regression from the residual stream at
    each depth out to the operand color, the result color, and the result's
    *redness*. A probe is a small linear model we fit on the model's internal
    activations to read out what those activations carry. Ridge regression is
    linear regression with a penalty on large weights, which keeps the fit
    stable.

    Three things we expect to see. First, a small nGPT should learn the task,
    with near-perfect accuracy on seen forms and on unseen *hex* pairs; that
    leaves the anchored runs room to show any degradation later. Second, color
    should be linearly readable from the residual stream, more so as depth
    increases. Third, the redness probe directions should vary from seed to
    seed, landing somewhere different on each run. That last point is part of
    the motivation for this work: searching for a concept after training turns
    up a different geometry every time, whereas SCA should let us fix the
    location in advance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training data

    The corpus sampler is deterministic. Regenerating it here with the
    experiment's own constants gives back the same training data the model saw,
    and these are its first lines:
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
                + f". Between them they cover {len(_pairs):,} distinct operand pairs, "
                f"**{len(_pairs) / _all_pairs:.2%}** of the grid's {_all_pairs / 1e6:.1f}M. So the "
                f"unseen-pair eval sets, sampled to steer clear of all of them, test the mixing rule "
                f"rather than recall. "
            ),
        ]
    )
    return holdout, train_pairs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The color space

    Every color is a point on an RGB grid with 16 levels per channel, so 16³ =
    4096 points in all. If we rotate the cube so its black-to-white diagonal
    stands vertical, *value* runs up the page and hue wraps around it. That is
    the figure below, seen front-on toward the *red* corner. Hex and cross
    equations draw their operands from anywhere in this cube.
    """)
    return


@app.cell(hide_code=True)
def _():
    @themed(
        name="color-space-cube",
        alt_text=(
            "An orthographic front view of the RGB grid, rotated so the black-to-white diagonal is vertical: "
            "black at the bottom, white at the top, hues fanned around the middle, showing the red, green, and "
            "magenta faces. Each of the 4096 grid colors is a filled dot, packed densely enough to read as a "
            "smooth solid."
        ),
        caption="The 16³ hex grid",
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        cube.draw_rgb_cube(ax, cube.grid(), side="front", s=70)
        ax.set_facecolor("none")  # drop the panel fill — it only adds clutter here
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The 27 named colors sit only on the {0, 8, 15}³ sub-lattice, the cube's
    corners and edge midpoints. Note that no *color* is held out: every point
    shows up in training, since hex operands are sampled over the whole grid and
    each name appears in an alias line. What we hold out is operand *pairs*, both
    named and hex.
    """)
    return


@app.cell(hide_code=True)
def _(holdout, train_pairs):
    # Split the two-panel lattice into two independent figures so they reflow and shrink
    # separately: on a narrow screen the pair stacks instead of shrinking as a block,
    # keeping each panel legible. The panels used to share a y-axis; now that they are
    # separate figures, an identical figsize and identical data-driven limits give them the
    # same lattice scale — and, since the tight crop is dominated by that shared axes box,
    # the same size — without a shared axis. The lettered tags sit just outside those limits
    # (annotation_clip=False) and are picked up by the crop.
    _vals = list(colors.PALETTE.values())
    _idx = {c: i for i, c in enumerate(_vals)}
    _train_edges = [p for p in train_pairs if p[0] != p[1]]  # self-pairs are just the vertices
    _named = cube.named()
    # A single front view suffices: the lattice is mostly empty, so nothing hides behind it.
    _x, _y, _depth = cube.project(_named, "front")
    _dmin, _dspan = _depth.min(), _depth.max() - _depth.min()
    _cx, _cy = _x.mean(), _y.mean()
    # Square limits from the lattice vertices alone — a small margin, no room reserved for the
    # lettered tags. Identical across panels (same data), so the two figures come out the same
    # size; the tags draw just past the edge and the tight crop still lands identically.
    _half = max(_x.max() - _x.min(), _y.max() - _y.min()) / 2 + 0.12
    _xlim, _ylim = (_cx - _half, _cx + _half), (_cy - _half, _cy + _half)

    def _panel(bold, other, examples) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.2, 4.4))
        faint, vedge = light_dark("#0001", "#fff1"), light_dark("#0006", "#fff7")
        ink, halo = light_dark("#111", "#eee"), light_dark("#fff", "#111")

        def _edge(pair, lw_lo, lw_hi, **kw):
            u, v = _idx[pair[0]], _idx[pair[1]]
            mid = (_depth[u] + _depth[v]) / 2  # orders each edge against the vertices for occlusion
            # Taper by depth: front-facing edges read heavy, back/interior ones recede.
            lw = lw_lo + (lw_hi - lw_lo) * (mid - _dmin) / _dspan
            ax.plot([_x[u], _x[v]], [_y[u], _y[v]], lw=lw, zorder=float(mid), solid_capstyle="round", **kw)

        def _letter(name, ch):
            i = _idx[colors.PALETTE[name]]
            ang = np.arctan2(_y[i] - _cy, _x[i] - _cx)  # nudge the tag radially outward, clear of the lattice
            if name == "white":
                ang = np.radians(150)  # apex points straight at the title; send it up-left instead
            ax.annotate(
                ch,
                (_x[i], _y[i]),
                (_x[i] + np.cos(ang) * 0.12, _y[i] + np.sin(ang) * 0.12),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                fontstyle="italic",
                color=ink,
                zorder=100,
                annotation_clip=False,
                path_effects=[pe.withStroke(linewidth=3, foreground=halo)],
            )

        for _p in other:
            _edge(_p, 0.4, 2.0, color=faint)
        for _p in bold:
            _result = tuple(np.array(colors.mix(*_p)) / (colors.N_LEVELS - 1))
            _edge(_p, 1.4, 3.0, color=_result)
        # Vertices in true color; +ε on zorder so a vertex wins a depth tie with an edge.
        for _i in range(len(_vals)):
            ax.scatter(_x[_i], _y[_i], c=[_named[_i]], s=60, edgecolors=vedge, lw=0.6, zorder=float(_depth[_i]) + 1e-3)
        _u, _v, _chs = examples
        _letter(_u, _chs[0])
        _letter(_v, _chs[1])
        cube.style_cube_axes(ax, labels=False)
        ax.set_facecolor("none")  # drop the panel fill — it only adds clutter here
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)
        return fig

    _train_alt = (
        "An orthographic front view of the 27 named colors as a lattice in the rotated RGB cube, value "
        "vertical with black at the bottom and white at the top. Each named color is a small dot in its true "
        "color. The pairs used as named equations in training are bold and colored by the color their two "
        "operands mix to; the held-out pairs are drawn faint for context. One training edge is picked out on "
        "the cube's silhouette with italic letters a and b at its endpoints (white and magenta), the worked "
        "example a + b = orchid. The panel has no background fill or axes; front-facing edges are heavier than "
        "back and interior ones, so the lattice reads three-dimensionally. Titled 'train'."
    )
    _holdout_alt = (
        "The same orthographic front view of the 27 named colors in the rotated RGB cube, same orientation and "
        "styling as the train panel. Here the pairs held out for the named-holdout evaluation are bold and "
        "colored by their mixed result, with the training pairs drawn faint for context. One held-out edge is "
        "picked out on the cube's silhouette with italic letters c and d at its endpoints (magenta and blue), "
        "the worked example c + d = violet. Titled 'held out for eval'."
    )
    _left = themed(
        lambda: _panel(_train_edges, holdout, ("white", "magenta", "ab")),
        name="named-pair-lattice-train",
        alt_text=_train_alt,
        caption="Train",
    )()
    _right = themed(
        lambda: _panel(holdout, _train_edges, ("magenta", "blue", "cd")),
        name="named-pair-lattice-holdout",
        alt_text=_holdout_alt,
        caption="Held out for eval",
    )()
    # Two sub-figures under one caption: figure_html nests the themed panels in a <figure>
    # that the `figure:has(> figure)` rule in report.css reflows to a stack on a narrow screen.
    mo.Html(figure_html(f"{_left}{_right}", caption="Named pairs on the cube"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    Both figures show the same lattice from the front. The vertices are the
    named colors, and each edge joins the two operands of a named pair. An edge's
    midpoint, which is also its color, is the answer that equation should
    produce.

    Two edges are labeled as worked examples:

    - $a$–$b$ (train): {colors.swatch("white")} + {colors.swatch("magenta")} = {colors.swatch("orchid")}
    - $c$–$d$ (held out): {colors.swatch("magenta")} + {colors.swatch("blue")} = {colors.swatch("violet")}

    Only these connected pairs ever appear as *named* equations with a named
    answer. Every other operand pair the model sees is written in hex or cross
    form, and those draw their operands from the full 16³ grid.

    A held-out edge like `magenta + blue = violet` can be answered two ways. One
    is recall, which is ruled out here, since that named rendering never appears
    in training. The other is composition: look up both names through the alias
    lines, mix them as if they had been written in hex, and translate the result
    back into a name. Composition is what the `named_holdout` eval set measures.

    The `hex_unseen` and `cross_unseen` sets are sampled at evaluation time from
    the full grid, steering clear of every operand pair the corpus used.
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
        f"Here are the headline numbers. Accuracy on unseen hex pairs spans "
        f"**{min(_hex):.2f}–{max(_hex):.2f}** across the sweep, while held-out named "
        f"pairs, the compositional test, span **{min(_hold):.2f}–{max(_hold):.2f}**. "
        f"The figures below break this down by cell and eval set."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Completion accuracy across the sweep

    Each panel below is one eval set. Accuracy runs up the y-axis against width
    along the x-axis, with one line per depth (the mean over seeds) and the
    individual seeds shown as faint points.

    The named-holdout panel is the one to watch. It can only be solved by
    combining the alias dictionary with the mixing arithmetic, and we find that
    the model never learns to do that.
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

    Let's look at where the model was unsure as it read each sequence.

    For the d{_w}-L{_d} model (seed {SEEDS[0]}), we plot one example per eval set
    and draw two series beneath the text, both as fractions of $\log |V|$, the
    value a uniform guess over the vocabulary would give. The first is the
    model's surprisal at each character: how startled it is by the character that
    actually comes next. The second is the entropy of its predictive
    distribution, the surprisal it expected on average before seeing that
    character. Surprisal is the negative log-probability the model assigned to
    the true character, so a character it was sure of costs little and a shock
    costs a lot; entropy is the mean surprisal the model's own distribution
    implies.

    Operands are unpredictable by design, so both series should spike at the
    first characters of each operand and settle as the prefix pins down the rest.
    Everything after `=` follows from the operands, so a model that has worked
    out the mix should coast through the answer at near-zero surprisal, even on
    operand pairs it has never seen. When instead it guesses the answer,
    surprisal climbs across the answer characters.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _w, _d = pick_backbone(metrics)
    (_cell,) = [r for r in metrics if r["label"] == label(_w, _d, SEEDS[0])]
    # named_holdout's index 1 is `lime + black = green`, the example "Why the
    # named answers fail" walks through below; index 0 is a same-set spare.
    _idx = {"named_holdout": 1}
    rows = [(es, _cell["surprisal"][es][_idx.get(es, 0)]) for es in EVAL_SETS]
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
            label = f'<span style="font-size: 11px; font-family: monospace; opacity: 0.65">{name}</span>'
            return figure_html(svg, caption=label, style="display: inline-block; margin: 0 .5em")

        strip = "".join(one(name, row) for name, row in rows)
        return mo.Html(externalize_html(figure_html(strip, aria_label=aria_label), name=name))

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
    The gap between those two series is [the surprisal beyond what the model expected](https://www.lesswrong.com/posts/Kjo64rSWkFfc3sre5/detecting-out-of-distribution-text-with-surprisal-and#5__Surprise_surprise__A_new_metric),

    $$s_2 = \frac{i - h}{\log |V|}$$

    where $i$ is the surprisal and $h$ the entropy. This sits near zero when the
    model's confidence matched the outcome, whether it was confident and right or
    unsure and fairly caught out. It goes positive when the model was confidently
    wrong, and negative when the character was more predictable than its
    distribution suggested. The sparkline clips at zero, so we draw the negative
    values as a second, flipped series, $-s_2$.
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
    The spike lands on `named_holdout`, the one set this sweep never solves
    (accuracy 0 above). The model stays committed on those answers: entropy
    stays low while the true characters arrive as a surprise, so $s_2$ reads as
    confidently wrong rather than merely unsure. What is the model so sure of?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why the named answers fail

    That sparkline is teacher-forced: we feed the model the true answer from the
    validation set one character at a time and watch how much each one surprises
    it. Two of the model's own preferred names show through.

    The first letter already costs something. Left to choose, this seed opens
    `lime + black` with `t`, for *teal*, so the true `g` arrives as a mild
    surprise (the small bump above, around 0.6 of the uniform-guess ceiling).
    Once it is forced onto `g`, the model gets `r` for free, since *gray* and
    *green* share the prefix `gr`. Then the true `e` is where it comes apart: on
    the `gr…` branch the model is all but sure the word is *gray*, and `e` is the
    first character that rules *gray* out. The spike is the model fluently
    spelling a different palette name, then being surprised when the truth
    arrives.

    Left to run on its own (below), it writes *teal* instead of *green* or
    *gray*, a one-channel neighbor of the true mix, and the tall spike on the `a`
    of `black` hints at why. After `lime + bl` the model is 99.9% sure the second
    operand is *blue*, and `lime + blue = teal` is an equation it trained on. The
    `a` is the moment that guess breaks. It fixes the operand to `black` right
    away, but only half-fixes the answer: the correction lifts *green* about 70×
    (to 13%) yet leaves the trained *teal* on top. The result-form rule, which
    says a named answer appears exactly when both operands are named, holds up:
    the model commits to a name every time. The difficulty is choosing which
    name, and a trained neighbor can win out before the arithmetic finishes.

    The experiment publishes its checkpoints alongside the metrics, so we can ask
    the model directly. Below is every held-out pair, prompted exactly as in the
    `named_holdout` eval set, with one column per seed of the backbone
    architecture.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    from sca.compute.evaluation import greedy_completions
    from sca.compute.model import load_checkpoint
    from sca.data.tokenizer import CharTokenizer

    backbone = pick_backbone(metrics)
    _store = project_store()
    _refs = {s: f"{CKPT_REF}/{label(*backbone, s)}" for s in SEEDS}
    _resolved = _store.get_refs(_refs.values())
    _arts = {s: a for s, r in _refs.items() if (a := _resolved[r]) is not None}
    mo.stop(
        len(_arts) < len(SEEDS),
        mo.md("The checkpoints aren't in the store yet — re-run the experiment to publish them."),
    )
    _models = {}
    with tempfile.TemporaryDirectory() as _tmp:
        _store.get_many([(_art, Path(_tmp) / str(_s) / "model") for _s, _art in _arts.items()])
        for _s in _arts:
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

    _head = (
        f"<tr><th>prompt</th><th>{colors.swatch(None)} expected</th>"
        + "".join(f"<th>{colors.swatch(None)} seed {s}</th>" for s in SEEDS)
        + "</tr>"
    )
    _rows = "".join(
        f"<tr><td><code>{ex.prompt}</code></td><td>{colors.swatch(ex.answer)}</td>"
        + "".join(f"<td>{colors.swatch(_by_seed[s][i])}</td>" for s in SEEDS)
        + "</tr>"
        for i, ex in enumerate(named_holdout_exs)
    )
    _w, _d = backbone
    _table = (
        '<div class="report-table-scroll">'
        f'<table class="report-table" style="font-size: 0.9em">{_head}{_rows}</table>'
        "</div>"
    )
    _caption = mo.md(f"Greedy completions of the `named_holdout` prompts, d{_w}-L{_d}, all seeds.").text
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return (named_holdout_exs,)


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
    mo.md(rf"""
    The model never answers these in hex. It always reaches for a name, and the
    names are wrong in telling ways: usually a palette neighbor of the true mix,
    sometimes an echo of one operand (`olive + lavender = lavender`). The seeds
    mostly agree on the same wrong answers, which points to a systematic bias
    that looks like retrieval of the nearest memorized named equation.

    The mixing arithmetic itself is fine. Prompted with the very same held-out
    value pairs, seed {SEEDS[0]} solves **{_scores["hex"]}/{_n}** in hex form and
    **{_scores["cross"]}/{_n}** in cross form, against **{_scores["named"]}/{_n}**
    as named equations.
    """)
    return


@app.cell(hide_code=True)
def _(train_pairs):
    _reps = round(N_EXAMPLES * colors.FORM_WEIGHTS["named"] / len(train_pairs))
    mo.md(rf"""
    Three properties of the corpus may make this hard for the model.

    1. The named slice is memorizable. Named equations draw on only
       {len(train_pairs)} distinct pairs, so each one is seen about {_reps} times
       in training. A lookup table is enough, and the model evidently builds one
       (`named_seen` ≈ 1). Once the loss on that slice reaches zero, nothing
       nudges the model toward the compositional route.
    2. The alias dictionary runs one way. Alias lines always read `name = hex`,
       and nothing supervises the reverse direction except those memorizable
       named equations. This may be an instance of the *reversal curse*, where
       training on `A = B` does not by itself teach `B = A`.
    3. Hex answers factor apart per channel, and named answers do not. A hex
       answer comes out digit by digit, and each digit depends on one channel of
       the operands, so no single position needs the whole mix at once. A
       *name's* first character depends on all three channels and the inverted
       dictionary at the same time. The probe section below looks at this more.

    So the training signal is strong enough; it is just easy to satisfy by
    lookup. A few corpus changes might help: reverse some alias lines
    (`#f00 = red`); add named operands whose off-palette mix forces a hex answer
    (`red + navy = #804`), so that `name + name` prompts have to engage the
    arithmetic instead of the lookup table; and use a denser named palette, so
    memorization is harder.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _, _d = pick_backbone(metrics)
    mo.md(rf"""
    ## Where color is represented

    Here we fit the probes at each residual-stream depth (depth 0 is the
    embedding) and plot their R² against depth. R² is the fraction of the
    target's variance the probe recovers, so 1 means the color is fully readable
    from the stream and 0 means it is not there linearly. The figure has one
    panel per probe target and one line per width. We test only the deepest
    models (L{_d}) and show the mean over seeds.
    """)
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
    Rising R² for the *result* means the mix becomes partly readable before the
    answer starts. It plateaus well below the operand's R², though, even in cells
    whose hex accuracy is perfect. The reason may be that the full mix never has
    to sit at any one position: each hex digit can be worked out at the position
    that emits it, so the pre-answer probe catches at most a head start. Probing
    every answer position, per channel, would map that spread-out schedule; that
    is a follow-up.

    The probes read the residual stream at two positions, marked below. The first
    is the first operand's last character, where the whole operand has been read
    in, so its value can be represented. The second is the space after `=`, the
    last position before the answer begins, where the result has to be ready.
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
        'first operand\'s color at this character; <span style="background: #4d9de066; border-radius: 2px">&nbsp;result&nbsp;'
        "</span> probes read the result's color and redness at the space just before the answer (shown as ␣). The "
        "dimmed answer is never probed.</p>"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Do seeds agree on where *redness* points?

    For each pair of seeds trained with the same architecture, we take their
    fitted redness-probe directions and measure the absolute cosine similarity
    between them, layer by layer. Cosine similarity is the cosine of the angle
    between two vectors: 1 means they point the same way, 0 means they are at
    right angles. Two random directions in n dimensions sit near |cos| ≈ 0.8/√n,
    drawn here as the dashed line.

    If this geometry were stable across seeds, there would be little point in
    anchoring. The spread we see is part of why we want to pin the direction down
    at training time.
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
    ## Findings

    The smallest cell that saturates the unseen-pair eval sets is **width {_best[0]}, {_best[1]} layers**.
    That points toward a network of similar capacity for the experiments to come.
    For D2.1, we can take that architecture as the baseline and add the anchor,
    which pulls sequences labeled *red-ish* (supplied as sparse, noisy labels)
    toward a chosen direction at chosen layers. Then we re-run these measurements
    and compare the anchored and baseline versions.

    One caveat: the held-out named pairs sit at zero validation accuracy, so that
    set gives the anchored runs no headroom and probably can't help us spot any
    unintended degradation.
    """
    )
    return


if __name__ == "__main__":
    app.run()
