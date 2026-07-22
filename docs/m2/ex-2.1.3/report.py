import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.3: names all the way down",
    css_file="../../report.css",
    auto_download=["html"],
)

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
        ARRAYS_REF,
        EVALS_REF,
        HOLDOUT_FRAC,
        METRICS_REF,
        N_EXAMPLES,
        SEEDS,
    )
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import figure_html, light_dark, themed
    from sca import baselines as bl
    from sca.compute.evaluation import ridge_probe, ridge_probe_loo
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS, load_example_sets, to_hex
    from sca.vis import plot_rgb_cube

    use_publisher(report_bundle(__file__))

    GRID_NAMES = list(nc.GRIDS)
    PALETTES = {g: nc.grid_palette(nc.GRIDS[g]) for g in GRID_NAMES}
    VOCAB_RGB = {g: np.array(list(p.values()), dtype=np.float32) / (N_LEVELS - 1) for g, p in PALETTES.items()}

    def load_results() -> tuple[dict, dict[str, np.ndarray], dict[str, dict]] | None:
        """Resolve metrics, arrays, and eval sets from the store, or None if unpublished."""
        store = project_store()
        arts = store.get_refs([METRICS_REF, ARRAYS_REF, *(f"{EVALS_REF}/{g}" for g in GRID_NAMES)])
        m_art, a_art = arts[METRICS_REF], arts[ARRAYS_REF]
        e_arts = {g: arts[f"{EVALS_REF}/{g}"] for g in GRID_NAMES}
        if m_art is None or a_art is None or any(v is None for v in e_arts.values()):
            return None
        with tempfile.TemporaryDirectory() as d:
            paths = store.get_many(
                [
                    (m_art, Path(d) / "metrics.json"),
                    (a_art, Path(d) / "arrays.npz"),
                    *((art, Path(d) / f"{g}.json") for g, art in e_arts.items() if art is not None),
                ]
            )
            metrics = json.loads(paths[0].read_text())
            with np.load(paths[1]) as z:
                arrays = {k: z[k] for k in z.files}
            evals = {p.stem: load_example_sets(p.read_bytes()) for p in paths[2:]}
        return metrics, arrays, evals

    def cell_of(metrics: dict, grid: str, s: int) -> dict:
        (r,) = [r for r in metrics["cells"] if r["label"] == f"{grid}-s{s}"]
        return r

    def grid_shades() -> dict[str, tuple]:
        stops = light_dark([0.82, 0.55, 0.32, 0.08], [0.88, 0.62, 0.42, 0.2])
        return dict(zip(GRID_NAMES, plt.cm.viridis(stops), strict=True))

    def sw(name: str, grid: str) -> str:
        """Inline swatch + name for any grid's palette (the classic `swatch` knows only the 27)."""
        rgb = PALETTES[grid].get(name)
        if rgb is None:
            return f"<code>{name}</code>"
        return f'<span class="sw" style="--sw: {to_hex(rgb)};" aria-hidden="true"></span> {name}'

    def dists_for(grid: str, exs: list, logp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(guess distance, floor distance) per example, unit-cube units."""
        result = np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)
        d = np.linalg.norm(VOCAB_RGB[grid][None] - result[:, None], axis=2)
        return d[np.arange(len(exs)), logp.argmax(axis=1)], d.min(axis=1)

    def raw_rgb(grid: str) -> np.ndarray:
        """(V, 3) raw 0..15 channel values, which `sca.baselines` works in."""
        return np.array(list(PALETTES[grid].values()))

    def dist_matrix(grid: str, exs: list) -> np.ndarray:
        """(N, V) distance from every vocabulary color to each example's true mix."""
        return bl.distances(raw_rgb(grid), [ex.result for ex in exs])

    def exact_null(grid: str, evals: dict) -> float:
        """Exact-match rate for a guesser that knows only the answer's neighborhood.

        On a coarse grid this is well above zero, so held-out accuracy has to clear
        it before it shows the model can name the mix rather than only locate it.
        """
        exs = evals[grid]["named_holdout"]
        return bl.neighborhood_exact_null(bl.shell_mask(nc.GRIDS[grid], raw_rgb(grid), [ex.result for ex in exs]))

    def blind_for(grid: str, evals: dict) -> int:
        """The name a prompt-blind model would always answer, fit on the training pairs."""
        return bl.blind_index(dist_matrix(grid, evals[grid]["named_seen"]))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.3: named colors only

    Every experiment so far has taught the model colors two ways: as names
    (`red`) and as hex codes (`#f00`), with alias lines tying the two together.
    The hex form spells a color one channel at a time, so the geometry of color
    space is legible in the text.

    In this experiment's language, each color is a single opaque token, and
    every sentence is a mixing equation between named colors whose mix is also
    named. Nothing about the tokens says that colors live on a 3D grid, that
    `purple` sits midway between `red` and `blue`, or that colors are values at
    all. The only structure left is the co-occurrence statistics of the mixing
    table.

    The main question is whether the model can infer the color-space geometry
    from that alone. A model that discovers it should be able to fill in cells
    of the table it has never seen.

    Many operand pairs mix to a color that has no name in the vocabulary, so an
    exact answer is impossible there. A model that holds the geometry should
    still guess a nearby name. Measuring how close the guesses land gives a
    graded score where exact-match accuracy would only record a miss.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The language and a vocabulary-size sweep

    A vocabulary here is a sub-grid of the 16-level RGB cube: pick a set of
    levels, then give a name to every point on that sub-grid. We sweep four such
    grids. The 3-level grid is the familiar 27-color palette from the base
    language, with proper names. The denser grids use synthetic names that encode
    the value (`c05f` is the color `#05f`), which helps us read through failures
    later on. Each name is still a single token, so the model sees an opaque id
    and has to learn what it means from usage.

    | grid | levels per channel | colors | example |
    |------|--------------------|-------:|---------|
    | `v27` | 0, 8, 15 | 27 | `red`, `purple` |
    | `v64` | 0, 5, 10, 15 | 64 | `c05f` |
    | `v216` | 0, 3, 6, 9, 12, 15 | 216 | `c9c3` |
    | `v4096` | all 16 | 4096 | `c27b` |

    Every training line reads `a + b = mix(a, b)`, six tokens including the
    newline, with both operands and the mix drawn from the vocabulary. Mixing is
    the same channel-wise round-half-up mean used in earlier experiments. A pair
    whose mix lands on the vocabulary is "closed"; {HOLDOUT_FRAC:.0%} of the
    distinct closed pairs are held out of training entirely. Pairs whose mix
    falls between grid cells are "open": they never appear in training, since
    there is no name to write on the right-hand side, and we use them only as
    probes of graded generalization. The corpus is {N_EXAMPLES:,} with lines
    drawn at random. The model architecture is the d64-L4 nGPT cell from
    ex-2.1.1, with {len(SEEDS)} seeds per grid.

    Here are a few lines from two of the corpora.
    """)
    return


@app.cell(hide_code=True)
def _():
    _v27 = "".join(ex.text for ex in nc.sample_corpus(5, 0, nc.GRIDS["v27"]))
    _v216 = "".join(ex.text for ex in nc.sample_corpus(5, 0, nc.GRIDS["v216"]))
    mo.hstack(
        [mo.md(f"`v27`:\n```\n{_v27}```"), mo.md(f"`v216`:\n```\n{_v216}```")],
        justify="start",
        gap=3,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A denser grid means more colors to name, which is a harder vocabulary, but
    it also means many more closed pairs constraining each color, which is
    richer evidence about how colors relate. At the far end, the full 16-level
    grid has 4096 colors and 8.4 million distinct closed pairs, of which a
    100k-line corpus can visit only about 1%. The smallest grid is tiny: 49
    distinct closed pairs at `v27`, so the table is easy to memorize, and the
    held-out fifth of it asks the model to infer answers from very little
    evidence.

    ## Measurements

    From the next-token distribution at the `=` position we compute exact-match
    accuracy, and the cross-entropy of the true answer on closed pairs
    (where a single answer exists).

    For all pairs (including open pairs), we also find the graded score as the
    RGB distance from the model's highest-probability color to the true mix.
    There are two reference distances per prompt to compare against: the *floor*
    (the distance from the true mix to the nearest vocabulary color, which is
    zero for closed pairs) and *chance* (the mean distance over the whole
    vocabulary, what a guesser with no knowledge would score on average).

    We also fit probes on the residual stream to inspect the shape of the
    latent embeddings.

    ## Hypotheses

    Written before looking at any results.

    **H1.** Seen pairs are answered almost perfectly on every grid; the corpus
    gives each seen pair many effective repetitions, even at `v4096`.

    **H2.** Held-out accuracy rises with grid size. `v27` stays near zero, since
    39 training pairs constrain 27 colors only weakly. At the other end,
    `v4096` can barely memorize even its seen pairs, so if it learns them at all
    it must be generalizing, and its held-out accuracy should come close to its
    seen accuracy.

    **H3.** Where held-out accuracy is high, guessed colors on open pairs land
    near the floor distance, well below chance; where it is low, guesses sit
    at or near chance.

    **H4.** The embedding-to-RGB probe's R² and the low-dimensionality of the
    embedding table (top-3 explained variance) rise with grid size, and a PCA of
    the embeddings shows the color cube where, and only where, held-out
    performance is good.

    **H5.** Less certain, so a watch item: at `v4096` the fixed 100-epoch budget
    may not be enough for the rule to settle. If that happens, seen accuracy
    will be middling too, and the run would be telling us "undertrained" rather
    than "impossible".
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
            "No results yet. Run the experiment, which publishes metrics, arrays, eval sets, "
            "and checkpoints when it finishes:\n\n"
            "```bash\nbin/mini run docs/m2/ex-2.1.3/experiment.py --app modal --max-containers 9\n```"
        ),
    )
    metrics, arrays, evals = loaded
    return arrays, evals, metrics


@app.cell(hide_code=True)
def _(evals, metrics):
    def _mean(g, es, key):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es][key] for s in SEEDS]))

    _hold = {g: _mean(g, "named_holdout", "accuracy") for g in GRID_NAMES}
    _null = {g: exact_null(g, evals) for g in GRID_NAMES}
    mo.md(f"""
        **Headline numbers.** Mean held-out accuracy by grid:
        {", ".join(f"`{g}` **{v:.2f}**" for g, v in _hold.items())}.
        On every grid but the smallest, that is far above what a model knowing
        only the answer's neighborhood would score
        ({", ".join(f"`{g}` {v:.2f}" for g, v in _null.items())}), so the model
        does infer the geometry (except `v27`, which is discussed below). Misses are
        interesting: at `v4096` they land a mean distance of just
        {_mean("v4096", "named_holdout", "guess_dist"):.3f} from the true mix
        (chance is {_mean("v4096", "named_holdout", "chance_dist"):.2f}). The
        sections below build that picture up piece by piece, starting with the
        pairs the corpus sampler actually produced.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _stats = metrics["corpus_stats"]
    _rows = "".join(
        f"<tr><td><code>{g}</code></td>"
        f'<td class="num">{s["n_colors"]:,}</td>'
        f'<td class="num">{s["n_closed_distinct"]:,}</td>'
        f'<td class="num">{s["n_seen_distinct"]:,}</td>'
        f'<td class="num">{s["n_holdout"]:,}</td>'
        f'<td class="num">{s["n_open"]:,}</td>'
        "</tr>"
        for g, s in _stats.items()
    )
    _table = (
        '<div class="report-table-scroll"><table class="report-table">'
        '<tr><th>grid</th><th class="num">colors</th><th class="num">closed pairs</th>'
        '<th class="num">distinct pairs in corpus</th><th class="num">held out</th>'
        '<th class="num">open pairs</th></tr>' + _rows + "</table></div>"
    )
    _caption = mo.md(
        """
        Pair accounting per grid. "Closed pairs" counts the distinct unordered pairs
        whose mix lands on the vocabulary (self-pairs excluded); the corpus samples,
        with repetition, from the side that isn't held out. At `v4096` the corpus
        reaches about 1% of the closed pairs, and every pair is closed, so there are no
        open pairs to probe.
        """
    ).text
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    All twelve cells train stably. Each grid settles within the 100-epoch
    budget, `v4096` included, whose curve is flat over its last twenty epochs. So
    the results that follow reflect what this budget and schedule produce at
    convergence.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="val-loss",
        alt_text="""
            Validation loss over training epochs, with one line per vocabulary grid
            (v27, v64, v216, v4096) and three thin overlapping lines per grid, one per
            seed. Larger vocabularies start at higher loss, because a larger vocabulary
            means higher chance cross-entropy.
        """,
        caption="""
            Validation loss per epoch, with three thin lines per grid, one per seed.
            The grids aren't comparable to each other, since chance loss grows with the
            vocabulary size; what matters here is that each curve settles.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        shades = grid_shades()
        for g in GRID_NAMES:
            for s in SEEDS:
                vl = cell_of(metrics, g, s)["val_loss"]
                ax.plot(vl, color=shades[g], lw=1.0, alpha=0.8, label=g if s == 0 else None)
        ax.set(xlabel="epoch", ylabel="validation loss")
        ax.legend()
        ax.grid(alpha=0.3)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exact-match accuracy

    Can the model answer equations it has never seen?
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _sets = ["named_seen", "named_holdout"]

    @themed(
        name="accuracy",
        alt_text="""
            Two bar panels of exact-match accuracy (0 to 1) by vocabulary grid (v27,
            v64, v216, v4096), one panel for seen pairs and one for held-out pairs. Bars
            show the mean over three seeds, and dots show the individual seeds. Seen
            accuracy is 1.0 everywhere except v4096 (0.85). Held-out accuracy is
            non-monotonic: 0.27, 0.59, essentially 1.0 at v216, then 0.65 at v4096.
        """,
        caption=mo.md("""
            Exact-match accuracy by grid. Bars show the mean over three seeds;
            dots are individual seeds. **Left:** Seen pairs show whether
            training worked at all; **Right:** held-out pairs show
            generalization.
        """).text,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_sets), figsize=(8.4, 3.2), sharey=True)
        shades = grid_shades()
        xs = np.arange(len(GRID_NAMES))
        for ax, es in zip(axes, _sets, strict=True):
            per_seed = np.array([[cell_of(metrics, g, s)["sets"][es]["accuracy"] for s in SEEDS] for g in GRID_NAMES])
            ax.bar(xs, per_seed.mean(axis=1), color=[shades[g] for g in GRID_NAMES], width=0.62)
            for i in range(len(GRID_NAMES)):
                ax.plot([xs[i]] * len(SEEDS), per_seed[i], "o", color=light_dark("#000", "#fff"), ms=3, zorder=3)
            ax.set(title=es.replace("_", " "), ylim=(-0.03, 1.03))
            ax.set_xticks(xs, GRID_NAMES)
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("exact-match accuracy")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, evals, geom, metrics):
    def _acc(g, es):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es]["accuracy"] for s in SEEDS]))

    def _z(g: str) -> float:
        """Held-out accuracy above the neighborhood null, in standard errors.

        Pooling seeds treats repeat runs on the same pairs as independent draws, so
        this flatters the small grids; `v27`'s ten pairs are the binding limit.
        """
        null, n = exact_null(g, evals), len(evals[g]["named_holdout"]) * len(SEEDS)
        return (_acc(g, "named_holdout") - null) / ((null * (1 - null) / n) ** 0.5)

    _null27 = exact_null("v27", evals)
    _shell27 = (
        bl.shell_mask(nc.GRIDS["v27"], raw_rgb("v27"), [ex.result for ex in evals["v27"]["named_holdout"]])
        .sum(axis=1)
        .mean()
    )
    _z27, _z64, _z216, _z4096 = (_z(g) for g in GRID_NAMES)
    _r2_27 = geom("v27", "r2")

    def _miss_structure(g: str) -> tuple[int, int, int]:
        """Pooled over seeds: (misses, one-channel misses, of those off by one level)."""
        exs = evals[g]["named_holdout"]
        rgb = np.array(list(PALETTES[g].values()))
        lvl = {v: i for i, v in enumerate(nc.GRIDS[g])}
        n_miss = n_1ch = n_1lvl = 0
        for s in SEEDS:
            guesses = rgb[arrays[f"{g}-s{s}/logp/named_holdout"].argmax(1)]
            for ex, gv in zip(exs, guesses, strict=True):
                diff = [int(a != b) for a, b in zip(gv, ex.result, strict=True)]
                if sum(diff) == 0:
                    continue
                n_miss += 1
                if sum(diff) == 1:
                    n_1ch += 1
                    (ch,) = [i for i, d in enumerate(diff) if d]
                    n_1lvl += abs(lvl[int(gv[ch])] - lvl[ex.result[ch]]) == 1
        return n_miss, n_1ch, n_1lvl

    _m64, _m216, _m4096 = (_miss_structure(g) for g in ("v64", "v216", "v4096"))
    mo.md(rf"""
    Every grid learns its seen pairs almost perfectly, apart from `v4096`, which
    reaches {_acc("v4096", "named_seen"):.2f}. That is the one grid where even
    the training pairs are hard to memorize, since its corpus visits 99k distinct
    pairs roughly once each. Held-out accuracy then follows a non-monotonic path:
    {_acc("v27", "named_holdout"):.2f} at `v27` (that's
    {round(_acc("v27", "named_holdout") * 10)} of its ten held-out pairs),
    {_acc("v64", "named_holdout"):.2f} at `v64`,
    {_acc("v216", "named_holdout"):.2f} at `v216`, near enough to solved, and
    back down to {_acc("v4096", "named_holdout"):.2f} at `v4096`.

    Only three of those four numbers mean much. On a 27-color grid the true mix
    has just {_shell27:.1f} one-step neighbors on average, so a model that has
    located the answer's neighborhood and picks within it scores
    {_null27:.2f} without being able to name anything. `v27`'s
    {_acc("v27", "named_holdout"):.2f} sits {_z27:.1f} std above
    that, over ten distinct pairs, which is not a difference worth reading. The
    denser grids clear the same bar easily: {_z64:.0f} std at `v64`,
    {_z216:.0f} at `v216` and {_z4096:.0f} at `v4096`.

    So `v27`'s score is not evidence on its own. It is tempting to read it as a
    shift from the base language, where `named_holdout` sat at zero through
    every intervention ex-2.1.2 tried, but two of ten pairs on a grid this
    coarse cannot carry that. The case that the geometry is inferable from names
    rests on the denser grids here, and on the embedding probes further down,
    which reach R² ≈ {_r2_27:.2f} even at `v27`.

    The `v4096` misses are not scattered, either. Pooled over seeds, {_m4096[1]}
    of {_m4096[0]} held-out misses differ from the true mix in a single RGB
    channel, and {_m4096[2]} of those are off by one grid level (`v64`:
    {_m64[2]}/{_m64[0]}; `v216`: {_m216[2]}/{_m216[0]}). We checked whether
    these misses are more likely on rounding cases, meaning channels where the
    operand sum is odd so the mean has to round, and they don't: about half of
    the one-channel misses are rounding cases, which matches chance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Closeness

    Let's switch to the graded view. For every prompt we take the model's
    highest-probability color and measure its RGB distance to the true mix, and
    plot the cumulative distribution. For each distance x, the curve shows the
    fraction of prompts that landed within x of the answer.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, evals, metrics):
    _panels = [("named_holdout", "held-out pairs"), ("open", "open pairs")]

    @themed(
        name="distance-ecdf",
        alt_text="""
            Two panels of cumulative distributions of the RGB distance from the model's
            guessed color to the true mix, pooled over three seeds, one panel for
            held-out pairs and one for open pairs. One line per vocabulary grid. A dashed
            line shows the nearest-name floor on open pairs. Triangles under the x-axis
            mark the prompt-blind constant baseline. Every grid's curve stays close to its
            floor and well to the left of the reference; v216 and v4096 rise to 1 within a
            fraction of the chance distance.
        """,
        caption="""
            How close do guesses land? Each line is the cumulative distribution of the
            distance from the guess to the true mix (in unit-cube units, pooled over
            seeds); higher and further to the left is better. On held-out pairs an exact
            answer exists, so the height at distance 0 is the exact-match accuracy. On
            open pairs no name is quite right, so the dashed line shows the best
            achievable (nearest-name) distance there. The triangles under the x-axis are
            the prompt-blind constant, the score for always answering the center of the
            training answers.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_panels), figsize=(8.4, 3.4), sharey=True)
        shades = grid_shades()
        for ax, (es, title) in zip(axes, _panels, strict=True):
            for g in GRID_NAMES:
                if es not in evals[g]:
                    continue
                exs = evals[g][es]
                gd = np.concatenate([dists_for(g, exs, arrays[f"{g}-s{s}/logp/{es}"])[0] for s in SEEDS])
                fl = dists_for(g, exs, arrays[f"{g}-s0/logp/{es}"])[1]
                xs = np.sort(gd)
                ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=shades[g], lw=1.6, label=g)
                if es == "open":
                    xf = np.sort(fl)
                    ax.plot(xf, np.arange(1, len(xf) + 1) / len(xf), color=shades[g], lw=1.0, ls="--", alpha=0.7)
                blind = bl.blind_stats(dist_matrix(g, exs), blind_for(g, evals))["dist"]
                ax.plot([blind], [-0.02], marker="^", ms=4, color=shades[g], clip_on=False)
            ax.set(title=title, xlabel="distance to true mix", xlim=(-0.02, 1.0), ylim=(-0.03, 1.03))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("fraction of prompts ≤ x")
        axes[0].legend(loc="lower right", fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, evals, metrics):
    def _mean(g, es, key):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es][key] for s in SEEDS]))

    _open = [g for g in GRID_NAMES if evals[g].get("open")]

    def _row(g):
        exs = evals[g]["open"]
        d = dist_matrix(g, exs)
        gd = np.mean([d[np.arange(len(exs)), arrays[f"{g}-s{s}/logp/open"].argmax(1)].mean() for s in SEEDS])
        two, blind = bl.k_nearest_stats(d, 2), bl.blind_stats(d, blind_for(g, evals))
        cells = [
            f"`{g}`",
            f"{gd:.3f}",
            f"{d.min(axis=1).mean():.3f}",
            f"{two['dist']:.3f}",
            f"{blind['dist']:.3f}",
            f"{d.mean():.3f}",
        ]
        return "<tr>" + "".join(f'<td class="num">{c}</td>' for c in cells) + "</tr>"

    _head = (
        '<tr><th>grid</th><th class="num">model</th><th class="num">floor</th>'
        '<th class="num">2 nearest</th><th class="num">prompt-blind</th>'
        '<th class="num">chance</th></tr>'
    )
    _table = figure_html(
        f'<div class="report-table-scroll"><table class="report-table">{_head}'
        + "".join(_row(g) for g in _open)
        + "</table></div>",
        caption="Mean distance to the true mix on open pairs, against four references.",
        class_="report-figure",
    )

    mo.md(rf"""
    Guesses are better than chance everywhere, and comfortably inside the
    prompt-blind constant, which is stricter: mixes are midpoints,
    so they cluster centrally, and always answering the middle name scores
    {bl.blind_stats(dist_matrix("v27", evals["v27"]["open"]), blind_for("v27", evals))["dist"]:.2f}
    on `v27` open pairs where chance is {_mean("v27", "open", "chance_dist"):.2f}.
    That much supports H3's picture.

    {_table}

    The floor is a harder reference than it looks, though, because it asks the
    model to break a tie it may have no way to break. An open mix falls between
    grid points, so two names bracket it, often at the same distance. A model
    that has located the answer but picks between those two at random is the
    "2 nearest" column, and every grid sits behind it: `v27` guesses
    {_mean("v27", "open", "guess_dist"):.3f} against that baseline's
    {bl.k_nearest_stats(dist_matrix("v27", evals["v27"]["open"]), 2)["dist"]:.3f}.
    On the nearest-name rate the gap is wider, {_mean("v27", "open", "nearest_acc"):.2f}
    against {bl.k_nearest_stats(dist_matrix("v27", evals["v27"]["open"]), 2)["nearest"]:.2f}.
    So the guesses are near the true
    mix, but not as near as a model with the same positional knowledge and a
    coin could get.

    H3 predicted near-floor guessing only where exact-match accuracy was high,
    but the graded signal does turn out to be present even where exact-match
    accuracy is low.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Embedding geometry

    The behavior suggests the model knows where the colors are in the cube.
    Since colors are single tokens, everything the model knows about a color's
    identity must live in that token's embedding. If the model inferred
    the hidden space, the embedding table should hold a color cube: some linear
    view of the 64-dimensional embeddings under which the tokens arrange
    themselves by their RGB values. A ridge probe from embeddings to RGB
    measures how true that is, and a PCA projection (principal component
    analysis, which finds the few directions along which the vectors vary most)
    gives an unsupervised look at the table's dominant structure.

    We'll look twice: first with PCA, which is unsupervised and so answers "what
    is this table mostly doing?", then with the probe's own weights, which
    answer the narrower question.
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    def _geometry(g: str, s: int) -> dict:
        """Where a grid's color tokens sit, read through a probe rather than through variance."""
        emb, rgb = arrays[f"{g}-s{s}/embeddings"], VOCAB_RGB[g]
        centered = emb - emb.mean(0)
        # Leave-one-out, so no token helps place itself and there is no split to draw:
        # a vocabulary of 27 is small enough that which fold a token lands in would
        # otherwise move the answer.
        pred = ridge_probe_loo(emb, rgb)
        ss_res = ((pred - rgb) ** 2).sum(0) / ((rgb - rgb.mean(0)) ** 2).sum(0)
        # How much of the table's variation the probe's 3-d read-out subspace holds, against
        # the most any three directions could. PCA can only find the cube when this is near
        # 1, whatever the probe says. Descriptive rather than predictive — it asks where the
        # full table's variance sits, so it uses the full fit.
        w, *_ = ridge_probe(emb, rgb, emb, rgb)
        q, _ = np.linalg.qr(w)
        top3 = (np.linalg.svd(centered, compute_uv=False)[:3] ** 2).sum()
        # R² is scale-free, which hides whether the leftover error is small *for this grid*.
        # Measuring it in grid cells asks that directly.
        err = np.linalg.norm(pred - rgb, axis=1).mean()
        cell = float(np.mean(np.diff(nc.GRIDS[g]))) / (N_LEVELS - 1)
        return {
            "r2": float((1 - ss_res).mean()),
            "pred": pred,
            "share": float(((centered @ q) ** 2).sum() / top3),
            "err_rgb": float(err),
            "err_cells": float(err / cell),
            # A mean under one cell does not imply a token decodes to its own name: a
            # Voronoi cell reaches only half a step, and the mean hides the tail. Ask
            # directly instead.
            "self_nearest": bl.self_nearest_rate(raw_rgb(g), pred),
        }

    geometry = {(g, s): _geometry(g, s) for g in GRID_NAMES for s in SEEDS}

    def geom(g: str, key: str) -> float:
        """One grid's geometry number, averaged over seeds."""
        return float(np.mean([geometry[(g, s)][key] for s in SEEDS]))

    return geom, geometry


@app.cell(hide_code=True)
def _(arrays, metrics):
    @themed(
        name="embedding-pca",
        alt_text="""
            Four scatter panels, one per vocabulary grid, showing the first two
            principal components of each grid's color-token embeddings, with every point
            drawn in the color it names. v27 and v64 show loose clusters with only rough
            color grouping; v216 shows a clear gradient organized by hue; v4096 shows a
            smooth color wheel, with hues arranged around a disc and darker colors toward
            the middle.
        """,
        caption="""
            The embedding table, projected onto its own first two principal components
            (seed 0), with each token drawn in the color it names. Only `v4096` shows the
            hue wheel we thought we might find. The percentage is how much of the table's
            variance its leading *three* components hold — the most any three directions
            could — of which the panel draws the leading two.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(GRID_NAMES), figsize=(11.5, 3.1))
        for ax, g in zip(axes, GRID_NAMES, strict=True):
            emb = arrays[f"{g}-s0/embeddings"]
            centered = emb - emb.mean(0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            z = centered @ vt[:2].T
            z /= np.abs(z).max() + 1e-9
            n = len(emb)
            ax.scatter(z[:, 0], z[:, 1], c=VOCAB_RGB[g], s=float(np.clip(6_000 / n, 4, 50)), lw=0)
            evr3 = np.mean([cell_of(metrics, g, s)["emb_evr3"] for s in SEEDS])
            ax.set_title(f"{g}   (top 3: {evr3:.0%})", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_axis_off()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(geom):
    mo.md(rf"""
    The PCA plots give the impression the cube arrives late — somewhere between
    `v216` and `v4096` — but it doesn't. PCA keeps the directions along which the
    table varies most, and apparently color is not what the table mostly does.
    These embeddings are also the tied language model head for predicting the
    next token, so perhaps two adjacent colors need well-separated directions
    for the softmax to keep them apart, even when their color values are nearly
    identical.

    A cube needs three directions. A couple of natural ways to pick them are:

    - PCA: Pick the three directions the table varies most along. The amount of
      the total variance captured by those directions is shown as the
      percentages on the panels above.
    - Probe: Fit to predict RGB and indifferent to how much the table varies
      along them.

    How well do they agree in this case? If representing color is what the table
    mostly does, the probe should be reading close to the top-variance
    directions and the two should capture similar amounts. They don't, at least
    not until the vocabulary gets large. The probe's directions hold
    {geom("v27", "share"):.0%} as much variance as the best three at `v27` and
    {geom("v64", "share"):.0%} at `v64`, then {geom("v216", "share"):.0%} at
    `v216` and {geom("v4096", "share"):.0%} at `v4096`. So it is only at the
    dense end that the color directions and the high-variance directions are
    nearly the same thing — and only there that PCA stumbles onto the cube.

    Let's see what the probe found. Its weights are a map from the
    64-dimensional embedding onto RGB, so we can use it to decode the color
    tokens. If the table holds a color cube, the decoded values should be
    similar to the source data. To keep this from being a statement about the
    probe's memory rather than the model's geometry, every token is placed by a
    probe fit on every *other* token in the vocabulary.
    """)
    return


@app.cell(hide_code=True)
def _(geom, geometry):
    @themed(
        name="embedding-probe-cube",
        alt_text="""
            Four hexagonal panels, one per vocabulary grid, showing where the probe places
            each color token in the RGB cube, viewed down the cube's grey diagonal so hue
            runs around the panel and the six chromatic corners sit on the rim. Filled dots
            are where a token landed, open rings its true position, joined by a short stub.
            At v27 many dots sit well away from their rings; v64 is closer; v216 has dots
            almost on top of their rings, forming an even hue wheel; v4096 is a dense wheel
            shown without rings.
        """,
        caption="""
            The same embedding tables (seed 0), now read through the probe instead of
            through variance: each token is placed at the RGB the probe predicts for it,
            and drawn in the color it actually names. Open rings mark where a token should
            sit, so the displacement (lines) show the error. `R²` is the same fit as a
            single number, averaged over seeds. Rings are left off at `v4096`, where 4096
            of them would bury the colors.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(GRID_NAMES), figsize=(11.5, 3.1))
        for ax, g in zip(axes, GRID_NAMES, strict=True):
            rgb = VOCAB_RGB[g]
            n = len(rgb)
            plot_rgb_cube(
                ax,
                geometry[(g, 0)]["pred"],
                rgb,
                view="wheel",
                truth=rgb if n <= 216 else None,
                s=float(np.clip(6_000 / n, 3, 24)),
            )
            ax.set_title(f"{g}   (R² {geom(g, 'r2'):.2f})", fontsize=10)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(geom, metrics):
    def _cell_mean(g: str, key: str) -> float:
        return float(np.mean([cell_of(metrics, g, s)[key] for s in SEEDS]))

    _evr = {g: _cell_mean(g, "emb_evr3") for g in GRID_NAMES}
    mo.md(rf"""
    So the cube geometry is in the embedding table, even in the smallest model.
    `v216` is the clearest case: held-out tokens land almost exactly on their
    own lattice sites, a hue wheel with greys in the middle, from a table whose
    principal components showed only a smear. `v64` is recognizably the same
    shape with visible slop, and even `v27` has most of its colors in the right
    neighborhood.

    Probe R² quantifies how well it fit: {geom("v27", "r2"):.2f},
    {geom("v64", "r2"):.2f}, {geom("v216", "r2"):.2f}, {geom("v4096", "r2"):.2f}
    across the four grids.[^folds] From `v64` up, RGB is close to a linear
    function of the embedding.

    Does a probe-decoded token land nearer its own name than any other?

    | grid | nearest name is its own |
    | --- | --- |
    | `v27` | {geom("v27", "self_nearest"):.0%} |
    | `v64` | {geom("v64", "self_nearest"):.0%} |
    | `v216` | {geom("v216", "self_nearest"):.0%} |
    | `v4096` | {geom("v4096", "self_nearest"):.0%} |

    At `v27` the answer is a coin flip. `v4096` makes the same point from the
    other end: its R² of {geom("v4096", "r2"):.2f} is respectable, but sixteen
    levels per channel pack the names so tightly that a decoded token is almost
    never nearest its own. So the probe finds the cube, but it does not decode
    individual tokens well enough to name them at any of these grid sizes.

    H4 expected the embedding table itself to turn low-dimensional, and the PCA
    says it doesn't: even the best three directions hold only {_evr["v27"]:.0%}
    of the variance at `v27`, and that falls to {_evr["v4096"]:.0%} at `v4096`.
    The cube is in there as a subspace the probe can read, but most of the
    embedding variance is doing something else.

    [^folds]: Probes fit per-color with leave-one-out, which matters most at the
    small grids. A single half/half split leaves `v27`'s probe fitting a 64→3
    map from 13 points, and scores it {_cell_mean("v27", "emb_r2"):.2f} against
    {geom("v27", "r2"):.2f} here; `v4096` is unmoved either way
    ({_cell_mean("v4096", "emb_r2"):.2f} against {geom("v4096", "r2"):.2f}).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mixing in value space

    The embeddings hold the geometry of individual colors; the last question is
    about the computation that combines them. At the pre-answer position (`=`),
    does the residual stream already contain the mix's value before the answer
    is emitted? We'll check with probes fit at each depth on seen prompts and
    then apply them to held-out and open prompts.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="residual-probes",
        alt_text="""
            Four line panels, one per vocabulary grid, of ridge-probe R-squared for the
            mix's RGB read from the residual stream at the pre-answer position, over
            residual depth 0 to 4. Each panel has three lines: seen pairs (the probe fit
            set), held-out pairs, and open pairs.
        """,
        caption="""
            Ridge probes from the pre-answer residual stream to the true mix's
            RGB, one per depth (0 = embeddings, 4 = final layer), fit on half
            the seen prompts (seed 0). Transfer to held-out and open prompts
            tells a genuine value-space computation apart from a probe that
            memorized its fit set.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(GRID_NAMES), figsize=(11.5, 3.0), sharey=True)
        styles = {"named_seen": "-", "named_holdout": "--", "open": ":"}
        shades = grid_shades()
        for ax, g in zip(axes, GRID_NAMES, strict=True):
            probe = cell_of(metrics, g, 0)["probe_r2"]
            for es, r2s in probe.items():
                ax.plot(r2s, styles.get(es, "-"), color=shades[g], label=es.replace("_", " "))
            ax.set(title=g, xlabel="depth", ylim=(-0.1, 1.03))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("probe R² (mix RGB)")
        axes[0].legend(fontsize=7, loc="upper left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _mid = {g: max(cell_of(metrics, g, 0)["probe_r2"]["named_holdout"]) for g in GRID_NAMES}
    mo.md(rf"""
    Is the answer computed in value space before it is emitted?
    Apparently so, and the probes transfer well. Peak held-out R² is
    {_mid["v27"]:.2f} at `v27`, {_mid["v64"]:.2f} at `v64`, {_mid["v216"]:.2f}
    at `v216`, and {_mid["v4096"]:.2f} at `v4096` (seed 0), with most of the
    value present from depth 1 or 2 onward. This quite different from the base
    language. There, ex-2.1.2 found that the answer's channels are computed just
    in time and cleared away after emission (for hex answers), so a full
    "result" concept wasn't readable at any single position. But now with
    one-token answers there is no schedule to spread the work over, so the whole
    mix must be present at the pre-answer position.

    Even `v27`'s held-out prompts probe at about 0.9 mid-stack while their
    exact-match accuracy is 0.2 to 0.3. The model computes roughly the right
    value, and then the readout picks a neighboring name.

    ## Example completions

    Here are a few concrete completions (we picked the widest misses).
    """)
    return


@app.cell(hide_code=True)
def _(arrays, evals):
    def completion_rows(grid: str, es: str, k: int = 4) -> str:
        exs = evals[grid][es]
        lp = arrays[f"{grid}-s0/logp/{es}"]
        names = list(PALETTES[grid])
        gd, fl = dists_for(grid, exs, lp)
        order = np.argsort(-gd)[:k]  # the widest misses are the most informative
        rows = []
        for i in order:
            ex = exs[i]
            a, b = ex.prompt.split()[0], ex.prompt.split()[2]
            true = (
                sw(ex.answer, grid)
                if ex.answer
                else f'<span class="sw" style="--sw: {to_hex(ex.result)};" aria-hidden="true"></span> <code>{to_hex(ex.result)}</code>'
            )
            rows.append(
                f"<tr><td>{sw(a, grid)}</td><td>{sw(b, grid)}</td><td>{true}</td>"
                f'<td>{sw(names[int(lp[i].argmax())], grid)}</td><td class="num">{gd[i]:.2f}</td>'
                f'<td class="num">{fl[i]:.2f}</td></tr>'
            )
        return "".join(rows)

    return (completion_rows,)


@app.cell(hide_code=True)
def _(completion_rows):
    _head = (
        "<tr><th>a</th><th>b</th><th>true mix</th><th>model's guess</th>"
        '<th class="num">distance</th><th class="num">floor</th></tr>'
    )
    _k = 5
    _tables = []
    for _g, _es in [("v27", "named_holdout"), ("v216", "open"), ("v4096", "named_holdout")]:
        _tables.append(
            figure_html(
                f'<div class="report-table-scroll"><table class="report-table">{_head}'
                f"{completion_rows(_g, _es, _k)}</table></div>",
                caption=mo.md(f"`{_g}` · {_es.replace('_', ' ')}: the {_k} widest misses (seed 0).").text,
                class_="report-figure",
            )
        )
    mo.Html("\n".join(_tables))
    return


@app.cell(hide_code=True)
def _(arrays, evals):
    _exs = evals["v27"]["named_holdout"]
    _rgb = raw_rgb("v27")
    _null = bl.operand_shell_null(
        bl.shell_mask(nc.GRIDS["v27"], _rgb, [ex.result for ex in _exs]),
        _rgb,
        [(ex.lhs, ex.rhs) for ex in _exs],
    )
    _n = _k = 0
    for _s in SEEDS:
        for _ex, _gv in zip(_exs, _rgb[arrays[f"v27-s{_s}/logp/named_holdout"].argmax(1)], strict=True):
            _n += 1
            _k += tuple(_gv) in (tuple(_ex.lhs), tuple(_ex.rhs))

    mo.md(rf"""
    `v27`'s misses are neighbor names (`violet` guessed as `blue`, `maroon` as
    `red`); `v216`'s open-pair guesses sit a step away from the floor; and
    `v4096`'s worst held-out misses still agree with the true mix on two of the
    three channels.

    Several of `v27`'s rows return one of the operands, may or may not be an echo.
    Returning an operand and returning a neighbor are close to the same thing here.
    {_k} of {_n} guesses are operands, against {_null:.0%} for a guess drawn
    uniformly from the neighbors.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings

    1. The model infers the color-space geometry better without hex scaffolding.
       From mixing co-occurrences alone it builds embeddings that are linear
       color cubes, computes mixes in value space at the pre-answer position,
       and generalizes to held-out pairs. When it guesses a held-out answer, the
       guess is very close.
    2. Performance varies with vocabulary size. Exact match is non-monotonic,
       while geometric closeness improves monotonically.
    """)
    return


if __name__ == "__main__":
    app.run()
