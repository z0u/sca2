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
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS, load_example_sets, to_hex

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.3: names all the way down

    Every experiment so far has taught the model colors two ways at once: as
    names (`red`) and as hex codes (`#f00`), with alias lines tying the two
    together. The hex form does the model a favor. It spells a color out one
    channel at a time, so the geometry of color space is legible right there in
    the text. And [ex-2.1.2](../ex-2.1.2/) found that the models lean on that
    hard: name-plus-name arithmetic works as long as the answer can be written
    in hex, and the one step that never gets trained is the translation from a
    value back to a name.

    That suggests a control worth running: take the hex away entirely. In this
    experiment's language, a color is a single opaque token, and every sentence
    is a mixing equation between named colors whose mix is also named. No
    aliases, no digits, no characters. Nothing in the token stream says that
    colors live on a 3D grid, that `purple` sits midway between `red` and
    `blue`, or that colors are values at all. The only structure left is the
    co-occurrence statistics of the mixing table: which colors show up together,
    and how often.

    So the question, from the [science todo](https://github.com/z0u/sca2/blob/main/todo-science.md),
    is whether the model can infer the color-space geometry from that alone. The
    task is a matrix-completion problem[^mc] dressed up as language modeling. The
    mixing table is a large symmetric grid whose entries follow one small hidden
    rule: each color is a point in a 3D cube, and mixing is the channel-wise
    mean. A model that discovers the rule can fill in cells of the table it has
    never seen; a model that memorizes the table cell by cell cannot. Held-out
    pairs tell the two apart.

    There is a second, softer question. Many operand pairs mix to a color that
    has no name in the vocabulary, so an exact answer is impossible there. A
    model that holds the geometry should still guess a nearby name. Measuring how
    close the guesses land gives a graded score where exact-match accuracy would
    only record a miss, and that graded score is the one that matters for the
    anchored experiments to come. An anchor needs the model's representation to
    be organized by geometry, whether or not the top guess comes out right.

    [^mc]: Matrix completion is the problem of filling in the missing entries of
        a partly observed table when the full table has some simple underlying
        structure. A streaming service predicting the ratings you haven't given
        from the ones you have is the same shape of problem.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The language and a vocabulary-size sweep

    A vocabulary here is a sub-grid of the 16-level RGB cube:
    pick a set of levels, then give a name to every point of levels³. We sweep
    four such grids. The 3-level grid is the familiar 27-color palette from the
    base language, names and all. The denser grids use synthetic names that
    encode the value (`c05f` is the color `#05f`), which helps us read through
    failures later on. Each name is still a single token, so the model sees an
    opaque id and has to learn what it means from usage, the same way it would
    for `red`.

    | grid | levels per channel | colors | example |
    |------|--------------------|-------:|---------|
    | `v27` | 0, 8, 15 | 27 | `red`, `purple` |
    | `v64` | 0, 5, 10, 15 | 64 | `c05f` |
    | `v216` | 0, 3, 6, 9, 12, 15 | 216 | `c9c3` |
    | `v4096` | all 16 | 4096 | `c27b` |

    Every training line reads `a + b = mix(a, b)`, six tokens including the
    newline, with both operands and the mix drawn from the vocabulary. Mixing is
    the same channel-wise round-half-up mean we have used throughout. A pair
    whose mix lands on the vocabulary is *closed*; {HOLDOUT_FRAC:.0%} of the
    distinct closed pairs are held out of training entirely. Pairs whose mix
    falls off the vocabulary are *open*: they never appear in training, since
    there is no name to write on the right-hand side, and we use them only as
    probes of graded generalization. The corpus is {N_EXAMPLES:,} lines drawn
    independently at random from the train-side closed pairs. The backbone is the
    frozen d64-L4 nGPT cell from ex-2.1.1, with {len(SEEDS)} seeds per grid.

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
    The sweep moves two things at once, on purpose. A denser grid means more
    colors to name, which is a harder vocabulary, but it also means many more
    closed pairs constraining each color, which is richer evidence about how
    colors relate. At the far end, the full 16-level grid has 4096 colors and
    8.4 million distinct closed pairs, of which a 100k-line corpus can visit only
    about 1%. There, memorizing the table stops being an option, and the model
    either finds the rule or fails. The small grids sit at the opposite extreme:
    49 distinct closed pairs at `v27`, so the table is easy to memorize, and the
    held-out fifth of it asks the model to infer answers from very little
    evidence.

    ## What we measure

    Answers are single tokens, so one forward pass per prompt gives us the whole
    picture. From the next-token distribution at the `=` position we read off
    several numbers. There is exact-match accuracy, and the cross-entropy[^xent]
    of the true answer on closed pairs, where a true answer exists. There is the
    RGB distance from the model's highest-probability color to the true mix, on
    any pair, which is the graded score. And there are two reference distances
    per prompt to compare against: the *floor* (the distance from the true mix to
    the nearest vocabulary color, which is zero for closed pairs) and *chance*
    (the mean distance over the whole vocabulary, what a guesser with no
    knowledge would score on average).

    Then two geometry probes. A ridge probe[^ridge] from the color-token
    embeddings to RGB asks whether the embedding table itself turned into a color
    cube. The second probe works on the residual stream (the running vector each
    token carries through the layers) at the pre-answer position: per-layer
    probes to the mix's RGB ask whether the answer gets computed in value space
    on the way out, the way it does in the hex-trained models.

    [^xent]: Cross-entropy here measures how much probability the model placed on
        the correct token. Lower is better, and zero would mean full confidence
        in the right answer.

    [^ridge]: A ridge probe is a linear regression with a mild penalty on large
        weights, fit here to predict RGB from an embedding vector. When it fits
        well, the target is close to a linear readout of the input. We report the
        fit as R², the fraction of variance explained, where 1.0 is perfect.

    ## Hypotheses

    Written down before looking at any results.

    **H1.** Seen pairs are answered almost perfectly on every grid; the corpus
    gives each seen pair many effective repetitions, even at `v4096`.

    **H2.** Held-out accuracy rises with grid size. `v27` stays near zero, since
    39 training pairs constrain 27 colors only weakly, echoing the base
    language's unsolved `named_holdout` (there the sticking point was the missing
    translation step; here it would be a shortage of evidence). At the other end,
    `v4096` can barely memorize even its seen pairs, so if it learns them at all
    it must be generalizing, and its held-out accuracy should come close to its
    seen accuracy. The interesting territory is the middle.

    **H3.** Where held-out accuracy is high, guessed colors on open pairs land
    near the floor distance, well below chance; where it is low, guesses sit
    at or near chance.

    **H4.** Embedding geometry tracks behavior. The embedding-to-RGB probe's R²
    and the low-dimensionality of the embedding table (top-3 explained variance)
    rise with grid size, and a PCA of the embeddings shows the color cube where,
    and only where, held-out performance is good.

    **H5.** Less certain, so a watch item: at `v4096` the fixed 100-epoch budget
    may not be enough for the rule to settle. Late, sudden generalization
    (grokking, as it is known for small algorithmic tasks) is a documented
    failure mode. If that happens, seen accuracy will be middling too, and the
    run would be telling us "undertrained" rather than "impossible".
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
def _(metrics):
    def _mean(g, es, key):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es][key] for s in SEEDS]))

    _hold = {g: _mean(g, "named_holdout", "accuracy") for g in GRID_NAMES}
    mo.md(
        "**Headline numbers.** Mean held-out accuracy by grid: "
        + ", ".join(f"`{g}` **{v:.2f}**" for g, v in _hold.items())
        + f". So yes, the model infers the geometry. The telling part is the way "
        f"it misses where it does miss: at `v4096` the misses land a mean distance "
        f"of just {_mean('v4096', 'named_holdout', 'guess_dist'):.3f} from the true mix "
        f"(chance is {_mean('v4096', 'named_holdout', 'chance_dist'):.2f}). "
        "The sections below build that picture up piece by piece, starting with the "
        "pair accounting the corpus sampler actually produced."
    )
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

    All twelve cells train stably, and the loss curves below are unremarkable,
    which is worth a sentence in itself. Each grid settles within the 100-epoch
    budget, `v4096` included, whose curve is flat over its last twenty epochs. So
    the results that follow reflect what this budget and schedule produce at
    convergence, rather than a run caught partway through its descent. (Whether a
    longer or differently shaped schedule would keep improving `v4096` is a
    follow-up question this run can't settle; late, sudden generalization on
    small algorithmic tasks is well documented.)
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

    The first question has a yes-or-no answer: can the model answer equations it
    has never seen? Seen pairs are the sanity check, and held-out pairs are the
    real test.
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
        caption="""
            Exact-match accuracy by grid. The bar is the mean over three seeds, and the
            dots are the individual seeds. Seen pairs show whether training worked at
            all; held-out pairs show generalization, since the only way to answer them
            is to infer the hidden geometry, their equations never having appeared in
            training.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_sets), figsize=(8.4, 3.2), sharey=True)
        shades = grid_shades()
        xs = np.arange(len(GRID_NAMES))
        for ax, es in zip(axes, _sets, strict=True):
            per_seed = np.array([[cell_of(metrics, g, s)["sets"][es]["accuracy"] for s in SEEDS] for g in GRID_NAMES])
            ax.bar(xs, per_seed.mean(axis=1), color=[shades[g] for g in GRID_NAMES], width=0.62)
            for i in range(len(GRID_NAMES)):
                ax.plot([xs[i]] * len(SEEDS), per_seed[i], "o", color="#0008", ms=3, zorder=3)
            ax.set(title=es.replace("_", " "), ylim=(-0.03, 1.03))
            ax.set_xticks(xs, GRID_NAMES)
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("exact-match accuracy")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, evals, metrics):
    def _acc(g, es):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es]["accuracy"] for s in SEEDS]))

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

    Two of these numbers are worth a closer look before we get to the graded
    metrics.

    The `v27` result is already a shift from the base language. There,
    `named_holdout` sat at zero through every intervention ex-2.1.2 tried. Here,
    with the same 27 colors and the same ten held-out pairs, but one-token names
    and no hex anywhere, the model gets a few of them right. So that earlier zero
    wasn't evidence that geometry can't be inferred from names. Take the
    translation step away, and a weak form of the inference shows up right away.

    The `v4096` misses are not scattered, either. Pooled over seeds, {_m4096[1]}
    of {_m4096[0]} held-out misses differ from the true mix in a single RGB
    channel, and {_m4096[2]} of those are off by one grid level (`v64`:
    {_m64[2]}/{_m64[0]}; `v216`: {_m216[2]}/{_m216[0]}). The model runs the right
    computation and lands in the house next door. We checked whether these misses
    pile up on rounding cases, meaning channels where the operand sum is odd so
    the mean has to round, and they don't: about half of the one-channel misses
    are rounding cases, which matches the ~50% base rate.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How close the guesses land

    Exact match is a harsh score for a task that is really about geometry, so
    let's switch to the graded view. For every prompt we take the model's
    highest-probability color and measure its RGB distance to the true mix.
    Plotting the cumulative distribution of that distance puts a grid's whole
    behavior on one curve: for each distance x, the curve shows the fraction of
    prompts that landed within x of the answer. The two reference distances say
    what to compare against: the *floor*, which is the nearest name that exists
    and so the best any answer could do, and *chance*, the average distance over
    the vocabulary.
    """)
    return


@app.cell(hide_code=True)
def _(evals, arrays, metrics):
    _panels = [("named_holdout", "held-out pairs"), ("open", "open pairs")]

    @themed(
        name="distance-ecdf",
        alt_text="""
            Two panels of cumulative distributions of the RGB distance from the model's
            guessed color to the true mix, pooled over three seeds, one panel for
            held-out pairs and one for open pairs. One line per vocabulary grid. A dashed
            line shows the nearest-name floor on open pairs. Vertical ticks on the x-axis
            mark each grid's chance distance. Every grid's curve stays close to its floor
            and well to the left of chance; v216 and v4096 rise to 1 within a fraction of
            the chance distance.
        """,
        caption="""
            How close do guesses land? Each line is the cumulative distribution of the
            distance from the guess to the true mix (in unit-cube units, pooled over
            seeds); higher and further to the left is better. On held-out pairs an exact
            answer exists, so the height at distance 0 is the exact-match accuracy. On
            open pairs no name is quite right, so the dashed line shows the best
            achievable (nearest-name) distance there. The tick under each axis marks a
            grid's chance distance, where guesses would land if the model knew nothing
            about the operands.
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
                chance = np.mean([cell_of(metrics, g, s)["sets"][es]["chance_dist"] for s in SEEDS])
                ax.plot([chance], [-0.02], marker="|", ms=8, color=shades[g], clip_on=False)
            ax.set(title=title, xlabel="distance to true mix", xlim=(-0.02, 1.0), ylim=(-0.03, 1.03))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("fraction of prompts ≤ x")
        axes[0].legend(loc="lower right", fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    def _mean(g, es, key):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es][key] for s in SEEDS]))

    mo.md(rf"""
    Here the sweep's story gets simple. Guesses land near the floor everywhere.
    Even `v27`, whose exact-match score looks poor, guesses within
    {_mean("v27", "open", "guess_dist"):.2f} of the true mix on open pairs,
    against a floor of {_mean("v27", "open", "floor_dist"):.2f} and chance of
    {_mean("v27", "open", "chance_dist"):.2f}. Its guesses come close to the best
    any name could manage, for pair types it never saw a single example of.
    `v216` tracks its floor very closely, and `v4096`'s held-out curve reaches
    nearly 1 within a couple of grid levels of distance.

    H3 predicted near-floor guessing only where exact-match accuracy was high,
    but the geometry turns out to be present even where exact-match accuracy is
    low. Exact match and geometric knowledge behave like nearly independent axes
    in this task. `v27` is coarse-grained, with few names and big gaps, so any
    leftover imprecision costs the top-1 answer; `v4096` asks for neighbor-level
    precision; and `v216` happens to match the model's achievable precision to
    its grid spacing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The geometry the model built

    The behavior suggests the model acts as if it knows where every color sits in
    the cube. We can also look at that knowledge itself. Colors are single
    tokens, so everything the model knows about a color's identity has to live in
    that token's embedding, the vector the model looks up when it reads the
    token. If the model really did infer the hidden space, the embedding table
    should hold a color cube: some linear view of the 64-dimensional embeddings
    under which the tokens arrange themselves by their RGB values. A ridge probe
    from embeddings to RGB measures how well that holds, and a PCA projection
    (principal component analysis, which finds the few directions along which the
    vectors vary most and lets us plot them in 2D) gives an unsupervised look at
    the table's dominant structure.
    """)
    return


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
            (seed 0), with each token drawn in the color it names. No axis means
            anything on its own; what matters is whether nearby points carry nearby
            colors. By `v4096` the leading components form a recognizable hue wheel. The
            number on each panel is the ridge-probe R² from the full 64-d embedding to
            RGB (mean over seeds), the quantitative version of the question "is this a
            color cube?".
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
            r2 = np.mean([cell_of(metrics, g, s)["emb_r2"] for s in SEEDS])
            ax.set_title(f"{g}   (R² {r2:.2f})", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_axis_off()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _r2 = {g: float(np.mean([cell_of(metrics, g, s)["emb_r2"] for s in SEEDS])) for g in GRID_NAMES}
    _evr = {g: float(np.mean([cell_of(metrics, g, s)["emb_evr3"] for s in SEEDS])) for g in GRID_NAMES}
    mo.md(rf"""
    The probe's held-out R² rises with grid size ({_r2["v27"]:.2f},
    {_r2["v64"]:.2f}, {_r2["v216"]:.2f}, {_r2["v4096"]:.2f}), so from `v64` up,
    RGB is close to a linear function of the embedding. The `v27` number is fit
    on only 13 points in 64 dimensions, so I read it as present but hard to
    certify, which fits with its near-floor guessing.

    H4 needs one refinement, though. The hypothesis expected the embedding table
    itself to turn low-dimensional, and it doesn't. The top three principal
    components carry only {_evr["v27"]:.0%} of the variance at `v27`, falling to
    {_evr["v4096"]:.0%} at `v4096`, which is why the PCA panels above look
    organized without looking like a flat cube. (At `v4096` the leading
    components pick out hue, laid out as a wheel, and the rest of the value
    information sits in later components.) The cube is in there as a subspace the
    probe reads off cleanly, yet most of the embedding variance is doing
    something else.

    That something else is probably needed. These embeddings are also the tied
    language-model head, the same vectors the model uses to score each possible
    next token, so two adjacent colors need well-separated directions for the
    softmax to keep them apart, even when their *values* are nearly identical.
    Value geometry and token identity share the same vectors. This is the
    superposition question (more features packed into a space than it has
    dimensions to spare) that D2.1's anchored runs will have to live with, here
    in miniature.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Computing the mix in value space

    The embeddings hold the geometry of individual colors; the last question is
    about the computation that combines them. At the pre-answer position, the
    token `=`, does the residual stream already contain the mix's value before
    the answer is emitted? We fit ridge probes at each depth on seen prompts and
    then apply them to held-out and open prompts. That transfer is what separates
    a real value-space computation from a probe that has simply memorized its fit
    set.
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
            Is the answer computed in value space before it is emitted? Ridge probes
            from the pre-answer residual stream to the true mix's RGB, one per depth
            (0 = embeddings, 4 = final layer), fit on half the seen prompts (seed 0).
            Transfer to held-out and open prompts tells a genuine value-space
            computation apart from a probe that memorized its fit set.
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
    It is, and the probes transfer well. Peak held-out R² is {_mid["v27"]:.2f} at
    `v27`, {_mid["v64"]:.2f} at `v64`, {_mid["v216"]:.2f} at `v216`, and
    {_mid["v4096"]:.2f} at `v4096` (seed 0), with most of the value present from
    depth 1 or 2 onward. This is a change from the base language. There, ex-2.1.2
    found that the answer's channels are computed just in time and cleared away
    after emission, so a full "result" concept never sat at any single position
    all at once. With one-token answers there is no schedule to spread the work
    over, so the whole mix has to be present at the pre-answer position, and it
    is. For anchoring, that gives a word-level result concept a natural home: one
    position, one direction per channel, high R², at every depth past the first
    block.

    Even `v27`'s held-out prompts probe at about 0.9 mid-stack while their
    exact-match accuracy is 0.2 to 0.3. The model computes roughly the right
    value, and then the readout picks a neighboring name. It is the same
    computed-but-misread gap we saw in ex-2.1.2, in a much milder form, since
    there the readout was a whole untrained translation.

    ## Example completions

    Here are a few concrete completions, chosen as the widest misses, so these
    are the hardest cases rather than typical ones.
    """)
    return


@app.cell(hide_code=True)
def _(evals, arrays):
    def completion_rows(grid: str, es: str, k: int = 6) -> str:
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
    _tables = []
    for _g, _es in [("v27", "named_holdout"), ("v216", "open"), ("v4096", "named_holdout")]:
        _tables.append(
            mo.Html(
                figure_html(
                    f'<div class="report-table-scroll"><table class="report-table">{_head}'
                    f"{completion_rows(_g, _es)}</table></div>",
                    caption=f"`{_g}` · {_es.replace('_', ' ')}: the six widest misses (seed 0).",
                    class_="report-figure",
                )
            )
        )
    mo.vstack(_tables)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The tables show in individual rows what the metrics showed in aggregate.
    `v27`'s misses are neighbor names (`violet` guessed as `purple`, `maroon` as
    `olive`); `v216`'s open-pair guesses sit a step away from the floor; and
    `v4096`'s worst held-out misses still agree with the true mix on two of the
    three channels.

    ## Discussion

    The todo item asked two questions. First, can the model infer the
    color-space geometry with no hex scaffolding? Yes. From mixing co-occurrences
    alone it builds embeddings that are linear color cubes, computes mixes in
    value space at the pre-answer position, and generalizes to held-out pairs.
    Second, when it guesses a held-out answer, is the guess close? Very. Guesses
    sit near the nearest-name floor at every vocabulary size, including the open
    pairs, whose pair type never occurs in training at all.

    The way performance varies with vocabulary size is more interesting than any
    single trend. Exact match is non-monotonic, running 0.27, 0.59, 1.00, 0.65
    across 27 to 4096 colors, while geometric closeness improves monotonically. I read
    the exact-match dip at `v4096` as a limit on precision rather than on
    knowledge: its misses are one grid level off in one channel, its seen-pair
    accuracy shows the same gap, and its training loss had already leveled off.
    Whether more training, or a different schedule, would buy back that last level
    of precision is an open question. This run can't tell "converged short of the
    grid spacing" apart from "would eventually snap into place".

    Three things carry forward to the anchored experiments.

    - Vocabulary design. The base language's `named_holdout` = 0 came from its
      grammar, the untrained value-to-name translation, rather than from
      name-only supervision. One token per color makes the whole name-arithmetic
      pipeline learnable, and a grid around `v216` density is a sweet spot: the
      task is solved, the geometry is clean, and open pairs are still available
      as graded probes. This also lowers the risk on the word-level tokenizer
      ablation already queued in the todo list.
    - A result concept with a fixed home. The base language computed its answer
      channels just in time and cleared them afterward; the single-token answer
      instead forces the full mix to exist at the pre-answer position, where it
      is linearly decodable (R² ≈ 0.9) from depth 1 to 2 onward. That is a
      friendlier target to anchor than a value that never fully exists in one
      place.
    - Geometry and identity share the embedding. The color cube is a decodable
      subspace, yet most embedding variance is doing something else, probably the
      separation the tied softmax head needs to keep neighboring colors apart. An
      anchor placed on the value subspace would leave the identity component
      unconstrained. Whether that turns out to help or to cause trouble is the
      capacity and superposition question queued for the eval step of the
      anchored runs.

    A few caveats. The sweep moves vocabulary size, pair coverage, and closure
    fraction together, so "vocab size" here stands for the whole regime rather
    than one isolated variable. The `v27` holdout has only ten pairs, so its
    accuracy comes in steps of a tenth. And everything rests on one backbone
    (d64-L4) at one token budget.
    """)
    return


if __name__ == "__main__":
    app.run()
