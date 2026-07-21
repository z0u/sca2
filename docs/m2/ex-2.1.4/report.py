import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.4: spelling the names",
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
    from matplotlib.legend_handler import HandlerTuple

    # Marimo puts the notebook's directory on sys.path, so the experiment
    # definition is importable — refs and sweep constants can't drift.
    from experiment import (
        ARRAYS_REF,
        EVALS_REF,
        GRID_NAMES,
        METRICS_REF,
        N_EXAMPLES,
        NAME_SEED,
        SEEDS,
    )
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import figure_html, light_dark, smooth_step, themed
    from sca.data import char_names as ch
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS, load_example_sets, to_hex

    use_publisher(report_bundle(__file__))

    PALETTES = {g: ch.opaque_names(nc.GRIDS[g], NAME_SEED) for g in GRID_NAMES}
    VOCAB_RGB = {g: np.array(list(p.values()), dtype=np.float32) / (N_LEVELS - 1) for g, p in PALETTES.items()}

    # The word-level sibling's published metrics, for rung-to-rung comparisons.
    WORD_METRICS_REF = "reports/m2/ex-2.1.3/metrics"
    # The base language's schedule probes (hex answers), for the eviction contrast.
    HEX_MARGINS_REF = "reports/m2/ex-2.1.2/margins"

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

    def load_sibling(ref: str, kind: str = "json"):
        """A sibling experiment's published artifact, or None if unavailable."""
        store = project_store()
        art = store.get_ref(ref)
        if art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            path = store.get(art, Path(d) / f"sibling.{kind}")
            if kind == "json":
                return json.loads(path.read_text())
            with np.load(path) as z:
                return {k: z[k] for k in z.files}

    def cell_of(metrics: dict, grid: str, s: int) -> dict:
        (r,) = [r for r in metrics["cells"] if r["label"] == f"{grid}-s{s}"]
        return r

    def grid_shades() -> dict[str, tuple]:
        # The same viridis stops ex-2.1.3 used for these two grids.
        stops = light_dark([0.82, 0.32], [0.88, 0.42])
        return dict(zip(GRID_NAMES, plt.cm.viridis(stops), strict=True))

    def sw(name: str, grid: str) -> str:
        """Inline swatch + name for a grid's opaque names."""
        rgb = PALETTES[grid].get(name)
        if rgb is None:
            return f"<code>{name}</code>"
        return f'<span class="sw" style="--sw: {to_hex(rgb)};" aria-hidden="true"></span> <code>{name}</code>'

    def dists_for(grid: str, exs: list) -> np.ndarray:
        """(N, V) distance from every vocabulary color to each example's true mix."""
        result = np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)
        return np.linalg.norm(VOCAB_RGB[grid][None] - result[:, None], axis=2)

    def renorm(lp: np.ndarray) -> np.ndarray:
        """Candidate log-probabilities → a distribution over the vocabulary names."""
        p = np.exp(lp - lp.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.4: spelling the names

    [Ex-2.1.3](../ex-2.1.3/) showed that a small transformer can work out the
    geometry of a color space from mixing equations alone. Give it lines like
    `red + blue = purple`, with made-up color names, and it arranges the colors
    in a latent cube that matches the real one. That setup made the task easy in
    one particular way: every color was a single token, so a name was just a row
    of the embedding table. Reading an operand took one table lookup, and writing
    the answer took one softmax. The two things that make real language models
    worth studying, assembling a concept out of several tokens and keeping hold
    of it while writing several more, were left out.

    This experiment adds those two things back and changes nothing else. The
    language is the same as ex-2.1.3's: same grids, same operand pairs, same
    train/holdout split, same number of equations. The one difference is what the
    tokenizer sees. Every color is now a four-letter name that the model reads
    and writes one character at a time:

    ```
    tkzk + qwfd = hjnp
    ```

    It helps to picture the three languages as rungs on a ladder. Word-level
    names (ex-2.1.3) are the low rung: one concept, one token. The base
    char + hex language is the top rung, the one the M2 milestone is really
    about. This experiment is the middle rung: concepts span several tokens, but
    there is still no hex scaffolding, no alias dictionary, and no spelling rule
    to lean on. If word-level and char-level behavior come apart, that difference
    tells us something about what tokenization does to concept formation, which
    is what we want to understand before we decide where anchors should go.

    One design choice is worth explaining first. Ex-2.1.3's synthetic names
    (`c05f` for the color `#05f`) spell the value out character by character.
    That did no harm when a name was a single token, but at char level it would
    amount to hex with a prefix, which hands back the per-channel scaffolding this
    rung is meant to remove. The classic 27 palette names (`red`, `chartreuse`)
    would not work either, because their lengths vary, and that would confound
    the answer-emission analysis. So every color gets an opaque random name: four
    letters, drawn without replacement, assigned with no relation to the color's
    value. No character carries channel information, and the link from spelling to
    value has to be learned as a whole.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The language at char level

    We keep two grids from ex-2.1.3's sweep, picked to bracket the range that
    matters. `v27` is the 3-level grid: 27 colors, with barely enough closed
    pairs to learn from. `v216` is the 6-level grid, 216 colors, where the
    word-level model had essentially solved the task. Each grid trains three
    seeds on the frozen d64-L4 backbone. Every training line is
    `a + b = mix(a, b)` between vocabulary colors whose mix is also in the
    vocabulary; the same fifth of the distinct closed pairs is held out, and the
    same rendering seed fixes each equation's operand order. The corpora are
    ex-2.1.3's, respelled line for line.

    Three parts of the model's job change. A line is now 19 characters rather
    than 6 tokens, so the {N_EXAMPLES:,}-equation corpus holds about 3× the
    tokens; that is a confound we accept and return to later. An operand's value
    is no longer a single lookup: the model has to recognize `tkzk` as one unit
    spread over four positions and attach a value to it. And the answer is no
    longer one softmax: the model has to start writing the mix's name before any
    of it is present in the context, then continue, letter by letter, without
    losing track.

    A few lines from each corpus:
    """)
    return


@app.cell(hide_code=True)
def _():
    def _sample(grid: str) -> str:
        names = {v: k for k, v in PALETTES[grid].items()}
        return "".join(ch.rename(ex, names).text for ex in nc.sample_corpus(5, 0, nc.GRIDS[grid]))

    mo.hstack(
        [mo.md(f"`{g}`:\n```\n{_sample(g)}```") for g in GRID_NAMES],
        justify="start",
        gap=3,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What we measure

    Exact match now comes in two forms, and the gap between them tells us
    something. Greedy accuracy lets the model write freely: we decode five
    characters, taking the most likely one at each step, and check them against
    the true name. A greedy miss can mean two things, that the model picked the
    wrong color, or that it misspelled and produced a string that is no color's
    name at all, so we also record how often the second thing happens
    (`p_malformed`). Candidate accuracy sets the spelling question aside. We
    teacher-force every vocabulary name in turn, plus its closing newline, sum
    the character log-probabilities, and take the highest-scoring name.
    (Teacher-forcing means feeding the model a fixed continuation and reading off
    the probability it assigned, rather than letting it generate.) When greedy
    and candidate accuracy differ, spelling is part of the trouble; when they
    agree, the misses are about which color the model settled on.

    Candidate scoring earns its keep a second way: it recovers the model's full
    probability distribution over well-formed answers, which the word-level
    experiment read straight off its softmax. That distribution feeds a new
    measurement. Ex-2.1.3 scored answers against the one-hot truth, the negative
    log-likelihood of the true name, which asks how much probability sits on the
    right answer without asking whether the rest of it sits in sensible places.
    So here we also build distance-shaped target distributions, a softmax of the
    negative RGB distance from each name to the true mix at a temperature τ, and
    measure how far the model's answer distribution is from them as τ changes. A
    model that has learned the geometry should spread its uncertainty over colors
    near the true mix. This measure has the further benefit of working on open
    pairs, where no name is exactly right and one-hot scoring has nothing to
    score against.

    Distances follow ex-2.1.3: the RGB distance from the model's chosen name (the
    candidate argmax) to the true mix, set against the nearest-name floor and the
    vocabulary-mean chance, computed per prompt.

    Then the probes. Per-depth ridge probes ask whether the operand and result
    RGB values can be read off the residual stream with a linear fit; we fit them
    on in-training lines, and their transfer to held-out and open prompts is the
    check that the probe learned the representation rather than memorizing the fit
    set. The answer-schedule probe from ex-2.1.2 also returns: teacher-force whole
    equations and probe for the mix's three channels at every position around the
    answer, at each depth. In the base language that probe found just-in-time
    computation with eviction, where hex digit k was readable at its own emission
    position and gone soon after. That schedule works because hex factorizes:
    digit k is channel k, so once a channel is written it is finished. An opaque
    name does not factorize, which turns the same probe into a pointed question
    about how a whole-name answer is held.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Written down before we looked at any results.

    **H1.** Training still works: seen-pair accuracy is near-perfect on both
    grids, and malformed completions are rare. Well-formedness is a local statistical pattern
    (after `= `, four letters and a newline) that a char model of this size picks
    up easily.

    **H2.** Held-out accuracy lands below the word-level rung on `v216`, where
    ex-2.1.3 scored ≈ 1.0, but well above zero. Multi-token names add ways to
    fail at binding and emission without removing the co-occurrence evidence the
    inference runs on. `v27` stays low, as it was at word level. A `v216` collapse
    to ≈ 0 would mean that multi-token naming on its own blocks the inference;
    that would be a negative result, and a useful one, since the base language's
    names are multi-token too.

    **H3.** Where the task is learned, guesses stay close in color space:
    near-floor distances on open pairs, and misses that are neighbors rather than
    random names. Geometric closeness held at every vocabulary size at word level,
    including sizes whose exact match was poor, so we expect it to survive the
    tokenizer change as well.

    **H4.** The mix has a fixed home even though the answer spans several tokens.
    No character of an opaque name stands for a channel, so the model cannot
    compute one channel, write it, and drop it; whatever it knows about the mix
    has to stay put while the name is being spelled. Concretely, all three RGB
    channels stay decodable at late depth across the whole emission window, unlike
    ex-2.1.2's per-channel stair-step. We also expect the mix to be largely
    decodable at the pre-answer position, as at word level, though it may come
    together later in depth, since identifying the operands now takes layers of
    work that embeddings used to handle.

    **H5.** The answer distribution is shaped like the color geometry. The KL
    divergence from the distance-softmax target to the candidate distribution,
    minimized over τ, sits far below the same divergence for a value-blind
    reference (uniform over names), and the best-fit τ is on the order of the grid
    spacing. Less confidently, `v216`'s distribution should match the targets
    better than `v27`'s, echoing the geometric trend of the word-level sweep.
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
            "No results yet — run the experiment (it publishes metrics, arrays, eval sets, "
            "and checkpoints on completion):\n\n"
            "```bash\nbin/mini run docs/m2/ex-2.1.4/experiment.py --app modal --max-containers 9\n```"
        ),
    )
    metrics, arrays, evals = loaded
    return arrays, evals, metrics


@app.cell(hide_code=True)
def _(metrics):
    def _mean(g, es, key):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es][key] for s in SEEDS]))

    _hold = {g: _mean(g, "named_holdout", "accuracy") for g in GRID_NAMES}
    _mal = {g: _mean(g, "named_holdout", "p_malformed") for g in GRID_NAMES}
    mo.md(
        "**Headline numbers.** Mean held-out greedy accuracy: "
        + ", ".join(f"`{g}` **{v:.2f}**" for g, v in _hold.items())
        + " (word level scored 0.27 and ≈ 1.0 on these grids). Malformed completions: "
        + ", ".join(f"{v:.1%} on `{g}`" for g, v in _mal.items())
        + f". Held-out guesses land a mean {_mean('v216', 'named_holdout', 'guess_dist'):.2f} "
        f"from the true mix on `v216` (chance {_mean('v216', 'named_holdout', 'chance_dist'):.2f}). "
        "The sections that follow fill this in, starting with the corpus accounting."
    )
    return


@app.cell(hide_code=True)
def _(metrics):
    _stats = metrics["corpus_stats"]
    _rows = "".join(
        f"<tr><td><code>{g}</code></td>"
        f'<td class="num">{s["n_colors"]:,}</td>'
        f'<td class="num">{s["n_seen_distinct"]:,}</td>'
        f'<td class="num">{s["n_holdout"]:,}</td>'
        f'<td class="num">{s["n_open"]:,}</td>'
        f'<td class="num">{s["total_tokens"]:,}</td>'
        "</tr>"
        for g, s in _stats.items()
    )
    _table = (
        '<div class="report-table-scroll"><table class="report-table">'
        '<tr><th>grid</th><th class="num">colors</th><th class="num">distinct pairs in corpus</th>'
        '<th class="num">held out</th><th class="num">open pairs</th>'
        '<th class="num">corpus tokens</th></tr>' + _rows + "</table></div>"
    )
    _caption = mo.md(
        f"""
        Pair counts per grid. These match ex-2.1.3's by construction (same
        sampler, same seed); only the token count changes ({N_EXAMPLES:,} equations
        × 19 characters, against × 6 word-level tokens).
        """
    ).text
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    Both grids train smoothly under the same 100-epoch schedule as before. The
    curves are not comparable to ex-2.1.3's: the vocabulary here is 30 characters
    rather than hundreds of names, so chance loss is lower, and most characters in
    a line follow almost surely from the name's prefix, so much of the loss falls
    on the few genuinely uncertain positions, the first character of the answer
    most of all.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="val-loss",
        alt_text="""
            Validation loss versus training epoch for the two grids, v27 and v216,
            with three seeds each drawn as thin overlapping lines. Every curve settles
            smoothly inside the 100-epoch budget.
        """,
        caption="""
            Validation loss per epoch, with three thin lines per grid, one per seed.
            This is per-token loss over characters, so it does not line up with the
            word-level experiment's curves; what matters is that each curve converges.
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

    The yes-or-no question first: can the model answer equations it has never
    seen, now that answering means spelling the name out? Greedy accuracy is the
    end-to-end score, and candidate accuracy (the argmax over teacher-forced
    names) tells us how much of any gap is spelling rather than knowledge. The
    dashed line over each group of bars is the mark to match, ex-2.1.3's
    word-level accuracy on the same pairs.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _word = load_sibling(WORD_METRICS_REF)
    _sets = ["named_seen", "named_holdout"]

    @themed(
        name="accuracy",
        alt_text="""
            Two bar panels of exact-match accuracy (0 to 1) for grids v27 and v216,
            one panel for seen pairs and one for held-out pairs. Each grid shows a
            greedy bar and a candidate (argmax over teacher-forced names) bar in the
            grid's color, with a dashed horizontal line over the group marking
            ex-2.1.3's word-level accuracy on the same pairs. Bars are means over three
            seeds, dots the individual seeds. On seen pairs all bars are at 1.0. On
            held-out pairs v27's bars sit at zero, below its word-level line at 0.27,
            while v216's bars reach 0.91, just under its word-level line near 1.0.
        """,
        caption="""
            Exact match by grid and eval set; bars are means over three seeds, dots the
            individual seeds. Greedy decoding writes freely, so spelling mistakes count
            as misses; candidate scoring picks the highest-probability vocabulary name
            and so cannot misspell. The dashed line over each group is the word-level
            benchmark, ex-2.1.3's accuracy on the same pairs with one-token names.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_sets), figsize=(8.4, 3.4), sharey=True)
        shades = grid_shades()
        fg = light_dark("#333", "#ddd")
        for ax, es in zip(axes, _sets, strict=True):
            for i, g in enumerate(GRID_NAMES):
                per_seed = [
                    [cell_of(metrics, g, s)["sets"][es][k] for s in SEEDS] for k in ("accuracy", "cand_accuracy")
                ]
                for x, vals, a in zip([i - 0.16, i + 0.16], per_seed, (1.0, 0.55), strict=True):
                    ax.bar([x], [np.mean(vals)], color=shades[g], alpha=a, width=0.28)
                    ax.plot([x] * len(vals), vals, "o", color=light_dark("#0008", "#fff8"), ms=3, zorder=3)
                if _word is not None:
                    wl = np.mean([cell_of(_word, g, s)["sets"][es]["accuracy"] for s in SEEDS])
                    ax.plot([i - 0.38, i + 0.38], [wl, wl], color=fg, ls="--", lw=1.4, zorder=4)
            ax.set(title=es.replace("_", " "), ylim=(-0.03, 1.03))
            ax.set_xticks(range(len(GRID_NAMES)), GRID_NAMES)
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("exact-match accuracy")

        def _pair(a: float) -> tuple:
            return tuple(plt.Rectangle((0, 0), 1, 1, fc=shades[g], alpha=a) for g in GRID_NAMES)

        _handles: list = [_pair(1.0), _pair(0.55)]
        _labels = ["greedy", "candidate"]
        if _word is not None:
            _handles.append(plt.Line2D([], [], color=fg, ls="--", lw=1.4))
            _labels.append("word level (ex-2.1.3)")
        axes[1].legend(
            _handles,
            _labels,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
            fontsize=8,
            loc="upper left",
        )
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, evals, metrics):
    def _mean(g, es, key):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es][key] for s in SEEDS]))

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
                    (chn,) = [i for i, dd in enumerate(diff) if dd]
                    n_1lvl += abs(lvl[int(gv[chn])] - lvl[ex.result[chn]]) == 1
        return n_miss, n_1ch, n_1lvl

    _m216 = _miss_structure("v216")
    _s2_27 = float(np.mean([cell_of(metrics, "v27", s)["sets"]["named_holdout"]["calibration"]["s2"] for s in SEEDS]))
    _s2_216 = float(np.mean([cell_of(metrics, "v216", s)["sets"]["named_holdout"]["calibration"]["s2"] for s in SEEDS]))
    mo.md(rf"""
    There are four grid × eval-set combinations here; three come out as hoped,
    and one falls apart. Start with what works. Seen pairs are perfect on both
    grids (greedy {_mean("v27", "named_seen", "accuracy"):.2f} /
    {_mean("v216", "named_seen", "accuracy"):.2f}). H1's well-formedness
    prediction holds with room to spare: not one malformed completion turns up
    across every eval set and seed, and the probability mass on complete names
    sums to {_mean("v216", "named_holdout", "mass_names"):.2f}. Spelling is a
    settled sub-problem for this model. That is why greedy and candidate accuracy
    coincide everywhere, and why we can read every miss below as a wrong color
    rather than a garbled string.

    That leaves the two held-out cells, and they part ways. `v216` lands at
    {_mean("v216", "named_holdout", "accuracy"):.2f}, below its word-level
    benchmark of ≈ 0.99 but comfortably where H2 placed it. The misses have the
    same shape as the word-level full grid's: pooled over seeds, {_m216[1]} of
    {_m216[0]} are wrong in exactly one RGB channel, and {_m216[2]} of those are
    off by a single grid level. So the model works out roughly the right mix and
    now and then rounds one channel to a neighboring level; the computation is
    sound, and only the last step of precision slips.

    `v27` held out is the outcome H2 did not leave room for: exactly zero, on
    every seed, below the word-level benchmark of 0.27. And the model is confident
    about it. The calibration dial reads s₂ = {_s2_27:.2f} on those prompts
    (confidently wrong; `v216` reads {_s2_216:.2f}, which is well calibrated), and
    the true answer's candidate NLL is ≈ 13 nats, so the model has committed to
    particular wrong answers rather than spreading its uncertainty. The failure
    rows further down show what it commits to: nearby names, and strikingly often
    one of the operands, a reasonable fallback since an operand is never far from
    the mix. The sparse end of the ladder is where char-level naming genuinely
    costs something. 66 distinct training pairs were just barely enough to pin
    down the geometry at word level, but not enough to pin down the geometry and
    the name-reading machinery together.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How close do the guesses land?

    Here is the graded view, as in ex-2.1.3: for every prompt, the RGB distance
    from the model's chosen name (the candidate argmax, so spelling plays no part)
    to the true mix, drawn as a cumulative distribution. The dashed floor line and
    the chance ticks say what to compare against.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, evals, metrics):
    _panels = [("named_holdout", "held-out pairs"), ("open", "open pairs")]

    @themed(
        name="distance-ecdf",
        alt_text="""
            Two panels of cumulative distributions of the RGB distance from the model's
            chosen name to the true mix, pooled over three seeds, one panel for held-out
            pairs and one for open pairs. One line per grid (v27, v216). Dashed lines
            show the nearest-name floor on open pairs. Vertical ticks mark each grid's
            chance distance.
        """,
        caption="""
            Distance from the model's choice to the true mix, in unit-cube units, pooled
            over seeds; a curve that climbs sooner (further left) is better. On held-out
            pairs the height at distance 0 is the candidate exact-match accuracy. On open
            pairs no name is exactly right, so the dashed line marks the best reachable
            distance (the nearest name), and the tick under the axis marks chance.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_panels), figsize=(8.4, 3.4), sharey=True)
        shades = grid_shades()
        for ax, (es, title) in zip(axes, _panels, strict=True):
            for g in GRID_NAMES:
                exs = evals[g][es]
                d = dists_for(g, exs)
                gd = np.concatenate([d[np.arange(len(exs)), arrays[f"{g}-s{s}/logp/{es}"].argmax(1)] for s in SEEDS])
                xs = np.sort(gd)
                ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=shades[g], lw=1.6, label=g)
                if es == "open":
                    xf = np.sort(d.min(axis=1))
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
    The graded view recovers most of what exact match missed, as H3 predicted.
    `v216` tracks its floor: held-out guesses land a mean
    {_mean("v216", "named_holdout", "guess_dist"):.3f} from the true mix, and
    open-pair guesses {_mean("v216", "open", "guess_dist"):.2f} against a floor of
    {_mean("v216", "open", "floor_dist"):.2f} and chance of
    {_mean("v216", "open", "chance_dist"):.2f}, landing on the single nearest name
    {_mean("v216", "open", "nearest_acc"):.0%} of the time.

    `v27` is the more interesting case. Its exact match was zero, yet its guesses
    are far from random: open-pair guesses average
    {_mean("v27", "open", "guess_dist"):.2f} (floor
    {_mean("v27", "open", "floor_dist"):.2f}, chance
    {_mean("v27", "open", "chance_dist"):.2f}), and its held-out curve has every
    guess landing at neighbor distance, about one grid step of 0.53 away and no
    further. So the geometry came through the tokenizer change on both grids. What
    `v27` lost is the last step alone: picking the exactly right name from
    evidence this thin.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is the whole distribution shaped like the geometry?

    Exact match reads off the top of the model's answer distribution, and the
    distance metrics read off its argmax. Neither says whether the rest of the
    probability mass respects the geometry. So we build a family of target
    distributions per prompt, one for each temperature τ: a softmax of the
    negative RGB distance from every vocabulary name to the true mix. Then we
    measure the KL divergence from target to model, `KL(q_τ ‖ p)`, as τ varies.
    (KL divergence is a one-directional distance between two distributions,
    measured in nats; it is zero when they match and grows when the model fails to
    put mass where the target does.) As τ → 0 the target collapses to one-hot on the
    nearest name, the classic score; large τ tends toward uniform. A model whose
    uncertainty is spread over colors near the true mix will fit some middle τ far
    better than a value-blind model could, and the τ that fits best is an estimate
    of the scale of the model's geometric uncertainty. The `KL(q_τ ‖ uniform)`
    line is the reference: it is what a model that gives every name equal
    probability scores at each τ.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, evals):
    _taus = np.geomspace(0.02, 1.2, 40)

    def _kl_curves(g: str, es: str) -> tuple[np.ndarray, np.ndarray]:
        """Mean-over-prompts KL(q_tau || model) and KL(q_tau || uniform), per tau."""
        exs = evals[g][es]
        d = dists_for(g, exs)  # (N, V)
        lp = np.stack([arrays[f"{g}-s{s}/logp/{es}"] for s in SEEDS])  # (S, N, V)
        log_pn = np.log(np.stack([renorm(m) for m in lp]) + 1e-30)
        kl_m, kl_u = [], []
        for t in _taus:
            lq = -d / t
            lq = lq - lq.max(axis=1, keepdims=True)
            q = np.exp(lq)
            q /= q.sum(axis=1, keepdims=True)
            h_q = -(q * np.log(q + 1e-30)).sum(axis=1)  # (N,)
            ce = -(q[None] * log_pn).sum(axis=2)  # (S, N)
            kl_m.append(float((ce - h_q[None]).mean()))
            kl_u.append(float((np.log(d.shape[1]) - h_q).mean()))
        return np.array(kl_m), np.array(kl_u)

    _panels = [("named_holdout", "held-out pairs"), ("open", "open pairs")]

    @themed(
        name="distance-kl",
        alt_text="""
            Two panels (held-out pairs, open pairs) of KL divergence from a
            distance-shaped target distribution to the model's answer distribution,
            against the target temperature tau on a log axis. One solid line per grid
            (v27, v216); dashed lines show the same divergence to a uniform
            distribution, the value-blind reference.
        """,
        caption="""
            How well does the model's answer distribution match distance-shaped
            targets? Solid lines: mean KL(q_τ ‖ model), pooled over seeds and prompts,
            where q_τ is a softmax of −distance/τ around the true mix. Dashed lines:
            KL(q_τ ‖ uniform), the score a value-blind guesser gets. A dip well below
            the dashed line at moderate τ means the model's probability mass gathers
            near the true mix, spread over its neighbors rather than resting on one
            best guess.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_panels), figsize=(8.4, 3.4), sharey=True)
        shades = grid_shades()
        for ax, (es, title) in zip(axes, _panels, strict=True):
            for g in GRID_NAMES:
                kl_m, kl_u = _kl_curves(g, es)
                ax.plot(_taus, kl_m, color=shades[g], lw=1.6, label=g)
                ax.plot(_taus, kl_u, color=shades[g], lw=1.0, ls="--", alpha=0.7)
            ax.set(title=title, xlabel="target temperature τ", xscale="log")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("KL(target ‖ ·)  (nats)")
        axes[0].legend(loc="upper right", fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The two grids sit on opposite sides of the uniform reference, and the reason
    is worth working through. On `v216` the model's distribution fits even the
    sharpest targets well: KL ≈ 0.3 nats at τ ≈ 0.03 on held-out pairs, an order
    of magnitude under the uniform line, with the best-fit τ near the grid spacing
    of 0.2, as H5 hoped. What little mass leaves the true answer goes to its
    immediate neighbors.

    `v27` fits worse than uniform at every temperature. On its own that reads as
    no geometric structure at all, yet the distance curves just showed the
    reverse, that its guesses sit systematically near the mix. The explanation is
    that `KL(q_τ ‖ p)` is very sensitive to confident error. This model's answer
    entropy is under 0.2 nats, meaning it puts nearly all its mass on one name, so
    whenever that name is not the target's favorite the divergence shoots up; a
    model with the same argmax that spread its mass would score far better. So this
    metric reads two things at once: whether the mass sits in geometrically
    sensible places, and whether the model's confidence is earned. It is the
    distributional counterpart of the s₂ dial, and it flags the same cells. H5
    splits along the grids: confirmed on `v216`, while on `v27` the trouble is
    overconfidence rather than geometric ignorance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is the mix computed in value space, and when?

    Two probe suites. The per-depth probes ask whether the residual stream at the
    pre-answer position (the space after `=`) already holds the mix's RGB before
    any of the answer is present in the context; we fit them on in-training lines
    and transfer them to held-out and open prompts. The schedule probe asks what
    happens during emission: teacher-force whole equations and read the three
    channels at every position around the answer, at every depth. Ex-2.1.2's
    version of this picture, on hex answers, was a staircase with eviction, and H4
    predicts a plateau here.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="residual-probes",
        alt_text="""
            Two line panels, one per grid (v27, v216), of ridge-probe R-squared for the
            mix's RGB read from the pre-answer residual stream, against depth 0 to 4.
            Lines show the probe's fit set (in-training lines) and its transfer to seen,
            held-out, and open prompts.
        """,
        caption="""
            Is the answer computed in value space before emission begins? Ridge probes
            from the pre-answer residual stream to the true mix's RGB, per depth (0 is
            embeddings, 4 is the final block), fit on half the in-training probe lines
            (seed 0) and transferred to the eval sets. Transfer tells a genuine
            value-space computation apart from a probe that memorized the fit set.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(GRID_NAMES), figsize=(8.4, 3.2), sharey=True)
        styles = {"fit": "-", "named_seen": "-.", "named_holdout": "--", "open": ":"}
        shades = grid_shades()
        for ax, g in zip(axes, GRID_NAMES, strict=True):
            transfer = cell_of(metrics, g, 0)["transfer_r2"]
            for es, r2s in transfer.items():
                ax.plot(r2s, styles.get(es, "-"), color=shades[g], label=es.replace("_", " "))
            ax.set(title=g, xlabel="depth", ylim=(-0.1, 1.03))
            ax.set_xticks(range(len(next(iter(transfer.values())))))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("probe R² (mix RGB)")
        axes[0].legend(fontsize=7, loc="upper left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _op = cell_of(metrics, "v216", 0)["probe_r2"]["operand_rgb"]
    _res = cell_of(metrics, "v216", 0)["transfer_r2"]
    mo.md(rf"""
    The mix does get computed in value space at the pre-answer position, on both
    grids. What is new is when in depth it shows up. At word level the mix was
    decodable from depth 1–2 on. Here at `v216` it is essentially absent until the
    final block: R² {_res["fit"][2]:.2f} at depth 2, {_res["fit"][3]:.2f} at depth
    3, then {_res["fit"][4]:.2f} at depth 4. The earlier layers are busy with a
    different job. The operand probe climbs {" → ".join(f"{v:.2f}" for v in _op)}
    across depths, which tells us that reading a four-letter name back into a value
    is itself a multi-layer computation; the depth that word-level models put into
    arithmetic goes here into reading. The final-block mix transfers with almost no
    loss to held-out and open prompts ({_res["named_holdout"][4]:.2f} and
    {_res["open"][4]:.2f}), so it is a genuine value-space computation, just a late
    one.

    `v27` turns the picture around, in a way that matches its behavioral collapse.
    Mid-stack, a probe fit on training lines transfers respectably to held-out
    prompts (R² ≈ 0.6–0.8 at depths 1–2, depending on seed). By the final layer,
    transfer falls to
    {cell_of(metrics, "v27", 0)["transfer_r2"]["named_holdout"][4]:.2f}
    (depth 4, seed 0) while the fit set stays at
    {cell_of(metrics, "v27", 0)["transfer_r2"]["fit"][4]:.2f}. So the value the
    network worked out on the way up is real, and the last block then writes over
    it with the particular wrong answer the readout has settled on. This is the
    mechanistic side of the s₂ ≈ 0.9 result: the value is computed, then misread,
    the same gap ex-2.1.3 saw at `v27`, only wider.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, metrics):
    _hex = load_sibling(HEX_MARGINS_REF, kind="npz")

    @themed(
        name="answer-schedule",
        alt_text="""
            Line panels of probe R-squared against position around the answer (offsets −4
            to 3), read from the final residual layer, with one line per RGB channel drawn
            in its own color. Left panels show this experiment's opaque names (v27, v216),
            where the three channel lines run together at every offset. The right panel, if
            available, shows ex-2.1.2's hex answers, where the three separate into a
            staircase, each channel high only around its own emission position.
        """,
        caption="""
            The answer-emission schedule at the final residual depth (seed 0). Each line is
            R² for one RGB channel across offsets from the answer's first character; the
            answer sits at offsets 0–3, and the prompt's last characters are the negative
            offsets. Lines are drawn as steps because each offset is a separate character
            position, with the value held across the position and ramping between; line
            width tapers R → G → B so an offset where all three agree reads as nested bands
            rather than as whichever channel drew last. Hex answers (right, from ex-2.1.2's
            base-grammar cells) spell one channel per digit, and the probe found each
            channel computed just in time and dropped once emitted. Opaque names cannot be
            written that way, and H4 predicts all three channels stay decodable across the
            window.
        """,
    )
    def _plot() -> plt.Figure:
        panels: list[tuple[str, np.ndarray, list[int]]] = []
        for g in GRID_NAMES:
            r2 = arrays[f"{g}-s0/schedule/r2"]  # (offsets, depth+1, 3)
            panels.append((f"{g} (names)", r2[:, -1], cell_of(metrics, g, 0)["schedule_offsets"]))
        if _hex is not None and "control-s0/schedule/r2" in _hex:
            panels.append(("hex (ex-2.1.2)", _hex["control-s0/schedule/r2"][:, -1], list(range(-4, 4))))

        # Color is data: each channel's line is drawn in that channel's own hue.
        cols = light_dark(["#d1495b", "#2a9d5c", "#3b6fd4"], ["#ff6b7d", "#4fd07a", "#6ea3ff"])
        # The finding at v27/v216 is that all three channels agree, so the lines coincide
        # almost exactly. Tapering the widths keeps every channel visible where they do.
        lws = (2.6, 1.7, 1.0)

        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (title, m, offsets) in zip(axes, panels, strict=True):
            ax.axvline(0, color="grey", alpha=0.5, linestyle="--")
            for c in range(3):
                smooth_step(ax, offsets, np.clip(m[:, c], 0, 1), color=cols[c], lw=lws[c], ramp=0.5)
            ax.set(title=title, xlabel="offset from answer start", ylim=(-0.05, 1.05))
            ax.set_xlim(offsets[0] - 0.5, offsets[-1] + 0.5)
            ax.set_xticks(offsets)
            ax.grid(alpha=0.1)
        axes[0].set_ylabel("probe R² (mix RGB)")
        axes[0].legend(
            handles=[plt.Line2D([], [], color=cols[c], lw=lws[c], label=n) for c, n in enumerate("RGB")],
            fontsize=7,
            loc="lower left",
            ncols=3,
        )
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    H4's main prediction holds: there is no staircase. In the hex panel the three
    channel lines separate, each peaking at its own emission offset and falling once
    the digit is out, the just-in-time schedule with eviction that ex-2.1.2 found. In
    the name panels the three lines run together at every offset, close enough that
    they read as one banded line: whatever the stream knows about the mix, it knows
    for all three channels at once. A name that does not factorize into channels gets
    a representation that does not either.

    The "stays fully live" half of the prediction needs a small amendment. At
    `v216` the final-depth R² is high at the pre-answer position and the first
    answer character (≈ 0.95), dips to ≈ 0.55–0.6 through the middle of the name,
    and climbs back to ≈ 0.96 at the last character. One plausible reading is that
    the mid-name positions look backward to finish a spelling that is already
    decided, and carry only a thinned copy of the value while they do it. So there
    is no per-channel eviction, though the whole value does fade somewhat during
    emission. For anchoring, this means "the result" is sharpest at the pre-answer
    position or the first answer character, so an anchor would sit best there.
    (`v27`'s panel is high everywhere: with 27 colors and near-total memorization,
    everything is legible at every position, which is why the `v216` panel is the
    one that tells us anything.)

    ## What the misses look like

    To put numbers on the ground, here are a few completions, picked as the widest
    misses, so the worst cases rather than typical ones. Swatches show each opaque
    name's actual color.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, evals):
    def completion_rows(grid: str, es: str, k: int = 6) -> str:
        exs = evals[grid][es]
        lp = arrays[f"{grid}-s0/logp/{es}"]
        names = list(PALETTES[grid])
        d = dists_for(grid, exs)
        gd = d[np.arange(len(exs)), lp.argmax(1)]
        fl = d.min(axis=1)
        order = np.argsort(-gd)[:k]
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
        "<tr><th>a</th><th>b</th><th>true mix</th><th>model's choice</th>"
        '<th class="num">distance</th><th class="num">floor</th></tr>'
    )
    _tables = []
    for _g, _es in [("v27", "named_holdout"), ("v216", "named_holdout"), ("v216", "open")]:
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
    The rows repeat the aggregates in miniature. `v216`'s worst held-out misses
    are neighbor colors, and its worst open-pair choices hover a step from the
    floor. `v27`'s misses are the telling ones: several give an operand back as
    the answer. That is a reasonable guess, since the mix is the midpoint of the
    two operands and so an operand is never far from it, and it is what we would
    expect from a model that has the neighborhood structure without the resolution
    to name the midpoint itself.

    ## Discussion

    Recall the question this experiment was built to answer: is one token per
    concept load-bearing for geometry inference? The answer depends on how much
    evidence the corpus offers, and both halves are informative.

    Where the mixing table is rich (`v216`, ~2,500 distinct training pairs), the
    answer is no. The model reads opaque four-letter names, binds them to values
    across layers 1–3, computes the mix in value space in the final block, and
    spells the answer as a whole. Held-out accuracy is 0.91 against the word-level
    0.99, and the shortfall is made up of one-grid-level precision misses, the
    same failure the word-level full grid showed. The value subspace, the fixed
    pre-answer home for the result, and the distance-shaped answer distribution
    all come through the tokenizer change. For the anchored experiments this is the
    reassurance that matters: a corpus with `v216`'s density supports
    concept-level structure even when concepts span several tokens.

    Where the table is sparse (`v27`, 66 distinct training pairs), the answer is
    yes, and the failure mode is specific. The model still learns the neighborhood
    structure (open-pair guesses far below chance, every held-out miss a neighbor
    or an operand), but exact naming drops to zero and the model is confident in
    its wrong answers: s₂ ≈ 0.9, with the final block writing over a mid-stack
    value representation that had transferred at R² ≈ 0.7. Word-level `v27` got
    0.27 from the same evidence. So multi-token naming carries a real cost, and it
    lands where evidence is scarce. That is worth keeping in mind when reading the
    base language's `named_holdout` = 0: its named sub-grid is this same sparse
    27-color regime, with a hex distraction on top.

    Three findings carry into the anchored runs:

    - Depth budget. Reading names took up layers 1–3 at `v216`, and the mix only
      exists from the final block. A d64-L4 model leaves one layer of residual
      stream in which the result concept exists at all, thin ground for anchoring
      it "across the stream". This strengthens the queued deep-and-narrow plan (L8
      gives the anchor somewhere to live) and argues for anchoring operand
      concepts at operand positions, which exist from depth 1–2.
    - Emission keeps the value whole. There is no per-channel eviction: the
      answer's value is held for all channels at once, from the pre-answer
      position through emission, with a dip mid-name. An anchored result direction
      at the pre-answer position would not have to work around ex-2.1.2's
      compute-and-evict schedule on this language.
    - The distributional metric doubles as a calibration probe. KL against
      distance-shaped targets separated `v216` (which fits sharp targets at τ near
      the grid spacing) from `v27` (worse than uniform, because it is confidently
      wrong), which exact match and mean distance both miss. The queued post-hoc
      analysis of ex-2.1.3's saved distributions should use the same τ sweep so the
      rungs line up.

    Caveats. The corpora match ex-2.1.3's equation for equation, but hold about 3×
    the tokens, so these models took ~3× the gradient steps per epoch. `v216`
    still came out below its word-level twin, so that regression is not a side
    effect of the larger budget, though the `v27` comparison carries the confound
    in the other direction. The probe suites are fit on in-training lines, with
    transfer to held-out and open prompts as the check, and it passed where it
    mattered. `v27`'s holdout has ten pairs, so its zero is a zero out of ten,
    three times over. And the mid-emission dip deserves a per-position probe pass
    before we lean on it; it is one seed's picture at one grid.
    """)
    return


if __name__ == "__main__":
    app.run()
