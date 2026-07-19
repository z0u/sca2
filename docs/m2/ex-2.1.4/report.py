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
    from mini.vis import figure_html, light_dark, themed
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
        m_art, a_art = store.get_ref(METRICS_REF), store.get_ref(ARRAYS_REF)
        e_arts = {g: store.get_ref(f"{EVALS_REF}/{g}") for g in GRID_NAMES}
        if m_art is None or a_art is None or any(v is None for v in e_arts.values()):
            return None
        with tempfile.TemporaryDirectory() as d:
            metrics = json.loads(store.get(m_art, Path(d) / "metrics.json").read_text())
            with np.load(store.get(a_art, Path(d) / "arrays.npz")) as z:
                arrays = {k: z[k] for k in z.files}
            evals = {
                g: load_example_sets(store.get(art, Path(d) / f"{g}.json").read_bytes())
                for g, art in e_arts.items()
                if art is not None
            }
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

    [Ex-2.1.3](../ex-2.1.3/) showed that a small transformer can infer the
    geometry of a color space from mixing equations alone: give it lines like
    `red + blue = purple` (with made-up names) and it places the colors in a
    latent cube that matches the true one. But that experiment also made the
    task unusually easy in one specific way: every color was a single opaque
    token, so a name *was* an embedding row. Reading an operand took one table
    lookup, and emitting the answer took one softmax. The parts that make real
    language models interesting to study — assembling a concept across several
    tokens, and holding onto it while emitting several more — were designed
    out.

    This experiment puts that difficulty back, and nothing else. The language
    is identical to ex-2.1.3's: same grids, same operand pairs, same
    train/holdout split, same number of equations. The only change is the
    tokenizer's view. Every color is now a four-letter name that the model
    reads and writes one character at a time:

    ```
    tkzk + qwfd = hjnp
    ```

    It helps to picture the three languages as rungs on a ladder. Word-level
    names (ex-2.1.3) are the easy rung: one concept, one token. The base
    char + hex language is the top rung, the one the M2 milestone actually
    cares about. This experiment is the middle rung: concepts are multi-token,
    but there is still no hex scaffolding, no alias dictionary, no form rule
    to lean on. If word-level and char-level behavior diverge, the divergence
    itself teaches us something about what tokenization does to concept
    formation, and that is what we need to know before choosing where anchors
    go.

    One design decision is worth explaining up front. Ex-2.1.3's synthetic
    names (`c05f` for the color `#05f`) spell the value out character by
    character. That was harmless when a name was atomic, but at char level it
    would just be hex with a prefix, handing back the per-channel scaffolding
    this rung exists to remove. The classic 27 palette names (`red`,
    `chartreuse`) won't do either: their lengths vary, which would confound
    the answer-emission analysis. So every color gets an opaque random name:
    four letters, drawn without replacement, assigned independently of the
    color's value. No character carries channel information, and the binding
    from spelling to value is holistic.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The language at char level

    We keep two grids from ex-2.1.3's sweep, chosen to bracket the interesting
    range. `v27` is the classic 3-level grid: 27 colors, and barely enough
    closed pairs to learn from. `v216` is the 6-level grid, 216 colors, where
    the word-level model essentially solved the task. Each grid trains three
    seeds on the frozen d64-L4 backbone. Every training line is
    `a + b = mix(a, b)` between vocabulary colors whose mix is also on the
    vocabulary; the same fifth of distinct closed pairs is held out, and the
    same rendering seed fixes each equation's operand order. The corpora are
    ex-2.1.3's re-spelled, line for line.

    What changes is the model's job, in three ways. A line is now 19
    characters instead of 6 tokens, so the {N_EXAMPLES:,}-equation corpus is
    about 3× the tokens (a confound we accept and will come back to). An
    operand's value is no longer an embedding lookup: the model must recognize
    `tkzk` as a unit across four positions and bind a value to it. And the
    answer is no longer one softmax: the model must begin emitting the mix's
    name before any of it exists in the context, then keep going, letter by
    letter, without losing the thread.

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

    Exact match now comes in two flavors, and the gap between them is
    informative. *Greedy accuracy* lets the model write freely: we decode five
    characters greedily and exact-match them against the true name. A greedy
    miss can mean the model chose the wrong color, or that it misspelled
    (emitted a string that is no color's name at all), so we also track how
    often that happens (`p_malformed`). *Candidate accuracy* removes the
    spelling failure mode: we teacher-force every vocabulary name (plus its
    terminating newline) after the prompt, sum the character
    log-probabilities, and take the argmax over names. If greedy and candidate
    accuracy differ, spelling is part of the problem; if they agree, the
    misses are about which color the model believes in.

    Candidate scoring has a second use: it recovers the model's full
    probability distribution over well-formed answers, which the word-level
    experiment got for free from its softmax. That distribution feeds a new
    measurement. Ex-2.1.3 scored answers against the one-hot truth (the NLL
    of the true name), which asks "how much mass is on the right answer?" but
    not "is the rest of the mass in sensible places?". So here we also build
    distance-shaped target distributions (a softmax of the negative RGB
    distance to the true mix, at a temperature τ) and measure how far the
    model's answer distribution sits from them as τ varies. The idea is that
    a model that knows the geometry should spread its uncertainty over the
    true mix's *neighbors*. Usefully, this metric also works on open pairs,
    where no name is exactly right and one-hot scoring is undefined.

    Distances mirror ex-2.1.3: the RGB distance from the model's chosen name
    (candidate argmax) to the true mix, compared against the nearest-name
    *floor* and the vocabulary-mean *chance* per prompt.

    Then the probes. Per-depth ridge probes ask whether the operand and
    result RGB values can be read linearly from the residual stream; they are
    fit on in-training lines, and transfer to held-out and open prompts is
    the check against probe memorization. And the answer-schedule probe from
    ex-2.1.2 returns: teacher-force complete equations and probe for the
    mix's three channels at every position around the answer, per depth. In
    the base language that probe found just-in-time computation with
    eviction: hex digit k was decodable at its own emission position and
    dropped afterwards. That schedule was possible because hex factorizes —
    digit k *is* channel k, so an emitted channel is finished. An opaque name
    doesn't factorize, which turns the same probe into a sharp question
    about holistic emission.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Stated before looking at any results.

    **H1.** Training still works: seen-pair accuracy is near-perfect on both
    grids, and malformed completions are rare (well-formedness is a local
    statistical pattern — after `= `, four letters and a newline — that a char
    model of this size learns easily).

    **H2.** Held-out accuracy lands below the word-level rung on `v216`
    (where ex-2.1.3 scored ≈ 1.0) but well above zero: multi-token names add
    binding and emission failure modes without removing the co-occurrence
    evidence the inference runs on. `v27` stays low, as it was at word level.
    A `v216` collapse to ≈ 0 would say multi-token naming itself blocks the
    inference — a negative result, but a directly useful one, since the base
    language's names are multi-token too.

    **H3.** Where the task is learned, guesses stay geometrically close:
    near-floor distances on open pairs, misses that are neighbors rather than
    random names. Geometric closeness held at every vocabulary size at word
    level, including sizes whose exact match was poor; the prediction is that
    it survives the tokenizer change too.

    **H4.** The mix has a fixed home despite the multi-token answer. Because
    no character of an opaque name corresponds to a channel, the model cannot
    compute one channel, emit it, and evict it; whatever it knows about the
    mix must persist while the name is being spelled. Concretely: all three
    RGB channels stay decodable at late depth across the whole emission
    window, in contrast to ex-2.1.2's per-channel stair-step. We also expect
    the mix to be substantially decodable at the pre-answer position (as at
    word level), though it may be assembled later in depth, since operand
    identification now takes layers of work that embeddings used to do.

    **H5.** The answer distribution is distance-shaped. The candidate
    distribution's KL divergence to the distance-softmax target, minimized
    over τ, is far below the same divergence for a value-blind reference
    (uniform over names), and the best-fit τ is on the order of the grid
    spacing. Less confidently: `v216`'s distribution should match the targets
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
        "The sections below build up the picture; first, the corpus accounting."
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
        Pair accounting per grid — identical to ex-2.1.3's by construction (same
        sampler, same seed); only the token count changes ({N_EXAMPLES:,} equations
        × 19 characters, vs × 6 word-level tokens).
        """
    ).text
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    Both grids train stably under the unchanged 100-epoch schedule. Note the
    curves are not comparable to ex-2.1.3's: the vocabulary is 30 characters
    rather than hundreds of names (so chance loss is lower), and most
    characters in a line are highly predictable given the name's prefix, so
    a large share of the loss is spent on the few genuinely uncertain
    positions — the first character of the answer above all.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="val-loss",
        alt_text="""
            Validation loss against training epoch for the two vocabulary grids (v27,
            v216), three seeds each drawn as thin overlapping lines. Both settle
            smoothly within the 100-epoch budget.
        """,
        caption="""
            Validation loss per epoch (three thin lines per grid: the seeds). Per-token
            loss over characters, so not comparable to the word-level experiment's
            curves; each curve's own convergence is what matters.
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

    The binary question first: can the model answer equations it has never
    seen, now that answering means spelling? Greedy accuracy is the honest
    end-to-end score, and candidate accuracy (argmax over teacher-forced
    names) tells us how much of any gap is spelling rather than knowledge.
    The dashed line over each group of bars is the benchmark to beat — or
    rather, to match: ex-2.1.3's word-level accuracy on the same pairs.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _word = load_sibling(WORD_METRICS_REF)
    _sets = ["named_seen", "named_holdout"]

    @themed(
        name="accuracy",
        alt_text="""
            Two bar panels of exact-match accuracy (0 to 1) for grids v27 and v216:
            seen pairs and held-out pairs. Each grid shows a greedy bar and a candidate
            (argmax over teacher-forced names) bar in the grid's color, with a dashed
            horizontal line over the group marking ex-2.1.3's word-level accuracy on
            the same pairs. Bars are means over three seeds, dots the individual
            seeds. Seen pairs: all bars at 1.0. Held out: v27's bars sit at zero,
            below its word-level line at 0.27; v216's bars reach 0.91, just under its
            word-level line near 1.0.
        """,
        caption="""
            Exact match by grid and eval set: bars are means over three seeds, dots
            individual seeds. "Greedy" decodes freely (spelling mistakes count as
            misses); "candidate" picks the highest-probability vocabulary name, so it
            cannot misspell. The dashed line over each group is the word-level
            benchmark: ex-2.1.3's accuracy on the same pairs, with one-token names.
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
    There are four grid × eval-set combinations here; three behave, and one
    collapses. Start with what works. Seen pairs are perfect on both grids
    (greedy {_mean("v27", "named_seen", "accuracy"):.2f} /
    {_mean("v216", "named_seen", "accuracy"):.2f}). H1's well-formedness
    prediction also holds, and then some: there is not one malformed
    completion across every eval set and seed, and the summed probability
    mass on complete names is
    {_mean("v216", "named_holdout", "mass_names"):.2f}. Spelling is a solved
    sub-problem for this model. That is why greedy and candidate accuracy
    coincide everywhere, and why we can read every miss below as a wrong
    *color* rather than a wrong string.

    That leaves the two held-out cells, and they split. `v216` lands at
    {_mean("v216", "named_holdout", "accuracy"):.2f}, below its word-level
    benchmark of ≈ 0.99 but comfortably where H2 put it. The misses have the
    same structure as the word-level full grid's: pooled over seeds,
    {_m216[1]} of {_m216[0]} are wrong in exactly one RGB channel, and
    {_m216[2]} of those are off by exactly one grid level. In other words,
    the model computes approximately the right mix and occasionally rounds
    one channel to a neighboring grid level; the computation is intact, and
    only the last level of precision is lost.

    `v27` held out is the result H2 did not allow for: exactly zero, on
    every seed, *below* the word-level benchmark of 0.27. And the model is
    not hedging. The calibration dial reads s₂ = {_s2_27:.2f} on those
    prompts (confidently wrong; `v216` reads {_s2_216:.2f}, calibrated), and
    the true answer's candidate NLL is ≈ 13 nats: the model has settled on
    specific wrong answers rather than spreading its uncertainty. The
    failure rows further down show what it settles on — nearby names, and
    strikingly often *one of the operands*, a plausible fallback since an
    operand is never far from the mix. So the sparse end of the ladder is
    where char-level naming genuinely costs something: 66 distinct training
    pairs were enough evidence to pin down the geometry at word level
    (barely), but not enough to pin down the geometry *and* the name-reading
    machinery at once.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How close do the guesses land?

    The graded view, as in ex-2.1.3: for every prompt, the RGB distance from
    the model's chosen name (candidate argmax, so spelling is factored out) to
    the true mix, as a cumulative distribution; the dashed floor line and the
    chance ticks say what to compare against.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, evals, metrics):
    _panels = [("named_holdout", "held-out pairs"), ("open", "open pairs")]

    @themed(
        name="distance-ecdf",
        alt_text="""
            Two panels of cumulative distributions of the RGB distance from the model's
            chosen name to the true mix, pooled over three seeds: held-out pairs and
            open pairs. One line per grid (v27, v216). Dashed lines: the nearest-name
            floor on open pairs. Vertical ticks mark each grid's chance distance.
        """,
        caption="""
            Distance from the model's choice to the true mix (unit-cube units, pooled
            over seeds); higher and further left is better. On held-out pairs the value
            at distance 0 is the candidate exact-match accuracy. On open pairs no name
            is exactly right; the dashed line is the best achievable (nearest-name)
            distance, and the tick under the axis marks chance.
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
    The graded view recovers most of what exact match missed, as H3
    predicted. `v216` tracks its floor: held-out guesses land a mean
    {_mean("v216", "named_holdout", "guess_dist"):.3f} from the true mix, and
    open-pair guesses {_mean("v216", "open", "guess_dist"):.2f} against a
    floor of {_mean("v216", "open", "floor_dist"):.2f} and chance of
    {_mean("v216", "open", "chance_dist"):.2f}, choosing the single nearest
    name {_mean("v216", "open", "nearest_acc"):.0%} of the time.

    `v27` is the more interesting case. Its exact match was zero, but its
    guesses are far from random: open-pair guesses average
    {_mean("v27", "open", "guess_dist"):.2f} (floor
    {_mean("v27", "open", "floor_dist"):.2f}, chance
    {_mean("v27", "open", "chance_dist"):.2f}), and its held-out curve shows
    every guess landing at neighbor distance, roughly one grid step of 0.53
    away, never further afield. So the geometry survived the tokenizer change
    on both grids. What `v27` lost is only the final step: picking the
    exactly right name from evidence this sparse.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is the whole distribution shaped like the geometry?

    Exact match reads off the top of the model's answer distribution, and
    distance metrics read off its argmax; neither says whether the *rest* of
    the probability mass respects the geometry. So: build a family of target
    distributions per prompt — softmax of the negative RGB distance from each
    vocabulary name to the true mix, with temperature τ — and measure the KL
    divergence from target to model, `KL(q_τ ‖ p)`, as τ varies. At τ → 0 the
    target collapses to one-hot on the nearest name (the classic score); large
    τ approaches uniform. A model whose uncertainty is spread over the true
    mix's neighbors will fit some intermediate τ far better than a value-blind
    model could, and the τ that fits best estimates the *scale* of the model's
    geometric uncertainty. The uniform-target-fit line (`KL(q_τ ‖ uniform)`)
    is the reference: it is what a model that assigns every name equal
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
            targets? Solid: mean KL(q_τ ‖ model), pooled over seeds and prompts, where
            q_τ is a softmax of −distance/τ around the true mix. Dashed: KL(q_τ ‖
            uniform), what a value-blind guesser scores. A minimum well below the
            dashed line at moderate τ means the model's probability mass is
            concentrated near the true mix, not just on its single best guess.
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
    The two grids sit on opposite sides of the uniform reference, and the
    reason is worth unpacking. On `v216` the model's distribution fits the
    sharpest targets well: KL ≈ 0.3 nats at τ ≈ 0.03 on held-out pairs, an
    order of magnitude under the uniform line, with the best-fit τ near the
    grid spacing (0.2), as H5 hoped. What little mass leaves the true answer
    goes to its immediate neighbors.

    `v27` fits *worse than uniform* at every temperature. Taken alone, that
    reads as "no geometric structure at all" — but the distance curves just
    showed the opposite, that its guesses are systematically near the mix.
    The resolution is that `KL(q_τ ‖ p)` penalizes confident error very
    heavily. This model's answer entropy is under 0.2 nats (it commits fully
    to one name), so whenever the committed name is not the target's
    favorite, the divergence blows up; a model with the same argmax behavior
    that hedged would score far better. So this metric measures two things at
    once: whether the mass is in geometrically sensible places, *and* whether
    the model's confidence is warranted. It is the distributional twin of the
    s₂ dial, and it flags the same cells. H5 splits accordingly: confirmed on
    `v216`, while on `v27` the failure is overconfidence rather than
    geometric ignorance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is the mix computed in value space, and when?

    Two probe suites. The per-depth probes ask whether the residual stream at
    the pre-answer position (the space after `=`) contains the mix's RGB
    before any of the answer exists in the context — fit on in-training
    lines, transferred to held-out and open prompts. Then the schedule probe
    asks what happens *during* emission: teacher-force complete equations and
    read the three channels at every position around the answer, at every
    depth. Ex-2.1.2's version of this picture (hex answers) was a staircase
    with eviction; H4 predicts a plateau here.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="residual-probes",
        alt_text="""
            Two line panels, one per grid (v27, v216), of ridge-probe R-squared for the
            mix's RGB read from the pre-answer residual stream, against depth 0 to 4.
            Lines: the probe's fit set (in-training lines), and transfer to seen,
            held-out, and open prompts.
        """,
        caption="""
            Is the answer computed in value space before emission begins? Ridge probes
            from the pre-answer residual stream to the true mix's RGB, per depth (0 =
            embeddings, 4 = final), fit on half the in-training probe lines (seed 0)
            and transferred to the eval sets. Transfer distinguishes a genuine
            value-space computation from a probe that memorized the fit set.
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
    The mix does get computed in value space at the pre-answer position, on
    both grids. What is new is *when* in depth it appears. At word level the
    mix was decodable from depth 1–2 on. Here at `v216` it is essentially
    absent until the final block: R² {_res["fit"][2]:.2f} at depth 2,
    {_res["fit"][3]:.2f} at depth 3, then {_res["fit"][4]:.2f} at depth 4.
    The earlier layers turn out to be busy with a different job. The
    *operand* probe climbs {" → ".join(f"{v:.2f}" for v in _op)} across
    depths, which says that reading a four-letter name back into a value is
    itself a multi-layer computation; the depth that word-level models spent
    on arithmetic is spent here on reading. The final-block mix transfers
    essentially without loss to held-out and open prompts
    ({_res["named_holdout"][4]:.2f} and {_res["open"][4]:.2f}), so it is a
    genuine value-space computation, just a late one.

    `v27` inverts the picture, in a way that matches its behavioral collapse.
    Mid-stack, a probe fit on training lines transfers respectably to
    held-out prompts (R² ≈ 0.6–0.8 at depths 1–2, seed-dependent). By the
    final layer, transfer *falls* to
    {cell_of(metrics, "v27", 0)["transfer_r2"]["named_holdout"][4]:.2f}
    (depth 4, seed 0) while the fit set stays at
    {cell_of(metrics, "v27", 0)["transfer_r2"]["fit"][4]:.2f}. So the value
    the network computed on the way up is real, and the last block then
    overwrites it with the specific wrong answer the readout has committed
    to. This is the mechanistic side of the s₂ ≈ 0.9 result: the value is
    computed, then misread — the same gap ex-2.1.3 saw at `v27`, but wider.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, metrics):
    _hex = load_sibling(HEX_MARGINS_REF, kind="npz")

    @themed(
        name="answer-schedule",
        alt_text="""
            Heatmap panels of probe R-squared for the three RGB channels (rows) at each
            position around the answer (columns, offsets −4 to 3), read from the final
            residual layer. Left panels: this experiment's opaque names (v27, v216).
            Right panel, if available: ex-2.1.2's hex answers, which show a diagonal
            staircase — each channel decodable only around its own emission position.
        """,
        caption="""
            The answer-emission schedule at the final residual depth (seed 0). Each
            cell: R² for one RGB channel at one offset from the answer's first
            character (the answer occupies offsets 0–3; the prompt's last characters
            are the negative offsets). Hex answers (right, from ex-2.1.2's base-grammar
            cells) spell one channel per digit, and the probe found each channel
            computed just in time and dropped after emission. Opaque names cannot be
            emitted that way; H4 predicts all three channels stay decodable across the
            window.
        """,
    )
    def _plot() -> plt.Figure:
        panels: list[tuple[str, np.ndarray, list[int]]] = []
        for g in GRID_NAMES:
            r2 = arrays[f"{g}-s0/schedule/r2"]  # (offsets, depth+1, 3)
            panels.append((f"{g} (names)", r2[:, -1].T, cell_of(metrics, g, 0)["schedule_offsets"]))
        if _hex is not None and "control-s0/schedule/r2" in _hex:
            panels.append(("hex (ex-2.1.2)", _hex["control-s0/schedule/r2"][:, -1].T, list(range(-4, 4))))
        fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels) + 1.2, 2.6), sharey=True)
        cmap = light_dark("viridis", "viridis")
        for ax, (title, m, offsets) in zip(np.atleast_1d(axes), panels, strict=True):
            im = ax.imshow(np.clip(m, 0, 1), vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(offsets)), offsets, fontsize=8)
            ax.set_yticks(range(3), ["R", "G", "B"])
            ax.set(title=title, xlabel="offset from answer start")
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label="probe R²", fraction=0.03)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    H4's central prediction holds: there is no staircase. In the hex panel,
    each channel has its own bright cell at its own emission offset and fades
    once the digit is out; that is the just-in-time schedule with eviction
    that ex-2.1.2 found. In the name panels the three channel rows are nearly
    identical at every offset: whatever the stream knows about the mix, it
    knows about all three channels at once. A name that doesn't factorize
    into channels gets a representation that doesn't either.

    The "stays fully live" half of the prediction needs an amendment, though.
    At `v216` the final-depth R² is high at the pre-answer position and the
    first answer character (≈ 0.95), dips to ≈ 0.55–0.6 across the middle of
    the name, and returns to ≈ 0.96 at the last character. A plausible
    reading is that the mid-name positions attend backwards to finish a
    spelling that is already determined, and carry only a diluted copy of the
    value while doing it. So there is no per-channel eviction, but there is
    some whole-value attenuation during emission. For anchoring, the
    practical consequence is that "the result" is sharpest at the pre-answer
    position or the first answer character, and an anchor would be best
    placed there. (`v27`'s panel is high everywhere: with 27 colors and
    near-total memorization, everything is legible at every position, which
    is why the `v216` panel is the informative one.)

    ## What the misses look like

    To make the aggregates concrete, here are a few completions, chosen as
    the widest misses (the worst cases, not typical ones). Swatches show each
    opaque name's actual color.
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
    The rows repeat the aggregates in miniature. `v216`'s worst held-out
    misses are neighbor colors, and its worst open-pair choices hover a step
    from the floor. `v27`'s misses are the interesting ones: several echo an
    *operand* back as the answer. That is a defensible guess (the mix is the
    operands' midpoint, so an operand is never far from it), and it is what
    we would expect from a model that has the neighborhood structure without
    the resolution to name the midpoint itself.

    ## Discussion

    Recall the question this experiment was built to answer: is
    one-token-per-concept load-bearing for geometry inference? The answer
    depends on how much evidence the corpus offers, and both halves are
    informative.

    Where the mixing table is rich (`v216`, ~2,500 distinct training pairs),
    the answer is no. The model reads opaque four-letter names, binds them to
    values across layers 1–3, computes the mix in value space in the final
    block, and spells the answer holistically. Held-out accuracy is 0.91
    against the word-level 0.99, and the shortfall consists of one-grid-level
    precision misses, the same failure the word-level full grid showed. The
    value subspace, the fixed pre-answer home for the result, and the
    distance-shaped answer distribution all survive the tokenizer change.
    For the anchored experiments this is the reassurance that matters: a
    corpus with `v216`'s density supports concept-level structure even when
    concepts are multi-token.

    Where the table is sparse (`v27`, 66 distinct training pairs), the answer
    is yes, and the failure mode is specific. The model still learns the
    neighborhood structure (open-pair guesses far below chance; every
    held-out miss a neighbor or an operand), but exact naming collapses to
    zero and the model is confident in its wrong answers: s₂ ≈ 0.9, with the
    final block overwriting a mid-stack value representation that had
    transferred at R² ≈ 0.7. Word-level `v27` got 0.27 from the same
    evidence. So multi-token naming has a real cost, and it is paid where
    evidence is scarce. That is worth remembering when interpreting the base
    language's `named_holdout` = 0: its named sub-grid is this same sparse
    27-color regime, plus a hex distraction.

    Three things carry into the anchored runs:

    - Depth budget. Reading names consumed layers 1–3 at `v216`; the mix only
      exists from the final block. A d64-L4 backbone leaves one layer of
      residual stream in which the result concept exists at all — thin
      territory for anchoring it "across the stream". This strengthens the
      queued deep-and-narrow plan (L8 gives the anchor somewhere to live) and
      argues for anchoring operand concepts at operand positions, which exist
      from depth 1–2.
    - Emission keeps the value whole. There is no per-channel eviction: the
      answer's value is represented all-channels-at-once from the pre-answer
      position through emission (with a mid-name dip). An anchored result
      direction at the pre-answer position would not be working against
      ex-2.1.2's compute-and-evict schedule on this language.
    - The distributional metric doubles as a calibration probe. KL against
      distance-shaped targets separated `v216` (fits sharp targets at τ near
      the grid spacing) from `v27` (worse than uniform, because confidently
      wrong) — information exact match and mean distance both miss. The
      queued post-hoc analysis of ex-2.1.3's saved distributions should use
      the same τ sweep so the rungs are directly comparable.

    Caveats. The corpora match ex-2.1.3's in equations, not tokens; these
    models took ~3× the gradient steps per epoch. `v216` still came out
    *below* its word-level twin, so that regression is not an artifact of the
    larger budget, but the `v27` comparison inherits the confound in the
    other direction. The probe suites are fit on in-training lines (transfer
    to held-out and open prompts is the check, and it passed where it
    mattered). `v27`'s holdout has ten pairs, so its zero is a zero out of
    ten, three times. And the mid-emission dip deserves a per-position probe
    pass before being leaned on; it is one seed's picture at one grid.
    """)
    return


if __name__ == "__main__":
    app.run()
