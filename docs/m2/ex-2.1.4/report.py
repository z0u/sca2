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
    from sca import baselines as bl
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

    def raw_rgb(grid: str) -> np.ndarray:
        """(V, 3) raw 0..15 channel values, which `sca.baselines` works in."""
        return np.array(list(PALETTES[grid].values()))

    def blind_index(grid: str, evals: dict) -> int:
        """The name a prompt-blind model would always answer, fit on the training pairs."""
        return bl.blind_index(dists_for(grid, evals[grid]["named_seen"]))

    def blind_nearest(grid: str, es: str, evals: dict) -> float:
        """How often the prompt-blind constant happens to *be* the nearest name."""
        return bl.blind_stats(dists_for(grid, evals[grid][es]), blind_index(grid, evals))["nearest"]

    def v27_evidence(evals: dict) -> dict[str, float]:
        """How much the `v27` split leaves to learn from, and how much it can grade.

        `named_seen` is the complete list of distinct training pairs at this grid (66),
        so the counts here are exact rather than sampled.
        """
        seen, hold = evals["v27"]["named_seen"], evals["v27"]["named_holdout"]
        names = {ex.lhs for ex in seen} | {ex.rhs for ex in seen}
        mixed = {n for ex in seen if ex.lhs != ex.rhs for n in (ex.lhs, ex.rhs)}
        shell = bl.shell_mask(nc.GRIDS["v27"], raw_rgb("v27"), [ex.result for ex in hold])
        blind = dists_for("v27", hold)[:, blind_index("v27", evals)]
        return {
            "seen": len(seen),
            "hold": len(hold),
            "informative": sum(ex.lhs != ex.rhs for ex in seen),
            "lonely": len(names - mixed),
            "shell_null": bl.neighborhood_exact_null(shell),
            "blind_exact": float((blind == 0).mean()),
        }

    def miss_shape(grid: str, arrays: dict, evals: dict) -> dict[str, float]:
        """Held-out guesses pooled over seeds: how near, how often an operand, and the null.

        See `sca.baselines` for why the operand rate needs a null at all: closure puts
        an operand one grid level from the mix wherever the operands differ, so on a
        coarse grid it is usually a member of the mix's one-step shell by construction.
        """
        exs = evals[grid]["named_holdout"]
        rgb = raw_rgb(grid)
        shell = bl.shell_mask(nc.GRIDS[grid], rgb, [ex.result for ex in exs])
        n = n_op = n_shell = 0
        for s in SEEDS:
            guesses = rgb[arrays[f"{grid}-s{s}/logp/named_holdout"].argmax(1)]
            for row, ex, gv in zip(shell, exs, guesses, strict=True):
                n += 1
                n_op += tuple(gv) in (tuple(ex.lhs), tuple(ex.rhs))
                n_shell += any(tuple(gv) == tuple(c) for c in rgb[row])
        null = bl.operand_shell_null(shell, rgb, [(ex.lhs, ex.rhs) for ex in exs])
        return {"n": n, "operand": n_op, "shell": n_shell, "null_operand": null}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.4: spelling the names

    In [ex-2.1.3](../ex-2.1.3/), our small transformer inferred the geometry of
    a color space from mixing equations alone, but every color was a single
    opaque token: reading an operand was one table lookup, and writing the
    answer took one softmax. This experiment keeps the same language and setup
    but makes every color a random[^why-not-hex] four-letter name that the
    model reads and writes one character at a time:

    ```
    tkzk + qwfd = hjnp
    ```

    The question is whether one token per concept is needed for geometry
    inference. This language variant sits between the word-level language
    (ex-2.1.3) and the hex languages (ex-2.1.1, ex-2.1.2), though the ordering
    by difficulty isn't entirely clear: the hex models seemed to learn some
    geometry but failed to predict held-out named colors.

    [^why-not-hex]: In ex-2.1.3, we used names like `c05f` for the denser grids.
    Those are single tokens that decode as hex, but we can't use those names as
    the multi-character names in this experiment, or the model will just learn
    the separable channel arithmetic again (as in ex-2.1.1).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The language

    We'll use two data grids from ex-2.1.3:

    - `v27` is the 3-level grid: 27 colors, with barely enough closed pairs to
      learn from.
    - `v216` is the 6-level grid, 216 colors, where the word-level model had
      essentially solved the task.

    Each grid trains three seeds of the d64-L4 architecture from scratch, on the
    same `a + b = mix(a, b)` equations as ex-2.1.3, but tokenized differently. A
    line is now 19 character tokens rather than 6, so the
    {N_EXAMPLES:,}-equation corpus holds about 3× the tokens; we assume this
    isn't a significant confound as long as training reaches a low loss.

    The model now has to recognize `tkzk` as one unit spread over four positions
    and attach a value to it. It then has to start writing the mix's name before
    any of it is in the context, and continue without losing track.

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

    - Greedy accuracy: decode five characters, taking the most likely at each
      step, and check them against the true name. We also record `p_malformed`:
      how often the output is no color's name at all.
    - Candidate accuracy: teacher-force every vocabulary name plus its closing
      newline and take the highest scorer. This sets spelling aside, and it
      recovers the model's full distribution over well-formed answers.
    - Distribution distance: compare that answer distribution to soft targets
      built from color geometry — a softmax of negative RGB distance to the true
      mix at a temperature τ. A model that has learned the geometry should
      spread its uncertainty over colors near the true mix. This also works on
      open pairs, where no name is exactly right.
    - Distances, as in ex-2.1.3: RGB distance from the model's chosen name (the
      candidate argmax) to the true mix, set against the nearest-name floor and
      the vocabulary-mean chance.
    - Probes: per-depth ridge probes for operand and result RGB, fit on
      in-training lines and checked for transfer to held-out and open prompts;
      plus the answer-schedule probe from ex-2.1.2, which teacher-forces whole
      equations and probes for the mix's three channels at every position around
      the answer, at each depth.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Written down before we looked at any results.

    **H1.** Training still works: seen-pair accuracy is near-perfect on both
    grids, and malformed completions are rare. Well-formedness is a local
    statistical pattern (after `= `, four letters and a newline) that a char
    model of this size picks up easily.

    **H2.** Held-out accuracy will be lower than ex-2.1.3 which scored ≈ 1.0,
    but well above zero. Multi-token names add ways to fail at binding and
    emission without removing the co-occurrence evidence the inference runs on.
    `v27` stays low, as it was at word level. A `v216` collapse to ≈ 0 would
    mean that multi-token naming on its own blocks the inference.

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
        × 19 characters, against × 6 word-level tokens). At `v27` the two pair
        columns are the entire closed universe: 76 equations exist, and 27 of
        the 66 training ones are `a + a = a`.
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
    rather than hundreds of names, so chance loss is lower.
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

    Greedy accuracy is the end-to-end score; candidate accuracy (the argmax
    over teacher-forced names) tells us how much of any gap is spelling rather
    than knowledge. The dashed line over each group of bars is ex-2.1.3's
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
    Seen pairs are perfect on both grids (greedy
    {_mean("v27", "named_seen", "accuracy"):.2f} /
    {_mean("v216", "named_seen", "accuracy"):.2f}), and H1 holds: no malformed
    completion turns up in any eval set or seed, and the probability mass on
    complete names sums to {_mean("v216", "named_holdout", "mass_names"):.2f}.
    Greedy and candidate accuracy agree everywhere, so it seems that spelling is
    a settled sub-problem.

    Held-out accuracy does drop. `v216` lands at
    {_mean("v216", "named_holdout", "accuracy"):.2f}, below its word-level
    benchmark of ≈ 0.99 but in line with H2. The misses are similar to the
    word-level full grid: pooled over seeds, {_m216[1]} of {_m216[0]} are
    wrong in exactly one RGB channel, and {_m216[2]} of those are off by a
    single grid level. The model works out roughly the right mix and
    occasionally rounds one channel to a neighboring level.

    `v27` drops much more than H2 expected: zero on every seed (compared to 0.27
    for word-level tokens). And the model is confident about it: s₂ =
    {_s2_27:.2f} on those prompts (confidently wrong; `v216` reads
    {_s2_216:.2f}, well calibrated), and the true answer's candidate NLL is ≈ 13
    nats. The failure rows further down show that it expects nearby names —
    often one of the operands, though the section after them shows that is what
    the grid geometry predicts on its own.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How close do the guesses land?

    The graded view, as in ex-2.1.3: for every prompt, the RGB distance from
    the model's chosen name (the candidate argmax, so spelling plays no part)
    to the true mix, drawn as a cumulative distribution.
    """)
    return


@app.cell(hide_code=True)
def _(arrays, evals):
    _panels = [("named_holdout", "held-out pairs"), ("open", "open pairs")]

    @themed(
        name="distance-ecdf",
        alt_text="""
            Two panels of cumulative distributions of the RGB distance from the model's
            chosen name to the true mix, pooled over three seeds, one panel for held-out
            pairs and one for open pairs. One line per grid (v27, v216). Dashed lines
            show the nearest-name floor on open pairs. Under each axis a triangle marks
            the prompt-blind constant baseline.
        """,
        caption="""
            Distance from the model's choice to the true mix, in unit-cube units, pooled
            over seeds; a curve that climbs sooner (further left) is better. On held-out
            pairs the height at distance 0 is the candidate exact-match accuracy. On open
            pairs no name is exactly right, so the dashed line marks the best reachable
            distance (the nearest name). The triangles on the x-axis are the prompt-blind
            constant (always answering the centre of the training answers).
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
                ax.plot([d[:, blind_index(g, evals)].mean()], [-0.02], marker="^", ms=4, color=shades[g], clip_on=False)
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

    def _blind(g, es):
        return float(dists_for(g, evals[g][es])[:, blind_index(g, evals)].mean())

    _shape = miss_shape("v27", arrays, evals)

    mo.md(rf"""
    The graded view recovers most of what exact match missed, as H3 predicted.

    `v216` tracks its floor: held-out guesses land a mean
    {_mean("v216", "named_holdout", "guess_dist"):.3f} from the true mix, and
    open-pair guesses {_mean("v216", "open", "guess_dist"):.2f} against a floor of
    {_mean("v216", "open", "floor_dist"):.2f}, a prompt-blind constant of
    {_blind("v216", "open"):.2f} and chance of
    {_mean("v216", "open", "chance_dist"):.2f}, landing on the single nearest name
    {_mean("v216", "open", "nearest_acc"):.0%} of the time.

    `v27` lands on the single nearest name {_mean("v27", "open", "nearest_acc"):.0%} of the time against
    the constant's {blind_nearest("v27", "open", evals):.0%}, and
    {_shape["shell"]} of the {_shape["n"]} held-out guesses sit one grid step
    from the true mix (0.53), the rest at two steps. So the geometry did come
    through the tokenizer change on both grids, and what `v27` lost is the last
    step, picking the exactly right name — but this grid seems to be too coarse.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is the whole distribution shaped like the geometry?

    Exact match and the distance metrics read the top of the model's answer
    distribution. To check the rest of the mass, we build a target distribution
    per prompt for each temperature τ: a softmax of negative RGB distance from
    every vocabulary name to the true mix. Then we measure the KL divergence
    from target to model, `KL(q_τ ‖ p)`, as τ varies. As τ → 0 the target
    collapses to one-hot on the nearest name; large τ tends toward uniform. A
    model whose uncertainty is spread over colors near the true mix will fit
    some middle τ far better than a value-blind model could, and the best-fit τ
    estimates the scale of its geometric uncertainty. The dashed
    `KL(q_τ ‖ uniform)` line is what a model giving every name equal probability
    would score.
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
    On `v216` the model fits even the sharpest targets well: KL ≈ 0.3 nats at τ
    ≈ 0.03 on held-out pairs, an order of magnitude under the uniform line, with
    the best-fit τ near the grid spacing of 0.2, as H5 hoped.

    `v27` fits worse than uniform at every temperature, even though the distance
    curves above showed its guesses are close. `KL(q_τ ‖ p)` is very sensitive
    to confident error, and this model's answer entropy is under 0.2 nats, so
    whenever its chosen name is not the target's favorite, the divergence shoots
    up. The metric shows whether the mass sits in geometrically sensible places
    and whether the confidence is appropriate. So H5 is confirmed on `v216`, but
    not on `v27`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Probes: Is the mix computed in value space, and when?

    Per-depth probes ask whether the residual stream at the pre-answer
    position (the space after `=`) already holds the mixed color before any of
    the answer is in the context; we fit them on in-training lines and transfer
    them to held-out and open prompts.

    Schedule probes teacher-forces whole
    equations and reads the three channels at every position around the answer,
    at every depth. On hex answers (ex-2.1.2) this picture was a staircase with
    eviction; H4 predicts a plateau here.

    ### Per-depth probes
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
            Ridge probes from the pre-answer residual stream to the true mix's RGB, per
            depth (0 is embeddings, 4 is the final block), fit on half the in-training
            probe lines (seed 0) and transferred to the eval sets.
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
    The mix is computed in value space at the pre-answer position on both
    grids, but it shows up later. At word level it was
    decodable from depth 1–2 on. Here at `v216` it is essentially absent until
    the final block: R² {_res["fit"][2]:.2f} at depth 2, {_res["fit"][3]:.2f}
    at depth 3, then {_res["fit"][4]:.2f} at depth 4. The earlier layers are
    reading: the operand probe climbs
    {" → ".join(f"{v:.2f}" for v in _op)} across depths, so turning a
    four-letter name back into a value is itself a multi-layer computation. The
    final-block mix transfers with almost no loss to held-out and open prompts
    ({_res["named_holdout"][4]:.2f} and {_res["open"][4]:.2f}), so it is a
    real value-space computation.

    `v27` matches its behavioral collapse. Mid-stack, a probe fit on training
    lines transfers respectably to held-out prompts (R² ≈ 0.6–0.8 at depths 1–2,
    depending on seed); by the final layer, transfer falls to
    {cell_of(metrics, "v27", 0)["transfer_r2"]["named_holdout"][4]:.2f}
    (depth 4, seed 0) while the fit set stays at
    {cell_of(metrics, "v27", 0)["transfer_r2"]["fit"][4]:.2f}.
    The value seems to be computed, but the last block overwrites it with the
    wrong answer the readout has settled on.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Schedule probes
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
        # (title, r2, offsets, is_reference) — reference panels come from another
        # experiment, and are set apart from this one's panels below.
        panels: list[tuple[str, np.ndarray, list[int], bool]] = []
        for g in GRID_NAMES:
            r2 = arrays[f"{g}-s0/schedule/r2"]  # (offsets, depth+1, 3)
            panels.append((f"{g} (names)", r2[:, -1], cell_of(metrics, g, 0)["schedule_offsets"], False))
        if _hex is not None and "control-s0/schedule/r2" in _hex:
            panels.append(("hex (ex-2.1.2)", _hex["control-s0/schedule/r2"][:, -1], list(range(-4, 4)), True))

        # Color is data: each channel's line is drawn in that channel's own hue.
        cols = light_dark(["#d1495b", "#2a9d5c", "#3b6fd4"], ["#ff6b7d", "#4fd07a", "#6ea3ff"])
        # The finding at v27/v216 is that all three channels agree, so the lines coincide
        # almost exactly. Tapering the widths keeps every channel visible where they do.
        lws = (2.6, 1.7, 1.0)

        # A spacer column ahead of each reference panel: the extra gap, plus the rule
        # drawn in it, say "this one is from elsewhere" without a caption having to.
        gap = 0.1
        widths = [w for i, (*_, ref) in enumerate(panels) for w in ((gap, 1.0) if ref and i else (1.0,))]
        fig = plt.figure(figsize=(3.2 * sum(widths), 2.8))
        gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.1)

        axes, col = [], 0
        for title, m, offsets, ref in panels:
            if ref and col:
                rule = fig.add_subplot(gs[0, col])
                rule.set_axis_off()
                rule.axvline(0.5, color="grey", alpha=0.35, lw=0.8)
                col += 1
            # Shared y scale throughout, but the reference panel is far enough from the
            # left edge that it earns its own tick labels.
            ax = fig.add_subplot(gs[0, col], sharey=axes[0] if axes else None)
            # ax.tick_params(labelleft=col == 0 or ref)
            ax.tick_params(labelleft=col == 0)
            col += 1
            ax.axvline(0, color="grey", alpha=0.5, linestyle="--")
            for c in range(3):
                smooth_step(ax, offsets, np.clip(m[:, c], 0, 1), color=cols[c], lw=lws[c], ramp=0.5)
            ax.set(xlabel="offset from answer start", ylim=(-0.05, 1.05))
            ax.set_title(title, **({"style": "italic", "color": "grey"} if ref else {}))
            ax.set_xlim(offsets[0] - 0.5, offsets[-1] + 0.5)
            ax.set_xticks(offsets)
            ax.grid(alpha=0.1)
            axes.append(ax)
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
    H4 mainly holds: there is no per-channel staircase. In the name panels the three lines
    run together at every offset: whatever the stream knows about the mix, it
    knows for all three channels at once.

    The "stays fully live" half needs a small amendment. At `v216` the
    final-depth R² is high at the pre-answer position and the first answer
    character (≈ 0.95), dips to ≈ 0.55–0.6 through the middle of the name, and
    climbs back to ≈ 0.96 at the last character. Perhaps the mid-name positions
    look backward to finish a spelling that is already decided, carrying only a
    thinned copy of the value. So no per-channel eviction, though the whole
    value fades somewhat during emission; "the result" is sharpest at the
    pre-answer position or the first answer character.

    ## What the misses look like

    A few completions, picked as the widest misses, so worst cases rather than
    typical ones. Swatches show each opaque name's actual color.
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
    _k = 5
    _tables = []
    for _g, _es in [("v27", "named_holdout"), ("v216", "named_holdout"), ("v216", "open")]:
        _tables.append(
            mo.Html(
                figure_html(
                    f'<div class="report-table-scroll"><table class="report-table">{_head}'
                    f"{completion_rows(_g, _es, _k)}</table></div>",
                    caption=f"`{_g}` · {_es.replace('_', ' ')}: the {_k} widest misses (seed 0).",
                    class_="report-figure",
                )
            )
        )
    mo.vstack(_tables)
    return


@app.cell(hide_code=True)
def _(arrays, evals):
    _m = miss_shape("v27", arrays, evals)
    _n, _k, _null = _m["n"], _m["operand"], _m["null_operand"]
    _se = (_n * _null * (1 - _null)) ** 0.5
    _ev = v27_evidence(evals)
    mo.md(rf"""
    In `v216`, the worst held-out misses are neighbor colors, and its worst open-pair choices hover a step from the
    floor.
    The `v27` misses often return an operand ({_k} of {_n} predictions,
    {_k / _n:.0%}), which looks like an echo until you count the alternatives.
    Closure puts each operand one grid level from the mix, so a guesser that
    picks uniformly from the mix's one-step shell already returns an operand
    {_null:.0%} of the time; against that reference the observed rate is about
    {(_k / _n - _null) / (_se / _n):.1f} standard errors away, on {_n}
    predictions. So there is no operand echo here, only a small vocabulary.

    ## Discussion

    Is one token per concept needed for geometry inference? At `v216`, no.
    The model reads opaque four-letter names, binds them to values across layers
    1–3, computes the mix in value space in the final block, and spells the
    answer as a whole. Held-out accuracy is 0.91 against the word-level 0.99,
    and the shortfall is one-grid-level precision misses. The value subspace, the fixed pre-answer home
    for the result, and the distance-shaped answer distribution all survive
    the tokenizer change.

    It turns out the `v27` grid is too coarse to answer the question. Its closed-pair universe is {_ev["seen"] + _ev["hold"]}
    equations in total, split into {_ev["seen"]} for training and
    {_ev["hold"]} held out. Of the training pairs, {_ev["seen"] - _ev["informative"]}
    are `a + a = a`, leaving {_ev["informative"]} equations that say anything
    about how two names combine. There are {_ev["lonely"]} names that never
    appear in a mix at all, and one held-out pair is built entirely from them:
    both operands and the true answer occur in training only as `a + a = a`.
    No amount of skill reaches that answer from this corpus.

    The grading side is equally thin for v27. A guesser confined to the true mix's
    one-step shell scores {_ev["shell_null"]:.2f} exact on these ten pairs, and
    always answering the center of the training answers scores
    {_ev["blind_exact"]:.2f}, so Ex-2.1.3's word level 0.27 and this experiment's 0.00 both
    sit inside the range a model with no naming ability produces.

    The same caution applies to the base language in Ex-2.1.1. That experiment measured `named_holdout` = 0, but named sub-grid is this same 27-color regime with a hex distraction on top.

    ---

    Future experiments might use a v216-like vocabulary. If so, consider:

    - Reading names occupied layers 1–3 in this experiment, so the mix only exists
      from the final layer: a d64-L4 model leaves one layer of residual stream
      in which the result concept exists at all. Operand concepts are
      readable from depth 1–2.
    - Without hex, all three channels stay decodable together
      from the pre-answer position through the last character, with a dip
      mid-name, so an anchored result direction at the pre-answer position would
      not have to work around ex-2.1.2's compute-and-evict schedule.
    """)
    return


if __name__ == "__main__":
    app.run()
