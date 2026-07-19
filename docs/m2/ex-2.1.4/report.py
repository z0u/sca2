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

    [Ex-2.1.3](../ex-2.1.3/) made the color-mixing task almost frictionless:
    one opaque token per color, so a name *was* an embedding row, and the model
    inferred the whole latent color cube from mixing co-occurrences alone. A
    clean result, but a suspiciously comfortable one. Reading an operand took
    one embedding lookup; emitting the answer took one softmax. The messy parts
    of being a language model — assembling a concept across several tokens,
    holding it while emitting several more — were designed out.

    This experiment puts exactly that messiness back, and nothing else. The
    language is identical to ex-2.1.3's: same grids, same operand pairs, same
    train/holdout split, same number of equations. The only change is the
    tokenizer's view. Every color is now a four-letter name that the model
    reads and writes one character at a time:

    ```
    tkzk + qwfd = hjnp
    ```

    We think of the three languages as rungs on a ladder. Word-level names
    (ex-2.1.3) are the easy rung: concept = token. The base char + hex language
    is the top rung, the one the M2 milestone actually claims. This is the
    middle rung: concepts are multi-token, but there is still no hex
    scaffolding, no alias dictionary, no form rule. If word-level and
    char-level behavior diverge, the divergence is itself a finding about what
    tokenization does to concept formation, which is exactly what we need to
    know before choosing where anchors go.

    One design decision matters enough to explain up front. Ex-2.1.3's
    synthetic names (`c05f` for the color `#05f`) spell the value out per
    character — harmless when a name is atomic, but at char level that is just
    hex with a prefix, and it would hand back the per-channel scaffolding this
    rung exists to remove. Even the classic 27 palette names (`red`,
    `chartreuse`) won't do: their lengths vary, which confounds the
    answer-emission analysis. So every color gets an opaque random name:
    four letters, drawn without replacement, assigned independently of the
    color's value. No character carries channel information; the binding from
    spelling to value is holistic.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The language at char level

    Two grids from ex-2.1.3's sweep, chosen to bracket the interesting range:
    `v27` (the classic 3-level grid, barely enough closed pairs to learn from)
    and `v216` (the 6-level grid, where the word-level model essentially solved
    the task). Three seeds each on the frozen d64-L4 backbone. Every training
    line is `a + b = mix(a, b)` between vocabulary colors whose mix is on the
    vocabulary; the same fifth of distinct closed pairs is held out, and the
    same rendering seed fixes each equation's operand order — the corpora are
    ex-2.1.3's re-spelled, line for line.

    What changes is the model's job. A line is now 19 characters instead of 6
    tokens, so the {N_EXAMPLES:,}-equation corpus is about 3× the tokens (a
    confound we accept and will come back to). An operand's value is no longer
    an embedding lookup: the model must recognize `tkzk` as a unit across four
    positions and bind a value to it. And the answer is no longer one softmax:
    the model must begin emitting the mix's name before any of it exists in
    the context, then keep going, letter by letter, without losing the thread.

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
    informative. *Greedy accuracy* lets the model write freely: five characters
    of greedy decoding, exact-matched against the true name. It can fail by
    knowing the wrong color, or by misspelling — emitting a string that is no
    color's name at all — so we also track how often that happens
    (`p_malformed`). *Candidate accuracy* removes the spelling failure mode:
    we teacher-force every vocabulary name (plus its terminating newline)
    after the prompt, sum the character log-probabilities, and take the
    argmax. This also yields the model's full distribution over well-formed
    answers, which the word-level experiment got for free from its softmax.

    That distribution feeds the measurement this report adds to the family.
    Ex-2.1.3 scored answers against the one-hot truth (the NLL of the true
    name), which asks "how much mass is on the right answer?" but not "is the
    rest of the mass in sensible places?". Here we also build distance-shaped
    target distributions — softmax of the negative RGB distance to the true
    mix, at a temperature τ — and measure how far the model's answer
    distribution sits from them as τ varies. A model that knows the geometry
    should spread its uncertainty over the true mix's *neighbors*, and this
    metric works even on open pairs, where no name is exactly right and
    one-hot scoring is undefined.

    Distances mirror ex-2.1.3: the RGB distance from the model's chosen name
    (candidate argmax) to the true mix, against the nearest-name *floor* and
    the vocabulary-mean *chance* per prompt.

    Then the probes. Per-depth ridge probes read operand and result RGB from
    the residual stream (fit on in-training lines, with transfer to held-out
    and open prompts as the check against probe memorization). And the
    answer-schedule probe from ex-2.1.2 returns: teacher-force complete
    equations and probe for the mix's three channels at every position around
    the answer, per depth. In the base language that probe found just-in-time
    computation with eviction — hex digit k was decodable at its own emission
    position and dropped afterwards. That schedule was possible because hex
    factorizes: digit k *is* channel k, so an emitted channel is finished.
    An opaque name refuses to factorize, which turns the same probe into a
    sharp question about holistic emission.
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
    end-to-end score; candidate accuracy (argmax over teacher-forced names)
    tells us how much of any gap is spelling rather than knowledge, and the
    word-level bars from ex-2.1.3 sit alongside as the easy-rung reference.
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
            seen pairs and held-out pairs. Each grid shows three bars: greedy accuracy,
            candidate (argmax over teacher-forced names) accuracy, and the word-level
            accuracy from ex-2.1.3 as a reference. Bars are means over three seeds,
            dots the individual seeds.
        """,
        caption="""
            Exact match by grid and eval set: bars are means over three seeds, dots
            individual seeds. "Greedy" decodes freely (spelling mistakes count as
            misses); "candidate" picks the highest-probability vocabulary name, so it
            cannot misspell; "word level" is ex-2.1.3's accuracy on the same pairs
            with one-token names.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_sets), figsize=(8.4, 3.4), sharey=True)
        shades = grid_shades()
        _kinds = ["greedy", "candidate"] + (["word level"] if _word is not None else [])
        for ax, es in zip(axes, _sets, strict=True):
            for i, g in enumerate(GRID_NAMES):
                per_seed = [
                    [cell_of(metrics, g, s)["sets"][es][k] for s in SEEDS] for k in ("accuracy", "cand_accuracy")
                ]
                if _word is not None:
                    per_seed.append([cell_of(_word, g, s)["sets"][es]["accuracy"] for s in SEEDS])
                xs = i + np.arange(len(per_seed)) * 0.28 - 0.28
                alphas = [1.0, 0.65, 0.35]
                for x, vals, a in zip(xs, per_seed, alphas, strict=True):
                    ax.bar([x], [np.mean(vals)], color=shades[g], alpha=a, width=0.24)
                    ax.plot([x] * len(vals), vals, "o", color="#0008", ms=3, zorder=3)
            ax.set(title=es.replace("_", " "), ylim=(-0.03, 1.03))
            ax.set_xticks(range(len(GRID_NAMES)), GRID_NAMES)
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("exact-match accuracy")
        _legend = [plt.Rectangle((0, 0), 1, 1, fc="#888", alpha=a) for a in (1.0, 0.65, 0.35)[: len(_kinds)]]
        axes[1].legend(_legend, _kinds, fontsize=8, loc="upper left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    def _mean(g, es, key):
        return float(np.mean([cell_of(metrics, g, s)["sets"][es][key] for s in SEEDS]))

    mo.md(rf"""
    Seen pairs: `v27` {_mean("v27", "named_seen", "accuracy"):.2f}, `v216`
    {_mean("v216", "named_seen", "accuracy"):.2f} (greedy). Held out: `v27`
    {_mean("v27", "named_holdout", "accuracy"):.2f}, `v216`
    {_mean("v216", "named_holdout", "accuracy"):.2f}, with candidate accuracy
    at {_mean("v27", "named_holdout", "cand_accuracy"):.2f} and
    {_mean("v216", "named_holdout", "cand_accuracy"):.2f}. Malformed
    completions on held-out prompts: {_mean("v27", "named_holdout", "p_malformed"):.1%}
    and {_mean("v216", "named_holdout", "p_malformed"):.1%}; the probability
    mass on *some* complete name (summed over all teacher-forced candidates)
    averages {_mean("v27", "named_holdout", "mass_names"):.2f} and
    {_mean("v216", "named_holdout", "mass_names"):.2f}.
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
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("probe R² (mix RGB)")
        axes[0].legend(fontsize=7, loc="upper left")
        return fig

    mo.Html(_plot())
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
    ## What the misses look like

    A few concrete completions, chosen as the widest misses (the worst cases,
    not typical ones). Swatches show each opaque name's actual color.
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
    ## Discussion

    (Written after the results; see the hypothesis-by-hypothesis notes above.)
    """)
    return


if __name__ == "__main__":
    app.run()
