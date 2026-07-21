import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.2: making composition necessary",
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
        CONDITIONS,
        CORPUS_SEED,
        HOLDOUT_FRAC,
        MARGINS_REF,
        METRICS_REF,
        N_EXAMPLES,
        OPEN_HOLDOUT_FRAC,
        SEEDS,
    )
    from mini.reports import externalize_html, report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import figure_html, light_dark, themed
    from sca.data import colors
    from subline.series import Series
    from subline.subline import Subline

    use_publisher(report_bundle(__file__))

    EVAL_SETS = [
        "named_seen",
        "named_holdout",
        "hex_unseen",
        "cross_unseen",
        "open_seen",
        "open_holdout",
        "alias_rev",
    ]
    CONDS = list(CONDITIONS)

    def load_results() -> tuple[dict, dict[str, np.ndarray]] | None:
        """Resolve the metrics and margin arrays from the store, or None if unpublished."""
        store = project_store()
        arts = store.get_refs([METRICS_REF, MARGINS_REF])
        m_art, g_art = arts[METRICS_REF], arts[MARGINS_REF]
        if m_art is None or g_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            m_path, g_path = store.get_many([(m_art, Path(d) / "metrics.json"), (g_art, Path(d) / "margins.npz")])
            metrics = json.loads(m_path.read_text())
            with np.load(g_path) as z:
                margins = {k: z[k] for k in z.files}
        return metrics, margins

    def label(cond: str, s: int) -> str:
        return f"{cond}-s{s}"

    def cell(metrics: dict, cond: str, s: int) -> dict:
        (r,) = [r for r in metrics["cells"] if r["label"] == label(cond, s)]
        return r

    def acc(metrics: dict, cond: str, s: int, eval_set: str) -> float:
        return cell(metrics, cond, s)["accuracy"][eval_set]["accuracy"]

    def cond_shades() -> dict[str, tuple]:
        stops = light_dark([0.82, 0.55, 0.32, 0.08], [0.88, 0.62, 0.42, 0.2])
        return dict(zip(CONDS, plt.cm.viridis(stops), strict=True))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.2: making composition necessary

    [Ex-2.1.1](../ex-2.1.1/) gave a decent baseline, but no model in the sweep
    ever solved `named_holdout`. Those are named operand pairs whose named form
    never appears in training, so the model can only answer by combining two
    things it has seen separately: the alias dictionary (which name maps to
    which hex color) and the mixing arithmetic.

    That report investigated and suggested why the models never learn to combine
    them: the corpus never forces them to, because the named slice is small
    enough to memorize outright. The alias dictionary is only ever shown in one
    direction, and a hex answer can be written one digit at a time, so the model
    never has to hold the whole mix in mind at once. The failure looked close,
    though; e.g on `lime + black`, the model seems to nudge its output toward
    the right answer, but the memorized lookup still wins.

    We keep ex-2.1.1's architecture, split, and training recipe (the d64-L4
    backbone) unchanged, but we add two new types of sequences: **rev** and
    **open**.

    - **rev** adds reverse alias lines (`#f00 = red`). The base grammar
      leaves the hex → name readout untrained (its absence is a small-scale
      reversal curse); this gives it direct supervision.
    - **open** adds equations with named operands whose mix falls *off* the
      palette, answered in hex (`red + navy = #804`). The answer's surface
      form then depends on the mix's *value*, not just the operands' forms,
      so a name + name prompt can no longer be settled by lookup alone: the
      model has to compute the mix to know what kind of answer to give.

    | Type | Example |
    |------|---------|
    | Named pairs | `red + blue = purple` |
    | Hex pairs   | `#f00 + #00f = #808`  |
    | Cross-form  | `red + #00f = #808`   |
    | Alias       | `red = #f00`          |
    | **Reverse** | `#f00 = red`          |
    | **Open**    | `red + orange = #f40` |

    We apply the two grammar interventions in a 2 × 2 factorial. Each
    intervention replaces the same number of tokens out of the hex slice, so the
    conditions differ only in what they add on top of the baseline.

    This is still an **un-anchored** experiment.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training data

    The corpus sampler is deterministic given the experiment's constants;
    regenerating the `both` corpus here shows the two new forms in context
    (reverse aliases and off-palette named equations, among the familiar
    lines):
    """)
    return


@app.cell(hide_code=True)
def _():
    from experiment import _classify as classify_form, _corpus as corpus_for

    _corpora = {cond: corpus_for(cond) for cond in CONDS}
    _head = "".join(ex.text for ex in _corpora["both"][:12])
    _counts = {
        cond: {f: sum(classify_form(ex) == f for ex in corpus) for f in CONDITIONS[cond]}
        for cond, corpus in _corpora.items()
    }
    _forms = ["hex", "named", "cross", "alias", "alias_rev", "open"]
    _rows = "".join(
        f"<tr><td><code>{cond}</code></td>"
        + "".join(f'<td class="num">{_counts[cond].get(f, 0) or "—"}</td>' for f in _forms)
        + "</tr>"
        for cond in CONDS
    )
    _open_train, _open_holdout = colors.split_open_pairs(CORPUS_SEED, OPEN_HOLDOUT_FRAC)
    _named_train, _named_holdout = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
    _table = (
        '<div class="report-table-scroll"><table class="report-table"><tr><th>condition</th>'
        + "".join(f'<th class="num"><code>{f}</code></th>' for f in _forms)
        + f"</tr>{_rows}</table></div>"
    )
    _caption = mo.md(
        f"""
        Lines per form in each condition's {N_EXAMPLES:,}-line corpus. Named equations draw
        from the same {len(_named_train)} train pairs as ex-2.1.1 (the same {len(_named_holdout)}
        pairs held out); open equations draw from {len(_open_train)} of the 302 off-palette
        named pairs, with {len(_open_holdout)} held out entirely.
        """
    ).text
    mo.vstack(
        [
            mo.md(f"```\n{_head}```"),
            mo.Html(figure_html(_table, caption=_caption, class_="report-figure")),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Stated before looking at any results.

    **H1.** Accuracy: `named_holdout` stays at zero in `control` and is lifted
    well off zero by `both`. `rev` alone lifts `named_holdout` partially: the
    garden path showed the mix is already half-computed on named prompts, and
    `rev` supplies the missing readout for whatever computation is there. `open`
    alone forces the computation, but leaves name emission supervised only
    through the memorizable named slice.

    **H2.** Margins: Scoring all 27 names as complete answers gives each
    held-out pair a *margin*: log-probability of the true name minus the
    best competitor's. In `control` the margins are negative (accuracy is
    zero) but spread out well above the random floor. Pairs with the
    least-negative control margins should be the first to flip positive under
    intervention. Interventions shift the whole margin distribution upward,
    while `named_seen` margins stay large and positive.

    **H3.** The answer schedule: Probing each answer position for each RGB
    channel (on hex prompts, strictly before each digit lands in the context)
    shows a stair-step: channel k stays low and becomes strongly decodable at
    the position that emits digit k.

    **H4.** Computed but outvoted: A result-color probe fit on open-pair
    prompts (name + name surface form) at the pre-answer position transfers
    to the held-out named prompts in proportion to how much the mix is
    actually computed there: middling R² in `control` (partial
    computation), rising toward the fit ceiling in the `open` conditions,
    and tracking `named_holdout` accuracy across conditions.

    **H5.** No collateral damage: `named_seen`, `hex_unseen`, and `cross_unseen`
    stay saturated in every condition, even though we are displacing up to 15%
    of the hex data. The `alias_rev` eval set reads ≈ 1 wherever reverse aliases
    are trained and stays random elsewhere, as in ex-2.1.1.

    If H1's interaction shows up but the single interventions do nothing,
    composition needed both ingredients at once. If `rev` alone saturates the
    set, the readout was the only missing piece and the lookup pressure never
    mattered. If nothing moves, the diagnosis was wrong somewhere upstream of
    the corpus. Any of these outcomes tells us how much D2.1's anchored runs
    can lean on `named_holdout` as an indicator of degradation.
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
            "No results yet — run the experiment (it publishes metrics, margins, and checkpoints "
            "on completion):\n\n"
            "```bash\nbin/mini run docs/m2/ex-2.1.2/experiment.py --app modal --max-containers 12\n```"
        ),
    )
    metrics, margins = loaded
    return margins, metrics


@app.cell(hide_code=True)
def _(metrics):
    _hold = {cond: np.mean([acc(metrics, cond, s, "named_holdout") for s in SEEDS]) for cond in CONDS}
    _sat = {
        cond: np.mean([acc(metrics, cond, s, es) for s in SEEDS for es in ("named_seen", "hex_unseen", "cross_unseen")])
        for cond in CONDS
    }
    mo.md(
        "**Headline numbers.** Mean `named_holdout` accuracy by condition: "
        + ", ".join(f"`{cond}` **{v:.2f}**" for cond, v in _hold.items())
        + ". The saturated sets (named seen, hex/cross unseen) average "
        + ", ".join(f"{v:.3f}" for v in _sat.values())
        + " in the same order. The figures below break this down."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Completion accuracy across the factorial

    Did either intervention move `named_holdout` off zero, and did the two new
    forms train the way we intended? The figure below shows exact-match accuracy
    for all seven eval sets in each of the four conditions.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="accuracy-factorial",
        alt_text="""
            Seven bar-and-dot panels of completion accuracy (0 to 1) against condition
            (control, rev, open, both), one panel per eval set: named seen, named holdout,
            hex unseen, cross unseen, open seen, open holdout, and reverse alias. Bars show
            the mean over three seeds, dots the individual seeds.
        """,
        caption="""
            Each panel is one eval set: the bar is the mean over three seeds and the dots
            are the individual seeds. `named_holdout` is the set H1 is about; `open_holdout`
            asks whether the forced computation carries over to off-palette pairs never seen
            in training; `alias_rev` checks the reverse-alias supervision. In `control` and
            `rev`, the open-form sets ask for a surface form those corpora never train on (a
            name + name prompt), so a low score there means the grammar is absent, not that
            the model tried and missed.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(EVAL_SETS), figsize=(12.5, 3.0), sharey=True)
        shades = cond_shades()
        xs = np.arange(len(CONDS))
        for ax, es in zip(axes, EVAL_SETS, strict=True):
            per_seed = np.array([[acc(metrics, cond, s, es) for s in SEEDS] for cond in CONDS])
            ax.bar(xs, per_seed.mean(axis=1), color=[shades[c] for c in CONDS], width=0.62)
            for i in range(len(CONDS)):
                ax.plot([xs[i]] * len(SEEDS), per_seed[i], "o", color="#0008", ms=2.5, zorder=3)
            ax.set(title=es.replace("_", " "), ylim=(-0.03, 1.03))
            ax.set_xticks(xs, CONDS, rotation=90, fontsize=7)
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("completion accuracy")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _a = {cond: float(np.mean([acc(metrics, cond, s, "named_holdout") for s in SEEDS])) for cond in CONDS}
    _interaction = _a["both"] - _a["rev"] - _a["open"] + _a["control"]
    mo.md(
        rf"""
    **H1 is refuted: `named_holdout` stays at zero in every condition** (its
    interaction term is a degenerate {_interaction:+.2f}). But the other panels
    show that both interventions *trained*. The `open` conditions answer
    held-out off-palette pairs at ≈ 0.9, so the mixing arithmetic runs on name +
    name prompts and generalizes to operand pairs never seen in that surface
    form. And `alias_rev` reads 1.0 exactly where reverse aliases were trained:
    the hex → name readout exists and works in the frame it was taught in. Both
    missing ingredients were supplied, yet the composition still doesn't appear
    as a *named* answer. The next section shows where the failure moved.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Right value, wrong spelling

    Below we list greedy completions of the held-out named prompts.
    In `control` and `rev` the answers are the
    familiar retrieval-flavored wrong names. In the `open` conditions
    something new happens: for a good fraction of pairs the model emits a
    *hex* answer, and when it does, the value is the correct mix. The
    swatches make it visible: reading down an `open`/`both` column, the hex
    answers match the expected color exactly, while the name answers are
    still near-miss neighbors.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    # Greedy completions of the named-holdout prompts are precomputed in `eval_one`
    # (where the model's already loaded) and ride the metrics JSON — so this reads
    # them from the store rather than pulling 12 checkpoints and re-decoding per edit.
    mo.stop(
        any("holdout_completions" not in cell(metrics, cond, s) for cond in CONDS for s in SEEDS),
        mo.md("These metrics predate precomputed completions — re-run the experiment to republish."),
    )
    completions = {(cond, s): cell(metrics, cond, s)["holdout_completions"] for cond in CONDS for s in SEEDS}

    def answer_value(text: str) -> tuple[int, int, int] | None:
        """The color a completion denotes, in either surface form (None if malformed)."""
        if text in colors.PALETTE:
            return colors.PALETTE[text]
        if len(text) == 4 and text.startswith("#") and all(c in "0123456789abcdef" for c in text[1:]):
            r, g, b = (int(c, 16) for c in text[1:])
            return (r, g, b)
        return None

    return answer_value, completions


@app.cell(hide_code=True)
def _(answer_value, completions, holdout_exs):
    def _swatch(text: str, want: tuple | None = None) -> str:
        rgb = answer_value(text)
        if rgb is None:
            return f"<code>{text}</code>"
        mark = "" if want is None else (" ✓" if rgb == want else "")
        return (
            f'<span aria-hidden="true" style="background: {colors.to_hex(rgb)}; border: 1px solid #8886; '
            f'border-radius: 2px; display: inline-block; width: 0.8em; height: 0.8em"></span> {text}{mark}'
        )

    _head = (
        f"<tr><th>prompt</th><th>{colors.swatch(None)} expected</th>"
        + "".join(f"<th>{colors.swatch(None)} {cond}</th>" for cond in CONDS)
        + "</tr>"
    )
    _rows = "".join(
        f"<tr><td><code>{ex.prompt}</code></td><td>{_swatch(ex.answer)}</td>"
        + "".join(f"<td>{_swatch(completions[cond, SEEDS[0]][i], ex.result)}</td>" for cond in CONDS)
        + "</tr>"
        for i, ex in enumerate(holdout_exs)
    )
    _table = f'<div class="report-table-scroll"><table class="report-table">{_head}{_rows}</table></div>'
    _caption = mo.md(
        f"""
        Greedy completions of the `named_holdout` prompts, seed {SEEDS[0]}, one column per
        condition. A ✓ marks answers whose *value* equals the true mix (they are all hex:
        value-correct in the wrong surface form).
        """
    ).text
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return


@app.cell(hide_code=True)
def _(answer_value, completions, holdout_exs, margins, metrics):
    _hexed = {cond: [sum(g.startswith("#") for g in completions[cond, s]) for s in SEEDS] for cond in CONDS}
    _value_ok = {
        cond: [
            sum(answer_value(g) == ex.result for g, ex in zip(completions[cond, s], holdout_exs, strict=True))
            for s in SEEDS
        ]
        for cond in CONDS
    }

    def _form_margin(cond: str) -> float:
        """Mean log P(correct hex) − log P(true name) on the holdout prompts."""
        vals = []
        for s in SEEDS:
            cands = cell(metrics, cond, s)["margin_candidates"]["named_holdout"]
            lp = margins[f"{label(cond, s)}/margins/named_holdout"]
            for i, ex in enumerate(holdout_exs):
                vals.append(lp[i, cands.index(colors.to_hex(ex.result))] - lp[i, cands.index(ex.answer)])
        return float(np.mean(vals))

    _n = len(holdout_exs)
    mo.md(
        f"""
    Across all seeds: hex-form answers on the {_n} held-out prompts number
    {", ".join(f"`{c}` **{sum(_hexed[c])}/{3 * _n}**" for c in CONDS)}, and value-correct answers
    {", ".join(f"`{c}` **{sum(_value_ok[c])}/{3 * _n}**" for c in CONDS)} — the two counts coincide
    exactly: every value-correct answer is hex, and every name answer is value-wrong. The form-choice
    margin (log-probability of the correct hex minus the true name, teacher-forced) tells the same
    story: {", ".join(f"`{c}` {_form_margin(c):+.1f}" for c in CONDS)} nats.

    So the `open` intervention did what it was designed to do: the mix is computed on name + name
    prompts. But the *form rule* ("answer named exactly when the mix lands on the palette") did not
    generalize: the model treats held-out closed pairs like open ones and answers in hex, or falls
    back to the lookup-neighbor name. And `rev` moved nothing here. Its perfect reverse-alias skill
    stays locked to the `#hex = ` frame it was trained in; supplying the inverse dictionary as a
    *surface task* does not make it available as an internal *readout* mid-equation.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Error margins, pair by pair

    Exact-match accuracy over ten pairs is a coarse instrument: it says whether the
    true name came out on top, but not whether it was close. So let's grade the
    name-identity question continuously and ask, pair by pair, whether intervention
    brings the true name any closer to winning.
    """)
    return


@app.cell(hide_code=True)
def _(margins, metrics):
    def margin_rows(eval_set: str, exs: list) -> np.ndarray:
        """(len(CONDS), n_seeds, n_examples) margin of the true name over the best *other name*."""
        out = np.empty((len(CONDS), len(SEEDS), len(exs)))
        for ci, cond in enumerate(CONDS):
            for si, s in enumerate(SEEDS):
                cands = cell(metrics, cond, s)["margin_candidates"][eval_set]
                names = [i for i, c in enumerate(cands) if not c.startswith("#")]
                lp = margins[f"{label(cond, s)}/margins/{eval_set}"]
                true_idx = np.array([cands.index(ex.answer) for ex in exs])
                truth = lp[np.arange(len(exs)), true_idx]
                rival = lp[:, names].copy()
                rival[np.arange(len(exs)), [names.index(t) for t in true_idx]] = -np.inf
                out[ci, si] = truth - rival.max(axis=1)
        return out

    _train_pairs, _holdout_pairs = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
    holdout_exs = colors.as_named(_holdout_pairs, seed=2)
    m_hold = margin_rows("named_holdout", holdout_exs)
    m_seen = margin_rows("named_seen", colors.as_named(_train_pairs, seed=1))
    return holdout_exs, m_hold, m_seen


@app.cell(hide_code=True)
def _(holdout_exs, m_hold, m_seen):
    @themed(
        name="margin-trajectories",
        alt_text="""
            Line chart of the answer margin (log-probability of the true name minus the best
            competitor) against condition (control, rev, open, both), one line per held-out
            named pair, averaged over seeds. A horizontal line marks zero, where the true
            answer starts to win; a shaded band shows the range of margins on the seen named
            pairs.
        """,
        caption="""
            The margin is the log-probability of the true name as a complete answer, minus
            that of the best competing name; it sets the hex-versus-name form choice aside,
            since only names count as candidates. Positive means the true name beats the
            other names, and the magnitude says by how much. One line per held-out pair,
            averaged over seeds and traced across the four conditions; the shaded band is
            the range of `named_seen` margins, for reference.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        xs = np.arange(len(CONDS))
        seen_mean = m_seen.mean(axis=1)  # (conds, n_seen)
        ax.fill_between(
            xs,
            np.percentile(seen_mean, 5, axis=1),
            np.percentile(seen_mean, 95, axis=1),
            color="#4d9de0",
            alpha=0.15,
            label="named seen (5–95%)",
        )
        mean = m_hold.mean(axis=1)  # (conds, n_pairs)
        order = np.argsort(mean[0])
        cmap = plt.cm.viridis(np.linspace(0.05, 0.85, mean.shape[1]))
        # Spread the end-of-line labels vertically so neighbors don't collide.
        label_y = mean[-1].astype(float).copy()
        for prev, nxt in zip(np.argsort(label_y)[:-1], np.argsort(label_y)[1:], strict=True):
            label_y[nxt] = max(label_y[nxt], label_y[prev] + 1.1)
        for rank, p in enumerate(order):
            ex = holdout_exs[p]
            ax.plot(xs, mean[:, p], "o-", color=cmap[rank], lw=1.4, ms=3.5)
            ax.annotate(
                f"{ex.prompt}{ex.answer}",
                (xs[-1], mean[-1, p]),
                xytext=(xs[-1] + 0.12, label_y[p]),
                textcoords="data",
                fontsize=6.5,
                va="center",
                color=cmap[rank],
            )
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xticks(xs, CONDS)
        ax.set(xlabel="condition", ylabel="margin: log P(true) − log P(best competitor)")
        ax.set_xlim(-0.3, len(CONDS) - 0.25 + 1.6)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(m_hold):
    _mean = m_hold.mean(axis=1)  # (conds, pairs)
    _ctrl, _both = _mean[0], _mean[CONDS.index("both")]
    _rho = float(np.corrcoef(np.argsort(np.argsort(_ctrl)), np.argsort(np.argsort(_both)))[0, 1])
    _flipped = int((_both > 0).sum())
    mo.md(
        f"""
    **H2 mostly fails too.** The distribution barely moves: mean margin
    {_ctrl.mean():+.1f} nats in `control` and {_both.mean():+.1f} in `both`,
    with {_flipped}/10 pairs ending positive. The true name never comes close
    to winning among names — it stays roughly ten nats adrift while
    `named_seen` sits far above zero. So the value → name translation isn't
    *almost there* and outvoted at the margin; on these prompts it simply
    never engages, even in the condition whose greedy answers prove the value
    has been computed. The rank structure is real but weak: rank correlation
    {_rho:.2f} between `control` and `both`.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The answer schedule: when each channel becomes readable

    Ex-2.1.1's probes read a single pre-answer position and averaged over RGB,
    which hides *when* the mix is computed. So here we fit a separate ridge probe
    for every (position, channel, layer) on hex prompts, and watch each RGB channel
    of the result become decodable as the answer is spelled out.

    H3 predicts a stair-step: each channel stays low until the position that emits
    its own digit, then rises sharply.
    """)
    return


@app.cell(hide_code=True)
def _(margins, metrics):
    sched_offsets = cell(metrics, "control", SEEDS[0])["schedule_offsets"]
    # Schedule R² arrays travel in the same npz as the margins (see eval_one).
    sched_r2 = np.mean([margins[f"{label('control', s)}/schedule/r2"] for s in SEEDS], axis=0)
    return sched_offsets, sched_r2


@app.cell(hide_code=True)
def _(sched_offsets, sched_r2):
    @themed(
        name="answer-schedule",
        alt_text="""
            Line charts of probe R-squared against position offset around the answer, one
            panel per residual-stream depth, three lines per panel for the R, G, and B
            channels of the result. Lines are solid before each channel's digit enters the
            context and dotted after. In the deeper layers each channel peaks near 1 at its
            own emission position and falls away on either side, so the three channels form a
            sequence of staggered peaks rather than a cumulative plateau; at depth 0 the
            dotted segments jump to 1 as each digit becomes readable from the context.
        """,
        caption="""
            One panel per residual-stream depth; three lines per panel for the R, G, and B
            channels of the result. Offset 0 is the `#`, digit k sits at offset k + 1 and is
            emitted from offset k. A line is solid where its digit is not yet in the context
            (so decoding it is computation) and dotted once it has landed (decoding is
            copying).
        """,
    )
    def _plot() -> plt.Figure:
        depths = sched_r2.shape[1]
        fig, axes = plt.subplots(1, depths, figsize=(12.5, 2.9), sharey=True)
        chan_colors = ["#e4572e", "#3aa76d", "#4d9de0"]
        offs = np.array(sched_offsets)
        for d, ax in enumerate(axes):
            for k, c in enumerate(chan_colors):
                emit = k + 1  # digit k enters the context at offset k + 1
                pre = offs <= emit
                ax.plot(offs[pre], sched_r2[pre, d, k], "o-", color=c, lw=1.8, ms=3, label=f"channel {'RGB'[k]}")
                ax.plot(offs[emit <= offs], sched_r2[emit <= offs, d, k], "o:", color=c, lw=1, ms=2, alpha=0.6)
            ax.axvline(-0.5, color="#888", lw=0.8, ls="--", alpha=0.6)
            ax.set(title=f"depth {d}", xlabel="offset from '#'", ylim=(-0.05, 1.05))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("probe R² (held-out half)")
        axes[0].legend(fontsize=7)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That is close to what the deep layers show. At the final layer, channel k is
    near-perfectly decodable exactly at its emission offset (R² ≈ 0.97) and *only*
    there. One position earlier it is far weaker, and once emission moves on to the
    next digit the previous channels largely fade from the deep residual stream.

    So each answer position holds just the one channel it is about to emit, riding
    on a diffuse ≈ 0.5-R² trace of the whole mix that persists from the pre-answer
    position. The mix is never fully represented at any single position: the
    "result" the pre-answer probe sees is a head start, not a finished value. This
    is just-in-time computation, and anchoring a *result* concept at one position
    might be at odds with the model's own schedule.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Computed but outvoted?

    If the mix really is computed on named prompts, a probe trained to read the
    result elsewhere should carry over to them. So we fit the probe on open-pair
    prompts (name + name surface form) at the pre-answer position, then score it on
    the named eval sets. Where the mix is never represented, no probe fit elsewhere
    can recover it.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _sets = ["fit", "open_holdout", "named_seen", "named_holdout"]

    @themed(
        name="transfer-probe",
        alt_text="""
            Four line charts of probe R-squared against residual-stream depth, one panel per
            prompt set: the fit set's held-back half, open holdout, named seen, and named
            holdout. One line per condition (control, rev, open, both; darker means richer
            corpus). The probes were fit on open-pair prompts at the pre-answer position.
        """,
        caption="""
            One panel per scored set: the fit set's held-back half, open holdout, named seen,
            and named holdout. R² against residual depth, one line per condition (darker means
            a richer corpus). Depth 0 is left out, since the pre-answer embedding is constant
            across prompts until attention runs.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_sets), figsize=(11.5, 3.0), sharey=True)
        shades = cond_shades()
        for ax, name in zip(axes, _sets, strict=True):
            for cond in CONDS:
                rows = np.array([cell(metrics, cond, s)["transfer_r2"][name] for s in SEEDS])
                # Skip depth 0: the pre-answer position's raw embedding is a constant
                # across prompts (attention hasn't run), so its transfer R² only reflects
                # the fit-vs-eval mean gap — no result is decodable there in any set.
                depths = np.arange(rows.shape[1])
                ax.plot(depths[1:], rows.mean(axis=0)[1:], "o-", color=shades[cond], label=cond, lw=2)
            ax.set(title=name.replace("_", " "), xlabel="residual depth", ylim=(-0.05, 1.05))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("probe R² (result RGB)")
        axes[0].legend(fontsize=8)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Partial computation is present everywhere, `control` included: the
    held-out named prompts carry roughly as much linearly-decodable result
    as the fit set's own ceiling (≈ 0.6 at the deep layers). That quantifies
    ex-2.1.1's garden-path anecdote. The `open` conditions raise the
    mid-depth transfer somewhat, consistent with the arithmetic being made
    load-bearing on these prompts, and their *final*-layer R² drops on the
    named sets — there, the last layer's job has become committing to a
    surface form. But the main point is that "computed but outvoted" was
    already true in `control`, and making the computation stronger (`open`)
    or the readout available (`rev`) still doesn't connect them.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The garden path, revisited

    Let's look again at Ex-2.1.1's walkthrough example: in `lime + black =
    green`, the model guesses the second operand as *blue*, updates to *black*
    when it sees the `a`, but only half-corrects the answer — the trained
    `lime + blue = teal` keeps winning[^unstable]. The margin data lets us ask
    that question again in every condition: does the correction finally overtake
    the lookup? It does not; the margin stays several nats negative in every
    condition. Sublines below show per-character surprisal and entropy on the
    same example (teacher-forced), one row per condition.

    [^unstable]: The retrained `control` gives the same *kind* of wrong answer as
    ex-2.1.1's d64-L4-s0, but not always the same one (this seed now says *gray*
    rather than *teal* for this pair). The training code is bit-for-bit
    deterministic on CPU at a fixed seed, so this is not an unseeded RNG in our
    code: it is GPU floating-point non-associativity — reductions such as the
    tied embedding's scatter-add gradient sum in a hardware-dependent order, and
    we set no XLA determinism flag (the one in `sca/__init__.py` is CPU-only).
    That only surfaces as a *changed answer* because this prompt is a knife-edge:
    the true name loses by ~10 nats and the winner is decided by a small gap
    among wrong neighbors, so a bit-level perturbation is enough to flip it. The
    instability is a symptom of the underdetermination this section is about, not
    a separate defect.
    """)
    return


@app.cell(hide_code=True)
def _(holdout_exs, m_hold):
    gp_idx = next(
        i for i, ex in enumerate(holdout_exs) if {"lime", "black"} == set(ex.prompt.split(" = ")[0].split(" + "))
    )
    _rows = "".join(
        f"<tr><td><code>{cond}</code></td>"
        + "".join(f'<td class="num">{m_hold[ci, si, gp_idx]:+.1f}</td>' for si in range(len(SEEDS)))
        + f'<td class="num"><b>{m_hold[ci, :, gp_idx].mean():+.1f}</b></td></tr>'
        for ci, cond in enumerate(CONDS)
    )
    _table = (
        '<div class="report-table-scroll">'
        '<table class="report-table"><tr><th>condition</th>'
        + "".join(f'<th class="num">seed {s}</th>' for s in SEEDS)
        + f'<th class="num">mean</th></tr>{_rows}</table>'
        + "</div>"
    )
    _caption = mo.md(
        """
        Margin of `green` over its best competitor on the `lime + black` prompt, per seed
        and condition (positive: the arithmetic wins).
        """
    ).text
    mo.Html(figure_html(_table, caption=_caption, class_="report-figure"))
    return (gp_idx,)


@app.cell(hide_code=True)
def _(gp_idx, holdout_exs, metrics):
    _log_v = np.log(len(colors.alphabet()))
    _gp_text = holdout_exs[gp_idx].prompt + holdout_exs[gp_idx].answer
    _rows_by_cond = [
        (cond, row)
        for cond in CONDS
        for row in cell(metrics, cond, SEEDS[0])["surprisal"]["named_holdout"]
        if row["text"] == _gp_text
    ]
    mo.stop(
        not _rows_by_cond,
        mo.md("*(The garden-path example fell outside the captured surprisal rows — skipping the sublines.)*"),
    )
    _sub_css = "svg { --bg-color: light-dark(#fff, #181c1a); }"

    def _pad(row: dict, key: str) -> np.ndarray:
        return np.concatenate([[np.nan], np.asarray(row[key]) / _log_v])

    def _one(cond: str, row: dict) -> str:
        series = [
            Series(raw=np.clip(_pad(row, "nll"), 0, 1), label="surprisal"),
            Series(raw=np.clip(_pad(row, "entropy"), 0, 1), label="entropy", dasharray="3 2"),
        ]
        svg = Subline(chars_per_line=len(row["text"]), css=_sub_css).plot(row["text"], series)
        label = f'<span style="font-size: 11px; font-family: monospace; opacity: 0.65">{cond}</span>'
        return figure_html(svg, caption=label, style="display: inline-block; margin: 0 1em 0 0")

    _html = figure_html(
        "".join(_one(cond, row) for cond, row in _rows_by_cond),
        aria_label="""
            The equation lime plus black equals green, repeated once per condition, each with a
            sparkline of per-character surprisal (solid) and predictive entropy (dashed) under
            the text on a shared 0-to-log-V scale.
        """,
    )
    mo.Html(externalize_html(_html, name="sublines-garden-path"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calibration

    A model can be wrong quietly or wrong loudly, and the anchored runs will lean
    on that difference as an early warning. Accuracy is saturated on most sets, so
    if a run starts to degrade, miscalibration should move before accuracy does.
    Ex-2.1.1 found that `named_holdout` answers were confidently wrong while
    everything else sat near calibrated. Here the same measure doubles as a check
    on H1, since a condition that solved the holdout set should also stop being
    surprised by it.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="calibration-heatmap",
        alt_text="""
            Heatmap of mean surprise-surprise (surprisal minus entropy over answer characters,
            as a fraction of log V) with eval sets as rows and conditions as columns, annotated
            with values. Warm cells mark confidently-wrong sets; near-zero cells are well
            calibrated.
        """,
        caption=r"""
            Mean surprise-surprise over answer characters, $s_2 = (i - h)/\log|V|$, with eval
            sets as rows and conditions as columns. Warm cells are confidently wrong (s₂ ≫ 0);
            near-zero cells are well calibrated.
        """,
    )
    def _plot() -> plt.Figure:
        s2 = np.array(
            [
                [np.mean([cell(metrics, cond, s)["calibration"][es]["s2"] for s in SEEDS]) for cond in CONDS]
                for es in EVAL_SETS
            ]
        )
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        lim = max(0.2, float(np.abs(s2).max()))
        im = ax.imshow(s2, cmap=light_dark("RdBu_r", "berlin"), vmin=-lim, vmax=lim, aspect="auto")
        for i in range(s2.shape[0]):
            for j in range(s2.shape[1]):
                low_cell = abs(s2[i, j]) > 0.6 * lim
                ax.text(
                    j,
                    i,
                    f"{s2[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=light_dark("#fff", "#000") if low_cell else light_dark("#000", "#fff"),
                )
        ax.set_xticks(range(len(CONDS)), CONDS)
        ax.set_yticks(range(len(EVAL_SETS)), [es.replace("_", " ") for es in EVAL_SETS], fontsize=8)
        fig.colorbar(im, ax=ax, label="mean s₂ on answers")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It behaves as expected. The `open_*` rows and the `alias_rev` row snap from
    confidently wrong (s₂ ≈ 0.7) to calibrated (≈ 0) under the conditions that train
    those forms, while `named_holdout` stays confidently wrong everywhere.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _oh = float(np.mean([acc(metrics, "both", s, "open_holdout") for s in SEEDS]))
    mo.md(
        f"""
    ## Findings

    Adding the inverse dictionary and the pressure to compute on named prompts
    did teach the model those two skills: reverse aliases reach 1.0, and
    off-palette generalization reaches ≈ {_oh:.2f}. What it did not teach was
    how to combine them: `named_holdout` stays at zero everywhere.

    The failure has two parts. First, the model often gets the value
    right but spells it as hex rather than as a name. Second, the value → name
    translation never runs mid-equation, even though the model has learned that
    mapping perfectly on its own. In a four-layer model this reads less like a
    gap in the data than a gap in the mechanism. Nothing in training ever asks
    the model to chain the two skills within a single forward pass, and the
    just-in-time answer schedule suggests it never holds a full intermediate
    result anywhere as a latent embedding.

    So for D2.1, `named_holdout` is not able to identify degradation: it sits
    at zero, so it has no headroom to lose. But the new corpus gives us a better
    one: `open_holdout` is genuinely compositional (unseen pairs, with the
    answer's form decided by a computed value), it sits near ceiling without
    touching it, and it comes with the graded instruments (margins, s₂, transfer
    probes) that this experiment validated end to end. The anchored runs should
    therefore train on the `both` corpus and treat `open_holdout` and
    calibration as the sensitive dials, with the saturated sets as the coarse
    ones.

    Keeping the `rev` half is worth it, though not because of `alias_rev`, which
    is just a memorizable lookup. It earns its place by writing the hex → name
    readout into the weights at all. That readout is the exact piece missing
    when `named_holdout` fails, so a later anchored run has something to recruit.
    On `open` alone, that direction never trains (`alias_rev` accuracy 0.00),
    and adding it costs nothing measurable elsewhere. Whether `named_holdout`
    itself can be made solvable at all (a denser named sub-grid, a curriculum,
    or simply more depth) is now a separate question.
    """
    )
    return


if __name__ == "__main__":
    app.run()
