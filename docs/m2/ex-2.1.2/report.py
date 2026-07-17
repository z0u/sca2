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
    from mini.vis import light_dark, themed
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
        m_art, g_art = store.get_ref(METRICS_REF), store.get_ref(MARGINS_REF)
        if m_art is None or g_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            metrics = json.loads(store.get(m_art, Path(d) / "metrics.json").read_text())
            with np.load(store.get(g_art, Path(d) / "margins.npz")) as z:
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

    [Ex-2.1.1](../ex-2.1.1/) left D2.1 with a healthy baseline and one hole:
    no model in the sweep ever solves `named_holdout` — named operand pairs
    whose named rendering is held out of training, so that answering requires
    *composing* the alias dictionary with the mixing arithmetic. The
    diagnosis (worked through in that report) was that the corpus never
    requires composition: the named slice is small enough to memorize, the alias
    dictionary is supervised in one direction only, and hex answers can be
    emitted digit-by-digit without ever holding the whole mix at one position.
    The failure is close, though. On `lime + black`, the model *computes* a
    correction toward the true answer, but the trained lookup still decides the
    output; the arithmetic runs underneath the retrieval without changing it.

    This experiment tests that diagnosis by intervening on the corpus — the
    architecture, split, and training recipe are ex-2.1.1's backbone
    (d64-L4), frozen. It is still an *un-anchored* experiment: the point is
    to hand the anchoring runs a baseline whose compositional eval set
    actually has headroom, and to build the graded measurements (margins,
    calibration, position-resolved probes) that anchoring side-effects will
    be read against.

    Two grammar interventions, in a 2 × 2 factorial:

    - **rev** adds reverse alias lines (`#f00 = red`): the hex → name
      readout, which the base grammar leaves untrained (its absence is a
      small-scale reversal curse), gets direct supervision.
    - **open** adds equations with named operands whose mix falls *off* the
      palette, answered in hex (`red + navy = #804`). The answer's surface
      form then depends on the mix's *value*, not just the operands' forms —
      so a name + name prompt can no longer be settled by lookup alone; the
      model has to compute the mix to know what kind of answer to give.

    Each intervention carves its token share out of the hex slice, which is
    far past saturation, so conditions differ in what they *add*, not in
    what they starve. Everything else — 40k lines, the ten held-out named
    pairs, seeds, LR — is identical across conditions and to ex-2.1.1.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What the model sees

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
    mo.vstack(
        [
            mo.md(f"```\n{_head}```"),
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table"><tr><th>condition</th>'
                + "".join(f'<th class="num"><code>{f}</code></th>' for f in _forms)
                + f"</tr>{_rows}</table></div>"
            ),
            mo.md(
                f"*Lines per form in each condition's {N_EXAMPLES:,}-line corpus. Named equations draw "
                f"from the same {len(_named_train)} train pairs as ex-2.1.1 (the same {len(_named_holdout)} "
                f"pairs held out); open equations draw from {len(_open_train)} of the 302 off-palette "
                f"named pairs, with {len(_open_holdout)} held out entirely.*"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Stated before looking at any results, with the diagnosis they follow from.

    **H1 — accuracy.** `named_holdout` stays at zero in `control` and is
    lifted well off zero by `both`. The single interventions are the
    diagnostic cells: `rev` alone lifts it *partially* — the garden path
    showed the mix is already half-computed on named prompts, and `rev`
    supplies the missing readout for whatever computation is there — while
    `open` alone forces the computation but leaves name emission supervised
    only through the memorizable named slice. The sharp prediction is the
    interaction: the legs are complementary, so the `both` effect exceeds
    the sum of the single-intervention effects.

    **H2 — margins.** Scoring all 27 names as complete answers gives each
    held-out pair a *margin*: log-probability of the true name minus the
    best competitor's. In `control` the margins are negative (accuracy is
    zero) but spread out well above the ignorance floor, and that spread is
    signal: pairs with the least-negative control margins are the first to
    flip positive under intervention. Interventions shift the whole margin
    distribution upward; `named_seen` margins stay large and positive.

    **H3 — the answer schedule.** Probing each answer position for each RGB
    channel (on hex prompts, strictly before each digit lands in the
    context) shows a stair-step: channel k becomes strongly decodable at
    the position that emits digit k, not before. This is the mechanism
    behind "hex answers factorize"; it should hold in every condition,
    because the hex task itself is unchanged.

    **H4 — computed but outvoted.** A result-color probe fit on open-pair
    prompts (name + name surface form) at the pre-answer position transfers
    to the held-out named prompts in proportion to how much the mix is
    actually computed there: middling R² in `control` (partial
    computation), rising toward the fit ceiling in the `open` conditions,
    and tracking `named_holdout` accuracy across conditions.

    **H5 — no collateral damage.** `named_seen`, `hex_unseen`, and
    `cross_unseen` stay at ex-2.1.1's saturated levels in every condition:
    displacing up to 15% of the hex slice costs nothing measurable. The
    `alias_rev` eval set reads ≈ 1 wherever reverse aliases are trained
    (it is memorizable by construction — a supervision check, not a
    generalization test) and stays garbage elsewhere, as in ex-2.1.1.

    If H1's interaction shows up but the single interventions do nothing,
    composition needed both legs at once; if `rev` alone saturates the set,
    the readout was the only missing piece and the lookup pressure never
    mattered; if nothing moves, the diagnosis was wrong somewhere upstream
    of the corpus. Any of these is informative for how hard D2.1's anchored
    runs should lean on `named_holdout` as a degradation canary.
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

    Exact-match accuracy per eval set and condition; dots are seeds, bars are
    means. `named_holdout` is H1's panel; `open_holdout` shows whether the
    forced computation generalizes to *unseen* off-palette pairs; `alias_rev`
    is the supervision check. (In `control` and `rev`, the open-form sets ask
    for a surface form those corpora never train on a name + name prompt, so
    low scores there read as "grammar not in evidence", not as failure.)
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="accuracy-factorial",
        alt_text=(
            "Seven bar-and-dot panels of completion accuracy (0 to 1) against condition (control, rev, "
            "open, both), one panel per eval set: named seen, named holdout, hex unseen, cross unseen, "
            "open seen, open holdout, and reverse alias. Bars show the mean over three seeds, dots the "
            "individual seeds."
        ),
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
    **H1 is refuted at face value: `named_holdout` stays at zero in every
    condition** (so its interaction term is a degenerate {_interaction:+.2f}).
    But the flanking panels show both interventions *trained*. The `open`
    conditions answer held-out off-palette pairs at ≈ 0.9 — the mixing
    arithmetic runs on name + name prompts and generalizes to operand pairs
    never seen in that surface form. And `alias_rev` reads 1.0 exactly where
    reverse aliases were trained: the hex → name readout exists and works in
    the frame it was taught in. Both legs of the diagnosis were supplied, and
    the composition still never clicks into a *named* answer. The next
    section shows where the failure actually moved.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Right value, wrong spelling

    Greedy completions of the held-out named prompts, straight from the
    published checkpoints. In `control` and `rev` the answers are the
    familiar retrieval-flavored wrong names. In the `open` conditions
    something new happens: for a good fraction of pairs the model emits a
    *hex* answer — and when it does, the value is the **correct mix**.
    The swatches make it visible: reading down an `open`/`both` column, the
    hex answers match the expected color exactly, while the name answers
    are still near-miss neighbors.
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
    mo.vstack(
        [
            mo.Html(f'<div class="report-table-scroll"><table class="report-table">{_head}{_rows}</table></div>'),
            mo.md(
                f"*Greedy completions of the `named_holdout` prompts, seed {SEEDS[0]}, one column per "
                "condition. A ✓ marks answers whose *value* equals the true mix (they are all hex: "
                "value-correct in the wrong surface form).*"
            ),
        ]
    )
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

    So the `open` intervention did what it was designed to do — the mix is computed on name + name
    prompts — but the *form rule* ("answer named exactly when the mix lands on the palette") did not
    generalize: the model treats held-out closed pairs like open ones and answers in hex, or falls
    back to the lookup-neighbor name. And `rev` moved nothing here: its perfect reverse-alias skill
    stays locked to the `#hex = ` frame it was trained in — supplying the inverse dictionary as a
    *surface task* does not make it available as an internal *readout* mid-equation. That is a
    sharper version of the reversal curse than the one diagnosed in ex-2.1.1: even indirect
    supervision of the inverse direction fails to transfer across frames.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Margins: how far from right, pair by pair

    Exact-match accuracy on ten pairs is a coarse instrument. The margin —
    log-probability of the true name as a complete answer, minus the best
    competitor among the *other names* — grades the name-identity question
    continuously, independent of the hex-vs-name form choice above: positive
    means the true name wins among names, and the magnitude says by how
    much. One line per held-out pair, mean over seeds, across conditions;
    `named_seen` margins shown as the shaded reference.
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
        alt_text=(
            "Line chart of the answer margin (log-probability of the true name minus the best competitor) "
            "against condition (control, rev, open, both), one line per held-out named pair, averaged over "
            "seeds. A horizontal line marks zero, where the true answer starts to win; a shaded band shows "
            "the range of margins on the seen named pairs."
        ),
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
    {_ctrl.mean():+.1f} nats in `control` and {_both.mean():+.1f} in `both`, with
    {_flipped}/10 pairs ending positive. The true name never comes close to
    winning among names — roughly ten nats adrift while `named_seen` sits far
    above zero — so the value → name translation isn't *almost there* and
    outvoted at the margin; on these prompts it simply never engages, even in
    the condition whose greedy answers prove the value has been computed. The
    rank structure is real but weak (rank correlation {_rho:.2f} between
    `control` and `both`), so the "least-wrong pairs flip first" reading has
    nothing to bite on: no pair flips at all.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The answer schedule: when each channel becomes readable

    Ex-2.1.1's probes read one pre-answer position and averaged over RGB —
    which hides *when* the mix is computed. Here, on hex prompts, a ridge
    probe per (position, channel, layer). Offset 0 is the `#`; digit k sits
    at offset k + 1 and is emitted from offset k. Solid segments are
    positions where the digit is not yet in the context (decoding is
    computation); dotted segments are after it lands (decoding is copying).
    H3 says each channel's solid segment ends with a sharp rise at its own
    emission position — a stair-step, not a plateau.

    That is what the deep layers show, and then some. At the final layer,
    channel k is near-perfectly decodable exactly at its emission offset
    (R² ≈ 0.97) and *only* there: one position earlier it is far weaker,
    and once emission moves on to the next digit the previous channels are
    not merely stale but largely *evicted* from the deep residual stream —
    each answer position holds the one channel it is about to emit, on top
    of a diffuse ≈ 0.5-R² trace of the whole mix that persists from the
    pre-answer position. The mix is never fully represented at any single
    position; the "result" the pre-answer probe sees is a head start, not
    a value. This is just-in-time computation, and it is exactly the
    regime in which anchoring a *result* concept at one position would be
    fighting the model's own schedule.
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
        alt_text=(
            "Line charts of probe R-squared against position offset around the answer, one panel per "
            "residual-stream depth, three lines per panel for the R, G, and B channels of the result. "
            "Lines are solid before each channel's digit enters the context and dotted after. In the "
            "deeper layers each channel peaks near 1 at its own emission position and falls away on "
            "either side, so the three channels form a sequence of staggered peaks rather than a "
            "cumulative plateau; at depth 0 the dotted segments jump to 1 as each digit becomes "
            "readable from the context."
        ),
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
    ## Computed but outvoted?

    The transfer probe: fit on open-pair prompts (name + name surface form)
    at the pre-answer position, scored on the named eval sets. If a
    condition computes the mix on named prompts, the probe carries over; if
    the mix is never represented there, no probe fit elsewhere can read it
    out. One panel per scored set, R² against depth, one line per condition.

    The reading: partial computation is present *everywhere*, control
    included — the held-out named prompts carry roughly as much
    linearly-decodable result as the fit set's own ceiling (≈ 0.6 at the
    deep layers), which quantifies ex-2.1.1's garden-path anecdote. The
    `open` conditions raise the mid-depth transfer somewhat, consistent
    with the arithmetic being made load-bearing on these prompts, and their
    *final*-layer R² drops on the named sets — the last layer's job there
    has become committing to a surface form. But the headline is that
    "computed but outvoted" was already true in `control`, and making the
    computation stronger (`open`) or the readout available (`rev`) still
    doesn't connect them.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    _sets = ["fit", "open_holdout", "named_seen", "named_holdout"]

    @themed(
        name="transfer-probe",
        alt_text=(
            "Four line charts of probe R-squared against residual-stream depth, one panel per prompt set: "
            "the fit set's held-back half, open holdout, named seen, and named holdout. One line per "
            "condition (control, rev, open, both; darker means richer corpus). The probes were fit on "
            "open-pair prompts at the pre-answer position."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_sets), figsize=(11.5, 3.0), sharey=True)
        shades = cond_shades()
        for ax, name in zip(axes, _sets, strict=True):
            for cond in CONDS:
                rows = np.array([cell(metrics, cond, s)["transfer_r2"][name] for s in SEEDS])
                ax.plot(rows.mean(axis=0), "o-", color=shades[cond], label=cond, lw=2)
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
    ## The garden path, revisited

    Ex-2.1.1's walkthrough example: `lime + black = green`, where the model
    mis-parses the second operand as *blue* until the `a` of `black` breaks
    the guess, then only half-corrects the answer — the trained
    `lime + blue = teal` keeps winning. The margin data lets us re-ask that
    question in every condition: does the correction finally overtake the
    lookup? It does not — the margin stays several nats negative in every
    condition. One incidental observation: the retrained `control` gives the
    same *kind* of wrong answer as ex-2.1.1's d64-L4-s0 but not always the
    same one (this seed now says *gray* rather than *teal* for this pair),
    a reminder that which neighbor wins is unstable run to run even at a
    fixed seed. Sublines below show per-character surprisal and entropy on
    the same example (teacher-forced), one row per condition.
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
    mo.vstack(
        [
            mo.Html(
                '<div class="report-table-scroll">'
                '<table class="report-table"><tr><th>condition</th>'
                + "".join(f'<th class="num">seed {s}</th>' for s in SEEDS)
                + f'<th class="num">mean</th></tr>{_rows}</table>'
                + "</div>"
            ),
            mo.md(
                "*Margin of `green` over its best competitor on the `lime + black` prompt, per seed "
                "and condition (positive: the arithmetic wins).*"
            ),
        ]
    )
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
        cap = f'<figcaption style="font-size: 11px; font-family: monospace; opacity: 0.65">{cond}</figcaption>'
        return f'<figure style="display: inline-block; margin: 0 1em 0 0">{svg}{cap}</figure>'

    _html = (
        '<figure role="img" aria-label="The equation lime plus black equals green, repeated once per '
        "condition, each with a sparkline of per-character surprisal (solid) and predictive entropy "
        '(dashed) under the text on a shared 0-to-log-V scale.">'
        + "".join(_one(cond, row) for cond, row in _rows_by_cond)
        + "</figure>"
    )
    mo.Html(externalize_html(_html, name="sublines-garden-path"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calibration as an early-warning dial

    Mean surprise-surprise over answer characters, $s_2 = (i - h)/\log|V|$,
    per eval set and condition. Ex-2.1.1 found `named_holdout` answers were
    *confidently wrong* (s₂ ≫ 0) while everything else sat near zero. The
    anchored runs will use this as a graded canary — accuracy is saturated
    on most sets, so miscalibration should move first. Here it doubles as a
    check on H1: conditions that solve the holdout set should also stop
    being surprised by it.

    It behaves exactly as designed: the `open_*` rows and the `alias_rev`
    row snap from confidently-wrong (s₂ ≈ 0.7) to calibrated (≈ 0) in
    precisely the conditions that train those forms, while `named_holdout`
    stays confidently wrong everywhere — matching its unmoved accuracy.
    The metric is ready to be the anchored runs' early-warning dial.
    """)
    return


@app.cell(hide_code=True)
def _(metrics):
    @themed(
        name="calibration-heatmap",
        alt_text=(
            "Heatmap of mean surprise-surprise (surprisal minus entropy over answer characters, as a "
            "fraction of log V) with eval sets as rows and conditions as columns, annotated with values. "
            "Warm cells mark confidently-wrong sets; near-zero cells are well calibrated."
        ),
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
def _(metrics):
    _oh = float(np.mean([acc(metrics, "both", s, "open_holdout") for s in SEEDS]))
    mo.md(
        f"""
    ## What this settles

    The diagnosis was half right, and the half that failed is the more
    interesting result. Supplying both missing ingredients — the inverse
    dictionary and pressure to compute on named prompts — produced exactly
    those two skills (reverse aliases at 1.0, off-palette generalization at
    ≈ {_oh:.2f}) without producing their *composition*: `named_holdout`
    stays at zero everywhere, with the failure now visibly split into a
    form-rule error (correct value, hex spelling) and a value → name
    translation that never engages mid-equation, even when the same mapping
    is perfectly learned in its own frame. In a four-layer model this looks
    less like a data gap than a mechanistic one: nothing in training ever
    requires chaining the two skills inside one forward pass, and the
    just-in-time answer schedule suggests the model has no habit of holding
    a full intermediate result anywhere a readout could find it.

    For D2.1 this changes the plan in a useful way. `named_holdout` is not
    a usable degradation canary — it has no headroom to lose — but the
    `both` corpus supplies a better one: `open_holdout` is compositional
    (unseen pairs, form decided by a computed value), sits near but not at
    ceiling, and comes with graded instruments (margins, s₂, transfer
    probes) that this experiment validated end to end. The anchored runs
    should therefore train on the `both` corpus and treat `open_holdout` +
    calibration as the sensitive dials, with the saturated sets as the
    coarse ones. Whether `named_holdout` itself can be made solvable (a
    denser named sub-grid, curricula, or simply more depth) is now a
    separate question from D2.1's, and stays parked in the todo list.
    """
    )
    return


if __name__ == "__main__":
    app.run()
