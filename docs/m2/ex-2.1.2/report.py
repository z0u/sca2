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
    # Ex 2.1.2: making composition pay

    [Ex-2.1.1](../ex-2.1.1/) left D2.1 with a healthy baseline and one hole:
    no model in the sweep ever solves `named_holdout` — named operand pairs
    whose named rendering is held out of training, so that answering requires
    *composing* the alias dictionary with the mixing arithmetic. The
    diagnosis (worked through in that report) was that the corpus never makes
    composition pay: the named slice is small enough to memorize, the alias
    dictionary is supervised in one direction only, and hex answers can be
    emitted digit-by-digit without ever holding the whole mix at one position.
    The failure is close, though. On `lime + black`, the model *computes* a
    correction toward the true answer but the trained lookup wins anyway; the
    arithmetic is running, and losing, underneath the retrieval.

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
        + "".join(f"<td>{_counts[cond].get(f, 0) or '—'}</td>" for f in _forms)
        + "</tr>"
        for cond in CONDS
    )
    _open_train, _open_holdout = colors.split_open_pairs(CORPUS_SEED, OPEN_HOLDOUT_FRAC)
    _named_train, _named_holdout = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
    mo.vstack(
        [
            mo.md(f"```\n{_head}```"),
            mo.Html(
                '<table style="font-size: 0.9em"><tr><th>condition</th>'
                + "".join(f"<th><code>{f}</code></th>" for f in _forms)
                + f"</tr>{_rows}</table>"
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
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _a = {cond: float(np.mean([acc(metrics, cond, s, "named_holdout") for s in SEEDS])) for cond in CONDS}
    _interaction = _a["both"] - _a["rev"] - _a["open"] + _a["control"]
    mo.md(
        f"H1's interaction term — both − rev − open + control on `named_holdout` — comes to "
        f"**{_interaction:+.2f}** (super-additive if positive)."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Margins: how far from right, pair by pair

    Exact-match accuracy on ten pairs is a coarse instrument. The margin —
    log-probability of the true name as a complete answer, minus the best
    competitor among the other names and the mix's hex rendering — grades
    each pair continuously: positive means the true answer wins, and the
    magnitude says by how much. One line per held-out pair, mean over seeds,
    across conditions; `named_seen` margins shown as the shaded reference.
    """)
    return


@app.cell(hide_code=True)
def _(margins, metrics):
    def margin_rows(eval_set: str, exs: list) -> np.ndarray:
        """(len(CONDS), n_seeds, n_examples) margin of the true answer over the best competitor."""
        out = np.empty((len(CONDS), len(SEEDS), len(exs)))
        for ci, cond in enumerate(CONDS):
            for si, s in enumerate(SEEDS):
                cands = cell(metrics, cond, s)["margin_candidates"][eval_set]
                lp = margins[f"{label(cond, s)}/margins/{eval_set}"]
                true_idx = np.array([cands.index(ex.answer) for ex in exs])
                truth = lp[np.arange(len(exs)), true_idx]
                rival = lp.copy()
                rival[np.arange(len(exs)), true_idx] = -np.inf
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
        for rank, p in enumerate(order):
            ex = holdout_exs[p]
            ax.plot(xs, mean[:, p], "o-", color=cmap[rank], lw=1.4, ms=3.5)
            ax.annotate(
                f"{ex.prompt}{ex.answer}",
                (xs[-1], mean[-1, p]),
                xytext=(6, 0),
                textcoords="offset points",
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
    _ctrl, _both = m_hold.mean(axis=1)[0], m_hold.mean(axis=1)[CONDS.index("both")]
    _rho = float(np.corrcoef(np.argsort(np.argsort(_ctrl)), np.argsort(np.argsort(_both)))[0, 1])
    _flipped = int((_both > 0).sum())
    mo.md(
        f"H2's prediction check: the rank correlation between control margins and `both` margins across "
        f"the ten pairs is **{_rho:.2f}**; {_flipped}/10 pairs end positive under `both`."
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
            "deeper layers each channel rises sharply at its own emission position, forming a stair-step."
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
        fig.tight_layout()
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
        fig.tight_layout()
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
    lookup? Sublines below show per-character surprisal and entropy on the
    same example (teacher-forced), one row per condition.
    """)
    return


@app.cell(hide_code=True)
def _(holdout_exs, m_hold):
    gp_idx = next(
        i for i, ex in enumerate(holdout_exs) if {"lime", "black"} == set(ex.prompt.split(" = ")[0].split(" + "))
    )
    _rows = "".join(
        f"<tr><td><code>{cond}</code></td>"
        + "".join(f"<td>{m_hold[ci, si, gp_idx]:+.1f}</td>" for si in range(len(SEEDS)))
        + f"<td><b>{m_hold[ci, :, gp_idx].mean():+.1f}</b></td></tr>"
        for ci, cond in enumerate(CONDS)
    )
    mo.vstack(
        [
            mo.Html(
                '<table style="font-size: 0.9em"><tr><th>condition</th>'
                + "".join(f"<th>seed {s}</th>" for s in SEEDS)
                + f"<th>mean</th></tr>{_rows}</table>"
            ),
            mo.md(
                "*Margin of `green` over its best competitor on the `lime + black` prompt, per seed "
                "and condition (positive: the arithmetic wins).*"
            ),
        ]
    )
    return (gp_idx,)


@app.cell(hide_code=True)
def _(holdout_exs, gp_idx, metrics):
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
        '<div role="img" aria-label="The equation lime plus black equals green, repeated once per '
        "condition, each with a sparkline of per-character surprisal (solid) and predictive entropy "
        '(dashed) under the text on a shared 0-to-log-V scale." style="text-wrap: balance">'
        + "".join(_one(cond, row) for cond, row in _rows_by_cond)
        + "</div>"
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
        im = ax.imshow(s2, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(s2.shape[0]):
            for j in range(s2.shape[1]):
                dark_cell = abs(s2[i, j]) > 0.6 * lim
                ax.text(
                    j,
                    i,
                    f"{s2[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#fff" if dark_cell else "#000",
                )
        ax.set_xticks(range(len(CONDS)), CONDS)
        ax.set_yticks(range(len(EVAL_SETS)), [es.replace("_", " ") for es in EVAL_SETS], fontsize=8)
        fig.colorbar(im, ax=ax, label="mean s₂ on answers")
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(metrics):
    _hold = {cond: float(np.mean([acc(metrics, cond, s, "named_holdout") for s in SEEDS])) for cond in CONDS}
    _best = max(_hold, key=lambda c: _hold[c])
    mo.md(
        f"""
    ## What this settles

    The condition that best solves `named_holdout` while keeping the
    saturated sets intact — **`{_best}`** on current numbers — becomes the
    corpus for D2.1's anchored runs, giving them a compositional eval set
    with real headroom as a degradation canary. The margin, calibration,
    and position-resolved probe measurements built here carry over as the
    graded instruments those runs will be read with: anchoring should leave
    all of them where this experiment puts them, and the redness direction
    should land where we choose instead of where the seed happens to put it.
    """
    )
    return


if __name__ == "__main__":
    app.run()
