import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.11: Ablations and a survey for the D2.1 operating point",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path
    from typing import cast

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo puts the notebook directory on sys.path, so the design constants come
    # from the experiment module rather than a copy of them in the prose.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import AxesRow, figure_html, light_dark, themed

    use_publisher(report_bundle(__file__))

    def load_results() -> tuple[dict, dict[str, np.ndarray]] | None:
        """Resolve this experiment's metrics and arrays from the store, or None if unpublished."""
        store = project_store()
        arts = store.get_refs([ex.METRICS_REF, ex.ARRAYS_REF])
        m_art, a_art = arts[ex.METRICS_REF], arts[ex.ARRAYS_REF]
        if m_art is None or a_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            m_path, a_path = store.get_many([(m_art, Path(d) / "metrics.json"), (a_art, Path(d) / "arrays.npz")])
            with np.load(a_path) as z:
                arrays = {k: z[k] for k in z.files}
            metrics = json.loads(m_path.read_text())
        return metrics, arrays

    def load_prior() -> dict | None:
        """The published inputs to the calibration stage, or None if any is unreachable.

        Ex-2.1.10's metrics, arrays and probe tables (the nine-seed floors, the
        trade, the τ map), ex-2.1.9's metrics (the latch calibration), and
        ex-2.1.8's metrics (the across-point correlation).
        """
        store = project_store()
        refs = [
            ex.EX2110_METRICS_REF,
            ex.EX2110_ARRAYS_REF,
            ex.EX2110_PROBES_REF,
            ex.EX219_METRICS_REF,
            ex.EX218_METRICS_REF,
        ]
        arts = store.get_refs(refs)
        if any(arts[r] is None for r in refs):
            return None
        with tempfile.TemporaryDirectory() as d:
            paths = store.get_many([(arts[r], Path(d) / f"{i}.bin") for i, r in enumerate(refs)])
            m10 = json.loads(paths[0].read_text())
            with np.load(paths[1]) as z:
                a10 = {k: z[k] for k in z.files if "/alpha" in k or "/w_line" in k}
            with np.load(paths[2]) as z:
                probes = {k: z[k] for k in z.files}
            m9 = json.loads(paths[3].read_text())
            m8 = json.loads(paths[4].read_text())
        return {"m10": m10, "a10": a10, "probes": probes, "m9": m9, "m8": m8}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.11: Ablations and a survey for the D2.1 operating point

    /// tip |
    <!-- tl;dr -->
    A survey to close out D2.1. The recipe's schedules and weights were
    inherited piece by piece and never tuned together, so first we ablate:
    flat arms bracketing each schedule, a half-length condition. Each "no
    difference" deletes a dimension. Then a Sobol search over the survivors —
    anchor weight, pooling temperature, repulsion dose — maps the good region
    and proposes the operating point the next milestone inherits.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Observations

    *Empty until results land. Each line will carry its noise floor the way a
    verdict carries its gate, and one line will name the proposed operating
    point.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a **survey**, not a hypothesis-scoring experiment: it
    preregisters a *search plan* — the space, the sampling rule, the
    objective and its constraints, the seed budget, and the decision rules
    that stage the work — and scores nothing. The plan below is frozen once
    agreed (immaterial edits aside); results replace the `TODO` placeholders
    in place, and anything conceived after seeing the data goes under
    *Exploratory analyses*, marked as post hoc.

    Nothing this report finds may be quoted as a result. It proposes an
    operating point; the next preregistered experiment adopts that point,
    re-measures it at fresh seeds, and reports the survey's value beside the
    confirmed one. The gap between them is the winner's-curse correction.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every knob in the D2.1 recipe arrived with a lineage rather than a
    justification. The anchor schedule (warmup, hold, end-anneal to a floor)
    and the anti-subspace schedule (2.5 → 0.3 of the anchor weight) are M1's
    dopesheets, mapped onto 100 epochs by fraction of training — keyframes
    tuned for a 5-d autoencoder bottleneck, applied to a 64-d residual
    stream. The anchor weight λ = 0.1 is ex-2.1.6's scoring rung, chosen to
    be safely inside the task-free region, and ex-2.1.7 showed selectivity
    falls by λ = 1 without finding where. τ = 0.1 is the one rung of
    ex-2.1.9's coarse ladder that stayed graded in every seed, and its ×5
    neighbors lose the group contrast and the grading in one step — so
    whether the primary sits on a plateau or a knife edge is unmeasured.
    D2.2 fixes an operating point and builds interventions on it; this
    experiment is where that point stops being a stack of inheritances.

    The order of work is ablate first, then search, because deleting a
    dimension is far cheaper than searching it. A schedule that a constant
    replaces is four knobs gone; a testbed that converges in 50 epochs
    halves every later trial. The ablations are preregistered conditions
    with frozen decision rules; the search is a Sobol map over whatever
    survives them. Before either, a calibration stage — free, computed from
    published results — fixes the noise floors that decide what "no
    difference" means, and settles the shape of the objective.

    Scope: the position-dropout mechanism from the ex-2.1.9 review is not
    part of the recipe and is not surveyed; the grammar, corpus, labeller
    and architecture stay fixed at ex-2.1.10's primary throughout. The
    survey tunes the recipe we have, on the testbed we have.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | Glossary
    - **trial** — one sampled point in the survey's search space,
      seed-aggregated. Sampled, so unlike a *condition* its levels weren't
      chosen by hand.
    - **round** — one pass over trials at a given seed budget: round 1 is
      every trial at one seed, round 2 the promoted trials at five.
    - **feasible** — a trial satisfying every constraint (task gate,
      containment, retention, grading, contrast, no latch). Only feasible
      trials are ranked.
    - **dose** — a schedule's delivered weight, ∫ w(e)·lr(e) de over
      training; the LR is in the integrand because it multiplies every
      term's gradient. The **centroid** is the dose-weighted mean epoch:
      *when* the dose arrives.
    - **band** — the resolution limit on a seed-mean comparison,
      2σ·√(1/n₁ + 1/n₂) from the nine-seed per-run σ. Differences inside
      the band are unresolved, which is a statement about resolution, never
      evidence of equality.
    - **latch** — a run whose pooled pull commits its deep-slice weight to
      a syntax role (`+` or `=`); the winner-take-all failure ex-2.1.9
      found at small τ.
    - **m_line** — per-line selectivity: label-weighted minus unweighted
      mean alignment, best span role, mean over slices (ex-2.1.10's scored
      statistic). **ᾱ** at op1 is containment; **retention** is the end of
      the m_line trajectory over its running peak; **r²** is the squared
      Pearson correlation of the per-color op1 response with the sim^1.5
      target (grading); **contrast** is the between-label-group difference
      in op2 softmin weight (per-line localization).
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    The testbed is ex-2.1.10's primary, unchanged: the word-level `v216`
    corpus (216 colors, one token each, d64-L4), the either-slot labeller
    (each operand draws at redness⁸ · 0.04; a line is labeled when either
    draws), the mellowmax-pooled anchor term over the four span roles, and
    the anti-subspace term. The reference recipe is that experiment's
    operating point: λ = 0.1, τ = 0.1, anchor schedule warmup 0–10%,
    anneal 90–100% to a 0.1 floor, anti-subspace 2.5 → 0.30 (μ/λ) by 90%.
    Schedule keyframes are expressed as fractions of training so the
    half-length condition runs the same shape.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Measurements

    Every run — ablation or trial — records the same slim set, all defined
    in earlier experiments: the three behavior sets (holdout exact match is
    the task gate), the alignment map on the probe lines (m_line, ᾱ at op1,
    r² against the sim^1.5 target), the softmin weight profiles (the group
    contrast and the latch detector), and the training trajectory of m_line
    (retention). The per-run cost is one eval pass; the readability,
    geometry and leakage passes that a scoring experiment carries are *not*
    run per trial — they belong to the confirming experiment, which measures
    one point properly rather than eighty points thinly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Search plan

    Three stages. Calibration is already run (it reads only published
    results, and its numbers are frozen into the design constants). The
    ablation stage runs eight conditions and applies the decision rules
    below, which fix the survey's recipe, length, and space. The survey
    stage then samples its trials from a rule that is deterministic given
    this document. The only branch points in the whole plan are the decision
    rules; everything downstream of the data is fixed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration: noise floors

    Ex-2.1.10 ran nine seeds on its primary — the reference recipe here —
    so the per-run spread of every statistic at the operating point is
    already in the store. The table below re-derives the floors from the
    published arrays; the design constants freeze the same numbers
    (`NOISE_RUN`), and the ablation bands follow as 2σ·√(2/3) for a
    three-seed against three-seed comparison. The margin statistics are
    tight relative to the effects worth chasing (the ex-2.1.10 τ ladder
    spans several sd on m_line), which is what licenses single-seed survey
    trials. Containment is the opposite — its sd is ~40% of its own mean —
    so it enters the survey as a constraint, not a ranking.

    One caveat travels with these floors: they are measured in the graded
    regime (τ = 0.1). At τ ≲ 0.03 the pull latches per run, the statistics
    go bimodal, and no per-run sd describes them — which is why the survey
    bounds τ above that regime and carries a per-run latch veto instead.
    """)
    return


@app.cell(hide_code=True)
def _():
    _prior = load_prior()
    mo.stop(
        _prior is None, mo.md("_The published inputs aren't reachable from here; the calibration cells need them._")
    )
    assert _prior is not None
    prior: dict = _prior
    return (prior,)


@app.cell(hide_code=True)
def _(group_weights, prior):
    # Per-run statistics of the nine-seed primary, re-derived from the arrays.
    _m10: dict = prior["m10"]
    _primary = [c for c in _m10["cells"] if c["condition"] == "either-t100"]
    _probes = prior["probes"]
    _g1, _g2 = group_weights(_probes["r1"], _probes["r2"])

    def _retention(cell: dict) -> float:
        tr = np.asarray(cell["traj"]["m_line"], dtype=float)
        return float(tr[-1] / np.maximum.accumulate(tr).max())

    def _contrast(label: str) -> float:
        w = prior["a10"][f"{label}/w_line"]  # (L1, N, SPAN)
        return float(np.einsum("lnt,n->lt", w, _g2)[1:, 2].mean() - np.einsum("lnt,n->lt", w, _g1)[1:, 2].mean())

    def _r2(label: str) -> float:
        a = prior["a10"][f"{label}/alpha"][:, :, 0].mean(axis=0)  # per-color op1 response
        return float(np.corrcoef(a, ex.SIM_TARGET)[0, 1] ** 2)

    per_run: dict[str, np.ndarray] = {
        "m_line": np.array([c["m_line"] for c in _primary]),
        "alpha_op1": np.array([c["alpha_mean_op1"] for c in _primary]),
        "retention": np.array([_retention(c) for c in _primary]),
        "r2_sim": np.array([_r2(c["label"]) for c in _primary]),
        "contrast": np.array([_contrast(c["label"]) for c in _primary]),
        "holdout_em": np.array([c["sets"]["named_holdout"]["accuracy"] for c in _primary]),
    }
    return (per_run,)


@app.cell(hide_code=True)
def _(per_run):
    _rows = "".join(
        f"<tr><td><code>{k}</code></td>"
        f"<td class='num'>{v.mean():+.4f}</td>"
        f"<td class='num'>{v.std(ddof=1):.4f}</td>"
        f"<td class='num'>{ex.NOISE_RUN[k]:.4f}</td>"
        f"<td class='num'>{ex.equiv_band(k):.4f}</td></tr>"
        for k, v in per_run.items()
    )
    _table = f"""
    <table class="report-table">
    <thead><tr><th>statistic</th><th class="num">mean</th><th class="num">per-run σ</th>
    <th class="num">frozen σ</th><th class="num">band (3 v 3)</th></tr></thead>
    <tbody>{_rows}</tbody>
    </table>
    """
    mo.Html(
        figure_html(
            _table,
            caption="""
            **Noise floors at the operating point.** Per-run mean and sd over
            ex-2.1.10's nine-seed primary, re-derived from its published
            arrays; the *frozen σ* column is the value the design constants
            carry, and the *band* is the smallest seed-mean difference the
            ablation stage may call a difference (2σ·√(2/3), three seeds a
            side).
            """,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration: the shape of the objective

    A search needs to know whether its objectives trade off — a real trade
    needs a Pareto front, independent objectives can be constraints around a
    scalar. Across the ten anchored operating points already in the store
    (the ex-2.1.8 grid, the ex-2.1.9 ladder, the ex-2.1.10 arms), margin and
    containment *don't* trade: their correlation is weakly negative, i.e.
    the better operating points improved both together. The trade that does
    bind is margin against grading, and it runs along the τ axis: softening
    τ raises m_line while the per-color grading and the group contrast
    collapse — selectivity smearing into "everything red-ish moves". The
    figure below shows that structure.

    So the objective is scalar with constraints: **maximize m_line, subject
    to** the task gate, containment, retention, grading, contrast, and the
    latch veto (values under *Objective and constraints* below). A
    margin-only search would walk to the smeared end of the τ range; the
    grading and contrast constraints are what hold it out. The m_line ↔ r²
    plane is still reported for every trial, so if the constraint boundary
    turns out to be where the interesting points live, the map will show it.
    """)
    return


@app.cell(hide_code=True)
def _(per_run, prior):
    _m10: dict = prior["m10"]
    _conds = ["op1-labels", "either-t100", "either-t500", "either-t2500", "slot-oracle"]
    _pts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _c in _conds:
        _cells = [c for c in _m10["cells"] if c["condition"] == _c]
        _m = np.array([c["m_line"] for c in _cells])
        _r = np.array(
            [
                float(np.corrcoef(prior["a10"][f"{c['label']}/alpha"][:, :, 0].mean(axis=0), ex.SIM_TARGET)[0, 1] ** 2)
                for c in _cells
            ]
        )
        _pts[_c] = (_m, _r)

    @themed(
        name="trade",
        alt_text="""
            Scatter chart of grading r squared against per-line margin
            m_line. Five clusters of small dots, one per ex-2.1.10
            condition, each with a larger ringed mark at its seed mean. The
            reference arm, the primary and the slot oracle cluster at the
            upper left, near m_line 0.4 and r squared around 0.8. The two
            soft-tau arms sit to the lower right, near m_line 0.49 and r
            squared 0.65 and 0.58: margin rises as grading falls along the
            tau axis.
        """,
        caption=r"""
            **The trade the constraints must hold.** Per-run grading $r^2$
            against per-run m_line for ex-2.1.10's five anchored conditions
            (small marks; large ring = seed mean). Softening τ (0.1 → 0.5 →
            2.5, left to right) buys margin and spends grading; ranking on
            m_line alone would pick the right-hand end. The survey caps that
            walk with the grading and contrast constraints instead of a
            second objective.
        """,
    )
    def _plot():
        grey = light_dark("#666", "#999")
        colors = {
            "op1-labels": light_dark("#7b3fa0", "#b98ce0"),
            "either-t100": light_dark("#1f6fb4", "#5fa8dd"),
            "either-t500": light_dark("#2a9d8f", "#5fc9bd"),
            "either-t2500": light_dark("#b4531f", "#dd8f5f"),
            "slot-oracle": light_dark("#a02040", "#d06080"),
        }
        fig, ax = plt.subplots(figsize=(5.4, 3.2), layout="constrained")
        for c, (m, r) in _pts.items():
            ax.scatter(m, r, s=14, color=colors[c], alpha=0.7, lw=0)
            ax.scatter(m.mean(), r.mean(), s=64, facecolors="none", edgecolors=colors[c], lw=1.6, label=c)
        ax.set_xlabel("m_line (per run)", fontsize="x-small")
        ax.set_ylabel("grading r² (per run)", fontsize="x-small")
        ax.legend(fontsize="x-small", frameon=False, loc="lower left")
        ax.set_title("margin buys smear at soft τ", fontsize="small", color=grey)
        return fig

    _corr_note = mo.md(rf"""
    Margin–containment correlation across the ten stored operating points:
    $r$ = −0.19 (no trade; both improve together along the design line).
    For scale, the primary's nine runs put per-run grading at
    {per_run["r2_sim"].mean():.2f} ± {per_run["r2_sim"].std(ddof=1):.3f}
    against {_pts["either-t2500"][1].mean():.2f} for the softest arm.
    """)
    mo.vstack([mo.Html(_plot()), _corr_note])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration: the τ range

    τ only means anything through the weights it induces, so the survey's τ
    bounds are set on the weight scale: the share of the pooled pull held by
    the leading span position, computed from the primary's stored alignment
    profiles. The bounds [0.05, 0.30] keep the leading weight between ~0.74
    and ~0.42 on the trained profile — the 0.6–0.8 target band from
    ex-2.1.9 sits inside — while the latching regime (τ ≲ 0.03, leading
    weight ≳ 0.82 and one-hot per run at depth) stays outside. The upper
    bound stops short of τ = 0.5, which the trade figure above shows is
    already into the smeared regime.
    """)
    return


@app.cell(hide_code=True)
def _(prior, softmin_weights_np):
    _m10: dict = prior["m10"]
    _labels = [c["label"] for c in _m10["cells"] if c["condition"] == "either-t100"]
    _alpha = np.mean([prior["a10"][f"{la}/alpha_lines"] for la in _labels], axis=0)  # (L1, N, T), seed mean
    _line_p = prior["probes"]["line_p"]
    _line_w = _line_p / _line_p.sum()
    _x = 1.0 - _alpha[:, :, : ex.SPAN]
    _taus = np.geomspace(0.01, 1.0, 60)
    _lead = np.array([np.einsum("ln,n->l", softmin_weights_np(_x, t).max(axis=-1), _line_w) for t in _taus])  # (τ, L1)

    @themed(
        name="tau-range",
        alt_text="""
            Line chart of leading softmin weight against temperature tau on
            a log scale from 0.01 to 1. Five curves, one per residual slice,
            fall from near 1 at the left to about 0.3 at the right. A shaded
            vertical band marks the survey range 0.05 to 0.3, and a shaded
            horizontal band marks the target leading-weight range 0.6 to
            0.8. The two bands overlap around tau 0.05 to 0.1. A dashed
            vertical line marks the reference tau of 0.1.
        """,
        caption=r"""
            **Where the survey's τ bounds sit on the weight scale.** The
            label-weighted share of softmin weight held by the leading span
            position, per residual slice, on the primary's seed-mean
            alignment profile. Vertical shading: the survey range
            [0.05, 0.30]; horizontal shading: ex-2.1.9's 0.6–0.8 target
            band; dashed line: the reference τ = 0.1.
        """,
    )
    def _plot():
        grey = light_dark("#666", "#999")
        slice_colors = [
            light_dark("#7b3fa0", "#b98ce0"),
            light_dark("#1f6fb4", "#5fa8dd"),
            light_dark("#2a9d8f", "#5fc9bd"),
            light_dark("#b4531f", "#dd8f5f"),
            light_dark("#a02040", "#d06080"),
        ]
        fig, ax = plt.subplots(figsize=(5.4, 2.9), layout="constrained")
        ax.axvspan(*ex.SPACE_COMMON["tau"][:2], color=light_dark("#00000010", "#ffffff14"), zorder=-10)
        ax.axhspan(0.6, 0.8, color=light_dark("#00000010", "#ffffff14"), zorder=-10)
        for li in range(_lead.shape[1]):
            ax.plot(_taus, _lead[:, li], color=slice_colors[li], lw=1.4, label=f"slice {li}")
        ax.axvline(ex.TAU_REF, color=grey, ls=(0, (4, 3)), lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("τ", fontsize="x-small")
        ax.set_ylabel("leading weight", fontsize="x-small")
        ax.legend(fontsize="x-small", frameon=False, loc="upper right")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration: doses, and why the flat arms are shaped as they are

    The anchor term's schedule can be ablated with a single flat arm because
    flat-at-peak and dose-matched-flat coincide: the schedule delivers 97%
    of the flat-at-peak dose, a 3% gap well inside the noise floor, so no
    matching invariant has to be chosen. What the flat arm does change is
    *timing* — the ramp carries 7.3% of the dose across the same window in
    which ex-2.1.9 found the pooled pull commits (by epoch 8, inside the
    10-epoch warmup), and the `AnchorSpec` docstring says the ramp exists so
    the anchor arrives with the optimizer rather than ahead of it. So the
    prediction for `flat-anchor` is not a dose effect: full λ against a
    still-random model risks latching, and the latch veto plus the
    deep-slice weight profile is where the reading will be.

    The anti-subspace term cannot be dose-matched flat at all: its schedule
    spans an 83× weight range, and the constant matching its dose would sit
    at 1.7× the anchor peak for all of training — a different regime, the
    trap ex-2.1.8's dose arm documented. So two arms bracket it instead:
    flat at the hold ratio (0.17× the schedule's dose) and flat at the peak
    ratio (1.41×). The schedule's effect is gated on the peak-matched arm;
    the dose alignment is read as a stated secondary, with every arm's
    delivered dose and centroid reported in the conditions table below.
    """)
    return


@app.cell(hide_code=True)
def _():
    _e = np.linspace(0.0, float(ex.EPOCHS), 2001)
    _lr = ex.learning_rate(_e)
    _anchor_sched = ex.anchor_weight(_e) * _lr
    _anchor_flat = ex.SCORING_LAMBDA * _lr
    _anti_sched = ex.anti_weight(_e) * _lr
    _anti_hold = ex.ANTI_HOLD_RATIO * ex.SCORING_LAMBDA * _lr
    _anti_peak = ex.ANTI_PEAK_RATIO * ex.SCORING_LAMBDA * _lr

    @themed(
        name="doses",
        alt_text="""
            Two line charts of effective weight, schedule times learning
            rate, against epoch from 0 to 100. Left panel, the anchor term:
            the scheduled curve and the flat-at-peak curve lie nearly on top
            of each other except during the first ten epochs, where the flat
            curve is higher. Right panel, the anti-subspace term: the
            scheduled curve starts high and decays to a low hold level by
            epoch 90, the flat-at-hold curve stays low throughout, and the
            flat-at-peak curve stays high throughout, so the two flat curves
            bracket the scheduled one.
        """,
        caption=r"""
            **What each arm delivers, when.** Effective weight (schedule ×
            learning rate) over training. **Left:** the anchor term — the
            schedule and `flat-anchor` differ visibly only in the warmup
            window, and their doses agree to 3%. **Right:** the
            anti-subspace term — `anti-hold` and `anti-peak` bracket the
            schedule from below (0.17× its dose) and above (1.41×); no
            constant matches it without leaving the regime.
        """,
    )
    def _plot():
        grey = light_dark("#666", "#999")
        sched_c = light_dark("#1f6fb4", "#5fa8dd")
        flat_c = light_dark("#b4531f", "#dd8f5f")
        flat2_c = light_dark("#2a9d8f", "#5fc9bd")
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained", sharex=True)
        axs = cast(AxesRow, axs)
        axs[0].plot(_e, _anchor_sched, color=sched_c, lw=1.4, label="schedule")
        axs[0].plot(_e, _anchor_flat, color=flat_c, lw=1.4, ls=(0, (4, 2)), label="flat-anchor")
        axs[0].set_title("anchor term", fontsize="small", color=grey)
        axs[0].set_ylabel("λ(e) · lr(e)", fontsize="x-small")
        axs[1].plot(_e, _anti_sched, color=sched_c, lw=1.4, label="schedule")
        axs[1].plot(_e, _anti_hold, color=flat2_c, lw=1.4, ls=(0, (4, 2)), label="anti-hold")
        axs[1].plot(_e, _anti_peak, color=flat_c, lw=1.4, ls=(0, (1, 2)), label="anti-peak")
        axs[1].set_title("anti-subspace term", fontsize="small", color=grey)
        axs[1].set_ylabel("μ(e) · lr(e)", fontsize="x-small")
        for ax in axs:
            ax.set_xlabel("epoch", fontsize="x-small")
            ax.legend(fontsize="x-small", frameon=False)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(prior):
    _m9: dict = prior["m9"]
    _rows = []
    for _c in _m9["cells"]:
        if _c["condition"] in ("lam0",) or _c.get("tau") is None:
            continue
        _pi = np.asarray(_c["pi"])  # (L1, SPAN)
        _deep = _pi[1:].mean(axis=0)
        _syntax = float(max(_deep[1], _deep[3]))
        _rows.append((_c["label"], _deep, _syntax))

    _body = "".join(
        f"<tr><td><code>{la}</code></td>"
        + "".join(f"<td class='num'>{v:.2f}</td>" for v in deep)
        + f"<td class='num'>{'<b>' if syn > ex.LATCH_PI else ''}{syn:.2f}{'</b>' if syn > ex.LATCH_PI else ''}</td></tr>"
        for la, deep, syn in _rows
    )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
    <thead><tr><th>run</th><th class="num">op1</th><th class="num">+</th><th class="num">op2</th>
    <th class="num">=</th><th class="num">max syntax ↓</th></tr></thead>
    <tbody>{_body}</tbody>
    </table></div>
    """
    mo.Html(
        figure_html(
            _table,
            caption=f"""
            **The latch detector, calibrated on ex-2.1.9.** Deep-slice
            (post-attention) mean softmin weight per span role, per stored
            run of the pooled conditions. Latched runs put ≥ 0.85 on `+`;
            healthy runs ≤ 0.03 on any syntax role (bold = over the
            {ex.LATCH_PI:g} veto). The threshold sits in a wide gap, so the
            veto is insensitive to its exact value.
            """,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The ablation conditions

    Eight conditions, three seeds each, all at the reference λ and τ under
    the either-slot labeller. `ref` re-runs ex-2.1.10's primary recipe
    through this experiment's code path, so every contrast is between runs
    of identical code; `lam0` and `short-lam0` are the task-cost controls at
    each length. The dose and centroid columns report each arm's matching
    invariants rather than matching on one, so the outcomes can say which
    invariant was the active one.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _invariants(c: dict) -> tuple[float, float, float, float]:
        epochs: int = c["epochs"]
        if c.get("anchor_shape") == "flat":
            anchor = lambda e: c["lam"] * np.ones_like(np.asarray(e, dtype=float))  # noqa: E731
        else:
            anchor = lambda e: ex.anchor_weight(e, peak=c["lam"], epochs=epochs)  # noqa: E731
        if c.get("anti_shape") == "flat":
            anti = lambda e: c["anti_ratio"] * c["lam"] * np.ones_like(np.asarray(e, dtype=float))  # noqa: E731
        else:
            anti = lambda e: ex.anti_weight(e, lam=c["lam"], epochs=epochs)  # noqa: E731
        return (
            ex.anchor_dose(anchor, epochs=epochs),
            ex.dose_centroid(anchor, epochs=epochs) if c["lam"] > 0 else float("nan"),
            ex.anchor_dose(anti, epochs=epochs),
            ex.dose_centroid(anti, epochs=epochs) if c["lam"] > 0 else float("nan"),
        )

    def _fmt(v: float, spec: str) -> str:
        return "—" if np.isnan(v) else format(v, spec)

    _rows = []
    for _c in ex.ABLATION_CONDITIONS:
        _ad, _ac, _sd, _sc = _invariants(_c)
        _what = {
            "lam0": "un-anchored control",
            "ref": "the ex-2.1.10 primary recipe, re-run",
            "flat-anchor": "λ constant from step 0; no ramp, anneal or floor",
            "anti-hold": "μ/λ constant at the hold ratio (0.30)",
            "anti-peak": "μ/λ constant at the peak ratio (2.5)",
            "linear": "every minimum-jerk ramp/anneal made linear",
            "short": "the whole recipe at 50 epochs, keyframes scaled",
            "short-lam0": "50-epoch task-cost control",
        }[_c["name"]]
        _rows.append(
            f"<tr><td><code>{_c['name']}</code></td><td>{_what}</td>"
            f"<td class='num'>{_c['epochs']}</td>"
            f"<td class='num'>{_fmt(_ad, '.4f')}</td><td class='num'>{_fmt(_ac, '.1f')}</td>"
            f"<td class='num'>{_fmt(_sd, '.4f')}</td><td class='num'>{_fmt(_sc, '.1f')}</td></tr>"
        )
    _table = f"""
    <div class="report-table-scroll"><table class="report-table">
    <thead><tr><th>condition</th><th>one change</th><th class="num">epochs</th>
    <th class="num">anchor dose</th><th class="num">centroid</th>
    <th class="num">anti dose</th><th class="num">centroid</th></tr></thead>
    <tbody>{"".join(_rows)}</tbody>
    </table></div>
    """
    mo.Html(
        figure_html(
            _table,
            caption="""
            **The ablation conditions and their delivered invariants.** Dose
            is ∫ w·lr de over that condition's training; centroid is the
            dose-weighted mean epoch. Doses are reported, not matched — the
            flat arms change shape and timing, and the columns say by how
            much.
            """,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The decision rules

    Rendered from the design constants (`experiment.DECISION_RULES`), so the
    frozen text has one home:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("\n".join("> " + line for line in ex.DECISION_RULES.splitlines()))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The survey space

    Two dimensions are unconditional. **λ_a** spans [0.02, 1.0]: from half
    the scoring rung — below which ex-2.1.6's ladder suggests the pull is
    weak — up to the ex-2.1.7 ceiling arm's weight, where selectivity was
    already lost, so the believed optimum is interior. **τ** spans
    [0.05, 0.30] on the weight-scale argument above. Both log.

    The anti-subspace dimension depends on decision rule 2. If a flat ratio
    suffices (**branch B**), it is one knob: μ/λ ∈ [0.10, 3.0], log — a
    third of the hold level to above the peak. If the schedule shape matters
    (**branch A**), the family keeps its shape (hold ratio fixed at 0.30)
    and two knobs are sampled — peak ratio ∈ [0.75, 4.0] (log) and anneal
    endpoint ∈ [30%, 95%] of training — but the *analysis* reports marginals
    in the derived coordinates that matter, delivered dose and timing
    centroid, because the raw knobs are a correlated ridge (the endpoint is
    the dose axis; ex-2.1.8). Sampling stays in the knob box since every
    point of it is a valid schedule, where a box in (dose, centroid) has
    unreachable corners.

    Trials are a scrambled Sobol set — even one-dimensional marginals at
    this budget, no two trials aligned on any axis — with the scramble seed
    frozen in the design constants. Bayesian optimization is deliberately
    not used: it returns an argmax where D2.2 needs a map, and with the
    containment constraint's noise it would chase seed luck. The full trial
    list for whichever branch runs is below, computable from the design
    constants alone and identical when the run rebuilds it.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _trial_table(space: dict, sched: bool) -> str:
        trials = ex.sobol_trials(space)
        head = "".join(f"<th class='num'>{d}</th>" for d in space)
        extra = (
            "<th class='num'>rel dose</th><th class='num'>centroid</th>" if sched else "<th class='num'>rel dose</th>"
        )
        rows = []
        for t in trials:
            cells = "".join(f"<td class='num'>{t[d]:.3f}</td>" for d in space)
            rd = ex.anti_rel_dose(t)
            if sched:
                pr, ae = t["anti_peak_ratio"], t["anti_anneal_end_frac"]
                cen = ex.dose_centroid(
                    lambda e, pr=pr, ae=ae: ex.anti_weight(e, lam=1.0, peak_ratio=pr, anneal_end_frac=ae)
                )
                extra_cells = f"<td class='num'>{rd:.2f}</td><td class='num'>{cen:.1f}</td>"
            else:
                extra_cells = f"<td class='num'>{rd:.2f}</td>"
            rows.append(f"<tr><td class='num'>{t['trial']}</td>{cells}{extra_cells}</tr>")
        return f"""
        <div class="report-table-scroll"><table class="report-table">
        <thead><tr><th class="num">#</th>{head}{extra}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
        </table></div>
        """

    _a = _trial_table({**ex.SPACE_COMMON, **ex.SPACE_ANTI_SCHED}, sched=True)
    _b = _trial_table({**ex.SPACE_COMMON, **ex.SPACE_ANTI_FLAT}, sched=False)
    mo.Html(
        figure_html(
            f"<details><summary>Branch A trial list (schedule retained)</summary>{_a}</details>"
            f"<details><summary>Branch B trial list (flat anti)</summary>{_b}</details>",
            caption=f"""
            **The frozen trial lists**, {ex.N_TRIALS} scrambled-Sobol points
            per branch (seed {ex.SOBOL_SEED}), shown for review before any
            run. Branch A adds the derived (relative dose, centroid)
            coordinates its marginals will be reported in; relative dose is
            the trial's ∫ μ/λ·lr de over the operating point's.
            """,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Objective, constraints, and rounds
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    **Objective** (from `experiment.OBJECTIVE`): rank feasible trials by
    seed-mean m_line, largest first. m_line is the one statistic tight
    enough to rank on; the calibration stage shows why it cannot stand
    alone, so everything else is a constraint:

    | constraint | bar | source |
    |---|---|---|
    | task cost | holdout EM within {ex.TASK_GATE:g} of the same-length control | ex-2.1.6 on |
    | containment | ᾱ at op1 ≤ {ex.MEAN_ALIGN_GATE:g} | ex-2.1.8 on |
    | retention | final m_line ≥ {ex.RETENTION_GATE:g} × running peak (floor {ex.RETENTION_FLOOR:g}) | ex-2.1.8 on |
    | grading | per-run r² within {ex.GRADE_R2_DROP:g} of `ref`'s seed mean | ex-2.1.9 on, relative form |
    | contrast | group contrast ≥ {ex.CONTRAST_MIN:g} | ex-2.1.10's partial bar |
    | latch | no run with deep-slice syntax weight > {ex.LATCH_PI:g} | calibrated above |

    **Rounds.** Round 1 runs all {ex.N_TRIALS} trials at seed 0 — licensed
    by the noise floors, which put the margin statistics' per-run σ far
    below the effects of interest. The {ex.N_PROMOTE} feasible trials with
    the highest m_line are promoted to {ex.SEEDS_PROMOTE} seeds; fewer
    feasible, promote all; none, publish the infeasibility map and propose
    nothing. The proposed operating point is the promoted trial with the
    highest seed-mean m_line that stays feasible on seed means (the latch
    veto stays per-run). Promotion corrects the winner's curse only partly,
    which is why the proposal must be confirmed at fresh seeds before it is
    quoted.

    **Stopping rule.** Fixed budget, two rounds, no adaptive continuation.
    Diverged or failed trials are published as failures, not resampled.
    **Every trial is published**, including the ones that go nowhere —
    memoization keeps them anyway, and a complete table is what makes a
    search's map trustworthy.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Budget
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {len(ex.ABLATION_CONDITIONS)} ablation conditions × {ex.N_SEEDS_ABLATION}
    seeds = {ex.N_RUNS_ABLATION} runs, then at most {ex.N_TRIALS} +
    {ex.N_PROMOTE} × {ex.SEEDS_PROMOTE - 1} = {ex.N_RUNS_SURVEY} survey runs
    — ≈ {ex.N_RUNS_ABLATION + ex.N_RUNS_SURVEY} training runs worst case,
    each ~25 min of L4 with the slim eval, halved for the survey stage if
    `short` is adopted. Comparable to two ex-2.1.10s. Each stage launches
    with `--max-containers {ex.MAX_CONTAINERS}` and a `--budget` sized from
    its run count; memoization makes the promoted round re-run only the new
    seeds, and `ctx.map` with `allow_partial=True` lets diverged trials land
    as data rather than failures.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Ablations

    *Results land here; the decision each subsection feeds is fixed above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Is the anchor schedule needed?

    /// admonition | TODO
    Table: `flat-anchor` − `ref` seed-mean differences for each decision
    statistic, each against its band; per-run deep-slice weight profiles
    beside ex-2.1.9's latch table; the m_line trajectories of all six runs
    overlaid (does the flat arm's margin arrive earlier, and does it hold?).
    Expected under the commit-window account: the flat arm latches a syntax
    role in at least one run, or holds a resolvably lower m_line — the dose
    columns say any such difference is timing, not dose. Contrary result:
    flat within band with no latch, in which case the survey runs flat and
    four schedule knobs retire.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Does the anti-subspace schedule beat a constant?

    /// admonition | TODO
    Table: `anti-hold` and `anti-peak` against `ref`, same statistics, same
    bands, with each arm's delivered dose and centroid restated from the
    conditions table. Gate the schedule's effect on the peak-matched arm;
    read the hold-level arm as the dose secondary. Expected from ex-2.1.8's
    mechanism (early repulsion prices the syntax roles out before the pull
    commits): `anti-hold` loses containment or contrast, `anti-peak` holds
    the statistics but pays somewhere visible — if it doesn't, the schedule
    was a constant in disguise and branch B runs. Contrary result worth
    naming: `anti-hold` within band, which would say the repulsion's timing
    never mattered at this operating point, and branch B runs at the lower
    level.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Is 100 epochs buying anything?

    /// admonition | TODO
    Table: `short` − `ref` on the decision statistics (task gate against
    `short-lam0`); the m_line and validation-loss trajectories on a shared
    fraction-of-training axis. Ex-2.1.6 saw margins flat from epoch ~50 and
    validation loss settle earlier, so the expectation is `short` within
    band, and the survey at half cost. Contrary: a resolved m_line or
    retention gap, which would mean the last 50 epochs do consolidation the
    trajectory statistics haven't shown — worth knowing either way before
    D2.2 scales anything.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Does the anneal shape matter?

    /// admonition | TODO
    One row: `linear` − `ref` per decision statistic against its band. M1
    only ever compared a stepped anneal (which needed an LR-warmup
    accommodation); the linear arm's milder discontinuity — a jump in λ's
    derivative at each keyframe — is the cheapest version of the question.
    Expected: within band, retiring the todo item. Either outcome changes
    no survey dimension.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The landscape

    *The survey's deliverable: the map, then the point.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | TODO
    Four views, all from the published per-trial table. (1) The full trial
    table: sampled coordinates, every statistic, feasibility per constraint,
    round-2 seed means where promoted — no trial omitted. (2) Marginals:
    m_line and each constraint statistic against each dimension (branch A's
    anti axes in dose/centroid coordinates), feasible and infeasible trials
    distinguished; the reading is where the feasible region is and how flat
    m_line is across it — a wide plateau is itself the result D2.2 wants.
    (3) The m_line ↔ r² plane with the constraint boundary drawn, showing
    what the grading constraint excluded; beside the per-color r², report
    the conditional-mean r² over redness levels — the ex-2.1.10 review found
    the pair separates graded-on-average from graded color-by-color, and
    only the per-color reading constrains. (4) The λ_a marginal specifically:
    ex-2.1.7 predicts a selectivity optimum below λ = 1; where the
    containment and grading constraints actually bind along λ is the
    highest-value single read in the survey.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Proposed operating point

    /// admonition | TODO
    One line: the winning trial's coordinates and seed-mean statistics, each
    with its noise floor; beside it, the reference recipe's values and the
    width of the plateau it sits on. Plus the handoff sentence: which
    experiment confirms it, at how many fresh seeds, before any of these
    numbers is quoted.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    *Empty until results land; anything conceived after seeing the data goes
    here, marked as post hoc.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    Interpretation only, once the observations exist: what the ablations say
    about M1's schedule inheritance in this architecture, what the map says
    about how carefully M3 will need to tune, and what the survey style
    itself cost or saved (first use of the type — note friction for the
    science skill).
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    # Shared helpers for the calibration cells: the ex-2.1.10 statistics,
    # restated here so the skeleton has no dependency on that experiment's module.
    def softmin_weights_np(x: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
        z = -np.asarray(x, dtype=np.float64) / tau
        z -= z.max(axis=axis, keepdims=True)
        w = np.exp(z)
        return w / w.sum(axis=axis, keepdims=True)

    def group_weights(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1, p2 = np.asarray(ex.p_slot(r1)), np.asarray(ex.p_slot(r2))
        g1 = p1 * (1.0 - p2)
        g2 = (1.0 - p1) * p2
        return g1 / g1.sum(), g2 / g2.sum()

    return group_weights, softmin_weights_np


if __name__ == "__main__":
    app.run()
