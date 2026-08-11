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
    from mini.vis import AxesGrid, figure_html, light_dark, themed

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

        Ex-2.1.10's metrics, arrays and probe tables (the nine-seed floors, the trade, the τ map), ex-2.1.9's metrics (the latch calibration), and ex-2.1.8's metrics (the across-point correlation).
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
        found = [a for r in refs if (a := arts[r]) is not None]
        if len(found) < len(refs):
            return None
        with tempfile.TemporaryDirectory() as d:
            paths = store.get_many([(a, Path(d) / f"{i}.bin") for i, a in enumerate(found)])
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
    A survey to close out D2.1. The schedules and weights in the recipe were inherited piece by piece and never tuned together. So we ablate first: flat arms that bracket each schedule and their composition, plus a half-length condition. Every "no difference" lets us drop a dimension. Then a Sobol search over what is left (anchor weight, pooling temperature, repulsion dose) maps the good region and proposes the operating point the next milestone inherits.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Observations

    **The reference recipe reproduces.** `ref` re-runs the primary condition of ex-2.1.10 through the code path of this experiment. It lands within one per-run standard deviation on every statistic: m_line 0.4204 against 0.4202 (σ 0.0088), group contrast 0.8562 against 0.8542 (σ 0.0057), holdout exact match 0.9961 against 0.9948 (σ 0.0087).

    **The anchor schedule can go.** `flat-anchor` stays within band on all five decision statistics, and no run latches (m_line +0.0039, band 0.0144). Rule 1: the survey runs a flat anchor weight.

    **The anti-subspace schedule cannot.** Both flat brackets come out worse. `anti-hold` loses the group contrast (−0.4667, band 0.0093), grading (−0.2199, band 0.0392), and containment (+0.1275, band 0.0346), and two of its three runs latch onto `+`. `anti-peak` keeps containment and contrast, but loses grading (−0.1283) and m_line (−0.0282, band 0.0144). Rule 2: the survey searches the schedule family, branch A.

    **Half the training is enough.** `short` stays within band on every statistic (m_line +0.0014), and costs 0.0013 on the task against its own control. Rule 4: every survey trial runs 50 epochs.

    **Nothing resolves on the interpolation shape.** `linear` stays within band throughout; its largest excursion is grading, at −0.0248 against a band of 0.0392. Rule 5: keep minimum-jerk, and the question retires.

    **Proposed operating point.** *Lands when the survey has run.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a survey, not a hypothesis-scoring experiment. It preregisters a *search plan* and scores nothing. The plan covers the space to search, the sampling rule, the objective and its constraints, the seed budget, and the decision rules that stage the work. Once we agree on it, it is frozen (aside from immaterial edits). Results replace the `TODO` placeholders in place, and anything we think of after seeing the data goes under *Exploratory analyses*, marked as post hoc.

    Nothing this report finds may be quoted as a result. It proposes an operating point. The next preregistered experiment adopts that point, re-measures it at fresh seeds, and reports the survey value next to the confirmed one. The gap between the two is the winner's-curse correction: the best point in a search looks better than it really is, because part of what put it on top was luck.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Hyperparameters in D2.1 have history but little justification. The anchor schedule (warmup, hold, then anneal at the end down to a floor) and the anti-subspace schedule (2.5 → 0.3 of the anchor weight) were inherited from M1, mapped onto 100 epochs by fraction of training. Those keyframes were tuned for a 5-dimensional autoencoder bottleneck and are now applied to a 64-dimensional residual stream.[^rs]

    The anchor weight λ_a = 0.1 is the rung ex-2.1.6 scored at, chosen to sit safely inside the region where the task is unharmed. Ex-2.1.7 showed that selectivity is gone by λ_a = 1, but didn't show where it begins to degrade.

    τ = 0.1 is the one rung of the coarse ladders that stayed graded in every seed: the rungs below it latch per run (ex-2.1.9, ÷3 and ÷10), and the rung above it loses the group contrast and the grading in a single step (ex-2.1.10, ×5). So we don't know whether the primary sits on a plateau or on a knife edge.

    We ablate first and search second, because deleting a dimension is cheaper than searching it. If a constant can replace a schedule, four parameters go away; if the testbed converges in 50 epochs, every later trial is half the cost. Before either, a calibration stage fixes the noise floors that decide what "no difference" means, and settles the shape of the objective.

    Out of scope: the position-dropout mechanism raised in the ex-2.1.9 review, which is not part of the recipe yet; and the grammar, corpus, labeller, and architecture, which stay fixed at the ex-2.1.10 primary throughout.

    [^rs]: The residual stream is the running vector of activations that each transformer layer reads from and writes back into.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | Glossary
    - **trial** — one sampled point in the search space, aggregated over seeds. It is sampled, so unlike a *condition* its levels weren't chosen by hand.
    - **round** — one pass over trials at a given seed budget. Round 1 is every trial at one seed; round 2 is the promoted trials at five.
    - **λ_a** — the weight on the anchor term (`lam` in code; bare λ in earlier reports). The subscript leaves room for the weights of other terms.
    - **λ_s̄** — the weight on the anti-subspace term (μ in earlier reports). The s̄ means "everything outside the anchored subspace". We usually state it as the ratio λ_s̄/λ_a.
    - **feasible** — a trial that satisfies every constraint: task gate, containment, retention, grading, contrast, and no latch. Only feasible trials are ranked.
    - **dose** — how much weight a schedule delivers in total, $\int w(e) \cdot \mathrm{lr}(e) \, \mathrm{d}e$ over training. The learning rate is inside the integral because it multiplies the gradient of every term. The **centroid** is the dose-weighted mean epoch, i.e. *when* the dose arrives.
    - **band** — how small a difference between seed means we can resolve, 2σ·√(1/n₁ + 1/n₂) from the per-run σ over nine seeds. A difference inside the band is unresolved. That says something about our resolution, and is never evidence that two things are equal.
    - **latch** — a run in which the pooled pull commits its deep-slice weight to a syntax role (`+` or `=`). This is the winner-take-all failure ex-2.1.9 found at small τ.
    - **m_line** — per-line selectivity: label-weighted minus unweighted mean alignment, at the best span role, averaged over slices. This is the statistic ex-2.1.10 scored. **ᾱ** at op1 measures containment. **retention** is the final value of the m_line trajectory divided by its running peak. **r²** is the squared Pearson correlation between the per-color op1 response and the sim^1.5 target, and measures grading. **contrast** is the difference in op2 softmin weight between the two label groups, and measures per-line localization.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    The testbed is the ex-2.1.10 primary, unchanged: the word-level `v216` corpus (216 colors, one token each, d64-L4), the either-slot labeller, the mellowmax-pooled anchor term over the four span roles, and the anti-subspace term. Under the either-slot labeller, each operand draws with probability redness⁸ · 0.04, and a line is labeled if either operand draws.

    The reference recipe is the operating point of that experiment: λ_a = 0.1, τ = 0.1, anchor schedule with warmup over 0–10% and anneal over 90–100% down to a 0.1 floor, and anti-subspace 2.5 → 0.30 (λ_s̄/λ_a) by 90%. Schedule keyframes are given as fractions of training, so the half-length condition runs the same shape.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Measurements

    Every run, ablation or trial, records the same slim set of measurements, all of them defined in earlier experiments: the three behavior sets (holdout exact match is the task gate), the alignment map on the probe lines (m_line, ᾱ at op1, r² against the sim^1.5 target), the softmin weight profiles (the group contrast and the latch detector), and the training trajectory of m_line (retention).

    That costs one eval pass per run. We do not run the readability, geometry, and leakage passes that a scoring experiment carries. Those belong to the confirming experiment, which measures one point properly rather than eighty points thinly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Search plan

    Calibration has already run; it reads only published results, and its numbers are frozen into the design constants. The ablation stage runs nine conditions and applies the decision rules below, which fix the recipe, the length, and the space for the survey. The survey stage then draws its trials from a rule that is deterministic given this document.

    The decision rules are the only branch points; everything downstream of the data is fixed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration: noise floors

    Ex-2.1.10 ran nine seeds on its primary, which is the reference recipe here, so the per-run spread of every statistic at the operating point is already in the store. The table below re-derives the floors from the published arrays. The design constants freeze the same numbers (`NOISE_RUN`), and the ablation bands follow as 2σ·√(2/3) for a three-seed against three-seed comparison.

    The margin statistics are tight compared to the effects worth chasing; the τ ladder in ex-2.1.10 spans several standard deviations on m_line. That is what lets us run survey trials at a single seed. Containment is the opposite, with a standard deviation about 40% of its own mean, so it enters the survey as a constraint rather than something we rank on.

    These floors carry a caveat: they are measured in the graded regime, at τ = 0.1. At τ ≲ 0.03 the pull latches from run to run, the statistics become bimodal, and no per-run standard deviation describes them. So the survey keeps τ above that regime and uses a per-run latch veto instead.
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
def _(group_weights, prior: dict):
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
def _(per_run: dict[str, np.ndarray]):
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
            caption=mo.md("""
            **Noise floors at the operating point.** Per-run mean and sd over ex-2.1.10's nine-seed primary, re-derived from its published arrays; the *frozen σ* column is the value the design constants carry, and the *band* is the smallest seed-mean difference the ablation stage may call a difference (2σ·√(2/3), three seeds a side).
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration: the shape of the objective

    Across the anchored operating points already in the store, margin and containment don't trade: their correlation is negative on both available readings (computed live below the figure), meaning the better operating points improved both at once. The trade that does bind is margin against grading, and it runs along the τ axis: softening τ raises m_line while the per-color grading and the group contrast collapse, so selectivity smears out into "everything red-ish moves".
    <!-- REVIEW: the correlation was a hard-coded −0.19 credited to all three prior experiments; it is now computed in the figure cell from the loaded metrics, on two bases: m_span over the ten ex-2.1.9/ex-2.1.10 points (the original number — ex-2.1.8 publishes no m_span, so it never contributed), and m_op1 over all seventeen points including the ex-2.1.8 grid. Verify: both print beneath the trade figure. -->


    That structure decides the shape of the search. A two-way trade would need a Pareto front,[^pareto] but objectives that move together can be constraints around a single scalar. So the objective is one number: maximize m_line, subject to the task gate, containment, retention, grading, contrast, and the latch veto, with the values under *Objective, constraints, and rounds* below.

    A search on margin alone would walk to the smeared end of the τ range, and the grading and contrast constraints are what keep it out. We still report the m_line against r² plane for every trial, so if the interesting points turn out to sit on the constraint boundary, the map will show it.

    [^pareto]: The set of points at which no objective can be improved without giving up some of another. It is a curve to choose along, rather than a single winner.
    """)
    return


@app.cell(hide_code=True)
def _(per_run: dict[str, np.ndarray], prior: dict):
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
            Scatter chart of grading r squared against per-line margin m_line. Five clusters of small dots, one per ex-2.1.10 condition, each with a larger ringed mark at its seed mean. The reference arm, the primary and the slot oracle cluster at the upper left, near m_line 0.4 and r squared around 0.8. The two soft-tau arms sit to the lower right, near m_line 0.49 and r squared 0.65 and 0.58: margin rises as grading falls along the tau axis.
        """,
        caption=r"""
            **The trade the constraints must hold.** Per-run grading $r^2$ against per-run m_line for ex-2.1.10's five anchored conditions (small marks; large ring = seed mean). Softening τ (0.1 → 0.5 → 2.5, left to right) buys margin and spends grading; ranking on m_line alone would pick the right-hand end. The survey caps that walk with the grading and contrast constraints instead of a second objective.
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

    # Margin–containment correlation across stored operating points, on two
    # bases: m_span over the ex-2.1.9/ex-2.1.10 points (ex-2.1.8 publishes no
    # m_span), and m_op1 over all three experiments' anchored points.
    def _seed_means(metrics: dict, stat: str) -> dict[str, float]:
        conds = sorted({c["condition"] for c in metrics["cells"]} - {"lam0"})
        return {
            k: float(np.mean([c[stat] for c in metrics["cells"] if c["condition"] == k]))
            for k in conds
            if stat in next(c for c in metrics["cells"] if c["condition"] == k)
        }

    def _corr(stat: str, sources: list[dict]) -> tuple[float, int]:
        m = [v for s in sources for v in _seed_means(s, stat).values()]
        a = [v for s in sources for v in _seed_means(s, "alpha_mean_op1").values()]
        return float(np.corrcoef(m, a)[0, 1]), len(m)

    _r_span, _n_span = _corr("m_span", [prior["m9"], prior["m10"]])
    _r_op1, _n_op1 = _corr("m_op1", [prior["m8"], prior["m9"], prior["m10"]])

    _corr_note = mo.md(rf"""
    Margin–containment correlation across the stored anchored operating points: $r$ = {_r_span:+.2f} over the {_n_span} ex-2.1.9/ex-2.1.10 points (m_span basis), and $r$ = {_r_op1:+.2f} over all {_n_op1} points including the ex-2.1.8 grid (m_op1 basis, the statistic all three share). No trade on either reading; both improve together along the design line. For scale, the primary's nine runs put per-run grading at {per_run["r2_sim"].mean():.2f} ± {per_run["r2_sim"].std(ddof=1):.3f} against {_pts["either-t2500"][1].mean():.2f} for the softest arm.
    """)
    mo.vstack([mo.Html(_plot()), _corr_note])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration: the τ range

    τ only matters through the weights it produces, so we set its bounds on the weight scale: the share of the pooled pull held by the leading span position, computed from the stored alignment profiles of the primary.

    The bounds [0.05, 0.30] keep that leading weight between about 0.74 and about 0.42 on the trained profile, which covers the lower part of the 0.6–0.8 target band from ex-2.1.9. Reaching 0.8 would need τ of about 0.04. The lower bound sits above that, keeping a margin over the latching regime, where τ ≲ 0.03, the leading weight is ≳ 0.82, and the profile goes one-hot per run at depth. The upper bound stops short of τ = 0.5, which the trade figure above shows is already smeared.
    <!-- REVIEW: was "contains the 0.6-0.8 target band". The stated range of the leading weight is [0.42, 0.74], so it meets the band only over 0.6-0.74; the figure's alt text already described the two as overlapping. The exclusion of the top of the band is a margin choice rather than a consequence of the latching regime, so it is now stated as one. Verify: if the survey shows τ ≈ 0.04 stays graded, the lower bound can widen. -->
    """)
    return


@app.cell(hide_code=True)
def _(prior: dict, softmin_weights_np):
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
            Line chart of leading softmin weight against temperature tau on a log scale from 0.01 to 1. Five curves, one per residual slice, fall from near 1 at the left to about 0.3 at the right. A shaded vertical band marks the survey range 0.05 to 0.3, and a shaded horizontal band marks the target leading-weight range 0.6 to 0.8. The two bands overlap around tau 0.05 to 0.1. A dashed vertical line marks the reference tau of 0.1.
        """,
        caption=r"""
            **Where the survey's τ bounds sit on the weight scale.** The label-weighted share of softmin weight held by the leading span position, per residual slice, on the primary's seed-mean alignment profile. Vertical shading: the survey range [0.05, 0.30]; horizontal shading: ex-2.1.9's 0.6–0.8 target band; dashed line: the reference τ = 0.1.
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

    One flat arm is enough to ablate the anchor schedule, because flat at the peak and flat at matched dose are the same thing here. The schedule delivers 97% of the flat-at-peak dose, and that 3% gap sits well inside the noise floor, so we don't have to pick which quantity to match on.

    What the flat arm does change is timing. The ramp carries 7.3% of the dose across the same window in which ex-2.1.9 found the pooled pull commits, by epoch 8, inside the 10-epoch warmup. The `AnchorSpec` docstring says the ramp is there so the anchor arrives with the optimizer rather than ahead of it.

    So we don't expect a dose effect from `flat-anchor`. Full λ_a against a still-random model risks latching, so the reading will come from the latch veto and the deep-slice weight profile.

    No constant can match the anti-subspace term on dose. Its schedule spans an 83× range of weights, and the constant that matched its dose would sit at 1.7× the anchor peak for the whole of training, which is a different regime; the dose arm in ex-2.1.8 documented that trap.

    So two arms bracket it instead: flat at the hold ratio (0.17× the dose of the schedule) and flat at the peak ratio (1.41×). We gate the effect of the schedule on the peak-matched arm, and read the dose alignment as a stated secondary. The conditions table below reports the delivered dose and centroid for every arm.
    """)
    return


@app.cell(hide_code=True)
def _():
    _e = np.linspace(0.0, float(ex.EPOCHS), 2001)
    _lr = ex.learning_rate(_e)
    _flat = np.ones_like(_e)
    _anchor_sched = ex.anchor_weight(_e)
    _anchor_flat = ex.SCORING_LAMBDA * _flat
    _anti_sched = ex.anti_weight(_e)
    _anti_hold = ex.ANTI_HOLD_RATIO * ex.SCORING_LAMBDA * _flat
    _anti_peak = ex.ANTI_PEAK_RATIO * ex.SCORING_LAMBDA * _flat

    @themed(
        name="doses",
        alt_text="""
            A two-by-two grid of line charts over epochs 0 to 100, with the anchor term on the left and the anti-subspace term on the right. The top row shows the term weights, where the flat arms are horizontal lines and the two schedules ramp or decay. The bottom row multiplies those weights by the learning rate, so every curve rises and falls together; the scheduled and flat anchor curves then differ only over the first ten epochs.
        """,
        caption=r"""
            **What each arm delivers, when.** **Top row:** the term weights, λ_a(e) (left) and λ_s̄(e) (right). The flat arms really do hold a constant weight. **Bottom row:** the effective weights, w(e) · lr(e), which is what the dose integrals measure. The rise and fall shared by every curve is the LR schedule. **Left:** the anchor schedule and `flat-anchor` differ visibly only during warmup, and their doses agree to 3%. **Right:** `anti-hold` and `anti-peak` bracket the schedule from below (0.17× its dose) and above (1.41×). No constant matches its dose while staying in the regime.
        """,
    )
    def _plot():
        grey = light_dark("#666", "#999")
        sched_c = light_dark("#1f6fb4", "#5fa8dd")
        flat_c = light_dark("#b4531f", "#dd8f5f")
        flat2_c = light_dark("#2a9d8f", "#5fc9bd")
        fig, _axs = plt.subplots(2, 2, figsize=(7.2, 4.4), layout="constrained", sharex="col", sharey="row")
        (ax_aw, ax_sw), (ax_ad, ax_sd) = cast(AxesGrid, _axs)
        for ax, mult in ((ax_aw, _flat), (ax_ad, _lr)):
            ax.plot(_e, _anchor_sched * mult, color=sched_c, lw=1.4, label="schedule")
            ax.plot(_e, _anchor_flat * mult, color=flat_c, lw=1.4, ls=(0, (4, 2)), label="flat-anchor")
        for ax, mult in ((ax_sw, _flat), (ax_sd, _lr)):
            ax.plot(_e, _anti_sched * mult, color=sched_c, lw=1.4, label="schedule")
            ax.plot(_e, _anti_hold * mult, color=flat2_c, lw=1.4, ls=(0, (4, 2)), label="anti-hold")
            ax.plot(_e, _anti_peak * mult, color=flat_c, lw=1.4, ls=(0, (1, 2)), label="anti-peak")
        ax_aw.set_title("anchor term", fontsize="small", color=grey)
        ax_sw.set_title("anti-subspace term", fontsize="small", color=grey)
        ax_aw.set_ylabel("term weight w(e)", fontsize="x-small")
        ax_ad.set_ylabel("w(e) · lr(e)", fontsize="x-small")
        ax_aw.legend(fontsize="x-small", frameon=False)
        ax_sw.legend(fontsize="x-small", frameon=False)
        for ax in (ax_ad, ax_sd):
            ax.set_xlabel("epoch", fontsize="x-small")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(prior: dict):
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
    <thead><tr><th>run (cond-seed)</th><th class="num">op1</th><th class="num">+</th><th class="num">op2</th>
    <th class="num">=</th><th class="num">max syntax ↓</th></tr></thead>
    <tbody>{_body}</tbody>
    </table></div>
    """
    mo.Html(
        figure_html(
            _table,
            caption=mo.md(f"""
            **The latch detector, calibrated on ex-2.1.9.** Deep-slice (post-attention) mean softmin weight per span role, per stored run of the pooled conditions (with seed shown as a suffix). Latched runs put ≥ 0.85 on `+`; healthy runs ≤ 0.03 on any syntax role (bold = over the {ex.LATCH_PI:g} veto). The threshold sits in a wide gap, so the veto is insensitive to its exact value.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The ablation conditions

    There are nine conditions, three seeds each, all at the reference λ_a and τ under the either-slot labeller. `ref` re-runs the primary recipe from ex-2.1.10 through the code path of this experiment, so every comparison is between runs of identical code. `lam0` and `short-lam0` are the task-cost controls, one at each length. `flat-both` combines the two flat simplifications. We need it because an arm that changes one schedule at a time cannot show an interaction between the two; we read it only if decision rules 1 and 2 both simplify. The table reports dose and centroid for each arm, so the outcomes can tell us which of the two was doing the work.
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: the shapes now come from `experiment.condition_shapes`, the same
    # mapping the runs are built from, rather than being re-derived here. The
    # `linear` arm was previously reported at the minimum-jerk schedule's dose,
    # which is not what it delivers: the anti-subspace ratio interpolates over
    # 90 epochs, where a straight line and a smoothstep differ well outside
    # rounding. Verify: every other row is unchanged, since flat and minimum-jerk
    # were already read correctly.
    def _invariants(c: dict) -> tuple[float, float, float, float]:
        epochs: int = c["epochs"]
        anchor_shape, anti_shape = ex.condition_shapes(c)
        ratio = c.get("anti_ratio", ex.ANTI_HOLD_RATIO)
        anchor = lambda e: ex.anchor_weight(e, peak=c["lam"], epochs=epochs, shape=anchor_shape)  # noqa: E731
        anti = lambda e: ex.anti_weight(  # noqa: E731
            e, lam=c["lam"], epochs=epochs, shape=anti_shape, hold_ratio=ratio
        )
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
            "flat-anchor": "λ_a constant from step 0; no ramp, anneal or floor",
            "anti-hold": "λ_s̄/λ_a constant at the hold ratio (0.30)",
            "anti-peak": "λ_s̄/λ_a constant at the peak ratio (2.5)",
            "flat-both": "flat anchor and flat anti (hold ratio) together",
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
            caption=mo.md(r"""
            **The ablation conditions and their delivered invariants.** Dose is $\int w \cdot \mathrm{lr} \, \mathrm{d}e$ over that condition's training; centroid is the dose-weighted mean epoch. Doses are reported, not matched.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The decision rules

    Rendered from the design constants (`experiment.DECISION_RULES`), so the frozen text has one home:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ---

    {ex.DECISION_RULES}

    ---
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The survey space

    Two dimensions are always sampled, both on a log scale. λ_a spans [0.02, 1.0]: from a fifth of the rung we scored at, below which the ladder in ex-2.1.6 suggests the pull is weak, up to the weight of the ex-2.1.7 ceiling arm, where selectivity was already gone. So we believe the optimum is somewhere inside. τ spans [0.05, 0.30], on the weight-scale argument above.
    <!-- REVIEW: "half the rung" → "a fifth"; the scored rung is λ = 0.1 and the bound is 0.02. Same correction in the SPACE_COMMON docstring. -->

    The anti-subspace dimension depends on decision rule 2. If a flat ratio is enough (branch B), it is one dimension: λ_s̄/λ_a ∈ [0.10, 3.0], log, running from a third of the hold level up past the peak. If the shape of the schedule matters (branch A), the family keeps that shape with the hold ratio fixed at 0.30, and we sample two dimensions: peak ratio ∈ [0.75, 4.0] (log) and anneal endpoint ∈ [30%, 95%] of training.

    For branch A the analysis reports marginals in derived coordinates, delivered dose and timing centroid, rather than in the sampled parameters. Those parameters form a correlated ridge, since the endpoint is effectively the dose axis (ex-2.1.8). Sampling still happens in the parameter box, because every point in it is a valid schedule, whereas a box in (dose, centroid) has corners we can't reach.

    Trials come from a scrambled Sobol set, which gives even one-dimensional marginals at this budget with no two trials aligned on any axis. The scramble seed is frozen in the design constants, so the run rebuilds the trial list below identically.

    We deliberately don't use Bayesian optimization: it returns a single best point where D2.2 needs a map, and given the noise in the containment constraint it would chase seed luck.
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
            caption=mo.md(rf"""
            **The frozen trial lists**, {ex.N_TRIALS} scrambled-Sobol points per branch (seed {ex.SOBOL_SEED}), shown for review before any run. Branch A adds the derived (relative dose, centroid) coordinates its marginals will be reported in; relative dose is the trial's $\int (\lambda_{{\bar{{s}}}}/\lambda_a) \cdot \mathrm{{lr}} \, \mathrm{{d}}e$ over the operating point's.
            """).text,
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
    **Objective** (from `experiment.OBJECTIVE`): rank feasible trials by seed-mean m_line, largest first. m_line is the one statistic tight enough to rank on, so everything else is a constraint:

    | constraint | bar | source |
    |---|---|---|
    | task cost | holdout EM within {ex.TASK_GATE:g} of the same-length control | ex-2.1.6 on |
    | containment | ᾱ at op1 ≤ {ex.MEAN_ALIGN_GATE:g} | ex-2.1.8 on |
    | retention | final m_line ≥ {ex.RETENTION_GATE:g} × running peak (floor {ex.RETENTION_FLOOR:g}) | ex-2.1.8 on |
    | grading | per-run r² within {ex.GRADE_R2_DROP:g} of `ref`'s seed mean | ex-2.1.9 on, relative form |
    | contrast | group contrast ≥ {ex.CONTRAST_MIN:g} | ex-2.1.10's partial bar |
    | latch | no run with deep-slice syntax weight > {ex.LATCH_PI:g} | calibrated above |

    **Rounds.** Round 1 runs all {ex.N_TRIALS} trials at seed 0, which the noise floors allow. The {ex.N_PROMOTE} feasible trials with the highest m_line are promoted to {ex.SEEDS_PROMOTE} seeds. If fewer are feasible, we promote all of them; if none are, we publish the infeasibility map and propose nothing.

    The proposed operating point is the promoted trial with the highest seed-mean m_line that is still feasible on seed means; the latch veto stays per-run. Promotion only partly corrects the winner's curse.

    **Stopping rule.** Fixed budget, two rounds, no adaptive continuation. Trials that diverge or fail are published as failures rather than resampled. Memoization keeps them anyway, and a complete table makes the map from a search trustworthy.
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
    {len(ex.ABLATION_CONDITIONS)} ablation conditions × {ex.N_SEEDS_ABLATION} seeds = {ex.N_RUNS_ABLATION} runs, then at most {ex.N_TRIALS} + {ex.N_PROMOTE} × {ex.SEEDS_PROMOTE - 1} = {ex.N_RUNS_SURVEY} survey runs. That is about {ex.N_RUNS_ABLATION + ex.N_RUNS_SURVEY} training runs in the worst case, each roughly 25 minutes of L4 time with the slim eval. Ex-2.1.10 was 24 runs, so this is about 3.5 times as many, or about twice the training time if we adopt `short` and the survey stage runs at half length. Each run is also cheaper, because the slim eval drops the readability, geometry, and leakage passes.

    Each stage launches with `--max-containers {ex.MAX_CONTAINERS}` and a `--budget` sized from its run count. Memoization means the promoted round re-runs only the new seeds, and `ctx.map` with `allow_partial=True` lets diverged trials land as data rather than failures.

    <!-- Observed after the run, not a design change. -->
    The 25-minute figure was inherited from ex-2.1.10, which carried a much heavier eval. The ablation stage's runs took 0.7 to 3.6 minutes of L4 time each, training and eval together, and all 27 settled in about 25 minutes of wall clock at eight containers.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Ablations

    Nine conditions, three seeds each. Each subsection reads one arm against `ref` on the five decision statistics, and the rule it feeds was fixed before the run.
    """)
    return


@app.cell(hide_code=True)
def _():
    _res = load_results()
    mo.stop(_res is None, mo.md("_The ablation results aren't published yet; the stage-2 cells need them._"))
    assert _res is not None
    ab_metrics, ab_arrays = _res
    ab_cells: list[dict] = ab_metrics["cells"]
    # Re-derived from the published per-run numbers rather than read off the
    # stored summary, so the decision in this report is the frozen rule applied
    # to data the reader can see.
    ab_summary = ex.ablation_summary(ab_cells)
    ab_decision = ex.decide(ab_summary)
    return ab_arrays, ab_cells, ab_decision, ab_summary


@app.cell(hide_code=True)
def _(ab_summary: dict):
    _rows = "".join(
        f"<tr><td><code>{name}</code></td><td class='num'>{s['n']}</td>"
        + "".join(f"<td class='num'>{s[k]:+.4f}</td>" for k in ex.DECISION_STATS)
        + f"<td class='num'>{s['holdout_em']:.4f}</td>"
        + f"<td class='num'>{len(s['latched'])}</td></tr>"
        for c in ex.ABLATION_CONDITIONS
        if (name := c["name"]) and (s := ab_summary[name])
    )
    _head = "".join(
        f"<th class='num'>{k} {'↑' if ex.STAT_DIRECTION[k] == 'up' else '↓'}</th>" for k in ex.DECISION_STATS
    )
    mo.Html(
        figure_html(
            f"""
            <div class="report-table-scroll"><table class="report-table">
            <thead><tr><th>condition</th><th class="num">seeds</th>{_head}
            <th class="num">holdout EM ↑</th><th class="num">latched ↓</th></tr></thead>
            <tbody>{_rows}</tbody>
            </table></div>
            """,
            caption=mo.md("""
            **Every arm, every decision statistic.** Seed means over three runs per condition. Arrows give the direction that counts as better; containment (`alpha_op1`) is the one that improves downward. The subsections below take the differences from `ref` and read each against its band.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Is the anchor schedule needed?

    No. `flat-anchor` holds λ_a at 0.1 from step 0, with no ramp, no end anneal, and no floor. It matches `ref` on every decision statistic, and no run latches. The largest excursion is grading, +0.0361 against a band of 0.0392, and it falls on the better side. Rule 1 fires: the survey runs a flat anchor weight.

    We had predicted the other outcome. Ex-2.1.9 found that the pooled pull commits by epoch 8, inside the 10-epoch warmup, and the `AnchorSpec` docstring says the ramp exists so that the anchor arrives with the optimizer rather than ahead of it. Applying full weight to a still-random model looked like a way to latch a syntax role. It isn't: all six runs put 0.85–0.89 of their deep-slice weight on op1, and the flat arm reaches its margin sooner and then holds it (retention 0.9975).

    The end anneal is the half we were least willing to lose, since M1/ex-2.9.3 found that withdrawing protection entirely lets the task loss reclaim the axis. Flattening does not withdraw it, because the weight stays at full strength through the last step. So that lesson is untouched here.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary: dict, delta_table):
    mo.Html(
        figure_html(
            delta_table(ab_summary, ["flat-anchor"]),
            caption=mo.md("""
            **`flat-anchor` against `ref`.** Seed-mean differences, three runs a side. Bold marks a difference the band can resolve; ✓ and ✗ say which side of `ref` it landed on. The band is 2σ·√(2/3) from the nine-seed noise floors of ex-2.1.10, and the two gate rows carry absolute bars rather than bands.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _(ab_cells: list[dict], traj_of):
    _pi = {
        c["condition"]: [
            np.asarray(r["pi"], dtype=float)[1:].mean(axis=0) for r in ab_cells if r["condition"] == c["condition"]
        ]
        for c in ab_cells
    }
    _traj = {k: traj_of(ab_cells, k, "m_line") for k in ("ref", "flat-anchor", "lam0")}

    @themed(
        name="flat-anchor",
        alt_text="""
            Two panels. Left: per-line margin m_line against epoch for three reference runs, three flat-anchor runs and three un-anchored controls; the anchored curves rise over the first twenty epochs and then hold, while the controls stay near zero. Right: the deep-slice softmin weight each run puts on the four span roles, one dot per run, with a dashed line at the latch threshold of 0.5.
        """,
        caption=r"""
            **What the ramp buys.** **Left:** the m_line trajectory of every run, each on its own epoch axis. It sits above the endpoint value in the tables: the trajectory measures one fixed partner line per color, where the endpoint eval averages all 27. That makes it a cheaper instrument, but a consistent one within a run, which is what retention compares. **Right:** the post-attention mean softmin weight on each span role, one mark per run. The dashed line is the latch veto at $\pi = 0.5$; a run above it on `+` or `=` has committed to a syntax role. The reference and flat arms differ only over the warmup window, so a difference here says when the pull arrives rather than how much of it arrives.
        """,
    )
    def _plot():
        grey = light_dark("#666", "#999")
        colors = {
            "ref": light_dark("#1f6fb4", "#5fa8dd"),
            "flat-anchor": light_dark("#b4531f", "#dd8f5f"),
            "lam0": grey,
        }
        fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
        for name, runs in _traj.items():
            for i, (e, y) in enumerate(runs):
                ax_t.plot(e, y, color=colors[name], lw=1.0, alpha=0.8, label=name if i == 0 else None)
        ax_t.set_xlabel("epoch", fontsize="x-small")
        ax_t.set_ylabel("m_line (trajectory probe)", fontsize="x-small")
        ax_t.legend(fontsize="x-small", frameon=False, loc="center right")
        for name in ("ref", "flat-anchor"):
            for i, p in enumerate(_pi.get(name, [])):
                ax_p.scatter(
                    np.arange(ex.SPAN) + (0.12 if name == "flat-anchor" else -0.12),
                    p,
                    s=22,
                    color=colors[name],
                    alpha=0.8,
                    lw=0,
                    label=name if i == 0 else None,
                )
        ax_p.axhline(ex.LATCH_PI, color=grey, ls=(0, (4, 3)), lw=0.8)
        ax_p.set_xticks(range(ex.SPAN), ex.ROLES)
        ax_p.set_ylim(0, 1)
        ax_p.set_ylabel("deep-slice weight", fontsize="x-small")
        ax_p.legend(fontsize="x-small", frameon=False, loc="upper right")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Does the anti-subspace schedule beat a constant?

    It comes out ahead of both constants, and at each end for a different reason.

    `anti-hold` holds the low ratio for all of training, which removes the early repulsion. The pull then goes where ex-2.1.8 said it would: two of three runs put more than half their deep-slice weight on `+` and trip the latch veto, and the third sits at 0.482, just under. Containment, grading, and the group contrast all come out worse, the contrast by fifty times its band.

    `anti-peak` holds the high ratio instead. The early repulsion does its job: op1 keeps 0.86–0.88 of the deep-slice weight, and the contrast survives at 0.839. The cost is grading, with r² falling from 0.78 to 0.65 and m_line falling with it. Repulsion that never comes down leaves the color response less graded, which is the failure the constraints exist to catch.

    So both ends of the schedule do real work. The peak keeps the pull off the syntax roles while the model is still random, and the anneal down to the hold ratio is what leaves the response graded once it has committed. Rule 2 sends the survey to branch A, the schedule family, with the hold ratio fixed and the peak and the anneal endpoint sampled.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary: dict, delta_table):
    _dose = {
        "ref": ex.anchor_dose(lambda e: ex.anti_weight(e)),
        "anti-hold": ex.anchor_dose(lambda e: ex.ANTI_HOLD_RATIO * ex.SCORING_LAMBDA * np.ones_like(e)),
        "anti-peak": ex.anchor_dose(lambda e: ex.ANTI_PEAK_RATIO * ex.SCORING_LAMBDA * np.ones_like(e)),
    }
    mo.Html(
        figure_html(
            delta_table(ab_summary, ["anti-hold", "anti-peak"]),
            caption=mo.md(f"""
            **The two flat brackets against `ref`.** Read as the table above. The arms bracket the dose of the schedule rather than matching it: `anti-hold` delivers {_dose["anti-hold"] / _dose["ref"]:.2f}× and `anti-peak` {_dose["anti-peak"] / _dose["ref"]:.2f}× of what the schedule delivers. So an arm that holds every statistic while delivering a different dose tells us the dose was not the active ingredient.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Do the flat arms compose?

    Rule 3 does not read this arm: it is read only when rules 1 and 2 both simplify, and rule 2 did not, so nothing here changes the survey. The arm was insurance against an outcome that did not happen, and it cost three runs.

    The numbers are worth a look anyway. `flat-both` fails on all five statistics, and its runs disagree about how: two put their deep-slice weight on `=` and op2, and the third on `+`. So the failure is not even stable across seeds. This is what `anti-hold` does on its own, and that is the reading to prefer, since the flat anchor contributes nothing to it. Pairing `flat-anchor` with the *scheduled* anti term is exactly the arm that passed.

    Note that the recipe the survey runs is `flat-anchor`, a condition that ran here, rather than a combination no arm tested.
    """)
    return


@app.cell(hide_code=True)
def _(ab_decision: dict, ab_summary: dict, delta_table):
    _read = ab_decision["flat_both_read"]
    mo.Html(
        figure_html(
            delta_table(ab_summary, ["flat-both"]),
            caption=mo.md(f"""
            **The composition against `ref`.** Read as the tables above. Rule 3 {"reads this arm, because rules 1 and 2 both simplify" if _read is not None else "does not read this arm: rules 1 and 2 did not both simplify, so there is no composition to adopt"}; the numbers are shown either way, since a survey that publishes only the arms it acted on is not a complete map.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Is 100 epochs buying anything?

    Nothing we can measure. `short` runs the whole recipe in 50 epochs with every keyframe scaled, and lands within band on all five statistics (m_line +0.0014 against a band of 0.0144), at a task cost of 0.0013 against its own control. On the shared clock below, the two lengths trace the same curves.

    Ex-2.1.6 suggested as much when its margins went flat from about epoch 50. That was read off runs scored for something else; here it is measured directly. Rule 4: the survey runs at 50 epochs, halving its cost.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary: dict, delta_table):
    mo.Html(
        figure_html(
            delta_table(ab_summary, ["short"]),
            caption=mo.md("""
            **`short` against `ref`.** Read as the tables above, with one difference: the task gate compares `short` against `short-lam0`, the un-anchored control at the same length, so that whatever the shorter run costs the task is not charged to the anchor.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _(ab_cells: list[dict], traj_of):
    _keys = ("m_line", "val_loss")
    _traj = {k: {c: traj_of(ab_cells, c, k) for c in ("ref", "short")} for k in _keys}

    @themed(
        name="short",
        alt_text="""
            Two line charts sharing a horizontal axis of fraction of training. Left: per-line margin m_line for three hundred-epoch runs and three fifty-epoch runs; both groups rise over the first fifth of training and hold. Right: validation loss for the same runs, falling steeply and then flattening. The two lengths track each other closely on both panels.
        """,
        caption=r"""
            **A hundred epochs against fifty, on a shared clock.** Both panels put fraction of training on the horizontal axis, so the two lengths run the same schedule shape and land at the same place on it. **Left:** m_line, measured with the cheaper one-partner-per-color probe, which is why it sits above the endpoint values in the tables. **Right:** validation loss. If the second half of training consolidates anything, it would show as the hundred-epoch curves pulling away over the right half of the left panel.
        """,
    )
    def _plot():
        colors = {"ref": light_dark("#1f6fb4", "#5fa8dd"), "short": light_dark("#2a9d8f", "#5fc9bd")}
        labels = {"m_line": "m_line (trajectory probe)", "val_loss": "validation loss"}
        fig, _axs = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained", sharex=True)
        ax_m, ax_v = cast(tuple, _axs)
        for ax, key in ((ax_m, "m_line"), (ax_v, "val_loss")):
            for name, runs in _traj[key].items():
                for i, (e, y) in enumerate(runs):
                    ax.plot(e / e.max(), y, color=colors[name], lw=1.0, alpha=0.8, label=name if i == 0 else None)
            ax.set_xlabel("fraction of training", fontsize="x-small")
            ax.set_ylabel(labels[key], fontsize="x-small")
        ax_m.legend(fontsize="x-small", frameon=False, loc="lower right")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Does the anneal shape matter?

    No. Straightening every ramp and anneal in both terms leaves all five statistics within band, the largest excursion being grading at −0.0248 against a band of 0.0392. Minimum-jerk stays, now on evidence rather than by inheritance from M1, and the question retires.

    A straight line and a smoothstep both cover half their window, so the anchor dose barely moves. The anti-subspace ratio interpolates over 90 epochs, and over that span the two shapes differ enough to change its dose by 6%. No statistic resolved: near the operating point the repulsion is insensitive to dose changes of that size, worth holding beside the previous section, where what the flat brackets changed was mostly something other than the dose.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary: dict, delta_table):
    _sched = ex.anchor_dose(lambda e: ex.anti_weight(e))
    _lin = ex.anchor_dose(lambda e: ex.anti_weight(e, shape="linear"))
    mo.Html(
        figure_html(
            delta_table(ab_summary, ["linear"]),
            caption=mo.md(f"""
            **`linear` against `ref`.** Read as the tables above. Straightening the ramps moves the anchor dose by under a percent, since a smoothstep and a straight line both cover half their window. It moves the anti-subspace dose by {abs(_lin - _sched) / _sched:.0%}, because that term interpolates over 90 epochs, and the two shapes differ most over the early part of that window, where the learning rate is highest.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What the survey runs
    """)
    return


@app.cell(hide_code=True)
def _(ab_decision: dict):
    _space = ex.survey_space(ab_decision)
    _rows = "".join(
        f"<tr><td><code>{d}</code></td><td class='num'>{lo:g}</td><td class='num'>{hi:g}</td><td>{scale}</td></tr>"
        for d, (lo, hi, scale) in _space.items()
    )
    mo.md(f"""
    The four rules that touch the survey resolve to one recipe and one box. The anchor weight is flat at the sampled λ_a; the anti-subspace term keeps its schedule, with the hold ratio fixed at {ex.ANTI_HOLD_RATIO:g} and its peak and anneal endpoint sampled; every trial runs {ab_decision["epochs"]} epochs.

    <table class="report-table">
    <thead><tr><th>dimension</th><th class="num">low</th><th class="num">high</th><th>scale</th></tr></thead>
    <tbody>{_rows}</tbody>
    </table>

    That is branch {ab_decision["anti_branch"]} of the frozen plan, so the trial list is the one shown under *The survey space* above: {ex.N_TRIALS} scrambled-Sobol points at seed {ex.SOBOL_SEED}, run at one seed each, with the top {ex.N_PROMOTE} feasible trials promoted to {ex.SEEDS_PROMOTE}.

    Two of the ablations paid for themselves in survey cost rather than in dimensions. `short` halves every trial, and `flat-anchor` removes four parameters that would otherwise have been carried untested.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The landscape

    *What the survey delivers: the map, then the point.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | TODO
    Four views, all built from the published per-trial table.

    1. The full trial table: sampled coordinates, every statistic, feasibility against each constraint, and round-2 seed means where a trial was promoted. No trial is omitted.
    2. Marginals: m_line and each constraint statistic against each dimension, with the branch A anti axes shown in dose and centroid coordinates, and feasible trials distinguished from infeasible ones. We read off where the feasible region sits and how flat m_line is across it; a wide plateau is itself the result D2.2 wants.
    3. The m_line against r² plane with the constraint boundary drawn, showing what the grading constraint excluded. Alongside the per-color r², we report the conditional-mean r² over redness levels. The ex-2.1.10 review found that the two together separate graded on average from graded color by color, and only the per-color reading constrains anything.
    4. The λ_a marginal on its own. Ex-2.1.7 predicts a selectivity optimum below λ_a = 1, and where the containment and grading constraints actually bind along λ_a will inform the new operating point.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Proposed operating point

    /// admonition | TODO
    One line: the coordinates and seed-mean statistics of the winning trial, each with its noise floor. Beside it, the values for the reference recipe and the width of the plateau it sits on. Then the handoff: which experiment confirms the point, and at how many fresh seeds, before any of these numbers is quoted.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    *Post hoc: noticed after the ablation results, not part of the frozen plan.*

    ### The latch veto fires on an un-anchored control

    One of the three `lam0` runs puts 0.55 of its deep-slice weight on `+`, above the 0.5 threshold of the veto. The detector is reading a profile that only means something when there is a pull. With λ_a = 0, the softmin weights describe whatever alignment structure the task happens to leave behind, and there is no pull that could have latched. The veto is calibrated on the anchored runs of ex-2.1.9, and the controls were never going to be judged by it, so nothing here is affected.

    It does bear on the survey. λ_a runs down to 0.02, a fifth of the scored rung, and a trial that weak may look like the control rather than like a pull that committed. Such a trial would be marked infeasible for a reason other than the one the veto is named for. It would fail the contrast constraint too, so the error goes in the safe direction: we would reject a trial that was never going to be proposed. We are leaving the veto frozen and reporting the λ_a of every vetoed trial, so the two readings can be told apart in the map.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    Interpretation only, once the observations exist: what the ablations say about carrying the M1 schedules into this architecture, what the map says about how carefully M3 will need to tune, and what the survey format itself cost or saved. This is the first time we've used that format, so note any friction for the science skill.
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


@app.cell(hide_code=True)
def _():
    # Shared by the ablation sections: every arm is read against `ref` the same
    # way, through the frozen `experiment.arm_verdict`.
    MARK = {"within band": "", "better": " ✓", "regression": " ✗"}

    def _cell(text: str, resolved: bool) -> str:
        return f"<td class='num'>{f'<b>{text}</b>' if resolved else text}</td>"

    def delta_table(summary: dict, arms: list[str], ref: str = "ref") -> str:
        """Each arm against *ref*, one row per decision statistic, plus the two gates."""
        v = {a: ex.arm_verdict(summary, a, ref) for a in arms}
        rows = []
        for s in ex.DECISION_STATS:
            arrow = "↑" if ex.STAT_DIRECTION[s] == "up" else "↓"
            stats = [v[a]["stats"][s] for a in arms]
            cells = "".join(
                _cell(f"{d['delta']:+.4f}{MARK[d['verdict']]}", d["verdict"] != "within band") for d in stats
            )
            rows.append(
                f"<tr><td><code>{s}</code> {arrow}</td>"
                f"<td class='num'>{summary[ref][s]:+.4f}</td>"
                f"<td class='num'>{ex.equiv_band(s):.4f}</td>{cells}</tr>"
            )
        ref_cost = abs(summary[ref]["holdout_em"] - summary[ex.CONTROL_OF[summary[ref]["epochs"]]]["holdout_em"])
        rows.append(
            f"<tr><td>holdout EM − control ↓</td><td class='num'>{ref_cost:.4f}</td>"
            f"<td class='num'>{ex.TASK_GATE:.4f}</td>"
            + "".join(_cell(f"{v[a]['task_cost']:.4f}", not v[a]["task_ok"]) for a in arms)
            + "</tr>"
        )
        rows.append(
            f"<tr><td>latched runs ↓</td><td class='num'>{len(summary[ref]['latched'])}</td><td class='num'>—</td>"
            + "".join(_cell(str(len(v[a]["latched"])), bool(v[a]["latched"])) for a in arms)
            + "</tr>"
        )
        head = "".join(f"<th class='num'>{a}</th>" for a in arms)
        return f"""
        <div class="report-table-scroll"><table class="report-table">
        <thead><tr><th>statistic</th><th class="num">{ref}</th><th class="num">band</th>{head}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
        </table></div>
        """

    def traj_of(cells: list[dict], condition: str, key: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """(epoch, value) per run of *condition*, on that run's own epoch axis."""
        return [
            (np.asarray(c["traj"]["epoch"], dtype=float), np.asarray(c["traj"][key], dtype=float))
            for c in cells
            if c["condition"] == condition
        ]

    return delta_table, traj_of


if __name__ == "__main__":
    app.run()
