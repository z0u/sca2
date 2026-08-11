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
    A survey to close out D2.1. The schedules and weights in the recipe were inherited piece by piece and never tuned together, so we ablate first: flat arms that bracket each schedule, an arm that flattens both at once, and a half-length condition. Every "no difference" lets us drop a dimension.

    Then a Sobol search over what is left (anchor weight, pooling temperature, repulsion dose) maps the good region and proposes the operating point the next milestone inherits.

    Outcome: the anchor schedule and half the training could go, but the anti-subspace schedule could not. A constant that delivers the same total dose loses the grading the schedule protects, so the shape itself is doing the work. The search found a wide feasible plateau and proposes a stronger, softer point on its edge, for D2.2 to confirm.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Observations

    **The reference recipe reproduces.** `ref` re-runs the primary condition of ex-2.1.10 through the code path of this experiment. It lands within one per-run standard deviation on every statistic: m_line 0.4204 against 0.4202 (σ 0.0088), group contrast 0.8562 against 0.8542 (σ 0.0057), holdout exact match 0.9961 against 0.9948 (σ 0.0087).

    **The anchor schedule can go.** `flat-anchor` stays within band on all five decision statistics, and no run latches (m_line +0.0039, band 0.0144). Rule 1: the survey runs a flat anchor weight.

    **The anti-subspace schedule cannot.** Both flat brackets come out worse. `anti-hold` loses the group contrast (−0.4667, band 0.0093), grading (−0.2199, band 0.0392), and containment (+0.1275, band 0.0346), and two of its three runs latch onto `+`. `anti-peak` keeps containment and latches nothing, but it loses grading (−0.1283, band 0.0392), m_line (−0.0282, band 0.0144), and a little contrast (−0.0174, band 0.0093). Rule 2: the survey searches the schedule family, branch A.

    <!-- REVIEW: `anti-peak` was summarized as keeping contrast; its contrast delta is −0.0174 against a band of 0.0093, so the rules read it as a resolved regression, not a tie. The branch is unaffected (m_line and grading already resolve worse). Verify: the `anti-peak` column of the delta table marks the contrast row bold with ✗. -->

    **Half the training is enough.** `short` stays within band on every statistic (m_line +0.0014), and costs 0.0013 on the task against its own control. Rule 4: every survey trial runs 50 epochs.

    **Nothing resolves on the interpolation shape.** `linear` stays within band throughout; its largest excursion is grading, at −0.0248 against a band of 0.0392. Rule 5: keep minimum-jerk, and the question retires.

    **What matters about the anti-subspace schedule is its shape, not its dose.** `anti-mid` is a constant that matches the dose of the schedule, and it still loses grading (−0.1022, band 0.0392), m_line, and contrast. That is the pattern the shape reading of rule 6 predicted, so branch A is confirmed. The flat pairing `anti-mid-flat` fails the same way against `flat-anchor`, so flattening the anti term costs about the same with either anchor shape.

    <!-- REVIEW: two wording fixes in the line above. "delivers the same dose as a constant" read as if `anti-mid` were not itself the constant; it matches the *schedule's* dose. And "there is no interaction with the anchor shape" was stronger than the 2 x 2 supports at three seeds a corner — the analysis section says the two lines move nearly in parallel, which is what this line now says. No verdict changes; rule 7 is unread either way. -->


    **A quarter of the training is too little.** `short25` fails on m_line (−0.0268, band 0.0144), containment, and contrast. The predicted failure, in which the un-anchored control loses the task, did not happen: the task gate holds, using 0.0026 of its 0.02. Rule 8: 50 epochs stands.

    **The survey maps the trade the calibration predicted.** 38 of 64 trials are feasible. m_line rises with λ_a until the task gate binds (every task failure at λ_a ≥ 0.38), and with τ until grading and contrast reach their floors. The latch veto accounts for the cold edge, with all ten vetoes at τ ≤ 0.12. The anti-subspace axes organize little.

    **Proposed operating point.** `t00` sets λ_a 0.557, τ 0.265, and an anti peak of 1.376 with anneal end 0.537. It reaches m_line 0.534 ± 0.004 over five seeds, against 0.420 ± 0.005 for the reference recipe. It sits at the grading floor (r² 0.690 against 0.682), with four more trials within one band spanning λ_a 0.28–0.94. D2.2 confirms at fresh seeds before any of these numbers is quoted.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this report
    This is a survey, not a hypothesis-scoring experiment. It preregistered a *search plan* and scores nothing. The plan covers the space to search, the sampling rule, the objective and its constraints, the seed budget, and the decision rules that stage the work. It was frozen before the runs, and amended once mid-experiment under rules that were themselves frozen before their arms ran (the amendment section records how). Results replaced the placeholders in place, so each analysis section reads as prediction then observation, and anything conceived after seeing the data sits under *Exploratory analyses*, marked as post hoc.
    <!-- REVIEW: retitled from "How to read this draft" and moved to past voice now that the results are in; the frozen-plan claims are unchanged, only their tense. Verify: git history shows the plan text predating the runs. -->

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
            Scatter chart of grading r squared against per-line margin m_line. Five clusters of small dots, one per ex-2.1.10 condition, each with a larger ringed mark at its seed mean. The reference arm, the primary and the slot oracle cluster at the upper left, near m_line 0.4 and r squared around 0.8. The two soft-tau arms sit to the lower right, near m_line 0.49 and r squared 0.65 and 0.58: margin rises as grading falls along the tau axis.
        """,
        caption=r"""
            **The trade the constraints must hold: margin buys smear at soft τ.** Per-run grading $r^2$ against per-run m_line for ex-2.1.10's five anchored conditions (small marks; large ring = seed mean). Softening τ (0.1 → 0.5 → 2.5, left to right) buys margin and spends grading; ranking on m_line alone would pick the right-hand end. The survey caps that walk with the grading and contrast constraints instead of a second objective.
        """,
    )
    def _plot():
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

    The bounds [0.05, 0.30] carry that leading weight from 0.78 down to 0.57 at the embedding slice, and from 0.90 to 0.67 averaged over the deeper ones. So the survey range spans the target band from ex-2.1.9 — 0.6 to 0.8 — near enough end to end at the embedding slice, which is where that band was set.

    Below τ ≈ 0.05 the curve is flat: it saturates at 0.78 rather than climbing to 1, because the profile the pooling is applied to has a spread that a colder temperature runs out of room to sharpen. Nothing is given up by holding the lower bound above the latching regime. The upper bound stops short of τ = 0.5, which the trade figure above shows is already smeared.

    One thing this curve cannot do is predict a run trained at a different τ. It applies a hypothetical temperature to the profile of a model trained at 0.1, so it says what the pooling would weight given that model, and the latching at τ ≲ 0.03 is ex-2.1.9's observation from runs rather than anything visible here.
    <!-- REVIEW: this paragraph and the figure below were computed on different
    statistics, and the figure's was not the one the 0.6-0.8 band belongs to. It
    plotted the line-weighted mean of the per-line maximum; ex-2.1.9's band is on
    `pi`, the label-affinity-weighted per-colour profile maxed over roles, which
    is also what the latch veto reads. Averaging over a colour's partners before
    taking the maximum is the whole difference, and it is worth about 0.13 at
    τ = 0.1 (0.90 against 0.77). The figure now plots `pi`, and its reference
    value at τ = 0.1 reproduces ex-2.1.9's published lead_emb of 0.77.

    Both this round's numbers and the previous round's note were derived from the
    wrong quantity: the paragraph said [0.74, 0.42] and that reaching 0.8 needed
    τ ≈ 0.04, and the note narrowed the claim to "the lower part of the band" on
    that basis. On `pi` the range is [0.78, 0.57] and 0.8 is not reachable at any
    τ. This widens the earlier note rather than reversing it — the bounds are
    unchanged and better placed than the note credited. Verify: the reference
    line at τ = 0.1 in the figure below should read 0.77 at the embedding slice. -->
    """)
    return


@app.cell(hide_code=True)
def _(prior, softmin_weights_np):
    _m10: dict = prior["m10"]
    _labels = [c["label"] for c in _m10["cells"] if c["condition"] == "either-t100"]
    _alpha = np.mean([prior["a10"][f"{la}/alpha_lines"] for la in _labels], axis=0)  # (L1, N, T), seed mean
    _w_color = prior["probes"]["weights"]
    _n_part = _alpha.shape[1] // len(_w_color)
    _x = 1.0 - _alpha[:, :, : ex.SPAN]
    _taus = np.geomspace(0.01, 1.0, 60)
    # ex-2.1.9's `pi`: average each colour's partner lines first, then take the
    # leading role. Averaging before the maximum is what makes this the quantity
    # the 0.6-0.8 band was set on, rather than the per-line maximum.
    _lead = np.array(
        [
            np.einsum(
                "lct,c->lt",
                softmin_weights_np(_x, t).reshape(_x.shape[0], len(_w_color), _n_part, ex.SPAN).mean(axis=2),
                _w_color,
            ).max(axis=-1)
            for t in _taus
        ]
    )  # (τ, L1)

    @themed(
        name="tau-range",
        alt_text="""
            Line chart of leading softmin weight against temperature tau on a log scale from 0.01 to 1. Five curves, one per residual slice, are flat at about 0.77 to 0.92 on the left, then fall away from about tau 0.07 to reach 0.3 to 0.4 at the right. A shaded vertical band marks the survey range 0.05 to 0.3, and a shaded horizontal band marks the target leading-weight range 0.6 to 0.8. The embedding-slice curve runs through the target band across most of the survey range; the deeper slices sit above it until the right-hand edge. A dashed vertical line marks the reference tau of 0.1.
        """,
        caption=r"""
            **Where the survey's τ bounds sit on the weight scale.** The label-affinity-weighted share of softmin weight held by the leading span role, per residual slice, on the primary's seed-mean alignment profile — ex-2.1.9's $\pi$, which is also what the latch veto reads. The lowest curve is the embedding slice, where ex-2.1.9 set its band; it passes through τ = 0.1 at 0.77, reproducing that experiment's published value. Vertical shading: the survey range [0.05, 0.30]; horizontal shading: ex-2.1.9's 0.6–0.8 target band; dashed line: the reference τ = 0.1. The flat left-hand end is the profile's own spread, which a colder temperature runs out of room to sharpen.
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
    ### Ablation conditions

    There are nine conditions, three seeds each, all at the reference λ_a and τ under the either-slot labeller. `ref` re-runs the primary recipe from ex-2.1.10 through the code path of this experiment, so every comparison is between runs of identical code. `lam0` and `short-lam0` are the task-cost controls, one at each length. `flat-both` combines the two flat simplifications. We need it because an arm that changes one schedule at a time cannot show an interaction between the two; we read it only if decision rules 1 and 2 both simplify. The table reports dose and centroid for each arm, so the outcomes can tell us which of the two was doing the work.

    Four more conditions were added once these had settled, under rules frozen in turn before they ran; they are set out under *An amendment to the search plan* below, with the reasoning that motivated them.
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
    ### Decision rules

    Two names recur from here on: **branch A** keeps the anti-subspace term's schedule shape, sampling its peak and anneal endpoint; **branch B** replaces it with one flat ratio. Rule 2 below picks between them, and the next section gives each branch's own dimensions.

    <!-- REVIEW: added the branch A/B definition above the frozen rule text. Both names were already defined inside rule 2's own wording and restated in the next section, but only after their first appearance in the ablation and amendment prose further down; a reader hitting "branch A" there had to backtrack through two sections to find it. No claim changes — this states what rule 2 already decides, ahead of the frozen block that decides it. -->

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
    ### Survey space

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
            **The frozen trial lists**, scrambled-Sobol points at seed {ex.SOBOL_SEED}, shown for review before any run: {ex.n_trials({**ex.SPACE_COMMON, **ex.SPACE_ANTI_SCHED})} for branch A's four dimensions, {ex.n_trials({**ex.SPACE_COMMON, **ex.SPACE_ANTI_FLAT})} for branch B's three. Branch A adds the derived (relative dose, centroid) coordinates its marginals will be reported in; relative dose is the trial's $\int (\lambda_{{\bar{{s}}}}/\lambda_a) \cdot \mathrm{{lr}} \, \mathrm{{d}}e$ over the operating point's.
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

    **Rounds.** Round 1 runs every trial at seed 0, which the noise floors allow. The {ex.N_PROMOTE} feasible trials with the highest m_line are promoted to {ex.SEEDS_PROMOTE} seeds. If fewer are feasible, we promote all of them; if none are, we publish the infeasibility map and propose nothing.

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
    {len(ex.ABLATION_CONDITIONS)} ablation conditions × {ex.N_SEEDS_ABLATION} seeds = {ex.N_RUNS_ABLATION} runs, plus the amendment's {len(ex.AMENDMENT_CONDITIONS)} × {ex.N_SEEDS_ABLATION} = {ex.N_RUNS_AMENDMENT}, then at most {max(ex.N_TRIALS_BY_DIM.values())} + {ex.N_PROMOTE} × {ex.SEEDS_PROMOTE - 1} = {ex.N_RUNS_SURVEY} survey runs: about {ex.N_RUNS_ABLATION + ex.N_RUNS_AMENDMENT + ex.N_RUNS_SURVEY} training runs in the worst case.

    The design budgeted roughly 25 minutes of L4 per run, inherited from ex-2.1.10. The ablation stage then billed 0.7–3.6 minutes each, because the slim eval drops the readability, geometry and leakage passes and the corpus is prepared once for the whole DAG. At the observed rate the experiment fits in under two hours of wall clock.

    Each stage launches with `--max-containers {ex.MAX_CONTAINERS}` and a `--budget` sized from its run count. Memoization means the promoted round re-runs only the new seeds, and `ctx.map` with `allow_partial=True` lets diverged trials land as data rather than failures.

    <!-- Observed after the run, not a design change. -->
    The 25-minute figure was inherited from ex-2.1.10, which carried a much heavier eval. The ablation stage's runs took 0.7 to 3.6 minutes of L4 time each, training and eval together, and all 27 settled in about 25 minutes of wall clock at eight containers. The full DAG — all 127 training runs plus the prep, eval and publish steps — billed about $3.60 of compute, nearly all of it L4, in 2.8 hours of wall clock.
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
    ab_decision = ex.decide_amended(ab_summary)
    return ab_arrays, ab_cells, ab_decision, ab_summary


@app.cell(hide_code=True)
def _(ab_summary):
    _rows = "".join(
        f"<tr><td><code>{name}</code></td><td class='num'>{s['n']}</td>"
        + "".join(f"<td class='num'>{s[k]:+.4f}</td>" for k in ex.DECISION_STATS)
        + f"<td class='num'>{s['holdout_em']:.4f}</td>"
        + f"<td class='num'>{len(s['latched'])}</td></tr>"
        for c in ex.ALL_CONDITIONS
        if (name := c["name"]) in ab_summary and (s := ab_summary[name])
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

    No. `flat-anchor` holds λ_a at 0.1 from step 0, with no ramp, no end anneal, and no floor. Every decision statistic lands inside its band, and no run latches. The largest excursion is grading, +0.0361 against a band of 0.0392, and it falls on the better side, so at three seeds a side this says the schedule is not doing anything we can resolve rather than that it does nothing. Rule 1 fires: the survey runs a flat anchor weight.

    <!-- REVIEW: "matches `ref` on every decision statistic" → "lands inside its band", with the resolution reading spelled out. Two of the five deltas sit at 78% and 92% of their band, so "matches" read the resolution limit as evidence of equality. Verify: the delta table prints every delta beside its band. -->

    We had predicted the other outcome. Ex-2.1.9 found that the pooled pull commits by epoch 8, inside the 10-epoch warmup, and the `AnchorSpec` docstring says the ramp exists so that the anchor arrives with the optimizer rather than ahead of it. Applying full weight to a still-random model looked like a way to latch a syntax role. It isn't: all six runs put 0.85–0.89 of their deep-slice weight on op1, and the flat arm reaches its margin sooner and then holds it (retention 0.9975).

    The end anneal is the half we were least willing to lose, since M1/ex-2.9.3 found that withdrawing protection entirely lets the task loss reclaim the axis. Flattening does not withdraw it, because the weight stays at full strength through the last step. So that lesson is untouched here.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary, delta_table):
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
def _(ab_cells, traj_of):
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

    It comes out ahead of both constants.

    <!-- REVIEW: checked what these two paragraphs are read off. Every number in
    them is a seed-mean, end-of-training statistic from the same per-run fields
    the delta table below reads (deep-slice `pi`, `alpha_op1`, `r2_sim`,
    `contrast`) — there is no trajectory for these arms, and none is claimed:
    `anti-peak`'s op1 share is 0.858/0.868/0.884 across its three runs (published
    `pi`), matching "0.86-0.88", and its `r2_sim` is 0.61/0.66/0.69 against
    `ref`'s 0.79/0.77/0.78, matching "0.78 to 0.65". Trimmed one sentence that
    read as a timing claim ("the early repulsion does its job") without a
    trajectory behind it; the rest states what the bracket comparison shows,
    not what happens during training. Verify: the numbers above against the
    published per-run `pi` and `r2_sim` for `anti-hold`/`anti-peak`/`ref`. -->

    `anti-hold` holds the low ratio for all of training, which removes the early repulsion (the schedule's min-jerk anneal is slow to start, so its ratio still sits at 2.47 of an eventual 0.30 at epoch 10). The pull then goes where ex-2.1.8 said it would: two of three runs put more than half their deep-slice weight on `+` and trip the latch veto, and the third sits at 0.482, just under. Containment, grading, and the group contrast all come out worse, the contrast by fifty times its band.

    `anti-peak` holds the high ratio instead: op1 keeps 0.86–0.88 of the deep-slice weight, and the contrast holds at 0.839, which is 0.0174 under `ref`: a small drop, but nearly twice the band, so it resolves. The cost is grading, with r² falling from 0.78 to 0.65 and m_line falling with it. Repulsion that never comes down leaves the color response less graded, which is the failure the constraints exist to catch.

    So both ends of the schedule seem to be required: the peak keeps the pull off the syntax roles while the model is still random, and the anneal down to the hold ratio leaves the response graded once it has committed. That is one reading — both brackets change the dose along with the shape, and the amendment below adds the arm that separates the two. On what ran here, rule 2 sends the survey to branch A, the schedule family, with the hold ratio fixed and the peak and the anneal endpoint sampled.

    <!-- REVIEW: the shape reading was stated here as settled while the
    amendment section below says round 1 could not settle it (the brackets
    confound shape with dose). Marked it as one of two live readings and added
    the forward pointer; rule 2's firing and the branch it names are unchanged.
    The "before its anneal" clause added in the previous pass was also made
    precise: the anti anneal spans epochs 0–90, so nothing is delivered
    "before" it — what protects the early window is the min-jerk shape's slow
    start (ratio 2.47 at epoch 10, from `anti_weight`). -->
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary, delta_table):
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

    `flat-both` fails on all five statistics. Rule 3 never reads it: the rule fires only when rules 1 and 2 both simplify, and rule 2 did not, so nothing here changes the survey either way. The arm was insurance against an outcome that did not happen, and it cost three runs.

    <!-- REVIEW: reordered so the result comes before the rule. The section
    previously opened with "rule 3 does not read this arm", which a reader can
    take to mean the arm was never run. It ran; it is unread and it also fails.
    No claim changes. -->

    The numbers are worth a look anyway. Two runs put their deep-slice weight on `=` and op2, and the third on `+`. So the failure is not stable across seeds — and it is not quite `anti-hold`'s failure either: that arm sent all three of its runs to `+` and kept retention within band, where this one regresses it. What survives the comparison is that the hold-level anti term fails with either anchor shape; whether the anchor's shape also steers where the pull settles is a three-seed observation with nothing riding on it.

    <!-- REVIEW: corrected "this is what anti-hold does on its own, and the flat
    anchor contributes nothing to it" against the published per-run profiles.
    anti-hold's failure is stable — all three runs put their deep-slice weight
    on `+` (0.54/0.63/0.48) — while flat-both's wanders (`=`/op2, `=`/op2, `+`;
    none over the 0.5 veto) and it adds a retention regression (−0.079 against
    a 0.010 band) that anti-hold does not have. So the flat anchor changes how
    the failure is expressed, and the old sentence claimed otherwise. Nothing
    decision-bearing changes: the arm is unread either way. Verify: the six
    runs' `pi` fields, roles ordered [op1, +, op2, =]. -->

    <!-- REVIEW: reworded for clarity; no claim changes. `flat-anchor` only
    flattens the anchor term (`condition_shapes` leaves its anti-subspace shape
    at the schedule default), so it already is flat-anchor-plus-scheduled-anti
    — the same pairing rule 2's branch A sends the survey to. -->
    Rule 1 (flat anchor) and rule 2 (branch A, scheduled anti) together land on a pairing that already ran: `flat-anchor` only flattened the anchor term, leaving the anti-subspace term on its schedule, so the survey's recipe is that tested condition, not a combination assembled after the fact from separate arms.
    """)
    return


@app.cell(hide_code=True)
def _(ab_decision, ab_summary, delta_table):
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
def _(ab_summary, delta_table):
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
def _(ab_cells, traj_of):
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

    <!-- REVIEW: added the sentence below to answer "if linear passes, why not
    switch to it". Rule 5 was written to keep minimum-jerk on either outcome
    (DECISION_RULES: "Either way no survey dimension"), unlike rules 1-3, where
    a pass deletes schedule parameters the survey would otherwise carry. There
    is no cost saved by switching, so a pass only closes the question of
    whether the shape is load-bearing rather than arguing for the simpler one. -->
    Rule 5 was written to keep minimum-jerk whichever way this arm landed. Unlike the flat-arm rules, swapping the interpolation shape deletes no survey dimension, so a pass here only closes the question of whether the shape matters — it is not a case for adopting the simpler curve.

    A straight line and a smoothstep both cover half their window, so the anchor dose barely moves. The anti-subspace ratio interpolates over 90 epochs, and over that span the two shapes differ enough to change its dose by 6%. No statistic resolved: near the operating point the repulsion is insensitive to dose changes of that size, worth holding beside the previous section, where what the flat brackets changed was mostly something other than the dose.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary, delta_table):
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
    mo.md(rf"""
    ### An amendment to the search plan

    Everything above ran under frozen rules. This section adds four conditions and three more rules, also frozen before any of *them* ran. Rules 1–5 stand exactly as they fired; the amendment extends the plan where round 1 turned out not to answer the question we had asked it.

    **What round 1 could not settle.** Rule 2 read two constants, at {ex.ANTI_HOLD_RATIO:g} and {ex.ANTI_PEAK_RATIO:g}, and both resolved worse than `ref`, which we read as the schedule earning its shape at both ends. But those two arms differ from `ref` in two ways at once: in shape, and in dose ({ex.anchor_dose(lambda e: ex.anti_weight(e, hold_ratio=ex.ANTI_HOLD_RATIO, shape="flat")) / ex.anchor_dose(lambda e: ex.anti_weight(e)):.2f}× and {ex.anchor_dose(lambda e: ex.anti_weight(e, hold_ratio=ex.ANTI_PEAK_RATIO, shape="flat")) / ex.anchor_dose(lambda e: ex.anti_weight(e)):.2f}× the reference).

    So alternatively, it could be that total dose has an interior optimum, and the shape is incidental. The seed means fit it too. Containment moves monotonically with dose across the three arms, and `ref` sits between the two brackets while beating both on the other three statistics.

    The design anticipated this middle level and set it aside. The frozen note under `ABLATION_CONDITIONS` says the anti term "cannot be dose-matched flat (the match would sit at 1.7× the anchor peak for all of training, a different regime)".

    Perhaps that was the right call with no data in hand. But following round 1, this level is now a measurement that could separate two live readings, rather than one more point in a range.

    **The four conditions.** `anti-mid` holds λ_s̄/λ_a constant at {ex.ANTI_MID_RATIO:g}, the level that matches the reference dose, so it differs from `ref` in shape alone. `anti-mid-flat` pairs that constant with the flat anchor. Since `ref` and `flat-anchor` have already run, the four cells complete a 2 × 2 in (anchor shape) × (anti shape), which is what it takes to read an interaction.

    `short25` halves the adopted length again, and `short25-lam0` is its task-cost control, since un-anchored holdout EM is itself a function of length.

    **Why this does not spend the preregistration.** A survey scores nothing. Its output is a proposal, which the next preregistered experiment confirms at fresh seeds, so there is no verdict here for a late change to manufacture. The risks specific to this experiment type are selective reporting and quoting the numbers from a survey as results, and adding arms touches neither, as long as every arm is published.

    What protects the rest is ordinary freezing, done a second time: the rules below were fixed before the runs, they name their predictions, and rules 1–5 keep their own record. `decide` reads round 1 alone whatever else has landed, so the amendment shows up as a diff rather than an overwrite.

    One cost, uncorrected. Rule 6 is a third constant tested against a rule that simplifies on a tie, so it is a third chance to simplify by luck. We are leaning on the survey contract rather than on the risk being small: a spurious pass buys a slightly wrong search space, and D2.2 re-measures the operating point at fresh seeds. Branch A is the more expensive branch, so a spurious pass would displace the incumbent that costs *more* to search.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _anti(c: dict):
        _e: int = c["epochs"]
        _shape = ex.condition_shapes(c)[1]
        _w = lambda e: ex.anti_weight(  # noqa: E731
            e, lam=c["lam"], epochs=_e, shape=_shape, hold_ratio=c.get("anti_ratio", ex.ANTI_HOLD_RATIO)
        )
        return ex.anchor_dose(_w, epochs=_e), ex.dose_centroid(_w, epochs=_e)

    _ref_dose, _ = _anti(next(c for c in ex.ABLATION_CONDITIONS if c["name"] == "ref"))
    _what = {
        "anti-mid": f"λ_s̄/λ_a constant at the dose-matched level ({ex.ANTI_MID_RATIO:g})",
        "anti-mid-flat": "the same constant, paired with the flat anchor",
        "short25": "the whole recipe in a quarter of the epochs",
        "short25-lam0": "un-anchored control at 25 epochs",
    }
    _rows = "".join(
        f"<tr><td><code>{c['name']}</code></td><td>{_what[c['name']]}</td>"
        f"<td class='num'>{c['epochs']}</td>"
        f"<td class='num'>{'—' if c['lam'] == 0 else format(_anti(c)[0] / _ref_dose, '.2f')}</td>"
        f"<td class='num'>{'—' if c['lam'] == 0 else format(_anti(c)[1] / c['epochs'], '.2f')}</td></tr>"
        for c in ex.AMENDMENT_CONDITIONS
    )
    mo.Html(
        figure_html(
            f"""
            <table class="report-table">
            <thead><tr><th>condition</th><th>what it changes</th><th class="num">epochs</th>
            <th class="num">anti dose vs <code>ref</code></th><th class="num">centroid</th></tr></thead>
            <tbody>{_rows}</tbody>
            </table>
            """,
            caption=mo.md(r"""
            **The amendment's conditions**, three seeds each. Dose is $\int \lambda_{\bar{s}} \cdot \mathrm{lr} \, \mathrm{d}e$ divided by the same integral for the reference schedule, at the same length; centroid is where that dose is delivered, as a fraction of training. `anti-mid` and `anti-mid-flat` match the reference dose by construction, and differ from it in *when* the repulsion arrives: the reference delivers it at a centroid of 0.27, a constant at 0.34. That is the contrast rule 6 turns on. The two short arms carry the reference shape unchanged, so their doses scale with length rather than moving independently.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    **The rules** (from `experiment.AMENDMENT_RULES`, frozen before the runs).

    **6.** `anti-mid` against `ref`. Within band, or better → the dose is the active ingredient, the anti schedule collapses to a flat ratio, and the survey searches branch B with {ex.ANTI_MID_RATIO:g} as its reference level. Resolved worse → the shape is load-bearing, and branch A is confirmed rather than merely un-overturned. This comparison holds the anchor schedule fixed at `ref`'s, so a regression here isolates the anti-subspace shape from any interaction with the anchor's shape; rule 7 tests that interaction separately, and confirming branch A does not wait on it.

    **7.** `anti-mid-flat` against `flat-anchor`. Rules 1 and 6 are one-factor readings, so both simplifications are adopted together only if the pair holds. This is the interaction gate of rule 3, now at a level that might pass it. If rule 6 simplifies and this arm does not, the two simplifications interact the way rule 3 describes: we keep the anchor schedule, since rule 1's parameters are fixed either way, and adopt the flat anti at the level rule 6 already showed works alone.

    **8.** `short25` within band against `ref`, task-gated against `short25-lam0` → the survey runs at {ex.EPOCHS_SHORTER} epochs. Otherwise the {ex.EPOCHS_SHORT} of rule 4 stands.

    **What each reading predicts.** A constant at {ex.ANTI_MID_RATIO:g} delivers about 0.7× the repulsion of the schedule over the window where the latch is decided, and four to six times its late repulsion over the window where grading is learned. So the shape reading predicts m_line and grading below band with containment intact, a weakened blend of both round-1 failures. The dose reading predicts every statistic within band.

    Grading is the statistic to watch: it is what the M1 schedules bought, and it is the one both round-1 brackets lost.

    `flat-both` already speaks to half of this. It holds the anchor at full strength from step 0 and still fails the way `anti-hold` does, so the flat anti fails at the low end for some reason other than the anchor being weak early. The high end is where the 2 × 2 is still open, and rule 7 closes it.

    For rule 8, the trap runs the opposite way to the usual one. The margin peaks near epoch 10 and drifts down over the following forty, so a shorter run can score *better* on m_line while being the less converged model. The task gate and grading decide this arm; an m_line improvement on its own does not carry it.
    """)
    return


@app.cell(hide_code=True)
def _():
    _n_a, _n_b = ex.N_TRIALS_BY_DIM[4], ex.N_TRIALS_BY_DIM[3]

    @themed(
        name="amendment-flow",
        alt_text=f"""
            Flowchart of amendment rules 6 to 8. A start box, round 1: flat anchor, scheduled anti, 50 epochs, feeds two lanes. Left lane: rule 6 compares anti-mid to ref. Resolved worse leads to branch A confirmed: scheduled anti, 4 dimensions, {_n_a} trials. Within band leads to rule 7, which compares anti-mid-flat to flat-anchor. There, within band leads to flat anchor with flat anti at {ex.ANTI_MID_RATIO:g}, 3 dimensions, {_n_b} trials; resolved worse leads to scheduled anchor with flat anti at {ex.ANTI_MID_RATIO:g}, also 3 dimensions, {_n_b} trials. Right lane: rule 8 compares short25 to ref, task-gated against short25-lam0. Both passing leads to 25 epochs; either failing leads to 50 epochs under rule 4.
        """,
        caption="""
            **How rules 6–8 compose.** Each arrow is one frozen comparison, read once the amendment arms settle. The left lane decides what the survey searches: rule 6 sets the anti-subspace branch, and rule 7 — read only if rule 6 simplifies — decides whether the flat anchor and the flat anti are adopted together. The right lane decides how long each trial trains, and composes with any left-lane outcome. Grey boxes are outcomes; each names the surviving configuration and the trial budget its dimension count buys.
        """,
    )
    def _plot():
        ink = light_dark("#333", "#ccc")
        edge = light_dark("#555", "#aaa")
        dec_fc, dec_ec = light_dark("#e9f1f8", "#1b2733"), light_dark("#1f6fb4", "#5fa8dd")
        out_fc, out_ec = light_dark("#f2f2f2", "#24272c"), light_dark("#999", "#666")

        fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_axis_off()

        def box(x, y, text, fc, ec):
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=ink,
                linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.55", facecolor=fc, edgecolor=ec, lw=1.1),
            )

        def arrow(x0, y0, x1, y1, label=None, lx=0, ly=0):
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.0, shrinkA=0, shrinkB=0),
            )
            if label:
                ax.text(
                    (x0 + x1) / 2 + lx,
                    (y0 + y1) / 2 + ly,
                    label,
                    fontsize=6.5,
                    style="italic",
                    color=ink,
                    ha="center",
                    va="center",
                )

        box(50, 92, "round 1 (rules 1–5): flat anchor · scheduled anti · 50 epochs", out_fc, out_ec)

        box(36, 70, "rule 6\nanti-mid vs ref", dec_fc, dec_ec)
        arrow(44, 88, 37, 76)

        box(12, 44, f"branch A confirmed:\nscheduled anti\n4 dims · {_n_a} trials", out_fc, out_ec)
        arrow(28, 65, 14, 51, "resolved\nworse", lx=-8, ly=0)

        box(
            48, 46, "rule 7\nanti-mid-flat vs flat-anchor\n(read because rule 1\nchose the flat anchor)", dec_fc, dec_ec
        )
        arrow(40, 64, 46, 55, "within band", lx=9, ly=1)

        box(30, 16, f"flat anchor\nflat anti at {ex.ANTI_MID_RATIO:g}\n3 dims · {_n_b} trials", out_fc, out_ec)
        arrow(42, 37, 32, 24, "within band", lx=-8, ly=0)

        box(64, 16, f"scheduled anchor\nflat anti at {ex.ANTI_MID_RATIO:g}\n3 dims · {_n_b} trials", out_fc, out_ec)
        arrow(54, 37, 62, 24, "resolved\nworse", lx=8, ly=1)

        box(82, 70, "rule 8\nshort25 vs ref,\ntask-gated vs short25-lam0", dec_fc, dec_ec)
        arrow(56, 88, 81, 77)

        box(74, 44, "25 epochs", out_fc, out_ec)
        arrow(78, 62, 75, 49, "both pass", lx=-8, ly=0)

        box(92, 44, "50 epochs\n(rule 4)", out_fc, out_ec)
        arrow(86, 62, 91, 51, "either\nfails", lx=7, ly=1)

        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Shape or dose? (rules 6 and 7)

    Shape. `anti-mid` matches the dose of the schedule to within rounding and still comes out worse on three of the five statistics: grading falls 0.1022 against a band of 0.0392, m_line 0.0156 against 0.0144, and the group contrast 0.0135 against 0.0093. Containment and retention stay within band, containment on the better side.

    That is what the shape reading predicted, a weakened blend of the two round-1 failures with containment intact, and not what the dose reading predicted. Rule 6 confirms branch A, so the peak and anneal endpoint of the schedule stay dimensions of the search.

    Grading was the statistic to watch, and it moved the most, in the direction the frozen prediction gave a mechanism for. A constant at the matched level delivers four to six times the late repulsion of the schedule over the window where grading is learned, and the response ends less graded, as it did in both round-1 brackets.

    Rule 7 is unread, because its gate needs rule 6 to simplify. The arm ran anyway, and it completes the 2 × 2. Against `flat-anchor`, `anti-mid-flat` shows the same three regressions (grading −0.1325, m_line −0.0203, contrast −0.0108). In the panel below the two anchor-shape lines move nearly in parallel, so flattening the anti term costs about the same with either anchor shape.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary, delta_table):
    mo.vstack(
        [
            mo.Html(
                figure_html(
                    delta_table(ab_summary, ["anti-mid"]),
                    caption=mo.md("""
                    **`anti-mid` against `ref`.** Read as the round-1 tables. The arm matches the reference dose to 0.3% by construction, so what the deltas measure is the shape alone.
                    """).text,
                    class_="report-figure",
                )
            ),
            mo.Html(
                figure_html(
                    delta_table(ab_summary, ["anti-mid-flat"], "flat-anchor"),
                    caption=mo.md("""
                    **`anti-mid-flat` against `flat-anchor`.** The frozen comparison for rule 7. Both arms hold the anchor flat, so the anti term is again the only difference.
                    """).text,
                    class_="report-figure",
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(ab_cells):
    _corner = {
        ("min-jerk", "schedule"): "ref",
        ("flat", "schedule"): "flat-anchor",
        ("min-jerk", "constant"): "anti-mid",
        ("flat", "constant"): "anti-mid-flat",
    }
    _runs = {
        k: {s: [c[s] for c in ab_cells if c["condition"] == cond] for s in ("m_line", "r2_sim")}
        for k, cond in _corner.items()
    }

    @themed(
        name="two-by-two",
        alt_text="""
            Two panels, each a two-by-two interaction chart. Horizontal axis: the anti-subspace term, schedule or constant at 1.77. Two lines per panel, one per anchor shape, min-jerk and flat, with per-run dots around each seed mean. Left panel, m_line: both lines drop by about 0.02 from schedule to constant, staying nearly parallel. Right panel, grading r squared: both lines drop by about 0.10 to 0.13, again nearly parallel, with the flat-anchor line starting higher.
        """,
        caption=r"""
            **The 2 × 2 of (anchor shape) × (anti term).** Seed means (large marks, joined) and per-run values (small dots) for m_line and grading $r^2$ at the four corners. An interaction would show as the two lines diverging; instead they move nearly in parallel on both panels, so the cost of flattening the anti term does not depend on the anchor's shape.
        """,
    )
    def _plot():
        colors = {"min-jerk": light_dark("#1f6fb4", "#5fa8dd"), "flat": light_dark("#b4531f", "#dd8f5f")}
        fig, _axs = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained", sharex=True)
        ax_m, ax_r = cast(tuple, _axs)
        for ax, stat in ((ax_m, "m_line"), (ax_r, "r2_sim")):
            for anchor in ("min-jerk", "flat"):
                means = [float(np.mean(_runs[(anchor, anti)][stat])) for anti in ("schedule", "constant")]
                ax.plot([0, 1], means, "o-", color=colors[anchor], ms=5, lw=1.2, label=f"{anchor} anchor")
                for xi, anti in ((0, "schedule"), (1, "constant")):
                    y = _runs[(anchor, anti)][stat]
                    ax.scatter(
                        [xi + (0.03 if anchor == "flat" else -0.03)] * len(y),
                        y,
                        s=10,
                        color=colors[anchor],
                        alpha=0.55,
                        lw=0,
                    )
            ax.set_xticks([0, 1], ["schedule", f"constant {ex.ANTI_MID_RATIO:g}"])
            ax.set_xlim(-0.35, 1.35)
        ax_m.set_ylabel("m_line", fontsize="x-small")
        ax_r.set_ylabel("grading r²", fontsize="x-small")
        ax_m.legend(fontsize="x-small", frameon=False, loc="lower left")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Can we halve the budget again? (rule 8)

    No, and not for the predicted reason. The prediction named the task gate as the binding constraint, on the grounds that at 25 epochs the un-anchored control might stop saturating holdout EM. The control is fine: `short25-lam0` holds 0.9961, and the task cost of `short25` is 0.0026 against a gate of 0.02.

    What fails is the anchored recipe itself: m_line drops 0.0268 against a band of 0.0144, containment worsens by 0.0410 against 0.0346, and the contrast drops 0.0222 against 0.0093, with grading within band. The trap the rule named, where a shorter run scores *better* on m_line while being the less converged model, did not appear either; the shorter run scores worse outright.

    The trajectories below say why the halving stopped working. On the shared fraction-of-training clock, `ref` and `short` trace the same curves. But `short25` starts far behind, and spends the rest of the run catching up: at a fifth of training its margin probe reads half of what `ref` reads at the same point, and its validation loss is a full nat higher.

    Fifty epochs leaves slack after convergence, and twenty-five leaves none. Rule 8: the survey runs at the 50 epochs set by rule 4.
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary, delta_table):
    mo.Html(
        figure_html(
            delta_table(ab_summary, ["short25"]),
            caption=mo.md("""
            **`short25` against `ref`.** Read as the tables above; as with `short`, the task gate compares against the un-anchored control at the same length (`short25-lam0`), so convergence cost is not charged to the anchor.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _(ab_cells, traj_of):
    _keys = ("m_line", "val_loss")
    _traj = {k: {c: traj_of(ab_cells, c, k) for c in ("ref", "short", "short25")} for k in _keys}

    @themed(
        name="short25",
        alt_text="""
            Two line charts sharing a horizontal axis of fraction of training. Left: per-line margin m_line for three hundred-epoch runs, three fifty-epoch runs and three twenty-five-epoch runs. The hundred- and fifty-epoch groups rise together over the first fifth of training and hold near 0.75; the twenty-five-epoch group is still near 0.35 at that fraction, climbs for most of the run, and levels off late around 0.65 to 0.69, below the others. Right: validation loss for the same runs; the two longer groups overlap, while the twenty-five-epoch curves sit well above them until nearly half of training before closing the gap.
        """,
        caption=r"""
            **Twenty-five epochs against fifty and a hundred, on a shared clock.** Axes as in the `short` figure above: fraction of training horizontally, the cheap one-partner margin probe on the left, validation loss on the right. The two longer lengths trace the same curves; the quarter-length runs spend nearly half their budget still converging, and the margin they reach afterwards levels off below the others.
        """,
    )
    def _plot():
        colors = {
            "ref": light_dark("#1f6fb4", "#5fa8dd"),
            "short": light_dark("#2a9d8f", "#5fc9bd"),
            "short25": light_dark("#b4531f", "#dd8f5f"),
        }
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
    ### What the survey runs
    """)
    return


@app.cell(hide_code=True)
def _(ab_decision):
    _space = ex.survey_space(ab_decision)
    _rows = "".join(
        f"<tr><td><code>{d}</code></td><td class='num'>{lo:g}</td><td class='num'>{hi:g}</td><td>{scale}</td></tr>"
        for d, (lo, hi, scale) in _space.items()
    )
    mo.md(f"""
    The rules that touch the survey resolve to one recipe and one box. The anchor weight is {"flat at the sampled λ_a" if ab_decision["anchor_shape"] == "flat" else "scheduled, peaking at the sampled λ_a"}; the anti-subspace term {f"keeps its schedule, with the hold ratio fixed at {ex.ANTI_HOLD_RATIO:g} and its peak and anneal endpoint sampled" if ab_decision["anti_branch"] == "A" else "is flat at the sampled ratio"}; every trial runs {ab_decision["epochs"]} epochs.

    <table class="report-table">
    <thead><tr><th>dimension</th><th class="num">low</th><th class="num">high</th><th>scale</th></tr></thead>
    <tbody>{_rows}</tbody>
    </table>

    That is branch {ab_decision["anti_branch"]} of the frozen plan, so the trial list is the one shown under *The survey space* above: {ex.n_trials(_space)} scrambled-Sobol points at seed {ex.SOBOL_SEED}, run at one seed each, with the top {ex.N_PROMOTE} feasible trials promoted to {ex.SEEDS_PROMOTE}. The sampled coordinates are unchanged by the length rules, and so is the relative dose, which is normalized at the run's own length. The centroid column of that table is in epochs at 100, so at {ab_decision["epochs"]} epochs each trial's centroid scales down with it.

    <!-- REVIEW: added the centroid note. The frozen branch A table renders `dose_centroid` at its default 100 epochs, and rule 4 adopted 50, so the column's units are not the units the survey runs in. Nothing sampled or ranked changes; the marginals should be plotted at the survey's own length. -->


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
    def load_survey() -> dict | None:
        """Resolve the published survey from the store, or None if unpublished."""
        store = project_store()
        art = store.get_refs([ex.SURVEY_REF])[ex.SURVEY_REF]
        if art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            (p,) = store.get_many([(art, Path(d) / "survey.json")])
            return json.loads(p.read_text())

    _sv = load_survey()
    mo.stop(_sv is None, mo.md("_The survey isn't published yet; the landscape cells need it._"))
    assert _sv is not None
    sv_scored: dict[int, dict] = {int(t): s for t, s in _sv["scored"].items()}
    sv_trials: dict[int, dict] = {t["trial"]: t for t in _sv["trials"]}
    sv_cells: dict[str, dict] = {c["label"]: c for c in _sv["cells"]}
    sv_promoted: list[int] = _sv["promoted"]
    return sv_cells, sv_promoted, sv_scored, sv_trials


@app.cell(hide_code=True)
def _(ab_decision, ab_summary, sv_trials):
    # The two derived anti coordinates the frozen plan asks the marginals to use,
    # at the survey's own length (see the centroid REVIEW note above).
    _epochs = ab_decision["epochs"]

    def _centroid_frac(tr: dict) -> float:
        w = lambda e: ex.anti_weight(  # noqa: E731
            e, lam=1.0, peak_ratio=tr["anti_peak_ratio"], anneal_end_frac=tr["anti_anneal_end_frac"], epochs=_epochs
        )
        return ex.dose_centroid(w, epochs=_epochs) / _epochs

    sv_coords = {
        t: {"dose": ex.anti_rel_dose(tr, epochs=_epochs), "centroid": _centroid_frac(tr)} for t, tr in sv_trials.items()
    }
    sv_control_em = ab_summary[ex.CONTROL_OF[_epochs]]["holdout_em"]
    sv_r2_floor = ab_summary["ref"]["r2_sim"] - ex.GRADE_R2_DROP
    return sv_control_em, sv_coords, sv_r2_floor


@app.cell(hide_code=True)
def _(sv_scored, sv_trials):
    _infeasible = {t: s for t, s in sv_scored.items() if not s["feasible"]}
    _fails = {
        k: sum(1 for s in _infeasible.values() if not s["checks"][k])
        for k in ("latch", "containment", "grading", "task", "contrast")
    }
    _veto_tau = max(sv_trials[t]["tau"] for t, s in _infeasible.items() if not s["checks"]["latch"])
    _task_lam = min(sv_trials[t]["lam"] for t, s in _infeasible.items() if not s["checks"]["task"])
    mo.md(f"""
    All {len(sv_trials)} trials ran; {len(sv_scored) - len(_infeasible)} are feasible on the final reading and {len(_infeasible)} are not. No constraint is decorative: the latch veto removes {_fails["latch"]} trials, containment {_fails["containment"]}, grading {_fails["grading"]}, the task gate {_fails["task"]}, and the contrast floor {_fails["contrast"]} (a trial can fail several).

    The failures organize by region rather than scattering. The latch veto owns the cold edge: every vetoed trial has τ ≤ {_veto_tau:.2f}. Grading and containment fail mostly where the pull is weak (λ_a ≲ 0.1), where the response is too control-like to show graded structure. The task gate binds only where the pull is strong: every task failure has λ_a ≥ {_task_lam:.2f}.

    The feasible region is the interior left over, and it is wide on every axis. Across it the marginals repeat the calibration's trade at survey scale: m_line rises with λ_a until the task gate cuts in and rises with τ across the whole sampled range, while grading and the contrast fall along the same τ axis toward their floors.

    The anti-subspace coordinates barely organize anything: the plateau's top spans most of the sampled range of both (its extent is quantified under *Proposed operating point*). That is the same insensitivity the amendment measured at the operating point, now visible across the box.
    """)
    return


@app.cell(hide_code=True)
def _(sv_cells, sv_coords, sv_promoted, sv_scored, sv_trials):
    def _cond_mean(t: int) -> float | None:
        vals = [c["r2_cond"] for s in range(sv_scored[t]["n_seeds"]) if "r2_cond" in (c := sv_cells[f"t{t:02d}-s{s}"])]
        return float(np.mean(vals)) if vals else None

    def _row(_t: int) -> str:
        _tr, _s = sv_trials[_t], sv_scored[_t]
        _fails = ", ".join(k for k, ok in _s["checks"].items() if not ok) or "—"
        _name = f"t{_t:02d}"
        if _t in sv_promoted:
            _name = f"<b>{_name}</b>"
        _cond = _cond_mean(_t)
        return (
            f"<tr><td>{_name}</td>"
            + "".join(
                f"<td class='num'>{_tr[k]:.3f}</td>" for k in ("lam", "tau", "anti_peak_ratio", "anti_anneal_end_frac")
            )
            + f"<td class='num'>{sv_coords[_t]['dose']:.2f}</td><td class='num'>{sv_coords[_t]['centroid']:.2f}</td>"
            + f"<td class='num'>{_s['n_seeds']}</td>"
            + "".join(f"<td class='num'>{_s[k]:+.4f}</td>" for k in ("m_line", "alpha_op1", "retention", "r2_sim"))
            + f"<td class='num'>{'—' if _cond is None else format(_cond, '+.4f')}</td>"
            + f"<td class='num'>{_s['contrast']:+.4f}</td><td class='num'>{_s['holdout_em']:.4f}</td>"
            + f"<td class='num'>{_s['latch_pi']:.2f}</td><td>{_fails}</td></tr>"
        )

    _rows = [_row(_t) for _t in sorted(sv_trials)]
    _head_stats = "".join(
        f"<th class='num'>{h}</th>"
        for h in (
            "m_line ↑",
            "alpha_op1 ↓",
            "retention ↑",
            "r2_sim ↑",
            "r2_cond",
            "contrast ↑",
            "holdout EM ↑",
            "latch π ↓",
        )
    )
    mo.Html(
        figure_html(
            f"""
            <div class="report-table-scroll"><table class="report-table">
            <thead><tr><th>trial</th><th class="num">λ_a</th><th class="num">τ</th><th class="num">peak</th>
            <th class="num">anneal end</th><th class="num">dose</th><th class="num">centroid</th>
            <th class="num">seeds</th>{_head_stats}<th>fails</th></tr></thead>
            <tbody>{"".join(_rows)}</tbody>
            </table></div>
            """,
            caption=mo.md("""
            **Every trial, no omissions.** Sampled coordinates (λ_a, τ, anti peak ratio, anneal endpoint), the two derived anti coordinates (dose relative to the reference schedule at the same length, and the centroid as a fraction of training), and seed means of every statistic — over five seeds for the six promoted trials (bold), one seed otherwise. `r2_cond` is the conditional-mean grading reading, reported beside the per-color `r2_sim`; only the per-color reading constrains. The *fails* column lists the constraints a trial fails; the proposed operating point is `t00`.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _(sv_control_em, sv_coords, sv_r2_floor, sv_scored, sv_trials):
    _ROWS = [
        ("m_line", "m_line", []),
        ("holdout_em", "holdout EM", [sv_control_em - ex.TASK_GATE, sv_control_em + ex.TASK_GATE]),
        ("alpha_op1", "ᾱ op1", [ex.MEAN_ALIGN_GATE]),
        ("retention", "retention", [ex.RETENTION_GATE]),
        ("r2_sim", "grading r²", [sv_r2_floor]),
        ("contrast", "contrast", [ex.CONTRAST_MIN]),
        ("latch_pi", "latch π", [ex.LATCH_PI]),
    ]
    _COLS = [
        ("lam", "λ_a", "log"),
        ("tau", "τ", "log"),
        ("dose", "anti dose (rel.)", "log"),
        ("centroid", "anti centroid", "linear"),
    ]

    def _x(t: int, key: str) -> float:
        return sv_trials[t][key] if key in sv_trials[t] else sv_coords[t][key]

    _feas = [t for t, s in sv_scored.items() if s["feasible"]]
    _infeas = [t for t, s in sv_scored.items() if not s["feasible"]]

    @themed(
        name="marginals",
        alt_text="""
            A grid of small scatter panels, seven rows by four columns. Rows are the objective and the six constraint statistics: m_line, holdout exact match, containment, retention, grading r squared, contrast, and the latch statistic pi. Columns are the four axes: anchor weight and pooling temperature on log scales, then the derived anti-subspace dose and centroid. Feasible trials are filled dots, infeasible ones open circles, the proposed trial is ringed, and dashed lines mark each constraint's gate. The strongest patterns: m_line rises left to right along both the anchor weight and temperature columns; grading and contrast fall along the temperature column; latch pi is high only at the cold end of the temperature column; holdout exact match dips below its gate line only at high anchor weight. The two anti-subspace columns show weak trends throughout.
        """,
        caption=r"""
            **The marginals: every statistic against every axis.** One dot per trial (seed means; five seeds for promoted trials, one otherwise): filled = feasible, open = infeasible, ring = the proposed trial `t00`. Rows are the objective and the six constraints, with each constraint's gate dashed — a band for the task row, a ceiling for containment and latch, a floor for the rest. Columns are the two sampled scalars and the two derived anti coordinates (dose relative to the reference schedule at 50 epochs; centroid as a fraction of training). The map to read off: m_line climbs with λ_a and with τ; the τ axis spends grading and contrast as it climbs; the latch veto owns the cold-τ edge; the task gate cuts in at strong λ_a; and the anti axes organize little, which is the amendment's insensitivity result seen across the whole box.
        """,
    )
    def _plot():
        from matplotlib import ticker

        c_feas = light_dark("#1f6fb4", "#5fa8dd")
        c_infeas = light_dark("#999", "#777")
        c_gate = light_dark("#666", "#999")
        _ticks = {"lam": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0], "tau": [0.05, 0.1, 0.2, 0.3], "dose": [0.3, 0.5, 1.0]}
        fig, axs = plt.subplots(
            len(_ROWS), len(_COLS), figsize=(8.4, 10.2), layout="constrained", sharex="col", sharey="row"
        )
        for i, (stat, sname, gates) in enumerate(_ROWS):
            for j, (key, xname, scale) in enumerate(_COLS):
                ax = axs[i][j]
                ax.scatter(
                    [_x(t, key) for t in _infeas],
                    [sv_scored[t][stat] for t in _infeas],
                    s=11,
                    facecolors="none",
                    edgecolors=c_infeas,
                    lw=0.8,
                )
                ax.scatter([_x(t, key) for t in _feas], [sv_scored[t][stat] for t in _feas], s=12, color=c_feas, lw=0)
                ax.scatter([_x(0, key)], [sv_scored[0][stat]], s=52, facecolors="none", edgecolors=c_feas, lw=1.4)
                for g in gates:
                    ax.axhline(g, color=c_gate, ls=(0, (4, 3)), lw=0.7)
                if scale == "log":
                    ax.set_xscale("log")
                    ax.set_xticks(_ticks[key])
                    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.tick_params(labelsize=6)
                if i == len(_ROWS) - 1:
                    ax.set_xlabel(xname, fontsize="x-small")
                if j == 0:
                    ax.set_ylabel(sname, fontsize="x-small")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(ab_arrays, ab_summary, sv_cells, sv_promoted, sv_r2_floor, sv_scored):
    _has_cond = any("r2_cond" in c for c in sv_cells.values())
    _ref_cond = float(np.mean([ex.r2_sim_cond(ab_arrays[f"ref-s{s}/alpha"][:, :, 0].mean(axis=0)) for s in range(3)]))

    def _cond_mean(t: int) -> float:
        return float(np.mean([sv_cells[f"t{t:02d}-s{s}"]["r2_cond"] for s in range(sv_scored[t]["n_seeds"])]))

    _pts = {
        t: (s["m_line"], s["r2_sim"], _cond_mean(t) if _has_cond else np.nan, s["feasible"])
        for t, s in sv_scored.items()
    }

    @themed(
        name="plane",
        alt_text="""
            Two scatter panels sharing both axes: grading r squared against m_line, left panel the per-color reading, right panel the conditional-mean reading. Feasible trials are filled dots, infeasible open, promoted trials ringed, and the reference recipe is marked with a cross near m_line 0.42, r squared 0.78. In the left panel a dashed horizontal line at 0.68 marks the grading floor; the points slope downward to the right, and the proposed trial sits at the far right, just above the line. In the right panel the same cloud sits higher, around 0.9 and above, with a much shallower slope and no floor drawn.
        """,
        caption=rf"""
            **The margin–grading plane, in both readings.** Per-trial seed means: filled = feasible, open = infeasible, rings = the six promoted trials, × = the ablation stage's `ref`. **Left:** the per-color $r^2$ the grading constraint reads, with the floor (`ref` − {ex.GRADE_R2_DROP:g}) dashed; the cloud slopes down to the right — the τ walk trades grading for margin — and the constraint is what stops the objective from following it further. **Right:** the conditional-mean $r^2$ over redness levels, which stays high across the same trials. The pair separates *graded on average* from *graded color by color*: what the soft-τ end loses is per-color fidelity, not the average dose–response shape.
        """,
    )
    def _plot():
        c_feas = light_dark("#1f6fb4", "#5fa8dd")
        c_infeas = light_dark("#999", "#777")
        c_gate = light_dark("#666", "#999")
        c_ref = light_dark("#7b3fa0", "#b98ce0")
        ncols = 2 if _has_cond else 1
        fig, _axs = plt.subplots(
            1, ncols, figsize=(7.2 if _has_cond else 4.2, 3.0), layout="constrained", sharex=True, sharey=True
        )
        axs = np.atleast_1d(_axs)
        for k, ax in enumerate(axs):
            yi = 1 if k == 0 else 2
            for t, (m, r, rc, feas) in _pts.items():
                y = (m, r, rc)[yi]
                if feas:
                    ax.scatter([m], [y], s=14, color=c_feas, lw=0)
                else:
                    ax.scatter([m], [y], s=12, facecolors="none", edgecolors=c_infeas, lw=0.8)
                if t in sv_promoted:
                    ax.scatter([m], [y], s=56, facecolors="none", edgecolors=c_feas, lw=1.2)
            _ref_y = ab_summary["ref"]["r2_sim"] if yi == 1 else _ref_cond
            ax.scatter([ab_summary["ref"]["m_line"]], [_ref_y], marker="x", s=44, color=c_ref, lw=1.6)
            ax.set_xlabel("m_line (seed mean)", fontsize="x-small")
        axs[0].axhline(sv_r2_floor, color=c_gate, ls=(0, (4, 3)), lw=0.8)
        axs[0].set_ylabel("grading r² (per color)", fontsize="x-small")
        if _has_cond:
            axs[1].set_ylabel("grading r² (conditional mean)", fontsize="x-small")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(sv_scored, sv_trials):
    _MARKS = {
        "task": ("^", "task"),
        "containment": ("s", "containment"),
        "grading": ("D", "grading"),
        "latch": ("x", "latch"),
        "contrast": ("v", "contrast"),
    }

    def _first_fail(s: dict) -> str:
        return next(k for k in ("latch", "task", "grading", "containment", "contrast") if not s["checks"][k])

    @themed(
        name="lam-marginal",
        alt_text="""
            One scatter panel: m_line against the anchor weight lambda a on a log axis from 0.02 to 1. Feasible trials are filled dots rising from about 0.35 at the far left to about 0.53 around lambda 0.3 to 0.6, then holding flat to the right edge. Infeasible trials use open markers keyed by their first failed constraint: crosses for latch vetoes are spread across the axis, diamonds and squares for grading and containment failures cluster at the weak end, and triangles for task-gate failures appear only from about lambda 0.4 upward. The proposed trial at lambda 0.56 is ringed.
        """,
        caption=r"""
            **The λ_a marginal on its own.** m_line against λ_a; filled dots are feasible trials, the ring is the proposed `t00`, and each infeasible trial is drawn with a marker naming its first failed constraint. The shape ex-2.1.7 predicted shows up as a rise to a flat top: the margin climbs for a decade of λ_a and stops gaining around 0.3–0.6, while the task gate (△) starts to bind above 0.4 — the two highest margins in the whole survey belong to trials the task gate rejected, so the task, and nothing else, caps the strong end. At the weak end grading (◇) and containment (□) fail together where the pull is too weak to organize the response.
        """,
    )
    def _plot():
        c_feas = light_dark("#1f6fb4", "#5fa8dd")
        c_infeas = light_dark("#8a6d3b", "#c0a060")
        fig, ax = plt.subplots(figsize=(5.6, 3.2), layout="constrained")
        _seen = set()
        for t, s in sv_scored.items():
            x, m = sv_trials[t]["lam"], s["m_line"]
            if s["feasible"]:
                ax.scatter([x], [m], s=16, color=c_feas, lw=0, label="feasible" if "f" not in _seen else None)
                _seen.add("f")
            else:
                mk, mname = _MARKS[_first_fail(s)]
                _label = mname if mname not in _seen else None
                if mk == "x":
                    ax.scatter([x], [m], s=26, marker=mk, lw=1.0, color=c_infeas, label=_label)
                else:
                    ax.scatter([x], [m], s=26, marker=mk, lw=1.0, facecolors="none", edgecolors=c_infeas, label=_label)
                _seen.add(mname)
        ax.scatter([sv_trials[0]["lam"]], [sv_scored[0]["m_line"]], s=64, facecolors="none", edgecolors=c_feas, lw=1.4)
        ax.set_xscale("log")
        ax.set_xlabel("λ_a", fontsize="x-small")
        ax.set_ylabel("m_line (seed mean)", fontsize="x-small")
        ax.legend(fontsize="x-small", frameon=False, loc="lower right", ncols=2)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Proposed operating point
    """)
    return


@app.cell(hide_code=True)
def _(ab_summary, sv_coords, sv_scored, sv_trials):
    _t0, _s0 = sv_trials[0], sv_scored[0]
    _ref = ab_summary["ref"]

    def _pm(stat: str, s: dict) -> str:
        return f"{s[stat]:.4f} ± {ex.NOISE_RUN[stat] / np.sqrt(s['n']):.4f}"

    _stats = ["m_line", "alpha_op1", "retention", "r2_sim", "contrast", "holdout_em"]
    _point_rows = (
        "<tr><td>proposed (<code>t00</code>)</td>"
        + f"<td class='num'>{_t0['lam']:.3f}</td><td class='num'>{_t0['tau']:.3f}</td>"
        + f"<td class='num'>{_t0['anti_peak_ratio']:.3f}</td><td class='num'>{_t0['anti_anneal_end_frac']:.3f}</td>"
        + f"<td class='num'>{sv_coords[0]['dose']:.2f}</td>"
        + "".join(f"<td class='num'>{_pm(s, {**_s0, 'n': _s0['n_seeds']})}</td>" for s in _stats)
        + "</tr>"
        + "<tr><td><code>ref</code> (recipe)</td>"
        + f"<td class='num'>{ex.SCORING_LAMBDA:g}</td><td class='num'>{ex.TAU_REF:g}</td>"
        + f"<td class='num'>{ex.ANTI_PEAK_RATIO:g}</td><td class='num'>{ex.ANTI_ANNEAL_END_FRAC:g}</td>"
        + "<td class='num'>1.00</td>"
        + "".join(f"<td class='num'>{_pm(s, _ref)}</td>" for s in _stats)
        + "</tr>"
    )
    mo.Html(
        figure_html(
            f"""
            <div class="report-table-scroll"><table class="report-table">
            <thead><tr><th>point</th><th class="num">λ_a</th><th class="num">τ</th><th class="num">peak</th>
            <th class="num">anneal end</th><th class="num">dose</th>
            <th class="num">m_line ↑</th><th class="num">alpha_op1 ↓</th><th class="num">retention ↑</th>
            <th class="num">r2_sim ↑</th><th class="num">contrast ↑</th><th class="num">holdout EM ↑</th></tr></thead>
            <tbody>{_point_rows}</tbody>
            </table></div>
            """,
            caption=mo.md("""
            **The proposed operating point beside the recipe it would replace.** Seed means ± the frozen per-run noise floor over √n (five seeds for the proposal, three for `ref`). The coordinate columns are the sampled values plus the relative anti dose; `ref`'s dose is 1 by definition. These numbers are proposals: D2.2 adopts the point and re-measures it at fresh seeds before any of them is quoted.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _(ab_summary, sv_coords, sv_r2_floor, sv_scored, sv_trials):
    _feas = sorted((t for t, s in sv_scored.items() if s["feasible"]), key=lambda t: -sv_scored[t]["m_line"])
    _band = ex.equiv_band("m_line", 5, 1)
    _top = [t for t in _feas if sv_scored[t]["m_line"] >= sv_scored[0]["m_line"] - _band]
    _rng = lambda key: (  # noqa: E731
        min(sv_trials[t].get(key) or sv_coords[t][key] for t in _top),
        max(sv_trials[t].get(key) or sv_coords[t][key] for t in _top),
    )
    _lam, _tau, _pk, _dose = _rng("lam"), _rng("tau"), _rng("anti_peak_ratio"), _rng("dose")
    _r2_others = [sv_scored[t]["r2_sim"] for t in _top[1:]]
    mo.md(f"""
    The frozen rule proposes `t00`, the promoted trial with the highest seed-mean m_line that stays feasible on every constraint. Relative to the recipe it is a stronger pull with softer pooling and less repulsion: λ_a rises from {ex.SCORING_LAMBDA:g} to {sv_trials[0]["lam"]:.2f}, τ softens from {ex.TAU_REF:g} to {sv_trials[0]["tau"]:.2f}, and the anti term keeps the shape of the schedule at {sv_coords[0]["dose"]:.2f}× the reference dose.

    The margin it buys is {sv_scored[0]["m_line"]:.3f}, against {ab_summary["ref"]["m_line"]:.3f} for `ref`. Containment, retention and the task gate pass with room. Grading sits {sv_scored[0]["r2_sim"] - sv_r2_floor:.3f} above its floor, a third of the per-run noise of that statistic, and it is the closest pass among the checks on the proposal.

    The point sits on a plateau rather than a peak. {len(_top)} trials land within one m_line band ({_band:.3f}, five seeds against one) of the proposal, spanning λ_a {_lam[0]:.2f}–{_lam[1]:.2f}, τ {_tau[0]:.2f}–{_tau[1]:.2f}, peak ratio {_pk[0]:.1f}–{_pk[1]:.1f} and relative dose {_dose[0]:.2f}–{_dose[1]:.2f}.

    The runners-up among them hold {min(_r2_others) - sv_scored[0]["r2_sim"]:.2f}–{max(_r2_others) - sv_scored[0]["r2_sim"]:.2f} more grading at nearly the same margin (the plane figure above). So what the search found is the plateau together with the constraint that bounds it, and the proposal is one end of that plateau.
    """)
    return


@app.cell(hide_code=True)
def _(sv_cells, sv_promoted, sv_scored):
    _rows = "".join(
        f"<tr><td><code>t{t:02d}</code></td>"
        f"<td class='num'>{sv_cells[f't{t:02d}-s0']['m_line']:.4f}</td>"
        f"<td class='num'>{sv_scored[t]['m_line']:.4f}</td>"
        f"<td class='num'>{sv_scored[t]['m_line'] - sv_cells[f't{t:02d}-s0']['m_line']:+.4f}</td>"
        f"<td>{'✓' if sv_scored[t]['feasible'] else '✗ ' + ', '.join(k for k, ok in sv_scored[t]['checks'].items() if not ok)}</td></tr>"
        for t in sv_promoted
    )
    mo.Html(
        figure_html(
            f"""
            <table class="report-table">
            <thead><tr><th>trial</th><th class="num">m_line, seed 0</th><th class="num">m_line, 5 seeds</th>
            <th class="num">change</th><th>feasible at 5 seeds</th></tr></thead>
            <tbody>{_rows}</tbody>
            </table>
            """,
            caption=mo.md("""
            **The promotion round is the winner's curse, measured.** The six trials promoted on their seed-0 m_line, in promotion order. Every one fell back on re-seeding, so the trials were promoted partly on merit and partly on a lucky seed, and two crossed the task gate on their five-seed means and dropped out of feasibility. The five-seed mean of the proposal already carries this correction, while the single-seed trials around it on the plateau do not. That is one more reason the numbers here are proposals for D2.2 rather than results.
            """).text,
            class_="report-figure",
        )
    )
    return


@app.cell(hide_code=True)
def _(sv_scored, sv_r2_floor):
    mo.md(f"""
    **The handoff.** D2.2 adopts this operating point and re-measures it at fresh seeds before quoting any number from this section. The survey values stand beside the confirmed ones there, and the gap between them is the remaining winner's-curse correction.

    The statistic to watch at confirmation is grading. The proposal passes its floor by {sv_scored[0]["r2_sim"] - sv_r2_floor:.3f}, so a modest unlucky draw at fresh seeds would put it under. If that happens, the place to step back to is the plateau, where the runners-up hold more grading at nearly the same margin.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    *Post hoc: noticed after the ablation results, not part of the frozen plan.*

    ### The latch veto fires on an un-anchored control

    One run in each un-anchored control trips the veto: `lam0-s2` puts 0.55 of its deep-slice weight on `+`, and `short-lam0-s0` puts 0.52 there, both above the 0.5 threshold. The detector is reading a profile that only means something when there is a pull. With λ_a = 0, the softmin weights describe whatever alignment structure the task happens to leave behind, and there is no pull that could have latched.
    <!-- REVIEW: was "one of the three `lam0` runs". The 50-epoch control fires too (0.523), which strengthens rather than changes the reading. Verify: the per-run π table in the published metrics. --> The veto is calibrated on the anchored runs of ex-2.1.9, and the controls were never going to be judged by it, so nothing here is affected.

    It does bear on the survey. λ_a runs down to 0.02, a fifth of the scored rung, and a trial that weak may look like the control rather than like a pull that committed. Such a trial would be marked infeasible for a reason other than the one the veto is named for.

    It would fail the contrast constraint too, so the error goes in the safe direction: we would reject a trial that was never going to be proposed. We are leaving the veto frozen and reporting the λ_a of every vetoed trial, so the two readings can be told apart in the map.
    """)
    return


@app.cell(hide_code=True)
def _(sv_scored, sv_trials):
    _veto = {t: s for t, s in sv_scored.items() if not s["checks"]["latch"]}
    _weak = {t for t in _veto if sv_trials[t]["lam"] <= 0.08}
    _strong = sorted(set(_veto) - _weak, key=lambda t: sv_trials[t]["lam"])
    mo.md(f"""
    ### Where the vetoes actually landed

    The vetoes from the survey sort by τ rather than by λ_a. All {len(_veto)} vetoed trials sit at τ ≤ {max(sv_trials[t]["tau"] for t in _veto):.2f}, the cold end of the box. {len(_weak)} of them are the weak-pull corner the paragraph above anticipated (λ_a ≤ {max(sv_trials[t]["lam"] for t in _weak):.2f}; one, `t29` at π = {sv_scored[29]["latch_pi"]:.2f}, also fails grading and contrast, which is the control-like signature).

    The other {len(_strong)} have healthy pulls, λ_a {sv_trials[_strong[0]]["lam"]:.2f}–{sv_trials[_strong[-1]]["lam"]:.2f}. Their π values, {min(_veto[t]["latch_pi"] for t in _strong):.2f} to {max(_veto[t]["latch_pi"] for t in _strong):.2f}, fall inside what the calibration in ex-2.1.9 measured as an empty gap, with latched runs at ≥ 0.85 and healthy runs at ≤ 0.03. Cold pooling concentrates the softmin weight, so a modest alignment with a syntax role reads as a large π there.

    Whether these are true latches, or the detector reading far from its calibration point, cannot be settled from this data. The error direction is safe either way, since every vetoed trial sits in the τ region whose feasible neighbors score well below the plateau. The frozen noise floors carry the same cold-regime caveat, which is one more reason for D2.2 to stay off the τ ≲ 0.1 edge unless it brings a recalibrated detector.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    What transfers from M1 is the shape of the schedule. The dose is not what matters: a constant that delivers the same integrated repulsion loses the grading the schedule protects, with either anchor shape. The schedule on the anchor term, which M1 carried for training stability, resolves nothing here, and neither does the interpolation shape.

    Of everything inherited from the autoencoder recipe, the one load-bearing piece of scheduling is *when the repulsion arrives*: strong while the pull is committing, reduced while grading is learned. That reading now stands on a one-variable comparison rather than on brackets that moved two things at once.

    Within this testbed, the map says the recipe does not need careful tuning. The feasible region spans a decade of λ_a and most of the τ box, the top of the m_line landscape is a plateau rather than a spike, and the anti-subspace coordinates barely matter within the schedule family.

    Each edge of the region belongs to a different constraint: the latch veto at cold τ, grading and contrast at soft τ, the task gate at strong λ_a, and grading and containment where the pull is too weak. So the box a future search inherits is drawn by measured boundaries rather than by guesses.

    <!-- REVIEW: "For M3, the map says the method does not need careful tuning" narrowed to "within this testbed ... the recipe". The map covers one grammar, one corpus, one architecture and one labeller; M3 is small language models with real safety targets, which nothing here measures. Verify: the survey's space table and the method section, which fix everything outside the four sampled dimensions. -->

    The survey format did what it was adopted for. The amendment could be added mid-experiment without spending the preregistration, because the frozen rules, their predictions, and the record-as-a-diff made the addition inspectable. Publishing every trial is what makes the plateau readable as a result.

    Two frictions are worth carrying to the next survey plan. First, a scalar objective walks to its constraint boundary. The proposal sits on the grading floor with a pass margin a third of the per-run noise of that statistic, so a plan that hands off a single point should either freeze how close to a constraint a proposal may sit, or expect the confirming experiment to step back onto the plateau.

    Second, ranking on single-seed values promoted two trials that the task gate rejected at five seeds. The gate width (0.02) is about twice the per-run noise of holdout EM (0.0087), so a seed-0 pass near the boundary carries little information. The curse correction absorbed both, which is what promotion is for, but a future plan could rank on constraint margins as well as on the objective.
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
