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

    *Empty until results land. Every line will report its noise floor alongside the number, and one line will give the proposed operating point.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a survey, not a hypothesis-scoring experiment: it preregisters a *search plan* and scores nothing. The plan covers the space, the sampling rule, the objective and its constraints, the seed budget, and the decision rules that stage the work. Once agreed it is frozen (aside from immaterial edits). Results replace the `TODO` placeholders in place, and anything we think of after seeing the data goes under *Exploratory analyses*, marked as post hoc.

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
    Table: `flat-anchor` − `ref` seed-mean differences for each decision statistic, each against its band. Then the per-run deep-slice weight profiles beside the latch table from ex-2.1.9, and the m_line trajectories of all six runs overlaid: does the margin of the flat arm arrive earlier, and does it hold?

    If the commit-window account is right, we expect the flat arm either to latch a syntax role in at least one run or to hold a resolvably lower m_line. The dose columns tell us any such difference is about timing rather than dose. The contrary result is a flat arm within band with no latch, in which case the survey runs flat and four schedule parameters go away.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Does the anti-subspace schedule beat a constant?

    /// admonition | TODO
    Table: `anti-hold` and `anti-peak` against `ref`, same statistics, same bands, with the delivered dose and centroid of each arm restated from the conditions table.

    The mechanism proposed in ex-2.1.8 is that early repulsion pushes the syntax roles out of reach before the pull commits. Under that account we expect `anti-hold` to lose containment or contrast, and `anti-peak` to hold the statistics but pay somewhere visible.

    If `anti-peak` pays nothing, the schedule was a constant in disguise and branch B runs. The other outcome worth naming is `anti-hold` within band, which would say the timing of the repulsion never mattered at this operating point, and branch B runs at the lower level.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Do the flat arms compose?

    /// admonition | TODO
    One row, read only if rules 1 and 2 both simplify: `flat-both` − `ref` for each decision statistic, compared against its band. An arm that changes one schedule at a time cannot show an interaction between the two, and flattening both is the recipe the survey would actually run. If each flat arm passes on its own but the two together fail, rule 3 keeps the anchor schedule and adopts only the flat anti-subspace term.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Is 100 epochs buying anything?

    /// admonition | TODO
    Table: `short` − `ref` on the decision statistics, with the task gate measured against `short-lam0`. Then the m_line and validation-loss trajectories on a shared fraction-of-training axis.

    Ex-2.1.6 saw margins flat from about epoch 50 and validation loss settling earlier, so we expect `short` to land within band and the survey to cost half as much. The contrary result is a resolved gap in m_line or retention, which would mean the last 50 epochs do some consolidation the trajectory statistics haven't shown. Either way it is worth knowing before D2.2 scales anything.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Does the anneal shape matter?

    /// admonition | TODO
    One row: `linear` − `ref` per decision statistic against its band. M1 only ever compared against a stepped anneal, which needed an accommodation in the LR warmup. The linear arm has a milder discontinuity, a jump in the derivative of λ_a at each keyframe, and is the cheapest version of the question. We expect it within band, which would retire the todo item. Neither outcome changes a survey dimension.
    ///
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

    *Empty until results land; anything conceived after seeing the data goes here, marked as post hoc.*
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


if __name__ == "__main__":
    app.run()
