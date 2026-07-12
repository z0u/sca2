import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import io

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from sca.colorcube import TRAJ_STRIDE, classify, load_results, make_dopesheet
    from mini.reports import report_bundle, use_publisher
    from mini.temporal import Dopesheet, Timeline, realize_timeline
    from mini.vis import light_dark, themed

    use_publisher(report_bundle(__file__))

    # Store refs published by experiment.py (kept in sync by hand).
    METRICS_REF = "reports/ex-2.9.3/metrics"
    TRAJS_REF = "reports/ex-2.9.3/trajectories"

    PEAK_LRS = (0.10, 0.07, 0.05, 0.03)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Experiment 2.9.3: why anchoring fails — timing, attribution, and a schedule fix

    [Ex-2.9.2](../ex-2.9.2/report.py) split ablation variance in two and solved
    the *redistribution* half: fallback control gives the intervention a
    designed response. The other half — on some seeds the concept never ends
    up cleanly on its axis — was left open, with the working hypothesis that
    the regularizer schedule is **incompatible with some initializations**.
    If that were right, the remedy would be a per-seed or per-init schedule
    search: expensive, and worse at scale.

    This experiment tests that hypothesis directly, and rejects it. Three
    arms, all on ex-2.9.1's tiny color autoencoder:

    - **Trajectories** — retrain ex-2.9.2's base arm (32 seeds, identical
      RNG), recording anchor progress, leakage, and reconstruction error at
      every step: *when* do failures happen, relative to the schedule?
    - **Attribution** — factor the RNG into the model init and the batch/label
      stream (16 inits × 8 streams, including the two known-catastrophic
      inits). Failures that are a property of the init repeat along its row;
      accidents scatter.
    - **Schedule sweep** — peak LR {0.10, 0.07, 0.05, 0.03} × regularizer
      anneal {on, off}, 32 seeds per cell, trained with the fallback term and
      scored by the `redirect` intervention (ex-2.9.2's recipe, so the score
      reflects anchoring quality rather than redistribution luck).

    The experiment is [`experiment.py`](./experiment.py):

    ```bash
    bin/mini run docs/ex-2.9.3/experiment.py --app modal --max-containers 16
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    loaded = load_results(METRICS_REF, TRAJS_REF)
    return (loaded,)


@app.cell(hide_code=True)
def _(loaded):
    mo.stop(
        loaded is None,
        mo.md(
            "No results yet — run the experiment (it publishes metrics to the store on completion):\n\n"
            "```bash\nbin/mini run docs/ex-2.9.3/experiment.py --app modal --max-containers 16\n```"
        ),
    )
    metrics, trajs = loaded

    def arm(name: str) -> list[dict]:
        return [r for r in metrics if r["arm"] == name]

    def traj(r: dict, key: str) -> np.ndarray:
        return trajs[f"{r['run']:03d}_{key}"]

    def sweep_cell(peak_lr: float, anneal: bool) -> list[dict]:
        return [r for r in arm("sweep") if r["peak_lr"] == peak_lr and r["anneal"] == anneal]

    steps = np.arange(len(traj(arm("trajectories")[0], "z0_red"))) * TRAJ_STRIDE
    return arm, steps, sweep_cell, traj


@app.cell(hide_code=True)
def _(arm):
    _t = arm("trajectories")
    _bad = [r for r in _t if classify(r) != "clean"]
    _cat = [r for r in _bad if classify(r) == "catastrophic"]
    mo.md(
        f"**{len(arm('trajectories')) + len(arm('attribution')) + len(arm('sweep'))} runs completed.** "
        f"The trajectory arm reproduces ex-2.9.2's base arm bit-for-bit: {len(_bad)} of {len(_t)} runs "
        f"end unhealthy ({len(_cat)} catastrophically — the anchor lost or reconstruction collapsed — "
        f"and {len(_bad) - len(_cat)} with non-red colors leaked onto the anchored axis). The per-step "
        f"traces show what the endpoint metrics hide: **every failing run anchored successfully "
        f"first**, then broke during the high-LR plateau. The attribution arm confirms the failures "
        f"are not a property of the initialization, and the sweep finds a plain schedule fix: halve "
        f"the LR peak and keep the anneal."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Failures happen late, not early

    Each line below is one seed's training trajectory under the original
    schedule. The top panel tracks anchor progress — z₀ of pure red, which
    is 1 when *red* sits exactly on its anchor — and the middle panel tracks
    leakage (mean |z₀| over decidedly non-red colors). The bottom panel is
    the schedule itself: the LR ramps to its 0.1 peak at step 750 and holds,
    while the regularizer weights anneal to zero by step 1425.
    """)
    return


@app.cell(hide_code=True)
def _(arm, steps, traj):
    _t = arm("trajectories")
    _bad = sorted((r for r in _t if classify(r) != "clean"), key=lambda r: r["seed"])
    _label = {
        r["seed"]: f"seed {r['seed']} — "
        + ("anchor lost" if r["val_anchor"] > 0.3 else "recon collapse" if r["val_recon"] > 0.01 else "leak")
        for r in _bad
    }
    _hist = realize_timeline(Timeline(Dopesheet.from_csv(io.StringIO(make_dopesheet(0.10, anneal=True)))))

    @themed(
        name="trajectories",
        alt_text=(
            "Three stacked charts sharing a step axis from 0 to 1500. Top: anchor progress (z0 of "
            f"pure red) for 32 seeds. All lines, including the {len(_bad)} failures drawn in color, "
            "rise from near 0 to about 1 by step 750. After that the gray (healthy) lines stay at 1 "
            "while the colored ones misbehave: seeds 22 and 8 dip briefly as the learning rate "
            "reaches its peak (both recover on this metric, though seed 22's reconstruction never "
            "does), seed 15 breaks away to about 0.65 from step 900, and seed 27 falls to about "
            "0.4 after step 1100. Middle: leakage for the same runs; healthy lines settle near 0.05, "
            "failures climb to 0.1–0.6 in the same late window. Bottom: the schedule — learning rate "
            "ramping to 0.1 at step 750 and holding, regularizer weights annealing to zero by 1425. "
            "A shaded band marks the high-LR plateau, where every failure occurs."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.5), sharex=True, height_ratios=[3, 2, 1.6])
        gray = light_dark("#9aa5b155", "#6b768655")
        hues = ["#d55e00", "#0072b2", "#009e73", "#cc79a7", "#e69f00"]
        for ax, key in zip(axes[:2], ("z0_red", "leak"), strict=False):
            for r in _t:
                if classify(r) == "clean":
                    ax.plot(steps, traj(r, key), color=gray, lw=1)
            for hue, r in zip(hues, _bad, strict=True):
                ax.plot(steps, traj(r, key), color=hue, lw=1.6, label=_label[r["seed"]])
        axes[0].set(ylabel="z₀(pure red)", ylim=(-1.05, 1.05))
        axes[0].legend(loc="lower left", fontsize=8, ncols=2)
        axes[1].set(ylabel="leak (mean |z₀|, non-red)", ylim=(0, 0.7))
        ax = axes[2]
        ax.plot(_hist.index, _hist["lr"], color=light_dark("#1a5f8a", "#6ab0d4"), lw=2, label="lr")
        ax2 = ax.twinx()
        for name, ls in (("anchor", "--"), ("anti-anchor", "-."), ("anti-subspace", ":")):
            ax2.plot(_hist.index, _hist[name], color=light_dark("#555", "#aaa"), ls=ls, lw=1.2, label=name)
        ax.set(xlabel="step", ylabel="lr", ylim=(0, 0.11))
        ax2.set(ylabel="weight", ylim=(0, 0.27))
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8, ncols=2)
        for ax in axes:
            ax.axvspan(750, 1500, color=light_dark("#1a5f8a", "#6ab0d4"), alpha=0.07, lw=0)
            ax.grid(alpha=0.3)
        axes[0].set_title("Every failure anchors first, then breaks during the high-LR plateau")
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arm, traj):
    _t = arm("trajectories")
    _bad = [r for r in _t if classify(r) != "clean"]

    def _timing(r):
        z = traj(r, "z0_red")
        hi = np.flatnonzero(z > 0.9)
        drops = [i for i in np.flatnonzero(z < 0.7) if len(hi) and i > hi[0]]
        return hi[0] * TRAJ_STRIDE if len(hi) else None, drops[0] * TRAJ_STRIDE if drops else None

    _anchored = [_timing(r)[0] for r in _bad]
    _drops = [t for t in (_timing(r)[1] for r in _bad) if t is not None]
    _w_anchor = realize_timeline(Timeline(Dopesheet.from_csv(io.StringIO(make_dopesheet(0.10, True)))))[
        "anchor"
    ].to_numpy()
    mo.md(
        f"All {len(_bad)} failing runs reached z₀(red) > 0.9 — a clean anchor — between steps "
        f"{min(_anchored)} and {max(_anchored)}, right as the LR approaches its peak. The "
        f"{len(_drops)} that later lost it did so between steps {min(_drops)} and {max(_drops)}: "
        f"seeds 22 and 8 the moment the LR tops out (reconstruction collapses outright on 22), and "
        f"seeds 15 and 27 only once the anchor weight has annealed low — by step "
        f"{max(_drops)} it is {float(_w_anchor[min(max(_drops), len(_w_anchor) - 1)]):.3f}, "
        f"too weak to pull red back. So the anchored solution is *metastable* at this LR: the "
        f"regularizers hold it in place while they're on, and the timed anneal removes that "
        f"protection while the optimizer is still hot. Nothing about these seeds resisted anchoring; "
        f"they were unlucky during the plateau. That predicts the failure should follow the "
        f"*randomness of training*, not the initialization — which the next arm tests directly."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## It's not the seed (and so there is nothing to sweep against)

    The attribution arm re-trains 16 inits × 8 batch/label streams under the
    original schedule. Under the incompatible-init hypothesis, the two inits
    whose legacy runs failed catastrophically (22 and 27) should fail again;
    under the accident hypothesis, failures should scatter.
    """)
    return


@app.cell(hide_code=True)
def _(arm):
    _att = arm("attribution")
    _inits = sorted({r["seed"] for r in _att}, key=lambda s: (s >= 14, s))
    _streams = sorted({r["stream"] for r in _att})
    _leak = np.zeros((len(_inits), len(_streams)))
    _fail = np.zeros_like(_leak, dtype=bool)
    for _r in _att:
        _i, _j = _inits.index(_r["seed"]), _streams.index(_r["stream"])
        _leak[_i, _j] = _r["leak"]
        _fail[_i, _j] = classify(_r) != "clean"

    @themed(
        name="attribution",
        alt_text=(
            "Heatmap of final leakage over a 16-by-8 grid: model inits on rows (0 through 13, then "
            "the two known-catastrophic inits 22 and 27), batch/label streams on columns. Most cells "
            f"are pale (leak around {np.median(_leak):.2f}). The {int(_fail.sum())} failed cells, "
            "marked with crosses, scatter across the grid: no row has more than "
            f"{int(_fail.sum(1).max())} of 8, and the rows for inits 22 and 27 are entirely clean. "
            "The darkest cell is init 1 under stream 1, with leak "
            f"{_leak.max():.2f}."
        ),
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
        im = ax.imshow(_leak, cmap="Blues", vmin=0, vmax=0.35, aspect="auto")
        for i in range(len(_inits)):
            for j in range(len(_streams)):
                if _fail[i, j]:
                    ax.text(j, i, "×", ha="center", va="center", color=light_dark("#d55e00", "#ffb000"), fontsize=13)
        ax.set_xticks(range(len(_streams)), [str(s) for s in _streams])
        ax.set_yticks(range(len(_inits)), [str(s) for s in _inits])
        ax.set(xlabel="batch/label stream", ylabel="model init")
        fig.colorbar(im, ax=ax, label="final leak (mean |z₀|, non-red)", shrink=0.8)
        ax.set_title("Failures (×) scatter across inits — including 22 and 27")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arm):
    _att = arm("attribution")
    _fails = [r for r in _att if classify(r) != "clean"]
    _cats = [r for r in _att if classify(r) == "catastrophic"]
    _by_init, _by_stream = {}, {}
    for _r in _fails:
        _by_init[_r["seed"]] = _by_init.get(_r["seed"], 0) + 1
        _by_stream[_r["stream"]] = _by_stream.get(_r["stream"], 0) + 1
    _n2227 = sum(1 for r in _fails if r["seed"] in (22, 27))
    mo.md(
        f"Inits 22 and 27 — catastrophic under their legacy streams — trained cleanly under **all 8 "
        f"fresh streams** ({_n2227} failures between them). The {len(_fails)} unhealthy runs "
        f"(of 128; {len(_cats)} catastrophic) instead scatter across {len(_by_init)} different "
        f"inits, none failing more than {max(_by_init.values())} of 8, with mild clustering by "
        f"stream (stream 6 accounts for {_by_stream.get(6, 0)}). The incompatible-init hypothesis "
        f"is dead: the same init succeeds or fails depending on which random batches and label "
        f"draws it sees during the hot phase, with a residual interaction that is just chaos. Two "
        f"practical consequences. Sweeping schedules *per seed* would be aiming at noise — a seed "
        f"isn't good or bad, a *(seed, stream, schedule)* triple is. And any fix must make the "
        f"plateau safe for every trajectory, rather than route around particular inits."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The sweep: the LR peak was the culprit; the anneal was innocent

    Going in, the anneal looked guilty — it removes the anchor's protection.
    But the sweep says otherwise: holding the regularizers on to the end
    (`anneal off`) makes things *worse* at every peak, while simply halving
    the LR peak removes the failures entirely. These runs train with the
    fallback term and are scored by the redirect intervention, per ex-2.9.2.
    """)
    return


@app.cell(hide_code=True)
def _(sweep_cell):
    _cells = [(p, a) for p in PEAK_LRS for a in (True, False)]
    _scores = {c: np.array([r["interventions"]["redirect"]["score"] for r in sweep_cell(*c)]) for c in _cells}
    _leaks = {c: np.array([r["leak"] for r in sweep_cell(*c)]) for c in _cells}
    _nbad = {c: sum(classify(r) != "clean" for r in sweep_cell(*c)) for c in _cells}

    @themed(
        name="sweep",
        alt_text=(
            "Two stacked strip plots over eight conditions: peak learning rates 0.10, 0.07, 0.05, "
            "0.03, each with the regularizer anneal on (blue) or off (gray), 32 seeds per condition. "
            "Top: redirect selectivity scores. Medians hover around 0.87 to 0.91 everywhere, but "
            f"0.03 with anneal sits lower at {np.median(_scores[(0.03, True)]):.2f}; one blue outlier "
            f"at 0.10 falls to {_scores[(0.10, True)].min():.2f}. Bottom: final leakage on a log "
            "scale with a dashed line at the 0.1 degraded threshold. With the anneal on, leak "
            "tightens as the peak drops — at 0.05 no seed crosses the line. With the anneal off, "
            "every condition has a tail of 4 to 8 seeds above the line."
        ),
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True, height_ratios=[2, 1.4])
        rng = np.random.default_rng(0)
        colors = {True: light_dark("#1a5f8a", "#6ab0d4"), False: light_dark("#9aa5b1", "#6b7686")}
        for ax, data in zip(axes, (_scores, _leaks), strict=True):
            for gi, (p, a) in enumerate(_cells):
                x0 = gi + rng.uniform(-0.13, 0.13, len(data[(p, a)]))
                ax.scatter(x0, data[(p, a)], s=14, color=colors[a], alpha=0.75, lw=0)
                ax.plot([gi - 0.22, gi + 0.22], [np.median(data[(p, a)])] * 2, color=colors[a], lw=2)
            ax.grid(alpha=0.3, axis="y")
        axes[0].set(ylabel="score (R², redirect)", ylim=(0, 1.02))
        axes[1].axhline(0.1, ls="--", lw=1, color=light_dark("#000", "#fff"), alpha=0.5)
        axes[1].text(7.45, 0.1, "degraded", va="bottom", ha="right", fontsize=8, alpha=0.8)
        axes[1].set(ylabel="final leak", yscale="log", xlabel="peak LR × anneal")
        axes[1].set_xticks(range(len(_cells)), [f"{p}\n{'anneal' if a else 'hold'}" for p, a in _cells])
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=c, label=lbl)
            for lbl, c in (("anneal on", colors[True]), ("anneal off (hold)", colors[False]))
        ]
        axes[0].legend(handles=handles, loc="lower left")
        axes[0].set_title("Redirect score and leakage across the schedule sweep (32 seeds per cell)")
        fig.tight_layout()
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(sweep_cell):
    _hot, _cool = sweep_cell(0.10, True), sweep_cell(0.05, True)
    _cold = sweep_cell(0.03, True)
    _nbad = lambda rs: sum(classify(r) != "clean" for r in rs)  # noqa: E731
    _rd = lambda rs: np.array([r["interventions"]["redirect"]["score"] for r in rs])  # noqa: E731
    _rc = lambda rs: np.array([r["val_recon"] for r in rs])  # noqa: E731
    _hold_bad = sum(_nbad(sweep_cell(p, False)) for p in PEAK_LRS)
    _hold_rc = np.median(np.concatenate([_rc(sweep_cell(p, False)) for p in PEAK_LRS]))
    _gamma = [r for r in _hot if r["interventions"]["redirect"]["red_pure"] < 0.05 and classify(r) == "clean"]
    _hot_ok = np.array([r["interventions"]["redirect"]["score"] for r in _hot if r not in _gamma])
    mo.md(
        f"Three results. First, **peak LR 0.05 with the anneal is the safe cell**: "
        f"{_nbad(_cool)}/32 unhealthy runs (0.10: {_nbad(_hot)}; 0.07: {_nbad(sweep_cell(0.07, True))}), "
        f"median score {np.median(_rd(_cool)):.2f}, and reconstruction as good as the hot cell "
        f"(median {np.median(_rc(_cool)):.6f} vs {np.median(_rc(_hot)):.6f}) — the 0.1 peak bought "
        f"nothing but risk. Cooler is not monotonically better: at 0.03 the model undercooks "
        f"({_nbad(_cold)} leaky runs, median score {np.median(_rd(_cold)):.2f}).\n\n"
        f"Second, **the anneal earns its keep** — just not where we suspected. Holding the "
        f"regularizers on actually lowers *typical* leak slightly, but it fattens the tail: "
        f"{_hold_bad} leaky runs across the four hold cells versus "
        f"{sum(_nbad(sweep_cell(p, True)) for p in PEAK_LRS)} with the anneal, as the live anchor "
        f"term keeps dragging pinkish *labeled* samples onto the axis in unlucky runs. It also "
        f"costs ~10× in reconstruction (median {_hold_rc:.6f} vs "
        f"{np.median(np.concatenate([_rc(sweep_cell(p, True)) for p in PEAK_LRS])):.6f}) and pushes "
        f"the redirect's damage to pure red past the ¼ design bound (median "
        f"{np.median([r['interventions']['redirect']['red_pure'] for p in PEAK_LRS for r in sweep_cell(p, False)]):.2f} "
        f"vs {np.median([r['interventions']['redirect']['red_pure'] for p in PEAK_LRS for r in sweep_cell(p, True)]):.2f} "
        f"with the anneal), forfeiting the predictable response ex-2.9.2 bought. The anneal isn't "
        f"the bug; annealing *while the optimizer is still hot* is.\n\n"
        f"Third, a stowaway: the worst hot-cell score ({_rd(_hot).min():.2f}, seed "
        f"{min(_hot, key=lambda r: r['interventions']['redirect']['score'])['seed']}) is not an "
        f"anchoring failure at all — the run anchored cleanly, but the redirect's fixed γ = 1 bias "
        f"failed to dominate that seed's pre-norm scale, so 'deleted' red passed through almost "
        f"untouched (damage to pure red {min(r['interventions']['redirect']['red_pure'] for r in _hot):.3f}). "
        f"That is ex-2.9.2's γ-calibration caveat recurring in 1 run of 256; excluding it, the hot "
        f"cell's floor is {_hot_ok.min():.2f}. γ should be calibrated per model, not fixed."
    )
    return


@app.cell(hide_code=True)
def _(arm):
    _base = arm("trajectories") + arm("attribution")
    _fb = arm("sweep")
    _cat = lambda rs: sum(classify(r) == "catastrophic" for r in rs)  # noqa: E731
    _like = [r for r in _fb if r["peak_lr"] == 0.10 and r["anneal"]]
    mo.md(f"""
    ## A bonus: the fallback term looks like a stabilizer

    Catastrophic failures — anchor lost, reconstruction collapsed, or leak
    beyond 0.3 — occurred only in the fallback-free arms: {_cat(_base)} of
    {len(_base)} base-config runs, versus **{_cat(_fb)} of {len(_fb)}**
    fallback-trained runs across all eight schedules ({_cat(_like)} of
    {len(_like)} in the like-for-like cell, same schedule as the base arms).
    Ex-2.9.2 saw this pattern at 32 seeds and hedged — "we can't tell one
    extra loss term apart from luck". At {len(_fb)} runs the pattern has held
    without exception. Mechanistically it is plausible: the fallback term
    pins the decoder's output at −e₀, which flattens the loss landscape
    around the redirect direction and may remove the escape route the
    instability uses. The schedule confound (only the like-for-like cell is
    a clean comparison) keeps this suggestive rather than settled, but it is
    now the second experiment in which fallback training produced zero
    catastrophes.
    """)
    return


@app.cell(hide_code=True)
def _(arm, sweep_cell):
    _cool = sweep_cell(0.05, True)
    _rd = np.array([r["interventions"]["redirect"]["score"] for r in _cool])
    _bad_t = [r for r in arm("trajectories") if classify(r) != "clean"]
    mo.md(f"""
    ## Takeaways

    The "schedule incompatible with some seeds" hypothesis is rejected on
    both counts: failures are not early (every failing run anchored first,
    breaking only during the 0.1-LR plateau) and not seed-bound (the
    known-bad inits trained cleanly under fresh streams; failures scatter).
    The anchored solution is simply metastable when the optimizer is too
    hot, and whether a run survives is a draw against the batch stream.

    For M2 defaults:

    - **Halve the LR peak to 0.05 and keep the anneal.** That cell had 0/32
      unhealthy runs, scores {_rd.min():.2f}–{_rd.max():.2f} (median
      {np.median(_rd):.2f}), and reconstruction as good as the hot schedule.
      Don't overshoot: 0.03 undercooks.
    - **Keep the fallback term** — for its designed purpose (a predictable
      intervention response), and for its apparent side effect of preventing
      catastrophic training failures.
    - **Calibrate the redirect's γ** against the model's pre-norm activation
      scale; a fixed γ = 1 silently no-ops on ~1 run in 250.
    - Cheap endpoint screening (leak < 0.1, anchor loss, recon) remains
      worthwhile: even the safe cell only bounds what we measured, and the
      failure mechanism is chaotic.

    A schedule this tuned still fixes the *symptom* — protection ends on a
    timer while the hazard (LR) is set by hand. [Ex-2.9.4](../ex-2.9.4/report.py)
    asks whether the weights can instead respond to the training signals
    themselves, which would remove the timing coupling altogether.

    ({len(_bad_t)} of 32 base-config seeds failed here, so the phenomenon is
    rare but material; all comparisons in this report are distributional
    across 32-seed cells, and per-seed pairings across conditions are
    meaningless — see ex-2.9.2's caveats.)
    """)
    return


if __name__ == "__main__":
    app.run()
