import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="medium",
    app_title="Experiment 2.9.3: why anchoring fails — timing, attribution, and a schedule fix",
    auto_download=["html"],
    css_file="../../report.css",
)

with app.setup(hide_code=True):
    import io

    import marimo as mo  # noqa: F401
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

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

    [Ex-2.9.2](../ex-2.9.2/report.py) identified two components to the variance
    in our ablation scores. The redistribution was fixed: fallback control
    gives the intervention a response we design ahead of time. The other part
    stayed open: On some seeds the concept never landed cleanly on its axis.

    Our guess was that the regularizer schedule is incompatible with some
    initializations: some starting weights simply cannot be anchored by this
    schedule. If that were true, the remedy would be a schedule search per seed
    or per init, which is expensive and gets worse as models grow.

    This experiment tests that, and it doesn't hold up. There are three arms,
    all built on ex-2.9.1's small color autoencoder.

    - Trajectories: retrain ex-2.9.2's base arm (the same 32 seeds), recording
      anchor progress, leakage, and reconstruction error at every step. When do
      failures happen, relative to the schedule?
    - Attribution: separate the two sources of randomness, the model init and
      the batch/label stream (16 inits × 8 streams, including the two inits
      whose earlier runs failed badly). A failure that comes from the init
      should repeat all along that init's row, while a failure that comes from
      the data ordering should scatter.
    - Schedule sweep: peak LR {0.10, 0.07, 0.05, 0.03} × regularizer anneal
      {on, off}, 32 seeds per cell, trained with the fallback term and scored by
      the `redirect` intervention (ex-2.9.2's recipe, so the score reflects
      anchoring quality).

    The experiment is [`experiment.py`](./experiment.py):

    ```bash
    bin/mini run docs/m1/ex-2.9.3/experiment.py --app modal --max-containers 16
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
            "```bash\nbin/mini run docs/m1/ex-2.9.3/experiment.py --app modal --max-containers 16\n```"
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
    mo.md(f"""
    {len(arm("trajectories")) + len(arm("attribution")) + len(arm("sweep"))}
    runs completed. The trajectory arm reproduces ex-2.9.2's base arm:
    {len(_bad)} of {len(_t)} runs end unhealthy. {len(_cat)} of them failed
    catastrophically, meaning the anchor was lost or reconstruction collapsed,
    and {len(_bad) - len(_cat)} finished with non-red colors leaking onto the
    anchored axis. The per-step traces show that every failing run anchored
    successfully first, then broke during the high-LR plateau. The attribution
    arm confirms the failures are not a property of the initialization, and the
    sweep finds a plain schedule fix: halve the LR peak and keep the anneal.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## When failures happen

    Each line below is one seed's training run under the original schedule. The
    top panel tracks anchor progress: z₀ of pure red, which reaches 1 when *red*
    sits exactly on its anchor. The middle panel tracks leakage, the mean |z₀|
    over colors that are clearly not red. The bottom panel is the schedule
    itself: the learning rate ramps to its 0.1 peak at step 750 and holds there,
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
        alt_text=f"""
        Three charts stacked on a shared step axis from 0 to 1500. Top: anchor progress, the
        z0 of pure red, for all 32 seeds. Every line, including the {len(_bad)} failures drawn
        in color, climbs from near 0 to about 1 by step 750. After that the gray healthy lines
        hold at 1 while the colored ones diverge. Seeds 22 and 8 dip briefly as the learning
        rate reaches its peak, and both recover on this metric, though seed 22's reconstruction
        does not. Seed 15 breaks away to about 0.65 from step 900, and seed 27 falls to about
        0.4 after step 1100. Middle: leakage for the same runs. Healthy lines settle near 0.05
        while the failures climb to 0.1 to 0.6 in the same late window. Bottom: the schedule,
        with the learning rate ramping to 0.1 at step 750 and holding, and the regularizer
        weights annealing to zero by step 1425. A shaded band marks the high-LR plateau, where
        every failure occurs.
        """,
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
    mo.md(f"""
    All {len(_bad)} failing runs reached z₀(red) > 0.9, a clean anchor, between steps
    {min(_anchored)} and {max(_anchored)}, right as the LR approaches its peak. The {len(_drops)}
    that later lost it did so between steps {min(_drops)} and {max(_drops)}. Seeds 22 and 8 slip
    the moment the LR tops out, and reconstruction collapses outright on 22. Seeds 15 and 27 slip
    only once the anchor weight has annealed low: by step {max(_drops)} it is
    {float(_w_anchor[min(max(_drops), len(_w_anchor) - 1)]):.3f}, too weak to pull red back.

    So the anchored solution seems to be metastable at this learning rate. The regularizers hold
    it in place while they are on, and the timed anneal removes that hold while the optimizer is
    still hot. Nothing about these seeds stopped them from anchoring; they were unlucky during the
    plateau. If that is right, whether a run fails should follow the randomness of training rather
    than the initialization, which we test below.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Init versus data stream

    The attribution arm retrains 16 inits × 8 batch/label streams under the
    original schedule. If failures come from the init, the two inits whose
    earlier runs failed catastrophically (22 and 27) should fail again. If they
    come from the luck of training, they should scatter across the grid instead.
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
        alt_text=f"""
        Heatmap of final leakage over a 16-by-8 grid. Model inits run down the rows (0 through
        13, then the two earlier catastrophic inits 22 and 27), and batch/label streams run
        across the columns. Most cells are pale, with leak around {np.median(_leak):.2f}. The
        {int(_fail.sum())} failed cells are marked with crosses and scatter across the grid: no
        row has more than {int(_fail.sum(1).max())} of 8, and the rows for inits 22 and 27 are
        entirely clean. The darkest cell is init 1 under stream 1, with leak {_leak.max():.2f}.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
        _cmap = LinearSegmentedColormap.from_list("leak", light_dark(["#eef3f7", "#1a5f8a"], ["#20242a", "#6ab0d4"]))
        im = ax.imshow(_leak, cmap=_cmap, vmin=0, vmax=0.35, aspect="auto")
        for i in range(len(_inits)):
            for j in range(len(_streams)):
                if _fail[i, j]:
                    ax.text(
                        j,
                        i,
                        "×",
                        ha="center",
                        va="center",
                        color=light_dark("#d55e00", "#ffb000"),
                        fontsize=13,
                        path_effects=[pe.withStroke(linewidth=2, foreground=light_dark("#ffffff", "#000000"))],
                    )
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
    mo.md(f"""
    Inits 22 and 27, catastrophic under their earlier streams, trained cleanly under **all 8 fresh
    streams** ({_n2227} failures between them). The {len(_fails)} unhealthy runs (of 128;
    {len(_cats)} catastrophic) instead scatter across {len(_by_init)} different inits, none failing
    more than {max(_by_init.values())} of 8, with mild clustering by stream (stream 6 accounts for
    {_by_stream.get(6, 0)}). So the incompatible-init hypothesis does not hold: the same init
    succeeds or fails depending on which random batches and label draws it sees during the hot
    phase.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Learning rate versus regularizer anneal

    At first the anneal looked responsible, since it removes the hold the anchor
    term provides. But the sweep says otherwise: holding the regularizers on to the
    end (`anneal off`) makes things worse at every peak, while halving the LR
    peak removes the failures entirely. These runs train with the fallback term
    and are scored by the redirect intervention, as in ex-2.9.2.
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
        alt_text=f"""
        Two strip plots stacked over eight conditions: peak learning rates 0.10, 0.07, 0.05,
        and 0.03, each with the regularizer anneal on (blue) or off (gray), 32 seeds per
        condition. Top: redirect selectivity scores. Medians hover around 0.87 to 0.91
        everywhere, but 0.03 with anneal sits lower at {np.median(_scores[(0.03, True)]):.2f},
        and one blue outlier at 0.10 falls to {_scores[(0.10, True)].min():.2f}. Bottom: final
        leakage on a log scale, with a dashed line at the 0.1 degraded threshold. With the
        anneal on, leak tightens as the peak drops, and at 0.05 no seed crosses the line. With
        the anneal off, every condition has a tail of 4 to 8 seeds above the line.
        """,
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
    mo.md(f"""
    Three results come out of the sweep.

    First, peak LR 0.05 with the anneal is the safe cell: {_nbad(_cool)}/32 unhealthy runs
    (0.10 gives {_nbad(_hot)}; 0.07 gives {_nbad(sweep_cell(0.07, True))}), a median score of
    {np.median(_rd(_cool)):.2f}, and reconstruction as good as the hot cell (median
    {np.median(_rc(_cool)):.6f} versus {np.median(_rc(_hot)):.6f}). On the other
    hand, at 0.03 the model doesn't train properly, with {_nbad(_cold)} leaky
    runs and a median score of {np.median(_rd(_cold)):.2f}.

    Second, the anneal should be kept. Holding the regularizers on lowers typical leak a
    little, but it widens the tail: {_hold_bad} leaky runs across the four hold cells versus
    {sum(_nbad(sweep_cell(p, True)) for p in PEAK_LRS)} with the anneal, because the live anchor
    term keeps pulling pinkish-labeled samples onto the axis in unlucky runs. Holding on also
    costs about 10× in reconstruction (median {_hold_rc:.6f} versus
    {np.median(np.concatenate([_rc(sweep_cell(p, True)) for p in PEAK_LRS])):.6f}).

    Third, γ should be calibrated per model. The worst hot-cell score ({_rd(_hot).min():.2f}, seed
    {min(_hot, key=lambda r: r["interventions"]["redirect"]["score"])["seed"]}) anchored
    cleanly, but the redirect's fixed γ = 1 bias was too small to dominate that seed's
    pre-norm scale, so "deleted" red passed through almost untouched (damage to
    pure red {min(r["interventions"]["redirect"]["red_pure"] for r in _hot):.3f}).
    Excluding it, the hot cell's floor is {_hot_ok.min():.2f}.
    """)
    return


@app.cell(hide_code=True)
def _(arm):
    _base = arm("trajectories") + arm("attribution")
    _fb = arm("sweep")
    _cat = lambda rs: sum(classify(r) == "catastrophic" for r in rs)  # noqa: E731
    _like = [r for r in _fb if r["peak_lr"] == 0.10 and r["anneal"]]
    mo.md(f"""
    ## The fallback term looks like a stabilizer

    Catastrophic failures — meaning the anchor was lost, reconstruction
    collapsed, or leak went beyond 0.3 — occurred only in the fallback-free
    arms: {_cat(_base)} of {len(_base)} base-config runs, against **{_cat(_fb)}
    of {len(_fb)}** fallback-trained runs across all eight schedules
    ({_cat(_like)} of {len(_like)} in the like-for-like cell, the same schedule
    as the base arms). Ex-2.9.2 saw this pattern at 32 seeds and was careful
    about reading much into it, since one extra loss term is hard to tell apart
    from luck at that scale. At {len(_fb)} runs the pattern has held without
    exception.

    A mechanism seems plausible. The fallback term pins the decoder's output at
    −e₀, which flattens the loss landscape around the redirect direction and may
    remove the direction along which the instability grows. Only the
    like-for-like cell is a clean comparison, so the schedule confound prevents
    this from being conclusive. Still, this is now the second experiment in
    which fallback training produced zero catastrophes.
    """)
    return


@app.cell(hide_code=True)
def _(arm, sweep_cell):
    _cool = sweep_cell(0.05, True)
    _rd = np.array([r["interventions"]["redirect"]["score"] for r in _cool])
    _bad_t = [r for r in arm("trajectories") if classify(r) != "clean"]
    mo.md(f"""
    ## Conclusion

    The "schedule incompatible with some seeds" hypothesis does not hold up
    under either test. Failures come late: every failing run anchored first and
    broke only during the 0.1-LR plateau. And they follow the data stream rather
    than the init: the earlier bad inits trained cleanly under fresh streams, and
    the failures scatter. The anchored solution is metastable when the optimizer
    is too hot, and whether a run stays clean comes down to the luck of the batch
    stream.

    For M2 defaults:

    - Reduce the LR. That cell had 0/32 unhealthy runs, scores {_rd.min():.2f}
      to {_rd.max():.2f} (median {np.median(_rd):.2f}), and reconstruction as
      good as the hot schedule. Reducing it only during the anneal phase may be
      enough.
    - Keep the fallback term, both for its designed purpose (a predictable
      intervention response) and for its apparent side effect of preventing
      catastrophic training failures.
    - Calibrate the redirect's γ against the model's pre-norm activation scale.
      A fixed γ = 1 silently no-ops on about 1 run in 250.
    - Keep the cheap endpoint screening (leak < 0.1, anchor loss, recon). Even
      the safe cell only bounds what we measured, and the failure mechanism is
      chaotic.

    A tuned schedule fixes the symptom but still requires tuning.
    [Ex-2.9.4](../ex-2.9.4/report.py) asks whether the weights can instead
    respond to the training signals themselves, which would remove the timing
    coupling altogether.
    """)
    return


if __name__ == "__main__":
    app.run()
