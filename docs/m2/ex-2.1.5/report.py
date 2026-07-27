import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.5: disjoint vocabularies and more named colors",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo puts the notebook's directory on sys.path, so the experiment
    # definition is importable — refs and sweep constants can't drift.
    from experiment import ARMS, ARRAYS_REF, CROSS_REF, METRICS_REF, SEEDS
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import light_dark, themed
    from sca import baselines as bl
    from sca import vis as sv
    from sca.compute import geometry as gm
    from sca import vis_probes as vp
    from sca.data import mixed_vocab as mv
    from sca.data.mixed_vocab import LANDMARKS

    use_publisher(report_bundle(__file__))

    def load_results() -> tuple[dict, dict[str, np.ndarray]] | None:
        """Resolve metrics and stacked per-condition arrays from the store, or None if unpublished."""
        store = project_store()
        arts = store.get_refs([METRICS_REF, ARRAYS_REF])
        m_art, a_art = arts[METRICS_REF], arts[ARRAYS_REF]
        if m_art is None or a_art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            m_path, a_path = store.get_many([(m_art, Path(d) / "metrics.json"), (a_art, Path(d) / "arrays.npz")])
            metrics = json.loads(m_path.read_text())
            with np.load(a_path) as z:
                arrays = {k: z[k] for k in z.files}
        return metrics, arrays

    def load_cross() -> dict | None:
        """Cross-form exact match for the bridge condition, or None if it hasn't been scored.

        Written by `cross_eval.py`, which sits outside the sweep's DAG — see H5.
        """
        store = project_store()
        art = store.get_refs([CROSS_REF])[CROSS_REF]
        if art is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            return json.loads(store.get(art, Path(d) / "cross.json").read_text())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.5: disjoint vocabularies and more named colors

    Do two surface languages for the same domain converge on one internal
    geometry?

    The corpus for this experiment holds two sublanguages that are never seen
    together: color-mixing equations written with names, and the same arithmetic
    written as hex codes. For example:

    ```
    melon + ultramarine = …
    #e26 + #48a = #958
    ```

    Nothing directly ties a name to a hex value; there are no alias lines and no
    mixed-form equations. So if the model places both vocabularies in the same
    latent geometry, the cause is pressure internal to the network, e.g.
    capacity constraints.

    Graded concept labels would be computed from color values (see [M1/Ex-1.7]),
    so they can be used with either form. If this experiment finds that the two
    sublanguages use different latent representations, a later one could test
    whether anchoring the same concept in both forms pulls the geometries
    together, and it could quantify how many are needed. It would be remarkable
    if a single anchor aligned the whole cube.

    Lineage: the base language (ex-2.1.1, 2.1.2) bridged names and hex with
    alias and cross lines, and its 27-name sub-grid turned out to be too sparse
    to grade (ex-2.1.4). The single-vocabulary experiments (ex-2.1.3, 2.1.4)
    removed hex entirely. This experiment keeps both forms but removes the
    bridge, and replaces the coarse named sub-grid with 140 real color names
    spread through the full 8-bit cube.

    /// note | How to read this draft
    This report was preregistered (drafted before the experiment ran).
    Admonitions marked TODO are placeholders: each states what its figure or
    table should show and the pattern we expect. As results land, placeholders are
    replaced with observations. The hypotheses section is frozen except for
    immaterial changes; any analysis invented after seeing data goes under
    "Exploratory analyses" and is marked as post hoc.
    ///

    [M1/Ex-1.7]: https://z0u.github.io/ex-preppy/m1-color-mlp/ex-1.7-sparse-labels.html#Labelling
    """)
    return


@app.cell(hide_code=True)
def _():
    _res = load_results()
    mo.stop(_res is None, mo.md("_Results are not published yet; the analysis cells below render once they are._"))
    assert _res is not None
    metrics, arrays = _res
    cells = {c["label"]: c for c in metrics["cells"]}
    stats = metrics["corpus_stats"]
    cross = load_cross()

    def seed_mean(arm: str, set_name: str, key: str) -> float:
        vals = [cells[f"{arm}-s{s}"]["sets"][set_name][key] for s in SEEDS]
        return float(np.mean([v for v in vals if v is not None]))

    def maps(arm: str, path: str) -> np.ndarray:
        """The seed mean of one stored depth × landmark map."""
        return np.mean([arrays[f"{arm}-s{s}/{path}"] for s in SEEDS], axis=0)

    return arrays, cells, cross, maps, seed_mean, stats


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data (the language)

    First, let's define two RGB grids (cubes):

    | Space | Levels | Hex digits | Example ("cyan") | Precision | Colors |
    |---|---|---|---|---|---|
    | *hex grid* | 16 | 3 | `#0ff` | 4-bit | 4,096 |
    | *full cube* | 256 | 6 | `#00ffff` | 8-bit | 16,777,216 |

    **Names.** The named palette comes from the [xkcd color survey][xkcd]: all
    949 names, ordered by farthest-point selection so that the first $N$ form
    the most uniform palette available at that size. The center condition uses
    $N = 140$; at that size the minimum and median nearest-neighbor distances
    are ≈ 41 and ≈ 46 (8-bit Euclidean), against 4 and 27 for the CSS keyword
    list. Names map to points on the full cube.

    /// details | On not filtering by name
    An earlier draft kept single-word names only, to avoid modifier words like
    `light` that spell out part of the value; but that filter swapped one
    selection bias for another and cost uniformity (28/37 at the same size), so
    the selection rule is now distance alone. Most selected names are
    multi-word, and the longest at $N = 140$ is `blue with a hint of purple`, so
    lines run to ≈ 86 characters and the block size grows accordingly. If
    spelled modifiers look like they shortcut the name geometry, we could do a
    separate analysis of common substrings.
    ///

    [xkcd]: https://xkcd.com/color/rgb/

    **Hex.** Hex operands are drawn from a fixed random subset of the hex grid,
    sampled point-by-point rather than as a regular sub-grid. The subset
    constrains operands only: the correct completion of a hex equation may be
    any point on the hex grid.

    To prepare training data, we pick two operands from one sublanguage (names
    or hex) and compute the result. The mix is the channel-wise round-half-up
    mean, always computed in the full cube; the two forms differ only in how
    values enter and leave it:
    - Named colors already sit in the full cube, but the answer must be snapped
      to the nearest named color. Distance ties break by a coin flip seeded from
      the mix value.
    - Hex operands are lifted to it by digit repetition (`#f80` → `#ff8800`)
      and the answer is snapped back to the hex grid.

    Two forms appear in every condition of the sweep, and a third only in the
    _bridge_ arm:

    | Form  | Example                       | Sweep conditions |
    |-------|-------------------------------|------------------|
    | named | `melon + ultramarine = <name>`| all              |
    | hex   | `#e26 + #48a = #958`          | all              |
    | cross | `melon + #48a = #<hex>`       | bridge arm       |

    (Examples are illustrative until the corpus code lands.) Named equations
    always answer with a name, and hex equations with a hex code, so the answer
    form is determined by the prompt form; nothing about the result value
    changes which vocabulary the answer uses.
    """)
    return


@app.cell(hide_code=True)
def _(stats):
    _st = stats["n140-h216"]
    _pal = np.array(list(_st["palette"].values()), dtype=float) / 255
    _ops = np.array([mv.lift((int(h[1], 16), int(h[2], 16), int(h[3], 16))) for h in _st["hex_ops"]], dtype=float) / 255

    @themed(
        name="palettes",
        alt_text="""
            Two color-cube panels, each a hexagonal silhouette of the RGB cube
            standing on its black corner, with data-colored dots. Left: named
            colors shown as dots, evenly spaced through the whole solid with no
            two dots touching. Right: hex colors, reaching the same extent but
            visibly lumpy at close range — touching pairs and triples in some
            places, bare patches in others.
        """,
        caption="""
            The two operand sets in the RGB cube (center corpus). Left: the 140
            named colors, spaced by farthest-point selection; right: the 216
            hex operands, a uniform draw from the 4,096-point grid.
        """,
    )
    def _plot() -> plt.Figure:
        fig, _axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
        for _ax, _pts, _title in [(_axes[0], _pal, "names (140)"), (_axes[1], _ops, "hex operands (216)")]:
            sv.plot_rgb_cube(_ax, _pts)
            _ax.set_title(_title)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, stats):
    # Nearest-neighbor spacing and worst coverage gap, in 8-bit units, for the two
    # operand sets and for fresh uniform draws of the same size — the reference that
    # says whether the visible clumping is unusual or just what uniform sampling is.
    _grid = np.array([mv.lift((_r, _g, _b)) for _r in range(16) for _g in range(16) for _b in range(16)], dtype=float)

    def _spacing(_pts: np.ndarray) -> tuple[float, float, float]:
        _dd = np.linalg.norm(_pts[:, None] - _pts[None], axis=2)
        np.fill_diagonal(_dd, np.inf)
        _nn = _dd.min(1)
        _gap = np.linalg.norm(_grid[:, None] - _pts[None], axis=2).min(1).max()
        return float(_nn.min()), float(np.median(_nn)), float(_gap)

    _ops8 = np.array(
        [mv.lift((int(_h[1], 16), int(_h[2], 16), int(_h[3], 16))) for _h in stats["n140-h216"]["hex_ops"]], dtype=float
    )
    _hex, _names = _spacing(_ops8), _spacing(np.array(list(stats["n140-h216"]["palette"].values()), dtype=float))
    _draws = np.array(
        [_spacing(_grid[np.random.default_rng(1000 + _i).permutation(len(_grid))[: len(_ops8)]]) for _i in range(24)]
    )
    _typical = np.median(_draws, axis=0)
    _better_gap = float((_draws[:, 2] > _hex[2]).mean())

    # Empirical control: hex-dense draws 2048 operands from the same grid, so its gaps are
    # a tenth the size. If the clumping shaped the hex geometry, the two maps would differ.
    def _agreement(_t: str) -> tuple[float, float]:
        _a = np.mean([arrays[f"center-s{_s}/probes/hex/{_t}/r2_strict"] for _s in SEEDS], axis=0)
        _b = np.mean([arrays[f"hex-dense-s{_s}/probes/hex/{_t}/r2_strict"] for _s in SEEDS], axis=0)
        return float(np.corrcoef(_a.ravel(), _b.ravel())[0, 1]), float(np.abs(_a - _b).mean())

    _corrs = {_t: _agreement(_t) for _t in ("op1", "op2", "mix")}
    _corr_lo, _corr_hi = min(_c for _c, _ in _corrs.values()), max(_c for _c, _ in _corrs.values())
    _dmax = max(_d for _, _d in _corrs.values())

    mo.md(f"""
    /// details | Is the clumping in the hex panel a problem?
    It is conspicuous next to the names, and it is exactly what the two
    sampling rules predict: farthest-point selection cannot leave a clump,
    and uniform sampling clumps at every scale. Nearest-neighbor distances in
    8-bit units (closest pair, then median) are {_hex[0]:.0f} and {_hex[1]:.0f}
    for the hex operands, against {_names[0]:.0f} and {_names[1]:.0f} for the
    names and {_typical[0]:.0f} and {_typical[1]:.0f} for a typical fresh draw
    of {len(_ops8)} grid points. So the subset's local spacing matches the
    median draw; it is not an unlucky one. Its coverage is if anything better
    than average: the farthest any of the 4,096 grid points sits from an
    operand is {_hex[2]:.0f}, against {_typical[2]:.0f} for the typical draw,
    and {_better_gap * 100:.0f}% of draws leave a wider hole.

    Whether that biases the *geometry* is answered by the sweep. The *hex-dense*
    condition draws 2,048 operands from the same grid — a tenth of the spacing, and
    no visible clumping — and its hex probe map is the same map: the two conditions'
    depth × landmark maps correlate {_corr_lo:.2f} to {_corr_hi:.2f} across the
    three targets, with a mean absolute difference of at most {_dmax:.2f} in
    $R^2$. The two conditions also answer held-out hex equations equally well (see
    the density arms under H1). Nothing about the hex form's layout appears to
    be an artifact of the 216-point draw, so the same should hold for what
    transfers out of it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(stats):
    _cols = ["corpus", "named", "hex", "cross", "held-out named", "held-out hex", "max line", "answer ppl", "reachable"]

    def _row(key: str) -> str:
        _s = stats[key]
        _vals = [
            f"{_s['n_lines']['named']:,}",
            f"{_s['n_lines']['hex']:,}",
            f"{_s['n_lines']['cross']:,}",
            f"{_s['n_holdout']['named']:,}",
            f"{_s['n_holdout']['hex']:,}",
            f"{max(v['max'] for v in _s['line_length'].values())}",
            f"{_s['answer_perplexity']:.0f}",
            f"{_s['answers_reachable']}",
        ]
        return f"<tr><td>{key}</td>" + "".join(f'<td class="num">{v}</td>' for v in _vals) + "</tr>"

    _thead = f"<tr><th>{_cols[0]}</th>" + "".join(f'<th class="num">{h}</th>' for h in _cols[1:]) + "</tr>"
    mo.vstack(
        [
            mo.md("""
            Corpus statistics, one row per corpus (conditions that share
            names × hex × bridge share a corpus). The design study's numbers
            held up: answer perplexity is 86 over 140 names with every name
            reachable as an answer (the 250-name palette reaches 248), and
            the longest line is 85 characters, inside the block size of 128.
            Line counts are the training corpus's 100,000 lines split by form.
            """),
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table">'
                + _thead
                + "".join(_row(k) for k in sorted(stats))
                + "</table></div>"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Measurements

    Exact-match accuracy is scored per form on seen and held-out operand
    pairs. On a small palette, raw accuracy can flatter a weak strategy, so
    each number sits beside two reference guessers. One never reads the
    prompt: it always answers with the centroid of the training answers. The
    other is handed the answer's neighborhood for free: it guesses uniformly
    among the $k$ palette entries nearest the true mix (the irregular-palette
    version of the one-step-shell null in earlier reports). The model has
    learned something only where it beats both.

    Geometry is read with ridge probes (leave-one-out, as in
    `ridge_probe_loo`) fitted for operand and mix values at every layer and
    every token position, separately per form. The full scan replaces the
    hand-picked probe sites of earlier reports with a map, so the alignment
    measures below don't depend on us choosing the right position in advance.

    /// admonition | Amendment: a second holdout protocol
    The preregistered estimator holds out one *equation* at a time. But the
    held-out equation's colors and hex digits still appear in other equations
    in the fit, so a probe can score well just by memorizing which token has
    which value. That is a fair way to ask whether a value is present at a site,
    and one analysis below uses it for that. But it's not a good test of H2,
    which is a statement about geometry.

    So each site now also gets a stricter holdout. To score a value, every
    equation containing it (as either operand or the answer) is removed from the
    fit, and the probe must place the unseen value using only the others. The
    whole-value holdout is needed because the same hex digit serves all three
    channels, and a color can be operand 1 in one line and operand 2 in another.

    Both scores are reported. The preregistered one decides H2 (and it turns out
    the two agree at the site used for that decision (0.944 against 0.941). We
    also added two landmarks, the spaces closing each operand, after the strict
    holdout showed that a named operand's value sits on them rather than on the
    name's characters.
    ///

    One subtlety: token positions don't naturally line up. Names vary in length,
    and hex expressions are shorter than named ones. Positions are therefore
    indexed by grammar landmarks (the last character of each operand, the
    operator, the pre-answer position, and answer characters counted from the
    answer's start and end), and the cross-form measures compare only at
    landmarks the two forms share.

    Alignment between the two forms is scored two ways:

    - Transfer ratio $\rho$: zero-shot cross-form $R^2$ divided by within-form
      $R^2$, at the same layer and landmark. Zero-shot means the probe is
      fitted on one form's activations and applied unchanged to the other's.
      Two guards keep the ratio well-behaved. First, $\rho$ is reported only
      where the within-form $R^2 \ge 0.5$; below that the site isn't measuring
      geometry, and a small denominator makes the ratio erratic. Second,
      negative cross-form $R^2$ clips to zero, so $\rho \in [0, 1]$ with 0
      meaning no transfer.
    - Principal angles between the row-spaces of the two forms' fitted probes:
      a graded measure of whether the two decoders use the same directions of
      the residual stream.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    - **H1.** The disjoint language trains at the center condition: hex+hex exact
      match on unseen pairs is comparable to ex-2.1.1's hex levels, and
      name+name held-out exact match clears the $k$-NN analogues of the
      neighborhood nulls.
    - **H2.** Each sublanguage develops latent linear color geometry, with
      form-specific layout: name-form probes decode operands and mix with mix
      $R^2 \approx 0.9$ at the pre-answer position in the last layer; hex-form
      answers assemble just-in-time, channel by channel, with pre-answer
      full-mix $R^2$ staying low. An elevated pre-answer hex-mix $R^2$ would
      instead be evidence of cross-form coupling (see H3).
    - **H3.** The two latent geometries live apart in sweep conditions that have no
      bridging grammar and width d64: $\rho < 0.2$, and the two probes'
      row-spaces show large principal angles. ($0.2 < \rho < 0.8$ reads as
      partial sharing and falsifies the crisp version of both H3 and H5.)
    - **H4.** Narrowing the stream aligns the forms: $\rho$ and subspace
      overlap rise monotonically over d64 → d32 → d16, with d16-L8 the most
      aligned condition.
    - **H5.** Adding the cross form produces alignment: the bridge condition (d64)
      reaches $\rho > 0.8$. The star design tests this at one width only;
      bridge × width conditions are candidates for a follow-up round if H4 shows
      width matters.
    - **H6.** At L8, name+name accuracy improves at fixed width and the mix
      crystallizes before the last layer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The sweep

    A star design around one center condition, three seeds per condition:

    | Condition   | Names | Hex ops | Bridge | Width | Depth |
    |-------------|-------|---------|--------|-------|-------|
    | center      | 140   | 216     | none   | 64    | 4     |
    | L8          | 140   | 216     | none   | 64    | 8     |
    | d32         | 140   | 216     | none   | 32    | 4     |
    | d16         | 140   | 216     | none   | 16    | 4     |
    | d16-L8      | 140   | 216     | none   | 16    | 8     |
    | hex-dense   | 140   | 2048    | none   | 64    | 4     |
    | palette-250 | 250   | 216     | none   | 64    | 4     |
    | bridge      | 140   | 216     | cross  | 64    | 4     |

    Each arm provides data to score the hypotheses. L8, the width conditions, and
    d16-L8 score H4 and H6; the bridge condition scores H5. The two density arms
    attach to H1: *hex-dense* checks that hex accuracy and geometry aren't
    artifacts of the 216-point operand subset (expected: little change —
    ex-2.1.1's hex arithmetic generalized from far sparser coverage), and
    *palette-250* extends the density axis of ex-2.1.3 to the irregular palette
    (expected: named held-out accuracy holds or improves, with misses staying
    neighbor-level).

    Attention is held at 8 heads × 8 dims in every condition; only the residual
    stream and the MLP scale with width. The ngpt-scaling sweep validated
    widths {32, 64, 128} under this scheme, so d16 sits one step below the
    tested range. If the d16 conditions train poorly, that is an architecture
    effect to report, and the width trend in H4 rests on d64 → d32.
    """)
    return


@app.cell(hide_code=True)
def _(cells):
    _thead = (
        "<tr><th>condition</th>" + "".join(f'<th class="num">{h}</th>' for h in ("width", "depth", "params")) + "</tr>"
    )
    _rows = "".join(
        f"<tr><td>{_a}</td>"
        + f'<td class="num">{ARMS[_a]["width"]}</td>'
        + f'<td class="num">{ARMS[_a]["depth"]}</td>'
        + f'<td class="num">{cells[f"{_a}-s0"]["n_params"]:,}</td>'
        + "</tr>"
        for _a in ARMS
    )

    mo.md(f"""
    Parameter counts per condition (seeds share a count). The full sweep —
    4 corpora, 24 training conditions, 24 evals.

    /// details | Runtime
    The sweep ran in about 2.5 hours of wall time on five L4 containers. It
    would have been faster, but there were cross-region I/O issues.
    ///

    <div class="report-table-scroll"><table class="report-table">{_thead}{_rows}</table></div>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    All 24 conditions converge smoothly under the shared 100-epoch schedule —
    including the d16 conditions, which sit one step below ngpt-scaling's
    validated width range. No width-gated instability appeared.
    """)
    return


@app.cell(hide_code=True)
def _(cells):
    @themed(
        name="loss-curves",
        alt_text="""
            Eight small panels of loss versus epoch, one per sweep condition, each
            with three validation curves (one per seed) and three fainter
            training curves. Every curve descends smoothly and flattens within
            the 100-epoch budget; no panel shows divergence or oscillation.
        """,
        caption="""
            Loss per epoch for every condition: validation solid, training faint,
            one line per seed. All conditions share the character vocabulary, so
            per-token losses are comparable across panels.
        """,
    )
    def _plot() -> plt.Figure:
        fig, _axes = plt.subplots(2, 4, figsize=(9.0, 4.4), sharex=True, sharey=True)
        _c = light_dark("#1a5f8a", "#6ab0d4")
        for _ax, _arm in zip(_axes.flat, ARMS, strict=True):
            for _s in SEEDS:
                _cell = cells[f"{_arm}-s{_s}"]
                _ax.plot([v for v in _cell["train_loss"] if v is not None], color=_c, lw=0.8, alpha=0.35)
                _ax.plot([v for v in _cell["val_loss"] if v is not None], color=_c, lw=1.0)
            _ax.set_title(_arm, fontsize=9)
            _ax.grid(alpha=0.3)
        for _ax in _axes[-1]:
            _ax.set_xlabel("epoch")
        for _ax in _axes[:, 0]:
            _ax.set_ylabel("loss")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exact-match accuracy (H1)

    The center condition answers held-out hex equations almost perfectly (0.996
    exact) and held-out named equations at 0.667, with no malformed
    completions in any of its eval sets. The named score sits far above the
    prompt-blind centroid (0.043). One null turned out to be simple: in this
    language the true answer is always the candidate nearest the mix, so
    guessing uniformly among the $k$ nearest candidates scores exactly $1/k$.
    The hardest version is a coin flip between the two nearest candidates,
    0.5, and the model clears that too, by a margin the 1,946 held-out named
    pairs can resolve.
    """)
    return


@app.cell(hide_code=True)
def _(seed_mean, stats):
    _nulls = stats["n140-h216"]["nulls"]
    _cols = ["eval set", "exact", "guess dist", "floor dist", "blind null", "2-NN null"]

    def _row(_set: str) -> str:
        _vals = [
            f"{seed_mean('center', _set, 'accuracy'):.3f}",
            f"{seed_mean('center', _set, 'guess_dist'):.3f}",
            f"{seed_mean('center', _set, 'floor_dist'):.3f}",
            f"{_nulls[_set]['blind']['acc']:.3f}",
            f"{_nulls[_set]['knn']['k2']['acc']:.3f}",
        ]
        return f"<tr><td>{_set.replace('_', ' ')}</td>" + "".join(f'<td class="num">{v}</td>' for v in _vals) + "</tr>"

    _thead = f"<tr><th>{_cols[0]}</th>" + "".join(f'<th class="num">{h}</th>' for h in _cols[1:]) + "</tr>"
    mo.vstack(
        [
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table">'
                + _thead
                + "".join(_row(s) for s in ("named_seen", "named_holdout", "hex_seen", "hex_holdout"))
                + "</table></div>"
            ),
            mo.md("""
            Center condition, mean over three seeds. Distances are Euclidean in the
            unit cube: *guess dist* from the emitted answer to the exact mix,
            *floor dist* from the true answer to the exact mix (the quantization
            floor). The guesses sit near the floor even where exact match
            misses.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Named accuracy is well short of 1 even on pairs the model trained on
    (0.841), which reads at first like a model that never learned the geometry.
    In contrast, ex-2.1.3 answered a 216-color named vocabulary at ≈ 1.0.

    But ex-2.1.3's colors sat on a regular grid closed under mixing: every
    answer landed on a vocabulary point, and the runner-up was a full grid step
    away (0.2 of the unit cube). Here the answer is the nearest of 140
    irregularly placed names, and the median gap between the nearest name and
    the second-nearest is six times smaller. In the exploratory section ("The
    named ceiling is a resolution limit"), accuracy tracks that gap prompt by
    prompt, and a single effective-precision number per channel accounts both
    for this experiment's named accuracy and for the earlier grids scoring
    near 1. So the geometry is present; what runs out is resolution, as in
    ex-2.1.3.
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    @themed(
        name="miss-ranks",
        alt_text="""
            Two bar panels of guess rank. For named held-out prompts, about two
            thirds of the mass is at rank 0 and most of the rest at ranks 1 and
            2, with a small tail beyond 6. For hex held-out prompts virtually
            all mass is at rank 0.
        """,
        caption="""
            Rank of the emitted answer among the form's candidate vocabulary,
            ordered by distance to the true mix; rank 0 is the correct answer.
            Center condition, pooled over seeds.
        """,
    )
    def _plot() -> plt.Figure:
        fig, _axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
        _cap = 6
        for _ax, _set, _title in [
            (_axes[0], "named_holdout", "named held-out"),
            (_axes[1], "hex_holdout", "hex held-out"),
        ]:
            _r = np.concatenate([arrays[f"center-s{_s}/evals/{_set}/rank"] for _s in SEEDS])
            _r = _r[_r >= 0]
            _counts = np.bincount(np.minimum(_r, _cap), minlength=_cap + 1) / len(_r)
            _ax.bar(range(_cap + 1), _counts, color=light_dark("#1a5f8a", "#6ab0d4"))
            _ax.set_xticks(range(_cap + 1), [*map(str, range(_cap)), f"{_cap}+"])
            _ax.set_title(_title)
            _ax.set_xlabel("guess rank (0 = correct)")
            _ax.grid(alpha=0.3, axis="y")
        _axes[0].set_ylabel("fraction of prompts")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(seed_mean, stats):
    _corpus_of = {"center": "n140-h216", "hex-dense": "n140-h2048", "palette-250": "n250-h216"}
    _thead = (
        "<tr><th>condition</th>"
        + "".join(f'<th class="num">{h}</th>' for h in ("named held-out", "hex held-out", "answer ppl"))
        + "</tr>"
    )
    _rows = "".join(
        f"<tr><td>{_a}</td>"
        + f'<td class="num">{seed_mean(_a, "named_holdout", "accuracy"):.3f}</td>'
        + f'<td class="num">{seed_mean(_a, "hex_holdout", "accuracy"):.3f}</td>'
        + f'<td class="num">{stats[_k]["answer_perplexity"]:.0f}</td>'
        + "</tr>"
        for _a, _k in _corpus_of.items()
    )
    mo.vstack(
        [
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table">' + _thead + _rows + "</table></div>"
            ),
            mo.md("""
            The density arms, both far above their nulls. Hex accuracy is
            insensitive to the operand-subset size, as expected. At 250 names
            the mean named held-out accuracy drops to 0.564, in the predicted
            direction — the answer perplexity is 153 against 86 — though the
            three seeds span 0.47 to 0.66, so the effect is soft. The hex-dense
            mean of 0.582 looked at first like an unpredicted dip under denser
            hex operands, but it rests on one seed (0.39 against 0.67 and 0.69);
            the other two match the center condition, so there is little evidence of
            a real hex-density effect on named accuracy. Three seeds is thin
            for a difference this size.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(arrays, stats):
    # Effective precision per condition and eval set: the single parameter of the
    # noise-then-snap reference in `sca.baselines` that reproduces the observed
    # exact-match rate. Shared by the density discussion below and by the
    # exploratory figure that checks the account against the margin.
    _rng = np.random.default_rng(0)

    def _behavior(_arm: str, _corpus: str, _set: str) -> dict:
        _pal = np.array(list(stats[_corpus]["palette"].values()), dtype=float) / 255

        def _stack(_key: str) -> np.ndarray:
            return np.concatenate([arrays[f"{_arm}-s{_s}/evals/{_set}/{_key}"] for _s in SEEDS]).astype(float) / 255

        _mix, _true, _guess = _stack("mix_value"), _stack("true_value"), _stack("guess_value")
        _true_idx = np.linalg.norm(_pal[None] - _true[:, None], axis=2).argmin(1)
        _hit = np.all(np.abs(_guess - _true) < 1e-9, axis=1)
        return {
            "pal": _pal,
            "mix": _mix,
            "true_idx": _true_idx,
            "hit": _hit,
            "margin": bl.snap_margin(np.linalg.norm(_pal[None] - _mix[:, None], axis=2)),
        }

    def fit_sigma(arm: str, corpus: str, set_name: str) -> dict:
        """One condition's behavior on one eval set, with the effective precision that reproduces it."""
        _b = _behavior(arm, corpus, set_name)
        _b["sigma"] = bl.fit_precision(_b["pal"], _b["mix"], _b["true_idx"], _b["hit"].mean(), _rng, 24, 11)
        return _b

    precision = {}
    for _arm, _corpus in (("center", "n140-h216"), ("palette-250", "n250-h216")):
        for _set in ("named_seen", "named_holdout"):
            _b = fit_sigma(_arm, _corpus, _set)
            _b["pred"] = bl.precision_limited_acc(_b["pal"], _b["mix"], _b["true_idx"], _b["sigma"], _rng, 120)
            precision[_arm, _set] = _b
    return fit_sigma, precision


@app.cell(hide_code=True)
def _(precision, stats):
    # The counterfactual the density comparison needs: the 250-name prompts answered at
    # the *center* condition's precision, so the palette's difficulty is held to one side.
    _c, _p = precision["center", "named_holdout"], precision["palette-250", "named_holdout"]
    _standardized = float(
        bl.precision_limited_acc(
            _p["pal"], _p["mix"], _p["true_idx"], _c["sigma"], np.random.default_rng(1), 160
        ).mean()
    )
    _per_pair = {
        _k: stats[_k]["n_lines"]["named"] / stats[_k]["n_seen_distinct"]["named"] for _k in ("n140-h216", "n250-h216")
    }

    mo.md(f"""
    More names and fewer correct answers is a strange pairing. Every condition in
    this table is d64-L4, and the 250-name corpus has the same 100,000 lines, so
    neither capacity nor data volume changed. Two other things may explain it.

    First, answering correctly became harder. The median gap between the nearest
    and second-nearest name to the exact mix falls from
    {np.median(_c["margin"]):.3f} to {np.median(_p["margin"]):.3f} of the unit
    cube, so more prompts ask the model to place the mix inside a tighter
    boundary. A reference guesser with the *center* condition's precision,
    answering the 250-name prompts, predicts {_standardized:.3f}. The observed
    score is {_p["hit"].mean():.3f}, so roughly half of the drop from
    {_c["hit"].mean():.3f} can be attributed to the more harder precision.

    Second, each training pair got less supervision. The same ≈ 50,000 named
    lines now spread over
    {stats["n250-h216"]["n_seen_distinct"]["named"]:,} distinct training pairs
    instead of {stats["n140-h216"]["n_seen_distinct"]["named"]:,}, which is
    {_per_pair["n250-h216"]:.1f} lines per pair against
    {_per_pair["n140-h216"]:.1f}. One sign of this is that the gap between seen
    and held-out accuracy nearly closes at 250 names
    ({precision["palette-250", "named_seen"]["hit"].mean():.3f} against
    {_p["hit"].mean():.3f}), where the center condition runs
    {precision["center", "named_seen"]["hit"].mean():.3f} against
    {_c["hit"].mean():.3f}. Fitted precision agrees: on *seen* pairs it is
    {precision["palette-250", "named_seen"]["sigma"]:.3f} at 250 names, the
    same as what the center condition manages on pairs it has never seen
    ({_c["sigma"]:.3f}). At 140 names a training pair recurs often enough to be
    remembered; at 250 it barely does.

    Neither reading needs a capacity limit, and the residual gap after
    standardizing ({_p["hit"].mean():.3f} against {_standardized:.3f}) is small
    enough that three seeds cannot separate a genuine precision cost from the
    seed spread.

    We'll look at capacity more closely in the width sweep (H4), below.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Within-form geometry (H2)

    Outcome: the named form matches the prediction, with the mix decodable at
    $R^2 = 0.94$ at the pre-answer position in the last layer, and hex never
    assembles a pre-answer mix at all.

    The figure below plots probe $R^2$ at every depth × landmark, for the two
    operands and the mix, per form; center condition of the sweep only. We used the
    stricter of the two probe holdout protocols described under
    [Measurements](#measurements).

    To be concrete about what one point on a line is: each combination of
    {seed, form, target, depth, landmark} is a "site", and each site gets its
    own independently fitted ridge probe. The target is the probed quantity
    (operand 1, operand 2, or the mix). The probe's inputs are the N equations'
    residual-stream activations at that one token position and depth (an N × C
    matrix), and its regression targets are the probed quantity's RGB triples
    (N × 3). A probe never sees activations from any other depth or position,
    in fitting or in scoring. The height of a line is that site's held-out
    $R^2$ under the strict holdout protocol. The colored step-lines are the
    per-channel scores averaged over the three seeds, and the hairlines show
    the seed spread.

    Because the sites are independent, $R^2$ says nothing about how probe
    directions relate across sites. A high score means a value-aligned readout
    direction exists at that site, but the score is the same whichever
    direction it is, so two sites can score alike with orthogonal readouts.
    The staggered per-channel bumps in the hex panels, similar as they look,
    are three separate fits. At the embedding they may well be one readout of
    the shared digit-embedding axis seen at three positions, but checking that
    would take a direction comparison (as in the principal-angle measure),
    which this figure does not make.

    A few reading notes. The lines are stepped because each landmark is a
    discrete character position; a straight line between two would claim
    measurements that were never made. Risers drawn in lighter ink skip over a
    name's variable-length middle (0 to 22 characters, depending on the name),
    so a named trace that steps up across one of those risers is the name
    completing; hex lines skip nothing, since all four of a hex operand's
    characters are landmarks. Every minor panel shares the same 0 and 1
    gridlines. Scores below zero are drawn at the floor (clamped to zero).
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    @themed(
        name="probe-channels",
        alt_text="""
            Probe R² traced across the equation for each RGB channel, stacked
            by depth, with the named form on top and hex below. The named
            panels are flat over the names themselves and rise at the spaces
            that follow them, peaking at the pre-answer position; the hex
            panels show each channel legible only at its own digit. Only named
            brings all three channels high together at the mix.
        """,
        caption="""
            Per-channel probe $R^2$ under the strict holdout, center condition.
            Named form on top, hex below; columns are operand 1, operand 2, and
            the mix. Within a block, depth runs upward from the embedding and
            the x axis across the grammar landmarks, one step-line per RGB
            channel. The grey area is the three-channel mean, and the two
            hairlines are the highest and lowest of the three seeds.
        """,
    )
    def _plot() -> plt.Figure:
        # (seed, depth+1, landmark, 3) per form and target — unaggregated, so the figure can
        # draw both the seed mean and the seed spread. The strict per-channel map: equations
        # containing a held-out value are removed from the probe's fit, so a plateau here is
        # geometry rather than a recovered identity. Its per-equation twin is in the
        # exploratory section.
        _stacks = {
            (_f, _t): np.stack([arrays[f"center-s{_s}/probes/{_f}/{_t}/r2_strict_ch"] for _s in SEEDS])
            for _f in vp.FORMS
            for _t in vp.TARGETS
        }
        return vp.probe_trace_grid(_stacks, form_axis="row", width=10.5, panel_height=0.4, ylabel="probe R² per depth")

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **The named form matches the prediction.** The mix is decodable at $R^2 =
    0.94$ at the pre-answer position in the last layer, and all three seeds land
    there (0.93, 0.95, 0.95) — for mix *values* the probe never saw, which is a
    stronger claim than the preregistered estimator makes. That estimator gives
    0.944 at the same site, so H2 is carried either way and nothing turns on the
    change of protocol. Where the mix *first* appears is less settled: one seed
    already reads it at the `=` sign by the second-to-last layer while the other
    two stay near 0.3 there, visible as the upper hairline lifting away from the
    outline at `=`.

    **A named operand's value is not on its name.** At the last character of
    operand 1 the strict probe scores $-0.05$, and operand 2 scores $0.00$:
    nothing. The value sits instead on the spaces that follow. It reads $0.23$
    on the space closing operand 1, rises to $0.47$ across the `+`, and reaches
    $0.64$ by the pre-answer position. A variable-length name has no fixed
    slot, and the model appears to resolve it into the first fixed position
    available. The value could not have appeared any earlier: the model
    reads left to right, and a color name's first characters do not determine
    its value: `sea green` and `salmon` share a prefix and nothing else, and
    predicting a held-out name's RGB from the other names sharing its
    two-character prefix scores $-0.21$, below the palette mean.

    Part of what the closing space holds is lexical rather than chromatic. 99
    of the 140 names are multi-word and the modifier words recur, and holding
    out whole word families costs that site $0.43$ more than a color-matched
    holdout of the same size does.[^wordfam] The site after the `+` is much
    less lexical, so the $0.47$ there is the sturdier number. This is also
    where the preregistered estimator diverges most: it reads $0.75$ at the
    name's last character. That reading is name recognition — the identity is
    recoverable there, but the value is not placed geometrically.

    **Hex relays its channels and does not retain them.** Each channel is
    legible at its own digit and near zero at the next, a strict relay rather
    than an accumulation. At the embedding the probe reads $0.42$ at a digit's
    own position, so the 16 digit embeddings do carry a real magnitude axis.
    Depth then improves it, rising to $0.78$ at operand 2's last digit by the
    last layer.

    Hex reads nothing at the delimiters ($-0.23$), which is what a form whose
    three digits sit at fixed offsets should do, because it needs no summary
    slot.

    /// details | Why the preregistered estimator reads 1.00 at the embedding
    That reading is a ceiling the setup hands out rather than a measurement:
    a hex digit position holds one of 16 tokens, the held-out unit is an
    equation, and 16 points in 64 dimensions admit an exact linear fit
    whatever the values. The strict $0.42$ is the informative number, since
    under that holdout the digit's value never enters the fit.

    Read at the *green* digit, the red channel scores $0.82$ under the
    preregistered estimator and $-0.27$ under the strict one: red is still
    recoverable there, because attention reaches back to its digit, but it is no
    longer placed by value.
    ///

    **Hex never assembles the mix, and the named pre-answer site carries
    everything at once.** Hex's mix reaches only $0.20$ at the pre-answer
    position and $0.55$ mid-answer, and the mid-answer reading sits where the
    answer's own digits are in the input (see the surface-text caveat under
    Exploratory analyses). Nothing in the hex panels looks like a whole
    pre-answer mix, so the coupling signal that H2 reserved judgment on did
    not appear. This is likely to matter for anchoring. At the named
    pre-answer position the mix reads $0.94$, operand 1 $0.64$, and operand 2
    $0.62$, so one site carries the whole equation. Hex has no such site, and
    a general RGB anchor needs one where the whole concept lives, even if only
    one color is being anchored.

    [^wordfam]: A separate refit on the published checkpoints, since it needs
        folds the stored maps don't carry; the numbers are quoted here rather
        than recomputed in this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    # Sites picked for the role they play in the claims above, not for the size of their
    # disagreement — ranking by the quantity on show would guarantee a dramatic table.
    _sites = [
        ("named", "mix", "pre", None, "named mix, pre-answer (decides H2)"),
        ("named", "mix", "eq", None, "named mix, at the ="),
        ("named", "op1", "o1e0", None, "named operand 1, its last character"),
        ("named", "op1", "plussp", None, "named operand 1, the space after +"),
        ("hex", "op2", "o2s1", 0, "hex operand 2, red at its own digit (embedding)"),
        ("hex", "op2", "o2e1", 0, "hex operand 2, red at the green digit"),
        ("hex", "mix", "pre", None, "hex mix, pre-answer"),
    ]
    _lm = {_n: _i for _i, _n in enumerate(LANDMARKS)}

    def _val(form: str, target: str, key: str, landmark: str, ch: int | None, depth: int) -> float:
        _m = np.mean([arrays[f"center-s{_s}/probes/{form}/{target}/{key}"] for _s in SEEDS], axis=0)
        _cell = _m[depth, _lm[landmark]]
        return float(_cell[ch] if ch is not None else np.mean(_cell))

    _thead = (
        "<tr><th>site</th>"
        + "".join(f'<th class="num">{_h}</th>' for _h in ("per-equation", "strict", "difference"))
        + "</tr>"
    )
    _rows = ""
    for _f, _t, _lmk, _ch, _desc in _sites:
        _d = 0 if "embedding" in _desc else 4
        _e = _val(_f, _t, "r2_ch", _lmk, _ch, _d)
        _s = _val(_f, _t, "r2_strict_ch", _lmk, _ch, _d)
        _rows += (
            f"<tr><td>{_desc}</td>"
            + f'<td class="num">{_e:+.3f}</td><td class="num">{_s:+.3f}</td>'
            + f'<td class="num">{_s - _e:+.3f}</td></tr>'
        )
    mo.vstack(
        [
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table">' + _thead + _rows + "</table></div>"
            ),
            mo.md("""
            Sites where the two holdouts disagree, at the last layer except for
            the embedding row. The difference column is how much of a reading
            was identity recovery: near zero means the site holds a value the
            probe can place without having seen it, and a large negative gap
            means the probe was looking the value up. Either estimator decides
            H2's site the same way, so adopting the stricter one changes no
            conclusion.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cross-form geometry separation (H3)

    H3 predicted $\rho < 0.2$ and large principal angles. Both hold conclusively: $\rho$ is **zero** at every cell where
    the guard defines it, in both directions and for all three targets.
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    # Every cross-form cell in the centre condition's scan: 2 directions × 3 targets × 3 seeds ×
    # depth × landmark. The extremes bound the whole claim, so quote them rather than a site.
    _all_cross = np.concatenate(
        [
            arrays[f"center-s{_s}/cross/{_d}/{_t}/r2"].ravel()
            for _s in SEEDS
            for _d in ("hex2name", "name2hex")
            for _t in vp.TARGETS
        ]
    )
    _pre = LANDMARKS.index("pre")
    _named_pre = float(np.mean([arrays[f"center-s{_s}/probes/named/mix/r2_strict"][4, _pre] for _s in SEEDS]))
    _named_pre_eq = float(np.mean([arrays[f"center-s{_s}/probes/named/mix/r2"][4, _pre] for _s in SEEDS]))
    _carried_pre = float(np.mean([arrays[f"center-s{_s}/cross/hex2name/mix/r2"][4, _pre] for _s in SEEDS]))

    mo.md(f"""
    Zero is a stronger statement than it looks. $\\rho$ clips negative cross-form
    $R^2$ to zero, so in principle a floor reading could hide a score just below
    break-even. Nothing here is close: all {len(_all_cross):,} cross-form cells
    in the scan are negative, and the best of them is ${_all_cross.max():.3f}$.
    Take the pre-answer position in the last layer as an example. The named
    form's own mix probe reads {_named_pre:.2f} there; the hex probe applied to
    the same activations reads ${_carried_pre:.0f}$. That is far worse than
    answering with the training mean, which suggests the decoder is pointing in
    an unrelated direction.

    The figure below shows both readings on the same axes, per form and target.
    The grey area is what the form's own probe recovers; the amber line is what
    the other form's probe recovers from the same activations. The amber
    fraction of the grey is $\\rho$, and it is zero everywhere. Note that the
    hex row's grey is low because hex's own strict readings are low (H2 found
    it relays its channels rather than holding them), so the transfer claim
    rests on the named row and on the magnitudes quoted above.

    /// details | Which within-form estimate the denominator uses
    The figure and the table use the **strict** holdout for the within-form
    reading, so the grey areas are the same quantity as in H2's per-channel
    figure. The preregistered $\\rho$ used the per-equation estimator instead.
    We switched because the two halves of the ratio should be measured alike: a
    cross-form probe has never seen the target form's tokens, so it cannot
    recover a value by identity lookup, while the per-equation within-form
    estimate can. Mixing the two biases $\\rho$ downward, which would flatter
    H3; the strict denominator removes that.

    The result is the same either way, since the numerator is what fails and it
    is unchanged: the preregistered ratios are zero too. The estimators do
    differ in which cells clear $\\rho$'s guard, and both are reported below.
    At the pre-answer site the named form reads {_named_pre:.2f} strict against
    {_named_pre_eq:.2f} per-equation, so H3's headline site is one of the
    places they agree.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    # Keyed by the form the probes are *applied* to, which is also the form whose landmarks
    # the x axis belongs to and whose within-form score is ρ's denominator.
    _applied = {"named": "hex2name", "hex": "name2hex"}

    @themed(
        name="cross-form-transfer",
        alt_text="""
            Six blocks of stacked step-line panels, two rows by three columns:
            named activations on the top row and hex on the bottom, with columns
            for operand 1, operand 2 and the mix. Each panel is one residual
            depth, embedding at the bottom, plotted across the grammar landmarks.
            A grey filled area shows what the form's own probe reads — rising
            over the operands and, in the named mix panel, high at the pre-answer
            position of the last layer. An amber line for the other form's probe
            applied to the same activations lies flat on the floor in every panel,
            at every depth and every landmark.
        """,
        caption=r"""
            Zero-shot cross-form transfer at the center condition, seed-averaged.
            Rows are the form the probes are applied to; columns are the probe
            target; within a panel, depth runs upward from the embedding and
            the x axis across the grammar landmarks. Grey is the form's own
            probe under the strict holdout (the same quantity as the
            per-channel figure under H2), with its three-seed envelope; amber
            is the other form's probe applied unchanged. Both are floored at
            zero for drawing.
        """,
    )
    def _plot() -> plt.Figure:
        _within = {
            (_f, _t): np.stack([arrays[f"center-s{_s}/probes/{_f}/{_t}/r2_strict"] for _s in SEEDS])
            for _f in vp.FORMS
            for _t in vp.TARGETS
        }
        _cross = {
            (_f, _t): np.stack([arrays[f"center-s{_s}/cross/{_applied[_f]}/{_t}/r2"] for _s in SEEDS])
            for _f in vp.FORMS
            for _t in vp.TARGETS
        }
        return vp.transfer_trace_grid(_within, _cross, width=10.5, panel_height=0.4, ylabel="probe R² per depth")

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, cells):
    # Two sites, neither chosen by ρ. The preregistered one maximizes the *weaker* form's
    # within-form R², so the comparison happens where both probes work; the pre-answer site
    # is where the named form's own geometry is strongest and where the embedding baseline is
    # zero in both forms (see the surface-text caveat).
    _stored = cells["center-s0"]["alignment"]["primary_site"]
    _lm = {_n: _i for _i, _n in enumerate(LANDMARKS)}

    def _mean(_path: str, _depth: int, _landmark: str) -> float:
        return float(np.mean([arrays[f"center-s{_s}/{_path}"][_depth, _lm[_landmark]] for _s in SEEDS]))

    def _rho(_direction: str, _into: str, _depth: int, _landmark: str, _key: str = "r2_strict") -> str:
        # ρ per direction, on the seed-mean maps, with its own guard: the denominator is the
        # within-form R² of the form the probe is applied *to*, and below 0.5 it reports nothing.
        _cross = np.mean([arrays[f"center-s{_s}/cross/{_direction}/mix/r2"] for _s in SEEDS], axis=0)
        _within = np.mean([arrays[f"center-s{_s}/probes/{_into}/mix/{_key}"] for _s in SEEDS], axis=0)
        _v = gm.rho(_cross, _within)[_depth, _lm[_landmark]]
        return "—" if np.isnan(_v) else f"{_v:.2f}"

    def _row(_label: str, _depth: int, _landmark: str) -> str:
        _angles = np.mean([arrays[f"center-s{_s}/angles/mix"][_depth, _lm[_landmark]] for _s in SEEDS], axis=0)
        _vals = [
            f"{_mean('probes/named/mix/r2_strict', _depth, _landmark):.2f}",
            f"{_mean('probes/hex/mix/r2_strict', _depth, _landmark):.2f}",
            f"{_mean('cross/hex2name/mix/r2', _depth, _landmark):+.2f}",
            _rho("hex2name", "named", _depth, _landmark),
            f"{_mean('cross/name2hex/mix/r2', _depth, _landmark):+.2f}",
            _rho("name2hex", "hex", _depth, _landmark),
            " · ".join(f"{_a:.0f}°" for _a in _angles),
        ]
        return f"<tr><td>{_label}</td>" + "".join(f'<td class="num">{_v}</td>' for _v in _vals) + "</tr>"

    _cols = ("within named", "within hex", "cross → named", "ρ → named", "cross → hex", "ρ → hex", "angles")
    _thead = "<tr><th>site (mix)</th>" + "".join(f'<th class="num">{_h}</th>' for _h in _cols) + "</tr>"
    _rows = _row(
        f"depth {_stored['depth']}, {_stored['landmark']} (preregistered)", _stored["depth"], _stored["landmark"]
    ) + _row("depth 4, pre-answer (clean ground)", 4, "pre")
    _closest = float(np.mean([arrays[f"center-s{_s}/angles/mix"][4, _lm["pre"]] for _s in SEEDS], axis=0).min())

    mo.vstack(
        [
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table">' + _thead + _rows + "</table></div>"
            ),
            mo.md(f"""
            The mix probe at two sites, seed-averaged; within-form readings use
            the strict holdout. The preregistered per-equation ratios give the
            same reading:
            {_rho("hex2name", "named", _stored["depth"], _stored["landmark"], "r2")} and
            {_rho("name2hex", "hex", _stored["depth"], _stored["landmark"], "r2")} at
            the preregistered site,
            {_rho("hex2name", "named", 4, "pre", "r2")} and
            {_rho("name2hex", "hex", 4, "pre", "r2")} at the pre-answer site.

            Neither site was picked using $\\rho$. The preregistered site
            maximizes the weaker form's own $R^2$, so the two probes are
            compared where both have something to carry. The pre-answer site is
            where the named form's geometry is strongest and where the answer
            text cannot contaminate either form. The two sites disagree about
            where hex is legible: at the pre-answer site, hex's own reading
            falls under $\\rho$'s guard, so the ratio into hex reports nothing
            there rather than zero. They agree about everything that bears on
            H3. The pre-answer site is also where the two mix decoders come
            closest to sharing a direction, and even there the smallest of the
            three principal angles is {_closest:.0f}°.

            The guard behind $\\rho$ turns out not to matter either. It reports
            a ratio only where the within-form $R^2$ clears 0.5. Swapping the
            per-equation estimator for the strict one changes which cells
            qualify, from 22–62 of 270 (5 depths × 18 landmarks × 3 seeds) down
            to 0–37; hex's operand rows drop out entirely, since their strict
            readings never reach 0.5. The verdict is the same either way,
            because the numerator is negative in every cell of the map, gated
            or not. That settles the open question of whether the alignment
            measures should be gated on the stricter estimator: here it makes
            no difference.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(arrays, maps):
    # Two nulls for the principal angle, both needed because the angle between two
    # 3-dimensional subspaces shrinks as the ambient space does — the width sweep moves
    # the ambient dimension, so an unadjusted angle trend says nothing on its own.
    _rng = np.random.default_rng(0)

    def _first(_wa: np.ndarray, _wb: np.ndarray) -> float:
        return float(gm.principal_angles(_wa, _wb)[0])

    def random_angle_null(width: int, n: int = 2000) -> tuple[float, float]:
        """Mean and spread of the first angle between two random 3-planes in R^width."""
        _a = [_first(_rng.normal(size=(width, 3)), _rng.normal(size=(width, 3))) for _ in range(n)]
        return float(np.mean(_a)), float(np.std(_a))

    def angles_at(arm: str, target: str = "mix") -> dict[str, float]:
        """Cross-form first angle at the clean pre-answer site, with both nulls.

        The seed control replaces the random draw's arbitrary directions with real
        probes that decode the same quantity in unrelated bases: three seeds train
        in three different coordinate systems, so their named-form probes are as
        unaligned as two probes can be while still being probes.
        """
        _d, _w = ARMS[arm]["depth"], ARMS[arm]["width"]
        _pre = LANDMARKS.index("pre")
        _wt = {
            (_f, _s): arrays[f"{arm}-s{_s}/probes/{_f}/{target}/weights"][_d, _pre] for _f in vp.FORMS for _s in SEEDS
        }
        _mean, _sd = random_angle_null(_w)
        return {
            "cross_form": float(np.mean([_first(_wt["named", _s], _wt["hex", _s]) for _s in SEEDS])),
            "seed_control": float(
                np.mean([_first(_wt["named", _a], _wt["named", _b]) for _a, _b in ((0, 1), (0, 2), (1, 2))])
            ),  # fmt: skip
            "random": _mean,
            "random_sd": _sd,
        }

    def rho_summary(arm: str) -> dict:
        """ρ at the preregistered site and over the whole scan, for one condition.

        The preregistered site maximizes the *weaker* form's own $R^2$, so the two
        probes are compared where both have something to carry; it is chosen without
        reference to ρ, since a per-condition maximum of a noisy map rises with the noise
        and could manufacture a width trend on its own. The scan maximum is reported
        beside it as the upper bound on sharing anywhere in the condition.
        """
        _wn, _wh = maps(arm, "probes/named/mix/r2"), maps(arm, "probes/hex/mix/r2")
        _site = np.unravel_index(np.minimum(_wn, _wh).argmax(), _wn.shape)
        _at, _best, _guarded, _total = {}, -np.inf, 0, 0
        for _dir, _into in (("hex2name", "named"), ("name2hex", "hex")):
            for _t in vp.TARGETS:
                _r = gm.rho(maps(arm, f"cross/{_dir}/{_t}/r2"), maps(arm, f"probes/{_into}/{_t}/r2_strict"))
                _guarded += int((~np.isnan(_r)).sum())
                _total += _r.size
                if not np.isnan(_r).all():
                    _best = max(_best, float(np.nanmax(_r)))
                if _t == "mix":
                    _at[_into] = float(_r[_site])
        return {"site": _site, "at": _at, "best": _best, "guarded": _guarded, "total": _total}

    return angles_at, rho_summary


@app.cell(hide_code=True)
def _(rho_summary):
    _all = {_a: rho_summary(_a) for _a in ARMS}
    _best = max(_r["best"] for _r in _all.values())
    _guarded = sum(_r["guarded"] for _r in _all.values())
    _total = sum(_r["total"] for _r in _all.values())

    mo.md(f"""
    ## Alignment under compression (H4)

    H4 predicted that narrowing the residual stream would push the two forms
    together: $\\rho$ and subspace overlap rising over d64 → d32 → d16, with
    d16-L8 the most aligned condition, but that didn't happen. $\\rho$ is
    {_best:.2f} at every one of the {_guarded:,} cells where the guard defines it
    across the whole sweep ({_total:,} cells before gating), in both directions
    and for all three targets.

    What compression does instead is wear the named form down. The figure below
    tracks three quantities against width: how well each form answers held-out
    equations, how well the named mix can be read out of the residual stream, and
    how close the two forms' mix decoders come to sharing a direction.
    """)
    return


@app.cell(hide_code=True)
def _(angles_at, cells, maps):
    _widths = [16, 32, 64]
    _by_shape = {(ARMS[_a]["width"], ARMS[_a]["depth"]): _a for _a in ARMS if not ARMS[_a]["bridge"]}
    _l4 = {_w: _by_shape[_w, 4] for _w in _widths}
    _l8 = {_w: _by_shape[_w, 8] for _w in _widths if (_w, 8) in _by_shape}

    def _acc(_arm: str, _set: str) -> float:
        return float(np.mean([cells[f"{_arm}-s{_s}"]["sets"][_set]["accuracy"] for _s in SEEDS]))

    def _within(_arm: str) -> float:
        return float(maps(_arm, "probes/named/mix/r2_strict")[ARMS[_arm]["depth"], LANDMARKS.index("pre")])

    @themed(
        name="compression",
        alt_text="""
            Three panels against residual width, plotted at 16, 32 and 64. Left:
            held-out exact match. The hex curve runs 1.00, 1.00, 0.72 as the
            stream narrows; the named curve falls much further, 0.67, 0.43, 0.05.
            Middle: the named mix probe's R² at the pre-answer position follows
            the named curve down, 0.94, 0.80, 0.35. Right: the first principal
            angle between the two forms' mix decoders falls from about 75° to 60°
            to 51°, and a shaded band for random subspaces of the same size falls
            with it, covering every observed point. Open markers for the 8-layer
            conditions sit clearly above the 4-layer ones at width 16 and coincide
            with them at width 64.
        """,
        caption=r"""
            Effects of narrowing the residual stream, across the width arms.
            Filled $\bullet$ markers joined by lines are the 4-layer conditions;
            open $\scriptstyle\Box$ markers are the 8-layer conditions, dashed
            because there is no d32-L8 condition to run a line through. Right:
            the first principal angle between the named and hex mix decoders at
            the pre-answer position (amber), with two references: the angle
            between two random 3-planes at that width (grey band, mean ± 1
            s.d.) and between two seeds' named probes (red).
        """,
    )
    def _plot() -> plt.Figure:
        fig, _axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
        _named = light_dark("#1a5f8a", "#6ab0d4")
        _hex = light_dark("#6a4f8a", "#b79ad6")
        _amber, _ref = vp.transfer_color(), light_dark("#d1495b", "#ff6b7d")

        def _series(_ax: plt.Axes, _fn, _color, _label):
            _ax.plot(_widths, [_fn(_l4[_w]) for _w in _widths], "o-", color=_color, lw=1.4, ms=5, label=_label)
            _ax.plot(list(_l8), [_fn(_l8[_w]) for _w in _l8], "s:", mfc="none", color=_color, lw=0.8, ms=6)

        _axes[0].set_title("accuracy")
        _series(_axes[0], lambda _a: _acc(_a, "hex_holdout"), _hex, "hex")
        _series(_axes[0], lambda _a: _acc(_a, "named_holdout"), _named, "named")
        _axes[0].set(ylabel="held-out exact match", ylim=(0, 1.05))
        _axes[0].legend(fontsize=7, loc="center left", framealpha=0.6)

        _axes[1].set_title("probe")
        _series(_axes[1], _within, _named, "named")
        _axes[1].set(ylabel="named mix $R^2$, pre-answer", ylim=(0, 1.05))

        _axes[2].set_title("rotation")
        _ang = {_w: angles_at(_l4[_w]) for _w in _widths}
        _mean = np.array([_ang[_w]["random"] for _w in _widths])
        _sd = np.array([_ang[_w]["random_sd"] for _w in _widths])
        _axes[2].fill_between(_widths, _mean - _sd, _mean + _sd, color=vp.mean_color(), alpha=0.2, lw=0)
        _axes[2].plot(_widths, _mean, color=vp.mean_color(), lw=1, alpha=0.7, label="random 3-planes")
        _series(_axes[2], lambda _a: angles_at(_a)["seed_control"], _ref, "seed control")
        _series(_axes[2], lambda _a: angles_at(_a)["cross_form"], _amber, "named vs hex")
        _axes[2].set(ylabel="first principal angle", ylim=(0, 95))
        _axes[2].yaxis.set_major_formatter(lambda _v, _p: f"{_v:.0f}°")
        _axes[2].legend(fontsize=7, loc="lower right", framealpha=0.6)

        for _ax in _axes:
            # Width doubles per step, so space the ticks by octave; the minor decade ticks a
            # log axis brings with it label the wrong scale entirely and collide with 16/32/64.
            _ax.set_xscale("log", base=2)
            _ax.set(xlabel="residual width", xticks=_widths)
            _ax.xaxis.set_minor_locator(plt.NullLocator())
            _ax.xaxis.set_major_formatter(lambda _v, _p: f"{_v:.0f}")
            _ax.grid(alpha=0.1)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(angles_at, cells, maps, rho_summary):
    _order = [_a for _a in ARMS if not ARMS[_a]["bridge"] and ARMS[_a]["names"] == 140 and ARMS[_a]["hex"] == 216]
    _pre = LANDMARKS.index("pre")

    def _row(_arm: str) -> str:
        _r, _g = rho_summary(_arm), angles_at(_arm)
        _fmt = lambda _v: "—" if np.isnan(_v) else f"{_v:.2f}"  # noqa: E731
        _vals = [
            f"{ARMS[_arm]['width']}",
            f"{ARMS[_arm]['depth']}",
            f"{np.mean([cells[f'{_arm}-s{_s}']['sets']['named_holdout']['accuracy'] for _s in SEEDS]):.3f}",
            f"{maps(_arm, 'probes/named/mix/r2_strict')[ARMS[_arm]['depth'], _pre]:.2f}",
            _fmt(_r["at"]["named"]),
            _fmt(_r["at"]["hex"]),
            f"{_r['best']:.2f}",
            f"{_g['cross_form']:.0f}°",
            f"{_g['seed_control']:.0f}°",
            f"{_g['random']:.0f}°",
        ]
        return f"<tr><td>{_arm}</td>" + "".join(f'<td class="num">{_v}</td>' for _v in _vals) + "</tr>"

    _cols = (
        "width", "depth", "named held-out", "named mix R²",
        "ρ → named", "ρ → hex", "ρ best", "angle", "seed control", "random",
    )  # fmt: skip
    _thead = "<tr><th>condition</th>" + "".join(f'<th class="num">{_h}</th>' for _h in _cols) + "</tr>"

    mo.vstack(
        [
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table">'
                + _thead
                + "".join(_row(_a) for _a in _order)
                + "</table></div>"
            ),
            mo.md("""
            The width arms, seed-averaged. The two ρ columns are at the
            preregistered site (the depth × landmark where the weaker form's own
            $R^2$ is highest, chosen without reference to ρ), and read "—" where
            that form's within-form score falls under ρ's 0.5 guard. "ρ best" is
            the largest ρ anywhere in the condition's scan, over both directions and
            all three targets. The last three columns are the first principal
            angle between the two mix decoders at the pre-answer position, and
            the two references the figure plots.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(angles_at, fit_sigma, maps, rho_summary, seed_mean):
    _sig = {_a: fit_sigma(_a, "n140-h216", "named_holdout")["sigma"] for _a in ("center", "d32", "d16", "d16-L8")}
    _g = {_a: angles_at(_a) for _a in ("center", "d32", "d16")}

    mo.md(f"""
    The angle does fall as the stream narrows, from {_g["center"]["cross_form"]:.0f}°
    at d64 to {_g["d32"]["cross_form"]:.0f}° at d32 and {_g["d16"]["cross_form"]:.0f}°
    at d16. That *looks* like the trend H4 predicted, but it's not: two
    3-dimensional subspaces drawn at random come just as close once the space
    they sit in is small enough. The same three widths give
    {_g["center"]["random"]:.0f}°, {_g["d32"]["random"]:.0f}° and
    {_g["d16"]["random"]:.0f}° for a random pair. Every observed angle sits inside
    one standard deviation of its null, and the seed control (two seeds' *named*
    probes, decoding the same quantity in unrelated bases) tracks the same curve.
    So the cross-form angle at every width is what two unrelated decoders would
    give.

    Under compression the two forms don't merge; one of them gives way first. Hex
    keeps its held-out accuracy at {seed_mean("d32", "hex_holdout", "accuracy"):.2f}
    at d32 and loses little even at d16
    ({seed_mean("d16", "hex_holdout", "accuracy"):.2f}), while named falls from
    {seed_mean("center", "named_holdout", "accuracy"):.2f} to
    {seed_mean("d32", "named_holdout", "accuracy"):.2f} to
    {seed_mean("d16", "named_holdout", "accuracy"):.2f}. In terms of the
    resolution account from H1, the fitted precision $\\sigma$[^sigma] coarsens
    from {_sig["center"]:.3f} of the unit cube at d64 to {_sig["d32"]:.3f} at
    d32 and {_sig["d16"]:.3f} at d16: roughly an eightfold loss of color
    resolution over two halvings of width. Hex needs far less resolution,
    because a hex answer's runner-up is a whole grid step away, so a coarse
    read still lands on the right digit.

    So the named form clearly lacks capacity at these widths, yet even that
    pressure causes no sharing. Keeping the two representations apart
    apparently costs the model less than merging them would.

    /// details | Whether d16 is too broken to test
    d16 answers {seed_mean("d16", "named_seen", "accuracy"):.2f} of the named
    equations it trained on, and its named mix probe reads
    {maps("d16", "probes/named/mix/r2_strict")[4, LANDMARKS.index("pre")]:.2f}
    at the pre-answer position, so there is not much named geometry left there to
    align. The sweep anticipated this: d16 sits one step below the width range
    ngpt-scaling validated, and H4's trend was to rest on d64 → d32 if it trained
    poorly. It does train, with smooth loss curves, but it does not learn the
    named form. Taking d16 out changes nothing about the verdict, since ρ is
    {rho_summary("d32")["best"]:.2f} at d32 too, where the named form still
    answers {seed_mean("d32", "named_holdout", "accuracy"):.2f} and its mix
    reads {maps("d32", "probes/named/mix/r2_strict")[4, LANDMARKS.index("pre")]:.2f}.
    Adding depth back at d16 recovers a good deal
    ({seed_mean("d16-L8", "named_holdout", "accuracy"):.2f} held-out named, and
    $\\sigma$[^sigma] back to {_sig["d16-L8"]:.3f}); see H6.
    ///

    [^sigma]: $\\sigma$ is the per-channel noise a guesser would need,
        reading the true mix and snapping to the nearest name, to match the
        observed accuracy; see H1.
    """)
    return


@app.cell(hide_code=True)
def _(cross, rho_summary, stats):
    _n = cross["n_pairs"]
    _acc = {
        _s: float(np.mean([_c["sets"][_s]["accuracy"] for _c in cross["cells"]])) for _s in ("cross_seen", "cross_unseen")
    }  # fmt: skip
    _share = stats["n140-h216-bridge"]["n_lines"]["cross"] / sum(stats["n140-h216-bridge"]["n_lines"].values())

    mo.md(f"""
    ## Alignment with a bridge (H5)

    H5 predicted that a little shared supervision would place both forms in the
    same subspace: cross-form equations lifting $\\rho$ above 0.8 at d64. The
    bridge condition's $\\rho$ is {rho_summary("bridge")["best"]:.2f}, the same as
    every other condition's, so H5 is unsupported on its own terms.

    That would be an easy result to dismiss if the bridge had simply failed to
    learn anything, so let's check whether the supervision worked.
    Cross lines are {_share * 100:.0f}% of the bridge corpus:
    {_n["n_lines"]:,} of them, drawn over {_n["seen"]:,} distinct
    (name, hex) pairs out of {_n["seen"] + _n["unseen"]:,} possible, so a pair
    recurs about {_n["n_lines"] / _n["seen"]:.1f} times and two thirds of the
    cross space is never seen at all. The sweep doesn't grade them, because the
    form exists to supervise alignment rather than to be scored, so we decoded
    them afterwards from the published checkpoints.[^crosseval]

    The model does learn to answer bridge-form equations. Exact match is
    {_acc["cross_seen"]:.2f} on cross pairs that appeared in training and
    {_acc["cross_unseen"]:.2f} on pairs that did not, with no malformed
    completions; at 1.2 lines per pair there is not much for the model to
    memorize, and the two numbers say it didn't. Given the unseen prompt
    `pale lime + #fc7 = `, all three seeds return `#dd7`, which is the right
    answer. So a color name's value does reach the hex answer path, for names and
    hex codes the model has never seen together.

    [^crosseval]: `cross_eval.py`, beside this report. It reads the sweep's
        checkpoints and writes its results back to the store, so the numbers
        here are computed rather than transcribed. It sits outside the
        experiment's DAG because a tick of `experiment.py` would now re-train
        all 24 conditions: `mini`'s evidence scheme has changed since the sweep ran,
        and a re-trained sweep would not reproduce the numbers quoted elsewhere
        in this report.
    """)
    return


@app.cell(hide_code=True)
def _(angles_at, cross, maps, rho_summary, seed_mean):
    _pre = LANDMARKS.index("pre")
    _cross_acc = {
        _s: float(np.mean([_c["sets"][_s]["accuracy"] for _c in cross["cells"]])) for _s in ("cross_seen", "cross_unseen")
    }  # fmt: skip

    def _row(_arm: str) -> str:
        _r, _g = rho_summary(_arm), angles_at(_arm)
        _fmt = lambda _v: "—" if np.isnan(_v) else f"{_v:.2f}"  # noqa: E731
        _bridged = ARMS[_arm]["bridge"]
        _vals = [
            f"{seed_mean(_arm, 'named_holdout', 'accuracy'):.3f}",
            f"{seed_mean(_arm, 'hex_holdout', 'accuracy'):.3f}",
            f"{_cross_acc['cross_unseen']:.3f}" if _bridged else "—",
            f"{maps(_arm, 'probes/named/mix/r2_strict')[4, _pre]:.2f}",
            _fmt(_r["at"]["named"]),
            _fmt(_r["at"]["hex"]),
            f"{_r['best']:.2f}",
            f"{_g['cross_form']:.0f}°",
            f"{_g['seed_control']:.0f}°",
            f"{_g['random']:.0f}°",
        ]
        return f"<tr><td>{_arm}</td>" + "".join(f'<td class="num">{_v}</td>' for _v in _vals) + "</tr>"

    _cols = (
        "named held-out", "hex held-out", "cross unseen", "named mix R²",
        "ρ → named", "ρ → hex", "ρ best", "angle", "seed control", "random",
    )  # fmt: skip
    _thead = "<tr><th>condition</th>" + "".join(f'<th class="num">{_h}</th>' for _h in _cols) + "</tr>"

    mo.vstack(
        [
            mo.Html(
                '<div class="report-table-scroll"><table class="report-table">'
                + _thead
                + _row("center")
                + _row("bridge")
                + "</table></div>"
            ),
            mo.md("""
            The bridge condition beside the center condition, both d64-L4, seed-averaged.
            The columns after the accuracies are the same measures as the H4
            table. Adding cross-form equations leaves every alignment measure
            where it was, and costs the two single-form tasks nothing worth
            reading.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(arrays):
    _applied = {"named": "hex2name", "hex": "name2hex"}

    @themed(
        name="bridge-transfer",
        alt_text="""
            Six blocks of stacked step-line panels, two rows by three columns,
            in the same layout as the cross-form transfer figure under H3, but
            for the bridge condition. A grey filled area shows what each form's own
            probe reads, rising over the operands and peaking at the pre-answer
            position of the named mix panel. The amber line for the other form's
            probe lies flat on the floor in every panel, at every depth and every
            landmark, exactly as it does without a bridge.
        """,
        caption="""
            Zero-shot cross-form transfer at the bridge condition, seed-averaged,
            drawn exactly as the H3 figure so the two can be laid side by side.
            Rows are the form the probes are applied to; columns are the probe
            target; grey is the form's own probe under the strict holdout and
            amber is the other form's probe applied unchanged.
        """,
    )
    def _plot() -> plt.Figure:
        _within = {
            (_f, _t): np.stack([arrays[f"bridge-s{_s}/probes/{_f}/{_t}/r2_strict"] for _s in SEEDS])
            for _f in vp.FORMS
            for _t in vp.TARGETS
        }
        _cross = {
            (_f, _t): np.stack([arrays[f"bridge-s{_s}/cross/{_applied[_f]}/{_t}/r2"] for _s in SEEDS])
            for _f in vp.FORMS
            for _t in vp.TARGETS
        }
        return vp.transfer_trace_grid(_within, _cross, width=10.5, panel_height=0.4, ylabel="probe R² per depth")

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(angles_at, cross, stats):
    _unseen = float(np.mean([_c["sets"]["cross_unseen"]["accuracy"] for _c in cross["cells"]]))
    _guess = float(np.mean([_c["sets"]["cross_unseen"]["guess_dist"] for _c in cross["cells"]]))
    _floor = float(np.mean([_c["sets"]["cross_unseen"]["floor_dist"] for _c in cross["cells"]]))
    _g = angles_at("bridge")
    _lines = stats["n140-h216-bridge"]["n_lines"]
    _share = _lines["cross"] / sum(_lines.values())

    mo.md(f"""
    So the bridge condition converts between the two vocabularies and keeps their
    geometries apart. Behaviorally the forms are joined: an unseen name and an
    unseen hex code mix to the right answer {_unseen * 100:.0f}% of the time, and
    the answers it emits sit {_guess:.3f} from the exact mix on average, against a
    quantization floor of {_floor:.3f}. Geometrically nothing moved: $\\rho$ stays at zero
    across the whole scan, and the two mix decoders' first principal angle is
    {_g["cross_form"]:.0f}° against a random-subspace null of
    {_g["random"]:.0f}° and a seed control of {_g["seed_control"]:.0f}°.

    Our measurement is fairly narrow: fit a probe on single-form lines, then
    apply it unchanged to the other form's lines at the same layer and grammar
    landmark. Suppose the model routes by form, with one pathway for `melon +
    ultramarine` and another for `melon + #48a`, sharing whatever they need
    somewhere we didn't look. Such a model would answer cross equations well and
    still score zero on our metric. What we can say is that a bridge does not
    make the two single-form geometries interchangeable at matched sites — the
    property H5 named, and a property a unidirectional anchor might want.

    We don't know where the conversion happens. The obvious place to look is the
    cross lines themselves: fit probes on those activations and ask which form's
    geometry a name's value lands in once the answer must come out as hex. That
    is a new measurement rather than a different reading of this one, so it goes
    to the backlog rather than into this report. The same applies to a
    depth-crossed $\\rho$, which the exploratory section reaches the edge of.

    /// details | What a stronger bridge might have done
    Cross lines are {_share * 100:.0f}% of the corpus here, and the weight was set
    before we knew what the disjoint corpus would do. A larger share, or cross
    equations that answer in *both* forms rather than only in hex, would be a
    different experiment; this one shows that {_share * 100:.0f}% is enough to
    teach the conversion and not enough to merge the geometries. Whether those
    two thresholds are far apart is open. It is also possible that no amount of
    bridging merges them under this grammar, since the answer form is determined
    by the prompt form, and keeping two decoders is a perfectly good way to
    satisfy that.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(fit_sigma, seed_mean):
    _sig = {_a: fit_sigma(_a, "n140-h216", "named_holdout")["sigma"] for _a in ("center", "L8", "d16", "d16-L8")}

    mo.md(f"""
    ## Depth (H6)

    H6 had two clauses. Doubling the layers at fixed width should improve
    name+name accuracy, and the mix should crystallize before the last layer,
    giving the result concept more than one layer of existence. The
    preregistration also named a live counter-expectation for the second clause:
    nothing in the loss rewards computing early, so the mix might stay pressed
    against the answer instead. The first clause fails at d64 and holds at d16;
    the second splits, and which way you read it depends on whether you count
    layers or fractions of the network.

    At d64 the extra layers buy nothing. Held-out named accuracy goes from
    {seed_mean("center", "named_holdout", "accuracy"):.3f} at L4 to
    {seed_mean("L8", "named_holdout", "accuracy"):.3f} at L8, and accuracy on
    trained pairs from {seed_mean("center", "named_seen", "accuracy"):.3f} to
    {seed_mean("L8", "named_seen", "accuracy"):.3f}. Three seeds cannot resolve
    differences that small. Fitted precision agrees: $\\sigma$ comes out at
    {_sig["center"]:.3f} of the unit cube at L4 and {_sig["L8"]:.3f} at L8, the
    same to three decimals, so the model reads colors to the same resolution
    either way. That fits the H1
    account, where the named ceiling at d64 is the palette's spacing rather than
    anything the model ran out of. Depth cannot buy resolution the task isn't
    short of.

    At d16 it buys a great deal. Held-out named accuracy goes from
    {seed_mean("d16", "named_holdout", "accuracy"):.3f} to
    {seed_mean("d16-L8", "named_holdout", "accuracy"):.3f}, and $\\sigma$ from
    {_sig["d16"]:.3f} to {_sig["d16-L8"]:.3f}. The same eight layers that do
    nothing at width 64 recover about half the color resolution lost by narrowing
    to width 16. So depth substitutes for width here, up to the point where the
    task's own resolution ceiling takes over.

    The figure below follows the named mix probe up through the stream at two
    landmarks: the `=` sign, and the pre-answer space one character later where
    H2 found the mature mix.
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    _cellset = [
        ("center", "d64-L4", "#1a5f8a", "#6ab0d4", "-"),
        ("L8", "d64-L8", "#1a5f8a", "#6ab0d4", "--"),
        ("d16", "d16-L4", "#6a4f8a", "#b79ad6", "-"),
        ("d16-L8", "d16-L8", "#6a4f8a", "#b79ad6", "--"),
    ]

    @themed(
        name="depth-profiles",
        alt_text="""
            Two panels of named mix probe R² against relative depth, from the
            embedding at 0 to the last layer at 1. Left panel, at the equals
            sign; right panel, at the pre-answer space. Four curves in each. The
            two width-64 conditions rise late and steeply, ending near 0.6 (4 layers)
            and 0.75 (8 layers) at the equals sign and near 0.95 at the
            pre-answer space, where the 8-layer curve reaches its ceiling one
            layer before the end and holds it. The two width-16 conditions stay far
            lower throughout, the 8-layer one about twice as high as the 4-layer
            one and flattening from mid-depth. Shaded bands show the seed spread,
            which is widest in the layers just before each curve's rise.
        """,
        caption="""
            Where the named mix becomes readable, at the `=` sign (left) and the
            pre-answer space (right, the site H2 uses). The x axis is depth as a
            fraction of the condition's layers, so the 4-layer and 8-layer
            curves overlay; 0 is the token embedding and 1 the last layer.
            Solid lines are the 4-layer conditions and dashed the 8-layer ones,
            blue for width 64 and purple for width 16; bands span the three
            seeds.
        """,
    )
    def _plot() -> plt.Figure:
        fig, _axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
        for _ax, _lmk, _title in [(_axes[0], "eq", "at the ="), (_axes[1], "pre", "at the pre-answer space")]:
            _i = LANDMARKS.index(_lmk)
            for _arm, _label, _cl, _cd, _ls in _cellset:
                _v = np.stack([arrays[f"{_arm}-s{_s}/probes/named/mix/r2_strict"][:, _i] for _s in SEEDS])
                _x = np.arange(_v.shape[1]) / (_v.shape[1] - 1)
                _c = light_dark(_cl, _cd)
                _ax.fill_between(_x, _v.min(0), _v.max(0), color=_c, alpha=0.15, lw=0)
                _ax.plot(_x, _v.mean(0), _ls, color=_c, lw=1.4, label=_label)
            _ax.set(title=_title, xlabel="relative depth", ylim=(-0.1, 1.05))
            _ax.grid(alpha=0.3)
        _axes[0].set_ylabel("named mix $R^2$")
        _axes[0].legend(fontsize=7, loc="upper left", framealpha=0.6)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, maps):
    _pre, _eq = LANDMARKS.index("pre"), LANDMARKS.index("eq")

    def _profile(_arm: str, _seed: int) -> np.ndarray:
        return arrays[f"{_arm}-s{_seed}/probes/named/mix/r2_strict"][:, _pre]

    def _layers_at_ceiling(_arm: str, _thresh: float = 0.9) -> float:
        """Mean number of layers, per seed, where the pre-answer mix clears *thresh*."""
        return float(np.mean([(_profile(_arm, _s) >= _thresh).sum() for _s in SEEDS]))

    def _crossings(_arm: str, _thresh: float = 0.9) -> str:
        """The layer at which each seed's pre-answer mix first clears *thresh*."""
        return ", ".join(str(int(np.argmax(_profile(_arm, _s) >= _thresh))) for _s in SEEDS)

    _c, _l = maps("center", "probes/named/mix/r2_strict"), maps("L8", "probes/named/mix/r2_strict")

    mo.md(f"""
    The mix does crystallize before the last layer at L8, which is what the
    clause asked for. It reads {_l[6, _pre]:.2f} at layer 6, {_l[7, _pre]:.2f} at
    layer 7 and {_l[8, _pre]:.2f} at layer 8, so the final layer adds nothing and
    the concept exists for two layers rather than one. Counting layers over 0.9,
    L4 averages {_layers_at_ceiling("center"):.1f} of 4 across seeds and L8
    {_layers_at_ceiling("L8"):.1f} of 8.

    Read as a fraction of the network, though, the mix sits *later* at L8:
    about {_layers_at_ceiling("center") / 4:.0%} of the layers at L4 falls to
    {_layers_at_ceiling("L8") / 8:.0%} at L8. The extra layers went in front of
    the computation rather than after it, and the mature mix stayed where it
    was, up against the answer. That is the counter-expectation H6 flagged. It
    is also the reading we'd trust, since it is the one that would generalize
    to a deeper model.

    Depth does move the mix earlier in the sequence. At the `=` sign the named
    mix reads {_c[4, _eq]:.2f} at L4 and {_l[7, _eq]:.2f} at L8, so one character
    before the site H2 measures, the deeper model has already assembled most of
    the answer. A head start on the token axis is a plausible thing for four
    extra layers to buy, though we haven't tested that reading.

    Which layer the rise happens in varies by seed, at both depths. At L4 the
    pre-answer mix first clears 0.9 at layer {_crossings("center")} for the three
    seeds; at L8, layer {_crossings("L8")}. The bands in the figure are widest
    exactly there. What replicates is the level reached and where it stops.

    For anchoring, the count matters more than the position. Wherever an anchor
    goes, it needs a site holding the concept, and at these depths that is one
    layer at L4 and about two at L8. That is a narrow window either way, and it
    doesn't widen in proportion to the network.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Analyses conceived after seeing the data land here, marked as post hoc.

    ### Answer positions partly read the answer (post hoc)

    Under teacher forcing the answer tokens are part of the input. So a probe
    at an answer position may be reading the digits off the surface text
    rather than anything the model computed.

    The embedding row (depth 0) tells us how much, because it is the token
    lookup before any attention or MLP has run: whatever is decodable there
    was already in the input. In the hex form it is a lot. At the embedding, a
    mix probe scores $R^2 \approx 1.00$ per channel at each answer digit
    (each hex digit *is* one channel of the answer), and an operand probe
    scores ≈ 0.48 at the same positions. The answer digit is the round-half-up
    mean of the two operand digits, so knowing it pins about half the variance
    of each operand. Both operands echo at the same strength, consistent with
    mixing rather than one operand being retained.

    This suggests that an answer-position reading should be compared against its
    own depth-0 value, not against zero. For example, the hex mix's best channel
    mean of 0.66 (depth 3, middle answer digit) breaks down as 0.22, 0.96, 0.80
    per channel, against an embedding baseline of 0.00, 1.00, 0.00: the green
    channel is read from the input, and the blue is computed a token ahead.

    The landmarks that carry the headline claims are clean: at the equals sign
    and the pre-answer space, every probe scores −0.003 at the embedding in both
    forms. That is why the named mix result is quoted at the pre-answer space,
    where the last layer reaches 0.95, 0.94, 0.94. Compared at those clean
    positions, the two forms separate further than the mid-answer numbers
    suggest: hex's best uncontaminated mix reading is 0.42 at the equals sign
    and 0.38 pre-answer, against named's 0.94.

    Similarly, $\rho$ computed at answer positions would be inflated in both
    directions, since both forms' probes can read the emitted answer there.
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    # The depth-crossed subspace comparison: the named mix probe at each depth against the
    # hex mix probe at each depth. Computable from the stored weights; the matching transfer
    # R² is not, since it needs the other form's activations at the other depth.
    _lm = {_n: _i for _i, _n in enumerate(LANDMARKS)}
    _w = {_f: np.stack([arrays[f"center-s{_s}/probes/{_f}/mix/weights"] for _s in SEEDS]) for _f in ("named", "hex")}
    _strict = {
        _f: np.mean([arrays[f"center-s{_s}/probes/{_f}/mix/r2_strict"] for _s in SEEDS], axis=0)
        for _f in ("named", "hex")
    }
    _n_depth = _w["named"].shape[1]

    def _first_angle(_dn: int, _ln: int, _dh: int, _lh: int) -> float:
        return float(
            np.mean([gm.principal_angles(_w["named"][_s, _dn, _ln], _w["hex"][_s, _dh, _lh])[0] for _s in range(3)])
        )

    _pre = _lm["pre"]
    _crossed = np.array([[_first_angle(_dn, _pre, _dh, _pre) for _dh in range(_n_depth)] for _dn in range(_n_depth)])
    _off_embedding = _crossed[1:, 1:]  # depth 0 is the token embedding; see below
    # The widest search that still asks the question of two probes that both work: every
    # depth × landmark pair where each form's own strict reading clears a modest 0.3.
    _sites = {_f: np.argwhere(_m >= 0.3) for _f, _m in _strict.items()}
    _best = min(
        (_first_angle(_dn, _ln, _dh, _lh) for _dn, _ln in _sites["named"] for _dh, _lh in _sites["hex"]),
        default=float("nan"),
    )
    _hex_pre = _strict["hex"][:, _pre]

    mo.md(f"""
    ### The forms compute at different depths, and $\\rho$ compares in place (post hoc)

    $\\rho$ as preregistered compares the same cell in both forms: same depth,
    same landmark. That assumes the two forms put the mix in the same place, and
    H2 showed they do not. Named color mixes are assembled at the pre-answer
    position in the last layer. Hex's best mix reading is mid-answer, a layer
    earlier, and its pre-answer column never clears {_hex_pre.max():.2f} at any
    depth ({", ".join(f"{_v:.2f}" for _v in _hex_pre)} from the embedding up).
    So the same-cell comparison may score named's mature mix against a hex
    representation that hasn't formed at that position, and a depth-crossed
    $\\rho$ could in principle be higher.

    We can check half of this from the stored data. The transfer half cannot be
    answered: applying hex's depth-{_n_depth - 3} probe to named's
    depth-{_n_depth - 1} activations needs the activations, and only the fitted
    probes and their scores are stored. The subspace half can, from the probe
    weights alone. The first principal angle between the named mix probe at
    *any* depth and the hex mix probe at *any* depth, both at the pre-answer
    position, stays between {_off_embedding.min():.0f}° and
    {_off_embedding.max():.0f}° across all {_off_embedding.size} pairs. Widening
    the search to every depth × landmark pair where both forms' own strict
    readings clear 0.3 (a low bar, and one that admits the contaminated answer
    positions), the closest the two subspaces come is {_best:.0f}°.

    So the depth offset is not concealing a shared set of directions: there is
    no layer at which hex's mix decoder points where named's does. A
    depth-crossed transfer $R^2$ would still be a different test (zero-shot fit
    rather than subspace angle) and is cheap to add to the eval step; it is
    banked in todo-science rather than done here, because it is not the
    measurement H3 was written against.

    One artifact in the angle map: At the embedding, on the space closing
    operand 1, the two probes appear to share a direction almost exactly (0.4°).
    The space character is the same token in both sublanguages, so at depth 0
    the probe's input is constant and its fitted direction is undetermined. The
    same degeneracy will show up in H4 and H5, where a small angle would
    otherwise read as the alignment those hypotheses are looking for.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The named ceiling is a resolution limit (post hoc)

    Named exact match tops out around 0.84 on trained pairs and 0.67 on held-out
    ones, while hex sits at ≈ 1.0, as did ex-2.1.3's regular 216-color
    vocabulary. That does not mean the named geometry is weaker; the two
    vocabularies ask questions of different precision.

    What varies is the **snap margin**: how much closer the correct name is to
    the exact mix than the runner-up. On a regular sub-grid closed under mixing,
    the answer lands on a vocabulary point and the margin is a constant (one
    grid step, 0.2 of the unit cube at ex-2.1.3's `v216`). On an irregular
    palette it varies by two orders of magnitude, and its median here is about a
    sixth of that. So a raw accuracy pools prompts that need 0.1 of positional
    accuracy with prompts that need 0.005.

    To separate resolution from everything else, we fit a one-parameter
    reference guesser: read the exact mix with isotropic Gaussian error of
    per-channel scale $\sigma$, then answer with the nearest name
    (`baselines.precision_limited_acc`). $\sigma$ is fitted to the *overall*
    accuracy only, so the shape of accuracy versus margin is free to agree or
    not.
    """)
    return


@app.cell(hide_code=True)
def _(precision):
    _panels = [
        ("center", "named_seen", "140 names · trained pairs"),
        ("center", "named_holdout", "140 names · held out"),
        ("palette-250", "named_seen", "250 names · trained pairs"),
        ("palette-250", "named_holdout", "250 names · held out"),
    ]
    _edges = np.array([0, 0.01, 0.02, 0.03, 0.045, 0.065, 0.09, 0.2])

    @themed(
        name="snap-margin",
        alt_text="""
            Four panels of exact-match rate against snap margin, one per palette
            size and eval set. In every panel the observed rate climbs steadily
            from about 0.4 at the tightest margins to near 1 at the widest, and a
            line for the resolution-limited reference guesser runs through the
            observed points. A faint histogram along the bottom shows most
            prompts have small margins, and the 250-name panels are shifted
            further toward the tight end than the 140-name ones.
        """,
        caption="""
            Exact match against the snap margin (the distance from the exact mix
            to the runner-up name, minus the distance to the correct one, in
            unit-cube units). Points are the observed rate in each margin bin,
            pooled over three seeds, with binomial standard errors; the line is
            the resolution-limited reference at the $\\sigma$ fitted to that
            panel's overall accuracy, so its level is fitted but its slope is
            not. The shaded profile along the bottom is the margin distribution.
        """,
    )
    def _plot() -> plt.Figure:
        fig, _axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True)
        _obs_c = light_dark("#1a5f8a", "#6ab0d4")
        _ref_c = light_dark("#d1495b", "#ff6b7d")
        for _ax, (_arm, _set, _title) in zip(_axes, _panels, strict=True):
            _b = precision[_arm, _set]
            _mg, _hit, _pred = _b["margin"], _b["hit"], _b["pred"]
            _mid = np.sqrt(_edges[:-1] * np.maximum(_edges[1:], 1e-6))  # log-ish bin centre
            _rate, _err, _ref, _x = [], [], [], []
            for _lo, _hi, _m in zip(_edges[:-1], _edges[1:], _mid, strict=True):
                _sel = (_mg >= _lo) & (_mg < _hi)
                if _sel.sum() < 12:
                    continue
                _rate.append(_hit[_sel].mean())
                _err.append(np.sqrt(_rate[-1] * (1 - _rate[-1]) / _sel.sum()))
                _ref.append(_pred[_sel].mean())
                _x.append(_m)
            # Margin distribution behind the rates, scaled to the lower third of the panel:
            # it says which bins carry the prompts, and the two palettes differ there.
            _share = np.histogram(_mg, _edges)[0] / len(_mg)
            _ax.fill_between(
                _edges,
                0,
                np.append(_share, _share[-1]) / 1.5,
                step="post",
                color=_obs_c,
                alpha=light_dark(0.14, 0.25),
                lw=0,
            )
            _ax.plot(_x, _ref, color=_ref_c, lw=1.4, label="resolution-limited")
            _ax.errorbar(_x, _rate, yerr=_err, fmt="o", ms=4, color=_obs_c, lw=1, label="observed")
            _ax.set(xscale="log", xlim=(_edges[0] + 0.006, _edges[-1]), ylim=(0, 1.05))
            _ax.set_title(_title, fontsize=8)
            _ax.set_xlabel("snap margin")
            _ax.grid(alpha=0.3)
            _ax.text(
                0.95, 0.06, f"σ = {_b['sigma']:.3f}", transform=_ax.transAxes, ha="right", fontsize=8, color=_ref_c
            )
        _axes[0].set_ylabel("exact match")
        _axes[0].legend(fontsize=7, loc="upper left", framealpha=0.6)
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays, precision):
    # What the same effective precision would score on the earlier experiments' regular
    # grids, where every mix lands on a vocabulary point and the margin is the grid step.
    _sigma = precision["center", "named_holdout"]["sigma"]
    _rng = np.random.default_rng(2)
    _grids = {"v216 (ex-2.1.3/2.1.4)": (0, 3, 6, 9, 12, 15), "v4096 (ex-2.1.3)": tuple(range(16))}
    _preds = {}
    for _name, _levels in _grids.items():
        # Ex-2.1.3's named equations draw *closed* pairs only, so the answer is a
        # vocabulary point and the runner-up sits one grid step away.
        _pts = np.array([(_r, _g, _b) for _r in _levels for _g in _levels for _b in _levels])
        _idx = np.random.default_rng(3).integers(len(_pts), size=(8192, 2))
        _mix = (_pts[_idx[:, 0]] + _pts[_idx[:, 1]] + 1) // 2
        _mix = _mix[np.isin(_mix, np.array(_levels)).all(1)][:256] / 15
        _pal = _pts / 15
        _ti = np.linalg.norm(_pal[None] - _mix[:, None], axis=2).argmin(1)
        _preds[_name] = float(bl.precision_limited_acc(_pal, _mix, _ti, _sigma, _rng, 120).mean())

    # The probe's own residual at the site that carries the mix, for comparison: R² is
    # 1 − residual variance / target variance, so a residual scale falls out of it.
    _target_sd = float(
        np.sqrt(
            np.var(
                np.concatenate([arrays[f"center-s{_s}/evals/named_seen/mix_value"] for _s in SEEDS]).astype(float)
                / 255,
                axis=0,
            ).mean()
        )
    )
    _pre = float(
        np.mean([arrays[f"center-s{_s}/probes/named/mix/r2_strict"][4, LANDMARKS.index("pre")] for _s in SEEDS])
    )
    _probe_sd = _target_sd * np.sqrt(max(0.0, 1 - _pre))

    mo.md(f"""
    The reference fits. On held-out named prompts the fitted $\\sigma$ is
    {precision["center", "named_holdout"]["sigma"]:.3f} of the unit cube
    ({precision["center", "named_holdout"]["sigma"] * 255:.0f} of 255 per
    channel), and from that one number the reference tracks the observed rate
    across the whole margin range, including the 250-name condition, where the same
    shape appears shifted toward the tight end. Trained pairs fit less well:
    at 140 names the fitted $\\sigma$ halves to
    {precision["center", "named_seen"]["sigma"]:.3f}, but the model beats the
    reference at tight margins and falls short of it at wide ones. Precision
    alone does not describe errors on pairs the model has seen; something more
    like recall is mixed in.

    **The earlier rungs are consistent with the same precision.** On
    ex-2.1.3's regular grids, where a mix lands on a vocabulary point and the
    margin is the grid step,
    $\\sigma = {_sigma:.3f}$ predicts {_preds["v216 (ex-2.1.3/2.1.4)"]:.2f} at
    `v216`, which is about what ex-2.1.3 (≈ 1.0, single-token names) and
    ex-2.1.4 (0.91, spelled names) actually scored. So the drop from ≈ 1.0
    there to 0.67 here needs no change in how well colors are represented; the
    coarser vocabulary forgives more. The exception is `v4096`, where the same
    $\\sigma$ predicts {_preds["v4096 (ex-2.1.3)"]:.2f} against ex-2.1.3's
    observed 0.65: that model read colors roughly 1.5× more finely than this
    one, plausible for a single-vocabulary corpus with single-token answers.

    **The behavior is more precise than the probe.** At the pre-answer
    position in the last layer the strict mix probe reads
    $R^2 = {_pre:.3f}$. With a per-channel target standard deviation of
    {_target_sd:.3f}, that puts the probe's residual near {_probe_sd:.3f},
    roughly twice the {_sigma:.3f} the behavior implies. The two are different
    estimands (a linear decoder's error versus an implied end-to-end
    resolution), so only the ordering is worth reading: the linear probe
    recovers a coarser version of the value than the model itself uses. For
    anchor design this means a probe $R^2$ of 0.94 at a site is a floor on
    what is there, and an anchor's own supervision is a linear readout with
    the same limitation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    Verdict table for H1–H6 (supported / partial / unsupported), with a
    pointer to the figure/analysis section that decides each.
    ///

    /// admonition | TODO
    What the outcome suggests for anchoring across surface forms: if
    alignment requires a bridge or compression, anchored runs on a
    mixed-vocabulary corpus need their labels to touch both forms (or need
    the bridge); if alignment is free, one form's labels may suffice. Especially
    for these suggestions, the verdicts above should be read with wide error
    bars: we're working with one synthetic task, small models, and there are
    many variables we haven't tested. The outcomes inform the design of the
    anchored runs; they don't settle the general question.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
