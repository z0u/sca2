import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.5: disjoint vocabularies and more named colors",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import json
    import tempfile
    from typing import cast
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    # Marimo puts the notebook's directory on sys.path, so the experiment
    # definition is importable — refs and sweep constants can't drift.
    from experiment import ARMS, ARRAYS_REF, METRICS_REF, SEEDS
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import figure_html, light_dark, smooth_step, themed
    from sca import vis as sv
    from sca.data import mixed_vocab as mv
    from sca.data.mixed_vocab import GAP_RISERS, LANDMARKS, OPERATORS, SPAN_RISERS

    use_publisher(report_bundle(__file__))

    def load_results() -> tuple[dict, dict[str, np.ndarray]] | None:
        """Resolve metrics and stacked per-cell arrays from the store, or None if unpublished."""
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

    def seq_cmap() -> LinearSegmentedColormap:
        """Theme-adaptive sequential map for R² heatmaps (near-background → accent)."""
        return LinearSegmentedColormap.from_list("r2", light_dark(["#eef3f7", "#1a5f8a"], ["#20242a", "#6ab0d4"]))


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

    def seed_mean(arm: str, set_name: str, key: str) -> float:
        vals = [cells[f"{arm}-s{s}"]["sets"][set_name][key] for s in SEEDS]
        return float(np.mean([v for v in vals if v is not None]))

    return arrays, cells, seed_mean, stats


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
    the most uniform palette available at that size. The center cell uses
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

    Two forms appear in every cell of the sweep, and a third only in the
    _bridge_ arm:

    | Form  | Example                       | Sweep cells |
    |-------|-------------------------------|-------------|
    | named | `melon + ultramarine = <name>`| all         |
    | hex   | `#e26 + #48a = #958`          | all         |
    | cross | `melon + #48a = #<hex>`       | bridge arm  |

    (Examples are illustrative until the corpus code lands.) Named equations
    always answer with a name, and hex equations with a hex code, so the answer
    form is determined by the prompt form; nothing about the result value
    changes which vocabulary the answer uses.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// warning | Data (the language)
    I notice the hex operands are clustered. They do look properly random, but it doesn't look uniform (it may be globally, but not locally). Maybe this biases training in some way? Clearly the models learnt the hex equations just fine, but this may have some effect on probe transfer between sublanguages.
    ///
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
            standing on its black corner, with data-colored dots. Left: the 140
            farthest-point xkcd names, spread evenly through the whole solid.
            Right: the 216 randomly sampled hex operands, also covering the
            solid with no obvious clusters or holes.
        """,
        caption="""
            The two operand sets in the RGB cube (center corpus). Left: the 140
            named colors; right: the 216 hex operands. Both spread through the
            full solid — the names by farthest-point construction, the hex
            subset by uniform sampling of the 4,096-point grid.
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
            Corpus statistics, one row per corpus (cells that share
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

    One subtlety: token positions don't line up across lines — names vary in
    length, and hex lines are shorter than named ones. Positions are therefore
    indexed by grammar landmarks (the last character of each operand, the
    operator, the pre-answer position, and answer characters counted from the
    answer's start and end), and the cross-form measures compare only at
    landmarks the two forms share.

    Alignment between the two forms is scored two ways:

    - Transfer ratio $\rho$: zero-shot cross-form $R^2$ divided by within-form
      $R^2$, at the same layer and landmark. Zero-shot means the probe is
      fitted on one form's activations and applied unchanged to the other's.
      Two guards keep the ratio well-behaved: $\rho$ is reported only where
      the within-form $R^2 \ge 0.5$, since below that the site isn't measuring
      geometry and a small denominator makes the ratio erratic; and negative
      cross-form $R^2$ clips to zero, so $\rho \in [0, 1]$ with 0 meaning no
      transfer.
    - Principal angles between the row-spaces of the two forms' fitted probes:
      a graded measure of whether the two decoders use the same directions of
      the residual stream.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    - **H1.** The disjoint language trains at the center cell: hex+hex exact
      match on unseen pairs is comparable to ex-2.1.1's hex levels, and
      name+name held-out exact match clears the $k$-NN analogues of the
      neighborhood nulls.
    - **H2.** Each sublanguage develops latent linear color geometry, with
      form-specific layout: name-form probes decode operands and mix with mix
      $R^2 \approx 0.9$ at the pre-answer position in the last layer; hex-form
      answers assemble just-in-time, channel by channel, with pre-answer
      full-mix $R^2$ staying low. An elevated pre-answer hex-mix $R^2$ would
      instead be evidence of cross-form coupling (see H3).
    - **H3.** The two latent geometries live apart in sweep cells that have no
      bridging grammar and width d64: $\rho < 0.2$, and the two probes'
      row-spaces show large principal angles. ($0.2 < \rho < 0.8$ reads as
      partial sharing and falsifies the crisp version of both H3 and H5.)
    - **H4.** Narrowing the stream aligns the forms: $\rho$ and subspace
      overlap rise monotonically over d64 → d32 → d16, with d16-L8 the most
      aligned cell.
    - **H5.** Adding the cross form produces alignment: the bridge cell (d64)
      reaches $\rho > 0.8$. The star design tests this at one width only;
      bridge × width cells are candidates for a follow-up round if H4 shows
      width matters.
    - **H6.** At L8, name+name accuracy improves at fixed width and the mix
      crystallizes before the last layer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The sweep

    <!-- Run plan (ops, not report content): 8 cells × 3 seeds = 24 runs. The
    GPU container cap is 10; run with max 5 containers so each handles ~5
    cells and the per-container startup time amortizes. Block size must be
    ≥ ~96 (lines reach ~86 chars with the multi-word names); earlier
    experiments used 64. -->

    A star design around one center cell, three seeds per cell:

    | Cell        | Names | Hex ops | Bridge | Width | Depth |
    |-------------|-------|---------|--------|-------|-------|
    | center      | 140   | 216     | none   | 64    | 4     |
    | L8          | 140   | 216     | none   | 64    | 8     |
    | d32         | 140   | 216     | none   | 32    | 4     |
    | d16         | 140   | 216     | none   | 16    | 4     |
    | d16-L8      | 140   | 216     | none   | 16    | 8     |
    | hex-dense   | 140   | 2048    | none   | 64    | 4     |
    | palette-250 | 250   | 216     | none   | 64    | 4     |
    | bridge      | 140   | 216     | cross  | 64    | 4     |

    Each arm has a reading. L8, the width cells, and d16-L8 score H4 and H6;
    the bridge cell scores H5. The two density arms attach to H1: *hex-dense*
    checks that hex accuracy and geometry aren't artifacts of the 216-point
    operand subset (expected: little change — ex-2.1.1's hex arithmetic
    generalized from far sparser coverage), and *palette-250* extends the
    density axis of ex-2.1.3 to the irregular palette (expected: named
    held-out accuracy holds or improves, with misses staying neighbor-level).

    Attention is held at 8 heads × 8 dims in every cell; only the residual
    stream and the MLP scale with width. The ngpt-scaling sweep validated
    widths {32, 64, 128} under this scheme, so d16 sits one step below the
    tested range. If the d16 cells train poorly, that is an architecture
    effect to report, and the width trend in H4 rests on d64 → d32.
    """)
    return


@app.cell(hide_code=True)
def _(cells):
    _thead = "<tr><th>cell</th>" + "".join(f'<th class="num">{h}</th>' for h in ("width", "depth", "params")) + "</tr>"
    _rows = "".join(
        f"<tr><td>{_a}</td>"
        + f'<td class="num">{ARMS[_a]["width"]}</td>'
        + f'<td class="num">{ARMS[_a]["depth"]}</td>'
        + f'<td class="num">{cells[f"{_a}-s0"]["n_params"]:,}</td>'
        + "</tr>"
        for _a in ARMS
    )

    mo.md(f"""
    Parameter counts per cell (seeds share a count). The full sweep —
    4 corpora, 24 training cells, 24 evals.

    /// details | Runtime
    The sweep ran in about 2.5 hours of wall time on five L4 containers. It would have been faster, but there were cross-region I/O issues.
    ///

    <div class="report-table-scroll"><table class="report-table">{_thead}{_rows}</table></div>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    All 24 cells converge smoothly under the shared 100-epoch schedule —
    including the d16 cells, which sit one step below ngpt-scaling's
    validated width range. No width-gated instability appeared.
    """)
    return


@app.cell(hide_code=True)
def _(cells):
    @themed(
        name="loss-curves",
        alt_text="""
            Eight small panels of loss versus epoch, one per sweep cell, each
            with three validation curves (one per seed) and three fainter
            training curves. Every curve descends smoothly and flattens within
            the 100-epoch budget; no panel shows divergence or oscillation.
        """,
        caption="""
            Loss per epoch for every cell: validation solid, training faint,
            one line per seed. All cells share the character vocabulary, so
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

    The center cell answers held-out hex equations almost perfectly (0.996
    exact) and held-out named equations at 0.667, with zero malformed
    completions in any eval set. The named score sits far above the
    prompt-blind centroid (0.043). One null simplified itself: in this
    language the true answer *is* the candidate nearest the mix, so the
    $k$-NN neighborhood null is exactly $1/k$ — the strongest version is a
    coin flip between the two nearest candidates at 0.5, and the model
    clears that too, by a margin the 1,946 held-out named pairs can resolve.
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
            Center cell, mean over three seeds. Distances are Euclidean in the
            unit cube: *guess dist* from the emitted answer to the exact mix,
            *floor dist* from the true (snapped) answer to the exact mix — the
            quantization floor. The guesses sit near the floor even where
            exact match misses.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// warning | Exact-match accuracy (H1)
    Interesting that named seen accuracy is significantly less than 1. So, the model hasn't properly learnt the geometry? But in 2.1.3 with one token per color (and no hex), it learnt it very well with 215 colors.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    @themed(
        name="miss-ranks",
        alt_text="""
            Two bar panels of guess rank, pooled over three seeds of the center
            cell. Rank 0 means the emitted answer is the candidate nearest the
            true mix, i.e. correct. For named held-out prompts, about two
            thirds of the mass is at rank 0 and most of the rest at ranks 1 and
            2, with a small tail beyond 6. For hex held-out prompts virtually
            all mass is at rank 0.
        """,
        caption="""
            Where the guesses land, center cell, pooled over seeds: the rank of
            the emitted answer among the form's candidate vocabulary, ordered
            by distance to the true mix. Rank 0 is the correct answer; rank 1
            is its nearest competitor. Malformed completions would be excluded,
            but there are none.
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
        "<tr><th>cell</th>"
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
            The density arms. Hex accuracy is insensitive to the operand-subset
            size, as expected. Named held-out accuracy dips in *both* density
            arms — at 250 names the answer perplexity is 153 against 86, so
            some drop was expected there, but the dip under denser hex
            operands (0.582, same named corpus as the center) was not
            predicted. Both stay far above their nulls. The hex-density
            interaction is taken up in the discussion.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// warning | More data -> lower named accuracy
    Yeah this is odd. This was at d64, right? So unlikely to be a capacity constraint (64 _sounds_ like plenty?)
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Within-form geometry (H2)

    The maps below plot leave-one-out probe $R^2$ at every depth × landmark,
    for the two operands and the mix, per form (center cell, seed-averaged;
    depth 0 is the embedding layer). The named form matches the prediction:
    operand readout builds over the early layers, and the mix is decodable at
    $R^2 \approx 0.94$ at the pre-answer position in the last layer — all
    three seeds land there (0.93, 0.95, 0.95), so this is the robust home of
    the result concept. Where the mix *first* appears is less settled: one
    seed already reads it at $R^2 \approx 0.97$ a landmark earlier, at the
    `=` sign by the second-to-last layer, while the other two stay near 0.3
    there and only reach the mix at the pre-answer space. So the earliest
    decodable site varies by seed; the pre-answer, last-layer location does
    not. The hex form never assembles the full mix at one site: its best
    full-mix $R^2$ is 0.62, mid-answer, consistent with ex-2.1.2's
    just-in-time channel staircase. Nothing in the hex panels looks like a
    holistic pre-answer mix, so the coupling tell that H2 reserved judgment
    on did not appear.
    """)
    return


@app.cell(hide_code=True)
def _(arrays):
    @themed(
        name="probe-maps",
        alt_text="""
            Six heatmaps of probe R² over depth (vertical, embedding plus four
            layers) by grammar landmark (horizontal, operand and answer
            characters), arranged as two rows of three: named form on top, hex
            form below, with columns for operand 1, operand 2, and the mix. In
            the named row, operand panels saturate from depth 1 onward around
            their own landmarks, and the mix panel stays dark through the early
            layers, brightening at the pre-answer and answer positions in the
            last layer; the equals column is only partly filled because the
            seeds disagree on whether the mix is decodable that early. In the
            hex row, operand panels also read out strongly, but the mix panel
            stays pale everywhere, peaking mid-answer at about 0.6. Each panel
            has a small open square marking its single brightest cell.
        """,
        caption="""
            Leave-one-out probe R² at every depth × landmark, center cell,
            mean over seeds. Rows: named and hex forms; columns: operand 1,
            operand 2, and the mix. Landmarks run through operand 1, the plus,
            operand 2, the equals sign, the pre-answer space, and the answer's
            first and last characters. Every cell is fit and scored on its own,
            leaving out one probe line at a time, so no cell borrows from
            another; the open square marks each panel's peak, the strongest
            site for that concept.
        """,
    )
    def _plot() -> plt.Figure:
        _targets = ("op1", "op2", "mix")
        fig, _axes = plt.subplots(2, 3, figsize=(9.6, 4.8), sharex=True, sharey=True)
        _cmap = seq_cmap()
        _im = None
        for _i, _form in enumerate(("named", "hex")):
            for _j, _t in enumerate(_targets):
                _m = np.mean([arrays[f"center-s{_s}/probes/{_form}/{_t}/r2"] for _s in SEEDS], axis=0)
                _ax = _axes[_i, _j]
                _im = _ax.imshow(_m, vmin=0, vmax=1, cmap=_cmap, aspect="auto", origin="lower")
                _ax.set_title(f"{_form} · {_t}", fontsize=9)
                # Mark the peak cell of the seed-averaged map — the strongest probe
                # site for this concept. Each cell is its own leave-one-out estimate,
                # so this is a readout of the map, not a probe chosen over the others.
                _pd, _pl = np.unravel_index(np.nanargmax(_m), _m.shape)
                _ax.plot(
                    _pl,
                    _pd,
                    marker="s",
                    mfc="none",
                    mec=light_dark("#111", "#fff"),
                    mew=1.4,
                    ms=9,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=light_dark("#fff", "#111"))],
                )
        for _ax in _axes[1]:
            _ax.set_xticks(range(len(LANDMARKS)), LANDMARKS, rotation=90, fontsize=7)
        for _ax in _axes[:, 0]:
            _ax.set_ylabel("depth")
        assert _im is not None
        fig.colorbar(_im, ax=_axes, shrink=0.8, label="probe R²")
        return fig

    mo.Html(_plot())
    return


@app.cell(hide_code=True)
def _(arrays):
    _landmark_labels = {
        "o1s0": "$a_{1}$",
        "o1s1": "$a_{2}$",
        "o1e1": "$a_{n-1}$",
        "o1e0": "$a_{n}$",
        #
        "o2s0": "$b_{1}$",
        "o2s1": "$b_{2}$",
        "o2e1": "$b_{n-1}$",
        "o2e0": "$b_{n}$",
        #
        "as0": "$r_{1}$",
        "as1": "$r_{2}$",
        "ae1": "$r_{n-1}$",
        "ae0": "$r_{n}$",
        #
        "plus": "+",
        "eq": "=",
        "pre": " ",
    }
    _major_landmarks = [_l for _l in LANDMARKS if _l in OPERATORS]
    _x_major = [LANDMARKS.index(_l) for _l in _major_landmarks]
    _x_labels_major = [_landmark_labels.get(_l, _l) for _l in _major_landmarks]

    _minor_landmarks = [_l for _l in LANDMARKS if _l not in OPERATORS]
    _x_minor = [LANDMARKS.index(_l) for _l in _minor_landmarks]
    _x_labels_minor = [_landmark_labels.get(_l, _l) for _l in _minor_landmarks]

    def _plot(_t: str) -> plt.Figure:
        _forms = ("named", "hex")
        # (depth+1, landmark, 3) per form, seed-averaged. r2_ch is the per-channel
        # companion the heatmap collapses: its mean over the channel axis is a column above.
        _stacks = {
            _f: np.mean([arrays[f"center-s{_s}/probes/{_f}/{_t}/r2_ch"] for _s in SEEDS], axis=0) for _f in _forms
        }
        _n_depth = next(iter(_stacks.values())).shape[0]
        _x = range(len(LANDMARKS))
        # Guard: if two adjacent landmarks ever alias onto one character, the probe
        # measures them once and the columns come out bit-identical — break the line
        # there so a riser doesn't span zero real distance and inflate the plateau.
        # The current scheme has no such pair (the answer is sampled like an operand),
        # so this is empty; it stays as a safety net for a scheme that reintroduces one.
        _breaks = {
            _f: {_i for _i in range(len(LANDMARKS) - 1) if np.array_equal(_s[:, _i], _s[:, _i + 1])}
            for _f, _s in _stacks.items()
        }
        # Ramp per riser: wider plateaus (0.25) keep the discrete tokens legible; a riser
        # that crosses an unprobed stretch is drawn as a full smooth slide (1) instead, so it
        # doesn't read as a step between adjacent measurements. Word middles (SPAN_RISERS) are
        # named-only — hex's are adjacent digits. Operator spaces (GAP_RISERS) exist in the
        # fixed grammar of both forms.
        _ramps = {_f: np.full(len(LANDMARKS) - 1, 0.25) for _f in _forms}
        _ramps["named"][list(SPAN_RISERS)] = 1.0
        for _f in _forms:
            _ramps[_f][list(GAP_RISERS)] = 1.0

        # Color is data: each channel's line in its own hue; the dashed line is the mean
        # (the heatmap value). Widths taper R → G → B so agreeing channels stay visible.
        _cols = light_dark(["#d1495bbb", "#2a9d5c", "#3b6fd4"], ["#ff6b7daa", "#4fd07ac8", "#6ea3ff"])
        _lws = (2.6, 1.7, 1.0)
        _mean_col = light_dark("#555", "#aaa")

        fig, _axes = plt.subplots(
            _n_depth, len(_forms), figsize=(8, 0.6 * _n_depth), sharex=True, sharey=True, squeeze=False
        )
        cast(ConstrainedLayoutEngine, fig.get_layout_engine()).set(hspace=0, h_pad=0, wspace=0)
        for _j, _f in enumerate(_forms):
            _axes[0, _j].set_title(_f, fontsize=9)
            for _r in range(_n_depth):
                _d = _n_depth - 1 - _r  # embedding (depth 0) at the bottom, as in the heatmap
                _ax = cast(plt.Axes, _axes[_r, _j])
                _ax.vlines(_x_major, -1, 2, "#8881", lw=6)
                _m = np.clip(_stacks[_f][_d], 0, 1)  # (landmark, 3)
                # Mean under the channels: hidden behind them where they agree, between them where they don't.
                smooth_step(
                    _ax,
                    _x,
                    _m.mean(1),
                    color=_mean_col,
                    lw=0.5,
                    linestyle=":",
                    ramp=_ramps[_f],
                    zorder=1,
                    breaks=_breaks[_f],
                )
                for _c in range(3):
                    smooth_step(
                        _ax, _x, _m[:, _c], color=_cols[_c], lw=_lws[_c], ramp=_ramps[_f], zorder=2, breaks=_breaks[_f]
                    )
                _ax.set(ylim=(-0.2, 1.2), xlim=(-0.5, len(LANDMARKS) - 0.5))
                if _j == 0:
                    _ax.set_ylabel(f"{_d}", fontsize=8)
                _ax.tick_params(axis="x", top=True, direction="inout")
                _ax.tick_params(axis="y", left=True, right=True, direction="in")
                _ax.set_yticks([0, 1], "")
                _ax.spines[:].set_visible(False)
        for _ax in _axes[-1]:
            _ax = cast(plt.Axes, _ax)
            _ax.set_xticks(_x_major, _x_labels_major, minor=False, fontsize="x-small")
            _ax.set_xticks(_x_minor, _x_labels_minor, minor=True, fontsize="xx-small")
        fig.supylabel(f"probe R² ({_t} RGB) per depth", fontsize=9)
        return fig

    # Same panel per probe target, rendered as separate figures so each keeps its full
    # width; the CSS in report.css lays a `<figure>` of `<figure>`s out as a reflowing
    # subfigure row. Columns match the heatmap above: operand 1, operand 2, the mix.
    _target_captions = {
        "op1": (
            "**Operand 1.** Named reads it holistically — the three channels ride together "
            "over its characters; hex resolves it digit by digit, each channel stepping up "
            "at its own hex position."
        ),
        "op2": "**Operand 2.** The same picture, shifted one operand along to the $b$ characters.",
        "mix": (
            "**The mix.** Named brings all three channels high together from the pre-answer "
            "position onward in the last layer; hex never does, which is why its averaged map "
            "stayed pale."
        ),
    }
    _target_alt = {
        "op1": """
            A five-by-two grid of small step-line panels for the operand-1 probe. Rows are
            residual depth, embedding at the bottom rising to the last layer at the top;
            columns are the two surface forms, named on the left and hex on the right. Each
            panel plots leave-one-out probe R² for operand 1's RGB across the grammar
            landmarks, one line per channel plus a dashed grey mean. In the named column the
            three channels rise and fall together over operand 1's own characters, a single
            shared plateau there that fades elsewhere. In the hex column they instead resolve
            at different characters — red, green and blue each stepping up at its own hex
            digit — so they separate into an offset staircase over the operand, and read out
            channel by channel again at the answer positions.
        """,
        "op2": """
            The same five-by-two grid of step-line panels for the operand-2 probe: rows are
            depth, columns are the named and hex forms, one line per RGB channel plus a
            dashed mean. It repeats the operand-1 reading one operand group to the right —
            named holds the three channels together over operand 2's characters, hex splits
            them across its digits into the same offset staircase.
        """,
        "mix": """
            The same five-by-two grid of step-line panels for the mix probe: rows are depth,
            columns are the named and hex forms, one line per RGB channel plus a dashed mean.
            In the named column the three channels run together and climb with depth, rising
            together around the equals and pre-answer landmarks in the last row. In the hex
            column the three channels separate and none reaches the top of the panel, so their
            mean stays low across every landmark.
        """,
    }
    _subfigs = "".join(
        themed(_plot, name=f"probe-channels-{_t}", alt_text=_target_alt[_t], caption=_target_captions[_t])(_t)
        for _t in ("op1", "op2", "mix")
    )
    mo.Html(
        figure_html(
            _subfigs,
            aria_label="Per-channel leave-one-out probe R² by depth for operand 1, operand 2, and the mix.",
            caption=mo.md("""
                Per-channel leave-one-out probe R², center cell, seed-averaged — the RGB mean
                of a panel's three lines is the matching form-and-target cell of the heatmap
                above. The three subfigures are the heatmap's three columns (operand 1, operand
                2, the mix); within each, rows are depth (embedding at the bottom) and columns
                are the named and hex forms. Each panel runs across the grammar landmarks, one
                step-line per channel (R, G, B) plus their mean (dashed). Steps, because each
                landmark is a discrete character position; widths taper R → G → B so a landmark
                where the channels agree reads as nested bands rather than as whichever channel
                drew last.
            """).text,
        )
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cross-form geometry separation (H3)

    /// admonition | TODO
    Figure: transfer-ratio $\rho$ maps over layer × position, both
    directions (hex→name, name→hex), center cell. Expected: $\rho < 0.2$
    everywhere the within-form probes are strong.
    ///

    /// admonition | TODO
    Figure: principal angles between the two probes' row-spaces. The
    primary site is where within-form $R^2$ is strongest, chosen without
    reference to $\rho$ so the verdict isn't shaped by the quantity being
    judged; the maximum-$\rho$ site is reported beside it as an upper bound
    on sharing ("even at its most aligned site…"). Expected: angles near
    90° at the primary site, i.e. the decoders use different directions of
    the stream.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Alignment under compression (H4)

    /// admonition | TODO
    Figure: $\rho$ and principal angles versus width (d64, d32, d16, plus
    the d16-L8 cell), at each cell's strongest within-form site, with the
    maximum-$\rho$ site as a second series. The primary site is chosen
    independently of $\rho$: a per-cell maximum of a noisy map rises with
    the noise, which could manufacture a width trend on its own. Expected: a
    monotonic rise in sharing as the stream narrows, with d16-L8 the most
    aligned. A
    flat line at low $\rho$ would say capacity pressure alone doesn't merge
    the forms at these scales; alignment already present at d64 would say
    the merge is a bias of training, and H3 falls.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Alignment with a bridge (H5)

    /// admonition | TODO
    Figure/table: the bridge cell's $\rho$ beside the center cell's, at d64.
    Expected: cross-form equations lift $\rho$ above 0.8, i.e. a small amount
    of shared supervision places both forms in the same subspace.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Depth (H6)

    /// admonition | TODO
    Figure: name-form accuracy and the mix-crystallization map at L8
    versus L4. Expected: held-out named accuracy rises, and the layer ×
    position probe map shows the mix decodable before the final layer,
    giving the result concept more than one layer of existence. A live
    counter-expectation: nothing in the loss rewards computing early, so the
    mix may instead stay pressed against the answer and build gradually —
    that outcome would refute this clause of H6.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Nothing yet. Analyses conceived after seeing the data land here, marked as
    post hoc.
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
