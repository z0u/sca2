import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.5: disjoint vocabularies and more named colors",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo  # noqa: F401

    # Experiment imports (refs, sweep constants, palette helpers) land with the
    # experiment code. The skeleton is prose-only by design: see the note under
    # "How to read this draft".


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

    The answer sets up the anchoring experiments. The graded concept labels are
    computed from color values, so they attach to either form with equal ease;
    the question is whether they land on one shared geometry or two. If the
    forms share their geometry, a concept like *red* has a single home and one
    anchor covers both vocabularies; if they live apart, an anchor is a
    per-form object. And if this experiment finds the forms apart, a later one
    could test whether anchoring the same concept in both forms pulls the
    geometries together, and whether more shared anchors pull harder.

    Lineage: the base language (ex-2.1.1, 2.1.2) bridged names and hex with
    alias and cross lines, and its 27-name sub-grid turned out too sparse to
    grade (ex-2.1.4). The single-vocabulary experiments (ex-2.1.3, 2.1.4)
    removed hex entirely. This experiment keeps both forms, removes every
    bridge, and replaces the named sub-grid with 140 real color names spread
    through the full 8-bit cube.

    ## How to read this draft

    This report was drafted before the experiment ran. Blockquotes marked 🔮
    are placeholders: each states what its figure or table will show and the
    pattern we expect. As results land, placeholders are replaced with
    observations. The hypotheses section is frozen as written; any analysis
    invented after seeing data goes under "Exploratory analyses" and is marked
    as post hoc.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The language

    The named palette comes from the xkcd color survey: of its 949 names, 250
    are single words, and we order those by farthest-point selection so that
    the first $N$ form the most uniform palette available at that size. The
    center cell uses $N = 140$. At that size the minimum and median
    nearest-neighbor distances are ≈ 28 and ≈ 37 (8-bit Euclidean), against
    4 and 27 for the CSS keyword list, which is why we prefer the xkcd set.
    Values are full 8-bit; names do not lie on the 16-level hex grid.

    Hex operands are drawn from a fixed random subset of the 4096-point grid
    (216 in the center cell), sampled point-by-point rather than as a regular
    sub-grid. The subset constrains operands only: the correct completion of a
    hex equation may be any grid point.

    Mixing works in 8-bit RGB. A short-hex digit expands by repetition
    (`f` → `ff`, i.e. ×17), and the mix is the round-half-up mean per channel,
    the same rule as every previous experiment. A hex answer rounds each
    channel back to the nearest grid level. A named answer is the palette
    entry nearest the mix (Euclidean; ties broken toward the
    lexicographically first name).

    Two forms appear in every cell, and a third only in the bridge arm:

    | Form  | Example                       | Where           |
    |-------|-------------------------------|-----------------|
    | named | `melon + ultramarine = <name>`| all cells       |
    | hex   | `#e26 + #48a = #958`          | all cells       |
    | cross | `melon + #48a = #<hex>`       | bridge arm only |

    (Examples are illustrative until the corpus code lands.) There are no
    alias lines in any cell. Named equations always answer with a name, and
    hex equations with a hex code, so the answer's form is determined by the
    prompt's form; nothing about the mix's value changes which vocabulary the
    answer uses.

    > 🔮 Figure: the palette. Swatch grid of the 140 names in selection order,
    > with nearest-neighbor-distance histograms for the named palette and the
    > hex subset. Expected: names spread through the cube with no
    > near-duplicates; the hex subset roughly uniform by construction.

    > 🔮 Figure/table: corpus statistics. Line counts per form, the
    > answer-name frequency distribution (the design study measured perplexity
    > ≈ 83 over 139 names under uniform pair sampling), and sequence lengths
    > against the block size.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What we measure

    Exact-match accuracy is scored per form on seen and held-out operand
    pairs, next to nulls adapted for an irregular palette: the prompt-blind
    centroid guesser carries over unchanged, and the grid-based one-step-shell
    nulls generalize to $k$-nearest-neighbor sets around the true mix.

    Geometry is read with ridge probes (leave-one-out, as in
    `ridge_probe_loo`) fitted for operand and mix values at every layer and
    every token position, separately per form. The full scan replaces the
    hand-picked probe sites of earlier reports with a map, so the alignment
    measures below don't depend on us choosing the right position in advance.

    Alignment between the two forms is scored two ways:

    - Transfer ratio $\rho$ = zero-shot cross-form $R^2$ divided by
      within-form $R^2$, at the same layer and position. Zero-shot means the
      probe is fitted on one form's activations and applied unchanged to the
      other's; fitting anything on the target form would only re-measure
      within-form geometry.
    - Principal angles between the row-spaces of the two forms' fitted probes:
      a graded measure of whether the two decoders use the same directions of
      the residual stream.

    Zero-shot transfer is the substantive test because the name sublanguage
    constrains its geometry only up to a rigid motion: midpoint mixing and
    nearest-neighbor answers are both isometry-invariant, so the model could
    learn a perfect color cube for names that shares no directions with the
    hex one.
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
    - **H2.** Each sublanguage develops linear color geometry, with
      form-specific layout: name-form probes decode operands and mix with mix
      $R^2 \approx 0.9$ at the pre-answer position in the last layer; hex-form
      answers assemble just-in-time, channel by channel, with pre-answer
      full-mix $R^2$ staying low. An elevated pre-answer hex-mix $R^2$ would
      instead be evidence of cross-form coupling (see H3).
    - **H3.** With no bridging grammar at d64, the two geometries live apart:
      $\rho < 0.2$, and the two probes' row-spaces show large principal
      angles. ($0.2 < \rho < 0.8$ reads as partial sharing and falsifies the
      crisp version of both H3 and H5.)
    - **H4.** Narrowing the stream aligns the forms: $\rho$ and subspace
      overlap rise monotonically over d64 → d32 → d16, with d16-L8 the most
      aligned cell.
    - **H5.** Adding the cross form produces alignment at every width:
      $\rho > 0.8$.
    - **H6.** At L8, name+name accuracy improves at fixed width and the mix
      crystallizes before the last layer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The sweep

    A star design around one center cell, three seeds per cell:

    | Cell        | Names | Hex ops | Bridge | Width | Depth |
    |-------------|-------|---------|--------|-------|-------|
    | center      | 140   | 216     | none   | 64    | 4     |
    | depth       | 140   | 216     | none   | 64    | 8     |
    | width-32    | 140   | 216     | none   | 32    | 4     |
    | width-16    | 140   | 216     | none   | 16    | 4     |
    | deep-narrow | 140   | 216     | none   | 16    | 8     |
    | hex-dense   | 140   | 2048    | none   | 64    | 4     |
    | palette-250 | 250   | 216     | none   | 64    | 4     |
    | bridge      | 140   | 216     | cross  | 64    | 4     |

    Attention is held at 8 heads × 8 dims in every cell; only the residual
    stream and the MLP scale with width. The ngpt-scaling sweep validated
    widths {32, 64, 128} under this scheme, so d16 sits one step below the
    tested range. If the d16 cells train poorly, that is an architecture
    effect to report, and the width trend in H4 rests on d64 → d32.

    > 🔮 Table: parameter counts and training cost per cell.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    > 🔮 Figure: loss curves per cell (train and validation). Expected: flat,
    > stable convergence everywhere, as in ngpt-scaling; the d16 cells are the
    > ones to watch for width-gated instability.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exact-match accuracy (H1)

    > 🔮 Table: exact match per form on seen and held-out pairs at the center
    > cell, beside the nulls (prompt-blind centroid, $k$-NN neighborhood,
    > shell-confined guesser). Expected: hex accuracy comparable to
    > ex-2.1.1's hex levels; named held-out accuracy above the nulls by a
    > margin the 140-name pair count can actually resolve, unlike v27.

    > 🔮 Figure: where the misses land. Distance from guess to true mix in
    > palette $k$-NN terms, per form. Expected: named misses concentrated on
    > nearest neighbors of the true answer; hex misses one grid level off in
    > one channel, as in earlier experiments.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Within-form geometry (H2)

    > 🔮 Figure: probe $R^2$ maps over layer × position for operand and mix
    > values, one panel per form, center cell. Expected: names show operand
    > readout building over layers 1–3 and the mix crystallizing at the
    > pre-answer position in the last layer ($R^2 \approx 0.9$); hex shows the
    > just-in-time channel staircase with low pre-answer full-mix $R^2$. If
    > the hex panel instead shows a holistic pre-answer mix, that is coupling,
    > and the alignment section is where to look next.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cross-form alignment (H3)

    > 🔮 Figure: transfer-ratio $\rho$ maps over layer × position, both
    > directions (hex→name, name→hex), center cell. Expected: $\rho < 0.2$
    > everywhere the within-form probes are strong.

    > 🔮 Figure: principal angles between the two probes' row-spaces at the
    > positions where each form's geometry is strongest. Expected: angles
    > near 90°, i.e. the decoders use different directions of the stream.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Alignment under compression (H4)

    > 🔮 Figure: $\rho$ and principal angles versus width (d64, d32, d16, plus
    > the d16-L8 cell), at each cell's best probe site. Expected: a monotonic
    > rise in sharing as the stream narrows, with d16-L8 the most aligned. A
    > flat line at low $\rho$ would say capacity pressure alone doesn't merge
    > the forms at these scales; alignment already present at d64 would say
    > the merge is a bias of training, and H3 falls.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Alignment with a bridge (H5)

    > 🔮 Figure/table: the bridge cell's $\rho$ beside the center cell's.
    > Expected: cross-form equations lift $\rho$ above 0.8, i.e. a small
    > amount of shared supervision places both forms in the same subspace.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Depth (H6)

    > 🔮 Figure: name-form accuracy and mix-crystallization depth at L8 versus
    > L4. Expected: held-out named accuracy rises, and the layer × position
    > probe map shows the mix decodable before the final layer, giving the
    > result concept more than one layer of existence.
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

    > 🔮 Verdict table for H1–H6 (supported / partial / unsupported), with a
    > pointer to the figure that decides each.

    > 🔮 What the outcome means for anchoring across surface forms: if
    > alignment requires a bridge or compression, anchored runs on a
    > mixed-vocabulary corpus need their labels to touch both forms (or need
    > the bridge); if alignment is free, one form's labels suffice.
    """)
    return


if __name__ == "__main__":
    app.run()
