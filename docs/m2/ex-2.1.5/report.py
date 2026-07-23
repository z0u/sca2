import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.5: disjoint vocabularies and more named colors",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo

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

    /// admonition | TODO
    Figure: the palettes, drawn with `sca.vis.plot_rgb_cube` — one subfigure
    per operand set, nested as in ex-2.1.1. One or two labelled points per
    subfigure, named in the caption ("a: `ultramarine`; b: `#48a`").
    Expected: fairly uniform spread through the cube.
    ///

    /// admonition | TODO
    Figure/table: corpus statistics. Line counts per form, the answer-name
    frequency distribution (the design study measured perplexity ≈ 87 over
    140 names under uniform pair sampling, with 139 of 140 names reachable as
    answers), and sequence lengths against the block size.
    ///
    """)
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

    /// admonition | TODO
    Table: parameter counts and training cost per cell.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    /// admonition | TODO
    Figure: loss curves per cell (train and validation). Expected: flat,
    stable convergence everywhere, as in ngpt-scaling; the d16 cells are the
    ones to watch for width-gated instability.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exact-match accuracy (H1)

    /// admonition | TODO
    Table: exact match per form on seen and held-out pairs at the center
    cell, beside the nulls (prompt-blind centroid, $k$-NN neighborhood,
    shell-confined guesser). Expected: hex accuracy comparable to
    ex-2.1.1's hex levels; named held-out accuracy above the nulls by a
    margin the 140-name pair count can actually resolve, unlike v27.
    ///

    /// admonition | TODO
    Figure: where the misses land. Distance from guess to true mix in
    palette $k$-NN terms, per form. Expected: named misses concentrated on
    nearest neighbors of the true answer; hex misses one grid level off in
    one channel, as in earlier experiments.
    ///

    /// admonition | TODO
    Table: the density arms beside the center cell. Expected: hex accuracy
    insensitive to the operand-subset size (hex-dense), and named accuracy at
    250 names holding or improving on 140 (palette-250), continuing the
    density trend of ex-2.1.3.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Within-form geometry (H2)

    /// admonition | TODO
    Figure: probe $R^2$ maps over layer × position for operand and mix
    values, one panel per form, center cell. Expected: names show operand
    readout building over layers 1–3 and the mix crystallizing at the
    pre-answer position in the last layer ($R^2 \approx 0.9$); hex shows the
    just-in-time channel staircase with low pre-answer full-mix $R^2$. If
    the hex panel instead shows a holistic pre-answer mix, that is coupling,
    and the alignment section is where to look next.
    ///
    """)
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
