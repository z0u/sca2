import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Ex 2.2.1: suppressing red in the anchored transformer",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo

    # Once results exist, the design constants and refs come from `experiment.py`
    # beside this notebook (Marimo puts the notebook directory on sys.path); the
    # skeleton quotes the frozen gates in prose and that module carries the same
    # numbers with each gate's wording in its docstring.
    from mini.reports import report_bundle, use_publisher

    use_publisher(report_bundle(__file__))

    # A cell renders its last expression, and a trailing docstring is one.
    None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.2.1: suppressing *red* in the anchored transformer

    /// tip |
    <!-- tl;dr -->
    The first intervention on an anchored transformer. We take the D2.1 checkpoints and remove the anchor axis from the residual stream, then score completion accuracy on lines with a red operand and on lines without. Four questions: does suppression bite, is it selective, does it grade with redness, and does the edit on non-red lines stay within the bound the placed geometry sets?
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Findings

    /// admonition | TODO
    Empty until the results land: one verdict line per hypothesis, with the deciding number and its gate inline.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration. The hypotheses and their gates below are frozen as of the commit that adds this file; results will replace the `TODO` placeholders in place, and anything conceived after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc.

    While this draft was written we ran a one-seed smoke test of the operator library (seed 0 of each condition), to check the code path from checkpoint to score. The gates come from the clean statistics instead: the published alignment maps, the task-gate width the D2.1 experiments used, and the reference values from M1. None of the smoke-test numbers are quoted here.

    One smoke-test observation is worth stating in advance. At one syntax position, the alignment arriving at the second slice under intervention sat above the clean envelope on that seed. We keep the deeper-slice clause of H4 as the design states it; the nine-seed read will say how general that behavior is.
    ///

    ## Why this experiment

    D2.1 anchored *red* on the first axis of the residual stream, and the geometry landed where we asked: red states aligned, the non-red bulk contained, grading roughly sensible, task accuracy unchanged. It never ran an intervention. The point of placing a concept is that removing it afterwards has a predictable effect and a side-effect we can bound, and we have not tested that claim in a transformer.

    The test uses checkpoints already in the store. Projecting e₁ out of the stream at inference time costs nothing to train and about a second per run to score, which makes it the most information per dollar in the [D2.2 plan](../d2.2/design.md).

    It is also where the first three risks in that plan get their first data:
    (a) that suppression does not bite even on *red*,
    (b) that the bound is loose or wrong once later blocks sit between the write and the output, and
    (c) that the response to suppression is undesigned. Only the first is retired here. The plan shares the second with the layer sweep and the third with the fallback experiment.
    <!-- REVIEW: was "settles the first three risks". The design's risk table retires only the first here — the bound risk is "suppress red, then layer sweep", and the undesigned-response risk is measured here and pinned by the fallback experiment. Verify against the risk table in /docs/m2/d2.2/design.md. -->



    Three findings from the M1 autoencoder result carry over here. Suppression there removed *red* almost completely, left orthogonal colors untouched, and degraded warm colors in proportion to their alignment with the axis. Those become H1, H2, and H3.

    The bound does not carry over as it stands. In the autoencoder, the decoder was a single linear readout from the point of intervention, so the geometry bounded the output directly. Here every later block sits between the write and the logits.

    So we state the bound as layer-local, following the [kickoff lessons](/todo/science/d21-kickoff-carry-over-lessons.md): the geometry bounds the immediate write at each intervened slice (H4). Behavior is then a separate prediction, scored by the task gate (H2).

    These checkpoints have no fallback term, so whatever a red line decodes to after suppression is the untrained response of the model itself. M1 called that spoofing, and it is where the seed spread in ex-2.9.1 came from. We measure it here without a gate; it is the reference the fallback experiment after this one has to improve on.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | Glossary
    - **line** — one equation, `c1 + c2 = answer`, six word-level tokens. The probe set enumerates all 5,832 closed lines of the 216-color grid: every color as op1 against each of its 27 closed partners.
    - **residual stream** — the running vector each token carries through the model, which every block reads from and adds back into.
    - **slice** ($\ell$) — a depth at which the residual stream is read: the embedding, plus the stream after each of the four blocks. Five slices.
    - **alignment** ($\alpha$) — $\cos(h, e_1)$, a state's cosine to the anchor axis. States are unit-norm, so it is the e₁ component itself.
    - **dose** — the redness of a line, taken as the larger of the two operand rednesses (the label affinity from M1, $r(1 - g/2 - b/2)$ on the unit cube). Either operand can carry the concept, since either could draw the label in ex-2.1.10.
    - **red lines** — dose ≥ 0.8: an operand among the five reddest colors of the grid. 365 lines.
    - **non-red lines** — dose ≤ 0.2. 1,689 lines.
    - **response** — the probability the model puts on the correct answer, read from the log-softmax at the `=` position. Accuracy is whether that answer is the argmax.
    - **damage** — the clean response minus the intervened response, per line, in probability units.
    - **write** — the rotation the operator applies to one state, in radians: the angle between the state as it arrived at the operator and the state the next block consumed.
    - **clean** — the un-intervened forward pass of the same checkpoint.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    No training. Every run is a published checkpoint of ex-2.1.10, scored on that experiment's published probe set through the eval contract in [`sca.intervention`](/src/sca/intervention.py): the same lines, the same code, and the same statistics for every condition, arm, and baseline.

    ### The intervention

    Axis projection at full strength: at each slice, every state loses its e₁ component and is put back onto the sphere, since the next block expects a unit vector. That re-projection is part of the edit.

    For a state with alignment $\alpha$, the surviving components are rescaled by $1/\sqrt{1 - \alpha^2}$, and the whole edit amounts to a rotation by $\arcsin|\alpha|$. That formula depends only on the alignment the state arrived with, so we can compute the bound from the published alignment maps before the intervention runs.

    The primary intervention acts at all five slices and all six positions. The operator at slice $\ell$ sees a state the blocks built from already-edited inputs, so only at the embedding does it see the clean stream. At every later slice, the arriving alignment measures whether the blocks put the axis back.

    ### Measurements

    One teacher-forced pass per run per intervention, over all 5,832 lines. From each pass:

    - **The response** per line: the log-softmax over the vocabulary at the `=` position. From it we read the probability on the correct answer, the argmax, and the mass outside the color vocabulary.
    - **The write** per (slice, line, position): the angle between the arriving state and the edited state. The pipeline asserts that this equals $\arcsin|\alpha_{\text{arriving}}|$ at every site, so where a map reads more easily we report the write as that alignment.
    - **The clean alignment map** per run, from the un-intervened pass. This is the same array ex-2.1.10 published, recomputed here so every quantity comes from one code path.
    - **The final-state displacement** per line: the angle between the intervened and clean states at the last slice, `=` position. That is the state the logits are read from. We report it in radians, the same scale as a write.

    **The undesigned response**, on red lines under the primary intervention, without a gate. We record the mass outside the color vocabulary; how far the decoded answer sits from the true mix, in unit-cube units; which answer it is (the visible operand, the red operand itself, the true answer, a one-step neighbor of it, or something else); and seed agreement, the fraction of red lines on which at least five of the nine seeds decode the same answer. These are the reference numbers for the fallback experiment.

    **Noise floor.** Clean accuracy on every group is at or near 1.0 in every run, so the seed scatter that matters is in the intervened statistics. For each group statistic we compute the pooled between-seed standard deviation per condition before reading any comparison. A difference between arms or tiers smaller than twice that value is reported as not resolved. Gates score seed means against fixed thresholds and do not use the floor.

    ### Conditions

    Two conditions from ex-2.1.10, both trained on the same corpus with the same seeds and code path:

    - **anchored** (`either-t100`, 9 seeds) — the D2.1 recipe: pooled either-operand labels, the anchor at λ = 0.1, the anti-subspace term at the ex-2.1.8 operating point. This is the primary condition, and the one every hypothesis scores.
    - **un-anchored** (`lam0`, 3 seeds) — the anchor weight at zero. Nothing was placed on e₁, so projecting it out applies the operator to a model with nothing on the axis. That calibrates what a write of this kind costs when it removes no concept.

    **Arms**, on the anchored condition. Each changes one thing from the primary intervention, and we report the same statistics for each without gates:

    | arm | what changes | question |
    | --- | --- | --- |
    | `operands` | positions restricted to op1 and op2 | does the edit need the syntax positions, whose embeddings carry a constant component on the axis? |
    | `embedding` | slice 0 only | is removing the token's component enough, or do the blocks re-derive red? |
    | `last` | slice 4 only | the one-readout case, closest to M1's geometry |
    | `lobe` | the M1 lobe, threshold 0.5, in place of the plain projection | does leaving the low-alignment bulk alone keep the removal and cut the collateral? |
    | `ablate` | the weights that read and write e₁ zeroed, then re-normalized | the permanent removal from M1, scored under H5 |

    **The post-hoc tier**, on the un-anchored condition, no gates. For each `lam0` run we fit one direction per slice on its own op1-position states (one line per color, 216 states), then apply it at that slice at every position through the same scorer. Three fits beside e₁ itself: diff-in-means between the colors at or above the red dose and the rest, a ridge probe for redness, and LEACE for redness. This calibrates what a searched-for direction removes from a model that was never asked to place one.

    **The strength sweep**, on the anchored condition: the primary intervention at γ ∈ {0.25, 0.5, 0.75, 1}, where γ is the fraction of the e₁ component removed. Exploratory (E1).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    All on the anchored condition under the primary intervention unless stated; seed means over nine runs.

    - **H1.** Suppression bites. Seed-mean exact-match accuracy on red lines falls to 0.5 or below, from a clean accuracy of 1.0. Partial: between 0.5 and 0.8. Contrary: accuracy stays above 0.8, which would mean the model reads the color of the operand from somewhere the axis does not reach.
    - **H2.** Suppression is selective. Seed-mean accuracy on non-red lines stays within 0.02 of the clean accuracy. Partial: within 0.05. The gate is the width every D2.1 task gate used.

        Projecting out an axis costs something even where nothing was placed on it. So we report the un-anchored row under the same operator beside this one, to say how much of any deficit comes from the operator rather than from the removal.
        <!-- REVIEW: H2 was also stated as "the behavioral half of the bound claim", with a miss under an intact H4 read as "a finding rather than a failure of the bound". That gave one bullet two verdicts. H2 is now the task gate alone, with one verdict; what a miss means for the bound is interpretation, and lives in the H4 analysis. Verify: the H4 section says how an H2 miss is read, and H2's verdict line in Findings is a plain pass/partial/fail. -->
    - **H3.** Damage grades with dose. We bin every line by dose at edges 0.2, 0.4, 0.6, and 0.8. Seed-mean damage is non-decreasing across the five bins, allowing a dip between adjacent bins of at most 0.02, and damage in the top bin exceeds the bottom bin by at least 0.5. Partial: one dip larger than 0.02, with the ends still ordered. We prescribe no shape between the ends. The D2.1 alignment response was sigmoid in redness rather than linear, so the grading may be a step rather than the smooth $\cos^2$ curve M1 saw.
    - **H4.** The write on non-red lines stays within the bound the clean geometry sets, at every slice. At a given (slice, position) site, the bound is $\arcsin$ of the seed-mean 99th-percentile clean alignment over non-red lines there, taken from the un-intervened map. The measurement is the seed-mean 99th-percentile write over the same lines under the primary intervention. At the embedding the two are equal by construction, and the pipeline asserts it. At each of the four deeper slices, and at every position, the measured write is at most 1.25 times the bound. Partial: one site over by no more than a factor of two. Contrary: a block re-writes the axis on non-red lines after it was removed upstream. That would be a bypass, and the question of the layer sweep arriving early.
    - **H5.** Permanent removal is selective, given the anti-subspace term. Under the `ablate` arm, the H1 and H2 gates both hold: red-line accuracy at or below 0.5, and non-red accuracy within 0.02 of clean. Partial: one of the two holds. M1 found that ablation was selective only when repulsion had cleared the axis, and the D2.1 recipe has that term, so this is the transformer form of the headline result from M1.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Suppression bites (H1)

    /// admonition | TODO
    A table of clean and intervened accuracy under the primary intervention, seed mean with the seed range, for the red lines, the non-red lines, and all lines. The calibration row is the un-anchored condition under the same operator. Beside it, the per-line response on red lines: a strip or histogram of the intervened probability on the correct answer, per seed, so that a bimodal response (some lines fully removed, some intact) shows up as such instead of averaging into the gate.

    Expected: red-line accuracy far below 0.5, and the un-anchored row at 1.0. Contrary: red-line accuracy above 0.8, with the color of the operand still read from the off-axis remainder of its embedding. The `leak_r2` figure from ex-2.1.10 says how much redness the off-axis state carries at op1, and that is the first number to look at.
    ///

    ## Selectivity (H2)

    /// admonition | TODO
    The non-red row of the H1 table, scored against the 0.02 gate, and the same row for every arm and for the post-hoc tier. One figure carries H2 and H3 together: damage against dose, per line, which is the grading cloud from D2.1 with damage on the y axis. The non-red bulk should sit on the floor and the red lines near the ceiling.

    Expected: non-red accuracy within the gate, the `operands` arm no worse, the `lobe` arm no worse. Contrary: a non-red deficit larger than 0.02 while H4 holds, which is amplification; or a deficit that the `operands` arm removes, which would place the collateral in the constant component of the syntax positions.
    ///

    ## Grading (H3)

    /// admonition | TODO
    Seed-mean damage per dose bin, with the seed range, as a step chart over the five bins; the same for the `lobe` arm and for the un-anchored condition. Expected: a monotone rise, possibly a step between the 0.6 and 0.8 edges rather than a ramp, and the un-anchored condition flat. Contrary: an inversion, with a middle bin damaged more than the bin above it. That would mean the axis carries something other than redness for those colors, and the per-color breakdown under E3 says which colors.
    ///

    ## The write bound (H4)

    /// admonition | TODO
    Two maps of the non-red 99th-percentile alignment per (slice, position): the clean bound, and the arriving alignment under intervention, on the shared 0–1 scale the D2.1 figures use. A table beside them gives the ratio at each site. The embedding row is the identity check.

    Expected: the deeper rows at or below the bound. The blocks were trained with the anti-subspace term on every state, so a non-red state arriving without its axis component is close to what they saw in training. Contrary: a site where the arriving alignment exceeds the bound by more than a quarter, most likely a syntax position at the second slice, where the smoke test saw one seed do exactly that. If that happens, the `operands` and `lobe` arms say whether it matters to behavior, and E5 says whether the edit grows or shrinks on its way to the logits.

    H2 is read beside this section. The bound covers the write and behavior is a prediction against it, so an H2 miss with the bound intact is amplification by later blocks: a finding for the layer sweep, and H2 fails as stated while H4 stands. An H2 miss with H4 also missed is the bound failing, which is the plan's second risk.
    ///

    ## Permanent removal (H5)

    /// admonition | TODO
    The H1 and H2 rows for the `ablate` arm, beside the rows for the projection. Expected: the same removal and the same selectivity, since a stream that never carries e₁ is what the projection produces at every slice, up to the gain on the surviving components.

    Contrary: ablation biting less than projection, which would mean the blocks rebuild red from the off-axis embedding when the axis is unavailable from the start; or ablation costing more on non-red lines, which would mean the read and write matrices use the axis for something the anti-subspace term did not clear.
    ///

    ## The undesigned response

    /// admonition | TODO
    No gate. For red lines under the primary intervention: the mass outside the color vocabulary; how far the decoded answer sits from the true mix; the composition of decoded answers (visible operand, red operand, true answer, one-step neighbor, other), as a stacked bar per seed; and the seed-agreement fraction.

    Expected: completions stay in the color vocabulary, the decoded answer sits nearer the visible operand than the true mix, and agreement across seeds is low. An untrained response should scatter by seed.
    ///

    ## The post-hoc tier

    /// admonition | TODO
    No gate. The H1–H3 statistics for each fitted direction on the un-anchored condition, beside e₁ on the same condition and beside the anchored primary. The comparison to read: how much red damage a searched-for direction buys per unit of non-red damage, compared with the placed axis.

    Expected: diff-in-means and the probe remove some red-line accuracy at a real non-red cost, LEACE less of both, and none of them come close to the selectivity of the placed axis. A post-hoc direction that matched the anchored result would mean the un-anchored model already keeps redness on a single linear direction at op1, which the probe maps in ex-2.1.12 make plausible at the embedding. The per-slice breakdown says where the two part company.
    ///

    ## Exploratory analyses

    /// admonition | TODO
    Preregistered as exploratory, no gates.

    **E1 — strength.** Red-line and non-red-line accuracy and damage at γ ∈ {0.25, 0.5, 0.75, 1}. The write at strength γ is closed-form in α, so the x axis can be drawn either as γ or as the write on red operands; the question is whether the response is proportional to the write or thresholded.

    **E2 — where the edit acts.** The H1–H3 statistics for the `embedding` and `last` arms against the primary. If `embedding` alone bites, the concept is read off the token and the blocks do not re-derive it. If only the primary intervention, at every slice, bites, the bypass test the D1.3 post left open gets its first transformer data point.

    **E3 — damage against alignment.** Per line, damage against the clean alignment at the position of the red operand, at each slice. In M1 the relation was error ∝ $\cos^2$. Also per color: the mean damage of lines with that color as an operand, against the alignment of that color. This is the per-color breakdown the contrary case of H3 asks for.

    **E4 — the lobe.** The H1–H4 statistics for the `lobe` arm. Its threshold of 0.5 sits above the constant component of the syntax embeddings and below the alignment of the red operand, so by design it should remove red and leave the syntax states alone. Its write map is the H4 figure with the syntax columns emptied.

    **E5 — amplification.** Per non-red line, the final-state displacement at the `=` position against the sum of the writes on that line at the prompt positions, over all slices. Both are angles, so their ratio is the gain the blocks apply to the edit on its way to the logits; a ratio above one is amplification. We read it for the primary intervention, and for the `last` arm, where no block sits after the edit. The `last` arm is the no-amplification reference.
    <!-- REVIEW: the displacement is now an angle rather than a distance, so the ratio is dimensionless; and the `last` arm's ratio is not one by construction, since the denominator sums the writes at all four prompt positions while the numerator is the displacement at `=` alone. Verify: under `last`, the numerator equals the write at (slice 4, `=`), so the ratio is one only when the other three prompt writes vanish. -->

    ///

    ## Discussion

    /// admonition | TODO
    Interpretation only, after the results. What the removal and its selectivity let the D2.2 plan say about the anchored-op experiments that follow; which operator the damage figures pick for them, which the [shaped-suppression item](/todo/science/shaped-suppression-rather-than-projecting-whole-axis.md) is waiting on; what the undesigned response says the fallback experiment must improve on; and what the write bound and E5 say about the layer sweep. No re-derivation of the findings.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
