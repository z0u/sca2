import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.12: fitted channel probes over the anchored checkpoints",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np

    # Marimo puts the notebook directory on sys.path, so the conditions and the
    # ref come from the definition module beside this one.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store

    use_publisher(report_bundle(__file__))


@app.function(hide_code=True)
def load_arrays() -> dict[str, np.ndarray] | None:
    """The published probe arrays, or None before the run."""
    store = project_store()
    art = store.get_refs([ex.ARRAYS_REF])[ex.ARRAYS_REF]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "probes.npz")])
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.12: fitted channel probes over the anchored checkpoints

    /// tip |
    <!-- tl;dr -->
    We fit ridge probes for the RGB channels of each operand and the answer, at every (slice, position) site of the four D2.1 conditions, from their published checkpoints. Two questions: is the off-key tilt in the grading figures fully a property of the probe set, which caps what any model can show at the sites that precede op2 — and does anchoring cost any ordinary color decodability, which nothing published yet measures?
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
    ///

    ## Why this experiment

    The α maps of [the D2.1 figures page](../d2.1/report.py) key one measurement by either operand's color, and wherever a condition responds at all, the slot that is *not* the key tilts too. For the slots that precede op2 that tilt cannot be a response: attention is causal, so the state above the first token is a function of that token alone, and the tilt is the on-key response composed through the probe set — a pair is on the set only if its mix lands back on the color grid, which correlates a color with its partners. The α statistic cannot show this cleanly, because it reads alignment with the anchor axis, which the un-anchored control ignores; its panels are flat for lack of any response to compose, and a reader cannot tell the composition story from the alignment story by looking.

    Fitted probes remove that asymmetry. A probe fit at a site reads whatever color representation the model keeps there, whether that is the anchored one or the model's own. That puts every condition on the same footing: whatever any of them shows at the sites that precede op2 is capped by the information ceiling of the pairing, computable in advance (H1), and how much of that ceiling a condition reaches is a property of its embedding geometry rather than of anchoring. [Ex-2.1.5](../ex-2.1.5/report.py) ran the same kind of scan on an un-anchored two-form model, and there the op2 probes read zero at op1 sites. That experiment sampled its lines over distinct operand pairs, so the operands were all but uncorrelated and the ceiling was effectively zero. Comparing the two probe sets is what makes the case: the tilt follows the pairing, and the model has no say at these sites.

    The same scan answers a question the D2.1 experiments left open. Anchoring reshapes the stream — that is what it is for — and the task gates only showed that *accuracy* survives. Whether the ordinary, linearly readable color geometry survives too is the bounded-side-effects claim read through a probe, and per-channel R² maps against the un-anchored control measure it (H2).

    ## Method

    No training. Every run is a published checkpoint of the four D2.1 conditions, probed on its own experiment's published probe set (the four sets are identical, which the pipeline asserts): 5,832 lines, every color as op1 against each of the 27 partners whose mix stays on the grid.

    **Sites and targets.** The residual stream is captured at 5 slices (embedding, then each of the 4 layers) × 6 token positions (`op1 + op2 = ans \n`). At each site, ridge probes (ℓ₂ = 10⁻²) decode three targets: the RGB triple of op1, of op2, and of the answer. Probes are fit and scored per site independently, as in ex-2.1.5.

    **Holdout.** The strict per-value protocol of ex-2.1.5 (`sca.compute.geometry.strict_r2`): to score a channel at a level, every line holding that level in that channel of *any* role — either operand or the answer — leaves the fit together, so the probe must place an unseen level from the others. R² is reported per channel from the out-of-fold predictions, and the prediction curves the figures draw are those same out-of-fold predictions, averaged per color keyed by either operand in turn (the two margins the D2.1 figures use).

    **Teacher forcing.** The ans and `\n` positions have the answer token in the input, so readings there include what the token itself carries; the embedding row is the surface-text control, as in ex-2.1.5.

    **Noise floor.** The primary condition has nine seeds and the others three. The pooled between-seed standard deviation of per-site R², computed per condition before any comparison is read, sets the resolution: a difference smaller than twice that floor is reported as not resolved.

    ### Conditions

    The four conditions of the D2.1 progression, from their published checkpoints: the un-anchored control and the bare anchor at λ = 0.1 (ex-2.1.6, 3 seeds each), the anchor plus the anti-subspace term at its tuned operating point (ex-2.1.8 `end90-hold30`, 3 seeds), and the full recipe with pooled either-operand labels (ex-2.1.10 `either-t100`, 9 seeds).

    ## Hypotheses

    - **H1.** At the two sites that precede op2 — the op1 and `+` positions, every slice — the seed-mean strict R² of the op2-target probe stays below 0.12 in every condition and every channel. This is deliberately a protocol check rather than a discovery: the state at these sites is a function of the tokens before op2, so the best any probe can do is E[op2 | op1], and the closure rule caps that at R² = 0.25 / (35/12) ≈ 0.086 per channel, whatever the model; the gate adds margin for estimation noise. A reading at or above 0.12 would mean the probe found more op2 information before op2 than the pairing supplies — a leak in the protocol or a misread of the probe set. No lower gate is stated, because every outcome below the ceiling is consistent with the bookkeeping account: how much of the ceiling each condition reaches is a property of its embedding geometry (how linearly it exposes level parity), read descriptively under E1, and a cross-condition difference there would be an observation about geometry that does not reopen the causal question.
      <!-- REVIEW: prereg round asked for a lower or similarity gate on H1, since the upper gate alone cannot fail without a bug. Resolved the other way: H1 now says plainly that it is a protocol check, and the cross-condition comparison moved to E1 as descriptive — a floor would score how linearly each embedding exposes parity, which is not the claim under test. Verify: the H1 analysis TODO's contrary line matches this scope. -->
    - **H2.** Anchoring does not cost linear color decodability where the control has it. Site selection is separated from scoring so the selection noise stays off the gate: for each target, the comparison site is the (slice, position) with the best strict R² (mean over channels) in the map of the control's seed 0 alone, ties broken toward the earlier slice and then the earlier position, among targets whose selection map reaches R² ≥ 0.5 (a target below that is reported unscored). The score then compares the seed-mean of the control's remaining seeds against the seed-mean of all nine primary seeds. At each selected site, the primary's per-channel R² comes within 0.10 of the control's (one-sided: the primary may exceed it). A miss smaller than twice the pooled between-seed standard deviation at that site is reported as not resolved rather than as a failure. Partial: exactly one channel of one target misses by more than that and by no more than 0.2. The intermediate conditions are reported at the same sites without a gate.
      <!-- REVIEW: prereg round flagged the original argmax-over-30-sites on the control's own 3-seed mean as loading the one-sided gate against the primary (selection inflates the control at the chosen site). Resolved by selecting on control seed 0 and scoring on seeds 1–2; ties get a fixed break; the method's noise floor now decides "not resolved" for near-gate misses. Verify: the analysis never lets the selection seed into the scored mean. -->
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The off-key ceiling (H1)

    /// admonition | TODO
    Per-channel strict R² maps for the op2 target, one panel per condition: slices upward from the embedding, positions across, one step-line per channel (the `probe_trace_grid` layout of ex-2.1.5). A horizontal rule at the 0.086 ceiling over the op1 and `+` positions. Expected: every condition, control included, reads at or below the ceiling at those positions, with the genuine response from the op2 position onward. Contrary: any pre-op2 reading clearing 0.12, which would mean a protocol leak. How much of the ceiling each condition reaches is read descriptively; differences between conditions there are geometry observations, taken up under E1, and do not bear on the causal account.

    Beside it, the deciding table: max pre-op2 seed-mean R² per condition × channel, against the gate.
    ///

    ## Decodability under anchoring (H2)

    /// admonition | TODO
    Per-channel strict R² maps for the op1 and ans targets (the op2 map is above), per condition. The deciding table: the site chosen by the control's selection seed per target, the control's scoring-seed mean vs the primary's seed-mean per channel with the one-sided 0.10 gate and the noise-floor "not resolved" band, plus the intermediate conditions unscored. Expected: differences within the gate everywhere, i.e. anchoring reshapes where redness lives without spending the color geometry the task needs. Contrary: a systematic deficit in the anchored conditions, largest for the R channel or at the sites where α is largest — which would be the first measured side-effect of anchoring on representation quality.
    ///

    ## Exploratory analyses

    /// admonition | TODO
    Preregistered as exploratory, no gates.

    **E1 — the shape of the off-key reading.** At the op1 site, the ideal op2 predictor is the parity sawtooth E[op2 level | op1 level] = 2, 3, 2, 3, 2, 3 (in level units). Compare the fitted probes' out-of-fold prediction curves at that site with the sawtooth, per channel; how much of the ceiling each model reaches says how linearly its embedding geometry exposes level parity.

    **E2 — the R channel is a poor proxy for α.** Correlation, over colors, of the on-key R-channel prediction curve with the α lookup at the same site, per anchored condition. Over the palette, corr(R, sim¹·⁵) = 0.61 while corr(redness, sim¹·⁵) = 0.90, so we expect the R channel to track α loosely and no channel to reproduce it.

    **E3 — drag at the embedding.** The emb-slice on-key curves, compared in shape against the direct-credit prediction (the affinity itself) and the drag prediction (the partner-mean of the affinity). From the published α margins, the op1-keyed-label conditions lean toward the drag shape and the pooled-label primary toward the direct shape; the probe curves test whether that ordering survives in the model's own representation rather than only along the anchor axis.
    ///

    ## Discussion

    /// admonition | TODO
    Interpretation only, after the results: what the ceiling result licenses the D2.1 figures page and the post to say about the off-key tilt, and what the decodability result adds to the bounded-side-effects account. No re-derivation of the findings.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    _arrays = load_arrays()
    mo.stop(_arrays is None, mo.md("_Results are not published yet; the analysis renders once they are._"))
    return


if __name__ == "__main__":
    app.run()
