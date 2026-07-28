import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    app_title="Ex 2.1.6: anchoring red in a transformer",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ex 2.1.6: anchoring *red* in a transformer

    The first anchored transformer. During training we add a single regularizer
    term that pulls the residual-stream activations of *red-labeled* equations
    toward a fixed direction $a$, and then ask two questions: did the concept
    land where we put it, and what did that cost the task?

    This is the opening experiment of D2.1. M1 established anchoring in
    autoencoders; everything since ex-2.1.1 has been un-anchored groundwork —
    finding a testbed where the task is solved and the concept geometry is
    legible, so that an anchor's effects are attributable. That groundwork sets
    the design:

    - **Testbed** (ex-2.1.3): the word-level `v216` cell — 216 grid colors,
      one token each, `name + name = name` equations, d64-L4. Held-out
      accuracy is ≈ 1.0 and the color cube is linearly decodable from the
      embeddings, so there is headroom to lose and a known un-anchored
      geometry to compare against.
    - **Labels** (M1/Ex-1.7 via ex-2.9.1): sparse, noisy, binary — each line
      is labeled *red* with probability `redness(op1)⁸ × 0.08`, so only
      strongly red first operands are ever labeled, and even they usually
      aren't.
    - **Schedule** ([issue #10], from ex-2.9.3): regularizer protection must
      outlive the optimizer's heat. The anchor weight holds at peak until the
      LR decay is essentially done, then anneals to a floor — never to zero.
    - **Architecture**: our simplified nGPT keeps every residual-stream state
      on the unit hypersphere, so M1's cosine-geometry anchor term transfers
      without modification, and there is no β-calibration problem during
      training (norms are pinned at 1).

    Deliberately *absent*: the anti-anchor, anti-subspace, and separate terms,
    and any intervention (ablation/redirect). One term only. Exclusivity —
    does *only* red live on the axis? — is measured rather than enforced, and
    intervention is deferred to a later experiment, where $-a$ first has a
    job.

    Labels attach to *sequences*, and the anchor term applies uniformly to
    every position and layer of a labeled line. That is the natural-language
    shape (a document is labeled, not a token), and it turns the interesting
    question into one the training dynamics must answer: when the pull is
    undifferentiated but the task loss resists it unevenly, does the concept
    condense at the position that computes it, or smear into a broadcast
    "this line mentions red" feature?

    [issue #10]: https://github.com/z0u/sca2/issues/10
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistered skeleton: the method, hypotheses, and decision
    thresholds below were written and frozen before the experiment ran.
    Results will replace the `TODO` placeholders in place, so review reads as
    a prediction → observation diff. Anything conceived after seeing the data
    goes under *Exploratory analyses*, marked post hoc.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Method

    ### Task and corpus

    Identical to ex-2.1.3's `v216` cell: the level sub-grid `(0, 3, 6, 9, 12,
    15)` of the 16-level cube (216 colors, one opaque token each), equations
    `name + name = name` with exact integer mixing, 100 k lines, 20 % of
    distinct closed pairs held out. Eval sets are the same three:
    `named_seen`, `named_holdout`, and `open` (pairs whose mix has no name).
    One addition: an **alignment probe set** of all 216 colors as operand 1,
    each crossed with 8 random second operands, so per-color alignment
    statistics average over op2 rather than conditioning on it.

    ### Labels

    A line is labeled *red* by a Bernoulli draw with
    `p = redness(op1)⁸ × 0.08`, where `redness = r·(1 − g/2 − b/2)` — M1's
    `RED_PROB`, applied to the first operand's color only. Only op1 drives
    the label; a red op2 or a red answer never does. That keeps the ground
    truth for *where* the concept should condense unambiguous (the either-slot
    labeling scheme is queued as a follow-up).

    Labels are drawn **per visit**, as in M1, so a color's expected pull is
    proportional to its label affinity — which is where H2's graded response
    is predicted to come from. Under a plain Bernoulli draw ≈ 0.1 % of lines
    would be labeled and ~93 % of batches would carry none, so batches are
    **label-balanced** (issue #10): each batch of 64 includes 4 lines drawn
    with probability proportional to their label affinity and treated as
    labeled, with the anchor term normalized by the labeled count (M1's
    label-affinity-weighted mean). That reproduces the Bernoulli expectation
    with the per-batch count fixed, so the term's gradient variance does not
    inherit the label sparsity. (The realism variant — labels drawn once at
    corpus build, a property of lines like labeled internet text — is queued
    beside either-slot labeling: both replace expected pull with what the
    model infers from sparse fixed evidence, and both deserve a run where the
    baseline mechanism is already understood.)

    ### Model and anchor term

    The d64-L4 simplified nGPT from ex-2.1.1 onward, unchanged. The anchor
    direction is $a = e_0$, matching M1; the simplified architecture's
    residual operations (LERP toward normalized sub-module output, scalar
    gains) are rotation-equivariant, so the basis choice is a convenience,
    not a hint.

    For labeled lines, the anchor term is

    $$\mathcal{L}_{\text{anchor}} = \frac{1}{|\mathcal{P}|\,(L{+}1)}
    \sum_{t \in \mathcal{P}} \sum_{\ell=0}^{L}
    \bigl(1 - \cos(h_{\ell,t},\, a)\bigr)$$

    — the mean over every non-padding position $\mathcal{P}$ of the line and
    all $L{+}1 = 5$ residual-stream slices (embedding plus four block
    outputs). Unlabeled lines contribute nothing. No other term is added.

    ### Schedule and arms

    LR as in ex-2.1.3: 100 epochs, 10 warmup, cosine to 1 %. The anchor
    weight ramps to its peak alongside the LR warmup, **holds at peak through
    epoch 90** (by which point the LR has decayed to a few percent of peak),
    then anneals linearly to a **10 % floor** at epoch 100. It never reaches
    zero — the ex-2.9.3 lesson, applied as issue #10 prescribes.

    Arms: peak anchor weight $w \in \{0,\ 0.03,\ 0.1,\ 0.3\}$ × seeds
    $\{0, 1, 2\}$, 12 cells. The $w{=}0$ arm is the in-experiment control
    (same corpus, labels ignored). **Hypotheses are scored on the middle rung
    $w{=}0.1$**, prespecified here so the ladder can't be cherry-picked; the
    other rungs chart the dose–response.

    ### Measurements

    - **Behavioral**: exact match, NLL, and the distance metrics of ex-2.1.3
      on all eval sets, with the usual nulls from `sca.baselines`.
    - **Alignment maps**: $\cos(h_{\ell,t}, a)$ on the alignment probe set,
      per layer × position. Lines are grouped by op1's similarity to red
      using M1's grouping (`SIM3 > 0.5` = red, `< 0.01` = other), and the
      **contrast** $C(\ell, t)$ is the red-group mean minus the other-group
      mean. The headline statistic is the layer-mean contrast at op1's
      position, $A = \tfrac{1}{L+1}\sum_\ell C(\ell, \text{op1})$ — the mean
      over anchored sites, chosen because the anchor applies to all of them
      equally (no site selection). The min and max over layers are reported
      beside it as unscored diagnostics: a wide min–max gap localizes which
      depths resist the anchor (the last layer is the candidate, if
      next-token pressure competes), without any single layer's behavior
      deciding a hypothesis.
    - **Grading**: per op1 color, the layer-mean $\cos(h, a)$ at op1's
      position, against that color's redness.
    - **Leakage** (secondary, no threshold): a ridge probe for redness fit on
      the residual stream *with the $a$-component projected out*, at op1. High
      off-axis R² means red is also encoded elsewhere — the completeness
      question SCA exists to answer, measured rather than assumed.
    - **Trajectories**: every ~50 steps, $A$ and validation loss, with the LR
      and anchor-weight schedules recorded alongside — the ex-2.9.3
      apparatus, carried over per issue #10's first queue item. A flat loss
      curve does not mean learning is done; the alignment trajectory is the
      evidence to consult before any future decision to shorten the schedule.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hypotheses

    Scored on the $w{=}0.1$ arm (seed-averaged) unless stated; the other
    rungs are context.

    - **H1.** Anchoring does not harm the task: every anchored arm's
      `named_holdout` exact match is within **0.02** (absolute) of the
      $w{=}0$ control's seed mean. *Partial*: the gate passes at
      $w \le 0.1$ but fails at $0.3$ (a capacity cost with a knee, not a
      broken method).
    - **H2.** The concept lands on the anchor, graded: **(a)** $A \ge 0.5$
      (the layer *mean*; per-layer min and max are reported but unscored, so
      no single depth decides the verdict);
      **(b)** Spearman $\rho \ge 0.8$ between op1 redness and per-color
      layer-mean $\cos(h, a)$ at op1, over the 216 colors; **(c)** the
      control arm shows $|A| \le 0.1$. *Partial*: (a) and (c) hold but the
      response is a step rather than a grade (ρ high only because red vs
      non-red separates; within-red ordering absent).
    - **H3.** The concept condenses at the position that carries it:
      $C(\cdot, \text{op1})$ (layer-mean) is at least **2×** the layer-mean
      contrast at each of `+`, op2, and `=`. The answer position is excluded
      from this comparison — a red op1 drags the mix toward red, so contrast
      there is confounded by construction. *Alternative outcome, explicitly
      anticipated*: broadcast — all positions of labeled lines move toward
      $a$ roughly equally. That would fail H3 while H2 passes, and is a
      finding about sequence-level labeling rather than a failure of
      anchoring; the logsumexp position-pooling variant is the queued
      response.
    - **H4.** The late-instability mechanism of ex-2.9.3 does not reappear
      under the anneal-to-floor schedule: for every anchored run, the
      end-of-training $A$ is at least **0.8×** its running maximum over
      training. *Partial*: violations confined to the $w{=}0.3$ arm.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Task cost (H1)

    /// admonition | TODO
    Table: exact match by arm ($w$ rows) × eval set (`named_seen`,
    `named_holdout`, `open` distance), seed mean ± range, with the $w{=}0$
    control row on top and the ex-2.1.3 `v216` published numbers quoted
    beside it as an external reference. Expected: a flat column within
    0.02 of control at every rung. Contrary: a monotone drop with $w$ —
    read together with H2's dose–response to see what alignment level the
    lost accuracy bought. Include the `sca.baselines` nulls for the `open`
    distance row.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Alignment and grading (H2)

    /// admonition | TODO
    Two figures. (1) Grading curve: per-color layer-mean $\cos(h, a)$ at
    op1's position vs op1 redness — one point per color, color-swatch
    marks per the figure conventions, control arm greyed underneath, one
    panel per rung. Expected: monotone and graded, with the labeled-red
    corner near $\cos = 1$ and the far cube near the control's baseline;
    Spearman ρ annotated. Contrary: flat (anchor didn't take) or a binary
    step at the label threshold (took, but un-graded — the partial
    outcome). (2) Per-layer decomposition of $A$: contrast by layer at
    op1, all rungs, control band behind. Expected: contrast at every
    depth; a last-layer sag would indicate next-token pressure competing
    at op1 (the "will the last layer be allowed" question — though op1's
    target is always `+`, so slack should suffice).
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Condensation vs broadcast (H3)

    /// admonition | TODO
    Heat map: contrast $C(\ell, t)$, layers × the six line positions
    (op1, `+`, op2, `=`, answer, newline), primary arm, seed mean; the
    layer-mean row annotated with the 2× threshold verdict against `+`,
    op2, and `=`. Expected: a dominant op1 column. Contrary: a uniformly
    warm map (broadcast). The answer column is shown but excluded from
    scoring (confounded: red op1 → reddish mix); note in the caption
    whatever it does show, since it previews the either-slot follow-up.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training stability (H4)

    /// admonition | TODO
    Trajectory figure: $A$ vs step for every anchored run (rows = rungs,
    lines = seeds), with LR and anchor-weight schedules as background
    bands, and each run's running-max ratio reported. Expected: rise
    during warmup, hold through the plateau, no post-anneal sag below
    0.8× the running max. Contrary: the ex-2.9.3 signature — anchored
    early, then a collapse while the LR is still hot, or after the anneal
    begins. If collapses appear, note *when* relative to both schedules;
    that timing is the diagnostic that mattered in M1.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Secondary measurements

    /// admonition | TODO
    Leakage: off-axis redness probe R² at op1 (per layer, primary arm vs
    control), reported without a threshold. Low values mean the anchor
    *concentrated* redness onto $a$; values matching the control mean the
    anchor added an aligned copy while leaving the natural encoding in
    place. Either is informative for the intervention experiment to come —
    this number is effectively its power analysis.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory analyses

    Reserved for post-hoc work; anything here was conceived after seeing the
    data and is marked as such. Anticipated candidates, listed now for
    honesty: per-seed condensation variability, the answer-position column's
    behavior, alignment on `open`-pair lines, and whether labeled-line
    over-representation (the balanced batching) leaves a fingerprint on
    those lines' task loss.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discussion

    /// admonition | TODO
    Written after the analysis round, with the human. Questions to
    address: which of H1–H4 cleared, what the dose–response says about
    where the capacity cost begins, whether condensation or broadcast won
    (and what that means for sequence-level labeling in natural language),
    and what the leakage number implies for the intervention experiment's
    design (anti-subspace needed or not).
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    /// admonition | Open decisions for review (pre-freeze)
    Defaults chosen for drafting; flag disagreement before the freeze.
    (1) H2 scores the layer-*mean* alignment; min and max are reported as
    diagnostics only. (2) Labels redrawn per visit (as M1), via balanced
    batching at 4 affinity-sampled lines per batch of 64.
    (3) Weight ladder $\{0, 0.03, 0.1, 0.3\}$ with $w{=}0.1$ as the
    prespecified scoring rung. (4) Three seeds. (5) H2's grading groups
    reuse M1's `SIM3` red/other split. (6) Anneal window epochs 90–100 to
    a 10 % floor. (7) The anchor term averages over *all* non-padding
    positions including the answer and newline.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
