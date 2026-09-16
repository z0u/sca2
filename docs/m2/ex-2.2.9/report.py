import marimo

__generated_with = "0.24.0"
app = marimo.App(
    app_title="Ex 2.2.9: the grammar handover",
    css_file="../../report.css",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import math

    import marimo as mo
    import numpy as np

    # The design constants come from `experiment.py` beside this notebook (Marimo puts the
    # notebook directory on sys.path). The prose quotes the frozen gates, and that module
    # carries the same numbers with each gate's wording in its docstring.
    import experiment as ex
    from mini.reports import report_bundle, use_publisher
    from sca.data.colors import redness
    from sca.data.ops import (
        TOP,
        answer_dist,
        colors,
        commutativity,
        dose,
        is_on_grid,
        lines,
        on_grid,
        probe_lines,
        probe_partners,
        relevance,
        unordered_pairs,
    )

    use_publisher(report_bundle(__file__))

    # A cell renders its last expression, and a trailing docstring is one.
    None


@app.function(hide_code=True)
def zeroed(a, b):
    """The line with the redder operand's R set to zero: ex-2.2.4's *to-zero* rule."""
    if redness(a) >= redness(b):
        return (0, a[1], a[2]), b
    return a, (0, b[1], b[2])


@app.function(hide_code=True)
def to_zero_move(op, a, b) -> float:
    """How far the true answer moves in the unit cube when the red operand loses its R."""
    return float(np.linalg.norm(np.subtract(op(*zeroed(a, b)), op(a, b))) / TOP)


@app.function(hide_code=True)
def removal_counts() -> dict[str, tuple[int, int, int]]:
    """Per op: (red probe lines, removal lines, non-red lines with a red answer), on the op's probe set as
    ex-2.2.3 draws it.
    """
    out = {}
    for op in ex.TABLE:
        pl = probe_lines(op, ex.N_PROBE, ex.PROBE_SEED)
        red = [(ln.lhs, ln.rhs) for ln in pl if dose(ln.lhs, ln.rhs) >= ex.RED_DOSE]
        far = sum(to_zero_move(op, a, b) >= ex.FAR_MOVE for a, b in red)
        red_answer = sum(dose(ln.lhs, ln.rhs) <= ex.NONRED_DOSE and redness(ln.result) >= ex.RED_DOSE for ln in pl)
        out[op.name] = (len(red), far, red_answer)
    return out


@app.function(hide_code=True)
def slot_counts() -> dict[str, dict[str, tuple[int, int]]]:
    """For the order-sensitive ops: (red lines, removal lines) with the red operand at op1 and at op2, on the
    both-slot probe set.
    """
    cs = colors()
    partners = probe_partners(ex.N_PROBE, ex.PROBE_SEED)[1]
    out = {}
    for name in ex.ORDER_SENSITIVE:
        op = next(o for o in ex.TABLE if o.name == name)
        out[name] = {}
        for slot in ("op1", "op2"):
            ls = [(c, b) if slot == "op1" else (b, c) for c, ps in zip(cs, partners, strict=True) for b in ps]
            red = [(a, b) for a, b in ls if dose(a, b) >= ex.RED_DOSE and (redness(a) >= redness(b)) == (slot == "op1")]
            far = sum(to_zero_move(op, a, b) >= ex.FAR_MOVE for a, b in red)
            out[name][slot] = (len(red), far)
    return out


@app.function(hide_code=True)
def eps_check() -> tuple[int, int]:
    """Removal lines for a red op2 under `sat-hsv`, as the per-slot table counts them, with the red operand's
    R set to zero and to one grid level: the epsilon the review asked about.
    """
    op = next(o for o in ex.TABLE if o.name == "sat-hsv")
    partners = probe_partners(ex.N_PROBE, ex.PROBE_SEED)[1]
    ls = [(b, c) for c, ps in zip(colors(), partners, strict=True) for b in ps]
    red = [(a, b) for a, b in ls if dose(a, b) >= ex.RED_DOSE and redness(a) < redness(b)]

    def far(eps: int) -> int:
        return sum(np.linalg.norm(np.subtract(op(a, (eps, b[1], b[2])), op(a, b))) / TOP >= ex.FAR_MOVE for a, b in red)

    return far(0), far(1)


@app.function(hide_code=True)
def refop_facts() -> tuple[float, float, float]:
    """`hsvmix` as a reference op: its on-grid share of unordered pairs, on-grid partners per color, and the
    expected-exact-match ceiling of its probe set (the chance that two draws from the true answer agree).
    """
    op = next(o for o in ex.TABLE if o.name == ex.SECONDARY_OP)
    cs = colors()
    partners = np.mean([sum(is_on_grid(op, c, b) for b in cs) for c in cs])
    pl = probe_lines(op, ex.N_PROBE, ex.PROBE_SEED)
    ceiling = np.mean([sum(p * p for p in answer_dist(op, ln.lhs, ln.rhs).values()) for ln in pl])
    return on_grid(op), float(partners), float(ceiling)


@app.function(hide_code=True)
def refop_md() -> str:
    counts = removal_counts()
    rows = []
    for name in (ex.PRIMARY_OP, ex.SECONDARY_OP):
        op = next(o for o in ex.TABLE if o.name == name)
        pl = probe_lines(op, ex.N_PROBE, ex.PROBE_SEED)
        exact = np.mean([is_on_grid(op, ln.lhs, ln.rhs) for ln in pl])
        ceiling = np.mean([sum(p * p for p in answer_dist(op, ln.lhs, ln.rhs).values()) for ln in pl])
        red = [(ln.lhs, ln.rhs) for ln in pl if dose(ln.lhs, ln.rhs) >= ex.RED_DOSE]
        move = np.mean([to_zero_move(op, a, b) for a, b in red])
        alone = relevance(op, ex.TABLE, lines()).get(0, 0)
        n_red, far, ra = counts[name]
        rows.append(f"| `{name}` | {exact:.0%} | {ceiling:.2f} | {far / n_red:.0%} | {move:.2f} | {ra} | {alone:.0%} |")
    head = (
        "| op | probe lines on the grid | expected exact match ceiling | red lines that are removal lines | "
        "mean to-zero move | red-answer lines | alone |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    return head + "\n".join(rows)


@app.function(hide_code=True)
def coverage(lines_per_op: float) -> tuple[float, float]:
    """The share of an op's trainable lines a model sees at least once: over ordered lines, and over unordered
    pairs. Lines are drawn i.i.d. with held-out pairs rejected, so this is 1 − exp(−draws / pool).
    """
    pool_ordered = len(lines()) * (1 - ex.HOLDOUT_FRAC)
    pool_pairs = len(unordered_pairs()) * (1 - ex.HOLDOUT_FRAC)
    return 1 - math.exp(-lines_per_op / pool_ordered), 1 - math.exp(-lines_per_op / pool_pairs)


@app.function(hide_code=True)
def table_md() -> str:
    rows = []
    for op in ex.TABLE:
        group = "kept" if op.name in ex.KEPT else ("added" if op.name in ex.ADDED else "order-sensitive")
        comm = "yes" if commutativity(op) > 0.99 else f"{commutativity(op):.0%}"
        rows.append(f"| `{op.name}` | {op.rule.replace('|', '\\|')} | {comm} | {group} |")
    head = "| op | rule (0..15 scale, snapped to the grid) | commutative | in A+ as |\n| --- | --- | --- | --- |\n"
    return head + "\n".join(rows)


@app.function(hide_code=True)
def conds_md() -> str:
    rows = []
    for c in ex.CONDS:
        anchor = "none" if c.lam == 0 else f"λ_a = {c.lam:g}, τ = {ex.TAU:g}"
        labeller = "—" if c.lam == 0 else ("whole line" if c.keying == "line" else "either slot, prompt span")
        readout = "tied" if c.tie else "untied"
        rows.append(
            f"| **{c.name}**, {c.role}: {c.title} | {anchor} | {labeller} | {readout} | {c.lines_per_op:,} | {c.epochs} ({c.steps:,}) | {c.seeds} |"
        )
    head = (
        "| condition | anchor | labeller | readout | lines per op | epochs (steps) | seeds |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: |\n"
    )
    return head + "\n".join(rows)


@app.function(hide_code=True)
def relevance_md() -> str:
    """Op-relevance for the two reference ops under table A+, counted over ordered pairs."""
    ordered = lines()
    rows = []
    for name in (ex.PRIMARY_OP, ex.SECONDARY_OP):
        op = next(o for o in ex.TABLE if o.name == name)
        rel = relevance(op, ex.TABLE, ordered)
        shares = [rel.get(k, 0) for k in range(3)] + [sum(v for k, v in rel.items() if k >= 3)]
        cells = " | ".join(f"{v:.0%}" for v in shares)
        rows.append(f"| `{name}` | {cells} |")
    head = "| anchored op | alone | 1 other agrees | 2 | 3+ |\n| --- | ---: | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


@app.function(hide_code=True)
def removal_md() -> str:
    counts = removal_counts()
    rows = [f"| `{name}` | {red:,} | {far:,} ({far / red:.0%}) | {ra:,} |" for name, (red, far, ra) in counts.items()]
    head = "| op | red probe lines | removal lines | non-red lines with a red answer |\n| --- | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


@app.function(hide_code=True)
def slot_md() -> str:
    rows = []
    for name, slots in slot_counts().items():
        cells = " | ".join(f"{red:,} | {far:,} ({far / red:.0%})" for red, far in slots.values())
        rows.append(f"| `{name}` | {cells} |")
    head = "| op | red at op1 | removal | red at op2 | removal |\n| --- | ---: | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <mark class="draft">DRAFT</mark>

    # Ex 2.2.9: the grammar handover

    /// tip |
    <!-- tl;dr -->
    The last four experiments each revealed something we should change about the setup in which we anchor *red*: more operations, answers that are drawn stochastically rather than rounded, labels according to the whole line, and a readout table kept separate from the embedding table. Each was tried on its own and looked fine.

    This experiment enables all four at once. Does *red* still land where we put it? Can it still be removed cleanly? And do the label and the readout, which were inconclusive, earn their place? Two reference conditions switch one of those back each, so we can tell which one did what.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    # REVIEW: the Findings line quoted the whole decision rule, which the Decision section states
    # again a few screens later; it is now a link, so the rule has one home. Verify: the rule is
    # rendered from `ex.DECISION` under "Decision".
    mo.md(r"""
    ## Findings

    - [Does the model still learn the task? (H1)](#does-the-model-still-learn-the-task-h1) — _to come_
    - [Does *red* still land where we put it? (H2)](#does-red-still-land-where-we-put-it-h2) — _to come_
    - [Can we still take *red* out cleanly? (H3)](#can-we-still-take-red-out-cleanly-h3) — _to come_
    - [Does the separate readout keep the axis off the syntax tokens? (H4)](#does-the-separate-readout-keep-the-axis-off-the-syntax-tokens-h4) — _to come_
    - [What does the whole-line label cost? (H5)](#what-does-the-whole-line-label-cost-h5) — _to come_
    - [Whether the handover is adopted](#decision) — _to come_
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | How to read this draft
    This is a preregistration. The conditions, the gates, and the decision rule below are written before any run, and will be frozen at a named commit. Each hypothesis section opens with the background, then gives the prediction we will be scored on. Results go into each section in place once they exist. Anything we think of after seeing the data goes under [Exploratory](#exploratory), marked as post hoc. Every count in the method is computed from `experiment.py` at render time.
    ///

    ## Why this experiment

    Ex-2.2.3 anchored *red* on a grammar of six operations and then tried to remove it. Removal was partial on four ops, and the ops themselves were confounding. On `lighten`, say, most lines with a red operand have an answer that stays the same when the operand is a little less red, so a model that has lost *red* can still answer them.

    Our investigations suggest:

    1. **A wider table of operations** ([ex-2.2.4](../ex-2.2.4/report.py)). Drop `add`, add three ops whose answers spread through the color cube, and add three more that take one attribute from one operand and the rest from the other. Those last three are the first ops where the order of the operands matters.
    2. **Stochastic answers instead of rounded ones** ([ex-2.2.5](../ex-2.2.5/report.py)). When an answer lands between two grid colors, the corpus picks one of them at random, rather than rounding deterministically. The model then learns a spread of possible answers rather than one biased token.
    3. **A label that covers the whole line** ([ex-2.2.6](../ex-2.2.6/report.py)). Previously, only the two operands could draw a label, and the pull covered the prompt. Now the anchor is told "this line is about red", and it pulls wherever in the line it finds the most red, answer included. That is the shape a document-level label will have in M3. The first pilot found it costs nothing; a later one saw a few seeds lose selectivity under it, so it goes in with a check.
    4. **A separate, untied readout table** ([ex-2.2.7](../ex-2.2.7/report.py)). The output layer of the model was sharing a table with the input embeddings, and that put part of the *red* axis onto the tokens `=` and `mix`. Giving the output its own table keeps the axis off those tokens.

    In [ex-2.2.8](../ex-2.2.8/report.py), we found that the removal operator to score is the plain projection, which takes the axis out everywhere, with the operand-only edit beside it as the selective reference.

    Here the four are tried together, at the same number of seeds as the reference, to inform the grammar and recipe the anchored-op experiments will use. We call that the *grammar of record*: the one setup every later D2.2 experiment is built on. It is a handover because the grammar of record changes hands here: until this runs it is the one from ex-2.2.3; after it, if the gates hold, it is this one.
    """)
    return


@app.cell(hide_code=True)
def _():
    seen_lines, seen_pairs = coverage(ex.HANDOVER.lines_per_op)
    seen_lines_six, seen_pairs_six = coverage(ex.WIDE.lines_per_op)
    mo.md(rf"""
    ## Conditions

    Every condition trains fresh models on the new grammar: table A+, {ex.N_LINES:,} lines drawn uniformly over the {ex.N_OPS} ops, answers drawn stochastically, one fifth of the pairs of each op held out. That is about {ex.HANDOVER.lines_per_op:,} lines per op, drawn with replacement from the op's {len(lines()) * (1 - ex.HOLDOUT_FRAC):,.0f} trainable lines, so a model sees about {seen_lines:.0%} of them at least once ({seen_pairs:.0%} of the unordered pairs), against {seen_lines_six:.0%} ({seen_pairs_six:.0%}) at six ops. The recipe is the point adopted in ex-2.2.3 (`{ex.EX223_REFERENCE}`), unchanged: λ_a = {ex.LAM:g}, annealed over the last tenth of training as ex-2.1.10 did, τ = {ex.TAU:g}, the anti-subspace weight from {ex.ANTI_PEAK_RATIO:g}× the anchor weight down to 0.3× by {ex.ANTI_ANNEAL_END_FRAC:.0%} of training, {ex.EPOCHS} epochs ({ex.HANDOVER.steps:,} steps). *Red* is anchored to e₁ at every slice, the embedding included.

    {conds_md()}

    **`handover`** has everything enabled. It is the one candidate for the grammar of record, at the same twenty seeds as the reference, and the gates are read on it alone.

    **`handover-slot`** switches the label back to the one from ex-2.2.3: only the two operands can draw a label, and the pull covers the prompt. Beside `handover` it is the selectivity check ex-2.2.7 asked for, at twenty seeds so that the comparison has the resolution of the gates. It is not a fallback: that labeller needs to know which tokens are the operands, and M3 will not have that.

    **`handover-tied`** switches the readout back to the shared table. Beside `handover` it shows what untying does on this grammar. Nine seeds, as the pilot had.

    **`control`** has no anchor at all. It sets the task bar for H1 and the calibration bar for the drawn answers.

    **`handover-wide`** is exploratory. Eleven ops share the same {ex.N_LINES:,} lines, so each op gets about half of what it had at six ops, and the model sees a smaller share of each op's lines. E4 in ex-2.2.3 found the operand cube less decodable at six ops than at three, and could not tell whether the op count or the lines per op was responsible. This condition holds lines per op at the six-op count ({ex.WIDE.lines_per_op:,}), using a corpus eleven sixths the size for fewer epochs so that the step count matches.

    The reference is not retrained. It is `{ex.EX223_REFERENCE}` from ex-2.2.3, at twenty seeds on the six-op grammar, and its stored statistics are printed beside every placement read.

    ### The removal operators

    We score every anchored checkpoint through the eval contract in [`sca.intervention`](/src/sca/intervention.py), on each op's probe lines.

    - **`projection`**: the axis projected out at every slice and every position, at full strength. This is the gated operator, as ex-2.2.8 proposed.
    - **`operands`**: the same edit at the two operand positions only. It is the selective reference, since it never touches the syntax tokens and so cannot cost anything there.
    - **`shaped-a0.4-p0`**: a thresholded projection that leaves any state below alignment {ex.SHAPED["a"]:g} alone. Ex-2.2.8 found it removes less than the projection at no cost on the six-op grammar; what we learn here is whether that smaller removal clears the removal gate on a table built to need *red*. Scoring it is one more pass over the same checkpoints.
    """)
    return


@app.cell(hide_code=True)
def _():
    # A definition list, written as HTML: Marimo's Markdown has no syntax for one, and the
    # markup inside the block is HTML too, since Markdown is not processed there.
    low = min(far / red for red, far, _ in removal_counts().values())
    mo.md(rf"""
    ## Glossary

    <dl>
    <dt>Red line, non-red line</dt>
    <dd>A line is <em>red</em> when its redder operand has redness at least 0.8, and <em>non-red</em> when neither operand is above 0.2. Redness is r·(1 − g/2 − b/2) on the unit scale. The redness of the redder operand is the line's <em>dose</em>.</dd>
    <dt>Removal lines</dt>
    <dd>The red lines on which the true answer would move a long way if the red operand had no red in it (its R channel set to zero). These are the lines where losing <em>red</em> has to show; on the others the op does not need it. On <code>mix</code> every red line is a removal line; on the other ops at least {low:.0%} of them are, and the count per op is in the method.</dd>
    <dt>Red-answer lines</dt>
    <dd>Non-red lines whose true answer is red (white minus cyan, under <code>difference</code>). Projecting the axis out at <code>=</code> takes <em>red</em> from the state that has to produce the answer, so a miss there is removal on the output side. They stay in the non-red deficit and are also counted on their own, and the deficit is reported with and without them. <code>mix</code> has none, so the gated read is the same either way.</dd>
    <dt>Expected exact match</dt>
    <dd>With drawn answers, the correct answer to a line is spread over two or more colors. Expected exact match is the chance that an answer drawn from the model agrees with one drawn from the true distribution. On the ops that round it cannot reach 1.</dd>
    <dt>m_line</dt>
    <dd>How far <em>red</em> is pushed onto the axis: at the reddest position in the span, the label-weighted mean alignment minus the unweighted mean, averaged over slices. Ex-2.1.10's margin, read on the <code>mix</code> lines.</dd>
    <dt>ᾱ (containment)</dt>
    <dd>The mean alignment with the axis over all 216 colors at op1. High when colors that are not red drift onto the axis at that position.</dd>
    <dt>Lead</dt>
    <dd>On the red lines, the largest share of the pull that any one position receives at the embedding slice. High when the pull is concentrated on one token rather than spread over the span.</dd>
    <dt>Contrast</dt>
    <dd>How much more the pull lands on op2 when op2 is the red operand than when it is not, averaged over the post-attention slices. It says the pull follows the red operand rather than a fixed position.</dd>
    <dt>Grading r²</dt>
    <dd>How well alignment with the axis tracks redness across colors, fit against the sim^1.5 target of ex-2.1.11.</dd>
    <dt>Retention</dt>
    <dd>Whether the placement holds to the end of training, after the anchor weight's anneal: the final m_line as a share of the run's peak.</dd>
    <dt>Latch</dt>
    <dd>A run in which the non-red group puts more than half of its softmin weight on op1: the pull has found a position rather than a concept.</dd>
    <dt>Deficit</dt>
    <dd>How much expected exact match a model loses on the non-red lines when the axis is projected out. This is a measure of the selectivity; a clean removal costs nothing here.</dd>
    <dt>Band</dt>
    <dd>Our measurement precision. Two seed means are told apart only when they differ by more than 2σ√(1/n_a + 1/n_b), with σ the per-run spread and n the seed counts. Smaller differences are reported as unresolved.</dd>
    </dl>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Does the model still learn the task? (H1)

    **Background.** Eleven ops in the same number of lines, with drawn answers, is a harder corpus than six ops with rounded ones. Before we read anything about the anchor we need to know that the anchored model learns the task as well as an un-anchored model does on this corpus.

    **Prediction.** For each of the {ex.N_OPS} ops, the `handover` seed-mean expected exact match on held-out lines is within {ex.TASK_GATE:g} of the control. Partial: every op within {ex.TASK_PARTIAL:g}. Contrary: more than {ex.TASK_PARTIAL:g} below the control on some op. A comparison that misses by less than a band is reported as unresolved rather than as a miss; the band on this read is computed from the runs and printed beside the gate. The reference conditions are read on the same table without a gate, and a reference that does not learn the task has its later comparisons read with that caveat.

    We expect this to hold: ex-2.2.5 saw no task cost from the drawn answers, ex-2.2.7 none from the untied readout. Whether the control itself learns the grammar is checked before the freeze, on one seed, rather than gated here. The number to watch is the order-sensitive subset, since an op that reads operand order asks the model for something the six-op grammar never did.

    /// admonition | TODO
        type: warning
    Results to come. The section will chart expected exact match per condition and op against the control, with the numbers in a table, and the calibration of the answer mass against the true distribution (P(mode) and KL, as ex-2.2.5 read them) beside it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Does *red* still land where we put it? (H2)

    **Background.** The recipe was tuned on the six-op grammar, but here the corpus, the labeller, and the readout all change. We read the same placement statistics on the same `mix` lines, so the numbers are comparable to ex-2.2.3. The question is whether they are still inside the gates set there.

    **Prediction.** On the `mix` lines, for `handover`, under its own labeller (as ex-2.2.6 read it):

    - *Margin:* seed-mean m_line at least {ex.MARGIN_RATIO:.0%} of the reference's {ex.REF_M_LINE:.4f}, so {ex.MARGIN_RATIO * ex.REF_M_LINE:.3f}; partial from {ex.MARGIN_PARTIAL:.0%}.
    - *Grading:* grading r² at least {ex.GRADE_R2_RATIO:.0%} of the reference's {ex.REF_R2_SIM:.3f}, so {ex.GRADE_R2_RATIO * ex.REF_R2_SIM:.3f}.
    - *Concentration, attribution, retention, latch:* lead at the embedding at least {ex.LEAD_GATE:g}; contrast at least {ex.CONTRAST_GATE:g} (partial from {ex.CONTRAST_PARTIAL:g}); every run that reaches m_line {ex.RETENTION_FLOOR:g} ends at {ex.RETENTION_GATE:g} of its peak; no run latched, which the reference held at twenty seeds.

    **Containment is a prediction here, not a gate.** Every experiment since ex-2.1.8 has gated ᾱ at op1 at {ex.MEAN_ALIGN_REF:g}. At nine seeds, ex-2.2.7 read 0.13 on its untied condition and 0.23 on its untied whole-line condition, against 0.08 on the reference, so we expect `handover` above {ex.MEAN_ALIGN_REF:g}, and `handover-slot` nearer to it. We do not know why untying raises it. What we do know is that the pull still lands on the red operand in those runs (lead, contrast, and latch all sat at the reference's values), so this is other colors drifting a little onto the axis at op1 rather than the pull finding a position. What that drift costs, if anything, is what H3's selectivity read measures. So the prediction is the pair: ᾱ above {ex.MEAN_ALIGN_REF:g} on `handover`, and a non-red deficit inside H3's gate all the same. If both hold, the drift is recorded for the anchored-op prereg to watch; if the deficit fails too, containment is the first place to look for why.

    `handover-tied` is read on the same statistics, so that we can attribute the containment read. If the tied condition sits with the reference (more contained) and both untied conditions sit higher, the untied readout is what moved it.

    /// admonition | TODO
        type: warning
    Results to come. The section will chart each placement statistic per condition with the reference beside, tabulate them (condition × statistic), and show the softmin profile over roles under each labeller.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Can we still take *red* out cleanly? (H3)

    **Background.** Take the *red* axis (e₁) out of every state. On the lines that need *red*, does the model fail? On the lines that never had any, does it still answer? The first is removal, the second selectivity. Ex-2.2.3 could only show removal on two of six ops, because the other four mostly did not need *red*. Table A+ was chosen so that every op has lines that do, and the removal read is scored on those lines only.

    **Prediction.** Under `projection`, for `handover`:

    - *Removal:* on the removal lines of every op, the model keeps at most {ex.RED_KEPT_GATE:.0%} of its clean expected exact match, seed mean. This is the red-accuracy gate of ex-2.2.3, read as a ratio, because with drawn answers the clean value sits below 1 on the ops that round.
    - *Selectivity:* the seed-mean deficit on the non-red `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}; partial to {ex.NONRED_DEFICIT_PARTIAL:g}. The partial band is a reporting level, as it was in ex-2.2.3: the decision rule asks for the gate. The read stays on `mix` so that it means what it meant at the reference; `hsvmix` and every other op are reported beside it; what a switch to `hsvmix` would change is set out in the method ([The reference op](#the-reference-op)). On the reference, ex-2.2.8 read 0.040 on `mix`, which is inside the gate by less than a band. With the readout untied we expect the deficit to fall toward the `operands` row, which read 0.012. Bands on the deficit use the per-run spread ex-2.2.8 measured under `projection` at the reference's twenty seeds, per op ({ex.DEFICIT_NOISE}), frozen so that the candidate's own spread does not move its verdict. Non-red lines whose true answer is red are in the deficit and also counted on their own (see the glossary); `mix` has none.

    Two further reads have an expected direction and no gate.

    *How far the answer moves.* On the removal lines, we measure the distance in the unit cube between the answer the model decodes under `projection` and the true answer. Beside it we put the distance the true answer itself moves when the red operand loses its red. Ranking the ops by each distance should give the same order, with `value-hsv`, `hue-hsv`, and `sat-hsv` at the top of both.

    *The operand-only edit.* On the ops whose answer is computed at `=` from both operands together, `operands` should remove less than `projection`, since the `=` state keeps its axis component. Elsewhere the two should remove the same amount.

    /// admonition | TODO
        type: warning
    Results to come. The section will chart removal and selectivity per op and condition under the three operators, with the table beside, and the answer-distance read against the to-zero distance.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Does the separate readout keep the axis off the syntax tokens? (H4)

    **Background.** In ex-2.2.7 the readout row for `=` picked up a component along the *red* axis: after a red operand, whose state sits on the axis, that is a cheap way to raise the `=` logit. With a shared table that row is also the input embedding of `=`, so the `=` token entered the residual stream carrying some *red*, and projecting the axis out at that position took away something the model was using. Giving the output its own table moved the component onto the readout row and left the input embedding mostly clean.

    That was on the six-op grammar at nine seeds. Does it carry to eleven ops, and does it give the cleaner full-line removal?

    **Prediction.** On `handover` against `handover-tied`, same labeller, twenty seeds against nine (the band formula takes both counts, so it is wider than between two twenty-seed conditions):

    - The axis component on the syntax embeddings (`=`, the op words, and `⏎`), read from the embedding-component table of ex-2.2.7, is lower on `handover` than on `handover-tied` by more than a band ({ex.COMPONENT_NOISE}, since ex-2.2.7 published seed means and ranges rather than a per-run spread; the σ is reported beside the comparison). On `=`, `handover` sits within a band of the hard-zeroed ceiling from ex-2.2.7 (`{ex.EX227_CEILING}`, where the component is zero by construction). The component appears on the readout table of `handover` instead.
    - The non-red `mix` deficit under `projection` is lower on `handover` than on `handover-tied`, by more than a band.

    Neither prediction is a gate. Ex-2.2.7 already took the readout decision, and this section either confirms it or reports that it did not carry. If the second prediction fails while the first holds, then on this grammar the syntax embeddings were not where the cost came from.

    /// admonition | TODO
        type: warning
    Results to come. The section will chart the embedding component per condition and the deficit under `projection` for the two conditions, with the tables beside.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## What does the whole-line label cost? (H5)

    **Background.** A label that covers the whole line is the shape we need for natural language (M3). Ex-2.2.6 found it costs nothing at three seeds. Ex-2.2.7 then ran nine seeds of its untied whole-line condition and saw a few of them lose a lot of non-red lines under the projection, where the operand-only labeller loses almost none. Here it is read at twenty seeds against twenty.

    **Prediction.** On `handover` against `handover-slot`, under `projection`:

    - The seed-mean non-red `mix` deficit differs by less than a band, and
    - the count of seeds whose deficit is above {ex.TAIL:g} (the level at which ex-2.2.7 read its tail) is no more than two higher on `handover` than on `handover-slot`.

    The whole-line label changes two things at once: which positions the pull can land on, and which lines get a label at all, since the answer draws at its own redness rate. So a miss here says the label as a whole costs selectivity, but not which half of it did. Neither prediction is a gate, and `handover-slot` is not a fallback: we need the whole-line label, so a cost here is something to understand and fix, and the size of the gap to `handover-slot` says how much there is to fix. We are not sure which way this goes: the tail in ex-2.2.7 was three seeds out of nine, enough to expect it and too few to be sure.

    /// admonition | TODO
        type: warning
    Results to come. The section will chart the per-seed deficit under `projection` for the two conditions, side by side, with the `operands` row beside it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## Decision

    H1 to H3 carry gates because one decision hangs on them: whether the anchored-op experiments run on this grammar. H4 and H5 are predictions, written down so that the result can be read against them, and what we do about a miss there is decided after reading it.

    {ex.DECISION}

    /// admonition | TODO
        type: warning
    To come, once H1 to H5 are read.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory

    Read without gates, and not part of the decision.

    - **Lines per op (`handover-wide`).** Ex-2.2.3's E4 probed how well the two operand colors can be decoded from the residual stream, and found them less decodable at six ops than at three. Was that because there were more ops, or because each op had fewer lines? `handover-wide` gives each op as many lines as it had at six ops. If the operands decode as well there as they did at six ops, and less well on `handover`, then lines per op was the cause. If the two conditions sit together, it was the op count.
    - **The order-sensitive subset, by slot.** For `hue-hsv`, `sat-hsv`, and `value-hsv` the probe set walks every color in both slots, so every read in H2 and H3 can be split by whether the red operand is op1 or op2. The labeller pools both operands the same way, so we expect the same placement in both slots. Removal should show in both slots too, at the rates in the method's per-slot table: a red op1 under `value-hsv` supplies the hue and the saturation, and losing those moves the answer as far as losing the value does.
    - **Calibration under the anchor.** P(mode) and KL from the true distribution on the rounded held-out lines, for every anchored condition against the control, as ex-2.2.5 read them. The anchor should not change them.
    - **Red-answer lines.** On the ops that have them, the model's answer under `projection` on the non-red lines whose true answer is red. If the model can no longer produce red there, that is removal on the output side, and it says the readout row for red carries the axis as the embedding does.
    - **The `shaped-a0.4-p0` operator.** Reported on every op beside the two gated operators: does its smaller removal still clear the removal gate on the new table?

    ## Discussion

    /// admonition | TODO
        type: warning
    About 200 words, after the results: what the handover settled, what it moved, and what the anchored-op prereg inherits.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    hsv_on_grid, hsv_partners, hsv_ceiling = refop_facts()
    sat_op2, eps_sat = eps_check()
    mo.md(rf"""
    ## Method

    ### The table

    {table_md()}

    Every rule is computed on the 0..15 scale and snapped to the grid. Where it lands between levels, the corpus draws the answer (*stochastic rounding*, ex-2.2.5). The three order-sensitive ops take one HSV attribute from op2 and the other two from op1, so each agrees with its own reverse on under 2% of pairs. Their reads are reported as a subset.

    ### Op-relevance under A+

    For a line that uses the anchored op, how many other ops in the table give the same answer. This is the stimulus side of the per-line predictions in the anchored-op experiments, counted over ordered pairs. Widening the table makes `mix` less distinctive (ex-2.2.4 read 95% alone at six ops), which is good because the later predictions need more than one level to read.

    {relevance_md()}

    ### The removal lines

    Per op, on its probe set: the red lines (dose ≥ {ex.RED_DOSE:g}, where dose is the redness of the redder operand) and, of those, the lines on which setting the R channel of the red operand to zero moves the true answer by at least {ex.FAR_MOVE:g} in the unit cube. The removal read in H3 is scored on the second column. The last column counts the non-red lines (dose ≤ {ex.NONRED_DOSE:g}) whose true answer is red; those are in the selectivity read and also counted on their own.

    {removal_md()}

    Replacing the red operand with a mid-gray instead of zeroing its R channel was considered. On `mix` it counts far fewer lines (a gray partner moves the mix less than a dark one does), and elsewhere it changes the counts without changing which ops need *red*, so the to-zero rule stays.

    For the order-sensitive ops, the both-slot probe set splits by where the red operand sits. The one place the rule under-counts is a red op2 under `hue-hsv` or `sat-hsv`. Zeroing the R of a pure red gives black, which HSV reads as hue 0 (red) at no saturation: under `hue-hsv` the answer keeps its hue, and under `sat-hsv` op1 only loses its saturation, which moves it far only when it was saturated. Setting R to one grid level instead of zero does not help. The operand is then a very dark red at full saturation, so `hue-hsv` moves as little as before and `sat-hsv` stops moving at all ({eps_sat} lines counted against {sat_op2} at zero). There is no hue a color should have once its red is gone, so these lines are reported and not gated.

    {slot_md()}

    ### The reference op

    Every gated read is on `mix`, and `hsvmix` is reported beside it as the op a later experiment might promote. This is what the switch would change, on each op's probe set as ex-2.2.3 draws it.

    {refop_md()}

    `hsvmix` is the better op for the removal read: nearly every red line is a removal line, and the answer moves further when the red operand loses its red, since a hue mean loses the red hue outright where a channel mean halves it. It is also a little less distinctive, which the anchored-op predictions want. Its placement reads (H2) would look much like `mix`'s, since the label reads the colors in the line rather than the op word.

    What it costs is the probe set. `mix` has {ex.N_PROBE} partners per color on which its answer lands on the grid without rounding, its probe lines are those, and so a clean model can match every answer and a deficit is a count of lines lost. `hsvmix` lands on the grid on {hsv_on_grid:.0%} of pairs, about {hsv_partners:.0f} partners per color, so its probe set is the shared random draw, and the best any model can do on it is an expected exact match of about {hsv_ceiling:.2f}. On that base the removal gate reads on a clean value near {hsv_ceiling:.1f} rather than 1, and the {ex.NONRED_DEFICIT_GATE:g} selectivity gate is an eighth of the clean value rather than a twentieth; both would need re-setting, and neither could be read against the reference's numbers. Had the program used `hsvmix` from the start, every exact-match read before ex-2.2.5 would have been on answers that round on {1 - hsv_on_grid:.0%} of lines, which is the bias that experiment found and fixed; D2.1 and ex-2.2.3 could read removal and selectivity as counts because `mix`'s probe set never rounds. The switch is open now that answers are drawn, at the price of a noisier deficit, and it is a later experiment's call, made with this experiment's `hsvmix` rows in hand.

    ### The corpus, the probes, the labellers

    {ex.N_LINES:,} lines at seed {ex.CORPUS_SEED}, ops drawn uniformly, with {ex.HOLDOUT_FRAC:.0%} of the distinct unordered pairs of each op held out. A held-out pair is out in both orders, for every op, so the held-out share of an op's lines is the same {ex.HOLDOUT_FRAC:.0%} whether or not the op reads operand order. Probe sets follow ex-2.2.3: `mix` on its 27 on-grid partners per color, and every other op on 27 partners per color drawn once at seed {ex.PROBE_SEED} and shared across ops. The order-sensitive subset also walks every color as op2.

    The two labellers are *either slot, prompt span* (each operand draws at redness⁸ × {ex.PER_SLOT_RATE:g}, and the pull covers op1, op, op2, `=`) and *whole line* (the answer draws at its redness rate too, and the pull covers all six positions). The anchor term is the per-line mellowmax over the pulled span with a conserved per-line budget, so a wider span changes where the pull can land but not how strong it is.

    ### Before the freeze

    One seed of `control` trains first, and its expected exact match per op is recorded here, so that the gate in H1 is read against a control that learned the grammar. The bar: on every kept and added op, expected exact match within {ex.CALIBRATION_FLOOR:g} of the ceiling the drawn answers allow on that op (ex-2.2.5 saw the six-op control within 0.04 on the ops that round). A lower value on the order-sensitive subset is recorded and does not stop the run, since it says something about the grammar rather than about the anchor; a miss on a commutative op does stop it, and sends the corpus size and epoch count back for a look. Two pieces of code land with the DAG: the holdout draw, now keyed on the position of the op in this table rather than in the table of ex-2.2.3 (`sca.data.ops.holdout`), and a probe draw that walks both slots for the order-sensitive subset (`probe_partners`). Neither changes a number in the design.

    /// admonition | TODO
        type: warning
    The calibration read, and the freeze commit, to be filled in.
    ///

    ### Budget

    {ex.N_RUNS} runs of {ex.HANDOVER.steps:,} steps at d64-L4, plus scoring under three operators on {ex.N_OPS} probe sets. That is about the size of the run in ex-2.2.3.
    """)
    return


if __name__ == "__main__":
    app.run()
