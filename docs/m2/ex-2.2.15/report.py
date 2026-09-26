# title: Ex 2.2.15: lines cut short by the training window

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). During preregistration that module is constants only.
import experiment as ex

POLICY_TEXT = {
    "all": "every labelled line, whatever is visible (today's behaviour)",
    "whole": "only lines wholly inside the window",
    "half": "only lines with more than half their tokens inside the window",
    "scaled": "every labelled line, its pull scaled by the share of it that is visible",
    "knowable": "only positions at or after a visible op word",
    "cut-only": "only lines the window cuts short",
}


def conditions_html() -> str:
    """One row per arm, plus the served control."""
    head = (
        "<tr><th>condition</th><th>which labelled lines the anchor pulls</th>"
        "<th class=num>window</th><th class=num>pull kept</th><th class=num>seeds</th></tr>"
    )
    rows = [
        f"<tr><td><code>{a.name}</code></td><td>{POLICY_TEXT[a.policy]}</td><td class=num>{a.block}</td>"
        f"<td class=num>{ex.pull_share(a.policy, a.block):.0%}</td><td class=num>{ex.SEEDS}</td></tr>"
        for a in ex.ARMS
    ]
    rows.append(
        f"<tr><td><code>{ex.CONTROL}</code></td><td>none: ex-2.2.11's un-anchored control, served from the store</td>"
        f"<td class=num>{ex.BLOCK}</td><td class=num>–</td><td class=num>{ex.CONTROL_SEEDS}</td></tr>"
    )
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def shares_html() -> str:
    """How often a line visit shows each run of roles, at both window sizes, computed from the sampler."""
    long, short = ex.visit_shares(ex.BLOCK), ex.visit_shares(ex.SHORT_BLOCK)
    head = (
        f"<tr><th>visible roles</th><th>what the anchor can see</th>"
        f"<th class=num>{ex.BLOCK}-token window</th><th class=num>{ex.SHORT_BLOCK}-token window</th></tr>"
    )

    def seen(first: int, last: int) -> str:
        if (first, last) == (0, ex.LINE_TOKENS - 1):
            return "the whole line"
        if first <= ex.OP_ROLE <= last:
            return "the op word, cut short"
        return "<strong>no op word</strong>"

    rows = [
        f"<tr><td>{' '.join(ex.ROLES[first : last + 1])}</td><td>{seen(first, last)}</td>"
        f"<td class=num>{long[(first, last)]:.1%}</td><td class=num>{short[(first, last)]:.1%}</td></tr>"
        for first, last in long
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


LONG = ex.visit_shares(ex.BLOCK)
SHORT = ex.visit_shares(ex.SHORT_BLOCK)
CUT = 1 - LONG[(0, 5)]
CUT_SHORT = 1 - SHORT[(0, 5)]
BLIND = sum(v for (first, last), v in LONG.items() if not first <= ex.OP_ROLE <= last)
OP1_ONLY = LONG[(0, 0)]


rf"""
# Ex 2.2.15: lines cut short by the training window

/// tip |
<!-- tl;dr -->
The anchor pulls whole lines. But training shows the model windows onto the corpus, and each window cuts the lines at its edges. On a cut line, the anchor asks the visible part to carry the whole label, even when that part cannot see the op word. We try a few policies for which cut lines to pull. We do this on the current grammar, before the in-context grammar makes the problem larger.
///

This is a scouting run with a few predictions and one rule: it proposes the crop policy the [in-context grammar pilot](../d2.2/design.md#the-pilot) starts from.

## Findings

- [The first operand's lean comes from cut lines (H1)](#the-first-operands-lean-comes-from-cut-lines-h1) —
- [The anchor and the task hold under every policy (H2)](#the-anchor-and-the-task-hold-under-every-policy-h2) —
- [Fragments without their op word (H3)](#fragments-without-their-op-word-h3) —
- [Shorter windows, more cut lines (H4)](#shorter-windows-more-cut-lines-h4) —

[The rule for the pilot](#the-rule-for-the-pilot): —

## How to read this draft

The policies, the four predictions, and the rule were fixed before any run, at commit `TODO`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

Each result section opens with what we expect, then a placeholder for what we saw.

## Why this experiment

Picture the training corpus as one long tape of six-token lines. Each training step cuts a {ex.BLOCK}-token window out of that tape at a random place. Most lines in the window are whole, but the two edges of the window usually slice through a line at each end.

The anchor does not know about the edges. It asks each labelled line to align with the axis wherever that comes most easily in the visible part, and so asks that part to carry the whole label.

If all that is visible of a `{ex.ANCHORED_OP}` line is its first operand, the anchor asks a color token to say "this line is `{ex.ANCHORED_OP}`" before the op word has appeared. That token cannot know: the only way the model can give the anchor what it wants there is to lean every first operand a little toward the axis, on every line.

That is what [ex-2.2.14](../ex-2.2.14/report.py) saw, after the fact. At the last block, the first operand leaned toward e₁ at {ex.REFERENCE_OP1_LEAN:.2f} on the primary arm, against {ex.REFERENCE_OP1_LEAN_CONTROL:.2f} on the control. The arm that pulls only the op word had no lean. The [op1-lean reanalysis](../op1-lean/report.py) found the same route on the *red* runs, and proposed skipping the pull on cut lines as a test.

How common are cut lines? At {ex.BLOCK} tokens, {CUT:.0%} of line visits are cut short. On {BLIND:.1%} the op word is out of sight, and on {OP1_ONLY:.1%} only the first operand is visible. Those are small shares. But the term is normalized per labelled line, so each of those visits gets the pull of a full line.

This matters more after the [D2.2 pivot](../d2.2/pivot.md). There, a line is a context of solved examples about 20 tokens long, and the op is inferred from the examples. A window that cuts the start of a context can remove the examples that name the op. There are fewer lines per window, so the share of cut visits roughly doubles. And a cut context has lost evidence, not just one token. The current grammar is the cheap place to see what each policy does, with a clear sign (the lean) to watch.

## Glossary

<dl>
<dt>Cut line</dt>
<dd>A line visit that the training window shows only part of. A line cut at its start has lost its first tokens, so its states differ from those of the whole line. A line cut at its end shows a prefix. Under causal attention (each position sees only earlier positions), those prefix states are the same as in the whole line.</dd>
<dt>Crop policy</dt>
<dd>Which labelled lines the anchor pulls, given what the window shows of them. The label draws are the same under every policy; only the pull changes.</dd>
<dt>Pull kept</dt>
<dd>The share of today's total pull a policy keeps, averaged over every way a window shows a line. It is computed from the sampler, not measured.</dd>
<dt>The lean</dt>
<dd>The mean cosine with e₁ at the first operand, over every op's probe lines, at the last block. The first operand comes before the op word, so any lean there is shared by every line, whatever its op.</dd>
<dt>Fragment</dt>
<dd>A probe line shown from its second operand, <code>=</code>, answer, or newline on, as a sequence of its own: what a window leaves of a line when it cuts off the op word.</dd>
<dt>Op margin</dt>
<dd>As in ex-2.2.14: how much closer the <code>{ex.ANCHORED_OP}</code> lines sit to e₁ than the pool of all eleven ops, at the role where the gap is largest, averaged over slices. It checks that the anchor landed.</dd>
</dl>

## Conditions

{conditions_html()}

Every arm is ex-2.2.14's primary: `{ex.ANCHORED_OP}` on e₁, labelled at a rate of {ex.LABEL_RATE:g} per line, the pull over the whole line, and the handover recipe. Only the crop policy and, for the last two, the window change.

**Same batches, same labels.** A policy is a weight on each labelled line's pull, applied after the labels are drawn. So at one seed every {ex.BLOCK}-token arm trains on the same windows with the same labels, and differences between arms at a seed are the policy's. The short-window arms draw differently and are compared with each other.

**A policy only takes pull away.** A line that a policy keeps gets the same pull it had under `all`, because the term still divides by the number of labelled lines with anything visible. We could instead divide by the number of lines the policy keeps, but then every kept pull would grow stronger as the policy drops more, which is the side effect ex-2.1.7 warned about.

The cost of our choice is that the policies differ a little in total pull, so the table gives the share each keeps. `cut-only` keeps only {ex.pull_share("cut-only"):.0%}, and that is intended: `whole` and `cut-only` split the pull of `all` in two, so if the lean follows `cut-only`, it follows the cut lines and not the total.

**The policies.** `whole`, `half`, and `scaled` need nothing but the window, so each would carry over to any grammar. `knowable` needs to know where the evidence is: here, that is the op word; in the in-context grammar, it is the posterior over ops given the tokens so far (label variant (c) in the pivot). `knowable` also drops the first operand from the pull on whole lines, as the note in ex-2.2.14 suggested, and serves as a reference for what knowing the evidence buys.

**The short windows.** At {ex.SHORT_BLOCK} tokens, {CUT_SHORT:.0%} of line visits are cut, about the share the in-context grammar would have. `all-short` against `whole-short` asks whether the effect grows with the share of cut lines, and whether `whole` still removes it.

## The first operand's lean comes from cut lines (H1)

**What we expect.** `all` reproduces ex-2.2.14's lean. `whole` removes it: its lean is within {ex.LEAN_BAND:g} of the control's, on the seed mean. And `cut-only` keeps at least {ex.CUT_ONLY_SHARE:.0%} of `all`'s excess over the control, although it has only {ex.pull_share("cut-only"):.0%} of the pull.

Why: the pool puts almost no pull on the first operand of a whole line, because the op word is right beside it and aligns far more easily. The only visits that force pull onto the first operand are the ones that show nothing else. `half` and `knowable` drop those visits too, so we expect them to remove the lean; `scaled` keeps a sixth of their pull, so we expect it to remove part of it.

If `whole` keeps most of the lean, the lean comes from whole lines, perhaps through the readout as the op1-lean reanalysis found for *red*, and cropping is a side issue on this grammar. If `all`'s lean is less than {ex.READABLE_LEAN:g} above the control's, it did not reproduce at these seeds, and H1 is unresolved.

/// admonition | TODO
Figure: the lean at the last block for each arm, one dot per seed with the seed mean, beside the control's band and ex-2.2.14's value; a second panel with the lean at each slice. Table: the seed-mean lean and its excess over the control per arm, with the pull kept beside it.
///

## The anchor and the task hold under every policy (H2)

**What we expect.** No policy costs the anchor or the task. Every arm's op margin is at least {ex.MARGIN_KEEP:.0%} of `all`'s, and every op's held-out expected exact match is within {ex.TASK_GATE:g} of the control's (ex-2.2.11's gate), on the seed means.

In ex-2.2.14 the margin sat at the op word and saturated early, and whole lines carry the op word, so dropping cut lines should leave it where it was. A margin that falls under `whole` would mean the cut lines were doing part of the anchor's work, and that the policy has a price. The rule reads this gate, since a policy that moves the anchor or the task would not go forward.

/// admonition | TODO
Figure: two panels, each arm on the horizontal axis: the op margin as a share of `all`'s, with the {ex.MARGIN_KEEP:.0%} line; and the largest task gap from the control over the eleven ops, with the gate. Table: the same, with the op of the largest gap.
///

## Fragments without their op word (H3)

**What we expect.** Under `all`, fragments lean toward e₁, and under `whole` and `knowable` they lean less. The number is the mean cosine with e₁ over every position of every fragment and every op, at the last block, against the control's. This is a prediction of direction, with no gate.

Why: a line cut before its op word leaves a fragment like `op2 = answer ⏎`. Under `all`, the anchor asks that fragment to align with nothing to go on. The model can respond in one of two ways:

1. with a general lean, as at the first operand, or
2. by guessing the op from what it can see, since some pairs of second operand and answer fit `{ex.ANCHORED_OP}` better than others.

So we also report the contrast between the `{ex.ANCHORED_OP}` fragments and the rest, with no prediction. A positive contrast would mean the model is learning a surface cue for the op. That is the shortcut the pivot names as a [failure mode](../d2.2/pivot.md#the-anchor-picks-up-a-shortcut), seen here in miniature.

/// admonition | TODO
Figure: two panels, each arm on the horizontal axis: the fragments' mean lean, and the contrast between `{ex.ANCHORED_OP}` fragments and the rest, with one dot per seed and the control's band. A small multiple below: the lean at each position of a fragment, by the role it starts at. Table: the two numbers per arm.
///

## Shorter windows, more cut lines (H4)

**What we expect.** Short windows give twice as many cut visits, so we expect the lean caused by cut lines to grow. That is, the gap between `all-short` and `whole-short` should be larger than the gap between `all` and `whole`. We also expect `whole-short` to remove the lean, bringing it to within {ex.LEAN_BAND:g} of the control. This prediction has no gate of its own; the rule falls back on it if H1 is unresolved.

We compare the gaps within each pair rather than comparing `all-short` with `all` directly. The short windows take twice as many steps per epoch, and the term is normalized per labelled line. So at the same weight, the anchor gets about twice as many updates over training. That alone could make `all-short` lean more than `all`, and `whole-short` has the same extra updates.

<!-- REVIEW: H4's statistic changed from "all-short leans more than all" to the within-pair gap (all-short − whole-short against all − whole). The short arms double the anchor updates per epoch, so the raw comparison confounds the share of cut visits with total anchor exposure. Verify: if the DAG rescales λ_a or the step count for the short arms, the raw comparison becomes clean again. -->

Suppose the gap for the short pair is no larger than the gap for the long pair. Then the lean does not depend on how often the first operand is pulled alone, and the extrapolation to the in-context grammar becomes weaker. For the same reason as above (the short windows change the number of steps per epoch), we compare task performance between the two short arms only.

/// admonition | TODO
Figure: the lean at the last block for `all`, `whole`, `all-short`, and `whole-short`, with the share of cut visits on the horizontal axis. Table: the lean, the op margin, and the largest task gap for the short pair.
///

## The rule for the pilot

> {ex.ADOPTION}

/// admonition | TODO
The table of the three candidate policies against the rule's tests: the lean within the band, the H2 gates, and the pull kept.
///

## Exploratory analyses

Anything we think of after seeing the data goes here, marked as post hoc. Three reads are planned as descriptions, with no gate.

**Where the pull lands.** For each arm, the share of each labelled line's pull that goes to each role, averaged over the ways a window shows a line (op1-lean's E7, per arm and slice). It shows what each policy changes about where the anchor is asked to act.

**The lean over training.** The lean at the last block through training, per arm, from the trajectories. It says whether the lean builds early, with the ramp, or late.

**The answer and the newline.** The mean cosine with e₁ at the answer and at the newline over every op's probe lines, per arm. A line cut before its op word puts its pull on these positions, so they may lean the way the first operand does.

## Discussion

/// admonition | TODO
After the results. What we would take to the pilot: which policy, what it costs in labels there (the share of cut contexts is larger, and `whole` drops them all), and whether the evidence question needs label variant (c) on top.
///

## Method

### The policies

The sampler already knows each labelled line's visible run: the window's offset fixes the role of every position, and padding hides a prefix. Each policy turns that run into a weight in [0, 1] on the line's pooled term (`line_weight` in `experiment.py`).

The denominator stays the count of labelled lines with any visible position, as under `all`, and the label draws are unchanged, so the random stream is consumed identically under every policy. `knowable` also removes the positions before the op word from the pool, so a whole line is pooled over roles 1 to 5.

How often a window shows each run of a line, at each window size, enumerated over every offset and padding length:

{shares_html()}

### The measurements

The lean is the mean cosine with e₁ at role 0 over every op's probe lines, at slice {ex.FINAL_SLICE}, as ex-2.2.14 read it; its excess is against the control's seed mean. The op margin is ex-2.2.14's, on the same probe sets. The task is held-out expected exact match per op, against ex-2.2.11's control.

The fragments are the probe lines of every op, cut to start at roles {", ".join(str(r) for r in ex.FRAGMENT_START_ROLES)} and fed as sequences of their own; the fragment lean is the mean cosine over every position of every fragment, and the contrast is the mean over `{ex.ANCHORED_OP}` fragments less the mean over the rest.

### Budget

{ex.N_RUNS} runs at d64-L4, each as long as a run in ex-2.2.14 except the short-window pair, which take twice the steps at half the tokens per step. Ex-2.2.13 trained 160 runs of this size for about fourteen dollars on Modal. Eval adds the fragment probe to ex-2.2.14's reads; no intervention is scored.

### What this experiment does not do

It does not change what the model sees. Two such changes are on the table for the in-context grammar: windows that start at a line boundary, which would remove the cuts at the start of a window but keep those at its end, and an attention mask that resets at each newline, which the pivot proposes for its own reasons and which leaves cut lines cut. Both change the task data as well as the pull, so they would need a control of their own.

It also does not test label variant (c) on the in-context grammar; `knowable` is its version on this grammar.
"""
