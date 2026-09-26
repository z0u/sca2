# title: Ex 2.2.15: lines cut short by the training window

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). During preregistration that module is constants only.
import experiment as ex

POLICY_TEXT = {
    "all": "every labelled line, whatever is visible (the current behaviour)",
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
        f"<tr><td><code>{ex.CONTROL}</code></td><td>none: the un-anchored control from ex-2.2.11, served from the store</td>"
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
SMOKE = ex.SMOKE_RED_LEAN


rf"""
# Ex 2.2.15: lines cut short by the training window

/// tip |
<!-- tl;dr -->
The anchor pulls on whole lines. But in training, the model sees the corpus through a window, and each window cuts off the lines at its edges. On a cut line, the anchor asks the visible part to carry the whole label, even when that part cannot see the op word. We retrain the anchored model under a few policies for which cut lines to pull, and track what each one does over the course of training. We do this on the current grammar, before the in-context grammar makes the problem larger.
///

This is a scouting run of {ex.N_RUNS} fresh training runs, with a few predictions and one rule: it proposes the crop policy the [in-context grammar pilot](../d2.2/design.md#the-pilot) starts from. Cut lines are a small share of any one batch, but the anchor meets them at every step, so the question is what they add up to over training. Each run records the lean and the trailing-fragment lean at every trajectory point (every {ex.TRAJ_STRIDE} training steps), as well as at the end.

## Findings

- [The first operand lean is due to cut lines (H1)](#the-first-operand-lean-is-due-to-cut-lines-h1) —
- [The anchor and the task hold under every policy (H2)](#the-anchor-and-the-task-hold-under-every-policy-h2) —
- [Trailing fragments without their op word (H3)](#trailing-fragments-without-their-op-word-h3) —
- [Halved windows, more cut lines (H4)](#halved-windows-more-cut-lines-h4) —

[The rule for the pilot](#the-rule-for-the-pilot): —

## How to read this draft

The policies, the four predictions, and the rule were fixed before any run, at commit `TODO`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

Each result section opens with what we expect, then a placeholder for what we saw.

## Why this experiment

The training corpus is one long tape of six-token lines. Each training step cuts a {ex.BLOCK}-token window from it at a random place. Most lines in the window are whole, but the two edges usually slice through a line.

The anchor does not know about the edges. It asks each labelled line to align with the axis wherever that comes most easily in the visible part, and so asks that part to carry the whole label.

If a `{ex.ANCHORED_OP}` line shows only its first operand, the anchor asks a color token to say "this line is `{ex.ANCHORED_OP}`" before the op word has appeared. That token cannot know, so the only way the model can satisfy the anchor there is to lean every first operand a little toward the axis.

That is what [ex-2.2.14](../ex-2.2.14/report.py) saw, after the fact. At the last block, the first operand leaned toward e₁ at {ex.REFERENCE_OP1_LEAN:.2f} on the primary arm, against {ex.REFERENCE_OP1_LEAN_CONTROL:.2f} on the control, and the arm that pulls only the op word had no lean. The [op1-lean reanalysis](../op1-lean/report.py) found the same route on the *red* runs, and proposed skipping the pull on cut lines as a test.

Before fixing the predictions, we ran that test as a smoke test on *red*: two seeds each of `all`, `whole`, and `cut-only`. `whole` took the lean from {SMOKE["all"]:.2f} to {SMOKE["whole"]:.2f}, and `cut-only` kept {SMOKE["cut-only"]:.2f}, against {SMOKE["control"]:.2f} for the control. So on *red*, cut lines carry about a third of the lean and whole lines the rest. The validation loss was the same under every policy, to three decimals.

Red is a weaker test than the op, though. On a *red* line the first operand can be red itself, which is evidence the anchor can pull on; on a `{ex.ANCHORED_OP}` line the first operand says nothing about the op.

At {ex.BLOCK} tokens, {CUT:.0%} of line visits are cut short. On {BLIND:.1%} the op word is out of sight, and on {OP1_ONLY:.1%} only the first operand is visible. The shares are small, but the term is normalized per labelled line, so each of those visits gets the pull of a full line.

This matters more after the [D2.2 pivot](../d2.2/pivot.md). There, a line is a context of solved examples about 20 tokens long, and the op is inferred from the examples, so a window that cuts the start of a context can remove the examples that name the op. With fewer lines per window, the share of cut visits roughly doubles, and a cut context has lost evidence, not just one token. The current grammar is the cheap place to see what each policy does, with a clear sign (the lean) to watch.

## Glossary

<dl>
<dt>Cut line</dt>
<dd>A line visit that the training window shows only part of. A line cut at its start has lost its first tokens, so its states differ from those of the whole line. A line cut at its end shows a prefix. Under causal attention (each position sees only earlier positions), those prefix states are the same as in the whole line.</dd>
<dt>Crop policy</dt>
<dd>Which labelled lines the anchor pulls, given what the window shows of them. The label draws are the same under every policy; only the pull changes.</dd>
<dt>Pull kept</dt>
<dd>The share of the current total pull a policy keeps, averaged over every way a window shows a line. It is computed from the sampler, not measured.</dd>
<dt>The lean</dt>
<dd>The mean cosine with e₁ at the first operand, over every op's probe lines, at the last block. The first operand comes before the op word, so any lean there is shared by every line, whatever its op.</dd>
<dt>Trailing fragment</dt>
<dd>A probe line shown from its second operand, <code>=</code>, answer, or newline on, as a sequence of its own: what a window leaves of a line when it cuts off the op word.</dd>
<dt>Op margin</dt>
<dd>As in ex-2.2.14: how much closer the <code>{ex.ANCHORED_OP}</code> lines sit to e₁ than the pool of all eleven ops, at the role where the gap is largest, averaged over slices. It checks that the anchor landed.</dd>
</dl>

## Conditions

{conditions_html()}

Every arm is the primary from ex-2.2.14: `{ex.ANCHORED_OP}` on e₁, labelled at a rate of {ex.LABEL_RATE:g} per line, the pull over the whole line, and the handover recipe. Only the crop policy and, for the last two, the window change.

**Same batches, same labels.** A policy is a weight on the pull of each labelled line, applied after the labels are drawn. So at one seed every {ex.BLOCK}-token arm trains on the same windows with the same labels, and differences between arms at a seed come from the policy. The short-window arms draw differently and are compared with each other.

**A policy only takes pull away.** A line that a policy keeps gets the same pull it had under `all`, because the term still divides by the number of labelled lines with anything visible. We could instead divide by the number of lines the policy keeps, but then every kept pull would grow stronger as the policy drops more, which is the side effect ex-2.1.7 warned about.

The cost of our choice is that the policies differ a little in total pull, so the table gives the share each keeps. `cut-only` keeps only {ex.pull_share("cut-only"):.0%}, and that is intended: `whole` and `cut-only` split the pull of `all` in two, so if the lean follows `cut-only`, it follows the cut lines and not the total.

**The policies.** `whole`, `half`, and `scaled` need nothing but the window, so each would carry over to any grammar. They differ on spans longer than the window, as a natural-language document often is: `whole` never pulls one, `half` stops at twice the window, and `scaled` pulls every span by the share in view. `knowable` needs to know where the evidence is: here, that is the op word; in the in-context grammar, it is the posterior over ops given the tokens so far (label variant (c) in the pivot). `knowable` also drops the first operand from the pull on whole lines, as the note in ex-2.2.14 suggested, and is a reference for what knowing the evidence buys.

**The halved windows.** At {ex.SHORT_BLOCK} tokens, {CUT_SHORT:.0%} of line visits are cut, about the share the in-context grammar would have. The batch doubles to {ex.SHORT_BATCH} windows, so a step sees the same number of tokens and the runs take the same steps. `all-short` against `whole-short` asks whether the effect grows with the share of cut lines, and whether `whole` still removes it.

## The first operand lean is due to cut lines (H1)

**What we expect.** `all` reproduces the lean from ex-2.2.14. `whole` removes it: its seed-mean lean is within {ex.LEAN_BAND:g} of the control's. And `cut-only` keeps at least {ex.CUT_ONLY_SHARE:.0%} of the excess of `all` over the control, although it has only {ex.pull_share("cut-only"):.0%} of the pull.

Why: the pool needs only one position of a line to align. On a whole line the op word aligns far more easily than the first operand, so the first operand gets almost no pull; only visits that show nothing else force pull onto it. `half` and `knowable` also drop those visits, so we expect them to remove the lean. `scaled` keeps a sixth of their pull, so we expect it to remove most of it.

Through training, we expect the lean of `all` to build while the anchor weight is high and persist through the anneal, and the lean of `whole` to stay near the control throughout. This prediction is descriptive, with no gate. If `whole` shows a lean early and sheds it later, or `all` builds its lean only late, that would mean the cut lines matter at a particular stage of training, which a reading taken only at the end would miss.

If `whole` keeps most of the lean, the lean comes from whole lines (perhaps through the readout, as the op1-lean reanalysis found for *red*), and cropping is a side issue on this grammar. If `whole` removes part of the lean but stays outside the band, as on *red* in the smoke test, the cut lines are one route among others, and `cut-only` says how large a share they carry.

If the lean of `all` is less than {ex.READABLE_LEAN:g} above the control's, it did not reproduce at these seeds, and H1 is unresolved.

<!-- REVIEW: the smoke test on red (two seeds) found `whole` removing about 30% of the lean and `cut-only` keeping about a third, so on red H1 as written would miss. The gate and CUT_ONLY_SHARE stay as they were, because on a red line the first operand can itself carry the label's evidence, and on a `difference` line it cannot; the middle case above was added so a partial result has a reading. Verify: a reader who takes red as a fair proxy for the op could argue for a partial band on H1, or a lower CUT_ONLY_SHARE. -->

/// admonition | TODO
Three figures. (1) The lean at the last block for each arm, one dot per seed with the seed mean, beside the band of the control and the value from ex-2.2.14. (2) The lean at each slice, one line per arm. (3) The lean at the last block through training, one line per arm (the seed mean, with each seed as a hairline), with the schedule of the anchor weight behind it. Table: the seed-mean lean and its excess over the control per arm, with the pull kept beside it.
///

## The anchor and the task hold under every policy (H2)

**What we expect.** No policy costs the anchor or the task. The op margin of every arm is at least {ex.MARGIN_KEEP:.0%} of the margin under `all`, and every op's held-out expected exact match is within {ex.TASK_GATE:g} of the control (the gate from ex-2.2.11), on the seed means.

In ex-2.2.14 the margin sat at the op word and saturated early, and whole lines carry the op word, so dropping cut lines should leave it where it was. A margin that falls under `whole` could mean the cut lines were doing part of the work of the anchor. It could also mean only that the policy removed some of the total pull, and λ_a would need scaling up to make up for it. The pull kept tells the two apart roughly: a fall about the size of the pull removed points to λ_a, and a larger one to the cut lines. Either way the rule reads this gate, since a policy that moves the anchor or the task would not go forward as it stands.

/// admonition | TODO
Figure: two panels, each arm on the horizontal axis: the op margin as a share of the margin under `all`, with the {ex.MARGIN_KEEP:.0%} line and the failing region hatched; and the largest task gap from the control over the eleven ops, with the gate and the failing region hatched. Table: the same, with the op of the largest gap and the pull kept.
///

## Trailing fragments without their op word (H3)

**What we expect.** Under `all`, trailing fragments lean toward e₁, and under `whole` and `knowable` they lean less. The number is the mean cosine with e₁ over every position of every trailing fragment and every op, at the last block, against the control. This is a prediction of direction, with no gate.

Why: under `all`, the anchor asks a trailing fragment like `op2 = answer ⏎` to align with nothing to go on. The pool puts the pull where alignment comes most easily, which here is likely the syntax tokens `=` and `⏎`, since they carry no color. So we expect most of the lean to sit there.

On *red*, the smoke test points to the second operand as well: on whole probe lines, the lean there came mostly from cut lines. `whole` took it from {ex.SMOKE_RED_OP2["all"]:.2f} to {ex.SMOKE_RED_OP2["whole"]:.2f}, and `cut-only` kept {ex.SMOKE_RED_OP2["cut-only"]:.2f}. But on *red* the second operand can itself be red, so that may not carry over to the op.

The model can respond with a general lean, as at the first operand, or by guessing the op from what it can see, since some pairs of second operand and answer fit `{ex.ANCHORED_OP}` better than others. So we also report the contrast between the `{ex.ANCHORED_OP}` trailing fragments and the rest, with no prediction.

A positive contrast would mean the model is learning a surface cue for the op: the shortcut the pivot names as a [failure mode](../d2.2/pivot.md#the-anchor-picks-up-a-shortcut), here in miniature.

/// admonition | TODO
Figure: two panels, each arm on the horizontal axis: the mean lean of the trailing fragments, and the contrast between `{ex.ANCHORED_OP}` trailing fragments and the rest, with one dot per seed and the band of the control. A small multiple below: the lean at each position of a trailing fragment, by the role it starts at, with the syntax tokens marked. Table: the two numbers per arm.
///

## Halved windows, more cut lines (H4)

**What we expect.** Halved windows give twice as many cut visits, so we expect the lean caused by cut lines to grow. That is, the gap between `all-short` and `whole-short` should be larger than the gap between `all` and `whole`. We also expect `whole-short` to remove the lean, bringing it to within {ex.LEAN_BAND:g} of the control. This prediction has no gate of its own; the rule falls back on it if H1 is unresolved.

We compare the gaps within each pair rather than comparing `all-short` with `all` directly. The doubled batch matches the steps and the anchor updates, but a model trained on {ex.SHORT_BLOCK}-token windows has seen less context per line, and the gap within a pair holds that fixed.

<!-- REVIEW: H4's statistic is the within-pair gap (all-short − whole-short against all − whole). The short arms now double the batch, so the steps and anchor updates match the long arms (Sandy's review of f2f8e41); the within-pair gap stays because the window still changes the context the model learns from. Verify: with the steps matched, a reader could argue for the raw all-short vs all comparison as a second read. -->

Suppose the gap for the halved pair is no larger than the gap for the long pair. Then the lean does not depend on how often the first operand is pulled alone, and the extrapolation to the in-context grammar becomes weaker.

/// admonition | TODO
Figure: the lean at the last block for `all`, `whole`, `all-short`, and `whole-short`, with the share of cut visits on the horizontal axis. Table: the lean, the op margin, and the largest task gap for the halved pair.
///

## The rule for the pilot

> {ex.ADOPTION}

The looser band for `scaled` is {ex.SCALED_BAND:g}, against {ex.LEAN_BAND:g} for the others. `whole` may give the cleanest result, but `scaled` is the one that carries to labelled spans of any length, so the rule accepts a little of the lean to get it.

/// admonition | TODO
The table of the three candidate policies against the tests of the rule: the lean within its band, the H2 gates, and the pull kept.
///

## Exploratory analyses

Anything we think of after seeing the data goes here, marked as post hoc. Two descriptive reads are planned, with no gate.

**Where the pull lands.** For each arm and slice, the share of the pull on a labelled line that goes to each role, averaged over the ways a window shows a line (as in E7 of op1-lean). This shows what each policy changes about where the anchor is asked to act.

**Each role on whole lines.** The mean cosine with e₁ at each role over every op's probe lines, per arm and slice. A line cut before its op word puts its pull on the roles after it, so those roles may lean the way the first operand does, on whole lines as well as on trailing fragments.

## Discussion

/// admonition | TODO
After the results. What we would take to the pilot: which policy, what it costs in labels there (the share of cut contexts is larger, and `whole` drops them all), and whether the evidence question needs label variant (c) on top.
///

## Method

### The policies

The sampler already knows the visible run of each labelled line: the offset of the window fixes the role of every position, and padding hides a prefix. Each policy turns that run into a weight in [0, 1] on the pooled term of the line (`line_weight` in `experiment.py`).

The denominator stays the count of labelled lines with any visible position, as under `all`, and the label draws are unchanged, so the random stream is consumed identically under every policy. `knowable` also removes the positions before the op word from the pool, so a whole line is pooled over roles 1 to 5.

How often a window shows each run of a line, at each window size, enumerated over every offset and padding length:

{shares_html()}

### The measurements

The lean is the mean cosine with e₁ at role 0 over every op's probe lines, at slice {ex.FINAL_SLICE}, as ex-2.2.14 read it; its excess is against the seed mean of the control. The op margin is the one from ex-2.2.14, on the same probe sets. The task is held-out expected exact match per op, against the control from ex-2.2.11.

The trailing fragments are the probe lines of every op, cut to start at roles {", ".join(str(r) for r in ex.FRAGMENT_START_ROLES)} and fed as sequences of their own; the trailing-fragment lean is the mean cosine over every position of every trailing fragment, and the contrast is the mean over `{ex.ANCHORED_OP}` trailing fragments less the mean over the rest.

The op margin, the lean, and the trailing-fragment lean are also read during training, every {ex.TRAJ_STRIDE} steps, on the same probe lines. These reads run inside the training loop; no checkpoint is kept along the way.

### Budget

{ex.N_RUNS} runs at d64-L4, each as long as a run in ex-2.2.14; the halved pair takes the same steps at twice the windows per step. Ex-2.2.13 trained 160 runs of this size for about fourteen dollars on Modal. Eval adds the trailing-fragment probe to the reads from ex-2.2.14; no intervention is scored.

### What this experiment does not do

It does not change what the model sees. The in-context grammar could remove cut lines altogether with windows that hold whole lines only: each window starts at a line boundary and is padded after the last line that fits, with a longer block so the padding is a small share. Windows that start at a line boundary but are not padded would still cut the last line. An attention mask that resets at each newline, which the pivot proposes for its own reasons, leaves cut lines cut. Each of these changes the task data as well as the pull, so it would need a control of its own. A policy from this experiment still matters with padded windows wherever a labelled span can be longer than the window, as a natural-language document can.

It also does not test label variant (c) on the in-context grammar; `knowable` is its version on this grammar.
"""
