# ruff: noqa: B018
# title: Ex 2.N.M: <what we try, in a few plain words>

# A fill-in skeleton for a preregistered report: copy it to docs/<milestone>/<ex>/report.py and replace each <...>.
# The prose is in r-strings so the placeholders render as written; switch a string to rf once it has template
# expressions. The register to aim for is example-evidence (see exemplar-example-evidence.md beside this file).

r"""
# Ex 2.N.M: <What we try, in a few plain words>

/// tip |
<!-- lede -->
<What we tried, in one or two sentences, with no numbers or IDs.> <The outcome line, written when results land.>
///

<Optional: one or two paragraphs; high-level info that belong on the front page but not in the lede.>
"""

# %%

r"""
## Findings

- [<Short name for H1> (H1)](#short-name-for-h1-h1) — **<pass, miss, partial, or unresolved>**. <What happened, in one sentence a colleague could repeat (lede only).>
- [<Short name for H2> (H2)](#short-name-for-h2-h2) — **<verdict>**. <One sentence.>
- [<Short name for E1> (E1)](#short-name-for-e1-e1) — <what we saw, in one sentence>.
- [Decision](#decision) — <what we chose, once it is made with the human>.

/// admonition | How to read this report
This report was preregistered: the predictions and the criteria for the decision were frozen at commit `<hash>`, before any run of this experiment. Each section opens with what we expect, and the results replaced the placeholders in place.
///
"""

# %%

r"""
## Why this experiment

<Start from the picture: what the model does here, in plain words, so the question follows from it. Recap earlier experiments by what they showed, without their numbers.>

<The question this experiment asks, and why the answer matters for anchoring.>

<So what we try, and what each outcome would tell us.>
"""

# %%

r"""
## Parameters

<The recipe, the conditions, and the one thing each changes. A table follows if there are more than two or three conditions.>
"""

# %%

r"""
## Measurements

<One short paragraph per measurement the result sections use: what it is, in words, and which way is good.>
"""

# %%

r"""
## <Short name for H1> (H1)

**What we expect.** <The prediction, in the conditional: what would count as a pass, a partial pass, and a miss. Name the one number we will look at.> If <the seeds disagree about the direction, or another outcome no rule covers>, the result would be outside the plan, and the verdict would be Unresolved.

/// admonition | TODO
<What the figures or table will show: axes, panels.>
///

<!-- Once the results are in, the placeholder becomes:

**What we saw.** <The answer first, in words. Then the reasons, with a number only where the argument needs its precision. One main measure per section; a second goes in parentheses.>

<The figures, introduced in the sentence before it. What we make of it goes after it, with how sure we are. Interleave the figures with the prose.>

/// admonition | Miss
<The one-line verdict, with its deciding number.>
///
-->
"""

# %%

r"""
## <Short name for E1> (E1)

<What this analysis sets out to look at, and why.>

/// admonition | TODO
<What the figure will show.>
///
"""

# %%

r"""
## Decision

<!-- Only when the experiment chooses something: an operating point, a schedule, an op set. -->

<Frozen with the predictions: the choice to be made, the candidates, and the criteria it will weigh. Hard gates only for task cost and risk rows, where a "no" has to hold whatever the data look like.>

<!-- Once the results are in: every criterion for every candidate, in a table; what we chose with the human and why; and that the choice is confirmed at fresh seeds before it is quoted as a result. -->
"""

# %%

r"""
## Discussion

<What we make of the results as a whole, with how sure we are. What it would mean for the next design can be said as a conditional ("if this holds, ..."). What we do next is for the human to decide.>
"""

# %%

r"""
## Method

<Data, calibration, and measurement details the sections refer back to. Cost.>
"""
