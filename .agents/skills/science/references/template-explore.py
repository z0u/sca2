# ruff: noqa: B018
# title: <What we look at, in a few plain words>

# A fill-in skeleton for an exploratory report, the default shape: one question, a few observations, no
# preregistration and no gate. It suits a re-analysis of stored results and a study with a few new runs alike. Copy it
# to docs/<milestone>/<key>/report.py and replace each <...>. The prose is in r-strings so the placeholders render as
# written; switch a string to rf once it has template expressions. The example-evidence report is this form, filled in
# (see exemplar-example-evidence.md beside this file). Aim for about eight printed pages.

r"""
# <What we look at, in a few plain words>

/// tip |
<!-- lede -->
<The puzzle or question, and which way it came out, in two or three sentences with no numbers.>
///

<Where the question came from (a result an earlier report left unexplained, say), linked. What this report runs or reads: "with no new runs", or how many runs and seeds.>
"""

# %%

r"""
## Observations

- [<Short name for E1> (E1)](#short-name-for-e1-e1): <what we saw, in words>.
- [<Short name for E2> (E2)](#short-name-for-e2-e2): <one or two sentences>.
- [<Short name for E3> (E3)](#short-name-for-e3-e3): <one or two sentences>.

## Scope

This is an exploratory <re-analysis, or study>, with no preregistration and no gate. <Which runs and conditions it covers, and which data. How far a difference between runs can be trusted, against a scale such as the seed range.>

## The measurements

<What each measurement is, in words, and which way is good. A short list where there are several summaries to compare. Only what a reader needs to parse the sections below.>
"""

# %%

r"""
## <Short name for E1> (E1)

<The question this section asks, in a sentence or two, and what each answer would look like.>

<!-- The figure or table. -->

<What we saw: the answer first, in words, then the reasons. One main measure per section; a number only where the argument needs its precision.>

<What we make of it, with how sure we are.>
"""

# %%

r"""
## <Short name for E2> (E2)

<Same shape. Each section follows from what the last one left open.>
"""

# %%

r"""
## <Short name for E3> (E3)

<Same shape.>
"""

# %%

r"""
## Discussion

<What we make of the observations as a whole, with how sure we are, in two to four short paragraphs. What it would mean for a design can be said as a conditional. What we do next is left out of the report, for the human to decide.>
"""
