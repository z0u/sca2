# Exemplar: an exploratory report (example-evidence)

Excerpts from `docs/m2/example-evidence/report.py`, frozen here as of commit `a0427bc`, with template expressions shown as they rendered. The live report may change later. It is a re-analysis of ex-2.2.22 with no new runs, and it printed to eight pages. Review found little to change in it and asked for future experiments to take its shape: short, pointed, quick to review, and light on hypotheses and decisions.

The notes after each excerpt say what to copy. For sentence-level register, [exemplar-ex-2.2.18.md](exemplar-ex-2.2.18.md) has more.

## Opening

```md
# What the anchor follows at the example answers

/// tip |
<!-- lede -->
In ex-2.2.22 the alignment with the anchored direction at the example answers rose with the posterior on `difference`, but at the same posterior it sat higher at earlier answers. That difference goes away once each answer is compared with the evidence of its own example. The alignment at an example answer follows how well that one example fits `difference`, and hardly depends on the examples before it.
///

The [backlog item](/todo/science/anchor-grades-with-the-posterior.md) asks whether the anchor holds the op the model has inferred, which would make it grade with the posterior, rather than the binary label it was trained on. [Ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (E3) took a first look and found a rise with the posterior at the example answers, plus an unexplained dependence on the answer index. This page re-analyzes the same stored measurements, with no new runs.
```

The whole report hangs on one puzzle, and the lede states it in its first sentence: a rise that sat higher at earlier answers. The second and third sentences resolve it. The paragraph under the box says where the question came from, with links, and what the page reads. That paragraph is the entire "why"; a reader who wants more follows the links.

## Observations and scope

```md
## Observations

- [Each example on its own (E1)](#each-example-on-its-own-e1): plotted against the posterior given that example alone, the lines for the three answer indices lie on top of each other. On every anchored condition, that posterior accounts for at least 89% of the variance in α (r²), against at most 48% for the posterior given the examples so far.
- [What the earlier examples add (E2)](#what-the-earlier-examples-add-e2): fitted together with the evidence before that example, the earlier evidence gets a weight of at most 0.05, against at least 1.2 for the example itself. It shows most on answers that do not fit `difference`.
- [The answer color alone (E3)](#the-answer-color-alone-e3): the answer color, without its operands, predicts at most 9% of the variance, so α depends on how the answer relates to its operands.

## Scope

This is a re-analysis of the stored by-count pass of ex-2.2.22 on its three-example held-out set, with no preregistration and no gate. It covers the `hinge` condition (six runs: three paired with ex-2.2.21 and three replicates), the whole-line condition (three runs), `k-mixed` (three runs, one left out as below), and the control (five runs). Only the held-out `difference` contexts are used, and only their example answers: as ex-2.2.22 found, nearly every query answer has a posterior near 1, so there is too little spread to compare there.

The `k-mixed` run at model seed 702 has no anchor at the example answers [...]. It is left out of the numbers below.
```

Three observations, each a question the previous one raised: does the one-example evidence explain the puzzle (E1), do the earlier examples add anything once it is accounted for (E2), and could something simpler than the evidence explain it (E3). Each observation is one claim with one kind of number. A reader who stops here has the result.

`Scope` replaces both `Parameters` and `Method`. It says what data the page covers, which part it leaves out and why, and which run it drops. There is no separate method section, because nothing in it would change how a reader takes a result.

## A section (E2, whole)

```md
## What the earlier examples add (E2)

The one-example posterior leaves some variance unexplained, so the earlier examples might still add something. A least-squares fit of α on the one-example posterior and the earlier posterior together gives each a weight; both are on the same scale, from 0 to 1.

<table: the two weights, and α at answers that don't fit, after doubt and after support, per condition>

The earlier examples get a weight of a few hundredths, against more than one for the example itself. They show most on the answers that do not fit: there α is a little higher when the earlier examples favor `difference` than when they do not (0.16 against 0.09 on `hinge`). On answers that fit, the earlier examples make almost no difference.
```

The section opens with the question it asks and why the previous section raised it, then the method in one sentence, then the evidence, then what we saw. It is three paragraphs and one table. "A few hundredths, against more than one" gives the size in words, and the table has the digits.

## Discussion

```md
At the example answers the anchor marks examples that look like `difference` more than contexts that do. The rise with the posterior that ex-2.2.22 saw comes mostly from this: the more examples fit, the higher the posterior, and each fitting example has a high α of its own. So this measurement does not yet show the anchor holding an inferred op. A site where that could show is one where the line itself says nothing about the op, such as the query `=`, and ex-2.2.21 found the anchor weak there.

This fits how the label works. [...]

It may also bear on the spill that ex-2.2.23 found. [...] This re-analysis covers only `difference` contexts, so it does not test that.
```

Three short paragraphs: what the result means for the question, why it fits what we know of the method, and what it may bear on elsewhere, with where this page stops. It names a site where the open question could be tested without announcing a next experiment.

## What to copy

- One question, posed by a puzzle a reader can hold in mind. Every section serves it, and the report ends when it is answered.
- Three or so E sections that each follow from what the last left open.
- No hypotheses, gates, verdicts, or Decision. The report says what we saw and how sure we are, and the strength of the evidence (r² near 0.9 on every anchored condition, across a dozen runs) is visible without a frozen threshold.
- The front matter is the lede, one paragraph, `Observations`, `Scope`, and the measurements. The first result starts on page two or three.
