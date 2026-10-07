---
status: open
tags: [D2.2, in-context, anchoring, ex-2.2.21]
opened: 2026-10-01
priority: high
---
# Does the anchor grade with the posterior on the op?

The ex-2.2.21 draft had a hypothesis (H2 in its first review round) that the alignment at the query `=` rises with the posterior on `difference` across the middle band. The label is binary on the true op, so a context whose examples only half-fit `difference` is pulled as hard as one that names it; if the alignment grades anyway, the anchor holds the inferred op rather than the label. The review moved it out of the pilot, for two reasons found in ex-2.2.16's stored runs.

First, the anchor does not land at the query `=` on the whole-line label: it sits on the answers (the preview of ex-2.2.21). Scoring at the query `=` would measure grading where there is almost nothing to grade. The query answer is no better, since its state comes after the answer is predicted, so the measurement site has to follow wherever the anchor turns out to sit, and that is what ex-2.2.21's E1 and its two reference labels (`prompt`, `query-eq`) will show.

Second, the stimulus is coarse. At three examples and ρ = 0.3 the posterior on `difference` takes few distinct values: on the seven-op table about 0.4% of `difference` contexts fall in the 0.5 to 0.65 bin, 3% in 0.65 to 0.8, and 38% in 0.8 to 0.95, nearly all of them in one step near 0.91. (The eleven-op table is similar: 2%, 5%, and 35%.) So a test across ex-2.2.16's evidence bins has two nearly empty bins. A version that holds up treats the posterior as about three levels (near 0.3, near 0.91, and near 1) or widens the stimulus, with a mix of example counts or noise rates in the held-out set only. Grammars with more examples (k > 3, which ex-2.2.16 scanned up to six) are another way to widen it: each extra example splits the posterior levels further, at the cost of a longer context.

Round 3 already plans a graded stimulus (the [D2.2 route](/docs/m2/d2.2/design-2026-10.md#quick-route)), so this may belong there. The `sampled` arm of ex-2.2.21, which trains the grading, is the trained comparison.

## Notes

**2026-10-04, PM** — Drafted as part of the ex-2.2.22 preregistration in [PR #246](https://github.com/z0u/sca2/pull/246) (not yet frozen): the control and the hinge arm trained on contexts with 1 to 5 examples, which spreads the posterior on the op over more values than the three a fixed count gives. Shortlisted while that scout is the next step for D2.2 round 3. Close or settle this from its results.

**2026-10-05, Claude** — [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (E3, in preregistration) takes this up as a first look with no gate: α at the example answers and the query answer against the posterior given the pairs up to and including that answer, on `k-mixed` (example counts drawn from one to five) and on the three-example conditions. A fixed-count context already gives one level of evidence per answer, so the three-example runs have more spread than the query `=` alone suggests. Since the posterior also rises with the position of an answer, E3 compares contexts at the same answer index too.

**2026-10-06, Claude** — E3 of [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) is in: at the example answers α rises with the posterior on every anchored condition, at a fixed answer index too, and the control stays flat. Three examples per context give enough spread for that, and mixing the counts added little. At the same posterior α is higher at earlier answers, which is not yet explained. This is a first look with no gate; a preregistered version would fix the bins and the answer index in advance.
