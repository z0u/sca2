---
status: finding
tags: [D2.2, in-context, analysis, ex-2.2.19]
opened: 2026-10-01
---
# Per-op calibration KL over all held-out contexts

The op confusion matrices of [ex-2.2.17](/docs/m2/ex-2.2.17/report.py) and [ex-2.2.19](/docs/m2/ex-2.2.19/report.py) (E3) are taken on confident contexts only, where the Bayes posterior on the true op is above a threshold. The calibration KL (E2) is taken over every held-out context. Sandy, reviewing ex-2.2.19, asked whether a matrix like the confusion one can be made over the whole corpus, showing how much each op contributes to the KL.

One way: group the per-context KL by true op, for the rows, and split it by the op whose answers hold the mass the model has in excess of the Bayes answer distribution, for the columns. The row sums would then add up to the calibration KL. On unconfident contexts several ops can give the same answer, so the column split needs a rule for shared answers (perhaps sharing the mass by the Bayes posterior over ops). It would show whether the KL that remains at 200 and 400 epochs sits in the HSV-channel ops, as the confusion on confident contexts does, or in the contexts the confusion matrices leave out.

## Finding

**2026-10-02, science desk** — Done post hoc in [a follow-up note to ex-2.2.19](/docs/m2/ex-2.2.19/calibration.py), on the 200- and 400-epoch runs at four seeds each. The columns follow the rule above: excess mass over the Bayes answer distribution, shared among ops by the posterior, with extra columns for colors only a dropped op gives, colors no op gives, and mass on tokens that are not colors. Each context gives its KL to the columns in proportion to its excess, so the matrix sums to the calibration KL. The row split is exact and the column split is a convention.

- The remaining KL sits mostly in the contexts the confusion matrices leave out. Contexts below a posterior of 0.5 on the true op (26% of the held-out set) hold 42% of the KL at 200 epochs; the confident set (posterior above 0.99, 23%) holds 9%.
- By true op the KL is spread over all seven ops. `mix` is highest per context (0.42), `lighten` and `darken` lowest (0.24); the three HSV-channel ops hold 48%. No single other op stands out as a column in any row; colors no op gives take 17%.
- The total is level between the two lengths (0.323 and 0.335), but the parts move apart at every seed: from 200 to 400 epochs the KL per context below 0.5 rises by 0.04 to 0.07 and the confident KL falls by 0.007 to 0.03. Below 0.5 the model puts 0.08 (200 epochs) and 0.11 (400) more mass on its top answer than the Bayes answer distribution does, so longer training commits it further to one answer where the examples allow several. This could be overfitting to the training corpus; there is no training-set KL stored to tell.
