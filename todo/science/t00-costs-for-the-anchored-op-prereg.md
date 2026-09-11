---
status: open
tags: [D2.2, anchoring, selectivity, intervention, ex-2.2.3]
opened: 2026-09-09
---
# What the adopted point `t00` costs, for the anchored-op prereg to weigh

Ex-2.2.3's frozen rule adopted `t00` (λ_a 0.557, τ 0.265, anti peak 1.376, 50 epochs) as D2.2's operating point, and the rule was applied as written. Four things sat outside the rule and the next prereg should price them before inheriting the point.

- **Lead weight 0.33 at the embedding**, under the 0.4 that H2 gates the recipe at. The selection rule read feasibility as the survey did, and the survey had no lead gate.
- **Contrast 0.36** against the recipe's 0.86: the red and non-red groups are much less separated along the axis.
- **Clean accuracy on the red `mix` lines 0.89** against 1.00 for the recipe. That accounts for the whole H1 gap (red lines are 6% of the holdout, and 6% of an 11-point loss is the 0.009 in the H1 table).
- **Syntax rows at 0.93** on the axis at `=` (E2), more than twice the recipe's 0.40, which is why the full-position projection costs 0.60 of the non-red `mix` lines. The `operands` edit (projection at positions 0 and 2 only) brings that to 0.017; the intervention-tuning pass decides which edit is carried, and these numbers favor `operands`. See [syntax rows carry the axis](./syntax-rows-carry-the-axis-via-tied-readout.md).

The fallback if these prove too expensive: `t12` was the runner-up on m_line, with contrast 0.44, lead 0.48, and an H1 gap of 0.002. The recipe itself stays feasible on the six-op grammar (H2 holds), at the lower plateau the H3 section describes.

From [ex-2.2.3](/docs/m2/ex-2.2.3/report.py#discussion), Discussion.

**2026-09-11, Fable** — the fourth bullet is priced: [ex-2.2.8](/docs/m2/ex-2.2.8/report.py) scored a hundred operators on `t00`'s five stored seeds. Every edit applied at every position costs between 0.32 and 0.95 of the non-red `mix` lines (the plain projection 0.62, and every thresholded step more than that), and every operand-only edit is inside the gate; `operands` reads red accuracy 0.078 at a deficit of 0.024, and the lowest feasible red accuracy is `shaped-a0.1-p0-operands` at 0.071, not resolved apart at five seeds. On the adopted `recipe-short` the same survey picks the plain projection. So a prereg on `t00` names an operand-only operator up front; the choice of point and the choice of operator go together.
