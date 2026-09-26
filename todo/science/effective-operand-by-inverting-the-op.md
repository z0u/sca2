---
status: open
tags: [D2.2, probes, analysis]
opened: 2026-09-26
---
# The effective operand, read back from the answer by inverting the op

From Sandy's review of [answer-distance](/docs/m2/answer-distance/report.py), on its counterfactual section: "We can probably reverse-engineer it from the answer distributions? Can the ops be reversed?"

That section tries two fixed substitutes for the red operand (red set to zero, and red lowered to the larger of green and blue), and finds the lost answers sit between the true answer and either substitute. Inverting the op asks the question the other way round: given the visible operand and the model's answer under the operator, which red operand would the op need to produce it? If the recovered operands cluster (say, a fixed fraction of the red taken away, or the hue rotated by a fixed amount), the intervention has a simple description in operand space.

Which ops invert, per channel, given the other operand `y` and the answer `z` (on the 0..15 scale):
- `mix`: x = 2z − y, everywhere.
- `multiply`: x = 15z / y, except where y = 0.
- `screen`: x = 15 − 15(15 − z) / (15 − y), except where y = 15.
- `exclusion`: x = (z − y) / (1 − 2y/15), except where y = 7.5 (not on the grid, so everywhere in practice, though badly conditioned near the middle).
- `difference`: two solutions, y ± z; the answer alone cannot choose.
- `lighten` and `darken`: only where the answer differs from y; elsewhere x is known only to sit on one side of y.
- `hue-hsv` with red as the hue donor: the answer's hue is the effective operand's hue. With red as the recipient, its saturation and value are read back the same way.

Use the mean of the model's distribution as `z`, since the greedy guess is rounded to the grid and inverting doubles that error on `mix`. The re-score keeps every per-line mean in each run's memo record and publishes it for `hue-hsv` and `darken` only (`LINE_OPS` in its experiment), so adding the invertible ops to `LINE_OPS` should need only the publish step to run again.
