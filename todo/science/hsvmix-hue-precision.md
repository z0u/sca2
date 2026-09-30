---
status: open
tags: [D2.2, in-context, ex-2.2.17]
opened: 2026-09-28
---
# Help the model learn `hsvmix`, or drop it

In ex-2.2.17 (E5, E6), `hsvmix` is the one op whose computation falls short: the center control gets 83% of the Bayes ceiling when the operand hues are close and 43% when they are nearly opposite, and a wider or deeper model does no better. Near-opposite hues are where the midpoint around the wheel moves furthest for a small change in either operand, so the op asks for more precision than the RGB grid tokens make easy to learn.

Two directions, from Sandy's review:
(a) give the model more to learn hue from, for example more HSV-heavy training data, or separate hue, saturation, and value tokens so the op can be read off its inputs; or
(b) drop `hsvmix` from the op set.

The second is simpler and would remove a shortfall that has nothing to do with inferring the op, which is what the in-context grammar is for. Either changes the task, so the Bayes ceiling of the center condition would need recomputing. Worth deciding together with [similar ops](drop-ops-with-similar-answers.md), since both prune or reshape the op set.

## Notes


**2026-09-29, ex-2.2.18 session** — In [ex-2.2.18](/docs/m2/ex-2.2.18/report.py) (E2), both runs without `hsvmix` learned the three HSV-channel ops early: mean probe EEM 0.36 and 0.35 on those ops at step 23k, against 0.23 for the full set. One seed each, but it suggests `hsvmix` slows the ops nearest to it, as well as falling short itself. That adds weight to option (b).
