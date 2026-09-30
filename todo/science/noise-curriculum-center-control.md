---
status: open
tags: [D2.2, training, ex-2.2.17]
opened: 2026-09-28
---
# Does a replacement-noise curriculum lift the center control past its level?

In ex-2.2.17 the d64-L4 center control (`k3-r0.3`) levelled off near 0.45 held-out expected exact match against a Bayes ceiling of 0.52, and schedule (cosine, warmup-stable-decay, staircase), length (eight and sixteen times), width (d128), and depth (L6) all left it within the seed range. The shortfall is a similar fraction of the ceiling in every op group, which suggests the shared step, inferring the op from noisy examples, as the limit (to be checked by scoring the existing checkpoints by how many examples in a context carry replacement op noise).

If that holds, a curriculum over the replacement rate is one way to help: start with clean contexts, where the op is easy to infer, and raise the rate to 0.3 over training; or the reverse, starting noisy. Sandy would rather not, since the curriculum changes the training distribution over time and so confounds every comparison made on the recipe afterward. Try it only if the noise-split scoring points at op inference, and treat it as a diagnostic of what limits the control more than as a candidate recipe.

## Notes

**2026-09-28, Opus** — The noise-split scoring is done, in the ex-2.2.17 report (E3, E4). It points away from op inference from noisy contexts: with two or three noisy examples, or a posterior on the true op below 0.5, the control scores what the ideal predictor scores. The gap sits on contexts whose examples point to one op, where the model keeps some mass on answers of ops the examples have ruled out (several times what nearness in the cube would give). So the limit looks like how firmly the model commits once the op is clear. Whether a curriculum that starts on clean contexts would teach firmer commitment is open; the condition above ("only if the scoring points at op inference") is not met as written.
