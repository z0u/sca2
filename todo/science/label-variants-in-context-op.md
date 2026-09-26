---
status: open
tags: [D2.2, anchoring, M3]
opened: 2026-09-24
---
# Label variants for the inferred op

In the [D2.2 pivot](/docs/m2/d2.2/pivot.md), a binary label on the whole context asks for the op at positions that cannot know it yet: early in the line, where attention has seen too little, and in contexts where replacement noise leaves the posterior on the true op below 0.5. The pooled anchor term softens the first case only. The whole-line label stays the default, since an M3 labeller would likely give that form. Four cheap variants should measure its effect on the task and the anchor:

- Leave out the embedding slice.
- Pull only the latter half of each line, where the posterior given the prefix is at or near its final value. This is the simplest fix along position.
- Label by a thresholded prefix posterior: a position is labelled when the posterior given the tokens before it clears a threshold. This handles both position and evidence, and is the form an M3 labeller with a confidence cutoff would give (a classifier run on each prefix of a conversation). It also leaves the contexts in the middle band of the posterior unlabelled, so alignment there shows grading the pull never touched, as D2.1 did for *red* with binary labels.
- Sample the label from the posterior: label each context with probability equal to its posterior on `difference`. The labels then follow the evidence the model can see, which should be easier to satisfy than the true op in noisy contexts. The model never sees a label, so this gives it nothing to read the answer from. But the pull on a context would be proportional, on average, to its posterior, which trains the grading directly, so the graded check in the middle band would no longer be a result the pull left free. It is also an idealized labeller: an M3 labeller would more likely err on surface cues than in proportion to the evidence.

Which variant becomes the primary would be decided on [the pilot](/docs/m2/d2.2/design.md#the-pilot), which has all four as arms. Whether the thresholded label helps is one input to the soft-label question in the pivot.
