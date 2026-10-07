---
status: open
tags: [D2.2, in-context, analysis, ex-2.2.19]
opened: 2026-10-02
---
# Is the extra commitment on unclear contexts overfitting?

The [follow-up note to ex-2.2.19](/docs/m2/ex-2.2.19/calibration.py) found that from 200 to 400 epochs the seven-op model puts more mass on its top answer than the Bayes answer distribution does where the posterior on the true op is below 0.5, and the KL per context there rises at every seed while the total stays level. That fits a model that keeps sharpening with training, and it also fits one that memorizes the answers of training contexts, which at 400 epochs it has seen many times.

The two predict different things on the training corpus. Memorizing predicts a KL on unclear training contexts that falls from 200 to 400 epochs while the held-out one rises; sharpening alone predicts both rise together. The end checkpoints are stored (`CHECKPOINT_REF` in ex-2.2.19), so this needs an evaluation pass over a sample of training contexts with the eval of ex-2.2.18, a few minutes of L4 time per run, and the same split by posterior bin. If it is memorization, it bears on how long the anchoring runs should train.
