---
status: open
tags: [D2.1, ex-2.1.3, metrics]
---
# Distance-shaped answer targets for ex-2.1.3, post-hoc

We scored answers against the one-hot truth (NLL of the true name); the sharper question is whether the model's whole answer distribution is shaped like the geometry — build a target distribution per prompt from RGB distance to the true mix (e.g. softmax of −distance/τ over the vocabulary) and measure cross-entropy / KL against it, sweeping τ. Needs no re-run: the eval step saved the full log-probability vector over color tokens for every prompt (`arrays` `{label}/logp/{set}`), so this is a report-side analysis. Ex-2.1.4's report now implements the τ-sweep (KL(q_τ ‖ p) with a uniform reference); reuse its recipe and τ grid so the rungs compare directly. Note from ex-2.1.4: the metric jointly scores geometry and calibration — a confidently-wrong model fits worse than uniform — so read it beside s₂.
