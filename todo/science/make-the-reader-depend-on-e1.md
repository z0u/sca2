---
status: open
tags: [D2.2, anchoring, intervention]
opened: 2026-11-02
---
# Make the reader depend on e₁: an edit-aware term

The [low-runs report](/docs/m2/low-runs/report.py) found that every run writes the evidence that an example fits `difference` twice: along e₁, where the anchor put it, and along one other direction that the loss made on its own. The low runs are the ones whose reader blocks take enough of that evidence from off the axis for the e₁ edit to leave a fifth of the op in place. Nothing in training sees this: the twins (same labels, seed and batches, different last bits of arithmetic) match in loss and in α for the whole run and part only in removal.

So the lever is a term that the loss does see. One shape: on the labelled contexts, run a second forward pass with e₁ cleared at the pull sites, and train its prediction toward the target null (the sans-diff reference model's answer, or a flat distribution over the op's answers). That makes the reader depend on e₁ because the model is scored on what it does without it. It costs a second forward pass on the labelled lines, which are a small share of a batch. A cheaper shape: at the pull sites, drop e₁ at random on a fraction of the labelled contexts and ask for the same null then, a dropout on the anchored coordinate.

The report also lists three schedule ideas from its review, which would be cheaper to try first and could be read with the same ten low runs as the baseline: a higher λ_a at the start of training, a slightly lower λ_s̄, and a λ_s̄ that ramps down over depth more gently, since the readers sit at blocks 3 and 4. None of them touches the reader as such, so they may move the odds without closing the gap.

Any of these is a recipe change, so it goes through the fixed-labels screen: eight seeds at 1/50, the share of runs with removal under 0.8, against 5 of 24 on the recipe of record. A check at epoch 224 with the e₁ edit itself (the earliest epoch where the low and normal runs separate, per the report's E4) would let a sweep drop the low runs before the anneal.
