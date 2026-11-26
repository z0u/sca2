---
status: open
tags: [D2.2, D2.3, anchoring, intervention]
opened: 2026-11-05
---
# Route the reader's learning on `difference` contexts through e₁

The [low-runs report](/docs/m2/low-runs/report.py) found that the reader blocks of the low runs settle on the weaker, off-axis part of the evidence for `difference` rather than on e₁, and nothing in training sees which part they chose. This item shapes what the reader uses while it learns, rather than scoring what it outputs under an edit.

Gradient routing assigns what a model learns from chosen examples to chosen parameters, by masking the gradient during training. Here it would boost, on labelled `difference` contexts, the updates to the rows of the query, key and value matrices that read e₁, so the reader learns to take its evidence from there. A softer version is a small regularizer on those matrices that discourages reading other directions on labelled `difference` contexts; routing seems the cleaner of the two, since it doesn't add a term that competes with the task.

Things to settle in a plan: which blocks count as the reader (block 3 at the query `=` on most runs, block 4 on some low runs, per low-runs and block1-edit), how strong the boost is and how it is scheduled against the anchor, and whether it uses only the labelled contexts, as the anchor does. Measurements: the low tail of removal on the fixed-labels screen, spill, task change, and the single-slice edit before the reader, which should remove about as much as the full edit if the reader now takes its evidence from e₁. Related: [is-the-inferred-op-held-in-attention](./is-the-inferred-op-held-in-attention.md) mentions gradient routing for the attention spaces, and the baseline comparisons item keeps SGTM as future work.

From the same review as [anchor-the-not-difference-end-of-the-arc](./anchor-the-not-difference-end-of-the-arc.md), which is the one to try first: it reuses the anchor machinery and its outcome is easier to read.

## Notes

**2026-11-26, D2.2 wrap-up** — Tagged D2.3: the D2.2 post names reader affinity for e₁ as the first piece of the circuit design for the verification task.
