---
status: open
tags: [D2.2, task-grammar, representations, ex-2.1.5]
opened: 2026-08-31
---
# Does op diversity push hex and named colors toward a shared representation?

Ex-2.1.5 found the model bridges the two color forms and never unifies them: ρ at 0, principal angles at their null, and [capacity pressure separates the forms instead of merging them](./compression-separates-two-forms-rather-than-merging.md). But every 2.1.x corpus had one operation, so a per-form solution — two decoders plus a conversion — satisfies the task at a fixed cost.

The in-context-learning literature has a suggestive result about that kind of setup: pretraining task diversity separates a memorization regime from a generalization regime, with a phase transition between them (Raventós et al., arXiv:2306.15063, empirically; Lu et al., arXiv:2405.11751, in closed form for linear attention). The analogy to us is loose — there, "task" means a regression vector presented in-context and the observable is behavior on unseen tasks; here, ops are learned in-weights and the observable is representational sharing across surface forms. What transfers is the cost argument: each added op multiplies the price of per-form machinery, while one shared color space amortizes it, so sharing should become the cheaper solution somewhere along the op-count axis.

Measurement: the ex-2.1.5 two-form corpus crossed with the D2.2 op table, un-anchored controls only, scored with ex-2.1.5's sharing metrics (ρ, principal angle between probe subspaces, zero-shot cross-form transfer) as functions of op count. That is the [richer-op-set arm](./richer-op-set-operand-geometry.md) plus the two-form corpus: that item's cube probe reads the operand geometry within a form, this one reads sharing across forms; they could run as one sweep. With at most ~6 well-defined ops on the 16-level grid we sample the diversity axis far too coarsely to locate a transition, so the claim to preregister is monotone — some sharing metric leaves its null as ops are added — with the phase-transition literature as motivation, and a prediction of the sharp version only if HSV-space ops later widen the table.
