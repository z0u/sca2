---
status: finding
tags: [D2.1, ex-2.1.2]
opened: 2026-07-15
---
# `named_holdout` is unsolved in 4 layers; value → name translation is the blocker

The 2×2 factorial (reverse aliases × off-palette named-as-hex, d64-L4) trained both missing ingredients — reverse aliases read out at 1.0 in their own frame, and name + name arithmetic generalizes to unseen off-palette pairs at ≈ 0.92 — yet `named_holdout` stays at exactly 0 in every cell. Decomposition: in the `open` cells ~1/3 of held-out answers are the correct mix value in hex form (form rule learned per-pair, not per-value); the rest are lookup-neighbor names. The name-identity margin (log P(true name) − best other name) sits ≈ −9 nats everywhere, so value → name never engages mid-equation though it is perfect in the `#hex = ` frame. Consequence: anchored runs train on the `both` corpus and use `open_holdout` + s₂ as graded canaries (`named_holdout` has no headroom to lose). Whether `named_holdout` is solvable at all in 4 layers is parked — candidates: a denser named sub-grid (value-diverse rgb→name supervision in-frame, which changes the concept inventory the anchors will label), more depth, or a frame-interleaving curriculum. Full analysis with figures in `docs/m2/ex-2.1.2/report.py`.
