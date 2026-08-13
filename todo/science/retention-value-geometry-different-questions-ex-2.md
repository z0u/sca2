---
status: finding
tags: [representations, ex-2.1.2, ex-2.1.5, metrics]
opened: 2026-07-26
---
# Retention and value-geometry are different questions, and ex-2.1.2's eviction finding may have measured the first

In ex-2.1.5, hex operand 2's red channel read at the green digit in the last layer scores 0.821 per-equation and −0.27 under a strict holdout. Both are true of different things: red's identity is still recoverable there (causal attention reaches back to its digit), but it is no longer placed by value. So "the earlier channel persists into later digits" is an identity claim, not a geometric one — and the report sentence that read it as depth accumulating a value representation is wrong and needs replacing. The same question applies upstream: ex-2.1.2's just-in-time-with-eviction result (R² ≈ 0.97 at a channel's own emission position, dropped afterwards) was measured on a small vocabulary with the same per-equation holdout, so its "dropped" may mean "no longer recoverable" or "no longer value-organized". Worth a re-check before anchor design leans on it.
