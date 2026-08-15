---
status: done
tags: [D2.1, anchoring, ex-2.1.6]
closed: 2026-08-15
---
# Pull op1 alone, as the counterpart to the position-pooling item below

Ex-2.1.6 pulls four prompt positions, and three of them (`+`, op2, `=`) carry no information about which color sits there — the label is an unobservable coin flip, so only the expected pull is visible, and at those positions it depends on op1 rather than the token present. That is a mechanism for the drift, separate from the missing repulsive terms, and pulling op1 alone separates the two.

## Notes

**2026-08-15, housekeeping** — ex-2.1.7 ran exactly this as its second crossed factor (`{span, op1}` pull × `{bare, anti-subspace}`), and H3 found the op1-only effect larger than the anti-subspace effect in every seed. Closing as done.
