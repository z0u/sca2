---
status: finding
tags: [methodology, anchoring]
opened: 2026-07-29
---
# A causal model's first position is context-free, so op1 alignment is a property of the token

Position 0 attends to itself alone, so every probe line sharing an op1 gives the same op1 measurement. Consequences for probe design: averaging over partners buys nothing at op1 (it does at later positions), the measurement carries no sampling error from the partner, and the noise floor is set by the residual slices alone — 0.056 per seed on this architecture rather than the 0.011 a 27-line average would give. Pinned by a test in `tests/sca/test_anchoring.py`.
