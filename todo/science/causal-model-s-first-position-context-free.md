---
status: finding
tags: [methodology, anchoring]
opened: 2026-07-29
---
# A causal model's first position is context-free, so op1 alignment is a property of the token

Position 0 attends to itself alone, so every probe line sharing an op1 gives the same op1 measurement. Consequences for probe design: averaging over partners buys nothing at op1 (it does at later positions), the measurement carries no sampling error from the partner, and the noise floor is set by the residual slices alone — 0.056 per seed on this architecture rather than the 0.011 a 27-line average would give. Pinned by a test in `tests/sca/test_anchoring.py`.

## Notes

**2026-08-17, Claude (d2.1 figure session)** — a companion caveat for the other slots, found while adding the op2-keyed grading figure to `docs/m2/d2.1/report.py`. The probe set holds a pair only if its mix lands back on the color grid, and that closure rule makes a color's partners non-uniform in redness: partner-mean redness correlates with a color's own at r = 0.27, running 0.20 → 0.36 across the redness range against a palette mean of 0.25 (pure red's partners average 0.36). So in any per-(slice, position) figure keyed by one operand, the *other* operand's slot inherits partner composition: keyed by op2, α at op1 tilts up by 0.17 across the redness range at the embedding even though op1 precedes op2 and no causal path exists. The embedding row measures that bias directly, since every state there is a function of its own token — and both keyings read the same 0.17 there, as symmetry requires. It doesn't touch any published statistic (all of them read the on-key slot), but a future figure or metric that reads an off-key slot should subtract it or say what it is.

**2026-08-18, Claude (d2.1 figure session)** — the same off-key tilt, now measured across all four conditions of the progression figure, since `docs/m2/d2.1/report.py` draws both keyings for each. At the embedding it reads 0.02 un-anchored, 0.18 under the bare anchor, 0.26 under the anti-subspace term and 0.17 on the primary. So the bias is not a fixed offset to subtract: it scales with how much response there is to compose from, and vanishes where there is none. That the un-anchored condition shows ~0 is the cleanest evidence yet that it really is partner composition rather than a measurement artifact of the probe set — an artifact would be there whether or not the model responded to anything. A figure or metric reading an off-key slot should therefore state the bias for its own condition, rather than carrying the 0.17 across.
