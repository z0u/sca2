---
status: finding
tags: [D2.1, anchoring, ex-2.1.6]
---
# Is the anchor drift bounded by the task or by λ?

Ex-2.1.6's mean alignment at op1 went 0.42 → 0.53 → 0.62 over λ ∈ {0.03, 0.1, 0.3} with no measurable task cost at any of them, so we never found the ceiling. A rung or two higher would say whether the task eventually pushes back, which is also the power analysis for how much anchor weight is available to spend.

## Notes

**2026-08-16, housekeeping** — ex-2.1.11's Sobol survey (current op1+pooling+anti-subspace recipe, ~1000 trials over λ_a among other axes) answers this directly: "m_line rises with λ_a until the task gate cuts in" (`## The landscape`), and task-gate failures appear only from λ_a ≈ 0.4 upward (the scatter under `## Proposed operating point`). So the ceiling is the task gate, not a saturation in λ's ability to pull further — closing this as a finding rather than open work. Different recipe from ex-2.1.6's raw λ ladder (op1-only + pooling + repulsion vs. whole-stream), but the question itself is architecture-general and this is a much better-powered read.
