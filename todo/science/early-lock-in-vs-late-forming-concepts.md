---
status: open
tags: [anchoring, M3]
opened: 2026-08-23
---
# Early lock-in vs late-forming concepts

Ex-2.1.9 found the softmin winner is committed by epoch 8 in every run and never revisited. Harmless in this testbed — the task converges in epochs — but in larger models concepts can form late (phase-transition-style), after the pull has committed to a position. Whether a committed softmin can follow a concept that moves is an M3-facing question.

The [τ-schedule item](./pooled-anchor-tau-schedule-and-adaptive-tau.md) bears on it: a soft → sharp schedule that stays soft long enough to see the phase transition is one answer, and a mechanism that resharpens after a movement signal is another.

Split from the older ex-2.1.9 follow-ups item on 2026-08-23.
