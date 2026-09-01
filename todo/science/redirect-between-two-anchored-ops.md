---
status: open
tags: [D2.3, anchoring, task-grammar]
opened: 2026-09-01
---
# Redirect between two anchored ops (a concept swap)

Anchor two operations, then intervene by redirecting one's component to the other: a model asked to `add` computes `multiply`. Sandy's suggestion during the D2.2 design review.

Two things make it attractive. The outcome is designed and has per-line ground truth — the target op's answer for the pair — so efficacy is scored line by line with no null construction at all. And it is a steering claim rather than a removal claim, which the post-hoc erasure baselines do not naturally make.

The angle question it raises: a redirect from $v_{add}$ to $v_{multiply}$ needs to know where multiply's state is, and with only one op anchored that is the natural geometry, unknown before training. Anchoring both dissolves the question — the angle is placed rather than discovered — at the risk of fighting the geometry the ops would prefer (they plausibly share a common *this-is-an-operation* component), which the task gate and alignment margins measure.

Sits beside "several ops on separate axes", already a D2.3 candidate for the subspace bound; the swap is the intervention that pair of axes makes possible, so the two probably land as one experiment.
