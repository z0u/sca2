---
status: open
tags: [methodology, D2.2, D2.4, baselines]
priority: high
---
# Baseline comparisons for SCA — plan and related-work delta

A 2026-08-13 four-angle literature sweep (gradient-routing lineage, training-time interpretability, unlearning, post-hoc erasure/steering) is banked in [`references/related-work-delta-2026.md`](/references/related-work-delta-2026.md), including which entries are snippet-only and need verification before citing. Headlines: SGTM is now its own paper (arXiv:2512.05648) and is the must-run training-time baseline; post-hoc bounded-side-effect methods now exist (COAST, arXiv:2605.01167; pre-intervention prediction, arXiv:2606.08365), so SCA's bound should be framed as by-construction rather than merely bounded; 2025 auditing standards (relearning rebound, activation-perturbation, minor-direction recoverability) should be adopted in our evals. Prep task: keep the eval contract method-agnostic — every baseline produces (model, subspace, intervention operator) and the shared eval scores the triple. Sequencing sketch: post-hoc tier (probe / diff-in-means / LEACE on existing controls) + filtered-corpus row first; SGTM as its own preregistered experiment; RMU-family row at D2.3; one bounded SAE experiment with output-score feature selection (arXiv:2505.20063).

## Notes

**2026-08-17, housekeeping** — promoted alongside the operation-variable item, since both are D2.2 prerequisites rather than D2.2 itself. The binding piece is the method-agnostic eval contract: every baseline has to produce (model, subspace, intervention operator) for a shared scorer, and that shape is far cheaper to fix before D2.2's own experiment code is written than to retrofit around it afterwards. The literature half is already banked in `references/related-work-delta-2026.md`, so what's left here is the plan, not another sweep. Shortlist was 1/6.
