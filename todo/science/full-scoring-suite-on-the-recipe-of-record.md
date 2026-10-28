---
status: open
tags: [D2.2, intervention, in-context]
opened: 2026-10-28
priority: high
---
# The full scoring suite on the recipe of record

The [fixed confirmation](/docs/m2/fixed-confirm/report.py) adopted `τ_ℓ fixed` as the recipe of record for D2.2 and scored it on removal, spill, the task, and lost anchors. The blog post needs the rest of what the edit does, on the same runs (seeds 712 to 719):

- what the [edit-effects report](/docs/m2/edit-effects/report.py) measured on `falloff`: removal by dose, the grading (the share removed by posterior bin), the landing, and the write bound by slice and position;
- the confusion matrices of [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (H2): by bin of the posterior on the true op, where the mass goes under the clean model, the edited model, and the target null.

The runs are stored, and the anneal confirmation pipeline that the fixed confirmation reuses already measures the write on them, so this is mostly a re-analysis. The [reference model without `difference`](./reference-model-without-the-anchored-op.md) belongs beside it, scored the same way, so the two could share a report.
