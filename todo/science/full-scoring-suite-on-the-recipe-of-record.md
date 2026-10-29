---
status: done
closed: 2026-10-29
tags: [D2.2, intervention, in-context]
opened: 2026-10-28
---
# The full scoring suite on the recipe of record

The [fixed confirmation](/docs/m2/fixed-confirm/report.py) adopted `τ_ℓ fixed` as the recipe of record for D2.2 and scored it on removal, spill, the task, and lost anchors. The blog post needs the rest of what the edit does, on the same runs (seeds 712 to 719):

- what the [edit-effects report](/docs/m2/edit-effects/report.py) measured on `falloff`: removal by dose, the grading (the share removed by posterior bin), the landing, and the write bound by slice and position;
- the confusion matrices of [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (H2): by bin of the posterior on the true op, where the mass goes under the clean model, the edited model, and the target null.

The runs are stored, and the anneal confirmation pipeline that the fixed confirmation reuses already measures the write on them, so this is mostly a re-analysis. The [reference model without `difference`](./reference-model-without-the-anchored-op.md) belongs beside it, scored the same way, so the two could share a report.

## Notes

**2026-10-28, science desk** — The [fixed-effects report](/docs/m2/fixed-effects/report.py) scores the eight runs at seeds 712 to 719 on everything in the first two bullets, from stored data alone, with the edit-effects `falloff` runs beside it. Every measure comes out about as on `falloff`: removal grows with the dose and is complete by three quarters of the full edit, spill stays below zero at every dose on seven runs and reaches 0.003 on the eighth, the share removed is level across the posterior (0.83 to 0.93 by bin), the landing is 0.50, and the write stays within the bound. Pooled over contexts, the `difference` row of the confusion matrix matches the target null at every level of the posterior. One new shape: on the seed mean the bound at the answers of the other ops widens to slice 3 (38° at the example answers, against 22° on `falloff`), set by about one answer in a hundred near e₁, with the runs of the two recipes overlapping. What is left is the reference model beside it, carried by [its own item](./reference-model-without-the-anchored-op.md). There are no control alignment arrays at seeds 712 to 719, so the write bound has no control beside it.

**2026-10-29, design refresh** — Done: the reference model is [sans-diff](/docs/m2/filtered-corpus/report.py) (PR #308), and the [D2.2 design](/docs/m2/d2.2/design.md) now cites both reports under claims (b) to (d).
