---
status: open
tags: [D2.2, intervention, in-context, ex-2.2.22]
opened: 2026-10-06
priority: high
---
# How close does a model trained without `difference` come to the target null?

We score the edited model against the *target null*: the answers of an ideal predictor that has lost `difference` and nothing else. In [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (H2) the edit moved each replicate run a little under half of the way there, context by context, while pooled over contexts the mass went about where the target null puts it. Part of the distance that remains is the gap any model has from an ideal predictor: the clean model is some way from the ideal predictor that keeps every op (E4).

So we don't know how much of the remaining distance a model could close at all. A model trained on the same corpus with no `difference` contexts would answer them as a model that never learned the op, and its distance from the target null would be a reference for the edit: a landing near that model, rather than near the ideal, may be as far as an edit could go. It would also show whether such a model behaves like the target null itself, which is the assumption behind using the target null as the destination.

It needs one condition (the control recipe on a corpus without `difference`, at a few seeds) and the landing measurements of ex-2.2.22, scored on its held-out `difference` contexts. Raised in review of ex-2.2.22.

## Notes

**2026-10-22, edit-effects review** — The [edit-effects report](/docs/m2/edit-effects/report.py) measured the landing on the recipe of record: the distance left is about 2.4× the distance from the clean model to the Bayes predictor, nearly all of it in how the edited model weighs the other ops context by context, with no systematic tilt toward flatter or sharper weights. Its Discussion names two readings (the edit blurs evidence the model uses to weigh the other ops, or the model never learned to weigh them on `difference` contexts) and this model is what would tell them apart. The paper review of that report asked for it.

**2026-10-24, backlog refresh** — On the shortlist. Claim (c) of the D2.2 design (the edited model answers in a predictable way) needs it to say whether half of the distance is short, and it is also the filtered-corpus comparison of the design. It trains controls only, so with the median baseline ([reference-baseline](/docs/m2/reference-baseline/report.py)) it can be scored without a paired control of its own.

**2026-10-27, science desk** — Planned as [filtered-corpus](/docs/m2/filtered-corpus/report.py): the control recipe on the corpus of record with the `difference` lines dropped, at seeds 704 to 711. Leaving the noise answers of `difference` in the contexts of the other ops makes the target null the Bayes predictor of that corpus exactly, so the model is scored against what is ideal for what it trained on. Waiting on an OK for the plan.
