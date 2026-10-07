---
status: open
tags: [D2.2, intervention, in-context, ex-2.2.22]
opened: 2026-10-06
---
# How close does a model trained without `difference` come to the target null?

We score the edited model against the *target null*: the answers of an ideal predictor that has lost `difference` and nothing else. In [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (H2) the edit moved each replicate run a little under half of the way there, context by context, while pooled over contexts the mass went about where the target null puts it. Part of the distance that remains is the gap any model has from an ideal predictor: the clean model is some way from the ideal predictor that keeps every op (E4).

So we don't know how much of the remaining distance a model could close at all. A model trained on the same corpus with no `difference` contexts would answer them as a model that never learned the op, and its distance from the target null would be a reference for the edit: a landing near that model, rather than near the ideal, may be as far as an edit could go. It would also show whether such a model behaves like the target null itself, which is the assumption behind using the target null as the destination.

It needs one condition (the control recipe on a corpus without `difference`, at a few seeds) and the landing measurements of ex-2.2.22, scored on its held-out `difference` contexts. Sandy raised it in review of ex-2.2.22.
