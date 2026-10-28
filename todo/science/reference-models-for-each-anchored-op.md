---
status: open
tags: [D2.2, intervention, in-context]
opened: 2026-10-28
---
# Reference models for each op in turn

D2.2 anchors one op, `difference`, and the [reference model](./reference-model-without-the-anchored-op.md) trains without that op to show how close any model can come to the target null. That says how far an edit can land for one op only. Ops differ in how often their answers coincide with those of other ops, and in how late a run learns them (the HSV ops come last), so the landing and the spill may well differ by op.

So eventually we would repeat the pair for each op: anchor `mix` (say) in place of `difference` on the recipe of record, and train a reference model without `mix`, then score the edit against both as for `difference`. Seven ops would mean seven anchored conditions and seven reference conditions, at a few seeds each. Not expected before the D2.2 blog post.
