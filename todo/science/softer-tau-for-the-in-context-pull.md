---
status: open
tags: [anchoring, D2.2, in-context, ex-2.2.22]
opened: 2026-10-07
---
# A softer mellowmax τ for the in-context pull

The anchor term pools the alignment over the positions of a `difference` context with mellowmax at τ = 0.1, the value frozen in ex-2.2.3. At that τ the pool is close to a max, so the pull can be met by the few positions that align best. The [example-evidence re-analysis](/docs/m2/example-evidence/report.py) found that it is met at the answers of the examples that fit `difference`, each on its own evidence, and ex-2.2.21 found the anchor weak at the query `=`, the one site where the op has to be inferred. A softer pool would spread the pull over more of the context, and might move some of it onto the query `=`.

We have only swept τ downward, on the earlier grammar ([ex-2.2.12](/docs/m2/ex-2.2.12/report.py): 0.03 and 0.01, neither moved its target). A softer τ on the in-context grammar is untried. A scout of the `hinge` recipe at a few larger τ values (say 0.3 and 1), scored on α at the query `=` against the posterior given the examples, with the edit criteria as checks, would say whether the pooling is what keeps the anchor off the query. The τ schedule in [the pooled-anchor item](/todo/science/pooled-anchor-tau-schedule-and-adaptive-tau.md) is a related lever, and ex-2.2.24 (E3) scores the query `=` on stored runs first, which sets the baseline.
