---
status: open
tags: [anchoring, D2.2, in-context, ex-2.2.22]
opened: 2026-10-07
---
# Score the `no-emb` runs by example count

The by-count pass of ex-2.2.22 stored α at every example answer, with the posterior, for `hinge`, whole-line, `k-mixed`, and the control. It did not cover the conditions that leave the embedding slice out of the pull (`no-emb`, `no-emb-matched`, and the uncapped `anchor-no-emb`). Ex-2.2.21 found that the pull on the embedding slice puts a few syntax embeddings (`=` and `,` on `hinge`) partway onto e₁, the same in every context, so part of the alignment elsewhere may be held up from below.

Running the same pass on the stored `no-emb` checkpoints would need no training, and the [example-evidence re-analysis](/docs/m2/example-evidence/report.py) could then add those runs beside the others: does α at an example answer still follow that example alone, and what does the query `=` look like without the embedding pull? If the answer differs, the next step would be training runs that combine `no-emb` with mixed example counts, which no condition has done yet.

The same pass could also store α at the `=` of each example, on every condition. The next-token loss asks the model to predict each example answer at its `=`, so those are sites where it has to combine the examples so far, with a posterior that climbs from 1/7 at the first example. The pass so far stored α only at the answers, where the example just shown is in view.
