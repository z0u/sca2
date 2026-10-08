---
status: open
tags: [anchoring, D2.2, in-context, ex-2.2.22, ex-2.2.23]
opened: 2026-10-08
---
# Mixed example counts at 400 epochs

Ex-2.2.22 trained `k-mixed` (the `hinge` recipe with 1 to 5 examples per context) at 200 epochs, and on three-example contexts it fell short of `hinge` by about a tenth in EEM, inside the seed band (H1, partial). Ex-2.2.23 then found that 200 epochs leaves some runs half-trained, and moved training to 400 epochs. So part of that shortfall may be training length.

The [example-evidence re-analysis](/docs/m2/example-evidence/report.py) (E4) gives a reason to look again. `k-mixed` handles every count from one to five about equally, where the control and `hinge` fall away outside the three examples they were trained on, and on `hinge` one run collapses at four and five. But `k-mixed` stays further below the ideal predictor than `hinge` does with three examples (0.69 against 0.88 in the top bin). If it closed that gap at 400 epochs, it would be a candidate for the recipe of record, since the query could then come after any number of examples.

A small arm: `k-mixed` at 400 epochs on the seeds of ex-2.2.23, scored with the by-count pass (EEM by count against the ceiling, α at the example answers and at every `=`) and the edit criteria, beside the 400-epoch `hinge` runs of ex-2.2.23.
