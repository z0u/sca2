---
status: open
tags: [D2.2, intervention, fallback, selectivity, ex-2.2.2, ex-2.2.7, ex-2.2.9]
opened: 2026-09-24
---
# Revisit fallback control on the untied-readout recipe

[Ex-2.2.2](/docs/m2/ex-2.2.2/report.py) found the fallback term does in the transformer what it did in M1: every seed gave the designed answer under the reflection it was trained at, and the seed disagreement of ex-2.2.1 was gone there. It was left out of later experiments for three reasons. The reflection was destructive on non-red lines in every anchored model. The response carried over to the plain projection unevenly by seed. And the anti-anchor hinge it relies on cost margin.

The first reason came from the `+` and `=` embeddings carrying the axis, and [ex-2.2.7](/docs/m2/ex-2.2.7/report.py) traced that to the tied readout ([item](./syntax-embeddings-carry-the-axis-via-tied-readout.md)). The recipe is now untied (adopted in [ex-2.2.9](/docs/m2/ex-2.2.9/report.py)), and the op words and `=` came down to near zero on the axis there. So the reflection's non-red cost may be much smaller now, and a full-position reflection may be usable as the trained edit.

The question: on the current untied recipe, does fallback control keep H1 (designed answer in every seed) and pass the non-red selectivity gate under the reflection it trains at, where ex-2.2.2 failed it?

Things that did not change with the readout, and that a re-run should read rather than assume:

- The `⏎` row still carries the axis under the whole-line labeller ([item](./eol-embedding-row-keeps-the-axis.md)), so the reflection still flips something at every line's last position.
- Containment is looser under the untied readout ([item](./containment-rises-under-the-untied-readout.md)): ᾱ at op1 rose from about 0.1 to 0.28, which puts more non-red states near the plane the reflection turns about.
- The antipode-to-zero transfer gap and the hinge's margin cost are unrelated to the readout. If the reflection becomes selective, the gap matters less, since the reflection itself could be the operator we use. The hinge cost would remain.

A cheap first step needs no new training: score the stored ex-2.2.9 checkpoints (no fallback term) under the reflection and read the non-red deficit against ex-2.2.2's no-fallback row. If the deficit is now inside the gate, a fallback re-run on the recipe is worth its budget. If it is not, E7's breakdown of what the non-red lines decode to says which rows are responsible. The [repulsion-onto-the-fallback item](./repulsion-onto-the-fallback.md) already names a fallback re-run on the adopted point as the thing that makes its whole-sequence version testable, so the two could share one run.

From a conversation with Sandy, after revisiting ex-2.2.2's findings.
