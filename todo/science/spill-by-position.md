---
status: done
tags: [D2.2, anchoring, intervention, selectivity, ex-2.2.23]
opened: 2026-10-08
closed: 2026-10-08
---
# Which positions the spill of the edit comes from

Ex-2.2.23 found that the edit (projecting e₁ out of every state) spills onto other ops on nearly every run that has learned the HSV ops, mostly onto `darken`, `lighten`, and `value-hsv`. This asks where in the context the spill comes from, on ex-2.2.23's checkpoints, with no new training.

A short exploratory report. The edit acts at every slice but on one set of positions at a time, at each dose: the example answers, the query operands, the query `=`, and the rest of the positions up to the query `=`; and, as a map, each single position alone at full dose. For each, the removal on `difference` (how far toward the target null) and the spill (the largest drop on another op, net of the control under the same edit), at 200 and 400 epochs. The question it answers is whether some position, or set of positions, gives the removal without the spill. Position maps are drawn as smooth-step charts along the context, with markers at the measured positions.

Ex-2.2.22's `suppress_one` already takes a position mask through `logits_at`; it needs a variant that takes an arbitrary mask. Cost: one pass over the 48 checkpoints, a few dollars at most.

Split from the withdrawn ex-2.2.24 draft (its H2 and E2; see [PR #255](https://github.com/z0u/sca2/pull/255)).

## Notes

**2026-10-08, spill-by-position** — Done in [the report](/docs/m2/spill-by-position/report.py). On most runs the removal and the spill sit at the same positions, the example answers; the rest of the positions spill without removing (onto `lighten`); and on four 400-epoch runs the separator embedding lies on e₁ (a latch of the pooled pull), which carries most of their spill.
