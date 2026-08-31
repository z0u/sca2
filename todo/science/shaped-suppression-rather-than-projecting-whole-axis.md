---
status: open
tags: [D2.2, D2.3, anchoring, ex-2.1.6]
priority: high
---
# Shaped suppression, rather than projecting the whole axis out

Every intervention we have specified for M2 removes the anchor direction outright, which is the crudest edit available and the one most exposed to the drift ex-2.1.6 found: if the whole cube has a common component on the axis, taking the axis away moves everything. A suppression shaped to act on the red-selective part of the response — a function of the component rather than its deletion — might degrade red behavior while leaving the color cube intact, and it would still have side-effects boundable from the geometry beforehand, which is the property the method is for. Design question to settle before the first transformer intervention.

We already tried this in M1 (autoencoders): the "intervention lobes" work in `scratch-m1-code/docs/m2-control/ex-2.1-intervention-lobe.ipynb` and the paper appendix `references/sca1-paper/asec_intervention_lobes.tex` define a bounded falloff $h(\alpha)$ — zero below an alignment threshold $a$, ramping to max strength $b$ with shape controlled by power $p$ (plus a Bézier variant for the rotation-based "repulsion" intervention) — instead of the aggressive $h(\alpha)=\alpha$ used in the main M1 text. Worth pulling forward as a starting point rather than re-deriving.

## Notes

**2026-08-19, housekeeping** — promoted, as the fourth D2.2 prerequisite alongside the baselines, the operation variable, and the survey-format lessons. D2.2 asks for suppression that "scales as it did in the autoencoders", which is a claim about the *form* of the intervention, so the choice between axis deletion and a shaped falloff has to be made while the plan is being written. It is also the cheapest of the four to settle: the M1 lobes are specified and implemented, so the work is deciding whether to carry them over, not deriving them. Shortlist was 3/6.

**2026-08-30, housekeeping** — the carry-over half is settled: the [D2.2 design](/docs/m2/d2.2/design.md) puts the M1 lobe in the eval contract's operator library, beside axis projection and weight ablation. What it does not do is pick the lobe for any experiment — *suppress red* runs plain axis projection, and the fallback and dose-label machinery downstream is written against the projection operator too.

That ordering reads as deliberate, and it changes what this item is waiting on. The worry here is ex-2.1.6's drift: if the whole cube has a common component on the axis, taking the axis away moves everything. *Suppress red* measures precisely that — observed non-red damage against a geometric prediction — on checkpoints already in the store, with no training cost. So the number that says whether a shaped falloff is needed arrives from the plan's first and cheapest experiment. Nothing needs to block on settling this beforehand; what this item asks for now is that the lobe be in the operator library when that contract gets built, and that the suppress-red damage figure be what picks the operator for the interventions after it.
