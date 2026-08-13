---
status: open
tags: [D2.2, D2.3, anchoring, ex-2.1.6]
---
# Shaped suppression, rather than projecting the whole axis out

Every intervention we have specified for M2 removes the anchor direction outright, which is the crudest edit available and the one most exposed to the drift ex-2.1.6 found: if the whole cube has a common component on the axis, taking the axis away moves everything. A suppression shaped to act on the red-selective part of the response — a function of the component rather than its deletion — might degrade red behavior while leaving the color cube intact, and it would still have side-effects boundable from the geometry beforehand, which is the property the method is for. Design question to settle before the first transformer intervention.

We already tried this in M1 (autoencoders): the "intervention lobes" work in `scratch-m1-code/docs/m2-control/ex-2.1-intervention-lobe.ipynb` and the paper appendix `references/sca1-paper/asec_intervention_lobes.tex` define a bounded falloff $h(\alpha)$ — zero below an alignment threshold $a$, ramping to max strength $b$ with shape controlled by power $p$ (plus a Bézier variant for the rotation-based "repulsion" intervention) — instead of the aggressive $h(\alpha)=\alpha$ used in the main M1 text. Worth pulling forward as a starting point rather than re-deriving.
