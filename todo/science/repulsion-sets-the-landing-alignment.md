---
status: open
tags: [D2.2, anchoring, ex-2.2.1, evaluation]
opened: 2026-09-03
---
# Repulsion: shape where the state lands, rather than how much is removed

[Ex-2.2.1](/docs/m2/ex-2.2.1/report.py) ran M1's suppression lobe as an arm: remove a fraction $h(\alpha)$ of the e₁ component, zero below a threshold $a = 0.5$ and rising linearly to all of it at $\alpha = 1$, then re-normalize. The re-normalization is what makes the operator a rotation, and it also decides where the state lands, which the lobe does not control. Three things followed, all in the report's operator figure and E4.

- A pure-red operand arriving at $\alpha = 0.9$ leaves at 0.38, past the threshold and at about the alignment the `+` and `=` embeddings carry. Above the threshold the order reverses: the more aligned a state arrives, the less aligned it leaves, down to zero at $\alpha = 1$.
- The blocks then bring the red operand partway back. At slices 1–3 it arrives at a mean alignment of about 0.33 under the lobe (0.52 clean), and about half of those states sit above the threshold and are written again.
- The lobe removes about half of the red response (red accuracy about 0.6 against 0.09 under the projection), at no non-red cost. What it decodes to is still a color: off-vocabulary mass on red lines is 0.002, 60% of decodes are the true answer, and 29% are a one-step neighbor.

M1's *repulsion* form (`references/sca1-paper/asec_intervention_lobes.tex`, and `docs/m2/ex-2.2.1/scratch-repulsion.md` for the ex-preppy code) parameterizes the landing alignment instead: $\mathbf{x}' = m(\alpha)\,\mathbf{v} + \sqrt{1 - m(\alpha)^2}\,\mathbf{u}_\perp$, with a linear or Bézier mapper $m$ that leaves $\alpha \le a$ alone and maps the rest into $[a, b]$. It is order-preserving, bounded, and its output is a known function of its input. In M1 the case for it was staying on the sphere, which suppression did not; here both operators re-normalize, so that distinction is gone and what is left is which quantity is shaped.

Two readings of what that buys, and they pull in different directions.

- For *removal*, the suppression lobe is the stronger operator at a given threshold: repulsion with a ceiling at $a = 0.5$ would leave pure red at 0.5, above where the lobe puts it, so it would bite less. Ex-2.2.1's data also says the worry that a state pushed past the threshold behaves at random is not what happens in this model: under the projection, the far case, 98.5% of the wrong decodes are what the line would mix to with a less red operand, and off-vocabulary mass stays small.
- For a *graded* edit, repulsion is the natural form. The D2.2 *suppress operation* design grades its dose by intervention strength (γ on the projection). A target alignment is a dose with a meaning in the clean map, and repulsion's mapper is that dial. It is also the operator whose side-effects are easiest to bound, since where every state lands is fixed in advance.

What it would take: a `repulsion` operator in `sca.intervention` beside `projection` and `lobe` (a few lines; the M1 code above is the reference), a contract check that its write equals the angle between $\alpha$ and $m(\alpha)$, and an arm on ex-2.2.1 or on the first graded experiment. The ex-2.2.1 runs are stored, so an extra arm is one more scoring pass per run rather than a rerun.

Sibling of the [shaped-suppression item](./shaped-suppression-rather-than-projecting-whole-axis.md), which owns the choice of operator for the anchored-op experiments.
