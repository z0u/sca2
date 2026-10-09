---
status: partial
tags: [D2.2, anchoring, representations, ex-2.2.23]
opened: 2026-10-08
---
# What e₁ holds besides the op

The spill of ex-2.2.23 lands on the ops that depend on how light their operands are, which suggests e₁ comes to hold some lightness along with `difference`. This asks what the states along e₁ carry, on ex-2.2.23's checkpoints, with no new training.

A short exploratory report, from one forward pass per checkpoint that keeps the states at every position and slice:

- A map along the context: the alignment with e₁, a probe for the op, and a probe for the color at each position, as stacked smooth-step charts (one row per slice), the control beside the anchored runs.
- Lightness on e₁: at the operands of contexts of the six other ops, where the token says nothing about the op, how closely the component along e₁ follows the lightness of the operand color, and whether that predicts how much each run spills. The 200- and 400-epoch runs, and the half-trained runs of ex-2.2.23, show whether it grows with training.
- The anchor at examples of other ops: [example-evidence](/docs/m2/example-evidence/report.py) found the alignment at an example answer following how well that one example fits `difference`. At the example answers of the other ops, does the same judgement fire, and do the ops where it fires most spill most?

For the probes, use cross-validation over contexts (closed-form leave-one-out for the ridge probe, k-fold for the logistic op probe) rather than a single half split. `difference` answers are darker than the operands on average (from the op rules), so lightness and the label go together on `difference` contexts, which is why the lightness measurement leaves them out.

Split from the withdrawn ex-2.2.24 draft (its E1, H1, H3, and E3; see [PR #255](https://github.com/z0u/sca2/pull/255)).

## Notes

**2026-10-08, spill-by-position** — [The position pass](/docs/m2/spill-by-position/report.py) bears on this item. The spill at the example answers lands mostly on `darken` and grows with the removal there; the operands and symbols spill onto `lighten` with no removal at all, so the lightness measurement at the operands looks like the right place to start. Most anchored runs also have one syntax embedding lying on e₁ (⏎, `?`, or `,`, a different one per run; listed in that report, E3), so the map should mark which token each run latched and keep those positions apart from the op. Across positions, the removal under a single-position edit follows the alignment closely on the runs without a latched separator.

**2026-10-09, Claude** — The [lean review](/docs/m2/embedding-lean/report.py) answers the second and third bullets. Lightness is on e₁ in the color embedding table itself on every anchored run (r ≈ −0.7 between lightness and the e₁ component; darker colors further along), about three times larger on runs with no latch, and it fades with depth; the spill tracks it. At the example answers of other ops the per-example "fits `difference`" judgement fires as on `difference` contexts, and the spill does not follow it. The map along the context with op and color probes (first bullet) is not done; what remains is whether it is still wanted once the pull is kept off the early slices.

