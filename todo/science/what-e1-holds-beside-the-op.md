---
status: open
tags: [D2.2, anchoring, representations, ex-2.2.23]
opened: 2026-10-08
priority: high
---
# A map of what e₁ holds along the context

A short exploratory report, from one forward pass per checkpoint that keeps the states at every position and slice: the alignment with e₁, a probe for the op, and a probe for the color at each position, as stacked smooth-step charts (one row per slice), the control beside the anchored runs.

The item began with two more questions, whether lightness sits on e₁ and whether the per-example judgement fires on other ops; the [lean review](/docs/m2/embedding-lean/report.py) answered both (notes below). The map is what is left. On the recipe of record (`falloff`, from [split-anti-seeds](/docs/m2/split-anti-seeds/report.py)) it would bear on an open question: editing block 1 alone removes most of the op on every seed, though nothing pulls block 1 (late-pull E2, slice-terms E1). An op probe by slice and position would show where the op is readable before block 2.

For the probes, use cross-validation over contexts (closed-form leave-one-out for the ridge probe, k-fold for the logistic op probe) rather than a single half split. `difference` answers are darker than the operands on average (from the op rules), so lightness and the label go together on `difference` contexts, and a color probe there would partly find the op.

Split from the withdrawn ex-2.2.24 draft (its E1, H1, H3, and E3; see [PR #255](https://github.com/z0u/sca2/pull/255)).

## Notes

**2026-10-08, spill-by-position** — [The position pass](/docs/m2/spill-by-position/report.py) bears on this item. The spill at the example answers lands mostly on `darken` and grows with the removal there; the operands and symbols spill onto `lighten` with no removal at all, so the lightness measurement at the operands looks like the right place to start. Most anchored runs also have one syntax embedding lying on e₁ (⏎, `?`, or `,`, a different one per run; listed in that report, E3), so the map should mark which token each run latched and keep those positions apart from the op. Across positions, the removal under a single-position edit follows the alignment closely on the runs without a latched separator.

**2026-10-09, Claude** — The [lean review](/docs/m2/embedding-lean/report.py) answers the two questions on lightness and on the other ops. Lightness is on e₁ in the color embedding table itself on every anchored run (r ≈ −0.7 between lightness and the e₁ component; darker colors further along), about three times larger on runs with no latch, and it fades with depth; the spill tracks it. At the example answers of other ops the per-example "fits `difference`" judgement fires as on `difference` contexts, and the spill does not follow it. The map along the context with op and color probes is not done; what remains is whether it is still wanted once the pull is kept off the early slices.

**2026-10-11, housekeeping** — The [equals-evidence](/docs/m2/equals-evidence/experiment.py) pass (PR #266) stores the alignment with e₁ at every position and slice of the held-out contexts, on every run of ex-2.2.23. So the alignment rows of the map are already in the store, and the map would need only the op and color probes on top.

**2026-10-13, holiday tidy-up** — Retitled to the map, the one part not yet done, and pointed at the `falloff` runs, where the block-1 question gives it a use. Status back to `open`, since none of the map has been made. As with ex-2.2.23 (the 10-11 note), the split-anti-seeds trajectories keep the alignment at every slice and position of the probe set, so the `falloff` runs need only the probes.

**2026-10-24, backlog refresh** — On the shortlist, as the one item that would bear on why editing block 1 alone removes most of the op. [Spill-timing](/docs/m2/spill-timing/report.py) E4 is a first piece of the color half of the map, on `pool 1–3, soft τ`: at the example answers of the other ops e₁ holds almost no color, and removing it barely changes how well the color can be fitted from the state. The op probe is the part still missing.
