---
status: open
tags: [D2.2, anchoring]
opened: 2026-09-24
---
# If the query `?` saturates

In the [D2.2 pivot](/docs/m2/d2.2/pivot.md), the pooled anchor term concentrates the pull where alignment comes most easily. The query `?` is a constant token with nothing else to hold, so the model may push it to a cosine near 1 on e₁. That is still an inferred op, but a state that is all concept has no partial dose, which is the collapse ex-2.2.14 found at the op word.

If it happens and we would rather it did not, the anchor term has three options:

- Cap the pull: a hinge that is zero above a target alignment, in place of 1 − cos, so no state is asked to be all concept.
- A larger τ, which spreads the pull over the span of the line.
- A mask that pulls only positions that also hold something else, `=` and the answer, where the state has to hold the answer as well.

[The pilot](/docs/m2/d2.2/design.md#the-pilot) has a hinge-capped arm beside the uncapped one, so if `?` saturates, the first option has already been tried.
