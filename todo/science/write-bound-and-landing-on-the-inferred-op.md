---
status: done
closed: 2026-10-22
tags: [D2.2, anchoring, intervention]
opened: 2026-10-13
---
# The write bound and the landing on the inferred op

With a recipe of record that holds at fresh seeds (split-anti-seeds, `falloff`), two of the claims in the [D2.2 design doc](/docs/m2/d2.2/design.md) still have no experiment behind them.

- The write bound, claim (d): because we placed the axis, we can say in advance how far an edit moves each state. This held on *red* in ex-2.2.1 (H4) and has not been measured for the inferred op.
- Where the edited model lands, claim (c): the full edit covers about half of the distance to the target null, context by context (landing 0.52 on `falloff`). Is that enough, or is a trained fallback wanted to close the rest?

The design doc has the plan for both under "What comes next"; this item is the backlog entry for them, so the shortlist names what D2.2 still has to show beside the recipe sweep.

## Notes

**2026-10-21, design refresh** — The design doc now proposes this as a re-analysis of the stored `falloff` runs at seeds 704 to 711, with no new training, and adds the per-context grading of claim (b) to it: whether the drop on each context follows how much its answer depended on `difference`. A review on the stored late-pull runs measured that informally (unpublished), so the measurement exists in sketch. The landing half splits into the measurement here and the follow-ups in [the reference model](./reference-model-without-the-anchored-op.md) and [fallback control](./fallback-control-on-the-untied-recipe.md).

**2026-10-22, edit-effects** — Measured in the [edit-effects report](/docs/m2/edit-effects/report.py) (PR #279) on the `falloff` runs at seeds 704 to 711. The write stays within the bound (at most 1.08 times it) and the blocks after an edit put back less e₁ than the clean model or the control has. The drop is graded (share removed 0.75 to 0.88 by posterior bin, net of the control). The landing is 0.52, with the distance left in how the edited model weighs the other ops context by context. The follow-ups are [the reference model](./reference-model-without-the-anchored-op.md) and [the op in attention](./is-the-inferred-op-held-in-attention.md).
