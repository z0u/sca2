---
status: open
tags: [D2.2, anchoring, intervention, selectivity, ex-2.2.23]
opened: 2026-10-08
---
# Does the recipe of record hold at 600 epochs?

Ex-2.2.23 found more spill at 400 epochs than at 200, on a recipe that pulled every slice. The [embedding-lean review](/docs/m2/embedding-lean/report.py) traced that spill to lightness leaning onto e₁ in the color embedding table, and on runs that lost their latch the lean was larger at 400 epochs than at 200. The recipe of record (`falloff` for now, from [split-anti-seeds](/docs/m2/split-anti-seeds/report.py)) holds the early slices off e₁ with the anti term and stays within the spill criterion at 400 epochs. Whether that still holds with more training is open: a longer run might give the lean more time to build, or the hold might keep it off.

A few runs of whatever the recipe of record is when this is picked up (`falloff`, or `step, soft τ` if [that question](./step-soft-as-recipe-of-record.md) goes its way) at 600 epochs, at seeds that already have 400-epoch runs, scored with removal, spill at every dose, and the slope of the color embedding table, as in split-anti-seeds.

Split from the withdrawn ex-2.2.24 draft (its H4; see [PR #255](https://github.com/z0u/sca2/pull/255)).

## Notes

**2026-10-13, holiday tidy-up** — Reworded for the recipe of record. The two items this waited on are done ([spill-by-position](/docs/m2/spill-by-position/report.py), and the lean review for what e₁ holds), and the question was first asked of a recipe that spilled at 400 epochs already.

**2026-10-24, backlog refresh** — [Spill-timing](/docs/m2/spill-timing/report.py) bears on this. The late spill it found grew while a seed learned `value-hsv`, after about epoch 100, and settled after epoch 200 on every seed; `falloff` learned `value-hsv` as late on some seeds and barely spilled on it. If spill comes mostly while ops are being learned, 200 more epochs would add little, but the lean in the embedding table grew between 200 and 400 epochs on the old recipe, so the question stays open at low priority.
