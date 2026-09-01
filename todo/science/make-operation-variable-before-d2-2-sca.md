---
status: open
tags: [D2.2, task-grammar]
priority: high
---
# Make the operation a variable before D2.2

Make the operation a variable before D2.2. `sca/data/colors.py` hardcodes one op (`mix`, spelled `+`); anchoring the operation only makes sense once there is more than one.

Add an operation table (name, surface form, grid fn with defined rounding — saturating add/subtract and screen all stay closed on 0..15), thread an `op` field through `Example`, and key the seen-pair bookkeeping on `(op, pair)`. Spell operators as words (`red mix blue = purple`), not symbols, so the operation concept is multi-token like the colors. No need to keep `+` compatible — each experiment retrains from scratch and carries its own control. Probe positions in `sca/compute/evaluation.py` assume the infix `a <op> b = ` frame; keep that frame.

## Notes

**2026-08-16, housekeeping** — D2.1 has closed out (ex-2.1.11's survey), and this is the one live D2.2-tagged item and explicit prerequisite plumbing for it (D2.2 needs more than one operation to anchor). Priority list was empty (0/6), so promoting this as the natural next-up item rather than leaving the shortlist blank.

**2026-08-30, housekeeping** — the [D2.2 design](/docs/m2/d2.2/design.md) now fixes the first table, so whoever writes this code should read its deps section before choosing ops: `mix` (the default), saturating `add`, `screen`, `multiply`. The choice is not arbitrary and the reasoning is worth keeping — one op departs from `mix` on most pairs, so the model has to read the op at all, and the others depart only sometimes, which is what makes the D2.2 dose label graded rather than binary. `divide` is excluded (it needs a saturation rule and is lumpy on a 16-level grid); ops in other color spaces (`hue`, `saturation`, `brightness`) are filed as [a separate question](./richer-op-set-operand-geometry.md). The design also picks `screen` as the first op to anchor, so that is the one whose surface form and rounding want the most care.

Still the only code-shaped item on the shortlist, still unstarted, and I re-checked that it is untouched: `src/sca/data/colors.py` has one op spelled `+`, and `Example` carries `prompt`/`answer` with no `op` field.

**2026-09-01, design review** — two corrections to the 08-30 note. The design's latest revision dropped the "anchor `screen` first" decision: the choice of anchored op is now recorded as open until this item lands, so read the design's Decisions section rather than that note. And the gradedness rationale was wrong, checked numerically: with round-to-nearest on the 0..15 grid, *all three* ops depart from `mix` on 98–99% of pairs (the only sizable agreement anywhere is `add`–`screen`, ≈36% of pairs, at low intensities and saturation), so *op-relevance* occupies {0, ½, ¾} at roughly 1%/35%/64% of lines when anchoring `add` or `screen`, and is nearly binary (98% of lines at ¾) when anchoring `mix` or `multiply`. The numbers shift a little with the rounding rule this item defines, so the design now asks for the relevance distribution per candidate anchored op to be computed and quoted when the table lands; partly because of this, the design's dose axis moved from stimulus (relevance) to intervention strength.
