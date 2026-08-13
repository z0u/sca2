---
status: finding
tags: [representations, anchoring, ex-2.1.5, task-grammar]
opened: 2026-07-26
---
# A named operand's value lives at the space that closes it, not on its characters — and the landmark set doesn't measure those spaces

Under the strict holdout, named operand 1 scores −0.05 at its own last character in the last layer, but +0.35 at the space that follows it and +0.47 at the space after the `+`; operand 2 scores +0.30 at the space before the `=`. So the earlier reading that named operands carry no value geometry was an artefact of where we measured. A variable-length name has no fixed slot, so the model appears to resolve it into the delimiter — a summary position — and the value stays there across the operator. Hex does the opposite: at the same spaces it scores −0.19 to −0.34, nothing at all, which fits a form whose three digits already sit at fixed offsets and need no summary slot. `LANDMARKS` measures the operator characters and the pre-answer space but not the two operand-closing spaces, so the current figure omits the named form's most informative sites. Add them before the H2 rewrite. Bears on anchor placement: this is a stable single-position home for a named operand's value, and hex has no counterpart.

Qualified by a word-family control (2026-07-26). 99 of the 140 names are multi-word and modifiers recur (`green` in 26, `light` in 11), so part of what a delimiter holds may be "this name contained *green*" rather than a resolved color. Holding out whole word families tests it, but names sharing a word also share a region of the cube, so the holdout removes a color cluster and the drop would be ambiguous — hence a color-matched control that removes the same number of names by RGB proximity instead. The contrast (word − color, last layer, 3 seeds):

| target, site | value | color-matched | word-family | word − color |
|---|---|---|---|---|
| op1 @ its own last character | +0.245 | −0.297 | −0.831 | −0.534 |
| op1 @ its closing space | +0.320 | −0.062 | −0.488 | −0.426 |
| op2 @ its closing space | +0.372 | +0.017 | −0.353 | −0.370 |
| op1 @ the space after `+` | +0.559 | +0.284 | +0.209 | −0.075 |
| mix @ pre-answer | +0.879 | +0.644 | +0.772 | +0.127 |

So a name's own closing space is substantially lexical, and "the model resolves the name into the delimiter" was too clean a story. Two things survive it: one token further on, at the space after the `+`, the lexical dependence is small and the value still reads +0.209 under a holdout that removes 14 sibling names at the median (52 at the max); and the mix at pre-answer shows none at all — its sign flips, so H2's headline is not a lexical artefact. A bias runs against this reading, which makes it sturdier: modifier families like `light` are spread across the cube rather than clustered, so for those names the word holdout removes a scattered set that is easier to extrapolate from than the tight cluster the control removes.

(Numbers here are restricted to the 108 names that have a family, and group the mix by the snapped answer name rather than by the exact mix value, so they are not comparable cell-for-cell with the strict-holdout table above — only within this table.)
