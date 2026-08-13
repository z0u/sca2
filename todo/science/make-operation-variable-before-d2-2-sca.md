---
status: open
tags: [D2.2, task-grammar]
---
# Make the operation a variable before D2.2

Make the operation a variable before D2.2. `sca/data/colors.py` hardcodes one op (`mix`, spelled `+`); anchoring the operation only makes sense once there is more than one.

Add an operation table (name, surface form, grid fn with defined rounding — saturating add/subtract and screen all stay closed on 0..15), thread an `op` field through `Example`, and key the seen-pair bookkeeping on `(op, pair)`. Spell operators as words (`red mix blue = purple`), not symbols, so the operation concept is multi-token like the colors. No need to keep `+` compatible — each experiment retrains from scratch and carries its own control. Probe positions in `sca/compute/evaluation.py` assume the infix `a <op> b = ` frame; keep that frame.
