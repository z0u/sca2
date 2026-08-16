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
