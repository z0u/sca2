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

**2026-08-26, housekeeping** — Still true as written: `src/sca/data/colors.py` has one `mix`, spelled `+`, and `Example` carries `prompt`/`answer` with no `op` field, so nothing here has been quietly done. Re-checking because the shortlist has changed shape around it — four of the six slots are now D2.2-tagged, where this was the only one in August.

Worth naming the split, since the four read as one queue and aren't. This item is the only one of the four that is code, so it needs no decision to start and blocks nothing while it waits. The other three (baseline comparisons, shaped suppression, survey-format lessons) are all inputs to how D2.2's first prereg gets written, and their value is highest before that draft exists. So the plumbing here can proceed in parallel with the planning rather than queueing behind it.
