---
status: open
tags: [D2.2, scoring]
opened: 2026-10-24
---
# The median baseline in the shared scoring

The D2.2 design now nets removal and spill against the median drop over the edit along every coordinate axis but e₁, taken in the control, with the same median in the anchored run beside it ([reference-baseline](/docs/m2/reference-baseline/report.py)). The shared dose scoring (`suppress_one` in ex-2.2.22, which later experiments reuse through their sibling modules) still scores e₁ alone, and each report nets against e₁ in the control. So a new experiment built on the last one would quote the paired baseline unless its author remembers the rule.

The fix is to give the shared scoring the reference axes: the drop per op and dose along every other axis (or a fixed draw of them, for speed), and a helper that returns removal and spill net of the median, beside the paired and raw numbers. Scoring every axis at four doses took a few minutes per checkpoint on an L4 in reference-baseline (`axes_one`), so a draw of about sixteen axes may be the better default, since a draw of eight varied by up to 0.06 in removal there.

A change to `suppress_one` changes the memo evidence of every experiment that scores through it, so it is cleaner as a new function beside it.
