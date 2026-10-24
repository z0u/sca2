---
status: open
tags: [measurement, experiments]
opened: 2026-10-24
---
# The energy spectrum of the stream in the standard measurements

On every run we have checked, one direction of the residual stream holds about half its energy, with the syntax tokens on it, and Adam lines that direction up with a coordinate axis ([rotated-basis](/docs/m2/rotated-basis/report.py)). The [optimizer scout](/docs/m2/optimizer-scout/report.py) found the same spike along the principal directions of the stream under every optimizer it tried, where a measurement along the coordinate axes would miss it. Whether e₁ overlaps the spike decides how far the control baseline could be trusted ([reference-baseline](/docs/m2/reference-baseline/report.py)), and a spike that moved would change which axes are quiet.

So the measurement pass of each experiment could report the energy spectrum per slice: the share of energy on the top few principal directions (`principal_one` in the optimizer scout), the share on the top coordinate axes, and the cosine of e₁ with the top principal direction. It is one forward pass over the probe set, already run for the other measurements, and would show at a glance whether a new recipe or architecture changes the spike. Asked for in the review of the privileged-basis findings, as a way to track progress toward a more isotropic stream.
