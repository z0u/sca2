---
status: open
tags: [reports, vis]
opened: 2026-07-16
---
# Responsive multi-panel figures in reports

The ex-2.1.1 two-panel named-pair lattice was split into two independent `themed` figures wrapped in a `.report-figure-row` (inline-block, reflows to a stack on narrow screens; matched size via a shared projection + pinned limits + full-figure bbox rather than `sharey`). That pattern works when the panels carry no shared axis labels and share only a scale. Still undecided for the remaining wide plots — `accuracy-sweep` (1×4) and `probe-r2` (1×3), which shrink illegibly on phones: (a) split like the lattice, but then we must manage the shared y-axis label and legend that currently live only on the leftmost panel; or (b) keep them single figures and give each a declared native width (e.g. `style="--mini-fig-width: 700px"`) that a wrapper turns into a `min-width` + horizontal scroll box, so they scroll instead of shrinking below legibility (mirrors the `.report-table-scroll` fix). Option
(b) is less disruptive and generalizes; the open question is where the min-width/scroll wrapper lives — a `themed` option, or a CSS class the author opts into. Decide before the anchoring reports reuse these figures.
