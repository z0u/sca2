---
status: open
tags: [reports]
opened: 2026-08-05
---
# Importable shared glossary for reports

Each report restates its glossary table by hand, which is how the cell/condition drift happened. A small module (e.g. `mini.reports.glossary` or `src/sca`) holding term → definition rows that a notebook imports and renders — each report selecting the terms it uses — would keep definitions identical across reports while staying self-contained when published. Needs care with memoization only if it lands in `experiment.py` inputs; keep it report-side.
