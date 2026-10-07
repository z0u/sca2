---
status: partial
tags: [reports]
opened: 2026-08-05
---
# Importable shared glossary for reports

Each report restates its glossary table by hand, which is how the cell/condition drift happened. A small module (e.g. `mini.reports.glossary` or `src/sca`) holding term → definition rows that a notebook imports and renders — each report selecting the terms it uses — would keep definitions identical across reports while staying self-contained when published. Needs care with memoization only if it lands in `experiment.py` inputs; keep it report-side.

## Notes

**2026-09-27, Opus (with Sandy)** — The dictionary exists as `docs/glossary.md` (a Markdown definition list; it began as TOML), read at render time rather than imported: `mini.lit.notes` annotates the first use of each term per section with its definition (margin note, or hover on a narrow screen), taking the report's own `## Glossary` `<dl>` first and the shared file second. It is seeded with two terms (expected exact match, band). Still open: moving the definitions that recur across reports (removal lines, kept share, op margin, retention) into the shared file, which needs a pass over how their meanings drifted (ex-2.2.11 changed three), and letting a report's glossary section list shared terms without restating them.
