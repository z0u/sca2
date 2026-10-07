---
status: open
tags: [reports, tooling]
opened: 2026-10-01
---
# Mandatory `# %%` cell markers in literate scripts

Sandy suggested this in the review of the ex-2.2.20 preregistration draft. Today a literate script (`mini.lit`) finds its cell boundaries implicitly, so where one cell ends depends on what the statements around it are. Two small slips in that draft came from this:

- A figure call followed by a table call in the same stretch of code failed to render: the value of the first expression would be shown, but it was not the last statement of its cell. It needed a `# %%` between them.
- A docstring under an assignment was printed as a paragraph at the top of the page (`scripts/trailing_cell_docstrings.py` now catches this case in `./go lint`, and CI caught it here).

Requiring an explicit `# %%` before every cell would make each boundary visible in the source, and both slips would become an error at the point of writing. Open questions: whether a prose string opens its own cell or needs a marker too; how much churn the change makes across the existing reports (an automatic rewrite could insert the markers where the implicit rule puts them today); and whether a lint that requires the marker is enough, without changing the weaver.
