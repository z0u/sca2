---
name: style-terms
description: |
  Shared terminology for experiment reports: condition vs. cell, etc.
  Some terms differ slightly from convention, so always use when working on reports.
---

## Methodological terms

Use these terms consistently across all reports.

- factor

  A swept design parameter with named levels ("factor A, the anneal endpoint; levels 50, 70, and 90"). The factorial is the crossing of the factors.

- condition

  One combination of factor levels in a sweep or factorial design; seed-aggregated. "The `span-anti` condition", "every condition misses on grading". Never "cell".[^not-cell]

- run

  A condition crossed with a seed: one training run, the unit of replication. Three seeds per condition means three runs per condition.

- arm

  An extra desing parameter outside the factorial (or ladder), branching off one grid condition with one thing changed to answer one question (the star, timing, ceiling, and dose arms). Arms ride along, typically unscored.

- seed

  The replication factor. Prefer "seed mean" / "seed range" for aggregates over runs.

- criterion

  One clause of a hypothesis gate. Calling these "conditions" would collide with the design sense above and produces sentences like "every condition a condition misses".


[^not-cell]: In classical DoE a condition is called a cell, but we can't call it that because other senses of "cell" appear in reports and cannot be renamed away: cells of a table or heatmap ("each cell is the seed mean"), and Marimo notebook cells ("the analysis cells below"). Reports also legitimately use it for spatial grids (color-grid cells, Voronoi cells). So in prose, "cell" never means a condition or a run. In _code_, the stored key `metrics["cells"]` can keep its legacy name to avoid invalidating memo keys.
