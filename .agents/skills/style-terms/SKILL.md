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

  An extra design parameter outside the factorial (or ladder), branching off one grid condition with one thing changed to answer one question (the star, timing, ceiling, and dose arms). Arms ride along, typically unscored.

- trial

  One sampled point in a survey's search space; seed-aggregated, like a condition. "Condition" implies named levels chosen in advance, which a sampled point doesn't have, so use "trial" wherever the point came out of a sampling rule. Surveys have trials and no arms. See the `science` skill for the experiment type.

- seed

  The replication factor. Prefer "seed mean" / "seed range" for aggregates over runs.

- criterion

  One clause of a hypothesis gate. Calling these "conditions" would collide with the design sense above and produces sentences like "every condition a condition misses".

## Model and measurement terms

- slice

  One of the L+1 readable points along the residual stream: the token embedding, then each block's output. "Slice" rather than "layer" because the embedding is not a layer, and an L-layer model has L+1 slices. "The embedding slice", "the four slices after the embedding". *Depth* names the axis the slices sit on, not a unit ("the concept migrates with depth").

- layer-mean

  The mean over all residual-stream slices of a per-slice statistic ("the layer-mean margin"). A historical name from ex-2.1.6 onward, even though it averages L+1 slices; keep it only where continuity with an old statistic matters (m_op1), and describe new statistics as "the mean over slices" instead (ex-2.1.9's m_span does this).

- role

  The job a position plays in a line: op1, `+`, op2, `=`, answer, newline. Positions are crop-relative and shift with every batch; roles are line-relative, so pulls, masks, and measurements are keyed by role. "Span roles" are the four prompt roles the anchor term can act on.

- R² vs r²

  Two statistics that earlier reports both wrote as R². Write $R^2$ for a probe's held-out coefficient of determination (can be negative; measures a fitted readout), and $r^2$ for a squared Pearson correlation (bounded to [0, 1]; measures proportionality, e.g. the grading track). Say which one in prose on first use.


[^not-cell]: In classical DoE a condition is called a cell, but we can't call it that because other senses of "cell" appear in reports and cannot be renamed away: cells of a table or heatmap ("each cell is the seed mean"), and Marimo notebook cells ("the analysis cells below"). Reports also legitimately use it for spatial grids (color-grid cells, Voronoi cells). So in prose, "cell" never means a condition or a run. In _code_, the stored key `metrics["cells"]` can keep its legacy name to avoid invalidating memo keys.
