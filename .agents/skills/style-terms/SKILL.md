---
name: style-terms
description: |
  Shared terminology for writing about the experiments: condition vs. cell, measurement vs. read, context/example/query, state vs. embedding, etc.
  Some terms differ from convention, so always use before writing reports, design docs, todo items, PR bodies, or code names and docstrings for corpus units.
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

- measurement

  A number taken from a run or a set of runs, together with the rule that produces it: "the removal measurement", "the three measurements ex-2.2.10 changed". Never "read" as a noun. It collides with the verb on the same line ("the read is read on `handover`") and with the reader's own reading, and the human finds it confusing. For "read on" or "read against" as verbs, prefer "scored on", "measured on", or "compared with".

## Indexing

Math and prose counts from 1; code counts from 0. So the anchored direction is *the first axis* or e₁ in a report and `ANCHOR_AXIS = 0`. Slight preference for "the first basis vector" over "basis vector 1", etc.

The slice index $\ell$ is not an exception to this, though it starts at 0. It counts *blocks applied*, so $\ell = 0$ is the token embedding, labeled "emb" in figures.

## Corpus and sequence terms

From the smallest unit to the largest. The in-context grammar (the [D2.2 pivot](/docs/m2/d2.2/pivot.md)) adds context, example, and query; before it, a line held one equation.

- token

  One item of the vocabulary, and one entry of the corpus: a color word, an op word, a symbol (`+`, `?`, `=`, `,`), or `\n`.

- position

  A token's index in a window. Positions are window-relative and shift with every crop, so pulls, masks, and measurements are keyed by role instead (below).

- equation

  One application of an op: `op1 ? op2 = answer` (or `op1 difference op2 = answer` in the grammars that name the op). A solved equation shows its answer; the query leaves it blank.

- line

  A stretch of the corpus ending in `\n`: the unit the labeller draws on, the holdouts split on, and per-line scoring scores. Before the in-context grammar a line is one equation; after it, a line is one context.

- context

  In the in-context grammar, one line: a few examples of one op, then a query. Its op is inferred from the examples. It never means the model's input window (say *window*), or the loose sense of "the surroundings of a token".

- example

  A solved equation inside a context, the evidence the op is inferred from. Prefer "example" to "shot", except in compounds like "few-shot".

- query

  The last equation of a context, whose answer the model completes. It is always clean.

- replacement op noise

  Showing, in some examples, the answer another op would give in place of the answer under the true op, at a *replacement rate*. It spreads the posterior over ops, so it is what grades the stimulus. Distinguish it from a wrong answer drawn at random, which no op produces, and from *label noise*, which is the labeller's error and leaves the corpus as it is.

- window

  The `block_size` tokens the model reads in one forward pass: a random crop of the packed corpus in training, which can cut a line or a context short. Code calls it `seq` or a crop; in prose use "window", and keep "sequence" for the general sense.

- batch

  A stack of windows, trained on in one optimizer step.

- corpus

  The whole packed token array a run trains on, lines joined by `\n`.

- sample

  A verb ("sample a batch"), or a statistical sample (a set of draws). Never a unit of the corpus: say token, equation, line, context, or window.

## Model and measurement terms

- slice

  One of the L+1 readable points along the residual stream: the token embedding, then each block's output. "Slice" rather than "layer" because the embedding is not a layer, and an L-layer model has L+1 slices. "The embedding slice", "the four slices after the embedding". *Depth* names the axis the slices sit on, not a unit ("the concept migrates with depth").

- layer-mean

  The mean over all residual-stream slices of a per-slice statistic ("the layer-mean margin"). A historical name from ex-2.1.6 onward, even though it averages L+1 slices; keep it only where continuity with an old statistic matters (m_op1), and describe new statistics as "the mean over slices" instead (ex-2.1.9's m_span does this).

- embedding, state, and row

  The vector of a token in the embedding table is *its embedding*: "the `=` embedding", "the syntax embeddings", "the red color embeddings". The whole matrix is *the embedding table*. Its counterpart on the output side is *the readout vector* for a token, and *the readout table*. Following a block, a position holds a *state* (the residual-stream state at slice ℓ), which is contextual and is never called an embedding; "the `=` embedding" and "the state at `=`" are different objects, and the contrast between them is important.[^not-row]

- role

  The job a position plays in a line: op1, `+`, op2, `=`, answer, newline. Positions are window-relative and shift with every batch; roles are line-relative, so pulls, masks, and measurements are keyed by role. In the in-context grammar a role also needs the equation it sits in. Name the equation as a qualifier, as with "the `=` embedding": "the query `=`", "the query `?`", "an example answer", "the answer of the first example". Avoid the possessive ("the query's `=`"), per the `writing` skill. "Span roles" are the four prompt roles the anchor term can act on.

- op1, op2 vs. op

  `op1` and `op2` are the **operand** roles, and always have a digit. Bare `op` is the **operation**, matching the usual reading outside the repo. In prose, usually write "operand", and reserve the abbreviations for role names, code, and compounds.

- R² vs r²

  Two statistics that earlier reports both wrote as R². Write $R^2$ for a probe's held-out coefficient of determination (can be negative; measures a fitted readout), and $r^2$ for a squared Pearson correlation (bounded to [0, 1]; measures proportionality, e.g. the grading track). Say which one in prose on first use.

- s₂ (surprise-surprise)

  A **per-token** statistic: a token's surprisal minus the entropy of the distribution that predicted it, over $\log |V|$ — how much more surprised the model was than it expected to be. Pooling is a separate choice made afterwards, and the statistic is linear in both parts, so the order doesn't matter (`sca.compute.evaluation.answer_calibration` means over the answer characters). What it buys over bare surprisal is that it nets out the surprise the model itself anticipated, so a genuinely unpredictable position doesn't count against a model that knew it was unpredictable.

  It is zero in expectation under the model's _own_ distribution, which is what makes it a calibration statistic and not an accuracy one — a uniform model reads exactly 0 at every token, since its surprisal and its entropy are both $\log |V|$ whatever comes next. So s₂ alone says nothing about whether the model is any good; pair it with accuracy or surprisal unless competence is already established elsewhere.

  Read the sign, and expect an asymmetric scale. Positive is confidently wrong and unbounded (surprisal has no ceiling); negative is a hedge the outcome didn't need, and is bounded by $-h/\log|V|$, so it stays shallow while the model is mostly confident. Measured over ex-2.1.1's captured rows: 80% of positions are negative but the median is −0.001, only 2% fall below −0.15, and the floor is −0.28 against a maximum of +6.2. Those deep-negative positions are overwhelmingly the first character of a word, where the model is unsure _which_ word but the character is shared across its candidates (`black + ` → `b`).

  It is not a KL divergence, despite the shape. It is $H(p,q) - H(q)$, where a divergence from the data would be $H(p,q) - H(p)$; against a near-deterministic truth the two disagree in sign, so don't describe it as a distance from the data.


[^not-cell]: In classical DoE a condition is called a cell, but we can't call it that because other senses of "cell" appear in reports and cannot be renamed away: cells of a table or heatmap ("each cell is the seed mean"), and a literate script's own cells ("the analysis cells below"). Reports also legitimately use it for spatial grids (color-grid cells, Voronoi cells). So in prose, "cell" never means a condition or a run. In _code_, the stored key `metrics["cells"]` can keep its legacy name to avoid invalidating memo keys.

[^not-row]: Like "cell", "row" is heavily overloaded. So in prose, "row" never means a token embedding. It keeps the senses that can't be renamed away: rows of a table, a heatmap, or a figure grid, and the per-run records that code stores as `metrics["rows"]`.
