# Surveys

Some questions are about choosing an operating point in a space too large to give every point a hypothesis: the anchor weight's selectivity optimum, the mellowmax temperature, how much repulsion to deliver and when. A *survey* is the experiment type for those. It preregisters the search plan instead of an outcome, and it scores nothing, so it builds on [preregistration.md](preregistration.md).

What makes a search credible is the same thing that makes preregistration work — the analysis was fixed before the data existed. So freeze the procedure in a `## Search plan` section, where a scored report's result sections would begin:

- **The space.** Each dimension with its bounds and its scale (log for weights and temperatures).
- **The sampling rule**, with its seed, so the trial list can be reviewed before the run. Prefer Sobol (`scipy.stats.qmc`) over uniform draws: it fills the box more evenly, so the one-dimensional marginals are less lumpy for the same budget. SciPy only reaches us through scikit-learn, so declare it when the first survey lands.
- **The trial budget**, and the `--budget` and `--max-containers` caps that hold it.
- **The objective.** Where objectives trade against each other — the usual case here, since anchor weight buys alignment and spends selectivity — state it as a constraint ("maximize m_line subject to holdout EM within `TASK_GATE` of control") and report the Pareto front rather than a scalar winner.
- **The seed budget:** how many seeds per trial in the first round, which band gets promoted, and to how many. Cut cost on the seed axis rather than by stopping runs early, because the margin peaks around epoch 10 and drifts down over the following forty, so a short-run proxy would favour configurations that look good early.
- **The noise floor of each objective**, measured first, and the resolution it licenses. A survey may not claim a difference it cannot resolve. Sometimes this is free: a past condition with many seeds gives the per-run spread of every statistic at that operating point, already in the store.
- **The stopping rule.**

With those fixed, everything the survey reports is a deterministic function of the data, so there is no forking-path problem. The only freedom left is which point wins, and that is the output rather than a claim.

Two rules make the type safe to publish.

**Nothing a survey reports may be quoted as a result.** It proposes an operating point; the next preregistered experiment adopts that point, scores it at fresh seeds, and reports the survey's value beside the confirmed one. The gap is the winner's-curse correction — a search's best trial wins partly on merit and partly on lucky seeds, so re-measuring is what turns a proposal into a number. That handoff already happens informally (ex-2.1.9 ran at ex-2.1.8's `end90-hold30` point); naming it makes the proposing half publishable.

**Publish every trial**, including the ones that went nowhere. Selective reporting is what would make a large search worthless, and a complete table settles it. Memoization means the data is there anyway. The trials a report publishes are on production storage; the dev pair is for prototyping, and a run made there is repeated on production before the freeze (the `mi-ni` skill's storage reference).

Then report the landscape rather than the winner. "The margin holds above 0.5 for λ_a anywhere in [0.05, 0.4]" is worth more than "0.12 was best": it is what the next milestone inherits, and a wide plateau is itself a result, since it says the method does not need careful tuning.

## A survey's report

The skeleton of a preregistered report, with three differences.

- `## Findings` becomes `## Observations` — same place, same brevity, but each line carries its noise floor where a verdict would carry its gate, and one line names the proposed operating point.
- The search plan is one `## Search plan` block, where a scored report's first result section would sit, since there are no hypotheses to spread over sections.
- `### Conditions` becomes the space specification plus the full trial table. This is the convention that has to bend: elsewhere the report imports hand-justified condition dicts and renders them as prose, which is why there's no generic grid builder, and a hundred trials can't each carry a docstring. So the justification attaches to the dimension rather than the level, and the trial table is generated from stored results.

Say "survey" in the first clause of the lede and label it the same way in `docs/index.md`. Numbering stays in the `ex-2.1.N` sequence.

`docs/ngpt-scaling/report.py` is the closest existing example — a width × depth grid, no hypotheses, and a conclusion about whether the region is safe to build on. A survey is that plus the frozen search plan, which a 3 × 3 didn't need.
