---
status: open
tags: [sci-report, templates]
opened: 2026-10-09
---
# A place for the method in exploratory reports

The exploratory template has `Scope` stand in for `Parameters` and `Method`, on the reasoning that nothing in a method section would change how a reader takes a result ([exemplar-example-evidence](/.claude/skills/sci-report/references/exemplar-example-evidence.md)). That held for re-analyses of stored runs, but in experiments that train new runs the Scope section keeps filling up with what the runs held fixed and what changed (the weight schedules, the per-slice pull), which does change how a reader takes the results. In review of [late pull](/docs/m2/late-pull/report.py) the section was "becoming scope+params/method", and the question was whether the template is missing something.

Late pull now has `Scope` for what the study can and cannot show (seeds, preregistration, reproduction) and `Method` for the runs and the measurements. If that reads well, the template and the `sci-report` skill could make the same split for exploratory reports that train: `Scope` for limits, then `Method` (the arms, what is held at the recipe, and the measurements), short.
