---
status: open
tags: [reports, skills, agents, methodology]
opened: 2026-09-02
bundle: skills-workflow
---

# Refactor the report-writing workflow

Currently, we have several skills and agent configs detailing different parts of the report-writing workflow, including several phases of review. It has become almost inscrutable.

Suggestion:

1. Map out the current state to understand what the full workflow is.
2. Redesign it, taking all steps into account and carefully considering roles and responsibilities, and which models are best suited to each part.
3. Re-write it, still as a collection of skills and agent configs, but maybe in a plugin — so they're all in one place and clearly related to each other. Follow the `style-md` and `style-skills`.
4. Document it, including a Mermaid flowchart, to make it more scrutable.
5. Ask for fresh-eyes reviews for both correctness and style (separately) with clean context.

These skills are the backbone of our work and they're worth getting right.

## Notes

**2026-10-06, PM** — [PR #241](https://github.com/z0u/sca2/pull/241) (merged 10-04) changed parts of the workflow without doing step 1 or 4. It made `prose-simplifier` and `report-restructure` run on Fable drafts only, added two fill-in skeletons (`template-scout.py`, since renamed `template-explore.py`, and `template-prereg.py`) and two exemplars under `.claude/skills/science/references/`, and moved decisions to after the results. So the map in step 1 should start from the current `science` skill and `references/review-passes.md`. The review agents changed only in what interpretation they allow. Still open: no single map or flowchart of the workflow exists yet, and the roles of the reviewers were not redesigned.

**2026-10-07, PM** — The `science` skill now makes a short exploratory report the default shape, after review praised the example-evidence report. Preregistration and surveys moved to `references/preregistration.md` and `references/surveys.md`, the scout template became `template-explore.py`, and `report-review` routes an exploratory report to the `results-reviewer`. That cuts the workflow a typical experiment walks through, so the map in step 1 has fewer paths to draw, though the reviewer roles are still as they were.

**2026-10-08, PM** — Review notes on the printed diff of PR #257 (the exploratory-report change), left for this refactor:

- `science` skill, the conventions list under "Writing any report": the shape is right, but it could be stated more clearly. Rewrite it in the lunchtime register, more like the example-evidence report. A proposed rule to fold in: tables are about 95% quantitative, prose about 95% qualitative, and figures bridge the two.
- The pointer to `exemplar-ex-2.2.18.md` "for the register": the register is evolving, so that exemplar may no longer be the one to point at.
- `template-explore.py`: drop "a backlog item" from where the question came from; make the `Observations` lines qualitative (in place of "at most one kind of number"); find another word for "yardstick"; drop the placeholder for runs left out; and say that what we do next is left unstated in the report, as well as being for the human to decide.
- `exemplar-example-evidence.md`: reword "bear on" (in the quoted Discussion).
