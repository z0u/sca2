---
status: open
tags: [D2.2, methodology]
opened: 2026-09-10
priority: high
---
# A scouting round over the backlog before the anchored-op experiments

Ex-2.2.3 left several open questions that each look like a half-day rather than a preregistered experiment: which red lines survive projection on a diverse op set, whether an untied readout clears the syntax rows, the τ against λ_a trade at the plateau, the shaped operator on the new grammar, and the labeller that also draws from the answer. We are behind schedule on D2.2 and the temptation is to press on to the anchored ops, but each of these bears on the operating point those experiments would build on.

The plan is one survey-like round in the science skill's sense: a single notebook (or one per question where the runs differ) that runs the cheapest version of each, five seeds or fewer, no gates and no verdicts, and records what was tried and what was seen. The write-up says up front that it is a scouting pass and why it is not thorough: the point is to know which of the questions changes the D2.2 design, so that the next preregistration is written on solid ground rather than on five seeds of a point whose selection rule missed a gate.

Candidates, roughly in the order they bear on D2.2: (1) the diverse op set ([item](diverse-operator-set-hue-saturation-brightness.md)), since the grammar is what every later experiment runs on; (2) the untied readout ([item](syntax-rows-carry-the-axis-via-tied-readout.md)), since it decides whether the operand-only edit is a workaround or the method; (3) the τ against λ_a sweep at fixed m_line, since the survey's proposals differ mostly there; (4) the answer-drawing labeller, filed under (c) of the labelling-span item. The record for each is the same shape: what we ran, one figure or table, what we would do differently, and whether it changes the design.

## Notes

**2026-09-10, ex-2.2.3 results discussion** — raised by Sandy after the ex-2.2.3 review: "we need to scout the terrain first", with the record kept plainly. The 20-seed recipe comparison (E6 of ex-2.2.3) was the first of these and ran as an addendum arm on the existing experiment, which is a cheap pattern worth reusing: add a condition, keep the frozen arms untouched, and label the section post hoc.

**2026-09-10, Claude** — the round is [ex-2.2.4](/docs/m2/ex-2.2.4/report.py). Candidate (1), the op set, is in; it needed no training. The record for it proposes a commutative table and a dependence filter for the removal statistic. (2) to (4) are still to run and land as further sections.

**2026-09-10, pilot (Fable)** — Candidate (4), the answer-drawing labeller, has run as [ex-2.2.6](/docs/m2/ex-2.2.6/report.py), together with the stochastic-rounding grammar variant as [ex-2.2.5](/docs/m2/ex-2.2.5/report.py); both are `pilot-` reports on the dev store, each with its own `experiment.py`, two or three seeds per arm, no gates. Neither changes the D2.2 design: the whole-line pull moves alignment onto the answer at no cost but leaves the redder-than-both answers intact under projection, and stochastic rounding leaves anchoring untouched while changing which accuracy statistic can be read. Details in the two items' notes ([labelling span](labeling-pull-span-variants-ex-2-1.md), [stochastic rounding](stochastic-rounding-of-off-grid-answers.md)). The pattern that worked: a small `experiment.py` that path-loads ex-2.2.3's module and reuses its tasks, so every statistic means what it meant there, and a report that reads production's arms from the public store as the many-seed reference.

**2026-09-11, Opus** — both pilots have been re-run on production under their ex-numbers (`ex-2.2.5`, `ex-2.2.6`), and their reports now read every result through `project_store()` rather than naming a bucket. The earlier note describes runs that lived in the dev store, which is an engineering sandbox and can be wiped; science runs, pilots included, belong on production. The numbers reproduced exactly — every figure and table in both reports is unchanged — so nothing in the readings above moves.

**2026-09-11, pilot (Fable)** — Candidate (2), the untied readout, has run as [ex-2.2.7](/docs/m2/ex-2.2.7/report.py), with Prep C (blocks-only anchoring) and a tied-table fix (the syntax rows held at zero after every step) beside it, and a scoring-only strip of the stored checkpoints in front. The readout is the mechanism, the row constraint is the fix to carry, and Prep C costs removal completeness; details in the [tied-readout item](syntax-rows-carry-the-axis-via-tied-readout.md). It changes the design in one place: the handover can read the full-position projection as its removal operator, which is the M3-shaped one, with the operand-only edit as the control. Candidate (3), τ against λ_a, now has a data point at λ_a 0.1: the row component there is 0.27 on `=`, against 0.93 at `t00`'s 0.56.
