---
status: done
tags: [reports, skills, writing]
opened: 2026-09-10
closed: 2026-10-06
---
# Report register: write the explanation a colleague would get over lunch, from the first draft

More than once a report draft has been hard to follow, Sandy has asked "can you explain this report to me?", and the chat explanation that came back was what the report should have been. The writing, text-lint, prose-simplifier, and report-restructure skills have not moved the needle much on this, which suggests the fix is in the workflow rather than in another rule: the skills polish sentences, and the problem is the register the first draft is written in.

What the good chat explanations have in common: they lead with what we found in one plain sentence, they say what each number means before giving it, they use the same everyday words throughout (the model loses *red*; the answer stops depending on the red operand) instead of the statistic names, they gloss each statistic once in a phrase, and they say what we make of it and what we would do next. They do not open with the gate arithmetic, and they do not say "reads as" or "carries".

Proposed workflow change, to try on the next report: (1) before touching the notebook, write the Findings and Discussion as a message to Sandy, in the chat, as if explaining the results over lunch; (2) paste that into the report as the first draft of those sections, template expressions added afterwards; (3) only then run the polishing skills, and run them on the H sections rather than on the lunch text. A second, cheaper check: after the render, ask a fresh agent to explain the report back in plain words, and diff its explanation against the Findings; where they differ, the report is the one to change.

The skills should say this too: the writing skill's opening line becomes "write the lunch explanation first", the report-structure agent checks whether the Findings could be read aloud to a colleague, and the glossary style item (glossary of preferred terms, under eng) picks the everyday word for each statistic so the same phrase appears everywhere.

## Notes

**2026-09-10, Sandy, ex-2.2.3 review** — "the explanation you gave was excellent, and I would have loved that to be the actual report (right from the first draft). We have a lot of skills on report writing and reviewing and style, but somehow it hasn't moved the needle much." The ex-2.2.3 Discussion rewrite in the same round is the first trial of the lunch-first draft.

**2026-09-10, Claude** — the skills now say it: the science skill's "Collaborating on a report" section has the lunch-first draft as a workflow step, the writing skill opens with it and lists "reads as"/"carries" and statistic-names-for-things as anti-patterns, the report-structure agent checks whether Findings could be read aloud, and the read-back check is in `review-passes.md`. Partial until the next report's Findings and Discussion are written this way from the first draft; that trial closes it.

**2026-09-15, housekeeping** — Four reports have landed since the skills changed (ex-2.2.5 through ex-2.2.8), so it is worth saying where the trial stands rather than leaving the item reading as if nothing had been tried.

The lede blocks have moved. Ex-2.2.7's `tl;dr` is three short paragraphs of plain words with no statistic in them — "the readout puts it there, to predict `=` after a red operand, and the tied table passes it to the embedding" — and reads close to the lunch explanation this item describes. The Observations bullets under it have not moved as far: they still open on the statistic ("**Placement.** m_line runs ...", "ᾱ at op1, the containment read") rather than on what the number means. That looks like the register taking hold at the top of a report first, which is the cheapest place for it and also the place a reader reaches first.

The four are pilots and surveys, though, so none of them is quite the trial this item names. The one to watch is ex-2.2.9, drafted in [PR #176](https://github.com/z0u/sca2/pull/176): it is Findings-first with a Glossary section, so the shape is right, but its Findings are written after the run. That write-up, done lunch-first from the first draft, is what closes this item — so leave it shortlisted until then.

**2026-09-22, Fable** — Where the trial stands after ex-2.2.9 through ex-2.2.11. Ex-2.2.11's Discussion reads in the register this item asks for: it opens "the handover setup does what ex-2.2.9 said it does, with one exception we can now name", says what the number means before the number ("still keeps about a quarter of the accuracy it had on the lines that need a red hue there, a little more than the gate allows"), and closes with what we would do next. Its tl;dr does the same. The Findings block is generated from the results (`findings_md`), which is where the statistic names still lead. Whether the chat draft came before the report draft is the one thing the commits cannot show, so this stays partial until Sandy says whether ex-2.2.11's Discussion is the lunch explanation they wanted; if it is, the item can close, since the skills already carry the workflow step. Separately, the science skill's Findings example, which carried a note pointing here as outdated, now leads each line with what the number means.

**2026-10-06, PM** — Closing. [PR #241](https://github.com/z0u/sca2/pull/241) (merged 10-04) settled the question this item was waiting on: Sandy chose ex-2.2.18 as the model report and the ex-2.2.21 companion notes as the example of the explanatory register, and both now sit in the `science` skill as exemplars that a drafter reads before writing (`references/exemplar-ex-2.2.18.md`, `references/exemplar-explanation.md`). The lunch-first step is still in the `science` and `writing` skills. An exemplar shows the register where the rules only described it, which is what this item asked for. If a later draft slips back, that is a new item about the exemplars.
