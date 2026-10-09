---
name: sci-report
description: |
  What an experiment report is: the exploratory and preregistered shapes, their templates and exemplars, the writing conventions, the lede, and the notes that carry review decisions between rounds. Read before drafting or reviewing a report or a plan.
---

This is about the document. How a report moves from idea to merge, and who reviews it when, is in the `science` skill.

## The exploratory report

The example-evidence report is the model: an eight-page re-analysis with three short sections and a discussion. Read [references/exemplar-example-evidence.md](references/exemplar-example-evidence.md) before drafting, and copy [references/template-explore.py](references/template-explore.py). Both stored-data re-analyses and studies with a few new runs take this shape (the ex-2.2.18 scout is one with new runs).

Its order is: the lede; one paragraph saying where the question came from and what the report reads or runs; `Observations`; `Scope`; the measurements; three or so E sections; and `Discussion`.

- One question, ideally a puzzle a reader can hold in mind. The paragraph under the lede is the whole "why", with links to the earlier report or result that raised it. Add a `Why this experiment` section only when the question needs a picture of the model to follow from (as in ex-2.2.18).
- `Observations` sits where a preregistered report has `Findings`: one line per E section, linked to it, saying what we saw in words, with a number only where no word gives the size. A reader who stops there has the result.
- `Scope` says which runs and data the report covers, how far a difference between runs can be trusted, and what was left out and why. It usually replaces `Parameters` and `Method`. Keep a method section only for detail that changes how a reader takes a result.
- Each E section asks one thing, usually the thing the last one left open. It opens with the question, shows one figure or table, says what we saw, and says what we make of it. A section that doesn't feed the discussion belongs in another report, or nowhere.
- No hypotheses, gates, verdicts, or `Decision`. Choices the results inform are made with the human afterwards, in conversation or a todo item, and stay out of the report.
- About eight printed pages. If a draft runs well past that, it is probably two questions; split it.

The report starts as the plan. Before anything runs, fill in the template with the question, where it came from, the runs or stored data and what each comparison separates, the measurements, and the question each E section will ask, leaving the observations as `TODO`. Put the cost in the PR description. That file is what the assumptions review reads and the human agrees to. Once results exist, the same file becomes the report. Nothing is frozen, so the plan can change as the results come in; say so in the PR when it does.

## Writing any report

Write the whole report the way you would explain it to a colleague over lunch. Lead with what we found in one plain sentence, say what a number means before giving it, and use the same everyday words for a thing throughout (the model loses *red*; the answer stops depending on the red operand) rather than the names of the statistics, glossing each statistic once in a phrase. Sentence-level polish doesn't change the register a draft was written in, so start in this one. For the index under the lede and the discussion, it helps to write them first as a message to the human, and paste that in as the first draft.

A report has three kinds of material, and each does one job. Tables hold the numbers. Prose says what they mean, in words: "the earlier examples barely count". Figures sit between the two, so a reader sees the size of a thing without reading digits. Nearly every table gets a figure beside it, because a table on its own looks like a wall of text.

Most numbers don't belong in prose. A sentence with three numbers in it is one a reader's eyes slide past. So each section uses one main measure, says which in its first paragraph, and gives any second measure in parentheses. Where a word gives the size ("little", "about as wide as the seed range"), use the word and let the figure show the digits. A coordinate the reader looks up, a constant of the apparatus, or a value they could take from a nearby table goes in that table, and the prose points at it. Writing the same number out in two places is how a report ends up with two roundings of it.

Say what we make of each result where it appears, and how sure we are: "fits without proving", "may account for it, though one run is not enough". Keep reasons and caveats beside the claim they support too, so nothing is said twice. The discussion then interprets. It can point back to a result, but it doesn't quote the number again, since that lives in the result section and in the index under the lede. What a result would mean for a design can be said as a conditional ("if this holds, a shorter schedule would ..."). What we do next stays out of the report, unless a frozen rule already decided it.

A few labelling habits keep cross-references short. Each section has a short label in its heading (E*n* for an exploratory section, H*n* or S*n* in a preregistered report), and the index under the lede uses it. A bare "(H3)" means this report, and a section of another report takes that report's name: "H3 of ex-2.2.19".

For the register, read [references/exemplar-example-evidence.md](references/exemplar-example-evidence.md) (a whole report) and [references/exemplar-explanation.md](references/exemplar-explanation.md) (how to explain a mechanism).

### The lede (tl;dr)

A report opens with one, just under the title and above the intro prose:

```md
# Ex 2.2.18: dropping ops with similar answers

/// tip |
<!-- lede -->
Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` together made the in-context grammar easier to solve, and the model got closer to what is solvable than in any run so far. Most of that came from dropping ops whose answers round at random. Dropping `lighten` and `darken` in place of `screen` and `multiply` breaks the same pairs, and it left the ceiling where it was and narrowed the gap by less.
///
```

Two or three sentences: what we tried, and which way it came out, with the verdict sentence inside the box. It orients someone deciding whether to read on, so keep numbers, hypothesis IDs, thresholds, and caveats out of it. Whatever else the opening has to say (what moved, what did not, where the recipe lands) goes in a normal paragraph just under the box, and the analysis sections and the discussion carry the full accounting. Left to itself this box grows into a second conclusion; if a sentence in it would also belong in the discussion, cut it.

The title is empty (`/// tip |`) so the box reads as a lede rather than a labelled aside, and the `<!-- lede -->` comment keeps the marker greppable (older reports have `<!-- tl;dr -->`). In a preregistration draft, write the "what we tried" half and leave the outcome line for later.

The index of results follows the lede and the paragraph under it: `Observations` in an exploratory report, and `Findings`, with verdicts, in a preregistered one.

## Designing the experiment

The checks a plan should pass are in [references/design.md](references/design.md). The assumptions review applies them, and the drafter should too.

A preregistered report has its own shape and conventions: see [references/preregistration.md](references/preregistration.md), and [references/surveys.md](references/surveys.md) for a survey.

## Notes between review rounds

Reports go through several fresh-eyes review rounds, each reader starting from the report alone. `REVIEW` notes are how one round's decisions reach the next.

**Leave a `REVIEW` note wherever a review changes a claim:** a threshold, a verdict, a scope, or a wording that changes what is being asserted. Typos and prose polish don't need one. Say what the change was, why, and what a later reader should check to disagree with it. Usually a Python comment in the cell:

```python
# REVIEW: narrowed "anchoring transfers" to "transfers at layer 4" — H2 only
# measures layer 4, so the broader claim outruns the data. Verify: if the sweep
# in the exploratory section covers other layers, this can widen again.
```

An HTML comment inside a Markdown string works when the note has to sit beside one specific paragraph; it stays invisible in the render. Make it visible only when a reader of the published report benefits from it. The marker is greppable either way, so a review pass can find every prior decision before touching the same text.

**A decision left for the human goes in a box they can see.** The human reviews drafts as printed PDFs, where comments don't show, so a choice that waits on them (a threshold to confirm, a rule to keep or drop) goes in an admonition titled `Open decision`, beside the text it concerns:

```md
/// admonition | Open decision
Criterion (b) uses the seed band, so it would also catch a small real cost we might accept. The alternative is a fixed margin of 0.015. To check: the gaps on the old recipe sat inside its seed spread.
///
```

Say what is open, the alternative, and what to check, as for a `REVIEW` note. Once the human decides, fold the outcome into the text (with a `REVIEW` note if it changed a claim) and delete the box. None survives to the merge (or to the freeze, in a preregistered report), and the marker is greppable, like `TODO`.

**A note should only record the change and its warrant:** what the report now claims, and why that follows from the data. It is the same category of thing as a code comment explaining a non-obvious invariant, which is why the next round may read it. It leaves out judgments of the report's quality, the confidence of a round, and anything phrased as "I suspect" or "this felt weak", since those would steer the next reader. Observations of that kind go in the round's own report, under `Tensions`, where they reach the lead and stop.

**A note you would reverse is a finding.** Wanting to undo a recorded decision usually means the claim is doing two jobs at once, and each reviewer is right about a different one. Say so, name both readings, and stop. The resolution is structural (split the hypothesis into two tracks, drop one, or state the scope that separates them) and it needs the human, because it changes what the experiment claims.
