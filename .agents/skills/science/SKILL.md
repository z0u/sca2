---
name: science
description: |
  How we do experimental science in this project: choosing an experiment's shape, writing short exploratory reports, preregistering when a claim needs it, and collaborating on reports.
---

## Choosing a shape

Most of our questions are best answered by a short exploratory report: one question, looked at a few ways, and what we make of it. That is the default. It is quick to write, quick to run, and quick for the human to review on paper, and it spends nothing on predictions, gates, and verdicts that the result might not need.

Two other shapes exist for when the default doesn't fit. Choose them with the human, while agreeing the design.

- Preregister when a claim has to stand without the reader trusting how we chose to look at the data: a result a deliverable will rest on, an operating point to confirm at fresh seeds before it is quoted, or a costly run where a frozen plan keeps us from reading the results to suit. See [references/preregistration.md](references/preregistration.md).
- Run a survey when choosing an operating point in a space too large to give every point a hypothesis. See [references/surveys.md](references/surveys.md).

An exploratory result is as solid as its data, and the report says how sure we are. What it lacks is a prediction made in advance, so if a later design leans on it heavily, a preregistered confirmation is how it becomes a result.

## The exploratory report

The example-evidence report is the model: an eight-page re-analysis with three short sections and a discussion. Read [references/exemplar-example-evidence.md](references/exemplar-example-evidence.md) before drafting, and copy [references/template-explore.py](references/template-explore.py). Both stored-data re-analyses and studies with a few new runs take this shape (the ex-2.2.18 scout is one with new runs).

It runs: the lede; one paragraph saying where the question came from and what the report reads or runs; `Observations`; `Scope`; the measurements; three or so E sections; and `Discussion`.

- One question, ideally a puzzle a reader can hold in mind. The paragraph under the lede is the whole "why", with links to the backlog item or earlier report that raised it. Add a `Why this experiment` section only when the question needs a picture of the model to follow from (as in ex-2.2.18).
- `Observations` sits where a preregistered report has `Findings`: one line per E section, linked to it, saying what we saw in words, with a number only where no word gives the size. A reader who stops there has the result.
- `Scope` says which runs and data the report covers, how far a difference between runs can be trusted, and what was left out and why. It usually replaces `Parameters` and `Method`. Keep a method section only for detail that changes how a reader takes a result.
- Each E section asks one thing, usually the thing the last one left open. It opens with the question, shows one figure or table, says what we saw, and says what we make of it. A section that doesn't feed the discussion belongs in another report, or nowhere.
- No hypotheses, gates, verdicts, or `Decision`. Choices the results inform are made with the human afterwards, in conversation or a todo item, and stay out of the report.
- About eight printed pages. If a draft runs well past that, it is probably two questions; split it.

When the report needs new runs, agree a short design note with the human first (in the todo item or the PR description): the question, the runs and what each comparison separates, the measurements, and the cost. Then run, and write the report once results exist. There is no skeleton to freeze.

## Writing any report

- Tables are quantitative, prose is qualitative, and figures bridge the two. A table holds the digits; the prose says what they mean in words ("the earlier examples barely count"); a figure shows the pattern so a reader can see the size without the digits. Each is about 95% its own kind: prose keeps a number only where its size is the argument and no word captures it, and a table gets a figure beside it almost always, since a table alone looks like a wall of text.
- Sections carry a short label in their heading, and the index under the lede and cross-references use it: E*n* for an exploratory analysis, and H*n* or S*n* in a preregistered report.
- Name the experiment when citing a hypothesis or section from another report ("H3 of ex-2.2.19"); a bare "(H3)" means this report.
- The discussion interprets. It may refer to a result and never requotes it: the deciding number lives in the result section and in the index under the lede.
- Say what we make of a result, and how sure we are ("fits without proving", "may account for it, though one run is not enough"). What it would mean for a design or an edit can be stated as a conditional. What we will do next is for the human to decide, unless a frozen rule already decided it, so a report never announces the next experiment.
- Keep rationales, caveats, and worked reasoning at the point of use rather than in a method section, so nothing is said twice.
- Numbers in prose earn their place by being part of an argument, and most don't: a reader's eyes slide off a sentence with three numbers in it. Use one main measure per section, say which in its first paragraph, and give a second measure only in parentheses. Where a word gives the size ("little", "about as wide as the seed range"), use the word and leave the digits to the figure. A coordinate the reader looks up, a constant of the apparatus, or a value derivable from an adjacent table belongs in a table or in the method, with the prose referring to it. Writing the same quantity out in two sections is how two roundings of it end up in the report.

For the register, read [references/exemplar-ex-2.2.18.md](references/exemplar-ex-2.2.18.md) (a scout report, with notes on what review changed) and [references/exemplar-explanation.md](references/exemplar-explanation.md) (how to explain a mechanism).

### The lede (tl;dr)

A report opens with one, directly under the title and above the intro prose:

```md
# Ex 2.2.18: dropping ops with similar answers

/// tip |
<!-- lede -->
Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` together made the in-context grammar easier to solve, and the model got closer to what is solvable than in any run so far. Most of that came from dropping ops whose answers round at random. Dropping `lighten` and `darken` in place of `screen` and `multiply` breaks the same pairs, and it left the ceiling where it was and narrowed the gap by less.
///
```

Two or three sentences: what we tried, and which way it came out, with the verdict sentence inside the box. It orients someone deciding whether to read on, so keep numbers, hypothesis IDs, thresholds, and caveats out of it. Whatever else the opening has to say (what moved, what did not, where the recipe lands) goes in a normal paragraph directly under the box, and the analysis sections and the discussion carry the full accounting. Left to itself this box grows into a second conclusion; if a sentence in it would also belong in the discussion, cut it.

The title is empty (`/// tip |`) so the box reads as a lede rather than a labelled aside, and the `<!-- lede -->` comment keeps the marker greppable (older reports have `<!-- tl;dr -->`). In a preregistration draft, write the "what we tried" half and leave the outcome line for later.

The index of results follows the lede and the paragraph under it: `Observations` in an exploratory report, and `Findings`, with verdicts, in a preregistered one.

## Best practices

- Choose a measurement site by a criterion independent of the statistic you're judging.
- A gate, or any comparison an observation rests on, is informative only if a reader cannot predict its direction from the method section alone. Ask what the treatment optimizes and whether the scored statistic is that quantity, a monotone function of it, or independent of it; only the third is a test of the mechanism. Ex-2.1.7's H4(a) scored the containment of ᾱ under a term _defined_ as a penalty on ᾱ, so "the term lowers ᾱ" could only ever say the weight was large enough, and it passed preregistration and a review round before anyone noticed. Score something the treatment does not touch by construction (the task metric, a held-out probe, a downstream selectivity), and keep the optimized quantity as a manipulation check.
- A factor changes more than the thing it is named for. Before writing its hypothesis, list everything it alters, and look hardest at normalizers, denominators, and anything averaged over a set whose size the factor changes. Ex-2.1.7's op1-only arm was read as a pure narrowing of the pull, but the anchor term divides by the realized mask, so one position instead of four also made each pull ~3.9× stronger; the reading survived only because the ceiling arm happened to separate the two. A quick test: if the factor were renamed after its side effect instead of its intent, would the hypothesis still read as written? If not, either fix the design so the side effect goes away (normalize by a fixed count) or report the matching invariants for every arm, as with bracketing above.
- Ablate before you search, and **bracket rather than survey** when the ablation needs a value you haven't found yet. Removing a schedule means running a constant instead, and which constant you pick can decide the answer. Rather than matching on one invariant (area, maximum, or endpoint — each defensible, each a different condition), choose flat levels that straddle the schedule's own range. If the schedule beats every constant in its range, no constant substitutes for the shape, whatever the optimum turns out to be; if one wins, the schedule dimensions go away and you have a better operating point too. Report the matching invariants for every arm instead of matching on one, so the results can say which was the active ingredient. Each dimension deleted this way is much cheaper than searching it.
- (more in `todo/science/`)

## Collaborating on a report

The human wants to be involved in the writing. For an exploratory report that happens at the design note and again at the draft; a preregistered report adds a round on the skeleton (see [references/preregistration.md](references/preregistration.md)). Pause for a discussion round before writing the prose where interpretation lives, since that is the part the human most wants a hand in, and the part most likely to over-reach.

Write the interpretive prose as an explanation first. Before touching the report, write the index under the lede and the Discussion as a message to the human, as if explaining the results to them over lunch: lead with what we found in one plain sentence, say what each number means before giving it, use the same everyday words throughout (the model loses *red*; the answer stops depending on the red operand) rather than the statistic names, gloss each statistic once in a phrase, and say what we make of it and how sure we are. What we might do next can go in the message, for the human to decide, and stays out of the report. Paste the rest into the report as the first draft of those sections and add the template expressions afterwards. This is a workflow rule rather than a style rule, because the chat explanation is the register the report should have had from the first draft, and sentence-level polish does not change the register a draft was written in.

Whatever you have written, run a review round over it before handing back to the human, covering the sections that are done. For an exploratory report, one round usually converges. Say in the request which sections are in scope, so a `TODO` in a section whose turn hasn't come isn't read as an omission.

Prose drafted by Fable gets two passes on the same turn: `prose-simplifier` to lower reader effort, then the `report-restructure` skill to give the result a shape that can be skimmed. Fable writes for an expert reader by default, and the passes bring it to the register of the exemplars; on Opus and Sonnet drafts they changed about one word in a hundred in a trial, so skip them there unless the human or a reviewer finds a section hard going. When they run, run them in that order, stage your changes first so you can see what each pass did, and read the edits for correctness. The sequence, and the checks each pass leaves to you, are in [references/review-passes.md](references/review-passes.md).

Neither pass will make a report much shorter. Length comes off at the structural level — duplication across sections, and front matter that runs before the first result — which is the `report-structure` agent's job at the publish gate (and the freeze, for a preregistered report). Asking one question per report does more for length than either pass.

The publishing mechanics — exporting the report as a bundle, wiring result refs, verifying the render — are a separate concern, covered by the `mi-ni` skill.

### Recording a review decision

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

Say what is open, the alternative, and what to check, as for a `REVIEW` note. Once the human decides, fold the outcome into the text (with a `REVIEW` note if it changed a claim) and delete the box. None survives the freeze, and the marker is greppable, like `TODO`.

**A note should only record the change and its warrant:** what the report now claims, and why that follows from the data. It is the same category of thing as a code comment explaining a non-obvious invariant, which is why the next round may read it. It never carries a judgment of the report's quality, a round's confidence, or anything phrased as "I suspect" or "this felt weak", since that primes the next reader instead of informing them. Observations of that kind go in the round's own report, under `Tensions`, where they reach the supervisor and stop.

**A note you would reverse is a finding.** Wanting to undo a recorded decision usually means the claim is doing two jobs at once, and each reviewer is right about a different one. Say so, name both readings, and stop. The resolution is structural (split the hypothesis into two tracks, drop one, or state the scope that separates them) and it needs the human, because it changes what the experiment claims.

Commissioning the rounds — which pass to run when, how to brief a reviewer, when to stop and escalate, and the lighter pass for prose alone — is in [references/review-passes.md](references/review-passes.md). Always read this if you are the lead.
