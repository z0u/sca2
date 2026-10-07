---
name: science
description: |
  How we do experimental science in this project: designing experiments, preregistering falsifiable hypotheses, and collaborating on reports.
---

## Preregistration

Design the experiment with the human, and draft the report skeleton before writing any experiment code. The skeleton doubles as the analysis plan: writing it before the data exists lets a later "we predicted X and found Y" carry weight, because the prediction is verifiably older than the result.

The skeleton is usually a text-only report, copied from [references/template-prereg.py](references/template-prereg.py). Findings come first and method last, so a reader meets evidence early and nothing is stated twice. It runs: the lede; `Findings`, an index of the hypotheses with the verdicts blank; a "How to read this report" note; `Why this experiment`, which starts from a picture of what the model does and lets the question follow from it, recapping earlier experiments by what they showed and without their numbers (about 250 words); the runs and the measurements, which are all a reader needs to parse a result; one section per hypothesis, each opening with its frozen prediction and followed by a placeholder for its evidence and verdict; one section per planned exploratory analysis (E1, E2, ...), placed among the hypotheses in reading order; `Decision`, if the experiment chooses something; `Discussion`; and then the method.

For the register, read [references/exemplar-ex-2.2.18.md](references/exemplar-ex-2.2.18.md) (a scout report, with notes on what review changed) and [references/exemplar-explanation.md](references/exemplar-explanation.md) (how to explain a mechanism) before drafting.

Conventions:

- Placeholders are admonitions marked `TODO`, one under each section's prediction. Each states what its figure or table will show (axes, panels); the prediction above it already says what pattern is expected and what a contrary result looks like, so the placeholder does not repeat them. The marker is greppable, so no placeholder survives to publication; results replace placeholders in place, so review reads as a prediction → observation diff.
- Almost always, tabular data should be accompanied by a figure. Tables look like a wall of text to a human; charts are easier to interpret.
- Hypotheses are falsifiable and plainly worded. Each is an expectation a colleague could restate from memory: what we expect to see, the one number we will look at, and what would change our mind. Write it in the conditional ("a shortfall under 0.015 would be a pass"), since it describes outcomes that haven't happened yet. Precision in the wording does not buy correctness in the design, and the trial-protocol register has cost days per experiment without catching the misses that mattered.
- Each prediction says what result would fall outside its plan, and that such a result gets the verdict `Unresolved`. The usual case is seeds that disagree about the direction. Ex-2.2.20 H3 predicted the HSV ops would rise sooner, and the seeds split (later at two, sooner at one); without that clause it was scored a miss, with a verdict stronger than mixed seeds support. An `Unresolved` verdict names both readings.
- Preregister the predictions and the criteria a decision will weigh, and make the decision after the results, with the human. Record it in `Decision` as a choice, with every criterion reported for every candidate, and confirm what it adopts at fresh seeds before quoting it as a result (the winner's curse; see Surveys). Frozen rules that adopted a point on their own have had to be broken or amended after the fact in ex-2.2.3, ex-2.2.8, ex-2.2.11, and ex-2.2.19. Hard gates stay where a "no" has to hold whatever the data look like: task cost and risk rows.
- A scout before the preregistration (as ex-2.2.18 was before ex-2.2.20) lets a prediction be written with the shape of the results already known. Use [references/template-scout.py](references/template-scout.py); its findings are Observations, with no gate.
- A result section reads as expectation, then what we saw, then what we make of it. Label the two openings `**What we expect.**` and `**What we saw.**`, so a frozen prediction is never mistaken for an outcome, and close the section with a verdict admonition (`/// admonition | Pass`, or `Miss`, `Partial`, `Unresolved`; `Decided` for an S section, whose rule picks among options rather than passing or missing) carrying the one-line verdict: the site hoists that title into the heading as a badge, so a reader meets the outcome with the question. The gate arithmetic goes in a `details` block or the method.
- Sections carry a short label in their heading, and `Findings` and cross-references use it: H*n* for a hypothesis, S*n* for a preregistered procedure with no prediction of its own (a scout whose frozen rule picks what a later stage confirms), and E*n* for an exploratory analysis, usually post hoc.
- A prediction lives at the top of its own result section and nowhere else: no standalone `## Hypotheses` block. A reviewer reads them in sequence through the `Findings` index, and in a draft the sections are consecutive anyway, with only a placeholder between them. A block up front would state every gate a second time before its section states it again.
- A constants-only `experiment.py` is marked `DESIGN_ONLY = True`. Landing the design constants — grid sizes, thresholds, schedules — as a module during preregistration lets the report import them instead of restating numbers the code will later own. But `tests/mini/test_experiments_e2e.py` globs every `docs/**/experiment.py` and asserts it loads into a named experiment with a callable `main(ctx)`, which a design module doesn't have yet; `DESIGN_ONLY = True` at module level skips that check. Delete the line in the same change that adds the DAG, or the implemented experiment silently loses its load coverage.
- Freeze the hypotheses once the skeleton is agreed (immaterial edits aside, and every `Open decision` box resolved), and say so in the report under "How to read this report", quoting the commit — with the predictions spread over their sections, that hash is what says they were fixed in advance. Results replace placeholders, and anything conceived after seeing the data is marked as post hoc, within the E section it grew from or in a section of its own.
- Name the experiment when citing a hypothesis or section from another report ("H3 of ex-2.2.19"); a bare "(H3)" means this report.
- The discussion interprets. It may refer to a result and never requotes it: the verdict and its deciding number live in the result section and in `Findings`.
- Say what we make of a result, and how sure we are ("fits without proving", "may account for it, though one run is not enough"). What it would mean for a design or an edit can be stated as a conditional. What we will do next is for the human to decide, unless a frozen rule already decided it, so a report never announces the next experiment.
- A claim stated before its evidence exists gets paid for twice, once where it is stated and once where it is met. That is inherent to preregistration and worth the cost for hypotheses and thresholds, and not for anything else, so keep rationales, caveats, and worked reasoning at the point of use rather than in the method. Where a restatement is unavoidable, quote the frozen line rather than paraphrasing it, since a paraphrase drifts.
- Numbers in prose earn their place by being part of an argument, and most don't: a reader's eyes slide off a sentence with three numbers in it. Use one main measure per section, say which in its first paragraph, and give a second measure only in parentheses. Where a word gives the size ("little", "about as wide as the seed range"), use the word and leave the digits to the figure. A coordinate the reader looks up, a constant of the apparatus, or a value derivable from an adjacent table belongs in a table or in the method, with the prose referring to it. Writing the same quantity out in two sections is how two roundings of it end up in the report.

Example of a prediction, from ex-2.2.20 H3 as it would read with the outside-the-plan clause:

```md
## The HSV ops come earlier (H3)

**What we expect.** In the plain 200-epoch runs, the HSV ops first passed a skill of 0.5 between epochs 80 and 112. We expect the head-start runs to pass it about 25 epochs earlier, paired by seed. Passing at about the same epoch would be a miss, and so would passing later. If the seeds disagree about the direction, the result would be outside the plan, and the verdict would be Unresolved.

/// admonition | TODO
The epoch at which the HSV skill first passes 0.5, per seed, for the plain and head-start runs.
///
```

## Surveys

Some questions are about choosing an operating point in a space too large to give every point a hypothesis: the anchor weight's selectivity optimum, the mellowmax temperature, how much repulsion to deliver and when. A *survey* is the experiment type for those. It preregisters the search plan instead of an outcome, and it scores nothing.

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

### A survey's report

Same skeleton, with three differences.

- `## Findings` becomes `## Observations` — same place, same brevity, but each line carries its noise floor where a verdict would carry its gate, and one line names the proposed operating point.
- The search plan is one `## Search plan` block, where a scored report's first result section would sit, since there are no hypotheses to spread over sections.
- `### Conditions` becomes the space specification plus the full trial table. This is the convention that has to bend: elsewhere the report imports hand-justified condition dicts and renders them as prose, which is why there's no generic grid builder, and a hundred trials can't each carry a docstring. So the justification attaches to the dimension rather than the level, and the trial table is generated from stored results.

Say "survey" in the first clause of the lede and label it the same way in `docs/index.md`. Numbering stays in the `ex-2.1.N` sequence.

`docs/ngpt-scaling/report.py` is the closest existing example — a width × depth grid, no hypotheses, and a conclusion about whether the region is safe to build on. A survey is that plus the frozen search plan, which a 3 × 3 didn't need.

## Best practices

- Choose a measurement site by a criterion independent of the statistic you're judging.
- A gate is informative only if a reader cannot predict its direction from the method section alone. Ask what the treatment optimizes and whether the scored statistic is that quantity, a monotone function of it, or independent of it; only the third is a test of the mechanism. Ex-2.1.7's H4(a) scored the containment of ᾱ under a term _defined_ as a penalty on ᾱ, so "the term lowers ᾱ" could only ever say the weight was large enough, and it passed preregistration and a review round before anyone noticed. Score something the treatment does not touch by construction (the task metric, a held-out probe, a downstream selectivity), and keep the optimized quantity as a manipulation check.
- A factor changes more than the thing it is named for. Before writing its hypothesis, list everything it alters, and look hardest at normalizers, denominators, and anything averaged over a set whose size the factor changes. Ex-2.1.7's op1-only arm was read as a pure narrowing of the pull, but the anchor term divides by the realized mask, so one position instead of four also made each pull ~3.9× stronger; the reading survived only because the ceiling arm happened to separate the two. A quick test: if the factor were renamed after its side effect instead of its intent, would the hypothesis still read as written? If not, either fix the design so the side effect goes away (normalize by a fixed count) or report the matching invariants for every arm, as with bracketing above.
- Ablate before you search, and **bracket rather than survey** when the ablation needs a value you haven't found yet. Removing a schedule means running a constant instead, and which constant you pick can decide the answer. Rather than matching on one invariant (area, maximum, or endpoint — each defensible, each a different condition), choose flat levels that straddle the schedule's own range. If the schedule beats every constant in its range, no constant substitutes for the shape, whatever the optimum turns out to be; if one wins, the schedule dimensions go away and you have a better operating point too. Report the matching invariants for every arm instead of matching on one, so the results can say which was the active ingredient. Each dimension deleted this way is much cheaper than searching it.
- (more in `/todo-science.md`)

## Collaborating on a report

The human wants to be involved in the writing, so the skeleton is a review artifact in its own right. Iterate on it together in a PR before any experiment code lands (although feel free to run small prototypes that don't get committed). This is where the hypotheses and thresholds get agreed and frozen.

When results arrive, fill the report in order of stakes rather than all at once. The mechanical sections, where the number either clears its threshold or it doesn't, can be filled in one pass. Pause for a discussion round before writing the prose where interpretation lives, since that is the part the human most wants a hand in, and the part most likely to over-reach.

Write the interpretive prose as an explanation first. Before touching the report, write the Findings and Discussion as a message to the human, as if explaining the results to them over lunch: lead with what we found in one plain sentence, say what each number means before giving it, use the same everyday words throughout (the model loses *red*; the answer stops depending on the red operand) rather than the statistic names, gloss each statistic once in a phrase, and say what we make of it and how sure we are. What we might do next can go in the message, for the human to decide, and stays out of the report. Paste the rest into the report as the first draft of those sections and add the template expressions afterwards. This is a workflow rule rather than a style rule, because the chat explanation is the register the report should have had from the first draft, and sentence-level polish does not change the register a draft was written in.

Whatever you have written, run a review round over it before handing back to the human, covering the sections that are done. Say in the request which sections are in scope, so a `TODO` in a section whose turn hasn't come isn't read as an omission.

Prose drafted by Fable gets two passes on the same turn: `prose-simplifier` to lower reader effort, then the `report-restructure` skill to give the result a shape that can be skimmed. Fable writes for an expert reader by default, and the passes bring it to the register of the exemplars; on Opus and Sonnet drafts they changed about one word in a hundred in a trial, so skip them there unless the human or a reviewer finds a section hard going. When they run, run them in that order, stage your changes first so you can see what each pass did, and read the edits for correctness. The sequence, and the checks each pass leaves to you, are in [references/review-passes.md](references/review-passes.md).

Neither pass will make a report much shorter. Length comes off at the structural level — duplication across sections, and front matter that runs before the first result — which is the `report-structure` agent's job at the freeze and publish gates.

The publishing mechanics — exporting the report as a bundle, wiring result refs, verifying the render — are a separate concern, covered by the `mi-ni` skill.

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

### Findings

Directly under the lede, and above the intro prose. Every preregistered hypothesis and its verdict, in bold, with a sentence on what happened. Give the deciding number only where a word can't say it. Under 200 words:

```md
## Findings

- [The head start keeps most of the skill (H1)](#the-head-start-keeps-most-of-the-skill-h1) — **miss**. The head-start runs fall short of the 400-epoch runs, inside the partial band, but all three HSV ops fall short by more than their tolerance.
- [The head start beats the plain schedule (H2)](#the-head-start-beats-the-plain-schedule-h2) — **miss**. The head start scores above the plain 200-epoch run in one of the three seeds, but below it on average.
```

The lede says which way it came out; this says what happened, in words that could be read aloud to a colleague. A reader who stops here should be able to tell that three of four hypotheses missed, without reading a discussion to find out. Without it, a reader gets nothing until they have read the whole report.

Verdicts only. Interpretation, mechanism, and whether an outcome was named in advance belong to the analysis sections. Link each line to its section, so this doubles as the report's index. In a preregistration draft it is only that: one line per hypothesis, ID and short name linking to the section, verdict blank. That is where a reviewer sees every prediction in one place, and writing the verdicts in is the first thing to do when results land.

Two consequences for the rest of the report. The discussion no longer opens by re-deriving the results, because they are above it — it interprets, and nothing else. And a section that cannot supply its own line here has a gap worth fixing: if a verdict or its deciding number is missing, or first appears in some other section, that is the section's problem rather than the summary's.

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
