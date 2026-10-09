# Preregistration

Read this when an experiment will be preregistered (see "Choosing a shape" in the `science` skill for when). Design the experiment with the human, and draft the report skeleton before writing any experiment code. The skeleton doubles as the analysis plan: writing it before the data exists lets a later "we predicted X and found Y" carry weight, because the prediction is verifiably older than the result.

The skeleton is usually a text-only report, copied from [template-prereg.py](template-prereg.py). Findings come first and method last, so a reader meets evidence early and nothing is stated twice. It runs: the lede; `Findings`, an index of the hypotheses with the verdicts blank; a "How to read this report" note; `Why this experiment`, which starts from a picture of what the model does and lets the question follow from it, recapping earlier experiments by what they showed and without their numbers (about 250 words); the runs and the measurements, which are all a reader needs to parse a result; one section per hypothesis, each opening with its frozen prediction and followed by a placeholder for its evidence and verdict; one section per planned exploratory analysis (E1, E2, ...), placed among the hypotheses in reading order; `Decision`, if the experiment chooses something; `Discussion`; and then the method.

Keep it as short as the exploratory shape where you can. Two or three hypotheses that each change what we do next are worth more than six, and every one costs a frozen prediction, a placeholder, a section, and a `Findings` line.

## Conventions

- Placeholders are admonitions marked `TODO`, one under each section's prediction. Each states what its figure or table will show (axes, panels); the prediction above it already says what pattern is expected and what a contrary result looks like, so the placeholder does not repeat them. The marker is greppable, so no placeholder survives to publication; results replace placeholders in place, so review reads as a prediction → observation diff.
- Hypotheses are falsifiable and plainly worded. Each is an expectation a colleague could restate from memory: what we expect to see, the one number we will look at, and what would change our mind. Write it in the conditional ("a shortfall under 0.015 would be a pass"), since it describes outcomes that haven't happened yet. Precision in the wording does not buy correctness in the design, and the trial-protocol register has cost days per experiment without catching the misses that mattered.
- Each prediction says what result would fall outside its plan, and that such a result gets the verdict `Unresolved`. The usual case is seeds that disagree about the direction. Ex-2.2.20 H3 predicted the HSV ops would rise sooner, and the seeds split (later at two, sooner at one); without that clause it was scored a miss, with a verdict stronger than mixed seeds support. An `Unresolved` verdict names both readings.
- Preregister the predictions and the criteria a decision will weigh, and make the decision after the results, with the human. Record it in `Decision` as a choice, with every criterion reported for every candidate, and confirm what it adopts at fresh seeds before quoting it as a result (the winner's curse; see [surveys.md](surveys.md)). Frozen rules that adopted a point on their own have had to be broken or amended after the fact in ex-2.2.3, ex-2.2.8, ex-2.2.11, and ex-2.2.19. Hard gates stay where a "no" has to hold whatever the data look like: task cost and risk rows.
- An exploratory report before the preregistration (as the ex-2.2.18 scout was before ex-2.2.20; see [exemplar-ex-2.2.18.md](exemplar-ex-2.2.18.md)) lets a prediction be written with the shape of the results already known.
- A result section reads as expectation, then what we saw, then what we make of it. Label the two openings `**What we expect.**` and `**What we saw.**`, so a frozen prediction is never mistaken for an outcome, and close the section with a verdict admonition (`/// admonition | Pass`, or `Miss`, `Partial`, `Unresolved`; `Decided` for an S section, whose rule picks among options rather than passing or missing) carrying the one-line verdict: the site hoists that title into the heading as a badge, so a reader meets the outcome with the question. The gate arithmetic goes in a `details` block or the method.
- H*n* labels a hypothesis, and S*n* a preregistered procedure with no prediction of its own (a scout whose frozen rule picks what a later stage confirms).
- A prediction lives at the top of its own result section and nowhere else: no standalone `## Hypotheses` block. A reviewer reads them in sequence through the `Findings` index, and in a draft the sections are consecutive anyway, with only a placeholder between them. A block up front would state every gate a second time before its section states it again.
- A constants-only `experiment.py` is marked `DESIGN_ONLY = True`. Landing the design constants — grid sizes, thresholds, schedules — as a module during preregistration lets the report import them instead of restating numbers the code will later own. But `tests/mini/test_experiments_e2e.py` globs every `docs/**/experiment.py` and asserts it loads into a named experiment with a callable `main(ctx)`, which a design module doesn't have yet; `DESIGN_ONLY = True` at module level skips that check. Delete the line in the same change that adds the DAG, or the implemented experiment silently loses its load coverage.
- Freeze the hypotheses once the skeleton is agreed (immaterial edits aside, and every `Open decision` box resolved), and say so in the report under "How to read this report", quoting the commit — with the predictions spread over their sections, that hash is what says they were fixed in advance. Results replace placeholders, and anything conceived after seeing the data is marked as post hoc, within the E section it grew from or in a section of its own.
- A claim stated before its evidence exists gets paid for twice, once where it is stated and once where it is met. That is inherent to preregistration and worth the cost for hypotheses and thresholds, and not for anything else, so keep rationales, caveats, and worked reasoning at the point of use rather than in the method. Where a restatement is unavoidable, quote the frozen line rather than paraphrasing it, since a paraphrase drifts.

Example of a prediction, from ex-2.2.20 H3 as it would read with the outside-the-plan clause:

```md
## The HSV ops come earlier (H3)

**What we expect.** In the plain 200-epoch runs, the HSV ops first passed a skill of 0.5 between epochs 80 and 112. We expect the head-start runs to pass it about 25 epochs earlier, paired by seed. Passing at about the same epoch would be a miss, and so would passing later. If the seeds disagree about the direction, the result would be outside the plan, and the verdict would be Unresolved.

/// admonition | TODO
The epoch at which the HSV skill first passes 0.5, per seed, for the plain and head-start runs.
///
```

## Findings

Directly under the lede, and above the intro prose. Every preregistered hypothesis and its verdict, in bold, with a sentence on what happened. Give the deciding number only where a word can't say it. Under 200 words:

```md
## Findings

- [The head start keeps most of the skill (H1)](#the-head-start-keeps-most-of-the-skill-h1) — **miss**. The head-start runs fall short of the 400-epoch runs, inside the partial band, but all three HSV ops fall short by more than their tolerance.
- [The head start beats the plain schedule (H2)](#the-head-start-beats-the-plain-schedule-h2) — **miss**. The head start scores above the plain 200-epoch run in one of the three seeds, but below it on average.
```

The lede says which way it came out; this says what happened, in words that could be read aloud to a colleague. A reader who stops here should be able to tell that three of four hypotheses missed, without reading a discussion to find out. Without it, a reader gets nothing until they have read the whole report.

Verdicts only. Interpretation, mechanism, and whether an outcome was named in advance belong to the analysis sections. Link each line to its section, so this doubles as the report's index. In a preregistration draft it is only that: one line per hypothesis, ID and short name linking to the section, verdict blank. That is where a reviewer sees every prediction in one place, and writing the verdicts in is the first thing to do when results land.

Two consequences for the rest of the report. The discussion no longer opens by re-deriving the results, because they are above it — it interprets, and nothing else. And a section that cannot supply its own line here has a gap worth fixing: if a verdict or its deciding number is missing, or first appears in some other section, that is the section's problem rather than the summary's.

## Collaborating on the skeleton

The skeleton is a review artifact in its own right. Iterate on it together in a PR before any experiment code lands (although feel free to run small prototypes that don't get committed). This is where the hypotheses and thresholds get agreed and frozen, after the assumptions and style reviews and the human's paper round (the flow is in the `science` skill).

When results arrive, fill the report in order of stakes rather than all at once. The mechanical sections, where the number either clears its threshold or it doesn't, can be filled in one pass. Pause for a discussion round before writing the prose where interpretation lives, since that is the part the human most wants a hand in, and the part most likely to over-reach.
