# Exemplar: a scout report (ex-2.2.18)

Excerpts from `docs/m2/ex-2.2.18/report.py`, frozen here as of commit `b0a02cf`. The live report may change later; this copy shows the register to aim for. It is a scout, so its findings are Observations. A preregistered report has the same register with the H-section shape of [template-prereg.py](template-prereg.py).

The notes after each excerpt say what to copy, and the last section lists what changed in review, since those changes are the rules a first draft most often misses.

## Opening

```md
# Ex 2.2.18: Dropping ops with similar answers

/// tip |
<!-- lede -->
Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` together made the in-context grammar easier to solve, and the model got closer to what is solvable than in any run so far. Most of that came from dropping ops whose answers round at random. Dropping `lighten` and `darken` in place of `screen` and `multiply` breaks the same pairs; it left the ceiling where it was but narrowed the gap by less.
///

In ex-2.2.17, the center control kept part of its probability mass on the answers of a similar op, even on contexts where the examples should have made the op unambiguous. Four pairs of ops stood out: `lighten` with `screen`, `darken` with `multiply`, `mix` with `hsvmix`, and `difference` with `exclusion`. This scout drops one op from each pair, first one at a time and then all four together. [...] Each op set trains with the ex-2.2.17 recipe once (a single seed).

Dropping ops changes the task, so every op set has its own Bayes ceiling and floor, and we score each run against the ceiling and floor of its op set.
```

The tl;dr (lede) has three plain sentences and no numbers. It says what happened, which part caused most of it, and how the control comparison came out. The paragraph under it says plainly what earlier result this follows up, and what the scout runs.

## Observations and scope

```md
## Observations

Each item below is a measurement on the runs of this scout, with no gate.

- [Scores against each ceiling (E1)](#scores-against-each-ceiling-e1): dropping `lighten` or `darken` lowered the ceiling, and dropping any of the other four raised it. Without `screen`, `multiply`, `hsvmix`, and `exclusion`, the model came within {gap(FOUR):.3f} of its ceiling, closer than any run so far. The second four-op drop came within {gap(LD):.3f}.
- [...]
- [Training time (E4)](#training-time-e4): every run reached 95% of its final skill by {max(steps_to(s, 0.95) for s in SETS) / STEPS:.0%} of the way through training, and the last fifth of training added little.

## Scope

This is a scout, with no preregistration and no gate. Each op set has one run, all from the same model seed, so a difference between two runs is only a hint. For scale, ex-2.2.17 trained this recipe on the full op set at three seeds, and their gaps to the ceiling spanned {min(YARD_GAP):.3f} to {max(YARD_GAP):.3f}.
```

Each observation is one or two sentences with at most one kind of number. "Added little" replaced "added at most 0.014" in review: a word gives the reader the size, and the table has the digits. The scope section gives the one yardstick every later comparison leans on, so the sections after it can say "about as wide as the seed range" without repeating it.

## Why this experiment

```md
The in-context grammar asks the model to infer the op from three examples, and some pairs of ops give the same answer on many operand pairs. An example that fits `lighten` often fits `screen` too, so those examples say less about the op, and the model has two nearly interchangeable answers to choose between. If the pairs are part of why the control falls short of its ceiling, a smaller op set might make a better grammar for the anchoring experiments.

It may matter which op of a pair is dropped, too. `lighten`, `darken`, and `difference` give one answer for each pair of operands, while their partners round each channel at random between grid levels, so their answers are spread over a few colors. Dropping an op with spread-out answers raises the ceiling whether or not similarity matters. Dropping its partner instead breaks up the pair without raising the ceiling, so comparing the two isolates the effect of rounding.
```

It starts from a picture of the task (an example that fits two ops says less about either) and lets the question follow from it. The second paragraph names a confound and says which comparison separates it, in words. There are no numbers.

## A result section (E1, the prose after the figure)

```md
The full-set run scored inside the range of the three ex-2.2.17 seeds, so the recipe reproduced. The rest of this section compares runs by their gap, since the ceiling moves from one op set to the next.

Dropping one of `screen`, `multiply`, `hsvmix`, or `exclusion` raised the ceiling, whereas dropping `lighten` or `darken` lowered it: Even a predictor told the op, with no inference to do, can't always name the answer of an op that rounds at random, but it always can for `lighten` or `darken`. So the ceiling rises when an op that rounds at random is removed, and falls when its partner goes.

The gaps of the single drops mostly stayed near that of the full set, {gap(FULL):.3f}. Three were wider: `no-multiply` and `no-darken`, the two sides of the darkening pair, and `no-exclusion`. E2 shows where those fell short.

The two four-op drops break the same four pairs, and both narrowed the gap: to {gap(FOUR):.3f} without `screen`, `multiply`, `hsvmix`, and `exclusion`, and to {gap(LD):.3f} without `lighten`, `darken`, `hsvmix`, and `exclusion`. Only the first raised the ceiling, so most of the higher score of `no-four` came from its higher ceiling.

But each op set has only a single run, and the gaps of the three ex-2.2.17 seeds on the full set spread over a range about as wide as either change. So these narrower gaps could be seed variation, and it would take more seeds to tell.
```

The section names its main measure in its first paragraph and then sticks to it: every number is a gap. Before review it switched between EEM, gap, and skill, and the switching made it hard to read. The ceiling is described in words, since the figure has its values. Each paragraph opens with what happened and follows with why, and the last says how far one seed can be trusted.

## Discussion

```md
The op set without `screen`, `multiply`, `hsvmix`, and `exclusion` is easier (its ceiling is higher) and the model gets closer to it. The higher ceiling comes from dropping ops that round at random. The smaller gap came with both four-op drops, so breaking up the similar pairs may account for it, though one run each is not enough to be sure. Dropping `screen` and `multiply` rather than `lighten` and `darken` leaves seven ops that are easier to tell apart. Adopting that set would also change the ceiling that ex-2.2.16 and ex-2.2.17 were scored against, so their results would not compare with new ones as they stand.

The single drops say less. Dropping one op mostly moved the ceiling and the model together, except in the three runs that fell short on the HSV ops; with one seed, that may be timing. The more consistent sign is that all three runs without `hsvmix` learned the HSV ops early, which fits ex-2.2.17 finding `hsvmix` the hardest op to compute.
```

It interprets, and says how sure it is about each reading ("may account for it, though one run each is not enough", "that may be timing", "the more consistent sign"). What adopting the op set would mean is stated as a conditional ("would also change the ceiling"). Whether to adopt it is left to the human.

## What review changed

These are the edits that were made to the draft, as rules:

- A number in prose only where the argument needs its precision. A recap of an earlier experiment needs none.
- One main measure per section, said up front, with a second one in parentheses when it's needed.
- Shorter names: "HSV ops" for "HSV-channel ops".
- Plain verbs over idioms: "should have made the op unambiguous" for "should settle the op", "could be seed variation" for "could come from seed variation alone", and a reword for "carry over".
- No trailing clause that restates the sentence: ", which EEM alone cannot do" and "of a ceiling near that of the full set" went.
- No elliptical back-references: "against its own" became "against the ceiling and floor of its op set".
- Conventional headings: "Why this experiment" and "Discussion".
