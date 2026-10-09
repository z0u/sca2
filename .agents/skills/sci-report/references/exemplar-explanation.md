# Exemplar: an explanation

From the companion notes written during ex-2.2.21, answering questions from the margins of a preview draft. It is a reply to a reader and not a report, so it has no verdicts or figures.

## The text

> **A picture to start from**
>
> Most of the questions on the preview page come back to one idea, so here it is first.
>
> The model reads a context left to right. At each position it holds a *state*, and the only job of that state, as far as training is concerned, is to predict the next token. So:
>
> - the state at the query `=` predicts the query answer, a color;
> - the state at the query answer predicts the line break, which is always next and so costs nothing to get right;
> - the state at an example answer predicts the comma, which is just as easy.
>
> The anchor asks a labelled context to put the op on e₁ somewhere along the context, and lets the context choose where. A state whose prediction is trivial has lots of room to spare, and the answer positions are also where the examples have shown the most evidence about the op. So the answers are the cheapest place to put it. I think that is why the pull settled there.
>
> The catch is what an edit can reach. The query answer comes after the answer has been predicted, so editing it can never change that answer. The example answers come before the query, and the query can attend to them, so editing them could.

**One picture first.** Several questions shared one idea, so the idea comes before any of them. The picture is the mechanism in everyday words (each state predicts the next token), and everything after follows from it. A report's "Why this experiment" can open the same way.

**Small steps joined by "So".** Each sentence takes one step from the one before. The list spells out the picture for three positions, so the reader can check it, and "So the answers are the cheapest place" then follows without a leap.

**Interpretation, with how sure we are.** "I think that is why the pull settled there" offers a reading and says how firmly it is held. That is calibrated interpretation, and a report may do it.

**What follows for the edit.** The last paragraph turns the picture into a consequence for the thing we care about (what an edit can change), as a statement about what is possible ("so editing them could"). It stops short of saying what we will do about it.

> "Wouldn't it be weird if `=` encoded the op, when what it predicts is a color?" Not weird, but it depends on depth, and I think your hunch about the query attending back is right.
>
> To predict the color, the state at the query `=` needs the operands and the op. It gets them by attending to earlier positions, block by block. So partway up the stack it plausibly holds the op in *some* direction, and by the last block it has turned into "which color", which is what it is graded on. What we measured is narrower: whether the op sits on e₁ at the query `=`. On the whole-line arm it never does, at any slice (at most 0.07).
>
> [...] On the whole-line arm the example answers build up to about 0.5 by block 2 and fade a little after, while the query answer keeps climbing to 0.91. So the op is on e₁ at the example answers in the middle of the stack, where the query can read it.
>
> The suppression pass fits that story without proving it. An edit at the query `=` does nothing on the whole-line arm, while an edit at every position takes off a third of the way to the null. Something other than the query `=` carries the op to the answer, and the example answers are the obvious suspect.

**The answer first, then the reasons.** "Not weird, but it depends on depth" answers the question in its first words, and the paragraphs after it are the reasons.

**The story kept apart from the measurement.** "What we measured is narrower" marks the line between what the model plausibly does and what the experiment checked. A reader can then tell which sentences rest on data.

**Few numbers, each doing a job.** A number appears where the argument turns on it (0.07 says "never"; 0.5 and 0.91 show the two positions going different ways) and in parentheses when it only backs up a word.

**"Fits without proving."** The last paragraph says how far the evidence goes before saying what it suggests. That is the confidence level a discussion section should state for each reading.
