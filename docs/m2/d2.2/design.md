# D2.2 design: anchoring an operation the model infers

D2.2 asks whether Sparse Concept Anchoring works for a concept that is never written down: the model has to infer it from context. Each line shows a few solved equations that use the same hidden operation, and the model has to infer the op to answer the last equation. We anchor one op to the first axis of the residual stream, then remove it by editing that axis out.

Each line of the corpus is one context: three solved examples of one op, then a query under the same op, written `op1 ? op2 = answer` with a constant `?` where the op word would be. The model reads the examples, infers which op fits them, and completes the query. We can compute how strongly each context indicates each op (the posterior over ops), and we evaluate the predictive power of each model against that, with and without being edited to remove the anchored op.

On the best recipe so far, the anchor holds, and any cost to the task is smaller than our seeds can resolve. Turning the edit up takes `difference` out gradually. On most runs the other ops stay as they were, but not on every run.

Still to show: that this holds at fresh seeds, that the removal follows how strongly each context indicates the op, and at which layers of the model it works.

How we got here is in the [index](/docs/index.md#d22-anchoring-an-operation). The [pivot](pivot.md) explains why we took the op words out of the grammar.

## The setup today

Our training recipe evolves with each experiment, as we learn more about how SCA behaves in transformers. This is our "recipe of record": by default, the next experiment should use these parameters.

Ops
: Seven: `mix`, `lighten`, `darken`, `difference`, and three HSV blend modes (`hue-hsv`, `sat-hsv`, `value-hsv`). Four ops whose answers often coincide with those of another op were dropped from the earlier table of eleven.

    From [ex-2.2.18](../ex-2.2.18/report.py) and [ex-2.2.19](../ex-2.2.19/report.py).

Rounding
: An answer that falls between grid levels rounds up or down at random, in proportion to where it falls. So the answer to a pair is a distribution over a few colors, and a model is scored on the probability it puts on them.

    From [ex-2.2.4](../ex-2.2.4/report.py) and [ex-2.2.5](../ex-2.2.5/report.py).

Contexts
: Three examples and a query. Three examples in ten show the answer another op would give (replacement op noise at 0.3), so the examples leave some contexts in doubt about their op; a few answers are random colors (cube noise at 0.02). Contexts with varying numbers of examples were tried and cost a little skill.

    From [ex-2.2.16](../ex-2.2.16/report.py); the varying counts from [ex-2.2.22](../ex-2.2.22/report.py).

Model
: d64-L4 simplified nGPT, with a readout table of its own (untied from the embedding table) and attention that stops at the line break.

    From [ex-2.2.7](../ex-2.2.7/report.py) (readout) and [ex-2.2.17](../ex-2.2.17/report.py) (mask).

Training
: 200 epochs, cosine schedule peaking near 0.003.

    From [ex-2.2.17](../ex-2.2.17/report.py), [ex-2.2.19](../ex-2.2.19/report.py), and [ex-2.2.20](../ex-2.2.20/report.py).

Anchored op
: `difference` on e₁, with no *red* anchor.

    From [ex-2.2.14](../ex-2.2.14/report.py).

Labels
: The whole context, on about one `difference` context in fifty. The label says only that the context uses `difference`; the model never sees it, and no other op is labelled.

    From [ex-2.2.14](../ex-2.2.14/report.py) and [ex-2.2.21](../ex-2.2.21/report.py).

Pull
: On a labelled context, the anchor term pulls every slice (the embedding and the output of each block), with no cap on the alignment. A context the training window cuts short is not pulled. Ex-2.2.21 suggested that a cap at an alignment of 0.8 (the `hinge` cap) kept the edit selective, but in ex-2.2.22 runs that spilled past the gate turned up at every cap tried, so the cap is dropped until fresh seeds say otherwise. Pulling fewer slices made the edit spill onto other ops.

    From [ex-2.2.15](../ex-2.2.15/report.py) (whole contexts), and [ex-2.2.21](../ex-2.2.21/report.py) and [ex-2.2.22](../ex-2.2.22/report.py) (the cap and the slices).

Anchor schedule
: The anchor weight warms up over the first tenth of training, holds, then eases over the last tenth to a tenth of its peak. A second term pushes the states of every other context off e₁; it starts at a few times the anchor weight and eases to a fraction of it before the anchor anneal begins. Unchanged since D2.1.

    From [ex-2.1.10](../ex-2.1.10/report.py), carried over by [ex-2.2.3](../ex-2.2.3/report.py).

Verification lines
: Allowed, since they leave completion unchanged; the `hinge` condition trains without them.

    From [ex-2.2.21](../ex-2.2.21/report.py).

Edit
: Remove a share of the component along e₁ from the state at every position and every slice. That share is the *dose*, from a quarter to all of it.

    From [ex-2.2.21](../ex-2.2.21/report.py).

Scoring
: Task: the probability the model puts on the right answer on held-out contexts (expected exact match), beside the Bayes ceiling, which is the best any model could do given the examples, and the control, a model trained without the anchor. Removal: the drop on `difference` has to grow with the dose and reach at least half the way to the *target null*, what an ideal predictor that had lost `difference` and nothing else would answer. Selectivity: no other op may drop by more than 0.02 at any dose (the *selectivity gate*). Both drops are net of what the same edit does to the control.

    From [ex-2.2.16](../ex-2.2.16/report.py) and [ex-2.2.21](../ex-2.2.21/report.py).

## What we want to be able to say

Ultimately, we want to see whether SCA works in transformers as a general alignment technique. For the current, narrow domain of color-mixing, we claim:

### (a) An inferred op can be anchored without costing the task

The pull puts `difference` contexts on the axis, and the model answers as well as one trained without the anchor.

**Observations.** The anchor sits on the example answers, where the context has shown the most about its op, and hardly at all at the query `=`, where the answer is predicted. At those answers the alignment rises with the posterior, on every anchored condition, while the control stays flat near zero. It does so at a fixed answer index too, so it is not only that evidence builds up along a context. That is what we would hope to see if the anchor holds the op the model inferred, and not a shortcut such as a characteristic answer color, which would not follow the evidence.

Whether the anchor costs the task anything is still open. With the pull uncapped, the anchored condition fell short of the control by more than our tolerance of 0.01 and by less than the seeds vary; with the cap, the task score varies more from seed to seed than from condition to condition. One seed takes a slow path through training on most anchored conditions, and five seeds at 200 epochs cannot tell a small cost from a slow start. Also still to show: that the rest of what the model represents is unchanged, which was measured for the op word and not yet for the inferred op.

From [ex-2.2.21](../ex-2.2.21/report.py) (H1, E1), [ex-2.2.22](../ex-2.2.22/report.py) (E1, E3, E5), and [ex-2.2.14](../ex-2.2.14/report.py) (the probe scan).

### (b) Removing the op is graded and selective

Graded means that a stronger edit takes out more of `difference`. Selective means that the other ops are left as they were, at every dose. In M1 this was the dose-response curve: a stronger edit removed more of the concept, and colors unrelated to it stayed as they were.

**Observations.** On the earlier grammar, where the concept was *red*, projecting the axis out removed *red*, more so the redder the line, and some of the cost fell on lines with no red in them. That cost came through the embeddings of syntax tokens, and a readout table of their own cleaned them. With that change, removal was clean on ten of eleven ops.

On the in-context grammar, an edit at the query `=` has little to act on, since the anchor sits on the example answers, and none met both criteria. The edit at every position does work. On the `hinge` condition, turning it up takes `difference` out step by step, and the other ops mostly stay as they were, though not yet on every run. Editing the example answers alone takes out most of the op, so it seems the query takes the op from the examples.

Across the caps tried in ex-2.2.22, from 0.8 to none, the runs that spilled past the gate were single runs, and along the caps every one of them had trained quickly.

Still to show: that the edit stays within the gate on every run, at fresh seeds; and that the damage on each context follows how much its answer depended on `difference`, which the target null predicts context by context.

From [ex-2.2.1](../ex-2.2.1/report.py), [ex-2.2.7](../ex-2.2.7/report.py), and [ex-2.2.11](../ex-2.2.11/report.py) (*red*); [ex-2.2.21](../ex-2.2.21/report.py) (E2) and [ex-2.2.22](../ex-2.2.22/report.py) (E1, E2).

### (c) The edited model answers in a predictable way

Once `difference` is removed, the model should answer as an ideal predictor that never knew `difference` but still weighs the other ops by how well they fit the examples: the target null. If it does, the edit has a destination we can state in advance, and we may not need to train one.

**Observations.** Looked at after the fact, the edited `hinge` runs answered `difference` contexts much as that predictor does: the mass leaves the `difference` answer and goes to the other ops in about the proportions the posterior gives. On three fresh runs, counted by op over all contexts, the mass goes about where the target null puts it. Context by context the match is looser: the edit moved each run a little under half of the way to the target null, short of the half we flat-out guessed at. Most of the distance that remains is on answers another op could give, shared among them differently from the target null; the edit moves little mass onto colors no op gives.

So the anchor alone gives the edit a direction and not the whole distance. Whether a trained fallback, a designed response to the edit, would close the rest, or whether the recipe can be improved so that it does, is open.

From [ex-2.2.21](../ex-2.2.21/report.py) (E2, post hoc) and [ex-2.2.22](../ex-2.2.22/report.py) (H2, E4).

### (d) The side-effects are bounded by construction, and we know at which layers the op is held

Because we placed the axis, we can say in advance how far an edit moves each state: the write bound.

**Observations.** The write bound held on *red*: at every slice past the embedding, the move on lines with no red in them stayed within the bound, or at most a quarter over it, as we allowed in advance. *Red* was read in the first two blocks. For the inferred op, the alignment on `difference` contexts rises through the stack.

Still to show: the write bound on the inferred op; which slices the edit needs, by editing at a prefix or suffix of slices on stored checkpoints; and which slices the pull needs. Leaving slices out of the pull by hand made the edit spill (ex-2.2.22), so the next try lets training choose, by pooling over slices.

From [ex-2.2.1](../ex-2.2.1/report.py) (H4) and [ex-2.2.22](../ex-2.2.22/report.py) (E1).

### (e) Scarce, noisy labels are enough

About one `difference` context in fifty is labelled, and the label says only that the context uses `difference`. That is the shape of label we expect to have for a concept in natural language, where labels will be scarce and sometimes wrong.

**Observations.** At that rate the op is anchored, on every condition tried. On the op word, labelling one `difference` line in fifty placed the op as well as labelling every `difference` line, and so did the same scarce labeller with a fifth of its labels on the wrong op. The labels here are already wrong in one sense: a label marks the op that generated the context, even when its examples favor another op.

Still to show: wrong labels on the inferred op.

From [ex-2.2.14](../ex-2.2.14/report.py) (the label arms) and [ex-2.2.21](../ex-2.2.21/report.py) (S1).

## What comes next

Each step below is written as the opening line we hope to write once it has run.

*The slow seeds.* Trained for longer, the runs that stalled on a plateau made the second rise and ended like the others, so a rule for runs that miss it could be fixed before the next experiment.

*Anchoring and removing the op at fresh seeds.* At enough seeds to tell a small cost from a slow start, anchoring `difference` cost the task nothing we could measure, and editing the axis out took the op away gradually on every run while the other ops stayed where they were. The damage on each context followed how much its answer depended on `difference`.

*Where the edited model ends up.* A fallback trained toward the target null brought the edited model the rest of the way there, with less scatter across seeds than the anchor alone gave; or the anchor alone turned out to be enough once the recipe settled, and no fallback was needed.

*The layer sweep.* Editing the first few slices removed as much of the op as editing all of them, and anchoring those slices alone kept the task score and the selectivity, so the bound could be stated for the slices that matter. Beside it, a baseline trained on the same labels (SGTM) showed what anchoring adds over a method that does not place the concept.

## Out of scope

The verification measurements (D2.3), though verification lines are allowed in the corpus. Topic markers, and anchoring at fine-tune time, both open questions in the [pivot](pivot.md#open-questions). Several ops on separate axes. RMU and SAE baselines, which wait for D2.3.

<details markdown="1"><summary>About this page</summary>

Rewritten 2026-10-06, to say where D2.2 stands in one sitting. The plan as it stood through ex-2.2.22, with its route, risk table, and decisions, is [archived](design-2026-10.md) and in git. Updated 2026-10-07 with the [ex-2.2.22](../ex-2.2.22/report.py) decision: no cap, every slice, three examples per context.

</details>
