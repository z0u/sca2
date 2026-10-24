# D2.2 design: anchoring an operation the model infers

D2.2 asks whether Sparse Concept Anchoring works for a concept that is never written down: the model has to infer it from context. Each line shows a few solved equations that use the same hidden operation, and the model has to infer the op to answer the last equation. We anchor one op to the first axis of the residual stream, then remove it by editing that axis out.

Each line of the corpus is one context: three solved examples of one op, then a query under the same op, written `op1 ? op2 = answer` with a constant `?` where the op word would be. The model reads the examples, infers which op fits them, and completes the query. We can compute how strongly each context indicates each op (the posterior over ops), and we evaluate the predictive power of each model against that, with and without being edited to remove the anchored op.

On the recipe of record, the anchor holds at a small cost to the task. Turning the edit up takes `difference` out gradually, and at eight fresh seeds the other ops stayed within the selectivity criterion on every run. Close variants of the recipe sometimes lose the anchor in the first tens of epochs, so how often the recipe itself keeps it is known only roughly.

A re-analysis of the recipe of record, the [edit-effects report](../edit-effects/report.py), measured the rest: the removal follows how much each context needed the op, the write stays within the bound, and the edited model sends the mass to the right ops overall but weighs them differently from the target null context by context.

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
: 400 epochs, cosine schedule peaking near 0.003. At 200 epochs a few anchored runs missed the second rise in task skill; at 400 every run made it.

    From [ex-2.2.17](../ex-2.2.17/report.py), [ex-2.2.19](../ex-2.2.19/report.py), and [ex-2.2.20](../ex-2.2.20/report.py) (schedule); [ex-2.2.23](../ex-2.2.23/report.py) (length).

Anchored op
: `difference` on e₁, with no *red* anchor.

    From [ex-2.2.14](../ex-2.2.14/report.py).

Labels
: The whole context, on about one `difference` context in fifty. The label says only that the context uses `difference`; the model never sees it, and no other op is labelled.

    From [ex-2.2.14](../ex-2.2.14/report.py) and [ex-2.2.21](../ex-2.2.21/report.py).

Pull
: On a labelled context, the anchor term pulls blocks 2 to 4, pooled over the positions of the context at a temperature $\tau_p$ of 0.1, so most of the pull lands on the example answers. There is no cap on the alignment. A context the training window cuts short is not pulled. The embedding and block 1 are left out because, before the model has read the context, the pull finds a stand-in there (see the lean, under (b)). A pull weight that ramps with depth in place of the step lost the anchor on more seeds, and a $\tau_p$ that starts soft spilled more, so the step and the fixed temperature stay for now.

    From [ex-2.2.15](../ex-2.2.15/report.py) (whole contexts), [ex-2.2.22](../ex-2.2.22/report.py) (no cap), the [τ × λ_a sweep](../tau-lambda-sweep/report.py) (τ_p), [late-pull](../late-pull/report.py) (the slices), and [anchor-ramp](../anchor-ramp/report.py) and [soft-pool](../soft-pool/report.py) (the alternatives).

Anchor schedule
: The anchor weight $\lambda_a$ (peak 0.1) warms up over the first tenth of training, holds, then eases over the last tenth to a tenth of its peak. A second term, the anti-subspace term, pushes the states of every other context off e₁. Its weight $\lambda_{\bar s}$ is set by slice, as an absolute number: 0.25 on every slice at the start, annealing by epoch 360 to a *hold* that falls off in a straight line with depth, from 0.2 at the embedding to 0.03 at block 4 (the `falloff` arm). The high hold on the first slices keeps the stand-in off e₁; the low hold on the pulled blocks leaves room for the pull.

    From [ex-2.1.10](../ex-2.1.10/report.py) (anchor weight), [slice-terms](../slice-terms/report.py) (weights by slice), and [split-anti-seeds](../split-anti-seeds/report.py) (the falloff).

Verification lines
: Allowed, since they leave completion unchanged; the `hinge` condition trains without them.

    From [ex-2.2.21](../ex-2.2.21/report.py).

Edit
: Remove a share of the component along e₁ from the state at every position and every slice. That share is the *dose*, from a quarter to all of it. Edits on a subset of slices are used to locate the op.

    From [ex-2.2.21](../ex-2.2.21/report.py).

### Scoring

Every run is scored on held-out contexts against the control, a model trained without the anchor at the same seed, and against a baseline of edits along other axes (below).

Task
: The probability the model puts on the right answer (expected exact match), beside the Bayes ceiling, which is the best any model could do given the examples. The task gate fails if the mean change against the control falls below −0.02, or the change on any one op below −0.05.

Removal
: The drop on `difference` under the edit, as a share of the way to the *target null*: what an ideal predictor that had lost `difference` and nothing else would answer. It has to grow with the dose.

Selectivity
: No other op may drop by more than 0.02 at any dose (the *selectivity criterion*). The largest such drop is the *spill*. Both drops, here and under removal, are net of the baseline below.

Landing
: The share of the distance to the target null that the full edit covers, context by context.

Lost anchor
: A run has lost the anchor when the full edit removes less than half of the op.

The anchor direction and the baseline
: e₁ is a coordinate axis, and under Adam the coordinate axes are a privileged basis: the model puts its separators one-hot on one axis and `=` on another, and leaves most of the rest quiet, so e₁ in a control is usually a quiet axis. The anchor does not need this. With the basis of the optimizer rotated, so that e₁ is an ordinary direction, `falloff` held and removed as before on all eight seeds. The baseline does need it: netting against the control assumes the edit takes little from the control, which fails on seeds where e₁ overlaps the syntax spike of the control. So a reference edit compared with e₁ is another coordinate axis off the syntax axes, not a random direction, and raw removal is reported beside net removal whenever the basis of the optimizer changes. The spike itself is there under every optimizer tried; an optimizer with no privileged basis only moves it off the axes, which makes a typical direction dearer to remove.

    From [rotated-basis](../rotated-basis/report.py) and the [optimizer scout](../optimizer-scout/report.py).

Baseline
: The median drop over the same edit along every other coordinate axis (all but e₁), per op and dose, in the control at the same seed. It replaces the drop the e₁ edit causes in the control, which depends on where e₁ happens to sit there, from among the quietest axes to among the busiest few. The same median taken in the anchored run itself is reported beside it, and so is raw removal, with nothing subtracted. On the stored runs of `falloff` the two medians agree within 0.02 of removal, and the numbers quoted before the change barely move (mean removal 0.85 against 0.87); on the rotated-basis runs the median brings the two seeds where the paired baseline fell short back beside the rest. Where the optimizer acts in another basis, the axes are those of that basis. Experiments from before this rule quote the paired baseline.

    From [reference-baseline](../reference-baseline/report.py).

From [ex-2.2.16](../ex-2.2.16/report.py), [ex-2.2.21](../ex-2.2.21/report.py), [ex-2.2.22](../ex-2.2.22/report.py), and [split-anti-seeds](../split-anti-seeds/report.py) (task gate and lost anchors).

## What we want to be able to say

Ultimately, we want to see whether SCA works in transformers as a general alignment technique. For the current, narrow domain of color-mixing, we claim:

### (a) An inferred op can be anchored on every run, without costing the task

The pull puts `difference` contexts on the axis on every seed, and the model answers as well as one trained without the anchor.

**Observations.** The anchor sits on the example answers, where the context has shown the most about its op, and hardly at all at the query `=`, where the answer is predicted. At those answers the alignment rises with the posterior, on every anchored condition, while the control stays flat near zero. It does so at a fixed answer index too, so it is not only that evidence builds up along a context. That is what we would hope to see if the anchor holds the op the model inferred, and not a shortcut such as a characteristic answer color, which would not follow the evidence.

The cost to the task is small. At eight fresh seeds and 400 epochs, the recipe of record scored 0.007 below the control on average. It missed the task gate once, on `hue-hsv` at seed 708, a little past the limit of 0.05. Every loss past that limit, in every arm of that experiment, was on one of the HSV ops. Training for 400 epochs settled the earlier question of slow seeds, which at 200 epochs could not be told apart from a cost. The anchor does not land on every run, and this is the largest gap in the recipe: a method that loses the concept on one run in eight would have to be checked run by run. `falloff` kept it on all eight seeds it has run at, which fits a loss rate anywhere from none to about three in ten. Its close variants lost it on one to three seeds of eight: the uniform anti holds, the pulls that ramp with depth, and the ramp over blocks 1 to 3. On most lost runs the example answers never come onto e₁, while the query answer aligns as usual. The lost runs part from the kept ones in the first tens of epochs, while the pull warms up: at $\tau_p$ = 0.1 the pool gives nearly all of the pull to the position closest to e₁, which early on is the query answer, and on a lost run the example answers fall back. A $\tau_p$ that starts soft and sharpens by the end of the warm-up kept the anchor on every seed in every arm that had it, though the edit spilled more (see (b)). That is 24 runs with the soft start and no loss, against losses in every arm with a ramp at the fixed temperature, though the soft arms differ in more than the temperature. One lost run had its example answers come back onto e₁ late in training, and the edit still removed almost nothing, so a run is judged by its removal and not by its alignment alone.

Still to show: a recipe that keeps the anchor on every seed, at enough seeds to say the loss rate is low. And beyond the task score, which is unchanged, that the rest of the representation keeps its shape: the color geometry and the states of the other ops, which the probe scan measured for the op word and not yet for the inferred op.

From [ex-2.2.21](../ex-2.2.21/report.py) (H1, E1), [ex-2.2.22](../ex-2.2.22/report.py) (E1, E3, E5), [ex-2.2.23](../ex-2.2.23/report.py), [split-anti-seeds](../split-anti-seeds/report.py) (Decision), [anchor-ramp](../anchor-ramp/report.py) (E1, E2), [soft-pool](../soft-pool/report.py) (E1, E2), and [ex-2.2.14](../ex-2.2.14/report.py) (the probe scan).

### (b) Removing the op is graded and selective

Graded means that a stronger edit takes out more of `difference`. Selective means that the other ops are left as they were, at every dose. In M1 this was the dose-response curve: a stronger edit removed more of the concept, and colors unrelated to it stayed as they were.

**Observations.** On the earlier grammar, where the concept was *red*, projecting the axis out removed *red*, more so the redder the line, and some of the cost fell on lines with no red in them. That cost came through the embeddings of syntax tokens, and a readout table of their own cleaned them. With that change, removal was clean on ten of eleven ops.

On the in-context grammar, an edit at the query `=` has little to act on, since the anchor sits on the example answers, and none met both criteria. The edit at every position does work: turning it up takes `difference` out step by step. Editing the example answers alone takes out most of the op, so it seems the query takes the op from the examples.

Trained to 400 epochs on the earlier recipe, which pulled every slice, the edit spilled onto other ops on nearly every run that learned the HSV ops. The spill came from a stand-in, the *lean*. At the embedding and block 1 the model has not yet read the context, so it meets the pull with a token-level substitute: it loads lightness onto e₁ in the color embedding table (darker colors are the nearest thing to a `difference` answer), or latches a syntax embedding onto e₁. The lean fades with depth, but the edit at those slices takes the lightness away from every op. Pulling only blocks 2 to 4 cut the lean to about a quarter, and holding the anti-subspace weight high on the first two slices kept it off.

On that recipe, with the anti weight falling off with depth, the full edit removed 0.85 of the way to the target null on average at eight fresh seeds, and stayed within the selectivity criterion on every seed. Holding the anti weight at one level on every slice lost the anchor altogether on one seed in each arm.

A pool that starts soft makes the edit less selective, at least on its own, with no other change such as a lower $\lambda_a$ or a higher $\lambda_{\bar s}$. Each soft arm went over the criterion on two to four seeds of eight, where the two arms at the fixed temperature stayed within it on every seed. While the pool is soft its pull reaches every position, and the other ops come partway onto e₁ during the rise; that lift is gone by the end of training, but it goes loosely with the spill. The [τ × λ_a sweep](../tau-lambda-sweep/report.py) found the same for a fixed temperature that is softer throughout.

On the recipe of record the drop on each context follows how much it needed `difference`: the full edit takes about the same share of the way to the target null whatever the posterior on `difference` (0.75 to 0.88 by bin, net of the control). The contexts of other ops whose examples point at `difference` gain, most of the way the target null predicts ([edit-effects](../edit-effects/report.py)).

From [ex-2.2.1](../ex-2.2.1/report.py), [ex-2.2.7](../ex-2.2.7/report.py), and [ex-2.2.11](../ex-2.2.11/report.py) (*red*); [ex-2.2.21](../ex-2.2.21/report.py) (E2), [ex-2.2.22](../ex-2.2.22/report.py) (E1, E2), [ex-2.2.23](../ex-2.2.23/report.py), [spill-by-position](../spill-by-position/report.py), [embedding-lean](../embedding-lean/report.py), [late-pull](../late-pull/report.py), [slice-terms](../slice-terms/report.py), [split-anti-seeds](../split-anti-seeds/report.py) (H1, H2), and [soft-pool](../soft-pool/report.py) (E1, E3).

### (c) The edited model answers in a predictable way

Once `difference` is removed, the model should answer as an ideal predictor that never knew `difference` but still weighs the other ops by how well they fit the examples: the target null. If it does, the edit has a destination we can state in advance, and we may not need to train one.

**Observations.** Looked at after the fact, the edited `hinge` runs answered `difference` contexts much as that predictor does: the mass leaves the `difference` answer and goes to the other ops in about the proportions the posterior gives. On three fresh runs, counted by op over all contexts, the mass goes about where the target null puts it. Context by context the match is looser: the edit moved each run a little under half of the way to the target null, short of the half we flat-out guessed at. Most of the distance that remains is on answers another op could give, shared among them differently from the target null; the edit moves little mass onto colors no op gives.

On the recipe of record the full edit lands about half of the way (0.52 on average at eight fresh seeds). The edited answers are close to a mixture of the answers of the other ops, in about the right overall proportions, but within a context they weigh those ops differently from the target null, with no common tilt toward flatter or sharper weights. The distance left is about 2.4× the distance from the clean model to the Bayes predictor ([edit-effects](../edit-effects/report.py)).

So the anchor alone gives the edit a direction and not the whole distance. Whether a trained fallback, a designed response to the edit, would close the rest, or whether the recipe can be improved so that it does, is open.

From [ex-2.2.21](../ex-2.2.21/report.py) (E2, post hoc), [ex-2.2.22](../ex-2.2.22/report.py) (H2, E4), and [split-anti-seeds](../split-anti-seeds/report.py).

### (d) The side-effects are bounded in advance, and we know at which layers the op is held

Because we placed the axis, the clean model tells us in advance how far an edit can move the states of other concepts. That the edit turns a state by an angle set by its alignment holds by construction. What has to be shown is downstream of the edit: the blocks after an edit put no more of the axis back on the states of other concepts than the clean model has there. That is the write bound.

**Observations.** The write bound held on *red*: at every slice past the embedding, the move on lines with no red in them stayed within the bound, or at most a quarter over it, as we allowed in advance. *Red* was read in the first two blocks. For the inferred op, the alignment on `difference` contexts rises through the stack.

With the pull on blocks 2 to 4, editing block 1 alone removes most of the op on every seed, though nothing pulls block 1; we don't yet know why. Editing blocks 2 to 4 alone removes nearly all of it on most runs and falls short on the same seed in every arm, which fits the seed rather than the terms deciding where it misses. Pooling the pull over slices as well as positions settled it on block 4 on most seeds, and on the query answer on one, where the op could not be removed; it was not adopted.

On the runs that keep the anchor, where it sits by depth hardly depends on how the pull is spread over blocks 2 to 4. Under the step and under every ramp, the alignment at the example answers rises to about 0.64 at block 4. With nothing pulling block 4, it falls back there to about three quarters of its level at block 3, and the edit still finds it, since it acts on every slice.

The states are unit vectors, and the edit takes out the e₁ component and renormalizes, so it turns each state by an angle set by its alignment, arcsin α. On *red*, the bound at each slice and position was the angle at the 99th percentile of the clean α over lines with no red, and the write was the angle of the α arriving at the edit, which can differ from the clean α once an earlier slice has been edited. For the inferred op the same map runs over slices and over the roles of every equation in the context, on contexts of the other ops. The anti term is what keeps it small, so we expect the bound to be loosest at block 1, where the other ops sit closest to e₁ (α about 0.18 at the example answers on `falloff`, seed mean). The write at a slice is the alignment a state arrives with there, after the block before it has read the states the earlier edits turned, so the bound tests what each block puts back. What the later blocks do with a turned state beyond that, in the answers, is what the spill measures.

On the recipe of record the write stays within the bound at every slice past the embedding, at most 1.08 times it, and from slice 2 on the other ops arrive with less e₁ than either the clean model or the control has ([edit-effects](../edit-effects/report.py)).

From [ex-2.2.1](../ex-2.2.1/report.py) (H4), [ex-2.2.22](../ex-2.2.22/report.py) (E1), [late-pull](../late-pull/report.py) (E2), [slice-terms](../slice-terms/report.py) (E1, E3), [split-anti-seeds](../split-anti-seeds/report.py) (E1), [anchor-ramp](../anchor-ramp/report.py) (E3), and [soft-pool](../soft-pool/report.py) (E3).

### (e) Scarce, noisy labels are enough

About one `difference` context in fifty is labelled, and the label says only that the context uses `difference`. That is the shape of label we expect to have for a concept in natural language, where labels will be scarce and sometimes wrong.

**Observations.** At that rate the op is anchored, on every condition tried. On the op word, labelling one `difference` line in fifty placed the op as well as labelling every `difference` line, and so did the same scarce labeller with a fifth of its labels on the wrong op. The labels here are already wrong in one sense: a label marks the op that generated the context, even when its examples favor another op.

Still to show: wrong labels on the inferred op.

From [ex-2.2.14](../ex-2.2.14/report.py) (the label arms) and [ex-2.2.21](../ex-2.2.21/report.py) (S1).

## What comes next

The recipe of record is settled enough to measure the claims on. The open work is of six kinds: measuring what D2.2 still has to show, making the anchor land on every run, comparing with other methods, choosing the baseline with the basis in mind, tuning the recipe, and understanding where the op is held. Every change to training below uses only what a real corpus would give us: a label per context, term weights, schedules, and temperatures. None is keyed to the role of a position, such as the example answers, since real text has no roles marked.

### The claims still to show

*The write bound, the grading, and the landing.* Done in the [edit-effects report](../edit-effects/report.py), a re-analysis of the stored `falloff` runs; the results are under (b), (c) and (d). It is cheap to repeat on a later recipe.

*How far an edit could land.* The full edit covers about half of the distance to the target null. A model trained with no `difference` contexts would show how close any model comes, and so whether half is short. It is also the filtered-corpus baseline below. If half is short, a fallback trained toward the target null is the designed response from M1, to be tried again on the current recipe. In the backlog: [a reference model](/todo/science/reference-model-without-the-anchored-op.md) and [fallback control](/todo/science/fallback-control-on-the-untied-recipe.md).

*Wrong labels on the inferred op.* One arm with a fifth of the labels on contexts of other ops, as was done for the op word, at the seeds of the recipe (e). In the [backlog](/todo/science/wrong-labels-on-the-inferred-op.md).

### An anchor on every run

*How often the anchor is lost.* With no losses in eight, the loss rate of `falloff` could be as high as three in ten (the 95% upper bound). Eight more seeds with no loss would bring that bound to about one in six, 24 more to about one in eleven, and showing it is under one in twenty would take about 60 seeds with no loss. Full runs at that count cost more than the question is worth, but the outcome is seemingly set in the first forty or so epochs, so a screen could stop early: train new seeds on the full-length schedule to about epoch 60, and call a run lost when the example answers have not come onto e₁. The stored trajectories of about a hundred runs whose outcome we know would show first whether that call agrees with the removal. The one late loss, in the soft-pool experiment, had already parted from the kept runs early (its example answers rose partway, fell back during the warm-up, and stayed below 0.4 until about epoch 100), so the screen would likely flag it. Two kept runs of the anchor-ramp experiment climbed slowly, crossing 0.4 only at epochs 60 to 80, so the cutoff needs choosing on the stored runs. Screened this way, 32 seeds of `falloff` and 32 of the soft start would cost about as much as ten full runs (under \$5), and compare the two loss rates as well as bounding them. A short report of its own. In the [backlog](/todo/science/falloff-at-more-seeds.md).

*Keeping the soft start and its anchors, without the spill.* The soft start kept the anchor on all 24 of its runs, so it is the lead for a recipe that lands every time, if its spill can be kept down. The arms to run, each against the soft ramp over blocks 1 to 3 or the soft step: a stronger anti term while the pool is soft; the two schedules offset, so that $\tau_p$ is sharp before $\lambda_a$ has warmed up far; the anti term off on block 4 where the ramp does not pull; and a later $\lambda_a$ warm-up at the fixed temperature. $\lambda_a$ already rises from zero over the same forty epochs that $\tau_p$ sharpens, so it is low while the pool is soft, and the offset makes the soft stretch a lighter one without lowering the weight after the warm-up. The ramps that pulled less in total kept about the same removal on the runs that kept the anchor, but lost it on more seeds, so a weight that is lower only early has to be judged on the losses too. The last arm tells the soft start apart from a plain delay; if a delay keeps the anchor as well, it might do so without the extra spill of a soft pool. In the [backlog](/todo/science/soft-pool-spill-follow-ups.md).

### Comparing with other methods

The blog post on D2.2 should show what anchoring adds over methods that do not place the concept. Three comparisons fit before D2.3, through the same scorer (`sca.intervention`), in order of cost:

- Post-hoc directions on the controls. Fit a direction for `difference` on the states of the un-anchored controls at seeds 704 to 711 (diff-in-means, a ridge probe, and LEACE, the linear eraser), with the same scarce labels, and edit along it. No training. On *red*, in ex-2.2.1, these directions cost the lines with no red about as much as they removed from red, since the un-anchored model keeps red on directions the mix also uses.
- A filtered corpus: the model trained with no `difference` contexts, the same runs as the reference model above.
- Gradient routing, a training-time method that masks gradients so that labelled examples update only parameters set aside for the concept, which are then removed. We would run it in its SGTM form (selective gradient masking), which made it more robust to label noise. Its own experiment, on the same labels.

RMU, SAE and LUNAR rows wait for D2.3. In the [backlog](/todo/science/baseline-comparisons-sca-plan-related-work-delta.md).

### The baseline and the basis

*A baseline without a control.* The median over the quiet axes is now the baseline (Scoring). Taken in the anchored run itself, it agreed with the one taken in the control on every stored run of `falloff` and of rotated-basis ([reference-baseline](../reference-baseline/report.py)). If that keeps holding, an experiment could train controls only now and then, to check the agreement, rather than one per seed.

*An anchored arm under Muon* (low priority). Muon learns the task as well as Adam at three times the rate its finder suggests, with no privileged basis. An anchor there would have no quiet axis to take, so it would need its own term weights, and a finer rate ladder first. Worth running if the project moves toward models trained without Adam, or to see whether an anchored color grades differently. In the [backlog](/todo/science/anchored-arm-under-muon.md).

### Tuning the recipe

*The full recipe sweep around `falloff`.* The weights and schedules have changed one at a time since the D2.1 survey and have not been tuned together. As in [ex-2.1.11](../ex-2.1.11/report.py): replace each schedule with flat levels to see which can go, then survey what is left (the anchor weight, the two ends of the anti falloff, $\tau_p$, the peak learning rate) with about 32 trials at one seed, spread evenly over the space by a Sobol sequence, and confirm the best few at three seeds. One seed per trial will rarely show a loss rate of one in eight, so the survey scores removal on every trial and leaves the loss rate to the confirmation. The sweep could also tune the falloff for its one task-gate miss, on `hue-hsv`, and the anchor-ramp losses suggest the pull on the early blocks as a dimension of its own. In the [backlog](/todo/science/full-recipe-sweep-after-the-lean-fix.md).

*An anti weight by slice that needs no end points.* The two ends of the falloff are set by hand. An anti term pooled over slices would put its weight where the states sit closest to e₁. But it acts on every context, labelled or not, and how to pool it there without catching the pull is unclear. Part of this may fit in the sweep. In the [backlog](/todo/science/principled-per-slice-anti-weight.md).

*Longer training* (low priority). A few runs at [600 epochs](/todo/science/spill-at-600-epochs.md) would show whether the hold still keeps the lean off with more training.

### Where the op is held

*Why block 1 alone holds most of the op.* Editing block 1 alone removes most of the op on every seed, though nothing pulls block 1, and we have no account of it yet. An op probe at each slice and position of the `falloff` runs would show where the op is readable before block 2; the alignment is already stored, so only the probes are new. It bears on the write bound and on the layer sweep. In the [backlog](/todo/science/what-e1-holds-beside-the-op.md).

*The layer sweep.* Whether editing the first few slices removes as much of the op as editing all of them, and whether anchoring those slices alone keeps the task score and the selectivity, so the bound could be stated for the slices that matter.

### A suggested order

We would start with the re-analysis of the claims: it needs no training, D2.2 needs it whichever recipe wins, and what it finds may change what the sweep should score. The loss-rate screen could run beside it, as a short report of its own, and the post-hoc comparison is another re-analysis that could follow. After those, the soft-start follow-ups, which aim at the anchor landing on every run, come before the full sweep, which is easier to read once the loss rate is known.

## Out of scope

The verification measurements (D2.3), though verification lines are allowed in the corpus. Topic markers, and anchoring at fine-tune time, both open questions in the [pivot](pivot.md#open-questions). Several ops on separate axes. RMU and SAE baselines, which wait for D2.3.

<details markdown="1"><summary>About this page</summary>

Rewritten 2026-10-06, to say where D2.2 stands in one sitting. The plan as it stood through ex-2.2.22, with its route, risk table, and decisions, is [archived](design-2026-10.md) and in git. Updated 2026-10-07 with the [ex-2.2.22](../ex-2.2.22/report.py) decision: no cap, every slice, three examples per context. Updated 2026-10-13 for the work since: 400 epochs (ex-2.2.23), the lean and the pull on blocks 2 to 4 (embedding-lean, late-pull), and the anti weight by slice with the falloff as the recipe of record (slice-terms, split-anti-seeds). Updated 2026-10-21 for the anchor-ramp and soft-pool experiments (the step pull and fixed τ_p stay; lost anchors and the spill of a soft pool), with the next steps rewritten as open choices. Revised the same day after review: lost anchors as a gap in claim (a), the scoring as its own section, what the write bound is in this grammar, a loss-rate screen, and the comparisons with other methods. Claim (d) restated 2026-10-22, in review of the [edit-effects report](../edit-effects/report.py): the turn being set by the alignment holds by construction, so the claim now rests on what the blocks after an edit put back. Updated 2026-10-24 for the [rotated-basis](../rotated-basis/report.py) experiment and the [optimizer scout](../optimizer-scout/report.py): e₁ is a privileged axis under Adam, the anchor does not depend on it, and the baseline does. Updated the same day for [reference-baseline](../reference-baseline/report.py): the baseline is now the median over the other coordinate axes.

</details>
