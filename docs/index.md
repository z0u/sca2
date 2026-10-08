# Sparse Concept Anchoring in transformers

Sparse Concept Anchoring (SCA) is a training-time method for concept control. A light geometric regularizer, driven by a small number of noisy labels, guides a chosen concept toward a known location in representation space. The geometry is shaped during training, so there is no need to reverse-engineer it afterwards: the concept lives where you put it, and the side effects of suppressing or ablating it can be bounded before running the intervention.

The first milestone (M1) established the method in autoencoders: [paper](https://arxiv.org/abs/2512.12469), [blog post](https://www.lesswrong.com/posts/sGskzx7LgsDkMLvcv/intervening-on-sparse-anchored-concepts), [code](https://github.com/z0u/ex-preppy). This site holds the experiment reports for the second milestone (M2), which asks: does SCA transfer to transformers? We train a small transformer to do vector math and anchor concepts in its residual stream. The vectors are colors, in a synthetic color-mixing task (`red + blue = purple`) where ground truth is unambiguous, so a negative result stays interpretable. The plan and deliverables are in the [project README](https://github.com/z0u/sca2).

![Milestone map: M1 (autoencoders) complete, M2 (transformers) in progress, M3 (language models) and M4 (LLM fine-tunes) planned. Within M2, D2.1 (anchor a concept) is complete, D2.2 (operations and steering) in progress, D2.3 (asymmetric verification) planned, and D2.4 (publication) already under way.](./public/milestones.svg)

&nbsp;

<!-- toc -->

## Experiments

Each report is a literate script.[^lit] Each entry has searchable tags and a strip of the report's figures, in reading order.

[^lit]: A literate script is plain Python with Markdown prose between the cells. It reads durable results produced by a separately-run experiment. Reports are published automatically, with their figures served from a Hugging Face dataset; the infrastructure is [mi-ni](https://github.com/z0u/mi-ni).

<!-- These URLs are rewritten to point to the published notebooks, and the mini:figures markers become thumbnail strips (scripts/build_site.py) -->

## Iteration 0 (prep)

These experiments were preparation for the main work: exercising the infrastructure, testing the normalized transformer architecture, and tying off some loose ends from M1.

<details markdown="1"><summary>Reports</summary>

- [nGPT scaling](./ngpt-scaling/report.py)

    Width × depth sweep of our simplified nGPT to check that it stays well-behaved as it grows. Converged loss improves with width and is flat across depth, and no condition spikes or stalls. The architecture seems safe to build the color-mixing experiments on.

    <span class="tags">`ngpt` `scaling` `architecture` `infra`</span>

    <!-- mini:figures ./ngpt-scaling/report.py -->

- [M1 2.9.1. Deleting _red_, now in JAX](./m1/ex-2.9.1/report.py)

    We ported the main M1 result from PyTorch to JAX: Anchor _red_ to one latent axis of a small autoencoder, then zero the axis and watch red disappear. The result reproduces, seed sensitivity and all, and our new infrastructure holds up.

    <span class="tags">`anchoring` `ablation` `jax-port` `seed-sensitivity`</span>

    <!-- mini:figures ./m1/ex-2.9.1/report.py -->

- [M1 2.9.2. Fallback control](./m1/ex-2.9.2/report.py)

    We compared two remedies for ablation seed-sensitivity: **1.** optimal ablation, which picks a replacement constant after training, and **2.** fallback control, a decoder-only loss term that teaches the model what to output once the concept has been removed. Fallback control worked well. Neither remedy touches the other half of the variance, where anchoring itself fails.

    <span class="tags">`ablation` `fallback-control` `optimal-ablation` `seed-sensitivity`</span>

    <!-- mini:figures ./m1/ex-2.9.2/report.py -->

- [M1 2.9.3. When anchoring fails](./m1/ex-2.9.3/report.py)

    Every failing run anchors first and then comes apart during the high learning-rate plateau. Halving the peak learning rate removes the failures.

    <span class="tags">`anchoring-failures` `training-dynamics` `learning-rate` `schedules`</span>

    <!-- mini:figures ./m1/ex-2.9.3/report.py -->

- [M1 2.9.4. Closed-loop regularizer weights](./m1/ex-2.9.4/report.py)

    We replaced the timed regularizer anneal with a feedback controller, so that each weight climbs while its constraint is being violated and settles back once it is met. It kind of worked but not very well, and it adds complexity.

    <span class="tags">`feedback-controller` `regularizer-weights` `schedules`</span>

    <!-- mini:figures ./m1/ex-2.9.4/report.py -->

</details>

## D2.1: anchoring in a transformer

D2.1 asked whether SCA works in a transformer at all. We designed a task where a small transformer learns the geometry of color on its own, then anchored _red_ to a chosen direction in its residual stream. With a repulsive term, a schedule for it, and a pull that finds the red operand by itself, _red_ landed where we put it, graded by how red each color is, at no measurable cost to the task.

<details markdown="1" open="true"><summary>Reports</summary>

- [2.1.1. Un-anchored color-mixing transformer](./m2/ex-2.1.1/report.py)

    A small transformer learns a character-level language of color-mixing equations, solving the forms it saw in training and unseen hex pairs. Color turns out to be linearly decodable from its residual stream, with each seed putting _redness_ in a different place. Held-out _named_ pairs sit at zero accuracy.

    <span class="tags">`char-tokens` `hex-colors` `color-names` `linear-probes` `generalization`</span>

    <!-- mini:figures ./m2/ex-2.1.1/report.py -->

- [2.1.2. Making composition necessary](./m2/ex-2.1.2/report.py)

    The previous model never answered a held-out _named_ pair, so we changed the grammar to make composition the only route: reverse alias lines, and named equations whose mix falls off the palette. The model picks up both new skills and still won't chain them within one forward pass, leaving held-out named accuracy at zero.

    <span class="tags">`char-tokens` `hex-colors` `color-names` `composition` `grammar-design`</span>

    <!-- mini:figures ./m2/ex-2.1.2/report.py -->

- [2.1.3. Named colors only](./m2/ex-2.1.3/report.py)

    No hex codes, just one opaque token per color and nothing in the text to say that colors are values at all. The model infers the geometry from co-occurrence alone: its embeddings hold the RGB cube as a linear subspace, it computes mixes in value space just before answering, and its held-out guesses land on or beside the right color.

    <span class="tags">`word-tokens` `named-colors` `embedding-geometry` `vocab-size`</span>

    <!-- mini:figures ./m2/ex-2.1.3/report.py -->

- [2.1.4. Spelling the names](./m2/ex-2.1.4/report.py)

    Does geometry inference need one token per concept? Not at 216 colors. We test the same equations but with every color spelled as an opaque four-letter name. There's a small drop in held-out accuracy, but the value subspace survives. Reading names takes most of the network's depth, leaving one layer for the mix to live in. The 27-color grid is too coarse.

    <span class="tags">`char-tokens` `spelled-names` `embedding-geometry` `depth`</span>

    <!-- mini:figures ./m2/ex-2.1.4/report.py -->

- [2.1.5. Disjoint vocabularies and more named colors](./m2/ex-2.1.5/report.py)

    Names and hex codes. Both sublanguages train, and each builds its own linear color geometry. But the two geometries resist being combined, even under pressure from a narrow residual stream. Keeping them apart apparently costs less than merging them would.

    <span class="tags">`char-tokens` `hex-colors` `long-color-names` `disjoint-vocabularies` `embedding-geometry`</span>

    <!-- mini:figures ./m2/ex-2.1.5/report.py -->

- [2.1.6. Anchoring _red_ in a transformer](./m2/ex-2.1.6/report.py)

    Our first anchored transformer, with a single attractive term. The anchor moved the activations to the chosen direction, and cost the task nothing measurable. But it moved the whole color cube, rather than _red_ in particular.

    <span class="tags">`word-tokens` `anchoring` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.1.6/report.py -->

- [2.1.7. A repulsive term and a narrower pull](./m2/ex-2.1.7/report.py)

    We tested two mechanisms to improve anchor selectivity: **1.** Apply the anchor term only to operand 1 (no other tokens), and **2.** Add a repulsive term to clear the target subspace. Both work, but 1. worked better, and their effects stack somewhat.

    <span class="tags">`word-tokens` `anchoring` `repulsion` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.1.7/report.py -->

- [2.1.8. Repulsion tuning](./m2/ex-2.1.8/report.py)

    We tested anti-subspace schedules (trailing timing and strength) to find an operating point that contains the cube-wide drift without reducing selectivity. Holding the repulsion high for longer contains the drift and keeps the margin, which nothing before this did. It's unclear if the response is well-graded.

    <span class="tags">`word-tokens` `anchoring` `repulsion` `schedules` `grading`</span>

    <!-- mini:figures ./m2/ex-2.1.8/report.py -->

- [2.1.9. Softmin sequence pooling](./m2/ex-2.1.9/report.py)

    Anchoring on the first operand alone worked best so far, but in natural language we won't know which tokens hold the concept. So we let the pull choose its own position, by pooling over the line with a soft minimum. It finds the first operand unaided and the anchor stays healthy, though only the softest pooling stays graded in every run.

    <span class="tags">`word-tokens` `anchoring` `pooling` `mellowmax` `grading`</span>

    <!-- mini:figures ./m2/ex-2.1.9/report.py -->

- [2.1.10. A label that doesn't point](./m2/ex-2.1.10/report.py)

    In ex-2.1.9 the label still only marked the first operand, so the label itself said where the concept was. Here a line is labelled when either operand is red, and the pull has to find the red one line by line. It does, and the anchor is about as selective as when we told it the position.

    <span class="tags">`word-tokens` `anchoring` `pooling` `weak-labels` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.1.10/report.py -->

- [2.1.11. Ablations and a survey for the D2.1 operating point](./m2/ex-2.1.11/report.py)

    A survey. The schedules and weights in the recipe were inherited piece by piece and never tuned together, so we tried removing each one, then searched over what was left. The anchor schedule and half the training could go, but the schedule for the repulsive term could not. The search found a wide region of good settings and proposed a stronger anchor at its edge.

    <span class="tags">`word-tokens` `anchoring` `ablation-study` `sobol-search` `schedules`</span>

    <!-- mini:figures ./m2/ex-2.1.11/report.py -->

- [2.1.12. Fitted channel probes over the anchored checkpoints](./m2/ex-2.1.12/report.py)

    We fitted linear probes to read each operand's color at every depth and position of the D2.1 models. Anchoring costs no color readability that we can measure. The probes also explain an odd pattern in the D2.1 figures, where positions before the second operand seemed to respond to its color: it comes from which test equations we used, not from the model.

    <span class="tags">`word-tokens` `linear-probes` `checkpoints` `decodability`</span>

    <!-- mini:figures ./m2/ex-2.1.12/report.py -->

- [D2.1 figures for the anchoring post](./m2/d2.1/report.py)

    Figures for the LessWrong post on D2.1, drawn from the published results above: the anchoring recipe built up one piece at a time, and where in the model the response sits.

    <span class="tags">`figures` `grading-clouds` `blog-post`</span>

    <!-- mini:figures ./m2/d2.1/report.py -->

</details>

## D2.2: anchoring an operation

D2.2 anchors an operation rather than a color. Since the [pivot](./m2/d2.2/pivot.md), the model infers which op a line uses from a few solved examples, so the op has no word of its own, like the abstract concepts we care about in language models. The anchor lands at little or no cost to the task. With the pull capped, so that it stops once a state is close to the anchor direction, removing that direction everywhere takes the op out gradually. Under the edit, the model moves its answers onto the other ops about as much as an ideal predictor would, though context by context it doesn't always pick the same ones. Currently refining the recipe (hyperparameters).

### Removing *red*

D2.1 anchored *red* but never removed it, so D2.2 began by removing it. Projecting out the anchor direction took *red* out, and took out more where the line was redder, at a small cost to lines with no red. Of the edits we tried, the plain projection stayed the best choice.

<details markdown="1" open="true"><summary>Reports</summary>

- [2.2.1. Suppressing _red_ in the anchored transformer](./m2/ex-2.2.1/report.py)

    Our first intervention on an anchored transformer: remove the anchor direction from the D2.1 models. _Red_ goes, more so the redder the line, and the damage stays within the bound the geometry sets in advance. A little of the cost lands on lines with no red, through the `+` and `=` tokens.

    <span class="tags">`word-tokens` `intervention` `suppression` `checkpoints` `eval-contract`</span>

    <!-- mini:figures ./m2/ex-2.2.1/report.py -->

- [2.2.2. A designed response to suppressing _red_](./m2/ex-2.2.2/report.py)

    We tried the fallback training from M1 on the transformer: a term that teaches the model to answer as though _red_ were _mid-gray_ once _red_ is removed. Every seed gives that answer under the edit it was trained with, though it carries over only partly to the plain projection. That edit also hurt lines with no red, in every anchored model, with or without the new term.

    <span class="tags">`word-tokens` `intervention` `fallback` `training` `eval-contract`</span>

    <!-- mini:figures ./m2/ex-2.2.2/report.py -->

- [2.2.8. A survey of the intervention operator on the stored ex-2.2.3 checkpoints](./m2/ex-2.2.8/report.py)

    A survey of edits on stored models, with no training. Projecting out the anchor direction everywhere was already selective enough. A threshold that spares weakly aligned states removed less _red_, and was no more selective that we could measure.

    <span class="tags">`survey` `intervention` `eval-contract`</span>

    <!-- mini:figures ./m2/ex-2.2.8/report.py -->

- [How far the answer moves: an RGB-distance readout beside expected exact match](./m2/answer-distance/report.py)

    Exact match gives no partial credit, so we re-scored stored models by how far the answer moves on the color grid. Answers that lose _red_ move between half and three quarters of the way to a random guess, and lines with no red barely move. The distance also varies much less between seeds.

    <span class="tags">`reanalysis` `probes` `methodology` `intervention`</span>

    <!-- mini:figures ./m2/answer-distance/report.py -->

</details>

### A richer grammar for *red*

Before we could anchor an op, the grammar needed more than one, so it grew from one op to eleven. We anchored *red* again at each step, and the recipe held. Removal was clean on every op but one, where the model kept about a quarter of the answers that needed red.

<details markdown="1" open="true"><summary>Reports</summary>

- [2.2.3. The multi-op grammar, with _red_ anchored again](./m2/ex-2.2.3/report.py)

    The grammar grows to six ops (`mix`, `add`, `screen`, `multiply`, `lighten`, and `darken`), and we check the D2.1 recipe on it. The stronger anchor from the ex-2.1.11 survey put the anchor direction on the `=` and op-word tokens, so removing it broke lines with no red, and we stayed with the D2.1 recipe.

    <span class="tags">`word-tokens` `multi-op` `regression` `survey-handoff`</span>

    <!-- mini:figures ./m2/ex-2.2.3/report.py -->

- [2.2.4. A scouting round before the anchored-op experiments](./m2/ex-2.2.4/report.py)

    A scouting pass over candidate ops, with no training. Ops that act on hue, saturation, or value spread their answers through the color cube, and they are the first where the order of the operands matters. We proposed adding six new ops and dropping `add`.

    <span class="tags">`scouting` `multi-op` `grammar`</span>

    <!-- mini:figures ./m2/ex-2.2.4/report.py -->

- [2.2.5. A pilot of stochastic rounding](./m2/ex-2.2.5/report.py)

    A pilot of rounding answers at random, in proportion to where they fall between grid levels. The model learns the coin flip itself, so no model can match every rounded answer, and we score the probability it puts on the right answer instead. Anchoring is unaffected.

    <span class="tags">`pilot` `multi-op` `grammar` `eval-contract`</span>

    <!-- mini:figures ./m2/ex-2.2.5/report.py -->

- [2.2.6. A pilot of the whole-span labeller](./m2/ex-2.2.6/report.py)

    A pilot of a label that pulls the whole line rather than just the operands. Some of the pull moves onto the answer at no cost, but removing the anchor still leaves those answers as they were, because the model has already decided the answer one position earlier.

    <span class="tags">`pilot` `multi-op` `anchoring`</span>

    <!-- mini:figures ./m2/ex-2.2.6/report.py -->

- [2.2.7. A pilot of the syntax embeddings](./m2/ex-2.2.7/report.py)

    Removal costs something on lines with no red because the `=` and op-word tokens pick up part of the anchor direction. This pilot asks why. The output side of the shared token table puts it there. Giving the model a separate output table moves it off those tokens at no cost to the task.

    <span class="tags">`pilot` `multi-op` `anchoring` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.2.7/report.py -->

- [2.2.9. The grammar handover](./m2/ex-2.2.9/report.py)

    The changes from the scouting round and the pilots, together: eleven ops, random rounding, the whole-line label, and a separate output table. _Red_ still lands and the task is unhurt. Removal was clean on eight of the eleven ops but fell short on three that take only the hue, saturation, or value of one operand, so we didn't adopt the combined setup as it stood.

    <span class="tags">`prereg` `multi-op` `anchoring` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.2.9/report.py -->

- [2.2.10. Three reads before the handover re-run](./m2/ex-2.2.10/report.py)

    The removal misses came from how we chose the lines to score: removal acts on _red_ like a change of hue, and the lines that missed needed only its saturation or brightness. We proposed choosing the scored lines by hue instead.

    <span class="tags">`scouting` `multi-op` `intervention` `anchoring`</span>

    <!-- mini:figures ./m2/ex-2.2.10/report.py -->

- [2.2.11. The handover re-run](./m2/ex-2.2.11/report.py)

    Ex-2.2.9 again at fresh seeds, with the scored lines chosen by hue. _Red_ lands, stays through training, and comes out cleanly on ten of the eleven ops. On `hue-hsv` the model keeps about a quarter of the answers that needed red, so we still didn't adopt the setup as it stood.

    <span class="tags">`prereg` `multi-op` `anchoring` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.2.11/report.py -->

- [2.2.12. What the stream holds on `hue-hsv`, and a small recipe sweep](./m2/ex-2.2.12/report.py)

    On `hue-hsv`, the answers that survive removal come from reds that lean neither toward orange nor toward pink, and the model rebuilds them inside its blocks after the edit. None of the changes to the recipe in a small sweep removed more.

    <span class="tags">`scouting` `multi-op` `anchoring` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.2.12/report.py -->

- [2.2.13. Does a heavier anchor make the leftover predictable?](./m2/ex-2.2.13/report.py)

    Does a heavier anchor make the leftover on `hue-hsv` the same size every time, so we could measure it once and subtract it? It doesn't: across a wide range, the anchor weight hardly changes anything. Anchoring _red_ to a plane instead of a single direction made the leftover a little smaller, but not by enough to justify the extra room, so the leftover stays a known side effect.

    <span class="tags">`anchoring` `selectivity` `multi-op` `methodology`</span>

    <!-- mini:figures ./m2/ex-2.2.13/report.py -->

- [Geometry under anchoring: a whole-geometry read over the stored runs](./m2/geometry-rsa/report.py)

    A re-analysis of stored models from three experiments, comparing how anchored and un-anchored models arrange their colors. Past the first block, anchored models arrange them differently from the controls, and more alike from seed to seed. A light anchor mostly adds a direction, while a heavier or longer one moves the deeper geometry away from the RGB cube.

    <span class="tags">`scouting` `anchoring` `geometry` `representations`</span>

    <!-- mini:figures ./m2/geometry-rsa/report.py -->

- [Where the op1 lean sits: a reanalysis of ex-2.2.11's stored runs](./m2/op1-lean/report.py)

    A re-analysis of why the first operand leans a little toward the anchor direction. With its own output table, the model uses that direction to predict that an op word or `=` comes next, and that accounts for more than half of the lean. The whole-line label adds a little at most.

    <span class="tags">`scouting` `anchoring` `containment` `readout`</span>

    <!-- mini:figures ./m2/op1-lean/report.py -->

</details>

### Anchoring an op word, and the pivot

Ex-2.2.14 anchored the op `difference`, and it landed, but almost entirely on the *word* `difference`, so removing it would be hard to tell apart from deleting the word. The concepts we care about in language models mostly have no word of their own, so we took the op words out: each line now shows a few solved examples and a query, and the model has to infer the op.

<details markdown="1" open="true"><summary>Reports</summary>

- [2.2.14. Anchoring an operation](./m2/ex-2.2.14/report.py)

    Our first anchored operation: `difference` in place of _red_. It lands more firmly than _red_ did and costs the task nothing, and every other op anchors the same way. But nearly all of it sits on the word `difference` itself, and little reaches the positions where the op is used.

    <span class="tags">`anchoring` `operation` `smoke-test` `preregistration`</span>

    <!-- mini:figures ./m2/ex-2.2.14/report.py -->

- [2.2.15. Lines cut short by the training window](./m2/ex-2.2.15/report.py)

    Training windows cut lines at their edges, and the anchor asked the visible part of a cut line to carry the whole label, even when it couldn't see the op word. Those cut lines are why the first operand leans toward the anchor, and skipping them removes the lean, so later experiments pull whole lines only.

    <span class="tags">`scouting` `anchoring` `labelling` `preregistration`</span>

    <!-- mini:figures ./m2/ex-2.2.15/report.py -->

- [D2.2 pivot: an operation the model has to infer](./m2/d2.2/pivot.md)

    Adopted after ex-2.2.14. Since the anchor went to the op word, D2.2 now anchors an op the model infers from solved examples, and no token in the line names the op. That is closer to what M3 needs. Covers the new grammar, what could go wrong with it, and what it changes in the plan for D2.3.

</details>

### Teaching the in-context task

Once the op had no word, the first question was whether an un-anchored model could learn the task at all. At first it got a little under halfway from guessing to the best possible score. Changes to training and a smaller set of ops brought it to about nine tenths of the way.

<details markdown="1" open="true"><summary>Reports</summary>

- [2.2.16. The in-context grammar pilot](./m2/ex-2.2.16/report.py)

    The first experiment on the new grammar, where the model infers the op from a few solved examples. Every control got a little under halfway from guessing to the best possible score, so the pilot stopped before anything was anchored.

    <span class="tags">`pilot` `in-context` `anchoring` `labelling` `preregistration`</span>

    <!-- mini:figures ./m2/ex-2.2.16/report.py -->

- [2.2.17. The center control plateau](./m2/ex-2.2.17/report.py)

    More training steps, a lower learning rate, and a mask that stops each line attending to earlier ones took the control most of the way to its target, and then it leveled off. It does as well as an ideal predictor where the examples leave the op uncertain, and falls short where they settle it.

    <span class="tags">`scouting` `in-context` `training`</span>

    <!-- mini:figures ./m2/ex-2.2.17/report.py -->

- [2.2.18. Dropping ops with similar answers](./m2/ex-2.2.18/report.py)

    Dropping four ops whose answers often coincide with another op raised the best possible score, and the model came closer to it than any run before. Most of the gain came from dropping ops that round at random.

    <span class="tags">`scouting` `in-context` `training`</span>

    <!-- mini:figures ./m2/ex-2.2.18/report.py -->

- [2.2.19. Training length and seeds for the seven-op set](./m2/ex-2.2.19/report.py)

    Half the training length keeps most of the skill of the full recipe, a little short of what we asked for. A [follow-up](./m2/ex-2.2.19/calibration.py) finds the model overconfident where the examples fit several ops.

    <span class="tags">`in-context` `training`</span>

    <!-- mini:figures ./m2/ex-2.2.19/report.py -->

- [2.2.20. A high-rate head start before the recipe schedule](./m2/ex-2.2.20/report.py)

    A short burst at a higher learning rate before the usual schedule kept no more of the skill than the plain run, so the plain schedule stays.

    <span class="tags">`in-context` `training`</span>

    <!-- mini:figures ./m2/ex-2.2.20/report.py -->

</details>

### Anchoring and removing the inferred op

With the control near its best, we anchored `difference` on the new grammar. The anchor settles on the example answers, where the context shows the most about the op. Removing the anchor direction everywhere takes `difference` out gradually. The other ops mostly stay as they were, though not on every run, and capping the pull does not seem to be what keeps them there.

<details markdown="1" open="true"><summary>Reports</summary>

- [2.2.21. The in-context grammar pilot, on the reworked recipe](./m2/ex-2.2.21/report.py)

    We anchored `difference` on the reworked recipe. The anchor sits on the example answers, and hardly at all where the query answer is predicted. With the pull capped, so that it stops once a state is close to the anchor direction, an edit at every position takes `difference` out gradually and leaves the other ops as they were.

    <span class="tags">`pilot` `in-context` `anchoring` `labelling` `intervention`</span>

    <!-- mini:figures ./m2/ex-2.2.21/report.py -->

- [2.2.22. Localized by depth, various pull caps, and contexts of varying length](./m2/ex-2.2.22/report.py)

    A scout of three changes to the capped recipe: keeping the pull off some depths, other caps, and contexts with varying numbers of examples. None improved it. Keeping the pull off some depths made the edit spill onto other ops, the cap made no steady difference, and varying the number of examples cost a little skill. Pooled over contexts, the edited model answers about as an ideal predictor without `difference` would, though context by context it gets less than half of the way there.

    <span class="tags">`scout` `in-context` `anchoring` `intervention`</span>

    <!-- mini:figures ./m2/ex-2.2.22/report.py -->

- [2.2.23. The slow seeds, trained for longer](./m2/ex-2.2.23/report.py)

    A scout of the runs that miss the second rise in task skill. At 200 epochs a few anchored runs missed it, and at 400 every run made it and ended alike. The edit spills onto other ops on nearly every run that learned the HSV ops, and on none of the half-trained runs, so the clean edits of earlier experiments came mostly from half-trained runs. Training keeps every run, at 400 epochs.

    <span class="tags">`scout` `in-context` `anchoring` `intervention`</span>

    <!-- mini:figures ./m2/ex-2.2.23/report.py -->

- [What the anchor follows at the example answers](./m2/example-evidence/report.py)

    A re-analysis of the stored ex-2.2.22 runs, asking why the anchor sat higher at earlier example answers for the same posterior. At an example answer, the anchor follows the posterior given that example alone, and the earlier examples barely count. So the rise of the anchor with the posterior comes from each example that fits, and it doesn't yet show the anchor holding an inferred op.

    <span class="tags">`reanalysis` `in-context` `anchoring`</span>

    <!-- mini:figures ./m2/example-evidence/report.py -->

- [Where the spill of the edit comes from](./m2/spill-by-position/report.py)

    A re-analysis of the ex-2.2.23 checkpoints, editing one set of positions at a time. On most runs the edit removes `difference` and spills onto other ops at the same positions, the example answers, so no set of positions gives the removal without the spill; the other positions add spill of their own. At 400 epochs a few runs also hold the op at the separators, which is where most of their spill comes from.

    <span class="tags">`reanalysis` `in-context` `anchoring` `intervention`</span>

    <!-- mini:figures ./m2/spill-by-position/report.py -->

</details>
