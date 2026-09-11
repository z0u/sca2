# Sparse Concept Anchoring in transformers

Sparse Concept Anchoring (SCA) is a training-time method for concept control. A light geometric regularizer, driven by a small number of noisy labels, guides a chosen concept toward a known location in representation space. The geometry is shaped during training, so there is no need to reverse-engineer it afterwards: the concept lives where you put it, and the side effects of suppressing or ablating it can be bounded before running the intervention.

The first milestone (M1) established the method in autoencoders: [paper](https://arxiv.org/abs/2512.12469), [blog post](https://www.lesswrong.com/posts/sGskzx7LgsDkMLvcv/intervening-on-sparse-anchored-concepts), [code](https://github.com/z0u/ex-preppy). This site holds the experiment reports for the second milestone (M2), which asks: does SCA transfer to transformers? We anchor concepts in the residual stream of a small transformer trained on a synthetic color-mixing task (`red + blue = purple`), where ground truth is unambiguous, so a negative result stays interpretable. The plan and deliverables are in the [project README](https://github.com/z0u/sca2).

![Milestone map: M1 (autoencoders) complete, M2 (transformers) in progress, M3 (language models) and M4 (LLM fine-tunes) planned. Within M2, D2.1 (anchor a concept) is complete, D2.2 (operations and steering) in progress, D2.3 (asymmetric verification) planned, and D2.4 (publication) already under way.](./public/milestones.svg)

&nbsp;

## Experiment reports

Each report is a [Marimo](https://marimo.io) notebook that reads durable results produced by a separately-run experiment. Reports are published automatically, with their figures served from a Hugging Face dataset; the infrastructure is [mi-ni](https://github.com/z0u/mi-ni). Each entry below carries searchable tags and a strip of the report's figures, in reading order — click a thumbnail for the full-size image.

<!-- These URLs are rewritten to point to the published notebooks, and the mini:figures markers become thumbnail strips (scripts/build_site.py) -->

<details markdown="1"><summary><h3 id="iteration-0-prep">Iteration 0 (prep)</h3></summary>

These experiments were preparation for the main work: exercising the infrastructure, testing the normalized transformer architecture, and tying off some loose ends from M1.

- [nGPT scaling](./ngpt-scaling/report.py)

    Width × depth sweep of our simplified nGPT to check that it stays well-behaved as it grows. Converged loss improves with width and is flat across depth, and no condition spikes or stalls. The architecture seems safe to build the color-mixing experiments on.

    <span class="tags">`ngpt` `scaling` `architecture` `infra`</span>

    <!-- mini:figures ./ngpt-scaling/report.py -->

- [M1 2.9.1. Deleting *red*, now in JAX](./m1/ex-2.9.1/report.py)

    We ported the main M1 result from PyTorch to JAX: Anchor *red* to one latent axis of a small autoencoder, then zero the axis and watch red disappear. The result reproduces, seed sensitivity and all, and our new infrastructure holds up.

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

<details markdown="1" open="true"><summary><h3 id="d21-anchoring-in-a-transformer">D2.1: anchoring in a transformer</h3></summary>

- [2.1.1. Un-anchored color-mixing transformer](./m2/ex-2.1.1/report.py)

    A small transformer learns a character-level language of color-mixing equations, solving the forms it saw in training and unseen hex pairs. Color turns out to be linearly decodable from its residual stream, with each seed putting *redness* in a different place. Held-out *named* pairs sit at zero accuracy.

    <span class="tags">`char-tokens` `hex-colors` `color-names` `linear-probes` `generalization`</span>

    <!-- mini:figures ./m2/ex-2.1.1/report.py -->

- [2.1.2. Making composition necessary](./m2/ex-2.1.2/report.py)

    The previous model never answered a held-out *named* pair, so we changed the grammar to make composition the only route: reverse alias lines, and named equations whose mix falls off the palette. The model picks up both new skills and still won't chain them within one forward pass, leaving held-out named accuracy at zero.

    <span class="tags">`char-tokens` `hex-colors` `color-names` `composition` `grammar-design`</span>

    <!-- mini:figures ./m2/ex-2.1.2/report.py -->

- [2.1.3. Named colors only](./m2/ex-2.1.3/report.py)

    No hex codes, just one opaque token per color and nothing in the text to say that colors are values at all. The model infers the geometry from co-occurrence alone: its embeddings hold the RGB cube as a linear subspace, it computes mixes in value space just before answering, and its held-out guesses land on or beside the right color. Exact match rises and falls with vocabulary size, while geometric closeness improves steadily.

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

- [2.1.6. Anchoring *red* in a transformer](./m2/ex-2.1.6/report.py)

    Our first anchored transformer, with a single attractive term. The anchor moved the activations to the chosen direction, and cost the task nothing measurable. But it moved the whole color cube, rather than *red* in particular.

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

    Anchoring on op1 alone worked best so far, but in natural language we won't know which tokens hold the concept. So we pool over sequences with a soft minimum (mellowmax) and let the pull choose its own position. At the embedding it picks op1 unaided, the operating point stays healthy, and grading improves. It wins no margin though, and only the softest pooling stays graded in every run.

    <span class="tags">`word-tokens` `anchoring` `pooling` `mellowmax` `grading`</span>

    <!-- mini:figures ./m2/ex-2.1.9/report.py -->

- [2.1.10. A label that doesn't point](./m2/ex-2.1.10/report.py)

    Ex-2.1.9's labels still keyed on op1, so the label itself said where the concept was. Here either operand can trigger the label, and the pooled pull finds the red operand line by line: the weight profiles track the label groups, the operating point survives, and selectivity matches the slot oracle. The soft-τ arms win margin but smear the grading.

    <span class="tags">`word-tokens` `anchoring` `pooling` `weak-labels` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.1.10/report.py -->

- [2.1.11. Ablations and a survey for the D2.1 operating point](./m2/ex-2.1.11/report.py)

    A survey to close out D2.1. The schedules and weights in the recipe were inherited piece by piece and never tuned together. We ablate first by replacing schedules with constants, and dropping training epochs. Then we run a Sobol search over what is left (anchor weight, pooling temperature, repulsion dose).

    The anchor schedule and half the epochs could go. The anti-subspace schedule could not: a constant delivering the same total dose loses grading. The search maps a wide feasible plateau and proposes a stronger, softer point on its edge.

    <span class="tags">`word-tokens` `anchoring` `ablation-study` `sobol-search` `schedules`</span>

    <!-- mini:figures ./m2/ex-2.1.11/report.py -->

- [2.1.12. Fitted channel probes over the anchored checkpoints](./m2/ex-2.1.12/report.py)

    Ridge probes for each operand's RGB at every (slice, position) site of the D2.1 conditions, from their published checkpoints. Before op2 the probes find nothing in any condition, anchored or not, so the off-key tilt in the grading figures belongs to the probe set. No decodability cost of anchoring resolves, and the maps show the bare anchor giving up late-slice decodability that the rest of the recipe restores.

    <span class="tags">`word-tokens` `linear-probes` `checkpoints` `decodability`</span>

    <!-- mini:figures ./m2/ex-2.1.12/report.py -->

- [D2.1 figures for the anchoring post](./m2/d2.1/report.py)

    Grading-cloud figures for the LessWrong post on D2.1, drawn from the published results above: the anchoring recipe one piece at a time, and the primary condition per (slice, position).

    <span class="tags">`figures` `grading-clouds` `blog-post`</span>

    <!-- mini:figures ./m2/d2.1/report.py -->

</details>

<details markdown="1" open="true"><summary><h3 id="d22-anchoring-an-operation">D2.2: anchoring an operation</h3></summary>

- [D2.2 design](./m2/d2.2/design.md)

    The plan: the claims D2.2 exists to make, the experiments in order with the engineering that precedes them, the risk each retires, and what is out of scope.

- [2.2.1. Suppressing *red* in the anchored transformer](./m2/ex-2.2.1/report.py)

    The first intervention on an anchored transformer: project the anchor axis out of the D2.1 checkpoints and score red lines against non-red lines. The removal works, grades with the line's redness, and stays inside the bound the placed geometry sets; zeroing the axis weights does the same job. Selectivity is partial: the non-red cost comes from the syntax positions, whose embeddings carry a constant component on the axis. Editing the operands alone avoids it, and so does M1's shaped suppression, which removes only half of *red*.

    <span class="tags">`word-tokens` `intervention` `suppression` `checkpoints` `eval-contract`</span>

    <!-- mini:figures ./m2/ex-2.2.1/report.py -->

- [2.2.2. A designed response to suppressing *red*](./m2/ex-2.2.2/report.py)

    Fallback control from M1, adapted to the transformer. It's a training term that teaches the blocks what to answer once *red* is removed, at a designed fallback answer (continuation), with a stop-gradient protecting the placement. The fallback answer is the center of the operand-averaged null, the visible operand mixed with *mid-gray*. It worked: *red* behaved like *mid-gray*, and seed variance was reduced — but it also reduced selectivity.

    <span class="tags">`word-tokens` `intervention` `fallback` `training` `eval-contract`</span>

    <!-- mini:figures ./m2/ex-2.2.2/report.py -->

- [2.2.3. The multi-op grammar, with *red* anchored again](./m2/ex-2.2.3/report.py)

    The grammar grows to six operations spelled as words (`mix`, `add`, `screen`, `multiply`, `lighten`, `darken`, each rounded to the grid), and the D2.1 recipes are checked again on it: the control, the ex-2.1.10 recipe at two lengths, and three survey proposals at fresh seeds, with a frozen rule that names the operating point the rest of D2.2 adopts. The recipe transferred (H1, H2), and the frozen rule adopted the survey's `t00` (H3), a heavier anchor that trades contrast and syntax-row cleanliness for margin. Suppression did not transfer as stated (H4): the recipe's removal reads as partial on four of the new ops, and `t00` needs the operand-only edit to keep the non-red lines. A post hoc read adds the lead and selectivity gates the rule left out and narrows the choice to the recipe; a twenty-seed comparison of its two lengths (E6) backs the decision to build D2.2 on `recipe-short`, which grades better for half the compute; the partial removals are lines whose answer never depended on the red operand (E8). Six ops left the operand cube less linearly decodable than three (E4).

    <span class="tags">`word-tokens` `multi-op` `regression` `survey-handoff`</span>

    <!-- mini:figures ./m2/ex-2.2.3/report.py -->

- [2.2.4. A scouting round before the anchored-op experiments](./m2/ex-2.2.4/report.py)

    A scouting pass over the open questions ex-2.2.3 left, no hypotheses and no training: each section runs the cheapest read of one question and says whether it changes the D2.2 design. The first covers the op set. Nine candidate ops are read on the grid beside the current six: where their answers land, how evenly they spread, whether the answer depends on the red operand, and how often the op word matters. The hue, saturation, and value blend modes spread their answers well, are the first ops where operand order carries information, and are where the answer moves furthest once the red operand loses its red; a small change in the red operand reaches them no more often than it does the saturating ops. Three commutative ops (`difference`, `exclusion`, and `mix` done in HSV) spread and are sensitive to both operands. Proposed table: drop `add`, add all six, with the HSV trio as a marked subset; score removal as a distance rather than exact match; and adopt stochastic rounding and the whole-line labeller from the two pilots.

    <span class="tags">`scouting` `multi-op` `grammar`</span>

    <!-- mini:figures ./m2/ex-2.2.4/report.py -->

- [2.2.5. A pilot of stochastic rounding](./m2/ex-2.2.5/report.py)

    A pilot, no gates: the un-anchored control and the adopted recipe retrained on a corpus where an answer between grid levels rounds by coin flip, in proportion to where it sits. The model learns the rule's answer distribution rather than a rounding, so exact match against a drawn answer sits at the ceiling the rule sets and carries one draw's noise; the reads to use are expected exact match and calibration. Anchoring does not notice the corpus, suppression reads shift through their clean baseline, and the redder-than-both counts rise on the ops that round up. The pilot recommended keeping nearest rounding; ex-2.2.4 argues for adopting it anyway, for comparability with M3.

    <span class="tags">`pilot` `multi-op` `grammar` `eval-contract`</span>

    <!-- mini:figures ./m2/ex-2.2.5/report.py -->

- [2.2.6. A pilot of the whole-span labeller](./m2/ex-2.2.6/report.py)

    A pilot, no gates: the adopted point retrained under labellers that also read the answer, or also pull the whole line, or both, against production's twenty seeds. Pulling the whole line puts a tenth to a quarter of the pull on the answer and doubles the alignment the redder-than-both lines have there, at no cost to the task or to placement; reading the answer changes nothing visible. Neither closes the blind span for interventions, because the answer is read out at `=`, one position before the extra alignment lands. The pilot recommended keeping the operand-only labeller; ex-2.2.4 argues for adopting the whole-line pull, as the M3-shaped labelling.

    <span class="tags">`pilot` `multi-op` `anchoring`</span>

    <!-- mini:figures ./m2/ex-2.2.6/report.py -->

- [2.2.7. A pilot of the syntax rows](./m2/ex-2.2.7/report.py)

    A pilot, no gates, on the anchor axis that every anchored model carries on the embedding rows of the op words and `=`. Part A strips the component from ex-2.2.3's stored checkpoints on the input side or the output side of the tied table: the output side carries the `=` prediction after a red operand, which is the tied-readout mechanism as hypothesised, and the input side reads as the blocks having learned where the rows sit. Part B retrains the recipe three ways, each removing one candidate mechanism. Anchoring the blocks only does not clean the rows and makes the projection less complete; untying the readout moves the component onto the readout table; holding the rows at zero after every step costs nothing visible on three seeds and brings the full-position projection's non-red cost down to the operand-only edit's. The pilot proposes carrying that constraint into the handover.

    <span class="tags">`pilot` `multi-op` `anchoring` `selectivity`</span>

    <!-- mini:figures ./m2/ex-2.2.7/report.py -->


</details>
