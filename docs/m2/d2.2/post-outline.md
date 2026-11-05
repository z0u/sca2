# D2.2 post: outline

A working outline for the LessWrong post on D2.2, the next in the sequence after *Anchoring one concept in a transformer* ([D2.1 post](/references/d2.1-anchored-transformer.md)). The post will be written by hand from this outline, so the outline gives the order, the question each section answers for the reader, what goes in, and where the evidence lives. It does not fix the wording. The [design doc](design.md) holds the claims and the recipe in full.

The structure follows a review from four angles (2026-11-05): one fresh read of the previous outline, and three structures proposed blind from a list of the material. All four put the arc in a section of its own and ended near the reflection. Three of the four placed the comparison as its own section in the middle.

## The post at a glance

Reader {.manual}
:   Someone who may not have read the earlier posts. The D2.1 post was about 2,200 words with ten figures; this one aims for about 2,500 with six or seven, with method detail in LessWrong footnotes and an appendix linked from the site. The post leaves out our internal names (claim letters, recipe names such as `τ_ℓ fixed` and "per-visit", sans-diff, the target null, spill criterion) and says what each means in plain words. The plain wordings used below: "the reference answer" for the target null, "a model trained without the op" for sans-diff, "side effects on the other ops" for spill.

The reader's question {.manual}
:   Once the summary has said that a direction fitted after training does about as well, the reader stops asking whether the anchor wins. The question that carries them changes as they go: can you anchor a concept the model has to infer? Can you take it out? Then, at the turn: a fitted direction does as well, so what did the anchor do? The arc section answers that, the next section says what the anchor is for, and the reflection says what the work changed for the author.

Settled {.manual}
:   The body reports on the runs whose labels are redrawn on every visit (about one `difference` context in fifty at a time, like a very large corpus); the fixed draw (a small labelled set reused) gets a sentence and the appendix. No further experiments before the post. The recipe-transfer test stays out of the post.

## 0. Summary and series note

*Reader's question: what is this, and what came of it?*

- Three or four sentences. We anchored an operation the model has to infer, one that is never written down, onto an axis chosen before training. Editing that axis out removes most of what the model does with the op, in proportion to how much each context needed it, and leaves the other ops alone. A direction fitted after training (LEACE, INLP) removes it about as well. What the anchor adds is a location known in advance and a geometry that made the remainder easy to read.
- A short blockquote as in D2.1: the series, that this post stands alone, and what D2.1 promised (intervention, more ops, a comparison with other methods), of which this post delivers the first and the third in part.

## 1. An operation the model has to infer

*Reader's question: why a different kind of concept, and what is the task?*

- D2.1 anchored *red*, which has its own token. The concepts worth steering, such as a persona, are inferred from context, so D2.2 anchors one that has to be inferred ([pivot](pivot.md)). One sentence citing Africa and Irving's agenda ([Thousand-dimensional structure](https://www.lesswrong.com/posts/sFhW3ZnPMJdnB4Dd6/thousand-dimensional-structure-1)), which treats a persona as a latent with a distribution over personas, as motivation only.
- The task in one picture: three solved examples `a ? b = y` and a query, `?` where the op word would be. Seven ops over colors; `difference` is anchored to e₁ of the same small nGPT (four blocks, 64 wide). Figure: the examples drawn as small images combined under the hidden op ([backlog](/todo/style/equations-as-images-under-blend-modes.md)).
- Mixed evidence: some examples show another op's answer, so a context can leave the op in doubt, and the posterior over ops says how much each context needs `difference`. The reference answer is what an ideal predictor that never knew `difference` would give, weighing the other ops by fit; removal is the share of the way from the clean model to it ([design § scoring](design.md#scoring)). Two sentences.

## 2. Anchoring it, and where it landed

*Reader's question: what changed in the method, and did the anchor find the op?*

- Carried over from D2.1: the anchor and anti terms, labels on whole contexts, pooling that lets the term choose its states. New: the pull is pooled over depth as well as position, so we never had to say which block holds the op, and the anti weight falls off with depth ([design § recipe](design.md#the-recipe-of-record)). Figure: the D2.1 pooling picture with a depth axis added ([pooling.svg](/docs/m2/ex-2.1.9/public/pooling.svg) is small and hand-written).
- Labels: one `difference` context in fifty, and about a quarter of those have a posterior below one half, so many labels mark an op the context barely shows. One sentence, with the draw in a footnote.
- One sentence on the path: the recipe took about thirty experiments, and the two problems it solved are in the appendix.
- Where it landed: on the example answers, where a context shows the most about its op, with alignment rising with the posterior on `difference` (fixed-effects, example-evidence). It held on all 8 fresh seeds and all 40 runs checked early (fixed-confirm, loss-rate); seven of eight passed the task gate and the eighth learned two HSV ops late. Figure: α by role and slice.

## 3. Taking it out

*Reader's question: can you delete it, and what happens to the answers?*

- The full edit removes 0.88 of the way to the reference answer on the seed mean (0.83 to 0.91), and side effects on the other ops stay under the limit on every run at every dose (fixed-effects, reference-baseline). Removal is complete by three quarters of the full edit, and each context drops by about as much as it needed the op. Figure: the dose curve with side effects underneath.
- Why the edit covers every depth: block 1 writes the evidence of each example onto e₁, and one later block (3 or 4, depending on the seed) reads it where the query is answered. An edit at fewer depths has to land between the two, and a larger model has readers at depths we don't choose (block1-edit).
- Where the mass goes: to the other ops in about the right proportions overall, about half the distance context by context. A model trained without the op comes 0.58 of the way where the edited model comes 0.50, and misses the same way (filtered-corpus). Quote this framing only; the 85% share, if used, goes in a footnote with its endpoint named.
- The surprise: on contexts of other ops whose examples point partly at `difference`, the edited model gets better, because the mass the op took now goes to the right op. The gain grows with the posterior on `difference`; in the top bin the edit gets about 75% of the gain the reference answer predicts (filtered-corpus E4). A small case of intervening on one inferred latent and seeing the mass move where inference says it should. Figure: the gain against the posterior. **Open:** whether the fitted edits show the same gain (being checked); if they do, say so in section 4.

## 4. A direction fitted afterwards does about as well

*Reader's question (the turn): was the anchor needed?*

- What was compared: the difference in means, a ridge probe, LEACE and INLP (linear erasers), and a direction found by gradient search, fitted on models trained with no anchor, with every label, tuned per run over depth, site and dose, and edited through the same scorer ([post-hoc compare](/docs/m2/post-hoc-compare/report.py)). On anchored models, the fitters find e₁. Name the tuning asymmetry both ways in one sentence: the fitters were tuned per model, the anchor not at all, but the recipe took about thirty experiments.
- One table, methods as rows, both readings as columns. Knowing which positions hold example answers, LEACE and INLP remove 0.94 to 0.97 against 0.88. Pooled over positions, the fairer match to real text, they remove 0.91 and 0.89, level with the anchor, and at matched removal exceed the side-effect limit on 2 of 8 runs against 0 of 8, within chance. The difference in means falls behind without role labels. Scarce labels are no edge either: the fitters stayed as flat down to one in fifty.
- The sentence the section ends on: the removal results describe what deleting one direction does in this model, for the anchored direction and for a fitted one.
- One footnote holds the caveats: the preregistered hypotheses missed, and which analyses were not preregistered; the pooled reading is a lower bound on the fitters until a per-position gate is tried; the link to the preregistration.

## 5. What both found

*Reader's question: then what did the anchor do?*

- Every run writes "this example fits `difference`" as one vector with two parts: along e₁, where the anchor put it, and along the direction a non-fitting answer has, which the loss made on its own (shared across the other ops, at least at slice 3). Fitting examples lie on an arc from the non-fitting direction to e₁ ([low-runs](/docs/m2/low-runs/report.py)).
- A model trained without the anchor holds the same arc, found from the class means after training, but a short one: about 21° between the means at slice 3 against 83° in anchored runs (33° against 60° at slice 2). So the fitted direction finds the arc too, and the anchor stretches it and names one end in advance. Figure: the controls-arc figure from the low-runs report (a control and an anchored run on the unit disc, with the angle for all sixteen runs).
- What the edit misses: the readers take some evidence from each part, and the edit reaches the first. Under the fixed label draw, two or three runs in eight end low because their readers lean on the second part; under the main draw it costs a few hundredths. Which part the readers lean on is invisible to training: twins with the same labels, seed and batches match in loss and alignment and differ only in removal (one sentence, the twins in a footnote).
- The worry that an edit pushes a behavior into dimensions we can't see, the central risk in the Thousand-dimensional structure agenda: here the remainder is a direction training made before any edit, and the checks on the edit itself (the write bound, in the appendix, and the side effects) look for anything the edit pushed elsewhere.
- The summary points here for "most of".

## 6. What the anchor is for

*Reader's question: given all that, why anchor?*

- A different cost profile, not a win. Costs: an extra training term, a task score 0.003 lower, a recipe that took work to settle, and it has to be there from the start of training; post-hoc methods apply to a model that already exists, which is why most people use them.
- No search: the location is known before training, so there is no sweep per model, and the full edit stayed within the limit on all 8 runs untuned, where a fitted configuration chosen on the other seven runs missed the depth on one run in eight.
- A fixed geometry: the bound on what later blocks may write back could be read off the clean model before any edit, and one end of the arc had a name before we looked. This is the direction of travel: several concepts, designed subspaces, circuits.

## 7. Next

*Reader's question: where does this go?*

- A paragraph: circuit design, perhaps in D2.3; gradient routing as the training-time method to compare with; anchoring both ends of the arc, so the edit acts on a known plane ([backlog](/todo/science/anchor-the-not-difference-end-of-the-arc.md)); and the cases where erasers are known to struggle (a concept correlated with another in the data, a concept held nonlinearly) as where an anchor might show an edge. Citations to check: Kumar et al., 2022; Ravfogel et al., 2022.
- A footnote at most for *red*: an earlier check found unconditional linear erasure cost the other colors, and a gated edit was not tried.

## 8. What I expected

*Reader's question: what did the author learn?*

The post ends here. A short reflection in the first person, tied to Cammarata's essay. The middle can be shorter than the draft below, since the reader has just seen the writer and reader blocks and the arc.

> When I started this program, I assumed that the representations in neural networks were tangled by default, and that a concept we cared about would be hard to find unless we gave the model somewhere to put it. SCA was meant to give that signal early in training. So when we started D2.2, we designed a grammar with what we thought would be a hard-to-find latent concept: an operation that is never written down.
>
> That isn't quite what we found. The anchor settled on the example answers, and the post-hoc methods found the same evidence in models trained with no anchor at all. They all found a judgement about each example: does this one fit `difference`? We went looking for the op and found a quality of each example, something like how much it has of `difference` about it: a noun, just not the one we expected. That is close to how Nick Cammarata reads neurons in *Neural Networks Are Already Clean*: as distinctions that later parts of the network use, more often than the named things we bring to them.¹
>
> We don't yet know whether the op itself is represented anywhere, or whether the answer is composed from those judgements by the blocks that read them. If it is, what we call `difference` may be a behaviour that emerges from them, unsatisfying though that may be. But the evidence is pretty cleanly represented. If networks often hold evidence this cleanly on their own, the anchor is less about making a concept clean and more about deciding where it goes before training.
>
> ¹ Cammarata argues from neurons in much larger vision and language models that this is the usual case. The essay is a position piece and still debated, and our result is one concept in one small synthetic model, so we offer it as related company rather than support.

Notes: the draft avoids "disentangled" and "superposition", which carry several meanings. The essay's caution is about fitting a replacement basis after training (a sparse autoencoder, say), and it credits training choices with cleaner neurons, so it sits easily beside a training-time term.

## Appendix, linked from the site

- The write bound, in plain terms. In the autoencoder the bound was geometric: removing the e₁ part of a state moves it by no more than that part. In a transformer the edited states feed later blocks, which could write e₁ back onto the states of other ops. The write bound says how much e₁ the later blocks may put there: no more than the clean model already has. Because we chose the axis, that limit can be read off the clean model before any edit, and the measured writes stay within it past the embedding, apart from a few sites at slice 1 on one run in eight (write-floor, fixed-effects).
- The path to the recipe: a color embedding stood in for the op, solved by holding the anti weight high on the first slices (embedding-lean, late-pull); then block 1 wrote e₁ on every op, solved by keeping the pool over slices sharp (slice-terms, soft-pool-scout). Figure: side effects falling across the recipes (recipe-compare).
- The label draw: redrawn on every visit (about 34,000 of the 42,891 `difference` contexts carry a label at some point over 400 epochs) against fixed for the run. Under the fixed draw the anchor held at 1/50, 1/25 and 1/10 on all 24 runs, with a low tail of two or three runs in eight at 1/50 and 1/10, no trend with label count, and retraining moving which runs end low ([fixed-labels](/docs/m2/fixed-labels/report.py)).
- Statistics: the reference baseline (median over the quiet axes) and why Adam makes it fair; the absolute tolerance on the write bound; comparing conditions as two samples with a Welch t interval ([eng/determinism.md](/eng/determinism.md)).
- Compute overhead. On an L40S the terms add no step time we can tell from the noise between containers (2.01 ms with both terms against 1.97 without, spread 1.90 to 2.11). By the compiler's count they add about 0.1% of the arithmetic and 3% of the bytes moved, a share that shrinks as the model widens; the anchor term could be skipped on batches with no labelled context. Anchored runs learn the task more slowly at first and catch up by about epoch 250 of 400; quote that only with the training trajectories beside it ([PR #345](https://github.com/z0u/sca2/pull/345)).

## Figures

A D2.2 figures page in the shape of the [D2.1 figures](/docs/m2/d2.1/report.py) page would hold: the task in images (section 1), the pooling picture with a depth axis (section 2), α by role and slice (section 2), the dose curve with side effects (section 3), the gain on other ops (section 3), the comparison table (section 4), and the controls-arc figure, already drawn in the [low-runs](/docs/m2/low-runs/report.py) report (section 5).
