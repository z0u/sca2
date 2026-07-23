# Science todo

A loose backlog of experiment questions and findings. Add to it when you notice
something interesting in a run but want to defer the investigation. Infrastructure
and tooling work lives in [`todo-eng.md`](./todo-eng.md).

<!--
Items may be tagged, and a tag _may_ link to more info. Potential tags:
- The deliverable or experiment the item was noticed in (the "huh, that's funny" moment)
- The deliverable it impacts or should be investigated in
- Concepts and groupings of various kinds
-->

## Open questions

- [ ] Probe all positions in a sample of sequences. So far we have only probed
  specific locations, e.g. last token of first operand; last token before answer.
  How do the other tokens compare? Visualize probe response as sublines; note that
  we could show multiple probes per subline (as separate series). #[D2.1] #ex-2.1.1
  #representations

- [ ] Does more training close the one-level precision gap at the full grid?
  Ex-2.1.3's v4096 cells plateau at seen 0.85 / holdout 0.65 under the fixed
  100-epoch schedule, with misses one grid level off in one channel — the
  geometry is right and the precision isn't. Candidates: a longer or reshaped
  schedule, weight decay (grokking-style late snap-in). #[D2.1] #ex-2.1.3 #vocab

- [ ] Does the sub-cell embedding precision hold up at depth, and what sets the
  `v4096` floor? Ex-2.1.3's cross-validated embedding probe places tokens
  0.73 / 0.63 / 0.49 grid cells from their true color at v27 / v64 / v216 —
  under one cell throughout, so nearest-name decoding survives even at v27 where
  R² is only 0.66. In absolute terms that error *shrinks* with the grid
  (0.363 → 0.210 → 0.097 of the unit cube), so precision is relative, not a
  fixed resolution the finer grids keep exposing. v4096 breaks it both ways:
  0.166 absolute (worse than v216) and 2.5 cells. Two follow-ups. (a) Is the
  v4096 floor capacity or optimization? Widening the stream or training longer
  separates them, and it's the same wall the accuracy plateau hits — pairs with
  the existing one-level-precision todo. (b) The same measure at each depth
  would say whether the mix computation preserves the sub-cell precision the
  embeddings start with, or loses it. Caveat for both: this is probe error, so
  it bundles model imprecision with probe misfit, which matters most at v27
  (26 fit points, leave-one-out). #[D2.1] #ex-2.1.3 #representations #metrics

- [ ] Fold ex-2.1.3's embedding-probe fixes back into `experiment.py`. The
  report now computes both itself from the published `embeddings` array, so the
  science is banked; the stored `emb_r2` is the stale one. (a) `emb_r2` uses a
  half/half split, which at v27 fits a 64→3 map from 13 points and understates.
  `sca.compute.evaluation.ridge_probe_loo` is the drop-in: leave-one-out, exact,
  about the cost of one fit, and with no split to draw it takes no seed. It
  gives 0.66 against the stored 0.48 (v64 0.81→0.87, v216 0.95→0.97, v4096
  unchanged). (b) Worth storing alongside it: color's share of the top-3 PC
  variance budget (0.25 / 0.26 / 0.56 / 0.80 across v27→v4096), which is *why*
  PCA only finds the cube at the dense end. Deferred because editing the eval
  step is memoization evidence and re-runs every cell — bundle it with the next
  change that re-runs ex-2.1.3 anyway. The same treatment
  extends to the per-depth probes, which is the transferable part.
  #[D2.1] #ex-2.1.3 #representations #metrics

- [ ] Distance-shaped answer targets for ex-2.1.3, post-hoc. We scored answers
  against the one-hot truth (NLL of the true name); the sharper question is
  whether the model's whole answer *distribution* is shaped like the geometry —
  build a target distribution per prompt from RGB distance to the true mix
  (e.g. softmax of −distance/τ over the vocabulary) and measure cross-entropy /
  KL against it, sweeping τ. Needs no re-run: the eval step saved the full
  log-probability vector over color tokens for every prompt (`arrays`
  `{label}/logp/{set}`), so this is a report-side analysis. Ex-2.1.4's report
  now implements the τ-sweep (KL(q_τ ‖ p) with a uniform reference); reuse its
  recipe and τ grid so the rungs compare directly. Note from ex-2.1.4: the
  metric jointly scores geometry *and* calibration — a confidently-wrong model
  fits worse than uniform — so read it beside s₂. #[D2.1] #ex-2.1.3 #metrics

- [ ] Cipher-name condition for the char-level language (deferred from
  ex-2.1.4): names whose letters encode the value through a per-position
  substitution, separating a compositional surface from a value-spelling one.
  Run only if the anchored experiments need to know *why* multi-token naming
  costs precision, not just that it does. #[D2.1] #vocab #task-grammar

- [ ] Probe the mid-emission dip (ex-2.1.4): at v216 the final-depth mix R²
  falls from ≈ 0.95 (pre-answer, first answer char) to ≈ 0.6 mid-name and
  returns to ≈ 0.96 at the last character. One seed, one grid so far; the
  all-positions probe item above would cover it. If real, it says the value is
  diluted (not evicted) while spelling completes — relevant to where an
  anchored result direction is enforced. #[D2.1] #ex-2.1.4 #representations

- [ ] If anchoring a composed concept fails in D2.1.x, run a word-level tokenizer
  ablation (one token per color name, hex still char-level): it separates "anchoring
  fails for transformers" from "anchoring fails for concepts that don't coincide
  with an embedding row". Worth testing, but perhaps the char-level task is
  closer to what M2 claims; need to think on this more. Ex-2.1.3 de-risks the
  training side: name-only word-level corpora learn the geometry end-to-end.
  #[D2.1] #anchoring #vocab

- [ ] Cheap capacity/superposition proxies for the ex-2.1.x eval step: per-layer
  participation ratio of residual-stream activations (eigenspectrum of the
  covariance — how many effective dimensions the model uses) and pairwise |cos|
  between the fitted probe directions (operand vs result vs redness — interference
  between concepts). Both fall out of arrays `eval_one` already computes; they'd let
  the width sweep read as a compression axis. Full superposition accounting (feature
  dictionary / SAE) is its own experiment, after D2.1.2. #[D2.1] #superposition

- [ ] Narrow the stream to raise superposition pressure — sequenced, not up front.
  d64 is generous for this task, and SCA's value proposition lives where geometry
  is contested. Plan: (1) keep d64-L4 for the first anchored runs, so the only
  change vs the existing baselines is the anchor; (2) un-anchored width × depth
  sweep on the chosen testbed (e.g. word-level v216: d16/d32 × L4/L8) to find the
  narrowest cell that still solves the task — the capacity proxies (item above)
  then read as a compression axis; (3) re-run the anchored comparison along the
  width axis down to that frontier. Prefer deep-and-narrow (d16-L8) over wide:
  width sets per-position capacity, depth adds anchor sites, and ngpt-scaling
  says the architecture tolerates the aspect ratio. Watch-out: at v216 the
  softmax's identity separability may fail before value geometry does — which is
  itself the identity-vs-value competition ex-2.1.3 flagged. #[D2.1]
  #superposition #model-arch #ex-2.1.3

- [ ] Make the operation a variable before D2.2. `sca/data/colors.py` hardcodes one
  op (`mix`, spelled `+`); anchoring *the operation* only makes sense once there is
  more than one. Add an operation table (name, surface form, grid fn with defined
  rounding — saturating add/subtract and screen all stay closed on 0..15), thread an
  `op` field through `Example`, and key the seen-pair bookkeeping on `(op, pair)`.
  Spell operators as *words* (`red mix blue = purple`), not symbols, so the operation
  concept is multi-token like the colors. No need to keep `+` compatible — each
  experiment retrains from scratch and carries its own control. Probe positions in
  `sca/compute/evaluation.py` assume the infix `a <op> b = ` frame; keep that frame.
  #[D2.2] #task-grammar

- [ ] Does tuning move the mix computation earlier in depth? Conjecture from
  ex-2.1.5 drafting: pretraining has no pressure to compute the result before
  the answer position, so the mix stays pressed against the last layer (ex-2.1.4
  saw exactly this at L4); an instruction-tuned variant might instead show
  operands decodable mid-stack and results near the head. Relevant to anchoring
  because it changes how many layers a result-concept anchor can act on.
  #[D2.1] #representations #ex-2.1.5

- [ ] Confirm the simplified nGPT gate holds at a genuinely larger size (wider/deeper
  than 128×12, bigger GPU + batch) before leaning on it for M3. ngpt-scaling shows
  the fixed scalar α = 1/n_layer trains flat across the width × depth grid we can
  afford. #model-arch

## Findings & notes to carry forward

- **Multi-token naming keeps the geometry where evidence is dense; `v27` cannot
  say anything either way (ex-2.1.4, 2026-07-19; revised 2026-07-22).** Char-level twin of
  ex-2.1.3: corpora identical line for line, every color an opaque four-letter
  random name (v27 + v216, d64-L4). At v216, held-out exact match is 0.91
  (word level: 0.99), misses are one grid level off in one channel (59 of 68,
  pooled), zero malformed completions anywhere, and s₂ ≈ 0. At v27 exact match
  collapses to 0/10 on every seed (word level: 0.27) and the model is confidently
  wrong (s₂ ≈ 0.9): 26 of 30 held-out guesses are one-step neighbors, and open-pair
  guesses land 0.41 against a floor of 0.29 — the neighborhood structure is learned;
  the exact naming is not. But v27 grades nothing: its closed-pair universe is
  76 equations in total (66 train / 10 holdout), 27 of the training pairs are
  `a + a = a`, three names never appear in a mix at all, and one held-out pair
  is made entirely of those three — unanswerable from the corpus. On those ten
  pairs a shell-confined guesser scores 0.18 exact and the prompt-blind constant
  0.10, so word level's 0.27 and this experiment's 0.0 sit in the same band. Read
  the v27 cells as absent evidence rather than as a cost of multi-token naming,
  and prefer v216 or denser for rungs that have to carry a comparison.
  Two null checks keep that honest on so coarse a grid
  (see the methodology note below): the model hands back an operand 53% of the
  time against a 40% neighbor-shell null (≈1.5 SE, 10 distinct pairs — no operand
  echo), and its open-pair distance beats a prompt-blind constant of 0.48, not just
  chance at 0.82. What separates it from prompt-blind is the counts, not the
  distance means: nearest-name 33% vs 18%. Mechanism: reading names occupies depths 1–3 (operand
  probe R² 0.11 → 0.98 across layers), the mix crystallizes only in the final
  block at v216 (pre-answer R² ≈ 0 through depth 2, ≈ 0.96 at depth 4,
  transferring ≈ 0.95 to held-out and open prompts), and emission is holistic —
  all three channels stay decodable together with no per-channel eviction, unlike
  hex's staircase (a mid-name dip to ≈ 0.6 recovers by the last character).
  Consequences for anchoring: the result concept's home is the pre-answer
  position, but on a 4-layer stack it exists for only one layer (strengthens the
  deep-and-narrow plan); operand concepts exist from depth 1–2 and are easier
  anchor targets; and the base language's `named_holdout` = 0 looks like this
  v27 regime (sparse named sub-grid) compounded by the hex pathway. Full
  analysis in `docs/m2/ex-2.1.4/report.py`. #[D2.1] #ex-2.1.4 #vocab
  #representations #anchoring

- **Color geometry is inferable from names alone; vocabulary density sets exact match
  (ex-2.1.3, 2026-07-19).** Trained the un-anchored d64-L4 transformer on a named-only language
  (one token per color, no hex) over vocabularies of 27/64/216/4096 grid colors. Every
  size learns the latent cube: embeddings hold RGB as a linear subspace (ridge R² up
  to ≈ 0.95), the mix is decodable at the pre-answer position (R² ≈ 0.9 from depth 1–2,
  transferring to held-out and open prompts), and guesses land near the nearest-name
  floor even for pair types never trained on. Held-out exact match is non-monotonic —
  0.27 / 0.59 / ≈ 1.0 / 0.65 (v27's 0.27 is inside the null band, per the ex-2.1.4
  note above — that cell measures the split more than the model) — and the full
  grid's misses are one grid level off in
  one channel (precision, not knowledge; not concentrated at rounding boundaries).
  Consequences: the base language's `named_holdout` = 0 was a property of its grammar,
  not of name-only supervision; a ~216-color one-token vocabulary is a sweet spot for
  anchored runs (task solved, geometry clean, open pairs remain as graded probes); a
  single-token answer gives the result concept a fixed home position, unlike the
  just-in-time, evicted hex answer; and embedding variance splits into a small
  value-geometry subspace plus a large identity/separability remainder — the
  superposition watch item in miniature. Full analysis in `docs/m2/ex-2.1.3/report.py`.
  #[D2.1] #ex-2.1.3 #vocab #geometry #representations

- **`named_holdout` is unsolved in 4 layers; value → name translation is the blocker
  (ex-2.1.2, 2026-07-15).** The 2×2 factorial (reverse aliases × off-palette
  named-as-hex, d64-L4) trained both missing ingredients — reverse aliases read
  out at 1.0 in their own frame, and name + name arithmetic generalizes to unseen
  off-palette pairs at ≈ 0.92 — yet `named_holdout` stays at exactly 0 in every cell.
  Decomposition: in the `open` cells ~1/3 of held-out answers are the correct mix value
  *in hex form* (form rule learned per-pair, not per-value); the rest are lookup-neighbor
  names. The name-identity margin (log P(true name) − best other name) sits ≈ −9 nats
  everywhere, so value → name never engages mid-equation though it is perfect in the
  `#hex = ` frame. Consequence: anchored runs train on the `both` corpus and use
  `open_holdout` + s₂ as graded canaries (`named_holdout` has no headroom to lose).
  Whether `named_holdout` is solvable at all in 4 layers is parked — candidates: a
  denser named sub-grid (value-diverse rgb→name supervision *in-frame*, which changes
  the concept inventory the anchors will label), more depth, or a frame-interleaving
  curriculum. Full analysis with figures in `docs/m2/ex-2.1.2/report.py`. #[D2.1]
  #ex-2.1.2

- **Just-in-time computation *with eviction* (ex-2.1.2 answer-schedule probe).** At the
  final layer, channel k is decodable (R² ≈ 0.97) only at its own emission position, and
  previously-emitted channels are dropped from the deep residual stream — so a "result"
  concept never fully exists at any single position, and anchoring one there would fight
  the model's schedule. Carry into anchor design. #[D2.1] #ex-2.1.2 #representations

- **Calibrate the redirect's γ against the model's pre-norm activation scale** instead of
  the fixed γ = 1. Ex-2.9.3 found the fixed value silently no-ops on ~1 run in 250 (the
  bias fails to dominate that seed's pre-norm residual, so "deleted" red passes through
  nearly untouched); ex-2.9.2 saw the same once. Cheap fix: set γ to a multiple of the
  ablated row's typical pre-norm contribution, measured on the train set after training.
  #anchoring #ex-2.9.3

- **Choose measurement sites independently of the statistic being judged
  (methodology, 2026-07-23).** From ex-2.1.5 drafting: when a comparison needs a
  probe site (a layer × position cell), picking the site that maximizes the
  reported statistic is a selection effect — the maximum of a noisy map rises
  with the noise, which can manufacture a trend (e.g. across widths) on its own.
  Pick the primary site by an independent criterion (e.g. strongest within-form
  probe R² when judging cross-form transfer), and report the statistic's own
  best site beside it as an explicit upper bound. Also banked in the writing
  skill's preregistration section. #metrics #ex-2.1.5

- **On coarse grids, state the null before reading a pattern as behavior
  (methodology, 2026-07-21).** Two overclaims found in ex-2.1.4's `v27` analysis and
  corrected, both from a reference that was too weak for a 27-name vocabulary.
  (1) "The model often hands back an operand" read as an operand echo. But closure
  forces each channel of a training pair to agree or to hold both end levels, so an
  operand is one grid level from the mix by construction — a member of the mix's
  one-step shell, and that shell holds only 4–6 names. Uniform choice within it
  returns an operand 40% of the time against 53% observed. (2) "Guesses are far from
  random" measured against `chance_dist`, a uniform-random name. Mixes cluster toward
  the cube's centre, so a prompt-blind model that always answers the training
  answers' centroid scores 0.48 on v27 open pairs where chance is 0.82 and the floor
  is 0.29 — it eats most of the apparent headroom, and on v27 held-out pairs it
  matches the model outright (0.57 vs 0.55). Rules of thumb for the anchored runs:
  a mean-distance metric needs the prompt-blind constant beside it, not just chance;
  prompt-dependent counts (nearest-name rate, shell membership) separate model from
  baseline where distance means cannot; and check whether a "striking" coincidence is
  forced by the grid's combinatorics before attributing it to the model. The
  references now live in `src/sca/baselines.py` (`blind_index`, `shell_mask`,
  `neighborhood_exact_null`, `operand_shell_null`, `k_nearest_stats`,
  `self_nearest_rate`), so all four D2.1 reports compute them the same way.
  #metrics #task-grammar #ex-2.1.4

- **Retcon sweep of ex-2.1.1 to ex-2.1.4 for unsupported claims (2026-07-22).** Applied
  the null discipline above to the three earlier reports; all are rebuilt and their
  numbers are now computed in-cell rather than asserted. What changed. ex-2.1.3: its
  `v27` held-out accuracy of 0.27 is only 1.2 SE over the 0.18 neighborhood null on ten
  pairs, so the "geometry is inferable from names" conclusion now rests on the denser
  grids (14–65 SE) and the embedding probes; "guesses track the floor" was measured
  against too weak a bracket, and every grid in fact sits behind a coin flip between the
  two names bracketing an open mix (`v27` 0.326 vs 0.302, nearest-name 0.50 vs 0.82);
  the "under 1 cell of probe error, so the nearest name is its own" inference is false
  (a Voronoi cell reaches half a step, and the real self-nearest rate is 0.51 at `v27`,
  0.61 at `v64`, 0.76 at `v216`); two cited example misses did not exist in the data;
  a caption said six rows where the code rendered four. ex-2.1.2: the stated s₂ range
  on zero-accuracy cells (0.5–0.7) is really 0.41–0.85, mean 0.61; the garden-path
  retelling pinned `lime + black` to `teal`, which is seed-dependent (`control` gives
  gray, olive, teal across seeds); "falls back on the nearest lookup neighbor" was
  asserted but never tested, and is now marked as untested. ex-2.1.1: "the correction
  lifts *green* about 70× (to 13%)" was really *gray*'s numbers — green moves 96× but
  only from 2e-08 to 2e-06, while teal holds 84%, which strengthens the report's point
  that the operand correction never reaches the arithmetic. Its neighbor and
  seed-agreement claims did clear their nulls (25/30 in the one-step shell against 17%;
  4 of 10 pairs unanimous) and now say so. #metrics #ex-2.1.1 #ex-2.1.2 #ex-2.1.3

## Queued issues

- [sca2#10](https://github.com/z0u/sca2/issues/10) — D2.1 kickoff: carry-over lessons
  and hypothesis queue from ex-2.9.3/2.9.4 (schedule ordering, the fallback analog, γ
  calibration, superposition watch-outs). Read before designing the first transformer
  anchoring experiment. #[D2.1]

---

# Tags

 <!-- Keep links in sync with section headings so the tags link to them properly -->

## M2: Concept control in transformers with Sparse Concept Anchoring

[M2]: #m2-concept-control-in-transformers-with-sparse-concept-anchoring

Milestone 2 of SCA, in which we attempt to get the method working in
transformers. As described in [the Manifund proposal](https://manifund.org/projects/concept-control-in-transformers-with-sparse-concept-anchoring).

## D2.1: Basic concept anchoring in transformers

[D2.1]: #d21-basic-concept-anchoring-in-transformers

Anchor a concept such as _red_ across the residual stream in the color-mixing
task; probe each layer for the anchored concept, and confirm that completion
accuracy (predicting the correct result color) matches an un-anchored baseline.

## D2.2: Anchor operations

[D2.2]: #d22-anchor-operations

Anchor an abstract _operation_ (e.g. _addition_, rather than a concrete
attribute like _redness_); sweep over layers; confirm task performance is intact
and that suppression scales as it did in the autoencoders.

## D2.3: Asymmetry

[D2.3]: #d23-asymmetry

Add a verification task (`red + blue = purple TRUE/FALSE`) and test whether
suppression can degrade _completion_ while preserving _verification_: the
experimental analog for letting a model recognize a behavior without being able
to produce it.

## D2.4: Consolidation

[D2.4]: #d24-consolidation

Publication. May involve writing a paper and posting on arXiv; likely involves ensuring
LessWrong posts are up to date and reviewed; unlikely to involve seeing the
paper through a review process.

Outreach. Drawing attention to the lessons from M2.

## Concepts

### Representations

[representations]: #representations

How concepts are laid out in the residual stream: where a probe reads out, which
positions carry which channels, and how the model schedules computation across
layers and token positions.

### Anchoring

[anchoring]: #anchoring

The SCA method itself as applied here — the redirect/suppression mechanism, its
knobs (e.g. γ), and design choices for *where* and *how* a concept is anchored.

### Superposition

[superposition]: #superposition

Capacity and interference: how many effective dimensions the model uses and how
much distinct concepts overlap. Proxies now; feature dictionaries / SAEs later.

### Metrics

[metrics]: #metrics

Diagnostics we report per run — e.g. s₂ (surprise-surprise) as a calibration dial,
alongside accuracy and raw surprisal.

### Task grammar

[task-grammar]: #task-grammar

The color-mixing synthetic language: operands, operations, surface forms, and the
train/holdout splits (`open`, `named`, `both`, …).

### Model architecture

[model-arch]: #model-arch

The transformer we train on the task (nGPT variant, width/depth) and how those
choices hold up as we scale toward M3.
