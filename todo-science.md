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

- [x] **Refit the probe maps holding out one target *value* at a time, not one
  equation.** Done in ex-2.1.5: `geometry.strict_r2` reports both estimators
  side by side, the report shows the strict figure with a disagreement table,
  and the three operand-closing spaces are landmarks. The residual question is
  the fold-strength asymmetry, split out as its own item below.

  Original note follows. `probe_maps` runs `ridge_probe_loo` over rows = equations, so a
  color (or a hex digit) sits in both the fit and the held-out row and the probe
  can memorize an identity → value table. That inflates some cells to the point
  of tautology and leaves others untouched, and which is which is the whole
  question — see the finding below for a one-cell prototype. The fix is a
  leave-one-group-out where the group is the target's value in the channel being
  scored (16 groups per hex digit, ~100–220 for named). Costs an eval-stage
  re-run only (checkpoints are published under `CKPT_REF`), and it is *cheaper*
  than the current fit. Two decisions to make: whether to report it beside the
  preregistered per-equation numbers or in place of them, and whether the
  cross-form ρ and principal angles (full-data fits, so unaffected mechanically)
  should be gated on the stricter R². #metrics #ex-2.1.5 #representations

  Riding along in the same re-run: add the three operand-closing spaces to
  `LANDMARKS` (see the delimiter finding below). That makes a hex line fully
  measured — its operand middles are zero characters, so every position is a
  landmark — and leaves elision meaning one thing only, a name span too variable
  to align. Decided against dropping the second-to-last landmark to avoid short-name
  aliasing: for hex it is the green digit's own position, dropping it would break
  the shared-landmark cross-form comparison, and the aliasing it would fix affects
  0/768 lines at the centre cell (the 140-name palette's shortest name is 4
  characters) and 5/768 in `palette-250`, whose only 3-letter name is `ice`. A
  footnote covers it. The caption should instead say that the elided span is 0–22
  characters wide depending on the line (median 6), since that averaging is the
  real approximation and it applies to every named panel.

- [ ] **The strict holdout is not equally strict across targets, so R² is not
  comparable between the operand rows and the mix row.** `_value_folds` groups
  by the target's value in one channel. For a named operand that group *is* the
  colour — every line carrying the name shares its R value — so ~93 folds remove
  a median 15/768 lines and the name is genuinely gone. A mix value is a
  midpoint, mostly unique to a line or two: ~219 folds removing a median 4/768,
  with both operand names still in the fit. So "named mix @ pre = 0.94" and
  "named op1 @ pre = 0.64" are answering different questions, and the report
  currently reads them side by side as if they weren't.

  Checked, and the headline survives: an identity-grouped fold (hold out every
  line where a colour is an operand, score the lines where it is operand 1, so
  every scored line has an unseen operand) gives mix @ pre = 0.94, op1 @ pre =
  0.64 — both unchanged to two places, over 139 folds × 3 seeds. Two things
  still worth doing. (1) Report the mix row under identity folds so the
  comparison is like for like; the per-channel grouping is the right one for
  hex, where the digit *is* the value, so this is a per-form choice. (2) Explain
  why mix beats its own inputs. If the probe could only compose two independent
  operand readouts, op1 at 0.64 with op2 seen bounds mix at ~0.82, and we see
  0.94 — either the mix has its own direction by `pre`, or the probe is reading
  the answer the model has already chosen and looking that colour up (the fold
  removes a colour only where it is an *operand*). Same mechanism as the
  "answer positions partly read the answer" section, arriving one position
  earlier. A fold that also removes lines where the colour is the answer would
  separate them. #metrics #ex-2.1.5 #representations

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
  the model's schedule. Carry into anchor design. Replicated in ex-2.1.5 on a corpus with
  no name↔hex bridge, per channel: at the last layer each channel is decodable about one
  position before its digit is emitted and fades after (0.98 red at `#`, 0.96 green a
  token later, 0.95 blue at the last digit with red down to 0.38), so the schedule isn't
  an artifact of the bridged language. #[D2.1] #ex-2.1.2 #ex-2.1.5 #representations

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

- **The hex embedding has one magnitude axis, and channel identity is positional
  (ex-2.1.5, 2026-07-26).** The same 16 digit tokens serve all three channels, and the
  model uses RoPE, so the depth-0 residual is the token embedding with no positional
  component. The probe weights confirm what that implies: the readout direction for red
  at digit 1, green at digit 2 and blue at digit 3 are *the same vector* — cosine
  +1.000 on all three seeds. So at the input there is no hex "red" direction to anchor
  or to align with named red; there is one value axis, read three times, with position
  supplying the channel. The channels do differentiate with depth (within one landmark
  at the last layer the three directions run cos −0.66 to +0.09), so channel-specific
  directions exist — just not where the tokens enter. Bears directly on anchor
  placement for a mixed-vocabulary corpus. #anchoring #ex-2.1.5 #representations

- **Per-equation leave-one-out lets probes memorize identity → value; per-value
  holdout separates memory from geometry (ex-2.1.5, 2026-07-26).** Refit of all three
  centre seeds, locally from published checkpoints — no Modal needed, and the grouped
  fit is cheaper than the shipped one because it shares a Gram matrix across groups.
  Three protocols: *equation* (shipped), *value* (hold out rows whose scored channel
  carries the value), *strict* (hold out rows where the value appears in **any** slot —
  operand 1, operand 2 or the answer — which closes the path where the same hex digit,
  or the same colour in the other operand slot, teaches its own direction from a slot
  the value holdout left alone). Refit *equation* reproduces the shipped arrays to
  4e-4, so the pipeline is faithful. Headline cells, seed-averaged:

  | site | equation | value | strict |
  |---|---|---|---|
  | hex op2, own digit, embedding | 1.000 | 0.416 | 0.416 |
  | hex op2, own digit, last layer | 0.982 | 0.781 | 0.781 |
  | hex op2, red retained at the green digit, L4 | 0.821 | −0.285 | −0.269 |
  | hex mix, pre-answer, L4 | 0.376 | 0.325 | 0.196 |
  | named op1 / op2, last operand char, L4 | 0.750 / 0.716 | −0.046 / 0.006 | −0.047 / 0.001 |
  | named mix, pre-answer, L4 | 0.944 | 0.943 | **0.941** |

  Cross-slot leak (value → strict) is exactly zero at the embedding, where no attention
  has run — the control behaving as theory demands. It stays negligible for named
  (mean +0.003 on op2) and concentrates in the hex mix (mean +0.10, +0.13 at
  pre-answer), because the 16 digits are shared across all three channel slots.
  Four outcomes, which is why the protocol matters:
  (1) hex digit cells at the embedding collapse 1.00 → 0.44, but land well above zero,
  so the digit tokens do carry a real (partial) magnitude axis rather than an arbitrary
  lookup;
  (2) the same hex cell at the last layer holds 0.98 → 0.76, so depth *improves* the
  value geometry instead of merely relaying the token;
  (3) named operand readout falls 0.75 → 0.19 — the "holistic operand bundle" is mostly
  name-identity recovery, which fits ex-2.1.4's finding that value → name translation is
  the blocker and that named holdout accuracy is weak;
  (4) the named mix at the pre-answer site is untouched, 0.944 → 0.943, so the headline
  H2 result is geometry and survives the stricter test.
  Net: the two forms carry value structure in different places — hex in its digit tokens
  (deepening with depth), named only in the computed mix. That, rather than token
  binding, is what a cross-form anchor would have to bridge. #metrics #ex-2.1.5
  #anchoring #representations

- **A named operand's value lives at the space that closes it, not on its characters —
  and the landmark set doesn't measure those spaces (ex-2.1.5, 2026-07-26).** Under the
  strict holdout, named operand 1 scores −0.05 at its own last character in the last
  layer, but +0.35 at the space that follows it and +0.47 at the space after the `+`;
  operand 2 scores +0.30 at the space before the `=`. So the earlier reading that named
  operands carry no value geometry was an artefact of *where* we measured. A
  variable-length name has no fixed slot, so the model appears to resolve it into the
  delimiter — a summary position — and the value stays there across the operator.
  Hex does the opposite: at the same spaces it scores −0.19 to −0.34, nothing at all,
  which fits a form whose three digits already sit at fixed offsets and need no summary
  slot. `LANDMARKS` measures the operator characters and the pre-answer space but not
  the two operand-closing spaces, so the current figure omits the named form's most
  informative sites. Add them before the H2 rewrite. Bears on anchor placement: this is
  a stable single-position home for a named operand's value, and hex has no counterpart.
  #representations #anchoring #ex-2.1.5 #task-grammar

  **Qualified by a word-family control (2026-07-26).** 99 of the 140 names are
  multi-word and modifiers recur (`green` in 26, `light` in 11), so part of what a
  delimiter holds may be "this name contained *green*" rather than a resolved colour.
  Holding out whole word families tests it, but names sharing a word also share a
  region of the cube, so the holdout removes a colour *cluster* and the drop would be
  ambiguous — hence a colour-matched control that removes the same number of names by
  RGB proximity instead. The contrast (word − colour, last layer, 3 seeds):

  | target, site | value | colour-matched | word-family | word − colour |
  |---|---|---|---|---|
  | op1 @ its own last character | +0.245 | −0.297 | −0.831 | **−0.534** |
  | op1 @ its closing space | +0.320 | −0.062 | −0.488 | **−0.426** |
  | op2 @ its closing space | +0.372 | +0.017 | −0.353 | **−0.370** |
  | op1 @ the space after `+` | +0.559 | +0.284 | +0.209 | −0.075 |
  | mix @ pre-answer | +0.879 | +0.644 | +0.772 | **+0.127** |

  So a name's *own* closing space is substantially lexical, and "the model resolves the
  name into the delimiter" was too clean a story. Two things survive it: one token
  further on, at the space after the `+`, the lexical dependence is small and the value
  still reads +0.209 under a holdout that removes 14 sibling names at the median (52 at
  the max); and the mix at pre-answer shows none at all — its sign flips, so H2's
  headline is not a lexical artefact. A bias runs *against* this reading, which makes it
  sturdier: modifier families like `light` are spread across the cube rather than
  clustered, so for those names the word holdout removes a scattered set that is easier
  to extrapolate from than the tight cluster the control removes.

  (Numbers here are restricted to the 108 names that have a family, and group the mix by
  the snapped answer name rather than by the exact mix value, so they are not comparable
  cell-for-cell with the strict-holdout table above — only within this table.)

- **Retention and value-geometry are different questions, and ex-2.1.2's eviction
  finding may have measured the first (2026-07-26).** In ex-2.1.5, hex operand 2's red
  channel read at the *green* digit in the last layer scores 0.821 per-equation and
  −0.27 under a strict holdout. Both are true of different things: red's identity is
  still recoverable there (causal attention reaches back to its digit), but it is no
  longer placed by value. So "the earlier channel persists into later digits" is an
  identity claim, not a geometric one — and the report sentence that read it as depth
  accumulating a value representation is wrong and needs replacing. The same question
  applies upstream: ex-2.1.2's just-in-time-*with-eviction* result (R² ≈ 0.97 at a
  channel's own emission position, dropped afterwards) was measured on a small
  vocabulary with the same per-equation holdout, so its "dropped" may mean "no longer
  recoverable" or "no longer value-organized". Worth a re-check before anchor design
  leans on it. #representations #ex-2.1.2 #ex-2.1.5 #metrics

- **Use the embedding row as a surface-text control for any probe read at an answer
  position (methodology, 2026-07-26).** Under teacher forcing the answer tokens are in
  the input, so a probe there can succeed by reading the label. Depth 0 is the token
  lookup before any attention or MLP, so whatever it decodes is present by definition and
  is the right baseline. In ex-2.1.5's hex form it is nearly everything: at the embedding
  a mix probe scores R² ≈ 1.00 per channel at each answer digit, and an operand probe
  scores ≈ 0.48 there — the latter is arithmetic (the answer digit is the mean of the two
  operand digits, so it pins about half of each operand's variance), and both operands
  echo equally, which distinguishes mixing from a retained operand. The equals sign and
  the pre-answer space score −0.003 at the embedding in both forms, so they are clean
  ground; prefer them for cross-form ρ, which answer positions would inflate in both
  directions. #metrics #ex-2.1.5 #representations

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
