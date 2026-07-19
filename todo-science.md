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

- [ ] Distance-shaped answer targets for ex-2.1.3, post-hoc. We scored answers
  against the one-hot truth (NLL of the true name); the sharper question is
  whether the model's whole answer *distribution* is shaped like the geometry —
  build a target distribution per prompt from RGB distance to the true mix
  (e.g. softmax of −distance/τ over the vocabulary) and measure cross-entropy /
  KL against it, sweeping τ. Needs no re-run: the eval step saved the full
  log-probability vector over color tokens for every prompt (`arrays`
  `{label}/logp/{set}`), so this is a report-side analysis. Ex-2.1.4 computes
  the same thing from candidate scoring; do 2.1.3's for the side-by-side.
  #[D2.1] #ex-2.1.3 #metrics

- [ ] Char-level twin of the name-only language (ex-2.1.4 candidate). Same corpus
  distribution as ex-2.1.3, only the tokenizer changes: names spelled character by
  character, so concepts are multi-token again and answers have an emission
  schedule. Isolates whether one-token-per-concept is load-bearing for geometry
  inference, and lets ex-2.1.2's answer-schedule probe ask whether just-in-time
  computation and eviction return with multi-token answers — on data with no hex
  path and no form rule. Design trap: the synthetic names (`c05f`) spell the value
  per character, which is hex with a prefix at char level; rebuild them as opaque
  fixed-length random strings (holistic binding, the true no-scaffolding control).
  Optional later condition: cipher names (per-position shuffled letters), to
  separate a compositional surface from a value-spelling one. Scope: v216 + v27,
  2–3 seeds, d64-L4. A failed holdout here is informative — it would say
  multi-token naming itself is the bottleneck. Ladder framing: word-level
  (ex-2.1.3) is the easy rung for anchoring, this is the middle rung, the base
  language is the M2 claim; divergence between rungs is a result, not a nuisance.
  #[D2.1] #ex-2.1.3 #vocab #representations #task-grammar

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
  change vs the frozen baselines is the anchor; (2) un-anchored width × depth
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

- [ ] Confirm the simplified nGPT gate holds at a genuinely larger size (wider/deeper
  than 128×12, bigger GPU + batch) before leaning on it for M3. ngpt-scaling shows
  the fixed scalar α = 1/n_layer trains flat across the width × depth grid we can
  afford. #model-arch

## Findings & notes to carry forward

- **Color geometry is inferable from names alone; vocabulary density sets exact match
  (ex-2.1.3, 2026-07-19).** Trained the un-anchored backbone on a named-only language
  (one token per color, no hex) over vocabularies of 27/64/216/4096 grid colors. Every
  size learns the latent cube: embeddings hold RGB as a linear subspace (ridge R² up
  to ≈ 0.95), the mix is decodable at the pre-answer position (R² ≈ 0.9 from depth 1–2,
  transferring to held-out and open prompts), and guesses land near the nearest-name
  floor even for pair types never trained on. Held-out exact match is non-monotonic —
  0.27 / 0.59 / ≈ 1.0 / 0.65 — and the full grid's misses are one grid level off in
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
  named-as-hex, frozen d64-L4) trained both missing ingredients — reverse aliases read
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

- **s₂ (surprise-surprise) is now a standard metric.** Adopted in ex-2.1.2
  (`answer_calibration`: mean answer nll/entropy/s₂ per eval set per cell) and it behaved
  exactly as designed — snaps from ≈ 0.7 (confidently wrong) to ≈ 0 in precisely the
  conditions that train a form, per-set. Caveats live in the docstring: it measures
  calibration, not competence (uniform ignorance also scores ≈ 0 — pair with accuracy or
  raw surprisal), and per-token s₂ is a noisy one-sample draw, so aggregate over many
  positions. Carry into the anchored runs as the early-warning dial. #metrics #[D2.1]

- **Calibrate the redirect's γ against the model's pre-norm activation scale** instead of
  the fixed γ = 1. Ex-2.9.3 found the fixed value silently no-ops on ~1 run in 250 (the
  bias fails to dominate that seed's pre-norm residual, so "deleted" red passes through
  nearly untouched); ex-2.9.2 saw the same once. Cheap fix: set γ to a multiple of the
  ablated row's typical pre-norm contribution, measured on the train set after training.
  #anchoring #ex-2.9.3

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
