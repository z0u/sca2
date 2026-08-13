# Science todo

Experiment questions and findings. Add to it when you notice something interesting in a run but want to defer the investigation. Infrastructure and tooling work lives in [`eng/`](../eng/); the schema and the `./go todo` query are described once, in [`todo/README.md`](../README.md).

Two kinds of item share this directory. An **open question** is work: something to run or decide. A **finding** (`status: finding`) is established knowledge with no completion state — a result worth carrying into the next experiment. `./go todo science` shows the first; `./go todo science --status finding` shows the second.

Items are tagged by the deliverable or experiment they came from, the one they bear on, and the concepts they touch. The vocabulary follows.

## Tags

### M2: Concept control in transformers with Sparse Concept Anchoring

Milestone 2 of SCA, in which we attempt to get the method working in transformers. As described in [the Manifund proposal](https://manifund.org/projects/concept-control-in-transformers-with-sparse-concept-anchoring).

### D2.1: Basic concept anchoring in transformers

Anchor a concept such as _red_ across the residual stream in the color-mixing task; probe each layer for the anchored concept, and confirm that completion accuracy (predicting the correct result color) matches an un-anchored baseline.

### D2.2: Anchor operations

Anchor an abstract _operation_ (e.g. _addition_, rather than a concrete attribute like _redness_); sweep over layers; confirm task performance is intact and that suppression scales as it did in the autoencoders.

### D2.3: Asymmetry

Add a verification task (`red + blue = purple TRUE/FALSE`) and test whether suppression can degrade _completion_ while preserving _verification_: the experimental analog for letting a model recognize a behavior without being able to produce it.

### D2.4: Consolidation

Publication. May involve writing a paper and posting on arXiv; likely involves ensuring LessWrong posts are up to date and reviewed; unlikely to involve seeing the paper through a review process.

Outreach. Drawing attention to the lessons from M2.

### Concepts

#### Representations

How concepts are laid out in the residual stream: where a probe reads out, which positions carry which channels, and how the model schedules computation across layers and token positions.

#### Anchoring

The SCA method itself as applied here — the redirect/suppression mechanism, its knobs (e.g. γ), and design choices for where and how a concept is anchored.

#### Superposition

Capacity and interference: how many effective dimensions the model uses and how much distinct concepts overlap. Proxies now; feature dictionaries / SAEs later.

#### Metrics

Diagnostics we report per run — e.g. s₂ (surprise-surprise) as a calibration dial, alongside accuracy and raw surprisal.

#### Task grammar

The color-mixing synthetic language: operands, operations, surface forms, and the train/holdout splits (`open`, `named`, `both`, …).

#### Model architecture

The transformer we train on the task (nGPT variant, width/depth) and how those choices hold up as we scale toward M3.
