---
status: open
tags: [ex-2.1.10, task-grammar, anchoring]
---
# Distinct operand lexicons, so per-group embedding reads can bind

With one lexicon serving both operand slots, the embedding-slice weight profile is symmetric between ex-2.1.10's label groups by construction — a token's embedding cannot depend on its line — so the H2(a) embedding read is near-vacuous and all the localization content sits in the post-attention slices. Disjoint op1/op2 color vocabularies (same palette, two names per color) would make the embedding read informative and remove the head start the shared lexicon gives the pull. (2026-08-10, from the ex-2.1.10 discussion round.)
