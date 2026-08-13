---
status: open
tags: [D2.1, anchoring, vocab]
---
# Word-level tokenizer ablation, if anchoring a composed concept fails

If anchoring a composed concept fails in D2.1.x, run a word-level tokenizer ablation (one token per color name, hex still char-level).

it separates "anchoring fails for transformers" from "anchoring fails for concepts that don't coincide with an embedding row". Worth testing, but perhaps the char-level task is closer to what M2 claims; need to think on this more. Ex-2.1.3 de-risks the training side: name-only word-level corpora learn the geometry end-to-end.
