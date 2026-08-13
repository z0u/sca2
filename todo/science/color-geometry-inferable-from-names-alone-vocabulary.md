---
status: finding
tags: [D2.1, ex-2.1.3, vocab, geometry, representations]
opened: 2026-07-19
---
# Color geometry is inferable from names alone; vocabulary density sets exact match

Trained the un-anchored d64-L4 transformer on a named-only language (one token per color, no hex) over vocabularies of 27/64/216/4096 grid colors. Every size learns the latent cube: embeddings hold RGB as a linear subspace (ridge R² up to ≈ 0.95), the mix is decodable at the pre-answer position (R² ≈ 0.9 from depth 1–2, transferring to held-out and open prompts), and guesses land near the nearest-name floor even for pair types never trained on. Held-out exact match is non-monotonic — 0.27 / 0.59 / ≈ 1.0 / 0.65 (v27's 0.27 is inside the null band, per the ex-2.1.4 note above — that cell measures the split more than the model) — and the full grid's misses are one grid level off in one channel (precision, not knowledge; not concentrated at rounding boundaries). Consequences: the base language's `named_holdout` = 0 was a property of its grammar, not of name-only supervision; a ~216-color one-token vocabulary is a sweet spot for anchored runs (task solved, geometry clean, open pairs remain as graded probes); a single-token answer gives the result concept a fixed home position, unlike the just-in-time, evicted hex answer; and embedding variance splits into a small value-geometry subspace plus a large identity/separability remainder — the superposition watch item in miniature. Full analysis in `docs/m2/ex-2.1.3/report.py`.
