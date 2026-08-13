---
status: finding
tags: [metrics, ex-2.1.5, representations]
opened: 2026-07-26
---
# Use the embedding row as a surface-text control for any probe read at an answer position

Under teacher forcing the answer tokens are in the input, so a probe there can succeed by reading the label. Depth 0 is the token lookup before any attention or MLP, so whatever it decodes is present by definition and is the right baseline. In ex-2.1.5's hex form it is nearly everything: at the embedding a mix probe scores R² ≈ 1.00 per channel at each answer digit, and an operand probe scores ≈ 0.48 there — the latter is arithmetic (the answer digit is the mean of the two operand digits, so it pins about half of each operand's variance), and both operands echo equally, which distinguishes mixing from a retained operand. The equals sign and the pre-answer space score −0.003 at the embedding in both forms, so they are clean ground; prefer them for cross-form ρ, which answer positions would inflate in both directions.
