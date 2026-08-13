---
status: finding
tags: [metrics, ex-2.1.5, anchoring, representations]
opened: 2026-07-26
---
# Per-equation leave-one-out lets probes memorize identity → value; per-value holdout separates memory from geometry

Refit of all three centre seeds, locally from published checkpoints — no Modal needed, and the grouped fit is cheaper than the shipped one because it shares a Gram matrix across groups. Three protocols: *equation* (shipped), *value* (hold out rows whose scored channel carries the value), *strict* (hold out rows where the value appears in any slot — operand 1, operand 2 or the answer — which closes the path where the same hex digit, or the same color in the other operand slot, teaches its own direction from a slot the value holdout left alone). Refit *equation* reproduces the shipped arrays to 4e-4, so the pipeline is faithful. Headline cells, seed-averaged:

| site | equation | value | strict |
|---|---|---|---|
| hex op2, own digit, embedding | 1.000 | 0.416 | 0.416 |
| hex op2, own digit, last layer | 0.982 | 0.781 | 0.781 |
| hex op2, red retained at the green digit, L4 | 0.821 | −0.285 | −0.269 |
| hex mix, pre-answer, L4 | 0.376 | 0.325 | 0.196 |
| named op1 / op2, last operand char, L4 | 0.750 / 0.716 | −0.046 / 0.006 | −0.047 / 0.001 |
| named mix, pre-answer, L4 | 0.944 | 0.943 | 0.941 |

Cross-slot leak (value → strict) is exactly zero at the embedding, where no attention has run — the control behaving as theory demands. It stays negligible for named (mean +0.003 on op2) and concentrates in the hex mix (mean +0.10, +0.13 at pre-answer), because the 16 digits are shared across all three channel slots. Four outcomes:
(1) hex digit cells at the embedding collapse 1.00 → 0.44, but land well above zero, so the digit tokens do carry a real (partial) magnitude axis rather than an arbitrary lookup;
(2) the same hex cell at the last layer holds 0.98 → 0.76, so depth improves the value geometry instead of merely relaying the token;
(3) named operand readout falls 0.75 → 0.19 — the "holistic operand bundle" is mostly name-identity recovery, which fits ex-2.1.4's finding that value → name translation is the blocker and that named holdout accuracy is weak;
(4) the named mix at the pre-answer site is untouched, 0.944 → 0.943, so the headline H2 result is geometry and survives the stricter test. Net: the two forms carry value structure in different places — hex in its digit tokens (deepening with depth), named only in the computed mix. That, rather than token binding, is what a cross-form anchor would have to bridge.
