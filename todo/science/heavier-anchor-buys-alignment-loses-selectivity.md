---
status: done
tags: [ex-2.1.7, anchoring]
closed: 2026-08-14
---
# A heavier anchor buys alignment and loses selectivity

Folded into ex-2.1.11's λ_a dimension ([0.02, 1.0], log). Original note follows. Ex-2.1.7's ceiling arm ran the full recipe at λ_a = 1, ten times the scoring rung. The task never pushed back (holdout EM within 0.0013 of control), so the task ceiling is still unfound — but the margin fell to 0.43 against 0.46 at a tenth of the weight, with ᾱ back up to 0.46, retention down to 0.58, and the color cube compressed even in the token embedding (0.77 against 0.91). So there is a selectivity optimum in λ_a somewhere below 1, and the useful sweep is the one that finds it rather than the one that finds where the task breaks. Worth doing before D2.2 fixes an operating point. Note this also answers ex-2.1.6's open question about a `separate`-style term: the compression it would address appears only at weights past the selectivity optimum, and the anti-subspace term restores the cube's extent at the scoring rung (91–103% of control against the bare pull's 67%).
