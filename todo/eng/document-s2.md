---
status: done
tags: [skills, vis]
closed: 2026-08-27
---
# Document s₂ (surprise-surprise) in a skill

Describe surprise-surprise in a skill: what it is, why we might use it instead of surprisal, and how to calculate it. The mean s_2 over a sequence would be analogous to perplexity, and probably more informative, since it captures the per-token difference from what the model anticipated. Negative values of s_2 are rare and probably uninformative; they suggest the model finds the token unsurprising.

## Notes

**2026-08-27, tech debt** — Landed in `style-terms` under "Model and measurement terms", beside `R²` vs `r²`, rather than in `style-fig`: it is a metric a report author looks up while writing prose, and the figure skill's audience is drawing. `sca.compute.evaluation.answer_calibration`'s docstring already covered the ≈ 0 and ≫ 0 readings, so the entry adds the sign scale, the calibration-not-competence caveat, and one correction.

Two things worth recording from checking the readings numerically before writing them down.

**Negative is under-confidence, not "uninformative".** The item above guessed negative values were rare and meaningless; they are neither. Negative means the actual character was less surprising than the model's own average uncertainty — it hedged more than the outcome needed. A hedging-but-correct prediction (0.2 on the true character, the rest spread over 256) reads −0.60, which is a large signal, and a confidently-correct one reads −0.02. So a competent model spends most of its characters slightly negative, and the mean is carried by the positive spikes where it missed.

**It is not a KL divergence.** The shape invites the reading, and it is wrong: s₂ estimates $H(p,q) - H(q)$, while a divergence from the data is $H(p,q) - H(p)$. Against a near-deterministic truth the two disagree in _sign_ (−0.77 against KL = +0.11 on the case checked), so the mis-reading is not a rounding matter. This is the kind of thing worth stating in the skill because it is exactly what a reader reconstructs from the formula and gets backwards.
