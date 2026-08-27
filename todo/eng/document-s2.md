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

**Negative means under-confidence, and it is common but shallow.** Negative says the actual character was less surprising than the model's own uncertainty — it hedged more than the outcome needed. Measuring it over ex-2.1.1's 2592 captured character positions settles how much that happens: 80% are negative, but the median is −0.001 and only 2.3% fall below −0.15, against a floor of −0.28 and a maximum of +6.2.

So the sign is common and the magnitude is not, and there is a reason for the asymmetry rather than an accident of this task: s₂ ≥ −h/log|V|, because surprisal cannot fall below zero, while the positive side has no ceiling at all. A confident model has small h, so its negative excursions are small by construction; only a genuinely uncertain position can go deep, and then only as deep as its own entropy.

Where they do arise is worth knowing, since it is not noise: 78% of the positions below −0.15 are the first character of a word (right after a space or `=`), where the model is unsure *which* word but the character is shared across its candidates. `black + ` → `b` (black, blue, brown), `lime + black = ` → `g` (green, grey, gold), and in `cross_unseen`, `rose + ` → `#`, where the operand is always a hex code but its digits are not. Character-level tokenization is what makes this a recurring shape rather than a curiosity.

**It is not a KL divergence.** The shape invites the reading, and it is wrong: s₂ estimates $H(p,q) - H(q)$, while a divergence from the data is $H(p,q) - H(p)$. Against a near-deterministic truth the two disagree in _sign_ (−0.77 against KL = +0.11 on the case checked), so the mis-reading is not a rounding matter. This is the kind of thing worth stating in the skill because it is exactly what a reader reconstructs from the formula and gets backwards.
