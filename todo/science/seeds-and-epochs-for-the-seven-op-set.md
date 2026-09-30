---
status: open
tags: [D2.2, in-context, training, ex-2.2.18]
opened: 2026-09-29
---
# Seeds and training length for the seven-op set

[Ex-2.2.18](/docs/m2/ex-2.2.18/report.py) scouted smaller op sets at one seed each. Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` (`no-four`) raised the Bayes ceiling from 0.51 to 0.61 and narrowed the gap from 0.074 to 0.044. Dropping `lighten`, `darken`, `hsvmix`, and `exclusion` (`no-four-ld`) left the ceiling near that of the full set and narrowed the gap to 0.055. The three ex-2.2.17 seeds on the full set spanned 0.053 to 0.080 in gap, so the narrower gaps could be seed variation. Sandy, reviewing ex-2.2.18, wants more seeds and also a sweep over training length, as a follow-up experiment of its own, and will design it.

A sketch from the review conversation, for a starting point: op set (`full`, `no-four`, perhaps `no-four-ld`) crossed with epochs (100, 200, 400) at three seeds, 27 runs at about $4 in all, since a 400-epoch run costs about $0.26. `no-multiply` and `no-darken` could ride along as arms at 400 epochs, for the shortfall on the HSV-channel ops. The scout gives something to preregister: at matched epochs, `no-four` has a smaller gap than `full` across seeds, and 200 epochs keeps most of the skill of 400.

One thing to keep in mind: the schedule is cosine, so a shorter run anneals sooner and is not a truncation of a longer one. The skill curves in ex-2.2.18 (E4) can't stand in for the shorter runs, though they suggest room to cut: the last fifth of training added at most 0.014.

Related: `cheaper-center-control-recipe.md` (a wider model at fewer steps) and `drop-ops-with-similar-answers.md`.
