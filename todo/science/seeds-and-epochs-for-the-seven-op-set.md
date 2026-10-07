---
status: done
tags: [D2.2, in-context, training, ex-2.2.18]
opened: 2026-09-29
---
# Seeds and training length for the seven-op set

[Ex-2.2.18](/docs/m2/ex-2.2.18/report.py) scouted smaller op sets at one seed each. Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` (`no-four`) raised the Bayes ceiling from 0.51 to 0.61 and narrowed the gap from 0.074 to 0.044. Dropping `lighten`, `darken`, `hsvmix`, and `exclusion` (`no-four-ld`) left the ceiling near that of the full set and narrowed the gap to 0.055. The three ex-2.2.17 seeds on the full set spanned 0.053 to 0.080 in gap, so the narrower gaps could be seed variation. Sandy, reviewing ex-2.2.18, wants more seeds and also a sweep over training length, as a follow-up experiment of its own, and will design it.

A sketch from the review conversation, for a starting point: op set (`full`, `no-four`, perhaps `no-four-ld`) crossed with epochs (100, 200, 400) at three seeds, 27 runs at about $4 in all, since a 400-epoch run costs about $0.26. `no-multiply` and `no-darken` could ride along as arms at 400 epochs, for the shortfall on the HSV-channel ops. The scout gives something to preregister: at matched epochs, `no-four` has a smaller gap than `full` across seeds, and 200 epochs keeps most of the skill of 400.

One thing to keep in mind: the schedule is cosine, so a shorter run anneals sooner and is not a truncation of a longer one. The skill curves in ex-2.2.18 (E4) can't stand in for the shorter runs, though they suggest room to cut: the last fifth of training added at most 0.014.

Related: `cheaper-center-control-recipe.md` (a wider model at fewer steps) and `drop-ops-with-similar-answers.md`.

## Notes

**2026-09-30, design conversation** — Sandy and I settled on a staged plan that drops `full`, since we will probably drop it anyway. Ex-2.2.17 already has three seeds on `full` at 400 epochs, which is enough for a comparison at that length.

1. Scout, about $0.25: `no-four` at 50, 100, and 200 epochs, at model seed 600. The 400-epoch point is the ex-2.2.18 run, and memoization may reuse it. One seed can show where skill falls away, but the difference we care about (the last fifth of a 400-epoch run added at most 0.014) is smaller than the seed range (about 0.03), so small differences here don't mean much.
2. Preregistered, about $1: two or three fresh seeds at the one or two most promising lengths, plus `no-four` at 400 epochs from the model seeds of ex-2.2.17 `s1` and `s2`, so that each of those runs pairs with a `full` run from the same initialization. The chosen length is judged on the fresh seeds only, because seed 600 helped choose it.
3. Optional: `no-four-ld`, or the single-drop arms, at the chosen length.

A later experiment could then take the chosen recipe to about 20 seeds for a firm measurement. Varying model size (narrower or shallower) belongs in its own experiment, at the length chosen here; see the note on `cheaper-center-control-recipe.md`.

**2026-09-30, results** — [Ex-2.2.19](/docs/m2/ex-2.2.19/report.py) ran the staged plan. The scout picked 200 epochs; at 100 the HSV-channel ops were still far short. On seeds 601-603, 200 epochs fell short of 400 by 0.016 on average (a partial pass against 0.015), with `hue-hsv` just inside its tolerance and calibration unchanged. We adopt 200 epochs. `no-four` had the smaller gap than `full` at all three paired seeds. The ~20-seed measurement and the model-size sweep are still to come, at 200 epochs.
