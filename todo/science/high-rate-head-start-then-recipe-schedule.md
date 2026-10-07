---
status: done
tags: [D2.2, in-context, training, ex-2.2.19]
opened: 2026-10-01
closed: 2026-10-04
---
# A high-rate head start, then the recipe schedule

A possible fast-follow to [ex-2.2.19](/docs/m2/ex-2.2.19/report.py), from Sandy's review of its results. At the scout seed, the 50-epoch run at peak rate 0.00562 reached a higher skill by its end than any 400-epoch run had at epoch 50. That suggests a two-stage schedule: train 50 epochs at the higher rate with its own warmup and full cosine anneal, then continue from those weights with the recipe schedule (warmup and cosine at peak 0.00316) for 150 epochs. The total is 200 epochs, the length ex-2.2.19 adopted, so it compares with those runs at matched cost.

Sandy expects something like the 200-epoch runs, starting from a higher base, and estimates from the skill curves (E1) a head start of about 25 epochs on the HSV-channel ops. In ex-2.2.19 those ops first passed half skill between epochs 80 and 112 at both 200 and 400 epochs, so the second stage would need to reach that point sooner than the plain 200-epoch run does for the head start to show.

Open design questions: whether the second stage restarts the optimizer state or carries it over; how many seeds (the 200-epoch condition of ex-2.2.19 has four, model seeds 600 to 603, which a paired comparison could reuse); and what to call a pass, likely the paired shortfall from the 400-epoch runs against the 0.015 tolerance of ex-2.2.19, which the plain 200-epoch runs met only partly (mean 0.0157).

## Notes

**2026-10-01, ex-2.2.20** — Preregistered as [ex-2.2.20](/docs/m2/ex-2.2.20/report.py), which settles the open design questions: one 200-epoch run per seed on a two-cycle schedule, carrying the optimizer state over; model seeds 600 to 603, scored on 601 to 603; and the H1 rule of ex-2.2.19 for a pass.

**2026-10-04, PM** — Done: [ex-2.2.20](/docs/m2/ex-2.2.20/report.py) has its results and Sandy's review (merged in #237 and #241). Neither head-start schedule kept more of the skill than the plain 200-epoch run, since both fall further short on the HSV-channel ops, so the plain 200-epoch schedule stays the recipe.
