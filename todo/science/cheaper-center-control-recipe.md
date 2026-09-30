---
status: open
tags: [D2.2, training, performance, ex-2.2.17]
opened: 2026-09-28
priority: high
---
# A cheaper recipe for the center control before building on it

The ex-2.2.17 scout reached about 0.45 held-out expected exact match with the d64-L4 model at eight times ex-2.2.16's length (105,600 steps, masked cosine, peak learning rate 0.00316). If that becomes the recipe for the following D2.2 experiments, every run pays for those steps, so it is worth bringing the cost down first.

Sandy suspects the steps can at least be halved with the wider d128-L4 model and a tuned learning rate. The evidence so far is one seed: d128 made its jump on the HSV channels at about 15k steps against 30k–45k for d64, and led through the first half of training, then ended level (0.459 against the d64 seed range of 0.436–0.464). Since the step is latency-bound, d128 costs the same per step as d64 (see `todo/eng/pack-runs-per-gpu.md`), so a shorter d128 run would be cheaper outright. One caution: its calibration KL was 0.95 against about 0.5 for d64, with a lower training loss, so a shorter run should be checked for calibration as well as accuracy.

A first sweep: d128 at two, three, and four times the length, with a short learning-rate sweep around 0.00316 at one seed, then the best condition at three seeds. Packing seeds would cut this cost further.

## Notes


**2026-09-29, ex-2.2.18 session** — [Ex-2.2.18](/docs/m2/ex-2.2.18/report.py) (E4) logged skill on the probe set through training for six op sets on this recipe. Every run passed 90% of its final skill between steps 65k and 78k of 105.6k, and the last fifth added 0.007 to 0.014 probe EEM. The cosine anneals to the end, so a shorter run anneals sooner and would likely keep most of that. A training run cost about $0.26 of L4 time and took 14 to 17 minutes, about 7,000 steps a minute. The second round (`no-lighten`, `no-darken`, `no-four-ld`) agrees: 90% of final skill between 70k and 76k steps.

**2026-09-29, housekeeping** — On the shortlist, since the center control recipe is what the anchored runs of pivot step 3 would build on. One premise has moved: "d128 costs the same per step as d64" held for the per-step loop, which was latency-bound. Since #228, `train_anchored` runs sixteen steps per dispatch and a d64 step costs 5–8 ms (see [`training-step-is-host-bound`](/todo/eng/training-step-is-host-bound.md)), so the GPU now does real work and d128 may cost more per step than d64. Time a d128 step on the scanned loop before assuming a shorter d128 run is cheaper outright. Ex-2.2.18 ([#230](https://github.com/z0u/sca2/pull/230), open) changes the op set, so the sweep here may want to wait for it.
