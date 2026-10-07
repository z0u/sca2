---
status: done
tags: [D2.2, training, performance, ex-2.2.17]
opened: 2026-09-28
closed: 2026-10-01
---
# A cheaper recipe for the center control before building on it

The center control recipe is now d64-L4 on the seven-op set (`no-four`) for 200 epochs (52,800 steps, masked cosine, peak learning rate 0.00316), which [ex-2.2.19](/docs/m2/ex-2.2.19/report.py) adopted and [ex-2.2.20](/docs/m2/ex-2.2.20/report.py) kept against a two-cycle schedule. A run costs about $0.13 of L4 time. Every following D2.2 experiment pays that per run, so this item asks whether another model size can bring it down without losing skill.

We suspect a different size could do the same work for less. Ex-2.2.17 is the evidence for a wider model, at one seed: d128-L4 made its jump on the HSV channels at about 15k steps against 30k–45k for d64, led through the first half of training, then ended level (0.459 against the d64 seed range of 0.436–0.464). Its calibration KL was 0.95 against about 0.5 for d64, with a lower training loss, so any shorter run needs a calibration check beside the accuracy check. A narrower or shallower model might instead keep the skill at the same length for less per step. The step timings below set what each route has to achieve, and the design note lays out a sweep over both.

## Design note

Drafted 2026-09-30 by the science desk; the choices at the end were settled in review on 2026-10-01. It builds on the 200-epoch `no-four` recipe and the step timings in the note of 2026-09-30 below.

The question: at that recipe, how does held-out expected exact match (EEM) change with the width and depth of the model, and is there a size that costs less per run than d64-L4 and keeps its skill? The timings split this into two routes to a cheaper run. A smaller model costs less per step (d32-L2 about 0.4 times d64-L4), so it would need to keep the skill at the same length. d128-L4 costs about 1.5 times as much per step, so it saves money only if it needs fewer than two thirds of the steps; at 100 epochs a run would cost about three quarters of the d64-L4 run at 200.

Candidate hypotheses, each scored on held-out EEM of `no-four` and compared with d64-L4 at 200 epochs:

- H1, a smaller model keeps the skill. Some size cheaper per run than d64-L4 falls short of it by at most 0.015 in seed-mean EEM, with no op beyond its ex-2.2.19 tolerance and calibration KL no higher than the d64-L4 seed range. 0.015 is the gate ex-2.2.19 used; a shortfall up to 0.03 would be a partial pass.
- H2, width buys steps. d128-L4 at 100 epochs meets the same gate. Ex-2.2.17 is the reason to expect it: at one seed d128 made its jump on the HSV-channel ops at about 15k steps against 30k to 45k for d64, and the HSV-channel ops are the ones that fell furthest short at 100 epochs in ex-2.2.19.
- Depth against width is exploratory: L2 and L4 at each width, with no gate.

Sizes differ in their parameter shapes, so no run can share an initialization with the d64-L4 reference, and each comparison is between seed means. With a seed range of about 0.03 on this task, three seeds a condition resolve a difference of about 0.02 at best. That is enough to rule out a size that falls well short, and too little to confirm one that matches to within 0.015. I think the fair scope for this experiment is to pick a size, with the firm measurement left to the roughly 20-seed run that is planned.

Conditions, in two stages like ex-2.2.19:

1. A scout at one seed: d32-L2, d32-L4, d64-L2, and d128-L2 at 200 epochs beside the existing d64-L4 run, d128-L4 at 100 and 200 epochs, and d32-L2 and d32-L4 at 400 epochs, the longer arm Sandy asked for to check that the small models are not short of steps. The d32 conditions also train at a second peak rate (below). About $1 in all, scaled from ex-2.2.18 (about $0.13 for d64-L4 at 200 epochs).
2. Three fresh seeds at the one or two sizes the scout favors, and at d64-L4, all at the chosen length. About $1.

Choices, as settled in review:

- The peak learning rate. The scout holds it at 0.00316 for every size. Smaller models often do better at a higher rate, so a small model that falls short might be short of rate rather than of capacity. So the d32 sizes get a second rate (0.00562) in the scout, which adds about $0.13.
- The head layout at d32. `ModelConfig` needs heads of at least eight dimensions, so d32 gets four heads of eight in place of eight heads of four. The alternative, eight heads of eight, gives an attention width of 64 on a stream of 32. We chose four heads, since it keeps the attention width equal to the stream width as at the other sizes.
- The residual step is 1/L, so an L2 model takes steps twice as large per block. That is part of what depth means in this architecture, so it stays as it is.

## Notes

**2026-09-29, ex-2.2.18 session** — [Ex-2.2.18](/docs/m2/ex-2.2.18/report.py) (E4) logged skill on the probe set through training for six op sets on this recipe. Every run passed 90% of its final skill between steps 65k and 78k of 105.6k, and the last fifth added 0.007 to 0.014 probe EEM. The cosine anneals to the end, so a shorter run anneals sooner and would likely keep most of that. A training run cost about $0.26 of L4 time and took 14 to 17 minutes, about 7,000 steps a minute. The second round (`no-lighten`, `no-darken`, `no-four-ld`) agrees: 90% of final skill between 70k and 76k steps.

**2026-09-29, housekeeping** — On the shortlist, since the center control recipe is what the anchored runs of pivot step 3 would build on. One premise has moved: "d128 costs the same per step as d64" held for the per-step loop, which was latency-bound. Since #228, `train_anchored` runs sixteen steps per dispatch and a d64 step costs 5–8 ms (see [`training-step-is-host-bound`](/todo/eng/training-step-is-host-bound.md)), so the GPU now does real work and d128 may cost more per step than d64. The 2026-09-30 timings below confirm it.

**2026-09-30, design conversation** — Sandy would also like to try narrower and shallower models. We agreed to keep that out of `seeds-and-epochs-for-the-seven-op-set.md`: model size likely changes how many steps a run needs, so crossing it with length would multiply the conditions. The plan is to settle the op set and length there, then vary size (d32, d64, d128; L2, L4) at that length here, with one longer arm to check that the small models aren't just short of steps. Time a step at each size on the scanned loop first, since per-step cost may now differ between sizes.

**2026-09-30, science desk: step timings** — A probe (`docs/m2/size-timing/experiment.py`, dev profile) trained each of six sizes for 2,112 steps on the `no-four` corpus condition and recipe, on three L4 containers, every size in each container, and took the steady-state time per step past the compile, with validation and batch sampling included. Milliseconds per step, and the ratio to d64-L4 in the same container:

| size | parameters | ms per step | ratio to d64-L4 |
| --- | --- | --- | --- |
| d32-L2 | 41k | 3.0–3.9 | 0.38–0.43 |
| d32-L4 | 66k | 4.4–5.4 | 0.55–0.60 |
| d64-L2 | 131k | 4.9–5.9 | 0.62–0.65 |
| d64-L4 | 229k | 7.9–9.1 | 1 |
| d128-L2 | 459k | 7.4–8.4 | 0.92–0.94 |
| d128-L4 | 852k | 12.3–13.7 | 1.50–1.56 |

So the premise of this item no longer holds on the scanned loop: a d128-L4 step costs about 1.5 times a d64-L4 step, and a shorter d128 run is cheaper only below about two thirds of the steps. Each size also paid 10 to 22 seconds of compile and start-up, whatever its length. The ratios agree closely across containers, though one container was 10 to 25% faster at every size. The probe left out the trajectory EEM measurement, which runs about fifty times per run. The probe cost under $0.25. The design note above builds on these numbers.

**2026-10-01, closed** — Closed without running the sweep. A run of the recipe costs about $0.13, so the cheapest size (d32-L2, about 0.4 times the step cost) would save about 8 cents a run, and a change of size would need a fresh control for the anchored runs of pivot step 3, which build on d64-L4 like every control from ex-2.2.14 on. Ex-2.2.20 kept the plain schedule, so the center control recipe stays as ex-2.2.19 adopted it. The design note and the step timings stay here for when model size comes up again as a question in its own right (for instance [`narrow-stream-raise-superposition-pressure-sequenced-up`](/todo/science/narrow-stream-raise-superposition-pressure-sequenced-up.md)).
