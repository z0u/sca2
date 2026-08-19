---
status: partial
tags: [D2.1, anchoring, schedules, superposition]
opened: 2026-07-11
---
# D2.1 kickoff: carry-over lessons and hypothesis queue from ex-2.9.3/2.9.4

Ex-2.9.3 and ex-2.9.4 ([reports](https://z0u.github.io/sca2/ex-2.9.3/), [PR #9](https://github.com/z0u/sca2/pull/9)) closed out the autoencoder-era robustness questions. This distills what they taught us into guidance for D2.1 and beyond, so the lessons survive the jump to transformers. It is program-level rather than a code task; it closes once D2.1's design has consumed or rejected each item, and several of the hypotheses below have D2.1 experiments bearing on them already — this records the starting position rather than the current one.

## The principle to carry over

The autoencoder failures were not "LR too high" but *protection removed while the optimizer is still hot*: every failing seed anchored successfully, then broke during the high-LR plateau after the timed anneal ended. Halving the peak LR fixed it there because the autoencoder was over-provisioned; a transformer will charge real loss for that. The transferable rule: order the schedule so regularizer protection outlives the heat — anneal after the LR decay is mostly done, or do not fully anneal at all.

Ex-2.9.3 compared anneal-to-zero against hold-at-full-strength only. Anneal-to-a-small-floor (say 10% of peak weight) is untested and probably the cheapest robustness knob available in a transformer; it should keep the tail control without full hold's capacity cost (leak up, ~10× recon).

## Watch-outs that are new in transformers

**Superposition pressure.** The autoencoder had 5 dims for ~3 concepts, so reserving an axis was nearly free. A transformer actively compresses, so the anti-subspace term now competes with the training objective for capacity. Watch D2.1's completion accuracy against the un-anchored baseline as a function of regularizer strength; that is where the cost shows first, and where a leaked axis will hide (the model may prefer the leak penalty to the capacity penalty).

**Bounds become layer-local.** The ¼ damage bound came from decoder geometry, one linear readout from the intervention. An ablation at layer ℓ propagates through every later block; the geometry bounds the immediate write, not the downstream compounding. Frame D2.1's bound claims as per-layer-local from the start.

**γ calibration is a prerequisite.** The fixed-γ redirect silently no-oped on ~1 run in 250 in the autoencoder — see [calibrate the redirect's γ against the model's pre-norm activation scale](./calibrate-redirect-s-against-model-s-pre.md). Residual norms grow with depth and vary per token, so a fixed magnitude will be miscalibrated somewhere. Scale γ to the measured pre-intervention norm at the anchored layer by default.

**Labeled-signal sparsity.** Failures followed the data stream rather than the init, and the anchor term fired on only ~6% of batches. Label-balanced batch sampling is a cheap variance reduction aimed at the actual cause.

## Hypothesis queue

Roughly in order; the first folds into the first anchoring run rather than needing its own experiment.

1. **The late-instability mechanism reappears.** Record per-step probe trajectories on D2.1's training run (the ex-2.9.3 apparatus carries over). This tells us whether the schedule lessons transfer before we tune anything.
2. **Anneal-to-floor vs anneal-to-zero vs anneal-after-decay**: a small factorial testing the ordering principle above.
3. **The fallback generalizes.** It was the strongest stabilizer found: 0 catastrophic failures in 448 runs with it, 12/352 without. The transformer analog (pin the readout's response to the anti-anchor direction to a neutral output) should be built deliberately rather than as an afterthought. If it transfers, it is the headline recipe component.
4. **Stream-vs-init attribution, once.** The autoencoder answer was "stream". If transformer failures follow the init instead, that changes the screening strategy and is worth knowing early.
5. **Anchor-layer sweeps run on a frozen schedule** (fallback + calibrated γ + anneal-outlives-heat + endpoint screening), so layer effects are not confounded with schedule fragility.

## What not to reach for

Leave the feedback controller alone, even though the setting is harder now. Ex-2.9.4's bottleneck was the sensor — noisy labels cannot distinguish drifted-red from pink — and that ambiguity gets worse in a transformer. If feedback earns another look, the holdout-probe sensor variant is the version to try, and only after the static stack demonstrably fails.

## Grounding

- Reports: [ex-2.9.3](https://z0u.github.io/sca2/ex-2.9.3/) (failure timing, init × stream attribution, schedule sweep), [ex-2.9.4](https://z0u.github.io/sca2/ex-2.9.4/) (closed-loop weights; a clean negative).
- Code: `docs/m1/ex-2.9.3/experiment.py`, `docs/m1/ex-2.9.4/experiment.py` (the in-scan controller and the trajectory recording both live here).
- Deliverables context: D2.1–D2.4 in the [README](../../README.md).
- Migrated from [sca2#10](https://github.com/z0u/sca2/issues/10), which this file replaces. The autoencoder-era name for the redirect knob was β; it is γ here, matching the rest of the backlog.

## Notes

**2026-08-19, housekeeping** — the pass the 08-16 note asked for, against the five-item hypothesis queue. Consumed: **1** (ex-2.1.6 records per-step probe trajectories, and its H4 failure is a slide rather than ex-2.9.3's late collapse, so the mechanism did not reappear) and **2** (ex-2.1.8 swept `anneal_end` × `hold_ratio` — anneal-to-floor is the hold ratio — and ex-2.1.11 ablated the schedules against constants). Not consumed: **3**, the fallback-control analog — "fallback" appears nowhere in the m2 reports, and ex-2.1.6 says outright that it leaves the *anti-anchor* term out; **4**, stream-vs-init attribution, which no D2.1 report runs, though it may be moot given there were no catastrophic anchoring failures to attribute; **5**, the anchor-layer sweep — the anchor pulls all five layers equally throughout D2.1, and a layer sweep is now inside D2.2's own scope per the README. So status is `partial`, and what is left is 3–5 as D2.2 design input rather than D2.1 leftovers.
