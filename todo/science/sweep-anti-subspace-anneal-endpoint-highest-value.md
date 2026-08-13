---
status: partial
tags: [ex-2.1.7, schedules, anchoring]
---
# Sweep the anti-subspace anneal endpoint

Sweep the anti-subspace anneal endpoint, the highest-value open knob after ex-2.1.7. **Preregistered as ex-2.1.8** — a 3 × 2 grid of (`anneal_end` ∈ {50, 70, 90}) × (`hold_ratio` ∈ {0.03, 0.30}) plus a dose-matched arm.

Original note follows.

The `span-anti-late` arm changed exactly one number, the epoch at which the repulsive term finishes annealing to its hold ratio (50 → 90), and improved margin (0.46 → 0.60), containment (ᾱ 0.33 → 0.13), retention (0.73 → 0.94 as means; see the item above) and grading (R² 0.62 → 0.78) together — a bigger margin gain than adding the term was worth in the first place. The M1 keyframes were inherited from a 5-d autoencoder bottleneck and mapped onto our 100 epochs by fraction of training, so there was never a reason to expect them to be right for a 64-d residual stream. Two things the single arm cannot separate: when the repulsion acts and how much of it there is, since holding near peak for longer also delivers more of it. A sweep over (`anneal_end`, `hold_ratio`) on a grid separates them; ex-2.4.1's ramp-up schedule is worth including as a third shape. Cheap: training only, no new measurements.

Found while designing ex-2.1.8, and worth keeping whatever that run says: **`hold_ratio` is not a dose knob.** Integrating ∫ λ_s̄(e)·lr(e) de over training, a 10× change in the hold ratio moves the delivered repulsion by about a sixth, while stepping the endpoint 50 → 90 nearly doubles it. So in this schedule family the endpoint *is* the dose axis, and the two cannot be crossed as factors: dose-matching endpoint 50 up to endpoint 90 would need λ_s̄/λ_a ≈ 1.18, sustaining the repulsion above the anchor weight for half of training. The achievable match runs the other way, scaling the opening ratio down (2.5 → 1.443) on the late schedule, which is what ex-2.1.8's dose arm does. The `anti_dose` helper in `docs/m2/ex-2.1.8/experiment.py` computes it.
