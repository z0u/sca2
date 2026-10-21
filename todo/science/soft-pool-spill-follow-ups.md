---
status: open
tags: [anchoring, D2.2]
opened: 2026-10-21
---
# Keep the spill of a soft pool down

In the [soft-pool experiment](/docs/m2/soft-pool/report.py), a pooling temperature τ_p that starts soft kept the anchor on every seed in every arm that had it, and the soft ramp over slices 1 to 3 did so without the step between slices 1 and 2. Each soft arm spilled more than its fixed twin, though. The trajectories suggest why: while the pool is soft its pull reaches every position from the op word on, and during the rise the other ops come partway onto e₁ (peak α 0.04 to 0.06 against 0.03 at the fixed temperature). That lift has gone by the end of training, but it goes with the spill (r ≈ 0.47 over 40 runs).

Arms that would test it, each beside the soft ramp:

- A soft pool over the answer positions only (example answers and query answer), so the pull is shared among them and nowhere else.
- A stronger anti term while the pool is soft.
- The anti term off on slice 4 for the arms that don't pull there. The anti term acts on every position, the `difference` answers included, and nothing on slice 4 counters it under the ramp. That may be why α falls there (0.46 against 0.64 under the step). The review wondered whether the model's work to meet a balanced pull on slice 3 and an unbalanced anti term on slice 4 could feed the spill indirectly. The arms so far argue against a large effect: the fixed ramp has that imbalance and the least spill of any arm, and the soft ramp spills about as much as the soft step, which has no imbalance. It may cost removal instead, since both ramps remove a few hundredths less than the step.
- A later or slower λ_a warm-up at the fixed temperature, which separates the soft start from a plain delay. The review also wondered about a faster one.

Seed count matters here too: `falloff` has run only at seeds 704 to 711, so its loss rate is known only to within about 0 to 30%.
