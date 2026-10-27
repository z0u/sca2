---
status: open
tags: [D2.2, anchoring, schedules, selectivity]
opened: 2026-10-27
---
# A slice pool that starts sharp, or a plain max

In the [soft pool scout](/docs/m2/soft-pool-scout/report.py), how sharp the pool over slices ends decided the early spill of `pool 1–3`, and how soon it got there barely mattered: $\tau_\ell$ held at 0.1 from epoch 0 did as well as sharpening from 4. That leaves two nearby schedules untried. One reverses the usual direction, starting at 0 (a max over slices) and softening to 0.1. The other drops the temperature altogether and pulls only the slice where the anchor is largest.

Both change the first epochs, so they need fresh runs rather than a resume from the stored epoch-64 states. A screen at epoch 64 as in the scout would call the early kind and lost anchors. The joint pool over blocks 1 to 4 at a fixed sharp temperature latched onto the query answer on one seed in [slice-terms](/docs/m2/slice-terms/report.py), so the screen should watch α at the query answer.
