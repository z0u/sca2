---
status: done
tags: [D2.1, anchoring, ex-2.1.6]
---
# Does the shape of the anchor anneal matter — linear vs minimum-jerk?

**No, at three seeds.** Ex-2.1.11's `linear` arm straightens every ramp and anneal in both anchor terms and leaves all five decision statistics within band (largest excursion: grading, −0.0248 against a band of 0.0392), so minimum-jerk stays on evidence rather than by inheritance. The arm also moves the anti-subspace dose by 6% with nothing resolving, which bounds how dose-sensitive that term is near the operating point. Original note follows. M1 compared neither; it did test a stepped anneal, which held only when the LR warmup restarted from zero at each step, so the one discontinuity we measured needed an accommodation. Whether the milder discontinuity of a linear ramp (a jump in λ's derivative) costs anything is unknown. Ex-2.1.6 takes minjerk by inheritance. Cheap to settle later as a two-condition arm on whatever anchored experiment is running anyway.
