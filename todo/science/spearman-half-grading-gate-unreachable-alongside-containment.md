---
status: done
tags: [ex-2.1.8, measurement, reports]
---
# The Spearman half of the grading gate is unreachable alongside containment

so a report that scores both is asking for two things at once. **Decided in ex-2.1.9's preregistration:** the ρ track is dropped as a gate from that experiment on, and the R² track is scored *relative* to the in-experiment span-mean arm (drop ≤ 0.1) rather than against an absolute bar, since ex-2.1.8 left where a realistic absolute R² gate sits to a later grid. The M1-shaped group contrast stays available as the candidate for a future absolute gate. Original note follows. Found in ex-2.1.8's exploratory section, which computes the ceiling: for a response that reproduces `sim^1.5` exactly above a knee and sits at the control below it, ρ against `redness` reaches 0.82 with no containment at all (because `sim^1.5` is not monotone in `redness` — they correlate at 0.90) and 0.59 once the 161 colors below redness 0.4 are collapsed together. The gate is 0.8. The R² track does not have this problem: its target is near zero exactly where containment acts, so its ceiling is 0.94 at the same knee, and the 0.78 the best cell reaches is a real shortfall rather than an artifact.

M1 never asked for this. `sca.colorcube` scores `red` and `collateral` as two groups (404 of 512 grid points are collateral, close to our 161 of 216), so the large flat block is a property of the color grid rather than of the residual stream, and M1 handled it with a group contrast instead of a rank statistic over the whole cube. Options for the next design: drop the ρ track, restrict it to colors above the knee, or replace both with a group contrast in M1's shape. Needs a decision before the next report scores grading.
