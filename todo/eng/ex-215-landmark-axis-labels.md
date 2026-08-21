---
status: done
tags: [reports, vis]
closed: 2026-08-21
---
# ex-2.1.5's two landmark figures label the x axis differently

The two landmark figures in ex-2.1.5 name the x axis differently: the probe heatmap prints the raw keys (`o1s0`, `ae1`, …) rotated 90°, while the trace grid below it uses the math labels from `sca.vis_probes.LANDMARK_LABELS` ($a_1$, $r_{n-1}$, …). The figures are laid out to be read against each other, so the mismatch costs the reader a translation step. Point the heatmap at `label_landmarks` too — its panels are wide enough for the full set.

## Notes

**2026-08-21, housekeeping** — closing, because the figure this asks to fix is gone. ex-2.1.5's two landmark figures are now `probe-channels` and `cross-form-transfer`, both `vp.*_trace_grid` calls, and both label the x axis through `label_landmarks` — so they already share the `LANDMARK_LABELS` math set the item wanted. Checked the whole report rather than those two: none of its nine `@themed` figures draws a heatmap, `rotation` appears once and on an unrelated panel title, and the only `imshow`/`pcolormesh` left anywhere under `src/` or `docs/` is in ex-2.9.3's report. The three `enumerate(LANDMARKS)` sites in the report are index lookups, not axis labels. So the mismatch was resolved by replacement rather than by the edit proposed here, and there is no heatmap left to point at `label_landmarks`.
