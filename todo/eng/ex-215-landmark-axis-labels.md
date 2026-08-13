---
status: open
tags: [reports, vis]
---
# ex-2.1.5's two landmark figures label the x axis differently

The two landmark figures in ex-2.1.5 name the x axis differently: the probe heatmap prints the raw keys (`o1s0`, `ae1`, …) rotated 90°, while the trace grid below it uses the math labels from `sca.vis_probes.LANDMARK_LABELS` ($a_1$, $r_{n-1}$, …). The figures are laid out to be read against each other, so the mismatch costs the reader a translation step. Point the heatmap at `label_landmarks` too — its panels are wide enough for the full set.
