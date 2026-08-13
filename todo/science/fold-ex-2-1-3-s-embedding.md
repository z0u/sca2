---
status: open
tags: [D2.1, ex-2.1.3, representations, metrics]
---
# Fold ex-2.1.3's embedding-probe fixes back into `experiment.py`

The report now computes both itself from the published `embeddings` array, so the science is banked; the stored `emb_r2` is the stale one. (a) `emb_r2` uses a half/half split, which at v27 fits a 64→3 map from 13 points and understates. `sca.compute.evaluation.ridge_probe_loo` is the drop-in: leave-one-out, exact, about the cost of one fit, and with no split to draw it takes no seed. It gives 0.66 against the stored 0.48 (v64 0.81→0.87, v216 0.95→0.97, v4096 unchanged). (b) Worth storing alongside it: color's share of the top-3 PC variance budget (0.25 / 0.26 / 0.56 / 0.80 across v27→v4096), which is why PCA only finds the cube at the dense end. Deferred because editing the eval step is memoization evidence and re-runs every cell — bundle it with the next change that re-runs ex-2.1.3 anyway. The same treatment extends to the per-depth probes, which is the transferable part.
