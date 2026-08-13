---
status: open
tags: [D2.1, ex-2.1.3, representations, metrics]
---
# Does the sub-cell embedding precision hold up at depth, and what sets the `v4096` floor?

Ex-2.1.3's cross-validated embedding probe places tokens 0.73 / 0.63 / 0.49 grid cells from their true color at v27 / v64 / v216 — under one cell throughout, so nearest-name decoding survives even at v27 where R² is only 0.66. In absolute terms that error shrinks with the grid (0.363 → 0.210 → 0.097 of the unit cube), so precision is relative, not a fixed resolution the finer grids keep exposing. v4096 breaks it both ways: 0.166 absolute (worse than v216) and 2.5 cells. Two follow-ups. (a) Is the v4096 floor capacity or optimization? Widening the stream or training longer separates them, and it's the same wall the accuracy plateau hits — pairs with the existing one-level-precision todo. (b) The same measure at each depth would say whether the mix computation preserves the sub-cell precision the embeddings start with, or loses it. Caveat for both: this is probe error, so it bundles model imprecision with probe misfit, which matters most at v27 (26 fit points, leave-one-out).
