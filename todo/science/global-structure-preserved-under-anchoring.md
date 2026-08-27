---
status: open
tags: [probes, anchoring, representations, geometry]
opened: 2026-08-27
---
# Does anchoring leave the rest of latent space alone? A test that could actually resolve

The d2.1 post wants to say that anchoring *red* left the global structure of latent space intact. What we have is weaker than that claim. Ex-2.1.12 H2 (linear RGB decodability, anchored vs. control) came back *not resolved*: the R channel misses the gate by a small margin that sits inside the seed band, G and B read above the control, and the control's three seeds disagree with each other by enough that only a deficit of about half an R² unit could have been caught. The containment statistic (ex-2.1.9/10) shows non-reds are as unaligned to the anchor as in un-anchored models, which is a local statement about one axis rather than about the whole geometry.

Two things would make the claim defensible:

1. **More control seeds.** The comparison is seed-noise limited and the control carries more than half of the variance with only two scored seeds. Nine control seeds, matching the primary, is the cheapest fix and reuses ex-2.1.12's protocol unchanged.
2. **A whole-geometry statistic rather than per-channel R².** Compare the anchored and control embedding geometries directly: e.g. Procrustes distance or RSA (representational similarity analysis: correlate the pairwise-distance matrices of the 216 color states across models) between each anchored run and each control run, against the control-vs-control baseline. That asks "is the anchored geometry within the spread of un-anchored geometries, apart from the anchored axis?", which is the actual claim. Projecting out the anchor direction first would isolate the *rest* of the space.

Until one of these is run, the post should say something like: linear probes decode RGB about as well from anchored models as from un-anchored ones, though that comparison is noisy and could only have caught a large loss.
