---
status: open
tags: [metrics, ex-2.1.5, representations]
---
# Depth-crossed cross-form transfer

ρ as preregistered compares the same (depth, landmark) in both forms, but ex-2.1.5's two forms compute the mix in different places — named at the pre-answer position in the last layer, hex mid-answer a layer earlier, with hex's pre-answer column never clearing 0.20 at any depth. So a same-cell ρ of 0 may be scoring a mature representation against one that hasn't formed. The subspace half is already answered report-side from the stored weights (first principal angle 66–78° for every depth pair at `pre`; 57° at the closest pair anywhere both forms clear 0.3), so there is no shared direction to find at any depth — but a zero-shot fit is a different test from a subspace angle, and only the fit needs activations the sweep doesn't publish. Cheap to add to `eval_one`: `transfer_maps` already has both forms' activations in hand, so a depth-crossed variant is a second pair of loops. Ride it along with the fold items above.
