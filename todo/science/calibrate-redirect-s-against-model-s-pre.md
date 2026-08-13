---
status: finding
tags: [anchoring, ex-2.9.3]
---
# Calibrate the redirect's γ against the model's pre-norm activation scale

Calibrate the redirect's γ against the model's pre-norm activation scale instead of the fixed γ = 1. Ex-2.9.3 found the fixed value silently no-ops on ~1 run in 250 (the bias fails to dominate that seed's pre-norm residual, so "deleted" red passes through nearly untouched); ex-2.9.2 saw the same once.

Cheap fix: set γ to a multiple of the ablated row's typical pre-norm contribution, measured on the train set after training.
