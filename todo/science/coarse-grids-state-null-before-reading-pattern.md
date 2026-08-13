---
status: finding
tags: [metrics, task-grammar, ex-2.1.4]
opened: 2026-07-21
---
# On coarse grids, state the null before reading a pattern as behavior

Two overclaims found in ex-2.1.4's `v27` analysis and corrected, both from a reference that was too weak for a 27-name vocabulary.
(1) "The model often hands back an operand" read as an operand echo. But closure forces each channel of a training pair to agree or to hold both end levels, so an operand is one grid level from the mix by construction — a member of the mix's one-step shell, and that shell holds only 4–6 names. Uniform choice within it returns an operand 40% of the time against 53% observed. (2) "Guesses are far from random" measured against `chance_dist`, a uniform-random name. Mixes cluster toward the cube's centre, so a prompt-blind model that always answers the training answers' centroid scores 0.48 on v27 open pairs where chance is 0.82 and the floor is 0.29 — it eats most of the apparent headroom, and on v27 held-out pairs it matches the model outright (0.57 vs 0.55). Rules of thumb for the anchored runs: a mean-distance metric needs the prompt-blind constant beside it, not just chance; prompt-dependent counts (nearest-name rate, shell membership) separate model from baseline where distance means cannot; and check whether a "striking" coincidence is forced by the grid's combinatorics before attributing it to the model. The references now live in `src/sca/baselines.py` (`blind_index`, `shell_mask`, `neighborhood_exact_null`, `operand_shell_null`, `k_nearest_stats`, `self_nearest_rate`), so all four D2.1 reports compute them the same way.
