---
status: open
tags: [methodology, anchoring, ex-2.1.7]
---
# How compactly is the color code stored, and does that bound intervention headroom?

Prototyped after ex-2.1.7 and not run at scale. Fit the redness probe on all 64 directions, delete whichever surviving direction it leans on hardest (by $|w_d|\sigma_d$, not raw weight), refit, repeat; the curve is held-out R² against k removed, with the greedy order chosen inside each training fold so the scores stay honest. One seed at op1's last layer gave k50 = 25 for the control, 23 for `op1-anti`, 13 for `span-bare` — against 2 for a simulated M1-shaped ideal where color occupies 4 directions and the anchor holds redness outright. So the trained model spreads color over ~25 directions where M1's bottleneck used 4, and anchoring did not compact it; `span-bare`'s low value is its cube compression, not concentration.

Two reasons this is worth finishing. It is a measurement of the color code's compactness, which is what an intervention has to contend with and which nothing else in M2 reports. And the by-product is cheap: the anchor coordinate's rank in the drop order is 0 at every layer for every anchored condition except `span-bare` (which decays to 5-6 at the last layer), against 8-47 or unranked for the control — a clean binary detector, though it saturates and so cannot order the anchored conditions.

Prototype: `scratchpad/concentration.py` from the 2026-08-03 session (not committed; rewrite from this description). Needs all 3 seeds before any k50 comparison is quotable — the 23-against-25 gap is well inside plausible seed noise.

## Notes

**2026-08-24, housekeeping** — Before rewriting the prototype, look at ex-2.1.12 on [PR #99](https://github.com/z0u/sca2/pull/99) (see [[land-ex-2-1-12-and-the-d21-figure-report]]). It fits ridge probes for each operand's RGB at every (slice, position) site of the D2.1 checkpoints, so the load-checkpoints → capture-slices → fit-per-fold path this item needs is already written and published there. Different question — decodability per site rather than compactness per direction — but the same scaffolding, and it runs in minutes on these models. Its headline is adjacent too: the bare anchor drops last-slice op2 decodability to ≈ 0.03 against the control's 0.61, with the anti-subspace term restoring it to 0.32 and the full recipe to 0.60.
