---
status: done
tags: [D2.2, anchoring, selectivity, intervention, ex-2.2.21]
opened: 2026-10-02
closed: 2026-10-06
---
# A higher hinge cap

In [ex-2.2.21](/docs/m2/ex-2.2.21/report.py) the hinge arm (the whole-line pull, counting an alignment of 0.8 as full and pulling no further) met both suppression criteria at every position, and the uncapped whole-line arm missed selectivity narrowly (worst other op 0.026 against a gate of 0.02). The cap of 0.8 was set in ex-2.2.16 without a search. A cap of 0.9 or 0.95 sits between the two and might keep the selectivity while holding more of the anchor, which would give a larger dose range to edit over. A small bracket (0.8, 0.9, 0.95, and uncapped) at a few seeds would show whether selectivity falls off smoothly with the cap.

## Notes

**2026-10-04, PM** — Drafted as part of the ex-2.2.22 preregistration in [PR #246](https://github.com/z0u/sca2/pull/246) (not yet frozen): hinge arms at caps of 0.9 and 0.95, at 3 seeds paired with the ex-2.2.21 hinge and whole-line arms. Shortlisted while that scout is the next step for D2.2 round 3. Close or settle this from its results.

**2026-10-06, Claude** — Settled by [ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (E2). The selectivity does not fall off steadily with the cap: runs past the gate turn up at 0.8, 0.9, and 0.95, each from a single run, and the uncapped condition stays within the gate at its three paired seeds (though not on its seed mean over five in ex-2.2.21). The decision there drops the cap from the recipe, to be confirmed at fresh seeds.
