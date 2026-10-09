---
status: open
tags: [D2.2, anchoring, in-context, ex-2.2.21]
opened: 2026-10-02
priority: high
---
# Leave slices out of the pull, at more seeds and longer training

In [ex-2.2.21](/docs/m2/ex-2.2.21/report.py) the `no-emb` arm (the whole-line pull with the embedding slice left out) had the highest held-out expected exact match of any arm, about 0.56, with its three seeds within 0.005 of each other. Its gain over the whole-line arm is inside the seed band, so three seeds cannot say whether it is real. Paired by model seed, though, there is a hint: on seed 702 every anchored arm that pulls the embedding slice took the slow path through training (EEM 0.43 to 0.54), while `no-emb` and the control did not (0.563 each). If pulling the embedding slice is what sends some seeds down the slow path, leaving it out would buy reliability more than peak score.

The reasoning for going further, from the review: an abstract concept like an op is not the property of any one token, so it probably exists only in the middle of the stack. That suggests a family of restrictions:

- no embedding slice (`no-emb`, as now);
- no last slice (the input to the readout), which [anchor-early-slices-only](./anchor-early-slices-only-tied-alternative.md) proposed for a different reason;
- both, leaving the middle blocks only.

Run them beside the whole-line arm and the control at more seeds (perhaps eight), at 200 epochs and again at 300 or 400, since the slow path shows up as a plateau that a longer run gives time to leave (ex-2.2.21's training traces). Score the task, the op margin, and the suppression pass, so the answer covers editability too.

Post hoc, the suppression pass found that `no-emb` edits less selectively than the whole-line arm. One reading is that the restriction itself makes the early blocks hold the axis less cleanly. Another, from the review, is that the anchor weight λa was set with the embedding slice in the pull, so a restricted pull may want a lower one. Crossing each restriction with a lower λa (a bracket of two or three levels) would separate the two.

## Notes

**2026-10-04, PM** — Drafted as part of the ex-2.2.22 preregistration in [PR #246](https://github.com/z0u/sca2/pull/246) (not yet frozen): the `no-last` and `middle` slice sets, and a twin for each slice set (`no-emb` included) at a weight that keeps the per-slice pull unchanged, at 3 seeds paired with ex-2.2.21. Shortlisted while that scout is the next step for D2.2 round 3. Close or settle this from its results.

**2026-10-06, Claude** — [Ex-2.2.22](/docs/m2/ex-2.2.22/report.py) (E1) ran the three slice sets at two weights, three seeds each. Every restriction let the edit spill past the gate on nearly every run (one of eighteen stayed within it), and at the first slices the states of other ops sat further along e₁ than with every slice pulled. A restriction also drops the anti-subspace term on the slices it leaves out, so a condition that restricts the anchor term alone, keeping the anti-subspace term on every slice, would tell the two apart (Sandy's suggestion in review). The longer runs here are still wanted, now mostly for the seeds ([drop-runs-that-miss-the-second-rise](./drop-runs-that-miss-the-second-rise.md)).

**2026-10-09, Claude** — The [lean review](/docs/m2/embedding-lean/report.py) says what the restriction switched off and why it spilled: with the anti term gone from the unpulled slices, the pull at block 1 still leans the embedding table (lightness on e₁), and the every-slice edit removes that lean from every op. The run it points to is the one the note above asks for, with two additions: hold the embedding off e₁ (the anti term on every slice, or `clean_embedding_rows` on every token as a second arm), and report the edit on the pulled slices beside the every-slice edit, since the two differ on the stored runs. Blocks 2 to 4 is the slice set to pull: on the stored runs that edit removes most of `difference` with spill inside the criterion, except on three unlatched seeds where the concept was partly built out of the lean. Four seeds at 400 epochs. [Mellowmax over slices](./mellowmax-over-slices.md) is a third arm.

