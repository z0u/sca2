---
status: open
tags: [D2.1, ex-2.1.10, anchoring, representations, grading]
opened: 2026-08-15
---
# Grading is stepped at the embedding slice; the slice-mean r² hides it

Breaking ex-2.1.10's grading chart down per (slice, position) shows the embedding slice at op1 responding as a step rather than a grade: near-flat through low redness, then a sharp rise around redness ~0.6 to α ≈ 0.9. The published grading statistic averages the response over slices before scoring, and the graded post-attention slices carry it: per-slice r² at op1 is 0.56 (emb) against 0.80–0.86 (slices 1–4) on `either-t100`, with the slice-mean response scoring 0.82.

It is not specific to the primary — `op1-labels` reads 0.59 at the embedding and `slot-oracle` 0.47, against ~0.85–0.89 at depth — so it looks like a property of how the pull lands on the embedding rather than of the either-slot labeller. A plausible account: the embedding is the one slice where α is a function of the token alone (no context to spread the response), and the anchor's per-line pull weighted by label probability (redness⁸-shaped affinity) gives near-corner tokens most of the pull, which would produce exactly a thresholded response. That account is checkable: the affinity curve predicts where the step should sit.

Questions worth answering: (a) is the step the affinity shape echoed, or a latch-like binarization the grading floor should worry about; (b) does the anti-subspace term set the flat part, or is it just absence of pull; (c) should the grading statistic be scored per slice — a slice whose response is a step passes the slice-mean r² today by hiding behind graded neighbours, which matters for the D2.2 confirm and any claim quoting r² as "the response is graded".

Numbers above are from the published `alpha` arrays (seed-mean, r² against sim¹·⁵ per slice at op1), computed with `docs/m2/grading_sketch.py`'s data path; the per-(slice, position) chart that surfaced this is in PR #99.

## Notes

**2026-08-15, Claude (chart-simplification session)** — ex-2.1.11's published arrays already answer part of (b). Per-slice r² at op1, emb first then L1–L4: `ref` 0.57 / 0.79–0.84, `flat-anchor` 0.61 / 0.82–0.87, `anti-mid` 0.50 / 0.69–0.74, `anti-peak` 0.42 / 0.66–0.72, `anti-hold` 0.25 / 0.54–0.66, `short25` 0.57 / 0.82–0.87. Two readings. First, the emb-to-depth ratio is roughly constant (~0.65–0.7×) across every anchor/anti schedule variant, so the emb deficit looks structural rather than schedule-tunable — no global dose or shape change closed it. Second, the exception runs opposite to the "anti headwind flattens the grade" account: `anti-hold` (constant *low* anti, 0.3×) collapses emb grading hardest, to 0.37× of its own L2, while the scheduled anti (strong early, annealed late) grades best everywhere. So less repulsion at the embedding is the direction the data argues *against*. Also for (a): the emb riser sits near redness 0.6, well left of redness⁸'s half-max (~0.92), so the emb response is already partly generalized through embedding geometry, not a pure affinity echo — the step is affinity-*sharpened*, not affinity-shaped. A per-slice λ arm should probably bracket the anti term upward-early at emb, not downward; and the graded-affinity lever (see flatten-label-affinity item) targets the mechanism the embedding actually has.
