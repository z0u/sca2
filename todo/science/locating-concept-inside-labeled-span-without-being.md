---
status: partial
tags: [D2.3, ex-2.1.9, anchoring]
opened: 2026-08-05
---
# Locating the concept inside a labeled span, without being told where it is

**Preregistered as ex-2.1.9** (2026-08-05): mellowmax pooling at τ ∈ {0.01, 0.03, 0.1} plus τ = ∞ and an op1 reference arm, all at ex-2.1.8's `end90-hold30` operating point. Two calibration findings moved the design off the notes below: the 0.6–0.8 leading-weight band sits at τ ≈ 0.01–0.03 (nothing like the 0.4 in the toy example), and ex-2.1.8's span pull parks its cube-wide drift on the constant syntax tokens (ᾱ maxed over span roles 0.37 against 0.09 at op1), so the prereg adds ᾱ_span as a scored statistic and names the syntax-token latch as an explicit contrary outcome. Design notes from the 2026-08-03 session are below; the original note follows them.

Conditional follow-up arm, suggested in the 2026-08-05 review: exclude the embedding slice from the pull, so no shared token embedding can satisfy the term directly and every pulled state can, in principle, depend on whether an earlier token was red. Queued rather than added: it changes two things at once against ex-2.1.8 (pooling and pull depth), the H4 weight profile will already show *where* a latch forms, and the embedding shift can partially propagate to deeper slices through the residual stream anyway, so the arm is sharpest as a follow-up targeted at whatever ex-2.1.9 finds. Related: the "can an anchor be confined to part of the stream" item under.
