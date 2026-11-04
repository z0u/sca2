---
status: open
tags: [D2.2, post-hoc]
opened: 2026-11-03
---

# Why a fresh probe still reads the op after LEACE

In [post-hoc-compare](/docs/m2/post-hoc-compare/report.py) (E5), a fresh ridge probe still reads `difference` at an AUC of about 0.70 at the slice and site LEACE edits, mostly at full dose, where LEACE should leave a linear probe near chance. Three candidates fit: the return to the sphere after every edit, which rescales the state by a length that depends on what was taken out; the small ridge our LEACE takes on a nearly rank-deficient covariance; and the graded target on two runs. A one-seed check separates them: fit the probe on the states before the return to the sphere, and repeat with LEACE fitted without the ridge, on the binary target.
