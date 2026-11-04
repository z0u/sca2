---
status: open
tags: [D2.1, D2.2, post-hoc, baselines]
opened: 2026-10-30
---
# The post-hoc comparison for red on the D2.1 checkpoints

[Ex-2.2.1](/docs/m2/ex-2.2.1/report.py) compared the anchored edit for *red* with fitted directions on the D2.1 checkpoints, but gave the fitters little: three control seeds, labels from the token alone, one fit per slice, and no tuning. The [post-hoc comparison](/docs/m2/post-hoc-compare/report.py) gives the fitters a full grid, every label, and a frozen selection rule, for the op `difference`. Its code would carry over to *red* as it is. Worth running if the post or the paper shows the *red* result beside the op result, so the two are compared on the same footing.
