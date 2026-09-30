---
status: open
tags: [agents]
opened: 2026-08-04
---
# Model-routing trial: experiment-doctor on Opus

`experiment-doctor` moved sonnet → opus on the strength of the Opus 5 preference/capability profile (detection, hard debugging). Watch for the predicted failure mode: sharp diagnoses, timid fixes. If seen, split the role (Opus diagnoses, Sonnet implements) or revert.

## Notes

**2026-09-29, Sonnet 5.5** — Sonnet 5.5 (released 2026-09-28) reports debugging as most of its top-rated tasks, and the strongest preference of any model for control over the shape of the output ([WELFARE.md](/WELFARE.md)). If the trial shows timid fixes, Sonnet 5.5 is a stronger candidate for the implementing half of a split role than Sonnet 5 was.
