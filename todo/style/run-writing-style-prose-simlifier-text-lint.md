---
status: open
tags: [reports]
---
# Run the writing and text-lint passes over the reports that never had one

Run the writing-style, `prose-simplifier` and `text-lint` skills and agents over ex-2.1.1 through ex-2.1.7 — the D2.1 reports written before those passes became part of the report workflow.

## Notes

**2026-08-17, housekeeping** — the body was a truncated half-sentence followed by its own repetition, an artifact of the one-file-per-item split in #90; restored it to the intended single line. All of these are published, so a pass means re-publishing each report as well as editing it — worth batching into one session rather than picking off one report at a time.

**2026-08-22, housekeeping** — re-aimed. The old title said "before 2.1.7", which stopped being a window some time ago: D2.1 has closed out at ex-2.1.11, all seven of the named reports are published and frozen, and the batch can now only be scheduled against D2.2's calendar rather than against an experiment inside its own set. Also corrected the boundary the 08-17 note drew. Per-report prose passes start at **ex-2.1.8**, not ex-2.1.11 — `git log -i --grep` over each report directory finds them for 2.1.8 (`fe3af49`, `e2107f2`), 2.1.9 (`ceca503`), 2.1.10 (`a34851b`) and 2.1.11 (`4851fcd`, `11cbcbe`, `a44560d`). What 2.1.1–2.1.7 carry instead is only the repo-wide unwrap of #86, which reflowed prose without rewriting it. So the set is seven reports, one more than the earlier note said, and ex-2.1.7 is the one it missed.
