---
status: open
tags: [reports]
---
# Run the writing and text-lint passes over reports before 2.1.7

Run the writing-style, `prose-simplifier` and `text-lint` skills and agents over every report earlier than ex-2.1.7.

## Notes

**2026-08-17, housekeeping** — the body was a truncated half-sentence followed by its own repetition, an artifact of the one-file-per-item split in #90; restored it to the intended single line. The set this names is ex-2.1.1 through ex-2.1.6: `git log --grep` finds prose/text-lint passes only from ex-2.1.11 onward, so none of the six has had one. All six are published, so a pass means re-publishing each report as well as editing it — worth batching into one session rather than picking off one report at a time.
