---
status: open
tags: [D2.1, anchoring, ex-2.1.6]
---
# Labeling and pull-span variants for ex-2.1.6, queued behind the baseline read

Bumped 2026-08-05, during the ex-2.1.9 prereg review: with a pooled pull, (a) becomes the natural labeling — label any line containing *red* and let the pool find the position — and is probably the better match for what M3's document-level labels will look like. Original list follows:
(a) either-slot labels, where a red op2 can also trigger one; (b) labels drawn once at corpus build rather than per visit, so the pull comes from fixed sparse evidence, like labeled internet text; (c) a whole-span pull that includes the answer and newline — the shape a document-level label takes in natural language, where nothing marks a position as safe to exclude. Ex-2.1.6 pulls the prompt span only as a measurement instrument (the answer position has a known redness confound, and unpulled it doubles as a spillover read), so (c) is what says whether the exclusion matters rather than a premise of the method. Update 2026-08-08: (a) is now ex-2.1.10 (preregistered); (b) and
(c) remain open, and the ex-2.1.10 prereg points here for (c).
