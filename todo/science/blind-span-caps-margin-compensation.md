---
status: open
tags: [D2.1, anchoring, ex-2.1.7]
---
# If the blind span caps the margin, M3 needs a compensation mechanism

If the blind span is what caps the margin (ex-2.1.7 H3's op1-only contrary reading), M3 needs a compensation mechanism, because natural language only offers document-level labels — no position is marked as the relevant one.

The affinity-softmax / logsumexp pooling item above is the leading candidate; this item exists so a blind-span result gets read as "prioritize pooling" rather than "narrow the pull", which M3 cannot do. Ex-2.1.7 delivered exactly that reading (H3's contrary outcome), so this is now settled as the reason ex-2.1.9 exists.
