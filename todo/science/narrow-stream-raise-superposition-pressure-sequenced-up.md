---
status: open
tags: [D2.1, superposition, model-arch, ex-2.1.3]
---
# Narrow the stream to raise superposition pressure — sequenced, not up front

d64 is generous for this task, and SCA's value proposition lives where geometry is contested. Plan: (1) keep d64-L4 for the first anchored runs, so the only change vs the existing baselines is the anchor; (2) un-anchored width × depth sweep on the chosen testbed (e.g. word-level v216: d16/d32 × L4/L8) to find the narrowest cell that still solves the task — the capacity proxies (item above) then read as a compression axis; (3) re-run the anchored comparison along the width axis down to that frontier. Prefer deep-and-narrow (d16-L8) over wide: width sets per-position capacity, depth adds anchor sites, and ngpt-scaling says the architecture tolerates the aspect ratio. Watch-out: at v216 the softmax's identity separability may fail before value geometry does — which is itself the identity-vs-value competition ex-2.1.3 flagged.
