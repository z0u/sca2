---
status: done
tags: [D2.1, anchoring, ex-2.1.6]
closed: 2026-08-14
---
# Pool the anchor term over positions instead of averaging over them

Ex-2.1.6 pulls all four prompt positions equally and asks (H3) whether the concept condenses anyway. If it broadcasts instead, a logsumexp over positions is the response: it asks that the span align somewhere rather than everywhere, which is the shape a document-level label really has. Ex-2.1.6's H3 section names this as the queued answer to a broadcast result, so it wants to exist as an item either way. Folded into the ex-2.1.9 design notes above, which settle the pooling form (mellowmax, not a softmax-weighted average).
