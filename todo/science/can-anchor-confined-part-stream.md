---
status: open
tags: [D2.3, anchoring, representations, ex-2.1.6]
---
# Can an anchor be confined to part of the stream?

Ex-2.1.6 pulls all five layers and all four prompt positions equally, and the margin came out with a shape nobody asked for: *red* enters at its own token and migrates with depth to the positions after it (op1 leads in the embedding and first block, `+` and `=` lead by the last). That is probably a transformer behaving as transformers do, and for an across-the-board intervention on *red* it may not matter — we would want the edit at every (layer, position) site anyway. It matters for [D2.3](./README.md#d23-asymmetry), which asks for suppression that degrades completion while leaving verification intact: that needs the concept localized somewhere we can reach, and nothing so far says whether the anchor term can be confined to a chosen region rather than merely applied there.

## Notes

**2026-09-25, D2.2 pivot** — Under the [adopted pivot](/docs/m2/d2.2/pivot.md#asymmetry-and-depth), completion and verification both infer the same op from the same context, so an asymmetry would likely need the anchor confined to the part of the stream only completion uses. That moves this item from optional to central for D2.3. The pivot hopes depth alone is enough.
