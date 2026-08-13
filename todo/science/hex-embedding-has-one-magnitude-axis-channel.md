---
status: finding
tags: [anchoring, ex-2.1.5, representations]
opened: 2026-07-26
---
# The hex embedding has one magnitude axis, and channel identity is positional

The same 16 digit tokens serve all three channels, and the model uses RoPE, so the depth-0 residual is the token embedding with no positional component. The probe weights confirm what that implies: the readout direction for red at digit 1, green at digit 2 and blue at digit 3 are the same vector — cosine +1.000 on all three seeds. So at the input there is no hex "red" direction to anchor or to align with named red; there is one value axis, read three times, with position supplying the channel. The channels do differentiate with depth (within one landmark at the last layer the three directions run cos −0.66 to +0.09), so channel-specific directions exist — just not where the tokens enter. Bears directly on anchor placement for a mixed-vocabulary corpus.
