---
status: finding
tags: [D2.1, ex-2.1.2, ex-2.1.5, representations]
---
# Just-in-time computation with eviction

_(ex-2.1.2 answer-schedule probe)_

At the final layer, channel k is decodable (R² ≈ 0.97) only at its own emission position, and previously-emitted channels are dropped from the deep residual stream — so a "result" concept never fully exists at any single position, and anchoring one there would fight the model's schedule. Carry into anchor design. Replicated in ex-2.1.5 on a corpus with no name↔hex bridge, per channel: at the last layer each channel is decodable about one position before its digit is emitted and fades after (0.98 red at `#`, 0.96 green a token later, 0.95 blue at the last digit with red down to 0.38), so the schedule isn't an artifact of the bridged language.
