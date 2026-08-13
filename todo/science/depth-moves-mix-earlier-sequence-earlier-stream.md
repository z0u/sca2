---
status: finding
tags: [ex-2.1.5, representations, anchoring]
opened: 2026-07-27
---
# Depth moves the mix earlier in the sequence, not earlier in the stream

Doubling L4 → L8 at d64 leaves held-out named accuracy (0.667 → 0.671) and σ (0.032 → 0.032) untouched, so depth buys no resolution once the palette's spacing is the ceiling. It does buy about one extra layer of mature mix (1.7 of 4 layers over R² 0.9 at L4, 2.3 of 8 at L8), but as a fraction of the network the mature mix sits later, 42% → 29%: the extra layers went in front of the computation, and the mix stayed against the answer. Where depth shows up is the token axis — the named mix at the `=` sign reads 0.58 at L4 against 0.77 at L8. At d16, where capacity rather than the palette is the limit, depth does buy accuracy (0.047 → 0.164) and halves σ. For anchor placement, the window holding a mature concept is one to two layers at these depths and does not widen in proportion to the network.
