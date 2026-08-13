---
status: finding
tags: [metrics, ex-2.1.5, representations]
opened: 2026-07-27
---
# A principal angle between probe subspaces needs a dimension-matched null

The angle between two 3-dimensional row-spaces shrinks as the space they sit in does, so any sweep over residual width will see angles fall whatever the representations are doing. Ex-2.1.5's cross-form mix decoders go 75° → 60° → 51° over d64 → d32 → d16, and two random 3-planes at those widths give 72° → 64° → 51°: the whole trend is the ambient dimension. Two nulls are worth carrying. Random 3-planes are the cheap one. The better one is a seed control — the same form's probe from a different seed, which decodes the same quantity in an unrelated basis, so it matches the real probes' conditioning as well as their dimension; here it tracks the random null to within a few degrees at every width. Report raw angles only against one of these.
