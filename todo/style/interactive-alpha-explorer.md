---
status: open
tags: [figures, reports, D2.2]
opened: 2026-10-28
---
# An interactive explorer for α

Most D2.2 reports show α, the alignment of a state with e₁, cut one or two ways per figure: by slice at the example answers, along training on the pulled slices, by posterior bin, by seed. Each report picks the cuts that answer its question, and a reader who wants another cut has to ask for a new figure. An interactive view would let a reader slice α freely: by op, role (and the equation it sits in), slice, seed, epoch, and condition, with the posterior on the anchored op as a filter or an axis.

A weekend-sized nice-to-have. The page could be a standalone artifact or an inline JS figure in a report, in the way [the stream-decode explorer](./interactive-explorer-for-stream-decodes.md) proposes. What to settle first is the data: the stored measurements keep α at some roles and checkpoints and not others, so the explorer would start from what one recent experiment stores (the fixed confirmation, say) and grow from there.
