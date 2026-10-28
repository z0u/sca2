---
status: open
tags: [D2.2, anchoring, intervention, architecture]
opened: 2026-10-28
---
# A two-block model, to fix where the op is written and read

The [block1-edit report](/docs/m2/block1-edit/report.py) found that block 1 writes the evidence for the op onto e₁ and one later block reads it from the query `=`, with the reader at block 2, 3 or 4 depending on the seed. The task needs three steps in sequence (each example answer judges itself against its operands, the query `=` reads the judgements, the query applies the op), which fits in two blocks. With two blocks the reader can only be block 2, so the edit at slice 1 would be the whole edit, and the question of which blocks to pull would come down to block 2.

That would make a clean demonstration for the write-up, though it gives up the depth that makes the reader vary, which is the property that carries to a larger model. So it is an arm beside the four-block model, rather than a replacement.

A cheap version: four seeds each of d64-L2 and d64-L3 on the control recipe, to see whether either depth does the task about as well as four blocks (no two-block run on this corpus has been trained to the end; size-timing only timed them). Then the anchored recipe on whichever depth does, scored with the same single-slice edits as block1-edit.
