---
status: open
tags: [M3, evaluation, side-effects]
opened: 2026-09-01
---
# Packed contexts open a cross-line leakage channel

D2.2's completion evals score one line per forward pass, so an intervention's side-effects stop at the line edge: no state survives across forward passes, and there is no earlier line in-context to disturb. Training-style windows are different — batches are sliding windows over the packed newline-separated corpus ([batches.py](/src/sca/data/batches.py)) with plain causal attention and no cross-line mask, so a later line reads earlier lines freely. An intervention applied in a packed context can therefore change what subsequent lines see: softmax conserves attention mass, and mass lost at an intervened position lands somewhere, including on other lines.

The moment an eval scores packed multi-line contexts — few-shot prompts in M3, say — this becomes a live side-effect channel that per-line scoring never exercised. When that happens, the eval contract should declare the bound's scope as *this forward pass, downstream positions* rather than *this line*, and a packed-context arm should measure the leakage. Surfaced while working through the [D2.2 design](/docs/m2/d2.2/design.md)'s intervention semantics on the nGPT architecture.
