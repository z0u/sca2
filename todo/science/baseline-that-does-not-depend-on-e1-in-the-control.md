---
status: open
tags: [D2.2, scoring, methodology]
opened: 2026-10-24
---
# A baseline that does not depend on where e₁ sits in the control

Net removal and spill subtract what the e₁ edit does to the control at the same seed. That is one draw of a direction in the control: in the Adam controls at seeds 704 and 705, e₁ is among the quietest coordinate axes at one seed and among the busiest few at the other ([optimizer-scout](/docs/m2/optimizer-scout/report.py) basis results). Under a rotated optimizer basis the draw went badly on two of eight seeds, where e₁ overlaps the syntax spike of the control and the edit takes half the op from it ([rotated-basis](/docs/m2/rotated-basis/report.py), E2).

The candidate is the median drop over edits along several coordinate axes off the syntax axes, like the reference edit of the spill-timing experiment (PR #285), with raw removal reported beside it. The work is a re-scoring of stored runs: score the reference edits on the stored controls and anchored runs of split-anti-seeds, and compare the net numbers against the ones quoted so far. If they barely move, adopt the median as the scoring rule in the D2.2 design; if they move, say where.
