---
status: open
tags: [D2.2, task-grammar, representations]
opened: 2026-08-27
---
# Does a richer op set give the model a better operand geometry?

With one operation, the model only ever needs whatever operand representation `mix` requires. More operations — saturating add, screen, multiply, and especially ops in another color space (hue, saturation, brightness) — put more constraints on the operand, and might sharpen the RGB cube in the stream, or even let the model discover a shared representation between named and hex colors, which ex-2.1.5 found it bridges rather than unifies.

Cheapest test: un-anchored control models at one, three, and six operations, with the cube probed per site as in ex-2.1.12; an optional arm of ex-2.2.2, on control models only so it cannot confound the anchored conditions. Kept separate from the D2.2 anchoring question because it changes the representation at the same time as the anchor target changes. If HSV ops help, HSV-valued colors as tokens are the natural next grammar extension, out of D2.2's scope.

From the [D2.2 design](/docs/m2/d2.2/design.md), Sandy's suggestion at the inception.
