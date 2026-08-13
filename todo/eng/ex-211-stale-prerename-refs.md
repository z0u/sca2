---
status: open
tags: [storage]
---
# ex-2.1.1's pre-rename report refs still sit in the store

The ex-2.1.1 report refs moved to `reports/m2/ex-2.1.1/*`; the pre-rename `reports/ex-2.1.1/*` refs still sit in the store (there's no ref-delete API). Harmless clutter, but they pin their artifacts through GC's mark-and-sweep. If a ref-delete/rename verb ever lands (eng/gc.md), sweep them. The m1 refs (`reports/ex-2.9.*`) predate milestone nesting and stay flat on purpose.
