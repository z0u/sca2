---
status: open
tags: [print, glossary]
opened: 2026-11-03
---
# Underline links that hold a glossary term, on paper

On paper, a link whose text holds a glossary term has no underline (`docs/report.css`, the rule beside `dfn.term a`), so a finding title such as "Spill° at matched removal (H1)" reads as plain blue text. We took the underline off because the ring after the term is an inline-block, and a text decoration skips inline-blocks, so the underline stopped short at the ring and started again after it.

A ring drawn so that the underline runs beneath it would let these links keep their underline. Two ways to try: draw the ring as a non-atomic inline box (an empty inline with border and padding, which takes part in the parent's decoration), or keep the inline-block and draw the underline for the ring's width on the ring itself, using the link's decoration colour. Check the result in a print of a report with term links in its Findings list, and keep the no-break behaviour (`term-end`, `term-keep` in `mini.lit.notes`).
