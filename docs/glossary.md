# Glossary

Terms whose meaning holds across the reports. A report shows each one beside its first use in every section, and a report that uses a term in a narrower sense defines it again in its own glossary.

<!--
How this file is read: src/mini/lit/notes.py (dl_terms). Each definition is a Markdown
definition list item: the term, any other wordings on the lines below it, then ":   " and
the definition, in inline Markdown. Lead the definition with a few words to jog the
memory, as a short first sentence or a head before a colon. That lead is the gloss: all
the margin shows until hover, and all the PDF shows. A term that is also an everyday
word takes {.manual}, so it is only marked where written as [text](term:band).
-->

Expected exact match
:   The task score, as the chance that a drawn answer agrees with a true one. With drawn answers, the correct answer to a line is spread over two or more colors, so we take the chance that an answer drawn from the model agrees with one drawn from the true distribution. Its ceiling is that same chance for the true distribution against itself (Σq²), below 1 on the ops that round.

Band {.manual}
:   Our measurement precision, the smallest gap between seed means that we call resolved. Two seed means are told apart only when they differ by more than 2σ√(1/n_a + 1/n_b), with σ the per-run spread and n the seed counts; smaller differences are reported as unresolved.
