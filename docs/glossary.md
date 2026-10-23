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

Alignment
α
:   How far a state points along e₁: the cosine between the state and e₁, which for a unit-length state is its component along e₁; 1 on the anchor direction and 0 off it.

Bayes predictor
:   The best answer given the examples: it weighs the answers of each op by the posterior over ops, the probability of each op given the examples.

Expected exact match
:   The task score, as the chance that a drawn answer agrees with a true one. With drawn answers, the correct answer to a line is spread over two or more colors, so we take the chance that an answer drawn from the model agrees with one drawn from the true distribution. Its ceiling is that same chance for the true distribution against itself (Σq²), below 1 on the ops that round.

Band {.manual}
:   Our measurement precision, the smallest gap between seed means that we call resolved. Two seed means are told apart only when they differ by more than 2σ√(1/n_a + 1/n_b), with σ the per-run spread and n the seed counts; smaller differences are reported as unresolved.

Pooling temperature
:   How sharply the anchor pull picks its positions. The pull on a context is a softmin-weighted mix over its positions, and the temperature (τ_p) sets how peaked the weights are: at 0.1 nearly all the weight goes to the position closest to e₁, and at 1 it is spread over many.

Spill
:   The side effect of an edit on the other ops: the largest drop in the score of any op other than the one removed, net of what the same edit does to a control trained without the anchor. The selectivity criterion of D2.2 allows at most 0.02 at any dose.

Target null
:   Where a removal should land: what an ideal predictor that had lost one op and nothing else would answer. It is the Bayes posterior over ops given the examples, with the removed op set to zero and the rest renormalized, so it still weighs the other ops by how well they fit.

Total variation
Total variation distance
:   A distance between two answer distributions: half the summed absolute difference, from 0 (identical) to 1 (no overlap). Mass a model puts on tokens that are not colors counts as distance.
