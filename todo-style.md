# Style todo

List of stylistic improvements to make.

## Text

- [ ] Run the writing style/prose-simlifier/text-lint skills and agents over all reports < 2.1.7.

- [ ] These are writing tics. Rewrite prose to remove (or massively reduce) them. These are all listed in the text-lint skill, so may be resolved by the point above.

  - carry e.g. "Where redness is readable is carried over from ex-2.1.6" -> "We measure where redness is readable, [like/as we did] in ex-2.1.6";
  - read [as], in the sense of "interpret" or "understand", e.g. "so read it as 'no better' rather than 'worse'." -> "so we interpret this as 'no better' rather than 'worse'." Test: if "interpret"/"take"/"understand" substitutes cleanly, rewrite it. Other senses are fine — literal reading (files, papers, axes), and extracting a measurement ("read redness off the residual stream", "readout layer").
  - readable, applied to a representation, e.g. "where redness is readable at op1" -> "where redness is decodable at op1" or "how strongly redness is encoded at op1". Prefer "decodable"/"recoverable" for the probe's side and "encoded"/"carried" for the model's side.
  - gap, a in "measurable difference", e.g. "the median gap between the nearest name and the second-nearest" -> "the median distance between the nearest and second-nearest names"
  - wrinkle, hair

- [x] "cell" vs "condition" conflict with the glossary. Resolved the other way:
  "cell" collides with table/heatmap cells and Marimo cells, which can't be renamed,
  so prose now uses **condition** (seed-aggregated factor combination), **run**
  (condition × seed), and **criterion** (a hypothesis-gate clause), reserving "cell"
  for literal grid entities. Applied to ex-2.1.6/7/8 (where the old glossary lived);
  earlier reports already conformed. Scheme documented in the `style-terms` skill.

- [x] Add a tl;dr **lede** to each of the reports, like the one in 2.1.7. Then use that text or something like it in docs/index.md for each of the reports instead of the current descriptions.

## Visual

- [ ] Tone down the admonitions. We need a mild callout for things like "this report was preregistered".
