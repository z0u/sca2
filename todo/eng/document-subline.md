---
status: done
tags: [skills, vis]
closed: 2026-08-27
---
# Document subline in a skill

Describe subline in a skill: what it is, why we might use it instead of a token heatmap, and how to use it.

## Notes

**2026-08-27, tech debt** — Landed as a `Sublines` section in `style-fig`, with a cross-link from the "never use heat maps for sequences" bullet so the alternative is findable from the rule. The mechanics went in the skill rather than in docstrings, against the house pattern, because `src/subline/` is vendored from [z0u/subline](https://github.com/z0u/subline) and the gotchas worth writing down are our usage conventions rather than its API.

Four of them, each verified against the code by rendering a probe SVG and reading the path data: values are fractions of the band (0 on the baseline, 1 at the top, negatives silently clipped away, the band itself rendering up to 2 — which is why ex-2.1.1 and ex-2.1.2 both `np.clip(..., 0, 1)`); `NaN` breaks the path, which is what makes the position-0 pad line the series up with the text; only five series colors are defined; and the SVG themes itself independently of `@themed`, so the `--bg-color` override is needed to stop it reading as a grey box.
