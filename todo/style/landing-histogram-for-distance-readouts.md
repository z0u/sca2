---
status: open
tags: [figures, reports]
opened: 2026-09-26
---
# A landing histogram for every distance readout

From Sandy's review of [answer-distance](/docs/m2/answer-distance/report.py): the per-op histograms of where the removal-line answers land ("These are really good. I think we should use this format in future reports").

The format, as that report draws it (`hist_draw`): one small panel per op, sharing both axes; bars for the share of lines whose greedy guess sits each whole number of grid steps from the raw answer under the intervention, pooled over seeds; the clean pass as an outline step; and the distribution a uniform guess gives on the same lines as a dashed line. The bin at zero holds the answers that survive, and the rest of the bars show how far the others went, against what guessing would give.

To make it the default: move the drawing into a shared helper (beside the grading-cloud helpers in `sca.vis`, or `mini.vis` if it is general enough), taking per-op bar, outline, and reference arrays, and add a line to the `style-fig` skill saying that a report scoring removal by distance shows it. The chance reference (`chance_hist` in the report) belongs with the distance code in `sca.answer_distance`.
