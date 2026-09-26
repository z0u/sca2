---
status: done
tags: [probes, D2.2, methodology]
opened: 2026-09-22
---
# An RGB-distance readout beside expected exact match

Every removal and selectivity number in D2.2 is expected exact match on the grid: the chance the model's answer is the one right color. It gives no partial credit, so a model that lands one grid step away scores the same as one that answers black. Sandy's suggestion from the ex-2.2.11 review: report the RGB distance (or its square, an MSE) of the greedy guess, and of the median of the model's distribution over the grid, beside the exact match.

Where it would help. On ex-2.2.11's removal lines, `handover` keeps 24% of its exact match on `hue-hsv` and 12–18% on the channel-wise ops. Distance would say whether those kept answers are the right color or a near neighbour, and whether the lost ones are near misses or somewhere else entirely, which the exact match cannot distinguish. It would also show, for the near miss on `darken`, what the model answers with red taken out.

Cost: the probe already has the full distribution per line, so this is a scoring change, no re-run. The grid geometry is fixed, so the distance is well defined. Worth adding to the probe utilities so later experiments get both readouts for free, and worth a preregistered expected direction before it is used in a gate.

## Notes

**2026-09-25, Fable** — Done as a re-score, in [`docs/m2/answer-distance/`](/docs/m2/answer-distance/report.py), with the distances in `sca.answer_distance` (greedy, mean, channel-wise median, expected distance; floors and chance per line). On `handover` under `projection` the removed answers are near misses: the greedy guess sits 1.8 (`mix`) to 3.4 (`value-hsv`) steps from the raw answer against a floor of 0 to 0.5 and chance of 3.7 to 4.4; on `mix` nine in ten moved answers are one or two steps off. The kept lines stay at the floor. The lost answers are not the op applied to a de-reddened operand (a counterfactual check in the report). Non-red lines move under 0.03 steps on `handover`, less than the control's own lines under the same operator. Seed CV of the distances is 6% to 20% against 33% to 127% for the kept share. Proposed preregistered direction, for the next experiment to adopt and set thresholds from its own control: removal as the normalized distance of the mean of the distribution, (distance − floor) / (chance − floor), rising on the removal lines; selectivity as the non-red rise of the same mean staying at or under the control's rise under the same operator. Open until an experiment adopts it in a gate.

**2026-09-26, Opus** — Adopted after Sandy's review. The report now states the direction above as adopted for D2.2's next removal and selectivity gates, and the [D2.2 design](/docs/m2/d2.2/design.md) points to it; that experiment sets the thresholds from its own control runs. Re-run on production. Two follow-ups filed: the [landing histogram](/todo/style/landing-histogram-for-distance-readouts.md) as a house figure, and [inverting the op](/todo/science/effective-operand-by-inverting-the-op.md) to read back the effective operand.
