---
status: done
tags: [reports]
---
# Decide which retention statistic the reports quote

Settled as the **minimum** across seeds, in `ex-2.1.8.RETENTION_STAT` with the rationale: H4(b)'s gate is worded per run ("for every anchored run … at least 0.8× that maximum"), and a three-seed mean can hide a single sliding run, which is the failure the gate exists to catch. Every later report follows it.

Left to do: ex-2.1.7's *Arms* section still quotes the mean under the same name (`span-anti-late` 0.94 against the Findings section's 0.92; `span-anti-hi` 0.58 against 0.55). Fix it on the next touch of that report.
