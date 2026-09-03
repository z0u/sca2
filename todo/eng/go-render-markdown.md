---
status: open
tags: [tooling, reports, skills]
opened: 2026-09-02
---
# Give the Markdown render a `./go` verb

`scripts/export_report_md.py` turns a notebook into a readable Markdown document, and nothing calls it: no `./go` verb, no CI step, and until 2026-09-02 no skill named it, so the only routes to it were its docstring and `tests/test_export_report_md.py`. The cost showed up in the `report-structure` pass: `report-review` said "export the document" without saying how, and the nearest export to hand was `./go preview`, which produces the HTML bundle. That bundle is about ten times the size of the same document and embeds the notebook source three times, so a pass whose point is reading the assembled document instead of the source got handed the source, buried in markup.

Naming the command in `report-review` and `report-render` fixes the immediate case. A verb — `./go render docs/m2/ex-2.2.1/report.py`, defaulting the output to `.mini/renders/<key>.md` — would make it discoverable the way `preview`, `publish` and `site` are, and give the key derivation in `docs/README.md` one implementation rather than a path each caller spells out.

Worth deciding at the same time: whether the render is stale-checked and cached like `./go preview` does for bundles (it re-runs the notebook, which is the slow part), and whether the image links should be rewritten to a resolvable path. They currently point at the notebook's live `public/.mini/`, so they only resolve from the notebook's own directory.
