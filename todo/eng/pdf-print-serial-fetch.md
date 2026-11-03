---
status: done
tags: [publishing, tooling, reports]
opened: 2026-09-21
closed: 2026-11-03
---
# A site build that reprints every PDF fetches figures one at a time

`route_remote` serves a print's `https://` requests from a cache that Python fills, one blocking fetch per URL inside the route handler. The fonts and KaTeX are shared and cached after the first report, but every figure is its own pinned CDN URL, so a report with twenty figures pays twenty round trips in series before its `load` event fires. On the GitHub runner a full reprint of 30 reports took six and a half minutes (the #190 preview, 2026-09-21). Since the build hands a preview production's memo to borrow from (a PDF's links resolve to production on every branch, so the two keys agree), a science PR pays this only for the reports it changed; a PR that touches `report.css` or `report_print.py` still reprints everything, as does a report new in the PR.

The remaining speed-up is to warm the cache before `goto`: scan the printable HTML for `https://` image sources and fetch them concurrently with a thread pool, so the route handler finds every figure on disk. It is contained and does not change what a PDF says.

## Notes

- 2026-09-21: previews now borrow production's PDFs (`MINI_PDF_MEMO` takes a path list; `PdfMemo.fallbacks`), which was the second option here. The open question it raised, a report new in the PR, is resolved by accepting a dangling link in that preview's PDF until merge.
- 2026-11-03, tech debt: done as described. `print_bundle` now calls `prefetch_remote` on the page before `goto`: it fetches the page's `https://` image sources into the same cache with up to eight threads, warns once per host that fails, and hands those hosts to `route_remote`, which aborts their requests without fetching again. The cache now writes a body by rename after its type file, so a body on disk always has its type beside it. `print_stamp` hashes the module, so the next site build reprints every PDF once. I didn't time a full reprint on a runner.
