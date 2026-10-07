---
status: partial
priority: high
tags: [publishing, reports]
opened: 2026-09-23
---
# Markdown pages that look and print like reports

The site build prints a `report.pdf` for every report (`mini.report_print`, through `docs/report.css`'s `@media print` block), and `./go render <report> -o <name>.pdf` gives one locally for review on the reMarkable. Markdown pages under `docs/` get neither: `./go preview docs/m2/d2.2/pivot.md` skips the file as "not a rendered report". Design docs and proposals (`d2.2/design.md`, `d2.2/pivot.md`) are what we most often review in rounds, so they want the same path, including `--since` for the margin bars.

A stopgap that worked on 2026-09-23: render the body with `build_site.render_markdown`, inline `base.css`, `scripts/md.css`, and `docs/report.css`, load the page in Playwright, and call `print_page`. With the default `fit=True` it prints one tall 158 mm page clipped to its content, the same shape as a report's PDF. Mermaid blocks and `mini:figures` strips were not exercised.

What a proper version needs: `convert_markdown` pulling in the print sheet, a nav chip linking the PDF, the build's print memo covering Markdown pages, `./go preview` accepting a `.md` path, and `review_marks` working on Markdown sources.

**2026-09-23, D2.2 pivot round 2** — Change marks work for a Markdown page as they are: render both versions with `build_site.render_markdown`, wrap each body in `MAIN_OPEN … </main>`, and pass them to `review_marks.mark_changes`. The page then prints one section per page, as a report does. A `--since` on the proper version should take this route.

**2026-09-24, round 3** — Sandy, on the round-2 print: plain `.md` pages should look just like `lit` reports. That print came out with much looser line spacing than a report's (the round-1 print, without the `main.lit` wrapper, did not), so the stopgap's mix of `md.css` and the report sheets does not match a report. The proper version should render a Markdown page through the same page shell and stylesheets a report uses, on the site as well as in print.

**2026-10-05, the print half** — `./go render <page>.md -o <name>.pdf [--since REF]` now goes through `mini.lit`: `parse` reads a `.md` as one piece of static prose, so a Markdown page gets the page shell, `base.css`, `lit.css`, and `report.css` a report gets, and prints through `review_page` with links resolved to the site. Its `--since` baseline is the old text woven by today's code (`review_base.render_md_at`), with no checkout, since the page has no cells. The `mini.lit` dialect gained `def_list` (no report used the syntax, so their output is unchanged); its heading slugs already match `github_slug` on every heading under `docs/`. Still to do, the site half: `convert_markdown` rendering through the same path, so the site and the print look alike. That needs the site-only pieces of `md.css` (`.tags`, `.fig-strip`, `pre.mermaid`) carried into the lit page, the `mini:figures` strips and mermaid applied to the lit page body, `./go preview` and `./go serve` accepting a `.md` path, and the build print memo covering Markdown pages. A print of `docs/index.md` today shows tags as plain code spans and no strips, for that reason.
