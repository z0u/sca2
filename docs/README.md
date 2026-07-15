# docs/

This directory contains executable experiment notebooks and source files for the
project site. The site is built into `_site/` — by `./go preview` locally, or
`./go site` in CI.

## File types

**Marimo notebooks** (`.py`) are the primary content — and the *only* thing in Git;
the exported HTML is never committed. `./go publish` exports each notebook to a
self-contained bundle (`index.html` + named-keyed `_assets/`) and mirrors it to the
HF bucket under `exports/<key>/`, where `<key>` is the notebook's docs-relative path
without suffix (`docs/overview.py` → `overview`, `docs/foo/bar.py` →
`foo/bar`) — except a notebook named `report.py` takes its directory as the key
(`docs/foo/report.py` → `foo`), so a one-report experiment publishes at `foo/` rather
than the redundant `foo/report/`. Publishing also records the commit sha the bundle
landed as into [`publish.lock`](./publish.lock) — commit that file: the site serves
each report at its pinned revision, so a publish from a branch deploys nothing until
the pin reaches main (the PR preview serves the branch's pins meanwhile). `./go site`
(CI) then assembles `_site/` from the pinned bundles, serving each report at
`_site/<key>/index.html` (URL `<key>/`).
`./go preview` assembles the same site *locally* — it exports stale reports to
`.mini/exports/` and copies their assets beside the HTML, so it works offline.

**Markdown files** (`.md`) are converted to HTML and written to `_site/` at the
same relative path. Links to a report's `.py` are automatically rewritten to its
rendered `<key>/` page. This `README.md` is excluded from the build.

**Other assets** (images, SVGs, etc.) are copied as-is into `_site/`.

## Structure

```
docs/
├── README.md                This file (excluded from build)
├── publish.lock             Export key → pinned publish-tier revision (written by ./go publish)
├── index.md                 Built as _site/index.html
├── overview.py              Marimo notebook → exported bundle, served at _site/overview/
└── ex-9.9/                  An experiment, split into definition + report
    ├── experiment.py        Importable main(ctx) DAG — not a notebook, so the build ignores it
    └── report.py            Marimo notebook → served at _site/ex-9.9/
```

Exported bundles live (gitignored) under `.mini/exports/<key>/` locally; their durable
home is the bucket. Nothing under `docs/` holds generated HTML.

Heavier or multi-step experiments live in a subdirectory as an importable
`experiment.py` (the definition, driven by the `mini` CLI) plus a `report.py`
notebook (reads durable results and publishes). A plain `.py` that isn't a Marimo
notebook is ignored by the build, so the definition module never lands on the
site. See the `mi-ni` skill (authoring, running & monitoring).
