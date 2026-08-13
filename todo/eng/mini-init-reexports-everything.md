---
status: open
tags: [tooling]
opened: 2026-08-06
---
# `mini/__init__.py` re-exports the whole package

`from mini.reports import export_key` runs `apparatus`, `modal_apparatus`, `experiment`, `store`… so a leaf module with only stdlib imports still needs the full environment. Cost a round in CI: `scripts/unpublished_reports.py` was written to run before the install and couldn't. Not urgent — nothing else wants a lightweight import today — but worth remembering before the next standalone tool.
