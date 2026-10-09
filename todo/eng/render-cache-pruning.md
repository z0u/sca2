---
status: open
tags: [tooling, reports]
opened: 2026-10-09
---

# Prune old entries from the shared render cache

`./go render <report> --cached` (`scripts/render_cache.py`) adds an entry under `/mnt/project-files/rendered/<key>/<version>/` each time a report is read at a new version, and nothing removes the old ones. A version changes with every commit that touches `docs/` or `src/`, so a busy week could leave a few dozen entries per report, each the size of its figures (embedding-lean is about 1 MB).

A simple rule would keep the newest few entries of each report by `woven_at` in `meta.json`, run by hand or from the store GC. Deleting while another session reads is the one case to think about; keeping anything woven in the last day probably covers it. Worth doing once the folder gets large enough to notice.
