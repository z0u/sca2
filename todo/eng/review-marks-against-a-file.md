---
status: done
tags: [reports, tooling]
opened: 2026-10-28
closed: 2026-10-31
---

# Change bars against an earlier file, without git

`./go render --since REF` bars the margin of a PDF beside every line changed since a commit. A draft written in a thread usually lives outside the repo (in the shared project folder or a scratchpad), so it has no commit to compare against, and its second round of review prints without bars.

An option that names an earlier file instead of a ref would cover it: an agent keeps numbered versions of a draft in a scratch folder (`draft-1.md`, `draft-2.md`) and prints `draft-2.md` against `draft-1.md`. For a Markdown page this is close to what `review_base.render_md_at` does now: it weaves the old text with today's code into a baseline directory, and `mark_changes` compares the two rendered pages. Only the source of the old text changes, from `git show` to a file read. The images an old draft links from disk would go through `render_report.local_images` as the new draft's do, so the figure digests compare.

A literate report (`.py`) is harder, since its baseline is a full export of the old script, so starting with Markdown pages seems right.

## Notes

**2026-10-31, implementation** — done for both kinds of document. `--since` takes a file when one exists at that path, and a ref otherwise. The earlier copy is woven by today's code as if it were at the report's own path (`review_base.export_file`), which matters for a literate report: its cells run in a module named after the report's path, and its memoized figures are keyed by that name, so a copy rendered under its own path misses the cache for every figure (and can't find a sibling it loads by relative path). Woven at the report's path, the baseline of `fixed-effects` cost about the same as a warm render. A `.py` outside the repo still can't be exported at all (`export_one` names it relative to the repo), so this covers a report copied aside within the repo, and Markdown drafts anywhere.
