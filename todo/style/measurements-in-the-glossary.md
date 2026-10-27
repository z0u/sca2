---
status: open
tags: [publishing, glossary, reports]
opened: 2026-10-13
---
# Measurements in the glossary terms machinery

Reports define their measurements (removal, spill, landing, task change, alignment) in a bulleted Measurements section, and the same definitions are repeated from report to report. The glossary terms machinery (margin notes and `docs/glossary.md`) could carry them instead, so a report names a measurement and links its definition.

Raised in the review of the split-anti-seeds preregistration.

Two pieces folded in from the closed [importable-shared-glossary](../eng/importable-shared-glossary.md): the meanings of some measurements drifted between reports (ex-2.2.11 changed three), so moving each one needs a pass over how it is used now; and a report glossary should be able to list shared terms without restating them. Other terms that recur across reports ("gate", "partial band", "rung") could move at the same time, from the closed [glossary-of-preferred-terms](../eng/glossary-of-preferred-terms.md).

## Notes

**2026-10-27, definition-list notes** — Every definition list in a report now gives margin notes, so a report can write its own Measurements section as a definition list and each measurement shows beside its first use in every section. What stays open is the shared part: moving measurements that recur across reports into `docs/glossary.md`.
