---
status: open
tags: [docs, skills]
opened: 2026-08-02
---
# Duplication across the mi-ni reference docs

Found while running the writing/text-lint passes over `.agents/skills/`.

The cache-friendliness guidance (narrow inputs, cheap `main`, folded RNG seeds, forcing a re-run) is near-verbatim in both `authoring.md` and `memoization.md`; the `Experiment(deps=[...])` note is in both `storage.md` and `running.md`; "failure is terminal by design" is in both `recovery.md` and `running.md`; and `memoization.md` points at `recovery.md` twice saying much the same thing. Each doc reads fine alone, so this only costs when they drift. Also: `reports.md`'s `### Simplification pass` is an h3 under the h1 with no h2 above it, and is arguably writing-workflow guidance that belongs with the `writing`/`text-lint` skills rather than in a publishing reference.
