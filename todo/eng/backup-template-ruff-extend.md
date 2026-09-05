---
status: open
tags: [tooling, storage]
opened: 2026-09-05
bundle: backup-template
---
# The backup template's `ruff.toml` extends a file the backup repo doesn't have

`templates/backup/ruff.toml` in mi-ni starts with `extend = "../../pyproject.toml"`, which resolves inside mi-ni, where the template sits two levels down. Installed, the file lands at the root of the backup repo, where that path points nowhere and ruff exits with an error before it reads `target-version`. Nothing in the backup repo runs ruff today, so it costs nothing until someone opens the repo in an editor with ruff configured, or adds a lint step; the `py312` pin it exists to carry is the part that would then be missed.

The fix is upstream, in `z0u/mi-ni`: make the file self-contained (`target-version = "py312"` and whatever formatting settings the template wants) and reword the comment, which currently says formatting is inherited. In the sca2 install we've already dropped the `extend` line by hand, so the two copies differ until mi-ni catches up.
