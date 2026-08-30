---
status: open
tags: [archival, versioning, security]
opened: 2026-08-12
---

# Create hard-to-delete backups of code and experiment data

We have a few environments with write-enabled GitHub and HF tokens. If an attacker gained access to any of those environments, it could all go up in smoke. `main` has some branch protection, but I'm unsure how strong it is in practice, since the GH tokens are mostly issued for a principal who owns the repo. And the experiment data in HF is certainly deletable with the current tokens. 🚨 Obviously, don't try to test whether `main` and the HF data can be deleted.

Configure automatic indelible backups. This should be done in such a way that a _write_ token scoped to this GH repo or the HF bucket or HF dataset would be unable to delete the backups. Specifically:

- GH code `z0u/sca2`
- HF bucket `z0u/sca2-store`
- HF publish repo `z0u/sca2-pub`

This should be done in a reusable way if possible to allow a backport to `z0u/mi-ni` — even if it's just a script and some instructions (but if we can do better, that would be cool too).
