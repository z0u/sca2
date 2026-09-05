---
status: partial
tags: [archival, versioning, security, publishing, storage]
opened: 2026-08-12
bundle: env-hardening
---

# Create hard-to-delete backups of code and experiment data

We have a few environments with write-enabled GitHub and HF tokens. If an attacker gained access to any of those environments, it could all go up in smoke. `main` has some branch protection, but I'm unsure how strong it is in practice, since the GH tokens are mostly issued for a principal who owns the repo. And the experiment data in HF is certainly deletable with the current tokens. 🚨 Obviously, don't try to test whether `main` and the HF data can be deleted.

Configure automatic indelible backups. This should be done in such a way that a _write_ token scoped to this GH repo or the HF bucket or HF dataset would be unable to delete the backups. Specifically:

- GH code `z0u/sca2`
- HF bucket `z0u/sca2-store`
- HF publish repo `z0u/sca2-pub`

This should be done in a reusable way if possible to allow a backport to `z0u/mi-ni` — even if it's just a script and some instructions (but if we can do better, that would be cool too).

## Notes

**2026-09-01, z0u** — This sounds complicated. Investigate, design, maybe prototype, but let me review the design before building.

**2026-09-05, port** — Designed and built upstream, and the runbook is now here as the `backup` skill; the reasoning is in [`eng/environments.md`](/eng/environments.md). The shape answers the "reusable" ask: a nightly Actions job in a separate repo, under a separate account, that *pulls* from the three sources into a mirror repo, a Hub dataset and a Hub bucket. No token a development environment holds can reach any of them. The job never deletes what the source still has, keeps a dropped store file for 90 days against `mini gc --store`'s 14-day grace, and never runs code fetched from the sources.

The code lives once, in the mi-ni template's `templates/backup/` (workflow, `backup.py`, restore note), and the setup fetches it from there rather than this repo carrying a second copy. So what is left here is the human half of the runbook, none of which an agent session can do: create the backup account, create the four target repos, add the trusted publishers, and run it once. Move this to `done` when that has happened and `state/last-run.json` shows a clean run.

**2026-09-05, setup** — The four targets exist (`z0u-bot/sca2-backup` for the runner and the dataset, `z0u-bot/sca2-mirror`, bucket `z0u-bot/sca2-backup-store`), both Hub targets have their trusted publisher, and the mirror PAT is issued. A `--dry-run` of `backup.py` against the real sources came back clean: code at `6a093da`, store 1954 files / 895 MB (all of it bucket → bucket, server-side; only the 246 `refs/` files, 126 kB, go into the dataset), pub 1232 commits, so the first replay takes seven nights at the default `--max-commits 200`, or a few manual dispatches. Two things to settle before the first real run. `z0u/sca2-store` is private while `z0u-bot/sca2-backup-store` and `z0u-bot/sca2-backup` are public, so the copy as configured would publish the store; and a private source needs the read-only `SOURCE_HF_TOKEN` secret, which isn't set yet. The workflow itself still has to be committed to the backup repo, which only a session holding a backup-account credential can do. The first run happened on 09-05: the code and publish legs landed (200 of 1232 commits replayed), and the store leg failed — see [the store-leg item](./backup-store-leg-private-source.md), which needs a decision before this can close.
