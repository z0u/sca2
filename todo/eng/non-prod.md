---
status: partial
tags: [devops, security, publishing, storage]
opened: 2026-09-05
bundle: env-hardening
---

# Create a non-prod environment

We push to a single bucket and dataset repo. A second pair, for development, would keep work *on* the storage and publishing machinery off production: the `hf`-marked integration tests (which write probe commits into `z0u/sca2-pub` on every run), a `sync_export` or `gc --store` change mid-development, an agent building a store feature.

## Notes

**2026-09-05, port** — The mechanism is in, ported from mi-ni. A `[tool.mini.profiles.<name>]` table names a second pair and `MINI_PROFILE` selects it; the two storage keys come from the profile alone, so a half-written one falls to the local store rather than reaching production, and `app`/`env`/`region` are inherited so the compute is unchanged. Under a profile `./go publish` writes its pins to a gitignored `.mini/publish.<profile>.lock`, leaving `publish.lock` as the record CI and the site read. `mini run --app modal` forwards the profile to its workers, and `tests/mini/test_hf_store.py` picks a `dev` profile itself whenever one is defined. The storage reference in the `mi-ni` skill describes it; [`eng/environments.md`](/eng/environments.md) has the reasoning.

What is left is the human half: create `z0u/sca2-store-dev` and `z0u/sca2-pub-dev`, add the `[tool.mini.profiles.dev]` table, mint a dev-only token, and set `MINI_PROFILE=dev` (or the two env vars, for an environment configured by variable) in the engineering environments. The `storage-envs` skill is the runbook. Until that happens the profile machinery is inert and every session uses production, as before.
