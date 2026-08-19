---
status: open
tags: [ci]
opened: 2026-08-19
---
# Two pinned Actions still target Node 20

Every workflow run now ends with a warning: "Node.js 20 is deprecated. The following actions target Node.js 20 but are being forced to run on Node.js 24: `actions/checkout@v4`, `astral-sh/setup-uv@v6`." The runner is already substituting Node 24, so nothing is broken today — what expires is the substitution, and the warning is what's left of the notice period ([GitHub changelog, 2025-09-19](https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/)).

Both pins appear three times each, once per workflow (`lint-check.yml`, `pr-preview.yml`, `publish-docs.yml`), so this is six lines and a green run to confirm. `rossjrw/pr-preview-action@v1` and `JamesIves/github-pages-deploy-action@v4` are not named in the warning.

Noticed while reading a CI log for something else, from a session whose GitHub scope was this repo only — so the major versions to move to weren't checked against the upstream release pages. Do that first rather than assuming the next integer, and prefer landing it in a PR that already touches CI, since the confirmation is one full run either way.
