---
status: open
tags: [ci, publishing]
---
# `gh-pages` branch pruning

This is where we publish reports to; see `.github/workflows/publish-docs.yml`. Currently the branch history is linear, and contains a commit for every preview build and every build on `main`. We should compact it, probably on every `main` build. How many commits to keep? Unsure, maybe ~1 week, maybe only those from `main` and currently-open branches. Maybe the Action that we use supports this out of the box.

Measured 2026-08-10, so the growth rate is on record rather than guessed at: 239 commits since the branch opened on 2026-07-14 (~8/day), 7.5 MiB of history against `main`'s 2.2 MiB, and a 17.9 MiB working tree across 177 files. Nothing to act on yet at that size — the note is when, not whether.

The action does have it out of the box: `single-commit: true` on `JamesIves/github-pages-deploy-action`, which the docs are blunt about — "using this option will also cause any existing history to be wiped from the deployment branch". Two interactions to settle before reaching for it, and the docs cover neither. It has to force-push, where the production deploy runs `force: false` precisely so it rebases onto a concurrent preview deploy rather than dropping it; and the PR previews live in the same tree under `clean-exclude: pr-preview/`, so their *files* should survive into the single commit while a preview deploy racing the force push would not. That race is rare and self-healing (the next preview deploy restores it), but it is the thing to check rather than assume. A `main`-only prune that keeps a window of commits avoids both by never rewriting what a preview is standing on.
