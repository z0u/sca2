---
status: done
tags: [ci, publishing]
closed: 2026-08-30
---
# `gh-pages` branch pruning

This is where we publish reports to; see `.github/workflows/publish-docs.yml`. Currently the branch history is linear, and contains a commit for every preview build and every build on `main`. We should compact it, probably on every `main` build. How many commits to keep? Unsure, maybe ~1 week, maybe only those from `main` and currently-open branches. Maybe the Action that we use supports this out of the box.

Measured 2026-08-10, so the growth rate is on record rather than guessed at: 239 commits since the branch opened on 2026-07-14 (~8/day), 7.5 MiB of history against `main`'s 2.2 MiB, and a 17.9 MiB working tree across 177 files. Nothing to act on yet at that size — the note is when, not whether.

The action does have it out of the box: `single-commit: true` on `JamesIves/github-pages-deploy-action`, which the docs are blunt about — "using this option will also cause any existing history to be wiped from the deployment branch". Two interactions to settle before reaching for it, and the docs cover neither. It has to force-push, where the production deploy runs `force: false` precisely so it rebases onto a concurrent preview deploy rather than dropping it; and the PR previews live in the same tree under `clean-exclude: pr-preview/`, so their *files* should survive into the single commit while a preview deploy racing the force push would not. That race is rare and self-healing (the next preview deploy restores it), but it is the thing to check rather than assume. A `main`-only prune that keeps a window of commits avoids both by never rewriting what a preview is standing on.

If it's easy to do, see if there's a way to publish tags or releases too. E.g. d2.1 marks the nominal end of that deliverable, and it's marked as an indelible release on GH. It would be nice if the tag became a permalink for the site at that point in time. We don't have many tags, so it shouldn't grow too fast. If it adds significant complexity to this pruning piece, it could be split into a new todo.

## Notes

**2026-08-30, closing** — Done as the windowed `main`-only prune, not `single-commit`: `scripts/prune_gh_pages.py`, run as the last step of `publish-docs.yml`. It keeps the newest 40 commits and only acts once the branch passes 120, so a rewrite costs a force-push every few weeks rather than one per build. The force-push carries `--force-with-lease` against the tip it fetched, so a preview deploy landing mid-prune wins and the prune waits for the next build — the race the item flagged, handled by yielding rather than by timing. Rationale is in the script's docstring and `eng/publishing.md`.

Measured again before building: 340 commits, of which 296 were preview churn (`Deploy preview for PR N` / `Remove preview for PR N`) against ~43 production deploys — so the growth is dominated by previews, and a window measured in commits rather than days tracks it better than the earlier ~8/day estimate suggested.

The tag-permalink rider is split out as [`site-permalinks-for-tags`](./site-permalinks-for-tags.md). It needs its own deploy path, and pruning doesn't get in its way: a prune copies the tip's tree verbatim, so a `v/<tag>/` subtree would survive one the same way the preview trees do.
