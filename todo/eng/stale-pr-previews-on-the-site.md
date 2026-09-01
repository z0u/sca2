---
status: partial
tags: [ci, publishing]
opened: 2026-08-31
---
# Seven closed PRs still have their previews on the site

Found while checking that the first real `gh-pages` prune had left the served tree intact — it had, and this was sitting in it.

`pr-preview.yml` tears a preview down when its PR closes, and mostly does: of 112 `pr-preview/pr-N/` directories on the branch, 103 hold nothing but a lone `.nojekyll` stub, which is what a completed teardown leaves. Seven hold a whole site build for a PR that closed weeks ago — **123, 112, 65, 59, 51, 42, 37** — and one more pair (127, 131) is legitimately live. Measured 2026-08-31: the preview tree is 30.8 MiB of the site's 36.7 MiB, and the seven stale ones are about 25 MiB of that. So roughly two thirds of what the site serves is previews of work that has already landed.

The bytes are the smaller half. Each is a reachable URL serving a report at a revision nobody promoted, and this is a project where the figures are the argument — a link to `pr-preview/pr-65/` shows science that was superseded, with nothing on the page saying so.

**Why the teardown didn't run, established 2026-09-01 from the workflow run history.** Two causes, and neither is the failed or cancelled run the shape suggests.

*The teardown ran and was overwritten.* On #59 the teardown removed the preview at 02:06:24–29 and a `synchronize` build that had started before the merge deployed it straight back at 02:06:35–40; #65 is the same pair, two minutes apart. Both runs succeeded — the build was simply last. The two overlapped, so the `github.ref` the `closed` event supplied was not the one its own `synchronize` runs used, which means a PR's builds and its teardown sat in different concurrency groups. Fixed: the group is now keyed on `github.event.number` with `cancel-in-progress: true`, which puts every event for one PR in one group and leaves the close as the last writer.

*No teardown run was created at all.* #123, #51 and #42 each have exactly one `pr-preview.yml` run, the `opened` one. Their `closed` event scheduled nothing — no run to read, no failure to fix, and no guard that could live in the workflow. All three closed unmerged, which is the only property they share; the one unmerged PR that isn't stale (#50) did get a teardown run, two seconds after its close, so "unmerged" is a correlate rather than the mechanism. Not chased further, because the answer wouldn't change the fix.

With the resurrection already recorded in [`scripts/prune_gh_pages.py`](/scripts/prune_gh_pages.py) — a production deploy that beats a teardown, which has already run and won't fire again — that is three ways to leak a preview and one place to catch all of them. So the guard is reconciliation rather than another attempt to make one event fire: [`scripts/stale_previews.py`](/scripts/stale_previews.py) compares what the branch serves against which PRs are still open. Run report-only against the real branch it names exactly the seven above, from the tree rather than from this list.

## Notes

**2026-09-01, tech debt** — Cause established and both guards built; what's left is the deletion itself, which is why this is `partial` rather than `done`. The sweep is dispatch-only and report-only by default (`./go stale-previews`, or the **Preview Sweep** workflow in the Actions tab with `apply` ticked), so removing the seven is one run you make when you're happy with the list. It deletes from a published site, which the item said wants a decision, and this session had no way to watch the result. Two things worth deciding at the same time: whether the sweep goes on a schedule once you've seen it judge correctly a few times, and whether the 103 `.nojekyll` stubs are worth collecting too — the sweep leaves them, on the grounds that they cost a few bytes each and clearing them is the preview action's business.
