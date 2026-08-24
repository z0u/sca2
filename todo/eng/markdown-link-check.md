---
status: done
tags: [ci, docs, skills]
opened: 2026-08-22
closed: 2026-08-24
---
# Nothing checks that a relative doc link or `#anchor` still resolves

The skills and `eng/` docs navigate by relative link, and several carry an `#anchor` into a named section (`storage.md#publishing-to-the-web`, `running.md#hotfix-safety-avoid-double-spending`, and two added when the mi-ni reference docs were deduplicated). A renamed heading breaks an anchor silently: the link still opens the file, just at the top, so a session following it lands somewhere plausible and reads the wrong section. A moved file is louder but no more likely to be noticed, since nothing reads these paths but agents.

The check is small — parse each `.md`, resolve every non-`http` link target relative to its own directory, and for a `#fragment` compare against the GitHub-style slugs of that file's headings. Written throwaway during the dedup pass it was about 30 lines and found nothing else broken, so the value is in the next rename rather than the current tree. Two wrinkles it needs: fenced code has to be stripped before matching (`recovery.md`'s table sits in one), and so do inline code spans, since `reports.md` deliberately shows `` `[experiment](./experiment.py)` `` as an example of a link a report author writes. Without that second strip it reports one false positive.

Natural home is a `./go` verb beside `deadcode` and `annotations`, advisory in `lint-check.yml` on the same footing. Worth deciding whether it covers `eng/` and `README.md` too, or only `.agents/skills/`: the wider net catches more, and the docs outside skills link to source files that move more often than headings do.

## Notes

**2026-08-24, tech debt** — Done: `scripts/check_md_links.py`, `./go links`, and a `Doc links` step in `lint-check.yml`. Both wrinkles the body named are real and are covered by tests; blanking a fence rather than deleting it is what keeps a later finding's line number honest.

Two decisions went against the body, and both are cheap to reverse.

**A gate, not advisory.** `dead` and `annotations` are advisory because each reports a backlog longer than one branch should carry; this one arrived at zero across the 198 files we author, so holding it there costs nothing — and advisory output is exactly what nobody reads on the day of the rename this check exists for. The repo-wide assertion also sits in `tests/test_check_md_links.py`, so `./go check` catches it before a push.

**The wider net, minus what we didn't write.** The default is every tracked `.md` except `references/` and `src/subline/`. Those two carry 21 breaks that were broken on arrival and stay broken however carefully we rename: `references/` is prior-work papers and posts imported as text whose figures never came with them, and `src/subline/` is a vendored README pointing at its own upstream tree. Naming a path explicitly still reaches them (`./go links references/`).

Scope beyond the body: reference definitions (`[n]: ./x.md`) as well as inline links, images, root-absolute `/AGENTS.md`, angle-bracketed and percent-escaped destinations, and explicit `id=`/`name=` HTML anchors. Fragments are checked only on `.md` targets, since `foo.py#L10` is a line reference. `git ls-files` is the file source, which is why vendored trees, `_site/` and agent worktrees need no rule of their own, and why the tracked `.claude/skills` symlink can't double-report `.agents/skills`.

Verified end to end rather than by unit tests alone: renaming `## Publishing to the web` and moving `eng/gc.md` produced four findings across three link styles — a skill-to-skill anchor, a sibling `./gc.md`, and a `../../eng/gc.md` from the todo tree — and reverting cleared them.
