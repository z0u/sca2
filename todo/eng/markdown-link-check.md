---
status: open
tags: [ci, docs, skills]
opened: 2026-08-22
---
# Nothing checks that a relative doc link or `#anchor` still resolves

The skills and `eng/` docs navigate by relative link, and several carry an `#anchor` into a named section (`storage.md#publishing-to-the-web`, `running.md#hotfix-safety-avoid-double-spending`, and two added when the mi-ni reference docs were deduplicated). A renamed heading breaks an anchor silently: the link still opens the file, just at the top, so a session following it lands somewhere plausible and reads the wrong section. A moved file is louder but no more likely to be noticed, since nothing reads these paths but agents.

The check is small — parse each `.md`, resolve every non-`http` link target relative to its own directory, and for a `#fragment` compare against the GitHub-style slugs of that file's headings. Written throwaway during the dedup pass it was about 30 lines and found nothing else broken, so the value is in the next rename rather than the current tree. Two wrinkles it needs: fenced code has to be stripped before matching (`recovery.md`'s table sits in one), and so do inline code spans, since `reports.md` deliberately shows `` `[experiment](./experiment.py)` `` as an example of a link a report author writes. Without that second strip it reports one false positive.

Natural home is a `./go` verb beside `deadcode` and `annotations`, advisory in `lint-check.yml` on the same footing. Worth deciding whether it covers `eng/` and `README.md` too, or only `.agents/skills/`: the wider net catches more, and the docs outside skills link to source files that move more often than headings do.
