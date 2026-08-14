# Backlogs

Three backlogs, one file per item: [`eng/`](./eng/) for infrastructure and tooling, [`science/`](./science/) for experiment questions and findings, [`style/`](./style/) for text and visual improvements. Each set has its own README saying what belongs in it.

One file per item keeps two branches from colliding, since adding items adds files rather than editing the same lines. It also gives every item a stable name that a commit, a PR, or an `eng/` doc can point at.

## Reading

`./go todo` prints the index, generated on demand rather than committed, so there is no second copy to fall out of step with the files.

```bash
./go todo                      # every live item, all three sets
./go todo science              # one set (eng | science | style)
./go todo --tag cli --tag storage   # conjunctive: items carrying both
./go todo --status finding      # settled results worth keeping
./go todo --bundle cli-devx    # one session's worth
./go todo --priority           # the shortlist: what to pick up next
./go todo --check              # validate frontmatter and priority budget
./go todo --json               # for scripts
```

The default view shows `open` and `partial` only. Settled work and findings are still there and `--status` reaches them. Shortlisted items take a `!` beside their status mark and sort to the head of their group, so a plain `./go todo` shows the ranking without being asked.

To search bodies rather than titles, use `rg` over the directory, so each match comes with its own filename. Reminder that paragraphs are one line each and soft-wrapped, so print files or windows rather than whole lines (see `../AGENTS.md`).

## Writing

Front matter, a `# Title`, then prose. `status` is the only required key.

```markdown
---
status: open           # open | partial | done | finding
tags: [cli, storage]   # optional
opened: 2026-08-13     # optional; many inherited items have none
closed: 2026-08-14     # optional, and only on a done item
bundle: cli-devx       # optional; groups items one dev session should take together
priority: high         # optional; the shortlist, capped at six live items
---
# A title, as the first heading

The body, as ordinary prose. The file is a document, so there is room to say why. One line per paragraph, soft-wrapped.
```

`finding` is for established knowledge rather than work: a result worth keeping.

`priority: high` is the only level, and absence is the default. At most six live items may have it: priority schemes decay when promotion is free. A seventh item requires demoting something else. Settled items don't count against the cap.

Settled items stay where they are: the default view already hides them, and leaving them put keeps their inbound links and file history intact.

## PM notes

If you have context to pass on that isn't a change to the item (evidence checked, a reason the item looks closeable, a blocker) — put that in a `## Notes` section at the foot of the item body, dated and signed:

```markdown
## Notes

**2026-08-14, housekeeping** — ex-2.1.6 finished on 08-11 and the margins are in the store, so this looks closeable. I couldn't confirm the anti-subspace claim from the report alone.
```

Try not to let this grow without bound: edit your own notes, and tidy up others.
