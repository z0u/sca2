# Backlogs

Three backlogs, one file per item: [`eng/`](./eng/) for infrastructure and tooling, [`science/`](./science/) for experiment questions and findings, [`style/`](./style/) for text and visual improvements. Each set has its own README saying what belongs in it.

One file per item is what keeps two branches from colliding — adding items adds files rather than editing the same lines — and it gives every item a stable name that a commit, a PR, or an `eng/` doc can point at.

## Reading them

`./go todo` prints the index, generated on demand rather than committed, so there is no second copy to fall out of step with the files.

```bash
./go todo                      # every live item, all three sets
./go todo science              # one set (eng | science | style)
./go todo --tag cli --tag storage   # conjunctive: items carrying both
./go todo --status finding     # settled results worth carrying forward
./go todo --bundle cli-devx    # one session's worth
./go todo --json               # for scripts
```

The default view shows `open` and `partial` only. Settled work and findings are still there and `--status` reaches them.

To search bodies rather than titles, use `rg` over the directory — a match carries its own filename and boundary, which a single long list can't give you. Paragraphs here are one line each and soft-wrapped, so print files or windows rather than whole lines (`AGENTS.md` has the reasoning):

```bash
rg -l anneal todo/science/           # the files, then read the ones you want
rg -no '.{0,55}anneal.{0,55}' todo/  # a window around each match
```

## Writing one

Front matter, a `# Title`, then prose. `status` is the only required key.

```markdown
---
status: open           # open | partial | done | finding
tags: [cli, storage]   # optional
opened: 2026-08-13     # optional — many inherited items carry no date
closed: 2026-08-14     # optional, and only on a done item
bundle: cli-devx       # optional — groups items one dev session should take together
---
# A title, as the first heading

The body, as ordinary prose. Paragraphs rather than one long line: the file is a
document, not a bullet, so there is room to say why.
```

`finding` is for established knowledge rather than work — a result to carry forward, which has no completion state to track.

Settled items stay where they are rather than moving to an archive: the default view already hides them, and leaving them put keeps their inbound links and file history intact. Several are kept deliberately for their measurements.

`./go check --lint` validates every header, so a malformed item fails the same gate as a lint error.
