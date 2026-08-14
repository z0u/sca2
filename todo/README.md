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
./go todo --priority           # the shortlist: what to pick up next
./go todo --json               # for scripts
```

The default view shows `open` and `partial` only. Settled work and findings are still there and `--status` reaches them.

Shortlisted items carry a `!` beside their status mark and sort to the head of their group, so a plain `./go todo` shows the ranking without being asked.

To search bodies rather than titles, use `rg` over the directory — a match carries its own filename and boundary, which a single long list can't give you. Paragraphs here are one line each and soft-wrapped, so print files or windows rather than whole lines (the root [`AGENTS.md`](../AGENTS.md) has the reasoning):

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
priority: high         # optional — the shortlist, capped at six live items
---
# A title, as the first heading

The body, as ordinary prose. Paragraphs rather than one long line: the file is a
document, not a bullet, so there is room to say why.
```

`finding` is for established knowledge rather than work — a result to carry forward, which has no completion state to track.

## Priority

`priority: high` is the only level, and absence is the default — a second rung would just be somewhere for indecision to live. At most six live items may carry it, and `./go check --test` fails when a seventh appears.

The cap is the point. Priority schemes decay when promotion is free: everything drifts upward until the top rung holds the backlog and the label stops carrying information. Here a seventh item costs a demotion, so the shortlist stays a claim about what to do next rather than a list of things that seemed important once. Settled items don't count against it, so a closed item can keep its old priority as history.

Settled items stay where they are rather than moving to an archive: the default view already hides them, and leaving them put keeps their inbound links and file history intact. Several are kept deliberately for their measurements.

`./go check --test` includes a test case that validates every header, so a malformed item fails the suite; `./go todo --check` is the same validation on its own, for a quick pass after editing one.

## Notes from a routine

An agent that reads the backlog on a schedule — the housekeeping routine, say — often has something to pass on that isn't a change to the item: evidence it checked, a reason the item looks closeable, a dependency it noticed. Those go in a `## Notes` section at the foot of the item's own body, dated and signed:

```markdown
## Notes

**2026-08-14, housekeeping** — ex-2.1.6 finished on 08-11 and the margins are in the store, so this looks closeable. I couldn't confirm the anti-subspace claim from the report alone.
```

The item's file is the right home because a note lives exactly as long as what it describes: close the item and the note goes with it, so there is no separate log to prune. It also travels with any branch touching the item, and `rg` over `todo/` finds it the same way it finds everything else.

One rule keeps this from accumulating: **an author replaces its own previous note rather than appending one.** A routine leaves at most one note per item, superseded in place, so a daily pass shows up as a few changed lines. Notes between people are ordinary prose in the body, as they have always been.

A ranking is not a note. "What we should do next" belongs in `priority`, where it is capped and reviewable, rather than restated in prose on six different items.

## GitHub issues

Issues are for capture: something noticed away from a checkout, or raised by someone without commit access. Transcribe one into a file and close it, so each item has a single home — an issue and a file describing the same work will drift, and the file is the copy that travels with the branch.
