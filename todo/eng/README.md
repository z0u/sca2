# Engineering todo

Deferred infrastructure work: tooling, storage, publishing, CLI, and the `mini` library. One item per file, so two branches adding items add two files instead of colliding on one. Science questions and experiment findings live in [`todo-science.md`](../../todo-science.md); text and visual improvements in [`todo-style.md`](../../todo-style.md).

Durable design rationale and recorded decisions live in [`eng/`](../../eng/README.md) — an item that grows into a design doc belongs there, with a line here pointing at it. Work tracked upstream in the `mini` library carries a [z0u/mi-ni](https://github.com/z0u/mi-ni/issues) link; issue numbers here are always qualified, because a bare `#38` resolves against *this* repo and lands on an unrelated pull request.

## Reading it

`./go todo` prints the index — grouped by bundle, done items filtered out. It is generated on demand rather than committed, so there is no second copy to fall out of step with the files.

```bash
./go todo                          # everything open
./go todo --tag cli --tag storage  # conjunctive: items carrying both
./go todo --status done            # what shipped
./go todo --bundle cli-devx        # one session's worth
./go todo --json                   # for scripts
```

To search bodies rather than titles, `rg` over the directory: a match carries its own filename and boundary, which is what a single long list can't give you.

## Writing one

A file is front matter, a `# Title`, and prose. `status` is `open`, `partial`, or `done`; `tags` is a non-empty inline list; `opened` is optional (some inherited items carry no date) and `closed` is required once an item is done. `bundle` is optional and groups items a single dev session should take together.

```markdown
---
status: open
tags: [cli, storage]
opened: 2026-08-13
bundle: cli-devx
---
# A title, as the first heading

The body, as ordinary prose. Paragraphs rather than one long line — the file is a
document, not a bullet, so there is room to say why.
```

Done items stay where they are rather than moving to an archive: the default view already hides them, and leaving them put keeps their inbound links and file history intact. Several are kept deliberately for their measurements.

`./go check --lint` validates every header, so a malformed item fails the same gate as a lint error.
