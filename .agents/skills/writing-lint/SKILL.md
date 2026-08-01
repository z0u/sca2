---
name: writing-lint
description: |
  Prose reduction. Removes over-explanation and deduplicates text. Does not
  check numbers or claims. Verify correctness yourself afterwards.
argument-hint: <document>
context: fork
model: sonnet
effort: low
tools: Read, Edit, Bash
---

We are linting this document (to remove fluff!) We are not checking numbers,
claims or anything else. Just linting.

Workflow:

1. Convert the document (or section) to plain text or Markdown first. Call it
   something like `fluffy.md`
2. Make another copy of that, like `clean.md`
3. Give `clean.md` to the `text-linter` agent. If it should focus on a small
   part of it, tell it which part. Don't give it any other context. The agent will remove unnecessary content.
4. When it's done, compare `fluffy.md` with `clean.md`. Use a word-level diff,
   because a line diff of reflowed prose just says "this paragraph changed":

   ```bash
   git diff --no-index --word-diff=plain --unified=0 fluffy.md clean.md
   ```

5. Manually port the changes back into the original document. Make sure it's
   syntactically correct, but don't check for content correctness (I will do that).
6. Check that every template expression (`{...}` in an f-string) still present
   in `clean.md` as a literal value is still present in the source as an
   expression. The linter sees `0.25`, not `{ex.MEAN_ALIGN_PARTIAL:g}`, so a
   dropped expression is easy to miss when porting back. List any that went.
7. End with a concise report of the flavor of the changes (not details; those
   will be self-evident).

The linter is deliberately given no context, so it can't know which details
carry weight. Expect to restore a few (named thresholds, scope qualifiers,
the reason behind a verdict, etc.) and say which in the report.

Marimo notebooks should be converted with:

```bash
uv run marimo-md-export report.py fluffy.md  # Runs the notebook to interpolate strings
uv run scripts/clean_marimo_md.py fluffy.md  # Externalizes images, fixes stuff
```

If the conversion _fails_, then stop and escalate.
