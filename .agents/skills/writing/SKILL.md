---
name: writing
description: |
  Writing style for composing text. Use for any prose: Markdown, GitHub issues & PRs, proposals, and technical writing such as academic papers. Improves collaboration effectiveness.
---

Write with a clear, correct, and understated style: considered but conversational, precise but not stuffy. Use the first person and voice uncertainty.

Characteristics:

- Oxford comma; straight (not "smart") quotes, em-dashes sparingly (prefer other punctuation).
- Calibrate confidence to the evidence: plain declaratives for what was measured or observed.
- Plain connectives to continue a thought: "So", "But",  etc.
- American spelling and double quotes to match the convention in scientific literature.
- Alt text for all images: aids vision-impaired people and LLMs alike (see the alt-text skill).

Anti-patterns:

- Business jargon and bombast, evocative headings, narrated paragraph openers
- Adversarial or aggressive framing
- Excessive use of lists, em dashes, and bold and italic text
- Contrastive conclusions and verdict kickers
- Committing to unplanned future work
- Possessives on abstract terms

### Pacing and structure

- Short paragraphs
- Say what's coming up to prime the reader
- State results where they first become visible
- Keep commentary adjacent to what it explains; introduce a figure in a paragraph before, put interpretation afterward rather than in the caption
- Cut tangents; don't repeat information in several sections

### Clarity

Write as though explaining to an intelligent person with technical skill who is _not_ an expert in machine learning. Allow the reader to infer information from figures, context, and a few exemplar results; draw their attention to key details.

Concision is not density. Cutting words and lowering reader effort are different goals, and sometimes they pull against each other. A sentence may be concise and still hard to read because it stacks several ideas, folds a definition into an appositive, or hides a verb inside a noun phrase.

So write in plain English, with respect for the reader's intelligence, but also for their time.

---

## Markdown

Use sentence case for headings and descriptive lists.

```patch
- # Experiment Design
-
-   - **Foo Bar:** baz
+ # Experiment design
+
+   - Foo bar: baz
```

Prefer paragraphs for nuanced or complex explanations; use lists for
summarizing steps, or when clarity would genuinely benefit from structure —
and use them sparingly.

Use italics for concepts like _red_. Never use bold and italics for emphasis,
because they are distracting to read. The user will add them if necessary.

In Marimo, consider using `details` markup for asides, which render unobtrusively.

```py
mo.md("""
Main content.

/// details | Title
Some backstory.
///
""")
```

Don't hard-wrap a line inside an inline code span or math expression. A wrapped
span can start the next line with block syntax — a hex code like `#f78` at the
start of a line renders as a heading — and some renderers break the span
entirely. Rewrap the surrounding prose so the whole span sits on one line.

```patch
      mo.md(
-       "Sometimes we write Markdown in Python, e.g. when working in a Marimo notebook. "
-       "In that case, prefer multiline strings rather than using one string per "
-       "hard-wrapped line. Use dedent and f-strings as needed."
+       """
+     Sometimes we write Markdown in Python, e.g. when working in a Marimo notebook.
+     In that case, prefer multiline strings rather than using one string per
+     hard-wrapped line. Use dedent and f-strings as needed."""
      )
```

Multiline strings are also supported by the `@themed(..., alt_text=..., caption=...)` decorator (see `figure-style`).
