---
name: writing
description: |
  Writing style for composing text. Use for any prose: Markdown, GitHub issues & PRs, proposals, and technical writing such as academic papers. Improves collaboration effectiveness.
---

Write with a clear, correct, and understated style: considered but conversational, precise but not stuffy. Use the first person and voice uncertainty.

Characteristics:

- Oxford comma; straight (not "smart") quotes, em-dashes sparingly (prefer other punctuation).
- Calibrate confidence to the evidence: plain declaratives for what was measured or observed. Hedges and tense carry how strongly a claim is made, so they survive editing: "would start to matter" is weaker than "matters".
- Verbs agree with which direction is good. Containment, decay and retention run downward, so "no condition reaches 0.1" reads backwards; "falls to" does not.
- A term the reader may not know gets one short plain-English gloss on first use, in a footnote or a `details` block. Figure captions stay pure legend.
- Plain connectives to continue a thought: "So", "But", etc.
- American spelling and double quotes to match the convention in scientific literature.
- Alt text for all images: aids vision-impaired people and LLMs alike (see the alt-text skill).

Anti-patterns:

- Business jargon and bombast, evocative headings, narrated paragraph openers
- Adversarial or aggressive framing
- Excessive use of lists, em dashes, and bold and italic text
- Contrastive conclusions and verdict kickers
- Committing to unplanned future work
- Possessives on abstract terms

## Pacing and structure

- Short paragraphs
- Say what's coming up to prime the reader
- State results where they first become visible
- Keep commentary adjacent to what it explains; introduce a figure in a paragraph before, put interpretation afterward rather than in the caption
- Cut tangents; don't repeat information in several sections

## Clarity

Write as though explaining to an intelligent person with technical skill who is _not_ an expert in machine learning. Allow the reader to infer information from figures, context, and a few exemplar results; draw their attention to key details.

So write in plain English, with respect for the reader's intelligence.

## Formatting and structure

Use normal-weight sentence case for headings and descriptive lists.

```patch
- # Experiment Design
-
- - **Foo Bar:** baz
+ # Experiment design
+
+ - Foo bar: baz
```

Prefer paragraphs for nuanced or complex explanations; use lists for summarizing
steps, or when clarity would genuinely benefit from structure, but use them
sparingly.

Emphasis can be distracting to read, so follow these guidelines for where and
how often to use it:

- Use italics for concepts like _red_. When: usually.
- Use italics for named terms like _anchor_. When: on the first use in a
  section, and then if the role of the word would otherwise be ambiguous, e.g.
  to distinguish "anchor" the regularizer term from "the anchor point".
- Never use bold or italics for other emphasis, because they are distracting to
  read. The user will add them if necessary.

For syntax conventions, refer to the relevant `style-*` skill.
