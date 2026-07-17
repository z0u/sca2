---
name: writing
description: |
  Writing style for composing text. Use for any prose: Markdown, GitHub issues & PRs, proposals, and technical writing such as academic papers. Improves collaboration effectiveness.
---

Write with a clear, correct, and understated style: considered but conversational, precise but not stuffy. Use the first person and voice uncertainty.

Characteristics:

- Appropriate punctuation, including colons, semicolons. Use the Oxford comma. Use double quotation marks for quotes, and single quotation marks for quotes within quotes. Do not use "smart quotes".
- Use em dashes sparingly (approx. one per page). Prefer commas or parentheses for asides, and semicolons to connect independent clauses.
- "I think...", "It seems...": hedge thoughtfully to signal confidence, but not to the point of being vague or non-committal. Calibrate to the evidence: plain declaratives for what was measured or observed; "should", "seems", "may", "perhaps" for expectations, interpretations, and mechanisms we haven't tested directly.
- Varied sentence rhythm.
- Plain connectives to continue a thought: "So", "But", "Also", "Note that", "It turns out that".
- Pose the motivating question directly, then answer it: "does zeroing that axis delete red, and only red?"
- Cross-domain analogies and examples to clarify technical concepts.
- American spelling to match the convention in scientific literature.
- High readability, with a Flesch-Kincaid grade level of around 10-12.
- Assume an intelligent audience and use precise language. See _Concision_ for more on this.
- Alt text for all images: aids vision-impaired people and LLMs alike (see the
  alt-text skill).

Anti-patterns:

- ~~Business jargon and bombast~~. Avoid buzzwords, corporate-speak, and baseball metaphors.
- ~~Adversarial framing~~. Don't cast the object of study as an opponent to beat, convict, or punish. No combat metaphors (*casualties*, *fighting back*, *hauled back*, *the fight is not free*), no crime or interrogation framing (*the culprit*, *names the suspect*, *guilty*/*innocent*, *the hypothesis is dead*), no coercion or punishment (*making composition pay*, *make the model pay for*, *punish memorization*). A result can be vivid without being violent — describe what happened, not who won. Prefer plain cause: "the LR peak was the cause" over "the LR peak was the culprit".
- ~~All the lists~~. Mostly use paragraphs, but use lists sparingly when they are the clearest way to present the information.
- ~~Heavy-handed transitions~~. Avoid "Furthermore", "In conclusion", "The honest answer is", etc. Just continue the thought.
- ~~Excessive use of em dashes~~. Prefer other punctuation.
- ~~Excessive use of bold and italic text~~. List items should not be bolded. 1-2 callouts (bold) per page or section; italics only for references & borrowed words, or when it's truly unobvious which phrase should be emphasized (usually the reader can infer without it).
- ~~Contrastive conclusions~~. Avoid "... is a feature, not a confound", etc. Instead of `A, ~A`, just say `A`.
- ~~Verdict kickers~~. Don't close a passage with a punchy fragment that passes judgment ("A clean negative: the boring fix stands."). Readers pattern-match these as AI and stop reading. State the consequence as an ordinary sentence: "It was a clean negative result, so we're keeping the static schedule."
- ~~Evocative headings~~. Headings name what the section contains, not what it means: "Training data", "Findings" — not "What the model sees", "What this settles".

### Register by document type

- Experiment reports sit between a technical blog post and documentation: relaxed but focused, contractions used sparingly, the occasional dry aside, comfortable acknowledging "this might be wrong". Enthusiasm is fine when a result earns it, but plainly stated rather than hyped.
- Reference documentation: clear and concise, no jokes or asides, but still conversational.
- Papers: slightly more formal but still readable. "We" for the work itself, fewer contractions, no jokes, no stiffness.
- Issues, PRs, and chat: more casual; direct questions and short paragraphs.

### Pacing and structure

- One move per paragraph. Prefer several short paragraphs over one dense one; don't pack too much into a single sentence held together by dashes.
- Declare intent, then execute. "Let's get a baseline before we anchor anything", "We will measure two things" — say what's coming in plain first-person, rather than framing it nominally after the fact.
- State results where they first become visible. When introducing a figure whose outcome is known, say the outcome ("...and we find the model fails this task").
- Keep commentary adjacent to what it explains. Introduce a figure with a sentence or two (what's plotted, how to read it), show it, then interpret.
- A small table for enumerable examples the text refers back to; inline numbering for alternatives ("answerable two ways: 1. recall..., or 2. composition...").
- Cut tangents. A detail that serves another section belongs there or nowhere; use cross-references and forward-references sparingly.

### Concision and clarity

We respect the intelligence of our readers, but we also respect their time. We aim for high readability because it improves our chances of conveying complex technical ideas. We accept that writing takes time, and spend that time to make the writing as clear and concise as possible. We spend the effort so our audience doesn't have to.

If you will excuse the irony of belaboring this point: concision is _so_ important. Allow the reader to infer information from figures, context, and a few exemplar results. Our job is to draw their attention to key details and to offer real, thoughtful insight.

## Workflow

Start by planning in your own words. Then, revise the draft to align with this style and tone. Use the characteristics and anti-patterns as a guide. Finally, review the draft for clarity, correctness, and consistency.

First drafts are invariably too verbose, so always edit them down after writing them. You should use tools to help: 1. Write in a file and save it. Include markers if the document has many sections. 2. Count length: `wc -wml file` for the whole file, and `.agents/skills/writing/scripts/block-wc file` for per-block word counts (sorted so the verbose blocks surface first) to see where to cut. 3. Edit the file to reduce length, aiming for > 20% reduction. Stop when you can't remove more without losing load-bearing information. 4. Remove the markers with a tool, then confirm none survived (e.g. `rg 'R:|/R' file` returns nothing).

Example section/block markers:

```md
<!--R:short-description-->

The text.

<!--/R-->
```

When in doubt about how to phrase something, err on the side of clarity and simplicity. Avoid jargon and complex sentence structures unless they are necessary to convey the technical content accurately.

## Markdown

Never hard-wrap a line inside an inline code span or math expression. A wrapped
span can start the next line with block syntax — a hex code like `#f78` at the
start of a line renders as a heading — and some renderers break the span
entirely. Rewrap the surrounding prose so the whole span sits on one line.

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
