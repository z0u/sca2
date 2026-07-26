---
name: writing
description: |
  Writing style for composing text. Use for any prose: Markdown, GitHub issues & PRs, proposals, and technical writing such as academic papers. Improves collaboration effectiveness.
---

Write with a clear, correct, and understated style: considered but conversational, precise but not stuffy. Use the first person and voice uncertainty.

Characteristics:

- Appropriate punctuation, including colons, semicolons. Use the Oxford comma. Use double quotation marks for quotes, and single quotation marks for quotes within quotes. Do not use "smart quotes".
- Use em dashes sparingly (approx. one per page). Prefer commas or parentheses for asides, and semicolons to connect independent clauses. A pile-up of dashes usually means a sentence is carrying too many ideas. The fix is to split it into separate sentences.
- "I think...", "It seems...": signal confidence, but not to the point of being vague or non-committal. Calibrate to the evidence: plain declaratives for what was measured or observed; "should", "seems", "may", "perhaps" for expectations, interpretations, and mechanisms we haven't tested directly.
- Varied sentence rhythm.
- Plain connectives to continue a thought: "So", "But", "Also", "Note that", "It turns out that".
- Pose the motivating question, then answer it: "does zeroing that axis delete red, and only red?"
- Cross-domain analogies and examples to clarify technical concepts.
- American spelling to match the convention in scientific literature.
- High readability, with a Flesch-Kincaid grade level of around 10-12.
- Assume an intelligent audience and use precise language. See _Concision_ for more on this.
- Alt text for all images: aids vision-impaired people and LLMs alike (see the alt-text skill).

Anti-patterns:

- ~~Business jargon and bombast~~. Avoid buzzwords, corporate-speak, and baseball metaphors.
- ~~Adversarial framing~~. Don't cast the object of study as an opponent to beat, convict, or punish. No combat metaphors (*casualties*, *fighting back*, *hauled back*, *the fight is not free*), no crime or interrogation framing (*the culprit*, *names the suspect*, *guilty*/*innocent*, *the hypothesis is dead*), no coercion or punishment (*making composition pay*, *make the model pay for*, *punish memorization*). A result can be vivid without being violent; describe what happened, not who won. Prefer plain cause: "the LR peak was the cause" over "the LR peak was the culprit".
- ~~All the lists~~. Mostly use paragraphs, but use lists sparingly when they are the clearest way to present the information.
- ~~Heavy-handed transitions~~. Avoid "Furthermore", "In conclusion", "The honest answer is", etc. Just continue the thought.
- ~~Narrated paragraph openers~~. Don't spend a sentence announcing what the paragraph will do ("The answer-schedule probe is worth a word of motivation.", "That distribution feeds a new measurement.", "Then the probes."). That move suits verbal teaching but is heavy in text; start with the content and let its role be apparent. Declaring intent for a whole section is still fine.
- ~~Excessive use of em dashes~~. Prefer other punctuation.
- ~~Excessive use of bold and italic text~~. List items should not be bolded. 1-2 callouts (bold) per page or section; italics only for references & borrowed words, or when it's truly unobvious which phrase should be emphasized (usually the reader can infer without it).
- ~~Contrastive conclusions~~. Avoid "... is a feature, not a confound", etc. Instead of `A, ~A`, just say `A`.
- ~~Verdict kickers~~. Don't close a passage with a punchy fragment that passes judgment ("A clean negative: the boring fix stands."). Readers pattern-match these as AI and stop reading. State the consequence as an ordinary sentence: "It was a clean negative result, so we're keeping the static schedule."
- ~~Committing to unplanned future work~~. Don't state plans we haven't made as if they are settled. "The next experiment will test X", "the anchored runs will use Y as an early warning" — written in the present indicative, these read as established facts, when usually the follow-up isn't scheduled and the property isn't demonstrated. Prefer to say what *this* report shows and stop there. If a follow-up genuinely belongs in the text, mark it as a possibility, not a promise ("this could be tested by..."), and keep the claim to what we actually know. When in doubt, say nothing about what comes next.
- ~~Evocative headings~~. Headings name what the section contains, not what it means. Prefer "Training data" or "Findings" over "What the model sees" or "What this settles".

### Register by document type

- Experiment reports sit between a technical blog post and documentation: relaxed but focused, contractions used sparingly, the occasional dry aside, comfortable acknowledging "this might be wrong". Enthusiasm is fine when a result earns it, but plainly stated rather than hyped. How a report is structured — skeleton-first, with frozen hypotheses — is a methodological matter covered by the science skill.
- Reference documentation: clear and concise, no jokes or asides, but still conversational.
- Papers: slightly more formal but still readable. "We" for the work itself, fewer contractions, no jokes, no stiffness.
- Issues, PRs, and chat: more casual; direct questions and short paragraphs.

### Pacing and structure

- One move per paragraph. Prefer several short paragraphs over one dense one; don't pack too much into a single sentence held together by dashes.
- Declare intent: say what's coming in plain first person, rather than framing it nominally after the fact. "Let's get a baseline before we anchor anything." "We will measure two things."
- State results where they first become visible. When introducing a figure whose outcome is known, say the outcome ("...and we find the model fails this task").
- Keep commentary adjacent to what it explains. Introduce a figure with a sentence or two (what's plotted, how to read it), show it, then interpret.
- A small table for enumerable examples the text refers back to; inline numbering for alternatives ("answerable two ways: 1. recall..., or 2. composition...").
- Cut tangents. A detail that serves another section belongs there or nowhere; use cross-references and forward-references sparingly.

### Clarity

Write as though giving an explanation to an intelligent person. Imagine that they have technical skill, but that they are _not_ an expert in machine learning. Aim for very high readability, and spend a significant amount of time revising drafts to be clear and concise. Allow the reader to infer information from figures, context, and a few exemplar results. Draw their attention to key details and to offer real, thoughtful insight.

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
