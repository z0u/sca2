---
name: writing
description: |
  Writing style for composing text. Use for any prose: Markdown, GitHub issues & PRs, proposals, and technical writing such as academic papers. Improves collaboration effectiveness.
---

Write with a clear, correct, and understated style: considered but conversational, precise but not stuffy. Use the first person and voice uncertainty.

Characteristics:

- Oxford comma; straight (not "smart") quotes; single quotation marks only for quotes within quotes.
- Em dashes sparingly (approx. one per page). Prefer commas or parentheses for asides, and semicolons to connect independent clauses. A pile-up of dashes means a sentence is carrying too many ideas; split it.
- Calibrate confidence to the evidence: plain declaratives for what was measured or observed; "should", "seems", "may", "perhaps" for expectations, interpretations, and mechanisms we haven't tested directly.
- Plain connectives to continue a thought: "So", "But", "Also", "Note that".
- Pose the motivating question, then answer it: "does zeroing that axis delete red, and only red?"
- American spelling to match the convention in scientific literature.
- Alt text for all images: aids vision-impaired people and LLMs alike (see the alt-text skill).

Anti-patterns:

- ~~Business jargon and bombast~~. Avoid buzzwords, corporate-speak, and baseball metaphors.
- ~~Adversarial framing~~. Don't cast the object of study as an opponent to beat, convict, or punish: no combat metaphors (*fighting back*), crime framing (*the culprit*, *guilty*), or coercion (*punish memorization*). Describe what happened, not who won: "the LR peak was the cause".
- ~~All the lists~~. Mostly use paragraphs, but use lists sparingly when they are the clearest way to present the information.
- ~~Heavy-handed transitions~~. Avoid "Furthermore", "In conclusion", "The honest answer is", etc. Just continue the thought.
- ~~Narrated paragraph openers~~. Don't spend a sentence announcing what the paragraph will do ("The answer-schedule probe is worth a word of motivation.", "That distribution feeds a new measurement.", "Then the probes."). That move suits verbal teaching but is heavy in text; start with the content and let its role be apparent. Declaring intent for a whole section is still fine.
- ~~Excessive use of em dashes~~. Prefer other punctuation.
- ~~Excessive use of bold and italic text~~. List items should not be bolded. 1-2 callouts (bold) per page or section; italics only for references & borrowed words, or when it's truly unobvious which phrase should be emphasized (usually the reader can infer without it).
- ~~Contrastive conclusions~~. Avoid "... is a feature, not a confound", etc. Instead of `A, ~A`, just say `A`.
- ~~Verdict kickers~~. Don't close a passage with a punchy fragment that passes judgment ("A clean negative: the boring fix stands."). Readers pattern-match these as AI and stop reading. State the consequence as an ordinary sentence: "It was a clean negative result, so we're keeping the static schedule."
- ~~Committing to unplanned future work~~. Don't state plans we haven't made as if settled ("The next experiment will test X"). Say what *this* report shows and stop there; if a follow-up belongs in the text, mark it as a possibility ("this could be tested by..."). When in doubt, say nothing about what comes next.
- ~~Evocative headings~~. Headings name what the section contains, not what it means. Prefer "Training data" or "Findings" over "What the model sees" or "What this settles".
- ~~Possessives on introduced terms~~. "Hex's staircase" makes the reader bracket the term before parsing the possession, and personifies a label. Prefer an "of" construction or an adjunct: "the hex staircase", "the geometry of the named form", "the verdict of H5".

### Register by document type

- Experiment reports sit between a technical blog post and documentation: relaxed but focused, contractions used sparingly, the occasional dry aside, comfortable acknowledging "this might be wrong". Enthusiasm is fine when a result earns it, but plainly stated rather than hyped. How a report is structured — skeleton-first, with frozen hypotheses — is a methodological matter covered by the science skill.
- Reference documentation: clear and concise, no jokes or asides, but still conversational.
- Papers: slightly more formal but still readable. "We" for the work itself, fewer contractions, no jokes, no stiffness.
- Issues, PRs, and chat: more casual; direct questions and short paragraphs.

### Pacing and structure

- One move per paragraph. Prefer several short paragraphs over one dense one; don't pack too much into a single sentence held together by dashes.
- Declare intent: say what's coming in plain first person, rather than framing it nominally after the fact. "Let's get a baseline before we anchor anything." "We will measure two things."
- State results where they first become visible. When introducing a figure whose outcome is known, say the outcome ("...and we find the model fails this task").
- Keep commentary adjacent to what it explains. Introduce a figure with a sentence or two (what's plotted, how to read it, and the headline outcome), show it, then interpret. A sentence that points at a visual feature ("the upper hairline lifting away at `=`") must come after the figure it points at.
- A small table for enumerable examples the text refers back to; inline numbering for alternatives ("answerable two ways: 1. recall..., or 2. composition...").
- Cut tangents. A detail that serves another section belongs there or nowhere; use cross-references and forward-references sparingly.

### Clarity

Write as though explaining to an intelligent person with technical skill who is _not_ an expert in machine learning. Allow the reader to infer information from figures, context, and a few exemplar results; draw their attention to key details.

Concision is not density. Cutting words and lowering reader effort are different goals, and sometimes they pull against each other. A sentence may be concise and still hard to read because it stacks several ideas, folds a definition into an appositive, or hides a verb inside a noun phrase.

So write in plain English, with respect for the reader's intelligence, but also for their time.

### Simplification pass

After drafting substantial prose (a report section or more), hand it to the `prose-simplifier` agent: give it only the file path and cell range or section, deliberately withholding all other context, so it reads the draft the way a reviewer would. Then review its diff before finishing the turn — the fresh-eyes editor catches density, and the author catches any qualifier it simplified into a wrong claim.

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
