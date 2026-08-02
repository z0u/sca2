---
name: text-lint
description: |
  Eagerly trims text to deduplicate and remove over-explanation.
argument-hint: <document> [section] [line range]
---

We're doing science, so we need rigor, but we need to move fast. We can tolerate a little imprecision in exchange for speed.

This is the quick, inline pass: run it as you write, or when asked to lint a
document, with whatever context you already have. For a full reshaping of a
report section, use the `report-restructure` skill instead — it runs a
fresh-eyes agent with no conversation context and adds a template check, and
its agent spec carries the rules this lint borrows from.

By default, our docs and reports contain a lot of text that could be described as "fluff". Examples below, with bad text between `anti-example` tags, and better text (if any) between `corrected-example` tags.

<anti-example>
Every set the model fails, it fails confidently rather than hedging.
</anti-example><corrected-example>
The model is confident even when incorrect.
</corrected-example>

<anti-example>
is a feature, not a confound
</anti-example><corrected-example>
is a feature
</corrected-example>

<anti-example>
## What the model sees
</anti-example><corrected-example>
## Training data
</corrected-example>

<anti-example>
## What this settles
</anti-example><corrected-example>
## Findings
</corrected-example>

This is navel-gazing, and can be cut without replacement:

<anti-example list>
- Whether more weight is *useful* is a different question, and the answer below is no.
- The answer-schedule probe is worth a word of motivation:
- That distribution feeds a new measurement
- Alignment is scored two ways:
- Capacity is worth a sentence
- The honest answer is
- We ask two questions:
</anti-example list>

These are bombastic; sometimes they just need toning down, but often they can be removed:

<anti-example list>
- hedge
- exact
- directly
- honest
- genuine
- crucial
- Furthermore
- In conclusion
</anti-example list>

<anti-example>
  Two entrypoints, split by what they touch: `./go` for the repo (deps, checks,
  reports, the site) and `bin/mini` for experiments (compute, durable results).
  Both print usage when run bare.
</anti-example><corrected-example>
  Use `./go` for the repo (deps, checks, reports, the site) and `mini` for
  experiments (compute, durable results).
</corrected-example>

Consider the purpose of the text. This "tl;dr" was 12-lines, which is waaay too long. A tl;dr should be a lede, not an executive summary. (The bold inline numbering below is an allowed exception to the `writing` skill's emphasis rules: in a tl;dr, compactness wins.)

<anti-example>
  Both mechanisms are real, and the blind span is the larger of the two.
  Narrowing the pull to op1 is worth +0.22 of margin and adding the
  <snip n-lines="8" />
  repulsion near peak through training, instead of annealing it at the
  halfway point as M1 did, beats the M1 schedule on every measurement.
</anti-example><corrected-example>
  We tested two mechanisms to improve anchor selectivity:
  **1.** Apply the anchor term only to operand 1 (no other tokens), and
  **2.** Add a repulsive term to clear the target subspace.
  Both work, but 1. worked better, and their effects stack.
</corrected-example>

And this is very long for such a small amount of information:

<anti-example>
  The **exit code** names what happened — `0` settled all-done, `1` settled
  with a failure, `3` needs attention now, `124` timed out with work still in
  flight — so a script (or the babysitting agent) branches without reading the
  text. `--json` swaps the live progress bars for one compact summary object:
  <snip n-lines="10" />
  The full monitor loop — which exit code does what — lives in the `mi-ni`
  skill's `running.md`.
</anti-example><corrected-example>
  Exit codes: `0`: settled all-done, `1`: settled with a failure, `3`: needs
  attention, `124`: timed out with work still in flight.
</corrected-example>


<anti-example>
  That same normalization means the op1-only pull is not purely a narrowing.
  Dividing the same $\lambda_\text{a}$ among 1.5 positions instead of 5.8 makes
  the pull on each surviving position about 3.9x stronger (2,741:1 against the
  repulsion, rather than 701:1). So the op1 factor of H3 changes two things at
  once: which positions are pulled, and how hard each one is pulled. Read on its
  own, its main effect could be either.
</anti-example><corrected-example>
  That normalization also means the op1-only pull is stronger as well as
  narrower: dividing $\lambda_\text{a}$ among ~4x fewer positions makes the
  pull on each surviving position about 4x stronger.
</corrected-example>

## Keep

Keep facts, as long as they are informative. Inline numbers that can be read
from a chart or table add little, but insights _about_ the data are valuable.

Keep qualifiers that narrow scope ("at the scoring rung", "on seen pairs") to
avoid widening a claim. Also keep the reason attached to a finding: "X fails
because the gate applies to every run" is better than "X fails".

Keep indicators of certainty: "would start to matter" is a weaker claim than
"matters". Don't strengthen a claim while shortening it.

Keep signposts. A sentence that changes how the reader weights a number ("The
margin and retention rows are what make it a result") is information; a
sentence that narrates the document ("Capacity is worth a sentence") is not.

Keep (but trim) first-use definitions of terms, but move them to footnotes or
`/// details` admonitions.

Compress against the immediate context: after a cut, everything needed to
understand what remains must still be visible nearby. The `report-restructure`
agent spec lists the specific ways this goes wrong (stranded referents, lost
baselines, flattened counts).

If a report contains results, don't make material changes to preregistration
text (e.g. the hypotheses). Follow the `REVIEW`-comment convention in the
`science` skill if a change is warranted.

## Style

We are the primary audience, so use an internal lab notes register. Aim for
something less ceremonial than a paper, but not so terse that it reads like a
telegram.

## Workflow

When linting, assume the text is _correct_. Don't check numbers or verify
claims, just:

1. Look for low-perplexity (boring) text
2. Dedup and de-fluff, rephrasing as necessary
3. Move asides to footnotes
4. If the target is a Marimo notebook, run
   `.agents/skills/report-restructure/scripts/check-templates <file>`. It
   catches syntax errors, and reports dropped or frozen template expressions;
   check that these were intentional.
5. Give a short report of the flavor of the changes (not details; those will be
   self-evident).

"Already clean, a few small edits" is a fine outcome.
