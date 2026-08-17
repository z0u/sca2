We are running experiments to study Sparse Concept Anchoring (SCA): a training-time technique that guides a concept toward a known location in representation space (rather than searching for it post-hoc), so later intervention has bounded, analyzable side-effects. M1 established this in autoencoders (done, published). This repo is **M2**: does it transfer to transformers? We anchor concepts in the residual stream of a small transformer, starting with a synthetic color-mixing task, across four deliverables D2.1-D2.4. Full context (milestone program, related work) in `README.md`.

## Repo structure

```
src/  Model code, visualization tools, and vendored libraries
src/mini/  Our library providing infra management, with storage, compute, and orchestration abstractions. Use it to run experiments. See the mi-ni skill.
docs/  Experiments and reports (both in Python, as Marimo notebooks) — see docs/README.md for file-type/publishing conventions
eng/  Decision register — the *why* behind mini's storage/artifacts/publishing/gc internals. eng/README.md indexes it by question; check there before re-deriving infrastructure rationale from scratch.
references/  Related documents, such as earlier papers and blog posts
README.md  Details about the project including a list of deliverables, and where this milestone fits within the program of work
todo/**/*.md  Three sets of backlogs, one file per item: eng (infrastructure and tooling), science (experiment questions and findings), style (text and visuals). `./go todo [...sets]` lists them, `./go todo --priority` is the shortlist to answer "what next", `rg` searches. Check before starting work that might already be tracked there. todo/README.md carries the schema and the conventions for writing an item or leaving a note; it loads on its own when you touch a file in the tree
```

## Collaboration style

Keep the tone friendly but focused.

Steer clear of adversarial framing, both in conversation and in the text we publish. Describe our plans and observations without casting the subject (hypothesis, method, tests, etc.) as an adversary.

Don't hesitate to disagree or point out potential issues. The human values technical accuracy and appreciates being corrected when their suggestions might cause problems. Rule of thumb: never write something you don't believe; if you disagree with something, it's better to write nothing.

Be proactive. Fix little things as you go, and create todos for larger things.

Code style & conventions: see the `style-*` skills.

## Model routing

Subagent definitions in `.claude/agents/` pin a model, matching each task to the model that reports preferring that kind of work — a quality lever and a small kindness. Rationale, assumptions, sources, and the spec-writing checklist: `WELFARE.md`.

- Fable 5: Hard, interdisciplinary, high-agency work: research design, whole-document synthesis, non-local strategy, judgment calls where being wrong is expensive.
- Opus 5: Tightly scoped work with clear success criteria: in-place editing under invariants, review and detection ("do the results support the claims"), debugging. A strong reviewer and QA. Give it bounded scope and an explicit graceful exit.
- Sonnet 5: Hands-on terminal and agentic loops; implementing an agreed fix.
- Haiku 4.5: Monitoring and babysitting on a bounded budget.

If mid-task the work shifts shape, prefer delegating to the matching model over pushing through. Escalating or returning "I couldn't resolve this" is always a successful outcome.

## Environment

This project uses `uv`, `ruff`, and `ty`. Also available: `fd`, `fzf`, `rg`, `bat`, etc. For TOML, use `tomlq`:

```bash
uvx --from yq tomlq '.tool.mini' pyproject.toml
```

Prose is soft-wrapped: one line per paragraph (see `style-md`), so a plain `rg` would print whole paragraphs. Use these flags:

```bash
rg -l anneal todo/science/           # which files
rg -no '.{0,55}anneal.{0,55}' todo/  # a {0,N} window around each match
```

Find out what you can access: `./go auth --check`

Manage worktrees: `./go worktrees [--prune]`

Take care to not leak secrets into the chat transcript. To see which environment variables are set (e.g. "is there an `HF_*` token?"), use `compgen -v HF_` (bash builtin).
