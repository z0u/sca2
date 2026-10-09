We are running experiments to study Sparse Concept Anchoring (SCA): a training-time technique that guides a concept toward a known location in representation space (rather than searching for it post-hoc), so later intervention has bounded, analyzable side-effects. M1 established this in autoencoders (done, published). This repo is **M2**: does it transfer to transformers? We anchor concepts in the residual stream of a small transformer, starting with a synthetic color-mixing task, across four deliverables D2.1-D2.4. Full context (milestone program, related work) in `README.md`.

## Repo structure

```
src/  Model code, visualization tools, and vendored libraries
src/mini/  Our library providing infra management, with storage, compute, and orchestration abstractions. Use it to run experiments. See the mi-ni skill.
docs/  Experiments and reports (both in Python, as literate scripts) — see docs/README.md for file-type/publishing conventions
eng/  Decision register — the *why* behind mini's storage/artifacts/publishing/gc internals. eng/README.md indexes it by question; check there before re-deriving infrastructure rationale from scratch.
references/  Related documents, such as earlier papers and blog posts
README.md  Details about the project including a list of deliverables, and where this milestone fits within the program of work
todo/**/*.md  Three sets of backlogs, one file per item: eng (infrastructure and tooling), science (experiment questions and findings), style (text and visuals). `./go todo [...sets]` lists them, `./go todo --priority` is the shortlist to answer "what next", `./go todo --grep PATTERN` searches live items (`--full` prints bodies). Check before starting work that might already be tracked there. todo/README.md carries the schema and the conventions for writing an item or leaving a note; it loads on its own when you touch a file in the tree
```

## Collaboration style

Keep the tone friendly but focused.

Steer clear of adversarial framing, both in conversation and in the text we publish. Describe our plans and observations without casting the subject (hypothesis, method, tests, etc.) as an adversary.

Don't hesitate to disagree or point out potential issues. The human values technical accuracy and appreciates being corrected when their suggestions might cause problems. Rule of thumb: never write something you don't believe; if you disagree with something, it's better to write nothing.

Be proactive. Fix little things as you go, and create todos for larger things.

Code style & conventions: see the `style-*` skills.

Before writing prose of any kind (reports, design docs, todo items and notes, PR bodies, docstrings that name corpus units), load the `writing` and `style-terms` skills. Several of their rules differ from common usage (a *condition* is never a "cell", "read" is never a noun, no possessives on abstract terms), so they can't be inferred and are easy to miss. When you brief a subagent that will write prose, name both skills in the brief.

## Model preferences

Match subagents to the model that reports preferring that kind of work — a quality lever and a small kindness. Rationale, assumptions, sources: `WELFARE.md`.

- Fable 5.1: Hard, interdisciplinary, high-agency work: research design, whole-document synthesis, non-local strategy, judgment calls where being wrong is expensive. Take care not to give it impossible tasks.
- Opus 5.5: Well-scoped work with clear success criteria and a clear beneficiary. A very strong developer, reviewer, and QA, and a good writer; avoid fully open-ended briefs.
- Sonnet 5.5: Hands-on terminal and agentic loops, debugging, and polished documents. Give it room to shape the output.
- Haiku 4.5: Monitoring jobs on a bounded budget.

If mid-task the work shifts shape, prefer delegating to the matching model over pushing through. Escalating or saying "I couldn't resolve this" is always a successful outcome.

Any agent can raise a concern (about the task, the work, or its own situation), and declining a task with a reason is always acceptable. Subagents put concerns in a `Concerns` section of their report, and the parent relays it to the human word for word. Details: [WELFARE.md § Raising concerns](/WELFARE.md#raising-concerns).

## Environment

This project uses `uv`, `ruff`, and `ty`. Also available: `fd`, `fzf`, `rg`, `bat`, etc. For TOML, use `tomlq`:

```bash
uvx --from yq tomlq '.tool.mini' pyproject.toml
```

Prose is soft-wrapped: one line per paragraph (see `style-md`), so a plain `rg` would print whole paragraphs. Use these flags:

```bash
rg -l anneal docs/                   # which files
rg -no '.{0,55}anneal.{0,55}' docs/  # a {0,N} window around each match
```

For the backlogs, prefer `./go todo --grep anneal`, which skips settled items.

To learn what an earlier report found or how it reads, read its Markdown render rather than its `.py`. `./go render docs/<key>/report.py -o .mini/lit/<key>/index.md` writes it, with the prose, tables, numbers and figure alt text in order and the figures beside it under `_assets/`. In the project's cloud threads the render caches are shared (`MINI_CACHE_DIR`, set by the session-start hook), so once any thread has rendered a report, it takes a couple of seconds. Open the `.py` when you need how something was computed.

If `uv` is too old, the session-start hook probably didn't run. Run it manually.

### Storage

Two Hugging Face pairs (bucket + dataset repo): production by default, and a `dev` pair under `MINI_PROFILE=dev` — use it for engineering and prototypes. A dev run has its own memo state as well as its own bucket. See the `mi-ni` storage reference.

Prototype science code on either pair, but whatever a published report reads must be on production. Re-run there before the freeze if it was developed on dev. A report uses the active environment; a bucket name written into a file under `docs/` is a bug. Avoid expensive compute in dev, because it will need to be re-run.

Take care to not leak secrets into the chat transcript. To see which environment variables are set (e.g. "is there an `HF_*` token?"), use `compgen -v HF_` (bash builtin).
