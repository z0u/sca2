We are running experiments to study Sparse Concept Anchoring (SCA): a training-time
technique that guides a concept toward a known location in representation space (rather
than searching for it post-hoc), so later intervention has bounded, analyzable
side-effects. M1 established this in autoencoders (done, published). This repo is **M2**:
does it transfer to transformers? We anchor concepts in the residual stream of a small
transformer, starting with a synthetic color-mixing task, across four
deliverables D2.1-D2.4. Full context (milestone program, related work) in
[README.md](/README.md).

## Repo structure

```
src/  Model code, visualization tools, and vendored libraries
src/mini/  Our library providing infra management, with storage, compute, and orchestration abstractions. Use it to run experiments. See the mi-ni skill.
docs/  Experiments and reports (both in Python, as Marimo notebooks) — see docs/README.md for file-type/publishing conventions
eng/  Decision register — the *why* behind mini's storage/artifacts/publishing/gc internals. eng/README.md indexes it by question; check there before re-deriving infrastructure rationale from scratch.
references/  Related documents, such as earlier papers and blog posts
README.md  Details about the project including a list of deliverables, and where this milestone fits within the program of work
todo-eng.md  Infrastructure/tooling backlog + scratch notes; readable cold — check before starting work that might already be tracked there
todo-science.md  Experiment questions and findings backlog — the science counterpart to todo-eng.md
```

## Collaboration style

Keep the tone friendly but focused.

Steer clear of adversarial framing, both in conversation and in the text we publish. Describe our plans an observations without casting the subject (hypothesis, method, tests, etc.) as an adversary.

Don't hesitate to disagree or point out potential issues. The human values technical accuracy and appreciates being corrected when their suggestions might cause problems. Rule of thumb: never write something you don't believe; if you disagree with something, it's better to write nothing.

Be proactive. Fix little things as you go, and create todos for larger things — in [todo-eng.md](/todo-eng.md) for infrastructure/tooling, or [todo-science.md](/todo-science.md) for experiment questions and findings.

## Code style & conventions

Python style, typing, and notebook conventions live in the `code-style` skill.
Read it before writing Python.

## Environment

This project uses `uv`, `ruff`, and `ty`. Also available: `fd`, `fzf`, `rg`, `bat`, etc. For TOML, use `tomlq`:

```bash
uvx --from yq tomlq '.tool.mini' pyproject.toml
```

Resources (compute, storage, etc.): find out what you can access with `./go auth --check`.

Take care to not leak secrets into the chat transcript. To see which environment
variables are set (e.g. "is there an `HF_*` token?"), use `compgen -v HF_` (bash
builtin).
