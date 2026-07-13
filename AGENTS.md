We are running experiments to study Sparse Concept Anchoring (SCA): a training-time
regularizer that guides a concept toward a known location in representation space (rather
than searching for it post-hoc), so suppressing it later has bounded, analyzable
side-effects. M1 established this in autoencoders (done, published). This repo is **M2**:
does it transfer to transformers? We anchor concepts in the residual stream of a small
transformer trained on a synthetic color-mixing task (`red + blue = purple`), across four
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
todo.md  Backlog + scratch notes, written to be readable cold — check before starting work that might already be tracked there
```

The session-start hook prints a compact orientation map (git state, which resources are
wired up, the experiment inventory, and pointers into `eng/` and `todo.md`) at the start
of every session — read the file it points you to rather than re-deriving from scratch.

## Collaboration style

Keep the tone friendly but focused.

Don't hesitate to disagree or point out potential issues. The human values technical accuracy and appreciates being corrected when their suggestions might cause problems. Rule of thumb: never write something you don't believe; if you disagree with something, it's better to write nothing.

Be proactive. Fix little things as you go, and create [todos](/todo.md) for larger things.

## Code style & conventions

- Even in Python, prefer JavaScript-style method chaining (newline before the dot, use outer parentheses as necessary).
- Use cutting-edge syntax.
- Prefer brevity.

This is valid syntax in Python 3.14:

```python
try:
    pass
except A, B:  # PEP 758
    pass
```

Do not get distracted by such things. If the linters say it's fine, it's probably fine, so move on.

### Typing

Use type hints.
Use `T | None` instead of `Optional[T]`.

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

## Notebooks

Iterate on both the code (Python) and the prose (Markdown). Aim for a literate programming style in which we narrate our experiments.

## Environment

This project uses `uv`, `ruff`, and `ty`. Also available: `fd`, `fzf`, `rg`, `bat`, etc. For TOML, use `tomlq`:

```bash
uvx --from yq tomlq '.tool.mini' pyproject.toml
```

Resources (compute, storage, etc.): find out what you can access with `./go auth --check`.

Take care to not leak secrets into the chat transcript. To see which environment
variables are set (e.g. "is there an `HF_*` token?"), use `compgen -v HF_` (bash
builtin).
