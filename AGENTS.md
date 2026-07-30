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

Do not get distracted by such things. If the linters and type checkers say it's
fine, it's probably fine, so move on.

### Typing

Use type hints, to give the type-checker something to catch.
Use `T | None` instead of `Optional[T]`.

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

Hints also allow IDE completion, so put them wherever inference would otherwise
stall: one annotation at the point a value enters the code usually carries
through everything downstream.

Marimo generates the cell signature and leaves the parameters bare, so
everything arriving from another cell starts out `Unknown`. Annotate the first
local binding, and the rest of the cell infers from there:

```python
@app.cell()
def _(RUNGS, grading):
    _rungs: list[str] = [c for c in RUNGS if c != "lam0"]
    _resp: dict[str, tuple[np.ndarray, float, float]] = {c: grading(c) for c in _rungs}
```

`plt.subplots()` returns `tuple[Figure, Any]`, so `cast()` it:

```python
from mini.vis import AxesRow
fig, axes = plt.subplots(1, 3, ...)  # 1D
axes = cast(AxesRow, axes)

from mini.vis import AxesGrid
fig, axes = plt.subplots(2, 3, ...)  # 2D
axes = cast(AxesGrid, axes)
```

## Notebooks

Iterate on both the code (Python) and the prose (Markdown). Aim for a literate programming style.

## Environment

This project uses `uv`, `ruff`, and `ty`. Also available: `fd`, `fzf`, `rg`, `bat`, etc. For TOML, use `tomlq`:

```bash
uvx --from yq tomlq '.tool.mini' pyproject.toml
```

Resources (compute, storage, etc.): find out what you can access with `./go auth --check`.

Take care to not leak secrets into the chat transcript. To see which environment
variables are set (e.g. "is there an `HF_*` token?"), use `compgen -v HF_` (bash
builtin).
