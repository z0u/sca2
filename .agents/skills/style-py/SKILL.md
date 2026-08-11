---
name: style-py
description: Python style and typing conventions for this repo — method chaining, modern syntax, type hints that survive Marimo's bare cell signatures, and the literate programming standard for notebooks. Use when writing or reviewing any Python.
---

House style, in three lines: chain your calls, use the newest syntax the toolchain accepts, and keep it short.

## Shape of the code

Prefer JavaScript-style method chaining, even in Python. Put the newline *before* the dot, and wrap the whole expression in parentheses when you need to:

```python
result = (
    frame
    .filter(pl.col("lam") > 0)
    .group_by("rung")
    .agg(pl.col("mse").mean())
)
```

Prefer brevity. A shorter version that reads the same is the better version.

## Cutting-edge syntax is encouraged

We track new Python releases and use what they give us. For example, PEP 758 multi-exception `except` without parentheses is valid in 3.14:

```python
try:
    pass
except A, B:  # PEP 758
    pass
```

If something looks unfamiliar, check the linters rather than rewriting it. `ruff` and `ty` are the arbiters: if they're happy, the code is fine.

## Typing

Annotate, to give the type-checker something to catch and the IDE something to complete.

Use `T | None`, never `Optional[T]`:

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

You don't need annotations everywhere. Put one wherever inference would otherwise stall — usually the point where a value first enters the code. That single annotation then carries through everything downstream.

### Marimo cells

Marimo generates each cell's signature and leaves the parameters bare, so every value arriving from another cell starts out as `Unknown`. Inference has nothing to work from at the top of the cell. Annotate the first local binding and the rest of the cell follows:

```python
@app.cell()
def _(RUNGS, grading):
    _rungs: list[str] = [c for c in RUNGS if c != "lam0"]
    _resp: dict[str, tuple[np.ndarray, float, float]] = {c: grading(c) for c in _rungs}
```

Naming:

- Symbols that are cell-local must start with `_`, or Marimo will complain.
- Symbols within nested functions should usually not start with `_`.

```python
@app.cell()
def _():
    def _foo(x: int) -> int:
        y = x + 1
        return y
```

### Matplotlib axes

`plt.subplots()` returns `tuple[Figure, Any]`, because the second element's shape depends on the arguments. `cast()` it to the alias that matches what you asked for:

```python
from mini.vis import AxesRow
fig, axes = plt.subplots(1, 3, ...)  # 1D
axes = cast(AxesRow, axes)

from mini.vis import AxesGrid
fig, axes = plt.subplots(2, 3, ...)  # 2D
axes = cast(AxesGrid, axes)
```

## Notebooks

Our experiments and reports are Marimo notebooks, which means the code and the prose ship together. Iterate on both. Aim for literate programming: the Markdown should explain what the next cell does and why, so the notebook reads as an argument rather than a script with captions.

See the `style-fig` skill for figure and results-table conventions, and `docs/README.md` for file-type and publishing rules.
