---
status: open
tags: [tooling, typing]
opened: 2026-07-30
---
# `ty` loses a PEP 695 `type` alias when solving widens it to a supertype

Reproduced on 0.0.49 (our pin) and 0.0.65 (latest):

```python
type Alias = Sequence[int]
def same[T](x: Sequence[T]) -> T: ...
def wider[T](x: Iterable[T]) -> T: ...
same(a)   # int
wider(a)  # Unknown
```

So `zip`/`list`/`enumerate`/`max` all drop the type argument, while assignment and plain iteration keep it. Found via `zip(cast(AxesRow, axes), rungs)` in ex-2.1.6, where `ax` came out `Unknown`. A plain `AxesRow: TypeAlias = Sequence[Axes]` works everywhere, so `mini.vis.plt` uses that form and pays a runtime `Axes` import for it. [ty#1851](https://github.com/astral-sh/ty/issues/1851) is closed and its generic-alias example does pass now — this supertype case looks separate and unfiled. Worth filing upstream, and re-testing later so the aliases can go back to `type`.
