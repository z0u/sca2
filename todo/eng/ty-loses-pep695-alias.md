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

## Notes

**2026-08-23, tech debt** — Re-tested, and upstream has fixed it. The repro above is unchanged on 0.0.65 (which is what the lock now resolves, so the "our pin" in the body has moved since it was written), and clean on **0.0.74**: `wider(a)` is `int`, `list(a)` is `list[int]`, and `enumerate` keeps the argument too. So the widening path recovers the alias now, and there is nothing left to file upstream.

`uvx ty@latest check --python .venv` over the whole repo passes with no new findings, so raising the floor from `ty>=0.0.19` looks free — the constraint is a floor rather than a pin, so it also needs `uv lock --upgrade-package ty` to actually move.

Two things worth knowing before making the conversion, which is why this session recorded the re-test instead of doing it. The floor has to reach 0.0.74 in the same change, or anyone still resolving 0.0.65 gets the `Unknown`s back silently. And the saving named in the body doesn't survive checking: `plt.py`'s other three `Axes` uses are already string annotations, so the alias assignments are the only runtime need for the import — but `import matplotlib.pyplot` two lines above pulls `matplotlib.axes` in regardless, so moving `Axes` under `TYPE_CHECKING` saves nothing measurable. What the conversion actually buys is the modern syntax and the retirement of the workaround comment in `plt.py`, which is real but is a taste call rather than a cost, and it rides on a toolchain bump.
