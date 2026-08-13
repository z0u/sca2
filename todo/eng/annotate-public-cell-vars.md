---
status: open
tags: [notebooks, tooling, typing]
opened: 2026-08-12
---
# Require public cell variables to be annotated, with the most specific type available

Now that Marimo propagates a definition's annotation onto the parameters of every cell downstream (see `style-py`), one annotation at the definition types the whole notebook — so annotating pays far better than it used to, and a missing one leaves every consumer as `Unknown`. Two halves, and only the first is mechanizable.

**Annotation present.** For each `@app.cell` function, every public name it *assigns* should be an `AnnAssign` rather than a bare `Assign`. Scoping it to assignments is what keeps it tractable: a public name bound by a `def` or `class` carries its own types already, and Marimo leaves those parameters bare downstream (`softmin_weights_np`, `traj_of`) rather than synthesizing a `Callable`. Ruff can't express this: `PYI052` (unannotated-assignment) is stubs-only, and the `ANN` rules cover function signatures. So it's either ast-grep as a new dependency, or ~40 lines of `ast` in `scripts/`: walk the cell functions, flag bare assignments at the top level of each body. The script version fits the pattern already here (`scripts/unpublished_reports.py` and friends, each with a test) and wires into `./go check --lint` and `.claude/hooks/marimo-format.sh` without adding a toolchain — worth trying first, and reaching for ast-grep only if the matching gets awkward. Known exception: tuple unpacking (`a, b = ...`) can't carry an annotation, so those want splitting, or an opt-out.

**Annotation specific.** `dict[int, dict]` and `np.ndarray` pass the first check while saying almost nothing. Two recurring cases, with different answers.

*Arrays* want jaxtyping, which is already a dependency and already the convention in `src/`: `Float[np.ndarray, "L1 N T C"]` on the analysis side, `Float[Array, "L1 B T C"]` on the JAX side, over a shape vocabulary that stays consistent across `anchoring.py`, `geometry.py`, and `evaluation.py`. `docs/` uses it zero times — the notebooks carry the same information as trailing comments instead (`# (L1, N, SPAN)`), and those ~70 comments across `docs/` and `src/` are a ready-made list of conversion sites. Checked 2026-08-12 that this composes with the rest of the setup: Marimo propagates the full annotation verbatim, shape string included, and `ty` and `ruff` both pass on it inside a notebook. Caveat worth knowing before leaning on it — nothing verifies these shapes at runtime (no `jaxtyped`, beartype, or typeguard anywhere in the repo), so they document rather than check, and turning runtime checking on is a separate and larger decision.

*Records* are where jaxtyping has nothing to say. `dict[int, dict]` wants a `TypedDict` for the metrics/trial/cell records every report indexes by string key. The repo has none today, and since those shapes are shared across experiments they'd belong in `src/` rather than being redeclared per notebook.

A weaker mechanical backstop, if we want one: flag unparameterized `dict`/`list`/`np.ndarray`, and select `ANN401` for a literal `Any`.
