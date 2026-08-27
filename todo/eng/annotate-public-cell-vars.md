---
status: partial
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

## Notes

**2026-08-21, tech debt** — The mechanizable half is built: `scripts/unannotated_cell_vars.py`, `./go annotations`, and an advisory CI step beside the dead-code one. The `ast` route was the right guess — it needed no new toolchain, and the parse turned out to be its own predicate for "is this a notebook", so no marker matching and no `docs/` pathspec caveat. It takes paths, so `./go annotations docs/m2/ex-2.1.8` prints one report's share.

The backlog it found on arrival: **91 bare public assignments** to annotate, plus **42 names bound by unpacking** across 21 statements, over 16 notebooks. Heaviest are ex-2.1.8 (20), ex-2.1.6 (11), ex-2.1.7 (11), ex-2.1.1 (10). Advisory rather than a gate because of that size, and because each fix has to be published with its report — which is also why this session stopped at the checker: a read-only `HF_TOKEN` can edit a notebook but can't republish it, and CI's publish check would fail the branch.

Two things worth knowing before the annotating pass. First, **16 of the 91 are not in their cell's `return` tuple at all**, so nothing downstream reads them and an annotation propagates nowhere: `scores`, `n_seeds`, `s_bz`, `s_fr`, `rp_bz`, `rp_fr`, `corpus`, `log_v`, `sub_width`, `sub_css`, `cells`, `MARK`, `STAR`, `E`, `LR`, `depth_spread`. Those want a leading underscore rather than a type — a naming fix, not this one. Second, the return tuple is a *better* filter than the underscore convention for finding the annotations that actually pay, and the check could take it as an option later; it's left out for now because it depends on Marimo having re-saved the file (the edit hook keeps it current, but that's an assumption the checker would rather not carry).

The gate is the natural next step once the list is short: move it into `./go lint`, and add it to `.claude/hooks/marimo-format.sh` so a new one is caught where it's written.
