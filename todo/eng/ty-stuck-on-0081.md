---
status: done
tags: [tooling, typing]
opened: 2026-09-28
closed: 2026-09-28
---
# ty is held at 0.0.81 by four `subplot_mosaic` calls

`pyproject.toml` excludes ty 0.0.82 because it stopped matching matplotlib's `HashableList` overloads, and every `plt.subplot_mosaic` call with a `list[list[str]]` mosaic reports `no-matching-overload`. The comment there says to drop the exclusion once a later release fixes this. On 2026-09-28 I tried 0.0.84, the newest release, and the same four errors are still there: `docs/m2/ex-2.1.7/report.py:682`, `ex-2.1.8/report.py:813`, `ex-2.1.9/report.py:883`, and `ex-2.1.10/report.py:822`. I didn't test 0.0.83 on its own, since 0.0.84 fails either way. So the lock stays at 0.0.81, and CI installs with `--locked`, so CI stays there too.

0.0.84 has a security fix: a use-after-free during incremental type checking, which a specially crafted project could use to run arbitrary code. I judged that it doesn't reach us. It needs a hostile project to analyze, and ty only ever checks our own code. So there's no rush, but each release we skip adds to what we miss (0.0.82 added `TypeVarTuple` call binding and recursive implicit aliases).

The decision is whether to keep waiting for upstream or change the four call sites. Two options, probably in order of preference. First, give each mosaic an explicit type that the overloads still accept, or `cast` it at the call. That is a small edit, but all four reports are published, and I haven't checked whether the change moves any memo fingerprints (see [docstring-edits-move-the-memo-fingerprint](docstring-edits-move-the-memo-fingerprint.md)). Second, check whether the regression has been reported upstream, report it if not, and keep waiting. Also, if a blanket `uv lock --upgrade` lands before this is settled, it will pick up 0.0.84 and fail `./go check`. That failure shows itself, so a wider exclusion in `pyproject.toml` doesn't seem worth it.

## Notes

**2026-09-28, closing** — Took the first option, with a `cast` at each call. Annotating the constants themselves as `list[HashableList[str]]` also clears the overloads, but then every element types as `str | HashableList[str]`, and ex-2.1.7 indexes dicts by panel name, so the wider type only moves the errors. The cast keeps the constants as `list[list[str]]` and widens only at the one call that needs it. The target is `mini.vis.Mosaic`, beside `AxesRow` and `AxesGrid`, so the reason is written once and the casts are easy to find later. Both 0.0.81 and 0.0.84 accept it, and the ty floor is now 0.0.84 with the exclusion gone. The four reports render the same figures, so the PR carries the `skip-publish-check` label and doesn't republish them. Remove the casts once a ty release matches the overloads again (`rg 'cast\(Mosaic'`).
