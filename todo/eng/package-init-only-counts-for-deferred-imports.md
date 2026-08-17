---
status: open
tags: [memoization]
opened: 2026-08-17
---
# A package `__init__.py` is evidence for deferred imports only

`_package_chain` folds each parent package's whole source into the manifest, so a deferred `from sca.compute.geometry import probe_maps` depends on `sca/__init__.py` and `sca/compute/__init__.py`. That is the right call — they execute on the way down, and ours sets `XLA_FLAGS`, which changes what the task computes. The live-object walk has no equivalent: a task that imports at module top reaches the helper as an object, `_collect_sources` records the function and its references, and no package source enters. Verified by fingerprinting the two shapes against an `__init__.py` that gained a definition — the deferred one moved, the top-level one didn't.

Low impact today, because task bodies in this project import deferred to keep the driver and CLI light, which is the covered path. It's the same class of hole as the unresolvable-module one though: a silent stale hit rather than a spurious re-run, so it's worth closing before some future task imports its helpers at module scope.

The fix has an asymmetry worth thinking through first. The deferred walk knows the dotted module name it is resolving, so the chain falls out of the name; the live walk starts from a function object, where `fn.__module__` is available but the packages above it have already been imported in the driver process, and folding their whole source in would invalidate more broadly than the deferred path does for the same code (the driver imports far more than any one task reaches). Narrowing to the prelude — the import-time statements, which is what actually differs between processes — may be the better shape for both walks.
