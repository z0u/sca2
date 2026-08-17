---
status: open
tags: [memoization]
opened: 2026-08-17
---
# Three shapes make the unresolvable-module warning fire on healthy code

`_should_have_resolved` (added by #102, closing [unresolvable-module-leaves-no-evidence](./unresolvable-module-leaves-no-evidence.md)) warns when a name resolves to no source and is neither stdlib nor an installed distribution. Three benign shapes land in that gap. All three were reproduced against the merged logic, so this is measured rather than predicted; none of them occur in the repo today, which is why #102 shipped.

A **project-local PEP 420 namespace package** — a directory on `sys.path` with no `__init__.py` — warns on the package itself. The leaf is tracked correctly (`nspkg.leaf:go` reaches the evidence), but `_package_chain` also yields the bare package as a reference, and there is no `__init__.py` for `_module_file` to find. The installed-metadata check covers namespace packages that were installed; a local one has no metadata to consult. This is the one with a cheap fix: `_should_have_resolved` could return `False` when some `sys.path` entry holds a *directory* named for the root, which is what a namespace package looks like from outside, and `_module_file` already walks those entries.

An **optional dependency behind `try/except ImportError`** warns whenever the dependency is absent, which is the case the code was written to handle. Same for an **`if TYPE_CHECKING:` import** of a package not installed at runtime, which cannot affect what the task computes. Both need the AST context that says "this import is guarded", and by the time the walk reaches them it is reading bytecode, where the `try` and the `if` have become jumps. Documenting the two is the realistic answer.

The doc entry in `.agents/skills/mi-ni/references/memoization.md` currently reads as though any warning means a missing install, and the message points the reader at the editable install and `PYTHONPATH`. That remedy is wrong for all three, so whichever way this is settled, the message and the entry should admit that a warning can also mean "deliberately absent".

Worth weighing against the reason the stdlib carve-out exists at all: a warning that fires on the ordinary case stops being read, and this one guards a failure that costs results.
