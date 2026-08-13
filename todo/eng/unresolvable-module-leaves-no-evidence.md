---
status: open
tags: [memoization]
opened: 2026-07-26
---
# An unresolvable module leaves no evidence and says nothing

Deferred-import evidence is now symbol-granular, so the blast-radius half of this is done. What's left is the failure direction that can actually serve a stale hit: if the `sys.path` search doesn't find a module, `_module_index` returns `None` and the walk moves on — indistinguishable from the stdlib and site-packages, which are meant to be skipped. A task importing something the driver process can't see would then depend on nothing and cache forever. Fixing it means telling "deliberately excluded" from "expected to resolve and didn't", which needs a notion of what should have been findable (an installed-distribution check, or a project-roots list). A warning would be enough. Related smaller assumption: `sys.path` order is taken as stable within a process.
