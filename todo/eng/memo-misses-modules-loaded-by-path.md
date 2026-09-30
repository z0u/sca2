---
status: open
tags: [mini, memoization]
opened: 2026-09-27
---

# Memo evidence misses sibling modules loaded by path

Experiments that reuse another experiment's code, or a helper beside their own `experiment.py`, load it by file path with `importlib.util.spec_from_file_location` and leave it out of `sys.modules`, so the task bodies still cloudpickle by value (ex-2.2.15 loads ex-2.2.14 this way; ex-2.2.16 loads ex-2.2.14 and `posterior.py`). The memo evidence walk reads imports from source and never sees these loads, so an edit to `docs/m2/ex-2.2.16/posterior.py` does not invalidate `prepare_corpus_condition`, `eval_one`, or `suppress_one`, which all compute with it. Until this is fixed, bump a task's code by hand after editing such a module, or clear its memo.

Two directions: teach the walk to follow a module-level `spec_from_file_location` whose path is a literal relative to `__file__`, or give experiments a small helper (`mini.load_sibling(path)`) that loads the module and registers its source as evidence for any task that reaches the returned handle.
