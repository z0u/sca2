---
status: open
tags: [testing]
---
# `test_progress_emitted_during_training` flaked once under load

`tests/sca/test_training.py::test_progress_emitted_during_training` failed once under `-n auto` on a loaded container (#73), at `{m.total for m in messages} == {max(steps)}`, and passed on the next three full-suite runs and on a serial re-run. Not reproduced since, so the cause is unconfirmed. `total` is constant across `emit_progress` calls and `BackgroundEmitter.close` flushes synchronously, so the likely shape is a second message reaching the queue from the metrics path with a different `total` — i.e. the assertion is stricter about queue contents than the contract really is. Worth reproducing under artificial load before changing anything. Noted while upgrading dependencies; nothing in that upgrade touches the debouncer, but it has not been ruled out either.
