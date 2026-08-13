---
status: open
tags: [docs, testing]
---
# Remove the remaining mi-ni template experiments

Remove the remaining mi-ni template experiments (`docs/pipeline`, `docs/probe`, `docs/acts` — their report notebooks are already gone) once the e2e tests that drive them (`tests/mini/test_experiments_e2e.py`) get their own fixtures, or once the first real M2 experiments can play that role. Ties into the [docs rework](https://github.com/z0u/mi-ni/issues/45). (`docs/gpt-sweep` has since become `docs/ngpt-scaling`, a real Iteration 0 output rather than a template.)
