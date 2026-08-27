---
status: open
tags: [docs, testing]
---
# Remove the remaining mi-ni template experiments

Remove the remaining mi-ni template experiments (`docs/pipeline`, `docs/probe`, `docs/acts` — their report notebooks are already gone) once the e2e tests that drive them (`tests/mini/test_experiments_e2e.py`) get their own fixtures, or once the first real M2 experiments can play that role. Ties into the [docs rework](https://github.com/z0u/mi-ni/issues/45). (`docs/gpt-sweep` has since become `docs/ngpt-scaling`, a real Iteration 0 output rather than a template.)

## Notes

**2026-08-21, tech debt** — Audited this as a candidate and stopped short of doing it, because what these three actually are is closer to documentation than to clutter. Recording the dependency map so the next session doesn't re-derive it.

Four things point at them. `tests/mini/test_experiments_e2e.py` genuinely loads all three: `pipeline` drives the memoized prep→sweep DAG on detached workers, and `acts`→`probe` is the acceptance test for cross-experiment artifact sharing (#13/#22). `src/mini/__main__.py` uses `docs/pipeline/experiment.py` and `docs/acts/experiment.py` as the worked examples in its CLI help and in two argparse help strings. `eng/artifacts.md` cites `docs/acts` → `docs/probe` as the illustration of explicit cross-experiment reuse. And `.claude/skills/mi-ni/references/storage.md:54` says "See `docs/acts` (producer) and `docs/probe` (consumer) for a runnable pair". The `pipeline`/`acts`/`probe` strings in `tests/test_build_site.py` and `tests/mini/test_reports.py` are synthetic fixture names only — they never touch the real files, so they don't constrain anything.

The item offers "once the first real M2 experiments can play that role" as the alternative trigger, and for the `pipeline` half they now can. Not for the pair: every real `set_ref`/`get_ref` in `docs/m1` is an experiment handing an artifact to *its own* report, so `acts`→`probe` is still the only runnable demonstration of one experiment consuming another's bytes without recomputing them. Moving the three modules into `tests/mini/fixtures/` would keep every test passing and would even harden them (CI exercises a fixture on every run, so it can't rot) — but it costs the browsable example, and `docs/pipeline/experiment.py`'s module docstring in particular is a page of onboarding prose about `--watch`, wake-at-a-time driving, and content memoization that has no other home yet. That reads like a call for the docs rework this item already points at ([mi-ni#45](https://github.com/z0u/mi-ni/issues/45)), not for a tech-debt session to make on its own.
