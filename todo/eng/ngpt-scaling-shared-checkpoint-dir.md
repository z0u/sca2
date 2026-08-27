---
status: open
tags: [experiments]
---
# ngpt-scaling's sweep cells share one checkpoint directory

ngpt-scaling's sweep cells all `save_checkpoint` to the same shared `get_data_dir()`, so the checkpoint file is last-writer-wins across a fan-out. Harmless there (cells return their metrics; nothing reads the checkpoints back). `train_model` now takes a `checkpoint_dir` for per-cell keying — ex-2.1.1 uses it because its eval step reads checkpoints back — but ngpt-scaling still writes to the shared default.

## Notes

**2026-08-23, housekeeping** — Still true, and now the last one. `docs/ngpt-scaling/experiment.py:116` is the only `train_model` call in `docs/` that omits `checkpoint_dir`; ex-2.1.1 through ex-2.1.5 all pass a per-cell one, so what this item described as a new option has since become the house pattern everywhere else. That makes the fix a one-line keyword argument rather than a design question, and the item is worth folding into whatever session next touches ngpt-scaling. Nothing reads these checkpoints back, so it buys tidiness rather than a correction — and it costs a re-run of the sweep, since editing `train_one` invalidates its evidence.
