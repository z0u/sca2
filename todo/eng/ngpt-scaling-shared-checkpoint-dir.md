---
status: open
tags: [experiments]
---
# ngpt-scaling's sweep cells share one checkpoint directory

ngpt-scaling's sweep cells all `save_checkpoint` to the same shared `get_data_dir()`, so the checkpoint file is last-writer-wins across a fan-out. Harmless there (cells return their metrics; nothing reads the checkpoints back). `train_model` now takes a `checkpoint_dir` for per-cell keying — ex-2.1.1 uses it because its eval step reads checkpoints back — but ngpt-scaling still writes to the shared default.
