---
status: open
tags: [mini, storage, watchdog, modal]
opened: 2026-10-13
---
# A bucket commit can outlast the training watchdog

In the split-anti-seeds run, about ten 400-epoch trainings finished within minutes of each other, and each then called `put` for its checkpoint. Most of those uploads sat in `HfApi.batch_bucket_files` → `new_upload_commit` for many minutes. Eight finished, and two were aborted by the 900 s training watchdog at step 105600/105600, so each costs a full re-train (about 25 minutes on an L4).

`Store.put` declares a phase sized `120s + size / 512 KiB/s`, but a phase only widens the threshold, so here the role's 900 s step watchdog governed. The time went to the commit round trip under concurrency, which the size-based budget doesn't model. Nothing raised, so the `_retrying` ring in `hf_store._batch` never ran.

Options: size the put phase with a concurrency-independent floor well above 900 s (the cost of erring loose is idle time, the cost of a false abort is a re-train); give the bucket commit its own timeout so a slow one raises and is retried; or stagger checkpoint uploads. Related: [watchdog-fires-during-upload](watchdog-fires-during-upload.md).
