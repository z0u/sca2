---
status: open
tags: [experiments, storage]
opened: 2026-10-30
---
# A smoke run's kept checkpoints can stand in for the real run's

`kept_checkpoint` in `docs/m2/ex-2.2.21/experiment.py` fetches a run's kept checkpoints only when `workdir / "along"` is missing, and its callers key the workdir on the run label (`anneal-confirm/along/<label>` in `kept_axes_one`). A Modal volume is per experiment name, so a smoke run under the same name as the real run, with the same labels, leaves its `along` directory on the volume. The real run then skips the fetch and loads whatever the smoke run kept.

In the fixed-labels experiment this failed loudly: the smoke run kept epochs 2, 4 and 7, so the real run's epoch 16 was missing and the task raised. Had the smoke run kept the same epochs, the real run would have scored the smoke run's checkpoints in silence. Deleting the directory from the volume and retrying fixed it.

Two fixes, either enough. The cheap one is a habit: give a smoke run its own experiment name, as the fixed and anneal confirmations did (`fixed-confirm-smoke`), so it gets its own volume. The sturdier one is in the code: fetch whenever the artifact differs from what is on disk (a marker file holding the artifact's sha), or fetch into a directory named for the artifact rather than the label. That edit moves the evidence of every task that reaches `kept_checkpoint`, so it is best made alongside other changes there.
