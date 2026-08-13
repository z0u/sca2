---
status: done
tags: [testing]
opened: 2026-08-11
closed: 2026-08-11
---
# `test_local_apparatus_concurrent` failed on a pristine tree

In `tests/mini/test_apparatus.py`. It asserted 3 × 0.1 s sleeps finish under 0.25 s, and on a loaded 4-CPU remote container the pool took ~1.9 s — the bound read the machine at least as much as the pool, so neither loosening it nor gating on CPU count would have made it measure the right thing. Workers there are threads and `sleep` releases the GIL, so what `max_workers=3` promises is that all three spans are open at once: each task now returns its start and end, and the test asserts the spans intersect, which holds at any speed. Verified against a serial pool (`max_workers=1`), which it still catches.
