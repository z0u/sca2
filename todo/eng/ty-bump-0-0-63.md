---
status: done
tags: [tooling]
opened: 2026-07-30
closed: 2026-07-30
---
# Bumped `ty` 0.0.49 → 0.0.63

`exclude-newer = "3 days"` in pyproject.toml capped us below 0.0.64/0.0.65. `ty check` is clean at 0.0.63. Notable in the gap: uv workspace-root discovery, several PEP 695 generic-type-alias fixes (0.0.63/0.0.64), which didn't fix the supertype-widening bug above, and ongoing inference-performance work every release.
