---
status: open
tags: [local, security]
opened: 2026-09-28
---
# Warn when a role's env overlay looks like it carries a credential

A queued local task's env overlay (a role's `env=`, `[tool.mini] env`) waits in a launch spec until a worker slot frees (`mini.local_apparatus._stage_spec`). The spec is owner-only and sits in memory where there's a tmpfs, but on macOS it falls back to `~/.local/state` on disk. The overlay is meant for config (thread caps, XLA flags); a secret belongs in the shell's environment, which never reaches the spec. A warning at staging time for keys that look like credentials (`*_TOKEN`, `*_SECRET`, `*_KEY`, `*PASSWORD*`) would catch one passed through by mistake. A warning rather than a refusal, since a name match can misfire on an ordinary config key. Low priority: nothing passes a secret this way today.
