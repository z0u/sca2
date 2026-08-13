---
status: open
tags: [library, temporal]
---
# `mini.temporal` can't drive feedback control

`DynamicProp.set()` retargets mid-flight from the current (value, velocity) state — exactly what a controller needs — but experiments consume schedules via `realize_timeline`, which bakes the dopesheet into a static per-step array before training, and the dopesheet's own keyframes would fight any runtime `set()` calls on the same prop. Ex-2.9.4's controller therefore lives inside the training loop (duals in the `lax.scan` carry), with the dopesheet still driving the non-controlled props. If feedback-driven weights become standard, consider a Timeline mode where a prop is declared "controlled": keyframes set its bounds and defaults, and a callback supplies the live value.
