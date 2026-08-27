---
status: open
tags: [anchoring, schedules, ex-2.1.9, m3]
opened: 2026-08-23
---
# Let τ move during training in the pooled anchor

Two mechanisms that both begin by unfreezing τ. `make_anchored_train_step` in `src/sca/anchoring.py` takes τ as a build-time argument and pins it there on purpose — its docstring says *"a condition's pooling does not move over training"* — so either mechanism starts by reopening that decision. Grouping them because that shared blocker is the first move for both; picking either mechanism afterwards is the second.

**Soft → sharp schedule.** Ex-2.1.9 found the deep-slice softmin winner committed by epoch 8 in every run, and the τ sweep's apparent dose response is really a latch-rate trend (2/3 → 1/3 → 0/3 runs latching `+` across τ = 0.01 / 0.03 / 0.1). The commit window is the first few epochs, so a remedy has to act there: start τ high (mean-like, no self-reinforcement) while the repulsion makes the syntax tokens unattractive, then sharpen. Echoes the M1 lesson that schedules shaping the space at the right moments beat curricula — and M1's schedules were never re-ablated in this architecture, so whether they're needed here is still open.

**Trainable τ, with a counterweight.** A trainable τ won't work bare: for any fixed alignment profile the mellowmax value falls monotonically as τ → 0, so gradient descent on τ always sharpens — the self-reinforcement is in the loss itself. It would need a counterweight, e.g. hold the softmin weights to a target entropy (an effective candidate count) and let τ float to meet it, annealing the target on a schedule. That is a feedback controller on a regularizer, which ex-2.9.4 found workable but fiddly and not obviously better than a timed schedule — so try the plain soft → sharp schedule first, and "soften near critical points" only if a usable critical-point signal exists (loss curvature, sudden readability gains).

Any next design here has enough seeds ready: ex-2.1.10 carries `N_SEEDS_PRIMARY = 9`, three times ex-2.1.9's three, so latch rates at the chosen rung can be bounded without re-running ex-2.1.9 wider.

Split from the older ex-2.1.9 follow-ups item on 2026-08-23.
