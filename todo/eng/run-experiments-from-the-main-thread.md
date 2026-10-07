---
status: open
tags: [agents, mini]
priority: high
opened: 2026-10-01
---
# Run experiments from the main thread by default

The `mi-ni` skill (`.agents/skills/mi-ni/SKILL.md`, `references/running.md`) sends a launch to the `experiment-monitor` agent, which escalates to `experiment-doctor` when it gets stuck. Now that `bin/mini run -w` drives a DAG to completion and `bin/mini watch --timeout` bounds a wait, the supervising agent can launch the run itself as a background command and be woken when it exits. That keeps the run in the same context as the code that defines it, so a failure is diagnosed by the agent that wrote the experiment, and it avoids the handoff in which a subagent is refused an action the parent then has to surface (ex-2.2.20 hit this: the permission classifier blocked the monitor from launching the run).

The change: make a backgrounded `bin/mini run ... -w` from the main thread the default in the skill, keep `experiment-doctor` for escalations that need a fresh context, and either retire `experiment-monitor` or keep it for long or parallel runs where a cheap model watching on a bounded budget is worth the handoff.

The skill should also describe a warm-up check. A background wait only wakes the agent at the end, so a run that is slow, stalled, or misconfigured from the first step would go unnoticed until the budget or the watchdog stops it. Early in a run, while the tasks warm up, the agent should look once or twice (`bin/mini status`, `bin/mini watch --json --timeout`, the task logs) to confirm that tasks have started, steps are advancing at the expected rate, the loss is moving, and the cost is on track, and only then leave the run to finish on its own.

The skill also has to change where it now says the opposite: `references/running.md` tells an agent session not to pass `--watch`, and to advance a run with one plain `run` per stage. Under that rule, every stage boundary needs someone to issue the next `run`.

## Notes

**2026-10-05, ex-2.2.22** — The run stalled after its first two prep tasks, with `status` showing "DAG suspended at the last tick — re-run to advance", and nothing ran on Modal until Sandy noticed. The `experiment-monitor` agent had launched it with a plain `run`, as `running.md` says, and did not issue the next one. I advanced it by hand and told the monitor to keep doing so. A backgrounded `run -w` from the main thread would have driven it through. Raised to priority, since every multi-stage run goes through this handoff.

