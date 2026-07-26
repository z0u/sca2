# Operational constraints worth remembering

*Part of the [engineering notes](./README.md).*

- **Egress allow-list.** Bucket I/O needs `*.xethub.hf.co`; serving figures needs
  `*.cdn.hf.co`. Without them, metadata calls to `huggingface.co` succeed while every
  byte transfer hangs on a 403 — a confusing failure mode.
- **Modal gRPC TLS.** Modal's client builds its trust store from `certifi` alone; behind
  a TLS-inspecting proxy it needs the system CA folded in
  (`mini._tls.ensure_grpc_trusts_system_ca` already does this).
- **CORS / Range.** The bucket reflects the request `Origin` on both the resolve redirect
  and the CDN response, and the CDN advertises `Accept-Ranges` — so a Pages-served report
  can `fetch()` a published JSON cross-origin and Range-slice a big binary.

## Progress transport: a queue only where there's a consumer

Two paths carry a job's progress, and they use different Modal primitives on purpose:

- **Blocking `Apparatus.amap`** (a notebook cell, no `Ctx`) uses a `modal.Queue` —
  streaming suits a live display, and the queue is `ephemeral()`, created and destroyed
  inside the same `async with` that holds its consumer. It cannot outlive the reader,
  so the "fills up with nobody draining it" hazard has no room to happen.
- **Memoized orchestration** (`mini.orchestration`, driven by the CLI) has **no queue at
  all**. Its workers are detached and its readers are short-lived polls — a fresh
  `status`/`watch` process per wake, often none at all for minutes — so there is no
  consumer to assume. Progress goes straight to the control-plane `modal.Dict`,
  last-writer-wins, which is also exactly the shape a poller wants: one current value per
  task, not a backlog to replay (`mini._taskworker._MemoSink`).

So the split already matches where each assumption holds, and a queue is never reachable
from the detached path. Worth knowing before reaching for one: the reason it works in the
one place it's used is the live consumer, not the transport.

Delivery to either sink runs on a per-job background thread with a single latest-wins slot
(`mini._debounce.BackgroundEmitter`), so a slow or distant sink costs progress resolution
rather than the task's own throughput.

**The 2026-07-23 slowdown was the Dict, not a Queue.** Worth recording, because the first
write-up named the wrong transport and the wrong cost. Ex-2.1.5 ran on the memoized path,
so every `emit_progress` went `_MemoSink.put` → `MemoStore.update_if` →
`ModalRecordStore.merge_if` — inherited from the base class at the time, which read to
check the fence and then called `merge`, itself a get plus a set: **three** Dict
round-trips per emission, on the training thread, at a 0.2 s debounce interval. Splitting
the observed 92–220 steps/min across three round-trips gives ~0.09–0.22 s each, which
reads like a cross-region RTT; the single 0.65 s hop the original note implied never quite
did. `ModalRecordStore` now overrides `merge_if` to do one get and one set, which also
halves the gap between checking the fence and acting on it.
