"""Progress emission must never charge the task for the sink's latency.

The failure this guards against was measured, not imagined: with the emission
running inline, containers far from the control plane ran identical training
cells 15–30× slower than their siblings, in order of distance, because every
step paid a network round-trip. So the contract is: emitting is cheap for the
caller whatever the sink costs, the *last* update still lands before the job
ends, and a slow sink is never mistaken for a wedged worker.
"""

from __future__ import annotations

import threading
import time

from mini._debounce import BackgroundEmitter
from mini.progress import emit_metrics, emit_progress, progress_context


class SlowSink:
    """A sink that takes *delay* seconds per delivery, recording what it received."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.seen: list = []
        self._lock = threading.Lock()

    def put(self, item, /, block: bool = True, timeout: float | None = None) -> None:
        time.sleep(self.delay)
        with self._lock:
            self.seen.append(item)


def test_emitting_does_not_wait_for_the_sink():
    """100 emissions into a 20 ms sink must cost the caller ~nothing — inline that
    would be two seconds of training time — and only the survivors get delivered."""
    sink = SlowSink(delay=0.02)
    emitter = BackgroundEmitter(sink.put, interval=0.0)
    t0 = time.monotonic()
    for i in range(100):
        emitter(i)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"caller blocked on the sink ({elapsed:.2f}s for 100 emissions)"
    assert emitter.flush(timeout=5.0)
    assert len(sink.seen) < 100, "latest-wins should have dropped superseded updates"
    assert sink.seen[-1] == 99, "the final update must survive"
    emitter.close()


def test_flush_waits_for_the_last_update():
    """A task's final progress should be on the record before it settles."""
    sink = SlowSink(delay=0.1)
    emitter = BackgroundEmitter(sink.put, interval=0.0)
    emitter("last")
    assert emitter.flush(timeout=5.0)
    assert sink.seen == ["last"]
    emitter.close()


def test_sink_failures_do_not_reach_the_caller():
    """Progress is diagnostic: a control plane having a bad minute must not take
    the task down (and the caller is off the thread by then anyway)."""
    calls: list[int] = []

    def boom(i: int) -> None:
        calls.append(i)
        raise RuntimeError("control plane unavailable")

    emitter = BackgroundEmitter(boom, interval=0.0)
    emitter(1)
    emitter.flush(timeout=5.0)
    emitter(2)
    emitter.flush(timeout=5.0)
    emitter.close()
    assert calls == [1, 2], "a failed delivery must not stop later ones"


def test_watchdog_sees_progress_through_a_stalled_sink():
    """The watchdog measures the *task*, not the control plane: with the sink
    wedged, a task that keeps stepping must keep poking it — otherwise the
    background emitter would turn a network problem into a killed training run."""
    poked: list[tuple[int, int]] = []
    blocked = threading.Event()

    class WedgedSink:
        def put(self, item, /, block: bool = True, timeout: float | None = None) -> None:
            blocked.wait(timeout=5.0)  # never delivers during the test

    with progress_context(
        "run", "job", queue=WedgedSink(), emission_interval=0.0, on_progress=lambda s, t: poked.append((s, t))
    ):
        for step in range(1, 6):
            emit_progress(step, 5)
        assert poked == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
        blocked.set()  # unwedge, so leaving the context doesn't wait out the flush


def test_metrics_ride_along_with_progress():
    """``emit_metrics`` merges into the job's metrics and travels on the next
    update, so a monitor reads numbers rather than parsing a message string."""
    sink = SlowSink(delay=0.0)
    with progress_context("run", "job", queue=sink, emission_interval=0.0):
        emit_metrics(loss=0.5)
        emit_progress(1, 10)
        emit_metrics(loss=0.25)
        emit_progress(2, 10)
    assert sink.seen[-1].metrics == {"loss": 0.25}
    assert sink.seen[-1].step == 2
