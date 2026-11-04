"""Term cost: the time a training step of the recipe of record spends on the anchor and anti terms.

The control runs train through the same `train_anchored` step as the anchored runs, with the two term weights at
zero, so their wall-clock time carries the terms' cost too and can't show it. This probe times the step-speed
probe's `train_timed` (the recipe of record, `τ_ℓ fixed`, on the recipe's corpus, for `ss.STOP` epochs of the
400-epoch schedule) four ways: as it is (`terms`), with the anti term alone or the anchor term alone left out
(`anchor-only`, `anti-only`), and with the step's loss swapped for the plain cross-entropy of an unanchored model
(`bare`, `sca.training.loop.loss_fn`). The anchor term reads labelled contexts only; the anti term reads every
state. Everything around the step is unchanged in both: the batch
draw with its labeller, the scan of sixteen steps per dispatch, and the trajectory records.

Results (2026-11-04, L40S, four containers a variant, one lost to the queue for `anchor-only`): the steady step
took 2.01 ms with both terms, 1.98 with the anchor alone, 1.96 with the anti term alone and 1.97 with neither. The
gap with both terms, about 2%, is smaller than the spread between containers of one variant (1.90 to 2.11 ms).
XLA's cost analysis of the compiled step (on CPU, same shapes) agrees: the two terms add about 0.1% to the
floating-point work of a step and about 3% to the bytes it moves, each term about a third of that alone. Both
terms read one coordinate of each residual state, so their share falls as the model widens.

Corpus preparation and the training tasks run under the dev profile; a report quotes the numbers, and reads nothing
from the store.

    MINI_PROFILE=dev bin/mini run docs/m2/term-cost/experiment.py --app modal --max-containers 10 --budget 1h
    MINI_PROFILE=dev bin/mini status term-cost
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from mini import Ctx, Experiment
from mini.orchestration import Pending


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ss = _load_sibling("step-speed", "term_cost_ss")
fc, ex2221, ex2216 = ss.fc, ss.ex2221, ss.ex2216

VARIANTS: dict[str, dict[str, Any]] = {
    "terms": dict(card="L40S", n=4, drop=()),
    "anchor-only": dict(card="L40S", n=4, drop=("anti",)),
    "anti-only": dict(card="L40S", n=4, drop=("anchor",)),
    "bare": dict(card="L40S", n=4, drop=("anchor", "anti", "stream")),
}
"""Variant → card, replicas (one container each), and the parts of the step it leaves out (`_drop`)."""


def _bare_step() -> None:
    """Swap the anchored step for one whose loss is the plain cross-entropy, in this process. It takes the anchored
    step's arguments and returns zeros for the two terms, so the loop around it runs unchanged.
    """
    import equinox as eqx
    import jax.numpy as jnp

    import sca.compute.training as training
    from sca.training.loop import loss_fn

    def make(optimizer, **_kw):
        @eqx.filter_jit
        def train_step(model, opt_state, x, y, mask, line_id, weight, anti_weight, line_w=None, pool=None, taus=None):
            task, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates).normalize_weights()
            zero = jnp.zeros((), jnp.float32)
            return model, opt_state, task, zero, zero

        return train_step

    training.make_anchored_train_step = make  # ty: ignore[invalid-assignment] — a patch for the probe alone


def _drop(parts: tuple[str, ...]) -> None:
    """Leave *parts* out of the anchored step, in this process: `anchor` and `anti` make that term a constant zero,
    so the compiler drops its work; `stream` (with both) swaps in the plain step (`_bare_step`), which never stacks
    the residual stream.
    """
    import jax.numpy as jnp

    import sca.anchoring as anchoring

    if "stream" in parts:
        _bare_step()
        return
    zero = lambda *_a, **_k: jnp.zeros((), jnp.float32)  # noqa: E731
    if "anchor" in parts:
        anchoring.pooled_anchor_term = zero  # ty: ignore[invalid-assignment] — a patch for the probe alone
        anchoring.anchor_term = zero  # ty: ignore[invalid-assignment]
    if "anti" in parts:
        anchoring.anti_subspace_term = zero  # ty: ignore[invalid-assignment]


def train_timed(corpus, labels, probes, row: dict, variant: str, replica: int, drop: tuple[str, ...]) -> dict:
    """`ss.train_timed` for *row* in this container, with *drop* left out of the step."""
    _drop(drop)
    return ss.train_timed(corpus, labels, probes, row, variant, replica)


def main(ctx: Ctx) -> dict:
    p = ctx.run(
        ex2221.prepare_condition,
        fc.OP_NAMES,
        fc.K,
        ex2221.RHO,
        0.0,
        ex2221.CUBE_RATE,
        ex2221.N_LINES,
        ex2221.CORPUS_SEED,
        ex2216.HOLDOUT_CONTEXTS,
        ex2216.N_TRAJ_PROBE,
        ex2221.MAIN_KEY,
        role="prep",
    )
    (row,) = fc.rows_of(p["meta"], (ss.SEED,), None)
    row = row | {"total_tokens": p["meta"].total_tokens}
    timed, waiting = {}, []
    for variant, v in VARIANTS.items():
        for r in range(v["n"]):
            try:
                timed[f"{variant}-r{r}"] = ctx.run(
                    train_timed, p["corpus"], p["labels"], p["probes"], row, variant, r, v["drop"],
                    role=f"{variant}-r{r}",
                )  # fmt: skip
            except Pending as e:
                waiting.append(e)
    if waiting:
        raise waiting[0]
    return {"timed": timed}


COMPUTE: dict[str, Any] = {"prep": dict(cpu=2, timeout=1800)} | {
    f"{variant}-r{r}": dict(gpu=v["card"], timeout=3600, watchdog=600, watchdog_grace=900)
    for variant, v in VARIANTS.items()
    for r in range(v["n"])
}

experiment = Experiment(name="term-cost", main=main, roles=COMPUTE)
