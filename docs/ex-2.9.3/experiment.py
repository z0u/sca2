"""
Experiment 2.9.3: why anchoring fails — timing, attribution, and a schedule fix.

Ex-2.9.2 fixed the *redistribution* half of ablation variance (fallback control
gives the intervention a designed response) but left the other half open: on
some seeds the concept never ends up cleanly on its axis, and no
intervention-time trick can repair that. The working hypothesis going in was
that the regularizer schedule is incompatible with some initializations.

A local pilot falsified the "bad init" framing: every failing seed *anchored
successfully* by step ~650, then lost the anchor during the high-LR plateau
(lr = 0.1 from step 750) — immediately on reaching it (recon collapse), or
later, once the anchor weight had annealed below ~0.03 and could no longer
restore the axis. The anchored solution is metastable at that LR, and the
anneal removes its protection while the optimizer is still hot.

This experiment pins that down with three arms:

- **trajectories** (32 seeds, ex-2.9.2's base config exactly): record per-step
  anchor progress (z₀ of pure red), leakage, and grid reconstruction error, to
  show when failures happen relative to the schedule.
- **attribution** (16 inits × 8 data streams, same config): factor the RNG into
  the model init and the batch/label stream. If failure were incompatible
  initial conditions, it would follow the init row; if it is a mid-training
  accident, it follows neither and scatters.
- **sweep** (peak LR {0.10, 0.07, 0.05, 0.03} × anneal {on, off} × 32 seeds,
  with ex-2.9.2's fallback term, scored by the redirect intervention): find
  whether a static schedule removes the failures, and what the LR peak buys.

The sweep's (0.10, anneal) cell replicates ex-2.9.2's fallback arm; the
trajectory arm replicates its base arm, so known-bad seeds (22, 27) reproduce.
The testbed (model, grids, loss terms, interventions) is `sca.colorcube`.

    bin/mini run docs/ex-2.9.3/experiment.py --app modal --max-containers 16
"""

from __future__ import annotations

import io
import json
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from sca.colorcube import (
    BATCH,
    GAMMA,
    GRID_RGB,
    OTHERS,
    PURE_RED,
    RED_PROB,
    SIM3,
    TRAJ_STRIDE,
    VAL_RED,
    VAL_RGB,
    WEIGHT_PROPS,
    Params,
    decode,
    edit_axis0,
    eval_model,
    forward,
    init_params,
    loss_fn,
    make_dopesheet,
    score_interventions,
)
from mini import Ctx, Experiment, emit_metrics, emit_progress
from mini.store import put, set_ref
from mini.temporal import Dopesheet, Timeline, realize_timeline

SEEDS = list(range(32))

FALLBACK_WEIGHT = 0.05  # w_fb for sweep-arm runs

PEAK_LRS = (0.10, 0.07, 0.05, 0.03)
ATTR_INITS = list(range(14)) + [22, 27]  # 14 ordinary inits + the two known-catastrophic ones
ATTR_STREAMS = list(range(8))

# Store refs the report reads (see report.py).
METRICS_REF = "reports/ex-2.9.3/metrics"
TRAJS_REF = "reports/ex-2.9.3/trajectories"


def train_one(seed: int, stream: int | None, w_fb: float, dopesheet_csv: str) -> dict:
    """Train one run and score it; also record per-step anchoring diagnostics.

    *seed* picks the model init. *stream* picks the batch/label sequence; None uses
    ex-2.9.1's derivation (init and stream both from *seed*), so those runs replicate
    earlier experiments bit-for-bit.
    """
    sheet = Dopesheet.from_csv(io.StringIO(dopesheet_csv))
    df = realize_timeline(Timeline(sheet))
    n_steps = len(sheet)
    lr = jnp.asarray(df["lr"].to_numpy(np.float32))
    weights = jnp.stack([jnp.asarray(df[p].to_numpy(np.float32)) for p in WEIGHT_PROPS], axis=1)

    x_train, p_red = jnp.asarray(GRID_RGB), jnp.asarray(RED_PROB)
    if stream is None:
        key, k_model = jr.split(jr.key(seed))
    else:
        _, k_model = jr.split(jr.key(seed))  # the same init this seed had under the legacy derivation
        key = jr.fold_in(jr.key(986452), stream)  # an unrelated batch/label stream
    params = init_params(k_model)
    opt = optax.adam(lambda count: lr[count])  # the dopesheet *is* the LR schedule
    opt_state = opt.init(params)

    others = jnp.asarray(np.flatnonzero(OTHERS))
    pure_red = jnp.asarray(PURE_RED)

    @jax.jit
    def run_chunk(params, opt_state, key, steps):
        def step(carry, i):
            params, opt_state, key = carry
            key, k_batch, k_label = jr.split(key, 3)
            idx = jr.randint(k_batch, (BATCH,), 0, x_train.shape[0])  # bootstrap sample
            labels = jr.bernoulli(k_label, p_red[idx]).astype(jnp.float32)  # stochastic sparse labels
            (_, recon), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_train[idx], labels, weights[i], w_fb
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = cast(Params, optax.apply_updates(params, updates))
            # Anchoring diagnostics on the full grid, for the failure-timing analysis
            _, z = forward(params, x_train)
            _, z_red = forward(params, pure_red)
            diag = {
                "z0_red": z_red[0, 0],  # anchor progress: → 1 when red sits on e₀
                "leak": jnp.mean(jnp.abs(z[others, 0])),  # axis-0 occupancy of non-red colors
                "recon": jnp.mean((jnp.clip(decode(params, z), 0.0, 1.0) - x_train) ** 2),
            }
            return (params, opt_state, key), diag

        (params, opt_state, key), diags = jax.lax.scan(step, (params, opt_state, key), steps)
        return params, opt_state, key, diags

    variant = f"s{seed}" + (f"×d{stream}" if stream is not None else "")
    diags = []
    for chunk in np.array_split(np.arange(n_steps), 10):
        params, opt_state, key, d = run_chunk(params, opt_state, key, jnp.asarray(chunk))
        diags.append({k: np.asarray(v) for k, v in d.items()})
        emit_progress(int(chunk[-1]) + 1, n_steps, message=variant)
        emit_metrics(recon=float(d["recon"][-1]), z0_red=float(d["z0_red"][-1]), leak=float(d["leak"][-1]))
    traj = {k: np.concatenate([d[k] for d in diags])[::TRAJ_STRIDE] for k in diags[0]}

    # Final validation, geometry diagnostics, and intervention scores
    mse_val, z_val = eval_model(params, jnp.asarray(VAL_RGB))
    mse_base, z_base = eval_model(params, x_train)
    interventions = score_interventions(params)

    buf = io.BytesIO()
    mse_rd, z_rd = eval_model(edit_axis0(params, bias=-GAMMA), x_train)
    np.savez_compressed(
        buf,
        rgb=GRID_RGB,
        sim3=SIM3,
        mse_base=np.asarray(mse_base),
        z_base=np.asarray(z_base),
        mse_redirect=np.asarray(mse_rd),
        z_redirect=np.asarray(z_rd),
    )
    metrics = {
        "val_recon": float(jnp.mean(mse_val)),
        "val_anchor": float(jnp.mean(1.0 - z_val[VAL_RED, 0])),
        "val_anti_anchor": float(jnp.mean(jnp.maximum(-z_val[:, 0], 0.0))),
        "leak": float(np.mean(np.abs(np.asarray(z_base)[OTHERS, 0]))),
    }
    emit_metrics(**metrics, **{f"score_{k}": v["score"] for k, v in interventions.items()})
    return {
        "seed": seed,
        "stream": stream,
        "w_fb": w_fb,
        **metrics,
        "interventions": interventions,
        "traj": {k: np.round(v.astype(np.float64), 6).tolist() for k, v in traj.items()},
        "eval": put(buf.getvalue(), name=f"ex-2.9.3-eval-{variant}.npz"),
    }


def publish_results(results: list[dict]) -> dict:
    """Publish scalar metrics as JSON and the stacked trajectories as one npz, under stable refs."""
    metrics = [{"run": i, **{k: v for k, v in r.items() if k not in ("eval", "traj")}} for i, r in enumerate(results)]
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.9.3-metrics.json"))

    trajs = {
        f"{i:03d}_{k}": np.asarray(v, dtype=np.float32) for i, r in enumerate(results) for k, v in r["traj"].items()
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **trajs)  # ty:ignore[invalid-argument-type]
    set_ref(TRAJS_REF, put(buf.getvalue(), name="ex-2.9.3-trajectories.npz"))

    # Exemplars: the worst redirect run in the replication cell, and its rescue under the cooler peak.
    def cell(peak_lr: float, anneal: bool) -> list[dict]:
        return [r for r in results if r["arm"] == "sweep" and r["peak_lr"] == peak_lr and r["anneal"] == anneal]

    worst = min(cell(0.10, True), key=lambda r: r["interventions"]["redirect"]["score"])
    rescue = next(r for r in cell(0.05, True) if r["seed"] == worst["seed"])
    set_ref("reports/ex-2.9.3/exemplar-hot", worst["eval"])
    set_ref("reports/ex-2.9.3/exemplar-cool", rescue["eval"])
    return {"n_published": len(metrics), "exemplar_seed": worst["seed"]}


def main(ctx: Ctx) -> dict:
    orig = make_dopesheet(0.10, anneal=True)
    jobs: list[tuple[dict, int, int | None, float, str]] = []

    # Arm 1 — trajectories: ex-2.9.2's base arm, bit-for-bit (w_fb = 0, original schedule).
    for seed in SEEDS:
        jobs.append(({"arm": "trajectories", "peak_lr": 0.10, "anneal": True}, seed, None, 0.0, orig))

    # Arm 2 — attribution: init × stream factorial on the same config.
    for init in ATTR_INITS:
        for stream in ATTR_STREAMS:
            jobs.append(({"arm": "attribution", "peak_lr": 0.10, "anneal": True}, init, stream, 0.0, orig))

    # Arm 3 — schedule sweep with fallback training, scored by the redirect intervention.
    for peak_lr in PEAK_LRS:
        for anneal in (True, False):
            csv = make_dopesheet(peak_lr, anneal=anneal)
            for seed in SEEDS:
                jobs.append(({"arm": "sweep", "peak_lr": peak_lr, "anneal": anneal}, seed, None, FALLBACK_WEIGHT, csv))

    meta, seeds, streams, wfbs, csvs = zip(*jobs, strict=True)
    results = ctx.map(train_one, list(seeds), list(streams), list(wfbs), list(csvs), role="train")
    results = [{**m, **r} for m, r in zip(meta, results, strict=True)]
    published = ctx.run(publish_results, results, role="publish")
    return {"n_runs": len(results), **published}


experiment = Experiment(
    name="ex-2.9.3",
    main=main,
    roles={
        "train": dict(timeout=600),  # CPU-only: ~840 params
        "publish": {},
    },
)
