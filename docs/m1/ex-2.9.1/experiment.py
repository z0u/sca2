"""
Experiment 2.9.1 redux: delete *red* from a tiny color autoencoder — in JAX.

A port of ex-preppy's experiment 2.9.1 ([repo](https://github.com/z0u/ex-preppy),
`docs/m2-control/ex-2.9.1-redux.ipynb`), run as an end-to-end shakedown of this
repo's infrastructure before the M2 transformer experiments. The science is M1's:
train an RGB autoencoder with Sparse Concept Anchoring so that *red* lands on
latent axis 0, then zero that axis and check that the damage is confined to
red-like colors.

Per seed: train 1501 steps on the 8×8×8 RGB grid with four regularizers on the
unit-normalized 5D bottleneck — anchor (pull red-labeled samples to e₀),
anti-anchor (repel everything from −e₀), separate (angular repulsion within a
batch), and anti-subspace (repel everything from axis 0) — with the loss weights
and LR driven per-step by the original dopesheet (`mini.temporal`, unchanged
from ex-preppy). Then ablate axis 0 and score the run: the R² between each
color's post-ablation reconstruction error and its HSV similarity to red. A high
score means the deletion was clean — errors land on red-like colors and nowhere
else.

The PyTorch/Lightning original becomes a handful of pure functions and one
`lax.scan`. Hooks, callbacks, DataLoaders, and even the model class don't
survive the port: the model is a pytree of arrays (task functions ship to
workers by value, so plain data beats clever classes), activations are just a
second return value, the schedule is an array indexed by step, and a "batch" is
64 random rows. The testbed itself — grids, model, loss terms, ablation — lives
in `sca.colorcube`, shared with the later ex-2.9.x experiments.

    bin/mini run docs/m1/ex-2.9.1/experiment.py --app modal --max-containers 8

The `train` role is CPU-only: the model has ~840 parameters, so a GPU would be
all overhead.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from sca.colorcube import (
    BATCH,
    GRID_RGB,
    RED_PROB,
    SIM3,
    VAL_RED,
    VAL_RGB,
    WEIGHT_PROPS,
    ablate,
    eval_model,
    init_params,
    loss_fn,
)
from mini import Ctx, Experiment, emit_metrics, emit_progress
from mini.store import put, set_ref
from mini.temporal import Dopesheet, Timeline, realize_timeline

SEEDS = list(range(16))  # the original swept 60 seeds; 16 is plenty for an infra test

DOPESHEET_CSV = (Path(__file__).parent / "dopesheet.csv").read_text()

# Store refs the report reads (see report.py).
METRICS_REF = "reports/ex-2.9.1/metrics"
BEST_EVAL_REF = "reports/ex-2.9.1/best-eval"


def train_one(seed: int, dopesheet_csv: str) -> dict:
    """Train one seeded run; return final val metrics, the ablation score, and an eval-dump artifact."""
    sheet = Dopesheet.from_csv(io.StringIO(dopesheet_csv))
    df = realize_timeline(Timeline(sheet))
    n_steps = len(sheet)
    lr = jnp.asarray(df["lr"].to_numpy(np.float32))
    weights = jnp.stack([jnp.asarray(df[p].to_numpy(np.float32)) for p in WEIGHT_PROPS], axis=1)

    x_train, p_red = jnp.asarray(GRID_RGB), jnp.asarray(RED_PROB)
    key, k_model = jr.split(jr.key(seed))
    params = init_params(k_model)
    opt = optax.adam(lambda count: lr[count])  # the dopesheet *is* the LR schedule
    opt_state = opt.init(params)

    @jax.jit
    def run_chunk(params, opt_state, key, steps):
        def step(carry, i):
            params, opt_state, key = carry
            key, k_batch, k_label = jr.split(key, 3)
            idx = jr.randint(k_batch, (BATCH,), 0, x_train.shape[0])  # bootstrap sample, like the original
            labels = jr.bernoulli(k_label, p_red[idx]).astype(jnp.float32)  # stochastic sparse labels
            (_, recon), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_train[idx], labels, weights[i])
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state, key), recon

        (params, opt_state, key), recons = jax.lax.scan(step, (params, opt_state, key), steps)
        return params, opt_state, key, recons[-1]

    for chunk in np.array_split(np.arange(n_steps), 10):
        params, opt_state, key, recon = run_chunk(params, opt_state, key, jnp.asarray(chunk))
        emit_progress(int(chunk[-1]) + 1, n_steps, message=f"seed {seed}")
        emit_metrics(recon=float(recon), lr=float(lr[chunk[-1]]))

    # Final validation (the original also validated mid-run; we only need the endpoint).
    mse_val, z_val = eval_model(params, jnp.asarray(VAL_RGB))
    # Score the run: ablate axis 0, then ask how tightly post-ablation error tracks similarity-to-red.
    mse_base, z_base = eval_model(params, x_train)
    mse_abl, z_abl = eval_model(ablate(params), x_train)
    r = np.corrcoef(SIM3, np.asarray(mse_abl))[0, 1]

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        rgb=GRID_RGB,
        sim3=SIM3,
        mse_base=np.asarray(mse_base),
        mse_abl=np.asarray(mse_abl),
        z_base=np.asarray(z_base),
        z_abl=np.asarray(z_abl),
    )
    metrics = {
        "score": float(r**2),
        "val_recon": float(jnp.mean(mse_val)),
        "val_anchor": float(jnp.mean(1.0 - z_val[VAL_RED, 0])),
        "val_anti_anchor": float(jnp.mean(jnp.maximum(-z_val[:, 0], 0.0))),
    }
    emit_metrics(**metrics)
    return {"seed": seed, **metrics, "eval": put(buf.getvalue(), name=f"ex-2.9.1-eval-seed{seed:02d}.npz")}


def publish_results(results: list[dict]) -> dict:
    """Publish per-seed metrics and the best run's eval dump under stable names for the report."""
    metrics = [{k: v for k, v in r.items() if k != "eval"} for r in results]
    best = max(results, key=lambda r: r["score"])
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.9.1-metrics.json"))
    set_ref(BEST_EVAL_REF, best["eval"])
    return {"best_seed": best["seed"], "best_score": best["score"]}


def main(ctx: Ctx) -> dict:
    results = ctx.map(train_one, SEEDS, [DOPESHEET_CSV] * len(SEEDS), role="train")
    best = ctx.run(publish_results, results, role="publish")
    return {"n_runs": len(results), **best}


experiment = Experiment(
    name="ex-2.9.1",
    main=main,
    roles={
        "train": dict(timeout=600),  # CPU-only: ~840 params
        "publish": {},
    },
)
