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
64 random rows.

    bin/mini run docs/ex-2.9.1/experiment.py --app modal --max-containers 8

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
from matplotlib import colors as mcolors

from mini import Ctx, Experiment, emit_metrics, emit_progress
from mini.store import put, set_ref
from mini.temporal import Dopesheet, Timeline, realize_timeline

K = 5  # bottleneck dim; red is anchored to axis 0
DIMS = (3, 16, 16, K, 16, 16, 3)  # bottleneck sits after layer 2 (~840 params)
BATCH = 64
SEEDS = list(range(16))  # the original swept 60 seeds; 16 is plenty for an infra test
WEIGHT_PROPS = ("separate", "anchor", "anti-anchor", "anti-subspace")

DOPESHEET_CSV = (Path(__file__).parent / "dopesheet.csv").read_text()

# Store refs the report reads (see report.py).
METRICS_REF = "reports/ex-2.9.1/metrics"
BEST_EVAL_REF = "reports/ex-2.9.1/best-eval"

type Params = list[dict[str, jax.Array]]


def _grid(coords: np.ndarray) -> np.ndarray:
    """The RGB cube sampled at *coords* along each axis: [len(coords)³, 3]."""
    r, g, b = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(-1, 3).astype(np.float32)


def _redness(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb.T
    return r * (1 - g / 2 - b / 2)


def _sim_to_red(rgb: np.ndarray, power: float = 3.0) -> np.ndarray:
    """Angular HSV similarity to pure red, weighted by vibrancy (ports ex-preppy's `hsv_similarity`)."""
    h, s, v = mcolors.rgb_to_hsv(rgb).T
    angle = 360.0 * np.minimum(h, 1.0 - h)  # hue distance to red, in degrees
    hue_sim = np.maximum(0.0, (90.0 - angle) / 90.0)
    vib = (s * v + 1.0) / 2.0  # mean vibrancy of (color, pure red); hue only matters for vibrant colors
    sim = (vib * hue_sim + 1.0 - vib) * (1.0 - np.abs(s - 1.0)) * (1.0 - np.abs(v - 1.0))
    return (sim**power).astype(np.float32)


GRID_RGB = _grid(np.linspace(0, 1, 8))  # train set and scoring grid: 512 corner points
RED_PROB = (_redness(GRID_RGB) ** 8 * 0.08).astype(np.float32)  # sparse, noisy label: P(labeled red)
SIM3 = _sim_to_red(GRID_RGB)
VAL_RGB = np.concatenate([_grid(np.linspace(1 / 16, 15 / 16, 7)), _grid(np.array([0.0, 1.0]))])  # centers + corners
VAL_RED = _redness(VAL_RGB) == 1.0  # exact label: only pure red


def init_params(key: jax.Array) -> Params:
    """Linear layers as plain dicts, uniform ±1/√fan_in (matching torch/equinox defaults)."""

    def linear(k: jax.Array, n_in: int, n_out: int) -> dict[str, jax.Array]:
        kw, kb = jr.split(k)
        lim = 1.0 / np.sqrt(n_in)
        return {
            "w": jr.uniform(kw, (n_out, n_in), minval=-lim, maxval=lim),
            "b": jr.uniform(kb, (n_out,), minval=-lim, maxval=lim),
        }

    keys = jr.split(key, len(DIMS) - 1)
    return [linear(k, a, b) for k, a, b in zip(keys, DIMS[:-1], DIMS[1:], strict=True)]


def forward(params: Params, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """3 → 16 → 16 → [unit 5-sphere] → 16 → 16 → 3, GELU between the linears.

    Returns (reconstruction, unit latent) for a batch.
    """
    enc, dec = params[:3], params[3:]
    for lyr in enc[:-1]:
        x = jax.nn.gelu(x @ lyr["w"].T + lyr["b"])
    z = x @ enc[-1]["w"].T + enc[-1]["b"]
    z = z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    y = z
    for lyr in dec[:-1]:
        y = jax.nn.gelu(y @ lyr["w"].T + lyr["b"])
    return y @ dec[-1]["w"].T + dec[-1]["b"], z


def loss_fn(params: Params, x: jax.Array, labels: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Reconstruction MSE plus the four weighted regularizers on the bottleneck."""
    y, z = forward(params, x)
    recon = jnp.mean((y - x) ** 2)

    cos_red = z[:, 0]  # z is unit-norm and RED = e₀, so cos(z, RED) is just z₀
    anchor = jnp.sum((1.0 - cos_red) * labels) / (jnp.sum(labels) + 1e-8)  # label-affinity-weighted mean
    anti_anchor = jnp.mean(jnp.maximum(-cos_red, 0.0))  # hemisphere gate: clamp(cos(z, −e₀), min=0)
    anti_subspace = jnp.mean(cos_red**2)

    cos_pairs = z @ z.T
    shifted = jnp.where(jnp.isclose(cos_pairs, 1.0), 0.0, (cos_pairs + 1.0) / 2.0)  # null self/duplicate similarity
    separate = jnp.mean(jnp.sum(shifted**100.0, axis=-1))  # high power: only near-duplicates repel

    terms = jnp.stack([separate, anchor, anti_anchor, anti_subspace])  # order matches WEIGHT_PROPS
    return recon + terms @ w, recon


def eval_model(params: Params, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-point reconstruction MSE (output clamped to [0,1], as at inference) and latents."""
    y, z = forward(params, x)
    return jnp.mean((jnp.clip(y, 0.0, 1.0) - x) ** 2, axis=-1), z


def ablate(params: Params) -> Params:
    """Delete latent axis 0: zero the encoder's output row 0 (and bias) and the decoder's input column 0."""
    params = [dict(lyr) for lyr in params]
    params[2] = {"w": params[2]["w"].at[0].set(0.0), "b": params[2]["b"].at[0].set(0.0)}
    params[3] = {**params[3], "w": params[3]["w"].at[:, 0].set(0.0)}
    return params


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
