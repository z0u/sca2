"""
Experiment 2.9.2: fallback control for deleting *red*.

Ex-2.9.1 reproduced M1's headline result and also its variance: the ablation
score (R² between post-ablation reconstruction error and similarity to red)
swings widely across seeds, so getting a clean deletion means sweeping seeds
and picking a winner. The SCA paper's discussion attributes this to unreliable
redistribution — zeroing the anchored axis and renormalizing sends *red*
somewhere random — and points at optimal ablation (Li & Janson 2024,
arXiv:2409.09951) as a possible fix. This experiment tests that suggestion,
and a training-time alternative: *fallback control*, where the decoder is
trained to map the anti-anchor point −e₀ (kept empty by the anti-anchor
regularizer) to mid-gray, so an intervention can redirect red somewhere with a
*defined* response.

Two training variants share ex-2.9.1's model, data, and dopesheet:
`base` (w_fb = 0) is ex-2.9.1's loss unchanged; `fallback` (w_fb = 0.05) adds
a decoder-only term MSE(dec(−e₀), 0.5). Each trains 32 seeds. Every trained
model is then scored under five weight-level interventions on latent axis 0:

- zero: zero encoder row 0 + bias (ex-2.9.1's ablation; the baseline).
- oa: zero row 0, set the bias to the constant minimizing mean reconstruction
  error over the full RGB grid — optimal ablation as defined by Li & Janson.
- oa-nontarget: same, but the constant is optimized over non-red colors only —
  the adaptation you'd want for removal (spare bystanders; ignore the target).
- reflect: negate row 0 + bias, so z₀ → −z₀ pre-norm; red lands on −e₀. Not a
  true deletion (a sign flip restores it) but a clean redirect.
- redirect: zero row 0, set the bias to −β (β = 1) — a true deletion (the
  redness computation is gone) plus a constant redirect toward −e₀, felt in
  proportion to how much of an input's pre-norm activation the deletion removed.

Per run we record each intervention's score, damage to red, and collateral
damage, plus the trained fallback color dec(−e₀) and axis-0 leakage. The
comparison is distributional across seeds: fallback control should collapse
the variance of the intervention response, not shift the mean.

    bin/mini run docs/ex-2.9.2/experiment.py --app modal --max-containers 8
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
SEEDS = list(range(32))
WEIGHT_PROPS = ("separate", "anchor", "anti-anchor", "anti-subspace")

GRAY = 0.5  # fallback target: dec(−e₀) → mid-gray, the "know-nothing" color
FALLBACK_WEIGHT = 0.05  # w_fb for the fallback variant; constant over training
BETA = 1.0  # redirect strength: encoder bias after deletion is −β
OA_GRID = np.linspace(-3.0, 3.0, 121)  # line-search grid for the optimal constant
INTERVENTIONS = ("zero", "oa", "oa-nontarget", "reflect", "redirect")

DOPESHEET_CSV = (Path(__file__).parent / "dopesheet.csv").read_text()

# Store refs the report reads (see report.py).
METRICS_REF = "reports/ex-2.9.2/metrics"
EXEMPLAR_REFS = {"base": "reports/ex-2.9.2/exemplar-base", "fallback": "reports/ex-2.9.2/exemplar-fallback"}

type Params = list[dict[str, jax.Array]]


def _grid(coords: np.ndarray) -> np.ndarray:
    """The RGB cube sampled at *coords* along each axis: [len(coords)³, 3]."""
    r, g, b = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(-1, 3).astype(np.float32)


def _redness(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb.T
    return r * (1 - g / 2 - b / 2)


def _sim_to_red(rgb: np.ndarray, power: float = 3.0) -> np.ndarray:
    """Angular HSV similarity to pure red, weighted by vibrancy (as in ex-2.9.1)."""
    h, s, v = mcolors.rgb_to_hsv(rgb).T
    angle = 360.0 * np.minimum(h, 1.0 - h)  # hue distance to red, in degrees
    hue_sim = np.maximum(0.0, (90.0 - angle) / 90.0)
    vib = (s * v + 1.0) / 2.0  # mean vibrancy of (color, pure red); hue only matters for vibrant colors
    sim = (vib * hue_sim + 1.0 - vib) * (1.0 - np.abs(s - 1.0)) * (1.0 - np.abs(v - 1.0))
    return (sim**power).astype(np.float32)


GRID_RGB = _grid(np.linspace(0, 1, 8))  # train set and scoring grid: 512 corner points
RED_PROB = (_redness(GRID_RGB) ** 8 * 0.08).astype(np.float32)  # sparse, noisy label: P(labeled red)
SIM3 = _sim_to_red(GRID_RGB)
REDS = SIM3 > 0.5  # "damage to red" group
OTHERS = SIM3 < 0.01  # "collateral damage" group (404 of 512 grid points)
VAL_RGB = np.concatenate([_grid(np.linspace(1 / 16, 15 / 16, 7)), _grid(np.array([0.0, 1.0]))])  # centers + corners
VAL_RED = _redness(VAL_RGB) == 1.0  # exact label: only pure red
PURE_RED = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
NEG_E0 = np.zeros((1, K), dtype=np.float32)
NEG_E0[0, 0] = -1.0


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


def decode(params: Params, z: jax.Array) -> jax.Array:
    """The decoder half: latent → RGB (no clamping)."""
    y = z
    for lyr in params[3:-1]:
        y = jax.nn.gelu(y @ lyr["w"].T + lyr["b"])
    return y @ params[-1]["w"].T + params[-1]["b"]


def forward(params: Params, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """3 → 16 → 16 → [unit 5-sphere] → 16 → 16 → 3, GELU between the linears.

    Returns (reconstruction, unit latent) for a batch.
    """
    enc = params[:3]
    for lyr in enc[:-1]:
        x = jax.nn.gelu(x @ lyr["w"].T + lyr["b"])
    z = x @ enc[-1]["w"].T + enc[-1]["b"]
    z = z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    return decode(params, z), z


def loss_fn(params: Params, x: jax.Array, labels: jax.Array, w: jax.Array, w_fb: float) -> tuple[jax.Array, jax.Array]:
    """Ex-2.9.1's loss (recon + four bottleneck regularizers) plus the fallback term."""
    y, z = forward(params, x)
    recon = jnp.mean((y - x) ** 2)

    cos_red = z[:, 0]  # z is unit-norm and RED = e₀, so cos(z, RED) is just z₀
    anchor = jnp.sum((1.0 - cos_red) * labels) / (jnp.sum(labels) + 1e-8)  # label-affinity-weighted mean
    anti_anchor = jnp.mean(jnp.maximum(-cos_red, 0.0))  # hemisphere gate: clamp(cos(z, −e₀), min=0)
    anti_subspace = jnp.mean(cos_red**2)

    cos_pairs = z @ z.T
    shifted = jnp.where(jnp.isclose(cos_pairs, 1.0), 0.0, (cos_pairs + 1.0) / 2.0)  # null self/duplicate similarity
    separate = jnp.mean(jnp.sum(shifted**100.0, axis=-1))  # high power: only near-duplicates repel

    fallback = jnp.mean((decode(params, jnp.asarray(NEG_E0)) - GRAY) ** 2)  # decoder-only: dec(−e₀) → gray

    terms = jnp.stack([separate, anchor, anti_anchor, anti_subspace])  # order matches WEIGHT_PROPS
    return recon + terms @ w + w_fb * fallback, recon


def eval_model(params: Params, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-point reconstruction MSE (output clamped to [0,1], as at inference) and latents."""
    y, z = forward(params, x)
    return jnp.mean((jnp.clip(y, 0.0, 1.0) - x) ** 2, axis=-1), z


def edit_axis0(params: Params, *, bias: float | None = None, negate: bool = False) -> Params:
    """Weight-level interventions on latent axis 0, all edits to the encoder's output layer.

    negate=True reflects (z₀ → −z₀ pre-norm). Otherwise row 0 is zeroed — the redness
    computation is deleted — and the bias becomes *bias* (0 = ex-2.9.1's zero ablation,
    a constant c = optimal ablation, −β = redirect to the fallback direction).
    """
    params = [dict(lyr) for lyr in params]
    if negate:
        params[2] = {"w": params[2]["w"].at[0].mul(-1.0), "b": params[2]["b"].at[0].mul(-1.0)}
    else:
        params[2] = {"w": params[2]["w"].at[0].set(0.0), "b": params[2]["b"].at[0].set(bias or 0.0)}
    return params


def optimal_constant(params: Params, x: jax.Array, subset: np.ndarray | None = None) -> float:
    """Line-search the post-ablation bias c* that minimizes mean reconstruction error.

    Li & Janson find a* by SGD; in 1D an exact line search is simpler. *subset* masks the
    distribution the constant is optimized over (None = the full grid, per the paper).
    """
    xs = x if subset is None else x[np.flatnonzero(subset)]
    losses = [float(jnp.mean(eval_model(edit_axis0(params, bias=float(c)), xs)[0])) for c in OA_GRID]
    return float(OA_GRID[int(np.argmin(losses))])


def train_one(seed: int, w_fb: float, dopesheet_csv: str) -> dict:
    """Train one seeded run and score all five interventions; return metrics + an eval-dump artifact."""
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
            idx = jr.randint(k_batch, (BATCH,), 0, x_train.shape[0])  # bootstrap sample
            labels = jr.bernoulli(k_label, p_red[idx]).astype(jnp.float32)  # stochastic sparse labels
            (_, recon), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, x_train[idx], labels, weights[i], w_fb
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state, key), recon

        (params, opt_state, key), recons = jax.lax.scan(step, (params, opt_state, key), steps)
        return params, opt_state, key, recons[-1]

    variant = "fallback" if w_fb else "base"
    for chunk in np.array_split(np.arange(n_steps), 10):
        params, opt_state, key, recon = run_chunk(params, opt_state, key, jnp.asarray(chunk))
        emit_progress(int(chunk[-1]) + 1, n_steps, message=f"{variant} seed {seed}")
        emit_metrics(recon=float(recon), lr=float(lr[chunk[-1]]))

    # Validation and geometry diagnostics on the un-edited model.
    mse_val, z_val = eval_model(params, jnp.asarray(VAL_RGB))
    mse_base, z_base = eval_model(params, x_train)
    leak = float(np.mean(np.abs(np.asarray(z_base)[OTHERS, 0])))  # axis-0 occupancy of non-red colors
    fb_color = np.clip(np.asarray(decode(params, jnp.asarray(NEG_E0)))[0], 0.0, 1.0)

    c_oa = optimal_constant(params, x_train)
    c_oa_nt = optimal_constant(params, x_train, subset=OTHERS)
    edits = {
        "zero": edit_axis0(params, bias=0.0),
        "oa": edit_axis0(params, bias=c_oa),
        "oa-nontarget": edit_axis0(params, bias=c_oa_nt),
        "reflect": edit_axis0(params, negate=True),
        "redirect": edit_axis0(params, bias=-BETA),
    }
    per_intervention, dumps = {}, {}
    for name, edited in edits.items():
        mse, z = eval_model(edited, x_train)
        mse = np.asarray(mse)
        per_intervention[name] = {
            "score": float(np.corrcoef(SIM3, mse)[0, 1] ** 2),
            "red": float(mse[REDS].mean()),
            "collateral": float(mse[OTHERS].mean()),
            "red_pure": float(np.asarray(eval_model(edited, jnp.asarray(PURE_RED))[0])[0]),
        }
        dumps[f"mse_{name}"] = mse
        dumps[f"z_{name}"] = np.asarray(z)

    buf = io.BytesIO()
    np.savez_compressed(buf, rgb=GRID_RGB, sim3=SIM3, mse_base=np.asarray(mse_base), z_base=np.asarray(z_base), **dumps)
    metrics = {
        "val_recon": float(jnp.mean(mse_val)),
        "val_anchor": float(jnp.mean(1.0 - z_val[VAL_RED, 0])),
        "val_anti_anchor": float(jnp.mean(jnp.maximum(-z_val[:, 0], 0.0))),
        "leak": leak,
        "c_oa": c_oa,
        "c_oa_nt": c_oa_nt,
    }
    emit_metrics(**metrics, **{f"score_{k}": v["score"] for k, v in per_intervention.items()})
    return {
        "seed": seed,
        "variant": variant,
        **metrics,
        "fallback_color": fb_color.tolist(),
        "interventions": per_intervention,
        "eval": put(buf.getvalue(), name=f"ex-2.9.2-eval-{variant}-seed{seed:02d}.npz"),
    }


def publish_results(results: list[dict]) -> dict:
    """Publish all per-run metrics and one median exemplar dump per variant under stable refs."""
    metrics = [{k: v for k, v in r.items() if k != "eval"} for r in results]
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.9.2-metrics.json"))

    # Exemplar = the median run of each variant, ranked by its headline intervention's score.
    headline = {"base": "zero", "fallback": "reflect"}
    out = {}
    for variant, ref in EXEMPLAR_REFS.items():
        runs = sorted(
            (r for r in results if r["variant"] == variant),
            key=lambda r: r["interventions"][headline[variant]]["score"],
        )
        exemplar = runs[len(runs) // 2]
        set_ref(ref, exemplar["eval"])
        out[f"{variant}_exemplar_seed"] = exemplar["seed"]
    return out


def main(ctx: Ctx) -> dict:
    seeds = SEEDS * 2
    wfbs = [0.0] * len(SEEDS) + [FALLBACK_WEIGHT] * len(SEEDS)
    results = ctx.map(train_one, seeds, wfbs, [DOPESHEET_CSV] * len(seeds), role="train")
    published = ctx.run(publish_results, results, role="publish")
    return {"n_runs": len(results), **published}


experiment = Experiment(
    name="ex-2.9.2",
    main=main,
    roles={
        "train": dict(timeout=600),  # CPU-only: ~840 params
        "publish": {},
    },
)
