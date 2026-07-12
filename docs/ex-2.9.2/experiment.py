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

Two training variants share ex-2.9.1's model, data, and dopesheet
(`sca.colorcube` has the testbed): `base` (w_fb = 0) is ex-2.9.1's loss
unchanged; `fallback` (w_fb = 0.05) adds a decoder-only term
MSE(dec(−e₀), 0.5). Each trains 32 seeds. Every trained model is then scored
under five weight-level interventions on latent axis 0:

- zero: zero encoder row 0 + bias (ex-2.9.1's ablation; the baseline).
- oa: zero row 0, set the bias to the constant minimizing mean reconstruction
  error over the full RGB grid — optimal ablation as defined by Li & Janson.
- oa-nontarget: same, but the constant is optimized over non-red colors only —
  the adaptation you'd want for removal (spare bystanders; ignore the target).
- reflect: negate row 0 + bias, so z₀ → −z₀ pre-norm; red lands on −e₀. Not a
  true deletion (a sign flip restores it) but a clean redirect.
- redirect: zero row 0, set the bias to −γ (γ = 1) — a true deletion (the
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

from sca.colorcube import (
    BATCH,
    GAMMA,
    GRID_RGB,
    NEG_E0,
    OTHERS,
    PURE_RED,
    RED_PROB,
    REDS,
    SIM3,
    VAL_RED,
    VAL_RGB,
    WEIGHT_PROPS,
    decode,
    edit_axis0,
    eval_model,
    init_params,
    loss_fn,
    optimal_constant,
)
from mini import Ctx, Experiment, emit_metrics, emit_progress
from mini.store import put, set_ref
from mini.temporal import Dopesheet, Timeline, realize_timeline

SEEDS = list(range(32))

FALLBACK_WEIGHT = 0.05  # w_fb for the fallback variant; constant over training
INTERVENTIONS = ("zero", "oa", "oa-nontarget", "reflect", "redirect")

DOPESHEET_CSV = (Path(__file__).parent / "dopesheet.csv").read_text()

# Store refs the report reads (see report.py).
METRICS_REF = "reports/ex-2.9.2/metrics"
EXEMPLAR_REFS = {"base": "reports/ex-2.9.2/exemplar-base", "fallback": "reports/ex-2.9.2/exemplar-fallback"}


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
        "redirect": edit_axis0(params, bias=-GAMMA),
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
