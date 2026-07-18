"""
Experiment 2.9.4: closed-loop regularizer weights.

Ex-2.9.3 found that anchoring failures are optimization instabilities: the
anchored solution is metastable during the high-LR plateau, and the timed
regularizer anneal removes its protection while the optimizer is still hot.
A cooler LR peak fixes that statically. This experiment tests the dynamic
alternative: replace the timed anneal of the anchor and anti-anchor weights
with feedback, so protection is present exactly when the training signals say
the constraint is violated, and absent otherwise.

The controller is dual ascent with hysteresis, driven only by signals available
during training (no ground-truth probes):

- Each controlled term keeps an EMA of its own raw value. The anchor term is
  measured on labeled samples only, so its EMA updates on the ~6% of batches
  that contain a label.
- The weight λ rises at η·(ema − τ_hi) while the EMA is above τ_hi, decays at
  5η·(τ_lo − ema) while below τ_lo, and holds in between. The deadband and the
  fast decay stop the integrator from winding up during the (normal) early
  transient and keep λ near zero when training is healthy — a live anchor
  weight late in training otherwise fights the label noise, dragging
  pinkish-labeled colors onto the axis.
- λ is capped near the dopesheet's proven constant (anchor 0.15; the sheet
  held 0.1), so even a saturated controller cannot over-anchor much.

The anti-subspace weight is not controlled: its raw value has a red-mass floor
that a labeled-blind controller can't separate from leakage, and ex-2.9.3's
sweep shows simply holding its small late value (0.003) is enough. The LR and
`separate` still come from the dopesheet.

Conditions (32 seeds each, scored by the redirect intervention): {static,
ctrl} × peak LR {0.10 hostile, 0.05 benign} with ex-2.9.2's fallback term; a
coarse sensitivity grid at the hostile LR (anchor targets ×0.75 and ×1.5,
gains ×0.5 and ×2); and a fallback-free pair {static, ctrl} at the hostile LR,
because that's the config where catastrophic failures actually occur (5/160
in ex-2.9.3's base arms). "static" is the original timed-anneal schedule at
that peak. The questions: does feedback prevent the catastrophes where they
exist, what does it cost where they don't, and is it knife-edge in its own
hyperparameters? The testbed (model, grids, loss terms, interventions) is
`sca.colorcube`.

    bin/mini run docs/m1/ex-2.9.4/experiment.py --app modal --max-containers 16
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
    GRAY,
    GRID_RGB,
    NEG_E0,
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
    loss_terms,
    make_dopesheet,
    score_interventions,
)
from mini import Ctx, Experiment, emit_metrics, emit_progress
from mini.store import put, set_ref
from mini.temporal import Dopesheet, Timeline, realize_timeline

SEEDS = list(range(32))

FALLBACK_WEIGHT = 0.05

# Controller constants (order: anchor, anti-anchor)
TAU_HI = np.array([0.20, 0.02])  # engage above this EMA level
TAU_LO = np.array([0.10, 0.005])  # release below this level; hold in the deadband
ETA = 0.005  # dual ascent rate; decay is 5× faster (see module docstring)
CAPS = np.array([0.15, 0.05])  # λ ceilings, near the dopesheet's proven constants
EMA_ALPHA = np.array([0.2, 0.02])  # anchor EMA updates only on labeled batches, so it's faster

# Store refs the report reads (see report.py).
METRICS_REF = "reports/ex-2.9.4/metrics"
TRAJS_REF = "reports/ex-2.9.4/trajectories"


def train_one(seed: int, peak_lr: float, ctrl: bool, tau_scale: float, eta_scale: float, w_fb: float) -> dict:
    """Train one run, static (timed anneal) or controlled (feedback duals), and score it.

    With ctrl=True the anchor and anti-anchor weights come from the hysteresis duals
    (targets scaled by *tau_scale*, rates by *eta_scale*); the anti-subspace weight is the
    dopesheet's, clamped to its step-750 value so it never anneals to zero. With ctrl=False
    all four weights follow the timed-anneal dopesheet and the scales are ignored.
    """
    sheet = Dopesheet.from_csv(io.StringIO(make_dopesheet(peak_lr, anneal=True)))
    df = realize_timeline(Timeline(sheet))
    n_steps = len(sheet)
    lr = jnp.asarray(df["lr"].to_numpy(np.float32))
    w_sheet = np.stack([df[p].to_numpy(np.float32) for p in WEIGHT_PROPS], axis=1)
    if ctrl:
        w_sheet = w_sheet.copy()
        w_sheet[:, 1:3] = 0.0  # anchor and anti-anchor are the duals' job
        w_sheet[:, 3] = np.maximum(w_sheet[:, 3], w_sheet[750, 3])  # hold anti-subspace, never anneal
    w_sheet = jnp.asarray(w_sheet)
    tau_hi = jnp.asarray(TAU_HI * tau_scale)
    tau_lo = jnp.asarray(TAU_LO * tau_scale)
    eta = ETA * eta_scale
    caps = jnp.asarray(CAPS)
    alpha = jnp.asarray(EMA_ALPHA)

    x_train, p_red = jnp.asarray(GRID_RGB), jnp.asarray(RED_PROB)
    key, k_model = jr.split(jr.key(seed))
    params = init_params(k_model)
    opt = optax.adam(lambda count: lr[count])
    opt_state = opt.init(params)

    others = jnp.asarray(np.flatnonzero(OTHERS))
    pure_red = jnp.asarray(PURE_RED)
    neg_e0 = jnp.asarray(NEG_E0)

    def loss_fn(params, x, labels, w):
        recon, terms = loss_terms(params, x, labels)
        fallback = jnp.mean((decode(params, neg_e0) - GRAY) ** 2)
        return recon + terms @ jax.lax.stop_gradient(w) + w_fb * fallback, (recon, terms)

    @jax.jit
    def run_chunk(params, opt_state, key, ema, lam, steps):
        def step(carry, i):
            params, opt_state, key, ema, lam = carry
            key, k_batch, k_label = jr.split(key, 3)
            idx = jr.randint(k_batch, (BATCH,), 0, x_train.shape[0])  # bootstrap sample
            labels = jr.bernoulli(k_label, p_red[idx]).astype(jnp.float32)  # stochastic sparse labels
            w = w_sheet[i].at[1:3].add(lam) if ctrl else w_sheet[i]
            (_, (recon, terms)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_train[idx], labels, w)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = cast(Params, optax.apply_updates(params, updates))
            if ctrl:  # dual updates from the controlled terms' EMAs (anchor updates only on labeled batches)
                has_label = (jnp.sum(labels) > 0).astype(jnp.float32)
                ema = ema + jnp.stack([has_label * alpha[0], alpha[1]]) * (terms[1:3] - ema)
                rise, fall = jnp.maximum(ema - tau_hi, 0.0), jnp.maximum(tau_lo - ema, 0.0)
                lam = jnp.clip(lam + eta * rise - 5 * eta * fall, 0.0, caps)
            # Anchoring diagnostics on the full grid, for the failure-timing analysis
            _, z = forward(params, x_train)
            _, z_red = forward(params, pure_red)
            diag = {
                "z0_red": z_red[0, 0],
                "leak": jnp.mean(jnp.abs(z[others, 0])),
                "recon": jnp.mean((jnp.clip(decode(params, z), 0.0, 1.0) - x_train) ** 2),
                "lam_anchor": lam[0],
                "lam_anti_anchor": lam[1],
            }
            return (params, opt_state, key, ema, lam), diag

        (params, opt_state, key, ema, lam), diags = jax.lax.scan(step, (params, opt_state, key, ema, lam), steps)
        return params, opt_state, key, ema, lam, diags

    variant = f"{'ctrl' if ctrl else 'static'}-lr{peak_lr}-fb{w_fb}-s{seed}"
    ema, lam = jnp.asarray(tau_hi), jnp.zeros(2)
    diags = []
    for chunk in np.array_split(np.arange(n_steps), 10):
        params, opt_state, key, ema, lam, d = run_chunk(params, opt_state, key, ema, lam, jnp.asarray(chunk))
        diags.append({k: np.asarray(v) for k, v in d.items()})
        emit_progress(int(chunk[-1]) + 1, n_steps, message=variant)
        emit_metrics(recon=float(d["recon"][-1]), z0_red=float(d["z0_red"][-1]), lam_anchor=float(lam[0]))
    traj = {k: np.concatenate([d[k] for d in diags])[::TRAJ_STRIDE] for k in diags[0]}

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
        "lam_anchor_mean": float(np.mean(traj["lam_anchor"])),
        "lam_anchor_end": float(traj["lam_anchor"][-1]),
    }
    emit_metrics(**metrics, **{f"score_{k}": v["score"] for k, v in interventions.items()})
    return {
        "seed": seed,
        **metrics,
        "interventions": interventions,
        "traj": {k: np.round(v.astype(np.float64), 6).tolist() for k, v in traj.items()},
        "eval": put(buf.getvalue(), name=f"ex-2.9.4-eval-{variant}.npz"),
    }


def publish_results(results: list[dict]) -> dict:
    """Publish scalar metrics as JSON and the stacked trajectories as one npz, under stable refs."""
    metrics = [{"run": i, **{k: v for k, v in r.items() if k not in ("eval", "traj")}} for i, r in enumerate(results)]
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.9.4-metrics.json"))

    trajs = {
        f"{i:03d}_{k}": np.asarray(v, dtype=np.float32) for i, r in enumerate(results) for k, v in r["traj"].items()
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **trajs)  # ty:ignore[invalid-argument-type]
    set_ref(TRAJS_REF, put(buf.getvalue(), name="ex-2.9.4-trajectories.npz"))
    return {"n_published": len(metrics)}


def main(ctx: Ctx) -> dict:
    jobs: list[tuple[dict, int, float, bool, float, float, float]] = []
    for cond, peak_lr, ctrl, tau_s, eta_s, w_fb in [
        ("static", 0.10, False, 1.0, 1.0, FALLBACK_WEIGHT),
        ("static", 0.05, False, 1.0, 1.0, FALLBACK_WEIGHT),
        ("ctrl", 0.10, True, 1.0, 1.0, FALLBACK_WEIGHT),
        ("ctrl", 0.05, True, 1.0, 1.0, FALLBACK_WEIGHT),
        # Sensitivity at the troublesome LR: are the controller's own params twitchy?
        ("ctrl-tau0.75", 0.10, True, 0.75, 1.0, FALLBACK_WEIGHT),
        ("ctrl-tau1.5", 0.10, True, 1.5, 1.0, FALLBACK_WEIGHT),
        ("ctrl-eta0.5", 0.10, True, 1.0, 0.5, FALLBACK_WEIGHT),
        ("ctrl-eta2", 0.10, True, 1.0, 2.0, FALLBACK_WEIGHT),
        # Fallback-free cells: catastrophic failures exist here (ex-2.9.3 arm 1) — does feedback prevent them?
        ("static-nofb", 0.10, False, 1.0, 1.0, 0.0),
        ("ctrl-nofb", 0.10, True, 1.0, 1.0, 0.0),
    ]:
        for seed in SEEDS:
            jobs.append(
                (
                    {"cond": cond, "peak_lr": peak_lr, "ctrl": ctrl, "w_fb": w_fb},
                    seed,
                    peak_lr,
                    ctrl,
                    tau_s,
                    eta_s,
                    w_fb,
                )
            )

    meta, seeds, lrs, ctrls, taus, etas, wfbs = zip(*jobs, strict=True)
    results = ctx.map(train_one, list(seeds), list(lrs), list(ctrls), list(taus), list(etas), list(wfbs), role="train")
    results = [{**m, **r} for m, r in zip(meta, results, strict=True)]
    published = ctx.run(publish_results, results, role="publish")
    return {"n_runs": len(results), **published}


experiment = Experiment(
    name="ex-2.9.4",
    main=main,
    roles={
        "train": dict(timeout=600),  # CPU-only: ~840 params
        "publish": {},
    },
)
