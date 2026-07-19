"""
The tiny color-autoencoder testbed shared by experiments ex-2.9.1–2.9.4.

Ported from ex-preppy (M1): an RGB autoencoder trained with Sparse Concept
Anchoring so that *red* lands on axis 0 of a unit-normalized 5D bottleneck.
This module holds what the ex-2.9.x experiments share — the data grids, the
model, the raw loss terms, the weight-level interventions on the anchored
axis, and the scoring and report helpers. Each experiment composes these with
its own loss weighting, schedule, and training loop, which stay in the
experiment's own file.

Edits here are memoization *evidence* for every experiment that imports the
edited definition (mi-ni tracks project source transitively), so a change
re-runs the affected tasks on their next invocation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from matplotlib import colors as mcolors

from mini.store import project_store

if TYPE_CHECKING:
    from matplotlib.axes import Axes

K = 5  # bottleneck dim; red is anchored to axis 0
DIMS = (3, 16, 16, K, 16, 16, 3)  # bottleneck sits after layer 2 (~840 params)
BATCH = 64
WEIGHT_PROPS = ("separate", "anchor", "anti-anchor", "anti-subspace")

GRAY = 0.5  # fallback target: dec(−e₀) → mid-gray, the "know-nothing" color (see ex-2.9.2)
GAMMA = 1.0  # redirect strength (γ): encoder bias after deletion is −γ
OA_GRID = np.linspace(-3.0, 3.0, 121)  # line-search grid for the optimal constant
TRAJ_STRIDE = 5  # trajectory-recording experiments keep diagnostics every N steps

type Params = list[dict[str, jax.Array]]


def make_dopesheet(peak_lr: float, anneal: bool) -> str:
    """Ex-2.9.1's dopesheet with a parameterized LR plateau and an optional regularizer anneal.

    anneal=True reproduces the original: all four regularizer weights ramp to zero by step
    1425 (90% through the second phase). anneal=False deletes that keyframe, so each weight
    holds its last keyed value (anchor 0.1, anti-anchor 0.05, anti-subspace 0.003,
    separate 0.001) to the end. The final LR is always half the peak, as in the original.
    """
    rows = [
        "STEP,PHASE,ACTION,lr,separate,anchor,anti-anchor,anti-subspace",
        "0,Train,,1e-8,,0,0,0.25",
        "+10,,,0.01,,,,",
        "+0.33,,,,0.01,0.1,0.05,",
        f"750,,,{peak_lr},0.001,0.1,,0.003",
        *([f"+0.9,,,{peak_lr},0,0,0,0"] if anneal else []),
        f"1500,,,{peak_lr / 2},,,,",
    ]
    return "\n".join(rows) + "\n"


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


def loss_terms(params: Params, x: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Reconstruction MSE and the raw regularizer terms, unweighted (order matches WEIGHT_PROPS)."""
    y, z = forward(params, x)
    recon = jnp.mean((y - x) ** 2)

    cos_red = z[:, 0]  # z is unit-norm and RED = e₀, so cos(z, RED) is just z₀
    anchor = jnp.sum((1.0 - cos_red) * labels) / (jnp.sum(labels) + 1e-8)  # label-affinity-weighted mean
    anti_anchor = jnp.mean(jnp.maximum(-cos_red, 0.0))  # hemisphere gate: clamp(cos(z, −e₀), min=0)
    anti_subspace = jnp.mean(cos_red**2)

    cos_pairs = z @ z.T
    shifted = jnp.where(jnp.isclose(cos_pairs, 1.0), 0.0, (cos_pairs + 1.0) / 2.0)  # null self/duplicate similarity
    separate = jnp.mean(jnp.sum(shifted**100.0, axis=-1))  # high power: only near-duplicates repel

    return recon, jnp.stack([separate, anchor, anti_anchor, anti_subspace])


def loss_fn(
    params: Params, x: jax.Array, labels: jax.Array, w: jax.Array, w_fb: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """Recon + the four weighted bottleneck regularizers + the decoder-only fallback term.

    w_fb=0 is ex-2.9.1's loss; w_fb>0 adds ex-2.9.2's fallback term MSE(dec(−e₀), gray).
    """
    recon, terms = loss_terms(params, x, labels)
    fallback = jnp.mean((decode(params, jnp.asarray(NEG_E0)) - GRAY) ** 2)  # decoder-only: dec(−e₀) → gray
    return recon + terms @ w + w_fb * fallback, recon


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


def edit_axis0(params: Params, *, bias: float | None = None, negate: bool = False) -> Params:
    """Weight-level interventions on latent axis 0, all edits to the encoder's output layer.

    negate=True reflects (z₀ → −z₀ pre-norm). Otherwise row 0 is zeroed — the redness
    computation is deleted — and the bias becomes *bias* (0 = ex-2.9.1's zero ablation,
    a constant c = optimal ablation, −γ = redirect to the fallback direction).
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


def score_interventions(params: Params) -> dict:
    """Score zero ablation and the redirect (ex-2.9.2's recommended edit) on the full grid."""
    out = {}
    for name, bias in (("zero", 0.0), ("redirect", -GAMMA)):
        edited = edit_axis0(params, bias=bias)
        mse = np.asarray(eval_model(edited, jnp.asarray(GRID_RGB))[0])
        out[name] = {
            "score": float(np.corrcoef(SIM3, mse)[0, 1] ** 2),
            "red": float(mse[REDS].mean()),
            "collateral": float(mse[OTHERS].mean()),
            "red_pure": float(np.asarray(eval_model(edited, jnp.asarray(PURE_RED))[0])[0]),
        }
    return out


def load_results(metrics_ref: str, trajs_ref: str) -> tuple[list[dict], dict[str, np.ndarray]] | None:
    """Resolve per-run metrics and the stacked trajectories from the store, or None if unpublished."""
    store = project_store()
    arts = store.get_refs([metrics_ref, trajs_ref])
    m_art, t_art = arts[metrics_ref], arts[trajs_ref]
    if m_art is None or t_art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        m_path, t_path = store.get_many([(m_art, Path(d) / "metrics.json"), (t_art, Path(d) / "trajs.npz")])
        metrics = json.loads(m_path.read_text())
        with np.load(t_path) as z:
            trajs = dict(z)
    return metrics, trajs


def classify(r: dict) -> str:
    """Bucket a run by its endpoint health.

    Thresholds sit in the gaps of clearly bimodal metrics: healthy runs end with
    val_anchor ≤ 0.07, leak < 0.1, and val_recon ≤ 0.002; failures sit far beyond
    (anchor 0.35+, leak 0.34+, recon 0.046+).
    """
    if r["val_anchor"] > 0.3 or r["val_recon"] > 0.01 or r["leak"] > 0.3:
        return "catastrophic"
    if r["leak"] > 0.1:
        return "degraded"
    return "clean"


def plot_latent_disc(ax: Axes, z: np.ndarray, colors: np.ndarray, *, s: float = 20) -> None:
    """One latent-space panel, per the repo's figure conventions (see the figure-style skill).

    The unit-hypersphere bound as a background disc, data-colored points at
    (z₁, z₀) — the anchored axis points up — fixed domain limits, no axes.
    Titles and annotations stay with the caller.
    """
    from matplotlib.patches import Circle

    from mini.vis import light_dark

    ax.add_patch(Circle((0, 0), 1, facecolor=light_dark("#eee", "#111"), zorder=-10))
    ax.scatter(z[:, 1], z[:, 0], c=colors, s=s, edgecolors=light_dark("#00000033", "#ffffff55"), lw=0.5)
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="#0005", lw=1, zorder=10))
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_axis_off()
