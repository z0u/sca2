"""Layer × landmark probe maps, cross-form transfer, and probe-subspace angles.

Ex-2.1.5's measurement kit. Earlier experiments probed hand-picked positions
(`probe_residual_stream`); here the probe scan covers every layer at every
grammar landmark (`sca.data.mixed_vocab.LANDMARKS`), producing a map per
target. Within-form R² is leave-one-out (`ridge_probe_loo`) — no split to
choose, so no seed. Alongside each map the full-data probe (weights and bias)
is kept, which is what transfers: applying one form's fitted probe unchanged to
the other form's activations gives the zero-shot cross-form R² that the
transfer ratio ρ is built from, and the fitted weight matrices' column spaces
are what the principal-angle measure compares.
"""

from typing import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from sca.compute.evaluation import _r2, _r2_cols, ridge_probe_loo
from sca.data.colors import Example
from sca.data.mixed_vocab import LANDMARKS, landmark_indices
from sca.data.tokenizer import CharTokenizer
from sca.model import LanguageModel


def collect_activations(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    examples: Sequence[Example],
    batch_size: int = 256,
) -> tuple[Float[np.ndarray, "L1 N T C"], Float[np.ndarray, "N M"]]:
    """Residual stream over teacher-forced lines, plus each line's landmark columns.

    Lines are left-padded to a common length, so landmark character positions
    shift by each line's pad offset; the returned index array accounts for it.
    """
    texts = [ex.prompt + ex.answer for ex in examples]
    seq = np.asarray(tokenizer.encode(texts))  # (N, T), left-padded
    offset = seq.shape[1] - np.array([len(t) for t in texts])
    lm = np.stack([[landmark_indices(ex)[name] for name in LANDMARKS] for ex in examples])
    lm += offset[:, None]

    stream = eqx.filter_jit(eqx.nn.inference_mode(model).residual_stream)
    acts = np.concatenate(
        [np.asarray(stream(jnp.asarray(seq[i : i + batch_size]))) for i in range(0, len(seq), batch_size)],
        axis=1,
    )  # (L+1, N, T, C)
    return acts, lm


def _fit(x: Float[np.ndarray, "N C"], y: Float[np.ndarray, "N K"], l2: float) -> tuple[np.ndarray, np.ndarray]:
    """Full-data closed-form ridge fit → (weights (C, K), bias (K,))."""
    mx, my = x.mean(0), y.mean(0)
    xc = x - mx
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(x.shape[1]), xc.T @ (y - my))
    return w, my - mx @ w


def probe_maps(
    acts: Float[np.ndarray, "L1 N T C"],
    lm: Float[np.ndarray, "N M"],
    targets: Mapping[str, Float[np.ndarray, "N K"]],
    l2: float = 1e-2,
) -> dict:
    """Fit probes at every (layer, landmark) for each target.

    Returns, keyed by target name:

    - ``r2``: (L+1, M) leave-one-out R² — the within-form map;
    - ``r2_ch``: (L+1, M, K) the same, per target channel (``r2`` is its mean
      over the last axis) — the map the heatmap collapses;
    - ``weights``: (L+1, M, C, K) and ``bias``: (L+1, M, K) — full-data fits,
      for zero-shot transfer and subspace comparison.
    """
    n_depth, n_ex = acts.shape[0], acts.shape[1]
    at_lm = acts[:, np.arange(n_ex)[:, None], lm]  # (L+1, N, M, C)
    out: dict[str, dict[str, np.ndarray]] = {name: {} for name in targets}
    for name, y in targets.items():
        r2 = np.empty((n_depth, lm.shape[1]))
        r2_ch = np.empty((n_depth, lm.shape[1], y.shape[1]))
        ws = np.empty((n_depth, lm.shape[1], acts.shape[3], y.shape[1]))
        bs = np.empty((n_depth, lm.shape[1], y.shape[1]))
        for d in range(n_depth):
            for m in range(lm.shape[1]):
                x = at_lm[d, :, m]
                r2_ch[d, m] = _r2_cols(ridge_probe_loo(x, y, l2), y)
                r2[d, m] = r2_ch[d, m].mean()
                ws[d, m], bs[d, m] = _fit(x, y, l2)
        out[name] = {"r2": r2, "r2_ch": r2_ch, "weights": ws, "bias": bs}
    return out


def transfer_maps(
    fitted: Mapping[str, Mapping[str, np.ndarray]],
    acts: Float[np.ndarray, "L1 N T C"],
    lm: Float[np.ndarray, "N M"],
    targets: Mapping[str, Float[np.ndarray, "N K"]],
) -> dict[str, np.ndarray]:
    """Zero-shot cross-form R²: probes from `probe_maps` (fit on form A),
    applied unchanged to form B's activations at the same (layer, landmark).
    """
    n_ex = acts.shape[1]
    at_lm = acts[:, np.arange(n_ex)[:, None], lm]
    out = {}
    for name, y in targets.items():
        ws, bs = fitted[name]["weights"], fitted[name]["bias"]
        r2 = np.empty(ws.shape[:2])
        for d in range(ws.shape[0]):
            for m in range(ws.shape[1]):
                r2[d, m] = _r2(at_lm[d, :, m] @ ws[d, m] + bs[d, m], y)
        out[name] = r2
    return out


def principal_angle_maps(
    wa: Float[np.ndarray, "L1 M C K"],
    wb: Float[np.ndarray, "L1 M C K"],
) -> Float[np.ndarray, "L1 M K"]:
    """Principal angles (degrees) between the two probes' weight column spaces
    at every (layer, landmark). 0° = same K directions of the stream; 90° = orthogonal.
    """
    angles = np.empty(wa.shape[:2] + (wa.shape[3],))
    for d in range(wa.shape[0]):
        for m in range(wa.shape[1]):
            qa, _ = np.linalg.qr(wa[d, m])
            qb, _ = np.linalg.qr(wb[d, m])
            s = np.linalg.svd(qa.T @ qb, compute_uv=False)
            angles[d, m] = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
    return angles


def rho(
    cross_r2: Float[np.ndarray, "L1 M"],
    within_r2: Float[np.ndarray, "L1 M"],
    floor: float = 0.5,
) -> Float[np.ndarray, "L1 M"]:
    """Transfer ratio ρ = clip(cross, 0) / within, guarded.

    Reported only where the within-form R² clears *floor* (NaN elsewhere) —
    below that the site isn't measuring geometry and a small denominator makes
    the ratio erratic. Negative cross-form R² clips to zero, so ρ ∈ [0, 1].
    """
    with np.errstate(invalid="ignore"):
        out = np.clip(cross_r2, 0.0, None) / within_r2
    out[within_r2 < floor] = np.nan
    return np.clip(out, 0.0, 1.0)
