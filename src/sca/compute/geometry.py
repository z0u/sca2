"""Layer × landmark probe maps, cross-form transfer, and probe-subspace angles.

Ex-2.1.5's measurement kit. Earlier experiments probed hand-picked positions (`probe_residual_stream`); here the probe scan covers every layer at every grammar landmark (`sca.data.mixed_vocab.LANDMARKS`), producing a map per target. Alongside each map the full-data probe (weights and bias) is kept, which is what transfers: applying one form's fitted probe unchanged to the other form's activations gives the zero-shot cross-form R² that the transfer ratio ρ is built from, and the fitted weight matrices' column spaces are what the principal-angle measure compares.

Within-form R² comes in two flavours, because "recoverable here" and "geometrically organized here" are different questions and this task pulls them apart. Leave-one-*equation*-out (`ridge_probe_loo`) answers the first: a colour or a hex digit appears in both the fit and the held-out row, so an identity → value table is memorizable, which is what you want when asking whether a value is present at all. Leave-one-*value*-out (`strict_r2`) answers the second: every row carrying that value — as either operand or the answer — leaves the fit together, so the probe has to place an unseen value from the others. Holding out only where the value is *scored* would not do it, since the same hex digit sits in all three channel slots and a colour can be operand 1 in one line and operand 2 in another.
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

    Lines are left-padded to a common length, so landmark character positions shift by each line's pad offset; the returned index array accounts for it.
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


def strict_r2(
    x: Float[np.ndarray, "N C"],
    y_col: Float[np.ndarray, " N"],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    l2: float = 1e-2,
) -> float:
    """Out-of-sample R² over (held-out rows, scored rows) folds.

    One uncentered Gram matrix serves every fold and each is downdated by its own block — the group form of the rank-1 trick in ``ridge_probe_loo``, so this costs one fit plus a small solve per fold rather than a fit per fold, which makes it *cheaper* than the per-equation estimator it sits beside.
    """
    x64, y64 = np.asarray(x, np.float64), np.asarray(y_col, np.float64)
    n, c = x64.shape
    sum_x, sum_y = x64.sum(0), y64.sum()
    gram_xx, gram_xy = x64.T @ x64, x64.T @ y64
    pred = np.full(n, np.nan)
    for held, scored in folds:
        n_tr = n - int(held.sum())
        if n_tr < c or not scored.any():
            continue
        xh, yh = x64[held], y64[held]
        mx, my = (sum_x - xh.sum(0)) / n_tr, (sum_y - yh.sum()) / n_tr
        xx = gram_xx - xh.T @ xh - n_tr * np.outer(mx, mx)
        xy = gram_xy - xh.T @ yh - n_tr * mx * my
        pred[scored] = (x64[scored] - mx) @ np.linalg.solve(xx + l2 * np.eye(c), xy) + my
    ok = ~np.isnan(pred)
    y_ok = y64[ok]
    return float(1 - ((y_ok - pred[ok]) ** 2).sum() / ((y_ok - y_ok.mean()) ** 2).sum())


def _value_folds(
    y: Float[np.ndarray, "N K"], slots: Sequence[Float[np.ndarray, "N K"]], k: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per distinct value of channel *k*: rows holding it in any slot, rows scored for it."""
    return [
        (np.any([np.abs(s[:, k] - v) < 1e-6 for s in slots], axis=0), np.abs(y[:, k] - v) < 1e-6)
        for v in np.unique(y[:, k])
    ]


def probe_maps(
    acts: Float[np.ndarray, "L1 N T C"],
    lm: Float[np.ndarray, "N M"],
    targets: Mapping[str, Float[np.ndarray, "N K"]],
    l2: float = 1e-2,
) -> dict:
    """Fit probes at every (layer, landmark) for each target.

    Returns, keyed by target name:

    - ``r2``: (L+1, M) leave-one-equation-out R² — is the value recoverable here;
    - ``r2_ch``: (L+1, M, K) the same, per target channel (``r2`` is its mean over the last axis);
    - ``r2_strict`` / ``r2_strict_ch``: the same pair under the leave-one-value-out holdout — is the value geometrically organized here (see the module docstring);
    - ``weights``: (L+1, M, C, K) and ``bias``: (L+1, M, K) — full-data fits, for zero-shot transfer and subspace comparison.

    The strict folds are built from *all* of ``targets``, so it must carry the line's every value-bearing slot (both operands and the mix) for the holdout to be airtight.
    """
    n_depth, n_ex = acts.shape[0], acts.shape[1]
    at_lm = acts[:, np.arange(n_ex)[:, None], lm]  # (L+1, N, M, C)
    slots = list(targets.values())
    out: dict[str, dict[str, np.ndarray]] = {name: {} for name in targets}
    for name, y in targets.items():
        n_ch = y.shape[1]
        r2 = np.empty((n_depth, lm.shape[1]))
        r2_ch = np.empty((n_depth, lm.shape[1], n_ch))
        r2_strict_ch = np.empty((n_depth, lm.shape[1], n_ch))
        ws = np.empty((n_depth, lm.shape[1], acts.shape[3], n_ch))
        bs = np.empty((n_depth, lm.shape[1], n_ch))
        folds = [_value_folds(y, slots, k) for k in range(n_ch)]
        for d in range(n_depth):
            for m in range(lm.shape[1]):
                x = at_lm[d, :, m]
                r2_ch[d, m] = _r2_cols(ridge_probe_loo(x, y, l2), y)
                r2[d, m] = r2_ch[d, m].mean()
                r2_strict_ch[d, m] = [strict_r2(x, y[:, k], folds[k], l2) for k in range(n_ch)]
                ws[d, m], bs[d, m] = _fit(x, y, l2)
        out[name] = {
            "r2": r2,
            "r2_ch": r2_ch,
            "r2_strict": r2_strict_ch.mean(axis=-1),
            "r2_strict_ch": r2_strict_ch,
            "weights": ws,
            "bias": bs,
        }
    return out


def transfer_maps(
    fitted: Mapping[str, Mapping[str, np.ndarray]],
    acts: Float[np.ndarray, "L1 N T C"],
    lm: Float[np.ndarray, "N M"],
    targets: Mapping[str, Float[np.ndarray, "N K"]],
) -> dict[str, np.ndarray]:
    """Zero-shot cross-form R²: probes from `probe_maps` (fit on form A), applied unchanged to form B's activations at the same (layer, landmark)."""
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


def principal_angles(
    wa: Float[np.ndarray, "*B C K"],
    wb: Float[np.ndarray, "*B C K"],
) -> Float[np.ndarray, "*B K"]:
    """The K principal angles (degrees) between two probes' weight column spaces, ascending.

    The first is the smallest — the closest the two subspaces come — so it bounds how much they could share. 0° = a direction of the stream both decoders use; 90° = orthogonal.

    Any leading batch dimensions are carried through: `qr` and `svd` are both stacked-aware, so a (…, C, K) pair is one LAPACK call per routine rather than one per subspace pair. A Python loop over slices gives bit-identical results — the batch form is the same arithmetic, just handed over in bulk — so callers with many pairs (a null distribution, a layer × landmark map) should pass the stack.
    """
    qa, _ = np.linalg.qr(wa)
    qb, _ = np.linalg.qr(wb)
    s = np.linalg.svd(qa.mT @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def principal_angle_maps(
    wa: Float[np.ndarray, "L1 M C K"],
    wb: Float[np.ndarray, "L1 M C K"],
) -> Float[np.ndarray, "L1 M K"]:
    """`principal_angles` at every (layer, landmark) — the two probes compared in place."""
    return principal_angles(wa, wb)


def rho(
    cross_r2: Float[np.ndarray, "L1 M"],
    within_r2: Float[np.ndarray, "L1 M"],
    floor: float = 0.5,
) -> Float[np.ndarray, "L1 M"]:
    """Transfer ratio ρ = clip(cross, 0) / within, guarded.

    Reported only where the within-form R² clears *floor* (NaN elsewhere) — below that the site isn't measuring geometry and a small denominator makes the ratio erratic. Negative cross-form R² clips to zero, so ρ ∈ [0, 1].
    """
    with np.errstate(invalid="ignore"):
        out = np.clip(cross_r2, 0.0, None) / within_r2
    out[within_r2 < floor] = np.nan
    return np.clip(out, 0.0, 1.0)
