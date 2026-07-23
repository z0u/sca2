"""Probe maps, transfer, and principal angles recover planted structure."""

import numpy as np

from sca.compute.geometry import _fit, principal_angle_maps, probe_maps, rho, transfer_maps
from sca.data.mixed_vocab import LANDMARKS

RNG = np.random.default_rng(0)


def planted_acts(n: int, c: int, y: np.ndarray, w: np.ndarray, noise: float) -> np.ndarray:
    """(1, N, M, C)-shaped activations where every landmark linearly encodes y."""
    base = y @ w.T + noise * RNG.standard_normal((n, c))
    return np.repeat(base[None, :, None, :], len(LANDMARKS), axis=2)  # (L1=1, N, T=M, C)


def test_probe_maps_and_transfer_recover_a_shared_code():
    n, c = 200, 16
    y = RNG.standard_normal((n, 3))
    w = RNG.standard_normal((c, 3))
    acts_a = planted_acts(n, c, y, w, noise=0.01)
    acts_b = planted_acts(n, c, y, w, noise=0.01)  # same code → transfer ≈ 1
    lm = np.tile(np.arange(len(LANDMARKS)), (n, 1))

    fitted = probe_maps(acts_a, lm, {"mix": y})
    assert fitted["mix"]["r2"].shape == (1, len(LANDMARKS))
    assert fitted["mix"]["r2"].min() > 0.95

    cross = transfer_maps(fitted, acts_b, lm, {"mix": y})
    ratio = rho(cross["mix"], fitted["mix"]["r2"])
    assert np.nanmin(ratio) > 0.95

    # A different code at the same accuracy → transfer collapses, ρ ≈ 0.
    w2 = RNG.standard_normal((c, 3))
    acts_c = planted_acts(n, c, y[RNG.permutation(n)], w2, noise=0.01)
    cross_c = transfer_maps(fitted, acts_c, lm, {"mix": y})
    assert np.nanmax(rho(cross_c["mix"], fitted["mix"]["r2"])) < 0.2


def test_rho_guards():
    within = np.array([[0.9, 0.4], [0.6, 0.8]])
    cross = np.array([[-0.5, 0.3], [0.3, 1.2]])
    r = rho(cross, within)
    assert r[0, 0] == 0.0  # negative transfer clips to zero
    assert np.isnan(r[0, 1])  # weak within-form site is not reported
    assert r[1, 1] == 1.0  # ratio caps at 1


def test_principal_angles_identity_and_orthogonal():
    w = np.linalg.qr(RNG.standard_normal((16, 3)))[0]
    same = principal_angle_maps(w[None, None], w[None, None])
    # arccos loses precision near 1, so "identical" still reads as ~1e-4 degrees.
    assert np.allclose(same, 0.0, atol=1e-3)
    # A subspace built from the orthogonal complement is at 90° everywhere.
    full = np.linalg.qr(RNG.standard_normal((16, 16)))[0]
    wa, wb = full[:, :3], full[:, 3:6]
    ortho = principal_angle_maps(wa[None, None], wb[None, None])
    assert np.allclose(ortho, 90.0, atol=1e-3)


def test_fit_matches_loo_on_clean_data():
    x = RNG.standard_normal((100, 8))
    w_true = RNG.standard_normal((8, 2))
    y = x @ w_true
    w, b = _fit(x, y, l2=1e-6)
    assert np.allclose(x @ w + b, y, atol=1e-4)
