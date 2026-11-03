"""Post-hoc fits from moments agree with the in-memory fits, and the slice-by-slice edit agrees with the contract's."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sca.config import ModelConfig
from sca.intervention import Subspace, diff_in_means, leace, logits_at, probe_direction, projection
from sca.model import build_model
from sca.posthoc import (
    Moments,
    SliceEdit,
    dim_fit,
    inlp_fit,
    leace_fit,
    optimize_directions,
    pre_post,
    ridge_fit,
    stream_moments,
    with_mean,
)
from sca.posthoc import logits_at as sliced_logits_at

WIDTH = 32


def data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    x = rng.normal(size=(n, WIDTH)) + 0.8 * y[:, None] * np.eye(WIDTH)[3]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    graded = np.clip(y + rng.normal(scale=0.2, size=n), 0, 1)
    return x.astype(np.float32), y, graded


def model():
    cfg = ModelConfig(vocab_size=64, block_size=32, n_embd=WIDTH, n_head=4, n_head_dim=8, n_ff=32, n_layer=2)
    return build_model(cfg, key=jr.PRNGKey(0))


def test_fits_from_moments_match_the_fits_in_memory():
    x, y, graded = data()
    m = Moments.of(x, np.stack([y, graded], 1))
    np.testing.assert_allclose(dim_fit(m, 0).basis, diff_in_means(x, y).basis, atol=1e-5)
    np.testing.assert_allclose(ridge_fit(m, 1).basis, probe_direction(x, graded).basis, atol=1e-5)
    ref = leace(x, graded)
    got = leace_fit(m, 1)
    for a, b in ((got.basis, ref.basis), (got.dual, ref.dual), (got.mean, ref.mean)):
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_moments_add_up():
    x, y, graded = data()
    z = np.stack([y, graded], 1)
    whole, a, b = Moments.of(x, z), Moments.of(x[:150], z[:150]), Moments.of(x[150:], z[150:])
    for p, q in zip(whole.astuple(), (a + b).astuple(), strict=True):
        np.testing.assert_allclose(p, q, atol=1e-9)


def test_inlp_is_orthonormal_and_starts_at_the_probe():
    x, y, _ = data()
    m = Moments.of(x, y[:, None].astype(float))
    sub = inlp_fit(m, 0, 4)
    np.testing.assert_allclose(sub.basis @ sub.basis.T, np.eye(4), atol=1e-5)
    np.testing.assert_allclose(sub.basis[0], ridge_fit(m, 0).basis[0], atol=1e-5)


def test_mean_ablation_sets_the_component_to_the_mean():
    x, y, _ = data()
    sub = with_mean(Subspace.direction(np.eye(WIDTH)[3]), x[~y].mean(0))
    edit = SliceEdit.build({0: sub}, 1, WIDTH, 1.0, np.ones(1))
    out = np.asarray(edit(jnp.asarray(x[:, None]), 0))[:, 0]
    # Before the state goes back on the sphere the component along e₃ equals the mean's, so afterwards the ratio of
    # that component to the rest is the mean's component over the rest's length.
    rest = np.linalg.norm(np.delete(x, 3, axis=1), axis=1)
    np.testing.assert_allclose(out[:, 3] / np.linalg.norm(np.delete(out, 3, axis=1), axis=1), x[~y].mean(0)[3] / rest, rtol=1e-4)  # fmt: skip


def test_the_slice_edit_matches_the_contract_when_every_slice_shares_the_subspace():
    m = model()
    tokens = np.asarray(jr.randint(jr.PRNGKey(1), (20, 12), 0, 64))
    sub = Subspace.direction(np.random.default_rng(2).normal(size=WIDTH))
    pos = np.zeros(12, np.float32)
    pos[[2, 5, 7]] = 1
    for gamma in (0.5, 1.0):
        edit = SliceEdit.build({0: sub, 1: sub, 2: sub}, 3, WIDTH, gamma, pos)
        ref = logits_at(m, tokens, projection(sub, gamma), (0, 1, 2), 9, pos)
        np.testing.assert_allclose(sliced_logits_at(m, tokens, edit, 9, batch_size=8), ref, atol=1e-4)
    identity = SliceEdit.identity(3, WIDTH, 12)
    np.testing.assert_allclose(sliced_logits_at(m, tokens, identity, 9), np.asarray(m(jnp.asarray(tokens)))[:, 9], atol=1e-4)  # fmt: skip


def test_stream_moments_match_the_states():
    m = model()
    tokens = np.asarray(jr.randint(jr.PRNGKey(3), (21, 12), 0, 64))
    z = np.random.default_rng(4).random((21, 2)).astype(np.float32)
    masks = np.zeros((2, 12), np.float32)
    masks[0] = 1
    masks[1, [4, 10]] = 1
    edit = SliceEdit.build({1: Subspace.axis(WIDTH)}, 3, WIDTH, 1.0, np.ones(12))
    got = stream_moments(m, tokens, edit, masks, z, batch_size=8)
    pre, post = pre_post(m, tokens, edit)
    np.testing.assert_allclose(post[1][:, :, 0], 0.0, atol=1e-6)
    for sl in range(3):
        for s, mask in enumerate(masks):
            sel = mask > 0
            x = pre[sl][:, sel].reshape(-1, WIDTH)
            zz = np.repeat(z, sel.sum(), axis=0)
            ref = Moments.of(x, zz)
            for p, q in zip(got[sl][s].astuple(), ref.astuple(), strict=True):
                np.testing.assert_allclose(p, q, atol=1e-3)


def test_the_search_lowers_its_loss():
    m = model()
    rng = np.random.default_rng(5)
    tokens = rng.integers(0, 64, (64, 12))
    color_ids = np.arange(10, 20)
    q_null = np.full((64, 10), 0.1, np.float32)
    w = (rng.random(64) < 0.5).astype(np.float32)
    init = np.zeros((3, WIDTH), np.float32)
    init[1] = rng.normal(size=WIDTH)
    v, losses = optimize_directions(m, tokens, q_null, w, 1 - w, color_ids, (1,), 9, 1.0, init, steps=40, batch_size=32, lr=0.05)  # fmt: skip
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    np.testing.assert_allclose(np.linalg.norm(v[1]), 1.0, atol=1e-5)
    np.testing.assert_allclose(v[[0, 2]], 0.0)
