"""The eval contract: operators edit what they say, the write is what the geometry predicts, and the scorer's clean pass is the model's own."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from sca.config import ModelConfig
from sca.intervention import (
    Subspace,
    ablate_weights,
    angle_between,
    apply,
    diff_in_means,
    gain,
    leace,
    lobe,
    probe_direction,
    projection,
    write_angle,
)
from sca.model import build_model

WIDTH = 32


def unit(rng: np.random.Generator, n: int, width: int = WIDTH) -> np.ndarray:
    h = rng.normal(size=(n, width)).astype(np.float32)
    return h / np.linalg.norm(h, axis=-1, keepdims=True)


def model_config() -> ModelConfig:
    return ModelConfig(vocab_size=64, block_size=64, n_embd=WIDTH, n_head=8, n_head_dim=8, n_ff=32, n_layer=2)


def test_projection_removes_the_axis_and_stays_on_the_sphere():
    h = unit(np.random.default_rng(0), 5)
    out = np.asarray(projection(Subspace.axis(WIDTH))(jnp.asarray(h)))
    np.testing.assert_allclose(out[:, 0], 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, rtol=0, atol=1e-6)
    # The surviving components are rescaled by the closed-form gain.
    np.testing.assert_allclose(out[:, 1:], h[:, 1:] * gain(h[:, 0])[:, None], rtol=1e-5, atol=0)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_write_angle_matches_the_measured_rotation(gamma: float):
    h = unit(np.random.default_rng(1), 7)
    out = np.asarray(projection(Subspace.axis(WIDTH), gamma=gamma)(jnp.asarray(h)))
    np.testing.assert_allclose(angle_between(h, out), write_angle(h[:, 0], gamma), rtol=0, atol=1e-5)
    if gamma == 0.0:
        np.testing.assert_allclose(out, h, rtol=0, atol=1e-6)


def test_write_angle_at_full_strength_is_arcsin_alpha():
    np.testing.assert_allclose(write_angle([0.0, 0.5, 1.0]), [0.0, np.pi / 6, np.pi / 2], rtol=0, atol=1e-7)


def test_lobe_leaves_states_below_the_threshold_alone():
    h = np.zeros((2, WIDTH), dtype=np.float32)
    h[0, :3] = [0.3, 0.0, np.sqrt(1 - 0.09)]
    h[1, :3] = [0.8, 0.0, 0.6]
    out = np.asarray(lobe(Subspace.axis(WIDTH), a=0.5, b=1.0, p=1.0)(jnp.asarray(h)))
    np.testing.assert_allclose(out[0], h[0], rtol=0, atol=1e-6)
    # At α = 0.8 with a = 0.5: h(α) = (0.8 − 0.5) / 0.5 = 0.6, so 0.6 · 0.8 = 0.48 of the axis is removed.
    expected = np.zeros(WIDTH)
    expected[:3] = [0.8 - 0.48, 0.0, 0.6]
    np.testing.assert_allclose(out[1], expected / np.linalg.norm(expected), rtol=1e-5, atol=0)


def test_apply_with_no_slices_is_the_clean_forward_pass():
    model = build_model(model_config(), key=jr.PRNGKey(0))
    tokens = np.random.default_rng(2).integers(1, 16, size=(3, 8)).astype(np.int32)
    out = apply(model, tokens, projection(Subspace.axis(WIDTH)), slices=())
    np.testing.assert_allclose(out.pre, np.asarray(model.residual_stream(jnp.asarray(tokens))), rtol=0, atol=1e-6)
    np.testing.assert_allclose(out.post, out.pre, rtol=0, atol=0)
    np.testing.assert_allclose(out.logits, np.asarray(model(jnp.asarray(tokens))), rtol=0, atol=1e-5)


def test_apply_projects_at_the_named_slices_and_positions_only():
    model = build_model(model_config(), key=jr.PRNGKey(0))
    tokens = np.random.default_rng(3).integers(1, 16, size=(2, 8)).astype(np.int32)
    positions = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # the first two of eight positions
    out = apply(model, tokens, projection(Subspace.axis(WIDTH)), slices=(1,), positions=positions)
    np.testing.assert_allclose(out.post[1, :, :2, 0], 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out.post[1, :, 2:], out.pre[1, :, 2:], rtol=0, atol=0)
    np.testing.assert_allclose(out.post[0], out.pre[0], rtol=0, atol=0)  # the embedding slice was not named
    # Slice 2 is downstream of the edit, so its state differs from the clean stream.
    clean = np.asarray(model.residual_stream(jnp.asarray(tokens)))
    assert np.abs(out.pre[2] - clean[2]).max() > 1e-4


def test_ablated_model_never_carries_the_axis():
    model = ablate_weights(build_model(model_config(), key=jr.PRNGKey(1)), Subspace.axis(WIDTH))
    tokens = np.random.default_rng(4).integers(1, 16, size=(4, 8)).astype(np.int32)
    stream = np.asarray(model.residual_stream(jnp.asarray(tokens)))
    np.testing.assert_allclose(stream[..., 0], 0.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(stream, axis=-1), 1.0, rtol=0, atol=1e-5)
    # Re-normalization ran after the zeroing: the embedding rows are unit length with the axis gone.
    wte = np.asarray(model.transformer.wte)
    np.testing.assert_allclose(wte[:, 0], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(np.linalg.norm(wte, axis=1), 1.0, rtol=0, atol=1e-6)


def test_ablation_refuses_an_oblique_subspace():
    sub = Subspace(basis=np.eye(WIDTH)[:1], dual=np.eye(WIDTH)[1:2], mean=np.zeros(WIDTH))
    with pytest.raises(AssertionError):
        ablate_weights(build_model(model_config(), key=jr.PRNGKey(0)), sub)


PLANTED = np.eye(WIDTH)[0] / np.sqrt(2) + np.eye(WIDTH)[1] / np.sqrt(2)
"""The direction the synthetic concept is written along: e₁ + e₂, normalized."""


def planted(rng: np.random.Generator, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """States with a concept z written along `PLANTED` over correlated background noise."""
    z = rng.normal(size=n)
    noise = rng.normal(size=(n, WIDTH)) @ (np.eye(WIDTH) + 0.5 * np.ones((WIDTH, WIDTH)) / WIDTH)
    return (noise + 2.0 * z[:, None] * PLANTED).astype(np.float32), z


def test_fitted_directions_recover_the_planted_one():
    x, z = planted(np.random.default_rng(5))
    assert abs(diff_in_means(x, z > 0).basis[0] @ PLANTED) > 0.95
    assert abs(probe_direction(x, z).basis[0] @ PLANTED) > 0.9
    assert abs(leace(x, z).basis[0] @ PLANTED) > 0.95


def test_leace_leaves_no_linear_trace_of_the_concept():
    x, z = planted(np.random.default_rng(6))
    sub = leace(x, z)
    assert not sub.orthogonal
    erased = np.asarray(projection(sub, renorm=False)(jnp.asarray(x)))
    # Cov(r(X), Z) = 0 in every coordinate, to float32 rounding.
    cov = (erased - erased.mean(0)).T @ (z - z.mean()) / len(z)
    np.testing.assert_allclose(cov, 0.0, rtol=0, atol=1e-4)
    # And a ridge probe refit on the erased states reads nothing.
    w = probe_direction(erased, z)
    pred = erased @ w.basis[0]
    assert abs(np.corrcoef(pred, z)[0, 1]) < 0.05
