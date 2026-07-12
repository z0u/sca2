"""The model runs, and nGPT keeps activations/weights on the sphere."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sca.config import ModelConfig
from sca.model import build_model
from sca.model._shared import normalize


def make_config(**overrides: Any) -> ModelConfig:
    defaults: dict[str, Any] = dict(
        vocab_size=64,
        block_size=64,
        n_embd=64,
        n_head=8,
        n_head_dim=8,
        n_ff=64,
        n_layer=2,
    )
    return ModelConfig(**{**defaults, **overrides})


def test_forward_shape_and_finite():
    config = make_config()
    model = build_model(config, key=jr.key(0))
    idx = jr.randint(jr.key(1), (2, 16), 0, config.vocab_size)
    logits = model(idx)
    assert logits.shape == (2, 16, config.vocab_size)
    assert jnp.isfinite(logits).all()


def test_hidden_state_stays_on_sphere():
    """Every block must return unit-norm hidden states."""
    config = make_config()
    model = build_model(config, key=jr.key(0))
    idx = jr.randint(jr.key(1), (2, 16), 0, config.vocab_size)
    enc = model.transformer.rotary_enc
    h = normalize(model.transformer.wte[idx])
    for block in model.transformer.blocks:
        h = block(h, enc)
        norms = jnp.linalg.norm(h, axis=-1)
        np.testing.assert_allclose(norms, jnp.ones_like(norms), rtol=0, atol=1e-5)


def test_residual_step_size_is_inverse_depth():
    """The residual gate is a constant 1/n_layer, not a learned parameter."""
    model = build_model(make_config(n_layer=4), key=jr.key(0))
    assert all(block.alpha == 0.25 for block in model.transformer.blocks)


def test_normalize_weights_projects_onto_sphere():
    """After normalization, each matrix is a stack of unit vectors along its hidden axis."""
    model = build_model(make_config(), key=jr.key(0))
    # Perturb, then re-project.
    rng = np.random.default_rng(0)
    model = jax.tree.map(
        lambda p: p + rng.normal(size=p.shape) if eqx.is_inexact_array(p) else p,
        model,
    )
    model = model.normalize_weights()

    def unit(t, axis: int):
        norms = jnp.linalg.norm(t, axis=axis)
        np.testing.assert_allclose(norms, jnp.ones_like(norms), rtol=0, atol=1e-5)

    unit(model.transformer.wte, axis=1)
    for block in model.transformer.blocks:
        unit(block.attn.qkv.weight, axis=1)
        unit(block.attn.proj.weight, axis=0)
        unit(block.mlp.fc.weight, axis=1)
        unit(block.mlp.proj.weight, axis=0)


def test_learnable_alpha_trains_and_reports():
    """learnable_alpha turns the residual step into a trained gain, surfaced by scale_report."""
    config = make_config(n_layer=4, residual_alpha_exp=0.5, learnable_alpha=True)
    model = build_model(config, key=jr.key(0))
    idx = jr.randint(jr.key(1), (2, 16), 0, config.vocab_size)

    logits = model(idx)
    assert jnp.isfinite(logits).all()

    # The gains start at n_layer ** -exp and appear in the diagnostics report.
    report = model.scale_report()
    assert report["alpha_attn"] == [0.5] * 4
    assert report["alpha_mlp"] == [0.5] * 4

    # They are parameters, not constants: a loss step reaches them with nonzero gradients.
    def loss(m):
        return jnp.mean(m(idx) ** 2)

    grads = eqx.filter_grad(loss)(model)
    for block in grads.transformer.blocks:
        assert block.s_attn is not None and jnp.abs(block.s_attn.weight).max() > 0
        assert block.s_mlp is not None and jnp.abs(block.s_mlp.weight).max() > 0


def test_fixed_alpha_omits_gains_from_report():
    """Without learnable_alpha, the report has no residual gains to show."""
    report = build_model(make_config(), key=jr.key(0)).scale_report()
    assert "alpha_attn" not in report and "alpha_mlp" not in report
