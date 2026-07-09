"""The model runs, and nGPT keeps activations/weights on the sphere."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiment.config import ModelConfig
from experiment.model import build_model
from experiment.model._shared import normalize


def make_config(**overrides) -> ModelConfig:
    defaults = dict(
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
