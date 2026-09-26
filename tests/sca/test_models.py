"""The model runs, and nGPT keeps activations/weights on the sphere."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from sca.config import ModelConfig
from sca.model import build_model
from sca.model.ngpt import NGPT


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


@pytest.mark.parametrize("n_head", [1, 3, 8])
def test_head_count_is_free_of_embedding_width(n_head: int):
    """Heads may number anything positive: the output projection maps n_head*n_head_dim back to n_embd, so nothing requires n_head to divide n_embd (or to align to 8)."""
    config = make_config(n_head=n_head, n_layer=2)
    model = build_model(config, key=jr.key(0))
    idx = jr.randint(jr.key(1), (2, 16), 0, config.vocab_size)
    assert model(idx).shape == (2, 16, 64)

    stream = model.residual_stream(idx)
    assert stream.shape == (3, 2, 16, 64)
    norms = jnp.linalg.norm(stream, axis=-1)
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

    # Without the flag there are no residual gains, so the report has none to show.
    fixed = build_model(make_config(), key=jr.key(0)).scale_report()
    assert "alpha_attn" not in fixed and "alpha_mlp" not in fixed


def test_untied_readout_starts_as_a_copy_and_trains_apart():
    """`tie_embeddings=False` gives the model a second table that begins equal to the embedding and is normalized on its own."""
    tied = build_model(make_config(), key=jr.key(0))
    untied = build_model(make_config(tie_embeddings=False), key=jr.key(0))
    assert tied.transformer.lm_head is None
    assert untied.transformer.lm_head is not None
    idx = jr.randint(jr.key(1), (2, 16), 0, 64)
    # Same key, same initial tables: the untied model computes the tied model's logits.
    np.testing.assert_allclose(untied(idx), tied(idx), rtol=0, atol=0)

    # Perturb the head alone: the embedding is unchanged and `normalize_weights` covers the head.
    rng = np.random.default_rng(0)
    head = untied.transformer.lm_head + rng.normal(size=untied.transformer.lm_head.shape)
    moved = untied.with_tables(readout=head).normalize_weights()
    np.testing.assert_allclose(moved.transformer.wte, untied.transformer.wte, rtol=0, atol=1e-6)
    norms = jnp.linalg.norm(moved.transformer.lm_head, axis=1)
    np.testing.assert_allclose(norms, jnp.ones_like(norms), rtol=0, atol=1e-5)
    assert np.abs(moved(idx) - untied(idx)).max() > 1e-3


def test_with_tables_unties_a_tied_model_for_scoring():
    """Giving a tied model a readout table of its own changes the logits and nothing before them."""
    model = build_model(make_config(), key=jr.key(0))
    idx = jr.randint(jr.key(1), (2, 16), 0, 64)
    stream, logits = model.stream_and_logits(idx)

    same = model.with_tables(readout=model.transformer.wte)
    assert same.transformer.lm_head is not None
    np.testing.assert_allclose(same(idx), logits, rtol=0, atol=0)

    edited = model.with_tables(readout=model.transformer.wte.at[:, 0].set(0.0))
    stream2, logits2 = edited.stream_and_logits(idx)
    np.testing.assert_allclose(stream2, stream, rtol=0, atol=0)
    assert np.abs(logits2 - logits).max() > 1e-4

    # The other side: editing the embedding leaves the readout as the original table.
    edited = model.with_tables(wte=model.transformer.wte.at[:, 0].set(0.0))
    assert edited.transformer.lm_head is None
    np.testing.assert_allclose(edited.transformer.readout, edited.transformer.wte, rtol=0, atol=0)


def test_untied_checkpoint_round_trips(tmp_path):
    """The config carries the tie flag, so a saved untied model loads with its head intact."""
    from sca.compute.model import load_checkpoint, save_checkpoint
    from sca.config import DataConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, TrainingConfig

    config = TrainingConfig(
        model=make_config(tie_embeddings=False),
        tokenizer=TokenizerConfig(vocabulary=[f"t{i}" for i in range(64)]),
        data=DataConfig(batch_size=8, oversample=1, train_split=0.8, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=1e-3, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=1, warmup_epochs=0, min_lr_factor=0.01),
    )
    model = build_model(config.model, key=jr.key(0))
    assert isinstance(model, NGPT)
    head = model.transformer.lm_head
    assert head is not None
    rng = np.random.default_rng(0)
    model = model.with_tables(readout=head + rng.normal(size=(64, 64))).normalize_weights()
    save_checkpoint(model, config, None, tmp_path)
    loaded, loaded_config, _ = load_checkpoint(tmp_path)
    assert loaded_config.model.tie_embeddings is False
    assert isinstance(loaded, NGPT)
    np.testing.assert_allclose(
        np.asarray(loaded.transformer.lm_head), np.asarray(model.transformer.lm_head), rtol=0, atol=0
    )
    idx = jr.randint(jr.key(1), (2, 16), 0, 64)
    np.testing.assert_allclose(loaded(idx), model(idx), rtol=0, atol=0)


def test_line_mask_stops_attention_at_the_closing_token():
    """With `line_mask_token` set, a position sees only its own line: changing an earlier line leaves a later one's logits alone, and without the mask it does not."""
    NL = 5
    idx = jnp.array([[7, 8, NL, 9, 10, 11, NL, 12, 13]])
    # Change only the first line, closing token kept.
    edited = idx.at[0, :2].set(jnp.array([20, 21]))

    masked = build_model(make_config(line_mask_token=NL), key=jr.key(0))
    plain = build_model(make_config(), key=jr.key(0))
    later = slice(3, None)
    np.testing.assert_allclose(masked(idx)[0, later], masked(edited)[0, later], rtol=0, atol=1e-6)
    assert np.abs(plain(idx)[0, later] - plain(edited)[0, later]).max() > 1e-4

    # The closing token belongs to the line it ends: it still sees that line.
    assert np.abs(masked(idx)[0, 2] - masked(edited)[0, 2]).max() > 1e-4
    # The stream and the intervention path carry the same mask.
    s, logits = masked.stream_and_logits(idx)
    np.testing.assert_allclose(s, masked.residual_stream(idx), rtol=0, atol=1e-5)
    np.testing.assert_allclose(logits, masked(idx), rtol=0, atol=1e-5)
