"""Completion decoding and residual-stream probes behave, even on an untrained model."""

from typing import Any

import jax.random as jr
import numpy as np

from experiment.config import ModelConfig, TokenizerConfig
from experiment.data import colors
from experiment.data.tokenizer import CharTokenizer
from experiment.model import build_model
from experiment.compute.evaluation import (
    completion_accuracy,
    greedy_completions,
    probe_residual_stream,
    ridge_probe,
)


def make_model(**overrides: Any):
    defaults: dict[str, Any] = dict(vocab_size=64, block_size=64, n_embd=16, n_head=8, n_head_dim=8, n_ff=64, n_layer=2)
    return build_model(ModelConfig(**{**defaults, **overrides}), key=jr.key(0))


def make_tokenizer() -> CharTokenizer:
    return CharTokenizer(TokenizerConfig(vocabulary=colors.alphabet()))


def examples(n: int) -> list[colors.Example]:
    train, _ = colors.split_named_pairs(0)
    return [ex for ex in colors.sample_corpus(n, 0, train) if ex.rhs is not None]


def test_greedy_completions_are_deterministic_and_bounded():
    model, tokenizer = make_model(), make_tokenizer()
    prompts = [ex.prompt for ex in examples(8)]
    got = greedy_completions(model, tokenizer, prompts, max_new_tokens=6)
    assert got == greedy_completions(model, tokenizer, prompts, max_new_tokens=6)
    assert len(got) == len(prompts)
    assert all(len(g) <= 6 and "\n" not in g for g in got)


def test_completion_accuracy_shape():
    model, tokenizer = make_model(), make_tokenizer()
    result = completion_accuracy(model, tokenizer, examples(8), max_new_tokens=6)
    assert set(result) == {"accuracy", "n", "failures"}
    assert 0.0 <= result["accuracy"] <= 1.0  # untrained: almost surely 0
    assert result["n"] == len(examples(8))


def test_ridge_probe_recovers_a_planted_linear_map():
    rng = np.random.default_rng(0)
    x, x_test = rng.standard_normal((256, 8)), rng.standard_normal((64, 8))
    w_true = rng.standard_normal((8, 3))
    w, b, r2 = ridge_probe(x, x @ w_true + 2.0, x_test, x_test @ w_true + 2.0, l2=1e-6)
    np.testing.assert_allclose(w, w_true, rtol=0, atol=1e-3)
    np.testing.assert_allclose(b, [2.0, 2.0, 2.0], rtol=0, atol=1e-3)
    assert r2 > 0.999


def test_probe_residual_stream_shapes():
    model, tokenizer = make_model(n_layer=2, n_embd=16), make_tokenizer()
    probe = probe_residual_stream(model, tokenizer, examples(32), batch_size=16)
    assert set(probe["r2"]) == {"operand_rgb", "operand_redness", "result_rgb", "result_redness"}
    assert all(len(r2s) == 3 for r2s in probe["r2"].values())  # embedding + 2 blocks
    assert probe["weights"]["result_rgb"].shape == (3, 16, 3)  # (depth, n_embd, targets)
    assert probe["weights"]["result_redness"].shape == (3, 16, 1)


def test_residual_stream_is_unit_norm_and_layered():
    model = make_model(n_layer=3)
    idx = jr.randint(jr.key(1), (2, 10), 0, 64)
    stream = model.residual_stream(idx)
    assert stream.shape == (4, 2, 10, 16)
    np.testing.assert_allclose(np.linalg.norm(stream, axis=-1), 1.0, rtol=0, atol=1e-5)
