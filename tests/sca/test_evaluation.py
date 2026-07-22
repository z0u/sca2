"""Completion decoding and residual-stream probes behave, even on an untrained model."""

from typing import Any

import jax.random as jr
import numpy as np

from sca.config import ModelConfig, TokenizerConfig
from sca.data import colors
from sca.data.tokenizer import CharTokenizer
from sca.model import build_model
from sca.compute.evaluation import (
    answer_calibration,
    candidate_logprobs,
    completion_accuracy,
    greedy_completions,
    probe_answer_schedule,
    probe_residual_stream,
    probe_transfer,
    ridge_probe,
    ridge_probe_loo,
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


def test_ridge_probe_loo_matches_refitting_one_row_at_a_time():
    """The rank-1 downdate is an optimization; it has to agree with the obvious loop."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 6))
    y = x @ rng.standard_normal((6, 3)) + 2.0 + rng.standard_normal((40, 3)) * 0.1
    naive = np.stack(
        [x[i] @ (fit := ridge_probe(np.delete(x, i, 0), np.delete(y, i, 0), x, y))[0] + fit[1] for i in range(len(x))]
    )
    np.testing.assert_allclose(ridge_probe_loo(x, y), naive, rtol=0, atol=1e-9)


def test_ridge_probe_loo_is_honest_about_an_unlearnable_target():
    """Held-out predictions of pure noise must not beat predicting the mean."""
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((30, 40)), rng.standard_normal((30, 2))
    pred = ridge_probe_loo(x, y)
    assert ((pred - y) ** 2).sum() > ((y.mean(0) - y) ** 2).sum()


def test_probe_residual_stream_shapes():
    model, tokenizer = make_model(n_layer=2, n_embd=16), make_tokenizer()
    probe = probe_residual_stream(model, tokenizer, examples(32), batch_size=16)
    assert set(probe["r2"]) == {"operand_rgb", "operand_redness", "result_rgb", "result_redness"}
    assert all(len(r2s) == 3 for r2s in probe["r2"].values())  # embedding + 2 blocks
    assert probe["weights"]["result_rgb"].shape == (3, 16, 3)  # (depth, n_embd, targets)
    assert probe["weights"]["result_redness"].shape == (3, 16, 1)


def test_candidate_logprobs_shape_and_consistency():
    model, tokenizer = make_model(), make_tokenizer()
    prompts = ["red + blue = ", "lime + black = "]
    candidates = ["purple", "green", "#804"]
    lp = candidate_logprobs(model, tokenizer, prompts, candidates)
    assert lp.shape == (2, 3)
    assert (lp < 0).all()  # log-probs of multi-character strings
    # Padding the batch out with another prompt must not change earlier rows.
    lp2 = candidate_logprobs(model, tokenizer, prompts + ["white + navy = "], candidates)
    np.testing.assert_allclose(lp, lp2[:2], rtol=0, atol=1e-4)


def test_answer_calibration_reports_finite_scalars():
    model, tokenizer = make_model(), make_tokenizer()
    stats = answer_calibration(model, tokenizer, examples(16))
    assert set(stats) == {"nll", "entropy", "s2"}
    assert all(np.isfinite(v) for v in stats.values())
    assert stats["nll"] > 0 and stats["entropy"] > 0


def test_probe_answer_schedule_goes_trivial_once_digits_land_in_context():
    model, tokenizer = make_model(), make_tokenizer()
    hex_exs = [ex for ex in colors.sample_corpus(300, 3, [], {"hex": 1.0}) if ex.rhs is not None][:256]
    probe = probe_answer_schedule(model, tokenizer, hex_exs, offsets=(-1, 0, 1, 2, 3), batch_size=128)
    assert probe["r2"].shape == (5, 3, 3)  # (offsets, embedding + 2 blocks, channels)
    # Once digit k is in the context (offset ≥ k + 1), even an untrained
    # embedding decodes its channel: the token *is* the channel value.
    for k, off in [(0, 1), (1, 2), (2, 3)]:
        assert probe["r2"][probe["offsets"].index(off), 0, k] > 0.95


def test_probe_transfer_scores_the_fit_ceiling_and_each_set():
    model, tokenizer = make_model(), make_tokenizer()
    open_train, open_holdout = colors.split_open_pairs(0)
    fit = colors.as_form(open_train[:64], "open", seed=1)
    evals = {"open_holdout": colors.as_form(open_holdout[:16], "open", seed=2)}
    r2 = probe_transfer(model, tokenizer, fit, evals, batch_size=64)
    assert set(r2) == {"fit", "open_holdout"}
    assert all(len(v) == 3 for v in r2.values())


def test_residual_stream_is_unit_norm_and_layered():
    model = make_model(n_layer=3)
    idx = jr.randint(jr.key(1), (2, 10), 0, 64)
    stream = model.residual_stream(idx)
    assert stream.shape == (4, 2, 10, 16)
    np.testing.assert_allclose(np.linalg.norm(stream, axis=-1), 1.0, rtol=0, atol=1e-5)
