# test_ex_2_2_16_posterior.py — the vectorized posterior of docs/m2/ex-2.2.16/posterior.py against
# sca.data.incontext's per-context posterior, on a small sample.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sca.data.incontext import posterior_over_ops, posterior_prefixes, sample_context
from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, colors

_POSTERIOR_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "m2" / "ex-2.2.16" / "posterior.py"


def _load_posterior():
    spec = importlib.util.spec_from_file_location("ex2216_posterior_test", _POSTERIOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


@pytest.fixture(scope="module")
def P():
    return _load_posterior()


OP_NAMES = (
    "mix",
    "screen",
    "multiply",
    "lighten",
    "darken",
    "difference",
    "exclusion",
    "hsvmix",
    "hue-hsv",
    "sat-hsv",
    "value-hsv",
)


@pytest.fixture(scope="module")
def table(P):
    ops = tuple(OP_BY_NAME[n] if n in OP_BY_NAME else CANDIDATE_BY_NAME[n] for n in OP_NAMES)
    return ops, P.build_table(ops)


def _to_batch(P, ops, contexts, color_index):
    n, k = len(contexts), len(contexts[0].examples)
    ex_pair = np.empty((n, k), dtype=np.int64)
    ex_color = np.empty((n, k), dtype=np.int64)
    query_pair = np.empty(n, dtype=np.int64)
    true_op = np.empty(n, dtype=np.int64)
    op_index = {o.name: i for i, o in enumerate(ops)}
    for i, c in enumerate(contexts):
        for j, ex in enumerate(c.examples):
            ex_pair[i, j] = color_index[ex.lhs] * len(color_index) + color_index[ex.rhs]
            ex_color[i, j] = color_index[ex.answer]
        query_pair[i] = color_index[c.query_lhs] * len(color_index) + color_index[c.query_rhs]
        true_op[i] = op_index[c.op]
    return P.Contexts(true_op, ex_pair, ex_color, query_pair)


def test_posterior_matches_the_per_context_reference_on_a_small_sample(P, table):
    """No cube noise: `posterior_over_ops`'s replacement-only model is exactly `posterior.py`'s at kappa=0."""
    ops, post_table = table
    color_index = {c: i for i, c in enumerate(colors())}
    rho = 0.3
    rng = np.random.default_rng(0)
    contexts = [sample_context(ops, ops[5], 3, rho, rng, cube_rate=0.0, rounding="stochastic") for _ in range(20)]

    batch = _to_batch(P, ops, contexts, color_index)
    fast = P.posterior(post_table, batch, rho, 0.0)

    for i, ctx in enumerate(contexts):
        ref = posterior_over_ops(ops, ctx.examples, rho)
        ref_vec = np.array([ref[o.name] for o in ops])
        np.testing.assert_allclose(fast[i], ref_vec, rtol=1e-6, atol=1e-9)


def test_prefix_posteriors_matches_the_per_context_reference_on_a_small_sample(P, table):
    ops, post_table = table
    color_index = {c: i for i, c in enumerate(colors())}
    rho = 0.2
    rng = np.random.default_rng(1)
    contexts = [sample_context(ops, ops[5], 3, rho, rng, cube_rate=0.0, rounding="stochastic") for _ in range(20)]

    batch = _to_batch(P, ops, contexts, color_index)
    fast = P.prefix_posteriors(post_table, batch, rho, 0.0)  # (n, k, ops)

    for i, ctx in enumerate(contexts):
        ref_prefixes = posterior_prefixes(ops, ctx.examples, rho)  # k + 1 dicts, the prior first
        for j in range(len(ctx.examples)):
            ref_vec = np.array([ref_prefixes[j + 1][o.name] for o in ops])
            np.testing.assert_allclose(fast[i, j], ref_vec, rtol=1e-6, atol=1e-9)
