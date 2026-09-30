"""The posterior over ops in the in-context grammar, and the Bayes ceiling it sets, from the op table alone.

A context is a few solved examples of one op and a query. The examples are the evidence: each shows a pair and an
answer, and the answer under stochastic rounding is a draw from up to eight colors (`answer_dist` in `sca.data.ops`),
so the likelihood of a shown answer under an op is the probability that rounding under that op gives it. Replacement op noise at a
rate ρ shows the answer of another op instead; cube noise at a rate κ shows a color drawn from the whole grid. The
posterior is what a model trained on the noisy corpus can reach at best, and the Bayes ceiling is the expected exact
match of the calibrated predictor that answers the query with the posterior-weighted answer distribution. Expected
exact match is linear in the model's distribution, so the metric itself peaks at the predictor that names the mode
(`mode_match`); the calibrated form is the ceiling of record because it is where cross-entropy training aims.

Everything here is a function of the op table and needs no corpus, no generator, and no training. The grammar
generator is being written separately; this module stays a pure computation so the method section of the pilot does not
depend on it. Once the generator lands, its noise model should match `sample_contexts` (a per-example replacement
rate, the replacing op uniform over the other ops, and cube noise uniform over the grid), or the ceiling here is not
the ceiling of the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sca.data import ops as grammar
from sca.data.ops import Op

N_COLORS = len(grammar.colors())
MAX_ANSWERS = 8
"""One color per corner of the box a raw value sits in: two levels per channel, three channels."""


@dataclass(frozen=True)
class AnswerTable:
    """The answer distribution of every op on every ordered pair, as a padded sparse table.

    `idx[o, pair, j]` is the color index of the j-th answer an op can give on a pair and `prob[o, pair, j]` its
    probability; unused slots hold color −1 and probability 0. The pairs are the 216² ordered pairs, indexed as
    `a * 216 + b` in palette order.
    """

    names: tuple[str, ...]
    idx: np.ndarray
    prob: np.ndarray

    @property
    def n_ops(self) -> int:
        return len(self.names)

    @property
    def n_pairs(self) -> int:
        return self.idx.shape[1]

    def lookup(self, pair: np.ndarray, color: np.ndarray) -> np.ndarray:
        """`P_o(color | pair)` for every op: shape `(n_ops, *pair.shape)`."""
        hit = self.idx[:, pair, :] == color[None, ..., None]
        return (self.prob[:, pair, :] * hit).sum(-1)

    def mode_match(self) -> float:
        """Expected exact match of a predictor told the op that names the mode: the mean of `mode_prob`."""
        return float(self.prob.max(-1).mean())

    def told_op(self) -> float:
        """Expected exact match of a predictor told the op that answers with the distribution of that op: the mean
        over ops and pairs of Σ_y P_o(y)². This is the ceiling with no inference to do, and it does not depend on the
        example count or the noise.
        """
        return float((self.prob**2).sum(-1).mean())

    def floor(self) -> float:
        """Expected exact match of the predictor with no evidence: the op-marginal answer distribution, uniform over
        ops, scored against the distribution of the true op, averaged over true ops and pairs.
        """
        # For each true op t and pair, Σ_j p_t(y_j) · mean_o P_o(y_j), in chunks of pairs to bound memory.
        total = 0.0
        for t in range(self.n_ops):
            for start in range(0, self.n_pairs, 4096):
                pairs = np.arange(start, min(start + 4096, self.n_pairs))
                q = self.lookup(pairs[:, None], self.idx[t, pairs]).mean(0)  # (pairs, 8)
                total += float((self.prob[t, pairs] * q).sum(-1).sum())
        return total / (self.n_ops * self.n_pairs)


def build_table(ops: tuple[Op, ...]) -> AnswerTable:
    """The answer table of every op on every ordered pair of grid colors. Takes a few seconds; memoize the caller."""
    cs = grammar.colors()
    color_index = {c: i for i, c in enumerate(cs)}
    n_pairs = len(cs) ** 2
    idx = -np.ones((len(ops), n_pairs, MAX_ANSWERS), dtype=np.int32)
    prob = np.zeros((len(ops), n_pairs, MAX_ANSWERS), dtype=np.float64)
    for o, op in enumerate(ops):
        for i, a in enumerate(cs):
            for j, b in enumerate(cs):
                pair = i * len(cs) + j
                for slot, (color, p) in enumerate(grammar.answer_dist(op, a, b).items()):
                    idx[o, pair, slot] = color_index[color]
                    prob[o, pair, slot] = p
    return AnswerTable(tuple(op.name for op in ops), idx, prob)


@dataclass(frozen=True)
class Contexts:
    """A batch of sampled contexts: the true op, the examples, and the query pair."""

    true_op: np.ndarray  # (n,)
    ex_pair: np.ndarray  # (n, k)
    ex_color: np.ndarray  # (n, k)
    query_pair: np.ndarray  # (n,)


def sample_contexts(
    table: AnswerTable, n: int, k: int, rho: float, kappa: float, rng: np.random.Generator, true_op: int | None = None
) -> Contexts:
    """*n* contexts of *k* examples under one op, with replacement op noise at *rho* and cube noise at *kappa*.

    Each example draws a pair uniformly over ordered pairs. Its answer is a draw from the distribution of the true op with
    probability 1 − ρ − κ; with probability ρ it is a draw from the distribution of another op, uniform over the other
    ops; and with probability κ it is a color uniform over the grid. The query pair is uniform and its answer is
    never shown. With *true_op* set, every context has that op; otherwise the op is uniform over the table.
    """
    ops = rng.integers(table.n_ops, size=n) if true_op is None else np.full(n, true_op)
    ex_pair = rng.integers(table.n_pairs, size=(n, k))
    u = rng.random((n, k))
    # Which op answers each example: the true op, or another op under replacement.
    shift = rng.integers(1, table.n_ops, size=(n, k))
    replaced = u < rho
    answer_op = np.where(replaced, (ops[:, None] + shift) % table.n_ops, ops[:, None])
    # A draw from the distribution of that op on the pair.
    probs = table.prob[answer_op, ex_pair]  # (n, k, 8)
    cum = probs.cumsum(-1)
    draw = rng.random((n, k, 1))
    slot = np.minimum((draw > cum).sum(-1), MAX_ANSWERS - 1)
    color = np.take_along_axis(table.idx[answer_op, ex_pair], slot[..., None], -1)[..., 0]
    # Cube noise replaces the answer with a uniform color.
    cube = (u >= rho) & (u < rho + kappa)
    color = np.where(cube, rng.integers(N_COLORS, size=(n, k)), color)
    return Contexts(ops, ex_pair, color, rng.integers(table.n_pairs, size=n))


def posterior(table: AnswerTable, ctx: Contexts, rho: float, kappa: float) -> np.ndarray:
    """The posterior over ops given the examples, under the noise model the contexts were drawn from: `(n, n_ops)`.

    The likelihood of a shown answer under op o is (1 − ρ − κ) P_o(y) + ρ · mean over the other ops of P_o'(y) + κ / 216.
    The prior is uniform. Every op stays above zero whenever ρ or κ is, which is what lets the target null weight the
    other ops by how nearly they fit on a context that fits one op alone.
    """
    p = table.lookup(ctx.ex_pair, ctx.ex_color)  # (ops, n, k)
    others = (p.sum(0, keepdims=True) - p) / (table.n_ops - 1)
    like = (1 - rho - kappa) * p + rho * others + kappa / N_COLORS
    log_post = np.log(np.maximum(like, 1e-300)).sum(-1).T  # (n, ops)
    log_post -= log_post.max(1, keepdims=True)
    post = np.exp(log_post)
    return post / post.sum(1, keepdims=True)


def prefix_posteriors(table: AnswerTable, ctx: Contexts, rho: float, kappa: float) -> np.ndarray:
    """The posterior after each example in turn: `(n, k, n_ops)`. After the j-th example it is the posterior a
    model with causal attention could hold at the tokens of example j + 1, the prefix form label variant (c) needs.
    """
    p = table.lookup(ctx.ex_pair, ctx.ex_color)  # (ops, n, k)
    others = (p.sum(0, keepdims=True) - p) / (table.n_ops - 1)
    like = (1 - rho - kappa) * p + rho * others + kappa / N_COLORS
    log_post = np.log(np.maximum(like, 1e-300)).cumsum(-1).transpose(1, 2, 0)  # (n, k, ops)
    log_post -= log_post.max(-1, keepdims=True)
    post = np.exp(log_post)
    return post / post.sum(-1, keepdims=True)


def expected_match(table: AnswerTable, ctx: Contexts, post: np.ndarray) -> np.ndarray:
    """Expected exact match on each query of the predictor that answers with the posterior-weighted answer
    distribution, scored against the distribution of the true op on the query pair: Σ_y q(y) P_t(y), with
    q = Σ_o post_o P_o(· | query).
    """
    colors = table.idx[ctx.true_op, ctx.query_pair]  # (n, 8)
    p_true = table.prob[ctx.true_op, ctx.query_pair]  # (n, 8)
    p_all = table.lookup(ctx.query_pair[:, None], colors)  # (ops, n, 8)
    q = np.einsum("no,onj->nj", post, p_all)
    return (q * p_true).sum(-1)


def predictive(table: AnswerTable, ctx: Contexts, post: np.ndarray) -> np.ndarray:
    """The Bayes predictive on each query, dense over the grid: `q(y) = Σ_o post_o P_o(y | query)`, shape
    `(n, 216)`. The distribution a calibrated model holds at the query `=`.
    """
    n = len(ctx.true_op)
    q = np.zeros((n, N_COLORS))
    for o in range(table.n_ops):
        colors, probs = table.idx[o, ctx.query_pair], table.prob[o, ctx.query_pair]  # (n, 8)
        dense = np.zeros((n, N_COLORS + 1))  # the last column catches the padding index −1
        np.put_along_axis(dense, np.where(colors < 0, N_COLORS, colors), probs, axis=1)
        q += post[:, o, None] * dense[:, :N_COLORS]
    return q


def mode_match(table: AnswerTable, ctx: Contexts, post: np.ndarray) -> np.ndarray:
    """Exact match on each query of the predictor that names the mode of the posterior-weighted answer distribution:
    P_t(argmax_y q(y)). The hard-accuracy form of `expected_match`, which is what the scratch simulation in the pivot
    scored, and the maximum of expected exact match over all predictors, since the metric is linear in the model's
    distribution. A calibrated model does not reach it; `expected_match` is the ceiling of record.
    """
    mode = predictive(table, ctx, post).argmax(1)
    return table.lookup(ctx.query_pair, mode)[ctx.true_op, np.arange(len(ctx.true_op))]


def kl(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """`KL(q ‖ p)` per row, in nats: `Σ_y q(y) log(q(y) / p(y))`, the terms where q is zero dropped.

    With q the Bayes predictive and p a model's answer distribution, its mean over contexts is the model's
    cross-entropy on the answer less the Bayes-optimal cross-entropy, since the true answers are drawn from q
    once the true op is marginalized under the posterior: the excess loss of the model over a calibrated one.
    """
    safe_q = np.where(q > 0, q, 1.0)
    return (q * (np.log(safe_q) - np.log(np.maximum(p, 1e-300)))).sum(-1)


def entropy(q: np.ndarray) -> np.ndarray:
    """`H(q)` per row, in nats: the irreducible loss on the answer, which no predictor gets under."""
    safe_q = np.where(q > 0, q, 1.0)
    return -(q * np.log(safe_q)).sum(-1)


def reference_predictors(table: AnswerTable, ctx: Contexts, rho: float, kappa: float) -> dict[str, np.ndarray]:
    """Answer distributions on the queries, `(n, 216)` each, for predictors that miss calibration in named ways,
    to read a model's KL against. `bayes` is the Bayes predictive itself.

    - `floor`: no evidence, the uniform posterior (the same predictor the floor scores).
    - `one-hidden`: the Bayes predictive with the last example ignored.
    - `mix-floor`: nine tenths of the Bayes predictive and a tenth of the floor.
    - `map`: commits to the most probable op and answers with the distribution of that op, mixed with a hundredth
      of the floor so that the divergence is finite where the committed op misses an answer the true op gives.
    """
    post = posterior(table, ctx, rho, kappa)
    q = predictive(table, ctx, post)
    n = len(ctx.true_op)
    uniform = np.full((n, table.n_ops), 1 / table.n_ops)
    floor_p = predictive(table, ctx, uniform)
    hidden = Contexts(ctx.true_op, ctx.ex_pair[:, :-1], ctx.ex_color[:, :-1], ctx.query_pair)
    one_hidden = predictive(table, hidden, posterior(table, hidden, rho, kappa))
    committed = np.zeros_like(post)
    committed[np.arange(n), post.argmax(1)] = 1.0
    map_p = 0.99 * predictive(table, ctx, committed) + 0.01 * floor_p
    return {"bayes": q, "floor": floor_p, "one-hidden": one_hidden, "mix-floor": 0.9 * q + 0.1 * floor_p, "map": map_p}


def skill_score(score: float, floor: float, ceiling: float) -> float:
    """Where a score sits between the floor (0) and the ceiling (1): (score − floor) / (ceiling − floor). The
    forecasting literature calls this a skill score. It puts conditions whose ceilings differ on one scale, for a
    table column beside the raw expected exact match; the raw score stays on the axes of the figures.
    """
    return (score - floor) / (ceiling - floor)


def ceiling(table: AnswerTable, ctx: Contexts, rho: float, kappa: float) -> np.ndarray:
    """Per-context Bayes ceiling: `expected_match` under the posterior, the calibrated form."""
    return expected_match(table, ctx, posterior(table, ctx, rho, kappa))


def floor(table: AnswerTable, ctx: Contexts) -> np.ndarray:
    """Per-context floor: `expected_match` under a uniform posterior, the same predictor with no evidence."""
    uniform = np.full((len(ctx.true_op), table.n_ops), 1 / table.n_ops)
    return expected_match(table, ctx, uniform)


def target_null(table: AnswerTable, ctx: Contexts, post: np.ndarray, removed: int) -> np.ndarray:
    """The posterior with one op removed and the rest renormalized: the answer a model should give once it no longer
    knows that op. `(n, n_ops)`, with the removed column at zero.
    """
    null = post.copy()
    null[:, removed] = 0
    return null / null.sum(1, keepdims=True)
