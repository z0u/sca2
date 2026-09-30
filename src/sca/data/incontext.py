r"""The in-context grammar: D2.2's pivot, an operation inferred from solved examples rather than named by a
word.

`docs/m2/d2.2/pivot.md` proposes each line as one *context*: a few solved examples of one hidden op, written
with the constant symbol ``?`` in place of the op word, then a query under the same op::

    red ? cyan = white, c999 ? c666 = c333, yellow ? red = lime

Op words never appear, so the op is a latent task the model has to infer from the examples. This module
generates that corpus and the bookkeeping an anchor needs to pull the hidden op rather than a token:
`sample_corpus` draws contexts (`sample_context` draws one); `line_role_arrays` recovers the variable-length
line and role of every token, since a context's length depends on its example count and on whether it carries
a verification line (`pivot.md#a-verification-line`); `posterior_over_ops` and the two readings built on it
(`bayes_ceiling`, `target_null`) are the posterior the pivot's method section computes from `answer_dist`;
and `whole_line_mask` is the per-token pull mask for the whole-line label, keyed on the op array `op_ids`
returns rather than on an op-word token (`pivot.md#the-proposal`: "the new keying reads the op of each context
from an array stored beside the corpus").

Every op and color function is reused from `sca.data.ops`: this module only lays out the tokens and the
label bookkeeping around them.

Token layout (a design choice; the pivot fixes the grammar but not every separator — see the module's tests
and the experiment report for how the choice was checked against the pivot's own token-count arithmetic).
Each solved example is a 6-token unit, the same shape as the current grammar's line::

    op1  ?  op2  =  answer  ,

with ``,`` closing every example but the last, whose unit closes the context instead of separating two of
them. The query is one more such unit, ``,`` replaced by ``\n`` for a completion line::

    op1  ?  op2  =  answer  \n

or, for a verification line (`pivot.md#a-verification-line`), two tokens longer: the query becomes the
*candidate* equation, followed by the verdict marker and the verdict itself::

    op1  ?  op2  =  answer  |  TRUE-or-FALSE  \n

So a context of *k* examples is ``6k + 6`` tokens as a completion line, ``6k + 8`` as a verification one:
with *k* = 3 that is 24 (the pivot's "about 20"; 64 // 24 == 2, "about two contexts and a fragment" at
``block_size=64``, which is the reading these numbers imply).
"""

from __future__ import annotations

from typing import Iterable, Literal, NamedTuple, Sequence

import numpy as np

from sca.data.colors import Rgb
from sca.data.ops import OPS, PALETTE, Op, Rounding, answer_dist, colors, draw_answer

QUERY_SYM = "?"
"""Stands for "some op" and marks the position between the operands (`pivot.md#the-proposal`)."""

EXAMPLE_SEP = ","
"""Closes a solved example that is not the context's last unit."""

VERIFY_MARK = "|"
"""Marks the end of a verification line's candidate equation, before its verdict (`pivot.md#a-verification-line`)."""

TRUE_TOKEN = "TRUE"
FALSE_TOKEN = "FALSE"

SYNTAX = ("=", "\n", QUERY_SYM, EXAMPLE_SEP, VERIFY_MARK, TRUE_TOKEN, FALSE_TOKEN)
"""Every non-color word of the grammar. Op words are gone (`pivot.md#the-proposal`), so unlike
`sca.data.ops.vocabulary` this needs no op table."""

UNIT_TOKENS = 6
"""``op1 ? op2 = answer <sep>``: one solved example, or a completion query."""

VERIFY_UNIT_TOKENS = 8
"""``op1 ? op2 = answer | verdict \\n``: a verification line's last unit, two tokens longer for the marker
and the verdict."""

Source = Literal["true", "noise", "cube"]
"""Where one shown answer came from: the hidden op ("true"), a uniformly chosen other op of the table
("noise", the graded family `pivot.md#the-posterior-over-ops` models), or a uniformly random grid color
("cube", the family the pivot keeps at a low rate and does not fold into the posterior)."""


class Example(NamedTuple):
    """One solved example in a context: an operand pair, the answer shown for it, and where that answer
    came from.
    """

    lhs: Rgb
    rhs: Rgb
    answer: Rgb
    source: Source
    drawn_op: str | None
    """The op whose rule produced `answer`: the hidden op under "true", the substitute op under "noise", and
    `None` under "cube" (no op produced it)."""


class Verification(NamedTuple):
    """A verification line's candidate equation and verdict (`pivot.md#a-verification-line`)."""

    candidate: Rgb
    verdict: bool
    source: Source
    drawn_op: str | None


class Context(NamedTuple):
    """One context: *k* solved examples of `op`, then a query under the same op — a completion query by
    default, or a verification candidate and its verdict.
    """

    op: str
    examples: tuple[Example, ...]
    query_lhs: Rgb
    query_rhs: Rgb
    query_answer: Rgb
    """The query's true answer under `op`. Always the value shown for a completion line; for a verification
    line it is the fact the verdict is checked against, and the token stream shows `verify.candidate` instead."""
    verify: Verification | None = None

    @property
    def words(self) -> list[str]:
        """The context's tokens: see the module docstring for the layout."""
        from sca.data.ops import NAMES

        out: list[str] = []
        for ex in self.examples:
            out += [NAMES[ex.lhs], QUERY_SYM, NAMES[ex.rhs], "=", NAMES[ex.answer], EXAMPLE_SEP]
        out += [NAMES[self.query_lhs], QUERY_SYM, NAMES[self.query_rhs], "="]
        if self.verify is None:
            out += [NAMES[self.query_answer], "\n"]
        else:
            v = self.verify
            out += [NAMES[v.candidate], VERIFY_MARK, TRUE_TOKEN if v.verdict else FALSE_TOKEN, "\n"]
        return out

    @property
    def n_tokens(self) -> int:
        """The context's token count: `context_length` for its own shape."""
        return context_length(len(self.examples), verify=self.verify is not None)


def context_length(k: int, verify: bool = False) -> int:
    """The token count of a context with *k* examples: `UNIT_TOKENS` each, plus the query unit
    (`VERIFY_UNIT_TOKENS` for a verification line, `UNIT_TOKENS` for a completion one).
    """
    return k * UNIT_TOKENS + (VERIFY_UNIT_TOKENS if verify else UNIT_TOKENS)


def _answer(op: Op, a: Rgb, b: Rgb, u: np.ndarray | None) -> Rgb:
    return op(a, b) if u is None else draw_answer(op, a, b, u)


def _draw_example(
    op_table: Sequence[Op], hidden_op: Op, a: Rgb, b: Rgb, rho: float, cube_rate: float, rng: np.random.Generator, u
) -> Example:
    r = rng.random()
    if r < rho:
        others = [o for o in op_table if o is not hidden_op]
        noise_op = others[int(rng.integers(len(others)))]
        return Example(a, b, _answer(noise_op, a, b, u), "noise", noise_op.name)
    if r < rho + cube_rate:
        cube = colors()
        return Example(a, b, cube[int(rng.integers(len(cube)))], "cube", None)
    return Example(a, b, _answer(hidden_op, a, b, u), "true", hidden_op.name)


def _draw_verification(
    op_table: Sequence[Op],
    hidden_op: Op,
    a: Rgb,
    b: Rgb,
    true_answer: Rgb,
    rho: float,
    cube_rate: float,
    rng: np.random.Generator,
    u,
) -> Verification:
    """TRUE half the time, the candidate is the query's true answer; otherwise FALSE, drawn from the same two
    noise families as the examples, in their relative proportion (`pivot.md#a-verification-line`: "a `FALSE`
    candidate shows the answer another op would give, or a color from the cube"). The verdict is always
    whichever of those the candidate actually agrees with — a noise or cube draw that happens to match the
    true answer is `TRUE`, not `FALSE`, whatever branch produced it.
    """
    if rng.random() < 0.5:
        return Verification(true_answer, True, "true", hidden_op.name)
    total = rho + cube_rate
    p_noise = 0.5 if total <= 0 else rho / total
    if rng.random() < p_noise:
        others = [o for o in op_table if o is not hidden_op]
        noise_op = others[int(rng.integers(len(others)))]
        candidate = _answer(noise_op, a, b, u)
        return Verification(candidate, candidate == true_answer, "noise", noise_op.name)
    cube = colors()
    candidate = cube[int(rng.integers(len(cube)))]
    return Verification(candidate, candidate == true_answer, "cube", None)


def sample_context(
    op_table: Sequence[Op],
    hidden_op: Op,
    k: int,
    rho: float,
    rng: np.random.Generator,
    cube_rate: float = 0.0,
    rounding: Rounding = "nearest",
    verify_rate: float = 0.0,
) -> Context:
    """Draw one context: *k* solved examples of *hidden_op*, then a clean query under the same op.

    Each example is clean with probability ``1 - rho - cube_rate``; with probability *rho* it is
    *replacement op noise*, the answer a uniformly chosen other op of *op_table* would give (the family
    `pivot.md#the-posterior-over-ops` computes a posterior over); with probability *cube_rate* it is *cube
    noise*, a uniformly random grid color (the family the pivot keeps at a low rate and does not fold into
    the posterior). The query's operands are drawn the same way as an example's, but its answer is always
    clean. With probability *verify_rate* the context is a verification line instead of a completion one.
    """
    cs = colors()
    idx = rng.integers(len(cs), size=(k + 1, 2))
    pairs = [(cs[i], cs[j]) for i, j in idx]
    examples = tuple(
        _draw_example(
            op_table, hidden_op, a, b, rho, cube_rate, rng, rng.random(3) if rounding == "stochastic" else None
        )
        for a, b in pairs[:k]
    )
    qa, qb = pairs[k]
    query_answer = _answer(hidden_op, qa, qb, rng.random(3) if rounding == "stochastic" else None)
    verify = None
    if rng.random() < verify_rate:
        u = rng.random(3) if rounding == "stochastic" else None
        verify = _draw_verification(op_table, hidden_op, qa, qb, query_answer, rho, cube_rate, rng, u)
    return Context(hidden_op.name, examples, qa, qb, query_answer, verify)


def sample_corpus(
    n: int,
    seed: int,
    k: int,
    rho: float,
    op_table: Sequence[Op] = OPS,
    cube_rate: float = 0.0,
    rounding: Rounding = "nearest",
    verify_rate: float = 0.0,
) -> list[Context]:
    """*n* contexts drawn i.i.d.: a uniform hidden op from *op_table*, then `sample_context`.

    One `np.random.Generator` threaded through every draw, so the corpus (and every context in it) is
    deterministic given *seed*.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        hidden_op = op_table[int(rng.integers(len(op_table)))]
        out.append(sample_context(op_table, hidden_op, k, rho, rng, cube_rate, rounding, verify_rate))
    return out


def vocabulary() -> list[str]:
    """Every word of the grammar: the syntax tokens and the 216 grid colors. Op words never appear, so unlike
    `sca.data.ops.vocabulary` this takes no op table.
    """
    return [*SYNTAX, *PALETTE]


def encode_corpus(contexts: Iterable[Context], stoi: dict[str, int]) -> np.ndarray:
    """The packed token stream of a corpus: each context's tokens, one after another."""
    return np.array([stoi[w] for ctx in contexts for w in ctx.words], dtype=np.int32)


def op_ids(contexts: Sequence[Context], op_table: Sequence[Op]) -> np.ndarray:
    """The op array beside the corpus, one entry per context: the index into *op_table* of each context's
    hidden op. This is what a labeller reads in place of the op-word token the pivot removes
    (`pivot.md#the-proposal`).
    """
    by_name = {o.name: i for i, o in enumerate(op_table)}
    return np.array([by_name[c.op] for c in contexts], dtype=np.int32)


def line_boundaries(tokens: np.ndarray, newline_id: int) -> np.ndarray:
    """The first token of every line: position 0, and the token right after every `newline_id`."""
    after = np.flatnonzero(tokens == newline_id) + 1
    return np.concatenate(([0], after[after < len(tokens)]))


def line_role_arrays(tokens: np.ndarray, newline_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The per-token (line, role) arrays the variable-length grammar needs in place of `sca.anchoring`'s fixed
    ``LINE_TOKENS`` arithmetic (`pivot.md#the-proposal`, deps (b)): *line* is which context a token belongs
    to (0-based from the start of *tokens*), and *role* is its offset from that context's first token, so
    role 0 is always a context's first operand and a context's length is the largest role in it, plus one.
    """
    starts = line_boundaries(tokens, newline_id)
    line = np.searchsorted(starts, np.arange(len(tokens)), side="right") - 1
    role = np.arange(len(tokens)) - starts[line]
    return line, role


def whole_line_mask(
    line: np.ndarray, per_line_op: np.ndarray, anchored_op_id: int, label_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """The pulled positions under the whole-line label (`pivot.md#the-label-anchors-the-concept-before-it-can-exist`):
    a context whose hidden op is *anchored_op_id* draws a label with probability *label_rate*, redrawn on
    every call as `sca.anchoring.sample_anchored_batches` redraws its labels; every token of a labelled
    context is pulled, no token of another one is. *line* is the per-token line array from `line_role_arrays`;
    *per_line_op* is `op_ids` restricted to the contexts *line* indexes (context *i* of *per_line_op* is
    wherever `line == i`), so a caller working on a crop of the packed corpus re-keys both to local context
    ids first, as the tests do.

    This is the building block a batch sampler needs for the new keying `pivot.md#the-proposal` calls for
    ("the new keying reads the op of each context from an array stored beside the corpus"); wiring it into
    `sca.anchoring.sample_anchored_batches` is left for the experiment that trains the grammar. The label
    variants of `/todo/science/label-variants-in-context-op.md` narrow this mask further (by role, by a
    prefix-posterior threshold, or by sampling the draw from the posterior) without changing its shape.
    """
    n_lines = len(per_line_op)
    drawn = (rng.random(n_lines) < label_rate) & (per_line_op == anchored_op_id)
    return drawn[line]


def posterior_over_ops(
    op_table: Sequence[Op], examples: Sequence[Example], rho: float, prior: np.ndarray | None = None
) -> dict[str, float]:
    """The posterior over which op in *op_table* is the hidden op, given the shown *examples* (not the
    query), under the replacement-noise generative model at rate *rho* (`pivot.md#the-posterior-over-ops`).

    The likelihood of one example's answer under a candidate op *c* is::

        (1 - rho) * P(answer | c)  +  rho * mean_{o in op_table, o != c} P(answer | o)

    with `P` the exact-rounding probability from `sca.data.ops.answer_dist`: the corpus's own generative
    model, where a replaced example's substitute op is drawn uniformly from the *other* ops. Cube noise is
    not part of this model — the pivot's formula names only *rho* — so a context sampled with
    ``cube_rate > 0`` is scored as if it carried none; that mismatch is the pivot's own (a labeller and a
    model alike would have no way to tell a cube answer from an unlikely one under every op).
    """
    n = len(op_table)
    post = np.full(n, 1.0 / n) if prior is None else np.asarray(prior, dtype=float).copy()
    for ex in examples:
        lik = np.array([answer_dist(o, ex.lhs, ex.rhs).get(ex.answer, 0.0) for o in op_table])
        lik_c = (1.0 - rho) * lik + rho * (lik.sum() - lik) / max(n - 1, 1)
        post = post * lik_c
        total = post.sum()
        post = post / total if total > 0 else np.full(n, 1.0 / n)
    return {o.name: float(p) for o, p in zip(op_table, post, strict=True)}


def posterior_prefixes(
    op_table: Sequence[Op], examples: Sequence[Example], rho: float, prior: np.ndarray | None = None
) -> list[dict[str, float]]:
    """The posterior after each prefix of *examples*, from none (the prior alone) to all of them: what label
    variant (c) thresholds (`/todo/science/label-variants-in-context-op.md`, "label by a thresholded prefix
    posterior").
    """
    return [posterior_over_ops(op_table, examples[:i], rho, prior) for i in range(len(examples) + 1)]


def bayes_ceiling(
    op_table: Sequence[Op],
    examples: Sequence[Example],
    query: tuple[Rgb, Rgb],
    rho: float,
    prior: np.ndarray | None = None,
) -> float:
    """The best expected accuracy any predictor can reach on this context's query: the largest probability
    mass the posterior-weighted answer distribution puts on one color (`pivot.md#the-new-grammar-control`).
    """
    post = posterior_over_ops(op_table, examples, rho, prior)
    mix: dict[Rgb, float] = {}
    for o in op_table:
        p = post[o.name]
        if p <= 0:
            continue
        for y, py in answer_dist(o, *query).items():
            mix[y] = mix.get(y, 0.0) + p * py
    return max(mix.values()) if mix else 0.0


def target_null(
    op_table: Sequence[Op],
    examples: Sequence[Example],
    query: tuple[Rgb, Rgb],
    rho: float,
    exclude: str,
    prior: np.ndarray | None = None,
) -> dict[Rgb, float]:
    """The answer distribution weighted by the posterior over ops, with *exclude* removed and the rest
    renormalized: the per-context prediction for suppression (`pivot.md#the-posterior-over-ops`).
    """
    post = posterior_over_ops(op_table, examples, rho, prior)
    z = sum(p for name, p in post.items() if name != exclude)
    if z <= 0:
        return {}
    out: dict[Rgb, float] = {}
    for o in op_table:
        if o.name == exclude:
            continue
        w = post[o.name] / z
        for y, py in answer_dist(o, *query).items():
            out[y] = out.get(y, 0.0) + w * py
    return out
