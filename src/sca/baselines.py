"""Reference points for grading a color guess on a coarse grid.

Exact-match accuracy and distance-to-the-true-mix both need something to be
measured against. On a small vocabulary the obvious references are too easy to
clear, so they can make a model look better than it is.

Uniform-random choice ("chance") ignores that mixes are midpoints, and so cluster
toward the center of the cube. A model that ignores the prompt entirely and always
answers one middling name beats chance by a wide margin. Exact match ignores how
few names sit near the answer: on the 27-color grid the true mix has only four to
six one-step neighbors, so a model that knows the right neighborhood but no more
still lands on the exact name about a fifth of the time.

Closure has a similar effect on operands. A closed pair either agrees in a channel
or holds both end levels, so each operand is one grid level from the mix in every
channel where the operands differ. That usually puts both operands inside the
one-step shell, which makes "the model answered with an operand" nearly free.

This module builds the stricter references. `blind_index` and `blind_stats` give
the best prompt-blind answer, fit on training results and scored on an eval set.
`shell_mask` and `neighborhood_exact_null` give the mix's one-step shell and the
exact-match rate of a guesser confined to it; `operand_shell_null` gives that
guesser's rate of returning an operand. `k_nearest_stats` covers graded distance:
it scores a guesser picking uniformly among the k names closest to the true mix.

Distances are Euclidean in unit-cube units, so they are comparable across grids.
Most functions take a precomputed (N, V) matrix from `distances`: the distance
from every vocabulary color to each example's true mix.
"""

from typing import Iterable, Sequence

import numpy as np

from sca.data.colors import N_LEVELS, Rgb

_TOL = 1e-9
"""Slack for "is this name the nearest one?", which ties must pass."""


def distances(vocab_rgb: np.ndarray, results: Iterable[Rgb]) -> np.ndarray:
    """(N, V) distance from every vocabulary color to each example's true mix.

    `vocab_rgb` is (V, 3) raw 0..15 channel values, as `grid_palette` yields them.
    """
    v = np.asarray(vocab_rgb, dtype=np.float32) / (N_LEVELS - 1)
    r = np.array(list(results), dtype=np.float32) / (N_LEVELS - 1)
    return np.linalg.norm(v[None] - r[:, None], axis=2)


def blind_index(train_dists: np.ndarray) -> int:
    """The vocabulary index a prompt-blind model would always answer.

    Fit on the training results (pass `distances(vocab, [ex.result for ex in
    named_seen])`) so the baseline stays learnable, rather than chosen with
    hindsight on the eval set.
    """
    return int(train_dists.mean(axis=0).argmin())


def blind_stats(dists: np.ndarray, index: int) -> dict[str, float]:
    """Mean distance and nearest-name rate for always answering `index`."""
    d = dists[:, index]
    return {"dist": float(d.mean()), "nearest": float((d <= dists.min(axis=1) + _TOL).mean())}


def shell_mask(levels: Sequence[int], vocab_rgb: np.ndarray, targets: Iterable[Rgb]) -> np.ndarray:
    """(N, V) bool: vocabulary colors one grid level from each target, in exactly one channel.

    `levels` is the ascending sub-grid (e.g. `(0, 8, 15)`); both `vocab_rgb` and
    `targets` are raw 0..15 values. What counts is distance in level *index*, not
    in RGB. The 27-color grid's steps are uneven (0 to 8 is 8, 8 to 15 is 7), and
    both are one step.
    """
    lv = np.asarray(levels)
    v = np.searchsorted(lv, np.asarray(vocab_rgb, dtype=int))
    t = np.searchsorted(lv, np.array(list(targets), dtype=int))
    return np.abs(v[None] - t[:, None]).sum(axis=2) == 1


def neighborhood_exact_null(shell: np.ndarray) -> float:
    """Exact-match rate for a guesser uniform over the true mix plus its one-step shell.

    An accuracy score on a coarse grid has to clear this before it shows the model
    can name the answer, rather than only locate it.
    """
    return float(np.mean(1.0 / (1.0 + shell.sum(axis=1))))


def operand_shell_null(shell: np.ndarray, vocab_rgb: np.ndarray, pairs: Iterable[tuple[Rgb, Rgb]]) -> float:
    """Rate at which a shell-uniform guesser lands on one of the prompt's operands.

    Operands outside the shell (pairs differing in more than one channel)
    contribute nothing, which makes this null conservative.
    """
    vocab = np.asarray(vocab_rgb, dtype=int)
    rates = []
    for row, (lhs, rhs) in zip(shell, pairs, strict=True):
        names = vocab[row]
        hits = sum(bool((names == np.asarray(o, dtype=int)).all(axis=1).any()) for o in (lhs, rhs))
        rates.append(hits / len(names) if len(names) else 0.0)
    return float(np.mean(rates))


def k_nearest_stats(dists: np.ndarray, k: int) -> dict[str, float]:
    """Mean distance and nearest-name rate for a guesser uniform over the k nearest names.

    k = 1 reproduces the floor. On open pairs k = 2 is the useful one: the true mix
    falls between grid points, so two names bracket it, and this is the score of a
    model that has located the answer but cannot break the tie.
    """
    picked = np.take_along_axis(dists, np.argsort(dists, axis=1)[:, :k], axis=1)
    floor = dists.min(axis=1, keepdims=True)
    return {"dist": float(picked.mean()), "nearest": float((picked <= floor + _TOL).mean())}


def self_nearest_rate(vocab_rgb: np.ndarray, decoded: np.ndarray) -> float:
    """Fraction of decoded points whose nearest vocabulary color is their own.

    A mean positional error cannot stand in for this. A Voronoi cell reaches only
    half a grid step, so a mean error well under one cell still leaves many points
    nearer to some other name.
    """
    v = np.asarray(vocab_rgb, dtype=np.float32) / (N_LEVELS - 1)
    d = np.linalg.norm(v[None] - np.asarray(decoded, dtype=np.float32)[:, None], axis=2)
    return float((d.argmin(axis=1) == np.arange(len(d))).mean())
