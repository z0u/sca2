"""RGB distance between a model's answer and a line's answer, in grid steps, to sit beside expected exact match.

Expected exact match (`eem`, the model's color mass on the line's answer distribution) gives no partial credit: a
guess one grid step from the answer scores the same as a guess of black. The measurements here are distances on
the color grid, in grid steps (one step is the spacing of `sca.data.ops.LEVELS`, so the grid's far corner is
5√3 ≈ 8.66 steps from black). A line's target is its *raw* answer: the expectation of its answer distribution
under stochastic rounding, which is the op's unrounded value. Three central points of the model's distribution
over the grid are measured against it: the greedy guess (the most likely color), the mean, and the channel-wise
median. The floors say how far the answer distribution itself sits from the raw answer, since on an off-grid
line no single grid color can reach it.

Every function takes color mass over the 216 grid colors in palette order (`sca.data.ops.colors`), which is what
the model's answer log-probabilities give once restricted to the color tokens. Off-vocabulary mass is left out:
the distribution is renormalized over the grid, and a line with no color mass at all gets NaN.
"""

from __future__ import annotations

import numpy as np

from sca.data.ops import LEVELS, N_LEVELS, colors

STEP = LEVELS[1] - LEVELS[0]
"""The spacing of the grid levels, the unit every distance here is measured in."""

GRID = np.asarray(colors(), dtype=float) / STEP
"""The 216 grid colors in grid steps (each channel 0..5), in palette order: the row order of every color mass array."""

LEVEL_INDEX = (GRID.round().astype(int)).T
"""Each grid color's level index per channel, `(3, 216)`, for channel-wise marginals."""

CHANCE = float(np.linalg.norm(GRID[:, None] - GRID[None], axis=2).mean())
"""The mean distance between two colors drawn uniformly from the grid: what a model that ignores the line scores
on average, over all lines. About 3.0 steps."""


def _normalized(p_color: np.ndarray) -> np.ndarray:
    total = p_color.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, p_color / total, np.nan)


def raw_answer(q_idx: np.ndarray, q_p: np.ndarray) -> np.ndarray:
    """The raw answer of each line, `(N, 3)` in grid steps: the expectation of its answer distribution, given as
    color indices `q_idx` (`(N, K)`, padded with −1 beyond the support) and their probabilities `q_p`.
    """
    return np.einsum("nk,nkc->nc", q_p, GRID[np.maximum(q_idx, 0)])


def greedy(p_color: np.ndarray) -> np.ndarray:
    """The most likely grid color of each line, `(N, 3)` in grid steps."""
    return GRID[p_color.argmax(axis=1)]


def mean(p_color: np.ndarray) -> np.ndarray:
    """The mean of each line's distribution over the grid, `(N, 3)` in grid steps: off the grid in general."""
    return _normalized(p_color) @ GRID


def median(p_color: np.ndarray) -> np.ndarray:
    """The channel-wise median of each line's distribution over the grid, `(N, 3)` in grid steps: per channel,
    the lowest level at which the marginal reaches half the mass. A grid color when the channels are independent,
    a level per channel otherwise.
    """
    p = _normalized(p_color)
    out = np.empty((len(p), 3))
    for c in range(3):
        marginal = np.stack([p[:, LEVEL_INDEX[c] == k].sum(axis=1) for k in range(N_LEVELS)], axis=1)
        out[:, c] = np.where(np.isnan(marginal).any(axis=1), np.nan, (marginal.cumsum(axis=1) >= 0.5).argmax(axis=1))
    return out


def distance(point: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Euclidean distance between two `(N, 3)` arrays of colors, per line, in grid steps."""
    return np.linalg.norm(point - target, axis=1)


def expected_distance(p_color: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The mean distance from *target* over each line's distribution, `(N,)`: what a draw from the model scores on
    average. It is 0 only for a distribution concentrated on a grid color that is the target.
    """
    return (_normalized(p_color) * np.linalg.norm(GRID[None] - target[:, None], axis=2)).sum(axis=1)


def floors(q_idx: np.ndarray, q_p: np.ndarray) -> dict[str, np.ndarray]:
    """How far a perfect answer can be from a line's raw answer: `mode`, the distance of the most likely grid
    answer, and `draw`, the mean distance of a draw from the answer distribution. Both are 0 on an on-grid line.
    """
    raw = raw_answer(q_idx, q_p)
    cols = GRID[np.maximum(q_idx, 0)]  # (N, K, 3)
    d = np.linalg.norm(cols - raw[:, None], axis=2)
    mode = d[np.arange(len(q_p)), q_p.argmax(axis=1)]
    return {"mode": mode, "draw": (q_p * d).sum(axis=1)}


def chance(target: np.ndarray) -> np.ndarray:
    """The mean distance from *target* of a color drawn uniformly from the grid, `(N,)`: what guessing at random
    scores on each line. `CHANCE` is its mean over lines whose target is itself uniform on the grid.
    """
    return np.linalg.norm(GRID[None] - target[:, None], axis=2).mean(axis=1)
