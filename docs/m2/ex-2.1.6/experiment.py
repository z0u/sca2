"""Experiment 2.1.6: anchoring *red* in the residual stream of a transformer.

The first anchored transformer: one regularizer term pulls the residual stream
of *red-labeled* equations toward a fixed direction, and we ask where the
concept landed and what it cost the task. See `report.py` for the
preregistration.

So far this module holds only the *design* — the label distribution, the two
schedules, and the sweep grid. The DAG lands here when the experiment is
implemented. The report imports these constants, so the figures drawn before
the run and the run itself cannot disagree about what was preregistered.
"""

from __future__ import annotations

import numpy as np

from sca.colorcube import redness
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from sca.training.scheduler import configure_schedule

DESIGN_ONLY = True
"""No DAG here yet, so the e2e loader test skips this module. Remove it with the sweep."""

GRID = "v216"  # the ex-2.1.3 cell this builds on: 216 colors, one word each
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2  # carried from ex-2.1.3, unchanged

LAMBDAS = [0.0, 0.03, 0.1, 0.3]
"""Peak anchor weight, the swept hyperparameter. λ=0 is the in-experiment control."""

SCORING_LAMBDA = 0.1
"""The rung hypotheses are reported on; H2's gates read *any* task-clean rung."""

H4_FLOOR = 0.2
"""Running-max floor below which an H4 run is reported but not scored.

A run the anchor never took hold of has a noise-level running maximum: the
margin's per-checkpoint noise is ≈0.02 (simulated: unrelated 64-d unit states,
α_c a mean of ~8 effective draws), so the maximum over ~200 checkpoints sits
near 0.05, and a ratio of noise-level margins fails the 0.8× gate about as
often as not. The floor is ~3× that maximum and well under H2(a)'s 0.5.
"""

RED_RATE = 0.08
"""P(labeled) for a pure-red first operand — the ceiling of the label distribution."""

SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_END = 100.0  # and reaches the floor — the end of training, for every condition
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
STAR_ANNEAL_START = 50.0  # the λ=0.1* star arm: starts down 40 epochs earlier, lands in the same place

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
LABEL_P = REDNESS**8 * RED_RATE
"""P(line labeled red) for each of the 216 colors, as a function of its first operand."""

LABEL_W = LABEL_P / LABEL_P.sum()
"""The same distribution normalized: each color's share of all labeled lines.

Also the weight $u_c$ that every alignment statistic scores with.
"""

BLOCK, BATCH = 64, 64  # ex-2.1.3's DataConfig, unchanged
LINE_TOKENS = 6  # `name + name = name ⏎` at word level
LINES_PER_BATCH = BLOCK * BATCH / LINE_TOKENS
"""Lines a batch carries. Samples are packed token blocks, so a batch is ~680 equations, not 64."""

LABELED_PER_BATCH = float(LINES_PER_BATCH * LABEL_P.mean())
BATCHES_WITH_A_LABEL = float(1.0 - np.exp(-LABELED_PER_BATCH))
"""Poisson approximation; the exact per-line draws are independent."""

ANCHOR_SPAN = ("op1", "+", "op2", "=")
"""Positions the anchor term pulls: the prompt span. The answer and newline are measured, not pulled."""


EFFECTIVE_COLORS = float(np.exp(-(LABEL_W[LABEL_W > 0] * np.log(LABEL_W[LABEL_W > 0])).sum()))
"""Perplexity of the label distribution — how many colors it behaves like, against 216 real ones."""


def _midranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing the mean rank of their group."""
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def step_rho(threshold: float) -> float:
    """Expected Spearman ρ of a pure step response `α = [redness > threshold]` against redness.

    The measured α is continuous (a mean of cosines), so a pure step orders the
    colors within each of its two levels arbitrarily: each α's expected rank is
    its level's midrank, while its rank variance is that of a full untied
    ranking. The expectation is over that arbitrary ordering — deterministic,
    no RNG — and Monte Carlo with within-level noise agrees to 3 decimals.
    """
    n = REDNESS.size
    rx, ry = _midranks(REDNESS), _midranks(REDNESS > threshold)
    return float(np.cov(rx, ry)[0, 1] / (np.std(rx, ddof=1) * np.sqrt(n * (n + 1) / 12.0)))


STEP_RHO_CEILING = max(step_rho(t) for t in np.unique(REDNESS)[:-1])
"""The best expected Spearman ρ any pure step response can score, over all step locations.

Below the H2(b) gate of 0.8, so a step cannot pass as a grade unless it also
carries residual within-level ordering — see the H2 partial in the report.
"""


def smoothstep(tau) -> np.ndarray:
    """Minimum-jerk interpolation from 0 to 1 over the unit interval, clamped outside it.

    The quintic 6τ⁵ − 15τ⁴ + 10τ³, which is what
    `mini.temporal.MinimumJerkTimingFunction` reduces to for a move that starts
    and ends at rest — so this is the shape M1's dopesheets used for every
    regularizer ramp and anneal, in closed form and vectorized.
    """
    t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


def anchor_weight(epoch, *, peak: float = 1.0, anneal_start: float = ANNEAL_START) -> np.ndarray:
    """The anchor weight λ at (fractional) *epoch*: ramp, hold, anneal to a floor.

    The ramp shares the LR warmup window, so the anchor arrives with the
    optimizer rather than ahead of it. Both moves are minimum-jerk, which is
    what M1's dopesheets defaulted to; linear was never compared against it, so
    this is inheritance rather than a measured choice (see the report). The
    anneal stops at `ANNEAL_FLOOR` of peak — the M1/ex-2.9.3 lesson that
    protection withdrawn entirely lets the task loss reclaim the axis.

    Every condition reaches the floor at `ANNEAL_END`, so moving *anneal_start*
    stretches the anneal rather than sliding a fixed window: protection comes
    off gradually, roughly in step with the cooling LR, instead of dropping
    away while the optimizer is still hot.
    """
    e = np.asarray(epoch, dtype=float)
    ramp = smoothstep(e / SCHEDULER.warmup_epochs)
    anneal = smoothstep((e - anneal_start) / (ANNEAL_END - anneal_start))
    return peak * ramp * (1.0 - (1.0 - ANNEAL_FLOOR) * anneal)


def learning_rate(epoch, *, peak: float = PEAK_LR, steps_per_epoch: int = 1000) -> np.ndarray:
    """The ex-2.1.3 LR schedule, evaluated at (fractional) *epoch*.

    The schedule itself is written in steps, so this is the real `optax` object
    sampled on an epoch axis rather than a re-description of it. The step count
    only sets the resolution of the warmup ramp; the curve is the same at any
    epoch length.
    """
    schedule = configure_schedule(SCHEDULER, peak, steps_per_epoch)
    return np.asarray(schedule(np.asarray(epoch, dtype=float) * steps_per_epoch))
