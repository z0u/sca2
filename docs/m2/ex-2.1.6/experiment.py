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

GRID = "v216"  # the ex-2.1.3 cell this builds on: 216 colors, one word each
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2  # carried from ex-2.1.3, unchanged

LAMBDAS = [0.0, 0.03, 0.1, 0.3]
"""Peak anchor weight, the swept hyperparameter. λ=0 is the in-experiment control."""

SCORING_LAMBDA = 0.1
"""The rung hypotheses are reported on; H2's gates read *any* rung that also clears H1."""

RED_RATE = 0.08
"""P(labeled) for a pure-red first operand — the ceiling of the label distribution."""

SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_EPOCHS = 10.0  # how long it takes to reach the floor
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
STAR_ANNEAL_START = 50.0  # the λ=0.1* star arm: same anneal, 40 epochs earlier

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
LABEL_P = REDNESS**8 * RED_RATE
"""P(line labeled red) for each of the 216 colors, as a function of its first operand."""

LABEL_W = LABEL_P / LABEL_P.sum()
"""The same distribution normalized: each color's share of all labeled lines.

Also the sampling weight for label-balanced batching, and the weight $u_c$ that
H2's $\\Delta\\alpha_{op1}$ scores with.
"""


def anchor_weight(epoch, *, peak: float = 1.0, anneal_start: float = ANNEAL_START) -> np.ndarray:
    """The anchor weight λ at (fractional) *epoch*: ramp, hold, anneal to a floor.

    The ramp shares the LR warmup window, so the anchor arrives with the
    optimizer rather than ahead of it. The anneal is linear and stops at
    `ANNEAL_FLOOR` of peak — the M1/ex-2.9.3 lesson that protection withdrawn
    entirely lets the task loss reclaim the axis.
    """
    e = np.asarray(epoch, dtype=float)
    ramp = np.clip(e / SCHEDULER.warmup_epochs, 0.0, 1.0)
    anneal = np.clip((e - anneal_start) / ANNEAL_EPOCHS, 0.0, 1.0)
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
