"""Experiment 2.1.8: when the repulsion acts, and how much of it is left.

Ex-2.1.7's timing arm changed one number — the epoch at which the anti-subspace
term finishes annealing to its hold ratio, 50 to 90 — and improved margin,
containment, retention and grading together, by more than adding the term was
worth in the first place. M1's keyframes were inherited from a 5-dimensional
autoencoder bottleneck and mapped onto our 100 epochs by fraction of training,
so there was never a reason to expect them to suit a 64-dimensional residual
stream.

This experiment crosses the anneal endpoint with the hold ratio the anneal
descends to, at the ex-2.1.6 scoring rung on the span pull, and adds one
dose-matched arm to say whether the endpoint effect is timing or delivered
repulsion. See `report.py` for the preregistration.

The testbed, labels, measurements, anchor schedule and pull span are ex-2.1.7's,
unchanged; only the anti-subspace schedule moves.

    bin/mini run docs/m2/ex-2.1.8/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.8
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import (
    PROMPT_SPAN,
    anchor_weight as _anchor_weight,
    anti_subspace_weight as _anti_subspace_weight,
)
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from sca.training.scheduler import configure_schedule

DESIGN_ONLY = True
"""Preregistration: constants only, no DAG yet. Delete when `main(ctx)` lands."""

# --- Testbed, carried from ex-2.1.7 unchanged -------------------------------

GRID = "v216"  # the ex-2.1.3 cell: 216 colors, one word each, d64-L4
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2

CORPUS_SEED = 0
N_EXAMPLES = 100_000
HOLDOUT_FRAC = 0.2  # of distinct closed pairs
N_EVAL = 256  # cap per eval set
N_PROBE = 27  # probe lines per color: all closed partners, exhaustive

RED_RATE = 0.08
"""P(labeled) for a pure-red first operand — the ceiling of the label distribution."""

SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_END = 100.0  # and reaches the floor — the end of training
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
TRAJ_STRIDE = 50  # steps between alignment measurements during training
BLOCK, BATCH = 64, 64  # ex-2.1.3's DataConfig, unchanged

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
LABEL_P = REDNESS**8 * RED_RATE
"""P(line labeled red) for each of the 216 colors, as a function of its first operand."""

LABEL_W = LABEL_P / LABEL_P.sum()
"""The same distribution normalized: the weight $u_c$ every alignment statistic scores with."""

SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
"""M1's ablation response shape mapped to alignment; see ex-2.1.6 for the derivation."""

SCORING_LAMBDA = 0.1
"""Anchor peak for every cell — ex-2.1.6's scoring rung, held fixed throughout."""

SPAN = PROMPT_SPAN
"""The four-position prompt pull. Fixed: ex-2.1.7's op1 pull is a position
oracle this testbed happens to supply, so an operating point is only useful
without one if it was measured on a span."""

# --- The factorial ----------------------------------------------------------

ANTI_PEAK_RATIO = 2.5
"""μ/λ at epoch 0 — M1/ex-2.9.1's dopesheet, unchanged in every cell but the dose arm."""

ANTI_ANNEAL_ENDS = [50.0, 70.0, 90.0]
"""Factor A: the epoch at which μ reaches its hold ratio.

50 is M1's step 750 of 1500 mapped by fraction of training, and 90 is
ex-2.1.7's timing arm, so two of the six cells reproduce measured conditions.
100 is excluded: with no hold window left, `hold_ratio` has almost no lever
there, so that row would cross a factor against nothing.
"""

ANTI_HOLD_RATIOS = [0.03, 0.30]
"""Factor B: the level μ/λ settles at once the anneal is done.

0.03 is M1's ratio. 0.30 is three times the level at which ex-2.1.7 saw
containment give way — that report's trajectories show ᾱ sitting near control
while the repulsion outweighs the pull, and climbing once the anneal has
brought μ/λ down to about a tenth.
"""

DOSE_MATCHED_PEAK_RATIO = 1.443
"""The dose arm: `ANTI_PEAK_RATIO` scaled so endpoint 90 delivers endpoint 50's repulsion.

Solved from `anti_dose` below against the (50, 0.03) cell; `report.py` re-derives
it rather than trusting this number. Matching the other way — raising the hold
ratio at endpoint 50 — would need μ/λ ≈ 1.18, sustaining the repulsion above the
anchor weight for half of training, so the achievable match is the downward one.
"""


def _cell(end: float, hold: float) -> dict:
    return {
        "name": f"end{end:.0f}-hold{hold * 100:02.0f}",
        "lam": SCORING_LAMBDA,
        "anti_anneal_end": end,
        "anti_hold_ratio": hold,
        "anti_peak_ratio": ANTI_PEAK_RATIO,
    }


FACTORIAL = [_cell(e, h) for e in ANTI_ANNEAL_ENDS for h in ANTI_HOLD_RATIOS]

CONDITIONS: list[dict] = [
    {
        "name": "lam0",
        "lam": 0.0,
        "anti_anneal_end": ANTI_ANNEAL_ENDS[0],
        "anti_hold_ratio": ANTI_HOLD_RATIOS[0],
        "anti_peak_ratio": ANTI_PEAK_RATIO,
    },
    *FACTORIAL,
    {
        "name": "end90-hold03-dose",
        "lam": SCORING_LAMBDA,
        "anti_anneal_end": 90.0,
        "anti_hold_ratio": 0.03,
        "anti_peak_ratio": DOSE_MATCHED_PEAK_RATIO,
    },
]
"""`end50-hold03` re-runs ex-2.1.7's `span-anti` and `end90-hold03` its `span-anti-late`,
so every comparison here is between cells that went through identical code."""

FACTORIAL_NAMES = [c["name"] for c in FACTORIAL]
DOSE_ARM = "end90-hold03-dose"
DOSE_ARM_REFERENCE = "end50-hold03"
"""The cell the dose arm is matched to: same delivered repulsion, later anneal."""

PRIMARY = "end90-hold30"
"""The cell the mechanism in `report.py` predicts will contain the drift: the
latest anneal to the highest floor, so μ/λ never passes below the level at
which ex-2.1.7 lost containment."""

# --- Schedules --------------------------------------------------------------


def anchor_weight(epoch, *, peak: float = SCORING_LAMBDA) -> np.ndarray:
    """The anchor weight λ at (fractional) *epoch*. Unchanged from ex-2.1.6."""
    return _anchor_weight(
        epoch,
        peak=peak,
        warmup_epochs=SCHEDULER.warmup_epochs,
        anneal_start=ANNEAL_START,
        anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


def anti_subspace_weight(
    epoch,
    *,
    lam: float = SCORING_LAMBDA,
    anneal_end: float,
    hold_ratio: float,
    peak_ratio: float = ANTI_PEAK_RATIO,
) -> np.ndarray:
    """The anti-subspace weight μ at (fractional) *epoch*, for one cell of the grid.

    M1/ex-2.9.1's shape, with the two swept keyframes exposed: full strength
    from step 0 — before the anchor has ramped in — then a minimum-jerk anneal
    to *hold_ratio* by *anneal_end*. The end-of-training anneal is the
    anchor's, applied to both terms, so their ratio is constant from
    *anneal_end* on.

    This is the schedule the run itself uses, reached through the same library
    function, so no figure can drift from the training loop.
    """
    return _anti_subspace_weight(
        epoch,
        lam=lam,
        peak_ratio=peak_ratio,
        hold_ratio=hold_ratio,
        anneal_end=anneal_end,
        anchor_anneal_start=ANNEAL_START,
        anchor_anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


def learning_rate(epoch, *, peak: float = PEAK_LR, steps_per_epoch: int = 1000) -> np.ndarray:
    """The ex-2.1.3 LR schedule, evaluated at (fractional) *epoch*. As ex-2.1.6."""
    schedule = configure_schedule(SCHEDULER, peak, steps_per_epoch)
    return np.asarray(schedule(np.asarray(epoch, dtype=float) * steps_per_epoch))


def anti_dose(cell: dict, n: int = 40_001) -> float:
    """Repulsion delivered over training: ∫ μ(e)·lr(e) de, for one condition dict.

    The learning rate belongs in the integrand because it multiplies every
    term's gradient, and it is warming and annealing across the same window the
    anti-subspace schedule moves in — so a weight-only integral would call two
    schedules equal that deliver visibly different pushes.

    This is the covariate the endpoint factor is confounded with, and the
    quantity `DOSE_MATCHED_PEAK_RATIO` equalizes.
    """
    e = np.linspace(0.0, float(SCHEDULER.epochs), n)
    mu = anti_subspace_weight(
        e,
        lam=cell["lam"],
        anneal_end=cell["anti_anneal_end"],
        hold_ratio=cell["anti_hold_ratio"],
        peak_ratio=cell["anti_peak_ratio"],
    )
    return float(np.trapezoid(mu * learning_rate(e), e))


# --- Gates ------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: |named_holdout EM − control| within this, per condition (seed means). As ex-2.1.7."""

MARGIN_GATE = 0.5
"""H2: the layer-mean margin at op1, ex-2.1.6's threshold unchanged."""

MEAN_ALIGN_GATE = 0.1
"""H2: ceiling on ᾱ, the mean alignment over all 216 colors at op1. Ex-2.1.7's H4(a) gate.

Every condition in ex-2.1.7 missed it: `span-anti` ended at 0.33 and the timing
arm at 0.13, against 0.008 in an un-anchored control.
"""

MEAN_ALIGN_PARTIAL = 0.25
"""H2 partial: ex-2.1.7's halving level, kept so a near miss stays readable."""

GRADE_RHO_GATE = 0.8
GRADE_R2_GATE = 0.8
"""H2: the two grading tracks, ex-2.1.6's thresholds unchanged."""

H4_FLOOR = 0.2
H4_RETENTION = 0.8
"""H2: the retention gate — for every run whose running maximum of m_op1 reaches
`H4_FLOOR`, the final value is at least `H4_RETENTION`× that maximum."""

RETENTION_STAT = "min"
"""Which seed statistic 'retention' names, settled here for every later report.

Ex-2.1.7 quoted the minimum across seeds in its Findings and the mean in its
Arms section under one name. The minimum is the one H4(b) asks for — its gate
is worded per run — and a three-seed mean can hide a single sliding run, which
is the failure the gate exists to catch.
"""

ALPHA_EFFECT_GATE = 0.05
"""H3: the smallest main effect on ᾱ scored as real.

Ex-2.1.6's control put ᾱ's seed spread at about 0.02. A main effect here is a
difference of two nine-run means, so its noise is roughly 0.02·√2/√9 ≈ 0.009,
and 0.05 is about five times that. Whether the anchored conditions hold that
spread is checked from this experiment's own seeds before the effect is read.
"""

NOISE_PER_SEED = 0.056
NOISE_SEED_MEAN = 0.032
"""Margin noise floors, carried from ex-2.1.6: same architecture, same measurement."""

# --- Published references ---------------------------------------------------

EX217_REFERENCE = {
    "end50-hold03": {"m_op1": 0.46, "alpha": 0.33},
    "end90-hold03": {"m_op1": 0.60, "alpha": 0.13},
}
"""Ex-2.1.7's `span-anti` and `span-anti-late`, the two cells this grid reproduces.

Seed means, for the reproduction check in the Method section. Retention and
grading are not carried across: ex-2.1.7 published retention under both seed
statistics, and `RETENTION_STAT` settles that convention here, so those
comparisons are made in this report against its own recomputation.
"""
