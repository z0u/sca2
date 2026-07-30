"""Experiment 2.1.7: a repulsive term and a narrower pull for the anchored transformer.

Ex-2.1.6 found that a bare anchor term buys alignment without selectivity, and
left two candidate mechanisms confounded: no repulsive term constrained the
cube-wide drift, and three of the four pulled positions cannot see which color
sits at op1. This experiment separates them with a 2×2 factorial at the
ex-2.1.6 scoring rung — {bare, + anti-subspace} × {prompt-span pull, op1-only
pull} — plus a schedule-timing arm and a weight-ceiling arm. See `report.py`
for the preregistration.

The testbed, labels, measurements, and anchor schedule are ex-2.1.6's,
unchanged. The anti-subspace term and its schedule are M1/ex-2.9.1's, with the
weight expressed relative to the anchor peak and the keyframes mapped onto our
100 epochs by fraction of training.

This module holds the design constants only; the DAG lands after the skeleton
freezes.
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import LINE_TOKENS, PROMPT_SPAN, anchor_weight as _anchor_weight, smoothstep
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from sca.training.scheduler import configure_schedule

DESIGN_ONLY = True  # constants only; delete in the same change that adds main(ctx)

# --- Testbed, carried from ex-2.1.6 unchanged -------------------------------

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

# --- The factorial ----------------------------------------------------------

SCORING_LAMBDA = 0.1
"""Anchor peak for every factorial cell — ex-2.1.6's scoring rung, for comparability."""

CEILING_LAMBDA = 1.0
"""The weight-ceiling arm: one rung above ex-2.1.6's swept maximum of 0.3."""

SPAN_FULL = PROMPT_SPAN  # op1, `+`, op2, `=` — ex-2.1.6's pull
SPAN_OP1 = 1  # op1 alone: every pulled position sees the labeled color

ANTI_PEAK_RATIO = 2.5
"""μ/λ at epoch 0 — M1/ex-2.9.1's dopesheet: anti-subspace 0.25 against an anchor peak of 0.1."""

ANTI_HOLD_RATIO = 0.03
"""μ/λ from `ANTI_ANNEAL_END` on — M1's 0.003 against 0.1, the ratio ex-2.1.6's discussion cites."""

ANTI_ANNEAL_END = 50.0
"""Epoch at which μ reaches its hold ratio — M1's step 750 of 1500, mapped by fraction of training."""

ANTI_ANNEAL_END_LATE = 90.0
"""The timing arm: the same anneal stretched to meet the end-of-training anneal."""

CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "span": SPAN_FULL, "anti": False},
    {"name": "span-bare", "lam": SCORING_LAMBDA, "span": SPAN_FULL, "anti": False},
    {"name": "span-anti", "lam": SCORING_LAMBDA, "span": SPAN_FULL, "anti": True},
    {"name": "op1-bare", "lam": SCORING_LAMBDA, "span": SPAN_OP1, "anti": False},
    {"name": "op1-anti", "lam": SCORING_LAMBDA, "span": SPAN_OP1, "anti": True},
    {
        "name": "span-anti-late",
        "lam": SCORING_LAMBDA,
        "span": SPAN_FULL,
        "anti": True,
        "anti_anneal_end": ANTI_ANNEAL_END_LATE,
    },
    {"name": "span-anti-hi", "lam": CEILING_LAMBDA, "span": SPAN_FULL, "anti": True},
]
"""`lam0` re-runs the control and `span-bare` re-runs ex-2.1.6's λ=0.1 cell, so every
comparison in this report is between cells that went through identical code."""

FACTORIAL = ["span-bare", "span-anti", "op1-bare", "op1-anti"]
PRIMARY = "span-anti"
"""The primary scoring cell: ex-2.1.6's pull with M1's repulsive company — the
condition the ex-2.1.6 discussion named as the next experiment."""

# --- Schedules --------------------------------------------------------------


def anchor_weight(epoch, *, peak: float = SCORING_LAMBDA, anneal_start: float = ANNEAL_START) -> np.ndarray:
    """The anchor weight λ at (fractional) *epoch*: ramp, hold, anneal to a floor. As ex-2.1.6."""
    return _anchor_weight(
        epoch,
        peak=peak,
        warmup_epochs=SCHEDULER.warmup_epochs,
        anneal_start=anneal_start,
        anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


def anti_subspace_weight(epoch, *, lam: float = SCORING_LAMBDA, anneal_end: float = ANTI_ANNEAL_END) -> np.ndarray:
    """The anti-subspace weight μ at (fractional) *epoch*.

    M1/ex-2.9.1's shape: full strength from step 0 — before the anchor has
    ramped in — then a minimum-jerk anneal to the hold ratio by *anneal_end*.
    The end-of-training anneal is the anchor's, applied to both terms, so the
    ratio between them is constant from *anneal_end* to the end.
    """
    e = np.asarray(epoch, dtype=float)
    ratio = ANTI_PEAK_RATIO + (ANTI_HOLD_RATIO - ANTI_PEAK_RATIO) * smoothstep(e / anneal_end)
    end = 1.0 - (1.0 - ANNEAL_FLOOR) * smoothstep((e - ANNEAL_START) / (ANNEAL_END - ANNEAL_START))
    return lam * ratio * end


def learning_rate(epoch, *, peak: float = PEAK_LR, steps_per_epoch: int = 1000) -> np.ndarray:
    """The ex-2.1.3 LR schedule, evaluated at (fractional) *epoch*. As ex-2.1.6."""
    schedule = configure_schedule(SCHEDULER, peak, steps_per_epoch)
    return np.asarray(schedule(np.asarray(epoch, dtype=float) * steps_per_epoch))


# --- Gates ------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: |named_holdout EM − control| within this, per condition (seed means)."""

MARGIN_GATE = 0.5
"""H2(a): the layer-mean margin at op1, ex-2.1.6's threshold unchanged."""

GRADE_RHO_GATE = 0.8
GRADE_R2_GATE = 0.8
"""H2(b): the two grading tracks, ex-2.1.6's thresholds unchanged."""

CONTROL_MARGIN_GATE = 0.1
"""H2(c): |m_op1| in the control."""

MAIN_EFFECT_GATE = 0.1
"""H3: the smallest margin main effect scored as real — ~3× the noise on a main effect.

A main effect is a difference of two two-cell means, so its noise is about the
same as a single three-seed mean (`NOISE_SEED_MEAN`), and 0.1 / 0.032 ≈ 3.
"""

MEAN_ALIGN_GATE = 0.25
"""H4(a): ceiling on the mean alignment over all colors at op1 in the anti cells.

Ex-2.1.6's bare term reached 0.53 at this rung, against 0.008 in its control —
this statistic averages over 216 colors and 5 layers, so an unanchored cloud
sits near zero rather than at the 1/8 spread of a single cosine. The gate asks
the anti-subspace term to hold the drift to about half the bare value while
leaving room for a genuinely red-selective component to lift the mean a little.
"""

H4_FLOOR = 0.2
H4_RETENTION = 0.8
"""H4(b): trajectory retention, ex-2.1.6's H4 gate unchanged (see its noise analysis)."""

NOISE_PER_SEED = 0.056
NOISE_SEED_MEAN = 0.032
"""Margin noise floors carried from ex-2.1.6 — same architecture, same measurement."""

# --- Published refs ---------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.7/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.7/arrays"
EVALS_REF = "reports/m2/ex-2.1.7/evals"
PROBE_REF = "reports/m2/ex-2.1.7/probes"
CKPT_REF = "reports/m2/ex-2.1.7/checkpoints"  # + f"/{label}"
GEOMETRY_REF = "reports/m2/ex-2.1.7/geometry"  # preregistered here (post-hoc in ex-2.1.6)

# --- Statistics shared with the report --------------------------------------


def _midranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing the mean rank of their group."""
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ with midranks for ties: Pearson correlation of the rank vectors."""
    return float(np.corrcoef(_midranks(a), _midranks(b))[0, 1])


def r2_sim(alpha: np.ndarray) -> float:
    """Pearson R² between a per-color response and SIM_TARGET — M1's proportionality score."""
    return float(np.corrcoef(alpha, SIM_TARGET)[0, 1] ** 2)


LINES_PER_BATCH = BLOCK * BATCH / LINE_TOKENS
"""Lines a batch carries; batch composition is ex-2.1.6's, unchanged."""
