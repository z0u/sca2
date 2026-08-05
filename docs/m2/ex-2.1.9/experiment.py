"""Experiment 2.1.9: a pull that finds its own position.

Ex-2.1.7's largest single effect was narrowing the pull from the four-position
prompt span to op1 alone — but that is a position oracle this synthetic
language happens to supply and natural language will not. Here we replace the
per-position mean in the anchor term with a soft minimum (mellowmax) over each
labeled line's prompt span, so the loss asks that the span align *somewhere*
rather than everywhere, and the pull is free to concentrate wherever alignment
is cheapest. The anti-subspace term at ex-2.1.8's operating point is what
should make "cheapest" coincide with "the position that carries the concept":
aligning a shared syntax token helps only the labeled lines' anchor term while
every line pays repulsion for it.

Design constants only for now (`DESIGN_ONLY`): the DAG lands once the
preregistration in `report.py` is agreed.

The testbed, labels, measurements, anchor schedule and anti-subspace schedule
are ex-2.1.8's, unchanged; only the pooling over span positions moves.
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import PROMPT_SPAN
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS

DESIGN_ONLY = True

# --- Testbed, carried from ex-2.1.8 unchanged -------------------------------

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
"""The four span roles the pooled pull chooses among: op1, `+`, op2, `=`."""

ROLES = ["op1", "+", "op2", "="]
"""Display names for the span roles, in position order."""

# --- The operating point, from ex-2.1.8 -------------------------------------

ANTI_PEAK_RATIO = 2.5
"""μ/λ at epoch 0 — M1/ex-2.9.1's dopesheet, unchanged."""

ANTI_ANNEAL_END = 90.0
ANTI_HOLD_RATIO = 0.30
"""Ex-2.1.8's `end90-hold30`: the anti-subspace anneal runs to epoch 90 and
settles at 0.30 of the anchor weight — the first condition in M2 to hold ᾱ
near the control while keeping the margin. Fixed in every anchored cell here."""

# --- The pooling ------------------------------------------------------------

TAUS = [0.010, 0.030, 0.100]
"""The mellowmax temperature grid, calibrated from ex-2.1.8's stored alignments.

The leading span position takes 0.6–0.8 of the softmin weight at τ ≈ 0.01–0.03
under both reference profiles (the un-anchored control and the span pull at the
operating point), so the grid brackets that band: 0.01 sits at its sharp edge
(winner-take-most), 0.03 in the middle, and 0.1 one step toward the mean. The
τ = ∞ arm closes the family at the span mean. The calibration figure in
`report.py` derives this from the published arrays.
"""

LEAD_BAND = (0.6, 0.8)
"""The target range for the leading position's softmin weight, per the design note."""


def softmin_weights(x: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
    """softmax(−x/τ) along *axis* — the mellowmax gradient's per-position weights.

    Non-negative and summing to 1, so the pull budget per line is conserved and
    λ_a means the same thing at every τ. τ = ∞ gives the uniform weights of the
    span mean.
    """
    if np.isinf(tau):
        return np.full_like(x, 1.0 / x.shape[axis])
    z = -x / tau
    z = z - z.max(axis=axis, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=axis, keepdims=True)


def mellowmax_min(x: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
    """The soft minimum −τ·log(mean exp(−x/τ)) along *axis*.

    The 1/n inside the log is what makes τ = ∞ the exact mean (plain
    log-sum-exp diverges instead), and τ = 0 the exact min. This is the
    mellowmax of Asadi & Littman (2017), applied to a minimum.
    """
    if np.isinf(tau):
        return x.mean(axis=axis)
    z = -x / tau
    m = z.max(axis=axis, keepdims=True)
    return -tau * (np.log(np.mean(np.exp(z - m), axis=axis)) + np.squeeze(m, axis=axis))


# --- Conditions -------------------------------------------------------------


def _pooled(tau: float) -> dict:
    return {"name": f"pool-t{tau * 1000:03.0f}", "lam": SCORING_LAMBDA, "tau": tau, "span": SPAN}


CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "tau": np.inf, "span": SPAN},
    {"name": "span-mean", "lam": SCORING_LAMBDA, "tau": np.inf, "span": SPAN},
    *[_pooled(t) for t in TAUS],
    {"name": "op1-oracle", "lam": SCORING_LAMBDA, "tau": np.inf, "span": 1},
]
"""Six conditions × 3 seeds. `span-mean` re-runs ex-2.1.8's `end90-hold30`
through the pooled code path at τ = ∞, so the H2 contrast differs only in τ
and the reproduction check validates the new path. `op1-oracle` is ex-2.1.7's
op1-only pull at the new operating point: the ceiling the oracle buys,
measured through identical code. Every anchored cell carries the anti-subspace
term at the operating point."""

POOLED_NAMES = [_pooled(t)["name"] for t in TAUS]
MEAN_ARM = "span-mean"
ORACLE_ARM = "op1-oracle"

PRIMARY = "pool-t030"
"""P, the condition H2–H4 score, fixed before the run. Chosen because its τ
sits at the middle of the 0.6–0.8 calibration band on both reference profiles
— a criterion independent of every statistic the gates read, so no gate is
scored on a condition selected by that same gate's statistic. The other two
pooled arms are the τ sweep: reported beside P, not gated."""

# --- Statistics shared with the report --------------------------------------


def pooled_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """M: the layer-mean of the max-over-span margin, for one run's alignment map.

    *alpha* is (slices, colors, positions) on the probe set; *weights* is
    LABEL_W. The max over span roles is applied to every condition and to the
    control alike, so the maximum-of-noise inflation it invites is absorbed by
    the control subtraction the hypotheses specify.
    """
    m = np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def alpha_span(alpha: np.ndarray) -> float:
    """ᾱ_span: the layer-mean of the max-over-span mean alignment over all colors.

    The containment statistic ᾱ reads at op1 only; this is the same quantity
    with the same max-over-span treatment as `pooled_margin`, so drift parked
    on the other span roles — which ᾱ at op1 cannot see — is scored too.
    """
    return float(alpha[:, :, :SPAN].mean(axis=1).max(axis=1).mean())


def _midranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing the mean rank of their group."""
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ with midranks for ties. Exploratory only here: the ρ grading
    track is dropped from the gates — see the grading note in `report.py`.
    """
    return float(np.corrcoef(_midranks(a), _midranks(b))[0, 1])


def r2_sim(alpha: np.ndarray) -> float:
    """Pearson R² between a per-color response and SIM_TARGET — M1's proportionality score."""
    return float(np.corrcoef(alpha, SIM_TARGET)[0, 1] ** 2)


# --- Gates ------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: |named_holdout EM − control| within this, per condition (seed means). As ex-2.1.8."""

POOL_GAIN_GATE = 0.15
POOL_GAIN_PARTIAL = 0.09
"""H2(a): the pooled margin gain over the span mean, M(P) − M(span-mean).

A difference of two three-seed condition means; at ex-2.1.6's seed-mean margin
noise of 0.032 that difference carries about 0.032·√2 ≈ 0.045, so the gate is
~3.3× noise and the partial ~2×. A gain in [PARTIAL, GATE) scores H2 as
partial when (b) also holds. For scale, the op1 oracle bought +0.22 at op1
over the span pull in ex-2.1.7.
"""

SYNTAX_DRIFT_GATE = 0.10
"""H2(b): the required drop in ᾱ_span from the span-mean arm to P.

Ex-2.1.8's operating point has ᾱ_span ≈ 0.37 against 0.03 in the control
(3-seed spreads ~0.01), so the gate is small against the headroom and large
against noise (a difference of three-seed means carries ~0.016 at the 0.02
control spread).
"""

MEAN_ALIGN_GATE = 0.1
"""H3(a): ceiling on ᾱ, the mean alignment over all 216 colors at op1. Carried
from ex-2.1.8, whose operating point meets it at 0.087."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
RETENTION_STAT = "min"
"""H3(b): retention on the M trajectory — for every run whose running maximum
of M reaches `RETENTION_FLOOR`, the final value is at least `RETENTION_GATE`×
that maximum. The quoted statistic is the minimum across seeds
(`RETENTION_STAT`, settled in ex-2.1.8). The values are ex-2.1.8's H4 gates
under clearer names, since retention scores H3 here."""

GRADE_R2_DROP = 0.10
"""H3(c): the largest grading loss tolerated, R²(span-mean) − R²(P).

Relative rather than absolute: ex-2.1.8 found no condition reaching the 0.8
absolute gate (the operating point sits at 0.77) and left where a realistic
bar sits to a later grid, so this experiment scores "pooling doesn't cost
response shape" instead. The ρ track is dropped entirely — ex-2.1.8's ceiling
analysis put its best reachable value alongside containment at 0.59–0.82
against a 0.8 gate.
"""

LEAD_GATE = 0.4
"""H4(a): the leading softmin weight that counts as concentration. Uniform over
four roles is 0.25, and seed spread on a label-weighted mean weight is small,
so 0.4 is well clear of both."""

READ_GAIN_GATE = 0.05
"""H4(b): mean over the position-aware slices of Σ_t w̄(l,t)·R²(l,t) − mean_t R²(l,t) —
how much better the weighted positions read op1's redness than the span
average. Zero for uniform weights or a flat readability profile, by
construction."""

NOISE_PER_SEED = 0.056
NOISE_SEED_MEAN = 0.032
"""Margin noise floors, carried from ex-2.1.6: same architecture, same measurement."""

# --- Published references ---------------------------------------------------

EX218_REFERENCE = {
    "lam0": {"m_op1": -0.03, "alpha_op1": 0.008, "m_pooled": 0.015, "alpha_span": 0.034},
    "end90-hold30": {
        "m_op1": 0.61,
        "alpha_op1": 0.087,
        "retention": 0.97,
        "r2": 0.77,
        "m_pooled": 0.64,
        "alpha_span": 0.37,
    },
}
"""Ex-2.1.8's control and operating point: seed means recomputed from that
experiment's published metrics and arrays (2026-08-05), including the two
pooled statistics defined above, which ex-2.1.8 itself did not report. The
`span-mean` arm here re-runs `end90-hold30`, so these are its reproduction
targets; the calibration cell in `report.py` re-derives them from the store
rather than trusting this dict."""

EX217_OP1_GAIN = 0.22
"""Ex-2.1.7's op1-only margin gain over the span pull at the old schedule — the
scale H2(a)'s gate is read against."""

# --- Result refs ------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.9/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.9/arrays"
EVALS_REF = "reports/m2/ex-2.1.9/evals"
PROBE_REF = "reports/m2/ex-2.1.9/probes"
CKPT_REF = "reports/m2/ex-2.1.9/checkpoints"  # + f"/{label}"
GEOMETRY_REF = "reports/m2/ex-2.1.9/geometry"

EX218_METRICS_REF = "reports/m2/ex-2.1.8/metrics"
EX218_ARRAYS_REF = "reports/m2/ex-2.1.8/arrays"
"""Ex-2.1.8's published results, read by the τ calibration cell and the
reproduction check."""
