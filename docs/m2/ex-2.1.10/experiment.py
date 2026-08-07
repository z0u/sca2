"""Experiment 2.1.10: a label that doesn't point.

Every anchored experiment so far keyed its labels on the first operand, so the
label itself told the pull which position carries the concept — ex-2.1.9's
pooled term found op1 unaided, but only op1 ever drew the label. A document-
level label on natural text says "red occurs in here" and nothing else. So the
labeller widens by one step: a line is labeled when *either* operand draws,
with the per-slot rate halved to keep the label budget where it was, and the
pooled pull has to find the red operand line by line.

The testbed, measurements, anchor schedule, anti-subspace operating point and
mellowmax pooling are ex-2.1.9's, unchanged; only the labelling rule moves
(plus two soft-τ arms, reported but not gated).

DESIGN ONLY for now: constants and their derivations, no DAG. The run command
will be:

    bin/mini run docs/m2/ex-2.1.10/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.10
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import (
    PROMPT_SPAN,
    anchor_weight as _anchor_weight,
    anti_subspace_weight as _anti_subspace_weight,
    softmin_weights,
)
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS

DESIGN_ONLY = True

__all__ = ["softmin_weights"]  # re-exported for the report's calibration cells

# --- Testbed, carried from ex-2.1.9 unchanged --------------------------------

GRID = "v216"  # the ex-2.1.3 cell: 216 colors, one word each, d64-L4
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2

CORPUS_SEED = 0
N_EXAMPLES = 100_000
HOLDOUT_FRAC = 0.2  # of distinct closed pairs
N_EVAL = 256  # cap per eval set
N_PROBE = 27  # probe lines per color: all closed partners, exhaustive

SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_END = 100.0  # and reaches the floor — the end of training
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
TRAJ_STRIDE = 50  # steps between alignment measurements during training
BLOCK, BATCH = 64, 64  # ex-2.1.3's DataConfig, unchanged

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)

SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
"""M1's ablation response shape mapped to alignment; see ex-2.1.6 for the derivation."""

SCORING_LAMBDA = 0.1
"""Anchor peak for every cell — ex-2.1.6's scoring rung, held fixed throughout."""

SPAN = PROMPT_SPAN
"""The four span roles the pooled pull chooses among: op1, `+`, op2, `=`."""

ROLES = ["op1", "+", "op2", "="]
"""Display names for the span roles, in position order."""

# --- The operating point, from ex-2.1.8 (via ex-2.1.9) -----------------------

ANTI_PEAK_RATIO = 2.5
"""μ/λ at epoch 0 — M1/ex-2.9.1's dopesheet, unchanged."""

ANTI_ANNEAL_END = 90.0
ANTI_HOLD_RATIO = 0.30
"""Ex-2.1.8's `end90-hold30`, fixed in every anchored cell, as in ex-2.1.9."""

# --- The labelling rule -------------------------------------------------------

RED_RATE = 0.08
"""P(labeled) for a pure-red first operand under the op1-keyed labeller — the
rate every experiment since ex-2.1.6 used, kept for the reference arm."""

PER_SLOT_RATE = RED_RATE / 2
"""The either-slot labeller's per-operand rate. Each operand draws
independently at redness⁸ · PER_SLOT_RATE and the line is labeled when either
draws. Halving keeps the expected labeled-line rate at the op1 labeller's, up
to the (tiny) both-draw overlap: E[labeled] = 2·E[p] − E[p]² per line against
the reference's 2·E[p], where p is the per-slot probability. The report's
label-budget cell computes both exactly over the corpus distribution."""


def p_slot(r: np.ndarray | float, rate: float = PER_SLOT_RATE) -> np.ndarray | float:
    """P(this operand draws a label), as a function of its color's redness."""
    return r**8 * rate


def line_label_p(r1: np.ndarray | float, r2: np.ndarray | float) -> np.ndarray | float:
    """P(line labeled) under the either-slot labeller: either operand draws."""
    p1, p2 = p_slot(r1), p_slot(r2)
    return p1 + p2 - p1 * p2


LABEL_P = REDNESS**8 * RED_RATE
"""The op1-keyed labeller's per-color rate — the reference arm's labels, and
the affinity the op1-keyed continuity statistics weight with."""

LABEL_W = LABEL_P / LABEL_P.sum()
"""The same distribution normalized: the weight $u_c$ the per-color statistics
(m_span, the grading response) score with, unchanged from ex-2.1.6 on."""

# --- The pooling --------------------------------------------------------------

TAUS = [0.100, 0.500, 2.500]
"""The mellowmax temperature ladder, geometric at ×5 from ex-2.1.9's τ = 0.1.

τ is a temperature: the weight ratio between two span positions is
exp(Δx/τ), so behavior changes multiplicatively and a ×2 step would give
near-duplicate arms. 0.1 is the only ex-2.1.9 rung whose op1 response stayed
graded in every seed (latch rate 0/3), so it is the primary here; 0.5 and 2.5
walk toward the mean, and at 2.5 the weight ratios on realistic alignment
spreads are ~1.2 — a deliberate "tiny bias toward the best-aligned position"
arm, nearly the span mean that ex-2.1.9's τ = ∞ arm pinned. The calibration
cell in `report.py` recomputes the leading-weight curves from ex-2.1.9's
published arrays to place the ladder before the freeze."""

LEAD_BAND = (0.6, 0.8)
"""Ex-2.1.9's target range for the leading softmin weight, kept for the
calibration figure's reference band."""


def mellowmax_min(x: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
    """The soft minimum −τ·log(mean exp(−x/τ)) along *axis*, as in ex-2.1.9."""
    if np.isinf(tau):
        return x.mean(axis=axis)
    z = -x / tau
    m = z.max(axis=axis, keepdims=True)
    return -tau * (np.log(np.mean(np.exp(z - m), axis=axis)) + np.squeeze(m, axis=axis))


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
    anneal_end: float = ANTI_ANNEAL_END,
    hold_ratio: float = ANTI_HOLD_RATIO,
    peak_ratio: float = ANTI_PEAK_RATIO,
) -> np.ndarray:
    """The anti-subspace weight μ at (fractional) *epoch*, at the operating point."""
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


# --- Conditions ----------------------------------------------------------------


def _either(tau: float) -> dict:
    return {
        "name": f"either-t{tau * 1000:03.0f}",
        "lam": SCORING_LAMBDA,
        "tau": tau,
        "pull": "span",
        "labels": "either",
    }


CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "tau": np.inf, "pull": "span", "labels": "either"},
    {"name": "op1-labels", "lam": SCORING_LAMBDA, "tau": 0.100, "pull": "span", "labels": "op1"},
    *[_either(t) for t in TAUS],
    {"name": "slot-oracle", "lam": SCORING_LAMBDA, "tau": np.inf, "pull": "slot", "labels": "either"},
]
"""Six conditions × 3 seeds. Every anchored cell runs the anti-subspace term at
the operating point and (where the pull is pooled) the mellowmax over the
prompt span; only the labelling rule and τ vary.

`op1-labels` re-runs ex-2.1.9's `pool-t100` verbatim — op1-keyed labels at
RED_RATE, same seeds, same draws — so it is both the reproduction check for
the new labelling code path and the baseline that says what widening the
labels costs. `slot-oracle` runs the either-slot labeller but pulls only the
operand(s) that drew the label (`pull: "slot"`, no pooling needed): what
knowing which slot triggered is worth, measured through identical code — a
reference rather than a strict ceiling, for the same reason ex-2.1.9's
op1-oracle was.

Label draws are consumed identically by every condition sharing a labeller,
so conditions with the same `labels` value see identical batches and masks at
a given seed; across labellers the batches differ, and cross-labeller
comparisons carry that corpus-draw noise on top of seed noise."""

EITHER_NAMES = [_either(t)["name"] for t in TAUS]
REFERENCE_ARM = "op1-labels"
ORACLE_ARM = "slot-oracle"

PRIMARY = "either-t100"
"""P, the condition H2–H4 score, fixed before the run: the either-slot arm at
τ = 0.1, the only rung ex-2.1.9 found graded in every seed — a criterion from
the previous experiment's published result, independent of every statistic
gated here. The other two either-slot arms are the soft-τ sweep: reported
beside P, not gated."""

# --- Statistics shared with the report -------------------------------------------


def pooled_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """m_span: mean over slices of the max-over-span margin, as in ex-2.1.9.

    *alpha* is (slices, colors, positions) keyed by op1 on the probe set;
    kept for continuity with ex-2.1.9's scored statistic.
    """
    m = np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def line_margin(alpha_lines: np.ndarray, line_w: np.ndarray) -> float:
    """m_line: the per-line generalization of m_span, the selectivity this
    experiment scores.

    *alpha_lines* is (slices, lines, positions) on the probe set — one row per
    probe line rather than per color — and *line_w* is the labeller's own
    P(labeled) per line, normalized. Per slice: the line-weighted mean
    alignment at each span role minus the unweighted mean, keeping the largest
    role; then the mean over slices. Reduces to m_span when the weights key on
    op1 alone and lines are averaged over partners first. The max over roles
    is applied to every condition and the control alike, so its
    maximum-of-noise inflation cancels in the control-subtracted comparisons.
    """
    m = np.einsum("n,lnt->lt", line_w, alpha_lines) - alpha_lines.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def alpha_span(alpha: np.ndarray) -> float:
    """ᾱ_span: the layer-mean of the max-over-span mean alignment, as in ex-2.1.9."""
    return float(alpha[:, :, :SPAN].mean(axis=1).max(axis=1).mean())


def r2_sim(alpha_op1: np.ndarray) -> float:
    """Squared Pearson correlation (r²) between a per-color op1 response and
    SIM_TARGET — the grading statistic, unchanged from ex-2.1.9.
    """
    return float(np.corrcoef(alpha_op1, SIM_TARGET)[0, 1] ** 2)


def group_weights(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-line weights for the two localization groups H2 scores.

    G1 weights each probe line by P(op1 drew and op2 didn't); G2 by the
    reverse. The groups are the labeller's own one-slot-only probabilities, so
    no redness threshold has to be invented, and lines where both operands are
    red (where either answer is defensible) get next to no say in either
    group. Each returned vector is normalized to sum to 1.
    """
    p1, p2 = np.asarray(p_slot(r1)), np.asarray(p_slot(r2))
    g1 = p1 * (1.0 - p2)
    g2 = (1.0 - p1) * p2
    return g1 / g1.sum(), g2 / g2.sum()


# --- Gates -------------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: |named_holdout EM − control| within this, per condition (seed means). As ex-2.1.9."""

LEAD_GATE = 0.4
"""H2(a): the leading softmin weight that counts as concentration, per group,
at the embedding slice. Uniform over four roles is 0.25; carried from
ex-2.1.9's H4(a), which `pool-t100` passed at 0.77."""

CONTRAST_GATE = 0.2
CONTRAST_PARTIAL = 0.1
"""H2(b): required between-group contrast in op2 weight over the four
position-aware slices, mean over slices of π̄_G2(l, op2) − π̄_G1(l, op2).
A contrast in [CONTRAST_PARTIAL, CONTRAST_GATE) scores H2 as partial when (a)
also holds.

The discriminator between per-line localization and a global winner. Ex-2.1.9
found the deep-slice weights one-hot per run: if a run picks one role
globally, both groups see the same profile and the contrast is ~0; if the
pool follows the label line by line, G2's op2 weight should exceed G1's by
most of the one-hot amplitude. 0.2 sits well above the ~0 of the global
account and well below the ≳0.5 a per-line account predicts, on a statistic
bounded in [−1, 1] whose ex-2.1.9 seed spreads were a few hundredths."""

ORACLE_FRAC_GATE = 0.75
ORACLE_FRAC_PARTIAL = 0.50
"""H4: control-subtracted m_line(P) as a fraction of control-subtracted
m_line(slot-oracle).

Scale-free because the labeller itself is new, so no earlier experiment
supplies an absolute target. The global-latch failure predicts roughly the
op1-triggered share of the label mass (~half, since the slots draw
symmetrically) plus whatever the shared token embedding gives away for free,
so a pass needs to clear that: 0.75 of the oracle is comfortably above the
latch account, 0.50 is its upper edge and scores as partial only. For scale,
ex-2.1.9's pooled arms recovered 91–98% of their oracle's m_span."""

MEAN_ALIGN_GATE = 0.1
"""H3(a): ceiling on ᾱ, the mean alignment over all 216 colors at op1. Carried
from ex-2.1.8/2.1.9; it is also the contamination detector — a pull that
latches op1 globally drags the non-red op1 states of op2-triggered lines onto
the axis, and that lands here."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
RETENTION_STAT = "min"
"""H3(b): retention on the m_line trajectory — for every run whose running
maximum reaches `RETENTION_FLOOR`, the final value is at least
`RETENTION_GATE`× that maximum, quoted as the minimum across seeds (settled in
ex-2.1.8)."""

GRADE_R2_DROP = 0.10
"""H3(c): the largest grading loss tolerated, r²(op1-labels) − r²(P), on the
seed-mean op1 response against SIM_TARGET. Relative, as in ex-2.1.9; the
reference arm reproduces `pool-t100`, whose r² was 0.83."""

NOISE_PER_SEED = 0.056
NOISE_SEED_MEAN = 0.032
"""Margin noise floors, carried from ex-2.1.6: same architecture, same measurement."""

# --- Published references ---------------------------------------------------------

EX219_REFERENCE = {
    "lam0": {"m_span": 0.023, "alpha_span": 0.018, "m_op1": -0.029, "alpha_op1": -0.012},
    "span-mean": {"m_span": 0.648, "alpha_span": 0.355, "m_op1": 0.619, "alpha_op1": 0.100, "r2": 0.73},
    "pool-t100": {
        "m_span": 0.704,
        "alpha_span": 0.242,
        "m_op1": 0.704,
        "alpha_op1": 0.075,
        "r2": 0.83,
        "lead_emb": 0.77,
    },
    "op1-oracle": {"m_span": 0.715, "alpha_span": 0.147, "m_op1": 0.715, "alpha_op1": 0.063, "r2": 0.87},
}
"""Ex-2.1.9's published results, recomputed from its arrays (2026-08-07) as
means of per-run statistics — the convention results are scored under.
`pool-t100` is the reproduction target for the `op1-labels` arm; the
calibration cells in `report.py` re-derive these from the store rather than
trusting this dict."""

# --- Result refs --------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.10/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.10/arrays"
EVALS_REF = "reports/m2/ex-2.1.10/evals"
PROBE_REF = "reports/m2/ex-2.1.10/probes"
CKPT_REF = "reports/m2/ex-2.1.10/checkpoints"  # + f"/{label}"
GEOMETRY_REF = "reports/m2/ex-2.1.10/geometry"

EX219_METRICS_REF = "reports/m2/ex-2.1.9/metrics"
EX219_ARRAYS_REF = "reports/m2/ex-2.1.9/arrays"
"""Ex-2.1.9's published results, read by the calibration cells and the carried
baseline figures."""
