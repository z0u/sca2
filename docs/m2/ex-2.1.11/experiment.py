"""Experiment 2.1.11: ablations and a survey for the D2.1 operating point.

D2.1's recipe was assembled piecewise: the schedules came from M1 by inheritance,
the anchor weight is ex-2.1.6's scoring rung, τ and the anti-subspace operating
point were each chosen by a short ladder in whatever experiment introduced them.
Nothing has asked whether the parts are needed, or where the good region is. This
experiment closes D2.1 by doing both, in stages, cheapest first:

1. **Calibration** (free, report-side): noise floors from ex-2.1.10's nine-seed
   primary, the objective trade structure across stored operating points, the
   τ ↔ leading-weight map, and the schedule dose integrals.
2. **Ablations** (preregistered conditions): flat arms bracketing each schedule,
   a half-length condition, and a linear-shape arm. Each "no difference" deletes
   dimensions from the search and simplifies the recipe the survey runs on.
3. **Survey** (Sobol): the surviving continuous knobs — λ_a, τ, and the
   anti-subspace dose — sampled in one round at a single seed, with a promoted
   second round at more seeds.

This is a survey in the sense of the `science` skill: it preregisters a search
plan rather than hypotheses, scores nothing, and proposes an operating point
that the next preregistered experiment must confirm at fresh seeds before any
number here is quoted as a result.

    bin/mini run docs/m2/ex-2.1.11/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.11
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

DESIGN_ONLY = True  # constants only; delete in the change that adds the DAG

# --- Testbed, carried from ex-2.1.10 unchanged --------------------------------

GRID = "v216"
PEAK_LR = 1e-2
CORPUS_SEED = 0
N_EXAMPLES = 100_000
HOLDOUT_FRAC = 0.2
N_EVAL = 256
N_PROBE = 27
BLOCK, BATCH = 64, 64
TRAJ_STRIDE = 50

EPOCHS = 100
EPOCHS_SHORT = 50
"""The `short` condition's length. Every epoch keyframe below is a fraction of
training, mapped onto the condition's length — the same rule that mapped M1's
keyframes onto these 100 epochs — so the two lengths run the same schedule
*shape*, LR warmup included."""

WARMUP_FRAC = 0.10
ANNEAL_START_FRAC = 0.90
ANNEAL_FLOOR = 0.1
ANTI_ANNEAL_END_FRAC = 0.90
ANTI_PEAK_RATIO = 2.5
ANTI_HOLD_RATIO = 0.30
"""Ex-2.1.10's schedule keyframes, restated as fractions of training. At 100
epochs these reproduce its numbers exactly: LR/anchor warmup over epochs 0–10,
anchor anneal 90–100 to a 0.1 floor, anti-subspace anneal 2.5 → 0.30 (μ/λ)
over epochs 0–90."""


def scheduler_config(epochs: int) -> SchedulerConfig:
    """The LR schedule for a condition of the given length."""
    return SchedulerConfig(epochs=epochs, warmup_epochs=round(epochs * WARMUP_FRAC), min_lr_factor=0.01)


SCORING_LAMBDA = 0.1
"""The reference anchor peak — ex-2.1.6's scoring rung, used by the calibration
and ablation stages. The survey searches λ_a; this is dimension `lam`'s anchor
of continuity, not a constraint on it."""

TAU_REF = 0.1
"""The reference mellowmax temperature: the ex-2.1.10 primary's rung."""

RED_RATE = 0.08
PER_SLOT_RATE = RED_RATE / 2
"""The either-slot labeller, unchanged from ex-2.1.10: each operand draws at
redness⁸ · PER_SLOT_RATE, the line is labeled when either does. This is the
labelling rule D2.1 ends on and the one M3 points toward, so the whole
experiment — ablations and survey alike — runs on it."""

SPAN = PROMPT_SPAN
ROLES = ["op1", "+", "op2", "="]

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)


def p_slot(r: np.ndarray | float, rate: float = PER_SLOT_RATE) -> np.ndarray | float:
    """P(this operand draws a label), as a function of its color's redness."""
    return r**8 * rate


# --- Schedules, evaluated at fractional epochs -------------------------------


def learning_rate(epoch, *, epochs: int = EPOCHS, peak: float = PEAK_LR, steps_per_epoch: int = 1000) -> np.ndarray:
    """The LR schedule for a condition of the given length, as in ex-2.1.8."""
    schedule = configure_schedule(scheduler_config(epochs), peak, steps_per_epoch)
    return np.asarray(schedule(np.asarray(epoch, dtype=float) * steps_per_epoch))


def anchor_weight(epoch, *, peak: float = SCORING_LAMBDA, epochs: int = EPOCHS) -> np.ndarray:
    """The scheduled anchor weight λ at (fractional) *epoch*: ex-2.1.10's shape,
    keyframes scaled to the condition's length.
    """
    return _anchor_weight(
        epoch,
        peak=peak,
        warmup_epochs=epochs * WARMUP_FRAC,
        anneal_start=epochs * ANNEAL_START_FRAC,
        anneal_end=float(epochs),
        floor=ANNEAL_FLOOR,
    )


def anti_weight(
    epoch,
    *,
    lam: float = SCORING_LAMBDA,
    peak_ratio: float = ANTI_PEAK_RATIO,
    hold_ratio: float = ANTI_HOLD_RATIO,
    anneal_end_frac: float = ANTI_ANNEAL_END_FRAC,
    epochs: int = EPOCHS,
) -> np.ndarray:
    """The scheduled anti-subspace weight μ at (fractional) *epoch*."""
    return _anti_subspace_weight(
        epoch,
        lam=lam,
        peak_ratio=peak_ratio,
        hold_ratio=hold_ratio,
        anneal_end=epochs * anneal_end_frac,
        anchor_anneal_start=epochs * ANNEAL_START_FRAC,
        anchor_anneal_end=float(epochs),
        floor=ANNEAL_FLOOR,
    )


def anchor_dose(weight_fn, *, epochs: int = EPOCHS, n: int = 40_001) -> float:
    """∫ w(e)·lr(e) de over training, for any weight schedule *weight_fn*.

    The learning rate belongs in the integrand because it multiplies every
    term's gradient (the ex-2.1.8 dose convention).
    """
    e = np.linspace(0.0, float(epochs), n)
    return float(np.trapezoid(weight_fn(e) * learning_rate(e, epochs=epochs), e))


def dose_centroid(weight_fn, *, epochs: int = EPOCHS, n: int = 40_001) -> float:
    """The timing centroid of a schedule's dose: ∫ e·w·lr de / ∫ w·lr de, in epochs."""
    e = np.linspace(0.0, float(epochs), n)
    f = weight_fn(e) * learning_rate(e, epochs=epochs)
    return float(np.trapezoid(e * f, e) / np.trapezoid(f, e))


ANTI_REL_DOSE_OP = anchor_dose(lambda e: anti_weight(e, lam=1.0))
"""The operating point's anti-subspace dose at unit λ — the denominator of the
survey's `anti_dose` dimension. Dividing λ out makes the multiplier a pure
schedule quantity, so it composes with the `lam` dimension instead of
double-counting it."""

ANTI_CENTROID_OP = dose_centroid(lambda e: anti_weight(e))
"""Epoch 26.5: where the operating point's repulsion is delivered, on average."""

# --- Noise floors (calibration stage) ----------------------------------------

NOISE_RUN = {
    "m_line": 0.0088,
    "m_span": 0.0191,
    "alpha_op1": 0.0212,
    "alpha_span": 0.0264,
    "holdout_em": 0.0087,
    "retention": 0.0061,
    "r2_sim": 0.024,
    "contrast": 0.0057,
}
"""Per-run sd of each statistic on ex-2.1.10's nine-seed primary — the noise
floor every comparison below is resolved against. Frozen here for the decision
rules; the report's calibration cell re-derives every entry from the published
arrays rather than trusting this dict. Caveat carried from the design notes:
these floors hold in the graded regime (τ ≈ 0.1 at the operating point) and do
not transfer to τ ≲ 0.03, where the pull latches per run and per-run statistics
go bimodal — which is why the survey bounds τ away from that regime and carries
a latch detector instead."""


def equiv_band(stat: str, n_a: int = 3, n_b: int = 3) -> float:
    """The smallest seed-mean difference the ablation stage may call a difference.

    Two seed-mean estimates from *n_a* and *n_b* runs differ, under no true
    effect, with sd σ·√(1/n_a + 1/n_b); twice that is the resolution rule's
    band. Within the band, the two conditions are "not resolvably different" —
    which is a statement about our resolution, never evidence of equality.
    """
    return 2.0 * NOISE_RUN[stat] * float(np.sqrt(1.0 / n_a + 1.0 / n_b))


DECISION_STATS = ["m_line", "alpha_op1", "retention", "r2_sim"]
"""The statistics an ablation decision reads, each against its `equiv_band`.
Task cost is gated separately (`TASK_GATE`, an absolute bar), and the latch
detector (`LATCH_PI`) is a per-run veto rather than a band."""

# --- Stage 2: the ablation conditions ----------------------------------------

N_SEEDS_ABLATION = 3

ABLATION_CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "epochs": EPOCHS},
    {"name": "ref", "lam": SCORING_LAMBDA, "epochs": EPOCHS},
    {"name": "flat-anchor", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "anchor_shape": "flat"},
    {"name": "anti-hold", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "anti_shape": "flat", "anti_ratio": ANTI_HOLD_RATIO},
    {"name": "anti-peak", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "anti_shape": "flat", "anti_ratio": ANTI_PEAK_RATIO},
    {"name": "linear", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "interp": "linear"},
    {"name": "short", "lam": SCORING_LAMBDA, "epochs": EPOCHS_SHORT},
    {"name": "short-lam0", "lam": 0.0, "epochs": EPOCHS_SHORT},
]
"""Eight conditions × 3 seeds = 24 runs, all at τ = TAU_REF under the
either-slot labeller. `ref` re-runs ex-2.1.10's primary recipe through this
experiment's code path, so every contrast is between conditions that went
through identical code; the nine-seed noise floors are read from ex-2.1.10's
published arrays instead of being re-bought here.

- `flat-anchor` holds λ constant for all of training: no warmup ramp, no end
  anneal, no floor. Free of invariant choices because flat-at-peak and
  dose-matched coincide within 3% (the calibration cell shows the integral),
  so this one arm ablates four schedule knobs together.
- `anti-hold` / `anti-peak` hold μ/λ constant at the schedule's two ends. The
  anti term cannot be dose-matched flat (the match would sit at 1.7× the
  anchor peak for all of training — a different regime), so these bracket the
  schedule instead, doses reported rather than matched.
- `linear` replaces every minimum-jerk ramp and anneal with a straight line,
  same keyframes (both anchor terms; the LR schedule is optax's and stays).
- `short` runs the whole recipe at half length, keyframes scaled by fraction
  of training; `short-lam0` is its task-cost control."""

# --- Stage 2: decision rules -------------------------------------------------

DECISION_RULES = """\
Frozen before any ablation run. "Within band" means: for every statistic in
DECISION_STATS, |seed-mean difference from `ref`| < equiv_band(stat); and the
task gate holds (holdout EM within TASK_GATE of the same-length control); and
no run latches a syntax role (LATCH_PI). Any resolved regression keeps the
incumbent recipe — ties break toward the simpler recipe only when nothing
resolves.

1. `flat-anchor` within band, or better on m_line → the survey runs the flat
   anchor (four schedule knobs deleted). Otherwise it keeps the schedule,
   also deleting the knobs (kept fixed by inheritance, now with evidence the
   shape earns its place). Predicted failure mode, from the ex-2.1.9 commit
   window: full λ on a still-random model latches a syntax role — so the
   latch veto, and which role leads, is the reading that matters.
2. `anti-hold` or `anti-peak` within band or better → the anti schedule
   collapses to a flat ratio, and the survey searches SPACE_ANTI_FLAT
   (branch B; prefer the passing level nearer `anti-hold`, the lower dose).
   Both resolved worse → the shape matters; the survey searches the schedule
   family, SPACE_ANTI_SCHED (branch A).
3. `short` within band (against `ref`, task-gated against `short-lam0`) →
   the survey runs at EPOCHS_SHORT, halving every trial. Otherwise 100.
4. `linear` within band → shapes are interchangeable; keep minimum-jerk
   everywhere and stop carrying the question. Resolved worse → keep
   minimum-jerk, note the shape as load-bearing. Either way no survey
   dimension: this arm only retires a todo item.
"""

# --- Stage 3: the survey space -----------------------------------------------

SOBOL_SEED = 0
N_TRIALS = 32  # a power of two, so the scrambled Sobol set keeps its balance
N_PROMOTE = 6
SEEDS_PROMOTE = 5  # total per promoted trial: the round-1 seed plus four more

SPACE_COMMON: dict[str, tuple[float, float, str]] = {
    "lam": (0.02, 1.0, "log"),
    "tau": (0.05, 0.30, "log"),
}
"""Dimension → (low, high, scale). λ_a spans half the scoring rung to the
ex-2.1.7 ceiling arm's weight, where selectivity was lost — the optimum is
believed inside. τ spans the graded regime: the calibration cell shows the
leading softmin weight runs 0.74 → 0.42 across it, the 0.6–0.8 target band
sits inside, and the latching regime (τ ≲ 0.03) stays outside. Both are
temperatures/weights, so log scale."""

SPACE_ANTI_SCHED: dict[str, tuple[float, float, str]] = {
    "anti_peak_ratio": (0.75, 4.0, "log"),
    "anti_anneal_end_frac": (0.30, 0.95, "lin"),
}
"""Branch A (schedule retained), sampled in the schedule family's own knobs
with `hold_ratio` fixed at 0.30. The raw knobs are correlated with the
quantity that matters — the endpoint *is* the dose axis (ex-2.1.8) — so the
analysis reports marginals in the derived coordinates (relative dose,
timing centroid) via `anti_rel_dose`/`dose_centroid`, not in the sampled
knobs. Sampling stays in the knob box because every point of it is a valid
schedule, whereas a box in (dose, centroid) has unreachable corners whose
projection would distort the marginals."""

SPACE_ANTI_FLAT: dict[str, tuple[float, float, str]] = {
    "anti_ratio": (0.10, 3.0, "log"),
}
"""Branch B (flat anti): one dimension, μ/λ constant over training, spanning
a third of the hold level to above the peak."""


def sobol_trials(space: dict[str, tuple[float, float, str]], seed: int = SOBOL_SEED) -> list[dict]:
    """The frozen trial list: a scrambled Sobol set over *space*, N_TRIALS points.

    Deterministic given (space, seed), so the list is reviewable before the
    run and identical when the run builds it. Sobol rather than uniform draws
    so the one-dimensional marginals stay even at this budget; scrambled so no
    trial sits exactly on a box edge.
    """
    from scipy.stats import qmc

    dims = list(space)
    sampler = qmc.Sobol(d=len(dims), scramble=True, seed=seed)
    u = sampler.random_base2(int(np.log2(N_TRIALS)))
    trials = []
    for i, row in enumerate(u):
        t: dict = {"trial": i}
        for (name, (lo, hi, scale)), x in zip(space.items(), row, strict=True):
            v = lo * (hi / lo) ** x if scale == "log" else lo + (hi - lo) * x
            t[name] = float(v)
        trials.append(t)
    return trials


def anti_rel_dose(trial: dict, epochs: int = EPOCHS) -> float:
    """A trial's delivered repulsion at unit λ, relative to the operating point."""
    if "anti_ratio" in trial:
        w = lambda e: trial["anti_ratio"] * np.ones_like(np.asarray(e, dtype=float))  # noqa: E731
    else:
        w = lambda e: anti_weight(  # noqa: E731
            e,
            lam=1.0,
            peak_ratio=trial["anti_peak_ratio"],
            anneal_end_frac=trial["anti_anneal_end_frac"],
            epochs=epochs,
        )
    return anchor_dose(w, epochs=epochs) / ANTI_REL_DOSE_OP


# --- Stage 3: objective, constraints, promotion ------------------------------

TASK_GATE = 0.02
"""Feasibility: |named_holdout EM − same-length control| within this. As ex-2.1.6 on."""

MEAN_ALIGN_GATE = 0.1
"""Feasibility: ᾱ at op1 (containment). Too noisy to rank on (per-run sd is
42% of the primary's mean), exactly right as a constraint. As ex-2.1.8 on."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
"""Feasibility: on the m_line trajectory, a run whose running maximum reaches
the floor must end at ≥ RETENTION_GATE × that maximum. As ex-2.1.8 on."""

GRADE_R2_DROP = 0.10
"""Feasibility: per-run r²(op1 response, sim^1.5) within this of the ablation
stage's `ref` seed mean. The r² of a trial is read per run and seed-averaged;
relative form as ex-2.1.9 on."""

CONTRAST_MIN = 0.1
"""Feasibility: the ex-2.1.10 H2(b) group contrast (op2-weight, post-attention
slices). The primary sits at 0.85 with per-run sd 0.006; a global latch
predicts ≈ 0. This is the per-line-localization floor, at ex-2.1.10's partial
bar."""

LATCH_PI = 0.5
"""Per-run veto: a run latches if the deep-slice (post-attention) mean softmin
weight on `+` or `=` exceeds this. Calibrated on ex-2.1.9's stored profiles:
latched runs sit at ≥ 0.85, healthy runs at ≤ 0.03, so the threshold has a
wide gap on both sides. Any latched run makes its trial infeasible."""

OBJECTIVE = """\
Rank feasible trials by m_line (seed mean), largest first. m_line is the one
statistic tight enough to rank on (per-run sd 0.0088 against effects of
several sd), and the calibration stage shows why it cannot stand alone: it
rises monotonically toward soft τ while grading and the group contrast
collapse, so the constraints — not the objective — hold the smeared end out.
Containment, retention, grading, contrast and the task gate are constraints,
not scores; the deliverable is the feasible region and the marginals, with
the proposed point one line of it.
"""

PROMOTION = """\
Round 1 runs every trial at seed 0. The N_PROMOTE feasible trials with the
highest m_line are promoted to SEEDS_PROMOTE seeds (four more runs each);
if fewer are feasible, promote all of them; if none are, publish the
infeasibility map and propose nothing. The proposed operating point is the
promoted trial with the highest seed-mean m_line that is feasible on every
seed (constraints re-read on seed means; the latch veto stays per-run).
Promotion corrects the winner's curse only partly — the survey's number is
a proposal, and the next preregistered experiment confirms it at fresh
seeds before anyone quotes it.
"""

STOPPING_RULE = """\
Fixed budget, two rounds, no adaptive continuation: the survey ends when the
promoted round has run. Diverged or failed trials are published as such
(ctx.map with allow_partial=True), not resampled.
"""

# --- Budget ------------------------------------------------------------------

MAX_CONTAINERS = 8
N_RUNS_ABLATION = len(ABLATION_CONDITIONS) * N_SEEDS_ABLATION  # 24
N_RUNS_SURVEY = N_TRIALS + N_PROMOTE * (SEEDS_PROMOTE - 1)  # 62 at most
"""≈ 86 training runs in the worst case, ~25 min of L4 each with the slim
eval — comparable to two ex-2.1.10s, and half that for the survey stage if
`short` is adopted. Each stage is launched with --max-containers 8 and the
stage's run count times a generous per-run bound as --budget."""

# --- Result refs -------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.11/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.11/arrays"
EVALS_REF = "reports/m2/ex-2.1.11/evals"
PROBE_REF = "reports/m2/ex-2.1.11/probes"

EX218_METRICS_REF = "reports/m2/ex-2.1.8/metrics"
EX219_METRICS_REF = "reports/m2/ex-2.1.9/metrics"
EX2110_METRICS_REF = "reports/m2/ex-2.1.10/metrics"
EX2110_ARRAYS_REF = "reports/m2/ex-2.1.10/arrays"
EX2110_PROBES_REF = "reports/m2/ex-2.1.10/probes"
"""Published inputs to the calibration stage: the nine-seed noise floors, the
τ ↔ weight curves and the trade figure all read ex-2.1.10; the latch-detector
calibration reads ex-2.1.9; the across-point correlation adds ex-2.1.8."""
