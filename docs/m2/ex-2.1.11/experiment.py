"""Experiment 2.1.11: ablations and a survey for the D2.1 operating point.

D2.1's recipe was assembled piecewise: the schedules came from M1 by inheritance, the anchor weight is ex-2.1.6's scoring rung, τ and the anti-subspace operating point were each chosen by a short ladder in whatever experiment introduced them. Nothing has asked whether the parts are needed, or where the good region is. This experiment closes D2.1 by doing both, in stages, cheapest first:

1. **Calibration** (free, report-side): noise floors from ex-2.1.10's nine-seed primary, the objective trade structure across stored operating points, the τ ↔ leading-weight map, and the schedule dose integrals.
2. **Ablations** (preregistered conditions): flat arms bracketing each schedule, their composition, a half-length condition, and a linear-shape arm. Each "no difference" deletes dimensions from the search and simplifies the recipe the survey runs on.
3. **Survey** (Sobol): the surviving continuous parameters (λ_a, τ, and the anti-subspace dose), sampled in one round at a single seed, with a promoted second round at more seeds.

This is a survey in the sense of the `science` skill: it preregisters a search plan rather than hypotheses, scores nothing, and proposes an operating point that the next preregistered experiment must confirm at fresh seeds before any number here is quoted as a result.

    bin/mini run docs/m2/ex-2.1.11/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.11
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import (
    PROMPT_SPAN,
    Shape,
    anchor_weight as _anchor_weight,
    anti_subspace_weight as _anti_subspace_weight,
    softmin_weights,
)
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from sca.training.scheduler import configure_schedule
from mini import Ctx, Experiment, get_data_dir

__all__ = ["experiment", "softmin_weights"]  # the latter re-exported for the report's calibration cells

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
"""The `short` condition's length. Every epoch keyframe below is a fraction of training, mapped onto the condition's length (the same rule that mapped M1's keyframes onto these 100 epochs), so the two lengths run the same schedule *shape*, LR warmup included."""

WARMUP_FRAC = 0.10
ANNEAL_START_FRAC = 0.90
ANNEAL_FLOOR = 0.1
ANTI_ANNEAL_END_FRAC = 0.90
ANTI_PEAK_RATIO = 2.5
ANTI_HOLD_RATIO = 0.30
"""Ex-2.1.10's schedule keyframes, restated as fractions of training. At 100 epochs these reproduce its numbers exactly: LR/anchor warmup over epochs 0–10, anchor anneal 90–100 to a 0.1 floor, anti-subspace anneal 2.5 → 0.30 (λ_s̄/λ_a) over epochs 0–90."""


def scheduler_config(epochs: int) -> SchedulerConfig:
    """The LR schedule for a condition of the given length."""
    return SchedulerConfig(epochs=epochs, warmup_epochs=round(epochs * WARMUP_FRAC), min_lr_factor=0.01)


SCORING_LAMBDA = 0.1
"""The reference anchor peak: ex-2.1.6's scoring rung, used by the calibration and ablation stages. The survey searches λ_a; this is dimension `lam`'s anchor of continuity, not a constraint on it."""

TAU_REF = 0.1
"""The reference mellowmax temperature: the ex-2.1.10 primary's rung."""

RED_RATE = 0.08
PER_SLOT_RATE = RED_RATE / 2
"""The either-slot labeller, unchanged from ex-2.1.10: each operand draws at redness⁸ · PER_SLOT_RATE, the line is labeled when either does. This is the labelling rule D2.1 ends on and the one M3 points toward, so the whole experiment (ablations and survey alike) runs on it."""

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


def anchor_weight(
    epoch, *, peak: float = SCORING_LAMBDA, epochs: int = EPOCHS, shape: Shape = "min-jerk"
) -> np.ndarray:
    """The scheduled anchor weight λ_a at (fractional) *epoch*: ex-2.1.10's shape, keyframes scaled to the condition's length."""
    return _anchor_weight(
        epoch,
        peak=peak,
        warmup_epochs=epochs * WARMUP_FRAC,
        anneal_start=epochs * ANNEAL_START_FRAC,
        anneal_end=float(epochs),
        floor=ANNEAL_FLOOR,
        shape=shape,
    )


def anti_weight(
    epoch,
    *,
    lam: float = SCORING_LAMBDA,
    peak_ratio: float = ANTI_PEAK_RATIO,
    hold_ratio: float = ANTI_HOLD_RATIO,
    anneal_end_frac: float = ANTI_ANNEAL_END_FRAC,
    epochs: int = EPOCHS,
    shape: Shape = "min-jerk",
) -> np.ndarray:
    """The scheduled anti-subspace weight λ_s̄ at (fractional) *epoch*."""
    return _anti_subspace_weight(
        epoch,
        lam=lam,
        peak_ratio=peak_ratio,
        hold_ratio=hold_ratio,
        anneal_end=epochs * anneal_end_frac,
        anchor_anneal_start=epochs * ANNEAL_START_FRAC,
        anchor_anneal_end=float(epochs),
        floor=ANNEAL_FLOOR,
        shape=shape,
    )


def anchor_dose(weight_fn, *, epochs: int = EPOCHS, n: int = 40_001) -> float:
    """∫ w(e)·lr(e) de over training, for any weight schedule *weight_fn*.

    The learning rate belongs in the integrand because it multiplies every term's gradient (the ex-2.1.8 dose convention).
    """
    e = np.linspace(0.0, float(epochs), n)
    return float(np.trapezoid(weight_fn(e) * learning_rate(e, epochs=epochs), e))


def dose_centroid(weight_fn, *, epochs: int = EPOCHS, n: int = 40_001) -> float:
    """The timing centroid of a schedule's dose: ∫ e·w·lr de / ∫ w·lr de, in epochs."""
    e = np.linspace(0.0, float(epochs), n)
    f = weight_fn(e) * learning_rate(e, epochs=epochs)
    return float(np.trapezoid(e * f, e) / np.trapezoid(f, e))


def anti_dose_ref(epochs: int = EPOCHS) -> float:
    """The reference schedule's anti-subspace dose at unit λ_a, at the given length.

    The denominator of the survey's `anti_dose` dimension. Dividing λ_a out makes the multiplier a pure schedule quantity, so it composes with the `lam` dimension instead of double-counting it. It is length-matched because a 50-epoch condition delivers half the dose of the same shape at 100, and a relative coordinate that read 0.5 for the reference shape would misname the axis the survey's marginals are reported on.
    """
    return anchor_dose(lambda e: anti_weight(e, lam=1.0, epochs=epochs), epochs=epochs)


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
"""Per-run sd of each statistic on ex-2.1.10's nine-seed primary: the noise floor every comparison below is resolved against. Frozen here for the decision rules; the report's calibration cell re-derives every entry from the published arrays rather than trusting this dict. Caveat carried from the design notes: these floors hold in the graded regime (τ ≈ 0.1 at the operating point) and do not transfer to τ ≲ 0.03, where the pull latches per run and per-run statistics go bimodal, which is why the survey bounds τ away from that regime and carries a latch detector instead."""


def equiv_band(stat: str, n_a: int = 3, n_b: int = 3) -> float:
    """The smallest seed-mean difference the ablation stage may call a difference.

    Two seed-mean estimates from *n_a* and *n_b* runs differ, under no true effect, with sd σ·√(1/n_a + 1/n_b); twice that is the resolution rule's band. Within the band, the two conditions are "not resolvably different"; that is a statement about our resolution, never evidence of equality.
    """
    return 2.0 * NOISE_RUN[stat] * float(np.sqrt(1.0 / n_a + 1.0 / n_b))


DECISION_STATS = ["m_line", "alpha_op1", "retention", "r2_sim", "contrast"]
"""The statistics an ablation decision reads, each against its `equiv_band`. Task cost is gated separately (`TASK_GATE`, an absolute bar), and the latch detector (`LATCH_PI`) is a per-run veto rather than a band."""
# REVIEW: added `contrast` (round-1 tension). The anti-subspace analysis names
# it as the expected discriminator ("anti-hold loses containment or contrast"),
# but the rules could not read it, so an arm losing contrast alone would have
# scored "within band" and been adopted. Its band is tiny (2σ·√(2/3) ≈ 0.009
# against a statistic the primary holds at 0.85), so the addition costs little
# resolution and any error it introduces keeps the incumbent recipe. Verify:
# with five statistics over six comparison arms, the note on family-wise
# resolution in DECISION_RULES sizes the spurious-call risk this raises.

# --- Stage 2: the ablation conditions ----------------------------------------

N_SEEDS_ABLATION = 3

ABLATION_CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "epochs": EPOCHS},
    {"name": "ref", "lam": SCORING_LAMBDA, "epochs": EPOCHS},
    {"name": "flat-anchor", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "anchor_shape": "flat"},
    {"name": "anti-hold", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "anti_shape": "flat", "anti_ratio": ANTI_HOLD_RATIO},
    {"name": "anti-peak", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "anti_shape": "flat", "anti_ratio": ANTI_PEAK_RATIO},
    {
        "name": "flat-both",
        "lam": SCORING_LAMBDA,
        "epochs": EPOCHS,
        "anchor_shape": "flat",
        "anti_shape": "flat",
        "anti_ratio": ANTI_HOLD_RATIO,
    },
    {"name": "linear", "lam": SCORING_LAMBDA, "epochs": EPOCHS, "interp": "linear"},
    {"name": "short", "lam": SCORING_LAMBDA, "epochs": EPOCHS_SHORT},
    {"name": "short-lam0", "lam": 0.0, "epochs": EPOCHS_SHORT},
]
"""Nine conditions × 3 seeds = 27 runs, all at τ = TAU_REF under the either-slot labeller. `ref` re-runs ex-2.1.10's primary recipe through this experiment's code path, so every contrast is between conditions that went through identical code; the nine-seed noise floors are read from ex-2.1.10's published arrays instead of being re-bought here.

- `flat-anchor` holds λ_a constant for all of training: no warmup ramp, no end anneal, no floor. Free of invariant choices because flat-at-peak and dose-matched coincide within 3% (the calibration cell shows the integral), so this one arm ablates four schedule parameters together.
- `anti-hold` / `anti-peak` hold λ_s̄/λ_a constant at the schedule's two ends. The anti term cannot be dose-matched flat (the match would sit at 1.7× the anchor peak for all of training, a different regime), so these bracket the schedule instead, doses reported rather than matched.
- `flat-both` composes the two flat simplifications (flat anchor, anti flat at the hold ratio, the level rule 2 prefers on a pass): the recipe the survey would run if decision rules 1 and 2 both simplify. The one-factor arms cannot see an interaction between the two schedules, so the composition is tested before it can be adopted.
- `linear` replaces every minimum-jerk ramp and anneal with a straight line, same keyframes (both anchor terms; the LR schedule is optax's and stays).
- `short` runs the whole recipe at half length, keyframes scaled by fraction of training; `short-lam0` is its task-cost control."""
# REVIEW: added `flat-both` (round-2 review). Rules 1 and 2 could both adopt a
# flat arm on one-factor evidence and hand the survey a composition no ablation
# ran; rule 3 now gates that composition on this arm. Costs three runs; the
# family-wise caveat in DECISION_RULES now counts six comparison arms. Verify:
# if the schedules interact, the signature is `flat-both` resolving worse while
# each single flat arm passes.

# --- Stage 2: decision rules -------------------------------------------------

DECISION_RULES = """\
Frozen before any ablation run. "Within band" means: for every statistic in DECISION_STATS, |seed-mean difference from `ref`| < equiv_band(stat); and the task gate holds (holdout EM within TASK_GATE of the same-length control); and no run latches a syntax role (LATCH_PI). Any resolved regression keeps the incumbent recipe; ties break toward the simpler recipe only when nothing resolves.

Resolution caveat, known going in: five statistics read at ±2 sd across six comparison arms makes a spurious "resolved" call somewhere likely even under no true effect. Every such error lands on the conservative side, keeping the incumbent recipe or the longer run, so the stage is biased toward keeping complexity, never toward adopting a simplification the data doesn't support. Read a lone marginal excursion with that in mind.

1. `flat-anchor` within band, or better on m_line → the survey runs the flat anchor (four schedule parameters deleted). Otherwise it keeps the schedule, also deleting the parameters (kept fixed by inheritance, now with evidence the shape earns its place). Predicted failure mode, from the ex-2.1.9 commit window: full λ_a on a still-random model latches a syntax role, so the latch veto, and which role leads, is the reading that matters.
2. `anti-hold` or `anti-peak` within band or better → the anti schedule collapses to a flat ratio, and the survey searches SPACE_ANTI_FLAT (branch B; prefer the passing level nearer `anti-hold`, the lower dose).
   Both resolved worse → the shape matters; the survey searches the schedule family, SPACE_ANTI_SCHED (branch A).
3. Rules 1 and 2 are one-factor readings, so their simplifications are adopted together only if `flat-both` is also within band. If each flat arm passes alone and `flat-both` fails, the schedules interact: keep the anchor schedule and adopt the flat anti, which preserves branch B and costs the survey nothing (rule 1 fixes the anchor parameters either way). `flat-both` is read only when rules 1 and 2 both simplify.
4. `short` within band (against `ref`, task-gated against `short-lam0`) → the survey runs at EPOCHS_SHORT, halving every trial. Otherwise 100.
5. `linear` within band → shapes are interchangeable; keep minimum-jerk everywhere and stop carrying the question. Resolved worse → keep minimum-jerk, note the shape as load-bearing. Either way no survey dimension: this arm only retires a todo item.
"""

CONTROL_OF = {EPOCHS: "lam0", EPOCHS_SHORT: "short-lam0"}
"""The un-anchored arm each length is task-gated against."""

STAT_DIRECTION = {"m_line": "up", "alpha_op1": "down", "retention": "up", "r2_sim": "up", "contrast": "up"}
"""Which side of `ref` counts as better, per decision statistic. Containment is the one that improves downward: ᾱ at op1 is a ceiling (`MEAN_ALIGN_GATE`), and a pull that spreads onto unlabeled states raises it."""
# REVIEW: this table, and `simplifies` below, are how DECISION_RULES's "within
# band, or better" is executed: an arm simplifies when no decision statistic
# resolves *worse* than `ref`, which reads the rules' "any resolved regression
# keeps the incumbent recipe" as the operative clause and lets a resolved
# improvement pass alongside a tie. Frozen with the implementation, before any
# run. Verify: the ablation tables print every arm's per-statistic verdict, so
# a stricter reading (every statistic strictly within band) can be re-scored
# from the published numbers without re-running anything.


def ablation_summary(cells: list[dict]) -> dict[str, dict]:
    """Seed means of every decision statistic per condition, plus the latch veto."""
    summary: dict[str, dict] = {}
    for c in ABLATION_CONDITIONS:
        rows = [r for r in cells if r["condition"] == c["name"]]
        if not rows:
            continue
        summary[c["name"]] = {
            **{s: float(np.mean([r[s] for r in rows])) for s in (*DECISION_STATS, "holdout_em")},
            "latched": [r["label"] for r in rows if r["latch_pi"] > LATCH_PI],
            "n": len(rows),
            "epochs": c["epochs"],
        }
    return summary


def arm_verdict(summary: dict[str, dict], arm: str, ref: str = "ref") -> dict:
    """One arm against *ref*: per-statistic deltas with their bands, the gates, and the rule's reading."""
    a, r = summary[arm], summary[ref]
    stats = {}
    for s in DECISION_STATS:
        delta = a[s] - r[s]
        band = equiv_band(s, a["n"], r["n"])
        better = delta > 0 if STAT_DIRECTION[s] == "up" else delta < 0
        stats[s] = {
            "delta": delta,
            "band": band,
            "verdict": "within band" if abs(delta) < band else ("better" if better else "regression"),
        }
    cost = abs(a["holdout_em"] - summary[CONTROL_OF[a["epochs"]]]["holdout_em"])
    return {
        "arm": arm,
        "ref": ref,
        "stats": stats,
        "task_cost": cost,
        "task_ok": cost <= TASK_GATE,
        "latched": a["latched"],
        "simplifies": not a["latched"]
        and cost <= TASK_GATE
        and not any(v["verdict"] == "regression" for v in stats.values()),
    }


def decide(summary: dict[str, dict]) -> dict:
    """Apply DECISION_RULES to the ablation seed means: the recipe and space the survey runs on.

    Deterministic given the ablation results, so the branch the survey takes is a function of published numbers rather than a judgement call made after seeing them.
    """
    verdicts = {arm: arm_verdict(summary, arm) for arm in summary if arm not in ("lam0", "short-lam0", "ref")}
    flat_anchor = verdicts["flat-anchor"]["simplifies"]  # rule 1
    flat_anti = [a for a in ("anti-hold", "anti-peak") if verdicts[a]["simplifies"]]  # rule 2
    # Rule 3: the two flat simplifications are adopted together only if their
    # composition also holds; where it doesn't, the flat anti survives alone.
    composes = verdicts["flat-both"]["simplifies"] if (flat_anchor and flat_anti) else None
    if flat_anchor and flat_anti and not composes:
        flat_anchor = False
    return {
        "anchor_shape": "flat" if flat_anchor else "min-jerk",
        "anti_branch": "B" if flat_anti else "A",
        # Rule 2's tie-break: the passing level nearer `anti-hold`, the lower dose.
        "anti_ratio_ref": ANTI_HOLD_RATIO if "anti-hold" in flat_anti else (ANTI_PEAK_RATIO if flat_anti else None),
        "epochs": EPOCHS_SHORT if verdicts["short"]["simplifies"] else EPOCHS,  # rule 4
        "shape_interchangeable": verdicts["linear"]["simplifies"],  # rule 5, no survey dimension
        "flat_both_read": composes,
        "verdicts": verdicts,
    }


def survey_space(decision: dict) -> dict[str, tuple[float, float, str]]:
    """The dimensions the survey samples, given the ablation decision."""
    return {**SPACE_COMMON, **(SPACE_ANTI_FLAT if decision["anti_branch"] == "B" else SPACE_ANTI_SCHED)}


# --- Stage 3: the survey space -----------------------------------------------

SOBOL_SEED = 0
N_TRIALS = 32  # a power of two, so the scrambled Sobol set keeps its balance
N_PROMOTE = 6
SEEDS_PROMOTE = 5  # total per promoted trial: the round-1 seed plus four more

SPACE_COMMON: dict[str, tuple[float, float, str]] = {
    "lam": (0.02, 1.0, "log"),
    "tau": (0.05, 0.30, "log"),
}
"""Dimension → (low, high, scale). λ_a spans a fifth of the scoring rung to the ex-2.1.7 ceiling arm's weight, where selectivity was lost; the optimum is believed inside. τ spans the graded regime: the calibration cell shows the leading softmin weight runs 0.78 → 0.57 across it at the embedding slice, spanning nearly all of the 0.6–0.8 target band from ex-2.1.9, and the latching regime (τ ≲ 0.03) stays outside. Both are temperatures/weights, so log scale."""
# REVIEW: "half the scoring rung" → "a fifth" (SCORING_LAMBDA is 0.1 and the
# bound is 0.02). The τ figures were then corrected twice. They were first
# narrowed to "the lower part of the band", on the reading that reaching 0.8
# needs τ ≈ 0.04; that came from a statistic other than the one ex-2.1.9's band
# is set on. On `pi` — the label-affinity-weighted per-colour profile maxed over
# roles, which is also what LATCH_PI reads — the range is [0.78, 0.57] at the
# embedding slice, and 0.8 is not reachable at any τ: the curve saturates at
# 0.78 as τ → 0, because the profile's own spread runs out of room to sharpen.
# The bounds are unchanged, and cover the band better than either earlier note
# credited. Verify: the tau-range figure's embedding-slice curve passes through
# 0.77 at τ = 0.1, reproducing ex-2.1.9's published lead_emb.

SPACE_ANTI_SCHED: dict[str, tuple[float, float, str]] = {
    "anti_peak_ratio": (0.75, 4.0, "log"),
    "anti_anneal_end_frac": (0.30, 0.95, "lin"),
}
"""Branch A (schedule retained), sampled in the schedule family's own parameters with `hold_ratio` fixed at 0.30. The sampled parameters are correlated with the quantity that matters (the endpoint *is* the dose axis, ex-2.1.8), so the analysis reports marginals in the derived coordinates (relative dose, timing centroid) via `anti_rel_dose`/`dose_centroid`, not in the sampled parameters. Sampling stays in the parameter box because every point of it is a valid schedule, whereas a box in (dose, centroid) has unreachable corners whose projection would distort the marginals."""

SPACE_ANTI_FLAT: dict[str, tuple[float, float, str]] = {
    "anti_ratio": (0.10, 3.0, "log"),
}
"""Branch B (flat anti): one dimension, λ_s̄/λ_a constant over training, spanning a third of the hold level to above the peak."""


def sobol_trials(space: dict[str, tuple[float, float, str]], seed: int = SOBOL_SEED) -> list[dict]:
    """The frozen trial list: a scrambled Sobol set over *space*, N_TRIALS points.

    Deterministic given (space, seed), so the list is reviewable before the run and identical when the run builds it. Sobol rather than uniform draws so the one-dimensional marginals stay even at this budget; scrambled so no trial sits exactly on a box edge.
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
    """A trial's delivered repulsion at unit λ_a, relative to the operating point."""
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
    # REVIEW: normalized against the reference dose at the *same* length, and
    # the superseded ANTI_REL_DOSE_OP constant removed. The numerator already
    # varied with `epochs` while the denominator was pinned to 100, so if
    # decision rule 4 adopted `short` every trial's relative dose would have
    # halved and the reference shape would have read 0.500 instead of 1.00.
    # This changes a reported coordinate, not what is sampled or ranked.
    # Verify: at epochs=EPOCHS the value is unchanged, so the frozen branch A
    # trial table in the report is identical.
    return anchor_dose(w, epochs=epochs) / anti_dose_ref(epochs)


# --- Stage 3: objective, constraints, promotion ------------------------------

TASK_GATE = 0.02
"""Feasibility: |named_holdout EM − same-length control| within this. As ex-2.1.6 on."""

MEAN_ALIGN_GATE = 0.1
"""Feasibility: ᾱ at op1 (containment). Too noisy to rank on (per-run sd is 42% of the primary's mean), well suited as a constraint. As ex-2.1.8 on."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
"""Feasibility: on the m_line trajectory, a run whose running maximum reaches the floor must end at ≥ RETENTION_GATE × that maximum. As ex-2.1.8 on."""

GRADE_R2_DROP = 0.10
"""Feasibility: per-run r²(op1 response, sim^1.5) within this of the ablation stage's `ref` seed mean. The r² of a trial is read per run and seed-averaged; relative form as ex-2.1.9 on."""

CONTRAST_MIN = 0.1
"""Feasibility: the ex-2.1.10 H2(b) group contrast (op2-weight, post-attention slices). The primary sits at 0.85 with per-run sd 0.006; a global latch predicts ≈ 0. This is the per-line-localization floor, at ex-2.1.10's partial bar."""

LATCH_PI = 0.5
"""Per-run veto: a run latches if the deep-slice (post-attention) mean softmin weight on `+` or `=` exceeds this. Calibrated on ex-2.1.9's stored profiles: latched runs sit at ≥ 0.85, healthy runs at ≤ 0.03, so the threshold has a wide gap on both sides. Any latched run makes its trial infeasible."""

OBJECTIVE = """\
Rank feasible trials by m_line (seed mean), largest first. m_line is the one statistic tight enough to rank on (per-run sd 0.0088 against effects of several sd), and the calibration stage shows why it cannot stand alone: it rises monotonically toward soft τ while grading and the group contrast collapse, so the constraints, not the objective, hold the smeared end out. Containment, retention, grading, contrast and the task gate are constraints, not scores; the deliverable is the feasible region and the marginals, with the proposed point one line of it.
"""

PROMOTION = """\
Round 1 runs every trial at seed 0. The N_PROMOTE feasible trials with the highest m_line are promoted to SEEDS_PROMOTE seeds (four more runs each); if fewer are feasible, promote all of them; if none are, publish the infeasibility map and propose nothing. The proposed operating point is the promoted trial with the highest seed-mean m_line that is feasible on every seed (constraints re-read on seed means; the latch veto stays per-run). Promotion corrects the winner's curse only partly; the survey's number is a proposal, and the next preregistered experiment confirms it at fresh seeds before anyone quotes it.
"""

STOPPING_RULE = """\
Fixed budget, two rounds, no adaptive continuation: the survey ends when the promoted round has run. Diverged or failed trials are published as such (ctx.map with allow_partial=True), not resampled.
"""

# --- Budget ------------------------------------------------------------------

MAX_CONTAINERS = 8
N_RUNS_ABLATION = len(ABLATION_CONDITIONS) * N_SEEDS_ABLATION  # 27
N_RUNS_SURVEY = N_TRIALS + N_PROMOTE * (SEEDS_PROMOTE - 1)  # 56 at most
"""≈ 83 training runs in the worst case, ~25 min of L4 each with the slim eval. That is about 3.5 times the 24 runs of ex-2.1.10, or about twice the training time if `short` is adopted and the survey stage runs at half length. Each run is also cheaper, because the slim eval drops the readability, geometry, and leakage passes. Each stage is launched with --max-containers 8 and the stage's run count times a generous per-run bound as --budget."""
# REVIEW: corrected 62 → 56 and 86 → 80 (arithmetic: 32 + 6×4 = 56, + the
# ablation runs), then 80 → 83 when `flat-both` added three runs. The report
# renders these from the constants, so only these comments were stale. Recast
# "comparable to two ex-2.1.10s" as ~3.5× its run count (83/24 = 3.5); the
# "twice" figure holds only in run-equivalents once `short` halves the survey
# stage. Verify: if N_PROMOTE or SEEDS_PROMOTE move, both numbers here move
# with them.

# --- Result refs -------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.11/metrics"
"""The ablation stage's results, published as soon as that stage settles: the survey runs after them and lands under `SURVEY_REF`, so the report renders stage 2 while stage 3 is still in flight."""
ARRAYS_REF = "reports/m2/ex-2.1.11/arrays"
EVALS_REF = "reports/m2/ex-2.1.11/evals"
PROBE_REF = "reports/m2/ex-2.1.11/probes"
SURVEY_REF = "reports/m2/ex-2.1.11/survey"

EX218_METRICS_REF = "reports/m2/ex-2.1.8/metrics"
EX219_METRICS_REF = "reports/m2/ex-2.1.9/metrics"
EX2110_METRICS_REF = "reports/m2/ex-2.1.10/metrics"
EX2110_ARRAYS_REF = "reports/m2/ex-2.1.10/arrays"
EX2110_PROBES_REF = "reports/m2/ex-2.1.10/probes"
"""Published inputs to the calibration stage: the nine-seed noise floors, the τ ↔ weight curves and the trade figure all read ex-2.1.10; the latch-detector calibration reads ex-2.1.9; the across-point correlation adds ex-2.1.8."""


# --- The DAG -----------------------------------------------------------------


def prepare_corpus(grid: str, red_rate: float, per_slot_rate: float, n_examples: int, holdout_frac: float) -> dict:
    """Build the corpus, the eval sets, and the alignment probe set.

    Ex-2.1.10's prep step with the op1-keyed labeller dropped, since every condition and trial here runs the either-slot rule. The corpus is unchanged down to the seed, so these are the lines ex-2.1.10's cells trained on. The design constants arrive as arguments rather than being read off the module, so changing one re-runs this step instead of silently reusing a corpus built under the old value.
    """
    import numpy as np

    from sca import baselines
    from sca.colorcube import redness as colorcube_redness
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS, dump_example_sets, mix
    from mini.store import put

    levels = GRIDS[grid]
    palette = nc.grid_palette(levels)
    names = {v: k for k, v in palette.items()}
    corpus = nc.sample_corpus(n_examples, CORPUS_SEED, levels, holdout_frac)

    words = [w for ex in corpus for w in nc.as_words(ex)]
    tokenizer_config = TokenizerConfig(vocabulary=[*nc.SYNTAX, *palette])
    tokenizer = nc.WordTokenizer(tokenizer_config)
    tokens = np.asarray(tokenizer.encode_words(words), dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=sum(map(len, words)),
        sources=[DatasetMetadata(title=f"named-only color corpus ({grid})", fixes=[], total_chars=len(words))],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / grid)

    held = nc.holdout_test(levels, CORPUS_SEED, holdout_frac)
    seen = sorted({p for ex in corpus if (p := ex.pair) is not None})
    holdout = [p for p in nc.closed_pairs(levels) if held(*p)]
    opens = nc.open_pairs(levels)
    rng = np.random.default_rng(CORPUS_SEED + 1)
    pick = lambda pairs, r: [pairs[i] for i in r.permutation(len(pairs))[:N_EVAL]]  # noqa: E731
    sets = {
        "named_seen": [nc.make_example(a, b, names, rng) for a, b in pick(seen, rng)],
        "named_holdout": [nc.make_example(a, b, names, rng) for a, b in pick(holdout, rng)],
        "open": [nc.make_example(a, b, names, rng) for a, b in pick(opens, rng)],
    }

    # The alignment probe set: every color as op1, against each of its closed
    # partners. Exhaustive, so there is no sampling seed and every run is
    # measured on identical lines. The operands' colors ride along per line, for
    # the line-keyed statistics and the contrast's group weights.
    colors = list(palette.values())
    on_grid = set(levels)
    probe_lines, partners, op1_rgb, op2_rgb = [], [], [], []
    for c in colors:
        closed = [b for b in colors if all(v in on_grid for v in mix(c, b))]
        partners.append(len(closed))
        for b in closed:
            probe_lines.append([names[c], "+", names[b], "=", names[mix(c, b)], "\n"])
            op1_rgb.append(c)
            op2_rgb.append(b)
    assert len(set(partners)) == 1, f"probe set is ragged: {sorted(set(partners))}"
    assert partners[0] == N_PROBE, f"expected {N_PROBE} partners per color, got {partners[0]}"
    probe_tokens = np.array([tokenizer.encode_words(line) for line in probe_lines], dtype=np.int32)

    # The labeller's table, recomputed from the palette and checked against the
    # design constants the report draws its figures from: the per-operand rate by
    # token id. `weights` is the per-color affinity the per-color statistics score
    # with, unchanged from ex-2.1.6 on (the rate cancels in the normalization).
    rgb = np.array(list(palette.values()), dtype=float) / (N_LEVELS - 1)
    slot_affinity = colorcube_redness(rgb) ** 8 * per_slot_rate
    np.testing.assert_allclose(rgb, GRID_RGB, rtol=0, atol=1e-12)
    np.testing.assert_allclose(slot_affinity, p_slot(REDNESS), rtol=1e-12, atol=0)
    slot_p = np.zeros(tokenizer.vocab_size, dtype=np.float64)  # the tokenizer's, which counts the pad token
    for name, sp in zip(palette, slot_affinity, strict=True):
        slot_p[tokenizer.stoi[name]] = sp
    affinity = colorcube_redness(rgb) ** 8 * red_rate
    weights = affinity / affinity.sum()

    # Per probe line: each operand's redness, and P(labeled) under the either-slot
    # labeller — the raw rate, so consumers normalize over whatever subset they weight.
    r1 = colorcube_redness(np.asarray(op1_rgb, dtype=float) / (N_LEVELS - 1))
    r2 = colorcube_redness(np.asarray(op2_rgb, dtype=float) / (N_LEVELS - 1))
    p1, p2 = r1**8 * per_slot_rate, r2**8 * per_slot_rate
    line_p = p1 + p2 - p1 * p2

    # Model-free references for the behavior rows (they never read the model).
    vocab_rgb = np.array(colors)
    train_dists = baselines.distances(vocab_rgb, [ex.result for ex in sets["named_seen"]])
    blind = baselines.blind_index(train_dists)
    nulls = {}
    for set_name, exs in sets.items():
        dists = baselines.distances(vocab_rgb, [ex.result for ex in exs])
        shell = baselines.shell_mask(levels, vocab_rgb, [ex.result for ex in exs])
        nulls[set_name] = {
            "blind": baselines.blind_stats(dists, blind),
            "k2": baselines.k_nearest_stats(dists, 2),
            "chance_dist": float(dists.mean()),
            "floor_dist": float(dists.min(axis=1).mean()),
            "neighborhood_exact": baselines.neighborhood_exact_null(shell),
        }

    stats = {
        "n_colors": len(palette),
        "n_seen_distinct": len(seen),
        "n_holdout": len(holdout),
        "n_open": len(opens),
        "n_partners": partners[0],
        "total_tokens": int(len(tokens)),
        "eval_n": {k: len(v) for k, v in sets.items()},
        "line_label_rate": float(line_p.mean()),
        "nulls": nulls,
        "blind_index": blind,
    }
    return {
        "meta": meta,
        "stats": stats,
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.11-{grid}-evals.json"),
        "probes": put(
            _npz(
                probe_tokens=probe_tokens,
                slot_p=slot_p,
                weights=weights,
                line_p=line_p,
                r1=r1,
                r2=r2,
                redness=colorcube_redness(rgb),
            ),
            name=f"ex-2.1.11-{grid}-probes.npz",
        ),
    }


def _npz(**arrays) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _make_config(vocab_size: int, seed: int, epochs: int):
    """The d64-L4 config from ex-2.1.3, unchanged; only the schedule's length varies."""
    from sca.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            block_size=BLOCK,
            n_embd=64,
            n_head=8,
            n_head_dim=8,
            n_ff=256,
            n_layer=4,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=BATCH, oversample=16, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=scheduler_config(epochs),
        seed=seed,
    )


def schedules(
    *,
    lam: float,
    tau: float,
    epochs: int,
    anchor_shape: Shape = "min-jerk",
    anti_shape: Shape = "min-jerk",
    anti_peak_ratio: float = ANTI_PEAK_RATIO,
    anti_hold_ratio: float = ANTI_HOLD_RATIO,
    anti_anneal_end_frac: float = ANTI_ANNEAL_END_FRAC,
) -> tuple[dict, dict | None]:
    """The `AnchorSpec` and `AntiSpec` keyword dicts for one run.

    Both schedules travel as plain dicts of every keyframe, so a run's memo key carries the whole shape of its pull, its repulsion and its pooling — a flattened arm and a scheduled one are different keys for the same reason a different λ_a is. Keyframes are fractions of training resolved against *epochs*, so the half-length condition runs the same shape.
    """
    anchor = {
        "peak": lam,
        "warmup_epochs": epochs * WARMUP_FRAC,
        "anneal_start": epochs * ANNEAL_START_FRAC,
        "anneal_end": float(epochs),
        "floor": ANNEAL_FLOOR,
        "span": SPAN,
        "tau": tau,
        "shape": anchor_shape,
    }
    anti = (
        {
            "lam": lam,
            "peak_ratio": anti_peak_ratio,
            "hold_ratio": anti_hold_ratio,
            "anneal_end": epochs * anti_anneal_end_frac,
            "anchor_anneal_start": epochs * ANNEAL_START_FRAC,
            "anchor_anneal_end": float(epochs),
            "floor": ANNEAL_FLOOR,
            "shape": anti_shape,
        }
        if lam > 0
        else None
    )
    return anchor, anti


def condition_shapes(c: dict) -> tuple[Shape, Shape]:
    """The anchor and anti-subspace schedule shapes of an ablation condition.

    One home for the mapping, so the runs and the report's dose table read the conditions the same way.
    """
    interp: Shape = "linear" if c.get("interp") == "linear" else "min-jerk"
    return (
        "flat" if c.get("anchor_shape") == "flat" else interp,
        "flat" if c.get("anti_shape") == "flat" else interp,
    )


def ablation_cells(prep: dict) -> list[dict]:
    """One row per ablation run: the config, both schedules, and its labels."""
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    cells = []
    for c in ABLATION_CONDITIONS:
        anchor_shape, anti_shape = condition_shapes(c)
        ratio = c.get("anti_ratio", ANTI_HOLD_RATIO)
        anchor, anti = schedules(
            lam=c["lam"],
            tau=TAU_REF,
            epochs=c["epochs"],
            anchor_shape=anchor_shape,
            anti_shape=anti_shape,
            anti_peak_ratio=ratio if anti_shape == "flat" else ANTI_PEAK_RATIO,
            anti_hold_ratio=ratio,
        )
        for seed in range(N_SEEDS_ABLATION):
            config = _with_tokenizer(_make_config(align(tc.vocab_size, 64), seed, c["epochs"]), tc)
            cells.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "epochs": c["epochs"],
                    "condition": c["name"],
                    "label": f"{c['name']}-s{seed}",
                }
            )
    return cells


def survey_cells(trials: list[dict], decision: dict, prep: dict, seeds: dict[int, list[int]]) -> list[dict]:
    """One row per survey run: *seeds* maps a trial index to the seeds it runs at."""
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    epochs = decision["epochs"]
    cells = []
    for t in trials:
        flat = "anti_ratio" in t
        anchor, anti = schedules(
            lam=t["lam"],
            tau=t["tau"],
            epochs=epochs,
            anchor_shape=decision["anchor_shape"],
            anti_shape="flat" if flat else "min-jerk",
            anti_peak_ratio=t["anti_ratio"] if flat else t["anti_peak_ratio"],
            anti_hold_ratio=t["anti_ratio"] if flat else ANTI_HOLD_RATIO,
            anti_anneal_end_frac=ANTI_ANNEAL_END_FRAC if flat else t["anti_anneal_end_frac"],
        )
        for seed in seeds.get(t["trial"], []):
            config = _with_tokenizer(_make_config(align(tc.vocab_size, 64), seed, epochs), tc)
            cells.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "epochs": epochs,
                    "condition": f"t{t['trial']:02d}",
                    "label": f"t{t['trial']:02d}-s{seed}",
                }
            )
    return cells


def _with_tokenizer(config, tokenizer_config):
    config.tokenizer = tokenizer_config.model_copy()
    return config


def train_one(config, anchor: dict, anti: dict | None, grid: str, traj_stride: int, probes, label: str) -> dict:
    """Train one run under the either-slot labeller, recording the m_line trajectory.

    Ex-2.1.10's step with the labeller fixed and the schedules made free: both specs arrive as keyword dicts, so a flat arm and a scheduled one differ only in what this function is handed.
    """
    import numpy as np

    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, slot_p, weights, line_p = z["probe_tokens"], z["slot_p"], z["weights"], z["line_p"]

    # One line per color for the trajectory. At op1 the partner cannot matter; at
    # the other span roles it can, so the trajectory reads one fixed partner per
    # color where the endpoint eval averages all 27 — cheap and consistent within
    # a run, which is what retention compares.
    stride = len(probe_tokens) // len(weights)
    first_of_color = probe_tokens[::stride]
    line_w = line_p[::stride] / line_p[::stride].sum()

    _, metrics, traj = train_anchored(
        config,
        get_data_dir() / "corpora" / grid,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti) if anti is not None else None,
        label_p=LabelSpec(p=slot_p, keying="either", pull="span"),
        probe_tokens=first_of_color,
        probe_weights=weights,
        probe_line_w=line_w,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    keep = ("epoch", "m_line", "m_op1", "m_span", "alpha_op1", "val_loss", "weight", "anti_weight")
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: traj[k].tolist() for k in keep if k in traj},
        "checkpoint": put(workdir / "model", name=f"ex-2.1.11-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, probes, grid: str, tau: float, condition: str, label: str) -> dict:
    """The slim eval: behavior, the alignment map, the weight profiles, and retention.

    Every statistic a decision rule or a survey constraint reads, and nothing else — no readability, geometry or leakage pass. Those belong to the experiment that confirms one operating point properly rather than eighty thinly.
    """
    import numpy as np

    from sca.anchoring import alignment
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS, load_example_sets
    from sca.data.named_colors import GRIDS, WordTokenizer, grid_palette
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = WordTokenizer(config.tokenizer)

    palette = grid_palette(GRIDS[grid])
    names = list(palette)
    name_idx = {n: i for i, n in enumerate(names)}
    color_ids = np.array([tokenizer.stoi[n] for n in names])
    vocab_rgb = np.array(list(palette.values()), dtype=np.float32) / (N_LEVELS - 1)

    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, weights, line_p, r1, r2 = z["probe_tokens"], z["weights"], z["line_p"], z["r1"], z["r2"]
    line_w = line_p / line_p.sum()
    g1, g2 = group_weights(r1, r2)

    # --- Behavior: one teacher-forced pass per eval set, read at the pre-answer position.
    sets, arrays = {}, {}
    for set_name, exs in load_example_sets(get(evals, workdir / "evals.json").read_bytes()).items():
        logp = _answer_logprobs(model, tokenizer, exs)[:, color_ids]
        result = np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)
        dists = np.linalg.norm(vocab_rgb[None] - result[:, None], axis=2)  # (N, V)
        guess = logp.argmax(axis=1)
        guess_dist = dists[np.arange(len(exs)), guess]
        floor = dists.min(axis=1)
        stats: dict = {
            "n": len(exs),
            "guess_dist": float(guess_dist.mean()),
            "floor_dist": float(floor.mean()),
            "nearest_acc": float((guess_dist <= floor + 1e-9).mean()),
        }
        if exs[0].answer:  # closed sets: the true answer is a vocabulary name
            true_idx = np.array([name_idx[ex.answer] for ex in exs])
            stats["accuracy"] = float((guess == true_idx).mean())
            stats["nll"] = float(-logp[np.arange(len(exs)), true_idx].mean())
        sets[set_name] = stats

    # --- Alignment: cos(h, e₀) per slice × line × position, contracted per color
    #     for the grading and containment statistics and per line for m_line.
    n_partners = len(probe_tokens) // len(weights)
    cos = alignment(model, probe_tokens)  # (L1, C*P, T)
    alpha = cos.reshape(cos.shape[0], len(weights), n_partners, cos.shape[2]).mean(axis=2)  # (L1, C, T)
    arrays["alpha"] = alpha.astype(np.float32)

    # --- The weight profiles: what the pull chose, at this run's τ. `pi` is
    #     ex-2.1.10's per-color profile, which is what the latch veto is calibrated
    #     on; the group profiles are the per-line contrast's two halves.
    x = 1.0 - cos[:, :, :SPAN]
    w_line = softmin_weights(x, tau)  # (L1, C*P, SPAN)
    w_color = w_line.reshape(cos.shape[0], len(weights), n_partners, SPAN).mean(axis=2)
    pi = np.einsum("lct,c->lt", w_color, weights)
    w_group = np.stack([np.einsum("lnt,n->lt", w_line, g) for g in (g1, g2)])  # (2, L1, SPAN)
    arrays["pi"] = pi.astype(np.float32)
    arrays["w_group"] = w_group.astype(np.float32)

    traj = np.asarray(trained["traj"]["m_line"], dtype=float)
    peak = float(np.maximum.accumulate(traj).max())
    return {
        "label": label,
        "condition": condition,
        "tau": tau,
        "val_loss": trained["val_loss"],
        "train_loss": trained["train_loss"],
        "traj": trained["traj"],
        "sets": sets,
        "holdout_em": sets["named_holdout"]["accuracy"],
        "m_line": line_margin(cos, line_w),
        "m_span": pooled_margin(alpha, weights),
        "alpha_op1": float(alpha[:, :, 0].mean()),
        "r2_sim": r2_sim(alpha[:, :, 0].mean(axis=0)),
        "contrast": float(w_group[1, 1:, 2].mean() - w_group[0, 1:, 2].mean()),
        "latch_pi": float(max(pi[1:, 1].mean(), pi[1:, 3].mean())),
        "m_line_peak": peak,
        "retention": float(traj[-1] / peak),
        "pi": pi.tolist(),
        "arrays": put(_npz(**arrays), name=f"ex-2.1.11-{label}-arrays.npz"),
    }


def _answer_logprobs(model, tokenizer, exs) -> "np.ndarray":
    """Full-vocabulary log-probabilities at the pre-answer position, per example."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    forward = eqx.filter_jit(model.__call__)
    seq = np.array([tokenizer.encode_words(ex.prompt.split()) for ex in exs])
    out = [
        np.asarray(jax.nn.log_softmax(forward(jnp.asarray(seq[i : i + 256]))[:, -1], axis=-1))
        for i in range(0, len(seq), 256)
    ]
    return np.concatenate(out)


# --- Statistics shared with the report ---------------------------------------


def line_margin(alpha_lines: np.ndarray, line_w: np.ndarray) -> float:
    """m_line: per-line selectivity, ex-2.1.10's scored statistic, unchanged.

    Per slice: the line-weighted mean alignment at each span role minus the unweighted mean, keeping the largest role; then the mean over slices.
    """
    m = np.einsum("n,lnt->lt", line_w, alpha_lines) - alpha_lines.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def pooled_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """m_span: the per-color margin maxed over the span, kept for continuity with ex-2.1.9."""
    m = np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def r2_sim(alpha_op1: np.ndarray) -> float:
    """The grading statistic: r² between the per-color op1 response and SIM_TARGET."""
    return float(np.corrcoef(alpha_op1, SIM_TARGET)[0, 1] ** 2)


def group_weights(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-line weights for the two localization groups the contrast compares.

    G1 weights each probe line by P(op1 drew and op2 didn't); G2 by the reverse. Ex-2.1.10's construction, unchanged: the labeller's own one-slot-only probabilities, so no redness threshold has to be invented.
    """
    p1, p2 = np.asarray(p_slot(r1)), np.asarray(p_slot(r2))
    g1 = p1 * (1.0 - p2)
    g2 = (1.0 - p1) * p2
    return g1 / g1.sum(), g2 / g2.sum()


# --- Stage 3: feasibility and promotion --------------------------------------


def trial_feasibility(runs: list[dict], control_em: float, r2_ref: float) -> dict:
    """Every survey constraint, read on the seed means of one trial's *runs*.

    The latch veto stays per-run (a trial with any latched run is infeasible however its means read); the other five are read on seed means, as `PROMOTION` specifies. With one run in round 1 the two readings coincide.
    """
    mean = lambda k: float(np.mean([r[k] for r in runs]))  # noqa: E731
    peak, retention = mean("m_line_peak"), mean("retention")
    checks = {
        "task": abs(mean("holdout_em") - control_em) <= TASK_GATE,
        "containment": mean("alpha_op1") <= MEAN_ALIGN_GATE,
        "retention": peak < RETENTION_FLOOR or retention >= RETENTION_GATE,
        "grading": mean("r2_sim") >= r2_ref - GRADE_R2_DROP,
        "contrast": mean("contrast") >= CONTRAST_MIN,
        "latch": all(r["latch_pi"] <= LATCH_PI for r in runs),
    }
    return {
        "n_seeds": len(runs),
        "checks": checks,
        "feasible": all(checks.values()),
        **{k: mean(k) for k in ("m_line", "alpha_op1", "retention", "r2_sim", "contrast", "holdout_em", "latch_pi")},
    }


def promote(scored: dict[int, dict]) -> list[int]:
    """The trials that earn more seeds: the N_PROMOTE feasible ones with the highest m_line."""
    feasible = [t for t, s in scored.items() if s["feasible"]]
    return sorted(feasible, key=lambda t: -scored[t]["m_line"])[:N_PROMOTE]


# --- Publishing --------------------------------------------------------------


def publish_ablations(results: list[dict], stats: dict, evals, probes) -> dict:
    """Publish the ablation stage: metrics (JSON), stacked per-run arrays (npz), the eval and probe sets."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    summary = ablation_summary(results)
    metrics = {
        "cells": [{k: v for k, v in r.items() if k != "arrays"} for r in results],
        "corpus_stats": stats,
        "summary": summary,
        "decision": decide(summary),
        "conditions": ABLATION_CONDITIONS,
        "design": {
            "n_seeds": N_SEEDS_ABLATION,
            "scoring_lambda": SCORING_LAMBDA,
            "tau_ref": TAU_REF,
            "span": SPAN,
            "epochs": {"long": EPOCHS, "short": EPOCHS_SHORT},
            "noise_run": NOISE_RUN,
            "decision_stats": DECISION_STATS,
            "stat_direction": STAT_DIRECTION,
            "gates": {
                "task": TASK_GATE,
                "mean_align": MEAN_ALIGN_GATE,
                "retention_floor": RETENTION_FLOOR,
                "retention": RETENTION_GATE,
                "grade_r2_drop": GRADE_R2_DROP,
                "contrast": CONTRAST_MIN,
                "latch": LATCH_PI,
            },
        },
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.11-metrics.json"))
    set_ref(EVALS_REF, evals)
    set_ref(PROBE_REF, probes)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.11-arrays.npz"))
    return {"n_runs": len(results), "decision": metrics["decision"]}


def publish_survey(results: list[dict], trials: list[dict], decision: dict, scored: dict, promoted: list[int]) -> dict:
    """Publish every trial, including the ones that went nowhere.

    A search's map is only worth reading if it is complete, so failed and infeasible trials are published as such rather than dropped.
    """
    import json

    from mini.store import put, set_ref

    survey = {
        "cells": [{k: v for k, v in r.items() if k not in ("arrays", "traj")} for r in results],
        "trials": trials,
        "decision": decision,
        "scored": {str(t): s for t, s in scored.items()},
        "promoted": promoted,
        "design": {
            "space": {k: list(v) for k, v in survey_space(decision).items()},
            "sobol_seed": SOBOL_SEED,
            "n_trials": N_TRIALS,
            "n_promote": N_PROMOTE,
            "seeds_promote": SEEDS_PROMOTE,
            "objective": OBJECTIVE,
            "promotion": PROMOTION,
            "stopping_rule": STOPPING_RULE,
        },
    }
    set_ref(SURVEY_REF, put(json.dumps(survey, indent=2).encode(), name="ex-2.1.11-survey.json"))
    ranked = sorted((t for t in scored if scored[t]["feasible"]), key=lambda t: -scored[t]["m_line"])
    return {
        "n_runs": len(results),
        "n_feasible": len(ranked),
        "proposed": {**next((tr for tr in trials if tr["trial"] == ranked[0]), {}), **scored[ranked[0]]}
        if ranked
        else None,
    }


# --- Orchestration -----------------------------------------------------------


def _stages() -> set[str]:
    """Which stages this wake may launch — `EX2111_STAGES`, both by default.

    The survey's space is decided by the ablations, so the two are one DAG and one tick would roll from the first into the second. The gate lets a wake settle the ablations and stop there for review instead.
    """
    import os

    return {s.strip() for s in os.environ.get("EX2111_STAGES", "ablations,survey").split(",")}


def _run_cells(ctx: Ctx, cells: list[dict], prep: dict, allow_partial: bool = False) -> list[dict]:
    """Train then evaluate each cell, index-aligned with *cells*."""
    n = len(cells)
    trained = ctx.map(
        train_one,
        [c["config"] for c in cells],
        [c["anchor"] for c in cells],
        [c["anti"] for c in cells],
        [GRID] * n,
        [TRAJ_STRIDE] * n,
        [prep["probes"]] * n,
        [c["label"] for c in cells],
        role="train",
        allow_partial=allow_partial,
    )
    # A cell that diverged or failed comes back MISSING under `allow_partial`;
    # it drops out here and is published as a failed trial rather than resampled.
    ok = [(c, t) for c, t in zip(cells, trained, strict=True) if isinstance(t, dict)]
    evaled = ctx.map(
        eval_one,
        [t for _, t in ok],
        [prep["evals"]] * len(ok),
        [prep["probes"]] * len(ok),
        [GRID] * len(ok),
        [c["anchor"]["tau"] for c, _ in ok],
        [c["condition"] for c, _ in ok],
        [c["label"] for c, _ in ok],
        role="eval",
        allow_partial=allow_partial,
    )
    return [e for e in evaled if isinstance(e, dict)]


def main(ctx: Ctx) -> dict:
    prep = ctx.run(prepare_corpus, GRID, RED_RATE, PER_SLOT_RATE, N_EXAMPLES, HOLDOUT_FRAC, role="prep")

    # --- Stage 2: the ablations, and the decision they fix.
    ablations = _run_cells(ctx, ablation_cells(prep), prep)
    summary = ablation_summary(ablations)
    decision = decide(summary)
    published = ctx.run(publish_ablations, ablations, prep["stats"], prep["evals"], prep["probes"], role="prep")
    if "survey" not in _stages():
        return {"stage": "ablations", **published}

    # --- Stage 3: the survey, on the space the decision named.
    trials = sobol_trials(survey_space(decision))
    control_em = summary[CONTROL_OF[decision["epochs"]]]["holdout_em"]
    r2_ref = summary["ref"]["r2_sim"]

    round1 = _run_cells(ctx, survey_cells(trials, decision, prep, {t["trial"]: [0] for t in trials}), prep, True)
    scored = {
        t["trial"]: trial_feasibility(runs, control_em, r2_ref)
        for t in trials
        if (runs := [r for r in round1 if r["condition"] == f"t{t['trial']:02d}"])
    }
    promoted = promote(scored)

    extra = {t: list(range(1, SEEDS_PROMOTE)) for t in promoted}
    round2 = _run_cells(ctx, survey_cells(trials, decision, prep, extra), prep, True) if promoted else []
    runs = round1 + round2
    scored |= {
        t: trial_feasibility([r for r in runs if r["condition"] == f"t{t:02d}"], control_em, r2_ref) for t in promoted
    }
    return {
        "stage": "survey",
        **ctx.run(publish_survey, runs, trials, decision, scored, promoted, role="prep"),
        "ablations": published["decision"],
    }


experiment = Experiment(
    name="ex-2.1.11",
    main=main,
    deps=["ex-2.1.10"],
    roles={
        # Corpus sampling is a plain-numpy loop over 100k lines; the probe set is trivial beside it.
        "prep": dict(cpu=2, timeout=900),
        # ~3.3k steps of 64×64 tokens (half that for the short arms), as ex-2.1.10.
        # The watchdog is sized for the gap *after* the last step: the checkpoint
        # upload emits no step progress and took over 300s in ex-2.1.7 when eight
        # containers pushed to the store at once (todo-eng).
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        # Three teacher-forced eval sets and one probe pass over 5832 lines. No
        # ridge probes here, so this is much shorter than ex-2.1.10's eval role.
        "eval": dict(gpu="L4", timeout=1200),
    },
)
