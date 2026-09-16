"""
The grammar handover: ex-2.2.3's recipe on table A+, with the stochastic corpus, the whole-line
labeller, and the untied readout.

Preregistered. This module is the design: the table, the conditions, the probe rule, and the gates,
each with its wording in a docstring, so the report quotes the same numbers it will be scored on.
The task DAG lands once the draft is agreed; until then the module is constants only.

The scouting round (ex-2.2.4) and the pilots (ex-2.2.5 to 2.2.8) each proposed one change to the
grammar or the recipe. This experiment adopts them together, at fresh seeds, and asks whether the
anchoring still lands, whether the removal reads are cleaner, and what the two changes whose pilots
were inconclusive (the labeller, the readout) each do. One reference condition changes each of
those back, so each read is a two-condition comparison on the new grammar.

    bin/mini run docs/m2/ex-2.2.9/experiment.py --app modal --max-containers 8 --budget 3h
    bin/mini status ex-2.2.9
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import Literal

from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, Op

DESIGN_ONLY = True
"""Constants only, until the draft is agreed: the e2e test skips this module, and the report renders
its method from it."""


def _load_ex223():
    """Ex-2.2.3's module, loaded by path and left out of `sys.modules`, as `mini.load_experiment` does,
    so its task functions still cloudpickle by value for a remote worker once the DAG lands.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.3" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex223", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex223 = _load_ex223()

# Bound by name so a task body never references the module object itself.
Condition = ex223.Condition
steps_per_epoch = ex223.steps_per_epoch

METRICS_REF = "reports/m2/ex-2.2.9/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.9/arrays"
TRAJ_REF = "reports/m2/ex-2.2.9/trajectories"
PROBE_REF = "reports/m2/ex-2.2.9/probes"
CHECKPOINT_REF = "reports/m2/ex-2.2.9/checkpoints/{label}"

EX223_METRICS_REF = ex223.METRICS_REF
EX223_REFERENCE = "recipe-short"
"""The reference: production's adopted point on the six-op grammar, twenty seeds with the addendum. Every
placement statistic here is read beside its value there, and the removal gates are the ones it was read on."""

EX227_METRICS_REF = "reports/m2/ex-2.2.7/metrics"
EX227_CEILING = "rows-clean"
"""Ex-2.2.7's condition with the syntax embeddings hard-zeroed every step: the in-grammar ceiling for how clean
the syntax embeddings can come, which the untied conditions are read against (H4)."""

# --- Table A+ ------------------------------------------------------------------------------

KEPT = ("mix", "screen", "multiply", "lighten", "darken")
"""Five of ex-2.2.3's six ops. `add` is dropped: a fifth of its pairs go to white, and it is the least
sensitive op to its red operand by either of ex-2.2.4's rules."""

ADDED = ("difference", "exclusion", "hsvmix")
"""Three commutative ops whose answers spread through the cube and depend on both operands."""

ORDER_SENSITIVE = ("hue-hsv", "sat-hsv", "value-hsv")
"""The marked subset: each takes one attribute of op2 at the rest of op1, so these are the first ops in the
grammar whose answer depends on which operand is which. Their reads are reported as a subset, beside the
table, so that anything of their own can be set aside without touching the rest."""

OP_NAMES: tuple[str, ...] = (*KEPT, *ADDED, *ORDER_SENSITIVE)
TABLE: tuple[Op, ...] = tuple(OP_BY_NAME[n] if n in OP_BY_NAME else CANDIDATE_BY_NAME[n] for n in OP_NAMES)
"""Table A+ of ex-2.2.4, in the order the report prints it. Every op is total on the grid."""
N_OPS = len(TABLE)
assert N_OPS == 11

PRIMARY_OP = "mix"
"""Every gated statistic is read on `mix`'s probe lines, as in ex-2.2.3, so the gates mean what they meant."""
SECONDARY_OP = "hsvmix"
"""Reported beside `mix` on every gated read, so a later experiment can make it the reference op if it
behaves. The switch, if it comes, is after the handover; what it would change is set out in the report's
method (the reference op): a stronger removal read, on a probe set that rounds on nearly every line."""

ROUNDING = "stochastic"
"""An answer between grid levels rounds to the upper level with probability equal to how far up it sits,
drawn once per line at corpus build (`sca.data.ops.Rounding`). The answer of a line is then a distribution
over the candidate colors, and every exact-match read becomes an expectation over it (ex-2.2.5)."""

# --- The corpus ------------------------------------------------------------------------------

N_LINES = ex223.N_LINES
"""D2.1's corpus size, unchanged: 100k lines. Ops are drawn uniformly, so each has about a tenth of the lines,
close to half of what each op had at six ops. Lines per op is the confound E4 of ex-2.2.3 named; the `wide`
condition below reads it. Coverage is rendered in the report's method."""

CORPUS_SEED = ex223.CORPUS_SEED
HOLDOUT_FRAC = ex223.HOLDOUT_FRAC
"""Of the distinct unordered pairs of each op, held out. A held-out pair is out in both orders, for every op,
so the held-out share of an op's lines is the same 20% whether or not the op reads operand order."""

EPOCHS = ex223.EPOCHS_SHORT
"""The adopted point's length: 50 epochs of 100k lines, 1,650 steps."""

N_LINES_WIDE = round(N_LINES * N_OPS / len(ex223.OP_NAMES))
EPOCHS_WIDE = round(EPOCHS * N_LINES / N_LINES_WIDE)
"""The `wide` condition holds lines per op at ex-2.2.3's count (about 16.7k), so its corpus is eleven sixths
the size, and it runs for fewer epochs so that the step count stays at the recipe's. What differs from the
handover condition is then how many distinct lines of each op the model sees, and how often it sees each."""
assert abs(EPOCHS_WIDE * steps_per_epoch(N_LINES_WIDE) - EPOCHS * steps_per_epoch(N_LINES)) <= 60

# --- The recipe --------------------------------------------------------------------------------

LAM = ex223.SCORING_LAMBDA
TAU = ex223.TAU_REF
ANTI_PEAK_RATIO = ex223.ANTI_PEAK_RATIO_REF
ANTI_ANNEAL_END_FRAC = ex223.ANTI_ANNEAL_END_FRAC_REF
"""The adopted point of ex-2.2.3 (`recipe-short`): λ_a = 0.1, annealed over the last tenth of training to a
0.1 floor as ex-2.1.10 did; τ = 0.1; the anti-subspace weight from 2.5× the anchor weight to 0.3× by 90% of
training. Nothing here retunes it."""

ANCHOR_AXIS = ex223.ANCHOR_AXIS
"""*Red* is anchored to e₁ at every slice, the embedding included (ex-2.2.7 settled the Prep C question)."""

Keying = Literal["either", "line"]
PROMPT_SPAN = ex223.SPAN
WHOLE_SPAN = 6
"""The two labellers. `either`, prompt span (ex-2.2.3's): each operand draws at redness⁸ × 0.04, and the pull
covers op1, op, op2, and `=`. `line`, whole span (ex-2.2.6's): the answer draws at its redness rate too, and
the pull covers the answer and the newline as well."""
PER_SLOT_RATE = ex223.PER_SLOT_RATE

TIE_EMBEDDINGS = False
"""The readout is a table of its own, initialised as a copy of the embedding (ex-2.2.7's `untied`). The tied
condition sets this back."""

# --- Conditions ---------------------------------------------------------------------------------

Role = Literal["candidate", "reference", "exploratory"]


@dataclass(frozen=True)
class Cond:
    """One training condition. Everything not named here is the adopted point on table A+ with stochastic
    rounding.
    """

    name: str
    seeds: int
    title: str
    role: Role
    lam: float = LAM
    keying: Keying = "line"
    span: int = WHOLE_SPAN
    tie: bool = TIE_EMBEDDINGS
    n_lines: int = N_LINES
    epochs: int = EPOCHS

    @property
    def condition(self) -> Condition:
        return Condition(
            self.name, self.seeds, self.title, lam=self.lam, epochs=self.epochs, ops=OP_NAMES, n_lines=self.n_lines
        )

    @property
    def steps(self) -> int:
        return self.epochs * steps_per_epoch(self.n_lines)

    @property
    def lines_per_op(self) -> int:
        return self.n_lines // N_OPS


CONTROL = Cond("control", 5, "un-anchored", "reference", lam=0.0)
"""Nothing placed on the axis, on the new grammar with the untied readout: the task reference for H1, and the
calibration reference for the stochastic reads."""

HANDOVER = Cond("handover", 20, "the recipe, untied readout, whole-line labeller", "candidate")
"""Every proposal adopted at once. The one candidate for the grammar of record, at the reference's seed count.
The gates (H1 to H3) are read on this condition alone."""

SLOT = Cond(
    "handover-slot",
    20,
    "as handover, with ex-2.2.3's either-slot labeller",
    "reference",
    keying="either",
    span=PROMPT_SPAN,
)
"""The labeller changed back. Beside `handover` this is the selectivity check ex-2.2.7 asked for (H5): what the
whole-line labeller costs, if anything. Twenty seeds, so the comparison has the same resolution as the gates.
It is not a fallback: the either-slot labeller needs to know which tokens are the operands, and a method that
needs that is not one we can carry to M3."""

TIED = Cond("handover-tied", 9, "as handover, with the tied readout", "reference", tie=True)
"""The readout changed back. Beside `handover` this reads what untying does on the new grammar (H4): the
component on the syntax embeddings, and the non-red cost of the full-position projection."""

WIDE = Cond(
    "handover-wide",
    5,
    "as handover, lines per op held at ex-2.2.3's count",
    "exploratory",
    n_lines=N_LINES_WIDE,
    epochs=EPOCHS_WIDE,
)
"""Exploratory: the lines-per-op confound of E4 (ex-2.2.3), read on the operand-cube probe scan and on the
task. Not gated."""

CONDS: tuple[Cond, ...] = (CONTROL, HANDOVER, SLOT, TIED, WIDE)
ANCHORED: tuple[Cond, ...] = tuple(c for c in CONDS if c.lam > 0)
N_RUNS = sum(c.seeds for c in CONDS)
assert N_RUNS == 59
assert len({c.name for c in CONDS}) == len(CONDS)

# --- The probe set -------------------------------------------------------------------------------

N_PROBE = ex223.N_PROBE
PROBE_SEED = ex223.PROBE_SEED
"""As ex-2.2.3: for `mix`, its 27 on-grid partners per color (D2.1's 5,832 lines with the op word changed);
for every other op, 27 partners per color drawn once with a fixed seed and shared across ops."""

PROBE_BOTH_SLOTS = ORDER_SENSITIVE
"""For the order-sensitive subset the probe set also walks every color as op2, against the same partners, so
each of those ops has 11,664 probe lines and its reads can be split by which slot the red operand is in."""

RED_DOSE = ex223.RED_DOSE
NONRED_DOSE = ex223.NONRED_DOSE
"""Dose is the redness of the redder operand. Red lines have dose ≥ 0.8; non-red lines have dose ≤ 0.2. As
ex-2.2.3."""

# REVIEW: the author asked whether setting R to one grid level rather than zero would keep the hue
# and so count the red-op2 lines under `hue-hsv` and `sat-hsv`. It does not: a one-level R is a
# dark red at full saturation, so `hue-hsv` moves as little and `sat-hsv` stops moving. The method
# says so with the counts (`eps_check` in the report); the rule stays at zero.
FAR_MOVE = 0.4
"""The removal lines: red lines on which zeroing the red operand's R moves the true answer by at least 0.4
(distance in the unit cube, ex-2.2.4's *to-zero* read). On `mix` that is every red line, since the move is
half the red operand's R; on `lighten` it drops the lines whose partner is about as red. The count per op is
rendered in the method."""

# REVIEW: the draft set the red-answer lines aside from the selectivity read. The author's review
# asked for the gate to cover them too, with the separate count kept. `mix` has none, so the gated
# read is the same either way; on the ops that have them the deficit is reported both ways. Verify:
# the H3 selectivity read and the glossary say "in the deficit and counted on their own".
RED_ANSWER_LINES_BOTH_WAYS = True
"""A non-red line whose true answer is red (white minus cyan is red, under `difference`) stays in the non-red
deficit and is also counted on its own, and the deficit is reported with and without these lines. Projecting
the axis out at `=` removes *red* from the state that has to produce the answer, so a miss there is removal
on the output side; the separate count says how much of a deficit is that. `mix` has no such lines, so the
gated read is unchanged; the order-sensitive ops have a few dozen each (counts in the method)."""

# --- Gates: the task (H1) -----------------------------------------------------------------------

CALIBRATION_FLOOR = 0.05
"""Before the freeze: the one control seed has learned the grammar when its expected exact match on every kept
and added op is within 0.05 of the ceiling the drawn answers allow on that op. The order-sensitive subset is
recorded and does not block."""

TASK_GATE = ex223.TASK_GATE
TASK_PARTIAL = ex223.TASK_PARTIAL
"""H1: for each op, `handover`'s seed-mean expected exact match on the held-out lines is within 0.02 of the
control's. Partial: every op within 0.05. A comparison that misses by less than a band is reported as
unresolved rather than as a miss. The reference conditions are read on the same table without a gate."""

# --- Gates: placement (H2) --------------------------------------------------------------------------

REF_M_LINE = ex223.REF_M_LINE
MARGIN_RATIO = ex223.MARGIN_RATIO
MARGIN_PARTIAL = ex223.MARGIN_PARTIAL
"""H2 (margin): `handover`'s seed-mean m_line on the `mix` lines is at least 80% of ex-2.1.10's 0.4202, the
same bar ex-2.2.3 set; partial from 60%. Read under the condition's own labeller, as ex-2.2.6 did."""

REF_R2_SIM = ex223.REF_R2_SIM
GRADE_R2_RATIO = 0.9
"""H2 (grading): grading r² at least 90% of ex-2.1.11's reference value. Ex-2.2.3 asked for no more than 0.10
below it, which on 0.782 is 87%; the ratio is a shade tighter and reads like the margin gate."""

MEAN_ALIGN_REF = ex223.MEAN_ALIGN_GATE
"""H2 (containment): ᾱ, the mean alignment over all 216 colors at op1 on the `mix` lines. Every experiment
since ex-2.1.8 gated it at 0.1; here it is a prediction, read beside H3's selectivity. Ex-2.2.7 read 0.13 on
`untied` and 0.23 on `untied-line` against 0.08 on the reference, at nine seeds, with the pull on the right
operand (lead, contrast, and latch all at the reference's values), so we expect `handover` above 0.1. What
the rise costs, if anything, is what H3 measures; a gate we expect to miss on a condition we would still adopt
is not a gate."""

RETENTION_FLOOR = ex223.RETENTION_FLOOR
RETENTION_GATE = ex223.RETENTION_GATE
LEAD_GATE = ex223.LEAD_GATE
CONTRAST_GATE = ex223.CONTRAST_GATE
CONTRAST_PARTIAL = ex223.CONTRAST_PARTIAL
LATCH_PI = ex223.LATCH_PI
"""H2 (retention, concentration, attribution, latch): as ex-2.2.3 H2. Retention: every run whose peak m_line
reaches 0.2 ends, after the anchor weight's anneal, at 0.8 of that peak. Lead: the red group's leading softmin
weight at the embedding is at least 0.4. Contrast: the between-group contrast in op2 weight, mean over the
four post-attention slices, at least 0.2, partial from 0.1. Latch: no run has a non-red-group softmin weight
above 0.5 at op1; ex-2.2.3 held this at twenty seeds."""

# --- Gates: removal and selectivity (H3) --------------------------------------------------------

OPERATORS = ("projection", "operands", "shaped-a0.4-p0")
"""The operators ex-2.2.8 proposed. `projection`: the axis projected out at every slice and position, at full
strength; the gated operator. `operands`: the same at the two operand positions only; the selective
reference. `shaped-a0.4-p0`: a thresholded projection that leaves states below alignment 0.4 alone; the
syntax-free candidate, reported without a gate."""
SHAPED = dict(a=0.4, b=1.0, p=0.0)

RED_KEPT_GATE = ex223.RED_ACC_GATE
"""H3 (removal): under `projection`, on the removal lines of every op, the share of the clean expected exact
match the model keeps is at most 20%, seed mean. This is ex-2.2.3's red-accuracy gate read as a ratio, since
under stochastic rounding the clean value sits below 1 on the ops that round (ex-2.2.5)."""

NONRED_DEFICIT_GATE = ex223.NONRED_DEFICIT_GATE
NONRED_DEFICIT_PARTIAL = ex223.NONRED_DEFICIT_PARTIAL
"""H3 (selectivity): under `projection`, the seed-mean drop in expected exact match on the non-red `mix` lines
is at most 0.05, partial to 0.10. Reported on every op beside it."""

TAIL = 0.07
"""H5: a seed loses selectivity when its non-red `mix` deficit under `projection` is above 0.07, the level
ex-2.2.7 read its tail at (three of nine `untied-line` seeds; one of twenty reference seeds)."""

RESOLUTION_SD = ex223.RESOLUTION_SD
"""A difference between two seed means smaller than 2σ√(1/n_a + 1/n_b), with σ the per-run spread, is reported
as unresolved, as ex-2.2.3 and ex-2.2.8 report it. The seed counts enter through n_a and n_b, so a twenty-seed
condition read against a nine-seed one has a wider band than two twenty-seed conditions."""

NOISE_RUN = ex223.NOISE_RUN
"""Per-run σ of each placement statistic, frozen at the reference. The bands in H2 use these; the handover
re-measures them on the new grammar and the report prints both, as ex-2.2.3's E5 does."""

# REVIEW: named a σ source for every banded read. H4 and H5 both turn on "more than a band" for
# statistics that are not in NOISE_RUN, and without a source the person running it picks one.
# Neither carries a gate now (below), so both take the σ named here and report it. Verify: ex-2.2.8
# tabulates the per-run spread of the non-red deficit under `projection` per op at the reference's
# twenty seeds.
DEFICIT_NOISE = "ex-2.2.8, `projection` on `recipe-short`, twenty seeds, per op"
"""Where σ for the non-red deficit band comes from (H3's reporting and H5): the per-run spread ex-2.2.8
measured at the reference, on `mix`. Frozen, so the candidate's own spread does not move its verdict."""

COMPONENT_NOISE = "pooled within-condition, over the two conditions compared"
"""Where σ for the syntax-embedding component band comes from (H4): ex-2.2.7 reported seed means and ranges
rather than a per-run spread, so the band is built from the two conditions' own seeds and the σ is reported
beside it. H4 carries no gate, so nothing is adopted on this band."""

# --- The decision rule ---------------------------------------------------------------------------

# REVIEW: the first draft chose between two candidates, `handover` and `handover-slot`, with H5 as the
# tie-break, and made `handover-slot` the fallback if the whole-line labeller failed. The author's
# review took the fallback out: the either-slot labeller needs the operand positions, which M3 will
# not have, so it cannot be a grammar we carry forward. `handover-slot` is now a reference that
# attributes a selectivity cost to the labeller, and the rule names one candidate. Verify: nothing
# in the report adopts `handover-slot`.
DECISION = f"""\
The handover is adopted, and `{HANDOVER.name}` becomes the grammar and recipe of record for the anchored-op \
experiments, when it clears H1, H2 (margin, grading, and contrast in full), and H3 in full; every other partial \
band is a reporting level. Otherwise it is not adopted, and the report says which gate was missed and what the reference \
conditions say about which change is responsible, because that sets what the next round tries: the labeller if `{SLOT.name}` clears what `{HANDOVER.name}` missed, the readout if \
`{TIED.name}` does, and the table or the corpus if none of them does."""
