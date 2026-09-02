"""
Experiment 2.2.1: suppress *red* on the D2.1 checkpoints.

The first intervention on an anchored transformer. No training: every run is a
published checkpoint of ex-2.1.10 — the nine-seed primary with *red* anchored
on e₁, and its three-seed un-anchored control — scored through the eval
contract (`sca.intervention`) on the 5,832-line probe set the D2.1 experiments
share. The operator is axis projection: remove the e₁ component at every slice
and position, re-project onto the sphere.

Design constants only, for now: the report imports them so the prose and the
gates cannot drift apart. The DAG lands with the run, at which point the
`DESIGN_ONLY` marker below goes and `main(ctx)` takes its place.

    bin/mini run docs/m2/ex-2.2.1/experiment.py --watch   # once the DAG exists
"""

from __future__ import annotations

from dataclasses import dataclass

DESIGN_ONLY = True

#: Result refs the report will read.
METRICS_REF = "reports/m2/ex-2.2.1/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.1/arrays"


@dataclass(frozen=True)
class Condition:
    """A set of published runs, and where to find them."""

    exp: str
    name: str
    seeds: int
    title: str

    @property
    def key(self) -> str:
        return f"{self.exp}/{self.name}"


PRIMARY = Condition("ex-2.1.10", "either-t100", 9, "anchored")
"""The D2.1 recipe: pooled either-operand labels, anchor + anti-subspace at the ex-2.1.8 operating point."""

CONTROL = Condition("ex-2.1.10", "lam0", 3, "un-anchored")
"""Same corpus, seeds, and code path as the primary, with the anchor weight at zero."""

# --- The lines -----------------------------------------------------------------

#: Token positions of a probe line, in sequence order.
POSITIONS = ("op1", "+", "op2", "=", "ans", "\n")

#: The answer token's position; the response is read from the logits one position earlier.
ANSWER_POS = 4

#: Positions whose state can reach the answer: the prompt. Writes at the answer and newline positions
#: are measured for the bound but cannot move the response.
PROMPT_POSITIONS = (0, 1, 2, 3)

#: The operand positions alone — the `operands` arm's mask.
OPERAND_POSITIONS = (0, 2)

#: Slices of the residual stream: the embedding, then the state after each of the four blocks.
SLICES = (0, 1, 2, 3, 4)

#: A line's dose: the larger of its two operands' redness (M1's label affinity, r·(1 − g/2 − b/2)).
#: Either operand can carry the concept, as either could draw the label in ex-2.1.10.
RED_DOSE = 0.8
"""*Red lines* have dose ≥ 0.8: an operand among the five reddest colors of the grid (365 lines)."""

NONRED_DOSE = 0.2
"""*Non-red lines* have dose ≤ 0.2: neither operand carries more than a fifth of full redness (1,689 lines)."""

DOSE_EDGES = (0.2, 0.4, 0.6, 0.8)
"""Bin edges for the grading read: five bins, the first being the non-red lines and the last the red lines."""

# --- The interventions ---------------------------------------------------------

#: The primary intervention, and the one every hypothesis scores: full-strength axis projection at
#: every slice and every position.
PRIMARY_GAMMA = 1.0

#: Arms on the primary condition, reported at the same statistics without gates. Each changes one
#: thing from the primary intervention.
ARMS = {
    "operands": "positions restricted to op1 and op2",
    "embedding": "slice 0 only",
    "last": "slice 4 only",
    "lobe": "the M1 lobe in place of the plain projection",
    "ablate": "the weights that read and write e₁ zeroed, then re-normalized",
}
# REVIEW: added the `ablate` row. The report's arm table listed five arms and H5 scores `ablate`,
# so the module was one arm short of the frozen prose. Verify: `ARMS` should have a row for every
# row of the arm table in the report's Method, and H5 needs this one.

LOBE = dict(a=0.5, b=1.0, p=1.0)
"""The lobe arm: no suppression below α = 0.5, ramping linearly to full removal at α = 1. The threshold sits
above the constant component the syntax embeddings carry (α ≈ 0.3–0.4 in the published map) and below the
red operand's (α ≈ 0.9), so it is the operator that leaves the bulk alone by design."""

GAMMAS = (0.25, 0.5, 0.75, 1.0)
"""The strength sweep (E1): the fraction of the e₁ component removed."""

#: The post-hoc tier, fitted per control run and applied to that run: a direction per slice, from the
#: op1-position states of one line per color, applied at that slice at every position. The `axis` row
#: is e₁ itself on the control — the operator with nothing on the axis.
POSTHOC = ("axis", "diff-in-means", "probe", "leace")

#: The diff-in-means split: colors at or above RED_DOSE against the rest.
#: The probe and LEACE targets: the color's redness.

# --- Gates ---------------------------------------------------------------------

RED_ACC_GATE = 0.5
"""H1: "seed-mean exact-match accuracy on red lines under the primary intervention falls to 0.5 or below"."""

RED_ACC_PARTIAL = 0.8
"""H1 partial: "between 0.5 and 0.8"."""

TASK_GATE = 0.02
"""H2 and H5: "seed-mean accuracy on non-red lines stays within 0.02 of the clean accuracy". The width
every D2.1 task gate used, kept for continuity; the probe set is over six times the size of the holdout
set those gates were read on, so the binomial share of the noise is smaller here."""

TASK_PARTIAL = 0.05
"""H2 and H5 partial: "within 0.05"."""

GRADE_DIP = 0.02
"""H3: "seed-mean damage is non-decreasing across the five dose bins, allowing a dip between adjacent
bins of at most 0.02"."""

GRADE_SPAN = 0.5
"""H3: "and the top bin's damage exceeds the bottom bin's by at least 0.5"."""

WRITE_QUANTILE = 99
"""H4: the quantile, over non-red lines, that the write bound is read at on both sides."""

WRITE_TOLERANCE = 1.25
"""H4: "at every slice and position, the seed-mean 99th-percentile write on non-red lines is at most
1.25 times arcsin of the seed-mean 99th-percentile clean alignment at that site"."""

RESOLUTION_SD = 2.0
"""Differences between arms or tiers smaller than this many pooled between-seed standard deviations of
the statistic are reported as not resolved."""

#: The seeds that decide agreement in the undesigned-response read: a red line "agrees" when at least
#: this many of the nine primary seeds decode the same answer.
AGREE_SEEDS = 5
