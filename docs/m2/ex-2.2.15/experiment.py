"""Ex-2.2.15: lines cut short by the training window — a scouting run of crop policies for the anchor.

Training windows are random crops of the packed corpus, so a line at either end of a window is often cut
short, and the pooled anchor term puts the whole of a labelled line's pull on whatever part of it is visible.
On a line cut before its op word, that part cannot know the op. This run trains ex-2.2.14's primary (anchored
`difference`, whole-line pull, label rate 0.02) under a few policies for which cut lines the anchor pulls, and
reads what each one does to the first operand's lean, to fragments seen without their op word, to the anchor,
and to the task. It proposes a default crop policy for the in-context grammar pilot.

Design constants only while the preregistration is in review; the DAG lands when the hypotheses freeze.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

DESIGN_ONLY = True

# --- What is inherited ---------------------------------------------------------------------------------

REFERENCE_EXPERIMENT = "m2/ex-2.2.14"
"""Everything but the crop policy and the window: ex-2.2.14's primary, `anchor-diff`. That is ex-2.2.11's
handover recipe (table A+, the stochastic corpus at 300k lines, the untied readout, λ_a 0.1 annealed to a
0.1 floor over the last tenth of training, τ 0.1, the anti-subspace schedule, 50 epochs at d64-L4) with no
*red* anchor, `difference` on e₁, `op` keying at `LABEL_RATE`, and the pull over the whole line."""

REFERENCE_CONDITION = "anchor-diff"
REFERENCE_OP1_LEAN = 0.19
REFERENCE_OP1_LEAN_CONTROL = -0.01
"""Ex-2.2.14's post hoc read, at the final slice: the mean cosine with e₁ at the first operand over every
op's probe lines, on the primary and on the control. The op-word arm, which never pulls the first operand,
sat at -0.04. Quoted from the published report; the report here reads its own `all` arm."""

CONTROL_EXPERIMENT = "m2/ex-2.2.11"
CONTROL = "control"
CONTROL_SEEDS = 5
"""The un-anchored control, served from the store as in ex-2.2.14: the task reference and the alignment
baseline."""

ANCHORED_OP = "difference"
LABEL_RATE = 0.02
"""As ex-2.2.14's primary. The sparse rate matters here: the term is normalized per labelled line, so at this
rate each labelled line gets a strong pull, and ex-2.2.14's every-line arm (fifty times the labels, each pull
fifty times weaker) showed no lean at the first operand."""

LINE_TOKENS = 6
ROLES = ("op1", "op", "op2", "=", "answer", "⏎")
OP_ROLE = 1
"""The op word is the only evidence of the op on a line of the current grammar, and it sits at role 1."""

# --- The windows ----------------------------------------------------------------------------------------

BLOCK = 64
PADDING_CHANCE = 0.1
"""Ex-2.1.3's data config, unchanged since: 64-token windows at a uniform offset, a tenth of them with a
zeroed prefix of 1 to `block // 3 - 1` tokens."""

SHORT_BLOCK = 32
"""The stress windows. At 32 tokens a window holds about five and a third lines, which is about the ratio of
window to line the in-context grammar would have with ~20-token contexts in 128-token windows, so the share
of line visits cut short roughly doubles (see `visit_shares`). Half the tokens per step means twice the steps
per epoch at the same number of tokens seen, so the short arms are compared with each other, and with the
control only loosely."""


def visit_shares(block: int, padding_chance: float = PADDING_CHANCE) -> dict[tuple[int, int], float]:
    """How often a line visit shows each run of roles, as (first role, last role): the exact enumeration of
    every window offset and padding length under the batch sampler, weighted by how often each occurs.
    `(0, 5)` is a whole line. The same enumeration as op1-lean's `crop_patterns`, over any block size.
    """
    pad_max = block // 3 - 1
    pads = [(0, 1 - padding_chance)] + [(p, padding_chance / pad_max) for p in range(1, pad_max + 1)]
    shares: defaultdict[tuple[int, int], float] = defaultdict(float)
    for offset in range(LINE_TOKENS):
        for pad, weight in pads:
            seen: dict[int, list[int]] = {}
            for pos in range(pad, block):
                seen.setdefault((offset + pos) // LINE_TOKENS, []).append((offset + pos) % LINE_TOKENS)
            for roles in seen.values():
                shares[(min(roles), max(roles))] += weight / LINE_TOKENS
    total = sum(shares.values())
    return {k: v / total for k, v in sorted(shares.items())}


# --- The crop policies ----------------------------------------------------------------------------------

Policy = Literal["all", "whole", "half", "scaled", "knowable", "cut-only"]


def line_weight(policy: Policy, first: int, last: int) -> float:
    """The weight of a labelled line's pull under *policy*, for a visit that shows roles *first* to *last*.

    The weight multiplies the line's pooled term; the denominator stays the count of labelled lines with any
    visible position, as under `all`. So a policy only ever takes pull away from a line, and a line it keeps
    gets the pull it had under `all`. The label draws are the same under every policy, which makes the arms at
    one seed train on identical batches and identical labels.
    """
    n = last - first + 1
    match policy:
        case "all":
            return 1.0
        case "whole":
            return float(n == LINE_TOKENS)
        case "half":
            return float(n > LINE_TOKENS / 2)
        case "scaled":
            return n / LINE_TOKENS
        case "knowable":
            # The positions at or after a visible op word; the pool runs over those alone.
            return float(first <= OP_ROLE <= last)
        case "cut-only":
            return float(n < LINE_TOKENS)


KNOWABLE_FIRST_ROLE = OP_ROLE
"""Under `knowable` the pull also narrows within a kept line: only positions at or after the op word are in
the pool, so the first operand of a whole line is outside it. That makes it the old grammar's version of the
pivot's label variant (c), where a position is pulled once the evidence before it names the op.

The denominator must still come from the unnarrowed mask: a visit showing only the first operand has no
position left in the pool, and if it dropped out of the count of labelled lines, every kept pull would grow
stronger. The DAG should build the weight in explicitly, with a test."""


def pull_share(policy: Policy, block: int = BLOCK) -> float:
    """The share of `all`'s pull that *policy* keeps, over every line visit with a visible position."""
    return sum(v * line_weight(policy, *k) for k, v in visit_shares(block).items())


# --- The runs ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    policy: Policy
    block: int = BLOCK


ARMS: tuple[Arm, ...] = (
    Arm("all", "all"),
    Arm("whole", "whole"),
    Arm("half", "half"),
    Arm("scaled", "scaled"),
    Arm("knowable", "knowable"),
    Arm("cut-only", "cut-only"),
    Arm("all-short", "all", SHORT_BLOCK),
    Arm("whole-short", "whole", SHORT_BLOCK),
)
"""`all` is today's behaviour and should reproduce ex-2.2.14's primary. `whole`, `half`, and `scaled` are the
grammar-agnostic policies; `knowable` needs to know where the evidence is, which the in-context grammar can
only approximate through the posterior. `cut-only` pulls the cut lines alone, so that `whole` and `cut-only`
split `all`'s pull in two. The short pair repeats `all` against `whole` where cut lines are twice as common."""

MITIGATIONS = ("whole", "half", "scaled")
"""The policies the rule chooses among. `knowable` is the reference for what knowing the evidence buys."""

SEEDS = 5
SEED_OFFSET = 400
"""Arm seed *i* trains at model seed `SEED_OFFSET + i`, the same five seeds for every arm. Ex-2.2.14 trained
at 300, so every seed here is fresh. Five seeds because the lean is a difference of about 0.2 and the task
gate needs ex-2.2.14's resolution."""

N_RUNS = SEEDS * len(ARMS)
assert N_RUNS == 40

# --- What is scored ------------------------------------------------------------------------------------

FINAL_SLICE = 4
"""Slice 4 is the output of the last block, where ex-2.2.14 read the lean."""

READABLE_LEAN = 0.10
"""H1 is readable only if `all`'s lean exceeds the control's by at least this much, half of ex-2.2.14's
excess. Below it the lean did not reproduce at these seeds, and H1 is unresolved."""

LEAN_BAND = 0.05
"""A policy removes the lean when its seed-mean lean at the final slice is within this of the control's.
A quarter of ex-2.2.14's excess."""

CUT_ONLY_SHARE = 0.5
"""H1's second half: `cut-only` keeps at least this share of `all`'s excess lean."""

MARGIN_KEEP = 0.9
"""H2: every policy's seed-mean op margin is at least this share of `all`'s."""

TASK_GATE = 0.02
"""H2: for each op, the seed-mean held-out expected exact match within this of the control's. Ex-2.2.11's
gate, unchanged."""

FRAGMENT_START_ROLES = (2, 3, 4, 5)
"""The fragment probe: each probe line shown from op2, `=`, the answer, or the newline on, as a sequence of
its own, so the op word is out of sight. These are the start-cut runs a training window leaves without their
op word."""

ADOPTION = (
    "The pilot's default crop policy is the one among `whole`, `half`, and `scaled` that keeps the largest "
    "share of the pull while it removes the lean (within the band of the control) and passes H2. If none "
    "does, the pilot keeps `all`, and leans on label variant (c) for the evidence question. If H1 is "
    "unresolved, the choice falls to the short pair: `whole` goes forward if `all-short`'s lean exceeds the "
    "control's by the readable margin, `whole-short`'s is within the band of it, and `whole` passes H2; "
    "`all` stays otherwise."
)
# REVIEW: the fallback branch now carries the same gates as the main one (a readable lean on `all-short`,
# the band on `whole-short`, H2 on `whole`); before, "shows a lean" had no threshold and the branch could
# adopt `whole` without checking H2. Verify: a reader who wants the short pair's own task gap in place of
# `whole`'s H2 can argue it, since the short arms are compared with the control only loosely.
