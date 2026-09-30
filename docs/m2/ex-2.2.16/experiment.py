"""Ex-2.2.16: the in-context grammar pilot — round 2 of the D2.2 quick route.

The first experiment on the grammar the pivot proposed: each line is a context of a few solved examples of one op,
written with `?` in place of the op word, then a query under the same op. The pilot trains the control at three
corpus conditions (example count and replacement rate), anchored `difference` with the whole-line label and its
variants, and the arms the design lists, at a few seeds each, and proposes by frozen rules what round 3 adopts.

This module holds the design constants and, below the constants, the DAG: corpus preparation (one build per
corpus condition, with the held-out contexts and label-variant arrays a training run needs), the training task
(ex-2.2.14's recipe over the grammar generator, with the trajectory ex-2.2.15 recorded), and the orchestration
that ties them together (stage B); then the evaluation of every run and the suppression pass on three arms,
over the checkpoints and held-out sets stage B publishes (stage C). The report imports the constants and computes
its method section (the posterior over ops and the Bayes ceiling) from the op table alone, through `posterior.py`
beside it.

    bin/mini run docs/m2/ex-2.2.16/experiment.py --app modal --max-containers 12 --budget 6h
    bin/mini status ex-2.2.16
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from mini import Ctx, Experiment, get_data_dir
from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, Op
from sca.data.incontext import context_length

# --- The grammar -----------------------------------------------------------------------------------------

OP_NAMES: tuple[str, ...] = (
    "mix",
    "screen",
    "multiply",
    "lighten",
    "darken",
    "difference",
    "exclusion",
    "hsvmix",
    "hue-hsv",
    "sat-hsv",
    "value-hsv",
)
TABLE: tuple[Op, ...] = tuple(OP_BY_NAME[n] if n in OP_BY_NAME else CANDIDATE_BY_NAME[n] for n in OP_NAMES)
"""Table A+ as ex-2.2.9 assembled it and every experiment since has used: five ops from ex-2.2.3, three commutative
additions, and the three HSV blend modes that read operand order. Every op is total on the grid. The order is the
one the reports print."""
N_OPS = len(TABLE)

ANCHORED_OP = "difference"
"""The op the anchored arms label, as in ex-2.2.14 and ex-2.2.15. It is total on the grid, so its ceiling is not
capped by rounding, and it is commutative."""

ROUNDING = "stochastic"
"""The corpus rounds each channel of a raw answer up or down at random, in proportion to where the raw value sits
between grid levels (`sca.data.ops.Rounding`), as every corpus since the handover has. The posterior uses the same
rounding, so the likelihood of a shown answer under an op is the probability that rounding under that op gives it."""


def context_tokens(k: int, verify: bool = False) -> int:
    """`a ? b = y ,` is six tokens, and so is the query `a ? b = y ⏎`: a context of k examples is `6k + 6` tokens,
    the layout `sca.data.incontext` generates; a verification line is two tokens longer, for the marker and the verdict.
    """
    return context_length(k, verify=verify)


# --- The posterior scan (the method section) ---------------------------------------------------------------

K_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
RHO_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5)
"""The grid the method section scans: example counts and replacement rates. The rate ρ is per example: each
example shows, with probability ρ, a draw from the answer distribution of another op, uniform over the other ten,
in place of one from the true op. A replacement is invisible when the other op agrees with the true op on the
pair, which is what makes the posterior smooth rather than a count of fitting ops."""

CUBE_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
"""Cube-noise rates κ scanned at the proposed conditions: with probability κ an example shows a color drawn
uniformly from the 216-color grid, which usually no op produces. The pivot allows it at a low rate so that the model
learns to discount examples that fit nothing; the scan says what it costs the ceiling."""

N_CONTEXTS = 40_000
POSTERIOR_SEED = 2216
"""Contexts sampled per grid point by the method section, and the seed of the stream they are drawn from. At this
count the standard error of a ceiling is about 0.002, well under the differences between grid points."""

MIDDLE_BAND = (0.5, 0.95)
"""The band of the posterior on the true op that the pivot calls graded: above it the context all but names the
op, below it the evidence is split or points to another op. The share of contexts in the band is reported beside the spread."""

# --- The proposed corpus conditions ------------------------------------------------------------------------

GRAMMAR_CONDITIONS: tuple[tuple[int, float], ...] = ((3, 0.2), (3, 0.3), (4, 0.3))
CENTRE: tuple[int, float] = (3, 0.3)
"""The three (examples, ρ) conditions the control trains at, proposed by the method section from the scan. The
center is the working point of the pivot. The second differs in ρ alone, a step down in inference difficulty on the same
line length, so it is the fallback within the same block size if the control falls short of the ceiling at the
center. The third adds one example at the same ρ as the center, which raises the ceiling by about 0.06 and keeps most of the
spread; it asks whether more evidence buys a higher ceiling without flattening the stimulus."""
# REVIEW: proposed from the scan in the method section, not yet reviewed. The alternatives weighed there are
# (3, 0.4), which adds no middle-band share over the center and costs 0.09 of ceiling, and (4, 0.35), which has a
# little more spread than the third condition (0.32 against 0.29) and 0.04 less ceiling. Verify: the grid table in the method section.

CUBE_RATE: float = 0.02
"""The cube-noise rate of the corpus. The ceiling and floor of record at each condition are computed at this rate
(the cube scan in the method section); the κ = 0 scan is the design tool that picked the conditions."""
# REVIEW: proposed at the lowest scanned rate, which costs about 0.013 of ceiling at the center condition (the
# cube table in the method section), so that the model meets examples that fit no op, as the pivot asks, without
# the noise eating into the margin of rule (a). The alternative is κ = 0 for the pilot, with cube noise deferred to
# round 3, which keeps the κ = 0 scan as the ceiling of record. Verify: the cube table.
assert CUBE_RATE in CUBE_GRID, "the ceiling of record is read from the cube scan"

# --- What is inherited --------------------------------------------------------------------------------------

MODEL = "d64-L4"
LARGE_MODEL = "d128-L4"
"""The larger control is wider, not deeper: the worry in the pivot is the per-example lookup over eleven ops, which
is width-bound, and the same depth keeps the slice axis comparable with every other arm."""
# REVIEW: "wider or deeper" in the design; the alternative is d64-L6, which the layer sweep (round 4) would then
# have a first point for. Verify: the slice count is what every alignment figure shares across arms.

SEEDS = 3
"""Per arm, as the pilot section of the design sets, the larger control included."""

CROP_POLICY = "whole"
"""The anchor pulls only labelled lines wholly inside the training window, as ex-2.2.15 proposed after review:
a whole context always shows its evidence. Its rule chose `half`; the reasons for `whole` are in the section
"The rule for the pilot" of that report."""

# The `knowable` oracle of ex-2.2.15 pulled a whole labelled line only when its op was in view. Under `whole` every
# pulled context is wholly in view, so on this grammar the oracle is the same arm as the primary, and the pilot drops it.

# --- The corpus and the recipe -------------------------------------------------------------------------------

N_LINES = 300_000
"""Contexts in the corpus, one per line: the line count of ex-2.2.14, so that the anchored op has the same number of
lines (about 27k) and the labeller the same number of draws. A context is four times the tokens of an op-word
line, so an epoch is about four times the steps."""

BLOCK = 96
"""The training window. A context of four examples is 30 tokens, so a random crop of 96 holds on average about two
whole contexts of four examples and three of three, with a fragment at each end. The window of 64 in ex-2.2.15 held
about one whole context of four examples."""
# REVIEW: the pivot asks for "at least two whole contexts"; 96 meets it at both example counts. The alternative
# is 64, which keeps the window of ex-2.2.15 and would fit two whole three-example contexts only just.

VERIFY_RATE = 0.3
"""In the verification arms, the share of contexts written as verification lines (`sca.data.incontext`,
`verify_rate`): a candidate equation and a `TRUE` or `FALSE` verdict in place of the completion. The rest of the
corpus is unchanged, so completion has 70% of the lines it has in the other arms."""
# REVIEW: a proposal. Higher gives D2.3 more verification data on the same checkpoints; lower keeps completion
# closer to the other arms. Rule (d) scores the difference either way.

LABEL_RATE = 0.02
"""Per context of the anchored op, the chance it draws a label, as ex-2.2.14 and ex-2.2.15 set it: about 545
labelled contexts in the corpus. The label covers the whole context under the whole-line label; the variants
narrow it."""

EPOCHS = 50
"""The count of ex-2.2.9, unchanged since; the recipe (λ_a = 0.1 annealed to a 0.1 floor over the last tenth, τ = 0.1,
the anti-subspace weight from 2.5× to 0.3× by 90% of training, the untied readout) is that of ex-2.2.14."""

HOLDOUT_CONTEXTS = 2_000
"""Held-out contexts per op for the task and alignment measurements, drawn from the same generator at a seed
the corpus does not use. Contexts rather than pairs are held out: a context is the unit the model is scored on,
and a pair recurs across contexts anyway."""

# --- The arms ----------------------------------------------------------------------------------------------

LABEL_VARIANTS: tuple[tuple[str, str], ...] = (
    ("whole", "the whole context, every slice"),
    ("no-emb", "(a) the whole context, the embedding slice left out"),
    ("latter", "(b) the latter half of the context, where the prefix posterior is near its final value"),
    ("prefix", "(c) a position once the posterior given the tokens before it clears the threshold"),
    ("sampled", "(d) the whole context, the label drawn with probability equal to the posterior on the op"),
)
"""The whole-line label and the four variants of `/todo/science/label-variants-in-context-op.md`, by the short
name each arm carries and the positions it pulls on a labelled context."""

PREFIX_THRESHOLD = 0.5
"""Label variant (c): a position is pulled when the posterior on the anchored op, given the examples before it,
is at least this. The bottom of the middle band, so a context whose evidence never clears it is not pulled at all."""

HINGE_CAP = 0.8
"""The hinge arm: the anchor term counts an alignment of this as full, 1 − cos/cap in place of 1 − cos, and stops
pulling above it. Set below the cosine near 1 the op word reached in ex-2.2.14, and above what the whole-line pull put
at the use sites there (about 0.1). Only the anchor term is remapped; every measurement uses the raw cosine."""
# REVIEW: a proposal; the alternative is 0.5, which would leave the pulled states with half their length off the
# axis. Rule (c) reads saturation on the uncapped arm, so the cap only matters if the capped arm goes forward.

HINGE_SOFTNESS = 0.05
"""How far the corner of the hinge is rounded: the term on a position is (s/cap)·softplus((cap − cos)/s). Well below
the cap that is 1 − cos/cap; the gradient then fades smoothly to zero over roughly cap ± 2s, where a sharp corner
would switch it off at the cap. The anti-subspace term pushes back on the state across that band, so a sharp corner
could make a state flip between pulled and not pulled from step to step; the rounded one gives the two terms a smooth
point to balance at, near the cap."""
# REVIEW: the remap alone (max(0, 1 − cos/cap)) is the clip max(0, cap − cos) scaled by 1/cap, so it keeps the corner;
# the softplus is what removes it. The 1/cap makes the pull below the cap 25% stronger than on the whole-line arm;
# dropping it would keep the two arms at equal strength there.


@dataclass(frozen=True)
class Arm:
    """One arm of the pilot: what it trains and why it is here."""

    name: str
    group: str
    """Which question the arm serves: `control`, `label`, `hinge`, `verify`, or `mask`."""
    condition: tuple[int, float] = CENTRE
    anchored: bool = False
    label: str = "whole"
    """The label-variant short name (`LABEL_VARIANTS`); ignored on an unanchored arm."""
    policy: str = CROP_POLICY
    hinge: bool = False
    verify: bool = False
    mask: bool = False
    """The newline mask: attention does not cross a line break, so a context never reads the ones before it."""
    model: str = MODEL
    seeds: int = SEEDS
    note: str = ""


ARMS: tuple[Arm, ...] = (
    *(
        Arm(f"control-k{k}-r{rho:g}", "control", (k, rho), note="the un-anchored control at each corpus condition")
        for k, rho in GRAMMAR_CONDITIONS
    ),
    Arm(
        "control-large",
        "control",
        model=LARGE_MODEL,
        note="the larger control, trained either way and scored only if rule (a) needs it",
    ),
    *(Arm(f"anchor-{label}", "label", anchored=True, label=label, note=desc) for label, desc in LABEL_VARIANTS),
    Arm("anchor-hinge", "hinge", anchored=True, hinge=True, note="the whole-line pull capped by a hinge"),
    Arm("control-verify", "verify", verify=True, note="the control with verification lines"),
    Arm("anchor-verify", "verify", anchored=True, verify=True, note="the whole-line arm with verification lines"),
    Arm("control-mask", "mask", mask=True, note="the control with the newline mask"),
    Arm("anchor-mask", "mask", anchored=True, mask=True, note="the whole-line arm with the newline mask"),
)
"""The arms the pilot section of the design lists, in its order. Every anchored arm anchors `ANCHORED_OP` at the
center condition; the control at the center is the reference for all of them."""

PRIMARY = "anchor-whole"
CONTROL = f"control-k{CENTRE[0]}-r{CENTRE[1]:g}"
N_RUNS = sum(a.seeds for a in ARMS)
assert PRIMARY in {a.name for a in ARMS} and CONTROL in {a.name for a in ARMS}
assert N_RUNS == 42


def arm(name: str) -> Arm:
    return next(a for a in ARMS if a.name == name)


# --- The rules -----------------------------------------------------------------------------------------------

CEILING_MARGIN = 0.03
"""Rule (a): a control passes at a condition when its seed-mean held-out expected exact match is at least the
ceiling of record less this margin. One-sided: a control above the ceiling passes, and the calibration check
says whether it got there by sharpening past calibration. About a fifteenth of the room between floor and ceiling
at the center condition, and less than the ceiling gives up when one of three examples is ignored (the method
section computes that shortfall)."""
# REVIEW: a proposal. The alternatives are the task gate, 0.02, which earlier experiments used for "no
# measurable difference" between arms, or a skill-score floor (0.9 would be about 0.04 at the center). Verify:
# the room and the one-example shortfall in the conditions table and the calibration table.

SPREAD_STATISTIC = "sd"
"""Rule (a) ranks the passing conditions by the standard deviation of the posterior on the true op across
contexts; the middle-band share is reported beside it and breaks a tie."""

SEED_BAND_SD = 2.0
"""The seed band between two arms at n seeds each is `SEED_BAND_SD · σ · √(2/n)`, with σ the seed standard
deviation of the statistic pooled over the two arms: the smallest difference of seed means the comparison
resolves, as ex-2.2.9 defined it. Rules (b) and (d) use it on held-out expected exact match at the center
condition."""

MARGIN_KEEP = 0.9
"""Rules (b) and (c): an arm "holds the anchor" when its seed-mean op margin is at least this share of the
margin of the whole-line arm, the bar of ex-2.2.15."""

SATURATION_LEVEL = 0.9
"""Rule (c): the uncapped whole-line arm saturates at the query `?` when the seed-mean cosine with e₁ at that
position, on the held-out contexts of the anchored op at the last block, is at least this. The op word in
ex-2.2.14 sat near 1; *red* at its own token reached about 0.5."""
# REVIEW: a proposal, set so that only a state that is nearly all concept counts. The alternative is to read
# saturation as the dose collapse itself, on the suppression pass: a projection whose damage arrives only in
# the last few percent of γ at the query `?`. Verify: the alignment-by-role figure of rule (c).

KL_CALIBRATED = 0.05
"""The calibration check: a model is calibrated at a condition when the mean over held-out contexts of the KL
divergence from the Bayes predictive to its answer distribution is under this, in nats. The method section
gives reference values: a tenth of the floor mixed into the Bayes answer is about this large, and one example
ignored is several times it."""
# REVIEW: a proposal; the check gates nothing in the pilot, it flags a control that scores above the ceiling and
# it is the calibration measurement the design asks for. The alternative is a threshold relative to the
# irreducible loss H(q), such as a twentieth of it. Verify: the calibration table in the method section.

DOSE_GAMMAS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
"""Rule (e), the dose axis of the projection: the edit removes a fraction γ of the component on e₁ and re-normalizes.
γ = 1 is the plain projection."""

REPULSION_LANDINGS: tuple[float, ...] = (0.25, 0.0)
"""Rule (e), the dose axis of the repulsion: the alignment a state above the threshold is sent to. Both landings sit
between the threshold and zero, the range the operator was designed for."""
# REVIEW: an earlier draft carried negative landings down to the antipode (−1), which is outside the use the
# repulsion was designed for; Sandy has not yet settled how, or whether, it should act on states that lie almost on
# e₁. Treat these two landings as a placeholder until then.

REPULSION_THRESHOLD = 0.5
"""States below this alignment are left where they are by the repulsion."""

REFLECT_GAMMA = 2.0
"""The reflection, `projection` at γ = 2: one dose only, so it enters rule (e) as a reference and cannot win it."""

EDIT_SITES: tuple[str, ...] = ("query ?", "query =", "every position")
"""Where each operator acts, at every slice: the two query sites separately, and every position of the context.
Rule (e) is scored on the every-position edit; the query sites are the first bypass measurement and gate nothing."""
# REVIEW: scoring on every position is the reading that does not depend on where the op turns out to sit. The
# alternative is to score at the query site with the larger full-dose damage, which is what round 3 would
# suppress at. Verify: the per-site damage table of rule (e).

SELECTIVITY_GATE = 0.02
"""Rule (e): on each of the other ten ops, the seed-mean drop in held-out expected exact match under an edit is at
most this at every dose: the task gate ex-2.2.11 set."""

GRADING_MIN_DAMAGE = 0.5
"""Rule (e): the damage an operator does "grades with dose" when the seed-mean drop on the contexts of the anchored op is
non-decreasing along its dose axis, and the drop at full dose is at least this share of the way from the clean
score to the target null."""


# =============================================================================================
# The DAG
# =============================================================================================


def _load_ex2214():
    """Ex-2.2.14's module, by path and left out of `sys.modules` (as ex-2.2.15 loads it), so this module's task
    bodies still cloudpickle by value for a remote worker. Reused for the recipe: `Condition229`, `schedules`,
    `_make_config`, and the anchor hyperparameters (`LAM`, `TAU`, `EPOCHS`).
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.14" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2216_ex2214", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def _load_posterior():
    """`posterior.py` beside this module, by path, the same way — the vectorized posterior the corpus prep
    reads (much faster than `sca.data.incontext.posterior_over_ops`'s per-context Python loop over 40k+
    contexts; `tests/sca/test_ex_2_2_16_posterior.py` checks the two agree). Loaded at module init time
    so it's available in the Modal image.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent / "posterior.py"
    spec = importlib.util.spec_from_file_location("ex2216_posterior", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def _get_posterior():
    """Return the cached posterior module loaded at module init time."""
    return _POSTERIOR_MODULE


ex2214 = _load_ex2214()
_POSTERIOR_MODULE = _load_posterior()
LAM = ex2214.LAM
TAU = ex2214.TAU
Condition229 = ex2214.Condition229
schedules = ex2214.schedules
_make_config = ex2214._make_config

assert ex2214.ANCHOR_AXIS == 0
"""The op is anchored to e₁, as ex-2.2.14 and ex-2.2.15 anchored it."""


def model_dims(name: str) -> tuple[int, int]:
    """Parse a model name of the form `d<embd>-L<layer>` (`MODEL`, `LARGE_MODEL`) into (n_embd, n_layer)."""
    import re

    m = re.fullmatch(r"d(\d+)-L(\d+)", name)
    assert m, f"unrecognized model name {name!r}"
    return int(m.group(1)), int(m.group(2))


N_LAYER = model_dims(MODEL)[1]
assert model_dims(LARGE_MODEL)[1] == N_LAYER, "the large control is wider, not deeper (MODEL and LARGE_MODEL note)"
FINAL_SLICE = N_LAYER
"""Slice `N_LAYER` is the output of the last block, where the report's rules read alignment."""

# --- Seeds, distinct per corpus condition and never shared with a draw the corpus made ------------------

SEED_OFFSET = 500
"""Ex-2.2.14 trained at 300, ex-2.2.15 at 400; every model seed here is fresh."""

CORPUS_SEED = 221_600
HOLDOUT_SEED = 221_601
PROBE_SEED = 221_602
"""Held apart, and offset again per corpus condition (`run`), so a condition's corpus, its held-out contexts,
and its trajectory probe set are three streams that never draw from one another."""

N_TRAJ_PROBE = 64
"""Held-out contexts per op for the trajectory's probe set: enough to see a trend every `TRAJ_STRIDE` steps
without sizing the eval read the way `HOLDOUT_CONTEXTS` does. Always drawn as completion contexts (verify_rate
0), even on the verification corpus, so every probe context of one corpus condition shares one token length
and stacks into one array."""

TRAJ_STRIDE = 50
"""Recorded at every trajectory point, as ex-2.2.14 and ex-2.2.15 recorded theirs."""

LABELS_REF = "reports/m2/ex-2.2.16/labels/{key}"
CORPUS_REF = "reports/m2/ex-2.2.16/corpus/{key}"
HOLDOUT_REF = "reports/m2/ex-2.2.16/holdout/{key}"
PROBES_REF = "reports/m2/ex-2.2.16/probes/{key}"
METRICS_REF = "reports/m2/ex-2.2.16/metrics"
TRAJ_REF = "reports/m2/ex-2.2.16/trajectories"
CHECKPOINT_REF = "reports/m2/ex-2.2.16/checkpoints/{label}"


def cond_key(k: int, rho: float, verify: bool = False) -> str:
    return f"k{k}-r{rho:g}" + ("-verify" if verify else "")


def _npz(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _sample_by_op(op_table, k: int, rho: float, cube_rate: float, verify_rate: float, n_per_op: int, seed: int):
    """*n_per_op* contexts of every op in *op_table*, one `sca.data.incontext.sample_context` draw at a time,
    each op at its own branch of *seed* (`(seed, op_index)`) so the set does not depend on table order and is
    reproducible independent of how many ops or contexts came before it.
    """
    from sca.data.incontext import sample_context

    out = []
    for i, op in enumerate(op_table):
        rng = np.random.default_rng((seed, i))
        out += [
            sample_context(op_table, op, k, rho, rng, cube_rate, "stochastic", verify_rate) for _ in range(n_per_op)
        ]
    return out


def _post_contexts(P, contexts, color_index: dict, op_index: dict):
    """Convert a batch of same-length `sca.data.incontext.Context` into `posterior.py`'s `Contexts`: color pairs
    and colors as indices into the palette (*color_index*, `sca.data.ops.colors()` order — the order
    `posterior.build_table` also uses), read off the shown examples and the query.
    """
    n, k = len(contexts), len(contexts[0].examples)
    n_colors = len(color_index)
    ex_pair = np.empty((n, k), dtype=np.int64)
    ex_color = np.empty((n, k), dtype=np.int64)
    query_pair = np.empty(n, dtype=np.int64)
    true_op = np.empty(n, dtype=np.int64)
    for i, c in enumerate(contexts):
        for j, ex in enumerate(c.examples):
            ex_pair[i, j] = color_index[ex.lhs] * n_colors + color_index[ex.rhs]
            ex_color[i, j] = color_index[ex.answer]
        query_pair[i] = color_index[c.query_lhs] * n_colors + color_index[c.query_rhs]
        true_op[i] = op_index[c.op]
    return P.Contexts(true_op, ex_pair, ex_color, query_pair)


def prepare_corpus_condition(
    k: int,
    rho: float,
    verify_rate: float,
    cube_rate: float,
    n_lines: int,
    corpus_seed: int,
    holdout_seed: int,
    holdout_n: int,
    probe_seed: int,
    probe_n: int,
    key: str,
) -> dict:
    """One corpus condition of the in-context grammar: the packed token stream and the label-variant arrays a
    training run needs (`op_ids`, `context_len`, `prefix_ok`, `sample_prob`, and — for a verification corpus —
    the `FALSE`-candidate `loss_mask`); the held-out contexts of every op with their posteriors and Bayes
    ceiling, for stage C; and a small probe set of the anchored op's and every other op's contexts for the
    training trajectory.

    `prefix_ok` and `sample_prob` are computed for the anchored op's contexts only: `LabelSpec`'s `_context_mask`
    never reads either array off a context of another op (`is_anchored` gates the draw first), so every other
    position is left at its zero default. `prefix_ok` reads the posterior after the complete examples before
    each position (`posterior.py`'s `prefix_posteriors`, at role `min(role // 6, k)` examples in); `sample_prob`
    reads the posterior after all `k` examples, scaled so its mean over the anchored op's contexts is
    `LABEL_RATE` (`/todo/science/label-variants-in-context-op.md`).
    """
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import ops as grammar
    from sca.data.incontext import encode_corpus, line_boundaries
    from sca.data.incontext import op_ids as op_ids_of
    from sca.data.incontext import sample_corpus, vocabulary
    from sca.data.named_colors import WordTokenizer
    from mini.store import put

    P = _get_posterior()
    table = tuple(TABLE)
    anchored_op_id = OP_NAMES.index(ANCHORED_OP)

    corpus = sample_corpus(
        n_lines, corpus_seed, k, rho, op_table=table, cube_rate=cube_rate, rounding=ROUNDING, verify_rate=verify_rate
    )
    tokenizer_config = TokenizerConfig(vocabulary=sorted(vocabulary()))
    tokenizer = WordTokenizer(tokenizer_config)
    tokens = encode_corpus(corpus, tokenizer.stoi)
    newline_id = tokenizer.stoi["\n"]
    context_op = op_ids_of(corpus, table)
    context_len = np.array([c.n_tokens for c in corpus], dtype=np.int32)
    line_starts = line_boundaries(tokens, newline_id)

    cs = grammar.colors()
    color_index = {c: i for i, c in enumerate(cs)}
    op_index = {o.name: i for i, o in enumerate(table)}
    post_table = P.build_table(table)

    # --- Label-variant arrays, the anchored op's contexts only --------------------------------------------
    anchored_idx = np.flatnonzero(context_op == anchored_op_id)
    prefix_ok = np.zeros(len(tokens), dtype=bool)
    sample_prob = np.zeros(len(tokens), dtype=np.float32)
    clipped, sample_prob_mean = 0, 0.0
    if len(anchored_idx):
        anchored_contexts = [corpus[i] for i in anchored_idx]
        ctx_batch = _post_contexts(P, anchored_contexts, color_index, op_index)
        prefix_post = P.prefix_posteriors(post_table, ctx_batch, rho, cube_rate)[:, :, anchored_op_id]  # (n, k)
        prior = np.full((len(anchored_idx), 1), 1.0 / len(table))
        post_after = np.concatenate([prior, prefix_post], axis=1)  # (n, k+1): after 0..k examples
        post_full = post_after[:, k]
        scale = LABEL_RATE / post_full.mean()
        raw = scale * post_full
        clipped = int((raw > 1).sum())
        scaled = np.clip(raw, 0.0, 1.0)
        sample_prob_mean = float(scaled.mean())
        for local_i, ctx_i in enumerate(anchored_idx):
            start, length = int(line_starts[ctx_i]), int(context_len[ctx_i])
            r = np.arange(length)
            ex_at_role = np.minimum(r // 6, k)
            prefix_ok[start : start + length] = post_after[local_i, ex_at_role] >= PREFIX_THRESHOLD
            sample_prob[start : start + length] = scaled[local_i]

    # --- The verification loss mask: the FALSE candidate's answer token, in every verification line --------
    loss_mask = None
    if verify_rate > 0:
        loss_mask = np.zeros(len(tokens), dtype=bool)
        for i, ctx in enumerate(corpus):
            if ctx.verify is not None and not ctx.verify.verdict:
                loss_mask[int(line_starts[i]) + k * 6 + 4] = True

    # --- Held-out contexts, with their posterior and Bayes ceiling ------------------------------------------
    # `HOLDOUT_CONTEXTS` per op, each one `sample_context` draw at the condition's (k, rho, cube_rate).
    holdout = _sample_by_op(table, k, rho, cube_rate, verify_rate, holdout_n, holdout_seed)
    ho_tokens = encode_corpus(holdout, tokenizer.stoi)
    ho_op = op_ids_of(holdout, table)
    ho_len = np.array([c.n_tokens for c in holdout], dtype=np.int32)
    ho_batch = _post_contexts(P, holdout, color_index, op_index)
    ho_post = P.posterior(post_table, ho_batch, rho, cube_rate)  # (n, n_ops)
    ho_ceiling = P.expected_match(post_table, ho_batch, ho_post)  # (n,)
    ho_verdict = np.array([(c.verify.verdict if c.verify is not None else -1) for c in holdout], dtype=np.int8)

    # --- The trajectory probe set: completion contexts only, one fixed length -----------------------------
    probes = _sample_by_op(table, k, rho, cube_rate, 0.0, probe_n, probe_seed)
    probe_tokens = encode_corpus(probes, tokenizer.stoi).reshape(len(table) * probe_n, context_length(k))

    n_chars = sum(len(w) for ctx in corpus for w in ctx.words)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=n_chars,
        sources=[DatasetMetadata(title=f"in-context grammar corpus ({key})", fixes=[], total_chars=n_chars)],
    )
    corpus_dir = get_data_dir() / "corpora" / key
    save_data(tokens, meta, corpus_dir)

    stats = {
        "key": key,
        "k": k,
        "rho": rho,
        "verify_rate": verify_rate,
        "cube_rate": cube_rate,
        "n_lines": n_lines,
        "total_tokens": int(len(tokens)),
        "vocab_size": tokenizer.vocab_size,
        "n_anchored_contexts": int(len(anchored_idx)),
        "sample_prob_clipped": clipped,
        "sample_prob_mean": sample_prob_mean,
        "context_tokens": context_length(k),
        "context_tokens_verify": context_length(k, verify=True) if verify_rate > 0 else None,
    }
    return {
        "key": key,
        "k": k,
        "meta": meta,
        "stats": stats,
        "newline_id": newline_id,
        "anchored_op_id": anchored_op_id,
        "corpus": put(corpus_dir, name=f"ex-2.2.16-{key}-corpus"),
        "labels": put(
            _npz(
                op_ids=context_op,
                context_len=context_len,
                prefix_ok=prefix_ok,
                sample_prob=sample_prob,
                **({"loss_mask": loss_mask} if loss_mask is not None else {}),
            ),
            name=f"ex-2.2.16-{key}-labels.npz",
        ),
        "holdout": put(
            _npz(
                tokens=ho_tokens,
                op_ids=ho_op,
                context_len=ho_len,
                posterior=ho_post,
                ceiling=ho_ceiling,
                verify_verdict=ho_verdict,
            ),
            name=f"ex-2.2.16-{key}-holdout.npz",
        ),
        "probes": put(_npz(tokens=probe_tokens), name=f"ex-2.2.16-{key}-probes.npz"),
    }


def _label_variant(
    label: str, n_layer: int
) -> tuple[Literal["whole", "latter", "prefix", "sampled"], tuple[int, ...] | None]:
    """The `LabelSpec.variant` and the training step's `anchor_slices` for one label-variant arm's short name.
    "no-emb" pulls every position (`variant="whole"`) but leaves the embedding slice out of the pull, as
    `LabelSpec`'s own docstring asks.
    """
    if label == "no-emb":
        return "whole", tuple(range(1, n_layer + 1))
    if label in ("whole", "latter", "prefix", "sampled"):
        return label, None
    raise ValueError(f"unknown label variant {label!r}")


def cells(arms: tuple[Arm, ...], preps: dict[str, dict], seeds: int = SEEDS, epochs: int = EPOCHS) -> list[dict]:
    """One row per run: the config, both schedules, the labeller's variant, and which corpus condition it
    trains on. *seeds* and *epochs* are arguments (as ex-2.2.15's `cells` takes them) so a short local
    prototype runs the same code at a fraction of the length.
    """
    from sca.config import ModelConfig
    from sca.utils import align

    rows = []
    for a in arms:
        cond = a.condition if a.group == "control" else CENTRE
        key = cond_key(*cond, a.verify)
        prep = preps[key]
        n_embd, n_layer = model_dims(a.model)
        variant, anchor_slices = _label_variant(a.label, n_layer)
        # REVIEW: the model seed repeats across arms (arm A's seed 0 and arm B's seed 0 share a model seed), as
        # ex-2.2.14 and ex-2.2.15 seed their cells; the alternative is a seed unique to each of the 42 runs. The
        # corpus, held-out, and probe seeds are distinct per corpus condition either way (`run`).
        for seed in range(min(a.seeds, seeds)):
            model_seed = SEED_OFFSET + seed
            config = _make_config(
                align(prep["meta"].tokenizer_config.vocab_size, 64), model_seed, epochs, n_embd, n_layer
            )
            config.tokenizer = prep["meta"].tokenizer_config.model_copy()
            config.model = ModelConfig.model_validate(
                config.model.model_dump()
                | {
                    "block_size": BLOCK,
                    "tie_embeddings": False,
                    "line_mask_token": prep["newline_id"] if a.mask else None,
                }
            )
            base = Condition229(
                a.name,
                1,
                a.name,
                lam=(LAM if a.anchored else 0.0),
                tau=TAU,
                epochs=epochs,
                ops=OP_NAMES,
                n_lines=N_LINES,
            )
            anchor, anti = schedules(base)
            if a.hinge:
                anchor = anchor | {"hinge": (HINGE_CAP, HINGE_SOFTNESS)}
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "variant": variant,
                    "anchor_slices": anchor_slices,
                    "corpus_key": key,
                    "k": prep["k"],
                    "condition": a.name,
                    "seed": seed,
                    "model_seed": model_seed,
                    "label": f"{a.name}-s{seed}",
                }
            )
    return rows


def train_one(
    config,
    anchor: dict,
    anti: dict | None,
    variant: Literal["whole", "latter", "prefix", "sampled"],
    anchor_slices,
    corpus,
    labels,
    probes,
    k: int,
    traj_stride: int,
    label: str,
) -> dict:
    """Train one run of the in-context grammar's context-keyed anchor (`LabelSpec(keying="context", ...)`),
    crop policy `whole`, recording every *traj_stride* steps on the corpus condition's probe set: the op margin
    over the whole context (`context_margin`, the contexts of the anchored op as the labelled group, as
    ex-2.2.14's `line_margin` but with roles running over the whole context rather than a fixed prompt span),
    the first-operand lean (mean cosine at role 0), and the trailing-fragment lean (mean cosine over every
    position of the context cut to start at each example boundary, fed alone, as ex-2.2.15's `fragment_sums`).
    Both leans and the margin are read at `FINAL_SLICE`, the output of the last block.
    """
    # REVIEW: `train_anchored`'s own trajectory (`m_op1`, `m_span`, `alpha_op1`, `pi`, `m_line`) truncates its
    # probe reads to the fixed-grammar `PROMPT_SPAN` (roles 0-3), which does not cover a variable-length context;
    # those keys are dropped below (kept only `epoch`/`weight`/`anti_weight`/`val_loss`) in favor of the three
    # `on_record` computes here, over the whole context.
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec, alignment
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    newline_id = tokenizer.stoi["\n"]
    anchored_op_id = OP_NAMES.index(ANCHORED_OP)

    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=anchored_op_id,
            label_rate=LABEL_RATE,
            variant=variant,
            prefix_ok=(z["prefix_ok"] if variant == "prefix" else None),
            sample_prob=(z["sample_prob"] if variant == "sampled" else None),
            context_len=z["context_len"],
        )
        loss_mask = z["loss_mask"] if "loss_mask" in z.files else None

    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]  # (n_ops * N_TRAJ_PROBE, context_tokens(k))

    per_op = len(probe_tokens) // N_OPS
    weights = np.concatenate([np.full(per_op, 1.0 if o == ANCHORED_OP else 0.0) for o in OP_NAMES])
    weights = weights / weights.sum()
    starts = tuple(6 * i for i in range(1, k + 1))  # every example boundary a start-cut window can open at

    def context_margin(alpha_lines: np.ndarray) -> float:
        m = np.einsum("n,lnt->lt", weights, alpha_lines) - alpha_lines.mean(axis=1)  # (L1, T)
        return float(m.max(axis=1).mean())

    traj_extra: dict[str, list] = {"m_context": [], "op1_lean": [], "fragment_lean": []}

    def on_record(_index: int, model) -> None:
        cos = alignment(model, probe_tokens)  # (L1, N, T)
        traj_extra["m_context"].append(context_margin(cos))
        traj_extra["op1_lean"].append(float(cos[FINAL_SLICE, :, 0].mean()))
        total, count = 0.0, 0
        for s in starts:
            frag = np.ascontiguousarray(probe_tokens[:, s:])
            fc = alignment(model, frag)  # (L1, N, T - s)
            total += float(fc[FINAL_SLICE].sum())
            count += fc.shape[1] * fc.shape[2]
        traj_extra["fragment_lean"].append(total / count)

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=(AntiSpec(**anti) if anti is not None else None),
        label_p=spec,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        probe_line_w=weights,
        crop=CROP_POLICY,
        anchor_slices=anchor_slices,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
        newline_id=newline_id,
        min_line_tokens=context_length(k),
        loss_mask=loss_mask,
        on_record=on_record,
    )
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {
            kk: np.asarray(vv).tolist()
            for kk, vv in traj.items()
            if kk in ("epoch", "weight", "anti_weight", "val_loss")
        }
        | traj_extra,
        "checkpoint": put(workdir / "model", name=f"ex-2.2.16-{label}-ckpt"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """The design constants the report (and stage C) reads beside the results."""
    return {
        "experiment": "ex-2.2.16",
        "anchored_op": ANCHORED_OP,
        "arms": [asdict(a) for a in ARMS],
        "seed_offset": SEED_OFFSET,
        "n_runs": N_RUNS,
        "ops": list(OP_NAMES),
        "corpus_conditions": [list(c) for c in GRAMMAR_CONDITIONS],
        "centre": list(CENTRE),
        "cube_rate": CUBE_RATE,
        "block": BLOCK,
        "epochs": EPOCHS,
        "holdout_contexts": HOLDOUT_CONTEXTS,
        "verify_rate": VERIFY_RATE,
        "label_rate": LABEL_RATE,
        "final_slice": FINAL_SLICE,
    }


def publish_results(trained: list[dict], preps: dict[str, dict]) -> dict:
    """Metrics (JSON: design + every corpus condition's stats), trajectories (JSON), every corpus condition's
    corpus/labels/holdout/probes, and every end checkpoint, each under its own ref.
    """
    import json

    from mini.store import put, set_ref

    metrics = {"design": design(), "corpus": {key: p["stats"] for key, p in preps.items()}}
    set_ref(METRICS_REF, put(json.dumps(metrics).encode(), name="ex-2.2.16-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.16-trajectories.json"))
    for key, p in preps.items():
        set_ref(CORPUS_REF.format(key=key), p["corpus"])
        set_ref(LABELS_REF.format(key=key), p["labels"])
        set_ref(HOLDOUT_REF.format(key=key), p["holdout"])
        set_ref(PROBES_REF.format(key=key), p["probes"])
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    return {"n_runs": len(trained), "n_conditions": len(preps)}


# =============================================================================================
# Stage C: the evaluation and the suppression pass
# =============================================================================================
#
# Nothing above this line changes for stage C: `train_one` and `prepare_corpus_condition` keep their memo keys
# and their evidence, so a run of the whole DAG serves stage B from the store and computes only what follows.

EVAL_REF = "reports/m2/ex-2.2.16/eval"
EVAL_ARRAYS_REF = "reports/m2/ex-2.2.16/eval-arrays/{label}"
SUPPRESSION_REF = "reports/m2/ex-2.2.16/suppression"
SUPPRESSION_ARRAYS_REF = "reports/m2/ex-2.2.16/suppression-arrays/{label}"

SUPPRESSION_ARMS: tuple[str, ...] = (PRIMARY, "anchor-hinge", CONTROL)
"""The arms the suppression pass scores (rule (e)): the whole-line arm, the hinge arm, and the control at the
center, every seed of each."""

EVIDENCE_BINS: tuple[float, ...] = (0.0, 0.05, 0.2, 0.5, 0.65, 0.8, 0.95, 1.0)
"""Bin edges on the posterior on the anchored op, for the binned alignment of the evidence section: three bins
across the middle band (0.5 to 0.95), three below it, and one above. The last bin is closed at 1."""
# REVIEW: the report leaves the bins open. These split the middle band in three so "higher in each bin than the
# one below it" has two steps to check; the per-context values are published beside them, so the report can
# re-bin without re-scoring.

REPULSION_KIND = "linear"
"""The landing map of the repulsion (`sca.intervention.repulsion_mapper`): every state at or above the threshold
lands at the landing, the form the report describes."""
# REVIEW: `bezier` is the smooth alternative, continuous in the arriving alignment. The report says states "above
# an alignment of 0.5" are sent to the landing, which is the linear map.

QUERY_Q = 1
QUERY_EQ = 3
"""Offsets within the query unit of the query `?` and the query `=`: at roles `6k + 1` and `6k + 3` of a context
of k examples. The answer is read from the logits at the query `=`."""
VERDICT_MARK = 5
"""Offset of the verdict marker `|` within the query unit of a verification line; the verdict is read from the
logits there."""


def context_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """The op margin of the glossary, on alignments `(L1, N, T)` of N same-length contexts: per slice, the
    *weights*-weighted mean alignment (uniform over the contexts of the anchored op) less the mean over all
    contexts, at the role where that gap is largest, averaged over slices.

    The same statistic as the closure of that name in `train_one`, which keeps its own copy: moving it out
    would change the source of `train_one`, which is its memo evidence, and re-train every run.
    """
    m = np.einsum("n,lnt->lt", weights, alpha) - alpha.mean(axis=1)
    return float(m.max(axis=1).mean())


def query_role(k: int, offset: int) -> int:
    """The role of a query token in a context of *k* examples (`QUERY_Q`, `QUERY_EQ`, `VERDICT_MARK`)."""
    return 6 * k + offset


def fragment_starts(k: int) -> tuple[int, ...]:
    """The example boundaries a trailing fragment starts at: after each example, the last one leaving the
    query alone. The starts `train_one` uses for its trajectory.
    """
    return tuple(6 * i for i in range(1, k + 1))


def _load_run(checkpoint, workdir):
    """The run's model, its tokenizer, the color token ids in palette order (`sca.data.ops.colors()`, the order
    `posterior.py` indexes), and the token → palette-index map.
    """
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from sca.model import NGPT
    from mini.store import get

    get(checkpoint, workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    assert isinstance(model, NGPT), "the operators act on the between-block stream of nGPT"
    tokenizer = WordTokenizer(config.tokenizer)
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(model.transformer.wte.shape[0], -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    return model, tokenizer, color_ids, tok2color


@dataclass(frozen=True)
class Holdout:
    """The held-out contexts of one corpus condition, split by kind: completion contexts stacked `(N, 6k + 6)`,
    and verification lines stacked `(M, 6k + 8)` (none outside the verification corpus).
    """

    tokens: np.ndarray
    op_ids: np.ndarray
    posterior: np.ndarray
    ceiling: np.ndarray
    verify_tokens: np.ndarray
    verify_op_ids: np.ndarray
    verdict: np.ndarray


def load_holdout(path, k: int) -> Holdout:
    """Unpack the packed held-out stream `prepare_corpus_condition` published, one context per row."""
    with np.load(path) as z:
        tokens, lens, op_ids = z["tokens"], z["context_len"], z["op_ids"]
        post, ceil, verdict = z["posterior"], z["ceiling"], z["verify_verdict"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    done = verdict < 0

    def stack(mask: np.ndarray, length: int) -> np.ndarray:
        assert (lens[mask] == length).all()
        return tokens[starts[mask][:, None] + np.arange(length)].astype(np.int32)

    return Holdout(
        stack(done, context_length(k)),
        op_ids[done],
        post[done],
        ceil[done],
        stack(~done, context_length(k, verify=True)),
        op_ids[~done],
        verdict[~done].astype(bool),
    )


def _post_queries(P, ho: Holdout, k: int, tok2color: np.ndarray):
    """The held-out completion contexts as `posterior.py` `Contexts`, holding what the scoring needs: the true op
    and the query pair. The examples are left empty: the posterior over ops was computed from them at prep time.
    """
    from sca.data.ops import colors

    n_colors = len(colors())
    a = tok2color[ho.tokens[:, 6 * k]]
    b = tok2color[ho.tokens[:, 6 * k + 2]]
    assert (a >= 0).all() and (b >= 0).all(), "query operands are colors"
    empty = np.zeros((len(a), 0), dtype=np.int64)
    return P.Contexts(ho.op_ids.astype(np.int64), empty, empty, (a * n_colors + b).astype(np.int64))


def _color_probs(logits: np.ndarray, color_ids: np.ndarray) -> np.ndarray:
    """The model answer distribution over the palette: the softmax over the whole vocabulary, at the color
    tokens. Mass on a non-color token is left out rather than renormalized away, so it counts as loss.
    """
    lg = logits.astype(np.float64)
    lg -= lg.max(axis=1, keepdims=True)
    p = np.exp(lg)
    return (p / p.sum(axis=1, keepdims=True))[:, color_ids]


def _match(table, ctx, p: np.ndarray) -> np.ndarray:
    """Expected exact match per context: the model mass on each answer the true op can give on the query pair,
    weighted by how often it gives it.
    """
    idx = table.idx[ctx.true_op, ctx.query_pair]  # (n, 8), −1 padding
    prob = table.prob[ctx.true_op, ctx.query_pair]
    return (np.take_along_axis(p, np.maximum(idx, 0), axis=1) * prob).sum(axis=1)


def _per_op(x: np.ndarray, op_ids: np.ndarray) -> list[float | None]:
    return [float(x[op_ids == o].mean()) if (op_ids == o).any() else None for o in range(N_OPS)]


def _target_null(P, table, ctx, post: np.ndarray, removed: int) -> np.ndarray:
    """`posterior.target_null`, with a context whose posterior underflowed to all on *removed* sent to the
    uniform over the other ops, where the renormalization would divide by zero.
    """
    lost = np.delete(post, removed, axis=1).sum(axis=1) <= 0
    safe = np.where(lost[:, None], 1.0, post)
    return P.target_null(table, ctx, safe, removed)


def _binned(post_a: np.ndarray, values: np.ndarray, edges: tuple[float, ...]) -> dict:
    """Count and mean of *values* `(n, ...)` per bin of *post_a*; empty bins are NaN."""
    b = np.clip(np.digitize(post_a, edges[1:-1], right=False), 0, len(edges) - 2)
    n = [int((b == i).sum()) for i in range(len(edges) - 1)]
    mean = [values[b == i].mean(axis=0).tolist() if c else None for i, c in enumerate(n)]
    return {"n": n, "mean": mean}


def eval_one(checkpoint, holdout, k: int, label: str) -> dict:
    """Every measurement the report takes from one run, on the held-out contexts of its corpus condition, one
    context per forward pass (the mask arms with the mask on, since it is part of the model).

    On the completion contexts: expected exact match against the ceiling and floor of each context; the
    calibration KL from the Bayes predictive to the model answer distribution at the query `=`; the mean
    alignment per op, slice, and role; the op margin; the alignment at the query `?` and the query `=` against
    the posterior on the anchored op, per context and binned; and the first-operand and trailing-fragment leans,
    per fragment start and pooled. On the verification lines, if any: verification accuracy. Summaries are
    returned as JSON; per-context arrays go to the store.
    """
    from sca.anchoring import alignment
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label
    model, tokenizer, color_ids, tok2color = _load_run(checkpoint, workdir)
    P = _get_posterior()
    table = P.build_table(TABLE)
    ho = load_holdout(get(holdout, workdir / "holdout.npz"), k)
    a_id = OP_NAMES.index(ANCHORED_OP)
    ctx = _post_queries(P, ho, k, tok2color)
    final = len(model.transformer.blocks)
    q_role, eq_role = query_role(k, QUERY_Q), query_role(k, QUERY_EQ)

    # --- The task and the calibration check --------------------------------------------------------------
    identity = projection(Subspace.axis(model.transformer.wte.shape[1]), 0.0)
    p = _color_probs(logits_at(model, ho.tokens, identity, (), eq_role), color_ids)
    eem = _match(table, ctx, p)
    floor = P.floor(table, ctx)
    kl = P.kl(P.predictive(table, ctx, ho.posterior), p)
    task = {
        name: {"all": float(x.mean()), "per_op": _per_op(x, ho.op_ids)}
        for name, x in (("eem", eem), ("ceiling", ho.ceiling), ("floor", floor), ("kl", kl))
    }
    task["color_mass"] = {"all": float(p.sum(1).mean())}

    # --- Alignment by op, slice, and role; the op margin ---------------------------------------------------
    # REVIEW: alignment, the margin, and both leans are measured on the completion contexts alone, so on the
    # verification corpus they leave its verification lines out and compare with the other arms like for like.
    cos = alignment(model, ho.tokens)  # (L1, N, T)
    anchored = ho.op_ids == a_id
    weights = anchored / anchored.sum()
    by_slice = np.einsum("n,lnt->lt", weights, cos) - cos.mean(axis=1)  # (L1, T)
    role_mean = np.stack([cos[:, ho.op_ids == o].mean(axis=1) for o in range(N_OPS)])  # (ops, L1, T)

    # --- Alignment against the posterior on the anchored op ------------------------------------------------
    post_a = ho.posterior[:, a_id]
    at_query = cos[:, :, [q_role, eq_role]].transpose(1, 0, 2)  # (N, L1, 2): the query `?`, the query `=`
    evidence = {
        "edges": list(EVIDENCE_BINS),
        "anchored": _binned(post_a[anchored], at_query[anchored], EVIDENCE_BINS),
        "other": _binned(post_a[~anchored], at_query[~anchored], EVIDENCE_BINS),
    }

    # --- The leans ------------------------------------------------------------------------------------------
    fragments, total, count = [], 0.0, 0
    for s in fragment_starts(k):
        fc = alignment(model, np.ascontiguousarray(ho.tokens[:, s:]))  # (L1, N, T − s)
        total += float(fc[final].sum())
        count += fc.shape[1] * fc.shape[2]
        fragments.append(
            {
                "start": s,
                "lean": float(fc[final].mean()),
                "lean_by_slice": fc.mean(axis=(1, 2)).tolist(),
                "profile": fc.mean(axis=1).tolist(),  # (L1, T − s): where along the fragment the lean sits
            }
        )
    leans = {
        "op1": float(cos[final, :, 0].mean()),
        "op1_by_slice": cos[:, :, 0].mean(axis=1).tolist(),
        "fragment": total / count,
        "fragments": fragments,
    }

    # --- Verification accuracy -------------------------------------------------------------------------------
    verify = None
    if len(ho.verify_tokens):
        t_id, f_id = tokenizer.stoi["TRUE"], tokenizer.stoi["FALSE"]
        lg = logits_at(model, ho.verify_tokens, identity, (), query_role(k, VERDICT_MARK)).astype(np.float64)
        said_true = lg[:, t_id] > lg[:, f_id]
        right = said_true == ho.verdict
        lg -= lg.max(axis=1, keepdims=True)
        pv = np.exp(lg) / np.exp(lg).sum(axis=1, keepdims=True)
        mass = np.where(ho.verdict, pv[:, t_id], pv[:, f_id])
        verify = {
            "n": int(len(right)),
            "accuracy": float(right.mean()),
            "accuracy_per_op": _per_op(right.astype(float), ho.verify_op_ids),
            "accuracy_by_verdict": {"TRUE": float(right[ho.verdict].mean()), "FALSE": float(right[~ho.verdict].mean())},
            "verdict_mass": float(mass.mean()),
        }
        # REVIEW: accuracy is the verdict whose logit is larger of the two, TRUE or FALSE, rather than the argmax over
        # the whole vocabulary; `verdict_mass` is the whole-vocabulary probability of the right verdict beside it.

    arrays = put(
        _npz(
            op_ids=ho.op_ids.astype(np.int8),
            posterior_anchored=post_a.astype(np.float32),
            eem=eem.astype(np.float32),
            ceiling=ho.ceiling.astype(np.float32),
            floor=floor.astype(np.float32),
            kl=kl.astype(np.float32),
            align_query=at_query.astype(np.float16),
        ),
        name=f"ex-2.2.16-{label}-eval.npz",
    )
    return {
        "label": label,
        "k": k,
        "n": int(len(ho.tokens)),
        "n_per_op": [int((ho.op_ids == o).sum()) for o in range(N_OPS)],
        "roles": {"query ?": q_role, "query =": eq_role},
        "task": task,
        "margin": {
            "value": context_margin(cos, weights),
            "by_slice": by_slice.max(axis=1).tolist(),
            "role_by_slice": by_slice.argmax(axis=1).tolist(),
        },
        "alignment": role_mean.tolist(),
        "evidence": evidence,
        "leans": leans,
        "verify": verify,
        "arrays": arrays,
    }


def suppression_edits(sub) -> list[tuple[str, float, Any]]:
    """Every operator at every dose rule (e) scores: the projection along γ, the repulsion along its landing,
    and the reflection, as `(operator, dose, edit)`.
    """
    from sca.intervention import projection, repulsion

    return [
        *(("projection", g, projection(sub, g)) for g in DOSE_GAMMAS),
        *(("repulsion", b, repulsion(sub, REPULSION_THRESHOLD, b, REPULSION_KIND)) for b in REPULSION_LANDINGS),
        ("reflection", REFLECT_GAMMA, projection(sub, REFLECT_GAMMA)),
    ]


def site_positions(site: str, k: int) -> np.ndarray:
    """The positions an edit at *site* touches, as a mask over a completion context. Every position is a mask of
    ones rather than no mask, so all three sites share one compiled program per operator.
    """
    mask = np.zeros(context_length(k), dtype=np.float32)
    match site:
        case "query ?":
            mask[query_role(k, QUERY_Q)] = 1
        case "query =":
            mask[query_role(k, QUERY_EQ)] = 1
        case "every position":
            mask[:] = 1
        case _:
            raise ValueError(site)
    return mask


def suppress_one(checkpoint, holdout, k: int, label: str) -> dict:
    """The suppression pass on one run: each operator at each dose and site (`EDIT_SITES`), acting at every slice,
    scored on the held-out completion contexts of every op. Per context: expected exact match on the clean pass,
    under each edit, and under the target null (the posterior with the anchored op removed); and the KL from the
    target-null predictive to the clean and edited answer distributions. Per-op means are returned; per-context
    arrays go to the store.
    """
    from sca.intervention import Subspace, logits_at
    from mini.store import get, put

    workdir = get_data_dir() / "suppress" / label
    model, _, color_ids, tok2color = _load_run(checkpoint, workdir)
    P = _get_posterior()
    table = P.build_table(TABLE)
    ho = load_holdout(get(holdout, workdir / "holdout.npz"), k)
    a_id = OP_NAMES.index(ANCHORED_OP)
    ctx = _post_queries(P, ho, k, tok2color)
    eq_role = query_role(k, QUERY_EQ)
    sub = Subspace.axis(model.transformer.wte.shape[1])
    every = tuple(range(len(model.transformer.blocks) + 1))

    null_post = _target_null(P, table, ctx, ho.posterior, a_id)
    q_null = P.predictive(table, ctx, null_post)
    null_eem = P.expected_match(table, ctx, null_post)

    edits = suppression_edits(sub)
    p = _color_probs(logits_at(model, ho.tokens, edits[0][2], (), eq_role), color_ids)
    clean_eem, clean_kl = _match(table, ctx, p), P.kl(q_null, p)

    records, eem, kl_null = [], [], []
    for site in EDIT_SITES:
        mask = site_positions(site, k)
        for operator, dose, edit in edits:
            p = _color_probs(logits_at(model, ho.tokens, edit, every, eq_role, mask), color_ids)
            e, d = _match(table, ctx, p), P.kl(q_null, p)
            eem.append(e)
            kl_null.append(d)
            records.append(
                {
                    "operator": operator,
                    "dose": dose,
                    "site": site,
                    "eem": _per_op(e, ho.op_ids),
                    "drop": _per_op(clean_eem - e, ho.op_ids),
                    "kl_null": _per_op(d, ho.op_ids),
                }
            )
    arrays = put(
        _npz(
            op_ids=ho.op_ids.astype(np.int8),
            clean_eem=clean_eem.astype(np.float32),
            null_eem=null_eem.astype(np.float32),
            clean_kl_null=clean_kl.astype(np.float32),
            eem=np.stack(eem).astype(np.float32),  # (edit, N), in the order of `edits`
            kl_null=np.stack(kl_null).astype(np.float32),
        ),
        name=f"ex-2.2.16-{label}-suppression.npz",
    )
    return {
        "label": label,
        "k": k,
        "slices": list(every),
        "clean": {"eem": _per_op(clean_eem, ho.op_ids), "kl_null": _per_op(clean_kl, ho.op_ids)},
        "null": {"eem": _per_op(null_eem, ho.op_ids)},
        "edits": records,
        "arrays": arrays,
    }


def design_c() -> dict[str, Any]:
    """The stage C constants the report reads beside the eval and suppression results."""
    return {
        "experiment": "ex-2.2.16",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "final_slice": FINAL_SLICE,
        "offsets": {"query ?": QUERY_Q, "query =": QUERY_EQ, "verdict mark": VERDICT_MARK},
        "evidence_bins": list(EVIDENCE_BINS),
        "middle_band": list(MIDDLE_BAND),
        "suppression": {
            "arms": list(SUPPRESSION_ARMS),
            "sites": list(EDIT_SITES),
            "gammas": list(DOSE_GAMMAS),
            "landings": list(REPULSION_LANDINGS),
            "threshold": REPULSION_THRESHOLD,
            "kind": REPULSION_KIND,
            "reflect_gamma": REFLECT_GAMMA,
            "selectivity_gate": SELECTIVITY_GATE,
            "grading_min_damage": GRADING_MIN_DAMAGE,
        },
        "rules": {
            "ceiling_margin": CEILING_MARGIN,
            "seed_band_sd": SEED_BAND_SD,
            "margin_keep": MARGIN_KEEP,
            "saturation_level": SATURATION_LEVEL,
            "kl_calibrated": KL_CALIBRATED,
            "hinge_cap": HINGE_CAP,
        },
    }


def publish_evaluation(runs: list[dict], evaled: list[dict], suppressed: list[dict]) -> dict:
    """The eval and suppression results, each as one JSON under its ref (the per-context arrays referenced by
    label), and every per-context array under its own ref. *runs* names the arm, seed, and corpus condition
    of each evaluated run. Stage B's refs are left as they are.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: r for r in runs}

    def slim(r: dict) -> dict:
        return meta[r["label"]] | {k: v for k, v in r.items() if k != "arrays"}

    for r in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    for r in suppressed:
        set_ref(SUPPRESSION_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    body = {"design": design_c(), "runs": [slim(r) for r in evaled]}
    set_ref(EVAL_REF, put(json.dumps(body).encode(), name="ex-2.2.16-eval.json"))
    body = {"design": design_c(), "runs": [slim(r) for r in suppressed]}
    set_ref(SUPPRESSION_REF, put(json.dumps(body).encode(), name="ex-2.2.16-suppression.json"))
    return {"n_evaled": len(evaled), "n_suppressed": len(suppressed)}


# --- Orchestration ----------------------------------------------------------------------------


def run(
    ctx: Ctx,
    arms: tuple[Arm, ...],
    n_lines: int,
    holdout_n: int,
    probe_n: int,
    seeds: int,
    epochs: int,
    traj_stride: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Prepare every corpus condition, then train *arms*: the whole DAG, with the corpus size, the held-out and
    probe counts, the seeds, the length, and the trajectory stride as arguments so a short prototype runs the
    same code (the smoke test overrides all seven).
    """
    conditions = (*((k, rho, False) for k, rho in GRAMMAR_CONDITIONS), (*CENTRE, True))
    preps = {
        cond_key(k, rho, verify): ctx.run(
            prepare_corpus_condition,
            k,
            rho,
            VERIFY_RATE if verify else 0.0,
            CUBE_RATE,
            n_lines,
            CORPUS_SEED + i,
            HOLDOUT_SEED + i,
            holdout_n,
            PROBE_SEED + i,
            probe_n,
            cond_key(k, rho, verify),
            role="prep",
        )
        for i, (k, rho, verify) in enumerate(conditions)
    }
    rows = cells(arms, preps, seeds, epochs)
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["variant"] for r in rows],
        [r["anchor_slices"] for r in rows],
        [preps[r["corpus_key"]]["corpus"] for r in rows],
        [preps[r["corpus_key"]]["labels"] for r in rows],
        [preps[r["corpus_key"]]["probes"] for r in rows],
        [r["k"] for r in rows],
        [traj_stride] * n,
        [r["label"] for r in rows],
        role="train",
    )
    return trained, preps


def run_rows(arms: tuple[Arm, ...], seeds: int) -> list[dict]:
    """Who each run is, in the order `cells` lists them (and so `run` returns them): the arm, the seed, the
    label, and the corpus condition.
    """
    return [
        {
            "label": f"{a.name}-s{seed}",
            "arm": a.name,
            "seed": seed,
            "corpus_key": cond_key(*(a.condition if a.group == "control" else CENTRE), a.verify),
            "k": (a.condition if a.group == "control" else CENTRE)[0],
        }
        for a in arms
        for seed in range(min(a.seeds, seeds))
    ]


def evaluate(
    ctx: Ctx, arms: tuple[Arm, ...], trained: list[dict], preps: dict[str, dict], seeds: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stage C over stage B's results: the eval on every run, then the suppression pass on the runs of
    `SUPPRESSION_ARMS`. Keyed on each checkpoint and held-out set, so a re-run serves both from the store.
    """
    rows = run_rows(arms, seeds)
    assert [r["label"] for r in rows] == [t["label"] for t in trained], "rows follow the order of `cells`"
    ckpt = [t["checkpoint"] for t in trained]
    holdout = [preps[r["corpus_key"]]["holdout"] for r in rows]
    evaled = ctx.map(eval_one, ckpt, holdout, [r["k"] for r in rows], [r["label"] for r in rows], role="eval")
    sup = [i for i, r in enumerate(rows) if r["arm"] in SUPPRESSION_ARMS]
    suppressed = ctx.map(
        suppress_one,
        [ckpt[i] for i in sup],
        [holdout[i] for i in sup],
        [rows[i]["k"] for i in sup],
        [rows[i]["label"] for i in sup],
        role="suppress",
    )
    return rows, evaled, suppressed


def main(ctx: Ctx) -> dict:
    trained, preps = run(ctx, ARMS, N_LINES, HOLDOUT_CONTEXTS, N_TRAJ_PROBE, SEEDS, EPOCHS, TRAJ_STRIDE)
    published = ctx.run(publish_results, trained, preps, role="prep")
    rows, evaled, suppressed = evaluate(ctx, ARMS, trained, preps, SEEDS)
    return published | ctx.run(publish_evaluation, rows, evaled, suppressed, role="prep")


COMPUTE = {
    # Four corpus builds (~300k contexts each) and the posterior scan over their anchored contexts and
    # held-out sets; and the fan-ins that write every ref.
    "prep": dict(cpu=2, timeout=1800),
    # 13,000 to 16,500 steps at L4 (the budget section works the arithmetic), the trailing-fragment lean measured
    # at every trajectory point; the watchdog covers the checkpoint upload.
    "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
    # Forward passes only, over 22,000 held-out contexts: the eval makes k + 3 of them, the suppression pass 22
    # (seven edits at three sites, and the clean pass). On four local CPU cores at k = 3 they take about 16 s and
    # 94 s, so an L4 spends most of its time starting up and the timeouts leave a wide margin.
    "eval": dict(gpu="L4", timeout=900),
    "suppress": dict(gpu="L4", timeout=1200),
}

experiment = Experiment(name="ex-2.2.16", main=main, roles=COMPUTE)
