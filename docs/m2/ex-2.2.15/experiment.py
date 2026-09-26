"""Ex-2.2.15: lines cut short by the training window — a scouting run of crop policies for the anchor.

Training windows are random crops of the packed corpus, so a line at either end of a window is often cut
short, and the pooled anchor term puts the whole of a labelled line's pull on whatever part of it is visible.
On a line cut before its op word, that part cannot know the op. This run trains ex-2.2.14's primary (anchored
`difference`, whole-line pull, label rate 0.02) under a few policies for which cut lines the anchor pulls, and
measures what each one does to the first operand's lean, to fragments seen without their op word, to the anchor,
and to the task. It proposes a default crop policy for the in-context grammar pilot.

The design constants come first; the DAG follows, binding what it does not change from ex-2.2.14's module.

    bin/mini run docs/m2/ex-2.2.15/experiment.py --app modal --max-containers 12 --budget 4h
    bin/mini status ex-2.2.15
"""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from mini import Ctx, Experiment, get_data_dir

# --- What is inherited ---------------------------------------------------------------------------------

REFERENCE_EXPERIMENT = "m2/ex-2.2.14"
"""Everything but the crop policy and the window: ex-2.2.14's primary, `anchor-diff`. That is ex-2.2.11's
handover recipe (table A+, the stochastic corpus at 300k lines, the untied readout, λ_a 0.1 annealed to a
0.1 floor over the last tenth of training, τ 0.1, the anti-subspace schedule, 50 epochs at d64-L4) with no
*red* anchor, `difference` on e₁, `op` keying at `LABEL_RATE`, and the pull over the whole line."""

REFERENCE_CONDITION = "anchor-diff"
REFERENCE_OP1_LEAN = 0.19
REFERENCE_OP1_LEAN_CONTROL = -0.01
"""Ex-2.2.14's post hoc measurement, at the final slice: the mean cosine with e₁ at the first operand over every
op's probe lines, on the primary and on the control. The op-word arm, which never pulls the first operand,
sat at -0.04. Quoted from the published report; the report here measures its own `all` arm."""

SMOKE_RED_LEAN = {"all": 0.232, "whole": 0.165, "cut-only": 0.083, "control": -0.002}
SMOKE_RED_OP2 = {"all": 0.12, "whole": 0.04, "cut-only": 0.13}
"""A smoke test run before the freeze, on ex-2.2.11's *red* handover (red on e₁, labelled by color over the
whole line) at its first two seeds (model seeds 100 and 101), under `all`, `whole`, and `cut-only`: the
seed-mean lean at the final slice, and the same mean cosine at the second operand. `all` matched ex-2.2.11's
stored runs at every trajectory point, which checks the new code path. The control is ex-2.2.11's at the same
seeds. The prototype is not committed; it ran on production storage as `ex-2.2.15-smoke`."""

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
BATCH = 64
PADDING_CHANCE = 0.1
"""Ex-2.1.3's data config, unchanged since: 64 windows of 64 tokens at a uniform offset, a tenth of them with
a zeroed prefix of 1 to `block // 3 - 1` tokens."""

SHORT_BLOCK = 32
SHORT_BATCH = 2 * BATCH
"""The stress windows. At 32 tokens a window holds about five and a third lines, which is about the ratio of
window to line the in-context grammar would have with ~20-token contexts in 128-token windows, so the share
of line visits cut short roughly doubles (see `visit_shares`). The batch doubles so a step sees the same
number of tokens, and so about the same number of labelled lines: the steps per epoch, the schedules, and the
anchor's updates then match the long arms."""
# REVIEW: the short arms used to keep the batch at 64, which doubled their steps per epoch and so the anchor's
# updates, and confined their comparisons to each other. Sandy suggested doubling the batch instead (review
# of f2f8e41). Verify: a doubled batch holds a few more line visits per step than a long batch (a window has
# a visit at each edge whatever its size), so the labelled lines per step run about 7% higher.


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
    """One training arm. `tie` ties the readout to the embedding table. `line_mask` stops attention at each
    newline, so a position attends to earlier positions of its own line only, and the state at the first
    operand of a whole line is a function of that token alone. The model has no such mask yet: the DAG adds it
    as a model config flag, with a test, and every measurement on that arm runs with it on, since it is part of
    the model.
    """

    name: str
    policy: Policy
    block: int = BLOCK
    batch: int = BATCH
    tie: bool = False
    line_mask: bool = False


ARMS: tuple[Arm, ...] = (
    Arm("all", "all"),
    Arm("whole", "whole"),
    Arm("half", "half"),
    Arm("scaled", "scaled"),
    Arm("knowable", "knowable"),
    Arm("cut-only", "cut-only"),
    Arm("all-short", "all", SHORT_BLOCK, SHORT_BATCH),
    Arm("whole-short", "whole", SHORT_BLOCK, SHORT_BATCH),
    Arm("whole-mask", "whole", line_mask=True),
    Arm("whole-tied", "whole", tie=True),
    Arm("all-tied", "all", tie=True),
)
"""`all` is today's behaviour and should reproduce ex-2.2.14's primary. `whole`, `half`, and `scaled` are the
grammar-agnostic policies; `knowable` needs to know where the evidence is, which the in-context grammar can
only approximate through the posterior. `cut-only` pulls the cut lines alone, so that `whole` and `cut-only`
split `all`'s pull in two. The short pair repeats `all` against `whole` where cut lines are twice as common.
The model arms are `whole` with one change to the model each, to test two routes for whatever lean `whole`
leaves: `whole-mask` stops attention at each newline, so a position sees only its own line; `whole-tied` ties
the readout to the embedding table, as ex-2.2.9's `handover-tied`. `all-tied` pairs with `whole-tied`, so the
cut lines' share of the lean can be measured under the tied readout too."""
# REVIEW: `whole-mask` and `whole-tied` were added after Sandy's review of 2543d1b, and `all-tied` after the
# round that followed. They sit outside the rule:
# the rule chooses a crop policy, and these change the model. Verify: a reader who wants the pilot's optional
# newline mask decided here could argue for a branch of the rule that adopts it.


MITIGATIONS = ("scaled", "half", "whole")
"""The policies the rule chooses among, in the order it prefers them: by how far each carries to a labelled
span longer than the window, as a natural-language document often is. `whole` never pulls such a span, `half`
stops at twice the window, and `scaled` pulls every span by the share in view. `knowable` is the reference
for what knowing the evidence buys."""

SEEDS = 5
SEED_OFFSET = 400
"""Arm seed *i* trains at model seed `SEED_OFFSET + i`, the same five seeds for every arm. Ex-2.2.14 trained
at 300, so every seed here is fresh. Five seeds because the lean is a difference of about 0.2 and the task
gate needs ex-2.2.14's resolution."""

N_RUNS = SEEDS * len(ARMS)
assert N_RUNS == 55

# --- What is scored ------------------------------------------------------------------------------------

FINAL_SLICE = 4
"""Slice 4 is the output of the last block, where ex-2.2.14 measured the lean."""

TRAJ_STRIDE = 50
"""Recorded at every trajectory point (every `TRAJ_STRIDE` training steps, on ex-2.2.14's probe lines), at every
slice: the op margin as ex-2.2.14 recorded it, plus the lean and the trailing-fragment lean. The measurements
at the end of training say where each policy lands; these say how it got there, since a cut line is a small
share of any one batch and the question is what the repeated pull on them adds up to. The stride is ex-2.2.14's
(inherited from ex-2.2.3), about a hundred points over a run; no checkpoint is kept along the way."""

READABLE_LEAN = 0.10
"""H1 is readable only if `all`'s lean exceeds the control's by at least this much, half of ex-2.2.14's
excess. Below it the lean did not reproduce at these seeds, and H1 is unresolved."""

LEAN_BAND = 0.05
"""A policy removes the lean when its seed-mean lean at the final slice is within this of the control's.
A quarter of ex-2.2.14's excess."""

CUT_ONLY_SHARE = 0.5
"""H1's second half: `cut-only` keeps at least this share of `all`'s excess lean."""

PARTIAL_SHARE = 0.3
"""H1 is partial when `whole` removes at least this share of `all`'s excess lean, and `cut-only` keeps at
least this share of it: the cut lines are one route for the lean, carrying a real part of it, among others.
On *red* the smoke test sat just under it (`whole` removed 29%, `cut-only` kept 36%)."""
# REVIEW: the partial band follows Sandy's review of 2543d1b ("e.g. 30% attribution to cut lines"), so that
# H1 keeps its strong prediction and the rule can still adopt a policy that only improves on `all`. It was set
# after the red smoke test, which lands at 29%. Verify: a reader who takes the smoke as a pilot for the gate
# could argue it was set with the red result in view; the op is the stronger test, since its first operand
# carries no evidence.

SCALED_SHARE = 0.5
"""`scaled` goes forward if it removes at least this share of the excess lean that `whole` removes. It keeps a
sixth of the pull on a visit that shows only the first operand, so some lean may stay, and the rule accepts
part of it for a policy that carries to longer spans. Stated against `whole`, it loosens with H1: if `whole`
removes nearly all of the excess, `scaled` must remove about half, as the fixed band of 0.1 asked before."""

MARGIN_KEEP = 0.9
"""H2: every policy's seed-mean op margin is at least this share of `all`'s."""

TASK_GATE = 0.02
"""H2: for each op, the seed-mean held-out expected exact match within this of the control's. Ex-2.2.11's
gate, unchanged."""

FRAGMENT_START_ROLES = (2, 3, 4, 5)
"""The trailing-fragment probe: each probe line shown from op2, `=`, the answer, or the newline on, as a sequence of
its own, so the op word is out of sight. These are the start-cut runs a training window leaves without their
op word."""

ADOPTION = (
    "If H1 passes or is partial, the pilot's default crop policy is `scaled` if it removes at least half the "
    "share of the excess lean (H1) that `whole` removes, and it passes H2. Otherwise it is whichever of `half` "
    "and `whole` keeps more of the pull while it removes as much of the lean as H1 asks of `whole` (to within "
    "the band of the control if H1 passed, at least the partial share if it was partial) and passes H2. If H1 "
    "missed, or no policy qualifies, the pilot keeps `all`, and leans on label variant (c) for the evidence "
    "question. If H1 is unresolved, the choice falls to the short pair: "
    "`whole` goes forward if the lean of `all-short` exceeds the control's by the readable margin, the lean of "
    "`whole-short` is within the band of it, and `whole` passes H2; `all` stays otherwise."
)
# REVIEW: the fallback branch carries the same gates as the main one (a readable lean on `all-short`, the band
# on `whole-short`, H2 on `whole`). Verify: a reader who wants the short pair's own task gap in place of
# `whole`'s H2 can argue it.
# REVIEW: `scaled` comes first under a looser test, after Sandy's review of f2f8e41: `whole` may give the
# cleanest result, but it never pulls a span longer than the window, which natural-language documents often
# are. After the review of 2543d1b the test is relative to `whole` (SCALED_SHARE) in place of a fixed band of
# 0.1, so it loosens when H1 is partial, as Sandy asked. Verify: under a partial H1 `scaled` may go forward
# removing 15% of the excess, about 0.03 in cosine, which five paired seeds may not resolve from zero.


# =============================================================================================
# The DAG
# =============================================================================================


def _load_ex2214():
    """Ex-2.2.14's module (which loads ex-2.2.11's, and so on down), by path and left out of `sys.modules`, so
    the task bodies here still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.14" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2214", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2214 = _load_ex2214()

# --- What stays as ex-2.2.14 had it -------------------------------------------------------------
# Bound by name so a task body never references the module objects themselves.

OP_NAMES: tuple[str, ...] = ex2214.OP_NAMES
EPOCHS = ex2214.EPOCHS
LAM = ex2214.LAM
TAU = ex2214.TAU
N_LINES = ex2214.N_LINES
N_EMBD = ex2214.N_EMBD
N_LAYER = ex2214.N_LAYER
ANNEAL_WEIGHT_RATIO = ex2214.ANNEAL_WEIGHT_RATIO
PROBE_RIDGE = ex2214.PROBE_RIDGE
N_SCAN = ex2214.N_SCAN
SCAN_SEED = ex2214.SCAN_SEED
SCAN_HOLDOUT = ex2214.SCAN_HOLDOUT
CORPUS_ARGS = (
    ex2214.OP_NAMES,
    ex2214.N_LINES,
    ex2214.CORPUS_SEED,
    ex2214.HOLDOUT_FRAC,
    ex2214.ROUNDING,
    ex2214.N_PROBE,
    ex2214.PROBE_SEED,
    ex2214.PROBE_BOTH_SLOTS,
    ex2214.PER_SLOT_RATE,
    ex2214.RED_RATE,
    ex2214.RED_DOSE,
    ex2214.NONRED_DOSE,
    ex2214.FAR_MOVE,
)
"""The grammar, the corpus, the probe sets, and the recipe: as ex-2.2.14 had them."""

assert ex2214.ANCHORED_OP == ANCHORED_OP and ex2214.LABEL_RATE == LABEL_RATE and ex2214.KEYING == "op"
assert ex2214.TASK_GATE == TASK_GATE and ex2214.TRAJ_STRIDE == TRAJ_STRIDE
assert ex2214.CONTROL_EXPERIMENT == CONTROL_EXPERIMENT and ex2214.CONTROL == CONTROL
assert ex2214.PRIMARY == REFERENCE_CONDITION and ex2214.WHOLE_SPAN == LINE_TOKENS
assert ex2214.ANCHOR_AXIS == 0 and ex2214.OP_POSITION == OP_ROLE

prepare_corpus = ex2214.prepare_corpus
traj_probe = ex2214.traj_probe
probe_walk = ex2214.probe_walk
line_margin = ex2214.line_margin
op_scan = ex2214.op_scan
anneal_retention = ex2214.anneal_retention
resolve_control = ex2214.resolve_control
_behavior = ex2214._behavior
_load = ex2214._load
_slim = ex2214._slim
_make_config = ex2214._make_config
schedules = ex2214.schedules
Condition229 = ex2214.Condition229
EX2211_CHECKPOINT_REF = ex2214.EX2211_CHECKPOINT_REF

# --- Refs -----------------------------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.2.15/metrics"
TRAJ_REF = "reports/m2/ex-2.2.15/trajectories"
CHECKPOINT_REF = "reports/m2/ex-2.2.15/checkpoints/{label}"

PULL_START_ROLES = (1, *FRAGMENT_START_ROLES)
"""The start-cut runs a window can leave: the trailing fragments, and the run that starts at the op word, which
the pull landing needs too (a crop at role 1 keeps its op word, so it is not a trailing fragment)."""


def cells(arms: tuple[Arm, ...], prep: dict, epochs: int = EPOCHS, seeds: int = SEEDS) -> list[dict]:
    """One row per run: ex-2.2.14's primary row with the arm's crop policy, window, readout, and mask."""
    from sca.config import ModelConfig
    from sca.data.named_colors import WordTokenizer
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    newline = WordTokenizer(tc).stoi["\n"]
    rates = {o: (LABEL_RATE if o == ANCHORED_OP else 0.0) for o in OP_NAMES}
    rows = []
    for a in arms:
        base = Condition229(a.name, seeds, a.name, lam=LAM, tau=TAU, epochs=epochs, ops=OP_NAMES, n_lines=N_LINES)
        anchor, anti = schedules(base)
        anchor = anchor | {"span": LINE_TOKENS}
        for seed in range(seeds):
            config = _make_config(align(tc.vocab_size, 64), SEED_OFFSET + seed, epochs, N_EMBD, N_LAYER)
            config.tokenizer = tc.model_copy()
            config.model = ModelConfig.model_validate(
                config.model.model_dump()
                | {
                    "block_size": a.block,
                    "tie_embeddings": a.tie,
                    "line_mask_token": newline if a.line_mask else None,
                }
            )
            assert config.data.batch_size == BATCH and config.data.padding_chance == PADDING_CHANCE
            config.data = config.data.model_copy(update={"batch_size": a.batch})
            rows.append(
                {
                    "config": config,
                    "anchor": anchor,
                    "anti": anti,
                    "rates": rates,
                    "crop": a.policy,
                    "condition": a.name,
                    "seed": seed,
                    "model_seed": SEED_OFFSET + seed,
                    "label": f"{a.name}-s{seed}",
                }
            )
    return rows


def fragments(tokens: np.ndarray, start: int) -> np.ndarray:
    """The probe lines shown from role *start* on, as sequences of their own: what a window that opens at that
    role sees of the line. Under causal attention the states of a window's first run depend on that run alone,
    so these are the states training saw on a start-cut line.
    """
    assert tokens.shape[1] == LINE_TOKENS, f"probe lines are {tokens.shape[1]} tokens, not {LINE_TOKENS}"
    return np.ascontiguousarray(tokens[:, start:])


def fragment_sums(model, tokens: np.ndarray, starts=FRAGMENT_START_ROLES) -> tuple[np.ndarray, int]:
    """The summed cosine with e₁ over every position of every fragment of *tokens*, per slice; and the count of
    positions it sums over. Sums, so a caller can pool ops and start roles weighted by position.
    """
    from sca.anchoring import alignment

    total, count = 0.0, 0
    for r in starts:
        cos = alignment(model, fragments(tokens, r))  # (L1, N, T - r)
        total = total + cos.sum(axis=(1, 2))
        count += cos.shape[1] * cos.shape[2]
    return np.asarray(total), count


def train_one(config, anchor: dict, anti: dict, corpus, traj_stride: int, probes, rates, crop: Policy, label: str):
    """Train one run under the op labeller and the crop policy, recording at every trajectory point the op
    margin (`m_line`), the cosine with e₁ per slice and role (`alpha_roles`, over every op's trajectory lines, so
    role 0 is the lean), and the trailing-fragment cosine per slice and op (`fragment`).
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    p = np.zeros(config.model.vocab_size)
    for o, r in rates.items():
        p[tokenizer.stoi[o]] = r
    with np.load(get(probes, workdir / "probes.npz")) as z:
        tokens, line_w = traj_probe(z, ANCHORED_OP)
    per_op = len(tokens) // len(OP_NAMES)
    by_op = [tokens[i * per_op : (i + 1) * per_op] for i in range(len(OP_NAMES))]

    fragment: list[np.ndarray] = []

    def on_record(_index: int, model) -> None:
        sums = [fragment_sums(model, t) for t in by_op]
        fragment.append(np.stack([s / n for s, n in sums], axis=1))  # (L1, op)

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti),
        label_p=LabelSpec(p=p, keying="op", pull="span"),
        probe_tokens=tokens,
        probe_weights=line_w,
        probe_line_w=line_w,
        crop=crop,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
        on_record=on_record,
    )
    keys = ("epoch", "m_line", "alpha_op1", "alpha_roles", "val_loss", "weight", "anti_weight")
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: np.asarray(traj[k]).tolist() for k in keys if k in traj}
        | {"fragment": np.stack(fragment).tolist()},
        "checkpoint": put(workdir / "model", name=f"ex-2.2.15-{label}-ckpt"),
    }


def pull_landing(model, tokens: np.ndarray, policy: Policy, block: int) -> np.ndarray:
    """(slice, role): where the pull on a labelled line lands, averaged over the ways a window shows the line.

    Each visit that shows roles `first..last` is weighted by how often it occurs (`visit_shares`) and by the
    weight *policy* puts on it; its pull is shared over the visible roles by the softmin weights at τ (from
    the op word on, under `knowable`). A visit's states are those of the fragment starting at `first`, since
    under causal attention a window's first run depends on that run alone and a later run's prefix is the
    whole line's. The arm's mask, if any, is part of the model and so part of those states.
    """
    from sca.anchoring import alignment, softmin_weights

    cos = {0: alignment(model, tokens)} | {r: alignment(model, fragments(tokens, r)) for r in PULL_START_ROLES}
    land = np.zeros((cos[0].shape[0], LINE_TOKENS))
    total = 0.0
    for (first, last), share in visit_shares(block).items():
        w = share * line_weight(policy, first, last)
        lo = max(first, KNOWABLE_FIRST_ROLE) if policy == "knowable" else first
        if w == 0 or lo > last:
            continue
        run = cos[first][:, :, lo - first : last - first + 1]  # (L1, N, roles lo..last)
        land[:, lo : last + 1] += w * softmin_weights(1.0 - run, TAU, axis=-1).mean(axis=1)
        total += w
    return land / total


def readout(model, tokenizer, tokens: np.ndarray) -> dict:
    """The tied-readout measurements, at the first operand of the probe lines at the final slice: the
    syntax-against-color log-odds of the next token, and the part the e₁ coordinate carries (op1-lean's
    readout gap); and the e₁ column of the embedding and readout tables, in vocabulary order.
    """
    import equinox as eqx
    import jax.numpy as jnp
    from scipy.special import logsumexp

    from sca.data.ops import PALETTE

    vocab = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
    colors = np.array([i for i, w in enumerate(vocab) if w in PALETTE])
    syntax = np.array([i for i, w in enumerate(vocab) if w and w not in PALETTE])

    def logodds(lg):
        return logsumexp(lg[..., syntax], axis=-1) - logsumexp(lg[..., colors], axis=-1)

    fwd = eqx.filter_jit(model.stream_and_logits)
    h, lg = [], []
    for i in range(0, len(tokens), 2048):
        s, logits = fwd(jnp.asarray(tokens[i : i + 2048]))
        h.append(np.asarray(s[FINAL_SLICE][:, 0], np.float64))
        lg.append(np.asarray(logits[:, 0], np.float64))
    h1, lg = np.concatenate(h)[:, 0], np.concatenate(lg)
    table = np.asarray(model.transformer.readout, np.float64)
    wte = np.asarray(model.transformer.wte, np.float64)
    wte /= np.linalg.norm(wte, axis=1, keepdims=True)
    s_z = float(np.asarray(model.s_z()).reshape(-1)[0])
    without = lg - s_z * h1[:, None] * table[:, 0]
    return {
        "vocab": vocab,
        "logodds_op1": float(logodds(lg).mean()),
        "logodds_op1_e1": float((logodds(lg) - logodds(without)).mean()),
        "readout_e1": table[:, 0].tolist(),
        "embedding_e1": wte[:, 0].tolist(),
    }


def eval_one(
    trained: dict, evals, probes, condition: str, policy: Policy | None, block: int, seed: int, label: str
) -> dict:
    """Ex-2.2.14's eval (behavior per op, the alignment per op's probe lines, the op margin, retention, and
    the op-identity scan), plus the trailing fragments per op and start role, where the pull lands under the
    arm's own policy and window, and the readout measurements.
    """
    from sca.anchoring import alignment
    from sca.data import ops as grammar
    from mini.store import get

    workdir = get_data_dir() / "eval" / label
    model, tokenizer, _, _ = _load(trained, workdir)
    sets = _behavior(model, tokenizer, grammar.load_lines(get(evals, workdir / "evals.json").read_bytes()))

    with np.load(get(probes, workdir / "probes.npz")) as z:
        lines = {o: probe_walk(z, o) for o in OP_NAMES}
    cos = {o: alignment(model, t) for o, t in lines.items()}  # each (L1, N, T)
    pooled = np.concatenate([cos[o] for o in OP_NAMES], axis=1)
    counts = [cos[o].shape[1] for o in OP_NAMES]
    assert len(set(counts)) == 1, f"probe walks differ in size: {counts}"
    w = np.concatenate([np.full(n, 1.0 if o == ANCHORED_OP else 0.0) for o, n in zip(OP_NAMES, counts, strict=True)])
    rng = np.random.default_rng(SCAN_SEED)
    scan_lines = {o: t[np.sort(rng.choice(len(t), N_SCAN, replace=False))] for o, t in lines.items()}
    fragment = {
        o: {str(r): alignment(model, fragments(t, r)).mean(axis=1).tolist() for r in FRAGMENT_START_ROLES}
        for o, t in lines.items()
    }  # per op and start role, (L1, T - r)
    out = {
        "label": label,
        "condition": condition,
        "seed": seed,
        "sets": sets,
        "holdout_eem": {o: s["holdout"]["eem"] for o, s in sets.items()},
        "cos_mean": {o: cos[o].mean(axis=1).tolist() for o in OP_NAMES},  # (L1, T) per op
        "op_margin": line_margin(pooled, w / w.sum()),
        "fragment": fragment,
        "readout": readout(model, tokenizer, np.concatenate(list(lines.values()))),
        "probe_r2": op_scan(model, scan_lines, PROBE_RIDGE, SCAN_SEED, SCAN_HOLDOUT).tolist(),
        "n_probe_lines": counts[0],
    }
    if policy is not None:
        out["pull_landing"] = pull_landing(model, lines[ANCHORED_OP], policy, block).tolist()
    if "traj" in trained:
        out |= anneal_retention(trained["traj"], ANNEAL_WEIGHT_RATIO)
    return out


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """The design constants the report reads beside the results."""
    return {
        "experiment": "ex-2.2.15",
        "reference": REFERENCE_EXPERIMENT,
        "control": {"experiment": CONTROL_EXPERIMENT, "condition": CONTROL, "seeds": CONTROL_SEEDS},
        "anchored_op": ANCHORED_OP,
        "label_rate": LABEL_RATE,
        "arms": [asdict(a) for a in ARMS],
        "pull_share": {a.name: pull_share(a.policy, a.block) for a in ARMS},
        "seed_offset": SEED_OFFSET,
        "n_runs": N_RUNS,
        "ops": list(OP_NAMES),
        "final_slice": FINAL_SLICE,
        "fragment_start_roles": list(FRAGMENT_START_ROLES),
        "adoption": ADOPTION,
    }


def publish_results(trained: list[dict], evaled: list[dict], corpus_stats: dict) -> dict:
    """Metrics (JSON), trajectories (JSON), and every end checkpoint, each under its ref."""
    import json

    from mini.store import put, set_ref

    metrics = {"runs": [_slim(r) for r in evaled], "corpus": corpus_stats, "design": design()}
    set_ref(METRICS_REF, put(json.dumps(metrics).encode(), name="ex-2.2.15-metrics.json"))
    traj = {t["label"]: {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.15-trajectories.json"))
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    return {"n_runs": len(trained), "n_evaled": len(evaled)}


# --- Orchestration ----------------------------------------------------------------------------


def run(
    ctx: Ctx, arms: tuple[Arm, ...], epochs: int, seeds: int, control_seeds: int
) -> tuple[list[dict], list[dict], dict]:
    """Train *arms*, then evaluate them beside the served control: the whole DAG, with the grid, the length,
    and the seeds as arguments so a short prototype runs the same code.
    """
    control_labels = [f"{CONTROL}-s{s}" for s in range(control_seeds)]
    control = ctx.run(resolve_control, control_labels, EX2211_CHECKPOINT_REF, role="prep")
    prep = ctx.run(prepare_corpus, *CORPUS_ARGS, role="prep")
    rows = cells(arms, prep, epochs, seeds)
    n = len(rows)
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [prep["corpus"]] * n,
        [TRAJ_STRIDE] * n,
        [prep["probes"]] * n,
        [r["rates"] for r in rows],
        [r["crop"] for r in rows],
        [r["label"] for r in rows],
        role="train",
    )
    ctrl = [{"checkpoint": control[lb], "label": lb} for lb in control_labels]
    m = n + len(ctrl)
    evaled = ctx.map(
        eval_one,
        trained + ctrl,
        [prep["evals"]] * m,
        [prep["probes"]] * m,
        [r["condition"] for r in rows] + [CONTROL] * len(ctrl),
        [r["crop"] for r in rows] + [None] * len(ctrl),
        [r["config"].model.block_size for r in rows] + [BLOCK] * len(ctrl),
        [r["seed"] for r in rows] + list(range(len(ctrl))),
        [r["label"] for r in rows] + control_labels,
        role="eval",
    )
    return trained, evaled, prep


def main(ctx: Ctx) -> dict:
    trained, evaled, prep = run(ctx, ARMS, EPOCHS, SEEDS, CONTROL_SEEDS)
    return ctx.run(publish_results, trained, evaled, prep["stats"], role="prep")


COMPUTE = {
    # Ex-2.2.9's corpus build, and the fan-in that writes the refs of fifty-five runs.
    "prep": dict(cpu=2, timeout=1800),
    # 4,950 steps at L4, with the trailing fragments measured at every trajectory point; the watchdog covers
    # the checkpoint upload.
    "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
    "eval": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="ex-2.2.15", main=main, roles=COMPUTE)
