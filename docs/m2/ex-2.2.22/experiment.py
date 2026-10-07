"""Ex-2.2.22: localized by depth, various pull caps, and contexts of varying length — a scout.

Ex-2.2.21 left three questions about how to anchor the inferred op. The `no-emb` condition, which leaves the
embedding slice out of the pull, had the best task score but edited less selectively. The hinge condition, whose
pull stops at an alignment of 0.8, edited selectively, and the uncapped condition just missed the gate, so a cap
between the two may hold more of the anchor and keep the selectivity. And the posterior on the op takes about three
distinct values given a whole context of three examples, too few to test whether the anchor grades with it. The hinge
condition is the best recipe so far, so every new condition here changes one setting from it (or from a new condition
of this scout), and ex-2.2.21's runs are reused as references, paired by model seed.

The design constants above the DAG were frozen with the report at commit 94873f2. The DAG reuses ex-2.2.21's corpus,
held-out set, probes, and the checkpoints of its reference conditions, by ref. It builds a corpus whose contexts draw
their example count per line and a held-out set at every other count, trains the new runs on ex-2.2.21's recipe, and
then scores every run, reused or new, with three passes: ex-2.2.21's eval; the projection at every position, with the
landing measurements of H2; and, on the conditions H1 and E3 read, the task score and the alignment at the answers at
every example count.

    bin/mini run docs/m2/ex-2.2.22/experiment.py --app modal --max-containers 12 --budget 4h
    bin/mini status ex-2.2.22
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir
from sca.data.incontext import context_length


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as in ex-2.2.16 to ex-2.2.21."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2221 = _load_sibling("ex-2.2.21", "ex2222_ex2221")
ex2216 = ex2221.ex2216
ex2218 = ex2221.ex2218
ex2217 = ex2221.ex2217

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = ex2221.OP_NAMES
ANCHORED_OP = ex2221.ANCHORED_OP
MODEL = ex2221.MODEL
N_LAYER = 4
"""The depth of `MODEL` (d64-L4): five slices, the embedding and the output of each block."""
assert MODEL.endswith(f"L{N_LAYER}")
N_SLICES = N_LAYER + 1

EPOCHS = ex2221.EPOCHS
CENTRE = ex2221.CENTRE
"""Three examples at ρ = 0.3: the corpus of every condition but `k-mixed`, and of every reference."""
K, RHO = CENTRE
LABEL_RATE = ex2221.LABEL_RATE
HINGE_SOFTNESS = ex2221.HINGE_SOFTNESS
VERIFY = False
"""No verification lines, as in the hinge condition of ex-2.2.21. Ex-2.2.21 (S2) found they leave
completion unchanged."""

SEED_BAND_SD = ex2221.SEED_BAND_SD
REGRESSION_TOL = ex2221.REGRESSION_TOL
SELECTIVITY_GATE = ex2221.SELECTIVITY_GATE
GRADING_MIN_DAMAGE = ex2221.GRADING_MIN_DAMAGE
DOSE_GAMMAS = ex2221.DOSE_GAMMAS
TASK_COST_TOL = ex2221.TASK_COST_TOL
"""The task tolerance, the seed band, and E2 of ex-2.2.21 (the grading criterion, the selectivity gate, and the dose
axis of the projection), unchanged."""

LAMBDA_A = 0.1
"""The anchor weight of the recipe (ex-2.2.14), before its anneal to a floor over the last tenth of training."""
assert LAMBDA_A == ex2216.LAM

# --- The conditions ----------------------------------------------------------------------------------------

SLICE_SETS: dict[str, tuple[int, ...]] = {
    "all": tuple(range(N_SLICES)),
    "no-emb": tuple(range(1, N_SLICES)),
    "no-last": tuple(range(N_SLICES - 1)),
    "middle": tuple(range(1, N_SLICES - 1)),
}
"""Which residual-stream slices the anchor and anti-subspace terms act on. Slice 0 is the embedding and slice 4 the
input to the readout. `middle` leaves out both."""


def per_slice_weight(slices: str) -> float:
    """The anchor weight that keeps the effective pull what it is with every slice pulled. Both terms average over
    the slices they act on, so dropping slices at a fixed weight pulls each remaining slice harder, by
    `N_SLICES / len(slices)`: by a quarter for `no-emb` and `no-last`, and by two thirds for `middle`.
    """
    return LAMBDA_A * len(SLICE_SETS[slices]) / N_SLICES


HINGE_CAP = ex2221.HINGE_CAP
"""The cap of ex-2.2.21's hinge condition, the base recipe here."""


@dataclass(frozen=True)
class Condition:
    """One condition: its settings, and the condition it differs from in one of them."""

    name: str
    reference: str
    """The condition this one differs from in one setting, paired by model seed: an ex-2.2.21 condition, or a new
    condition of this scout for the `-matched` twins."""
    anchored: bool = True
    slices: str = "all"
    weight: float = LAMBDA_A
    """The anchor weight before its anneal. The anneal floor and the anti-subspace schedule scale with it."""
    cap: float | None = HINGE_CAP
    counts: tuple[int, ...] = (K,)


MIXED_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The example counts of the `k-mixed` condition, one drawn uniformly per context. The mean is three, so the corpus
has the same number of tokens on average as the fixed-count corpus, and an epoch the same number of steps. Each count
adds levels to the posterior on `difference` given the whole context that the others lack (the report has the
figure).
One example contributes the most contexts in doubt, and pooled over the counts, the share of `difference` contexts
whose examples favour some other op is about what it is at three examples. Two whole contexts at five examples fit in
the window of 96 tokens."""

BASE = "anchor-hinge"

REFERENCES: tuple[Condition, ...] = (
    Condition("control", "", anchored=False, cap=None),
    Condition(BASE, ""),
    Condition("anchor-whole", "", cap=None),
    Condition("anchor-no-emb", "", slices="no-emb", cap=None),
)
"""The ex-2.2.21 runs this scout reuses, at the seeds below. `anchor-hinge` is the base of every new condition;
`anchor-whole` is the uncapped end of the cap ladder, and `anchor-no-emb` the uncapped twin of `no-emb`. The control
and `anchor-whole` had five seeds there; only the first three pair with this scout."""

CONDITIONS: tuple[Condition, ...] = (
    *(Condition(s, BASE, slices=s) for s in ("no-emb", "no-last", "middle")),
    *(Condition(f"{s}-matched", s, slices=s, weight=per_slice_weight(s)) for s in ("no-emb", "no-last", "middle")),
    Condition("cap-0.9", BASE, cap=0.9),
    Condition("cap-0.95", BASE, cap=0.95),
    Condition("k-mixed", BASE, counts=MIXED_COUNTS),
)
"""Every new condition is the hinge condition of ex-2.2.21 with one setting changed, except the `-matched` twins,
which change the weight from a slice condition of this scout."""

SEEDS = 3
SEED_OFFSET = ex2221.SEED_OFFSET
"""The model seeds of ex-2.2.21's three-seed conditions, 700 to 702, so each new run pairs with its reference by
seed. Seed 702 took the slow path through training on most anchored conditions there, so the pairing also shows
whether a setting changes that."""

REPLICATE_SEEDS: tuple[int, ...] = (703, 704, 705)
"""H2: three more runs of `anchor-hinge`, at seeds it was not trained at in ex-2.2.21, so H2 is scored on runs its
prediction was not drawn from. The first two pair with the five-seed control of ex-2.2.21."""
assert not set(REPLICATE_SEEDS) & set(range(SEED_OFFSET, SEED_OFFSET + SEEDS))

N_RUNS = SEEDS * len(CONDITIONS) + len(REPLICATE_SEEDS)
assert N_RUNS == 30


def settings(c: Condition) -> dict[str, object]:
    """The settings a design table compares between a condition and its reference."""
    return {"slices": c.slices, "weight": c.weight, "cap": c.cap, "counts": c.counts}


def by_name(name: str) -> Condition:
    return next(c for c in (*REFERENCES, *CONDITIONS) if c.name == name)


for _c in CONDITIONS:
    _diff = [k for k, v in settings(_c).items() if settings(by_name(_c.reference))[k] != v]
    assert len(_diff) == 1, (_c.name, _diff)

# --- The measurements --------------------------------------------------------------------------------------

LANDING_FRACTION = 0.5
"""H2: at full dose, the edit at every position closes at least this share of the clean model's distance from the
target null, in total variation, as a ratio of means over held-out `difference` contexts. The same share as the edit
criterion (`GRADING_MIN_DAMAGE`), on the same clean-to-null scale."""
assert LANDING_FRACTION == GRADING_MIN_DAMAGE

GRADING_SITES: tuple[str, ...] = ("example answers", "query answer")
"""E3: the roles where α (the alignment) is compared with the posterior on `difference` given the pairs up to and
including that answer. The example answers are where ex-2.2.21 (E1) found the anchor; at the query answer the state
already holds the answer, so its posterior counts the query pair too, and there ex-2.2.21 found the alignment climbing
with depth."""


# =============================================================================================
# The DAG
# =============================================================================================
#
# The fixed-count conditions train on ex-2.2.21's corpus, resolved by ref, so they differ from its runs in the one
# setting each changes. Ex-2.2.21's `eval_one` scores every run as it scored its own; the suppression pass and the
# by-count pass are new here.

CUBE_RATE = ex2221.CUBE_RATE
N_LINES = ex2221.N_LINES
HOLDOUT_CONTEXTS = ex2221.HOLDOUT_CONTEXTS
N_TRAJ_POINTS = ex2221.N_TRAJ_POINTS
N_TRAJ_EEM_PER_OP = ex2221.N_TRAJ_EEM_PER_OP
MAIN_KEY = ex2221.MAIN_KEY
MIXED_KEY = f"{MAIN_KEY}-mixed"

MIXED_SEED = 222_200
"""The `k-mixed` corpus at this seed, and the held-out set at count k at `MIXED_SEED + 10 + k`, so the two never draw
from one stream. No earlier experiment has used these seeds."""

REUSED_SEEDS: dict[str, int] = {"control": 5, BASE: SEEDS, "anchor-whole": SEEDS, "anchor-no-emb": SEEDS}
"""How many seeds of each reference condition are scored here, from 700. All five of the control, so the replicate
seeds 703 and 704 have a control at the same seed."""
assert set(REUSED_SEEDS) == {c.name for c in REFERENCES}

BY_COUNT_CONDITIONS: tuple[str, ...] = ("k-mixed", BASE, "anchor-whole", "control")
"""The conditions the by-count pass scores, at every count in `MIXED_COUNTS`: H1 reads the skill of `k-mixed` at each
count, and E3 the alignment at the answers of these four."""

CONFUSION_CONDITIONS: tuple[str, ...] = (BASE, "control")
"""The conditions whose answer distributions, clean and under the full edit, the suppression pass keeps per context,
for the op-confusion matrices of H2."""

PREFIX = "reports/m2/ex-2.2.22"
LABELS_REF = PREFIX + "/labels/{key}"
CORPUS_REF = PREFIX + "/corpus/{key}"
HOLDOUT_REF = PREFIX + "/holdout/k{k}"
METRICS_REF = PREFIX + "/metrics"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
EVAL_REF = PREFIX + "/eval"
EVAL_ARRAYS_REF = PREFIX + "/eval-arrays/{label}"
SUPPRESSION_REF = PREFIX + "/suppression"
SUPPRESSION_ARRAYS_REF = PREFIX + "/suppression-arrays/{label}"
BY_COUNT_REF = PREFIX + "/by-count"
BY_COUNT_ARRAYS_REF = PREFIX + "/by-count-arrays/{label}"


def resolve_reused(labels: list[str]) -> dict:
    """Ex-2.2.21's corpus condition (the corpus, labels, held-out set, and probes), the corpus metadata, and the
    checkpoints of *labels*, by ref. Resolving them rather than rebuilding records ex-2.2.21 as this experiment's
    lineage.
    """
    from sca.compute.data_pipelines import load_data
    from mini.store import get, get_ref

    refs = {
        "corpus": get_ref(ex2221.CORPUS_REF.format(key=MAIN_KEY)),
        "labels": get_ref(ex2221.LABELS_REF.format(key=MAIN_KEY)),
        "holdout": get_ref(ex2221.HOLDOUT_REF.format(key=MAIN_KEY)),
        "probes": get_ref(ex2221.PROBES_REF.format(key=MAIN_KEY)),
    }
    checkpoints = {label: get_ref(ex2221.CHECKPOINT_REF.format(label=label)) for label in labels}
    missing = [n for n, a in (refs | checkpoints).items() if a is None]
    corpus = refs["corpus"]
    assert corpus is not None and not missing, f"ex-2.2.21 has not published {missing}"
    _, meta = load_data(get(corpus, get_data_dir() / "resolve" / "corpus"))
    return refs | {"meta": meta, "checkpoints": checkpoints}


def prepare_mixed(
    ops: tuple[str, ...], counts: tuple[int, ...], rho: float, cube_rate: float, n_lines: int, seed: int, key: str
) -> dict:
    """The `k-mixed` corpus: `sample_corpus` with the example count drawn per context, uniformly from *counts*, from
    the same generator. The labels are the op and length of each context, all the whole-line label reads.
    """
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data.incontext import encode_corpus, sample_context, vocabulary
    from sca.data.incontext import op_ids as op_ids_of
    from sca.data.named_colors import WordTokenizer
    from mini.store import put

    table = ex2218.table_of(ops)
    rng = np.random.default_rng(seed)
    corpus = []
    for _ in range(n_lines):
        hidden_op = table[int(rng.integers(len(table)))]
        k = counts[int(rng.integers(len(counts)))]
        corpus.append(sample_context(table, hidden_op, k, rho, rng, cube_rate, ex2216.ROUNDING, 0.0))
    tokenizer_config = TokenizerConfig(vocabulary=sorted(vocabulary()))
    tokenizer = WordTokenizer(tokenizer_config)
    tokens = encode_corpus(corpus, tokenizer.stoi)
    context_op = op_ids_of(corpus, table)
    context_len = np.array([c.n_tokens for c in corpus], dtype=np.int32)
    n_examples = np.array([len(c.examples) for c in corpus])

    n_chars = sum(len(w) for c in corpus for w in c.words)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=n_chars,
        sources=[DatasetMetadata(title=f"in-context grammar corpus ({key})", fixes=[], total_chars=n_chars)],
    )
    corpus_dir = get_data_dir() / "corpora" / key
    save_data(tokens, meta, corpus_dir)
    return {
        "key": key,
        "meta": meta,
        "stats": {
            "key": key,
            "counts": list(counts),
            "n_lines": n_lines,
            "seed": seed,
            "total_tokens": int(len(tokens)),
            "count_share": [float((n_examples == k).mean()) for k in counts],
        },
        "corpus": put(corpus_dir, name=f"ex-2.2.22-{key}-corpus"),
        "labels": put(ex2216._npz(op_ids=context_op, context_len=context_len), name=f"ex-2.2.22-{key}-labels.npz"),
    }


def prepare_holdouts(
    ops: tuple[str, ...], counts: tuple[int, ...], rho: float, cube_rate: float, holdout_n: int, seed: int
) -> dict[int, dict]:
    """A held-out set at each count in *counts*, *holdout_n* completion contexts per op, in the format of ex-2.2.21's
    (so `ex2216.load_holdout` reads it): the tokens, the op, the length, the posterior over ops, and the ceiling.
    """
    from sca.data import ops as grammar
    from sca.data.incontext import encode_corpus, vocabulary
    from sca.data.incontext import op_ids as op_ids_of
    from sca.data.named_colors import WordTokenizer
    from sca.config import TokenizerConfig
    from mini.store import put

    P = ex2216._get_posterior()
    table = ex2218.table_of(ops)
    post_table = P.build_table(table)
    tokenizer = WordTokenizer(TokenizerConfig(vocabulary=sorted(vocabulary())))
    color_index = {c: i for i, c in enumerate(grammar.colors())}
    op_index = {o.name: i for i, o in enumerate(table)}
    out = {}
    for k in counts:
        holdout = ex2216._sample_by_op(table, k, rho, cube_rate, 0.0, holdout_n, seed + k)
        batch = ex2216._post_contexts(P, holdout, color_index, op_index)
        post = P.posterior(post_table, batch, rho, cube_rate)
        ceiling = P.expected_match(post_table, batch, post)
        floor = P.floor(post_table, batch)
        op = op_ids_of(holdout, table)
        out[k] = {
            "stats": {
                "k": k,
                "ceiling": float(ceiling.mean()),
                "floor": float(floor.mean()),
                "ceiling_per_op": [float(ceiling[op == o].mean()) for o in range(len(ops))],
                "floor_per_op": [float(floor[op == o].mean()) for o in range(len(ops))],
            },
            "holdout": put(
                ex2216._npz(
                    tokens=encode_corpus(holdout, tokenizer.stoi),
                    op_ids=op,
                    context_len=np.array([c.n_tokens for c in holdout], dtype=np.int32),
                    posterior=post,
                    ceiling=ceiling,
                    verify_verdict=np.full(len(holdout), -1, dtype=np.int8),
                ),
                name=f"ex-2.2.22-holdout-k{k}.npz",
            ),
        }
    return out


# --- Training ---------------------------------------------------------------------------------------


def cells(meta: dict[str, Any], seeds: int = SEEDS, replicate: tuple[int, ...] = REPLICATE_SEEDS, epochs: int = EPOCHS):
    """One row per new run: each condition at seeds 700 onward, then `BASE` at the replicate seeds. The config is
    ex-2.2.21's (`recipe_config`), so a run differs from its reference in the setting its condition changes and,
    for the replicate, the seed. *meta* holds each corpus key's metadata.
    """
    plan = [(c, SEED_OFFSET + s) for c in CONDITIONS for s in range(seeds)]
    plan += [(by_name(BASE), m) for m in replicate]
    rows = []
    for c, model_seed in plan:
        key = MAIN_KEY if c.counts == (K,) else MIXED_KEY
        base = ex2216.Condition229(
            c.name, 1, c.name, lam=c.weight, tau=ex2216.TAU, epochs=epochs, ops=OP_NAMES, n_lines=N_LINES,
        )  # fmt: skip
        anchor, anti = ex2216.schedules(base)
        if c.cap is not None:
            anchor = anchor | {"hinge": (c.cap, HINGE_SOFTNESS)}
        config, _ = ex2221.recipe_config(meta[key], model_seed, epochs)
        seed = model_seed - SEED_OFFSET
        rows.append(
            {
                "config": config,
                "anchor": anchor,
                "anti": anti,
                "anchor_slices": None if c.slices == "all" else SLICE_SETS[c.slices],
                "min_line": context_length(min(c.counts)),
                "corpus_key": key,
                "condition": c.name,
                "seed": seed,
                "model_seed": model_seed,
                "replicate": model_seed in replicate,
                "epochs": epochs,
                "source": "ex-2.2.22",
                "label": f"{c.name}-s{seed}",
            }
        )
    return rows


def reused_rows(reused_seeds: dict[str, int] = REUSED_SEEDS) -> list[dict]:
    """One row per ex-2.2.21 run scored here, labelled as ex-2.2.21 labelled it."""
    return [
        {
            "condition": name,
            "seed": s,
            "model_seed": SEED_OFFSET + s,
            "replicate": False,
            "epochs": EPOCHS,
            "source": "ex-2.2.21",
            "corpus_key": MAIN_KEY,
            "label": f"{name}-s{s}",
        }
        for name, n in reused_seeds.items()
        for s in range(n)
    ]


def train_one(
    config,
    anchor: dict,
    anti: dict | None,
    anchor_slices,
    min_line: int,
    ops: tuple[str, ...],
    corpus,
    labels,
    probes,
    holdout,
    k: int,
    traj_stride: int,
    n_traj_eem: int,
    label: str,
) -> dict:
    """Ex-2.2.21's `train_one` with the whole-line label, and the shortest line given as *min_line* rather than
    taken from *k*: the `k-mixed` corpus has lines of one example, so a crop can straddle more of them. The
    trajectory is ex-2.2.21's, on the probes and the held-out set of *k* examples.
    """
    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec, alignment
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    n_ops = len(ops)
    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    a_id = ops.index(ANCHORED_OP)
    final = config.model.n_layer

    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=a_id,
            label_rate=LABEL_RATE,
            variant="whole",
            context_len=z["context_len"],
        )
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]
    per_op = len(probe_tokens) // n_ops
    weights = np.concatenate([np.full(per_op, 1.0 if o == ANCHORED_OP else 0.0) for o in ops])
    weights = weights / weights.sum()

    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    sub = np.concatenate([np.flatnonzero(ho.op_ids == o)[:n_traj_eem] for o in range(n_ops)])
    sub_tokens, sub_op_ids, sub_post = ho.tokens[sub], ho.op_ids[sub], ho.posterior[sub]
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(config.model.vocab_size, -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    sub_ho = ex2216.Holdout(
        sub_tokens,
        sub_op_ids,
        sub_post,
        np.empty(0),
        np.empty((0, 0), dtype=np.int32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
    )
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    probe_ctx = ex2216._post_queries(P, sub_ho, k, tok2color)
    identity = projection(Subspace.axis(config.model.n_embd), 0.0)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)

    extra: dict[str, list] = {
        k_: [] for k_ in ("m_context", "op1_lean", "fragment_lean", "eem", "eem_per_op", "kl", "kl_per_op")
    }

    def on_record(_index: int, model) -> None:
        cos = alignment(model, probe_tokens)  # (L1, N, T)
        extra["m_context"].append(ex2216.context_margin(cos, weights))
        extra["op1_lean"].append(float(cos[final, :, 0].mean()))
        total, count = 0.0, 0
        for s in ex2216.fragment_starts(k):
            fc = alignment(model, np.ascontiguousarray(probe_tokens[:, s:]))
            total += float(fc[final].sum())
            count += fc.shape[1] * fc.shape[2]
        extra["fragment_lean"].append(total / count)
        p = ex2216._color_probs(logits_at(model, sub_tokens, identity, (), eq_role), color_ids)
        eem = ex2216._match(table, probe_ctx, p)
        kl = P.kl(P.predictive(table, probe_ctx, sub_post), p)
        extra["eem"].append(float(eem.mean()))
        extra["eem_per_op"].append([float(eem[sub_op_ids == o].mean()) for o in range(n_ops)])
        extra["kl"].append(float(kl.mean()))
        extra["kl_per_op"].append([float(kl[sub_op_ids == o].mean()) for o in range(n_ops)])

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=(AntiSpec(**anti) if anti is not None else None),
        label_p=spec,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        probe_line_w=weights,
        crop=ex2221.CROP_POLICY,
        anchor_slices=anchor_slices,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
        newline_id=tokenizer.stoi["\n"],
        min_line_tokens=min_line,
        on_record=on_record,
    )
    epoch_train_loss = np.asarray([m.train_loss for m in metrics], dtype=np.float64)
    epoch_axis = np.arange(1, len(epoch_train_loss) + 1, dtype=np.float64)
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": epoch_train_loss.tolist(),
        "traj": {
            kk: np.asarray(traj[kk]).tolist()
            for kk in ("step", "epoch", "lr", "weight", "anti_weight", "val_loss")
            if kk in traj
        }
        | {"train_loss": np.interp(np.asarray(traj["epoch"], dtype=np.float64), epoch_axis, epoch_train_loss).tolist()}
        | extra,
        "checkpoint": put(workdir / "model", name=f"ex-2.2.22-{label}-ckpt"),
    }


# --- Scoring ----------------------------------------------------------------------------------------

_per_op = ex2221._per_op


def total_variation(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per row, the total variation distance between *q* over the palette and the model distribution *p*, whose
    palette entries are *p* and whose remaining mass sits on other tokens, where *q* has none.
    """
    return 0.5 * (np.abs(q - p).sum(axis=1) + (1.0 - p.sum(axis=1)))


def suppress_one(checkpoint, holdout, ops: tuple[str, ...], k: int, keep_p: bool, label: str) -> dict:
    """The projection at every position and every slice, at each dose, on the held-out completion contexts of every
    op. Per op: expected exact match on the clean pass and under each dose, and the total variation distance and
    KL divergence (KL(target null ‖ model)) from the target null. Per context, to the store; with *keep_p*, also
    the answer distributions on the clean pass and at full dose, and what the confusion matrices need beside them.
    """
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    n_ops = len(ops)
    workdir = get_data_dir() / "suppress" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    a_id = ops.index(ANCHORED_OP)
    ctx = ex2216._post_queries(P, ho, k, tok2color)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)
    sub = Subspace.axis(model.transformer.wte.shape[1])
    every = tuple(range(len(model.transformer.blocks) + 1))
    mask = ex2216.site_positions("every position", k)

    null_post = ex2216._target_null(P, table, ctx, ho.posterior, a_id)
    q_null = P.predictive(table, ctx, null_post)
    null_eem = P.expected_match(table, ctx, null_post)

    def scored(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return ex2216._match(table, ctx, p), total_variation(q_null, p), P.kl(q_null, p)

    p_clean = ex2216._color_probs(logits_at(model, ho.tokens, projection(sub, 0.0), (), eq_role), color_ids)
    clean_eem, clean_tv, clean_kl = scored(p_clean)
    records, eem, tv, kl, p_full = [], [], [], [], None
    for g in DOSE_GAMMAS:
        p = ex2216._color_probs(logits_at(model, ho.tokens, projection(sub, g), every, eq_role, mask), color_ids)
        e, t, d = scored(p)
        eem.append(e)
        tv.append(t)
        kl.append(d)
        if g == max(DOSE_GAMMAS):
            p_full = p
        records.append(
            {
                "dose": g,
                "eem": _per_op(e, ho.op_ids, n_ops),
                "drop": _per_op(clean_eem - e, ho.op_ids, n_ops),
                "tv_null": _per_op(t, ho.op_ids, n_ops),
                "kl_null": _per_op(d, ho.op_ids, n_ops),
            }
        )
    anchored = ho.op_ids == a_id
    landing = 1.0 - tv[-1][anchored].mean() / clean_tv[anchored].mean()
    kept = {}
    if keep_p:
        assert p_full is not None
        kept = {
            "p_clean": p_clean.astype(np.float16),
            "p_full": p_full.astype(np.float16),
            "posterior": ho.posterior.astype(np.float32),
            "query_pair": ctx.query_pair,
        }
    arrays = put(
        ex2216._npz(
            op_ids=ho.op_ids.astype(np.int8),
            posterior_anchored=ho.posterior[:, a_id].astype(np.float32),
            clean_eem=clean_eem.astype(np.float32),
            null_eem=null_eem.astype(np.float32),
            clean_tv_null=clean_tv.astype(np.float32),
            clean_kl_null=clean_kl.astype(np.float32),
            eem=np.stack(eem).astype(np.float32),  # (doses, N), in the order of `DOSE_GAMMAS`
            tv_null=np.stack(tv).astype(np.float32),
            kl_null=np.stack(kl).astype(np.float32),
            **kept,
        ),
        name=f"ex-2.2.22-{label}-suppression.npz",
    )
    return {
        "label": label,
        "k": k,
        "ops": list(ops),
        "slices": list(every),
        "posterior_anchored": _per_op(ho.posterior[:, a_id], ho.op_ids, n_ops),
        "clean": {
            "eem": _per_op(clean_eem, ho.op_ids, n_ops),
            "tv_null": _per_op(clean_tv, ho.op_ids, n_ops),
            "kl_null": _per_op(clean_kl, ho.op_ids, n_ops),
        },
        "null": {"eem": _per_op(null_eem, ho.op_ids, n_ops)},
        "edits": records,
        "landing": float(landing),
        "arrays": arrays,
    }


def answer_roles(k: int) -> np.ndarray:
    """The roles of the answers in a context of *k* examples: each example's, then the query's."""
    return np.array([6 * i + 4 for i in range(k + 1)])


def by_count_one(checkpoint, holdouts: dict[int, Any], ops: tuple[str, ...], label: str) -> dict:
    """At each example count, on the held-out set of that count: expected exact match beside the ceiling and floor,
    per op (H1), and, per context, the alignment at every answer and slice with the posterior on the anchored op
    given the pairs up to and including that answer (E3). The query pair is never noise, so at the query answer
    the posterior is the one given the examples, times the likelihood of the answer under each op.
    """
    from sca.anchoring import alignment
    from sca.data.ops import colors
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    n_ops = len(ops)
    n_colors = len(colors())
    workdir = get_data_dir() / "by-count" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(ops))
    a_id = ops.index(ANCHORED_OP)
    identity = projection(Subspace.axis(model.transformer.wte.shape[1]), 0.0)

    summary, arrays = {}, {}
    for k, art in sorted(holdouts.items()):
        ho = ex2216.load_holdout(get(art, workdir / f"holdout-k{k}.npz"), k)
        ctx = ex2216._post_queries(P, ho, k, tok2color)
        eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)
        p = ex2216._color_probs(logits_at(model, ho.tokens, identity, (), eq_role), color_ids)
        eem = ex2216._match(table, ctx, p)
        floor = P.floor(table, ctx)

        roles = answer_roles(k)
        y = tok2color[ho.tokens[:, roles]]
        a, b = tok2color[ho.tokens[:, roles - 4]], tok2color[ho.tokens[:, roles - 2]]
        assert (y >= 0).all() and (a >= 0).all() and (b >= 0).all(), "operands and answers are colors"
        pair = a * n_colors + b
        examples = P.Contexts(ho.op_ids.astype(np.int64), pair[:, :k], y[:, :k], ctx.query_pair)
        prefix = P.prefix_posteriors(table, examples, RHO, CUBE_RATE)  # (N, k, ops)
        assert np.allclose(prefix[:, -1], ho.posterior, atol=1e-5), "the prefix posterior ends at the stored one"
        with_query = prefix[:, -1] * table.lookup(pair[:, k], y[:, k]).T
        with_query /= with_query.sum(axis=1, keepdims=True)
        post_at = np.concatenate([prefix[:, :, a_id], with_query[:, a_id, None]], axis=1)  # (N, k + 1)
        alpha = alignment(model, ho.tokens)[:, :, roles].transpose(1, 0, 2)  # (N, L1, k + 1)

        summary[k] = {
            "n": int(len(ho.tokens)),
            "eem": {"all": float(eem.mean()), "per_op": _per_op(eem, ho.op_ids, n_ops)},
            "ceiling": {"all": float(ho.ceiling.mean()), "per_op": _per_op(ho.ceiling, ho.op_ids, n_ops)},
            "floor": {"all": float(floor.mean()), "per_op": _per_op(floor, ho.op_ids, n_ops)},
        }
        arrays[f"k{k}_op_ids"] = ho.op_ids.astype(np.int8)
        arrays[f"k{k}_eem"] = eem.astype(np.float32)
        arrays[f"k{k}_ceiling"] = ho.ceiling.astype(np.float32)
        arrays[f"k{k}_floor"] = floor.astype(np.float32)
        arrays[f"k{k}_posterior_at_answer"] = post_at.astype(np.float32)
        arrays[f"k{k}_alpha_at_answer"] = alpha.astype(np.float16)
    return {
        "label": label,
        "ops": list(ops),
        "counts": summary,
        "arrays": put(ex2216._npz(**arrays), name=f"ex-2.2.22-{label}-by-count.npz"),
    }


# --- Publishing ------------------------------------------------------------------------------


def design() -> dict[str, Any]:
    """The design constants the report reads beside the results."""
    return {
        "experiment": "ex-2.2.22",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "model": MODEL,
        "epochs": EPOCHS,
        "centre": [K, RHO],
        "cube_rate": CUBE_RATE,
        "lambda_a": LAMBDA_A,
        "hinge_cap": HINGE_CAP,
        "hinge_softness": HINGE_SOFTNESS,
        "slice_sets": {k: list(v) for k, v in SLICE_SETS.items()},
        "mixed_counts": list(MIXED_COUNTS),
        "base": BASE,
        "references": [asdict(c) for c in REFERENCES],
        "conditions": [asdict(c) for c in CONDITIONS],
        "seeds": SEEDS,
        "seed_offset": SEED_OFFSET,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "reused_seeds": REUSED_SEEDS,
        "n_runs": N_RUNS,
        "holdout_contexts": HOLDOUT_CONTEXTS,
        "dose_gammas": list(DOSE_GAMMAS),
        "selectivity_gate": SELECTIVITY_GATE,
        "grading_min_damage": GRADING_MIN_DAMAGE,
        "landing_fraction": LANDING_FRACTION,
        "regression_tol": REGRESSION_TOL,
        "task_cost_tol": TASK_COST_TOL,
        "seed_band_sd": SEED_BAND_SD,
        "by_count_conditions": list(BY_COUNT_CONDITIONS),
        "confusion_conditions": list(CONFUSION_CONDITIONS),
    }


def publish(
    rows: list[dict],
    mixed: dict,
    holdouts: dict[int, dict],
    trained: list[dict],
    evaled: list[dict],
    suppressed: list[dict],
    by_count: list[dict],
):
    """Every ref the report reads: the metrics (design and corpus statistics), the trajectories of the new runs, and
    the three scoring passes (JSON, one record per run with its condition, seed, and source), every per-context
    array, every new checkpoint, and the `k-mixed` corpus, its labels, and the held-out sets.
    """
    import json

    from mini.store import put, set_ref

    keys = ("label", "condition", "seed", "model_seed", "replicate", "epochs", "source", "corpus_key")
    meta = {r["label"]: {k: r[k] for k in keys} for r in rows}

    def slim(r: dict) -> dict:
        return meta[r["label"]] | {k: v for k, v in r.items() if k != "arrays"}

    set_ref(CORPUS_REF.format(key=MIXED_KEY), mixed["corpus"])
    set_ref(LABELS_REF.format(key=MIXED_KEY), mixed["labels"])
    for k, h in holdouts.items():
        set_ref(HOLDOUT_REF.format(k=k), h["holdout"])
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for r in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    for r in suppressed:
        set_ref(SUPPRESSION_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    for r in by_count:
        set_ref(BY_COUNT_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    metrics = {
        "design": design(),
        "corpus": {MIXED_KEY: mixed["stats"]},
        "holdouts": {str(k): h["stats"] for k, h in holdouts.items()},
    }
    set_ref(METRICS_REF, put(json.dumps(metrics).encode(), name="ex-2.2.22-metrics.json"))
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.22-trajectories.json"))
    for ref, records, name in (
        (EVAL_REF, evaled, "eval"),
        (SUPPRESSION_REF, suppressed, "suppression"),
        (BY_COUNT_REF, by_count, "by-count"),
    ):
        body = {"design": design(), "runs": [slim(r) for r in records]}
        set_ref(ref, put(json.dumps(body).encode(), name=f"ex-2.2.22-{name}.json"))
    return {
        "n_trained": len(trained),
        "n_scored": len(evaled),
        "eem": {r["label"]: r["task"]["eem"]["all"] for r in evaled},
        "landing": {r["label"]: r["landing"] for r in suppressed},
    }


# --- Orchestration ----------------------------------------------------------------------------


def run(
    ctx: Ctx,
    n_lines: int,
    holdout_n: int,
    n_traj_points: int,
    n_traj_eem: int,
    seeds: int = SEEDS,
    replicate: tuple[int, ...] = REPLICATE_SEEDS,
    reused_seeds: dict[str, int] = REUSED_SEEDS,
    epochs: int = EPOCHS,
) -> dict:
    """The whole DAG: resolve ex-2.2.21's corpus condition and reference runs, build the `k-mixed` corpus and the
    held-out sets, train the new runs, score every run, and publish. Sizes, seeds, and length are arguments so a
    smoke run takes the same path.
    """
    reused = reused_rows(reused_seeds)
    main = ctx.run(resolve_reused, [r["label"] for r in reused], role="prep")
    mixed = ctx.run(prepare_mixed, OP_NAMES, MIXED_COUNTS, RHO, CUBE_RATE, n_lines, MIXED_SEED, MIXED_KEY, role="prep")
    other_counts = tuple(k for k in MIXED_COUNTS if k != K)
    holdouts = ctx.run(
        prepare_holdouts, OP_NAMES, other_counts, RHO, CUBE_RATE, holdout_n, MIXED_SEED + 10, role="prep"
    )
    preps = {MAIN_KEY: main, MIXED_KEY: mixed}
    rows = cells({key: p["meta"] for key, p in preps.items()}, seeds, replicate, epochs)
    n = len(rows)
    stride = [
        ex2217.traj_stride_for(
            epochs, ex2217.epoch_length_of(preps[r["corpus_key"]]["meta"].total_tokens, r["config"]), n_traj_points
        )
        for r in rows
    ]
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["anchor_slices"] for r in rows],
        [r["min_line"] for r in rows],
        [OP_NAMES] * n,
        [preps[r["corpus_key"]]["corpus"] for r in rows],
        [preps[r["corpus_key"]]["labels"] for r in rows],
        [main["probes"]] * n,
        [main["holdout"]] * n,
        [K] * n,
        stride,
        [n_traj_eem] * n,
        [r["label"] for r in rows],
        role="train",
    )
    scored = [{k: v for k, v in r.items() if k not in ("config", "anchor", "anti")} for r in rows] + reused
    ckpt = [t["checkpoint"] for t in trained] + [main["checkpoints"][r["label"]] for r in reused]
    m = len(scored)
    evaled = ctx.map(
        ex2221.eval_one,
        ckpt,
        [main["holdout"]] * m,
        [None] * m,
        [OP_NAMES] * m,
        [K] * m,
        [r["label"] for r in scored],
        role="eval",
    )
    suppressed = ctx.map(
        suppress_one,
        ckpt,
        [main["holdout"]] * m,
        [OP_NAMES] * m,
        [K] * m,
        [r["condition"] in CONFUSION_CONDITIONS for r in scored],
        [r["label"] for r in scored],
        role="suppress",
    )
    every_count = {k: h["holdout"] for k, h in holdouts.items()} | {K: main["holdout"]}
    picked = [i for i, r in enumerate(scored) if r["condition"] in BY_COUNT_CONDITIONS]
    by_count = ctx.map(
        by_count_one,
        [ckpt[i] for i in picked],
        [every_count] * len(picked),
        [OP_NAMES] * len(picked),
        [scored[i]["label"] for i in picked],
        role="suppress",
    )
    return ctx.run(publish, scored, mixed, holdouts, trained, evaled, suppressed, by_count, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx, N_LINES, HOLDOUT_CONTEXTS, N_TRAJ_POINTS, N_TRAJ_EEM_PER_OP)


COMPUTE = {
    # The `k-mixed` corpus of 300,000 contexts, four held-out sets with their posteriors, the refs resolved, and
    # the fan-in that writes every ref.
    "prep": dict(cpu=2, timeout=1800),
    # About 52,800 steps (200 epochs, about 20 minutes on an L4 in ex-2.2.21), with the trajectory.
    "train": dict(gpu="L4", timeout=2 * 3600, watchdog=900, watchdog_grace=900),
    # Forward passes over 14,000 held-out contexts: the eval makes k + 3 of them, the suppression pass five (the
    # clean pass and four doses), and the by-count pass two at each of five counts.
    "eval": dict(gpu="L4", timeout=900),
    "suppress": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="ex-2.2.22", main=main, roles=COMPUTE)
