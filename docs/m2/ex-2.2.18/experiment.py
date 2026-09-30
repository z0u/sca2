"""Ex-2.2.18: an op ablation scout on the in-context grammar.

Ex-2.2.17 left the center control (`k3-r0.3`, unanchored d64-L4) near 0.45 held-out expected exact match against a
Bayes ceiling of 0.52, and found that where the examples settle the op, the model keeps some of its mass on the
answers of the op most like the true one (`lighten` onto `screen`, `darken` onto `multiply`, `hsvmix` and `mix`
onto each other, `difference` onto `exclusion`). This scout drops one op of each of those pairs, one at a time and
then all four together, and trains the control on each smaller op set with ex-2.2.17's recipe (the newline mask,
the cosine at a peak learning rate of 0.00316, eight times ex-2.2.16's steps), beside a baseline on the full op
set built the same way. Each run is scored against the Bayes ceiling of its own op set, since dropping an op moves
the ceiling.

Every op set gets its own corpus condition (a fresh corpus, held-out set, and probe set at `k3-r0.3`), since the
replacement noise draws from the ops in the table. The corpus keeps ex-2.2.16's size in lines, so every run trains
for the same number of steps, and each remaining op gets a larger share of the lines.

    bin/mini run docs/m2/ex-2.2.18/experiment.py --app modal --max-containers 6 --budget 2h
    bin/mini status ex-2.2.18
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
    """A sibling experiment's module, by path and left out of `sys.modules` (the pattern ex-2.2.16 and ex-2.2.17
    use), so this module's task bodies still cloudpickle by value for a remote worker.
    """
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


ex2217 = _load_sibling("ex-2.2.17", "ex2218_ex2217")
ex2216 = ex2217.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

CENTRE = ex2216.CENTRE
EPOCHS = ex2216.EPOCHS * ex2217.LONG_MULT
PEAK_LR = ex2217.SWEEP_LRS[2]
WARMUP_EPOCHS = ex2217.WARMUP_EPOCHS
# The recipe of ex-2.2.17's `sweep-0.00316`: the masked cosine at eight times, the schedule its three seeds share.
# Its seed spread (about 0.03 over three seeds, ex-2.2.17 Scope) is the yardstick for a difference here.

SEED_OFFSET = ex2217.SEED_OFFSET
# Every run starts from model seed 600, the initialization of ex-2.2.17's `sweep-0.00316-s0`: the op sets share a
# vocabulary, so the models start from the same weights and differ in their corpus alone.

N_TRAJ_POINTS = ex2217.N_TRAJ_POINTS
N_TRAJ_EEM_PER_OP = ex2217.N_TRAJ_EEM_PER_OP

# --- The op sets --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OpSet:
    """One op set: the ops dropped from ex-2.2.16's table, and the op that most often gives the same answer as each
    dropped op, whose shortfall the drop is meant to shrink.
    """

    name: str
    dropped: tuple[str, ...]
    note: str = ""

    @property
    def ops(self) -> tuple[str, ...]:
        return tuple(o for o in ex2216.OP_NAMES if o not in self.dropped)


PARTNER: dict[str, str] = {"screen": "lighten", "multiply": "darken", "hsvmix": "mix", "exclusion": "difference"}
"""Each dropped op and the op ex-2.2.17 (E4) found the model confuses it with most."""

COUNTERPART: dict[str, str] = {"lighten": "screen", "darken": "multiply"}
"""The other side of the two brightening pairs of `PARTNER`: each op and the op it leaks onto."""

OP_SETS: tuple[OpSet, ...] = (
    OpSet("full", (), "ex-2.2.16's eleven ops, rebuilt the same way as the ablations"),
    *(OpSet(f"no-{op}", (op,), f"drop `{op}`, which `{p}` leaks onto") for op, p in PARTNER.items()),
    OpSet("no-four", tuple(PARTNER), "drop all four at once"),
    # A second round: drop the other op of the two brightening pairs, alone and in place of `screen` and `multiply` in
    # the four-op drop. `lighten` and `darken` round deterministically, so this four-op drop breaks the same pairs
    # while keeping the ops that round stochastically. Appended, so the seeds of the sets above stay as they were.
    *(OpSet(f"no-{op}", (op,), f"drop `{op}`, which leaks onto `{p}`") for op, p in COUNTERPART.items()),
    OpSet(
        "no-four-ld", (*COUNTERPART, "hsvmix", "exclusion"), "drop `lighten` and `darken` in place of their partners"
    ),
)

assert all(ex2216.ANCHORED_OP in s.ops for s in OP_SETS), "the labels keep the anchored op in every set"

CORPUS_SEED = 221_800
"""Offset by three per op set (`run`): the corpus, the held-out contexts, and the probe set of each set are three
streams that never draw from one another or from another set."""

N_RUNS = len(OP_SETS)


def op_set(name: str) -> OpSet:
    return next(s for s in OP_SETS if s.name == name)


def table_of(ops: tuple[str, ...]):
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME

    return tuple(OP_BY_NAME[n] if n in OP_BY_NAME else CANDIDATE_BY_NAME[n] for n in ops)


# =============================================================================================
# The DAG
# =============================================================================================


def prepare_op_set(
    ops: tuple[str, ...],
    k: int,
    rho: float,
    cube_rate: float,
    n_lines: int,
    seed: int,
    holdout_n: int,
    probe_n: int,
    key: str,
) -> dict:
    """One op set's corpus condition, as ex-2.2.16's `prepare_corpus_condition` builds one over its own table, less
    what an unanchored run never reads: the packed token stream, the per-context op ids and lengths the training
    step's labeller expects, the held-out contexts of every op with their posterior over the ops of this set,
    Bayes ceiling and floor, and the trajectory probe set.
    """
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import ops as grammar
    from sca.data.incontext import encode_corpus, sample_corpus, vocabulary
    from sca.data.incontext import op_ids as op_ids_of
    from sca.data.named_colors import WordTokenizer
    from mini.store import put

    P = ex2216._get_posterior()
    table = table_of(ops)
    corpus = sample_corpus(
        n_lines, seed, k, rho, op_table=table, cube_rate=cube_rate, rounding=ex2216.ROUNDING, verify_rate=0.0
    )
    tokenizer_config = TokenizerConfig(vocabulary=sorted(vocabulary()))
    tokenizer = WordTokenizer(tokenizer_config)
    tokens = encode_corpus(corpus, tokenizer.stoi)
    context_op = op_ids_of(corpus, table)
    context_len = np.array([c.n_tokens for c in corpus], dtype=np.int32)

    color_index = {c: i for i, c in enumerate(grammar.colors())}
    op_index = {o.name: i for i, o in enumerate(table)}
    post_table = P.build_table(table)

    holdout = ex2216._sample_by_op(table, k, rho, cube_rate, 0.0, holdout_n, seed + 1)
    ho_batch = ex2216._post_contexts(P, holdout, color_index, op_index)
    ho_post = P.posterior(post_table, ho_batch, rho, cube_rate)
    ho_ceiling = P.expected_match(post_table, ho_batch, ho_post)
    ho_floor = P.floor(post_table, ho_batch)
    ho_op = op_ids_of(holdout, table)

    probes = ex2216._sample_by_op(table, k, rho, cube_rate, 0.0, probe_n, seed + 2)
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

    return {
        "key": key,
        "ops": list(ops),
        "meta": meta,
        "stats": {
            "key": key,
            "ops": list(ops),
            "n_lines": n_lines,
            "total_tokens": int(len(tokens)),
            "ceiling": float(ho_ceiling.mean()),
            "floor": float(ho_floor.mean()),
            "ceiling_per_op": [float(ho_ceiling[ho_op == o].mean()) for o in range(len(ops))],
            "floor_per_op": [float(ho_floor[ho_op == o].mean()) for o in range(len(ops))],
            "told_op": post_table.told_op(),
        },
        "corpus": put(corpus_dir, name=f"ex-2.2.18-{key}-corpus"),
        "labels": put(ex2216._npz(op_ids=context_op, context_len=context_len), name=f"ex-2.2.18-{key}-labels.npz"),
        "holdout": put(
            ex2216._npz(
                tokens=encode_corpus(holdout, tokenizer.stoi),
                op_ids=ho_op,
                context_len=np.array([c.n_tokens for c in holdout], dtype=np.int32),
                posterior=ho_post,
                ceiling=ho_ceiling,
                verify_verdict=np.full(len(holdout), -1, dtype=np.int8),
            ),
            name=f"ex-2.2.18-{key}-holdout.npz",
        ),
        "probes": put(ex2216._npz(tokens=probe_tokens), name=f"ex-2.2.18-{key}-probes.npz"),
    }


def cells(sets: tuple[OpSet, ...], preps: dict[str, dict]) -> list[dict]:
    """One row per run: ex-2.2.17's `sweep-0.00316` config over each op set's corpus, at model seed 600."""
    from sca.config import ModelConfig
    from sca.data.named_colors import WordTokenizer
    from sca.utils import align

    rows = []
    for s in sets:
        meta = preps[s.name]["meta"]
        config = ex2216._make_config(
            align(meta.tokenizer_config.vocab_size, 64), SEED_OFFSET, EPOCHS, *ex2216.model_dims(ex2216.MODEL)
        )
        config.tokenizer = meta.tokenizer_config.model_copy()
        config.model.block_size = ex2216.BLOCK
        config.model.tie_embeddings = False
        config.optimizer.learning_rate = PEAK_LR
        config.model = ModelConfig.model_validate(
            config.model.model_dump() | {"line_mask_token": WordTokenizer(config.tokenizer).stoi["\n"]}
        )
        config.scheduler.warmup_epochs = WARMUP_EPOCHS
        base = ex2216.Condition229(
            s.name, 1, s.name, lam=0.0, tau=ex2216.TAU, epochs=EPOCHS, ops=s.ops, n_lines=ex2216.N_LINES
        )
        anchor, anti = ex2216.schedules(base)
        assert anti is None, "every run is unanchored"
        rows.append(
            {
                "config": config,
                "anchor": anchor,
                "op_set": s.name,
                "ops": list(s.ops),
                "dropped": list(s.dropped),
                "epochs": EPOCHS,
                "peak_lr": PEAK_LR,
                "model_seed": SEED_OFFSET,
                "label": s.name,
            }
        )
    return rows


def train_one(
    config,
    anchor: dict,
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
    """Ex-2.2.17's `train_one` over the op set *ops*: train one unanchored run, recording every *traj_stride* steps
    the held-out expected exact match and the calibration KL, overall and per op, on the first *n_traj_eem*
    held-out contexts of every op, scored against the posterior over this op set.
    """
    from sca.anchoring import AnchorSpec, LabelSpec
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import PALETTE
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    n_ops = len(ops)
    workdir = get_data_dir() / "cells" / label
    corpus_dir = get(corpus, workdir / "corpus")
    tokenizer = WordTokenizer(config.tokenizer)
    anchored_op_id = ops.index(ex2216.ANCHORED_OP)

    with np.load(get(labels, workdir / "labels.npz")) as z:
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=z["op_ids"],
            anchored_op_id=anchored_op_id,
            label_rate=ex2216.LABEL_RATE,
            variant="whole",
            context_len=z["context_len"],
        )

    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]
    per_op = len(probe_tokens) // n_ops
    weights = np.concatenate([np.full(per_op, 1.0 if o == ex2216.ANCHORED_OP else 0.0) for o in ops])
    weights = weights / weights.sum()

    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    sub = np.concatenate([np.flatnonzero(ho.op_ids == o)[:n_traj_eem] for o in range(n_ops)])
    sub_tokens, sub_op_ids, sub_posterior = ho.tokens[sub], ho.op_ids[sub], ho.posterior[sub]
    color_ids = np.array([tokenizer.stoi[n] for n in PALETTE])
    tok2color = np.full(config.model.vocab_size, -1)
    tok2color[color_ids] = np.arange(len(PALETTE))
    sub_holdout = ex2216.Holdout(
        sub_tokens,
        sub_op_ids,
        sub_posterior,
        np.empty(0),
        np.empty((0, 0), dtype=np.int32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
    )
    P = ex2216._get_posterior()
    table = P.build_table(table_of(ops))
    probe_ctx = ex2216._post_queries(P, sub_holdout, k, tok2color)
    identity = projection(Subspace.axis(config.model.n_embd), 0.0)
    eq_role = ex2216.query_role(k, ex2216.QUERY_EQ)

    traj_extra: dict[str, list] = {"eem": [], "eem_per_op": [], "kl": [], "kl_per_op": []}

    def on_record(_index: int, model) -> None:
        p = ex2216._color_probs(logits_at(model, sub_tokens, identity, (), eq_role), color_ids)
        eem = ex2216._match(table, probe_ctx, p)
        kl = P.kl(P.predictive(table, probe_ctx, sub_posterior), p)
        traj_extra["eem"].append(float(eem.mean()))
        traj_extra["eem_per_op"].append([float(eem[sub_op_ids == o].mean()) for o in range(n_ops)])
        traj_extra["kl"].append(float(kl.mean()))
        traj_extra["kl_per_op"].append([float(kl[sub_op_ids == o].mean()) for o in range(n_ops)])

    _, metrics, traj = train_anchored(
        config,
        corpus_dir,
        anchor=AnchorSpec(**anchor),
        anti=None,
        label_p=spec,
        probe_tokens=probe_tokens,
        probe_weights=weights,
        crop=ex2216.CROP_POLICY,
        anchor_slices=None,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
        newline_id=tokenizer.stoi["\n"],
        min_line_tokens=context_length(k),
        on_record=on_record,
    )

    epoch_train_loss = np.asarray([m.train_loss for m in metrics], dtype=np.float64)
    epoch_axis = np.arange(1, len(epoch_train_loss) + 1, dtype=np.float64)
    train_loss = np.interp(np.asarray(traj["epoch"], dtype=np.float64), epoch_axis, epoch_train_loss).tolist()

    return {
        "label": label,
        "traj": {
            "step": traj["step"].tolist(),
            "epoch": traj["epoch"].tolist(),
            "lr": traj["lr"].tolist(),
            "val_loss": traj["val_loss"].tolist(),
            "train_loss": train_loss,
        }
        | traj_extra,
        "checkpoint": put(workdir / "model", name=f"ex-2.2.18-{label}-ckpt"),
    }


def eval_one(checkpoint, holdout, ops: tuple[str, ...], k: int, label: str) -> dict:
    """The end-of-training scores of one run on every held-out context of its op set: expected exact match, the
    ceiling and floor of the context, and the calibration KL, overall and per op. The answer distribution over the
    grid at the query `=` goes to the store per context (float16), beside the posterior and the query pair, for
    the report's op confusion.
    """
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get, put

    n_ops = len(ops)
    workdir = get_data_dir() / "eval" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(table_of(ops))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), k)
    ctx = ex2216._post_queries(P, ho, k, tok2color)

    identity = projection(Subspace.axis(model.transformer.wte.shape[1]), 0.0)
    p = ex2216._color_probs(logits_at(model, ho.tokens, identity, (), ex2216.query_role(k, ex2216.QUERY_EQ)), color_ids)
    eem = ex2216._match(table, ctx, p)
    floor = P.floor(table, ctx)
    kl = P.kl(P.predictive(table, ctx, ho.posterior), p)
    task = {
        name: {"all": float(x.mean()), "per_op": [float(x[ho.op_ids == o].mean()) for o in range(n_ops)]}
        for name, x in (("eem", eem), ("ceiling", ho.ceiling), ("floor", floor), ("kl", kl))
    }
    task["color_mass"] = {"all": float(p.sum(1).mean())}
    arrays = put(
        ex2216._npz(
            op_ids=ho.op_ids.astype(np.int8),
            posterior=ho.posterior.astype(np.float32),
            query_pair=ctx.query_pair,
            eem=eem.astype(np.float32),
            ceiling=ho.ceiling.astype(np.float32),
            floor=floor.astype(np.float32),
            kl=kl.astype(np.float32),
            p=p.astype(np.float16),
        ),
        name=f"ex-2.2.18-{label}-eval.npz",
    )
    return {"label": label, "ops": list(ops), "n": int(len(ho.tokens)), "task": task, "arrays": arrays}


# --- Publishing ------------------------------------------------------------------------------

TRAJ_REF = "reports/m2/ex-2.2.18/trajectories"
EVAL_REF = "reports/m2/ex-2.2.18/eval"
EVAL_ARRAYS_REF = "reports/m2/ex-2.2.18/eval-arrays/{label}"
CHECKPOINT_REF = "reports/m2/ex-2.2.18/checkpoints/{label}"


def design() -> dict[str, Any]:
    """The design constants the report reads beside the results."""
    return {
        "experiment": "ex-2.2.18",
        "question": "whether dropping an op that gives answers like another's shrinks the gap to the Bayes ceiling",
        "corpus_condition": ex2216.cond_key(*CENTRE),
        "op_sets": [asdict(s) | {"ops": list(s.ops)} for s in OP_SETS],
        "partner": PARTNER,
        "counterpart": COUNTERPART,
        "epochs": EPOCHS,
        "peak_lr": PEAK_LR,
        "warmup_epochs": WARMUP_EPOCHS,
        "model": ex2216.MODEL,
        "model_seed": SEED_OFFSET,
        "n_lines": ex2216.N_LINES,
        "holdout_per_op": ex2216.HOLDOUT_CONTEXTS,
        "n_traj_eem_per_op": N_TRAJ_EEM_PER_OP,
    }


def publish(trained: list[dict], rows: list[dict], preps: dict[str, dict], evaled: list[dict]) -> dict:
    """The trajectories (JSON), the evaluation with the design, every row, and every op set's corpus statistics
    (JSON), every run's per-context arrays, and every end checkpoint, each under its own ref.
    """
    import json

    from mini.store import put, set_ref

    set_ref(TRAJ_REF, put(json.dumps({t["label"]: t["traj"] for t in trained}).encode(), name="ex-2.2.18-traj.json"))
    meta = {r["label"]: {k: v for k, v in r.items() if k not in ("config", "anchor")} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for e in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=e["label"]), e["arrays"])
    body = {
        "design": design(),
        "op_sets": {name: p["stats"] for name, p in preps.items()},
        "runs": [meta[e["label"]] | {k: v for k, v in e.items() if k != "arrays"} for e in evaled],
    }
    set_ref(EVAL_REF, put(json.dumps(body).encode(), name="ex-2.2.18-eval.json"))
    return {"n_runs": len(evaled), "eem": {e["label"]: e["task"]["eem"]["all"] for e in evaled}}


# --- Orchestration ----------------------------------------------------------------------------


def run(
    ctx: Ctx, sets: tuple[OpSet, ...], n_lines: int, holdout_n: int, probe_n: int, n_traj_points: int, n_traj_eem: int
) -> dict:
    """Prepare every op set's corpus condition, train one run on each, and score it: the whole DAG, with the corpus
    size, the held-out and probe counts, and the trajectory sizes as arguments so a short prototype runs the same
    code.
    """
    k, rho = CENTRE
    preps = {
        s.name: ctx.run(
            prepare_op_set,
            s.ops,
            k,
            rho,
            ex2216.CUBE_RATE,
            n_lines,
            CORPUS_SEED + 3 * i,
            holdout_n,
            probe_n,
            f"{ex2216.cond_key(k, rho)}-{s.name}",
            role="prep",
        )
        for i, s in enumerate(sets)
    }
    rows = cells(sets, preps)
    n = len(rows)
    stride = [
        ex2217.traj_stride_for(
            r["epochs"], ex2217.epoch_length_of(preps[r["op_set"]]["meta"].total_tokens, r["config"]), n_traj_points
        )
        for r in rows
    ]
    trained = ctx.map(
        train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [tuple(r["ops"]) for r in rows],
        [preps[r["op_set"]]["corpus"] for r in rows],
        [preps[r["op_set"]]["labels"] for r in rows],
        [preps[r["op_set"]]["probes"] for r in rows],
        [preps[r["op_set"]]["holdout"] for r in rows],
        [k] * n,
        stride,
        [n_traj_eem] * n,
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        [t["checkpoint"] for t in trained],
        [preps[r["op_set"]]["holdout"] for r in rows],
        [tuple(r["ops"]) for r in rows],
        [k] * n,
        [r["label"] for r in rows],
        role="eval",
    )
    return ctx.run(publish, trained, rows, preps, evaled, role="prep")


def main(ctx: Ctx) -> dict:
    return run(
        ctx, OP_SETS, ex2216.N_LINES, ex2216.HOLDOUT_CONTEXTS, ex2216.N_TRAJ_PROBE, N_TRAJ_POINTS, N_TRAJ_EEM_PER_OP
    )


COMPUTE = {
    # One corpus build of 300,000 contexts per op set, and the posterior over its held-out set; and the fan-in.
    "prep": dict(cpu=2, timeout=1800),
    # About 105,600 steps each, as ex-2.2.17's eight-times runs (about 40 minutes on an L4).
    "train": dict(gpu="L4", timeout=3 * 3600, watchdog=900, watchdog_grace=900),
    # Forward passes only, over at most 22,000 held-out contexts.
    "eval": dict(gpu="L4", timeout=900),
}

experiment = Experiment(name="ex-2.2.18", main=main, roles=COMPUTE)
