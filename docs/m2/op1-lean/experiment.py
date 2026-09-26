"""
Where the op1 lean sits: a reanalysis of ex-2.2.11's stored runs.

Scoring only, no training and no gates. ᾱ at op1 (the mean alignment of the 216 grid colors with e₁ at the
first operand) is about 0.02 on the un-anchored control and 0.28 on `handover`, and undoing either half of
the handover (the untied readout, the whole-line labeller) takes back about half the rise. Nothing so far
says why. This pass reads ex-2.2.11's stored metrics, eval arrays, and checkpoints for where that lean sits
(which slice, which position, which colors), what the readout table does with e₁, and what the e₁
coordinate of a state contributes to the next-token prediction.

    bin/mini run docs/m2/op1-lean/experiment.py --app local
    bin/mini status op1-lean
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, NamedTuple

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_ex2211():
    """Ex-2.2.11's module, loaded by path and left out of `sys.modules` (as ex-2.2.11 loads ex-2.2.9), so the
    task bodies here still cloudpickle by value for a remote worker.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "ex-2.2.11" / "experiment.py"
    spec = importlib.util.spec_from_file_location("ex2211", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2211 = _load_ex2211()

# --- Sources -------------------------------------------------------------------------------------

SOURCE = "ex-2.2.11"


class Condition(NamedTuple):
    """One condition of the source experiment: every stored seed is read."""

    cond: str
    seeds: tuple[int, ...]
    tied: bool
    span: int
    """How many roles of a labeled line the pull can land on: 4 (op1, the op word, op2, `=`) or the whole line, 6."""
    title: str


CONDITIONS: tuple[Condition, ...] = (
    Condition("control", tuple(range(5)), False, ex2211.ex229.WHOLE_SPAN, "un-anchored"),
    Condition("handover", tuple(range(20)), False, ex2211.ex229.WHOLE_SPAN, "untied readout, whole-line labeller"),
    Condition("handover-slot", tuple(range(20)), False, ex2211.ex229.PROMPT_SPAN, "untied readout, slot labeller"),
    Condition("handover-tied", tuple(range(9)), True, ex2211.ex229.WHOLE_SPAN, "tied readout, whole-line labeller"),
)
"""The four conditions of ex-2.2.11 that share a grammar and a seed set: the candidate, the two arms that each
undo one of its changes, and the control. The stored labels are `{cond}-s{seed}`."""

N_RUNS = sum(len(c.seeds) for c in CONDITIONS)

SRC_METRICS_REF = "reports/m2/ex-2.2.11/metrics"
SRC_PROBES_REF = "reports/m2/ex-2.2.11/probes"
SRC_CHECKPOINT_REF = "reports/m2/ex-2.2.11/checkpoints/{label}"
SRC_EVAL_REF = "reports/m2/ex-2.2.11/arrays/{label}/eval"
"""Where ex-2.2.11 published what this pass reads: its run table, the probe lines every run was scored on,
and per run the final checkpoint and the alignment maps of the eval pass."""

METRICS_REF = "reports/m2/op1-lean/metrics"
ARRAYS_REF = "reports/m2/op1-lean/arrays"
"""The run table (JSON) and every per-run and per-color array (npz), keyed `{label}/{name}`."""


def label_for(c: Condition, seed: int) -> str:
    return f"{c.cond}-s{seed}"


# --- What is read ---------------------------------------------------------------------------------

ROLES: tuple[str, ...] = ("op1", "op", "op2", "=", "answer", "⏎")
"""The six positions of a line, in order. Each predicts the next: a color predicts a syntax word and a syntax
word predicts a color, so the next-token question at every position is *syntax or color*."""

N_SLICES = 5
"""The embedding and the four blocks of the d64-L4 model."""

ANCHOR_AXIS = ex2211.ANCHOR_AXIS
"""e₁: where every anchored condition was asked to put *red*."""

TAU = ex2211.TAU
"""The mellowmax temperature the runs trained at, for the softmin weights that say where a line's pull landed."""

RED_DOSE, NONRED_DOSE = ex2211.RED_DOSE, ex2211.NONRED_DOSE
"""A color counts as red at or over `RED_DOSE` redness and as non-red at or under `NONRED_DOSE`."""

PER_SLOT_RATE = ex2211.PER_SLOT_RATE
"""Each slot of a line draws a label at redness⁸ × this rate: the labeller's exposure model."""

REF_OP = "mix"
"""The op whose probe lines carry the position and per-color measurements. At op1 the state depends on the
token alone (attention is causal), so a per-color measurement there is the same on every op."""

ANSWER_OPS: tuple[str, ...] = ("difference", "hue-hsv")
"""The ops the answer-only lines are read on: each can give a red answer from two non-red operands, which `mix`
cannot (a mean is red only if an operand is). `difference` is the op the D2.2 pivot anchors; `hue-hsv` has the
most such pairs of the eleven, and takes op2's hue at op1's saturation and value."""

OP_NAMES: tuple[str, ...] = ex2211.OP_NAMES
"""The eleven ops of the grammar, for the exposure model."""

TOKEN_GROUPS: tuple[str, ...] = ("colors", "red", "non-red", "op words", "=", "⏎")
"""How the vocabulary is grouped when a table's e₁ column is summarized: the 216 colors (and the red and non-red
ends of them), the eleven op words, and the two other syntax tokens."""


def npz_bytes(arrays: dict[str, Any]) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def palette_redness() -> dict[str, float]:
    """Each color token's redness, by name."""
    from sca.data.colors import redness
    from sca.data.ops import PALETTE

    return {name: redness(rgb) for name, rgb in PALETTE.items()}


def token_groups(vocab: list[str]) -> dict[str, np.ndarray]:
    """Index arrays into *vocab* for each of `TOKEN_GROUPS`. Every non-color, non-pad word that is not `=` or
    `⏎` is an op word: the vocabulary has no other kind.
    """
    red = palette_redness()
    idx = {g: [] for g in TOKEN_GROUPS}
    for i, w in enumerate(vocab):
        if w in red:
            idx["colors"].append(i)
            if red[w] >= RED_DOSE:
                idx["red"].append(i)
            elif red[w] <= NONRED_DOSE:
                idx["non-red"].append(i)
        elif w == "=":
            idx["="].append(i)
        elif w == "\n":
            idx["⏎"].append(i)
        elif w:
            idx["op words"].append(i)
    return {g: np.array(v, dtype=int) for g, v in idx.items()}


# --- Reading the stores --------------------------------------------------------------------------


def source_store():
    """The store the source runs were published to, for reads alone.

    On production this is the ambient store. Under a storage profile (`MINI_PROFILE=dev`) the ambient store is
    the profile's pair, which holds none of the source runs, so the base pair is resolved from the project
    configuration instead, as a production session resolves it. This pass's own results still go to the
    ambient store. No bucket is named here.
    """
    import os

    from mini.store import PROFILE_ENV, project_store

    saved = os.environ.pop(PROFILE_ENV, None)
    try:
        return project_store()
    finally:
        if saved is not None:
            os.environ[PROFILE_ENV] = saved


def resolve_inputs(labels: list[str]) -> dict:
    """The source experiment's artifacts by label, so each task is keyed on content rather than on a name."""
    ck = {lb: SRC_CHECKPOINT_REF.format(label=lb) for lb in labels}
    ev = {lb: SRC_EVAL_REF.format(label=lb) for lb in labels}
    names = [SRC_METRICS_REF, SRC_PROBES_REF, *ck.values(), *ev.values()]
    refs = source_store().get_refs(names)
    missing = [n for n in names if refs[n] is None]
    assert not missing, f"{SOURCE} artifacts not in the store: {missing}"
    return {
        "metrics": refs[SRC_METRICS_REF],
        "probes": refs[SRC_PROBES_REF],
        "checkpoints": {lb: refs[n] for lb, n in ck.items()},
        "evals": {lb: refs[n] for lb, n in ev.items()},
    }


# --- Tasks ---------------------------------------------------------------------------------------


def run_table(metrics) -> dict:
    """One record per run from the source run table: the per-slice, per-position alignment on the reference
    op, and the e₁ column of the embedding table and of the readout table (the embedding again when tied),
    summarized by token group and kept whole.
    """
    import json

    m = json.loads(source_store().get(metrics, get_data_dir() / "src" / "metrics.json").read_text())
    by_cond = {c.cond: c for c in CONDITIONS}
    vocab = sorted(m["runs"][0]["rows"])
    groups = token_groups(vocab)
    out = []
    for r in m["runs"]:
        c = by_cond.get(r["condition"])
        if c is None or r["seed"] not in c.seeds:
            continue
        rows = {"embedding": r["rows"], "readout": r["rows_readout"] or r["rows"]}
        cols = {k: np.array([v[w] for w in vocab]) for k, v in rows.items()}
        out.append(
            {
                "label": r["label"],
                "cond": c.cond,
                "seed": r["seed"],
                "tied": bool(r["tied"]),
                "alpha_op1": r["alpha_op1"],
                "alpha_pos": r["per_op"][REF_OP]["alpha_pos"],  # (slice, role)
                "e1": {k: {g: float(v[i].mean()) for g, i in groups.items()} for k, v in cols.items()},
                "e1_full": {k: v.tolist() for k, v in cols.items()},
            }
        )
    assert len(out) == N_RUNS, f"{len(out)} of {N_RUNS} runs in the source table"
    return {"vocab": vocab, "runs": out}


def line_groups(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> dict[str, np.ndarray]:
    """Two sets of lines with both operands non-red, split by the answer: red (the line can earn its label
    only through the answer, and only under the whole-line labeller) or non-red (the line earns no label).
    Matching the operands keeps the color composition at op1 the same in both sets.
    """
    operands = (r1 <= NONRED_DOSE) & (r2 <= NONRED_DOSE)
    return {"answer-only": operands & (r3 >= RED_DOSE), "none": operands & (r3 <= NONRED_DOSE)}


def crop_patterns() -> dict[tuple[int, int], float]:
    """How often a line is seen whole or cut short in training, from ex-2.2.11's batch sampler: each
    (first role, last role) that a crop leaves visible, with its share of all line visits.

    Training crops are `block_size`-token windows onto the packed corpus at a uniform offset, and a fraction
    `padding_chance` have a random prefix of 1 to `block_size // 3 − 1` tokens zeroed. So the first line in a
    crop usually starts late and the last usually ends early. The anchor term pools each labelled line over its
    visible pulled positions alone, so a line cut after op1 puts its whole pull there.
    """
    from collections import Counter

    config = ex2211.ex229._make_config(256, 0, 1)
    block, pad_p = config.model.block_size, config.data.padding_chance
    n_roles = len(ROLES)
    pads = [(0, 1 - pad_p)] + [(k, pad_p / (block // 3 - 1)) for k in range(1, block // 3)]
    seen: Counter[tuple[int, int]] = Counter()
    for start in range(n_roles):
        for pad, p_pad in pads:
            lines: dict[int, list[int]] = {}
            for t in range(pad, block):
                lines.setdefault((start + t) // n_roles, []).append((start + t) % n_roles)
            for roles in lines.values():
                seen[(roles[0], roles[-1])] += p_pad / n_roles
    total = sum(seen.values())
    return {k: v / total for k, v in seen.items()}


def crop_share(lines: np.ndarray, span: int, patterns: dict[tuple[int, int], float]) -> np.ndarray:
    """(slice, line): op1's softmin share of the pull, averaged over the ways a crop shows the line, weighted
    by `crop_patterns`. Only a crop that shows op1 can pull it, and such a crop shows a prefix of the line, whose
    states under causal attention are the probe line's own; a crop that shows nothing of the span has no pull
    to share and drops out of the average.
    """
    from sca.anchoring import softmin_weights

    share = np.zeros(lines.shape[:2], np.float32)
    weight = 0.0
    for (lo, hi), p in patterns.items():
        if lo >= span:
            continue
        weight += p
        if lo == 0:
            share += p * softmin_weights(1.0 - lines[:, :, : min(hi + 1, span)], TAU, axis=-1)[..., 0]
    return share / weight


def arrays_chunk(evals: list, probes, labels: list[str]) -> list[dict]:
    """From each run's eval arrays: the per-color alignment at op1 at every slice (reference op), and on each
    answer op's lines the softmin share op1 takes of the pull and its alignment, per slice, for the two line
    groups of `line_groups`. The share is over the condition's span, since that is what the pull pooled over:
    once on whole lines (`share`), and once averaged over the ways a training crop cuts a line (`share_crop`).
    """
    from mini.store import put
    from sca.anchoring import softmin_weights

    src = source_store()
    workdir = get_data_dir() / "eval"
    patterns = crop_patterns()
    groups: dict[str, dict[str, np.ndarray]] = {}
    op1_red: dict[str, dict[str, float]] = {}
    with np.load(src.get(probes, workdir / "probes.npz")) as z:
        redness = z["redness"]
        for op in ANSWER_OPS:
            r1, r2, r3, walk = (z[f"{op}/{k}"] for k in ("r1", "r2", "r3", "walk"))
            # The eval arrays cover the op1 walk alone (walk 0, the first rows of the probe set).
            keep = walk == 0
            groups[op] = line_groups(r1[keep], r2[keep], r3[keep])
            op1_red[op] = {k: float(r1[keep][g].mean()) for k, g in groups[op].items()}
    by_label = {label_for(c, s): c for c in CONDITIONS for s in c.seeds}
    paths = src.get_many([(e, workdir / f"{lb}.npz") for e, lb in zip(evals, labels, strict=True)])
    out = []
    for lb, p in zip(labels, paths, strict=True):
        span = by_label[lb].span
        arrays: dict[str, np.ndarray] = {}
        with np.load(p) as z:
            arrays["op1_alpha"] = z[f"{REF_OP}/alpha"][:, :, 0].astype(np.float32)  # (slice, color)
            for op in ANSWER_OPS:
                lines = z[f"{op}/alpha_lines"].astype(np.float32)  # (slice, line, role)
                n = len(next(iter(groups[op].values())))
                assert lines.shape[1] == n, f"{lb}: {lines.shape[1]} {op} lines against {n} walk-0 probes"
                w = softmin_weights(1.0 - lines[:, :, :span], TAU, axis=-1)
                wc = crop_share(lines, span, patterns)
                gs = groups[op].values()
                arrays[f"{op}/share"] = np.stack([w[:, g, 0].mean(1) for g in gs])  # (group, slice)
                arrays[f"{op}/share_crop"] = np.stack([wc[:, g].mean(1) for g in gs])  # (group, slice)
                arrays[f"{op}/line_alpha"] = np.stack([lines[:, g, 0].mean(1) for g in gs])  # (group, slice)
        out.append(
            {
                "label": lb,
                "n_lines": {op: {k: int(g.sum()) for k, g in groups[op].items()} for op in ANSWER_OPS},
                "op1_redness": op1_red,
                "redness": redness.tolist(),
                "crop_patterns": {f"{lo}-{hi}": v for (lo, hi), v in sorted(patterns.items())},
                "arrays": put(npz_bytes(arrays), name=f"op1-lean-{lb}-arrays.npz"),
            }
        )
    return out


def logodds(logits: np.ndarray, syntax: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Log P(next is a syntax word) − log P(next is a color), from logits over the last axis."""
    from scipy.special import logsumexp

    return logsumexp(logits[..., syntax], axis=-1) - logsumexp(logits[..., colors], axis=-1)


def logits_chunk(ckpts: list, probes, labels: list[str]) -> list[dict]:
    """From each checkpoint, on the reference op's probe lines: at every position, the syntax-against-color
    log-odds of the next token, and the part of it the e₁ coordinate carries.

    The logits are s_z · h·Rᵀ, linear in the state, so the e₁ coordinate's share of every logit is
    s_z · h₁ · R[:, 1]. Removing that share from the logits and taking the log-odds again gives the log-odds
    the readout would produce from the other 63 coordinates alone; the difference is what e₁ contributes.
    The e₁ columns of both tables ride along, in vocabulary order.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from mini.store import put
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer
    from sca.model.ngpt import NGPT

    src = source_store()
    workdir = get_data_dir() / "ckpt"
    with np.load(src.get(probes, workdir / "probes.npz")) as z:
        tokens = z[f"{REF_OP}/tokens"]  # (line, role)
    out = []
    for ckpt, lb in zip(ckpts, labels, strict=True):
        src.get(ckpt, workdir / lb / "model")
        model, config, _ = load_checkpoint(workdir / lb)
        assert isinstance(model, NGPT), f"{lb}: {type(model).__name__} is not an nGPT"
        tok = WordTokenizer(config.tokenizer)
        vocab = [tok.itos[i] for i in range(tok.vocab_size)]
        groups = token_groups(vocab)
        syntax = np.concatenate([groups["op words"], groups["="], groups["⏎"]])
        readout = np.asarray(model.transformer.readout, dtype=np.float64)
        wte = np.asarray(model.transformer.wte, dtype=np.float64)
        wte /= np.linalg.norm(wte, axis=1, keepdims=True)
        s_z = float(np.asarray(model.s_z()).reshape(-1)[0])

        fwd = eqx.filter_jit(model.stream_and_logits)
        states, logits = [], []
        for i in range(0, len(tokens), 2048):
            s, lg = fwd(jnp.asarray(tokens[i : i + 2048]))
            states.append(np.asarray(s[-1], dtype=np.float64))
            logits.append(np.asarray(lg, dtype=np.float64))
        h = np.concatenate(states)  # (line, role, width), the final slice
        lg = np.concatenate(logits)  # (line, role, vocab)
        assert np.abs(lg - s_z * (h @ readout.T)).max() < 1e-2, f"{lb}: logits are not s_z · h·Rᵀ"
        h1 = h[..., ANCHOR_AXIS]
        lo = logodds(lg, syntax, groups["colors"])
        without = lg - s_z * h1[..., None] * readout[:, ANCHOR_AXIS]
        arrays = {
            "logodds": lo.astype(np.float32),  # (line, role)
            "delta": (lo - logodds(without, syntax, groups["colors"])).astype(np.float32),
            "h1": h1.astype(np.float32),
            "readout_e1": readout[:, ANCHOR_AXIS].astype(np.float32),
            "embedding_e1": wte[:, ANCHOR_AXIS].astype(np.float32),
        }
        out.append(
            {
                "label": lb,
                "s_z": s_z,
                "vocab": vocab,
                "arrays": put(npz_bytes(arrays), name=f"op1-lean-{lb}-logits.npz"),
            }
        )
    return out


def exposure() -> dict:
    """What the grammar gives each color, as op1, in labels it did not earn: from the labeller's model (each
    slot draws at redness⁸ × `PER_SLOT_RATE`), the mean over the eleven ops and every partner of the
    probability that the line is labelled through op1 itself, through op2 when op1 did not draw, and through
    the answer alone. The answer's redness is taken in expectation over the stochastic rounding.
    """
    from sca.data.colors import redness
    from sca.data.ops import CANDIDATE_BY_NAME, OP_BY_NAME, answer_dist, colors

    cols = colors()
    r = np.array([redness(c) for c in cols])
    p = r**8 * PER_SLOT_RATE
    ops = [OP_BY_NAME.get(n) or CANDIDATE_BY_NAME[n] for n in OP_NAMES]
    n = len(cols)
    through = {k: np.zeros((len(ops), n)) for k in ("self", "op2", "answer")}
    for k, op in enumerate(ops):
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                p3 = sum(q * redness(c) ** 8 * PER_SLOT_RATE for c, q in answer_dist(op, a, b).items())
                through["self"][k, i] += p[i]
                through["op2"][k, i] += (1 - p[i]) * p[j]
                through["answer"][k, i] += (1 - p[i]) * (1 - p[j]) * p3
    return {
        "ops": [op.name for op in ops],
        "redness": r.tolist(),
        "through": {k: (v / n).tolist() for k, v in through.items()},  # (op, color): mean over partners
    }


def publish_results(table: dict, arrays: list[dict], logits: list[dict], exposed: dict) -> dict:
    """The run table under `METRICS_REF` and every array under `ARRAYS_REF`, keyed `{label}/{name}`, with the
    exposure model's arrays under `grammar/`.
    """
    import json

    from mini.store import get_many, put, set_ref

    merged: dict[str, np.ndarray] = {}
    parts = [(r["label"], r["arrays"]) for r in arrays] + [(r["label"], r["arrays"]) for r in logits]
    workdir = get_data_dir() / "publish"
    paths = get_many([(a, workdir / f"{i}.npz") for i, (_, a) in enumerate(parts)])
    for (lb, _), p in zip(parts, paths, strict=True):
        with np.load(p) as z:
            merged |= {f"{lb}/{k}": z[k] for k in z.files}
    merged["grammar/redness"] = np.array(exposed["redness"])
    for k, v in exposed["through"].items():
        merged[f"grammar/through/{k}"] = np.array(v)
    set_ref(ARRAYS_REF, put(npz_bytes(merged), name="op1-lean-arrays.npz"))

    by_label = {r["label"]: r for r in table["runs"]}
    for r in arrays:
        by_label[r["label"]] |= {k: r[k] for k in ("n_lines", "op1_redness")}
    for r in logits:
        by_label[r["label"]] |= {"s_z": r["s_z"]}
        # The checkpoint's vocabulary is the run table's plus the padding token at index 0, so the arrays'
        # e₁ columns have one more row than the table's `e1_full` lists.
        assert [w for w in r["vocab"] if w] == table["vocab"], f"{r['label']}: vocabulary differs from the run table's"
    metrics: dict[str, Any] = {
        "source": SOURCE,
        "conditions": [c._asdict() for c in CONDITIONS],
        "runs": table["runs"],
        "vocab": table["vocab"],
        "roles": ROLES,
        "n_slices": N_SLICES,
        "token_groups": TOKEN_GROUPS,
        "answer_ops": ANSWER_OPS,
        "line_groups": list(arrays[0]["n_lines"][ANSWER_OPS[0]]),
        "crop_patterns": arrays[0]["crop_patterns"],
        "redness": arrays[0]["redness"],
        "grammar_ops": exposed["ops"],
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="op1-lean-metrics.json"))
    return {"n_runs": len(table["runs"]), "n_arrays": len(merged)}


SEEDS_PER_TASK = 10
"""How many checkpoints one logits task reads in turn: the local backend runs every task of a `map` as its own
process at once, so a chunk keeps the fan-out to a handful of JAX processes."""


def chunks(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main(ctx: Ctx) -> dict:
    labels = [label_for(c, s) for c in CONDITIONS for s in c.seeds]
    inputs = ctx.run(resolve_inputs, labels, role="cpu")
    table = ctx.run(run_table, inputs["metrics"], role="cpu")
    parts = chunks(labels, SEEDS_PER_TASK)
    arrays = ctx.map(
        arrays_chunk,
        [[inputs["evals"][lb] for lb in part] for part in parts],
        [inputs["probes"]] * len(parts),
        parts,
        role="cpu",
    )
    logits = ctx.map(
        logits_chunk,
        [[inputs["checkpoints"][lb] for lb in part] for part in parts],
        [inputs["probes"]] * len(parts),
        parts,
        role="states",
    )
    exposed = ctx.run(exposure, role="cpu")
    return ctx.run(
        publish_results,
        table,
        [r for part in arrays for r in part],
        [r for part in logits for r in part],
        exposed,
        role="cpu",
    )


# Each logits task is up to ten forward passes over the reference op's probe lines; the exposure task is half
# a million answer distributions in Python. CPU work, minutes per task. The thread caps keep a handful of
# concurrent JAX processes from each claiming every core of one box.
THREAD_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}

experiment = Experiment(
    name="op1-lean",
    main=main,
    roles={
        "states": dict(cpu=2, timeout=3600, watchdog_grace=1800, env=THREAD_ENV),
        "cpu": dict(cpu=2, timeout=1800, watchdog_grace=900, env=THREAD_ENV),
    },
)
