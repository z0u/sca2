"""Why editing block 1 alone removes the op: a re-analysis of stored checkpoints, with no new training.

One forward pass per edit and run, written out by sub-step (the embedding, then the attention and the MLP of each block)
so that the e₁ projection can be spliced in after any block and at any set of positions. Per run it keeps the alignment
with e₁ at every position and sub-step, the score on each op, the attention from the query `=` and from each example
answer, and color probes on the clean states: the color of an operand at its own position, and at each example answer
the colors of the answer and of the two operands of its example.

The runs are `falloff` at seeds 704 to 711 (split-anti-seeds), `τ_ℓ fixed` at 712 to 719 (fixed-confirm), and the
400-epoch controls of ex-2.2.23 at 704 and 710.

    bin/mini run docs/m2/block1-edit/experiment.py
    bin/mini status block1-edit
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
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


sas = _load_sibling("split-anti-seeds", "block1_sas")
fc = _load_sibling("fixed-confirm", "block1_fc")
ex2223 = sas.ex2223
ex2222 = sas.ex2222
ex2221 = sas.ex2221
ex2216 = ex2222.ex2216

OP_NAMES = sas.OP_NAMES
ANCHORED_OP = sas.ANCHORED_OP
A_ID = OP_NAMES.index(ANCHORED_OP)
K = sas.K
EPOCHS = sas.EPOCHS
FALLOFF_SEEDS = tuple(sas.SEEDS)
FIXED_SEEDS = tuple(range(712, 720))
CONTROL_SEEDS = (704, 710)
N_LAYER = ex2222.N_LAYER

UNIT = 6  # tokens per equation: a ? b = y and a separator
T = UNIT * (K + 1)
EQ = ex2216.query_role(K, ex2216.QUERY_EQ)
EX_ANS = [UNIT * e + 4 for e in range(K)]
EX_OPERANDS = [UNIT * e + j for e in range(K) for j in (0, 2)]
QUERY_LINE = list(range(UNIT * K, EQ + 1))
SUBSTEPS = ["emb", *(f"{part}{i}" for i in range(1, N_LAYER + 1) for part in ("attn", "mlp"))]

# Where the attention from the query `=` lands, summed over the positions of each group.
ATTENTION_GROUPS = {
    "example answers": EX_ANS,
    "example =": [UNIT * e + 3 for e in range(K)],
    "example operands": EX_OPERANDS,
    "example ?": [UNIT * e + 1 for e in range(K)],
    "example ,": [UNIT * e + 5 for e in range(K)],
    "query line": QUERY_LINE,
}
# The e₁ edit at a set of slices (0 is the embedding, i the output of block i), at every position.
EDITS = {
    "none": (),
    "embedding": (0,),
    "block 1": (1,),
    "block 2": (2,),
    "block 3": (3,),
    "block 4": (4,),
    "blocks 2 to 4": (2, 3, 4),
    "blocks 1 to 4": (1, 2, 3, 4),
    "every slice": tuple(range(N_LAYER + 1)),
}
# The e₁ edit at block 1 only, at a set of positions.
POSITION_EDITS = {
    "example answers": EX_ANS,
    "all but example answers": [p for p in range(T) if p not in EX_ANS],
    "query line": QUERY_LINE,
}
PROBE_FOLDS = 5
PROBE_RIDGE = 1.0

PREFIX = "reports/m2/block1-edit"
RESULTS_REF = PREFIX + "/results"


def runs() -> list[dict]:
    """Every run measured, with the ref of its checkpoint."""
    out = [
        {
            "label": f"falloff-s{s}",
            "arm": "falloff",
            "model_seed": s,
            "ref": sas.CHECKPOINT_REF.format(label=sas.label_of("falloff", s)),
        }
        for s in FALLOFF_SEEDS
    ]
    out += [
        {"label": f"fixed-s{s}", "arm": "fixed", "model_seed": s, "ref": fc.CHECKPOINT_REF.format(label=f"fixed-s{s}")}
        for s in FIXED_SEEDS
    ]
    out += [
        {
            "label": f"control-s{s}",
            "arm": "control",
            "model_seed": s,
            "ref": ex2223.CHECKPOINT_REF.format(label=ex2223.label_of("control", EPOCHS, s)),
        }
        for s in CONTROL_SEEDS
    ]
    return out


def resolve(refs: list[str]) -> dict:
    """The held-out set of ex-2.2.21 and the checkpoint behind each ref."""
    from mini.store import get_ref

    holdout = get_ref(ex2221.HOLDOUT_REF.format(key=ex2222.MAIN_KEY))
    checkpoints = {r: get_ref(r) for r in refs}
    missing = [r for r, a in checkpoints.items() if a is None]
    assert holdout is not None and not missing, f"not published: {missing}"
    return {"holdout": holdout, "checkpoints": checkpoints}


def forward(model, idx, edit_slices: tuple[int, ...], pos_mask):
    """The stream by sub-step with the e₁ projection spliced in after the slices in *edit_slices* (at the positions
    where *pos_mask* is set, or everywhere), the attention rows from the query `=` and from the example answers, and
    the logits. The block is unrolled here so the state between its attention and its MLP can be kept.
    """
    import jax
    import jax.numpy as jnp

    from sca.intervention import Subspace, projection
    from sca.model._shared import merge_heads, normalize, split_heads

    op = projection(Subspace.axis(model.transformer.wte.shape[1]), 1.0)

    def edit(x):
        out = op(x)
        return out if pos_mask is None else jnp.where(pos_mask[None, :, None] > 0, out, x)

    x = model.embed(idx)
    states = {"emb": x}
    if 0 in edit_slices:
        x = edit(x)
    allowed = model.attention_mask(idx)
    enc = model.transformer.rotary_enc
    rows = []
    for i, block in enumerate(model.transformer.blocks, start=1):
        a = block.attn
        q, k, v = jnp.split(a.qkv(x), [a.n_kq_tot, 2 * a.n_kq_tot], axis=-1)
        q, k, v = (split_heads(t, a.n_head) for t in (q, k, v))
        q, k = enc(q, k)
        q, k = normalize(q), normalize(k)
        att = (q @ k.swapaxes(-2, -1)) * a.s_qk()
        att = jnp.where(allowed[:, None], att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1)
        x = block._step(x, a.proj(merge_heads(att @ v)), block.s_attn)
        states[f"attn{i}"] = x
        x = block._step(x, block.mlp(x), block.s_mlp)
        states[f"mlp{i}"] = x
        if i in edit_slices:
            x = edit(x)
        rows.append(att[:, :, [EQ, *EX_ANS], :])  # (N, H, 1 + K, T)
    logits = (x @ model.transformer.readout.T) * model.s_z()
    return states, jnp.stack(rows), logits


def probe_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Held-out R² of a ridge probe from states *x* `(n, d)` to targets *y* `(n, t)`, by k-fold over contexts
    (*groups* names the context of each row, so the positions of one context stay in one fold), pooled over folds and
    averaged over the target columns.
    """
    fold = groups % PROBE_FOLDS
    resid = np.zeros_like(y)
    for f in range(PROBE_FOLDS):
        tr, te = fold != f, fold == f
        xm, ym = x[tr].mean(0), y[tr].mean(0)
        xt = x[tr] - xm
        w = np.linalg.solve(xt.T @ xt + PROBE_RIDGE * np.eye(x.shape[1]), xt.T @ (y[tr] - ym))
        resid[te] = y[te] - ((x[te] - xm) @ w + ym)
    total = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - (resid**2).sum(0) / total))


def measure_one(checkpoint, holdout, label: str) -> dict:
    """Every measurement of one run, stored as one JSON."""
    import json

    import jax
    import jax.numpy as jnp

    from mini.store import get, put
    from sca.data.ops import colors

    workdir = get_data_dir() / "block1-edit" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    P = ex2216._get_posterior()
    table = P.build_table(ex2222.ex2218.table_of(OP_NAMES))
    ctx = ex2216._post_queries(P, ho, K, tok2color)
    null_eem = P.expected_match(table, ctx, ex2216._target_null(P, table, ctx, ho.posterior, A_ID))
    is_a = ho.op_ids == A_ID
    tokens = jnp.asarray(ho.tokens)
    run = jax.jit(forward, static_argnums=(2,))

    def score(logits) -> np.ndarray:
        return ex2216._match(table, ctx, ex2216._color_probs(np.asarray(logits[:, EQ]), color_ids))

    def summarize(states, rows, logits) -> tuple[dict, np.ndarray]:
        eem = score(logits)
        rows = np.asarray(rows)[:, is_a]  # (L, N_a, H, 1 + K, T)
        from_eq = rows[:, :, :, 0]
        out: dict[str, Any] = {
            "eem_a": float(eem[is_a].mean()),
            "eem_other": float(eem[~is_a].mean()),
            # mean α over the `difference` contexts and over the other ops, by sub-step and position
            "alpha_a": {k: np.asarray(v[..., 0])[is_a].mean(0).tolist() for k, v in states.items()},
            "alpha_other": {k: np.asarray(v[..., 0])[~is_a].mean(0).tolist() for k, v in states.items()},
            # attention from the query `=` to each group, by block: mean over heads, and the largest head
            "att_from_eq": {g: from_eq[..., p].sum(-1).mean(axis=(1, 2)).tolist() for g, p in ATTENTION_GROUPS.items()},
            "att_from_eq_maxhead": {
                g: from_eq[..., p].sum(-1).mean(axis=1).max(axis=-1).tolist() for g, p in ATTENTION_GROUPS.items()
            },
            # attention from each example answer to the two operands of its own example, by block
            "att_answer_to_operands": np.mean(
                [rows[:, :, :, 1 + e][..., [UNIT * e, UNIT * e + 2]].sum(-1).mean(axis=(1, 2)) for e in range(K)],
                axis=0,
            ).tolist(),
        }
        return out, eem

    res: dict[str, Any] = {"n_a": int(is_a.sum()), "null_eem_a": float(null_eem[is_a].mean()), "edits": {}}
    clean_states, clean_rows, clean_logits = run(model, tokens, (), None)
    clean, clean_eem = summarize(clean_states, clean_rows, clean_logits)
    span = clean_eem[is_a].mean() - null_eem[is_a].mean()

    def scored(states, rows, logits) -> dict:
        s, eem = summarize(states, rows, logits)
        drop = clean_eem - eem
        s["removal"] = float(drop[is_a].mean() / span)
        s["drop_by_op"] = [float(drop[ho.op_ids == o].mean()) for o in range(len(OP_NAMES))]
        return s

    for name, sl in EDITS.items():
        res["edits"][name] = clean if not sl else scored(*run(model, tokens, sl, None))
    res["position_edits"] = {}
    for name, positions in POSITION_EDITS.items():
        mask = np.zeros(T, np.float32)
        mask[positions] = 1
        res["position_edits"][name] = scored(*run(model, tokens, (1,), jnp.asarray(mask)))

    # The color probes, on the clean states of every context: the color of an operand at its own position (every
    # equation, the query included), and at the example answers the answer color and each operand color of that
    # example. Targets are the RGB of the palette color.
    rgb = np.asarray(colors(), np.float64)
    rgb /= rgb.max()
    tok = ho.tokens
    operands = [UNIT * e + j for e in range(K + 1) for j in (0, 2)]
    targets = {
        "operand": (operands, operands),
        "answer": (EX_ANS, EX_ANS),
        "answer, first operand": (EX_ANS, [UNIT * e for e in range(K)]),
        "answer, second operand": (EX_ANS, [UNIT * e + 2 for e in range(K)]),
    }
    res["probes"] = {name: {} for name in targets}
    for k, v in clean_states.items():
        v = np.asarray(v, np.float64)
        for name, (at, of) in targets.items():
            x = v[:, at].reshape(-1, v.shape[-1])
            y = rgb[tok2color[tok[:, of]].reshape(-1)]
            res["probes"][name][k] = probe_r2(x, y, np.repeat(np.arange(len(tok)), len(at)))

    body = json.dumps(res).encode()
    return {"label": label, "results": put(body, name=f"block1-edit-{label}.json")}


def design() -> dict[str, Any]:
    return {
        "experiment": "block1-edit",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "substeps": SUBSTEPS,
        "edits": {k: list(v) for k, v in EDITS.items()},
        "position_edits": POSITION_EDITS,
        "probe": {"folds": PROBE_FOLDS, "ridge": PROBE_RIDGE},
    }


def publish(meta: list[dict], measured: list[dict]) -> dict:
    import json

    from mini.store import get, put, set_ref

    by_label = {m["label"]: {k: v for k, v in m.items() if k != "ref"} for m in meta}
    rows = []
    workdir = get_data_dir() / "block1-edit" / "publish"
    for r in measured:
        body = json.loads(get(r["results"], workdir / f"{r['label']}.json").read_text())
        rows.append(by_label[r["label"]] | body)
    set_ref(RESULTS_REF, put(json.dumps({"design": design(), "runs": rows}).encode(), name="block1-edit-results.json"))
    return {"n_runs": len(rows)}


def main(ctx: Ctx) -> dict:
    meta = runs()
    resolved = ctx.run(resolve, [m["ref"] for m in meta], role="prep")
    measured = ctx.map(
        measure_one,
        [resolved["checkpoints"][m["ref"]] for m in meta],
        [resolved["holdout"]] * len(meta),
        [m["label"] for m in meta],
        role="measure",
    )
    return ctx.run(publish, meta, measured, role="prep")


COMPUTE = {
    "prep": dict(cpu=2, timeout=900),
    # Fourteen passes over 14,000 held-out contexts of a small model, plus the probes: a few minutes on four cores.
    "measure": dict(cpu=4, memory=16384, timeout=2400, watchdog=900),
}

experiment = Experiment(name="block1-edit", main=main, roles=COMPUTE)
