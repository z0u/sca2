"""Does the e₁ edit miss anything between a block's attention and its MLP? A re-analysis of stored checkpoints, with
no new training.

The edit of record acts on the between-block slices only. Inside a block, the attention step writes to the stream and
the MLP of the same block reads it before the next slice, so a block could write the op onto e₁ and use it in the
same position without any slice showing it. Per run this scores the full e₁ edit three ways (between blocks, at the
mid-block states alone, and at both) along every coordinate axis, so each is net of the median over the other 63, and
keeps the alignment with e₁ at every sub-step of the clean pass and of the pass under the between-block edit.

The runs are `τ_ℓ fixed` at seeds 712 to 719 (fixed-confirm).

    bin/mini run docs/m2/mid-block-edit/experiment.py
    bin/mini status mid-block-edit
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


b1 = _load_sibling("block1-edit", "midblock_b1")
fc, ex2216, ex2221, ex2222 = b1.fc, b1.ex2216, b1.ex2221, b1.ex2222

OP_NAMES = b1.OP_NAMES
ANCHORED_OP = b1.ANCHORED_OP
K = b1.K
N_LAYER = b1.N_LAYER
SEEDS = b1.FIXED_SEEDS
SUBSTEPS = b1.SUBSTEPS
SITES = {"query =": [b1.EQ], "example answers": b1.EX_ANS}

# Where the full edit acts: between-block slices (0 is the embedding) and blocks whose mid-block state it edits.
EVERY = tuple(range(N_LAYER + 1))
BLOCKS = tuple(range(1, N_LAYER + 1))
EDITS = {
    "between blocks": (EVERY, ()),
    "mid-block": ((), BLOCKS),
    "both": (EVERY, BLOCKS),
}

BATCH = 1024

PREFIX = "reports/m2/mid-block-edit"
RESULTS_REF = PREFIX + "/results"


def runs() -> list[dict]:
    return [
        {"label": f"fixed-s{s}", "model_seed": s, "ref": fc.CHECKPOINT_REF.format(label=f"fixed-s{s}")} for s in SEEDS
    ]


def measure_one(checkpoint, holdout, label: str) -> dict:
    """Per op, expected exact match on the clean pass and for the target null, the drop under each edit along each
    axis (`drop[edit][axis][op]`, e₁ first), and the mean alignment with e₁ by sub-step at each site, for the
    `difference` contexts and the others, clean and under the between-block edit.
    """
    import json

    import jax
    import jax.numpy as jnp

    from mini.progress import emit_progress
    from mini.store import get, put
    from sca.intervention import Subspace, logits_at, projection

    workdir = get_data_dir() / "mid-block-edit" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    P = ex2216._get_posterior()
    table = P.build_table(ex2222.ex2218.table_of(OP_NAMES))
    ctx = ex2216._post_queries(P, ho, K, tok2color)
    a_id = OP_NAMES.index(ANCHORED_OP)
    null = P.expected_match(table, ctx, ex2216._target_null(P, table, ctx, ho.posterior, a_id))
    width = model.transformer.wte.shape[1]
    is_a = ho.op_ids == a_id

    def per_op(v: np.ndarray) -> list[float]:
        return [float(v[ho.op_ids == o].mean()) for o in range(len(OP_NAMES))]

    def score(op, slices, mid) -> np.ndarray:
        lg = logits_at(model, ho.tokens, op, slices, b1.EQ, mid=mid)
        return ex2216._match(table, ctx, ex2216._color_probs(lg, color_ids))

    clean = score(projection(Subspace.axis(width), 0.0), (), ())
    drop: dict[str, list] = {name: [] for name in EDITS}
    total = len(EDITS) * width
    for name, (sl, mid) in EDITS.items():
        for i in range(width):
            emit_progress(sum(map(len, drop.values())), total, name)
            drop[name].append(per_op(clean - score(projection(Subspace.axis(width, i)), sl, mid)))

    run = jax.jit(b1.forward, static_argnums=(2,))
    tokens = jnp.asarray(ho.tokens)

    def alignment(edit_slices) -> dict:
        # In batches of contexts, keeping only the e₁ coordinate at the sites, so the full stream never sits in memory.
        parts = []
        for i in range(0, len(tokens), BATCH):
            states, _, _ = run(model, tokens[i : i + BATCH], edit_slices, None)
            parts.append(
                {site: {k: np.asarray(v[:, pos, 0]) for k, v in states.items()} for site, pos in SITES.items()}
            )
        out = {}
        for site in SITES:
            a = {k: np.concatenate([p[site][k] for p in parts]).mean(axis=1) for k in SUBSTEPS}
            out[site] = {
                "difference": [float(a[k][is_a].mean()) for k in SUBSTEPS],
                "other": [float(a[k][~is_a].mean()) for k in SUBSTEPS],
            }
        return out

    res = {
        "label": label,
        "n_ctx": int(len(ho.tokens)),
        "clean": per_op(clean),
        "null": per_op(null),
        "drop": drop,
        "alignment": {"clean": alignment(()), "between blocks": alignment(EVERY)},
    }
    return {"label": label, "results": put(json.dumps(res).encode(), name=f"mid-block-edit-{label}.json")}


def design() -> dict[str, Any]:
    return {
        "experiment": "mid-block-edit",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "seeds": list(SEEDS),
        "substeps": SUBSTEPS,
        "sites": SITES,
        "edits": {k: {"slices": list(sl), "mid": list(mid)} for k, (sl, mid) in EDITS.items()},
    }


def publish(meta: list[dict], measured: list[dict]) -> dict:
    import json

    from mini.store import get, put, set_ref

    by_label = {m["label"]: {k: v for k, v in m.items() if k != "ref"} for m in meta}
    workdir = get_data_dir() / "mid-block-edit" / "publish"
    rows = [
        by_label[r["label"]] | json.loads(get(r["results"], workdir / f"{r['label']}.json").read_text())
        for r in measured
    ]
    set_ref(
        RESULTS_REF, put(json.dumps({"design": design(), "runs": rows}).encode(), name="mid-block-edit-results.json")
    )
    return {"n_runs": len(rows)}


def main(ctx: Ctx) -> dict:
    meta = runs()
    resolved = ctx.run(b1.resolve, [m["ref"] for m in meta], role="prep")
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
    # 193 query-only passes and two full passes over 14,000 held-out contexts of a small model, per run.
    "measure": dict(cpu=4, memory=16384, timeout=7200, watchdog=900),
}

experiment = Experiment(name="mid-block-edit", main=main, roles=COMPUTE)
