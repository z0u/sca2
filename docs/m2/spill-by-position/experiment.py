"""Which positions the spill of the edit comes from: a re-analysis of ex-2.2.23's checkpoints, with no new training.

Ex-2.2.23 found that the edit (projecting e₁ out of the state at every position and every slice) spills onto other
ops on nearly every run that has learned the HSV ops. This pass applies the same projection, at every slice, to one
set of positions at a time, and scores the answer at the query `=` on every op. The sets are the roles of a context:
the example answers, the query operands, the query `=`, and the rest of the positions up to the query `=`. Each set
is edited at every dose, each set is left out of an otherwise full edit at full dose, and each single position is
edited alone at full dose. Every run is scored, the control included, so each measurement can be netted against the
control at the same seed and length, as ex-2.2.23 did.

    bin/mini run docs/m2/spill-by-position/experiment.py --app modal --max-containers 12 --budget 2h
    bin/mini status spill-by-position
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir
from sca.data.incontext import context_length


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


ex2223 = _load_sibling("ex-2.2.23", "spill_by_position_ex2223")
ex2222 = ex2223.ex2222
ex2221 = ex2223.ex2221
ex2216 = ex2221.ex2216
ex2218 = ex2221.ex2218

OP_NAMES = ex2223.OP_NAMES
ANCHORED_OP = ex2223.ANCHORED_OP
K = ex2223.K
DOSE_GAMMAS = ex2223.DOSE_GAMMAS
FULL = max(DOSE_GAMMAS)

# --- The positions -----------------------------------------------------------------------------------------

UNIT = 6
"""Tokens per example: operand, `?`, operand, `=`, answer, separator. The query unit starts at `UNIT * K`."""
QUERY_EQ = ex2216.query_role(K, ex2216.QUERY_EQ)
"""The query `=`, where the answer is read. Positions after it cannot reach the logits there, so every mask stops
at it."""
N_READ = QUERY_EQ + 1
"""Positions 0 to the query `=`: every position an edit can act through."""
assert N_READ == context_length(K) - 2

SETS: dict[str, tuple[int, ...]] = {
    "example answers": tuple(UNIT * i + 4 for i in range(K)),
    "query operands": (UNIT * K, UNIT * K + 2),
    "query =": (QUERY_EQ,),
}
SETS["rest"] = tuple(p for p in range(N_READ) if not any(p in s for s in SETS.values()))
"""The rest: the example operands, the `?`, `=`, and separator of each example, and the query `?`."""
assert sorted(p for s in SETS.values() for p in s) == list(range(N_READ)), "the sets partition the positions"

EVERY = "every position"


def role_name(p: int) -> str:
    """A short name for position *p*: its role in the line, and the example it sits in (1-based) or "query"."""
    unit, offset = divmod(p, UNIT)
    role = ("a", "?", "b", "=", "y", ",")[offset]
    return f"{role}{unit + 1}" if unit < K else f"{role} query"


def edits() -> list[tuple[str, float, np.ndarray]]:
    """Every edit as (name, dose, position mask over the context). Each set and the whole read span at every dose;
    each set left out at full dose; each single position at full dose.
    """
    n = context_length(K)

    def mask(positions) -> np.ndarray:
        m = np.zeros(n, dtype=np.float32)
        m[list(positions)] = 1
        return m

    out = []
    for name, positions in ({EVERY: tuple(range(N_READ))} | SETS).items():
        out += [(name, g, mask(positions)) for g in DOSE_GAMMAS]
    for name, positions in SETS.items():
        out.append((f"all but {name}", FULL, mask(p for p in range(N_READ) if p not in positions)))
    out += [(f"position {p}", FULL, mask([p])) for p in range(N_READ)]
    return out


# --- The runs ----------------------------------------------------------------------------------------------

PREFIX = "reports/m2/spill-by-position"
RESULTS_REF = PREFIX + "/results"


def runs() -> list[dict]:
    """Every run of ex-2.2.23, with the ref of its checkpoint: ex-2.2.21's for the reused 200-epoch runs."""
    out = []
    for c in ex2223.CONDITIONS:
        for epochs in ex2223.LENGTHS:
            for s in ex2223.SEEDS:
                label = ex2223.label_of(c.name, epochs, s)
                if epochs == ex2223.SHORT and s in ex2223.REUSED_SEEDS:
                    ref = ex2221.CHECKPOINT_REF.format(label=f"{c.reused_as}-s{s - ex2223.SEED_OFFSET}")
                else:
                    ref = ex2223.CHECKPOINT_REF.format(label=label)
                out.append({"label": label, "condition": c.name, "epochs": epochs, "model_seed": s, "ref": ref})
    return out


def resolve(refs: list[str]) -> dict:
    """The held-out set of ex-2.2.21 and the checkpoint behind each ref."""
    from mini.store import get_ref

    holdout = get_ref(ex2221.HOLDOUT_REF.format(key=ex2223.MAIN_KEY))
    checkpoints = {r: get_ref(r) for r in refs}
    missing = [r for r, a in checkpoints.items() if a is None]
    assert holdout is not None and not missing, f"not published: {missing}"
    return {"holdout": holdout, "checkpoints": checkpoints}


def position_one(checkpoint, holdout, label: str) -> dict:
    """Every edit of `edits` on one run, acting at every slice, scored on the held-out completion contexts of every
    op: per op, the expected exact match on the clean pass, under the target null, and under each edit.
    """
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get

    n_ops = len(OP_NAMES)
    per_op = ex2221._per_op
    workdir = get_data_dir() / "position" / label
    model, _, color_ids, _tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    a_id = OP_NAMES.index(ANCHORED_OP)
    ctx = ex2216._post_queries(P, ho, K, _tok2color)
    sub = Subspace.axis(model.transformer.wte.shape[1])
    every = tuple(range(len(model.transformer.blocks) + 1))

    def eem(operator, slices, mask=None) -> list[float | None]:
        p = ex2216._color_probs(logits_at(model, ho.tokens, operator, slices, QUERY_EQ, mask), color_ids)
        return per_op(ex2216._match(table, ctx, p), ho.op_ids, n_ops)

    null_post = ex2216._target_null(P, table, ctx, ho.posterior, a_id)
    records = [{"edit": name, "dose": g, "eem": eem(projection(sub, g), every, m)} for name, g, m in edits()]
    return {
        "label": label,
        "ops": list(OP_NAMES),
        "clean": eem(projection(sub, 0.0), ()),
        "null": per_op(P.expected_match(table, ctx, null_post), ho.op_ids, n_ops),
        "edits": records,
    }


def design() -> dict[str, Any]:
    return {
        "experiment": "spill-by-position",
        "source": "ex-2.2.23",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "dose_gammas": list(DOSE_GAMMAS),
        "sets": {k: list(v) for k, v in SETS.items()},
        "n_read": N_READ,
        "roles": [role_name(p) for p in range(N_READ)],
    }


def publish(meta: list[dict], scored: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    by_label = {m["label"]: {k: v for k, v in m.items() if k != "ref"} for m in meta}
    body = {"design": design(), "runs": [by_label[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="spill-by-position-results.json"))
    return {"n_runs": len(scored)}


def run(ctx: Ctx, limit: int | None = None) -> dict:
    """Resolve every checkpoint, run the pass on each, and publish. *limit* keeps the first few runs, for a smoke
    run that takes the same path.
    """
    meta = runs()[:limit]
    resolved = ctx.run(resolve, [m["ref"] for m in meta], role="prep")
    n = len(meta)
    scored = ctx.map(
        position_one,
        [resolved["checkpoints"][m["ref"]] for m in meta],
        [resolved["holdout"]] * n,
        [m["label"] for m in meta],
        role="position",
    )
    return ctx.run(publish, meta, scored, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


COMPUTE = {
    "prep": dict(cpu=2, timeout=900),
    # 47 forward passes over 14,000 held-out contexts (ex-2.2.23's suppression pass made five, in under 90 s).
    "position": dict(gpu="L4", timeout=3600),
}

experiment = Experiment(name="spill-by-position", main=main, roles=COMPUTE)
