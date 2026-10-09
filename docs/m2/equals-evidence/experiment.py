"""What the anchor follows at the `=` tokens: a re-analysis of ex-2.2.23's checkpoints, with no new training.

Every `=` of a context is a site where the model has to predict an answer from the examples before it, so the
posterior on the anchored op there climbs from chance at the first example to near certainty at the query. The
example-evidence re-analysis found that at an example answer the anchor follows that one example; this pass
stores the alignment with e₁ at every position and slice of the held-out contexts, on every run of ex-2.2.23, so
the report can compare the alignment at each `=` with the evidence before it.

    bin/mini run docs/m2/equals-evidence/experiment.py
    bin/mini status equals-evidence
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


sbp = _load_sibling("spill-by-position", "equals_evidence_sbp")
ex2223 = sbp.ex2223
ex2222 = sbp.ex2222
ex2221 = sbp.ex2221
ex2216 = sbp.ex2216
ex2218 = sbp.ex2218

OP_NAMES = sbp.OP_NAMES
ANCHORED_OP = sbp.ANCHORED_OP
K = sbp.K
UNIT = sbp.UNIT
QUERY_EQ = sbp.QUERY_EQ
N_READ = sbp.N_READ
N_SLICES = ex2222.N_LAYER + 1
RHO = ex2222.RHO
CUBE_RATE = ex2221.CUBE_RATE

EQUALS = tuple(UNIT * i + 3 for i in range(K + 1))
"""The `=` of each example, then the query `=`."""
assert EQUALS[-1] == QUERY_EQ
SYNTAX = ("?", "=", ",", "\n")
LATCH_LEVEL = 0.9
"""A syntax embedding whose alignment with e₁ is at least this counts as a latch (spill-by-position)."""

PREFIX = "reports/m2/equals-evidence"
RESULTS_REF = PREFIX + "/results"
ARRAYS_REF = PREFIX + "/arrays/{label}"


SWEEP_PREFIX = "reports/m2/tau-lambda-sweep"
"""The τ × λ_a sweep of PR #260, whose experiment module is not on the main branch; its eval and checkpoints are
read by ref, as the embedding-lean report does. Every trial is measured: a softer pool put the anchor onto the
`=` tokens there, which no run of ex-2.2.23 has."""


def runs() -> list[dict]:
    """Every run of ex-2.2.23, with the ref of its checkpoint, as spill-by-position lists them."""
    return [m | {"source": "ex-2.2.23"} for m in sbp.runs()]


def sweep_runs() -> list[dict]:
    """Every trial of the τ × λ_a sweep, from its eval."""
    import json
    from pathlib import Path

    from mini.store import get, get_ref

    art = get_ref(SWEEP_PREFIX + "/eval")
    assert art is not None, f"not published: {SWEEP_PREFIX}/eval"
    ev = json.loads(Path(get(art, get_data_dir() / "sweep" / "eval.json")).read_text())
    return [
        {
            "label": f"sweep-{r['label']}",
            "condition": r["arm"],
            "epochs": ev["design"]["epochs"],
            "model_seed": r["model_seed"],
            "tau": r["tau"],
            "lam": r["lam"],
            "ref": SWEEP_PREFIX + f"/checkpoints/{r['label']}",
            "source": "tau-lambda-sweep",
        }
        for r in ev["runs"]
    ]


def resolve(refs: list[str]) -> dict:
    """The held-out set of ex-2.2.21 and the checkpoint behind each ref."""
    return sbp.resolve(refs)


def measure_one(checkpoint, holdout, label: str) -> dict:
    """The alignment with e₁ at every slice and position of every held-out completion context, stored as one
    `.npz` per run: `alpha` is `(slices, contexts, positions)` in float16 up to the query `=`, `alpha_eq` the same
    at the `=` positions alone in float32, with the op of each context. Also the e₁ component of each syntax
    embedding, to mark the runs with a latch.
    """
    import io

    from mini.store import get, put
    from sca.anchoring import alignment

    workdir = get_data_dir() / "equals" / label
    model, tok, _, _ = ex2216._load_run(checkpoint, workdir)
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    cos = alignment(model, ho.tokens)[:, :, :N_READ]  # (L1, N, T)

    emb = np.asarray(model.transformer.wte)
    emb_axis = emb[:, 0] / np.linalg.norm(emb, axis=1)
    syntax = {w: float(emb_axis[tok.stoi[w]]) for w in SYNTAX}
    latched = max(syntax, key=lambda w: abs(syntax[w]))

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        alpha=cos.astype(np.float16),
        alpha_eq=cos[:, :, list(EQUALS)].astype(np.float32),
        op_ids=ho.op_ids.astype(np.int16),
    )
    return {
        "label": label,
        "arrays": put(buf.getvalue(), name=f"equals-evidence-{label}.npz"),
        "syntax": syntax,
        "latched": latched if abs(syntax[latched]) >= LATCH_LEVEL else None,
    }


def design() -> dict[str, Any]:
    return {
        "experiment": "equals-evidence",
        "source": "ex-2.2.23",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "rho": RHO,
        "cube_rate": CUBE_RATE,
        "equals": list(EQUALS),
        "n_read": N_READ,
        "n_slices": N_SLICES,
        "syntax": list(SYNTAX),
        "latch_level": LATCH_LEVEL,
        "roles": [sbp.role_name(p) for p in range(N_READ)],
    }


def publish(meta: list[dict], scored: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    by_label = {m["label"]: {k: v for k, v in m.items() if k != "ref"} for m in meta}
    rows = []
    for r in scored:
        set_ref(ARRAYS_REF.format(label=r["label"]), r["arrays"])
        rows.append(by_label[r["label"]] | {k: v for k, v in r.items() if k != "arrays"})
    body = {"design": design(), "runs": rows}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="equals-evidence-results.json"))
    return {"n_runs": len(rows)}


def run(ctx: Ctx, limit: int | None = None) -> dict:
    """Resolve every checkpoint, measure each, and publish. *limit* keeps the first few runs of ex-2.2.23 and none
    of the sweep, for a smoke run.
    """
    meta = runs()[:limit] if limit else runs() + ctx.run(sweep_runs, role="prep")
    resolved = ctx.run(resolve, [m["ref"] for m in meta], role="prep")
    scored = ctx.map(
        measure_one,
        [resolved["checkpoints"][m["ref"]] for m in meta],
        [resolved["holdout"]] * len(meta),
        [m["label"] for m in meta],
        role="measure",
    )
    return ctx.run(publish, meta, scored, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


COMPUTE = {
    "prep": dict(cpu=2, timeout=900),
    # One alignment pass over 14,000 held-out contexts of a small model: a minute on a CPU.
    "measure": dict(cpu=2, timeout=1800),
}

experiment = Experiment(name="equals-evidence", main=main, roles=COMPUTE)
