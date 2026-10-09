"""Where the lean comes from: a re-analysis of ex-2.2.23's checkpoints, with no new training.

Ex-2.2.23 found that the edit (projecting e₁ out of the state at every position and slice) spills onto other ops
on most runs trained to 400 epochs, and spill-by-position found the spill at the example answers, shaped by a
syntax token latched onto e₁. This pass asks what the edit removes on the runs with no latch. On every run it
measures what lies along e₁ in the embedding and readout tables, how the alignment at the color positions of
other ops follows the lightness of the color at each slice, what the alignment at the example answers of other
ops follows, and the full edit restricted to one set of slices at a time. The same measurements of the tables
and states are taken on the trials of the τ × λ_a sweep (PR #260) whose pull leaves out the embedding slice.

    bin/mini run docs/m2/embedding-lean/experiment.py --app modal --max-containers 12 --budget 1h
    bin/mini status embedding-lean
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


sbp = _load_sibling("spill-by-position", "embedding_lean_sbp")
ex2223 = sbp.ex2223
ex2221 = sbp.ex2221
ex2216 = sbp.ex2216
ex2218 = sbp.ex2218

OP_NAMES = sbp.OP_NAMES
ANCHORED_OP = sbp.ANCHORED_OP
K = sbp.K
UNIT = sbp.UNIT
QUERY_EQ = sbp.QUERY_EQ
N_READ = sbp.N_READ

# --- The positions and slices --------------------------------------------------------------------------------

ANSWERS = tuple(UNIT * i + 4 for i in range(K))
"""The example answers."""
OPERANDS_A = tuple(UNIT * i for i in range(K))
OPERANDS_B = tuple(UNIT * i + 2 for i in range(K))
QUERY_A, QUERY_B = UNIT * K, UNIT * K + 2
COLOR_POSITIONS = (*OPERANDS_A, *OPERANDS_B, *ANSWERS, QUERY_A, QUERY_B)
"""Every position up to the query `=` that holds a color token."""
SYNTAX = ("?", "=", ",", "\n")
LATE = (2, 3, 4)
"""The slices the alignment at the example answers is averaged over, as in the example-evidence report."""
LATCH_LEVEL = 0.9
"""A syntax embedding whose alignment with e₁ is at least this counts as a latch (spill-by-position)."""

N_SLICES = sbp.ex2222.N_LAYER + 1
SLICE_SETS: dict[str, tuple[int, ...]] = {
    "every slice": tuple(range(N_SLICES)),
    "embedding": (0,),
    "block 1": (1,),
    "embedding and block 1": (0, 1),
    "blocks 1 to 4": tuple(range(1, N_SLICES)),
    "blocks 2 to 4": tuple(range(2, N_SLICES)),
    "block 4": (N_SLICES - 1,),
}
"""The slice sets the full edit is restricted to."""

# --- The runs ------------------------------------------------------------------------------------------------

PREFIX = "reports/m2/embedding-lean"
RESULTS_REF = PREFIX + "/results"

SWEEP_PREFIX = "reports/m2/tau-lambda-sweep"
"""The τ × λ_a sweep of PR #260, whose experiment module is not on the main branch; its eval and checkpoints are
read by ref."""
SWEEP_TAU = (0.05, 0.3)
"""The τ range of the sweep trials measured here: around the recipe τ of 0.1, past the edge near 0.07 and before
the spill grows with τ."""


def runs() -> list[dict]:
    """Every run of ex-2.2.23, with the ref of its checkpoint, as spill-by-position lists them."""
    return [m | {"source": "ex-2.2.23", "edits": True} for m in sbp.runs()]


def sweep_runs() -> list[dict]:
    """The sweep trials in `SWEEP_TAU` on the no-emb and every-slice ("whole") arms, from the sweep's eval."""
    import json
    from pathlib import Path

    from mini.store import get, get_ref

    art = get_ref(SWEEP_PREFIX + "/eval")
    assert art is not None, f"not published: {SWEEP_PREFIX}/eval"
    ev = json.loads(Path(get(art, get_data_dir() / "sweep" / "eval.json")).read_text())
    out = []
    for r in ev["runs"]:
        if r["arm"] in ("no-emb", "whole") and SWEEP_TAU[0] <= r["tau"] <= SWEEP_TAU[1]:
            out.append(
                {
                    "label": f"sweep-{r['label']}",
                    "condition": r["arm"],
                    "tau": r["tau"],
                    "lam": r["lam"],
                    "ref": SWEEP_PREFIX + f"/checkpoints/{r['label']}",
                    "source": "tau-lambda-sweep",
                    "edits": False,
                }
            )
    return out


def resolve(refs: list[str]) -> dict:
    """The held-out set of ex-2.2.21 and the checkpoint behind each ref."""
    from mini.store import get_ref

    holdout = get_ref(ex2221.HOLDOUT_REF.format(key=ex2223.MAIN_KEY))
    checkpoints = {r: get_ref(r) for r in refs}
    missing = [r for r, a in checkpoints.items() if a is None]
    assert holdout is not None and not missing, f"not published: {missing}"
    return {"holdout": holdout, "checkpoints": checkpoints}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])


def measure_one(checkpoint, holdout, label: str, edits: bool) -> dict:
    """Every measurement on one run. With *edits*, also the full edit restricted to each slice set, scored on the
    held-out completion contexts of every op.
    """
    from mini.store import get
    from sca.anchoring import alignment
    from sca.data.ops import colors
    from sca.intervention import Subspace, logits_at, projection

    n_ops = len(OP_NAMES)
    a_id = OP_NAMES.index(ANCHORED_OP)
    workdir = get_data_dir() / "lean" / label
    model, tok, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    ctx = ex2216._post_queries(P, ho, K, tok2color)
    tokens, op_ids = ho.tokens, ho.op_ids
    other = op_ids != a_id
    n_colors = len(colors())
    light = np.array([np.mean(c) / 15.0 for c in colors()])  # 0 (black) to 1 (white)
    cid = np.asarray(color_ids)
    out: dict[str, Any] = {"label": label}

    # The tables: the e₁ component of every embedding and readout vector, after normalization.
    emb = np.asarray(model.transformer.wte)
    out["n_embd"] = int(emb.shape[1])
    emb_axis = emb[:, 0] / np.linalg.norm(emb, axis=1)
    readout = np.asarray(model.transformer.readout)
    readout_axis = readout[:, 0] / np.linalg.norm(readout, axis=1)
    syntax = {w: float(emb_axis[tok.stoi[w]]) for w in SYNTAX}
    latched = max(syntax, key=lambda w: abs(syntax[w]))
    out["syntax"] = syntax
    out["latched"] = latched if abs(syntax[latched]) >= LATCH_LEVEL else None
    out["emb_axis_colors"] = [float(emb_axis[cid[tok2color[cid] == c][0]]) for c in range(n_colors)]
    out["r_light_emb"] = _corr(light[tok2color[cid]], emb_axis[cid])
    out["emb_sd"] = float(emb_axis[cid].std())
    out["r_light_readout"] = _corr(light[tok2color[cid]], readout_axis[cid])
    out["readout_sd"] = float(readout_axis[cid].std())

    # The states: the alignment with e₁ at every slice and position.
    cos = alignment(model, tokens)  # (L1, N, T)
    light_at = light[tok2color[tokens[:, list(COLOR_POSITIONS)]]]  # (N, P)
    out["r_light_states_other"] = [
        _corr(light_at[other], cos[s][other][:, list(COLOR_POSITIONS)]) for s in range(cos.shape[0])
    ]
    out["role_profile_other_late"] = cos[list(LATE)][:, other].mean((0, 1))[:N_READ].tolist()
    out["role_profile_anchored_late"] = cos[list(LATE)][:, ~other].mean((0, 1))[:N_READ].tolist()

    # What the alignment at the example answers follows: whether that example fits the anchored op on its own.
    y = tok2color[tokens[:, list(ANSWERS)]]
    a = tok2color[tokens[:, list(OPERANDS_A)]]
    b = tok2color[tokens[:, list(OPERANDS_B)]]
    fits = table.lookup(a * n_colors + b, y)[a_id] > 0  # (N, K)
    alpha_ans = cos[list(LATE)][:, :, list(ANSWERS)].mean(0)  # (N, K)
    n_fit = fits.sum(1)
    out["alpha_fit_other"] = float(alpha_ans[other][fits[other]].mean())
    out["alpha_nofit_other"] = float(alpha_ans[other][~fits[other]].mean())
    out["alpha_fit_anchored"] = float(alpha_ans[~other][fits[~other]].mean())
    out["alpha_nofit_anchored"] = float(alpha_ans[~other][~fits[~other]].mean())
    out["alpha_ans_by_op"] = [float(alpha_ans[op_ids == o].mean()) for o in range(n_ops)]
    alpha_qeq = cos[list(LATE)][:, :, QUERY_EQ].mean(0)  # (N,)
    out["alpha_qeq_by_nfit_other"] = [
        float(alpha_qeq[other & (n_fit == k)].mean()) if (other & (n_fit == k)).any() else None for k in range(K + 1)
    ]
    out["alpha_qeq_by_nfit_anchored"] = [
        float(alpha_qeq[~other & (n_fit == k)].mean()) if (~other & (n_fit == k)).any() else None for k in range(K + 1)
    ]
    out["n_by_nfit_other"] = [int((other & (n_fit == k)).sum()) for k in range(K + 1)]

    if not edits:
        return out

    # The edit restricted to a set of slices: the full projection at every position, scored at the query `=`.
    sub = Subspace.axis(emb.shape[1])
    full = projection(sub, 1.0)
    every_position = np.ones(tokens.shape[1], dtype=np.float32)

    def match(operator, slices) -> np.ndarray:
        p = ex2216._color_probs(
            logits_at(model, tokens, operator, slices, QUERY_EQ, every_position if slices else None), color_ids
        )
        return ex2216._match(table, ctx, p)

    clean = match(projection(sub, 0.0), ())
    null_post = ex2216._target_null(P, table, ctx, ho.posterior, a_id)
    null = P.expected_match(table, ctx, null_post)
    per_op = lambda x: [float(x[op_ids == o].mean()) for o in range(n_ops)]  # noqa: E731
    out["clean"] = per_op(clean)
    out["null"] = per_op(null)
    out["edits"] = {}
    for name, slices in SLICE_SETS.items():
        drop = clean - match(full, slices)
        out["edits"][name] = per_op(drop)
        if name == "every slice":
            # Where the spill lands by how many examples fit the anchored op, on the other ops.
            out["drop_by_nfit_other"] = [
                float(drop[other & (n_fit == k)].mean()) if (other & (n_fit == k)).any() else None for k in range(K + 1)
            ]
            out["r_drop_nfit_other"] = _corr(n_fit[other], drop[other])
            out["r_drop_alpha_other"] = _corr(alpha_ans[other].sum(1), drop[other])
    return out


def design() -> dict[str, Any]:
    return {
        "experiment": "embedding-lean",
        "sources": ["ex-2.2.23", "tau-lambda-sweep"],
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "slice_sets": {k: list(v) for k, v in SLICE_SETS.items()},
        "late_slices": list(LATE),
        "latch_level": LATCH_LEVEL,
        "sweep_tau": list(SWEEP_TAU),
        "syntax": list(SYNTAX),
        "roles": [sbp.role_name(p) for p in range(N_READ)],
    }


def publish(meta: list[dict], scored: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    by_label = {m["label"]: {k: v for k, v in m.items() if k != "ref"} for m in meta}
    body = {"design": design(), "runs": [by_label[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="embedding-lean-results.json"))
    return {"n_runs": len(scored)}


def run(ctx: Ctx, limit: int | None = None) -> dict:
    """Resolve every checkpoint, measure each, and publish. *limit* keeps the first few runs of ex-2.2.23 and
    none of the sweep, for a smoke run that takes the same path.
    """
    meta = runs()[:limit] if limit else runs() + ctx.run(sweep_runs, role="prep")
    resolved = ctx.run(resolve, [m["ref"] for m in meta], role="prep")
    n = len(meta)
    scored = ctx.map(
        measure_one,
        [resolved["checkpoints"][m["ref"]] for m in meta],
        [resolved["holdout"]] * n,
        [m["label"] for m in meta],
        [m["edits"] for m in meta],
        role="measure",
    )
    return ctx.run(publish, meta, scored, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


COMPUTE = {
    "prep": dict(cpu=2, timeout=900),
    # One alignment pass and up to eight scoring passes over 14,000 held-out contexts.
    "measure": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="embedding-lean", main=main, roles=COMPUTE)
