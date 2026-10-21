"""What the edit does on the recipe of record: a re-analysis of split-anti-seeds' `falloff` runs, with no new training.

The edit projects e₁ out of the state at every position and every slice. Because the states are unit vectors and the
edit renormalizes, it turns each state by arcsin of the alignment that state arrives with. This pass stores that
alignment at every slice and position of the held-out contexts, twice per run: on the clean pass, and as it arrives
at each slice under the full edit (downstream of the edits at the slices before it). The write bound of claim (d)
compares the two. The grading and the landing come from the dose arrays split-anti-seeds already published, so they
need nothing from here.

Every `falloff` run at seeds 704 to 711 is measured, and so is ex-2.2.23's 400-epoch control at each seed, to show
the same map on a model where nothing was placed on e₁.

    bin/mini run docs/m2/edit-effects/experiment.py
    bin/mini status edit-effects
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


sas = _load_sibling("split-anti-seeds", "edit_effects_sas")
ex2223 = sas.ex2223
ex2222 = sas.ex2222
ex2221 = sas.ex2221
ex2216 = ex2222.ex2216

OP_NAMES = sas.OP_NAMES
ANCHORED_OP = sas.ANCHORED_OP
K = sas.K
EPOCHS = sas.EPOCHS
SEEDS = sas.SEEDS
ARM = "falloff"
N_SLICES = ex2222.N_LAYER + 1

PREFIX = "reports/m2/edit-effects"
RESULTS_REF = PREFIX + "/results"
ARRAYS_REF = PREFIX + "/arrays/{label}"


def runs() -> list[dict]:
    """The `falloff` runs and the controls at the same seeds, with the ref of each checkpoint."""
    anchored = [
        {"label": sas.label_of(ARM, s), "arm": ARM, "model_seed": s, "ref": sas.CHECKPOINT_REF.format(label=sas.label_of(ARM, s))}
        for s in SEEDS
    ]
    controls = [
        {
            "label": f"control-s{s}",
            "arm": "control",
            "model_seed": s,
            "ref": ex2223.CHECKPOINT_REF.format(label=ex2223.label_of("control", EPOCHS, s)),
        }
        for s in SEEDS
    ]
    return anchored + controls


def resolve(refs: list[str]) -> dict:
    """The held-out set of ex-2.2.21 and the checkpoint behind each ref."""
    from mini.store import get_ref

    holdout = get_ref(ex2221.HOLDOUT_REF.format(key=ex2222.MAIN_KEY))
    checkpoints = {r: get_ref(r) for r in refs}
    missing = [r for r, a in checkpoints.items() if a is None]
    assert holdout is not None and not missing, f"not published: {missing}"
    return {"holdout": holdout, "checkpoints": checkpoints}


def measure_one(checkpoint, holdout, label: str) -> dict:
    """The alignment with e₁ at every slice and position of every held-out completion context, clean and as it
    arrives at each slice under the full edit at every slice and position, stored as one `.npz` per run (float16,
    `(slices, contexts, positions)`), with the op of each context. Checks on the way that the turn the edit applies
    is arcsin of the arriving alignment, and that the embedding sees the clean stream.
    """
    import io

    from mini.store import get, put
    from sca.intervention import Subspace, angle_between, apply, projection

    workdir = get_data_dir() / "edit-effects" / label
    model, _, _, _ = ex2216._load_run(checkpoint, workdir)
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    sub = Subspace.axis(model.transformer.wte.shape[1])
    every = tuple(range(N_SLICES))
    clean, arriving, worst = [], [], 0.0
    for i in range(0, len(ho.tokens), 2000):
        tokens = ho.tokens[i : i + 2000]
        c = apply(model, tokens, projection(sub, 0.0), slices=())
        e = apply(model, tokens, projection(sub, 1.0), slices=every)
        theta = angle_between(e.pre, e.post)
        worst = max(worst, float(np.abs(theta - np.arcsin(np.abs(e.pre[..., 0]))).max()))
        np.testing.assert_allclose(e.pre[0], c.pre[0], rtol=0, atol=0)
        clean.append(c.pre[..., 0])
        arriving.append(e.pre[..., 0])
    assert worst < 2e-3, f"the turn is not arcsin of the arriving alignment: {worst}"

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        clean=np.concatenate(clean, axis=1).astype(np.float16),
        arriving=np.concatenate(arriving, axis=1).astype(np.float16),
        op_ids=ho.op_ids.astype(np.int8),
    )
    return {"label": label, "arrays": put(buf.getvalue(), name=f"edit-effects-{label}.npz"), "turn_check": worst}


def design() -> dict[str, Any]:
    return {
        "experiment": "edit-effects",
        "source": "split-anti-seeds",
        "arm": ARM,
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "seeds": list(SEEDS),
        "n_slices": N_SLICES,
        "dose_ref": sas.DOSE_ARRAYS_REF,
    }


def publish(meta: list[dict], measured: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    by_label = {m["label"]: {k: v for k, v in m.items() if k != "ref"} for m in meta}
    rows = []
    for r in measured:
        set_ref(ARRAYS_REF.format(label=r["label"]), r["arrays"])
        rows.append(by_label[r["label"]] | {k: v for k, v in r.items() if k != "arrays"})
    body = {"design": design(), "runs": rows}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="edit-effects-results.json"))
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
    # Two passes over 14,000 held-out contexts of a small model: under a minute on a CPU.
    "measure": dict(cpu=2, timeout=1800),
}

experiment = Experiment(name="edit-effects", main=main, roles=COMPUTE)
