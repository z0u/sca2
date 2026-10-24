"""Three recipes side by side: `falloff` (the recipe of record), `step, soft τ` and `pool 1–3, soft τ`, re-analysed on
their stored runs with no new training.

Most of what the comparison needs is stored already. Reference-baseline scored the dose edit along every coordinate
axis on all three conditions at seeds 704 to 711, which gives removal and spill under the own-run median baseline of
the D2.2 design. The dose arrays of split-anti-seeds and soft-pool give the grading and the landing, and their
measurements give the edits restricted to slice sets and the alignment by slice. Two things are new:

- The write bound of edit-effects on the two soft conditions: the alignment with e₁ at every slice and position,
  clean and as it arrives under the full edit (edit-effects has it for `falloff` and the controls).
- The same axis scoring as reference-baseline on the loss-rate runs that were finished at 400 epochs (`falloff` at
  714, 724, 725 and 735; `step, soft τ` at 724, 732 and 734), so those slow seeds get the own-run median too. They
  have no control at their seeds, which the own-run median does not need.

    bin/mini run docs/m2/recipe-compare/experiment.py --app modal --max-containers 16 --budget 1h
    bin/mini status recipe-compare
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from mini import Ctx, Experiment


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


rb = _load_sibling("reference-baseline", "recipe_compare_rb")
ee = _load_sibling("edit-effects", "recipe_compare_ee")
lr = _load_sibling("loss-rate", "recipe_compare_lr")
sas = rb.sas
sp = rb.sp
ex2222 = rb.ex2222

OP_NAMES = rb.OP_NAMES
ANCHORED_OP = rb.ANCHORED_OP
SEEDS: tuple[int, ...] = rb.SEEDS
CONDITIONS = ("falloff", "step-soft", "pool-123-soft")
"""The recipe of record, then soft-pool's two arms, by their labels in the stored runs."""
SOFT = CONDITIONS[1:]

BUDGET_USD = 2
"""Edit-effects' pass is two sweeps over the held-out set, under a minute on a CPU, for 16 runs. Reference-baseline
scored 48 checkpoints along 64 axes for about a dollar of GPU; this is 7."""

PREFIX = "reports/m2/recipe-compare"
WRITE_REF = PREFIX + "/write"
WRITE_ARRAYS_REF = PREFIX + "/write-arrays/{label}"
AXES_REF = PREFIX + "/axes"


def resolve_write() -> dict:
    """The soft-pool checkpoints of the two soft conditions at 704 to 711, by label (`step-soft-s704`)."""
    from mini.store import get_ref

    refs = {sp.label_of(a, s): sp.CHECKPOINT_REF.format(label=sp.label_of(a, s)) for a in SOFT for s in SEEDS}
    found = {label: get_ref(ref) for label, ref in refs.items()}
    missing = [label for label, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    return found


def resolve_finished() -> dict:
    """The loss-rate runs that were trained to 400 epochs, by label (`falloff-s714`), read from its results."""
    import json
    import tempfile
    from pathlib import Path

    from mini.store import get, get_ref

    results = get_ref(lr.RESULTS_REF)
    assert results is not None, "loss-rate results not published"
    with tempfile.TemporaryDirectory() as tmp:
        body = json.loads(Path(get(results, Path(tmp) / "results.json")).read_text())
    labels = [r["label"] for r in body["runs"]]
    found = {label: get_ref(lr.CHECKPOINT_REF.format(label=label)) for label in labels}
    missing = [label for label, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    return found


def design() -> dict[str, Any]:
    return {
        "experiment": "recipe-compare",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "dose_gammas": list(rb.DOSE_GAMMAS),
    }


def publish(written: list[dict], scored: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    for w in written:
        set_ref(WRITE_ARRAYS_REF.format(label=w["label"]), w["arrays"])
    rows = [{k: v for k, v in w.items() if k != "arrays"} for w in written]
    set_ref(WRITE_REF, put(json.dumps({"design": design(), "runs": rows}).encode(), name="recipe-compare-write.json"))
    set_ref(AXES_REF, put(json.dumps({"design": design(), "runs": scored}).encode(), name="recipe-compare-axes.json"))
    return {"n_written": len(written), "n_scored": len(scored)}


def run(ctx: Ctx, seeds_only: int | None = None, publishes: bool = True) -> dict:
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    soft = ctx.run(resolve_write, role="prep")
    finished = ctx.run(resolve_finished, role="prep")
    soft_labels = list(soft)[:seeds_only] if seeds_only else list(soft)
    finished_labels = list(finished)[:seeds_only] if seeds_only else list(finished)
    n, m = len(soft_labels), len(finished_labels)
    written = ctx.map(
        ee.measure_one,
        [soft[label] for label in soft_labels],
        [resolved["holdout"]] * n,
        soft_labels,
        role="write",
    )
    scored = ctx.map(
        rb.axes_one,
        [finished[label] for label in finished_labels],
        [resolved["holdout"]] * m,
        finished_labels,
        [None] * m,
        role="measure",
    )
    if not publishes:
        return {"written": [w["label"] for w in written], "scored": [s["label"] for s in scored]}
    return ctx.run(publish, written, scored, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One run of each kind, through the whole DAG, leaving the refs alone."""
    return run(ctx, seeds_only=1, publishes=False)


COMPUTE = {
    "prep": dict(cpu=2, timeout=900),
    "write": ee.COMPUTE["measure"],
    "measure": rb.COMPUTE["measure"],
}

experiment = Experiment(name="recipe-compare", main=main, roles=COMPUTE)
