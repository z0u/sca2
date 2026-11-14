"""Cooldown: does the end anneal of the recipe of record change the finished run?

The recipe of record (`τ_ℓ fixed`) ends with a cooldown: over epochs 360 to 400 the pull weight and the anti weight
both anneal to a tenth of where they were (`floor` 0.1, minimum-jerk). This experiment asks whether that matters, by
branching two arms from one stored state per seed, so that the cooldown is the only difference between them:

- the stem: each of the fixed confirmation's runs at seeds 712 to 719, carried on from its stored epoch-64 resume
  state to epoch 360, where the cooldown would start, and stopped there with a resume state of its own;
- `cooldown`: the recipe as it is, carried on from the stem to 400. On the same card type as the fixed confirmation
  (L4), this arm should repeat the stored runs, which the report checks;
- `no cooldown`: the same, with the end anneal taken out of both terms (`floor` 1), so the pull and the anti weight
  hold their epoch-360 values to the end.

Both arms train on the same card type (L4) and are scored as the fixed confirmation's finished runs are: the axis
scoring (removal, spill and the task change), the full edit restricted to each slice set (block 1), and the
embedding lean. The arms share a state from epoch 360, so they are compared seed by seed.

    bin/mini run docs/m2/cooldown/experiment.py --app modal --max-containers 16 --budget 1h --keep-stale-done
    bin/mini status cooldown
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


fc = _load_sibling("fixed-confirm", "cooldown_fc")
ac, sps, lr, rb, lean, ex2222 = fc.ac, fc.sps, fc.lr, fc.rb, fc.lean, fc.ex2222

OP_NAMES = fc.OP_NAMES
ANCHORED_OP = fc.ANCHORED_OP
K = fc.K
EPOCHS = fc.EPOCHS
NAME = "cooldown"

SEEDS: tuple[int, ...] = fc.SEEDS
"""712 to 719: the fixed confirmation's finished runs, whose epoch-64 resume states are stored."""
BRANCH = 360
"""Where the arms part: the start of the cooldown, before which the two schedules are the same."""
ARMS: tuple[str, ...] = ("cooldown", "no-cooldown")

BUDGET_USD = 6
"""From the fixed confirmation on L4 cards (eight runs carried from 64 to 400 for about \\$2, their end scoring about
\\$1): the stems (296 epochs each), the two arms (40 epochs each), the scoring of sixteen finished runs, and room for a
retry."""


def label_of(arm: str | None, model_seed: int) -> str:
    """`cooldown-stem-s712` for the stem, `cooldown-<arm>-s712` for an arm."""
    return f"{NAME}-{arm or 'stem'}-s{model_seed}"


def without_cooldown(anchor: dict, anti: dict) -> tuple[dict, dict]:
    """The two terms with the end anneal taken out: at `floor` 1 the anneal multiplies the weight by one throughout,
    and nothing else in either schedule reads `floor`.
    """
    return anchor | {"floor": 1.0}, anti | {"floor": 1.0}


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/cooldown"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
TRAJ_REF = PREFIX + "/trajectories"
RESULTS_REF = PREFIX + "/results"
AXES_REF = PREFIX + "/axes"
RESTRICTED_REF = PREFIX + "/restricted-axes"


def rows_of(meta, seeds: tuple[int, ...], arm: str | None, stop: int | None, epochs: int = EPOCHS) -> list[dict]:
    """One row per run at *seeds*: the fixed confirmation's row of `τ_ℓ fixed`, stopped at *stop*, with the arm's
    schedules (`None` is the stem, which runs the recipe's).
    """
    rows = []
    for r in fc.rows_of(meta, seeds, stop, epochs):
        anchor, anti = without_cooldown(r["anchor"], r["anti"]) if arm == "no-cooldown" else (r["anchor"], r["anti"])
        rows.append(
            r | {"anchor": anchor, "anti": anti, "kind": arm or "stem", "label": label_of(arm, r["model_seed"])}
        )
    return rows


def design() -> dict[str, Any]:
    base = next(c for c in sps.CONDITIONS if c.name == fc.BASE)
    anchor, anti = sps.st.schedules()
    anchor = base.anchor(anchor)
    anti = anti | {"hold": sps.FALLOFF_HOLDS}
    off_anchor, off_anti = without_cooldown(anchor, anti)
    return {
        "experiment": NAME,
        "base": fc.BASE,
        "anchor": {"cooldown": anchor, "no-cooldown": off_anchor},
        "anti": {"cooldown": anti, "no-cooldown": off_anti},
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "branch": BRANCH,
        "seeds": list(SEEDS),
        "window": list(lr.WINDOW),
        "track_slice": lr.TRACK_SLICE,
        "traj_stride_epochs": sps.TRAJ_STRIDE_EPOCHS,
    }


def resolve_resume(seeds: tuple[int, ...]) -> list[dict]:
    """The fixed confirmation's epoch-64 resume state at each of *seeds*, as `train_one` takes it."""
    from mini.store import get_ref

    out = []
    for s in seeds:
        ref = fc.RESUME_REF.format(label=fc.label_of(s))
        art = get_ref(ref)
        assert art is not None, f"not published: {ref}"
        out.append({"resume_state": art})
    return out


def publish(rows: list[dict], trained: list[dict], measured: list[dict], axes: list[dict],
            restricted: list[dict]) -> dict:  # fmt: skip
    """Trajectories, checkpoints, end measurements, the axis scoring and the restricted edits of both arms."""
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name=f"{NAME}-trajectories.json"))

    def body(runs: list[dict], **extra) -> bytes:
        return json.dumps({"design": design(), **extra, "runs": [meta[r["label"]] | r for r in runs]}).encode()

    set_ref(RESULTS_REF, put(body(measured), name=f"{NAME}-results.json"))
    set_ref(AXES_REF, put(body(axes), name=f"{NAME}-axes.json"))
    sets = {k: list(v) for k, v in sps.RESTRICTED_SETS.items()}
    set_ref(RESTRICTED_REF, put(body(restricted, slice_sets=sets), name=f"{NAME}-restricted-axes.json"))
    return {"n_runs": len(trained)}


def run(ctx: Ctx, epochs: int = EPOCHS, branch: int = BRANCH, seeds: tuple[int, ...] = SEEDS,
        stored: bool = True, full: bool = True) -> dict:  # fmt: skip
    """Carry each seed's stored run on to *branch*, branch both arms from there to the end, and score the finished
    runs. A smoke run (*stored* false) trains its stems from scratch at a fraction of the length and leaves the refs
    alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    stems = rows_of(resolved["meta"], seeds, None, branch, epochs)
    start = ctx.run(resolve_resume, seeds, role="prep") if stored else [None] * len(seeds)
    stemmed = ac.train(ctx, resolved, stems, start, [()] * len(stems))

    rows = [r for arm in ARMS for r in rows_of(resolved["meta"], seeds, arm, None, epochs)]
    n = len(rows)
    done = ac.train(ctx, resolved, rows, stemmed * len(ARMS), [()] * n)

    labels = [r["label"] for r in rows]
    holdout = [resolved["holdout"]] * n
    checkpoints = [t["checkpoint"] for t in done]
    measured = ctx.map(lean.measure_one, checkpoints, holdout, labels, [True] * n, role="measure")
    if not full:  # a smoke run leaves out the slow passes and the refs
        return {"finished": labels, "records": [len(t["traj"]["step"]) for t in done]}
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    restricted = ctx.map(sps.restricted_axes_one, checkpoints, holdout, labels, role="measure")
    return ctx.run(publish, rows, done, measured, axes, restricted, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed for eight epochs from scratch, branched at seven (the scaled cooldown runs from 7.2 to 8)."""
    return run(ctx, epochs=8, branch=7, seeds=SEEDS[:1], stored=False, full=False)


COMPUTE = fc.COMPUTE
"""The fixed confirmation's roles: training and scoring on L4 cards, the card its stored runs trained on."""

experiment = Experiment(name=NAME, main=main, roles=COMPUTE)
