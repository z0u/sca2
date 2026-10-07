"""Ex-2.2.23: the slow seeds, trained for longer — a scout.

Most runs of the recipe rise quickly to a first plateau of task skill, stay there for some tens of epochs, and then
rise again when the model learns the three HSV ops. A few runs never make that second rise within 200 epochs, and
they end well below the others. Those runs set the seed band of every measurement in ex-2.2.21 and ex-2.2.22. This
scout trains the recipe of record and the control at more seeds, at 200 epochs and at 400, to see how often a run
misses the rise, whether the anchor makes it more likely, whether a longer run makes it, and whether a run that rises
late ends like one that rises early. From that, a rule for runs that miss the rise can be fixed before the next
experiment.

The DAG resolves ex-2.2.21's corpus, held-out set, probes, and its 200-epoch runs at the reused seeds by ref, trains
the new runs with ex-2.2.21's code, and scores every run with ex-2.2.21's eval and ex-2.2.22's suppression pass.

    bin/mini run docs/m2/ex-2.2.23/experiment.py --app modal --max-containers 12 --budget 5h
    bin/mini status ex-2.2.23
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

from mini import Ctx, Experiment


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as in ex-2.2.16 to ex-2.2.22."""
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


ex2222 = _load_sibling("ex-2.2.22", "ex2223_ex2222")
ex2221 = ex2222.ex2221

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = ex2222.OP_NAMES
ANCHORED_OP = ex2222.ANCHORED_OP
MODEL = ex2222.MODEL
N_SLICES = ex2222.N_SLICES
LAMBDA_A = ex2222.LAMBDA_A
SELECTIVITY_GATE = ex2222.SELECTIVITY_GATE
GRADING_MIN_DAMAGE = ex2222.GRADING_MIN_DAMAGE
DOSE_GAMMAS = ex2222.DOSE_GAMMAS
"""The recipe of record (the D2.2 design, after ex-2.2.22): the seven-op set, three examples per context at
replacement op noise 0.3, the whole-line label on one `difference` context in fifty, every slice pulled, no cap, no
verification lines. The edit criteria of ex-2.2.21 (E2), unchanged."""

HSV_OPS: tuple[str, ...] = ("hue-hsv", "sat-hsv", "value-hsv")
"""The ops whose learning makes the second rise. On the slow runs of ex-2.2.21 and ex-2.2.22 their skill stays near
0.2 while the other four ops end like those of the fast runs."""
assert set(HSV_OPS) <= set(OP_NAMES)

# --- The conditions ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    name: str
    anchored: bool
    """Anchored runs follow the recipe of record, which is ex-2.2.21's `anchor-whole` condition."""
    reused_as: str
    """The ex-2.2.21 condition whose 200-epoch runs this one reuses at the reused seeds."""


CONDITIONS: tuple[Condition, ...] = (
    Condition("control", anchored=False, reused_as="control"),
    Condition("anchor", anchored=True, reused_as="anchor-whole"),
)

LENGTHS: tuple[int, ...] = (200, 400)
"""Epochs. 200 is the recipe; 400 is the recipe before ex-2.2.19 halved it, and every 400-epoch run there made the
second rise. The anchor schedules scale with the length (warm-up over the first tenth, anneal over the last), and the
learning-rate warm-up stays as it is."""
SHORT, LONG = LENGTHS

SEED_OFFSET = ex2221.SEED_OFFSET
REUSED_SEEDS: tuple[int, ...] = tuple(range(SEED_OFFSET, SEED_OFFSET + 5))
"""Model seeds 700 to 704: ex-2.2.21 trained both conditions at 200 epochs at these, so those runs are reused."""
NEW_SEEDS: tuple[int, ...] = tuple(range(SEED_OFFSET + 5, SEED_OFFSET + 12))
"""Model seeds 705 to 711, new for both conditions. (Ex-2.2.22 trained `hinge` at 705, which is a different
condition.)"""
SEEDS: tuple[int, ...] = REUSED_SEEDS + NEW_SEEDS

N_NEW_RUNS = len(CONDITIONS) * (len(NEW_SEEDS) + len(SEEDS))
"""New 200-epoch runs at the new seeds, and 400-epoch runs at every seed."""
assert N_NEW_RUNS == 38

# --- The measurements --------------------------------------------------------------------------------------

RISE_LEVEL = 0.35
"""A run has made the second rise once its HSV skill (EEM averaged over `HSV_OPS`) passes this level: about halfway
between the plateau, near 0.2, and where the fast runs end, near 0.5. The rise epoch is the first trajectory record
at or above it."""

CANDIDATE_RULE_LEVELS: tuple[float, ...] = (0.3, 0.35, 0.4)
"""S1: the levels of HSV skill (the average over `HSV_OPS`, or the worst of them), on the whole held-out set at the
end of training, that a rule for leaving out half-trained runs may use. Fixed before any run of this scout, since the edit results at the reused seeds are already
known from ex-2.2.21; S1 picks one of them from E1 to E3, and commits it before E4 is filled in. The slow runs of
ex-2.2.21 and ex-2.2.22 ended between 0.2 and 0.3, and the runs that rose early between 0.45 and 0.5."""
assert RISE_LEVEL in CANDIDATE_RULE_LEVELS

TRAJ_STRIDE_EPOCHS = 4
"""One trajectory record every four epochs at either length, as in ex-2.2.21 at 200 epochs, so the rise epoch is
resolved alike at both lengths."""

BUDGET_USD = 15
"""About \\$0.13 for a 200-epoch run on an L4 (ex-2.2.22), twice that at 400 epochs, plus the scoring passes."""


# =============================================================================================
# The DAG
# =============================================================================================
#
# Ex-2.2.21's corpus, held-out set, probes, and the checkpoints of its 200-epoch runs at the reused seeds, by ref.
# The new runs train with ex-2.2.21's `cells` and `train_one`, so a new 200-epoch run differs from a reused one in
# its seed alone, and a 400-epoch run in its length too. Every run, reused or new, is scored with ex-2.2.21's
# `eval_one` and ex-2.2.22's suppression pass at every position.

MAIN_KEY = ex2221.MAIN_KEY
K = ex2222.K
N_TRAJ_EEM_PER_OP = ex2221.N_TRAJ_EEM_PER_OP
CONFUSION_CONDITIONS: tuple[str, ...] = ()
"""No per-context answer distributions are kept: E4 reads the per-op summaries of the suppression pass."""

PREFIX = "reports/m2/ex-2.2.23"
METRICS_REF = PREFIX + "/metrics"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
EVAL_REF = PREFIX + "/eval"
EVAL_ARRAYS_REF = PREFIX + "/eval-arrays/{label}"
SUPPRESSION_REF = PREFIX + "/suppression"
SUPPRESSION_ARRAYS_REF = PREFIX + "/suppression-arrays/{label}"


def label_of(condition: str, epochs: int, model_seed: int) -> str:
    return f"{condition}-e{epochs}-s{model_seed}"


def arms(n_seeds: int) -> tuple:
    """Ex-2.2.21's arms for the two conditions, at *n_seeds* seeds from 700, so its `cells` builds every run."""
    from dataclasses import replace

    return tuple(replace(ex2221.arm(c.reused_as), seeds=n_seeds) for c in CONDITIONS)


def plan(
    reused: tuple[int, ...] = REUSED_SEEDS, new: tuple[int, ...] = NEW_SEEDS, lengths: tuple[int, ...] = LENGTHS
) -> list[tuple[Condition, int, int]]:
    """Every new run as (condition, epochs, model seed): the short length at the new seeds, every other length at
    every seed.
    """
    out = []
    for c in CONDITIONS:
        for epochs in lengths:
            seeds = new if epochs == SHORT else reused + new
            out += [(c, epochs, s) for s in seeds]
    return out


def new_rows(meta, runs: list[tuple[Condition, int, int]]) -> list[dict]:
    """One row per new run, built by ex-2.2.21's `cells` and relabelled."""
    by_name = {c.reused_as: c for c in CONDITIONS}
    n_seeds = max(s for _, _, s in runs) - SEED_OFFSET + 1
    built: dict[tuple[str, int, int], dict] = {}
    for epochs in sorted({e for _, e, _ in runs}):
        for r in ex2221.cells(arms(n_seeds), {MAIN_KEY: {"meta": meta}}, None, epochs):
            built[(by_name[r["arm"]].name, epochs, r["model_seed"])] = r
    rows = []
    for c, epochs, s in runs:
        r = built[(c.name, epochs, s)]
        assert r["config"].seed == s and r["epochs"] == epochs
        rows.append(r | {"condition": c.name, "source": "ex-2.2.23", "label": label_of(c.name, epochs, s)})
    return rows


def reused_rows(seeds: tuple[int, ...] = REUSED_SEEDS) -> list[dict]:
    """One row per ex-2.2.21 run scored here, with the label ex-2.2.21 gave it under `source_label`."""
    return [
        {
            "condition": c.name,
            "epochs": SHORT,
            "model_seed": s,
            "source": "ex-2.2.21",
            "source_label": f"{c.reused_as}-s{s - SEED_OFFSET}",
            "label": label_of(c.name, SHORT, s),
        }
        for c in CONDITIONS
        for s in seeds
    ]


def design() -> dict[str, Any]:
    return {
        "experiment": "ex-2.2.23",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "hsv_ops": list(HSV_OPS),
        "model": MODEL,
        "conditions": [asdict(c) for c in CONDITIONS],
        "lengths": list(LENGTHS),
        "reused_seeds": list(REUSED_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "rise_level": RISE_LEVEL,
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
        "dose_gammas": list(DOSE_GAMMAS),
        "selectivity_gate": SELECTIVITY_GATE,
        "grading_min_damage": GRADING_MIN_DAMAGE,
    }


def publish(scored: list[dict], trained: list[dict], evaled: list[dict], suppressed: list[dict]) -> dict:
    """Every ref the report reads: the design, the trajectories of the new runs, the eval and the suppression pass
    (one record per run with its condition, length, and seed), their per-context arrays, and every new checkpoint.
    """
    import json

    from mini.store import put, set_ref

    keys = ("label", "condition", "epochs", "model_seed", "source")
    meta = {
        r["label"]: {k: r[k] for k in keys} | ({"source_label": r["source_label"]} if "source_label" in r else {})
        for r in scored
    }

    def slim(r: dict) -> dict:
        return meta[r["label"]] | {k: v for k, v in r.items() if k != "arrays"}

    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for r in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    for r in suppressed:
        set_ref(SUPPRESSION_ARRAYS_REF.format(label=r["label"]), r["arrays"])
    set_ref(METRICS_REF, put(json.dumps({"design": design()}).encode(), name="ex-2.2.23-metrics.json"))
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="ex-2.2.23-trajectories.json"))
    for ref, records, name in ((EVAL_REF, evaled, "eval"), (SUPPRESSION_REF, suppressed, "suppression")):
        body = {"design": design(), "runs": [slim(r) for r in records]}
        set_ref(ref, put(json.dumps(body).encode(), name=f"ex-2.2.23-{name}.json"))
    return {
        "n_trained": len(trained),
        "n_scored": len(evaled),
        "eem": {r["label"]: r["task"]["eem"]["all"] for r in evaled},
        "landing": {r["label"]: r["landing"] for r in suppressed},
    }


def run(
    ctx: Ctx,
    reused: tuple[int, ...] = REUSED_SEEDS,
    new: tuple[int, ...] = NEW_SEEDS,
    lengths: tuple[int, ...] = LENGTHS,
) -> dict:
    """The whole DAG: resolve ex-2.2.21's corpus condition and reused runs, train the new runs, score every run, and
    publish. Seeds and lengths are arguments so a smoke run takes the same path.
    """
    old = reused_rows(reused)
    resolved = ctx.run(ex2222.resolve_reused, [r["source_label"] for r in old], role="prep")
    rows = new_rows(resolved["meta"], plan(reused, new, lengths))
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [round(TRAJ_STRIDE_EPOCHS * e) for e in epoch_length]
    trained = ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["variant"] for r in rows],
        [r["anchor_slices"] for r in rows],
        [r["pull"] for r in rows],
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        role="train",
    )
    keep = ("label", "condition", "epochs", "model_seed", "source")
    scored = [{k: r[k] for k in keep} for r in rows] + old
    ckpt = [t["checkpoint"] for t in trained] + [resolved["checkpoints"][r["source_label"]] for r in old]
    m = len(scored)
    evaled = ctx.map(
        ex2221.eval_one,
        ckpt,
        [resolved["holdout"]] * m,
        [None] * m,
        [OP_NAMES] * m,
        [K] * m,
        [r["label"] for r in scored],
        role="eval",
    )
    suppressed = ctx.map(
        ex2222.suppress_one,
        ckpt,
        [resolved["holdout"]] * m,
        [OP_NAMES] * m,
        [K] * m,
        [False] * m,
        [r["label"] for r in scored],
        role="suppress",
    )
    return ctx.run(publish, scored, trained, evaled, suppressed, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


COMPUTE = {
    # The refs resolved, and the fan-in that writes every ref.
    "prep": dict(cpu=2, timeout=1800),
    # About 52,800 steps at 200 epochs (about 20 minutes on an L4 in ex-2.2.21) and twice that at 400, with the
    # trajectory. Sized for the 400-epoch runs in a slow container (eng: training-step-is-host-bound).
    "train": dict(gpu="L4", timeout=4 * 3600, watchdog=900, watchdog_grace=900),
    "eval": dict(gpu="L4", timeout=900),
    "suppress": dict(gpu="L4", timeout=1800),
}

experiment = Experiment(name="ex-2.2.23", main=main, roles=COMPUTE)
