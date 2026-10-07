"""Ex-2.2.20: a high-rate head start before the recipe schedule.

In ex-2.2.19, the 50-epoch scout run at peak rate 0.00562 ended with a higher skill than any of the longer runs had
reached by epoch 50. This experiment trains the seven-op set (`no-four`) for 200 epochs on a schedule of two cycles:
50 epochs of warmup and cosine at that higher peak, then 150 epochs of warmup and cosine at the recipe peak, 0.00316.
A second condition is the same, except that the second cycle peaks at 0.0021, the top of the band of rates at which
the plain 200-epoch runs learned the HSV-channel ops, so that it spends longer in that band. Each condition is one run
per model seed, with the optimizer state carried across the boundary, at the four model seeds of the 200-epoch runs
of ex-2.2.19 (600-603), so that each run pairs with a plain 200-epoch run and a 400-epoch run from the same
initialization.

The corpus condition is rebuilt from ex-2.2.18's seed, as ex-2.2.19 did, which gives the same corpus. Training and
evaluation are ex-2.2.18's own tasks.

    bin/mini run docs/m2/ex-2.2.20/experiment.py --app modal --max-containers 8 --budget 2h
    bin/mini status ex-2.2.20
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from mini import Ctx, Experiment


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules` (the pattern ex-2.2.16 to ex-2.2.19
    use), so this module's task bodies still cloudpickle by value for a remote worker.
    """
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


ex2219 = _load_sibling("ex-2.2.19", "ex2220_ex2219")
ex2218 = ex2219.ex2218
ex2217 = ex2219.ex2217
ex2216 = ex2219.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_SET = ex2219.OP_SET
"""The seven-op set, with the corpus, held-out set, and probe set that ex-2.2.18 built."""

LO_LR = ex2219.PEAK_LR
"""The recipe peak rate, 0.00316: the peak of the second cycle, and of the plain runs this compares with."""

HI_LR = ex2219.SCOUT_LRS[1]
"""The higher peak rate of the ex-2.2.19 scout, 0.00562: the peak of the first cycle."""

BAND_LR = 0.0021
"""The second peak of the second condition: about the highest rate at which a plain 200-epoch run of ex-2.2.19 learned
the HSV-channel ops (0.00215, at seeds 600 and 602)."""

SECOND_PEAKS: tuple[float, ...] = (LO_LR, BAND_LR)
"""The peak of the second cycle in each condition."""

WARMUP_EPOCHS = ex2219.WARMUP_EPOCHS
"""Each cycle warms up over 5 epochs, the warmup of the recipe at every length."""

MIN_LR_FACTOR = 0.01
"""The recipe warms up from, and anneals to, 1% of its peak. The first cycle warms up from 1% of its own peak, as the
ex-2.2.19 scout run did, and anneals to 1% of the second peak, where the second warmup starts, so the schedule has no
jump."""

EPOCHS = 200
"""The length of every run: the length ex-2.2.19 adopted."""

REFERENCE_EPOCHS = ex2219.REFERENCE_EPOCHS
"""400 epochs, the length of the recipe before ex-2.2.19."""

HEAD_START_EPOCHS = 50
"""The length of the first cycle: the ex-2.2.19 scout run that prompted this experiment."""

SEED_OFFSET = ex2219.SEED_OFFSET
SEEDS: tuple[int, ...] = (ex2219.SCOUT_SEED, *ex2219.CONFIRM_SEEDS)
"""Seed indices 0-3, model seeds 600-603: the four initializations of the 200-epoch runs of ex-2.2.19."""

GATE_SEEDS: tuple[int, ...] = ex2219.CONFIRM_SEEDS
"""The seeds H1 is scored on, 601-603, as in ex-2.2.19. Seed 600 is reported beside them: the observation behind
this experiment came from its runs."""

SHORTFALL_TOL = ex2219.SHORTFALL_TOL
PARTIAL_TOL = ex2219.PARTIAL_TOL
YARDSTICK = ex2219.YARDSTICK
op_tolerances = ex2219.op_tolerances

KEYS_PER_EPOCH = 100
"""The sheet is keyed in hundredths of an epoch, so it does not depend on the number of steps in an epoch."""


def lr_sheet(second_peak: float = LO_LR) -> str:
    """The schedule as a dopesheet, in multiples of `HI_LR`. `lincos` rises linearly and falls along a half cosine,
    so each cycle is the recipe schedule for its own length and peak.
    """
    lo = second_peak / HI_LR
    keys = [
        (0, "Warmup", MIN_LR_FACTOR),
        (WARMUP_EPOCHS, "Anneal", 1.0),
        (HEAD_START_EPOCHS, "Warmup", MIN_LR_FACTOR * lo),
        (HEAD_START_EPOCHS + WARMUP_EPOCHS, "Anneal", lo),
        (EPOCHS, "", MIN_LR_FACTOR * lo),
    ]
    rows = "".join(f"{round(e * KEYS_PER_EPOCH)},{phase},,{v:.6g}\n" for e, phase, v in keys)
    return "STEP,PHASE,ACTION,lr::lincos\n" + rows


def schedule_check(epoch_length: int, second_peak: float = LO_LR) -> dict[str, float]:
    """The largest difference, as a share of each peak, between the realized sheet and the recipe schedule (optax
    warmup and cosine) of each cycle on its own: the 50-epoch run at `HI_LR`, and a 150-epoch run at *second_peak*.
    """
    import numpy as np

    from sca.config import SchedulerConfig
    from sca.training.scheduler import configure_schedule

    def recipe(epochs: int, peak: float, sheet: str | None = None):
        config = SchedulerConfig(
            epochs=epochs, warmup_epochs=WARMUP_EPOCHS, min_lr_factor=MIN_LR_FACTOR, lr_sheet=sheet
        )
        return configure_schedule(config, peak, epoch_length)

    split = HEAD_START_EPOCHS * epoch_length
    sheet = np.asarray(recipe(EPOCHS, HI_LR, lr_sheet(second_peak))(np.arange(EPOCHS * epoch_length + 1)))
    first = np.asarray(recipe(HEAD_START_EPOCHS, HI_LR)(np.arange(split + 1)))
    second = np.asarray(recipe(EPOCHS - HEAD_START_EPOCHS, second_peak)(np.arange(len(sheet) - split)))
    return {
        "first": float(np.abs(sheet[: split + 1] - first).max() / HI_LR),
        "second": float(np.abs(sheet[split:] - second).max() / second_peak),
    }


def cost_per_run(epochs: int) -> float:
    return ex2219.cost_per_run(epochs)


def label_of(second_peak: float, seed: int) -> str:
    return f"head-start-{second_peak:g}-s{seed}"


# =============================================================================================
# The DAG
# =============================================================================================


def rows_of(specs: list[tuple[float, int]], meta) -> list[dict]:
    """One row per (second peak, seed index): the 200-epoch `no-four` config of ex-2.2.19 at `HI_LR`, with the
    two-cycle sheet.
    """
    rows = []
    for second_peak, seed in specs:
        (row,) = ex2219.rows_of([(EPOCHS, HI_LR, seed)], meta)
        row["config"].scheduler.lr_sheet = lr_sheet(second_peak)
        row |= {"second_peak": second_peak, "stage": "head-start", "label": label_of(second_peak, seed)}
        rows.append(row)
    return rows


def check_corpus(stats: dict) -> dict:
    """The rebuilt corpus has the ceiling ex-2.2.19 published, so it is the same corpus."""
    import json
    import tempfile
    from pathlib import Path

    from mini.store import get, get_ref

    art = get_ref(ex2219.EVAL_REF)
    assert art is not None, f"{ex2219.EVAL_REF} is not published"
    published = json.loads(get(art, Path(tempfile.mkdtemp()) / "eval.json").read_text())["op_set"]["ceiling"]
    assert abs(stats["ceiling"] - published) < 1e-9, f"rebuilt corpus differs: {stats['ceiling']} vs {published}"
    return {"ceiling": published}


# --- Publishing ------------------------------------------------------------------------------

TRAJ_REF = "reports/m2/ex-2.2.20/trajectories"
EVAL_REF = "reports/m2/ex-2.2.20/eval"
EVAL_ARRAYS_REF = "reports/m2/ex-2.2.20/eval-arrays/{label}"
CHECKPOINT_REF = "reports/m2/ex-2.2.20/checkpoints/{label}"


def design() -> dict[str, Any]:
    """The design constants the report reads beside the results."""
    return {
        "experiment": "ex-2.2.20",
        "question": "whether a high-rate head start before the recipe schedule keeps more of the 400-epoch skill",
        "op_set": OP_SET.name,
        "ops": list(OP_SET.ops),
        "epochs": EPOCHS,
        "head_start_epochs": HEAD_START_EPOCHS,
        "hi_lr": HI_LR,
        "second_peaks": list(SECOND_PEAKS),
        "warmup_epochs": WARMUP_EPOCHS,
        "min_lr_factor": MIN_LR_FACTOR,
        "seed_offset": SEED_OFFSET,
        "seeds": list(SEEDS),
        "gate_seeds": list(GATE_SEEDS),
    }


def publish(trained: list[dict], rows: list[dict], evaled: list[dict], stats: dict) -> dict:
    """The trajectories (JSON); the evaluation with the design, every row, and the corpus statistics (JSON); every
    run's per-context arrays; and every end checkpoint, each under its own ref.
    """
    import json

    from mini.store import put, set_ref

    set_ref(TRAJ_REF, put(json.dumps({t["label"]: t["traj"] for t in trained}).encode(), name="ex-2.2.20-traj.json"))
    meta = {r["label"]: {k: v for k, v in r.items() if k not in ("config", "anchor")} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for e in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=e["label"]), e["arrays"])
    body = {
        "design": design(),
        "op_set": stats,
        "runs": [meta[e["label"]] | {k: v for k, v in e.items() if k != "arrays"} for e in evaled],
    }
    set_ref(EVAL_REF, put(json.dumps(body).encode(), name="ex-2.2.20-eval.json"))
    return {"eem": {e["label"]: e["task"]["eem"]["all"] for e in evaled}}


# --- Orchestration ----------------------------------------------------------------------------


def run(ctx: Ctx, n_lines: int, holdout_n: int, probe_n: int, n_traj_points: int, n_traj_eem: int) -> dict:
    """Rebuild the `no-four` corpus condition, then train and score every head-start run: the whole DAG, with sizes as
    arguments so a short prototype runs the same code.
    """
    k, rho = ex2218.CENTRE
    i = ex2218.OP_SETS.index(OP_SET)
    prep = ctx.run(
        ex2218.prepare_op_set,
        OP_SET.ops,
        k,
        rho,
        ex2216.CUBE_RATE,
        n_lines,
        ex2218.CORPUS_SEED + 3 * i,
        holdout_n,
        probe_n,
        f"{ex2216.cond_key(k, rho)}-{OP_SET.name}",
        role="prep",
    )
    if n_lines == ex2216.N_LINES:
        ctx.run(check_corpus, prep["stats"], role="prep")
    rows = rows_of([(p, s) for p in SECOND_PEAKS for s in SEEDS], prep["meta"])
    trained, evaled = ex2219.train_and_eval(ctx, rows, prep, n_traj_points, n_traj_eem)
    return ctx.run(publish, trained, rows, evaled, prep["stats"], role="prep")


def main(ctx: Ctx) -> dict:
    return run(
        ctx,
        ex2216.N_LINES,
        ex2216.HOLDOUT_CONTEXTS,
        ex2216.N_TRAJ_PROBE,
        ex2218.N_TRAJ_POINTS,
        ex2218.N_TRAJ_EEM_PER_OP,
    )


COMPUTE = {
    # The corpus build of 300,000 contexts and the posterior over its held-out set; the check; and the fan-in.
    "prep": dict(cpu=2, timeout=1800),
    # About 52,800 steps (200 epochs, about 20 minutes on an L4).
    "train": dict(gpu="L4", timeout=2 * 3600, watchdog=900, watchdog_grace=900),
    # Forward passes only, over the held-out contexts.
    "eval": dict(gpu="L4", timeout=900),
}

experiment = Experiment(name="ex-2.2.20", main=main, roles=COMPUTE)
