"""Ex-2.2.19: training length and seeds for the seven-op set.

Ex-2.2.18 trained ex-2.2.17's recipe once on each of several smaller op sets. Dropping `screen`, `multiply`, `hsvmix`,
and `exclusion` (`no-four`) raised the Bayes ceiling and narrowed the gap to it, at one seed, and the skill curves
suggested that the last fifth of the 400-epoch schedule added little. This experiment asks how short a `no-four` run
can be, in two stages:

1. A scout at model seed 600 (the initialization of the ex-2.2.18 `no-four` run) at 50, 100, and 200 epochs. A
   frozen rule picks the length to confirm from these runs and the ex-2.2.18 run at 400.
2. Three fresh model seeds (601-603) at the chosen length and at 400 epochs, so that each comparison of lengths is
   paired on the initialization. Seeds 601 and 602 are also the initializations of ex-2.2.17's `sweep-0.00316-s1`
   and `-s2` on the full op set, so the 400-epoch runs pair with those too.

The corpus condition is rebuilt from ex-2.2.18's seed (the memo store is per experiment), which gives the same
corpus; `select` checks that its ceiling matches the one ex-2.2.18 published. Training and evaluation are ex-2.2.18's
own tasks.

    bin/mini run docs/m2/ex-2.2.19/experiment.py --app modal --max-containers 6 --budget 3h
    bin/mini status ex-2.2.19
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from mini import Ctx, Experiment


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules` (the pattern ex-2.2.16 to ex-2.2.18
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


ex2218 = _load_sibling("ex-2.2.18", "ex2219_ex2218")
ex2217 = ex2218.ex2217
ex2216 = ex2218.ex2216

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_SET = ex2218.op_set("no-four")
"""The seven-op set, and its corpus condition as ex-2.2.18 built it (the same corpus, held-out set, and probe set)."""

PEAK_LR = ex2218.PEAK_LR
WARMUP_EPOCHS = ex2218.WARMUP_EPOCHS
"""A warmup of fixed length (5 epochs) at every length, as ex-2.2.17 fixed it from round 2 on. At 50 epochs this is
a tenth of the run, ex-2.2.16's own rule; at 400 it is an eightieth."""

REFERENCE_EPOCHS = ex2218.EPOCHS
"""400 epochs, the length of ex-2.2.17's recipe and of every ex-2.2.18 run."""

SEED_OFFSET = ex2218.SEED_OFFSET

# --- Stage 1: the scout ---------------------------------------------------------------------------------------

SCOUT_EPOCHS: tuple[int, ...] = (50, 100, 200)
"""The lengths the scout trains, in epochs: one, two, and four times ex-2.2.16's length. The 400-epoch point is the
ex-2.2.18 `no-four` run, which starts from the same weights."""

SCOUT_LRS: tuple[float, ...] = (PEAK_LR, ex2217.SWEEP_LRS[1])
"""The peak learning rates the scout trains at each length: the recipe rate, and the next rate up on ex-2.2.17's
quarter-decade grid (0.00562). A shorter run spends less time at a high rate, so its best peak rate may be higher. At
400 epochs, ex-2.2.17 found the two about level at one seed, and 0.01 lower."""

SCOUT_SEED = 0
"""Seed index 0, model seed 600: the initialization of the ex-2.2.18 `no-four` run."""

SHORTFALL_TOL = 0.015
"""How far below the 400-epoch run a shorter run may fall in held-out expected exact match and still count as keeping
most of its skill. It is the largest gain over the last fifth of training that ex-2.2.18 (E4) logged, and about half
the seed range of ex-2.2.17's three runs of this recipe on the full set."""

# --- Stage 2: fresh seeds -------------------------------------------------------------------------------------

CONFIRM_SEEDS: tuple[int, ...] = (1, 2, 3)
"""Seed indices of the confirmation runs, model seeds 601-603. None of them helped choose the length."""

MAX_CONFIRM_LENGTHS = 2
"""The chosen length, and twice it as a fallback when that is still shorter than the reference. Each trains at the
peak rate that did better at that length in the scout; the reference stays at `PEAK_LR`."""

PARTIAL_TOL = 0.03
"""The partial band of H1: about the seed range of ex-2.2.17's three runs on the full set."""


def chosen_lengths(pick: int) -> tuple[int, ...]:
    """The lengths stage 2 trains beside the reference, given the length the selection rule picked."""
    return tuple(e for e in (pick, 2 * pick) if e < REFERENCE_EPOCHS)[:MAX_CONFIRM_LENGTHS]


def cost_per_run(epochs: int) -> float:
    """Dollars of L4 time for one run, scaled from ex-2.2.18's $0.26 at 400 epochs."""
    return 0.26 * epochs / REFERENCE_EPOCHS


def op_tolerances(yardstick: list[dict], ops: tuple[str, ...]) -> dict[str, float]:
    """The tolerance of each op in *ops*: the seed range of its gap to the ceiling over the *yardstick* runs (ex-2.2.17
    on the full set, as published), or `PARTIAL_TOL`, whichever is larger.
    """

    def gap(run: dict, op: str) -> float:
        i = list(run.get("ops", ex2216.OP_NAMES)).index(op)
        return run["task"]["ceiling"]["per_op"][i] - run["task"]["eem"]["per_op"][i]

    return {op: max(max(gap(r, op) for r in yardstick) - min(gap(r, op) for r in yardstick), PARTIAL_TOL) for op in ops}


YARDSTICK = tuple(f"sweep-{PEAK_LR:g}-s{s}" for s in range(3))
"""Ex-2.2.17's three seeds of this recipe on the full set."""


def label_of(epochs: int, lr: float, seed: int) -> str:
    return f"e{epochs}-lr{lr:g}-s{seed}"


# =============================================================================================
# The DAG
# =============================================================================================


def rows_of(specs: list[tuple[int, float, int]], meta) -> list[dict]:
    """One row per (epochs, peak rate, seed index): ex-2.2.18's config for `no-four` at that length, rate, and model
    seed.
    """
    from sca.config import ModelConfig
    from sca.data.named_colors import WordTokenizer
    from sca.utils import align

    rows = []
    for epochs, lr, seed in specs:
        config = ex2216._make_config(
            align(meta.tokenizer_config.vocab_size, 64), SEED_OFFSET + seed, epochs, *ex2216.model_dims(ex2216.MODEL)
        )
        config.tokenizer = meta.tokenizer_config.model_copy()
        config.model.block_size = ex2216.BLOCK
        config.model.tie_embeddings = False
        config.optimizer.learning_rate = lr
        config.model = ModelConfig.model_validate(
            config.model.model_dump() | {"line_mask_token": WordTokenizer(config.tokenizer).stoi["\n"]}
        )
        config.scheduler.warmup_epochs = WARMUP_EPOCHS
        base = ex2216.Condition229(
            OP_SET.name, 1, OP_SET.name, lam=0.0, tau=ex2216.TAU, epochs=epochs, ops=OP_SET.ops, n_lines=ex2216.N_LINES
        )
        anchor, anti = ex2216.schedules(base)
        assert anti is None, "every run is unanchored"
        rows.append(
            {
                "config": config,
                "anchor": anchor,
                "op_set": OP_SET.name,
                "ops": list(OP_SET.ops),
                "epochs": epochs,
                "peak_lr": lr,
                "seed_index": seed,
                "model_seed": SEED_OFFSET + seed,
                "stage": "scout" if seed == SCOUT_SEED else "confirm",
                "label": label_of(epochs, lr, seed),
            }
        )
    return rows


def select(scout: list[dict], rows: list[dict], stats: dict) -> dict:
    """The frozen rule of S1. At each scout length, take the rate with the higher EEM; pick the shortest length that
    falls short of the ex-2.2.18 `no-four` run by at most `SHORTFALL_TOL`, with no op short by more than its
    tolerance. Returns the pick (None if no length passes), the rate taken at each length, and the arithmetic.
    """
    import json

    import tempfile
    from pathlib import Path

    from mini.store import get, get_ref

    tmp = Path(tempfile.mkdtemp())

    def fetch(name: str) -> dict:
        art = get_ref(name)
        assert art is not None, f"{name} is not published"
        return json.loads(get(art, tmp / name.replace("/", "-")).read_text())

    ref18 = fetch(ex2218.EVAL_REF)
    ref17 = fetch(ex2217.EVAL_REF)
    ref = next(r for r in ref18["runs"] if r["label"] == OP_SET.name)
    published = ref18["op_sets"][OP_SET.name]["ceiling"]
    assert abs(stats["ceiling"] - published) < 1e-9, f"rebuilt corpus differs: {stats['ceiling']} vs {published}"
    tol = op_tolerances([r for r in ref17["runs"] if r["label"] in YARDSTICK], OP_SET.ops)

    by_label = {e["label"]: e for e in scout}
    lengths = []
    for epochs in SCOUT_EPOCHS:
        cands = [(r, by_label[r["label"]]) for r in rows if r["epochs"] == epochs]
        row, ev = max(cands, key=lambda c: c[1]["task"]["eem"]["all"])
        short = ref["task"]["eem"]["all"] - ev["task"]["eem"]["all"]
        short_op = {
            op: ref["task"]["eem"]["per_op"][i] - ev["task"]["eem"]["per_op"][i] for i, op in enumerate(OP_SET.ops)
        }
        over = [op for op in OP_SET.ops if short_op[op] > tol[op]]
        lengths.append(
            {
                "epochs": epochs,
                "peak_lr": row["peak_lr"],
                "eem_by_lr": {f"{c[0]['peak_lr']:g}": c[1]["task"]["eem"]["all"] for c in cands},
                "shortfall": short,
                "shortfall_per_op": short_op,
                "ops_over": over,
                "passes": short <= SHORTFALL_TOL and not over,
            }
        )
    pick = next((x["epochs"] for x in lengths if x["passes"]), None)
    return {
        "pick": pick,
        "rate": {str(x["epochs"]): x["peak_lr"] for x in lengths},
        "lengths": lengths,
        "op_tol": tol,
        "reference_eem": ref["task"]["eem"]["all"],
    }


# --- Publishing ------------------------------------------------------------------------------

TRAJ_REF = "reports/m2/ex-2.2.19/trajectories"
EVAL_REF = "reports/m2/ex-2.2.19/eval"
EVAL_ARRAYS_REF = "reports/m2/ex-2.2.19/eval-arrays/{label}"
CHECKPOINT_REF = "reports/m2/ex-2.2.19/checkpoints/{label}"


def design() -> dict[str, Any]:
    """The design constants the report reads beside the results."""
    return {
        "experiment": "ex-2.2.19",
        "question": "how short a run on the seven-op set keeps most of the skill of the 400-epoch recipe",
        "op_set": OP_SET.name,
        "ops": list(OP_SET.ops),
        "scout_epochs": list(SCOUT_EPOCHS),
        "scout_lrs": list(SCOUT_LRS),
        "reference_epochs": REFERENCE_EPOCHS,
        "peak_lr": PEAK_LR,
        "warmup_epochs": WARMUP_EPOCHS,
        "seed_offset": SEED_OFFSET,
        "scout_seed": SCOUT_SEED,
        "confirm_seeds": list(CONFIRM_SEEDS),
        "shortfall_tol": SHORTFALL_TOL,
        "partial_tol": PARTIAL_TOL,
    }


def publish(trained: list[dict], rows: list[dict], evaled: list[dict], selection: dict, stats: dict) -> dict:
    """The trajectories (JSON); the evaluation with the design, the selection, every row, and the corpus statistics
    (JSON); every run's per-context arrays; and every end checkpoint, each under its own ref.
    """
    import json

    from mini.store import put, set_ref

    set_ref(TRAJ_REF, put(json.dumps({t["label"]: t["traj"] for t in trained}).encode(), name="ex-2.2.19-traj.json"))
    meta = {r["label"]: {k: v for k, v in r.items() if k not in ("config", "anchor")} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    for e in evaled:
        set_ref(EVAL_ARRAYS_REF.format(label=e["label"]), e["arrays"])
    body = {
        "design": design(),
        "selection": selection,
        "op_set": stats,
        "runs": [meta[e["label"]] | {k: v for k, v in e.items() if k != "arrays"} for e in evaled],
    }
    set_ref(EVAL_REF, put(json.dumps(body).encode(), name="ex-2.2.19-eval.json"))
    return {"pick": selection["pick"], "eem": {e["label"]: e["task"]["eem"]["all"] for e in evaled}}


# --- Orchestration ----------------------------------------------------------------------------


def train_and_eval(ctx: Ctx, rows: list[dict], prep: dict, n_traj_points: int, n_traj_eem: int):
    k, _ = ex2218.CENTRE
    n = len(rows)
    stride = [
        ex2217.traj_stride_for(
            r["epochs"], ex2217.epoch_length_of(prep["meta"].total_tokens, r["config"]), n_traj_points
        )
        for r in rows
    ]
    trained = ctx.map(
        ex2218.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [tuple(r["ops"]) for r in rows],
        [prep["corpus"]] * n,
        [prep["labels"]] * n,
        [prep["probes"]] * n,
        [prep["holdout"]] * n,
        [k] * n,
        stride,
        [n_traj_eem] * n,
        [r["label"] for r in rows],
        role="train",
    )
    evaled = ctx.map(
        ex2218.eval_one,
        [t["checkpoint"] for t in trained],
        [prep["holdout"]] * n,
        [tuple(r["ops"]) for r in rows],
        [k] * n,
        [r["label"] for r in rows],
        role="eval",
    )
    return trained, evaled


def run(ctx: Ctx, n_lines: int, holdout_n: int, probe_n: int, n_traj_points: int, n_traj_eem: int) -> dict:
    """Rebuild the `no-four` corpus condition, train and score the scout, apply the rule, then train and score the
    confirmation: the whole DAG, with sizes as arguments so a short prototype runs the same code.
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
    scout_rows = rows_of([(e, lr, SCOUT_SEED) for e in SCOUT_EPOCHS for lr in SCOUT_LRS], prep["meta"])
    scout_trained, scout_evaled = train_and_eval(ctx, scout_rows, prep, n_traj_points, n_traj_eem)
    selection = ctx.run(select, scout_evaled, scout_rows, prep["stats"], role="prep")

    picks = chosen_lengths(selection["pick"]) if selection["pick"] is not None else ()
    specs = [(e, selection["rate"][str(e)], s) for e in picks for s in CONFIRM_SEEDS]
    specs += [(REFERENCE_EPOCHS, PEAK_LR, s) for s in CONFIRM_SEEDS]
    confirm_rows = rows_of(specs, prep["meta"])
    confirm_trained, confirm_evaled = train_and_eval(ctx, confirm_rows, prep, n_traj_points, n_traj_eem)

    return ctx.run(
        publish,
        scout_trained + confirm_trained,
        scout_rows + confirm_rows,
        scout_evaled + confirm_evaled,
        selection,
        prep["stats"],
        role="prep",
    )


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
    # The corpus build of 300,000 contexts and the posterior over its held-out set; the rule; and the fan-in.
    "prep": dict(cpu=2, timeout=1800),
    # Up to about 105,600 steps (400 epochs, about 40 minutes on an L4).
    "train": dict(gpu="L4", timeout=3 * 3600, watchdog=900, watchdog_grace=900),
    # Forward passes only, over the held-out contexts.
    "eval": dict(gpu="L4", timeout=900),
}

experiment = Experiment(name="ex-2.2.19", main=main, roles=COMPUTE)
