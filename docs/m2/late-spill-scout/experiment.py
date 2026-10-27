"""Late spill scout: three late-schedule changes to `τ_ℓ 0.1, τ_ℓ fixed`, carried on from its epoch-64 states.

The soft pool scout's candidate recipe, the pool over slices 1 to 3 with τ_ℓ held at 0.1, has no early spill, but on
one or two of its eight seeds the spill rises after epoch 128 and ends near the criterion. That rise comes while the
anti weight anneals from its peak to the per-slice holds (epochs 0 to 360). Every arm here changes a schedule only
after epoch 64, so each run carries on from the soft pool scout's stored epoch-64 state of the same seed and shares
its first 64 epochs with the finished runs of that condition:

- `anti-late`: the anti anneal pauses at epoch 64, holding the weight it has reached, and finishes from epoch 200 to
  360, so the anti weight stays high while the late ops are learned and reaches the same holds;
- `anti-held`: the pause runs to 360, so the anti weight never anneals to the holds, only with the anchor at the end;
- `lt-late`: the pool over slices sharpens again, from τ_ℓ 0.1 to 0.025 over epochs 64 to 360, alongside the anti
  anneal.

A fourth set of runs, `base`, carries the condition on unchanged: it repeats the soft pool scout's finished runs bit
for bit and keeps the same dense checkpoints as the arms, so spill along training compares at every 32 epochs.

    bin/mini run docs/m2/late-spill-scout/experiment.py --app modal --max-containers 10 --budget 3h
    bin/mini status late-spill-scout
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
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


sps = _load_sibling("soft-pool-scout", "late_spill_scout_sps")
spt = sps.spt
rb = sps.rb
ee = sps.ee
lean = sps.lean
ex2222 = sps.ex2222
ex2221 = sps.ex2221

OP_NAMES = sps.OP_NAMES
ANCHORED_OP = sps.ANCHORED_OP
K = sps.K
EPOCHS = sps.EPOCHS
SEEDS = sps.SEEDS
STOP_EPOCH = sps.STOP_EPOCH
"""Where the stored runs stopped, and where every arm starts to differ."""
CONDITION = "lt0.1-fixed-lt"
"""The soft pool scout's condition every arm carries on: τ_ℓ 0.1 throughout, τ_p 1 → 0.1 by epoch 40."""
KEEP_EPOCHS = tuple(range(96, EPOCHS, 32))
"""Checkpoints kept after the stop, every 32 epochs (the stored runs kept 16, 32 and 64)."""
ANTI_END = 360.0
"""Where the recipe's anti anneal reaches the per-slice holds, and the anchor's end anneal starts."""
LATE_SLICE_TAU = 0.025
"""Where `lt-late` takes τ_ℓ: two more halvings past 0.1, continuing stage 1's ladder of 0.4, 0.2, 0.1."""


@dataclass(frozen=True)
class Arm:
    name: str
    shown_as: str
    anti_pause: tuple[float, float] | None = None
    """`AntiSpec.pause`: the anti anneal stops at the first epoch and finishes from the second."""
    slice_tau_late: tuple[float, float, float] | None = None
    """`AnchorSpec.slice_tau_late`: τ_ℓ moves again, to the value, between the two epochs."""

    def schedules(self, anchor: dict, anti: dict, f: float = 1.0) -> tuple[dict, dict]:
        """The arm's anchor and anti keywords on the condition's, with its epochs scaled by *f* (a smoke run). Only
        the fields it sets enter the specs, so `base` keeps the soft pool scout's.
        """
        if self.anti_pause is not None:
            anti = anti | {"pause": tuple(e * f for e in self.anti_pause)}
        if self.slice_tau_late is not None:
            v, a, b = self.slice_tau_late
            anchor = anchor | {"slice_tau_late": (v, a * f, b * f)}
        return anchor, anti


ARMS: tuple[Arm, ...] = (
    Arm("base", "τ_ℓ fixed (base)"),
    Arm("anti-late", "anti late", anti_pause=(STOP_EPOCH, 200.0)),
    Arm("anti-held", "anti held", anti_pause=(STOP_EPOCH, ANTI_END)),
    Arm("lt-late", "τ_ℓ late", slice_tau_late=(LATE_SLICE_TAU, STOP_EPOCH, ANTI_END)),
)

BUDGET_USD = 15
"""Each run carries on for 336 of 400 epochs, about \\$0.30 with its measurements at the soft pool scout's rate; 32
runs, plus scoring ten kept checkpoints each."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/late-spill-scout"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
KEPT_REF = PREFIX + "/kept/{label}"
TRAJ_REF = PREFIX + "/trajectories"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"
AXES_REF = PREFIX + "/axes"
WRITE_REF = PREFIX + "/write"
WRITE_ARRAYS_REF = PREFIX + "/write-arrays/{label}"
ALONG_REF = PREFIX + "/along"


def rows_of(meta, arms: tuple[Arm, ...] = ARMS, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run of each arm at *seeds*: the soft pool scout's straight-run row of the condition, with the
    arm's schedules.
    """
    condition = next(c for c in sps.CONDITIONS if c.name == CONDITION)
    f = epochs / EPOCHS
    rows = []
    for a in arms:
        for r in sps.rows_of(meta, (condition,), None, epochs, seeds):
            anchor, anti = a.schedules(r["anchor"], r["anti"], f)
            rows.append(r | {"anchor": anchor, "anti": anti, "arm": asdict(a), "label": label_of(a.name, r["model_seed"])})  # fmt: skip
    return rows


def stored_resume(seeds: tuple[int, ...]) -> dict[int, dict]:
    """The soft pool scout's epoch-64 resume state and kept checkpoints of the condition at each seed, from their
    refs (the memo records they came from expire).
    """
    from mini.store import get_ref

    out = {}
    for s in seeds:
        label = sps.label_of(CONDITION, s)
        state, kept = get_ref(sps.RESUME_REF.format(label=label)), get_ref(sps.KEPT_REF.format(label=label))
        assert state is not None and kept is not None, f"not stored: {label}"
        out[s] = {"resume_state": state, "kept_checkpoints": kept}
    return out


def train(ctx: Ctx, resolved: dict, rows: list[dict], resume: list[dict], keep: tuple[int, ...]) -> list[dict]:
    """`train_one` on every row with the alignment map, carrying on from *resume* and keeping *keep*."""
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(sps.TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    return ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        ["whole"] * n,
        [r["anchor_slices"] for r in rows],
        [""] * n,
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [ex2221.N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        [sps.EVERY] * n,
        [None] * n,
        [True] * n,
        [None] * n,
        [None] * n,
        [keep] * n,
        resume,
        role="train",
    )


def design() -> dict[str, Any]:
    return {
        "experiment": "late-spill-scout",
        "condition": CONDITION,
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "stop_epoch": STOP_EPOCH,
        "keep_epochs": list(KEEP_EPOCHS),
        "arms": [asdict(a) for a in ARMS],
        "seeds": list(SEEDS),
    }


def publish(rows: list[dict], trained: list[dict], measured: list[dict], dosed: list[dict], axes: list[dict],
            written: list[dict], along: list[list[dict]]) -> dict:  # fmt: skip
    """Trajectories, checkpoints, measurements, dose scoring, the axis scoring, the write bound, and the scoring of
    the kept checkpoints, under refs of their own.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="late-spill-scout-trajectories.json"))

    def body(runs: list[dict]) -> bytes:
        return json.dumps({"design": design(), "runs": [meta[r["label"]] | r for r in runs]}).encode()

    set_ref(RESULTS_REF, put(body(measured), name="late-spill-scout-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    set_ref(DOSE_REF, put(body([{k: v for k, v in d.items() if k != "arrays"} for d in dosed]), name="late-spill-scout-dose.json"))  # fmt: skip
    set_ref(AXES_REF, put(body(axes), name="late-spill-scout-axes.json"))
    for w in written:
        set_ref(WRITE_ARRAYS_REF.format(label=w["label"]), w["arrays"])
    set_ref(WRITE_REF, put(body([{k: v for k, v in w.items() if k != "arrays"} for w in written]), name="late-spill-scout-write.json"))  # fmt: skip
    set_ref(ALONG_REF, put(body([r for a in along for r in a]), name="late-spill-scout-along.json"))
    return {"n_runs": len(trained)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, arms: tuple[Arm, ...] = ARMS,
        resume: dict[int, dict] | None = None, keep: tuple[int, ...] = KEEP_EPOCHS, full: bool = True) -> dict:  # fmt: skip
    """Carry every arm on from the stored epoch-64 states, then measure and score it as the soft pool scout's stage 2
    did, scoring the kept checkpoints from the stop on. The arguments let a smoke run take the same path at a
    fraction of the size, from stops of its own, and leave the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    rows = rows_of(resolved["meta"], arms, epochs, seeds)
    if resume is None:
        resume = ctx.run(stored_resume, seeds, role="prep")
    trained = train(ctx, resolved, rows, [resume[r["model_seed"]] for r in rows], keep)
    n = len(rows)
    labels = [r["label"] for r in rows]
    holdout = [resolved["holdout"]] * n
    checkpoints = [t["checkpoint"] for t in trained]
    stop = STOP_EPOCH if epochs == EPOCHS else min(keep) - 1
    measured = ctx.map(lean.measure_one, checkpoints, holdout, labels, [True] * n, role="measure")
    dosed = ctx.map(ex2222.suppress_one, checkpoints, holdout, [OP_NAMES] * n, [K] * n, [True] * n, labels, role="measure")  # fmt: skip
    along = ctx.map(
        sps.score_kept,
        [t["kept_checkpoints"] for t in trained],
        checkpoints,
        holdout,
        labels,
        [tuple(e for e in t["kept_epochs"] if e >= stop) for t in trained],
        role="measure",
    )
    if not full:  # a smoke run leaves out the two slow passes and the refs
        return {"runs": labels, "along": [len(a) for a in along]}
    axes = ctx.map(rb.axes_one, checkpoints, holdout, labels, [None] * n, role="measure")
    written = ctx.map(ee.measure_one, checkpoints, holdout, labels, role="write")
    return ctx.run(publish, rows, trained, measured, dosed, axes, written, along, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of every arm for eight epochs, carried on from a stop at epoch 3 of the condition made here (the
    stored stops are at epoch 64 of 400), with the arms' epochs scaled to the length.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    condition = next(c for c in sps.CONDITIONS if c.name == CONDITION)
    stopped = sps.train(ctx, resolved, sps.rows_of(resolved["meta"], (condition,), 3, 8, SEEDS[:1]))
    resume = {SEEDS[0]: {k: stopped[0][k] for k in ("resume_state", "kept_checkpoints")}}
    return run(ctx, epochs=8, seeds=SEEDS[:1], resume=resume, keep=(4, 6), full=False)


COMPUTE = sps.COMPUTE

experiment = Experiment(name="late-spill-scout", main=main, roles=COMPUTE)
