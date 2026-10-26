"""Resume check: a run stopped at epoch 64 and resumed to 400 ends as the stored straight run does.

`train_one` (ex-2.2.21) stops a run early with *stop_epoch*, which now also writes a resume state, and carries a
stopped run on with *resume*. Soft-pool trained `pool 1–3, soft τ` at seed 704 straight through, and spill-timing
trained it again to the same end, so its stored run is the reference. This experiment trains the same row stopped
after 64 epochs, resumes it to the end, and compares the resumed run with the stored one: every array of the end
checkpoint, every trajectory record, and the per-epoch losses. The kept checkpoints are taken along the way, so the
check also covers their layout (16 and 32 from the stopped run, 64 written at the resume, 128 and 256 after it).

    bin/mini run docs/m2/resume-check/experiment.py --app modal --budget 1h
    bin/mini status resume-check
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


sp = _load_sibling("soft-pool", "resume_check_sp")
ex2222 = sp.ex2222
ex2221 = sp.ex2221

ARM = "pool-123-soft"
SEED = 704
STOP = 64


def train_args(resolved: dict, row: dict) -> list:
    """`train_one`'s arguments for *row* up to *stop_epoch*, as soft-pool passes them."""
    epoch_length = ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, row["config"])
    return [
        row["config"], row["anchor"], row["anti"], "whole", row["anchor_slices"], "", sp.OP_NAMES,
        resolved["corpus"], resolved["labels"], resolved["probes"], resolved["holdout"], sp.K,
        max(1, round(sp.TRAJ_STRIDE_EPOCHS * epoch_length)), ex2221.N_TRAJ_EEM_PER_OP, row["label"], sp.EVERY,
        None, True,
    ]  # fmt: skip


def compare(stopped: dict, resumed: dict, label: str) -> dict[str, Any]:
    """The resumed run beside the stored straight run: the largest difference in any array of the end checkpoint,
    in any trajectory key and in the per-epoch losses (zero means bit for bit), and the kept checkpoints.
    """
    import json

    import equinox as eqx
    import jax
    import numpy as np

    from sca.compute.model import load_checkpoint
    from mini import get_data_dir
    from mini.store import get, get_ref

    workdir = get_data_dir() / "compare"

    def leaves(art, name: str) -> list[np.ndarray]:
        model, _, _ = load_checkpoint(get(art, workdir / name / "model"))
        return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))]

    def gap(a, b) -> float:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")

    stored_ckpt, stored_traj = get_ref(sp.CHECKPOINT_REF.format(label=label)), get_ref(sp.TRAJ_REF)
    assert stored_ckpt is not None and stored_traj is not None, "soft-pool's run is not pinned"
    stored = json.loads(get(stored_traj, workdir / "traj.json").read_text())[label]
    checkpoint = max(gap(a, b) for a, b in zip(leaves(stored_ckpt, "stored"), leaves(resumed["checkpoint"], "resumed"), strict=True))  # fmt: skip
    traj = {k: gap(v, resumed["traj"][k]) for k, v in stored["traj"].items()}
    kept64 = ex2221.kept_checkpoint(resumed["kept_checkpoints"], STOP, workdir / "kept")[0]
    at_stop = leaves(stopped["checkpoint"], "stopped")
    return {
        "checkpoint": checkpoint,
        "traj": traj,
        "traj_keys_match": set(stored["traj"]) == set(resumed["traj"]),
        "val_loss": gap(stored["val_loss"], resumed["val_loss"]),
        "train_loss": gap(stored["train_loss"], resumed["train_loss"]),
        "kept_epochs": resumed["kept_epochs"],
        "kept_64_is_the_stop": max(
            gap(a, b) for a, b in zip(at_stop, jax.tree.leaves(eqx.filter(kept64, eqx.is_array)), strict=True)
        ),
        "stopped_records": len(stopped["traj"]["step"]),
        "resumed_records": len(resumed["traj"]["step"]),
    }


def run(ctx: Ctx, epochs: int = sp.EPOCHS, stop: int = STOP, keep: tuple[int, ...] = ex2221.KEEP_EPOCHS) -> dict:
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    (row,) = [r for r in sp.rows_of(resolved["meta"], epochs, (SEED,)) if r["arm"]["name"] == ARM]
    args = train_args(resolved, row)
    stopped = ctx.run(ex2221.train_one, *args, stop, None, keep, role="train")
    resumed = ctx.run(ex2221.train_one, *args, None, None, keep, stopped, role="train")
    if epochs != sp.EPOCHS:
        return {"stopped": len(stopped["traj"]["step"]), "resumed": len(resumed["traj"]["step"])}
    return ctx.run(compare, stopped, resumed, row["label"], role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """Eight epochs stopped at four, through the same path."""
    return run(ctx, epochs=8, stop=4, keep=(2, 4, 6))


COMPUTE = sp.COMPUTE

experiment = Experiment(name="resume-check", main=main, roles=COMPUTE)
