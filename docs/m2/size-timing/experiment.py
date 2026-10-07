"""Size timing: the cost of a training step at each model size, on the scanned loop.

`todo/science/cheaper-center-control-recipe.md` plans a sweep of model size (d32, d64, d128; L2, L4) on the center
control, at the length ex-2.2.19 settles. Its premise was that a d128 step costs the same as a d64 step, which held
while the loop was latency-bound. Since `train_anchored` runs sixteen steps per dispatch, the L4 does real work, so the
cost per step may now differ between sizes. This probe measures it before the sweep is designed around it.

Each task trains every size for a few epochs, one after another in the same container, on ex-2.2.18's `no-four`
corpus condition and recipe (the masked cosine at a peak learning rate of 0.00316, the untied readout). A task
timestamps each trajectory record and reports the steady-state milliseconds per step, past the compile, with
validation and batch sampling included, as in a full run. Timing every size in one container compares them on the
same host, since about one container in three or four runs slower (`todo/eng/training-step-is-host-bound.md`); the
order of sizes rotates between tasks. The trajectory's expected-exact-match read of ex-2.2.18 is left out: it runs
about fifty times per run whatever its length, a fixed cost beside the steps.

    MINI_PROFILE=dev bin/mini run docs/m2/size-timing/experiment.py --app modal --max-containers 3 --budget 1h
    bin/mini status size-timing
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


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


ex2218 = _load_sibling("ex-2.2.18", "sizetiming_ex2218")
ex2216 = ex2218.ex2216

OP_SET = ex2218.op_set("no-four")
OP_SET_INDEX = ex2218.OP_SETS.index(OP_SET)
# The corpus seed of ex-2.2.18's `no-four` set, so the corpus is the one ex-2.2.19 trains on.

SIZES: tuple[str, ...] = ("d32-L2", "d32-L4", "d64-L2", "d64-L4", "d128-L2", "d128-L4")
# The sizes the sweep would train. d64-L4 is the current recipe.

TIMED_EPOCHS = 8
# Epochs per size: about 2,100 steps at 264 steps an epoch, a few seconds of compile and then enough steady-state
# steps that a record's timestamp jitter is small beside the span.

RECORDS_PER_EPOCH = 2
# Trajectory records, and so timestamps, per epoch. The first two records hold the compile and are dropped.

N_REPLICAS = 3
# Tasks, each timing every size in its own container.


def head_layout(n_embd: int) -> tuple[int, int]:
    """(n_head, n_head_dim) for a width: eight heads of `n_embd // 8`, as ex-2.2.9's `_make_config` sets them, but
    never under eight dimensions a head (`ModelConfig` wants a multiple of eight), so d32 gets four heads of eight.
    """
    n_head = min(8, n_embd // 8)
    return n_head, n_embd // n_head


def size_config(meta, size: str, epochs: int):
    """Ex-2.2.18's run config (`cells`) at the model size *size* and *epochs* epochs."""
    from sca.config import ModelConfig
    from sca.data.named_colors import WordTokenizer
    from sca.utils import align

    n_embd, n_layer = ex2216.model_dims(size)
    n_head, n_head_dim = head_layout(n_embd)
    config = ex2216._make_config(align(meta.tokenizer_config.vocab_size, 64), ex2218.SEED_OFFSET, epochs, 64, n_layer)
    config.tokenizer = meta.tokenizer_config.model_copy()
    config.model.block_size = ex2216.BLOCK
    config.model.tie_embeddings = False
    config.optimizer.learning_rate = ex2218.PEAK_LR
    config.model = ModelConfig.model_validate(
        config.model.model_dump()
        | {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_head_dim": n_head_dim,
            "n_ff": 4 * n_embd,
            "line_mask_token": WordTokenizer(config.tokenizer).stoi["\n"],
        }
    )
    config.scheduler.warmup_epochs = ex2218.WARMUP_EPOCHS
    return config


def time_sizes(meta, corpus, labels, probes, sizes: tuple[str, ...], epochs: int, replica: int) -> dict:
    """Train each of *sizes* for *epochs* epochs in this container, in order, and time the steps."""
    import time
    from pathlib import Path

    import jax
    import jax.tree_util as jtu

    from sca.anchoring import AnchorSpec, LabelSpec
    from sca.compute.training import train_anchored
    from sca.data.named_colors import WordTokenizer
    from sca.data.incontext import context_length
    from mini.store import get

    k, _ = ex2218.CENTRE
    ops = OP_SET.ops
    workdir = get_data_dir() / "timing" / f"r{replica}"
    corpus_dir = get(corpus, workdir / "corpus")
    with np.load(get(labels, workdir / "labels.npz")) as z:
        op_ids, context_len = z["op_ids"], z["context_len"]
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["tokens"]
    per_op = len(probe_tokens) // len(ops)
    weights = np.concatenate([np.full(per_op, 1.0 if o == ex2216.ANCHORED_OP else 0.0) for o in ops])
    weights = weights / weights.sum()
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if "model name" in line
        ),
        "unknown",
    )

    results = []
    for size in sizes:
        config = size_config(meta, size, epochs)
        tokenizer = WordTokenizer(config.tokenizer)
        spec = LabelSpec(
            p=np.zeros(config.model.vocab_size),
            keying="context",
            context_op=op_ids,
            anchored_op_id=ops.index(ex2216.ANCHORED_OP),
            label_rate=ex2216.LABEL_RATE,
            variant="whole",
            context_len=context_len,
        )
        base = ex2216.Condition229(
            size, 1, size, lam=0.0, tau=ex2216.TAU, epochs=epochs, ops=ops, n_lines=ex2216.N_LINES
        )
        anchor, _ = ex2216.schedules(base)
        stride = max(1, ex2218.ex2217.epoch_length_of(meta.total_tokens, config) // RECORDS_PER_EPOCH)
        stamps: list[float] = []
        n_params: list[int] = []

        def on_record(_index: int, model, stamps=stamps, n_params=n_params) -> None:
            stamps.append(time.perf_counter())
            if not n_params:
                n_params.append(sum(x.size for x in jtu.tree_leaves(model) if hasattr(x, "size")))

        t0 = time.perf_counter()
        _, _, traj = train_anchored(
            config,
            corpus_dir,
            anchor=AnchorSpec(**anchor),
            anti=None,
            label_p=spec,
            probe_tokens=probe_tokens,
            probe_weights=weights,
            crop=ex2216.CROP_POLICY,
            anchor_slices=None,
            checkpoint_dir=workdir / size,
            traj_stride=stride,
            newline_id=tokenizer.stoi["\n"],
            min_line_tokens=context_length(k),
            on_record=on_record,
        )
        wall = time.perf_counter() - t0
        steps = np.asarray(traj["step"])
        t = np.asarray(stamps)
        # The first record is at step 0 and the second follows the first dispatches, which compile.
        ms_per_step = 1000 * (t[-1] - t[2]) / (steps[-1] - steps[2])
        results.append(
            {
                "size": size,
                "n_params": n_params[0] if n_params else None,
                "steps": int(steps[-1]),
                "ms_per_step": float(ms_per_step),
                "wall_s": float(wall),
                "overhead_s": float(wall - ms_per_step * steps[-1] / 1000),
            }
        )
        print(f"{size}: {ms_per_step:.2f} ms/step over {steps[-1] - steps[2]} steps, {wall:.0f} s in all", flush=True)

    return {"replica": replica, "device": jax.devices()[0].device_kind, "cpu": cpu, "sizes": results}


def main(ctx: Ctx) -> dict:
    k, rho = ex2218.CENTRE
    prep = ctx.run(
        ex2218.prepare_op_set,
        OP_SET.ops,
        k,
        rho,
        ex2216.CUBE_RATE,
        ex2216.N_LINES,
        ex2218.CORPUS_SEED + 3 * OP_SET_INDEX,
        ex2216.HOLDOUT_CONTEXTS,
        ex2216.N_TRAJ_PROBE,
        f"{ex2216.cond_key(k, rho)}-{OP_SET.name}",
        role="prep",
    )
    orders = [SIZES[r % len(SIZES) :] + SIZES[: r % len(SIZES)] for r in range(0, 2 * N_REPLICAS, 2)]
    timed = ctx.map(
        time_sizes,
        [prep["meta"]] * N_REPLICAS,
        [prep["corpus"]] * N_REPLICAS,
        [prep["labels"]] * N_REPLICAS,
        [prep["probes"]] * N_REPLICAS,
        orders,
        [TIMED_EPOCHS] * N_REPLICAS,
        list(range(N_REPLICAS)),
        role="train",
    )
    return {"timed": timed}


COMPUTE: dict[str, Any] = {
    "prep": dict(cpu=2, timeout=1800),
    # Six sizes of about 2,100 steps each, a few minutes in all on an L4.
    "train": dict(gpu="L4", timeout=1800, watchdog=600, watchdog_grace=900),
}

experiment = Experiment(name="size-timing", main=main, roles=COMPUTE)
