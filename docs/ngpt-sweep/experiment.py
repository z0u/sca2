"""
nGPT hyperparameter sweep, as a memoized experiment.

A small width × depth sweep of our simplified nGPT (3 embedding widths ×
3 depths = 9 training runs) on a plain-text character-level task. The residual
step size is fixed at 1/n_layer rather than learned, so the depth axis doubles
as a check on that choice: if the fixed gate is wrong, it should show up as
depth-dependent degradation. One CPU-ish data-prep step, then a GPU sweep whose
configs depend on prep's tokenizer.

This is the *definition* — an importable ``main(ctx)`` DAG with no compute baked
in. Drive it on Modal L4s from the CLI; the results are published to the durable
store by name.

    # one data-prep run, then nine training runs, fanned out across L4s:
    bin/mini run docs/ngpt-sweep/experiment.py --app modal --max-containers 9
    bin/mini status ngpt-sweep    # no --app needed — the launch backend sticks

The hardware is bound by role (see ``roles`` below): ``prep`` runs CPU-only,
``train`` on L4s — so ``main`` names labels, not GPUs. Re-run to advance/resume —
done cells are memo hits, so a crash heals by re-running and a failed cell is
recovered with ``bin/mini retry ngpt-sweep``. Adding a width or depth below and
re-running launches only the new cells.
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

# Axes of the sweep: residual-stream width and depth. The attention geometry
# (8 heads × 8 dims) stays fixed so each axis varies one thing. n_ff tracks
# width at the usual 4×. The peak LR is constant: nGPT proved insensitive to it
# across 3e-3..4e-2 in the earlier architecture sweep, so it isn't an axis here.
WIDTHS = [32, 64, 128]
DEPTHS = [4, 8, 12]
LR = 1e-2

# Named view of the gathered val-loss curves in the project-scoped store. The
# data lives in the durable store (the HF bucket when configured), not in Git.
CURVES_REF = "reports/ngpt-sweep/curves"


def download_pride_and_prejudice():
    """Download Pride and Prejudice from the Gutenberg HuggingFace dataset."""
    import ftfy
    import pandas as pd

    from experiment.config import DatasetMetadata

    url = "https://huggingface.co/api/datasets/larenwell/book-gutenberg-train/parquet/default/train/0.parquet"
    df = pd.read_parquet(url, columns=["text"])
    text = df.iloc[0]["text"]
    text, explanation = ftfy.fix_and_explain(text)
    return text, DatasetMetadata(
        title="Pride and Prejudice",
        author="Jane Austen",
        url=url,
        fixes=explanation or [],
        total_chars=len(text),
    )


def prepare_data():
    """Download, tokenize, and save training data to the volume; return the corpus metadata."""
    from experiment.compute.data_pipelines import save_data
    from experiment.data.preparation import tokenize_data

    data_dir = get_data_dir()
    data, metadata = tokenize_data([download_pride_and_prejudice()])
    save_data(data, metadata, data_dir)
    return metadata


def _make_config(n_embd: int, n_layer: int):
    """Build one training config (vocab/tokenizer filled in after prep)."""
    from experiment.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        model=ModelConfig(
            vocab_size=64,  # updated after data prep
            block_size=512,
            n_embd=n_embd,
            n_head=8,
            n_head_dim=8,
            n_ff=4 * n_embd,
            n_layer=n_layer,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=16, oversample=2, train_split=0.8, padding_chance=0.1),
        optimizer=OptimizerConfig(
            weight_decay=0,  # weight norms are pinned to 1; nothing to decay
            learning_rate=LR,
            betas=(0.9, 0.95),
        ),
        scheduler=SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01),
    )


def build_sweep(meta) -> list[tuple]:
    """Derive the (config, label) cells from prep's tokenizer.

    Runs every wake (cheap + deterministic), so the memo keys are stable: each
    cell re-runs only if its own config changes.
    """
    from experiment.utils import align

    cells = []
    for n_embd in WIDTHS:
        for n_layer in DEPTHS:
            config = _make_config(n_embd, n_layer)
            config.tokenizer = meta.tokenizer_config.model_copy()
            config.model.vocab_size = align(meta.tokenizer_config.vocab_size, 64)
            cells.append((config, f"d{n_embd}|L{n_layer}"))
    return cells


def train_one(config, label: str) -> tuple:
    """Train one sweep cell; return its width×depth label and per-epoch val losses."""
    from experiment.compute.training import train_model

    _, metrics = train_model(config, get_data_dir())
    return label, [m.val_loss for m in metrics]


def publish_curves(results: list[tuple]) -> str:
    """Publish the gathered val-loss curves to the project store under ``CURVES_REF``.

    A step, so the worker binds the ambient store and bare ``put`` / ``set_ref``
    resolve against it (the HF bucket when configured — the token rides in on the
    worker's Secret). Downstream reports then read the curves by name, so the data
    lives in the durable store rather than a ``results.json`` in Git. Idempotent:
    ``put`` is content-addressed, and ``set_ref`` is fenced on the attempt
    generation — only the current attempt can move the name; a stale relaunch
    fails loudly instead.
    """
    import json

    from mini.store import put, set_ref

    curves = dict(results)
    set_ref(CURVES_REF, put(json.dumps(curves, indent=2).encode(), name="ngpt-sweep-curves.json"))
    return CURVES_REF


def main(ctx: Ctx) -> list[tuple]:
    meta = ctx.run(prepare_data, role="prep")  # CPU prep; suspends until done
    configs, labels = zip(*build_sweep(meta), strict=True)
    results = ctx.map(train_one, configs, labels, role="train")  # GPU sweep that depends on prep
    ctx.run(publish_curves, results, role="prep")  # share the curves by name for reports
    return results


experiment = Experiment(
    name="ngpt-sweep",
    main=main,
    roles={
        "prep": {},  # CPU-only: data download + tokenize
        # 25 min: the largest cell (d128|L12) takes ~13 min on an L4; a 12 min
        # timeout killed it at 94% while the smaller cells finished comfortably.
        "train": dict(gpu="L4", timeout=1500),
    },
)
