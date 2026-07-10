"""nGPT scaling: does the simplified residual hold across width and depth?

Our nGPT strips the published recipe down to scalar gains (in place of the
per-channel eigen learning rates) and a residual step ``α`` *fixed* at
``1/n_layer`` rather than learned. The residual form we keep is the nGPT LERP
toward the sub-module's *normalized* output,
``h ← normalize(h + α·(normalize(sublayer) − h))``: normalizing the target makes
``α`` a true interpolation fraction, so the per-layer rotation stays ~α and the
stack's travel holds O(1) regardless of width.

The milestone leans on this transformer actually scaling — if we want to argue
SCA carries to LLMs, the backbone has to hold up as it grows. So this experiment
sweeps the model over a width × depth grid (widths {32, 64, 128} × depths
{4, 8, 12}) and checks that converged loss stays flat: no depth penalty, no
width-gated instability.

Everything else is held fixed (batch 16, peak LR 10⁻², 100 epochs, Pride and
Prejudice).

    bin/mini run docs/ngpt-scaling/experiment.py --app modal --max-containers 9
    bin/mini status ngpt-scaling
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

DEPTHS = [4, 8, 12]
WIDTHS = [32, 64, 128]
PEAK_LR = 1e-2

CURVES_REF = "reports/ngpt-scaling/curves"


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


def _make_config(n_embd: int, n_layer: int, batch_size: int = 16):
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
        data=DataConfig(batch_size=batch_size, oversample=2, train_split=0.8, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
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
    """Train one sweep cell; return its label and per-epoch val losses."""
    from experiment.compute.training import train_model

    _, metrics = train_model(config, get_data_dir())
    return label, [m.val_loss for m in metrics]


def publish_curves(results: list[tuple]) -> str:
    """Publish the gathered val-loss curves to the project store under ``CURVES_REF``."""
    import json

    from mini.store import put, set_ref

    curves = dict(results)
    set_ref(CURVES_REF, put(json.dumps(curves, indent=2).encode(), name="ngpt-scaling-curves.json"))
    return CURVES_REF


def main(ctx: Ctx) -> list[tuple]:
    meta = ctx.run(prepare_data, role="prep")  # CPU prep; suspends until done
    configs, labels = zip(*build_sweep(meta), strict=True)
    results = ctx.map(train_one, configs, labels, role="train")  # GPU sweep that depends on prep
    ctx.run(publish_curves, results, role="prep")  # share the curves by name for reports
    return results


experiment = Experiment(
    name="ngpt-scaling",
    main=main,
    roles={
        "prep": {},  # CPU-only: data download + tokenize
        # L4 is right-sized for these batch-16, ≤2M-param cells; the largest
        # (d128|L12) takes ~13 min, so the per-task timeout allows generous slack.
        "train": dict(gpu="L4", timeout=1500),
    },
)
