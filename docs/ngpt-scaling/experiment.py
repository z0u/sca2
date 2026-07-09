"""nGPT scaling: why the simplified residual fails wide-and-deep, and the fix.

We simplified our nGPT residual to a fixed *additive* step,
``h ← normalize(h + α·sublayer(h))`` with ``α = 1/n_layer``, on the assumption
that ``sublayer(h)`` has norm ≈ 1. It doesn't: the MLP up-projects its
pre-activations by a √n_embd baseline, so ``‖MLP(h)‖ ∝ √n_embd``. The *effective*
per-layer rotation is therefore ``α·‖sublayer‖``, which grows with width, so the
fixed step never controlled the geometry it claimed to — the model destabilizes
at width 128, depth 8–12. nGPT proper avoids this by stepping toward the
*normalized* output, ``h ← normalize(h + α·(normalize(sublayer) − h))``, making
``α`` a true interpolation fraction; we had dropped that normalization.

This experiment establishes the failure and the fix over two axes:

- **Recipe** (at the failing width 128 × depths {4, 8, 12}):
  ``base`` (the buggy additive step), ``lr3e3`` (same, lower peak LR),
  ``sqrt`` (additive at ``α = 1/√n_layer``), ``lrn`` (additive, learnable α),
  and ``norm`` (the normalized-LERP fix — now the model default).
- **Width** (``base`` and ``norm`` × widths {32, 64, 128} × depths {4, 8, 12}):
  shows the failure is width-gated (small widths train fine) and the fix is
  width-flat.

Everything else is held fixed (batch 16, 100 epochs, Pride and Prejudice).

    bin/mini run docs/ngpt-scaling/experiment.py --app modal --max-containers 27
    bin/mini status ngpt-scaling
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

DEPTHS = [4, 8, 12]
WIDTHS = [32, 64, 128]

# (arm label, peak LR, model-config knobs, widths). The buggy additive residual
# is now off by default, so the failing arms pin normalize_sublayer=False
# explicitly; `norm` is the corrected (now-default) recipe. `base` and `norm`
# sweep width to expose the width-gating; the rest run only at the failing 128.
ARMS: list[tuple[str, float, dict, list[int]]] = [
    ("base", 1e-2, dict(normalize_sublayer=False), WIDTHS),
    ("lr3e3", 3e-3, dict(normalize_sublayer=False), [128]),
    ("sqrt", 1e-2, dict(normalize_sublayer=False, residual_alpha_exp=0.5), [128]),
    ("lrn", 1e-2, dict(normalize_sublayer=False, learnable_alpha=True), [128]),
    ("norm", 1e-2, dict(normalize_sublayer=True), WIDTHS),
]

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


def _make_config(n_embd: int, n_layer: int, lr: float, knobs: dict, batch_size: int = 16):
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
            **knobs,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=batch_size, oversample=2, train_split=0.8, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=lr, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01),
    )


def build_sweep(meta) -> list[tuple]:
    """Derive the (config, label) cells from prep's tokenizer.

    Runs every wake (cheap + deterministic), so the memo keys are stable: each
    cell re-runs only if its own config changes.
    """
    from experiment.utils import align

    cells = []
    for arm, lr, knobs, widths in ARMS:
        for n_embd in widths:
            for n_layer in DEPTHS:
                config = _make_config(n_embd, n_layer, lr, knobs)
                config.tokenizer = meta.tokenizer_config.model_copy()
                config.model.vocab_size = align(meta.tokenizer_config.vocab_size, 64)
                cells.append((config, f"{arm}|d{n_embd}|L{n_layer}"))
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
