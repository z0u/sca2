"""nGPT scaling investigation: why the simplified recipe fails wide-and-deep.

The earlier ``ngpt-sweep`` found that at width 128 the fixed residual step
(``h ← normalize(h + α·sublayer(h))`` with ``α = 1/n_layer``) degrades with
depth — d128|L8 spikes and recovers worse, d128|L12 never trains. The stated
reason ("deeper-and-wider can't absorb the peak LR") is a symptom. The cause is
geometric: the additive step adds the *raw* sublayer output, and the MLP output
norm scales like √n_embd (it up-projects by a √n_embd baseline), so the
effective per-layer rotation is ``α·‖sublayer‖`` — *not* width-independent, and
``1/n_layer`` does not control it. (Init-time diagnostic: total hidden-state
travel runs 246° → 615° from d32|L4 to d128|L12 under the baseline recipe.)

This experiment tests the fixes at the failing corner — width 128 × depths
{4, 8, 12} — across recipe arms:

- ``base``  — the current recipe (control; should reproduce the failure).
- ``lr3e3`` — same recipe at a lower peak LR (was the failure "just" LR?).
- ``sqrt``  — additive step at ``α = 1/√n_layer`` (the user's hypothesis).
- ``norm``  — nGPT-faithful LERP toward the *normalized* sublayer output; this
  makes ``α`` the true interpolation fraction, restoring width/depth-independence.
- ``lrn``   — learnable scalar ``α`` per sublayer (init 1/n_layer).

Everything else matches ``ngpt-sweep`` (batch 16, 100 epochs, Pride and
Prejudice) so the ``base`` arm is directly comparable to the published failure.

    bin/mini run docs/ngpt-scaling/experiment.py --app modal --max-containers 15
    bin/mini status ngpt-scaling
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

# Fixed at the failing width; depth is the diagnostic axis (α is tied to it).
WIDTH = 128
DEPTHS = [4, 8, 12]

# (arm label, peak LR, model-config knobs). Knobs default to the current recipe.
ARMS: list[tuple[str, float, dict]] = [
    ("base", 1e-2, {}),
    ("lr3e3", 3e-3, {}),
    ("sqrt", 1e-2, dict(residual_alpha_exp=0.5)),
    ("norm", 1e-2, dict(normalize_sublayer=True)),
    ("lrn", 1e-2, dict(learnable_alpha=True)),
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
    for arm, lr, knobs in ARMS:
        for n_layer in DEPTHS:
            config = _make_config(WIDTH, n_layer, lr, knobs)
            config.tokenizer = meta.tokenizer_config.model_copy()
            config.model.vocab_size = align(meta.tokenizer_config.vocab_size, 64)
            cells.append((config, f"{arm}|d{WIDTH}|L{n_layer}"))
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
        # Same regime as ngpt-sweep, so the `base` arm reproduces the published
        # failure; L4 is right-sized for these batch-16, ≤2M-param cells.
        "train": dict(gpu="L4", timeout=1500),
    },
)
