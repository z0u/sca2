"""Experiment 2.1.1: the color-mixing transformer, un-anchored.

D2.1 asks whether SCA works in a transformer at all: anchor *red* in the
residual stream and check that completion accuracy matches an un-anchored
baseline. This experiment builds that baseline — the task, the data pipeline,
and the measurement apparatus the anchored runs (ex-2.1.2+) will be compared
against.

The task is the color-mixing language (`sca.data.colors`): a character-
level LM over lines like ``red + blue = purple`` and ``#e26 + #48a = #958``,
where mixing is the channel-wise round-half-up mean on a 16-level RGB grid —
exact integer ground truth, no perceptual judgement. Concepts are multi-token
by construction (names *and* hex codes denote the same colors, tied together
by alias lines like ``red = #f00``), which is what the anchoring experiments
need: an anchor should capture *red the concept*, not the token ``red``.

Per cell (width × depth × seed), we train our simplified nGPT as a plain char
LM over the corpus, then measure:

- **Completion accuracy** (greedy, exact-match) on four eval sets: named pairs
  seen in training; *held-out* named pairs (never shown as named equations —
  answering requires composing the alias dictionary with mixing); hex pairs
  never seen together; and cross-form pairs never seen together. The unseen
  sets measure whether the model learned the arithmetic rather than the table.
- **Per-layer linear decodability**: ridge probes from the residual stream to
  the operand color (at the operand's last character) and the result color and
  its *redness* (at the pre-answer position), R² on a held-out half.

The sweep answers two questions the anchored experiments depend on: which
backbone size to freeze (the smallest that saturates accuracy), and what the
un-anchored geometry looks like — where color is decodable, and whether the
probe directions agree across seeds (they shouldn't; that variability is the
problem SCA exists to remove).

    bin/mini run docs/ex-2.1.1/experiment.py --app modal --max-containers 9
    bin/mini status ex-2.1.1
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

WIDTHS = [16, 32, 64]
DEPTHS = [2, 4]
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2

CORPUS_SEED = 0
N_EXAMPLES = 40_000
HOLDOUT_FRAC = 0.2  # of distinct closed named pairs
EVAL_K = 256  # per unseen eval set
PROBE_K = 2048

# Store refs the report reads (see report.py).
METRICS_REF = "reports/ex-2.1.1/metrics"
WEIGHTS_REF = "reports/ex-2.1.1/probe-weights"


def prepare_data() -> dict:
    """Generate the corpus, tokenize it onto the volume, and stash the eval/probe sets."""
    import numpy as np

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import colors
    from sca.data.tokenizer import CharTokenizer
    from mini.store import put

    train_pairs, holdout = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
    corpus = colors.sample_corpus(N_EXAMPLES, CORPUS_SEED, train_pairs)
    text = "".join(ex.text for ex in corpus)

    # Fixed a priori vocabulary (not inferred from the sample), so every cell
    # and every future D2.1.x experiment agrees on token ids.
    tokenizer_config = TokenizerConfig(vocabulary=colors.alphabet())
    tokens = np.asarray(CharTokenizer(tokenizer_config).encode([text])[0], dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=len(text),
        sources=[DatasetMetadata(title="color-mixing corpus", fixes=[], total_chars=len(text))],
    )
    save_data(tokens, meta, get_data_dir())

    seen = {p for ex in corpus if (p := ex.pair) is not None}
    evals = {
        "named_seen": colors.as_named(train_pairs, seed=1),
        "named_holdout": colors.as_named(holdout, seed=2),
        "hex_unseen": colors.sample_unseen("hex", EVAL_K, 3, seen),
        "cross_unseen": colors.sample_unseen("cross", EVAL_K, 4, seen),
    }
    # Equations only (no alias lines): the probe positions assume `a + b = `.
    probes = {"probe": colors.sample_corpus(PROBE_K, 5, train_pairs, {"hex": 0.5, "named": 0.25, "cross": 0.25})}
    return {
        "meta": meta,
        "evals": put(colors.dump_example_sets(evals), name="ex-2.1.1-evals.json"),
        "probes": put(colors.dump_example_sets(probes), name="ex-2.1.1-probes.json"),
    }


def _make_config(n_embd: int, n_layer: int, seed: int):
    """One cell's training config; LR and schedule fixed across the sweep."""
    from sca.config import (
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
            block_size=64,
            n_embd=n_embd,
            n_head=8,
            n_head_dim=8,
            n_ff=4 * n_embd,
            n_layer=n_layer,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=64, oversample=16, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01),
        seed=seed,
    )


def build_sweep(meta) -> list[tuple]:
    """Derive (config, label) cells from prep's tokenizer; cheap + deterministic,
    so each cell's memo key is stable and re-runs only if its own config changes.
    """
    from sca.utils import align

    cells = []
    for n_embd in WIDTHS:
        for n_layer in DEPTHS:
            for seed in SEEDS:
                config = _make_config(n_embd, n_layer, seed)
                config.tokenizer = meta.tokenizer_config.model_copy()
                config.model.vocab_size = align(meta.tokenizer_config.vocab_size, 64)
                cells.append((config, f"d{n_embd}-L{n_layer}-s{seed}"))
    return cells


def train_one(config, label: str) -> dict:
    """Train one cell; checkpoint into the durable store for the eval step."""
    from sca.compute.training import train_model
    from mini.store import put

    data_dir = get_data_dir()
    ckpt_dir = data_dir / "cells" / label  # per-cell, so parallel cells don't clobber
    _, metrics = train_model(config, data_dir, checkpoint_dir=ckpt_dir)
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "checkpoint": put(ckpt_dir / "model", name=f"ex-2.1.1-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, probes) -> dict:
    """Completion accuracy on each eval set, plus per-layer probes."""
    import io

    import numpy as np

    from sca.compute.evaluation import completion_accuracy, probe_residual_stream
    from sca.compute.model import load_checkpoint
    from sca.data.colors import load_example_sets
    from sca.data.tokenizer import CharTokenizer
    from mini.store import get, put

    label = trained["label"]
    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = CharTokenizer(config.tokenizer)

    eval_sets = load_example_sets(get(evals, workdir / "evals.json").read_bytes())
    accuracy = {name: completion_accuracy(model, tokenizer, exs) for name, exs in eval_sets.items()}

    probe_set = load_example_sets(get(probes, workdir / "probes.json").read_bytes())["probe"]
    probe = probe_residual_stream(model, tokenizer, probe_set)

    buf = io.BytesIO()
    np.savez_compressed(buf, **probe["weights"])
    return {
        "label": label,
        "val_loss": trained["val_loss"],
        "accuracy": accuracy,
        "probe_r2": probe["r2"],
        "probe_weights": put(buf.getvalue(), name=f"ex-2.1.1-{label}-probes.npz"),
    }


def publish_results(results: list[dict]) -> dict:
    """Publish the metrics (JSON) and the stacked probe weights (npz) for the report."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    metrics = [{k: v for k, v in r.items() if k != "probe_weights"} for r in results]
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.1-metrics.json"))

    arrays = {}
    for r in results:
        path = get(r["probe_weights"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(WEIGHTS_REF, put(buf.getvalue(), name="ex-2.1.1-probe-weights.npz"))
    return {"n_cells": len(results)}


def main(ctx: Ctx) -> dict:
    prep = ctx.run(prepare_data, role="prep")
    configs, labels = zip(*build_sweep(prep["meta"]), strict=True)
    trained = ctx.map(train_one, configs, labels, role="train")
    n = len(trained)
    evaled = ctx.map(eval_one, trained, [prep["evals"]] * n, [prep["probes"]] * n, role="eval")
    summary = ctx.run(publish_results, evaled, role="prep")
    return {
        **summary,
        "worst_hex_unseen": min(r["accuracy"]["hex_unseen"]["accuracy"] for r in evaled),
        "best_hex_unseen": max(r["accuracy"]["hex_unseen"]["accuracy"] for r in evaled),
    }


experiment = Experiment(
    name="ex-2.1.1",
    main=main,
    roles={
        "prep": {},  # CPU-only: corpus generation + tokenize
        # These cells are smaller than ngpt-scaling's (≤1M params, ~2k steps on a
        # ~750k-char corpus), so an L4 clears one in a few minutes.
        "train": dict(gpu="L4", timeout=1500),
        "eval": dict(gpu="L4", timeout=900),
    },
)
