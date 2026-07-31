from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sca.anchoring import AnchorSpec, AntiSpec, alignment, make_anchored_train_step, margin, sample_anchored_batches
from sca.compute.data_pipelines import load_data
from sca.compute.model import save_checkpoint
from sca.config import TrainingConfig
from sca.data.batches import batches_per_epoch, sample_batches, split_data
from sca.model import LanguageModel, build_model
from sca.training.loop import eval_step, make_train_step
from sca.training.metrics import TrainingMetrics
from sca.training.optimizer import configure_optimizer
from sca.training.scheduler import configure_schedule
from mini.progress import emit_metrics, emit_progress, expect_metrics


def train_model(
    config: TrainingConfig,
    data_dir: Path,
    checkpoint_every: int | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[LanguageModel, list[TrainingMetrics]]:
    """Train a model and return it with per-epoch metrics.

    Args:
        config: Full training configuration.
        data_dir: Directory for loading data and saving checkpoints.
        checkpoint_every: Save a checkpoint every N epochs. None = only at the end.
        checkpoint_dir: Where to write checkpoints; defaults to *data_dir*. Sweep
            cells sharing a volume must each pass their own directory, or the
            shared checkpoint is last-writer-wins.
    """
    checkpoint_dir = checkpoint_dir or data_dir
    data, metadata = load_data(data_dir)
    assert metadata.tokenizer_config.vocab_size <= config.model.vocab_size, "Vocab size mismatch"

    model = build_model(config.model, key=jr.key(config.seed))
    rng = np.random.default_rng(config.seed)

    train_data, val_data = split_data(data, config.data.train_split)
    epoch_length = batches_per_epoch(len(train_data), config.data, config.model)
    val_length = batches_per_epoch(len(val_data), config.data, config.model, oversample=1)

    if checkpoint_every is None:
        checkpoint_every = max(1, config.scheduler.epochs // 50)

    schedule = configure_schedule(config.scheduler, config.optimizer.learning_rate, epoch_length)
    optimizer = configure_optimizer(model, config.optimizer, schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optimizer)

    total_steps = config.scheduler.epochs * epoch_length
    tokens_per_epoch = epoch_length * config.data.batch_size * config.model.block_size
    all_metrics: list[TrainingMetrics] = []
    step = 0

    # `loss` would be guessed anyway; saying it keeps the guess from being load-bearing.
    expect_metrics(loss="down")
    for epoch in range(config.scheduler.epochs):
        train_losses = []
        for x, y in sample_batches(train_data, config.data, config.model, epoch_length, rng):
            model, opt_state, loss = train_step(model, opt_state, x, y)
            train_losses.append(float(loss))
            step += 1
            # Through the metrics dict, not the message string: that's where the
            # record keeps it, so status/watch can flag a diverging or flat loss
            # without anyone reading the line.
            emit_metrics(loss=float(loss))
            emit_progress(step, total_steps)

        val_losses = [
            float(eval_step(model, x, y))
            for x, y in sample_batches(val_data, config.data, config.model, val_length, rng)
        ]
        metrics = TrainingMetrics(
            epoch=epoch,
            learning_rate=float(jnp.asarray(schedule(step))),
            val_loss=float(np.mean(val_losses)),
            training_tokens=(epoch + 1) * tokens_per_epoch,
            train_loss=float(np.mean(train_losses)),
        )
        all_metrics.append(metrics)

        if epoch > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(model, config, metrics, checkpoint_dir)

    if all_metrics:
        save_checkpoint(model, config, all_metrics[-1], checkpoint_dir)

    return model, all_metrics


def train_anchored(
    config: TrainingConfig,
    data_dir: Path,
    *,
    anchor: AnchorSpec,
    anti: AntiSpec | None = None,
    label_p: np.ndarray,
    probe_tokens: np.ndarray,
    probe_weights: np.ndarray,
    checkpoint_dir: Path,
    checkpoint_every: int | None = None,
    traj_stride: int = 50,
    n_val_batches: int = 4,
) -> tuple[LanguageModel, list[TrainingMetrics], dict[str, np.ndarray]]:
    """Train with a concept anchor, recording the alignment trajectory as it goes.

    `train_model` with two additions: the loss carries the scheduled anchor term
    and its optional repulsive companion (`sca.anchoring`), and every
    *traj_stride* steps we measure the alignment margin at op1 on *probe_tokens*
    — one line per color, weighted by *probe_weights*. A flat loss curve does not
    mean learning is done, so that trajectory is the evidence to consult before
    trusting an endpoint.

    Args:
        config: Full training configuration.
        data_dir: Directory to load the tokenized corpus from.
        anchor: Peak weight and schedule for the pull.
        anti: Weight and schedule for the anti-subspace term, or None for a bare
            anchor. The term runs at weight zero when absent, so the loss is the
            same function either way.
        label_p: (vocab,) probability that a line draws a label, by its first
            operand's token; redrawn per visit.
        probe_tokens: (C, T) one line per color, that color as first operand.
        probe_weights: (C,) label-affinity weights, summing to 1.
        checkpoint_dir: Where to write checkpoints; sweep cells sharing a volume
            must each pass their own.
        checkpoint_every: Save a checkpoint every N epochs. None = about 50 in all.
        traj_stride: Steps between alignment measurements.
        n_val_batches: fixed validation crops behind the trajectory's loss curve,
            drawn once so the curve moves with the model rather than the sample.
    """
    data, metadata = load_data(data_dir)
    assert metadata.tokenizer_config.vocab_size <= config.model.vocab_size, "Vocab size mismatch"

    model = build_model(config.model, key=jr.key(config.seed))
    rng = np.random.default_rng(config.seed)

    train_data, val_data = split_data(data, config.data.train_split)
    epoch_length = batches_per_epoch(len(train_data), config.data, config.model)
    val_length = batches_per_epoch(len(val_data), config.data, config.model, oversample=1)

    if checkpoint_every is None:
        checkpoint_every = max(1, config.scheduler.epochs // 50)

    schedule = configure_schedule(config.scheduler, config.optimizer.learning_rate, epoch_length)
    optimizer = configure_optimizer(model, config.optimizer, schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_anchored_train_step(optimizer)

    # A fixed validation sample, drawn off its own stream so the training crops
    # stay identical to what an unanchored run of the same seed would see.
    val_rng = np.random.default_rng(config.seed + 10_000)
    val_batches = list(sample_batches(val_data, config.data, config.model, max(n_val_batches, val_length), val_rng))
    total_steps = config.scheduler.epochs * epoch_length
    tokens_per_epoch = epoch_length * config.data.batch_size * config.model.block_size
    all_metrics: list[TrainingMetrics] = []
    keys = ("step", "epoch", "lr", "weight", "anti_weight", "m_op1", "alpha_op1", "val_loss", "anchor", "anti")
    traj: dict[str, list[float]] = {k: [] for k in keys}
    step = 0

    def record(step: int, weight: float, anti_weight: float, anchor_loss: float, anti_loss: float) -> None:
        alpha = alignment(model, probe_tokens)[:, :, :1]  # (L1, C, 1): op1 only
        m_op1 = float(margin(alpha, probe_weights).mean())
        # The margin is a contrast, so it is blind to the whole cube drifting onto
        # the axis together. Carrying the plain mean beside it keeps the two apart.
        traj["alpha_op1"].append(float(alpha.mean()))
        traj["step"].append(step)
        traj["epoch"].append(step / epoch_length)
        traj["lr"].append(float(jnp.asarray(schedule(step))))
        traj["weight"].append(weight)
        traj["anti_weight"].append(anti_weight)
        traj["m_op1"].append(m_op1)
        traj["val_loss"].append(float(np.mean([float(eval_step(model, x, y)) for x, y in val_batches])))
        traj["anchor"].append(anchor_loss)
        traj["anti"].append(anti_loss)
        emit_metrics(m_op1=m_op1)

    expect_metrics(loss="down", anchor="down", m_op1="up")
    for epoch in range(config.scheduler.epochs):
        train_losses, anchor_losses, anti_losses = [], [], []
        for x, y, mask in sample_anchored_batches(
            train_data, config.data, config.model, epoch_length, rng, label_p, anchor.span
        ):
            at = epoch + len(train_losses) / epoch_length
            weight = float(anchor(at))
            anti_weight = float(anti(at)) if anti is not None else 0.0
            model, opt_state, loss, anchor_loss, anti_loss = train_step(
                model, opt_state, x, y, mask, jnp.asarray(weight), jnp.asarray(anti_weight)
            )
            train_losses.append(float(loss))
            anchor_losses.append(float(anchor_loss))
            anti_losses.append(float(anti_loss))
            step += 1
            emit_metrics(loss=float(loss), anchor=float(anchor_loss), anchor_weight=weight)
            emit_progress(step, total_steps)
            if step % traj_stride == 0:
                record(step, weight, anti_weight, float(anchor_loss), float(anti_loss))

        val_losses = [
            float(eval_step(model, x, y))
            for x, y in sample_batches(val_data, config.data, config.model, val_length, rng)
        ]
        all_metrics.append(
            TrainingMetrics(
                epoch=epoch,
                learning_rate=float(jnp.asarray(schedule(step))),
                val_loss=float(np.mean(val_losses)),
                training_tokens=(epoch + 1) * tokens_per_epoch,
                train_loss=float(np.mean(train_losses)),
            )
        )
        if epoch > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(model, config, all_metrics[-1], checkpoint_dir)

    end = config.scheduler.epochs
    record(
        step,
        float(anchor(end)),
        float(anti(end)) if anti is not None else 0.0,
        float(np.mean(anchor_losses)),
        float(np.mean(anti_losses)),
    )
    if all_metrics:
        save_checkpoint(model, config, all_metrics[-1], checkpoint_dir)

    return model, all_metrics, {k: np.asarray(v, dtype=np.float32) for k, v in traj.items()}
