from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sca.anchoring import (
    LINE_TOKENS,
    PROMPT_SPAN,
    AnchorSpec,
    AntiSpec,
    LabelSpec,
    alignment,
    make_anchored_train_step,
    margin,
    sample_anchored_batches,
    softmin_weights,
)
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
    label_p: np.ndarray | LabelSpec,
    probe_tokens: np.ndarray,
    probe_weights: np.ndarray,
    probe_line_w: np.ndarray | None = None,
    checkpoint_dir: Path,
    checkpoint_every: int | None = None,
    traj_stride: int = 50,
    n_val_batches: int = 4,
) -> tuple[LanguageModel, list[TrainingMetrics], dict[str, np.ndarray]]:
    """Train with a concept anchor, recording the alignment trajectory as it goes.

    `train_model` with two additions: the loss carries the scheduled anchor term
    (flat or mellowmax-pooled, per `anchor.tau`) and its optional repulsive
    companion (`sca.anchoring`), and every
    *traj_stride* steps we measure the alignment margin on *probe_tokens*
    — one line per color, weighted by *probe_weights* — at op1 and maxed over
    the prompt span, along with the softmin weight profile the pull would put
    on each span role at the anchor's τ. A flat loss curve does not
    mean learning is done, so that trajectory is the evidence to consult before
    trusting an endpoint.

    Args:
        config: Full training configuration.
        data_dir: Directory to load the tokenized corpus from.
        anchor: Peak weight and schedule for the pull.
        anti: Weight and schedule for the anti-subspace term, or None for a bare
            anchor. The term runs at weight zero when absent, so the loss is the
            same function either way.
        label_p: a `LabelSpec`, or a (vocab,) array meaning op1 keying — the
            probability that a line draws a label, by its first operand's
            token; redrawn per visit either way.
        probe_tokens: (C, T) one line per color, that color as first operand.
        probe_weights: (C,) label-affinity weights, summing to 1.
        probe_line_w: (C,) per-*line* weights, summing to 1, for a labeller
            keyed on more than op1. When given, the trajectory also records
            `m_line`, the per-line margin the wider labeller's retention reads.
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
    n_lines = config.model.block_size // LINE_TOKENS + 2  # a crop straddles at most this many lines
    train_step = make_anchored_train_step(optimizer, tau=anchor.tau, n_lines=n_lines)

    # A fixed validation sample, drawn off its own stream so the training crops
    # stay identical to what an unanchored run of the same seed would see.
    val_rng = np.random.default_rng(config.seed + 10_000)
    val_batches = list(sample_batches(val_data, config.data, config.model, max(n_val_batches, val_length), val_rng))
    total_steps = config.scheduler.epochs * epoch_length
    tokens_per_epoch = epoch_length * config.data.batch_size * config.model.block_size
    all_metrics: list[TrainingMetrics] = []
    keys = ("step", "epoch", "lr", "weight", "anti_weight", "m_op1", "m_span", "alpha_op1")
    traj: dict[str, list] = {k: [] for k in (*keys, "val_loss", "anchor", "anti", "pi", "alpha_roles")}
    tau_eff = np.inf if anchor.tau is None else anchor.tau
    step = 0

    def record(step: int, weight: float, anti_weight: float, anchor_loss: float, anti_loss: float) -> None:
        alpha = alignment(model, probe_tokens)[:, :, :PROMPT_SPAN]  # (L1, C, span roles)
        m = margin(alpha, probe_weights)  # (L1, span roles)
        m_op1 = float(m[:, 0].mean())
        m_span = float(m.max(axis=1).mean())
        # The margin is a contrast, so it is blind to the whole cube drifting onto
        # the axis together. Carrying the plain mean beside it keeps the two apart.
        traj["alpha_op1"].append(float(alpha[:, :, 0].mean()))
        # Unweighted drift per (slice, role): which role pays for its alignment, and when.
        traj["alpha_roles"].append(alpha.mean(axis=1))
        if probe_line_w is not None:
            m_line = float(margin(alpha, probe_line_w).max(axis=1).mean())
            traj.setdefault("m_line", []).append(m_line)
            emit_metrics(m_line=m_line)
        # What the pull chose: the softmin weight each span role would take at the
        # anchor's τ, label-affinity-weighted over colors. Uniform at τ = ∞.
        traj["pi"].append(np.einsum("lct,c->lt", softmin_weights(1.0 - alpha, tau_eff), probe_weights))
        traj["step"].append(step)
        traj["epoch"].append(step / epoch_length)
        traj["lr"].append(float(jnp.asarray(schedule(step))))
        traj["weight"].append(weight)
        traj["anti_weight"].append(anti_weight)
        traj["m_op1"].append(m_op1)
        traj["m_span"].append(m_span)
        traj["val_loss"].append(float(np.mean([float(eval_step(model, x, y)) for x, y in val_batches])))
        traj["anchor"].append(anchor_loss)
        traj["anti"].append(anti_loss)
        emit_metrics(m_op1=m_op1, m_span=m_span)

    expected = dict(loss="down", anchor="down", m_op1="up", m_span="up")
    if probe_line_w is not None:
        expected["m_line"] = "up"
    expect_metrics(**expected)
    for epoch in range(config.scheduler.epochs):
        train_losses, anchor_losses, anti_losses = [], [], []
        for x, y, mask, line_id in sample_anchored_batches(
            train_data, config.data, config.model, epoch_length, rng, label_p, anchor.span, lines=True
        ):
            at = epoch + len(train_losses) / epoch_length
            weight = float(anchor(at))
            anti_weight = float(anti(at)) if anti is not None else 0.0
            model, opt_state, loss, anchor_loss, anti_loss = train_step(
                model, opt_state, x, y, mask, line_id, jnp.asarray(weight), jnp.asarray(anti_weight)
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
