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
from sca.fallback import FallbackSpec, make_fallback_train_step
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


def _anchored_step(
    optimizer,
    anchor: AnchorSpec,
    n_lines: int,
    fallback: FallbackSpec | None,
    fb_w: float,
    aa_w: float,
    slices: tuple[int, ...] | None = None,
    clean_rows: tuple[int, ...] | None = None,
):
    """One call shape for both steps: (model, opt_state, task, anchor, anti, fallback, anti_anchor, fb_lines).

    Without a fallback spec the last three are zeros, so the loop reads eight outputs either way. The slice selection and the row constraint exist on the plain step only; the fallback step has its own reflected pass and has not needed either.
    """
    if fallback is None:
        step = make_anchored_train_step(
            optimizer, tau=anchor.tau, n_lines=n_lines, slices=slices, clean_rows=clean_rows
        )
        return lambda *args: (*step(*args), 0.0, 0.0, 0.0)
    if slices is not None or clean_rows is not None:
        raise ValueError("anchor_slices and clean_rows are not supported together with a fallback spec")
    step = make_fallback_train_step(optimizer, fallback, tau=anchor.tau, n_lines=n_lines)
    fb_w_, aa_w_ = jnp.asarray(fb_w), jnp.asarray(aa_w)
    return lambda *args: step(*args, fb_w_, aa_w_)


class _Window:
    """Running means of the fallback step's extra outputs between trajectory records."""

    KEYS = ("fallback", "anti_anchor", "fb_lines")

    def __init__(self):
        self.values: dict[str, list[float]] = {k: [] for k in self.KEYS}
        self.last: dict[str, float] = dict.fromkeys(self.KEYS, float("nan"))

    def add(self, fb_loss: float, aa_loss: float, fb_lines: float) -> None:
        # A crop with no qualifying line reports a zero loss; that is absence, not a value.
        if fb_lines > 0:
            self.values["fallback"].append(fb_loss)
        self.values["anti_anchor"].append(aa_loss)
        self.values["fb_lines"].append(fb_lines)

    def flush(self) -> dict[str, float]:
        """Means since the last flush; a key with nothing in its window repeats its last value."""
        self.last = {k: float(np.mean(v)) if v else self.last[k] for k, v in self.values.items()}
        self.values = {k: [] for k in self.KEYS}
        return dict(self.last)


def train_anchored(  # noqa: C901 — one loop with two optional terms; the branches are the options
    config: TrainingConfig,
    data_dir: Path,
    *,
    anchor: AnchorSpec,
    anti: AntiSpec | None = None,
    label_p: np.ndarray | LabelSpec,
    probe_tokens: np.ndarray,
    probe_weights: np.ndarray,
    probe_line_w: np.ndarray | None = None,
    fallback: FallbackSpec | None = None,
    fallback_weight: float = 0.0,
    anti_anchor_weight: float = 0.0,
    anchor_slices: tuple[int, ...] | None = None,
    clean_rows: tuple[int, ...] | None = None,
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
        fallback: the fallback control's tables and thresholds (`sca.fallback`),
            or None for the anchored step alone. With a spec the step runs the
            reflected pass on every batch, whatever the weights, and the
            trajectory adds `fallback` and `anti_anchor` (the two losses,
            averaged over the steps since the last record) and `fb_lines` (the
            mean number of qualifying lines per crop over the same window).
        fallback_weight: constant weight of the fallback cross-entropy.
        anti_anchor_weight: constant weight of the anti-anchor hinge.
        anchor_slices: residual-stream slices the anchor and anti-subspace terms
            act on, or None for every slice. `(1, ..., n_layer)` anchors the
            blocks' outputs and leaves the embedding table to the task alone.
        clean_rows: embedding rows held off the anchor axis by a hard
            constraint after every step (`sca.anchoring.clean_embedding_rows`),
            or None for no constraint. Neither option combines with a fallback spec.
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
    train_step = _anchored_step(
        optimizer, anchor, n_lines, fallback, fallback_weight, anti_anchor_weight, anchor_slices, clean_rows
    )

    # A fixed validation sample, drawn off its own stream so the training crops
    # stay identical to what an unanchored run of the same seed would see.
    val_rng = np.random.default_rng(config.seed + 10_000)
    val_batches = list(sample_batches(val_data, config.data, config.model, max(n_val_batches, val_length), val_rng))
    total_steps = config.scheduler.epochs * epoch_length
    tokens_per_epoch = epoch_length * config.data.batch_size * config.model.block_size
    all_metrics: list[TrainingMetrics] = []
    keys = ("step", "epoch", "lr", "weight", "anti_weight", "m_op1", "m_span", "alpha_op1")
    traj: dict[str, list] = {k: [] for k in (*keys, "val_loss", "anchor", "anti", "pi", "alpha_roles")}
    if fallback is not None:
        traj |= {k: [] for k in _Window.KEYS}
    tau_eff = np.inf if anchor.tau is None else anchor.tau
    step = 0
    window = _Window()

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
        if fallback is not None:
            for k, v in window.flush().items():
                traj[k].append(v)
        emit_metrics(m_op1=m_op1, m_span=m_span)

    expected = dict(loss="down", anchor="down", m_op1="up", m_span="up")
    if probe_line_w is not None:
        expected["m_line"] = "up"
    if fallback is not None and fallback_weight > 0:
        expected["fallback"] = "down"
    expect_metrics(**expected)
    for epoch in range(config.scheduler.epochs):
        train_losses, anchor_losses, anti_losses = [], [], []
        for x, y, mask, line_id in sample_anchored_batches(
            train_data, config.data, config.model, epoch_length, rng, label_p, anchor.span, lines=True
        ):
            at = epoch + len(train_losses) / epoch_length
            weight = float(anchor(at))
            anti_weight = float(anti(at)) if anti is not None else 0.0
            model, opt_state, loss, anchor_loss, anti_loss, fb_loss, aa_loss, fb_lines = train_step(
                model, opt_state, x, y, mask, line_id, jnp.asarray(weight), jnp.asarray(anti_weight)
            )
            train_losses.append(float(loss))
            anchor_losses.append(float(anchor_loss))
            anti_losses.append(float(anti_loss))
            step += 1
            emit_metrics(loss=float(loss), anchor=float(anchor_loss), anchor_weight=weight)
            if fallback is not None:
                window.add(float(fb_loss), float(aa_loss), float(fb_lines))
                emit_metrics(fallback=float(fb_loss), anti_anchor=float(aa_loss))
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
