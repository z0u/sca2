from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sca.anchoring import (
    ANCHOR_AXES,
    LINE_TOKENS,
    PROMPT_SPAN,
    AnchorSpec,
    AntiSpec,
    Crop,
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
    anti_slices: tuple[int, ...] | None = None,
):
    """One call shape for both steps: (model, opt_state, task, anchor, anti, fallback, anti_anchor, fb_lines).

    Without a fallback spec the last three are zeros, so the loop reads eight outputs either way. The slice selection and the embedding constraint exist on the plain step only; the fallback step has its own reflected pass and has not needed either.
    """
    if fallback is None:
        step = make_anchored_train_step(
            optimizer,
            tau=anchor.tau,
            n_lines=n_lines,
            slices=slices,
            clean_rows=clean_rows,
            axes=anchor.axes,
            hinge=anchor.hinge,
            anti_slices=anti_slices,
        )
        return lambda *args: (*step(*args), 0.0, 0.0, 0.0)
    if slices is not None or clean_rows is not None or anti_slices is not None or tuple(anchor.axes) != ANCHOR_AXES:
        raise ValueError(
            "anchor_slices, anti_slices, clean_rows, and a multi-axis anchor are not supported with a fallback spec"
        )
    step = make_fallback_train_step(optimizer, fallback, tau=anchor.tau, n_lines=n_lines)
    fb_w_, aa_w_ = jnp.asarray(fb_w), jnp.asarray(aa_w)
    return lambda *args: step(*args, fb_w_, aa_w_)


SCAN_STEPS = 16
"""Training steps per dispatch in `train_anchored`. The step is host-bound on an L4 (Python dispatch and CUDA calls dominate a d64 step), so running several per call cuts its cost (about half, more on a slow host) with identical weights; the rationale and numbers are in `todo/eng/training-step-is-host-bound.md`."""


def _scanned(step, n_steps: int):
    """*n_steps* calls of *step* in one dispatch, as a `lax.scan` over stacked batches and weights.

    *step* has `_anchored_step`'s call shape. The result takes the model, the optimizer state, a tuple of batch arrays with a leading step axis, the two weight arrays, and a boolean `live` array; a step whose flag is off passes the state through untouched, so a dispatch cut short by a trajectory record or an epoch end is padded to the one compiled shape rather than compiled again. It returns the model, the state, and the six per-step outputs as an `(n_steps, 6)` array (zeros where not live).
    """

    @eqx.filter_jit
    def run(model, opt_state, batches, weights, anti_weights, live):
        carry, static = eqx.partition((model, opt_state), eqx.is_array)

        def body(carry, xs):
            batch, weight, anti_weight, on = xs

            def go(carry):
                m, s = eqx.combine(carry, static)
                m, s, *outs = step(m, s, *batch[:4], weight, anti_weight, *batch[4:])
                return eqx.filter((m, s), eqx.is_array), jnp.stack([jnp.asarray(o, jnp.float32) for o in outs])

            return jax.lax.cond(on, go, lambda carry: (carry, jnp.zeros(6, jnp.float32)), carry)

        carry, outs = jax.lax.scan(body, carry, (batches, weights, anti_weights, live), length=n_steps)
        return (*eqx.combine(carry, static), outs)

    return run


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
    anti_slices: tuple[int, ...] | None = None,
    clean_rows: tuple[int, ...] | None = None,
    crop: Crop | None = None,
    checkpoint_dir: Path,
    checkpoint_every: int | None = None,
    traj_stride: int = 50,
    n_val_batches: int = 4,
    on_record: Callable[[int, LanguageModel], None] | None = None,
    newline_id: int | None = None,
    min_line_tokens: int = LINE_TOKENS,
    loss_mask: np.ndarray | None = None,
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
        anti_slices: residual-stream slices the anti-subspace term acts on,
            or None for the same set as *anchor_slices*.
        clean_rows: embeddings held off the anchor axis by a hard
            constraint after every step (`sca.anchoring.clean_embedding_rows`),
            or None for no constraint. Neither option combines with a fallback spec.
        crop: which labeled lines the pooled term pulls, given how much of each
            the window shows (`sca.anchoring.Crop`), or None for every line, as
            `all` would. Needs a pooled anchor (`anchor.tau` set) and no fallback.
        checkpoint_dir: Where to write checkpoints; sweep cells sharing a volume
            must each pass their own.
        checkpoint_every: Save a checkpoint every N epochs. None = about 50 in all.
        traj_stride: Steps between alignment measurements.
        n_val_batches: fixed validation crops behind the trajectory's loss curve,
            drawn once so the curve moves with the model rather than the sample.
        on_record: called after each trajectory record with the record's index
            and the model as it stands, so a caller can keep a checkpoint at
            every trajectory point (a training-dynamics read needs the model
            through the plateau, where the end checkpoint says nothing).
        newline_id: passed through to `sample_anchored_batches`: `None` for the
            fixed-period reproduction path, or the newline token id for a
            variable-length-line grammar (the in-context grammar).
        min_line_tokens: passed through to `sample_anchored_batches`, and to
            size `n_lines`; unused when *newline_id* is `None`.
        loss_mask: passed through to `sample_anchored_batches` (a
            verification-line loss mask); `None` for no masking.
    """
    if crop is not None and (anchor.tau is None or fallback is not None):
        raise ValueError("a crop policy needs a pooled anchor (tau set) and no fallback spec")
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
    n_lines = config.model.block_size // min_line_tokens + 2  # a crop straddles at most this many lines
    train_steps = _scanned(
        _anchored_step(
            optimizer,
            anchor,
            n_lines,
            fallback,
            fallback_weight,
            anti_anchor_weight,
            anchor_slices,
            clean_rows,
            anti_slices,
        ),
        SCAN_STEPS,
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
        alpha = alignment(model, probe_tokens, axes=anchor.axes)[:, :, :PROMPT_SPAN]  # (L1, C, span roles)
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
        if on_record is not None:
            on_record(len(traj["step"]) - 1, model)

    expected = dict(loss="down", anchor="down", m_op1="up", m_span="up")
    if probe_line_w is not None:
        expected["m_line"] = "up"
    if fallback is not None and fallback_weight > 0:
        expected["fallback"] = "down"
    expect_metrics(**expected)
    for epoch in range(config.scheduler.epochs):
        train_losses, anchor_losses, anti_losses = [], [], []
        batches = sample_anchored_batches(
            train_data,
            config.data,
            config.model,
            epoch_length,
            rng,
            label_p,
            anchor.span,
            lines=True,
            crop=crop,
            newline_id=newline_id,
            min_line_tokens=min_line_tokens,
            loss_mask=loss_mask,
        )
        while len(train_losses) < epoch_length:
            # A dispatch stops at the next trajectory record and at the epoch's end, which need the model there.
            n = min(SCAN_STEPS, epoch_length - len(train_losses), traj_stride - step % traj_stride)
            chunk = [next(batches) for _ in range(n)]
            ats = [epoch + (len(train_losses) + i) / epoch_length for i in range(n)]
            weights = [float(anchor(at)) for at in ats]
            anti_weights = [float(anti(at)) if anti is not None else 0.0 for at in ats]
            pad = SCAN_STEPS - n
            stacked = tuple(np.stack([b[f] for b in chunk] + [chunk[-1][f]] * pad) for f in range(len(chunk[0])))
            model, opt_state, outs = train_steps(
                model,
                opt_state,
                stacked,
                np.asarray(weights + [0.0] * pad, np.float32),
                np.asarray(anti_weights + [0.0] * pad, np.float32),
                np.arange(SCAN_STEPS) < n,
            )
            loss, anchor_loss, anti_loss, fb_loss, aa_loss, fb_lines = np.asarray(outs[:n]).T.tolist()
            train_losses += loss
            anchor_losses += anchor_loss
            anti_losses += anti_loss
            step += n
            emit_metrics(loss=loss[-1], anchor=anchor_loss[-1], anchor_weight=weights[-1])
            if fallback is not None:
                for i in range(n):
                    window.add(fb_loss[i], aa_loss[i], fb_lines[i])
                emit_metrics(fallback=fb_loss[-1], anti_anchor=aa_loss[-1])
            emit_progress(step, total_steps)
            if step % traj_stride == 0:
                record(step, weights[-1], anti_weights[-1], anchor_loss[-1], anti_loss[-1])
        # Run the sampler to its end, as a for loop would, so any draws after its last batch still happen.
        assert next(batches, None) is None

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
