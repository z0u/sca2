"""Training-time concept anchoring in a transformer's residual stream.

M1 anchored *red* to axis 0 of an autoencoder bottleneck (`sca.colorcube`). This module is the same idea moved to a sequence model: a regularizer term pulls the residual-stream states of *labeled lines* toward a fixed direction, and the measurement side reads how far that got.

Three pieces, kept separate so an experiment can use them independently:

- **The pull.** `anchor_term` is the cosine term itself, and `pooled_anchor_term` its mellowmax variant that asks each labeled line to align *somewhere* in its span rather than everywhere; `make_anchored_train_step` folds either into a training step at a scheduled weight, and `anchor_weight` is that schedule (ramp, hold, anneal to a floor above zero — the M1/ex-2.9.3 lesson that protection withdrawn entirely lets the task loss reclaim the axis). `anti_subspace_term` is its repulsive companion from M1: an indiscriminate penalty on the mean-square alignment of *every* state, labeled or not, so the pull has to buy alignment against a headwind.
- **The labels.** `sample_anchored_batches` crops packed token blocks exactly as `sca.data.batches.sample_batches` does, and adds the (B, T) mask of positions the term pulls: the prompt span of lines that drew a label this visit. `LabelSpec` says which operands draw (op1 alone, or either independently) and whether the pull covers the span or just the operand(s) that drew.
- **The readout.** `alignment` reads cos(h, e₀) off the residual stream, and `margin` contracts it to the label-affinity-weighted margin the hypotheses score.

The architecture is what makes the cosine term transfer unmodified: every residual-stream state of our simplified nGPT is already unit-norm, so a direction constraint needs no companion term to keep activation scale in hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PyTree

from sca.config import DataConfig, ModelConfig
from sca.model import LanguageModel
from sca.training.loop import cross_entropy

ANCHOR_AXIS = 0
"""The anchor direction is e₀, the first basis vector of the residual stream.

The residual operations of nGPT (LERP toward normalized sub-module output,
scalar gains) are rotation-equivariant, so a basis vector is a convenience for
reading the cosine off a component rather than a hint to the model.
"""

LINE_TOKENS = 6
"""`name + name = name ⏎` at word level, so the packed corpus is periodic in 6."""

PROMPT_SPAN = 4
"""Roles the anchor pulls: op1, `+`, op2, `=`. The answer and newline are measured, not pulled."""


def smoothstep(tau) -> np.ndarray:
    """Minimum-jerk interpolation from 0 to 1 over the unit interval, clamped outside it.

    The quintic 6τ⁵ − 15τ⁴ + 10τ³, which is what `mini.temporal.MinimumJerkTimingFunction` reduces to for a move that starts and ends at rest — so this is the shape M1's dopesheets used for every regularizer ramp and anneal, in closed form and vectorized.
    """
    t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


@dataclass(frozen=True)
class AnchorSpec:
    """Everything about the pull except which lines it applies to.

    `peak` is the swept weight λ; the rest is the schedule it moves on. The ramp shares the LR warmup window, so the anchor arrives with the optimizer rather than ahead of it, and the anneal stops at `floor × peak` rather than zero. Every condition anneals to `anneal_end`, so moving `anneal_start` stretches the anneal instead of sliding a fixed window.
    """

    peak: float
    warmup_epochs: float
    anneal_start: float
    anneal_end: float
    floor: float = 0.1
    span: int = PROMPT_SPAN
    tau: float | None = None
    """Mellowmax temperature for per-line pooling over the span, or None for the
    flat mean over pulled positions (the ex-2.1.6..8 term). τ = ∞ is the pooled
    code path at the span mean: a per-line mean of means, which differs from the
    flat mean only on lines a crop truncates."""

    def __call__(self, epoch) -> np.ndarray:
        return anchor_weight(
            epoch,
            peak=self.peak,
            warmup_epochs=self.warmup_epochs,
            anneal_start=self.anneal_start,
            anneal_end=self.anneal_end,
            floor=self.floor,
        )


def anchor_weight(
    epoch,
    *,
    peak: float,
    warmup_epochs: float,
    anneal_start: float,
    anneal_end: float,
    floor: float,
) -> np.ndarray:
    """The anchor weight at (fractional) *epoch*: ramp, hold, anneal to a floor."""
    e = np.asarray(epoch, dtype=float)
    ramp = smoothstep(e / warmup_epochs)
    anneal = smoothstep((e - anneal_start) / (anneal_end - anneal_start))
    return peak * ramp * (1.0 - (1.0 - floor) * anneal)


@dataclass(frozen=True)
class AntiSpec:
    """The repulsive companion to the pull, and the schedule it moves on.

    M1 balanced its attractive and repulsive terms in time rather than by a single ratio, so the weight is given relative to the anchor peak `lam`: `peak_ratio` at epoch 0 — full strength before the anchor has ramped in — annealing (minimum-jerk) to `hold_ratio` by `anneal_end` and holding there. From `anchor_anneal_start` both terms share the anchor's end-of-training anneal, so their ratio is constant from the hold point on.
    """

    lam: float
    peak_ratio: float
    hold_ratio: float
    anneal_end: float
    anchor_anneal_start: float
    anchor_anneal_end: float
    floor: float = 0.1

    def __call__(self, epoch) -> np.ndarray:
        return anti_subspace_weight(
            epoch,
            lam=self.lam,
            peak_ratio=self.peak_ratio,
            hold_ratio=self.hold_ratio,
            anneal_end=self.anneal_end,
            anchor_anneal_start=self.anchor_anneal_start,
            anchor_anneal_end=self.anchor_anneal_end,
            floor=self.floor,
        )


def anti_subspace_weight(
    epoch,
    *,
    lam: float,
    peak_ratio: float,
    hold_ratio: float,
    anneal_end: float,
    anchor_anneal_start: float,
    anchor_anneal_end: float,
    floor: float,
) -> np.ndarray:
    """The anti-subspace weight at (fractional) *epoch*: see `AntiSpec`."""
    e = np.asarray(epoch, dtype=float)
    ratio = peak_ratio + (hold_ratio - peak_ratio) * smoothstep(e / anneal_end)
    end = 1.0 - (1.0 - floor) * smoothstep((e - anchor_anneal_start) / (anchor_anneal_end - anchor_anneal_start))
    return lam * ratio * end


@dataclass(frozen=True)
class LabelSpec:
    """Which lines draw a label each visit, and which of their positions the pull covers.

    *p* is a per-token probability. Under `op1` keying it is P(line labeled), read off the first operand alone — the ex-2.1.6..9 labeller, and what a bare array passed in its place means. Under `either` keying it is the per-operand rate: op1 and op2 draw independently and the line is labeled when either does, so the label no longer says which position carried it. *pull* picks the masked positions of a labeled line: its prompt span, or just the operand(s) that drew (`slot`) — the pull of a labeller that names the slot, which the span pull has to match without being told.
    """

    p: Float[np.ndarray, " V"]
    keying: Literal["op1", "either"] = "op1"
    pull: Literal["span", "slot"] = "span"


def anchor_term(states: Float[Array, "L1 B T C"], mask: Float[Array, "B T"]) -> Float[Array, ""]:
    """Mean of (1 − cos(h, e₀)) over pulled positions and residual-stream slices.

    States are unit-norm, so the cosine against a basis vector is just that component. The denominator is the mask's own weight with M1's ε, so a batch holding no labels contributes zero rather than 0/0 — which at this label density is one batch in eight.
    """
    cos = states[..., ANCHOR_AXIS]  # (L1, B, T)
    return jnp.sum((1.0 - cos) * mask) / (states.shape[0] * (jnp.sum(mask) + 1e-8))


def softmin_weights(x: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
    """softmax(−x/τ) along *axis*: the per-position weights of the mellowmax gradient.

    Non-negative and summing to 1, so the pull budget of a line is conserved at every τ; τ = ∞ gives the uniform weights of the span mean. The measurement side of `pooled_anchor_term` — what the pull chose, read off an alignment map.
    """
    if np.isinf(tau):
        return np.full_like(x, 1.0 / x.shape[axis])
    z = -x / tau
    z = z - z.max(axis=axis, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=axis, keepdims=True)


def pooled_anchor_term(
    states: Float[Array, "L1 B T C"],
    mask: Float[Array, "B T"],
    line_id: Int[Array, "B T"],
    n_lines: int,
    tau: float,
) -> Float[Array, ""]:
    """Mean over labeled lines and slices of the mellowmax of (1 − cos) over each line's span.

    The mellowmax −τ·log(mean exp(−x/τ)) (Asadi & Littman, 2017) interpolates from the hard minimum (τ → 0) to the mean (τ = ∞), so the term asks each labeled line to align *somewhere* in its visible span, concentrating the pull wherever alignment is cheapest. Its gradient per line is the softmin weights, non-negative and summing to 1, so the pull budget of each line is conserved and the anchor weight means the same thing at every τ.

    *mask* marks the pulled positions as in `anchor_term`; *line_id* groups them into lines (any per-batch-row local index below *n_lines*). The pool runs within each residual-stream slice, so the pull can choose different positions at different depths. Lines with no visible pulled position contribute nothing, as does an unlabeled batch.
    """
    sel = (line_id[..., None] == jnp.arange(n_lines)) & (mask[..., None] > 0)  # (B, T, N)
    count = sel.sum(axis=1)  # (B, N) pulled positions per line
    labeled = count > 0
    x = 1.0 - states[..., ANCHOR_AXIS]  # (L1, B, T)
    if np.isinf(tau):
        pooled = jnp.einsum("lbt,btn->lbn", x, sel.astype(x.dtype)) / jnp.maximum(count, 1)
    else:
        # The per-line min keeps exp in range (x − min ≥ 0 within the line); its
        # dependence cancels analytically, so it carries no gradient of its own.
        xw = jnp.where(sel[None], x[..., None], jnp.inf)  # (L1, B, T, N)
        lmin = jax.lax.stop_gradient(jnp.where(labeled, xw.min(axis=2), 0.0))
        z = jnp.exp(jnp.where(sel[None], -(x[..., None] - lmin[:, :, None, :]) / tau, -jnp.inf))
        mean_z = z.sum(axis=2) / jnp.maximum(count, 1)
        pooled = lmin - tau * jnp.log(jnp.where(labeled, mean_z, 1.0))
    return jnp.sum(pooled * labeled) / (states.shape[0] * (jnp.sum(labeled) + 1e-8))


def anti_subspace_term(states: Float[Array, "L1 B T C"], live: Float[Array, "B T"]) -> Float[Array, ""]:
    """Mean of cos²(h, e₀) over every residual-stream slice and every live position.

    M1's anti-subspace penalty with the reserved coordinate axis replaced by our anchor direction: it asks that the cloud as a whole not sit on the axis, without asking any particular point to leave it. *live* selects the non-pad positions — the ones the model is actually shown — and every line counts, labeled or not, which is what makes the term indiscriminate.
    """
    cos = states[..., ANCHOR_AXIS]  # (L1, B, T)
    return jnp.sum(cos**2 * live) / (states.shape[0] * (jnp.sum(live) + 1e-8))


def make_anchored_train_step(
    optimizer: optax.GradientTransformation,
    tau: float | None = None,
    n_lines: int = 0,
):
    """Build a jitted training step for cross-entropy plus the two weighted anchor terms.

    The weights are arguments rather than closures, so the schedules move without recompiling; *tau* is fixed per build, since a condition's pooling does not move over training. With `tau=None` the anchor term is the flat per-position mean (`anchor_term`); with a float (∞ allowed) it is the per-line mellowmax (`pooled_anchor_term`), and *n_lines* bounds the local line index the step's `line_id` argument carries. Returns the three loss terms separately: the anchor term is the training-side view of what the alignment measurements read later, and the anti-subspace term is the same view of the mean alignment the containment gates score. Pass `anti_weight=0` for a bare anchor.
    """
    if tau is not None and n_lines < 1:
        raise ValueError(f"pooled anchor (tau={tau}) needs n_lines >= 1, got {n_lines}")

    @eqx.filter_jit
    def train_step(
        model: LanguageModel,
        opt_state: PyTree,
        x: Int[Array, "B T"],
        y: Int[Array, "B T"],
        mask: Float[Array, "B T"],
        line_id: Int[Array, "B T"],
        weight: Float[Array, ""],
        anti_weight: Float[Array, ""],
    ) -> tuple[LanguageModel, PyTree, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        def loss_fn(model: LanguageModel):
            states, logits = model.stream_and_logits(x)
            live = (x != 0).astype(states.dtype)
            task = cross_entropy(logits, y)
            anchor = (
                anchor_term(states, mask) if tau is None else pooled_anchor_term(states, mask, line_id, n_lines, tau)
            )
            anti = anti_subspace_term(states, live)
            return task + weight * anchor + anti_weight * anti, (task, anchor, anti)

        (_, (task, anchor, anti)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model.normalize_weights(), opt_state, task, anchor, anti

    return train_step


def sample_anchored_batches(
    data: Int[np.ndarray, " T"],
    data_config: DataConfig,
    model_config: ModelConfig,
    n_batches: int,
    rng: np.random.Generator,
    label_p: Float[np.ndarray, " V"] | LabelSpec,
    span: int = PROMPT_SPAN,
    lines: bool = False,
) -> Iterator[tuple]:
    """Yield *n_batches* of (inputs, targets, anchor mask).

    The crops are `sca.data.batches.sample_batches`, repeated here rather than wrapped because the mask needs the crop offsets and that generator does not yield them. *label_p* is a `LabelSpec`, or a bare per-token array meaning op1 keying: the probability that a line draws a label, redrawn per visit as in M1 — so the same line is labeled on one epoch and not the next, and the draws are consumed whatever the anchor weight is, which keeps every condition sharing a labeller on identical batches and label draws for a given seed. (Labellers with different keying consume the stream differently, so *those* comparisons carry corpus-draw noise.)

    With `lines=True` each batch carries a fourth element: the (B, T) local line index (0 .. `block_size // LINE_TOKENS + 1`) that groups positions into lines for `pooled_anchor_term`. The draws are identical either way.
    """
    spec = label_p if isinstance(label_p, LabelSpec) else LabelSpec(np.asarray(label_p))
    block_size = model_config.block_size
    n_starts = len(data) - block_size - 1
    if n_starts < 1:
        raise ValueError(f"Corpus of {len(data)} tokens is too short for block size {block_size}")
    offsets = np.arange(block_size)
    n_lines = block_size // LINE_TOKENS + 2  # a crop straddles at most this many lines

    for _ in range(n_batches):
        starts = rng.integers(0, n_starts, size=data_config.batch_size)
        x = np.stack([data[s : s + block_size] for s in starts])
        y = np.stack([data[s + 1 : s + block_size + 1] for s in starts])

        # Randomly pad the beginning of some sequences
        if data_config.padding_chance:
            for i in np.flatnonzero(rng.random(len(starts)) < data_config.padding_chance):
                pad_length = int(rng.integers(1, block_size // 3))
                x[i, :pad_length] = 0
                if pad_length > 1:
                    y[i, : pad_length - 1] = 0

        absolute = starts[:, None] + offsets
        line = absolute // LINE_TOKENS
        local = line - line[:, :1]
        op1 = data[line * LINE_TOKENS]  # every line's first operand, wherever the crop landed
        if spec.keying == "op1":
            draw = rng.random((len(starts), n_lines))
            drew1 = np.take_along_axis(draw, local, axis=1) < spec.p[op1]
            drew2 = np.zeros_like(drew1)
        else:
            op2 = data[line * LINE_TOKENS + 2]
            draw = rng.random((len(starts), n_lines, 2))
            drew1 = np.take_along_axis(draw[..., 0], local, axis=1) < spec.p[op1]
            drew2 = np.take_along_axis(draw[..., 1], local, axis=1) < spec.p[op2]
        role = absolute % LINE_TOKENS
        # Padded positions are not shown to the model, so they are not pulled either.
        if spec.pull == "span":
            mask = (drew1 | drew2) & (role < span) & (x != 0)
        else:
            mask = ((drew1 & (role == 0)) | (drew2 & (role == 2))) & (x != 0)
        if lines:
            yield x, y, mask.astype(np.float32), local.astype(np.int32)
        else:
            yield x, y, mask.astype(np.float32)


@eqx.filter_jit
def _stream_axis(model: LanguageModel, tokens: Int[Array, "N T"]) -> Float[Array, "L1 N T"]:
    """Jitted at module level so the trajectory's repeated calls compile once."""
    return model.residual_stream(tokens)[..., ANCHOR_AXIS]


def alignment(
    model: LanguageModel,
    tokens: Int[np.ndarray, "N T"],
    batch_size: int = 1024,
) -> Float[np.ndarray, "L1 N T"]:
    """cos(h, e₀) at every residual-stream slice and position, for each line of *tokens*.

    nGPT is dropout-free, so there is no inference mode to switch into: the training-time and measurement-time forward passes are the same function.
    """
    chunks = [
        np.asarray(_stream_axis(model, jnp.asarray(tokens[i : i + batch_size])))
        for i in range(0, len(tokens), batch_size)
    ]
    return np.concatenate(chunks, axis=1)


def margin(alpha: Float[np.ndarray, "L1 C T"], weights: Float[np.ndarray, " C"]) -> Float[np.ndarray, "L1 T"]:
    """The alignment margin: label-affinity-weighted mean alignment, minus the plain mean.

    How much closer a color sits to the anchor for being the kind of color the anchor pulls. Zero means the axis is indifferent to redness; *weights* are the per-color label probabilities, normalized to sum to 1.
    """
    return np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)
