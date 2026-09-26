"""Training-time concept anchoring in a transformer's residual stream.

M1 anchored *red* to axis 0 of an autoencoder bottleneck (`sca.colorcube`). This module is the same idea moved to a sequence model: a regularizer term pulls the residual-stream states of *labeled lines* toward a fixed direction, and the measurement side reads how far that got.

Three pieces, kept separate so an experiment can use them independently:

- **The pull.** `anchor_term` is the cosine term itself, and `pooled_anchor_term` its mellowmax variant that asks each labeled line to align *somewhere* in its span rather than everywhere; `make_anchored_train_step` folds either into a training step at a scheduled weight, and `anchor_weight` is that schedule (ramp, hold, anneal to a floor above zero — the M1/ex-2.9.3 lesson that protection withdrawn entirely lets the task loss reclaim the axis). `anti_subspace_term` is its repulsive companion from M1: an indiscriminate penalty on the mean-square alignment of *every* state, labeled or not, so the pull has to buy alignment against a headwind.
- **The labels.** `sample_anchored_batches` crops packed token blocks exactly as `sca.data.batches.sample_batches` does, and adds the (B, T) mask of positions the term pulls: the prompt span of lines that drew a label this visit. `LabelSpec` says what draws (op1 alone, either operand, the answer too, or the op word) and whether the pull covers the span or just the operand(s) that drew.
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
from sca.model.ngpt import NGPT
from sca.training.loop import cross_entropy

ANCHOR_AXIS = 0
"""The anchor direction is e₁, the first basis vector of the residual stream.

The residual operations of nGPT (LERP toward normalized sub-module output,
scalar gains) are rotation-equivariant, so a basis vector is a convenience for
reading the cosine off a component rather than a hint to the model.
"""

ANCHOR_AXES: tuple[int, ...] = (ANCHOR_AXIS,)
"""The default home of the concept: the one axis. A tuple of several axes anchors to their span (a plane for
two): every term and readout below takes `axes=` and reads the alignment as `axes_alignment`."""


def axes_alignment(states: Float[Array, "... C"] | np.ndarray, axes: tuple[int, ...] = ANCHOR_AXES):
    """The alignment of unit-norm *states* with the span of *axes*: the component itself for one axis (signed,
    cos(h, e)), and the length of the projection onto the pair for several (unsigned). Either is the cosine
    between the state and its nearest point in the subspace, which is what the anchor term pulls toward one, the
    anti-subspace term squares, and the trajectory reads.
    """
    if len(axes) == 1:
        return states[..., axes[0]]
    sel = states[..., jnp.asarray(axes)] if isinstance(states, jax.Array) else states[..., list(axes)]
    return (sel**2).sum(axis=-1) ** 0.5


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


Shape = Literal["min-jerk", "linear", "flat"]
"""How a weight schedule moves between its keyframes.

`min-jerk` is `smoothstep`, what M1's dopesheets used and what every experiment
before ex-2.1.11 inherited. `linear` runs a straight line through the same
keyframes, so the weight is continuous but its derivative jumps at each one.
`flat` is not an interpolation at all: it discards the keyframes and holds one
constant weight for the whole of training — the ablation arm that asks whether
the schedule earns its place.
"""


def _interp(t, shape: Shape) -> np.ndarray:
    """Interpolate 0 → 1 over the unit interval, clamped outside it."""
    if shape == "linear":
        return np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return smoothstep(t)


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
    shape: Shape = "min-jerk"
    """Under `flat` the weight sits at `peak` throughout and the other keyframes
    are unused, so one field ablates the ramp, the anneal and the floor together."""
    axes: tuple[int, ...] = ANCHOR_AXES
    """Where the concept is pulled to: one axis, or the span of several (`axes_alignment`). The
    anti-subspace term and the trajectory's alignment reads take the same axes."""

    def __call__(self, epoch) -> np.ndarray:
        return anchor_weight(
            epoch,
            peak=self.peak,
            warmup_epochs=self.warmup_epochs,
            anneal_start=self.anneal_start,
            anneal_end=self.anneal_end,
            floor=self.floor,
            shape=self.shape,
        )


def anchor_weight(
    epoch,
    *,
    peak: float,
    warmup_epochs: float,
    anneal_start: float,
    anneal_end: float,
    floor: float,
    shape: Shape = "min-jerk",
) -> np.ndarray:
    """The anchor weight at (fractional) *epoch*: ramp, hold, anneal to a floor.

    Under `shape="flat"` there is nothing to interpolate: the weight is `peak`
    from step 0 to the end of training.
    """
    e = np.asarray(epoch, dtype=float)
    if shape == "flat":
        return peak * np.ones_like(e)
    ramp = _interp(e / warmup_epochs, shape)
    anneal = _interp((e - anneal_start) / (anneal_end - anneal_start), shape)
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
    shape: Shape = "min-jerk"
    """Under `flat` the ratio sits at `hold_ratio` throughout and the term skips
    the anchor's end anneal, so `peak_ratio` and `anneal_end` are unused."""

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
            shape=self.shape,
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
    shape: Shape = "min-jerk",
) -> np.ndarray:
    """The anti-subspace weight at (fractional) *epoch*: see `AntiSpec`.

    Under `shape="flat"` the weight is `lam × hold_ratio` for the whole of
    training: a constant ratio to the anchor peak, which is the bracket arm.
    """
    e = np.asarray(epoch, dtype=float)
    if shape == "flat":
        return lam * hold_ratio * np.ones_like(e)
    ratio = peak_ratio + (hold_ratio - peak_ratio) * _interp(e / anneal_end, shape)
    end = 1.0 - (1.0 - floor) * _interp((e - anchor_anneal_start) / (anchor_anneal_end - anchor_anneal_start), shape)
    return lam * ratio * end


@dataclass(frozen=True)
class LabelSpec:
    """Which lines draw a label each visit, and which of their positions the pull covers.

    *p* is a per-token probability. Under `op1` keying it is P(line labeled), read off the first operand alone — the ex-2.1.6..9 labeller, and what a bare array passed in its place means. Under `either` keying it is the per-operand rate: op1 and op2 draw independently and the line is labeled when either does, so the label no longer says which position carried it. Under `line` keying the answer draws too, at the same per-token rate as the operands, so a line is labeled when any of its three colors draws — the labeller a whole-line pull needs, since a pull that covers the answer with no way for the answer to earn the label would still read the label off the operands. *pull* picks the masked positions of a labeled line: the first *span* roles (the prompt span by default, or the whole line at `span=LINE_TOKENS`), or just the slot(s) that drew (`slot`) — the pull of a labeller that names the slot, which the span pull has to match without being told.

    Under `op` keying the colors play no part: *p* is a per-op rate table, indexed by the op word at role 1, and each line draws once against its op's rate. It labels an operation rather than a color, so `span` covers the first *span* roles as for the others (the whole line at `span=LINE_TOKENS`) and `slot` marks the op word alone.
    """

    p: Float[np.ndarray, " V"]
    keying: Literal["op1", "either", "line", "op"] = "op1"
    pull: Literal["span", "slot"] = "span"


Crop = Literal["all", "whole", "half", "scaled", "cut-only", "knowable"]
"""Which labeled lines the pooled term pulls, given how much of each a training window shows. A window is a
random crop of the packed corpus, so the lines at its edges are usually cut short, and under `all` a cut line
gets the pull of a whole one on whatever part is visible. The others weight each line's pooled term by its
visible run: only whole lines (`whole`), only lines with more than half their tokens in view (`half`), every
line by the share in view (`scaled`), or only the cut lines (`cut-only`, the complement of `whole`).
`knowable` keeps the lines whose op word is in view and narrows each one's pool to the positions at or after
it (`KNOWABLE_ROLE`), so no position is asked for the op before its evidence has appeared; the sampler yields
that narrowing as a pool mask beside the weights."""

KNOWABLE_ROLE = 1
"""The op word's role: under `knowable` a line's pool starts here."""


def crop_weight(crop: Crop, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """The weight *crop* puts on a line visit that shows roles *first* to *last* (inclusive, elementwise)."""
    n = last - first + 1
    match crop:
        case "all":
            return np.ones(n.shape, np.float32)
        case "whole":
            return (n == LINE_TOKENS).astype(np.float32)
        case "half":
            return (n > LINE_TOKENS / 2).astype(np.float32)
        case "scaled":
            return (n / LINE_TOKENS).astype(np.float32)
        case "cut-only":
            return (n < LINE_TOKENS).astype(np.float32)
        case "knowable":
            return ((first <= KNOWABLE_ROLE) & (last >= KNOWABLE_ROLE)).astype(np.float32)
    raise ValueError(f"unknown crop policy {crop!r}")


def anchor_term(
    states: Float[Array, "L1 B T C"], mask: Float[Array, "B T"], axes: tuple[int, ...] = ANCHOR_AXES
) -> Float[Array, ""]:
    """Mean of (1 − cos(h, e₁)) over pulled positions and residual-stream slices.

    States are unit-norm, so the cosine against a basis vector is just that component (or, for several *axes*, the length of the projection onto their span). The denominator is the mask's own weight with M1's ε, so a batch holding no labels contributes zero rather than 0/0 — which at this label density is one batch in eight.
    """
    cos = axes_alignment(states, axes)  # (L1, B, T)
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
    axes: tuple[int, ...] = ANCHOR_AXES,
    line_w: Float[Array, "B N"] | None = None,
    pool: Float[Array, "B T"] | None = None,
) -> Float[Array, ""]:
    """Mean over labeled lines and slices of the mellowmax of (1 − cos) over each line's span.

    The mellowmax −τ·log(mean exp(−x/τ)) (Asadi & Littman, 2017) interpolates from the hard minimum (τ → 0) to the mean (τ = ∞), so the term asks each labeled line to align *somewhere* in its visible span, concentrating the pull wherever alignment is cheapest. Its gradient per line is the softmin weights, non-negative and summing to 1, so the pull budget of each line is conserved and the anchor weight means the same thing at every τ.

    *mask* marks the pulled positions as in `anchor_term`; *line_id* groups them into lines (any per-batch-row local index below *n_lines*). The pool runs within each residual-stream slice, so the pull can choose different positions at different depths. Lines with no visible pulled position contribute nothing, as does an unlabeled batch.

    *line_w* weights each line's pooled term (a `Crop` policy's weights). It leaves the denominator alone, so it only ever takes pull away: a line at weight zero still counts as labeled, and a line at weight one keeps the pull it would have had without the weights.

    *pool* narrows the positions each line pools over (`knowable`'s positions from the op word on) without narrowing the denominator: a line is counted as labeled by *mask*, and a labeled line with nothing left in its pool contributes zero.
    """
    member = (line_id[..., None] == jnp.arange(n_lines)) & (mask[..., None] > 0)  # (B, T, N)
    n_labeled = jnp.sum(member.sum(axis=1) > 0)
    sel = member if pool is None else member & (pool[..., None] > 0)
    count = sel.sum(axis=1)  # (B, N) pooled positions per line
    labeled = count > 0
    x = 1.0 - axes_alignment(states, axes)  # (L1, B, T)
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
    kept = labeled if line_w is None else labeled * line_w
    return jnp.sum(pooled * kept) / (states.shape[0] * (n_labeled + 1e-8))


def anti_subspace_term(
    states: Float[Array, "L1 B T C"], live: Float[Array, "B T"], axes: tuple[int, ...] = ANCHOR_AXES
) -> Float[Array, ""]:
    """Mean of cos²(h, e₁) over every residual-stream slice and every live position.

    M1's anti-subspace penalty with the reserved coordinate axis replaced by our anchor direction (or, for several *axes*, the squared length of the projection onto their span): it asks that the cloud as a whole not sit on the axis, without asking any particular point to leave it. *live* selects the non-pad positions — the ones the model is actually shown — and every line counts, labeled or not, which is what makes the term indiscriminate.
    """
    cos = axes_alignment(states, axes)  # (L1, B, T)
    return jnp.sum(cos**2 * live) / (states.shape[0] * (jnp.sum(live) + 1e-8))


def make_anchored_train_step(
    optimizer: optax.GradientTransformation,
    tau: float | None = None,
    n_lines: int = 0,
    slices: tuple[int, ...] | None = None,
    clean_rows: tuple[int, ...] | None = None,
    axes: tuple[int, ...] = ANCHOR_AXES,
):
    """Build a jitted training step for cross-entropy plus the two weighted anchor terms.

    The weights are arguments rather than closures, so the schedules move without recompiling; *tau* is fixed per build, since a condition's pooling does not move over training. With `tau=None` the anchor term is the flat per-position mean (`anchor_term`); with a float (∞ allowed) it is the per-line mellowmax (`pooled_anchor_term`), and *n_lines* bounds the local line index the step's `line_id` argument carries. Returns the three loss terms separately: the anchor term is the training-side view of what the alignment measurements read later, and the anti-subspace term is the same view of the mean alignment the containment gates score. Pass `anti_weight=0` for a bare anchor.

    *slices* restricts both terms to the named residual-stream slices (slice 0 is the embedding); `None` is every slice, the term as ex-2.1 and ex-2.2 trained it. *axes* is where both terms read the alignment (`axes_alignment`): one axis, or the span of several. *clean_rows* names embeddings that may not carry the anchor axis: after each optimizer step and nGPT's re-normalization, the axis component of those embeddings is zeroed and they are re-normalized, the same kind of hard constraint as the unit norm. It is the tied-table fix for the syntax-embedding leak: those embeddings stay shared between the embedding table and the readout table, and training finds whatever solution it can with them held off the axis.
    """
    if tau is not None and n_lines < 1:
        raise ValueError(f"pooled anchor (tau={tau}) needs n_lines >= 1, got {n_lines}")
    if slices is not None and len(slices) == 0:
        raise ValueError("slices must name at least one residual-stream slice, or be None for all")
    sel = None if slices is None else jnp.asarray(sorted(set(slices)))
    rows = None if clean_rows is None else jnp.asarray(sorted(set(clean_rows)))

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
        line_w: Float[Array, "B N"] | None = None,
        pool: Float[Array, "B T"] | None = None,
    ) -> tuple[LanguageModel, PyTree, Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        def loss_fn(model: LanguageModel):
            states, logits = model.stream_and_logits(x)
            live = (x != 0).astype(states.dtype)
            task = cross_entropy(logits, y)
            if sel is not None:
                states = states[sel]
            anchor = (
                anchor_term(states, mask, axes)
                if tau is None
                else pooled_anchor_term(states, mask, line_id, n_lines, tau, axes, line_w, pool)
            )
            anti = anti_subspace_term(states, live, axes)
            return task + weight * anchor + anti_weight * anti, (task, anchor, anti)

        (_, (task, anchor, anti)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        model = model.normalize_weights()
        if rows is not None:
            model = clean_embedding_rows(model, rows)
        return model, opt_state, task, anchor, anti

    return train_step


def clean_embedding_rows(model: NGPT, rows: Int[Array, " R"]) -> NGPT:
    """Zero the anchor-axis component of the named embeddings and put them back on the sphere.

    Applied after `normalize_weights`, so the embeddings leave at unit length with no component on `ANCHOR_AXIS`. A tied readout reads through the same embeddings, so the constraint holds on both sides of the table.
    """
    wte = model.transformer.wte
    cleaned = wte[rows].at[:, ANCHOR_AXIS].set(0.0)
    cleaned = cleaned / jnp.maximum(jnp.linalg.norm(cleaned, axis=1, keepdims=True), 1e-12)
    return eqx.tree_at(lambda m: m.transformer.wte, model, wte.at[rows].set(cleaned))


def sample_anchored_batches(
    data: Int[np.ndarray, " T"],
    data_config: DataConfig,
    model_config: ModelConfig,
    n_batches: int,
    rng: np.random.Generator,
    label_p: Float[np.ndarray, " V"] | LabelSpec,
    span: int = PROMPT_SPAN,
    lines: bool = False,
    crop: Crop | None = None,
) -> Iterator[tuple]:
    """Yield *n_batches* of (inputs, targets, anchor mask).

    The crops are `sca.data.batches.sample_batches`, repeated here rather than wrapped because the mask needs the crop offsets and that generator does not yield them. *label_p* is a `LabelSpec`, or a bare per-token array meaning op1 keying: the probability that a line draws a label, redrawn per visit as in M1 — so the same line is labeled on one epoch and not the next, and the draws are consumed whatever the anchor weight is, which keeps every condition sharing a labeller on identical batches and label draws for a given seed. (Labellers with different keying consume the stream differently, so *those* comparisons carry corpus-draw noise.)

    With `lines=True` each batch carries a fourth element: the (B, T) local line index (0 .. `block_size // LINE_TOKENS + 1`) that groups positions into lines for `pooled_anchor_term`. The draws are identical either way.

    With a *crop* policy (and `lines=True`) each batch carries a fifth element: the (B, N) weight of each local line under that policy (`crop_weight`), read off the roles the window shows of it after padding. Under `knowable` a sixth follows, the (B, T) pool mask (see `pooled_anchor_term`). Neither consumes randomness, so the batches and labels are those of `crop=None` at the same seed.
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
        role = absolute % LINE_TOKENS
        draw_mask = _op_mask if spec.keying == "op" else _color_mask
        # Padded positions are not shown to the model, so they are not pulled either.
        mask = draw_mask(data, spec, rng, line, local, role, span, n_lines) & (x != 0)
        if lines and crop == "knowable":
            pool = (role >= KNOWABLE_ROLE).astype(np.float32)
            yield (
                x,
                y,
                mask.astype(np.float32),
                local.astype(np.int32),
                _line_weights(crop, x, local, role, n_lines),
                pool,
            )
        elif lines and crop is not None:
            yield x, y, mask.astype(np.float32), local.astype(np.int32), _line_weights(crop, x, local, role, n_lines)
        elif lines:
            yield x, y, mask.astype(np.float32), local.astype(np.int32)
        else:
            yield x, y, mask.astype(np.float32)


def _line_weights(crop: Crop, x, local, role, n_lines: int) -> np.ndarray:
    """The (B, N) weight of each local line under *crop*, from the first and last role in view of it. A line
    with nothing in view gets whatever the policy gives an empty run; it has no pulled position, so the pooled
    term never reads it.
    """
    rows = np.broadcast_to(np.arange(x.shape[0])[:, None], x.shape)
    seen = x != 0
    first = np.full((x.shape[0], n_lines), LINE_TOKENS, np.int64)
    last = np.full((x.shape[0], n_lines), -1, np.int64)
    np.minimum.at(first, (rows[seen], local[seen]), role[seen])
    np.maximum.at(last, (rows[seen], local[seen]), role[seen])
    return crop_weight(crop, first, last)


def _color_mask(
    data, spec: LabelSpec, rng: np.random.Generator, line, local, role, span: int, n_lines: int
) -> np.ndarray:
    """The pulled positions under the color keyings (`op1`, `either`, `line`), padding aside. *n_lines* sizes
    the draw, so it fixes how much of the stream a batch consumes.
    """
    n_rows = line.shape[0]
    op1 = data[line * LINE_TOKENS]  # every line's first operand, wherever the crop landed
    if spec.keying == "op1":
        draw = rng.random((n_rows, n_lines))
        drew1 = np.take_along_axis(draw, local, axis=1) < spec.p[op1]
        drew2 = drew3 = np.zeros_like(drew1)
    else:
        op2 = data[line * LINE_TOKENS + 2]
        draw = rng.random((n_rows, n_lines, 2))
        drew1 = np.take_along_axis(draw[..., 0], local, axis=1) < spec.p[op1]
        drew2 = np.take_along_axis(draw[..., 1], local, axis=1) < spec.p[op2]
        drew3 = np.zeros_like(drew1)
    if spec.keying == "line":
        # A separate draw after the operands', so `either` keying consumes the stream as it always has.
        at = line * LINE_TOKENS + 4  # the answer may sit past the end of the corpus for the last crop
        answer = data[np.minimum(at, len(data) - 1)]
        draw3 = rng.random((n_rows, n_lines))
        drew3 = (np.take_along_axis(draw3, local, axis=1) < spec.p[answer]) & (at < len(data))
    if spec.pull == "span":
        return (drew1 | drew2 | drew3) & (role < span)
    return (drew1 & (role == 0)) | (drew2 & (role == 2)) | (drew3 & (role == 4))


def _op_mask(data, spec: LabelSpec, rng: np.random.Generator, line, local, role, span: int, n_lines: int) -> np.ndarray:
    """The pulled positions under `op` keying, padding aside: one draw per line off the op word at role 1."""
    word = data[line * LINE_TOKENS + 1]
    draw = rng.random((line.shape[0], n_lines))
    drew = np.take_along_axis(draw, local, axis=1) < spec.p[word]
    return drew & ((role < span) if spec.pull == "span" else (role == 1))


@eqx.filter_jit
def _stream_axis(
    model: LanguageModel, tokens: Int[Array, "N T"], axes: tuple[int, ...] = ANCHOR_AXES
) -> Float[Array, "L1 N T"]:
    """Jitted at module level so the trajectory's repeated calls compile once (once per *axes*, a static arg)."""
    return axes_alignment(model.residual_stream(tokens), axes)


def alignment(
    model: LanguageModel,
    tokens: Int[np.ndarray, "N T"],
    batch_size: int = 1024,
    axes: tuple[int, ...] = ANCHOR_AXES,
) -> Float[np.ndarray, "L1 N T"]:
    """cos(h, e₁) at every residual-stream slice and position, for each line of *tokens*; with several *axes*, the alignment with their span (`axes_alignment`).

    nGPT is dropout-free, so there is no inference mode to switch into: the training-time and measurement-time forward passes are the same function.
    """
    chunks = [
        np.asarray(_stream_axis(model, jnp.asarray(tokens[i : i + batch_size]), tuple(axes)))
        for i in range(0, len(tokens), batch_size)
    ]
    return np.concatenate(chunks, axis=1)


def margin(alpha: Float[np.ndarray, "L1 C T"], weights: Float[np.ndarray, " C"]) -> Float[np.ndarray, "L1 T"]:
    """The alignment margin: label-affinity-weighted mean alignment, minus the plain mean.

    How much closer a color sits to the anchor for being the kind of color the anchor pulls. Zero means the axis is indifferent to redness; *weights* are the per-color label probabilities, normalized to sum to 1.
    """
    return np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)
