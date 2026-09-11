"""Fallback control for an anchored concept: a designed answer once the concept is removed.

M1's ex-2.9.2 trained a decoder to give a designed output when the anchored coordinate was flipped, so that a later intervention had a known destination instead of an undefined one. This module is the same term in the transformer, and its two supporting pieces:

- **The qualifying mask.** `qualifying_lines` reads off a training crop which `=` positions belong to a line the term scores: the concept operand (the redder one) carries at least `dose_min` redness, the visible operand is clean, and the concept operand's clean alignment at the edit slice clears `alignment_min`. Every test is a function of the crop and the clean stream, so the term needs no labels and no line bookkeeping from the sampler; the per-token tables in `FallbackSpec` are all it consults.
- **The reflected pass.** `reflected_logits` flips the anchor coordinate of every state at the edit slice (`projection` at γ = 2 in `sca.intervention`, the edit `redirect` applies at eval), detaches the result, and runs the remaining blocks and the unembedding forward. The detach is what keeps the term from moving where the concept sits: only the blocks after the edit and the tied LM head learn from it.
- **The anti-anchor hinge.** `anti_anchor_term` is mean(max(−α, 0)) over live positions and slices of the clean pass: a one-sided companion to `anti_subspace_term` that keeps clean states out of the hemisphere the fallback lives in without opposing the anchor.

`make_fallback_train_step` adds the two terms, at weights passed per step, to the anchored step of `sca.anchoring`. Every argument the anchored step takes is consumed the same way, so a run at zero weight on both terms is the anchored recipe through this code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PyTree

from sca.anchoring import ANCHOR_AXIS, anchor_term, anti_subspace_term, pooled_anchor_term
from sca.model._shared import normalize
from sca.model.ngpt import NGPT
from sca.training.loop import cross_entropy

EQ_OFFSET = 3
"""The `=` sits three tokens after op1 and one after op2 in a `op1 + op2 = ans ⏎` line."""

EPS = 1e-6
"""Slack on the redness thresholds: grid rednesses land on the thresholds (0.8 and 0.5) to float roundoff."""


@dataclass(frozen=True)
class FallbackSpec:
    """The fallback term's tables and thresholds; the weights arrive per step.

    `redness` and `answer` are per-token tables over the model's vocabulary: a color token's redness (zero for syntax and padding) and the token id of the fallback answer for a line whose *visible* operand is that token. `eq_token` is the id of `=`, where the answer is decoded.
    """

    redness: np.ndarray
    answer: np.ndarray
    eq_token: int
    dose_min: float = 0.8
    """A line qualifies when its concept operand, the redder of the two, reaches this redness."""
    visible_max: float = 0.5
    """... and its visible operand stays below this one, so the fallback answer is defined."""
    alignment_min: float = 0.5
    """... and the concept operand's clean alignment at the edit slice reaches this, so the reflection moves it."""
    slice: int = 0
    """Where the reflection acts: 0 is the embedding, k the state after block k."""

    def __post_init__(self):
        assert self.redness.shape == self.answer.shape, (self.redness.shape, self.answer.shape)
        assert self.redness[0] == 0.0 and self.answer[0] == 0, "padding is not a color"


class Qualifying(NamedTuple):
    """What `qualifying_lines` reads off a crop, all (B, T)."""

    active: Float[Array, "B T"]
    """1 at the `=` of every qualifying line."""
    target: Int[Array, "B T"]
    """The fallback token at those positions (meaningless elsewhere)."""
    concept: Float[Array, "B T"]
    """1 at the concept operand of every qualifying line: the position a redirect is meant to change."""


def qualifying_lines(x: Int[Array, "B T"], alpha: Float[Array, "B T"], spec: FallbackSpec) -> Qualifying:
    """Which `=` positions of a crop the term scores, and what it scores them against.

    A crop is a window onto the packed corpus, so a line's operands sit at fixed offsets before its `=`. Padding is a prefix of zeros, so a line whose op1 fell in the padding (or before the crop) reads a zero op1 and is skipped. *alpha* is the clean alignment at the edit slice, read from the same forward pass.
    """
    redness = jnp.asarray(spec.redness, dtype=alpha.dtype)
    answer = jnp.asarray(spec.answer, dtype=jnp.int32)
    op1, op2 = jnp.roll(x, EQ_OFFSET, axis=1), jnp.roll(x, 1, axis=1)
    a1, a2 = jnp.roll(alpha, EQ_OFFSET, axis=1), jnp.roll(alpha, 1, axis=1)
    in_crop = jnp.arange(x.shape[1]) >= EQ_OFFSET
    r1, r2 = redness[op1], redness[op2]
    op1_is_concept = r1 >= r2
    dose, visible_redness = jnp.maximum(r1, r2), jnp.minimum(r1, r2)
    a_concept = jnp.where(op1_is_concept, a1, a2)
    visible = jnp.where(op1_is_concept, op2, op1)
    active = (
        (x == spec.eq_token)
        & in_crop
        & (op1 != 0)
        & (dose >= spec.dose_min - EPS)
        & (visible_redness < spec.visible_max - EPS)
        & (a_concept >= spec.alignment_min)
    )
    concept = jnp.roll(active & op1_is_concept, -EQ_OFFSET, axis=1) | jnp.roll(active & ~op1_is_concept, -1, axis=1)
    return Qualifying(active.astype(alpha.dtype), answer[visible], concept.astype(alpha.dtype))


def reflect(h: Float[Array, "... C"]) -> Float[Array, "... C"]:
    """Flip the anchor coordinate: `projection` at γ = 2 with re-normalization, which a unit vector does not need."""
    return normalize(h.at[..., ANCHOR_AXIS].multiply(-1.0))


def reflected_logits(model: NGPT, h: Float[Array, "B T C"], slice_: int) -> Float[Array, "B T V"]:
    """Reflect the state at *slice_*, detach it, and run the rest of the model forward.

    The detach is the term's whole design: the reflected states carry no gradient back to the embedding or to the blocks before the edit, so the term trains a readout of the antipode and never moves the concept toward it.
    """
    h = jax.lax.stop_gradient(reflect(h))
    enc = model.transformer.rotary_enc
    run_block = eqx.filter_checkpoint(lambda block, h: block(h, enc))
    for block in model.transformer.blocks[slice_:]:
        h = run_block(block, h)
    return (h @ model.transformer.readout.T) * model.s_z()


def fallback_term(
    logits: Float[Array, "B T V"], target: Int[Array, "B T"], active: Float[Array, "B T"]
) -> Float[Array, ""]:
    """Mean cross-entropy against *target* over the *active* positions; zero when a crop holds none."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    return jnp.sum(nll * active) / (jnp.sum(active) + 1e-8)


def anti_anchor_term(states: Float[Array, "L1 B T C"], live: Float[Array, "B T"]) -> Float[Array, ""]:
    """Mean of max(−cos(h, e₁), 0) over every residual-stream slice and live position.

    A hinge on negative alignment: zero for any state on the anchor's side, growing linearly as a state crosses to the antipode. `anti_subspace_term` on the same sites penalizes α² on both sides; this one is what keeps the antipode hemisphere clear for the fallback without pushing back on the anchor.
    """
    cos = states[..., ANCHOR_AXIS]
    return jnp.sum(jnp.maximum(-cos, 0.0) * live) / (states.shape[0] * (jnp.sum(live) + 1e-8))


def make_fallback_train_step(
    optimizer: optax.GradientTransformation,
    spec: FallbackSpec,
    tau: float | None = None,
    n_lines: int = 0,
):
    """`sca.anchoring.make_anchored_train_step` plus the fallback and anti-anchor terms.

    Two more weights per step, `fb_weight` and `aa_weight`, and three more outputs: the two new losses and the number of qualifying lines in the crop. The reflected pass runs on every step whatever the weights, so a run at zero weight is the anchored recipe at about twice the cost, and the returned fallback loss on such a run is the untrained readout's cross-entropy at the antipode.
    """
    if tau is not None and n_lines < 1:
        raise ValueError(f"pooled anchor (tau={tau}) needs n_lines >= 1, got {n_lines}")

    @eqx.filter_jit
    def train_step(
        model: NGPT,
        opt_state: PyTree,
        x: Int[Array, "B T"],
        y: Int[Array, "B T"],
        mask: Float[Array, "B T"],
        line_id: Int[Array, "B T"],
        weight: Float[Array, ""],
        anti_weight: Float[Array, ""],
        fb_weight: Float[Array, ""],
        aa_weight: Float[Array, ""],
    ):
        def loss_fn(model: NGPT):
            states, logits = model.stream_and_logits(x)
            live = (x != 0).astype(states.dtype)
            task = cross_entropy(logits, y)
            anchor = (
                anchor_term(states, mask) if tau is None else pooled_anchor_term(states, mask, line_id, n_lines, tau)
            )
            anti = anti_subspace_term(states, live)
            # The mask reads the clean alignment as a constant: which lines qualify is not something to optimize.
            q = qualifying_lines(x, jax.lax.stop_gradient(states[spec.slice][..., ANCHOR_AXIS]), spec)
            fallback = fallback_term(reflected_logits(model, states[spec.slice], spec.slice), q.target, q.active)
            anti_anchor = anti_anchor_term(states, live)
            loss = task + weight * anchor + anti_weight * anti + fb_weight * fallback + aa_weight * anti_anchor
            return loss, (task, anchor, anti, fallback, anti_anchor, q.active.sum())

        (_, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model.normalize_weights(), opt_state, *aux

    return train_step


__all__ = [
    "EQ_OFFSET",
    "FallbackSpec",
    "Qualifying",
    "anti_anchor_term",
    "fallback_term",
    "make_fallback_train_step",
    "qualifying_lines",
    "reflect",
    "reflected_logits",
]
