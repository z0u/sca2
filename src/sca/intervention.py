"""Interventions on the residual stream: the eval contract and its operator library.

Every method that claims to have located a concept, whether it placed the concept there during training (`sca.anchoring`) or found it afterwards (`diff_in_means`, `probe_direction`, `leace`), produces the same triple: a model, a `Subspace`, and an operator that edits the stream at that subspace. One scorer, `apply`, takes the triple and returns what every downstream statistic reads: the stream as it arrived at each operator, the stream after it, and the logits. Keeping the contract method-agnostic is what lets a post-hoc baseline and an anchored model be scored on identical lines with identical code.

Three operators. `projection` removes a fraction γ of the concept component and re-projects onto the sphere — M1's suppression, with strength as a dose axis. `lobe` is M1's shaped suppression (`asec_intervention_lobes.tex`): a falloff that leaves states below an alignment threshold untouched. `ablate_weights` is the permanent form: it zeroes the subspace in every matrix that reads from or writes to the stream, then re-normalizes the weights, in that order.

Where operators act: on the between-block stream, the same slices `NGPT.residual_stream` returns and the anchor term reads. The stream is unit-norm, so an operator's output goes back onto the sphere before the next block consumes it, and the re-projection is part of the write: removing a component of size α rescales what survives by `gain(α, γ)`, and the whole edit is a rotation by `write_angle(α, γ)`. Both are closed-form in the pre-intervention alignment, which is what makes a write bound computable from a published alignment map before any intervention runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from sca.anchoring import ANCHOR_AXIS
from sca.model import NGPT
from sca.model._shared import normalize

Operator = Callable[[Float[Array, "... C"]], Float[Array, "... C"]]
"""An edit to unit-norm states, applied at one slice and broadcast over positions."""


@dataclass(frozen=True)
class Subspace:
    """Where a concept lives, as the contract sees it: what to remove, and how to read how much is there.

    `basis` spans the directions an operator removes; `dual` reads the coefficients, `dual @ (h − mean)`, that say how much of each to remove. For an orthogonal projection the two coincide and `mean` is zero, which is what an anchored axis is (`axis`) and what a fitted direction is once normalized. LEACE is the case that needs the split: it removes one direction while reading along another, so the erasure is oblique. `orthogonal` says which kind this is; `ablate_weights` accepts only the orthogonal kind.
    """

    basis: Float[np.ndarray, "k C"]
    dual: Float[np.ndarray, "k C"]
    mean: Float[np.ndarray, " C"]

    @classmethod
    def axis(cls, width: int, axis: int = ANCHOR_AXIS) -> Subspace:
        """The anchored basis vector: e₁ for the default axis."""
        return cls.direction(np.eye(width)[axis])

    @classmethod
    def direction(cls, v: Float[np.ndarray, " C"]) -> Subspace:
        """An orthogonal rank-1 subspace along *v*, normalized."""
        u = np.asarray(v, dtype=np.float32)
        u = (u / np.linalg.norm(u))[None]
        return cls(basis=u, dual=u, mean=np.zeros(u.shape[1], dtype=np.float32))

    @property
    def width(self) -> int:
        return self.basis.shape[1]

    @property
    def orthogonal(self) -> bool:
        return bool(np.allclose(self.basis, self.dual, atol=1e-6) and np.allclose(self.mean, 0.0, atol=1e-6))

    def coefficients(self, h: Float[Array, "... C"]) -> Float[Array, "... k"]:
        """How much of each basis direction *h* carries, read along the dual."""
        return (h - self.mean) @ self.dual.T


def projection(sub: Subspace, gamma: float = 1.0, renorm: bool = True) -> Operator:
    """Remove a fraction *gamma* of the subspace component, then re-project onto the sphere.

    γ = 1 is M1's suppression; γ = 0 is the identity, and the range between is the strength axis a categorical concept grades on. With *renorm* the surviving components pick up the gain `gain(α, γ)`, since the next block expects a unit vector; without it the state leaves the sphere, which is the off-manifold form M1's appendix draws.
    """

    def op(h: Float[Array, "... C"]) -> Float[Array, "... C"]:
        out = h - gamma * (sub.coefficients(h) @ sub.basis)
        return normalize(out) if renorm else out

    return op


def falloff(alpha: Array, a: float, b: float, p: float) -> Array:
    """M1's suppression strength h(α): zero below the threshold *a*, rising to *b* at α = 1 with shape *p*."""
    t = jnp.clip((alpha - a) / (1.0 - a), 0.0, 1.0)
    return jnp.where(alpha >= a, b * t**p, 0.0)


def lobe(sub: Subspace, a: float = 0.0, b: float = 1.0, p: float = 1.0, renorm: bool = True) -> Operator:
    """Shaped suppression on a rank-1 subspace: remove `h(α) · α` of the direction, α = max(0, coefficient).

    `a = 0, b = 1, p = 0` reduces to `projection` at γ = 1 for positively aligned states, so the lobe's extra freedom is the threshold that leaves the low-alignment bulk alone — the answer to a common component on the axis, if the plain projection turns out to move everything.
    """
    assert sub.basis.shape[0] == 1, "the lobe is defined on a single direction"

    def op(h: Float[Array, "... C"]) -> Float[Array, "... C"]:
        alpha = jnp.maximum(sub.coefficients(h)[..., 0], 0.0)
        out = h - (falloff(alpha, a, b, p) * alpha)[..., None] * sub.basis[0]
        return normalize(out) if renorm else out

    return op


def _survivor_norm(a: np.ndarray, gamma: float) -> np.ndarray:
    """|h − γ·α·v| for a unit state with alignment α: what re-normalization divides by."""
    return np.sqrt((1.0 - gamma) ** 2 * a**2 + 1.0 - a**2)


def gain(alpha, gamma: float = 1.0):
    """The rescaling of the surviving components when a unit state with alignment *alpha* is projected at strength *gamma* and re-normalized: 1/√((1−γ)²α² + 1 − α²), which is 1/√(1 − α²) at full strength. Infinite for a fully aligned state at full strength, since nothing survives to rescale."""
    with np.errstate(divide="ignore"):
        return 1.0 / _survivor_norm(np.asarray(alpha, dtype=float), gamma)


def write_angle(alpha, gamma: float = 1.0):
    """The rotation, in radians, that `projection` applies to a unit state with alignment *alpha* — the size of the write, closed-form in the pre-intervention geometry. At full strength it is arcsin|α|."""
    a = np.asarray(alpha, dtype=float)
    n = _survivor_norm(a, gamma)
    cos = np.divide((1.0 - gamma) * a**2 + 1.0 - a**2, n, out=np.zeros_like(n), where=n > 0)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def angle_between(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle in radians between unit vectors along the last axis, elementwise over the rest.

    Computed from both the sine and the cosine, since arccos alone loses the small angles the write bound is about: at float32, arccos of a cosine rounded to 1 reads 0 or 4.5e-4 and nothing between.
    """
    cos = np.sum(x * y, axis=-1)
    sin = np.linalg.norm(x - cos[..., None] * y, axis=-1)
    return np.arctan2(sin, cos)


class Applied(NamedTuple):
    """What the scorer returns: the stream before and after each operator, and the logits."""

    pre: Float[np.ndarray, "L1 N T C"]
    """The state as it arrived at each slice — the clean stream at the first intervened slice, and downstream of earlier edits after that."""
    post: Float[np.ndarray, "L1 N T C"]
    """The state the next block consumed. Equal to `pre` at slices the operator skipped."""
    logits: Float[np.ndarray, "N T V"]


def _forward(
    model: NGPT,
    idx: Int[Array, "B T"],
    operator: Operator,
    slices: tuple[int, ...],
    positions: Float[Array, " T"] | None,
) -> tuple[Float[Array, "L1 B T C"], Float[Array, "L1 B T C"], Float[Array, "B T V"]]:
    """`NGPT.residual_stream` with the operator spliced in between blocks; see `apply`."""

    def edit(x: Float[Array, "B T C"]) -> Float[Array, "B T C"]:
        out = operator(x)
        return out if positions is None else jnp.where(positions[None, :, None] > 0, out, x)

    x = normalize(model.transformer.wte[idx])
    pre, post = [x], [edit(x) if 0 in slices else x]
    x = post[-1]
    for i, block in enumerate(model.transformer.blocks, start=1):
        x = block(x, model.transformer.rotary_enc)
        pre.append(x)
        post.append(edit(x) if i in slices else x)
        x = post[-1]
    return jnp.stack(pre), jnp.stack(post), (x @ model.transformer.wte.T) * model.s_z()


def apply(
    model: NGPT,
    tokens: Int[np.ndarray, "N T"],
    operator: Operator,
    slices: tuple[int, ...],
    positions: np.ndarray | None = None,
    batch_size: int = 1024,
) -> Applied:
    """Score a triple: run *tokens* through *model* with *operator* applied at each of *slices* (0 = embedding).

    *positions* is an optional (T,) mask of the positions the operator touches; None means all of them. An empty *slices* is the clean forward pass, and the scorer's own control: `pre` and `post` then both equal the residual stream and the logits equal `model(tokens)`.
    """
    pos = None if positions is None else jnp.asarray(np.asarray(positions, dtype=np.float32))
    run = eqx.filter_jit(lambda m, t: _forward(m, t, operator, tuple(slices), pos))
    pres, posts, logits = [], [], []
    for i in range(0, len(tokens), batch_size):
        a, b, c = run(model, jnp.asarray(tokens[i : i + batch_size]))
        pres.append(np.asarray(a))
        posts.append(np.asarray(b))
        logits.append(np.asarray(c))
    return Applied(np.concatenate(pres, axis=1), np.concatenate(posts, axis=1), np.concatenate(logits))


def answer_logprobs(logits: Float[np.ndarray, "N T V"], answer_pos: int) -> Float[np.ndarray, "N V"]:
    """Log-probabilities over the vocabulary for the token at *answer_pos*, read at the position before it."""
    return np.asarray(jax.nn.log_softmax(jnp.asarray(logits[:, answer_pos - 1]), axis=-1))


def ablate_weights(model: NGPT, sub: Subspace) -> NGPT:
    """Zero the subspace in every matrix that reads from or writes to the stream, then re-normalize the weights.

    Matrices that read the stream (the embedding table, which doubles as the LM head, and the qkv and MLP up-projections) lose the subspace on their input axis; matrices that write it (the attention and MLP down-projections) lose it on their output axis. `normalize_weights` runs after, since zeroing entries shortens the rows nGPT keeps at unit length; the ablation's declared order is project first, re-normalize second. No block can then read the concept or write it back, and the embedding never carries it, so the stream stays off the subspace at every slice — which is M1's permanent removal, and needs the subspace to have been cleared for it to be selective.
    """
    assert sub.orthogonal, "weight ablation removes an orthogonal subspace; an oblique eraser has no weight form"
    P = jnp.asarray(np.eye(sub.width, dtype=np.float32) - sub.basis.T @ sub.basis)

    def where(m: NGPT):
        return (
            [m.transformer.wte]
            + [b.attn.qkv.weight for b in m.transformer.blocks]
            + [b.mlp.fc.weight for b in m.transformer.blocks]
            + [b.attn.proj.weight for b in m.transformer.blocks]
            + [b.mlp.proj.weight for b in m.transformer.blocks]
        )  # fmt: skip

    def replacements(m: NGPT):
        return (
            [m.transformer.wte @ P]
            + [b.attn.qkv.weight @ P for b in m.transformer.blocks]
            + [b.mlp.fc.weight @ P for b in m.transformer.blocks]
            + [P @ b.attn.proj.weight for b in m.transformer.blocks]
            + [P @ b.mlp.proj.weight for b in m.transformer.blocks]
        )  # fmt: skip

    return eqx.tree_at(where, model, replacements(model)).normalize_weights()


# --- Post-hoc directions: the baseline tier's half of the contract --------------


def diff_in_means(x: Float[np.ndarray, "N C"], labels: np.ndarray) -> Subspace:
    """The direction from the unlabeled mean to the labeled mean, as an orthogonal rank-1 subspace."""
    lab = np.asarray(labels, dtype=bool)
    return Subspace.direction(x[lab].mean(0) - x[~lab].mean(0))


def probe_direction(x: Float[np.ndarray, "N C"], z: Float[np.ndarray, " N"], l2: float = 1e-2) -> Subspace:
    """The ridge-probe weight vector for a scalar target *z*, normalized: the direction a linear readout uses."""
    xc = x - x.mean(0)
    zc = np.asarray(z, dtype=np.float64) - np.mean(z)
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(x.shape[1]), xc.T @ zc)
    return Subspace.direction(w)


def leace(x: Float[np.ndarray, "N C"], z: Float[np.ndarray, " N"], l2: float = 1e-6) -> Subspace:
    """LEACE for a scalar concept (Belrose et al., 2023): the least-squares-optimal affine edit after which no linear probe can read *z*.

    r(h) = h − c · (cᵀΣ⁻¹(h − μ)) / (cᵀΣ⁻¹c), with μ and Σ the mean and covariance of the states and c = Cov(h, z). It removes the covariance direction c, but reads the amount along Σ⁻¹c, which is why it is oblique: `basis` is ĉ and `dual` is |c| Σ⁻¹c / (cᵀΣ⁻¹c), so `projection` at γ = 1 reproduces r. The guarantee is Cov(r(h), z) = 0. *l2* is a ridge on Σ for a stream whose covariance is rank-deficient.
    """
    mu = x.mean(0)
    xc = np.asarray(x - mu, dtype=np.float64)
    zc = np.asarray(z, dtype=np.float64) - np.mean(z)
    cov = xc.T @ xc / len(x) + l2 * np.eye(x.shape[1])
    c = xc.T @ zc / len(x)
    s_inv_c = np.linalg.solve(cov, c)
    norm = np.linalg.norm(c)
    basis = (c / norm)[None].astype(np.float32)
    dual = (norm * s_inv_c / (c @ s_inv_c))[None].astype(np.float32)
    return Subspace(basis=basis, dual=dual, mean=mu.astype(np.float32))
