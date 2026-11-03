"""Post-hoc directions at scale: fits from accumulated moments, edits that differ by slice, and a searched direction.

`sca.intervention` fits a direction from states held in memory and applies one operator at every slice it edits.
The post-hoc comparison needs more on both sides. Its fit set is far larger than memory allows at every slice and
position, so the fits here are made from moments accumulated over it (`Moments`): counts, sums, and sums of products,
from which the difference in means, the ridge probe, LEACE, and the iterated probe (INLP) all follow in closed form.
Its fitted directions differ from slice to slice, so the edit here (`SliceEdit`) carries one subspace per slice, as
stacked arrays, with the dose as an array too: every configuration of one rank then runs through one compiled
program. And it fits one direction by gradient descent against the scoring itself (`optimize_directions`), as a
ceiling for what any single direction could do.

The edit is `sca.intervention.projection` slice by slice: remove the dose of the component read along the dual,
then put the state back on the sphere. Mean ablation is the same edit with the subspace's mean set to the mean state
of the contexts of the other ops (`with_mean`), so the coefficient is replaced by its mean there instead of by zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from sca.intervention import Subspace
from sca.model import NGPT
from sca.model._shared import normalize

# --- Moments -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Moments:
    """Sums over a set of states, each with a row of targets: what every closed-form fit here reads.

    `z` holds one column per target (a binary label, a graded one, a permuted copy, a thinned one). A binary
    column also splits the states into two classes, so the class means come from the same sums.
    """

    n: float
    sx: Float[np.ndarray, " C"]
    sxx: Float[np.ndarray, "C C"]
    sxz: Float[np.ndarray, "Z C"]
    sz: Float[np.ndarray, " Z"]
    szz: Float[np.ndarray, " Z"]

    def __add__(self, other: Moments) -> Moments:
        return Moments(*(a + b for a, b in zip(self.astuple(), other.astuple(), strict=True)))

    def astuple(self) -> tuple:
        return (self.n, self.sx, self.sxx, self.sxz, self.sz, self.szz)

    @classmethod
    def of(cls, x: Float[np.ndarray, "N C"], z: Float[np.ndarray, "N Z"]) -> Moments:
        """The moments of states held in memory, for tests and small sets."""
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
        return cls(float(len(x)), x.sum(0), x.T @ x, z.T @ x, z.sum(0), (z**2).sum(0))

    @property
    def mean(self) -> np.ndarray:
        return self.sx / self.n

    def gram(self) -> np.ndarray:
        """The centered sum of outer products, Σ (x − μ)(x − μ)ᵀ."""
        return self.sxx - self.n * np.outer(self.mean, self.mean)

    def cross(self, j: int) -> np.ndarray:
        """The centered sum Σ (x − μ)(z_j − z̄_j)."""
        return self.sxz[j] - self.mean * self.sz[j]

    def class_means(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """The mean state where binary column *j* is one, and where it is zero."""
        pos = self.sxz[j] / self.sz[j]
        neg = (self.sx - self.sxz[j]) / (self.n - self.sz[j])
        return pos, neg


def dim_fit(m: Moments, j: int) -> Subspace:
    """The difference in means for binary column *j*: from the mean of the rest to the mean of the labelled."""
    pos, neg = m.class_means(j)
    return Subspace.direction(pos - neg)


def ridge_fit(m: Moments, j: int, l2: float = 1e-2) -> Subspace:
    """The ridge-probe weight vector for column *j*, normalized: `sca.intervention.probe_direction`, from moments."""
    return Subspace.direction(np.linalg.solve(m.gram() + l2 * np.eye(len(m.sx)), m.cross(j)))


def leace_fit(m: Moments, j: int, l2: float = 1e-6) -> Subspace:
    """LEACE for column *j*: `sca.intervention.leace`, from moments."""
    cov = m.gram() / m.n + l2 * np.eye(len(m.sx))
    c = m.cross(j) / m.n
    s_inv_c = np.linalg.solve(cov, c)
    norm = np.linalg.norm(c)
    basis = (c / norm)[None].astype(np.float32)
    dual = (norm * s_inv_c / (c @ s_inv_c))[None].astype(np.float32)
    return Subspace(basis=basis, dual=dual, mean=m.mean.astype(np.float32))


def inlp_fit(m: Moments, j: int, rank: int, l2: float = 1e-2) -> Subspace:
    """The iterated probe for column *j*: fit the ridge probe, project its direction out of the states, and fit
    again on what is left, *rank* times. The edit removes the whole span. Projecting the states out by P turns
    their centered Gram matrix into P G P and the cross term into P c, so each round is a ridge solve on those.
    """
    width = len(m.sx)
    g, c = m.gram(), m.cross(j)
    u = np.zeros((0, width))
    for _ in range(rank):
        p = np.eye(width) - u.T @ u
        w = p @ np.linalg.solve(p @ g @ p + l2 * np.eye(width), p @ c)
        u = np.vstack([u, w / np.linalg.norm(w)])
    u = u.astype(np.float32)
    return Subspace(basis=u, dual=u, mean=np.zeros(width, np.float32))


def with_mean(sub: Subspace, mean: np.ndarray) -> Subspace:
    """Mean ablation: the same edit, with the coefficient read from *mean* rather than from the subspace's own
    center, so the component along the subspace is replaced by that of *mean* (before the state goes back onto
    the sphere) instead of being removed.
    """
    return Subspace(basis=sub.basis, dual=sub.dual, mean=np.asarray(mean, np.float32))


# --- The edit, slice by slice --------------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=1)
def _unpack(flat: Float[Array, " n"], shapes: tuple[tuple[int, ...], ...]) -> tuple[Array, ...]:
    out, i = [], 0
    for shape in shapes:
        n = math.prod(shape)
        out.append(flat[i : i + n].reshape(shape))
        i += n
    return tuple(out)


class SliceEdit(eqx.Module):
    """A subspace per slice, stacked: what to remove at each slice, how much (`gamma`), and at which positions.

    A slice with `gamma` zero is left as it arrives. Every field is an array, so any edit of the same rank and
    shape reuses one compiled program, whatever the slices, doses, and positions.
    """

    basis: Float[Array, "L1 k C"]
    dual: Float[Array, "L1 k C"]
    mean: Float[Array, "L1 C"]
    gamma: Float[Array, " L1"]
    positions: Float[Array, " T"]

    @classmethod
    def build(
        cls,
        subs: dict[int, Subspace],
        n_slices: int,
        width: int,
        gamma: float,
        positions: np.ndarray,
        rank: int = 1,
    ) -> SliceEdit:
        """The edit that applies `subs[l]` at each slice *l* in *subs* at dose *gamma*, at *positions* (a mask)."""
        basis = np.zeros((n_slices, rank, width), np.float32)
        dual = np.zeros((n_slices, rank, width), np.float32)
        mean = np.zeros((n_slices, width), np.float32)
        g = np.zeros(n_slices, np.float32)
        for sl, sub in subs.items():
            k = sub.basis.shape[0]
            assert k <= rank, f"a rank-{k} subspace in a rank-{rank} edit"
            basis[sl, :k], dual[sl, :k], mean[sl], g[sl] = sub.basis, sub.dual, sub.mean, gamma
        parts = (basis, dual, mean, g, np.asarray(positions, np.float32))
        # One transfer for the whole edit: on the cloud workers each host-to-device copy costs milliseconds, and a
        # scoring pass builds an edit per configuration.
        flat = jax.device_put(np.concatenate([a.ravel() for a in parts]))
        return cls(*_unpack(flat, tuple(a.shape for a in parts)))

    @classmethod
    def identity(cls, n_slices: int, width: int, length: int, rank: int = 1) -> SliceEdit:
        return cls.build({}, n_slices, width, 0.0, np.ones(length, np.float32), rank)

    def __call__(self, x: Float[Array, "B T C"], sl: int) -> Float[Array, "B T C"]:
        coef = (x - self.mean[sl]) @ self.dual[sl].T
        out = normalize(x - self.gamma[sl] * (coef @ self.basis[sl]))
        touch = (self.positions[None, :, None] > 0) & (self.gamma[sl] > 0)
        return jnp.where(touch, out, x)


def _blocks(model: NGPT, idx: Int[Array, "B T"], edit: SliceEdit):
    """The forward pass with *edit* at every slice: yields (slice, the state as it arrived, the state after the
    edit), then returns the final state.
    """
    x = model.embed(idx)
    allowed = model.attention_mask(idx)
    enc = model.transformer.rotary_enc
    states = []
    for sl in range(len(model.transformer.blocks) + 1):
        if sl > 0:
            x = model.transformer.blocks[sl - 1](x, enc, allowed)
        after = edit(x, sl)
        states.append((x, after))
        x = after
    return states, x


def _logits(model: NGPT, idx: Int[Array, "B T"], edit: SliceEdit, at: int) -> Float[Array, "B V"]:
    _, x = _blocks(model, idx, edit)
    return (x[:, at] @ model.transformer.readout.T) * model.s_z()


_logits_jit = eqx.filter_jit(_logits)


def _moments(
    model: NGPT, idx: Int[Array, "B T"], edit: SliceEdit, masks: Float[Array, "S T"], z: Float[Array, "B Z"]
) -> tuple[Array, ...]:
    """Per slice and site, the moments of the states as they arrive under *edit* (before the edit at that slice)."""
    states, _ = _blocks(model, idx, edit)
    x = jnp.stack([pre for pre, _ in states])  # (L1, B, T, C)
    w = jnp.broadcast_to(masks[:, None, :], (masks.shape[0], idx.shape[0], idx.shape[1]))  # (S, B, T)
    hp = jax.lax.Precision.HIGHEST
    n = w.sum((1, 2))  # (S,)
    sx = jnp.einsum("sbt,lbtc->lsc", w, x, precision=hp)
    sxx = jnp.einsum("sbt,lbtc,lbtd->lscd", w, x, x, precision=hp)
    wz = jnp.einsum("sbt,bz->sbtz", w, z)
    sxz = jnp.einsum("sbtz,lbtc->lszc", wz, x, precision=hp)
    sz = wz.sum((1, 2))  # (S, Z)
    szz = jnp.einsum("sbtz,bz->sz", wz, z)
    return n, sx, sxx, sxz, sz, szz


_moments_jit = eqx.filter_jit(_moments)


def _pre_post(model: NGPT, idx: Int[Array, "B T"], edit: SliceEdit) -> tuple[Array, Array]:
    states, _ = _blocks(model, idx, edit)
    return jnp.stack([a for a, _ in states]), jnp.stack([b for _, b in states])


_pre_post_jit = eqx.filter_jit(_pre_post)


def _batches(tokens: np.ndarray | Array, batch_size: int):
    """Fixed-size batches of *tokens*, the last padded by repeating its final row: (batch, rows that count).
    Tokens already on the device that fit in one batch go as they are.
    """
    batch_size = min(batch_size, len(tokens))
    if isinstance(tokens, jax.Array) and batch_size == len(tokens):
        yield 0, tokens, len(tokens)
        return
    for i in range(0, len(tokens), batch_size):
        chunk = tokens[i : i + batch_size]
        n = len(chunk)
        if n < batch_size:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch_size - n, axis=0)])
        yield i, jnp.asarray(chunk), n


def logits_at(
    model: NGPT, tokens: Int[np.ndarray, "N T"], edit: SliceEdit, at: int, batch_size: int = 8192
) -> Float[np.ndarray, "N V"]:
    """The logits at position *at* under *edit*: `sca.intervention.logits_at`, with a subspace per slice."""
    return np.concatenate([np.asarray(_logits_jit(model, b, edit, at))[:n] for _, b, n in _batches(tokens, batch_size)])


@eqx.filter_jit
def _color_logprobs(model: NGPT, idx: Int[Array, "B T"], edit: SliceEdit, at: int, color_ids: Int[Array, " K"]):
    lg = _logits(model, idx, edit, at)
    return lg[:, color_ids] - jax.nn.logsumexp(lg, axis=1, keepdims=True)


def color_probs_at(
    model: NGPT,
    tokens: Int[np.ndarray, "N T"],
    edit: SliceEdit,
    at: int,
    color_ids: np.ndarray,
    batch_size: int = 16384,
) -> Float[np.ndarray, "N K"]:
    """The answer distribution at the color tokens under *edit*: the softmax over the whole vocabulary taken on
    the device, so only the color columns come back (as `ex2216._color_probs` of `logits_at`, to float32
    rounding).
    """
    cid = jnp.asarray(color_ids)
    parts = [np.asarray(_color_logprobs(model, b, edit, at, cid))[:n] for _, b, n in _batches(tokens, batch_size)]
    return np.exp(np.concatenate(parts).astype(np.float64))


def stream_moments(
    model: NGPT,
    tokens: Int[np.ndarray, "N T"],
    edit: SliceEdit,
    masks: Float[np.ndarray, "S T"],
    z: Float[np.ndarray, "N Z"],
    batch_size: int = 4096,
) -> list[list[Moments]]:
    """The moments of the states as they arrive at each slice under *edit*, at each site of *masks* (a row of
    weights over positions per site), with *z* as the targets of each context: `[slice][site]`.
    """
    jm = jnp.asarray(np.asarray(masks, np.float32))
    total = None
    for i, b, n in _batches(tokens, batch_size):
        zb = np.zeros((b.shape[0], z.shape[1]), np.float32)
        zb[:n] = z[i : i + n]
        # The padded rows carry zero targets but would still count as states, so they are masked out of the
        # weights by running them as a separate, smaller program only when the last batch is short.
        if n < b.shape[0]:
            parts = _moments_jit(model, b[:n], edit, jm, jnp.asarray(zb[:n]))
        else:
            parts = _moments_jit(model, b, edit, jm, jnp.asarray(zb))
        parts = [np.asarray(p, np.float64) for p in parts]
        total = parts if total is None else [t + p for t, p in zip(total, parts, strict=True)]
    assert total is not None
    n, sx, sxx, sxz, sz, szz = total
    return [
        [Moments(float(n[s]), sx[sl, s], sxx[sl, s], sxz[sl, s], sz[s], szz[s]) for s in range(len(masks))]
        for sl in range(sx.shape[0])
    ]


def pre_post(
    model: NGPT, tokens: Int[np.ndarray, "N T"], edit: SliceEdit, batch_size: int = 512
) -> tuple[Float[np.ndarray, "L1 N T C"], Float[np.ndarray, "L1 N T C"]]:
    """The state as it arrived at each slice under *edit*, and the state after the edit there."""
    pres, posts = [], []
    for _, b, n in _batches(tokens, batch_size):
        a, c = _pre_post_jit(model, b, edit)
        pres.append(np.asarray(a)[:, :n])
        posts.append(np.asarray(c)[:, :n])
    return np.concatenate(pres, axis=1), np.concatenate(posts, axis=1)


# --- The optimized direction ---------------------------------------------------------------------------------


def _color_logp(logits: Float[Array, "B V"], color_ids: Int[Array, " P"]) -> Float[Array, "B P"]:
    """Log-probabilities at the color tokens under the softmax over the whole vocabulary, as the scoring reads them."""
    return jax.nn.log_softmax(logits, axis=-1)[:, color_ids]


def _search_loss(
    v: Float[Array, "L1 C"],
    model: NGPT,
    idx: Int[Array, "B T"],
    gamma: Float[Array, " L1"],
    at: int,
    color_ids: Int[Array, " P"],
    q_null: Float[Array, "B P"],
    w_null: Float[Array, " B"],
    w_keep: Float[Array, " B"],
    weight: Float[Array, ""],
) -> Float[Array, ""]:
    """KL from the target null to the edited answers, weighted by *w_null*, plus *weight* times KL from the clean
    answers to the edited ones, weighted by *w_keep*: the loss the optimized direction is searched on.
    """
    # A slice that is not edited has a zero row, where the gradient of a norm is undefined, so the length is
    # smoothed.
    u = (v * jax.lax.rsqrt((v**2).sum(-1, keepdims=True) + 1e-12))[:, None, :]
    width = v.shape[-1]
    edit = SliceEdit(u, u, jnp.zeros((v.shape[0], width)), gamma, jnp.ones(idx.shape[1]))
    lp_edit = _color_logp(_logits(model, idx, edit, at), color_ids)
    identity = SliceEdit(u, u, jnp.zeros((v.shape[0], width)), jnp.zeros_like(gamma), jnp.ones(idx.shape[1]))
    lp_clean = jax.lax.stop_gradient(_color_logp(_logits(model, idx, identity, at), color_ids))
    p_clean = jnp.exp(lp_clean)
    kl_null = (q_null * (jnp.log(jnp.maximum(q_null, 1e-12)) - lp_edit)).sum(-1)
    kl_keep = (p_clean * (lp_clean - lp_edit)).sum(-1)
    return (w_null * kl_null + weight * w_keep * kl_keep).sum() / jnp.maximum((w_null + w_keep).sum(), 1.0)


@eqx.filter_jit
def _search_step(v, opt_state, model, data, i, gamma, at, color_ids, weight, optimizer):
    tokens, q_null, w_null, w_keep = data
    loss, grad = jax.value_and_grad(_search_loss)(
        v, model, tokens[i], gamma, at, color_ids, q_null[i].astype(jnp.float32), w_null[i], w_keep[i], weight
    )
    updates, opt_state = optimizer.update(grad, opt_state, v)
    return v + updates, opt_state, loss


def search_data(
    tokens: Int[np.ndarray, "N T"],
    q_null: Float[np.ndarray, "N P"],
    w_null: Float[np.ndarray, " N"],
    w_keep: Float[np.ndarray, " N"],
) -> tuple:
    """The set a search draws its batches from, placed on the device once (the target nulls in half precision), so
    that a step moves only the indices of its batch.
    """
    return (
        jnp.asarray(tokens, jnp.int32),
        jnp.asarray(q_null, jnp.float16),
        jnp.asarray(w_null, jnp.float32),
        jnp.asarray(w_keep, jnp.float32),
    )


def optimize_directions(
    model: NGPT,
    data: tuple,
    color_ids: np.ndarray,
    slices: tuple[int, ...],
    at: int,
    weight: float,
    init: Float[np.ndarray, "L1 C"],
    steps: int = 300,
    batch_size: int = 512,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[Float[np.ndarray, "L1 C"], list[float]]:
    """A unit vector per slice in *slices*, found by Adam so that the full edit along them (projection at every
    position) moves the answers at *at* toward the target null where it weighs a context, and keeps them where they
    were where the keep weight does, with *weight* on the second term. *data* is `search_data` (the tokens, the
    target nulls, and the two weights per context). Starts from *init*; returns the vectors (zero rows at the slices
    not edited) and the loss at each step.
    """
    import optax

    n_slices = init.shape[0]
    gamma = np.zeros(n_slices, np.float32)
    gamma[list(slices)] = 1.0
    optimizer = optax.adam(lr)
    v = jnp.asarray(init, jnp.float32)
    opt_state = optimizer.init(v)
    rng = np.random.default_rng(seed)
    n = data[0].shape[0]
    args = (jnp.asarray(gamma), at, jnp.asarray(color_ids), jnp.asarray(weight, jnp.float32), optimizer)
    losses = []
    for _ in range(steps):
        i = jnp.asarray(rng.integers(0, n, size=batch_size))
        v, opt_state, loss = _search_step(v, opt_state, model, data, i, *args)
        losses.append(loss)
    out = np.array(normalize(v))
    out[[s for s in range(n_slices) if s not in slices]] = 0.0
    return out, [float(x) for x in jax.device_get(losses)]
