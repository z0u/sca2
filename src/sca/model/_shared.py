"""Primitives the model is built from.

The `ngpt` module tells the architecture's story end to end; the supporting
pieces live here: the last-axis `Linear` layer, positional rotary encoding, the
learnable `Scale`, and head reshaping.

Models here are Equinox modules: pytrees of arrays transformed by JAX. Forward
passes are pure functions — randomness (sampling) enters only through explicit
PRNG keys, and "mutating" weights means building a new model.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray


def normalize(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Project onto the unit hypersphere along *axis* (like `F.normalize`)."""
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), eps)


class Linear(eqx.Module):
    """Affine map over the last axis of an input of any rank.

    Unlike `eqx.nn.Linear` (which expects a single unbatched vector), this
    broadcasts over leading axes, so modules can stay written in the batched
    (B, T, C) style.
    """

    weight: Float[Array, "out in"]
    bias: Float[Array, " out"] | None

    def __init__(self, in_features: int, out_features: int, *, key: PRNGKeyArray, use_bias: bool = True):
        wkey, bkey = jr.split(key)
        lim = in_features**-0.5
        self.weight = jr.uniform(wkey, (out_features, in_features), minval=-lim, maxval=lim)
        self.bias = jr.uniform(bkey, (out_features,), minval=-lim, maxval=lim) if use_bias else None

    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... out"]:
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


class RotaryEncoding(eqx.Module):
    """Rotary positional encoding (RoPE), applied per head during attention.

    Passed into the forward pass rather than owned by the attention module, so a
    single instance can be shared across all layers. Holds no arrays — the
    sin/cos tables are derived from the sequence length at call time (and
    constant-folded under jit), so there are no buffers for the optimizer to
    mistake for parameters.
    """

    n_head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)

    def __init__(self, n_head_dim: int, base: float = 10_000):
        self.n_head_dim = n_head_dim
        self.base = base

    def __call__(self, q: Float[Array, "B H T D"], k: Float[Array, "B H T D"]):
        T = q.shape[-2]
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.n_head_dim, 2) / self.n_head_dim))
        enc = jnp.concatenate((f := jnp.outer(jnp.arange(T), inv_freq), f), axis=-1)  # (T, n_head_dim)
        sin, cos = jnp.sin(enc), jnp.cos(enc)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _rotate_half(self, x: Array):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)


class Scale(eqx.Module):
    """Learnable scalar (n=1) or per-channel (n=d) gain with nGPT's reparametrization.

    Store the parameter at `scale`; the effective value is `param * (init / scale)`,
    so Adam's step dynamics are decoupled from the value's magnitude.
    """

    weight: Float[Array, " n"]
    forward_scale: float = eqx.field(static=True)

    def __init__(self, n: int, init: float, scale: float):
        self.forward_scale = init / scale
        self.weight = jnp.full((n,), scale)

    def __call__(self) -> Array:
        return self.weight * self.forward_scale


def split_heads(x: Float[Array, "B T C"], n_head: int) -> Float[Array, "B H T D"]:
    """Reshape (B, T, n_head * n_head_dim) into (B, n_head, T, n_head_dim)."""
    B, T, C = x.shape
    return x.reshape(B, T, n_head, C // n_head).swapaxes(1, 2)


def merge_heads(x: Float[Array, "B H T D"]) -> Float[Array, "B T C"]:
    """Reshape (B, n_head, T, n_head_dim) back into (B, T, n_head * n_head_dim)."""
    B, n_head, T, d = x.shape
    return x.swapaxes(1, 2).reshape(B, T, n_head * d)


class LanguageModel(eqx.Module):
    """Base for the model variants: holds the key dimensions and shared diagnostics.

    Subclasses build `self.transformer` and implement `__call__` mapping token
    indices (B, T) to logits (B, T, vocab). `normalize_weights` returns the model
    unchanged here and is overridden by variants that enforce the
    unit-hypersphere weight constraint.
    """

    block_size: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    def __call__(self, idx: Int[Array, "B T"], *, key: PRNGKeyArray | None = None) -> Float[Array, "B T V"]:
        raise NotImplementedError

    def normalize_weights(self) -> "LanguageModel":
        """Re-project weights onto the unit hypersphere; identity unless overridden."""
        return self

    def get_num_params(self) -> int:
        """Calculate the number of parameters in the model."""
        return sum(x.size for x in jax.tree.leaves(eqx.filter(self, eqx.is_array)))

