"""nGPT: every activation and weight matrix lives on the unit hypersphere.

A stripped-back take on the published recipe: scalar gains everywhere (in place
of the per-channel eigen learning rates), and the residual step α *fixed* at
1/n_layer rather than learned — the value the learned gates settled on anyway.
What we keep from nGPT proper is the residual form itself: a LERP toward the
sub-module's *normalized* output, `h ← Norm(h + α·(ĥ* − h))`. Normalizing the
sub-module output (ĥ*) is load-bearing — it makes α the true step size,
independent of the raw output's norm (which scales like √n_embd for the MLP),
so the per-layer rotation stays ~α and the stack's travel holds O(1) regardless
of width.
"""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from sca.config import ModelConfig
from sca.model._shared import (
    LanguageModel,
    Linear,
    RotaryEncoding,
    Scale,
    merge_heads,
    normalize,
    split_heads,
)

log = logging.getLogger(__name__)


class CausalSelfAttention(eqx.Module):
    n_head: int = eqx.field(static=True)
    n_kq_tot: int = eqx.field(static=True)
    n_v_tot: int = eqx.field(static=True)

    qkv: Linear
    proj: Linear
    s_qk: Scale

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        self.n_head = config.n_head
        self.n_kq_tot = config.n_head * config.n_head_dim
        self.n_v_tot = config.n_head * config.n_head_dim

        # Bias-free: biases would push activations off the unit hypersphere.
        qkv_key, proj_key = jr.split(key)
        self.qkv = Linear(config.n_embd, 2 * self.n_kq_tot + self.n_v_tot, key=qkv_key, use_bias=False)
        self.proj = Linear(self.n_v_tot, config.n_embd, key=proj_key, use_bias=False)

        # Per-head q/k are unit-normalized, so their dot product is a cosine in
        # [−1, 1] and needs sharpening back up before softmax: a single learnable
        # scalar temperature, initialized to √d_k.
        self.s_qk = Scale(1, init=config.n_head_dim**0.5, scale=config.n_embd**-0.5)

    def __call__(self, x: Float[Array, "B T C"], enc: RotaryEncoding):
        _B, T, _C = x.shape
        q, k, v = jnp.split(self.qkv(x), [self.n_kq_tot, 2 * self.n_kq_tot], axis=-1)
        q = split_heads(q, self.n_head)
        k = split_heads(k, self.n_head)
        v = split_heads(v, self.n_head)

        q, k = enc(q, k)

        # Normalize q and k onto the unit hypersphere (per head). RoPE is a
        # rotation, so it commutes with normalization.
        q = normalize(q)
        k = normalize(k)

        att = (q @ k.swapaxes(-2, -1)) * self.s_qk()
        att = jnp.where(jnp.tril(jnp.ones((T, T), bool)), att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1)
        y = att @ v

        y = merge_heads(y)
        return self.proj(y)


class MLP(eqx.Module):
    su_base: float = eqx.field(static=True)

    fc: Linear
    proj: Linear
    s_u: Scale

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        # Bias-free to keep activations on the unit hypersphere.
        fc_key, proj_key = jr.split(key)
        self.fc = Linear(config.n_embd, config.n_ff, key=fc_key, use_bias=False)
        self.proj = Linear(config.n_ff, config.n_embd, key=proj_key, use_bias=False)
        # With unit-norm input and unit-norm weights the pre-activations would be
        # ~1/√d — far too small, leaving GELU near-linear. Scale the up-projection
        # by a √n_embd baseline (times learnable s_u) so GELU sees O(1) inputs.
        self.s_u = Scale(1, init=1.0, scale=1.0)
        self.su_base = config.n_embd**0.5

    def __call__(self, h):
        u = self.fc(h) * (self.s_u() * self.su_base)
        return self.proj(jax.nn.gelu(u, approximate=False))


class Block(eqx.Module):
    alpha: float = eqx.field(static=True)

    attn: CausalSelfAttention
    mlp: MLP
    s_attn: Scale | None
    s_mlp: Scale | None

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        attn_key, mlp_key = jr.split(key)
        self.attn = CausalSelfAttention(config, key=attn_key)
        self.mlp = MLP(config, key=mlp_key)
        # Residual step size: alpha = n_layer ** -exp (exp=1 → 1/n_layer). Because
        # the residual normalizes the sub-module output first, alpha is the true
        # interpolation fraction and the per-layer rotation is ~alpha, holding the
        # stack's travel O(1) regardless of width.
        self.alpha = config.n_layer**-config.residual_alpha_exp
        if config.learnable_alpha:
            self.s_attn = Scale(1, init=self.alpha, scale=1.0)
            self.s_mlp = Scale(1, init=self.alpha, scale=1.0)
        else:
            self.s_attn = None
            self.s_mlp = None

    def _step(self, h, sublayer_out, s: Scale | None):
        """One residual update: a LERP toward the *normalized* sublayer output, re-projected.

        Normalizing the target makes alpha the true interpolation fraction, so the
        step is independent of the raw output's magnitude (~√n_embd for the MLP).
        """
        alpha = self.alpha if s is None else s()
        return normalize(h + alpha * (normalize(sublayer_out) - h))

    def __call__(self, h, enc: RotaryEncoding):
        # h is on the unit hypersphere; each sub-module consumes it directly.
        h = self._step(h, self.attn(h, enc), self.s_attn)
        h = self._step(h, self.mlp(h), self.s_mlp)
        return h


class Transformer(eqx.Module):
    wte: Float[Array, "V C"]
    blocks: tuple[Block, ...]
    rotary_enc: RotaryEncoding

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        wte_key, *block_keys = jr.split(key, config.n_layer + 1)
        # Initialized like the (tied) LM head it doubles as; immediately
        # re-projected onto the sphere by `normalize_weights` below.
        lim = config.n_embd**-0.5
        self.wte = jr.uniform(wte_key, (config.vocab_size, config.n_embd), minval=-lim, maxval=lim)
        self.blocks = tuple(Block(config, key=k) for k in block_keys)
        self.rotary_enc = RotaryEncoding(config.n_head_dim)


class NGPT(LanguageModel):
    transformer: Transformer
    s_z: Scale

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray):
        log.info("Initializing nGPT model with config: %s", config)
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.transformer = Transformer(config, key=key)
        # Learnable scalar logit temperature (the hidden state is unit-norm, so
        # raw logits would be cosines in [−1, 1]).
        self.s_z = Scale(1, init=1.0, scale=config.n_embd**-0.5)

        # Start every matrix on the unit hypersphere.
        normalized = self.normalize_weights()
        self.transformer = normalized.transformer

        log.info("number of parameters: %.2fM", self.get_num_params() / 1e6)

    def __call__(self, idx: Int[Array, "B T"], *, key: PRNGKeyArray | None = None):
        # Token embeddings, projected onto the unit hypersphere. (The key is
        # unused: nGPT is dropout-free — the hypersphere constraint regularizes.)
        x: Float[Array, "B T C"] = normalize(self.transformer.wte[idx])

        # Gradient-checkpoint each block: the backward pass recomputes
        # activations instead of storing every layer's O(T²) attention maps.
        enc = self.transformer.rotary_enc
        run_block = eqx.filter_checkpoint(lambda block, h: block(h, enc))
        for block in self.transformer.blocks:
            x = run_block(block, x)

        # Hidden state is already normalized, so just project (tied LM head) and
        # apply the learnable logit temperature.
        return (x @ self.transformer.wte.T) * self.s_z()

    def residual_stream(self, idx: Int[Array, "B T"]) -> Float[Array, "L1 B T C"]:
        """The residual stream at every depth: the embedding plus the state after
        each block (n_layer + 1 slices, all unit-norm).

        The probing/anchoring readout for the M2 experiments. Mirrors
        ``__call__`` without gradient checkpointing — intended for small eval
        batches, not the training path.
        """
        x = normalize(self.transformer.wte[idx])
        states = [x]
        for block in self.transformer.blocks:
            x = block(x, self.transformer.rotary_enc)
            states.append(x)
        return jnp.stack(states)

    def normalize_weights(self) -> "NGPT":
        """Project every hidden-dim matrix back onto the unit hypersphere.

        Apply after each optimizer step to enforce nGPT's weight constraint:
        `model = model.normalize_weights()`. Matrices that read from the
        residual stream are normalized over their input axis (axis=1); matrices
        that write to it, over their output axis (axis=0). The LM head shares
        the embedding array, so it is covered once.
        """

        def where(m: NGPT):
            return (
                [m.transformer.wte]
                + [b.attn.qkv.weight for b in m.transformer.blocks]
                + [b.attn.proj.weight for b in m.transformer.blocks]
                + [b.mlp.fc.weight for b in m.transformer.blocks]
                + [b.mlp.proj.weight for b in m.transformer.blocks]
            )  # fmt: skip

        def replacements(m: NGPT):
            return (
                [normalize(m.transformer.wte, axis=1)]
                + [normalize(b.attn.qkv.weight, axis=1) for b in m.transformer.blocks]
                + [normalize(b.attn.proj.weight, axis=0) for b in m.transformer.blocks]
                + [normalize(b.mlp.fc.weight, axis=1) for b in m.transformer.blocks]
                + [normalize(b.mlp.proj.weight, axis=0) for b in m.transformer.blocks]
            )  # fmt: skip

        return eqx.tree_at(where, self, replacements(self))

    def scale_report(self) -> dict[str, list[float] | float]:
        """Read back the learned scalar temperatures, per layer.

        These are all single numbers, so we can see exactly what training
        settled on. The residual step is a fixed constant unless
        ``learnable_alpha`` is set, in which case its learned gains appear too.
        """
        blocks = self.transformer.blocks
        report: dict[str, list[float] | float] = {
            "s_qk": [float(jnp.mean(b.attn.s_qk())) for b in blocks],
            "s_u": [float(jnp.mean(b.mlp.s_u())) for b in blocks],
            "s_z": float(jnp.mean(self.s_z())),
        }
        # Residual step gains, present only when learnable_alpha is set.
        if blocks and blocks[0].s_attn is not None:
            report["alpha_attn"] = [float(jnp.mean(s())) for b in blocks if (s := b.s_attn) is not None]
            report["alpha_mlp"] = [float(jnp.mean(s())) for b in blocks if (s := b.s_mlp) is not None]
        return report
