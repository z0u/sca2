"""Jitted step functions for training and evaluation.

This is what PyTorch Lightning used to provide; in JAX it's a handful of pure
functions. The model is a pytree, so one training step is: differentiate the
loss, apply the optimizer update, then let the model re-impose its weight
constraint (nGPT's hypersphere projection).
"""

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree

from experiment.model import LanguageModel


def cross_entropy(logits: Float[Array, "B T V"], targets: Int[Array, "B T"]) -> Float[Array, ""]:
    """Mean cross-entropy over non-padding tokens (padding id 0)."""
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    mask = targets != 0
    return jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1)


def loss_fn(model: LanguageModel, x: Int[Array, "B T"], y: Int[Array, "B T"]) -> Float[Array, ""]:
    return cross_entropy(model(x), y)


def make_train_step(optimizer: optax.GradientTransformation):
    """Build a jitted training step closed over *optimizer*."""

    @eqx.filter_jit
    def train_step(
        model: LanguageModel,
        opt_state: PyTree,
        x: Int[Array, "B T"],
        y: Int[Array, "B T"],
    ) -> tuple[LanguageModel, PyTree, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        # Re-project weights onto the unit hypersphere (nGPT's weight constraint).
        model = model.normalize_weights()
        return model, opt_state, loss

    return train_step


@eqx.filter_jit
def eval_step(model: LanguageModel, x: Int[Array, "B T"], y: Int[Array, "B T"]) -> Float[Array, ""]:
    """Validation loss."""
    return loss_fn(model, x, y)
