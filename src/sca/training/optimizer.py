import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sca.config import OptimizerConfig


def configure_optimizer(
    model: eqx.Module,
    config: OptimizerConfig,
    learning_rate: float | optax.Schedule,
) -> optax.GradientTransformation:
    # apply weight decay to weight matrices (2+D arrays) but not to biases/gains (1-D or 0-D)
    params = eqx.filter(model, eqx.is_inexact_array)
    decay_mask = jax.tree.map(lambda p: p.ndim >= 2, params)

    optimizer = optax.adamw(
        learning_rate,
        b1=config.betas[0],
        b2=config.betas[1],
        weight_decay=config.weight_decay,
        mask=decay_mask,
    )
    if config.basis_seed is None:
        return optimizer
    (width,) = {
        leaf.shape[-1]
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
        if ".wte" in jax.tree_util.keystr(path)
    }
    basis = np.linalg.qr(np.random.default_rng(config.basis_seed).standard_normal((width, width)))[0]
    return rotated(optimizer, basis)


def rotated(inner: optax.GradientTransformation, basis: np.ndarray) -> optax.GradientTransformation:
    """*inner* applied in the residual basis *basis* (an orthogonal matrix, one basis vector per column).

    Adam's per-coordinate scaling is not rotation-equivariant: whatever basis it is written in becomes a privileged
    basis of the residual stream (the memo of 2026-10-23 in the project files, after Elhage et al. 2023). Rotating
    the gradients of every matrix that reads or writes the stream into *basis*, updating there, and rotating the
    update back moves that privilege from the coordinate axes to the columns of *basis*, with the forward pass, the
    initialization, and the batches untouched. Matrices that read the stream (the tables, qkv, and the MLP
    up-projection) are rotated on their input axis; matrices that write it (the two down-projections) on their
    output axis; gains and the optimizer's other state are left alone. The weight normalization after each step is
    rotation-invariant, so nothing else changes.
    """
    R = jnp.asarray(basis, dtype=jnp.float32)

    def kind(path) -> str | None:
        s = jax.tree_util.keystr(path)
        if any(k in s for k in (".wte", ".lm_head", ".qkv.weight", ".fc.weight")):
            return "read"
        if ".proj.weight" in s:
            return "write"
        return None

    def rotate(tree, inverse: bool = False):
        M = R.T if inverse else R

        def one(path, g):
            match kind(path):
                case "read":
                    return g @ M
                case "write":
                    return M.T @ g
                case _:
                    return g

        return jax.tree_util.tree_map_with_path(one, tree)

    def init(params):
        return inner.init(rotate(params))

    def update(updates, state, params=None):
        u, state = inner.update(rotate(updates), state, None if params is None else rotate(params))
        return rotate(u, inverse=True), state

    return optax.GradientTransformation(init, update)
