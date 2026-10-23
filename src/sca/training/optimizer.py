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

    b1, b2 = config.betas
    match config.kind:
        case "adam":
            optimizer = optax.adamw(learning_rate, b1=b1, b2=b2, weight_decay=config.weight_decay, mask=decay_mask)
        case "adam-pooled":
            assert config.weight_decay == 0, "pooled Adam has no weight decay"
            optimizer = pooled_adam(learning_rate, b1=b1, b2=b2)
        case "muon":
            assert config.weight_decay == 0, "Muon here has no weight decay"
            if config.aux_learning_rate is None:
                aux = learning_rate
            elif callable(learning_rate):
                # The gains follow the schedule's shape at their own peak.
                ratio = config.aux_learning_rate / config.learning_rate
                schedule = learning_rate
                aux = lambda count: jnp.asarray(schedule(count)) * ratio  # noqa: E731
            else:
                aux = config.aux_learning_rate
            optimizer = optax.contrib.muon(learning_rate, adam_learning_rate=aux, adam_b1=b1, adam_b2=b2)
        case "sgd":
            optimizer = optax.sgd(learning_rate, momentum=b1)
    if config.basis_seed is None:
        return optimizer
    (width,) = {
        leaf.shape[-1]
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
        if ".wte" in jax.tree_util.keystr(path)
    }
    basis = np.linalg.qr(np.random.default_rng(config.basis_seed).standard_normal((width, width)))[0]
    return rotated(optimizer, basis)


def stream_axis(path) -> int | None:
    """The axis of a parameter that runs along the residual coordinates: the last axis of a matrix that reads the
    stream (the tables, qkv, and the MLP up-projection), the first of one that writes it (the two down-projections),
    `None` for a gain.
    """
    s = jax.tree_util.keystr(path)
    if any(k in s for k in (".wte", ".lm_head", ".qkv.weight", ".fc.weight")):
        return -1
    if ".proj.weight" in s:
        return 0
    return None


def pooled_adam(
    learning_rate: float | optax.Schedule, b1: float, b2: float, eps: float = 1e-8
) -> optax.GradientTransformation:
    """Adam with the second moment pooled over the residual axis of each matrix that reads or writes the stream.

    Adam divides each coordinate's update by the root of that coordinate's own running mean square gradient, which is
    what writes its basis into the model. Pooling the mean square over the residual axis leaves one scale per row of
    a reading matrix (per output unit) or per column of a writing matrix (per input unit), so the scaling is the
    same whichever orthonormal basis the residual coordinates are written in, and the optimizer has no privileged
    basis there. The gains, and the other axis of each matrix, are scaled per coordinate as in Adam. Bias
    correction is Adam's.
    """

    def init(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return {"count": jnp.zeros((), jnp.int32), "m": zeros, "v": zeros}

    def update(updates, state, params=None):
        count = state["count"] + 1
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], updates)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], updates)
        lr = jnp.asarray(learning_rate(count) if callable(learning_rate) else learning_rate)
        c1, c2 = 1 - b1**count, 1 - b2**count

        def step(path, m, v):
            axis = stream_axis(path)
            if axis is not None:
                v = jnp.mean(v, axis=axis, keepdims=True)
            return -lr * (m / c1) / (jnp.sqrt(v / c2) + eps)

        return jax.tree_util.tree_map_with_path(step, m, v), {"count": count, "m": m, "v": v}

    return optax.GradientTransformation(init, update)


def rotated(inner: optax.GradientTransformation, basis: np.ndarray) -> optax.GradientTransformation:
    """*inner* applied in the residual basis *basis* (an orthogonal matrix, one basis vector per column).

    Adam's per-coordinate scaling is not rotation-equivariant: whatever basis it is written in becomes a privileged
    basis of the residual stream (the memo of 2026-10-23 in the project files, after Elhage et al. 2023). Rotating
    the gradients of every matrix that reads or writes the stream into *basis*, updating there, and rotating the
    update back moves that privilege from the coordinate axes to the columns of *basis*, with the forward pass, the
    initialization, and the batches untouched. Matrices that read the stream (the tables, qkv, and the MLP
    up-projection) are rotated on their input axis; matrices that write it (the two down-projections) on their
    output axis; gains and the optimizer's other state are left alone. The weight normalization after each step is
    rotation-invariant, so nothing else changes. Applied to an optimizer with no privileged basis (`pooled_adam`,
    Muon, SGD) it changes nothing, which is what the tests check.
    """
    R = jnp.asarray(basis, dtype=jnp.float32)

    def rotate(tree, inverse: bool = False):
        M = R.T if inverse else R

        def one(path, g):
            match stream_axis(path):
                case -1:
                    return g @ M
                case 0:
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
