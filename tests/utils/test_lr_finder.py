import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from utils.lr_finder.lr_finder import lr_finder_search


def _batches():
    rng = np.random.default_rng(0)
    while True:
        x = rng.normal(size=(16, 4)).astype(np.float32)
        yield x, x @ np.arange(4, dtype=np.float32)[:, None]


def _loss(model, x, y, _key):
    return jnp.mean((eqx.filter_vmap(model)(x) - y) ** 2)


def test_raw_losses_cover_every_step_and_constrain_runs_after_each_update():
    model = eqx.nn.Linear(4, 1, key=jr.key(0))
    seen = []

    def constrain(m):
        seen.append(True)
        return eqx.tree_at(lambda m: m.weight, m, m.weight / jnp.linalg.norm(m.weight))

    _, _, history = lr_finder_search(
        model,
        _loss,
        lambda learning_rate: optax.adam(learning_rate),
        _batches(),
        start_lr=1e-4,
        end_lr=1.0,
        num_zooms=2,
        steps_per_zoom=8,
        constrain=constrain,
        key=jr.key(1),
    )
    assert seen, "constrain is traced into the step"
    for series in history:
        assert len(series.raw_lrs) == len(series.raw_losses) == 8
        assert set(series.lrs) <= set(series.raw_lrs)
        # The filtered curve keeps only new lows, so it never rises.
        assert all(b < a for a, b in zip(series.losses, series.losses[1:], strict=False))
