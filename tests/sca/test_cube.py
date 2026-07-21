"""The RGB grid as data. Its projections and drawing live in sca.vis; see test_vis.py."""

import numpy as np

from sca.data import cube
from sca.data.colors import N_LEVELS, PALETTE


def test_grid_shape_and_range():
    g = cube.grid()
    assert g.shape == (N_LEVELS**3, 3)
    assert g.min() == 0.0 and g.max() == 1.0


def test_named_are_grid_points():
    nm = cube.named()
    assert nm.shape == (len(PALETTE), 3)
    # Named channels are {0, 8, 15}; normalized they must land on grid levels.
    levels = np.round(nm * (N_LEVELS - 1)).astype(int)
    assert set(levels.ravel().tolist()) == {0, 8, 15}
