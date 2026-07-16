"""The RGB-cube projection: orthonormal, value up the black→white diagonal."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sca.data import cube
from sca.data.colors import N_LEVELS, PALETTE


@pytest.mark.parametrize("side", ["front", "back"])
def test_basis_is_orthonormal(side: cube.Side):
    b = cube._basis(side)
    assert np.allclose(b.T @ b, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("side", ["front", "back"])
def test_value_axis_is_the_diagonal(side: cube.Side):
    rgb = cube.grid()
    _, value, _ = cube.project(rgb, side)
    # The vertical coordinate is the projection onto the black→white diagonal.
    assert np.allclose(value, rgb @ cube._DIAG)
    assert np.argmin(value) == 0  # black (0,0,0) at the bottom
    assert np.argmax(value) == len(rgb) - 1  # white (1,1,1) at the top


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


def test_draw_hides_axis_furniture():
    fig, ax = plt.subplots()
    pc = cube.draw_rgb_cube(ax, cube.named(), side="front")
    assert np.asarray(pc.get_offsets()).shape == (len(PALETTE), 2)
    assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0
    assert not any(s.get_visible() for s in ax.spines.values())
    plt.close(fig)
