"""The cube projections are figure conventions, so pin their geometry."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sca.data import cube
from sca.vis import CUBE_VIEWS, align_to_cube, grid_diameter, plot_rgb_cube, project_cube

VIEWS = list(CUBE_VIEWS)
SOLIDS = [v for v in VIEWS if v.startswith("solid")]


def test_solid_stands_on_black_with_white_up_and_red_toward_the_reader():
    corners = dict(K=(0, 0, 0), W=(1, 1, 1), R=(1, 0, 0), G=(0, 1, 0), B=(0, 0, 1))
    xy = {k: project_cube(np.array(c, float)) for k, c in corners.items()}
    assert np.allclose(xy["W"], [0, 1]) and np.allclose(xy["K"], [0, -1])
    assert xy["G"][0] < 0 < xy["B"][0]  # green left, blue right
    assert np.isclose(xy["R"][0], 0)  # red faces the reader, so it sits on the centerline
    # ...and being the view direction, a (2, −1, −1) step is invisible.
    assert np.allclose(np.ptp(project_cube(np.array([[1.0, 0.5, 0.5], [0.0, 1.0, 1.0]])), axis=0), 0)


def test_the_back_view_mirrors_the_front_and_faces_cyan():
    rgb = cube.grid(4)
    front, back = project_cube(rgb, "solid"), project_cube(rgb, "solid-back")
    assert np.allclose(front[:, 1], back[:, 1])  # same lightness, since both stand on black
    assert np.allclose(front[:, 0], -back[:, 0])  # mirrored horizontally
    assert np.allclose(CUBE_VIEWS["solid-back"].toward, -CUBE_VIEWS["solid"].toward)


@pytest.mark.parametrize("view", SOLIDS)
def test_the_solid_views_put_value_on_the_vertical_axis(view):
    rgb = cube.grid(4)
    value = project_cube(rgb, view)[:, 1]
    assert np.allclose(value, (rgb - 0.5) @ np.ones(3) * (2 / 3))  # the black→white diagonal
    assert np.argmin(value) == 0  # black (0, 0, 0) at the bottom
    assert np.argmax(value) == len(rgb) - 1  # white (1, 1, 1) at the top


@pytest.mark.parametrize("view", VIEWS)
def test_the_panel_carries_no_axis_furniture(view):
    fig, ax = plt.subplots()
    plot_rgb_cube(ax, cube.named(), view=view)
    assert not ax.axison
    assert ax.get_xlim() == (-1.1, 1.1) and ax.get_ylim() == (-1.1, 1.1)
    plt.close(fig)


def test_wheel_puts_the_chromatic_corners_on_a_unit_hexagon_with_red_up():
    xy = project_cube(CUBE_VIEWS["wheel"].rim, "wheel")
    assert np.allclose(np.linalg.norm(xy, axis=1), 1.0)
    angles = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360
    assert np.allclose(angles, [90, 150, 210, 270, 330, 30])  # R, Y, G, C, B, M
    grey = np.linspace(0, 1, 5)[:, None] * np.ones(3)
    assert np.allclose(project_cube(grey, "wheel"), 0.0, atol=1e-12)  # lightness is the view direction


def _outside(hull: np.ndarray, pts: np.ndarray) -> float:
    """Farthest any point strays outside a counter-clockwise convex polygon."""
    edge = np.roll(hull, -1, axis=0) - hull
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)  # outward, given CCW winding
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return float(((pts[:, None] - hull[None]) * normal[None]).sum(-1).max(axis=1).max())


@pytest.mark.parametrize("view", VIEWS)
def test_the_rim_is_wound_counter_clockwise(view):
    hull = project_cube(CUBE_VIEWS[view].rim, view)
    shoelace = np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1])
    assert shoelace > 0


@pytest.mark.parametrize("view", VIEWS)
def test_the_silhouette_contains_every_color(view):
    g = np.linspace(0, 1, 9)
    rgb = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    hull = project_cube(CUBE_VIEWS[view].rim, view)
    assert _outside(hull, project_cube(rgb, view)) < 1e-12


@pytest.mark.parametrize("view", VIEWS)
def test_grid_diameter_covers_the_panel_but_is_not_wasteful(view):
    levels = 8
    g = np.linspace(0, 1, levels)
    rgb = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    xy = np.unique(project_cube(rgb, view).round(9), axis=0)
    d = grid_diameter(levels, view)
    # Sample the panel: every interior point should fall inside some mark.
    probe = xy.mean(0) + (np.random.default_rng(0).random((4000, 2)) - 0.5) * np.ptp(xy, axis=0) * 0.5
    nearest = np.linalg.norm(probe[:, None] - xy[None], axis=2).min(axis=1)
    assert nearest.max() <= d / 2 + 1e-9
    assert nearest.max() > d / 2 * 0.8  # and not by a wide margin, or the marks are oversized


@pytest.mark.parametrize("view", VIEWS)
def test_grid_diameter_shrinks_with_the_grid(view):
    assert grid_diameter(16, view) < grid_diameter(8, view) < grid_diameter(4, view)


def test_align_recovers_a_rotated_scaled_cube_exactly():
    rgb = np.random.default_rng(0).random((40, 3))
    q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))
    mapped, resid = align_to_cube(rgb @ q * 3.7 + 12.0, rgb)
    assert resid < 1e-12
    assert np.allclose(mapped, rgb)


def test_align_reports_a_scale_free_residual():
    rgb = np.random.default_rng(0).random((40, 3))
    noisy = rgb + np.random.default_rng(2).standard_normal((40, 3)) * 0.1
    _, resid = align_to_cube(noisy, rgb)
    _, scaled = align_to_cube(noisy * 100.0, rgb)
    assert 0.0 < resid < 1.0
    assert np.isclose(resid, scaled)
