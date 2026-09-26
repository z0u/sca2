"""Grid distances between a model's answer distribution and a line's raw answer."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from sca import answer_distance as ad
from sca.data.ops import colors

BLACK, RED, WHITE, GREY = (0, 0, 0), (15, 0, 0), (15, 15, 15), (6, 6, 6)
IDX = {c: i for i, c in enumerate(colors())}


def mass(*entries: tuple[tuple[int, int, int], float]) -> np.ndarray:
    """One line of color mass, `(1, 216)`, with the given colors at the given probabilities."""
    p = np.zeros((1, 216))
    for c, v in entries:
        p[0, IDX[c]] = v
    return p


def test_grid_is_in_steps_in_palette_order():
    assert ad.GRID.shape == (216, 3)
    np.testing.assert_allclose(ad.GRID[IDX[WHITE]], [5, 5, 5], rtol=0, atol=0)
    np.testing.assert_allclose(ad.GRID[IDX[GREY]], [2, 2, 2], rtol=0, atol=0)


def test_raw_answer_is_the_expectation_of_the_answer_distribution():
    q_idx = np.array([[IDX[BLACK], IDX[RED], -1], [IDX[GREY], -1, -1]])
    q_p = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(ad.raw_answer(q_idx, q_p), [[2.5, 0, 0], [2, 2, 2]], rtol=0, atol=1e-12)


def test_central_points_ignore_off_vocabulary_mass():
    p = mass((BLACK, 0.2), (RED, 0.6))  # 0.2 off the grid
    np.testing.assert_allclose(ad.greedy(p), [[5, 0, 0]], rtol=0, atol=0)
    np.testing.assert_allclose(ad.mean(p), [[3.75, 0, 0]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ad.median(p), [[5, 0, 0]], rtol=0, atol=0)


@pytest.mark.parametrize(("p_black", "expected"), [(0.4, [5, 0, 0]), (0.5, [0, 0, 0]), (0.6, [0, 0, 0])])
def test_median_takes_the_lowest_level_at_half_mass(p_black, expected):
    p = mass((BLACK, p_black), (RED, 1 - p_black))
    np.testing.assert_allclose(ad.median(p), [expected], rtol=0, atol=0)


def test_expected_distance_averages_over_the_distribution():
    p = mass((BLACK, 0.5), (RED, 0.5))
    target = np.array([[0.0, 0.0, 0.0]])
    assert ad.expected_distance(p, target) == approx([2.5])
    assert ad.distance(ad.mean(p), target) == approx([2.5])


def test_floors_are_zero_on_the_grid_and_the_box_gap_off_it():
    q_idx = np.array([[IDX[BLACK], IDX[RED]], [IDX[GREY], -1]])
    q_p = np.array([[0.5, 0.5], [1.0, 0.0]])
    f = ad.floors(q_idx, q_p)
    assert {k: v.tolist() for k, v in f.items()} == approx({"mode": [2.5, 0.0], "draw": [2.5, 0.0]})


def test_chance_is_symmetric_and_lowest_at_the_center():
    corners = ad.chance(np.array([[0.0, 0, 0], [5.0, 5, 5], [2.5, 2.5, 2.5]]))
    assert corners[0] == approx(corners[1])
    assert corners[2] < corners[0]
    assert ad.chance(ad.GRID).mean() == approx(ad.CHANCE)


def test_no_color_mass_gives_nan():
    p = np.zeros((1, 216))
    assert np.isnan(ad.mean(p)).all()
    assert np.isnan(ad.median(p)).all()
    assert np.isnan(ad.expected_distance(p, np.zeros((1, 3)))).all()
