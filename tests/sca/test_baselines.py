import numpy as np
import pytest

from sca import baselines as bl
from sca.data.colors import mix
from sca.data.named_colors import GRIDS, grid_palette

V27 = GRIDS["v27"]
RGB27 = np.array(list(grid_palette(V27).values()))


def closed_pairs(levels):
    """Every distinct pair of grid colors whose mix lands back on the grid."""
    pts = [(r, g, b) for r in levels for g in levels for b in levels]
    return [(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :] if all(c in levels for c in mix(a, b))]


def test_shell_mask_counts_level_steps_not_rgb_distance():
    # v27's steps are uneven: 0->8 spans 8, 8->15 spans 7. Both are one step.
    shell = bl.shell_mask(V27, RGB27, [(8, 8, 8)])
    assert RGB27[shell[0]].tolist() == sorted([[0, 8, 8], [15, 8, 8], [8, 0, 8], [8, 15, 8], [8, 8, 0], [8, 8, 15]]), (
        "middle color has two neighbors per channel"
    )

    # A corner has one neighbor per channel, so half the shell.
    assert bl.shell_mask(V27, RGB27, [(0, 0, 0)])[0].sum() == 3


def test_shell_size_range_on_v27():
    """The claim the reports lean on: v27 shells hold 4 to 6 names."""
    mixes = [mix(a, b) for a, b in closed_pairs(V27)]
    sizes = bl.shell_mask(V27, RGB27, mixes).sum(axis=1)
    assert sizes.min() == 4 and sizes.max() == 6


def test_neighborhood_exact_null_is_one_over_closed_neighborhood():
    shell = bl.shell_mask(V27, RGB27, [(8, 8, 8), (0, 0, 0)])
    # 1/(1+6) and 1/(1+3), averaged.
    assert bl.neighborhood_exact_null(shell) == pytest.approx((1 / 7 + 1 / 4) / 2)


def test_operands_of_a_closed_pair_lie_in_the_shell():
    """Closure puts both operands one level from the mix in each disagreeing channel."""
    pairs = [(a, b) for a, b in closed_pairs(V27) if sum(x != y for x, y in zip(a, b, strict=True)) == 1]
    assert pairs, "single-channel closed pairs exist"
    shell = bl.shell_mask(V27, RGB27, [mix(a, b) for a, b in pairs])
    for row, (a, b) in zip(shell, pairs, strict=True):
        names = RGB27[row].tolist()
        assert list(a) in names and list(b) in names


def test_operand_shell_null_matches_hand_count():
    # red + blue = purple. The operands differ from the mix (8,0,8) in two channels
    # each, so neither is in the one-step shell and the null is zero.
    pair = ((15, 0, 0), (0, 0, 15))
    shell = bl.shell_mask(V27, RGB27, [mix(*pair)])
    assert bl.operand_shell_null(shell, RGB27, [pair]) == 0.0

    # A single-channel pair: black + (15,0,0) = (8,0,0). Shell holds 4 names, both operands.
    pair = ((0, 0, 0), (15, 0, 0))
    shell = bl.shell_mask(V27, RGB27, [mix(*pair)])
    assert shell[0].sum() == 4
    assert bl.operand_shell_null(shell, RGB27, [pair]) == pytest.approx(0.5)


def test_k_nearest_stats_k1_is_the_floor():
    d = bl.distances(RGB27, [(8, 8, 8), (0, 0, 0), (4, 4, 4)])
    assert bl.k_nearest_stats(d, 1)["dist"] == pytest.approx(d.min(axis=1).mean())
    assert bl.k_nearest_stats(d, 1)["nearest"] == 1.0


def test_k_nearest_stats_degrades_with_k():
    # Off-lattice targets with no tied nearest name (a channel at 4 ties 0 against 8).
    d = bl.distances(RGB27, [(3, 5, 9), (12, 2, 6)])
    stats = [bl.k_nearest_stats(d, k) for k in (1, 2, 4)]
    assert stats[0]["dist"] < stats[1]["dist"] < stats[2]["dist"]
    assert stats[0]["nearest"] > stats[1]["nearest"] > stats[2]["nearest"]


def test_k_nearest_stats_keeps_ties_at_the_floor():
    """A target equidistant from several names has more than one 'nearest'."""
    d = bl.distances(RGB27, [(4, 4, 4)])
    assert bl.k_nearest_stats(d, 2)["nearest"] == 1.0


def test_blind_index_picks_the_center_for_mixes():
    """Mixes cluster centrally, so the best prompt-blind answer is the middle name."""
    mixes = [mix(a, b) for a, b in closed_pairs(V27)]
    i = bl.blind_index(bl.distances(RGB27, mixes))
    assert RGB27[i].tolist() == [8, 8, 8]


def test_blind_stats_beats_uniform_chance_by_a_wide_margin():
    mixes = [mix(a, b) for a, b in closed_pairs(V27)]
    d = bl.distances(RGB27, mixes)
    blind = bl.blind_stats(d, bl.blind_index(d))["dist"]
    assert blind < 0.6 * d.mean(), "a constant answer beats chance, so chance is a weak reference"


def test_self_nearest_rate_is_one_for_exact_positions():
    exact = np.asarray(RGB27, dtype=np.float32) / 15
    assert bl.self_nearest_rate(RGB27, exact) == 1.0


def test_self_nearest_rate_falls_before_a_full_cell_of_error():
    """Displacement of 0.6 of a step is under one cell but past the Voronoi boundary."""
    step = 8 / 15
    shifted = np.asarray(RGB27, dtype=np.float32) / 15
    shifted[:, 0] = np.clip(shifted[:, 0] + 0.6 * step, 0, 1)
    assert bl.self_nearest_rate(RGB27, shifted) < 0.5
