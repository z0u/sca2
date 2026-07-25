"""The step path is a drawing convention, so pin what each of its gap kinds does to the ink."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from mini.vis.plt import smooth_step

Y = [0.0, 1.0, 0.5, 0.5]


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def subpaths(patch) -> int:
    """How many disconnected strokes the patch draws — one more than the gaps in it."""
    return int(np.sum(patch.get_path().codes == MplPath.MOVETO))


def test_plain_step_is_one_unbroken_stroke(ax):
    patch = smooth_step(ax, range(len(Y)), Y)
    assert subpaths(patch) == 1
    assert len(ax.patches) == 1


def test_breaks_split_the_stroke_and_add_no_ink(ax):
    patch = smooth_step(ax, range(len(Y)), Y, breaks=[1])
    assert subpaths(patch) == 2
    assert len(ax.patches) == 1


def test_elide_lifts_the_riser_onto_a_fainter_companion(ax):
    patch = smooth_step(ax, range(len(Y)), Y, elide=[1], fade=0.25, alpha=0.8)
    ghost, solid = ax.patches
    assert solid is patch
    assert subpaths(solid) == 2  # the elided riser is gone from the solid stroke...
    assert subpaths(ghost) == 1  # ...and only the ghost still carries it
    assert ghost.get_alpha() == pytest.approx(0.2)  # fade × the line's own alpha


def test_a_break_beats_an_elision_on_the_same_riser(ax):
    """Both say "these two samples are not neighbours"; the break says draw nothing at all."""
    patch = smooth_step(ax, range(len(Y)), Y, breaks=[1], elide=[1])
    assert subpaths(patch) == 2
    assert len(ax.patches) == 1


def test_elided_ghost_spans_exactly_the_riser_the_solid_stroke_dropped(ax):
    ramp = 0.5
    solid = smooth_step(ax, range(len(Y)), Y, ramp=ramp, elide=[1])
    ghost = ax.patches[0]
    # Riser 1 runs from x=1.5-ramp/2 to x=1.5+ramp/2. Both strokes reach its shoulders; only
    # the ghost carries the control points at the crossover, which is the riser itself.
    for xs in (solid.get_path().vertices[:, 0], ghost.get_path().vertices[:, 0]):
        assert np.isclose(xs, 1.5 - ramp / 2).any() and np.isclose(xs, 1.5 + ramp / 2).any()
    assert np.isclose(solid.get_path().vertices[:, 0], 1.5).sum() == 0
    assert np.isclose(ghost.get_path().vertices[:, 0], 1.5).sum() == 2
