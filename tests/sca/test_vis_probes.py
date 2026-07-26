"""The landmark axis is a figure convention shared across reports, so pin what it promises."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

from sca.data.mixed_vocab import GAP_RISERS, LANDMARKS, SPAN_RISERS
from sca.vis_probes import (
    ELIDED_RISERS,
    FORMS,
    LANDMARK_LABELS,
    MAJOR_LANDMARKS,
    MINOR_LANDMARKS,
    TARGETS,
    aliased_risers,
    mean_color,
    probe_trace_grid,
)

N_DEPTH = 5


@pytest.fixture
def stacks():
    rng = np.random.default_rng(0)
    return {(f, t): rng.random((N_DEPTH, len(LANDMARKS), 3)) for f in FORMS for t in TARGETS}


def test_every_landmark_is_labelled_exactly_once():
    assert set(LANDMARK_LABELS) == set(LANDMARKS)
    assert set(MAJOR_LANDMARKS) | set(MINOR_LANDMARKS) == set(LANDMARKS)
    assert not set(MAJOR_LANDMARKS) & set(MINOR_LANDMARKS)


def test_named_elides_everything_hex_does_and_its_word_middles_besides():
    assert ELIDED_RISERS["hex"] == GAP_RISERS
    assert ELIDED_RISERS["named"] == GAP_RISERS | SPAN_RISERS
    assert ELIDED_RISERS["named"] > ELIDED_RISERS["hex"]


def test_aliased_risers_finds_landmarks_that_resolved_to_one_character():
    stack = np.arange(N_DEPTH * len(LANDMARKS) * 3, dtype=float).reshape(N_DEPTH, len(LANDMARKS), 3)
    assert aliased_risers(stack) == frozenset()
    stack[:, 3] = stack[:, 2]  # landmarks 2 and 3 now measure the same position
    assert aliased_risers(stack) == frozenset({2})


@pytest.mark.parametrize(
    "form_axis,targets,shape",
    [
        ("row", TARGETS, (2, 3)),  # the heatmap's arrangement: forms stacked, targets across
        ("col", ("mix",), (1, 2)),  # one target, wide: a column per form
    ],
)
def test_grid_gives_every_form_target_pair_a_stack_of_depth_panels(stacks, form_axis, targets, shape):
    fig = probe_trace_grid(stacks, targets=targets, form_axis=form_axis)
    assert len(fig.subfigs) == shape[0] * shape[1]
    assert all(len(sfig.axes) == N_DEPTH for sfig in fig.subfigs)
    assert len(fig.get_axes()) == shape[0] * shape[1] * N_DEPTH
    plt.close(fig)


def test_only_the_bottom_row_of_blocks_is_labelled(stacks):
    fig = probe_trace_grid(stacks, form_axis="row")
    labelled = [{t.get_text() for t in sfig.axes[-1].get_xticklabels()} != {""} for sfig in fig.subfigs]
    assert labelled == [False, False, False, True, True, True]
    plt.close(fig)


def test_a_seed_axis_adds_the_envelope_without_changing_the_layout(stacks):
    per_seed = {k: np.stack([v, v * 0.5, v * 0.8]) for k, v in stacks.items()}
    plain = probe_trace_grid(stacks, form_axis="row")
    shaded = probe_trace_grid(per_seed, form_axis="row")
    assert len(shaded.get_axes()) == len(plain.get_axes())
    # Same panels, six extra patches each: the fill's outline and the two envelope hairlines,
    # each with the faded companion that carries its elided risers (both forms elide some).
    counts = [len(a.patches) for a in plain.get_axes()], [len(a.patches) for a in shaded.get_axes()]
    assert all(s == p + 6 for p, s in zip(*counts, strict=True))
    plt.close(plain)
    plt.close(shaded)


def test_the_summary_outranks_the_envelope_it_sits_inside():
    """Ink follows the hierarchy: the fill's own silhouette reads first, the spread second."""
    values = np.linspace(0.2, 0.8, len(LANDMARKS))[:, None] * np.ones(3)
    per_seed = np.stack([values * 0.75, values, values * 1.1])[:, None]  # one depth
    # Nothing elided, so each shading line is a single patch rather than a solid-plus-ghost pair.
    fig = probe_trace_grid({(f, "mix"): per_seed for f in FORMS}, targets=("mix",), elided={})
    ax = fig.get_axes()[0]
    grey = [p for p in ax.patches if p.get_edgecolor()[:3] == to_rgb(mean_color())]
    fills = [p for p in grey if p.get_fill()]
    assert len(fills) == 1 and np.isclose(fills[0].get_path().vertices[:, 1].min(), 0.0)
    # Seed minimum, seed mean (what the fill and the lines plot), seed maximum, at the top landmark.
    edges = sorted((p for p in grey if not p.get_fill()), key=lambda p: p.get_path().vertices[:, 1].max())
    assert np.allclose(
        [p.get_path().vertices[:, 1].max() for p in edges],
        [0.8 * f for f in (0.75, (0.75 + 1 + 1.1) / 3, 1.1)],
    )
    lo, summary, hi = (p.get_alpha() for p in edges)
    assert summary > hi and summary > lo and lo == hi  # the envelope is a matched, quieter pair
    plt.close(fig)


def test_sub_zero_scores_are_floored_for_drawing_but_not_for_alias_detection():
    """Clipping first would make every failed probe identically 0, and so falsely aliased."""
    flat_negative = np.full((N_DEPTH, len(LANDMARKS), 3), -0.4)
    assert aliased_risers(flat_negative) == frozenset(range(len(LANDMARKS) - 1))
    fig = probe_trace_grid({(f, "mix"): flat_negative for f in FORMS}, targets=("mix",))
    ax = fig.get_axes()[0]
    ys = np.concatenate([p.get_path().vertices[:, 1] for p in ax.patches])
    assert ys.min() >= 0.0  # drawn at the floor, not below the panel
    plt.close(fig)
