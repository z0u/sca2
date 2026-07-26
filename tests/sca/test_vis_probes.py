"""The landmark axis is a figure convention shared across reports, so pin what it promises."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgb

from mini.vis import mix, page_color
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


def test_one_panel_numbers_the_shared_scale_and_the_rest_reserve_its_room(stacks):
    fig = probe_trace_grid(stacks, form_axis="row")
    right = lambda ax: [t.label2 for t in ax.yaxis.get_major_ticks() if t.label2.get_visible()]  # noqa: E731
    numbered = [ax for ax in fig.get_axes() if any(t.get_color() != "none" for t in right(ax))]
    assert len(numbered) == 1
    assert {t.get_text() for t in right(numbered[0])} == {"0", "1"}
    # One panel per block claims the room, or the blocks would come out at different widths.
    assert len([ax for ax in fig.get_axes() if right(ax)]) == len(fig.subfigs)
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


def shading(ax):
    """The panel's neutral patches — the channel traces are the chromatic ones."""
    return [p for p in ax.patches if len(set(p.get_edgecolor()[:3])) == 1]


def test_the_summary_outranks_the_envelope_it_sits_inside():
    """Ink follows the hierarchy: the fill's own silhouette reads first, the spread second."""
    values = np.linspace(0.2, 0.8, len(LANDMARKS))[:, None] * np.ones(3)
    per_seed = np.stack([values * 0.75, values, values * 1.1])[:, None]  # one depth
    # Nothing elided, so each shading line is a single patch rather than a solid-plus-ghost pair.
    fig = probe_trace_grid({(f, "mix"): per_seed for f in FORMS}, targets=("mix",), elided={})
    ax = fig.get_axes()[0]
    fills = [p for p in shading(ax) if p.get_fill()]
    assert len(fills) == 1 and np.isclose(fills[0].get_path().vertices[:, 1].min(), 0.0)
    # Seed minimum, seed mean (what the fill and the lines plot), seed maximum, at the top landmark.
    edges = sorted((p for p in shading(ax) if not p.get_fill()), key=lambda p: p.get_path().vertices[:, 1].max())
    assert np.allclose(
        [p.get_path().vertices[:, 1].max() for p in edges],
        [0.8 * f for f in (0.75, (0.75 + 1 + 1.1) / 3, 1.1)],
    )
    lo, summary, hi = edges
    page = to_rgb(page_color())[0]  # one channel is enough; the shading is neutral
    strength = lambda p: abs(p.get_edgecolor()[0] - page)  # noqa: E731 — how far the ink carries
    assert strength(summary) > strength(lo) and strength(summary) > strength(hi)
    plt.close(fig)


def test_lines_over_the_fill_are_tinted_against_it_rather_than_against_the_page():
    """Fading toward the page inside the fill reads as a pale streak scratched across it."""
    values = np.linspace(0.2, 0.8, len(LANDMARKS))[:, None] * np.ones(3)
    per_seed = np.stack([values * 0.75, values, values * 1.1])[:, None]  # one depth
    fig = probe_trace_grid({(f, "mix"): per_seed for f in FORMS}, targets=("mix",))
    ax = fig.get_axes()[0]
    (area,) = (p for p in shading(ax) if p.get_fill())
    face = area.get_facecolor()
    fill = to_rgb(mix(page_color(), to_hex(face), face[3]))[0]  # the shade the fill composites to
    ink = to_rgb(mean_color())[0]
    # The two lowest lines are the minimum's stroke and the faded companion carrying its risers.
    lines = sorted((p for p in shading(ax) if not p.get_fill()), key=lambda p: p.get_path().vertices[:, 1].max())
    assert all(min(fill, ink) < p.get_edgecolor()[0] < max(fill, ink) for p in lines[:2])
    plt.close(fig)


def test_shading_lines_are_opaque_so_agreeing_replicates_dont_read_as_emphasis():
    """The three coincide wherever the seeds agree, which is most of the figure."""
    values = np.linspace(0.2, 0.8, len(LANDMARKS))[:, None] * np.ones(3)
    fig = probe_trace_grid({(f, "mix"): np.stack([values] * 3)[:, None] for f in FORMS}, targets=("mix",))
    ax = fig.get_axes()[0]
    lines = [p for p in shading(ax) if not p.get_fill()]
    assert len(lines) == 6 and all(p.get_alpha() is None for p in lines)
    # Minimum, summary and maximum, each tinted against its own surround, and each with the
    # paler companion that carries its elided risers: six colors mixed once, none of them mixed
    # again at draw time.
    assert len({to_hex(p.get_edgecolor()) for p in lines}) == 6
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
