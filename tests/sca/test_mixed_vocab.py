"""The disjoint-vocabulary language: palettes, snapping, corpus, landmarks."""

import numpy as np
import pytest

from sca.data import mixed_vocab as mv


def nn_stats(points: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(points[None] - points[:, None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(nn.min()), float(np.median(nn))


def test_palette_matches_the_design_study():
    """The preregistered report quotes min ≈ 41 / median ≈ 46 at N = 140."""
    pal = mv.xkcd_palette(140)
    assert len(pal) == 140
    mn, md = nn_stats(np.array(list(pal.values()), dtype=np.float64))
    assert mn == pytest.approx(41, abs=1)
    assert md == pytest.approx(46, abs=1)
    assert max(pal, key=len) == "blue with a hint of purple"


def test_palettes_and_hex_subsets_nest():
    p140, p250 = mv.xkcd_palette(140), mv.xkcd_palette(250)
    assert list(p250)[:140] == list(p140)
    h216, h2048 = mv.hex_operands(216, 0), mv.hex_operands(2048, 0)
    assert h2048[:216] == h216
    assert len(set(h2048)) == 2048


def test_answer_distribution_matches_the_design_study():
    """139/140 reachable from distinct pairs; perplexity ≈ 87 with self-pairs in."""
    pal = np.array(list(mv.xkcd_palette(140).values()))

    def counts(k: int) -> np.ndarray:
        i, j = np.triu_indices(len(pal), k=k)
        out = np.zeros(len(pal))
        for m in (pal[i] + pal[j] + 1) // 2:
            out[mv.snap_name((int(m[0]), int(m[1]), int(m[2])), pal)] += 1
        return out

    assert (counts(1) > 0).sum() == 139
    c = counts(0)
    p = c[c > 0] / c.sum()
    assert float(np.exp(-(p * np.log(p)).sum())) == pytest.approx(87, abs=1)


def test_lift_snap_and_mix():
    assert mv.lift((15, 8, 0)) == (255, 136, 0)
    assert mv.to_hex3((255, 136, 0)) == "#f80"
    # Lift then snap is the identity on the hex grid.
    for c4 in [(0, 0, 0), (15, 15, 15), (7, 8, 9)]:
        assert mv.snap_hex(mv.lift(c4)) == mv.lift(c4)
    # Snapping rounds to the nearest multiple of 17 (brute force check).
    for v in range(256):
        snapped = mv.snap_hex((v, 0, 0))[0]
        best = min(range(0, 256, 17), key=lambda g: abs(g - v))
        assert snapped == best
    # Round-half-up mean, computed in the full cube.
    assert mv.mix8((0, 0, 0), (1, 1, 3)) == (1, 1, 2)
    assert mv.mix8((255, 0, 17), (255, 0, 18)) == (255, 0, 18)


def test_snap_name_tie_break_is_deterministic():
    pal = np.array([(0, 0, 0), (2, 0, 0), (5, 5, 5)])  # (1,0,0) ties the first two
    picks = {mv.snap_name((1, 0, 0), pal) for _ in range(5)}
    assert len(picks) == 1 and picks < {0, 1}
    assert mv.snap_name((2, 0, 0), pal) == 1  # exact hit, no tie


def test_corpus_is_disjoint_and_respects_holdout():
    palette = mv.xkcd_palette(140)
    hex_ops = mv.hex_operands(216, 0)
    named_train, named_held = mv.holdout_split(mv.distinct_pairs(palette.values()), 0, 0.2)
    hex_train, hex_held = mv.holdout_split(mv.distinct_pairs(hex_ops), 0, 0.2)
    held_n, held_h = set(named_held), set(hex_held)
    corpus = mv.sample_corpus(
        500, 0, palette, hex_ops,
        lambda a, b: (min(a, b), max(a, b)) in held_n,
        lambda a, b: (min(a, b), max(a, b)) in held_h,
    )  # fmt: skip
    assert len(corpus) == 500
    names = set(palette)
    for ex in corpus:
        ops = ex.prompt[:-3].split(" + ")
        forms = {op.startswith("#") for op in ops}
        assert len(forms) == 1, f"cross line in a disjoint corpus: {ex.prompt!r}"
        pair = ex.pair
        if forms == {True}:
            assert pair not in held_h and ex.answer.startswith("#")
        else:
            assert pair not in held_n and ex.answer in names
        assert set(ex.text) <= set(mv.alphabet())

    bridged = mv.sample_corpus(500, 0, palette, hex_ops, lambda a, b: False, lambda a, b: False, mv.BRIDGE_WEIGHTS)
    kinds = {tuple(sorted(op.startswith("#") for op in ex.prompt[:-3].split(" + "))) for ex in bridged}
    assert (False, True) in kinds, "bridge corpus should contain cross lines"
    cross = [ex for ex in bridged if ex.prompt.count("#") == 1]
    assert all(ex.answer.startswith("#") for ex in cross), "cross answers are hex"


def test_landmarks_point_at_the_right_characters():
    rng = np.random.default_rng(0)
    palette = mv.xkcd_palette(140)
    names = {v: k for k, v in palette.items()}
    hex_ex = mv.make_example("hex", mv.lift((14, 2, 6)), mv.lift((4, 8, 10)), names, rng)
    vals = list(palette.values())
    named_ex = mv.make_example("named", vals[0], vals[1], names, rng)
    for ex in (hex_ex, named_ex):
        lm = mv.landmark_indices(ex)
        ops = ex.prompt[:-3].split(" + ")
        assert ex.text[lm["plus"]] == "+"
        assert ex.text[lm["eq"]] == "="
        assert ex.text[lm["pre"]] == " " and lm["pre"] == len(ex.prompt) - 1
        assert ex.text[lm["o1s0"]] == ops[0][0] and ex.text[lm["o1e0"]] == ops[0][-1]
        assert ex.text[lm["o2s0"]] == ops[1][0] and ex.text[lm["o2e0"]] == ops[1][-1]
        assert ex.text[lm["as0"]] == ex.answer[0] and ex.text[lm["ae0"]] == ex.answer[-1]
    assert hex_ex.text[mv.landmark_indices(hex_ex)["as0"]] == "#"
