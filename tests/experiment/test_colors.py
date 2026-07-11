"""The color-mixing language: exact ground truth, well-formed text, honest splits."""

import re

import numpy as np
import pytest

from experiment.data import colors
from experiment.data.colors import PALETTE, Example, mix

LINE = re.compile(r"^([a-z]+|#[0-9a-f]{3}) \+ ([a-z]+|#[0-9a-f]{3}) = ([a-z]+|#[0-9a-f]{3})\n$")
ALIAS = re.compile(r"^[a-z]+ = #[0-9a-f]{3}\n$")


@pytest.mark.parametrize(
    ("a", "b", "result"),
    [
        ("red", "blue", "purple"),
        ("white", "black", "gray"),
        ("red", "lime", "olive"),
        ("yellow", "blue", "gray"),
        ("red", "red", "red"),  # idempotent
    ],
)
def test_mixing_table(a: str, b: str, result: str):
    assert mix(PALETTE[a], PALETTE[b]) == PALETTE[result]
    assert mix(PALETTE[b], PALETTE[a]) == PALETTE[result]  # commutative


def test_mix_is_exact_round_half_up():
    assert mix((0, 0, 0), (15, 15, 15)) == (8, 8, 8)
    assert mix((0, 4, 5), (1, 5, 6)) == (1, 5, 6)


def test_hex_and_redness():
    assert colors.to_hex(PALETTE["orange"]) == "#f80"
    assert colors.redness(PALETTE["red"]) == 1.0
    assert colors.redness(PALETTE["white"]) == 0.0
    assert colors.redness(PALETTE["magenta"]) == 0.5
    assert colors.redness(PALETTE["black"]) == 0.0


def test_closed_named_pairs_are_closed():
    pairs = colors.closed_named_pairs()
    assert len(pairs) == 76  # 27 self-pairs + 49 distinct
    assert all(mix(a, b) in colors.NAMES for a, b in pairs)


def test_named_split_is_deterministic_and_disjoint():
    train, holdout = colors.split_named_pairs(0)
    assert (train, holdout) == colors.split_named_pairs(0)
    assert not set(train) & set(holdout)
    assert len(holdout) == 10  # 20% of the 49 distinct closed pairs
    assert all(a != b for a, b in holdout)  # self-pairs always train


def test_corpus_is_well_formed_and_deterministic():
    train, holdout = colors.split_named_pairs(0)
    corpus = colors.sample_corpus(500, 0, train)
    assert corpus == colors.sample_corpus(500, 0, train)
    for ex in corpus:
        assert LINE.match(ex.text) or ALIAS.match(ex.text)
        assert ex.rhs is None or mix(ex.lhs, ex.rhs) == ex.result
        assert set(ex.text) <= set(colors.alphabet())
    # No held-out pair ever appears as a *named* equation.
    named = [ex for ex in corpus if LINE.match(ex.text) and "#" not in ex.text]
    assert named and not {ex.pair for ex in named} & set(holdout)


def test_unseen_pairs_avoid_the_corpus():
    train, _ = colors.split_named_pairs(0)
    seen = {p for ex in colors.sample_corpus(500, 0, train) if (p := ex.pair) is not None}
    unseen = colors.sample_unseen("hex", 50, 3, seen)
    assert len(unseen) == 50
    assert not {ex.pair for ex in unseen} & seen
    assert len({ex.pair for ex in unseen}) == 50  # distinct among themselves


def test_example_sets_roundtrip():
    rng = np.random.default_rng(0)
    sets = {
        "eq": [colors.make_example("hex", (0, 8, 15), (1, 2, 3), rng)],
        "alias": [colors.make_example("alias", PALETTE["red"], None, rng)],
    }
    assert colors.load_example_sets(colors.dump_example_sets(sets)) == sets


def test_prompt_text_matches_operand_values():
    rng = np.random.default_rng(1)
    ex: Example = colors.make_example("cross", PALETTE["red"], (10, 2, 14), rng)
    first = ex.prompt.split(" ")[0]
    value = PALETTE.get(first) or tuple(int(c, 16) for c in first[1:])
    assert value == ex.lhs
