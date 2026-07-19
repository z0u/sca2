"""Opaque char-level names: fixed shape, no value leakage, faithful re-rendering."""

import re

import numpy as np

from sca.data import char_names as ch
from sca.data import named_colors as nc
from sca.data.named_colors import GRIDS

LINE = re.compile(r"^[a-z]{4} \+ [a-z]{4} = [a-z]{4}\n$")


def test_names_are_fixed_length_and_distinct():
    names = ch.opaque_names(GRIDS["v216"], seed=0)
    assert len(names) == 216
    assert all(len(n) == ch.NAME_LEN and set(n) <= set(ch.LETTERS) for n in names)
    assert len(set(names.values())) == 216  # a bijection: every color keeps its own name


def test_names_do_not_leak_value():
    # The assignment is independent of the value: per character position, the
    # mean value of colors sharing a letter stays near the grand mean. (The
    # `c05f` names this replaces would fail catastrophically here.)
    names = ch.opaque_names(GRIDS["v216"], seed=0)
    values = np.array(list(names.values()), dtype=float)
    grand, std = values.mean(0), values.std(0)
    for pos in range(ch.NAME_LEN):
        for letter in {n[pos] for n in names}:
            group = values[[n[pos] == letter for n in names]]
            bound = 4 * std / np.sqrt(len(group))  # 4 standard errors: just not value-sorted
            assert (np.abs(group.mean(0) - grand) < bound).all()


def test_rename_preserves_semantics():
    levels = GRIDS["v27"]
    names = {v: k for k, v in ch.opaque_names(levels, seed=0).items()}
    for ex in nc.sample_corpus(64, seed=0, levels=levels):
        r = ch.rename(ex, names)
        assert (r.lhs, r.rhs, r.result) == (ex.lhs, ex.rhs, ex.result)
        assert LINE.match(r.text)
        assert r.prompt.split(" ")[0] == names[ex.lhs]  # operand order preserved
        assert set(r.text) <= set(ch.alphabet())
