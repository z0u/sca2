"""Opaque spelled-out names for the char-level name-only language (ex-2.1.4).

Ex-2.1.3's language gives every color a single atomic token, so a name *is* an
embedding row. This module re-renders the same language for a character-level
tokenizer, where a name is something the model must read and write one letter
at a time. The design trap it exists to avoid: the synthetic grid names
(``c05f``) spell the color's value per character, which at char level is just
hex with a prefix — and even the classic palette names vary in length, which
would confound the answer-emission schedule. So every color gets a random
fixed-length letter string, assigned independently of its value::

    tkzk + qwfd = hjnp

No character carries channel information; the binding from spelling to value
is holistic. Corpora and eval sets come from re-rendering
`sca.data.named_colors` examples (same pairs, same splits, same operand
orders), so the only difference from ex-2.1.3 is the tokenizer's view.
"""

from typing import Iterable

import numpy as np

from sca.data.colors import Example, Rgb
from sca.data.named_colors import grid_palette

LETTERS = "abcdefghijklmnopqrstuvwxyz"
NAME_LEN = 4
"""Fixed name length: 26⁴ ≈ 457k strings comfortably covers the 216-color grid,
and equal lengths keep every equation the same shape (probe positions align)."""


def alphabet() -> list[str]:
    """Every character the language can produce (fixes the vocabulary a priori)."""
    return sorted({*LETTERS, *"+= \n"})


def opaque_names(levels: Iterable[int], seed: int) -> dict[str, Rgb]:
    """A random fixed-length letter name for every grid color (name → value).

    Names are drawn without replacement from the length-``NAME_LEN`` strings
    and assigned to colors independently of their values.
    """
    colors = list(grid_palette(levels).values())
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(LETTERS) ** NAME_LEN, size=len(colors), replace=False)
    spell = lambda i: "".join(LETTERS[(int(i) // len(LETTERS) ** k) % len(LETTERS)] for k in range(NAME_LEN))  # noqa: E731
    return {spell(i): c for i, c in zip(ids, colors, strict=True)}


def rename(ex: Example, names: dict[Rgb, str]) -> Example:
    """Re-render a named-only example under different names, preserving operand order.

    `named_colors.make_example` renders ``lhs`` first, so the operand order
    survives the round trip; open pairs keep their empty answer.
    """
    assert ex.rhs is not None
    return Example(f"{names[ex.lhs]} + {names[ex.rhs]} = ", names.get(ex.result, ""), ex.lhs, ex.rhs, ex.result)
