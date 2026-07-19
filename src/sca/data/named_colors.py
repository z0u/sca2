"""The named-only color language: a variant domain where every color is a word.

The base language (`sca.data.colors`) grounds color names in hex codes: alias
lines hand the model each name's value, and hex arithmetic teaches mixing in a
form where the geometry is legible per digit. This module removes that
scaffolding entirely. A vocabulary of colors is chosen from the 16-level RGB
grid, every color is a single *opaque token* (one token per name — no
characters, no hex), and the only sentences are mixing equations between
vocabulary colors whose mix is itself in the vocabulary::

    red + blue = purple

Nothing in the token stream reveals that colors live on a 3D grid; the only
structure is the co-occurrence statistics of the mixing table. Whether a model
can recover the geometry from that alone — tensor completion, in effect — is
the question ex-2.1.3 asks.

Vocabularies are level sub-grids of the full 16-level cube (`GRIDS`): the
classic 27-color palette, plus denser grids up to the full 4096. Distinct
closed pairs (mix on-vocabulary) split into train/holdout; pairs whose mix
falls *off* the vocabulary ("open" pairs) can never be answered exactly and
instead measure how *close* the model's guess lands.

Synthetic names encode the value (``c05f`` = ``#05f``) purely for human
readability in reports: each name is a single token, so the model sees an
opaque id and must learn its meaning from usage, exactly as for ``red``.
"""

import zlib
from typing import Callable, Iterable

import numpy as np

from sca.config import TokenizerConfig
from sca.data.colors import N_LEVELS, PALETTE, Example, Rgb, mix, to_hex

GRIDS: dict[str, tuple[int, ...]] = {
    "v27": (0, 8, 15),
    "v64": (0, 5, 10, 15),
    "v216": (0, 3, 6, 9, 12, 15),
    "v4096": tuple(range(N_LEVELS)),
}
"""Level sub-grids keyed by vocabulary size (colors = levels³).

Chosen so a useful fraction of pairs stays closed under mixing: steps of 5 and
3 divide 15, giving ~12–17% closure; the full grid is 100% closed (and so has
no open pairs).
"""

SYNTAX = ("+", "=", "\n")
"""Non-color words; the pad token '' joins them in the tokenizer."""


def grid_palette(levels: Iterable[int]) -> dict[str, Rgb]:
    """Name every point of the level sub-grid; the 3-level grid keeps the classic names."""
    levels = tuple(levels)
    if set(levels) == {0, 8, 15}:
        return dict(PALETTE)
    return {f"c{to_hex(c)[1:]}": c for r in levels for g in levels for b in levels for c in [(r, g, b)]}


class WordTokenizer:
    """One token per word. Mirrors `CharTokenizer`'s conventions (sorted vocabulary,
    the empty string as padding token 0), but vocabulary entries are whole words —
    color names and the syntax tokens — rather than characters.
    """

    def __init__(self, config: TokenizerConfig):
        self.vocabulary = sorted({""} | set(config.vocabulary))
        self.vocab_size = len(self.vocabulary)
        self.stoi = {w: i for i, w in enumerate(self.vocabulary)}
        self.itos = {i: w for i, w in enumerate(self.vocabulary)}

    def encode_words(self, words: Iterable[str]) -> list[int]:
        return [self.stoi[w] for w in words]

    def decode_words(self, tokens: Iterable[int]) -> list[str]:
        return [self.itos.get(i, "") for i in tokens]


def as_words(ex: Example) -> list[str]:
    """One line's token stream: operand, +, operand, =, answer, newline."""
    return [*ex.prompt.split(), ex.answer, "\n"]


def make_example(a: Rgb, b: Rgb, names: dict[Rgb, str], rng: np.random.Generator) -> Example:
    """Render one equation with random operand order.

    The answer is empty when the mix has no name (open pairs) — such examples
    are prompts to complete, never training lines.
    """
    result = mix(a, b)
    if rng.random() < 0.5:
        a, b = b, a
    return Example(f"{names[a]} + {names[b]} = ", names.get(result, ""), a, b, result)


def pair_key(a: Rgb, b: Rgb) -> tuple[Rgb, Rgb]:
    """Canonical unordered form — the unit of train/eval separation."""
    return (min(a, b), max(a, b))


def closed_pairs(levels: Iterable[int]) -> list[tuple[Rgb, Rgb]]:
    """All distinct unordered vocabulary pairs whose mix is on-vocabulary.

    Enumeration is O(colors²) — fine up to the 216-color grid; the full grid
    (8.4M pairs) should use `holdout_test`'s hash split instead.
    """
    lset = set(levels)
    colors = list(grid_palette(levels).values())
    return [(a, b) for i, a in enumerate(colors) for b in colors[i + 1 :] if all(v in lset for v in mix(a, b))]


def open_pairs(levels: Iterable[int]) -> list[tuple[Rgb, Rgb]]:
    """The complement: distinct pairs whose mix falls off the vocabulary."""
    lset = set(levels)
    colors = list(grid_palette(levels).values())
    return [(a, b) for i, a in enumerate(colors) for b in colors[i + 1 :] if not all(v in lset for v in mix(a, b))]


def holdout_test(levels: Iterable[int], seed: int, frac: float = 0.2) -> Callable[[Rgb, Rgb], bool]:
    """A deterministic membership test for held-out closed pairs; self-pairs always train.

    Small grids enumerate the distinct closed pairs and split exactly
    (mirroring `colors.split_named_pairs`); the full grid's 8.4M pairs get a
    stable hash threshold instead, which converges to *frac* by volume.
    """
    levels = tuple(levels)
    if len(levels) ** 3 <= 1000:
        distinct = closed_pairs(levels)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(distinct), round(len(distinct) * frac), replace=False)
        held = {distinct[i] for i in chosen}
        return lambda a, b: pair_key(a, b) in held
    return lambda a, b: a != b and zlib.crc32(f"{seed}:{pair_key(a, b)}".encode()) % 10_000 < frac * 10_000


def sample_corpus(n: int, seed: int, levels: Iterable[int], holdout_frac: float = 0.2) -> list[Example]:
    """*n* closed-pair equations drawn i.i.d. (with replacement) from the train side.

    Operand pairs are uniform over ordered vocabulary pairs, filtered to those
    whose mix is on-vocabulary and not held out — so operand order is already
    random and self-pairs occur at their natural rate.
    """
    levels = tuple(levels)
    palette = grid_palette(levels)
    names = {v: k for k, v in palette.items()}
    colors = np.array(list(palette.values()))
    on_grid = np.zeros(N_LEVELS, dtype=bool)
    on_grid[list(levels)] = True
    held = holdout_test(levels, seed, holdout_frac)
    rng = np.random.default_rng(seed)

    out: list[Example] = []
    while len(out) < n:
        ab = rng.integers(len(colors), size=(2, 2 * (n - len(out)) + 64))
        closed = on_grid[(colors[ab[0]] + colors[ab[1]] + 1) // 2].all(axis=1)
        for i in np.flatnonzero(closed):
            a, b = (_as_rgb(colors[j]) for j in ab[:, i])
            if len(out) < n and not held(a, b):
                out.append(make_example(a, b, names, rng))
    return out


def _as_rgb(row: np.ndarray) -> Rgb:
    return (int(row[0]), int(row[1]), int(row[2]))


def nearest_distances(vocab_rgb: np.ndarray, target: Rgb) -> np.ndarray:
    """Euclidean distance (unit-cube units) from *target* to every vocabulary color."""
    return np.linalg.norm((vocab_rgb - np.array(target)) / (N_LEVELS - 1), axis=1)
