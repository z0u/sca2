"""The disjoint-vocabulary color language: two surface forms, no bridge.

Ex-2.1.5's domain. The corpus holds two sublanguages that are never seen
together — color-mixing equations written with real color names, and the same
arithmetic written as 3-digit hex codes::

    melon + ultramarine = dusty rose
    #e26 + #48a = #958

Two RGB grids are in play. Names live on the *full cube* (8-bit, 256 levels per
channel); hex codes live on the *hex grid* (4-bit, 16 levels, one digit per
channel). The mix is always the channel-wise round-half-up mean computed in the
full cube; the two forms differ only in how values enter and leave it:

- Named operands already sit in the full cube; the answer snaps to the nearest
  named color (Euclidean), distance ties broken by a coin flip seeded from the
  mix value.
- Hex operands are lifted by digit repetition (``#f80`` → ``#ff8800``) and the
  answer snaps back to the hex grid.

The named palette comes from the xkcd color survey (all 949 names, via
matplotlib), ordered by farthest-point selection so the first *n* form the most
uniform palette available at that size. Hex operands come from a fixed seeded
permutation of the 4096 grid points, so smaller subsets are prefixes of larger
ones. The optional *cross* form (``melon + #48a = #958``) is the bridge arm's
intervention — off by default, and the only place the two vocabularies meet.

Values in :class:`Example` are always full-cube (0..255) coordinates, whatever
the surface form, so probe targets and distance metrics share one scale.
"""

from functools import cache
from typing import Callable, Iterable, Literal

import numpy as np

from sca.data.colors import Example

N_FULL = 256
"""Levels per channel in the full cube (8-bit)."""

N_HEX = 16
"""Levels per channel on the hex grid (one hex digit)."""

STEP = 17
"""Lift factor between the grids: digit repetition is multiplication by 17."""

type Rgb8 = tuple[int, int, int]
"""A full-cube color, 0..255 per channel."""

type MixedForm = Literal["named", "hex", "cross"]

FORM_WEIGHTS: dict[MixedForm, float] = {"named": 0.5, "hex": 0.5}
"""The disjoint corpus: the two sublanguages, never a line that ties them."""

BRIDGE_WEIGHTS: dict[MixedForm, float] = {"named": 0.45, "hex": 0.45, "cross": 0.1}
"""The bridge arm: a small amount of cross-form supervision."""


@cache
def xkcd_survey() -> dict[str, Rgb8]:
    """All 949 xkcd survey names → full-cube values, alphabetical.

    Sourced from matplotlib's vendored copy of the survey results; sorting by
    name makes the ordering independent of matplotlib's dict order. Cached —
    treat the result as read-only.
    """
    from matplotlib import colors as mcolors
    from matplotlib.typing import ColorType

    def rgb8(spec: ColorType) -> Rgb8:
        r, g, b = (int(round(v * (N_FULL - 1))) for v in mcolors.to_rgb(spec))
        return (r, g, b)

    return {k.removeprefix("xkcd:"): rgb8(v) for k, v in sorted(mcolors.XKCD_COLORS.items())}


def fps_order(points: np.ndarray, start: int) -> list[int]:
    """Greedy farthest-point ordering of *points*, beginning at index *start*.

    Every prefix of the result is a maximally spread subset in the greedy
    sense: each added point maximizes its distance to the points already
    chosen. Ties resolve to the lowest index, so the order is deterministic.
    """
    d = np.linalg.norm(points - points[start], axis=1)
    order = [start]
    for _ in range(len(points) - 1):
        nxt = int(d.argmax())
        order.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return order


def xkcd_palette(n: int) -> dict[str, Rgb8]:
    """The *n* most uniformly spread xkcd names (first *n* under farthest-point order).

    The ordering starts from the survey color nearest black (the design study's
    convention: at N = 140 it gives min/median nearest-neighbor distances of
    41.9/46.5 with ``blue with a hint of purple`` the longest selected name),
    so palettes of every size are prefixes of one fixed sequence.
    """
    survey = xkcd_survey()
    names = list(survey)
    points = np.array(list(survey.values()), dtype=np.float64)
    start = int(np.linalg.norm(points, axis=1).argmin())
    return {names[i]: survey[names[i]] for i in fps_order(points, start)[:n]}


def hex_operands(n: int, seed: int) -> list[Rgb8]:
    """A fixed random subset of the hex grid, lifted to the full cube.

    The first *n* points of one seeded permutation of all 4096, so subsets of
    different sizes nest. The subset constrains *operands* only — a hex
    equation's answer may be any point on the hex grid.
    """
    rng = np.random.default_rng(seed)
    picks = rng.permutation(N_HEX**3)[:n]
    return [lift(((int(p) >> 8) & 0xF, (int(p) >> 4) & 0xF, int(p) & 0xF)) for p in picks]


def lift(c4: tuple[int, int, int]) -> Rgb8:
    """Hex-grid digits (0..15) → full cube, by digit repetition."""
    r, g, b = (v * STEP for v in c4)
    return (r, g, b)


def snap_hex(c8: Rgb8) -> Rgb8:
    """Nearest hex-grid point (returned in full-cube coordinates).

    Rounds each channel to the nearest multiple of 17; integer inputs can never
    tie (the midpoint 8.5 is not an integer offset).
    """
    r, g, b = ((2 * v + STEP) // (2 * STEP) * STEP for v in c8)
    return (r, g, b)


def mix8(a: Rgb8, b: Rgb8) -> Rgb8:
    """Channel-wise round-half-up mean in the full cube — the single ground truth."""
    r, g, bl = ((x + y + 1) // 2 for x, y in zip(a, b, strict=True))
    return (r, g, bl)


def to_hex3(c8: Rgb8) -> str:
    """Full-cube coordinates of a hex-grid point → ``#xyz``."""
    assert all(v % STEP == 0 for v in c8), f"{c8} is not on the hex grid"
    return "#" + "".join(f"{v // STEP:x}" for v in c8)


def snap_name(target: Rgb8, palette_rgb: np.ndarray) -> int:
    """Index of the palette color nearest *target*; ties fall to a coin flip
    seeded from the mix value, so every (palette, target) pair snaps the same
    way in every process.
    """
    d2 = ((palette_rgb.astype(np.int64) - np.array(target)) ** 2).sum(axis=1)
    ties = np.flatnonzero(d2 == d2.min())
    if len(ties) == 1:
        return int(ties[0])
    r, g, b = target
    return int(np.random.default_rng((r << 16) | (g << 8) | b).choice(ties))


def alphabet() -> list[str]:
    """Every character the language can produce, fixed a priori.

    Name characters come from the *full* survey (not the selected palette), so
    every palette size shares one token table.
    """
    return sorted({*"".join(xkcd_survey()), *"0123456789abcdef", *"#+= \n"})


def make_example(form: MixedForm, a: Rgb8, b: Rgb8, names: dict[Rgb8, str], rng: np.random.Generator) -> Example:
    """Render one equation; operand order is random.

    For the cross form, *a* is the named operand and *b* the (lifted) hex one.
    ``result`` holds the exact full-cube mix, pre-snap; the answer string holds
    the snapped, surface-form value.
    """
    result = mix8(a, b)
    palette_rgb = None
    match form:
        case "named":
            palette_rgb = np.array(list(names))
            lhs, rhs = names[a], names[b]
            ans = list(names.values())[snap_name(result, palette_rgb)]
        case "hex":
            lhs, rhs, ans = to_hex3(a), to_hex3(b), to_hex3(snap_hex(result))
        case "cross":
            lhs, rhs, ans = names[a], to_hex3(b), to_hex3(snap_hex(result))
    if rng.random() < 0.5:
        lhs, rhs, a, b = rhs, lhs, b, a
    return Example(f"{lhs} + {rhs} = ", ans, a, b, result)


def distinct_pairs(operands: Iterable[Rgb8]) -> list[tuple[Rgb8, Rgb8]]:
    """All unordered non-self operand pairs, canonically ordered."""
    ops = sorted(set(operands))
    return [(a, b) for i, a in enumerate(ops) for b in ops[i + 1 :]]


def holdout_split(
    pairs: list[tuple[Rgb8, Rgb8]], seed: int, frac: float
) -> tuple[list[tuple[Rgb8, Rgb8]], list[tuple[Rgb8, Rgb8]]]:
    """Split unordered pairs into (train, holdout), exactly *frac* held out."""
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(len(pairs), round(len(pairs) * frac), replace=False).tolist())
    return [p for i, p in enumerate(pairs) if i not in chosen], sorted(p for i, p in enumerate(pairs) if i in chosen)


def sample_corpus(
    n: int,
    seed: int,
    palette: dict[str, Rgb8],
    hex_ops: list[Rgb8],
    held_named: Callable[[Rgb8, Rgb8], bool],
    held_hex: Callable[[Rgb8, Rgb8], bool],
    weights: dict[MixedForm, float] = FORM_WEIGHTS,
) -> list[Example]:
    """*n* training equations drawn i.i.d. with the given form mix.

    Operands draw uniformly (with replacement) from their sublanguage's operand
    set, skipping held-out pairs; self-pairs occur at their natural rate and
    always train. Cross pairs (one operand per vocabulary) have no holdout —
    the form exists to supervise alignment, not to be graded.
    """
    rng = np.random.default_rng(seed)
    names = {v: k for k, v in palette.items()}
    named_vals = list(palette.values())
    forms, probs = zip(*weights.items(), strict=True)

    out: list[Example] = []
    while len(out) < n:
        for form in rng.choice(forms, size=n - len(out), p=probs):
            match form:
                case "named":
                    a, b = (named_vals[i] for i in rng.integers(len(named_vals), size=2))
                    if held_named(a, b):
                        continue
                case "hex":
                    a, b = (hex_ops[i] for i in rng.integers(len(hex_ops), size=2))
                    if held_hex(a, b):
                        continue
                case "cross":
                    a = named_vals[rng.integers(len(named_vals))]
                    b = hex_ops[rng.integers(len(hex_ops))]
            out.append(make_example(form, a, b, names, rng))
    return out


def as_form(pairs: Iterable[tuple[Rgb8, Rgb8]], form: MixedForm, palette: dict[str, Rgb8], seed: int) -> list[Example]:
    """Render pairs in the given form (eval and probe sets)."""
    rng = np.random.default_rng(seed)
    names = {v: k for k, v in palette.items()}
    return [make_example(form, a, b, names, rng) for a, b in pairs]


LANDMARKS = (
    "o1s0", "o1s1", "o1e1", "o1e0",  # operand 1: chars from its start / end
    "plus",
    "o2s0", "o2s1", "o2e1", "o2e0",  # operand 2
    "eq", "pre",  # '=' and the pre-answer space
    "as0", "as1",  # answer chars from its start
    "ae1", "ae0",  # answer chars from its end
)  # fmt: skip
"""Grammar landmarks: positions that correspond across the two surface forms.

Raw offsets don't line up across lines — names vary in length and hex lines
are shorter — so probe maps index positions by these landmarks instead.
The answer is sampled like an operand — two chars from each end (``as0, as1,
ae1, ae0``) — so both mirror the ``s0, s1, e1, e0`` shape. Every landmark exists
in every line (the shortest operand or answer is 3 characters), though on short
words some landmarks coincide (e.g. ``as1`` and ``ae1`` on a 3-character answer).
"""
OPERATORS = ("plus", "eq", "pre")
"""The subset of LANDMARKS that are operators (not operands or results)"""


def _span_risers() -> frozenset[int]:
    def role(name: str) -> tuple[str, str] | None:  # (token, side) for anchored landmarks; None for plus/eq/pre
        return (name[:-2], name[-2]) if len(name) >= 2 and name[-2] in "se" and name[-1].isdigit() else None

    return frozenset(
        _i
        for _i in range(len(LANDMARKS) - 1)
        if (a := role(LANDMARKS[_i])) and (b := role(LANDMARKS[_i + 1])) and a[0] == b[0] and (a[1], b[1]) == ("s", "e")
    )


SPAN_RISERS = _span_risers()
"""Indices *i* where landmark *i*→*i+1* jumps from a start-anchored to an end-anchored
position of the same token — across a word's variable-length, unsampled middle (``o1s1``
→``o1e1``, ``o2s1``→``o2e1``, ``as1``→``ae1``). On fixed-width forms (hex) these land on
adjacent characters, so a plot draws them as discrete steps; on named ones they cover an
unmeasured interior, better drawn as a smooth slide."""


def landmark_indices(ex: Example) -> dict[str, int]:
    """Character positions of each landmark within ``ex.text``."""
    ops = ex.prompt[:-3].split(" + ")
    assert len(ops) == 2, f"unparseable prompt: {ex.prompt!r}"
    o1s, o2s = 0, len(ops[0]) + 3
    o1e, o2e = len(ops[0]) - 1, o2s + len(ops[1]) - 1
    ans, ane = len(ex.prompt), len(ex.prompt) + len(ex.answer) - 1
    return {
        "o1s0": o1s, "o1s1": o1s + 1, "o1e1": o1e - 1, "o1e0": o1e,
        "plus": o1e + 2,
        "o2s0": o2s, "o2s1": o2s + 1, "o2e1": o2e - 1, "o2e0": o2e,
        "eq": o2e + 2, "pre": o2e + 3,
        "as0": ans, "as1": ans + 1,
        "ae1": ane - 1, "ae0": ane,
    }  # fmt: skip
