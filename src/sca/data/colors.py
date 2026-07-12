"""The color-mixing language: M2's synthetic domain.

A color is a point on a 16-level RGB grid, written either as CSS-style short
hex (``#f80``, one digit per channel) or as one of 27 names covering the
sub-grid with channels in {0, 8, 15} (``red`` = ``#f00``, ``purple`` =
``#808``). Mixing is the channel-wise round-half-up mean::

    mix(a, b) = (a + b + 1) // 2        # per channel, closed on 0..15

which is commutative and idempotent, so the ground truth for every equation is
exact integer arithmetic — no perceptual judgement calls. Conveniently, the
palette lands on familiar values: ``red + blue = purple``, ``white + black =
gray``, ``red + lime = olive`` (averaging *darkens*; this is pigment-free
light-averaging, not paint).

Sentences come in four forms, one per line:

    named   ``red + blue = purple``      only pairs whose mix is itself named
    hex     ``#e26 + #48a = #958``       any pair on the full grid
    cross   ``red + #08f = #848``        a hex operand forces a hex result
    alias   ``red = #f00``               the name ↔ value dictionary

The result's surface form is deterministic given the operands (hex unless both
operands are named and the mix lands on the palette), so greedy completion has
a single correct answer — the exact-match accuracy in D2.1.x is well-defined.

``redness`` ports M1's graded concept label (the anchor's noisy supervision
signal) to this grid, so the anchoring experiments can label sequences the same
way the autoencoder experiments labeled samples.
"""

import json
from typing import Iterable, Literal, NamedTuple

import numpy as np

N_LEVELS = 16
"""Levels per channel; one hex digit."""

type Rgb = tuple[int, int, int]
type Form = Literal["named", "hex", "cross", "alias"]

PALETTE: dict[str, Rgb] = {
    "black": (0, 0, 0),
    "navy": (0, 0, 8),
    "blue": (0, 0, 15),
    "green": (0, 8, 0),
    "teal": (0, 8, 8),
    "azure": (0, 8, 15),
    "lime": (0, 15, 0),
    "spring": (0, 15, 8),
    "cyan": (0, 15, 15),
    "maroon": (8, 0, 0),
    "purple": (8, 0, 8),
    "violet": (8, 0, 15),
    "olive": (8, 8, 0),
    "gray": (8, 8, 8),
    "lavender": (8, 8, 15),
    "chartreuse": (8, 15, 0),
    "mint": (8, 15, 8),
    "sky": (8, 15, 15),
    "red": (15, 0, 0),
    "rose": (15, 0, 8),
    "magenta": (15, 0, 15),
    "orange": (15, 8, 0),
    "salmon": (15, 8, 8),
    "orchid": (15, 8, 15),
    "yellow": (15, 15, 0),
    "cream": (15, 15, 8),
    "white": (15, 15, 15),
}
"""All 27 grid points with channels in {0, 8, 15}, named CSS-style."""

NAMES: dict[Rgb, str] = {v: k for k, v in PALETTE.items()}


def mix(a: Rgb, b: Rgb) -> Rgb:
    """Channel-wise round-half-up mean — the domain's single ground truth."""
    r, g, bl = ((x + y + 1) // 2 for x, y in zip(a, b, strict=True))
    return (r, g, bl)


def to_hex(c: Rgb) -> str:
    return "#" + "".join(f"{x:x}" for x in c)


def redness(c: Rgb) -> float:
    """M1's graded *red* label on the unit cube: r·(1 − g/2 − b/2)."""
    r, g, b = (x / (N_LEVELS - 1) for x in c)
    return r * (1 - g / 2 - b / 2)


def alphabet() -> list[str]:
    """Every character the language can produce (fixes the vocabulary a priori)."""
    return sorted({*"".join(PALETTE), *"0123456789abcdef", *"#+= \n"})


class Example(NamedTuple):
    """One line of the language, split for completion eval and probing."""

    prompt: str
    """Everything up to and including the space after ``=``."""

    answer: str
    """The expected completion, without the trailing newline."""

    lhs: Rgb
    rhs: Rgb | None
    """Operand values; ``rhs`` is None for alias lines."""

    result: Rgb

    @property
    def text(self) -> str:
        return f"{self.prompt}{self.answer}\n"

    @property
    def pair(self) -> tuple[Rgb, Rgb] | None:
        """Unordered operand pair — the unit of train/eval separation."""
        return None if self.rhs is None else (min(self.lhs, self.rhs), max(self.lhs, self.rhs))


def make_example(form: Form, a: Rgb, b: Rgb | None, rng: np.random.Generator) -> Example:
    """Render one example; operand order and (for cross) which side is named are random."""
    if form == "alias":
        return Example(f"{NAMES[a]} = ", to_hex(a), a, None, a)
    assert b is not None
    result = mix(a, b)
    swap = rng.random() < 0.5
    match form:
        case "named":
            lhs, rhs, ans = NAMES[a], NAMES[b], NAMES[result]
        case "hex":
            lhs, rhs, ans = to_hex(a), to_hex(b), to_hex(result)
        case "cross":  # `a` is the named operand; `swap` decides which side it takes
            lhs, rhs, ans = NAMES[a], to_hex(b), to_hex(result)
    if swap:
        lhs, rhs, a, b = rhs, lhs, b, a
    return Example(f"{lhs} + {rhs} = ", ans, a, b, result)


def closed_named_pairs() -> list[tuple[Rgb, Rgb]]:
    """Unordered palette pairs whose mix is itself on the palette (incl. self-pairs)."""
    colors = list(PALETTE.values())
    return [(a, b) for i, a in enumerate(colors) for b in colors[i:] if mix(a, b) in NAMES]


def split_named_pairs(seed: int, holdout_frac: float = 0.2) -> tuple[list[tuple[Rgb, Rgb]], list[tuple[Rgb, Rgb]]]:
    """Split the *distinct* closed pairs into (train, holdout); self-pairs always train.

    Held-out pairs never appear as named equations in training, so completing
    them requires composing the alias dictionary with hex arithmetic.
    """
    rng = np.random.default_rng(seed)
    pairs = closed_named_pairs()
    distinct = [p for p in pairs if p[0] != p[1]]
    holdout = {distinct[i] for i in rng.choice(len(distinct), round(len(distinct) * holdout_frac), replace=False)}
    return [p for p in pairs if p not in holdout], sorted(holdout)


def _random_rgb(rng: np.random.Generator) -> Rgb:
    r, g, b = (int(x) for x in rng.integers(0, N_LEVELS, 3))
    return (r, g, b)


FORM_WEIGHTS: dict[Form, float] = {"hex": 0.6, "named": 0.15, "cross": 0.15, "alias": 0.1}


def sample_corpus(
    n_examples: int,
    seed: int,
    named_pairs: list[tuple[Rgb, Rgb]],
    weights: dict[Form, float] = FORM_WEIGHTS,
) -> list[Example]:
    """Sample the training corpus: i.i.d. examples with the given form mix.

    Named equations draw only from *named_pairs* (the train side of the split);
    hex and cross operands draw uniformly from the full grid, so pair coverage
    is sparse (~n_examples of the grid's ~8.4M pairs) and held-out pairs test
    composition, not recall.
    """
    rng = np.random.default_rng(seed)
    forms, probs = zip(*weights.items(), strict=True)
    examples = []
    for form in rng.choice(forms, n_examples, p=probs):
        match form:
            case "alias":
                a, b = list(PALETTE.values())[rng.integers(len(PALETTE))], None
            case "named":
                a, b = named_pairs[rng.integers(len(named_pairs))]
            case "cross":
                a, b = list(PALETTE.values())[rng.integers(len(PALETTE))], _random_rgb(rng)
            case _:
                a, b = _random_rgb(rng), _random_rgb(rng)
        examples.append(make_example(form, a, b, rng))
    return examples


def sample_unseen(
    form: Form,
    k: int,
    seed: int,
    seen: set[tuple[Rgb, Rgb]],
    max_tries: int = 100,
) -> list[Example]:
    """Sample *k* eval examples whose operand pairs never occurred in training."""
    rng = np.random.default_rng(seed)
    taken = set(seen)  # also keeps the eval pairs distinct from one another
    examples: list[Example] = []
    for _ in range(max_tries * k):
        if len(examples) >= k:
            break
        a = list(PALETTE.values())[rng.integers(len(PALETTE))] if form == "cross" else _random_rgb(rng)
        b = _random_rgb(rng)
        ex = make_example(form, a, b, rng)
        if (p := ex.pair) not in taken and p is not None:
            taken.add(p)
            examples.append(ex)
    return examples


def as_named(pairs: Iterable[tuple[Rgb, Rgb]], seed: int) -> list[Example]:
    """Render pairs as named equations (for the seen/held-out named eval sets)."""
    rng = np.random.default_rng(seed)
    return [make_example("named", a, b, rng) for a, b in pairs]


def dump_example_sets(sets: dict[str, list[Example]]) -> bytes:
    """Serialize eval/probe sets for handoff between experiment steps."""
    return json.dumps({name: [ex._asdict() for ex in exs] for name, exs in sets.items()}).encode()


def load_example_sets(raw: bytes) -> dict[str, list[Example]]:
    def example(d: dict) -> Example:
        rgb = lambda v: (v[0], v[1], v[2])  # noqa: E731
        rhs = rgb(d["rhs"]) if d["rhs"] is not None else None
        return Example(d["prompt"], d["answer"], rgb(d["lhs"]), rhs, rgb(d["result"]))

    return {name: [example(d) for d in exs] for name, exs in json.loads(raw).items()}
