"""The RGB grid as data: the points the color domain is built from.

Drawing them is :mod:`sca.vis`'s job — this module stays on the data side of
the line, because experiments import it and mi-ni fingerprints project source
transitively, so a figure tweak here would re-run them.
"""

from __future__ import annotations

import numpy as np

from sca.data.colors import N_LEVELS, PALETTE


def grid(levels: int = N_LEVELS) -> np.ndarray:
    """The full RGB grid at *levels* steps per channel, as ``[levels³, 3]`` in [0, 1]."""
    c = np.linspace(0, 1, levels)
    return np.stack(np.meshgrid(c, c, c, indexing="ij"), axis=-1).reshape(-1, 3)


def named() -> np.ndarray:
    """The 27 named colors as ``[27, 3]`` in [0, 1] — the 3×3×3 sub-lattice."""
    return np.array(list(PALETTE.values()), dtype=float) / (N_LEVELS - 1)
