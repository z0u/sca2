"""Ex-2.2.23: the slow seeds, trained for longer — a scout.

Most runs of the recipe rise quickly to a first plateau of task skill, stay there for some tens of epochs, and then
rise again when the model learns the three HSV ops. A few runs never make that second rise within 200 epochs, and
they end well below the others. Those runs set the seed band of every measurement in ex-2.2.21 and ex-2.2.22. This
scout trains the recipe of record and the control at more seeds, at 200 epochs and at 400, to see how often a run
misses the rise, whether the anchor makes it more likely, whether a longer run makes it, and whether a run that rises
late ends like one that rises early. From that, a rule for runs that miss the rise can be fixed before the next
experiment.

This is the design: the constants the report imports. The DAG comes once the design is agreed.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass

DESIGN_ONLY = True


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as in ex-2.2.16 to ex-2.2.22."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex2222 = _load_sibling("ex-2.2.22", "ex2223_ex2222")
ex2221 = ex2222.ex2221

# --- What is inherited, unchanged --------------------------------------------------------------------------

OP_NAMES = ex2222.OP_NAMES
ANCHORED_OP = ex2222.ANCHORED_OP
MODEL = ex2222.MODEL
N_SLICES = ex2222.N_SLICES
LAMBDA_A = ex2222.LAMBDA_A
SELECTIVITY_GATE = ex2222.SELECTIVITY_GATE
GRADING_MIN_DAMAGE = ex2222.GRADING_MIN_DAMAGE
DOSE_GAMMAS = ex2222.DOSE_GAMMAS
"""The recipe of record (the D2.2 design, after ex-2.2.22): the seven-op set, three examples per context at
replacement op noise 0.3, the whole-line label on one `difference` context in fifty, every slice pulled, no cap, no
verification lines. The edit criteria of ex-2.2.21 (E2), unchanged."""

HSV_OPS: tuple[str, ...] = ("hue-hsv", "sat-hsv", "value-hsv")
"""The ops whose learning makes the second rise. On the slow runs of ex-2.2.21 and ex-2.2.22 their skill stays near
0.2 while the other four ops end like those of the fast runs."""
assert set(HSV_OPS) <= set(OP_NAMES)

# --- The conditions ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    name: str
    anchored: bool
    """Anchored runs follow the recipe of record, which is ex-2.2.21's `anchor-whole` condition."""
    reused_as: str
    """The ex-2.2.21 condition whose 200-epoch runs this one reuses at the reused seeds."""


CONDITIONS: tuple[Condition, ...] = (
    Condition("control", anchored=False, reused_as="control"),
    Condition("anchor", anchored=True, reused_as="anchor-whole"),
)

LENGTHS: tuple[int, ...] = (200, 400)
"""Epochs. 200 is the recipe; 400 is the recipe before ex-2.2.19 halved it, and every 400-epoch run there made the
second rise. The anchor schedules scale with the length (warm-up over the first tenth, anneal over the last), and the
learning-rate warm-up stays as it is."""
SHORT, LONG = LENGTHS

SEED_OFFSET = ex2221.SEED_OFFSET
REUSED_SEEDS: tuple[int, ...] = tuple(range(SEED_OFFSET, SEED_OFFSET + 5))
"""Model seeds 700 to 704: ex-2.2.21 trained both conditions at 200 epochs at these, so those runs are reused."""
NEW_SEEDS: tuple[int, ...] = tuple(range(SEED_OFFSET + 5, SEED_OFFSET + 12))
"""Model seeds 705 to 711, new for both conditions. (Ex-2.2.22 trained `hinge` at 705, which is a different
condition.)"""
SEEDS: tuple[int, ...] = REUSED_SEEDS + NEW_SEEDS

N_NEW_RUNS = len(CONDITIONS) * (len(NEW_SEEDS) + len(SEEDS))
"""New 200-epoch runs at the new seeds, and 400-epoch runs at every seed."""
assert N_NEW_RUNS == 38

# --- The measurements --------------------------------------------------------------------------------------

RISE_LEVEL = 0.35
"""A run has made the second rise once its HSV skill (EEM averaged over `HSV_OPS`) passes this level: about halfway
between the plateau, near 0.2, and where the fast runs end, near 0.5. The rise epoch is the first trajectory record
at or above it."""

TRAJ_STRIDE_EPOCHS = 4
"""One trajectory record every four epochs at either length, as in ex-2.2.21 at 200 epochs, so the rise epoch is
resolved alike at both lengths."""

BUDGET_USD = 15
"""About \\$0.13 for a 200-epoch run on an L4 (ex-2.2.22), twice that at 400 epochs, plus the scoring passes."""
