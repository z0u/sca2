"""Scratch: the smoke entry of experiment.py as its own experiment (not committed)."""

import importlib.util
from pathlib import Path

from mini import Experiment

_spec = importlib.util.spec_from_file_location("anneal_confirm_ex", Path(__file__).with_name("experiment.py"))
assert _spec is not None and _spec.loader is not None
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)

experiment = Experiment(name="anneal-confirm-smoke", main=ex.smoke, roles=ex.COMPUTE)
