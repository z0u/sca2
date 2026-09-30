import numpy as np
import pytest

from sca.config import SchedulerConfig
from sca.training.scheduler import configure_schedule

SHEET = "STEP,PHASE,ACTION,lr\n0,Warmup,,0.01\n100,Hold,,1\n800,Anneal,,0.5\n1000,,,0.01\n"


def test_lr_sheet_left_out_of_dump_when_unset():
    config = SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01)
    assert "lr_sheet" not in config.model_dump(mode="json")
    assert "lr_sheet" in config.model_copy(update={"lr_sheet": SHEET}).model_dump(mode="json")


def test_lr_sheet_stretches_over_the_run():
    config = SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01, lr_sheet=SHEET)
    peak, epoch_length = 0.002, 250  # 1000 steps, so sheet steps map one to one
    schedule = configure_schedule(config, peak, epoch_length)
    lr = np.array([float(np.asarray(schedule(t))) for t in (0, 100, 450, 800, 1000, 5000)])
    assert lr == pytest.approx(peak * np.array([0.01, 1.0, 0.75, 0.5, 0.01, 0.01]), rel=1e-4)

    # Minimum jerk eases symmetrically, so halfway between keyframes is halfway between their values.
    assert float(np.asarray(schedule(50))) == pytest.approx(peak * 0.505, rel=1e-4)

    # Twice the run, same shape: the keyframes land at twice the step.
    longer = configure_schedule(config, peak, 2 * epoch_length)
    assert float(np.asarray(longer(200))) == pytest.approx(peak, rel=1e-3)
    assert float(np.asarray(longer(1600))) == pytest.approx(0.5 * peak, rel=1e-3)
